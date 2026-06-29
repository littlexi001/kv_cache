from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
SOURCE_PROJECT = REPO_ROOT / "ymluo/projects/qwen3_top2_head_limit3_ppl/src"
sys.path.insert(0, str(SOURCE_PROJECT))


def normalize_text(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def load_samples(jsonl_path: Path, lengths: set[int], depths: set[float], max_cases: int) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if lengths and int(row["context_length_words_target"]) not in lengths:
                continue
            if depths and float(row["target_depth_percent"]) not in depths:
                continue
            samples.append(row)
            if max_cases > 0 and len(samples) >= max_cases:
                break
    return samples


def generate_one(
    model: Any,
    tokenizer: Any,
    input_device: torch.device,
    sample: dict[str, Any],
    mode: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    mode_kind, mode_max_heads = parse_mode_config(mode)
    reuse_state = ReuseCandidateState() if mode_kind in {"qabs_reuse_rerank", "lagged_reuse_rerank"} else None
    sparq_mean_state = SparQMeanState() if mode_kind == "sparq_fast_attention" else None
    prompt = sample["prompt"] + "\nAnswer:"
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    encoded = {key: value.to(input_device) for key, value in encoded.items()}
    started = time.perf_counter()
    with torch.inference_mode():
        with attention_mode(
            mode=mode,
            top_fraction=args.top_fraction,
            max_heads_per_token=mode_max_heads if mode_max_heads is not None else args.max_heads_per_token,
            always_keep_self=True,
            protect_sink_tokens=args.protect_sink_tokens,
            protect_recent_tokens=args.protect_recent_tokens,
            load_stats=None,
            reuse_state=reuse_state,
            sparq_mean_state=sparq_mean_state,
            qabs_fast_path=True,
            qabs_cuda_final_kernel=args.qabs_cuda_final_kernel,
            qabs_cuda_candidate_kernel=args.qabs_cuda_candidate_kernel,
            qabs_cuda_reuse_select_kernel=args.qabs_cuda_reuse_select_kernel,
        ):
            output_ids = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )
    seconds = time.perf_counter() - started
    generated_ids = output_ids[0, encoded["input_ids"].shape[-1] :]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    expected = sample["expected_answer"]
    normalized_answer = normalize_text(answer)
    normalized_expected = normalize_text(expected)
    exact_contains = normalized_expected in normalized_answer
    token_overlap = 0.0
    expected_tokens = set(normalized_expected.split())
    if expected_tokens:
        token_overlap = len(expected_tokens & set(normalized_answer.split())) / len(expected_tokens)
    return {
        "sample_id": sample["sample_id"],
        "context_length_words_target": sample["context_length_words_target"],
        "target_depth_percent": sample["target_depth_percent"],
        "mode": mode,
        "expected_answer": expected,
        "generated_answer": answer.replace("\n", "\\n"),
        "contains_expected": int(exact_contains),
        "expected_token_overlap": f"{token_overlap:.6f}",
        "prompt_tokens": int(encoded["input_ids"].shape[-1]),
        "seconds": f"{seconds:.6f}",
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault((row["mode"], str(row["context_length_words_target"])), []).append(row)
    summary: list[dict[str, Any]] = []
    for (mode, length), bucket in sorted(buckets.items()):
        count = len(bucket)
        contains = sum(int(row["contains_expected"]) for row in bucket)
        overlap = sum(float(row["expected_token_overlap"]) for row in bucket) / max(1, count)
        seconds = sum(float(row["seconds"]) for row in bucket)
        summary.append(
            {
                "mode": mode,
                "context_length_words_target": length,
                "case_count": count,
                "contains_expected_rate": f"{contains / max(1, count):.6f}",
                "mean_expected_token_overlap": f"{overlap:.6f}",
                "total_seconds": f"{seconds:.6f}",
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate needle-in-a-haystack generation accuracy by attention mode.")
    parser.add_argument("--model_name_or_path", default="/mnt/workspace/Qwen3-0.6B")
    parser.add_argument("--jsonl_path", default=str(REPO_ROOT / "ymluo/projects/qwen3_kcache_l2_neighbor_analysis/data/needle_in_haystack/needle_in_haystack.jsonl"))
    parser.add_argument("--output_dir", default=str(REPO_ROOT / "ymluo/projects/qabs8cand3reuse_quality_suite/outputs/needle_generation"))
    parser.add_argument("--modes", default="baseline,qabs8cand3reuse,sparqfast8cand3")
    parser.add_argument("--lengths", default="1000,2000,4000,8000")
    parser.add_argument("--depths", default="0,25,50,75,100")
    parser.add_argument("--max_cases", type=int, default=12)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--top_fraction", type=float, default=0.02)
    parser.add_argument("--max_heads_per_token", type=int, default=3)
    parser.add_argument("--protect_sink_tokens", type=int, default=10)
    parser.add_argument("--protect_recent_tokens", type=int, default=10)
    parser.add_argument("--qabs_cuda_final_kernel", type=str2bool, default=True)
    parser.add_argument("--qabs_cuda_candidate_kernel", type=str2bool, default=False)
    parser.add_argument("--qabs_cuda_reuse_select_kernel", type=str2bool, default=False)
    args = parser.parse_args()

    global torch
    global AutoModelForCausalLM
    global AutoTokenizer
    global ReuseCandidateState
    global SparQMeanState
    global attention_mode
    global install_qwen3_attention_patch
    global parse_mode_config
    global pick_input_device
    global resolve_dtype

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from evaluate_qwen3_top2_head_limit3_ppl import (
        ReuseCandidateState,
        SparQMeanState,
        attention_mode,
        install_qwen3_attention_patch,
        parse_mode_config,
        pick_input_device,
        resolve_dtype,
    )

    lengths = {int(value) for value in args.lengths.split(",") if value}
    depths = {float(value) for value in args.depths.split(",") if value}
    samples = load_samples(Path(args.jsonl_path), lengths, depths, args.max_cases)
    if not samples:
        raise RuntimeError(f"No needle samples found in {args.jsonl_path}.")

    requested_device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model_dtype = resolve_dtype(args.dtype, requested_device)
    load_kwargs: dict[str, Any] = {"trust_remote_code": True, "torch_dtype": model_dtype}
    if args.device_map.lower() != "none":
        load_kwargs["device_map"] = args.device_map
    if args.attn_implementation.lower() != "auto":
        load_kwargs["attn_implementation"] = args.attn_implementation
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **load_kwargs)
    if args.device_map.lower() == "none":
        model = model.to(requested_device)
    model.eval()
    model.config.use_cache = True
    install_qwen3_attention_patch()
    input_device = pick_input_device(model, requested_device)

    rows: list[dict[str, Any]] = []
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    for sample in samples:
        for mode in modes:
            print(f"=== needle generation {sample['sample_id']} mode={mode} ===", flush=True)
            rows.append(generate_one(model, tokenizer, input_device, sample, mode, args))

    output_dir = Path(args.output_dir)
    result_fields = [
        "sample_id",
        "context_length_words_target",
        "target_depth_percent",
        "mode",
        "expected_answer",
        "generated_answer",
        "contains_expected",
        "expected_token_overlap",
        "prompt_tokens",
        "seconds",
    ]
    write_csv(output_dir / "needle_generation_results.csv", rows, result_fields)
    write_csv(
        output_dir / "needle_generation_summary.csv",
        summarize(rows),
        [
            "mode",
            "context_length_words_target",
            "case_count",
            "contains_expected_rate",
            "mean_expected_token_overlap",
            "total_seconds",
        ],
    )
    print(f"needle generation outputs: {output_dir}")


if __name__ == "__main__":
    main()
