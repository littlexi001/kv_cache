from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import run_top1_category_ablation as exp
from analyze_redefined_logits import token_rows, write_csv

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    from transformers import AutoModelWithLMHead as AutoModelForCausalLM
    from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replace drop_answer X singular values and evaluate logits.")
    parser.add_argument("--model_name_or_path", default=str(exp.REPO_ROOT / "ymluo" / "models" / "Qwen3-0.6B"))
    parser.add_argument("--data_path", default=exp.DEFAULT_DATA_PATH)
    parser.add_argument(
        "--output_dir",
        default=str(
            exp.REPO_ROOT
            / "ymluo"
            / "projects"
            / "qwen3_top1_category_ablation"
            / "outputs"
            / "local_8sample_4k_redefined_svd_surgery_drop_answer"
        ),
    )
    parser.add_argument("--max_samples", type=int, default=8)
    parser.add_argument("--max_context_chars", type=int, default=4000)
    parser.add_argument("--top_ratio", type=float, default=0.01)
    parser.add_argument("--lab2_start_rank", type=int, default=45)
    parser.add_argument("--lab2_end_rank", type=int, default=55)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--trust_remote_code", type=exp.str2bool, default=True)
    return parser.parse_args()


@torch.inference_mode()
def hidden_for_answer(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    prompt_token_count: int,
    answer_token_count: int,
    input_device: torch.device,
    context: exp.AblationContext | None,
) -> torch.Tensor:
    input_ids = input_ids.to(input_device)
    with exp.active_context(context):
        outputs = model(
            input_ids=input_ids,
            return_dict=True,
            use_cache=False,
            output_hidden_states=True,
            output_attentions=False,
        )
    answer_start = prompt_token_count - 1
    hidden = outputs.hidden_states[-1][0, answer_start : answer_start + answer_token_count, :].detach().float().cpu()
    del outputs
    return hidden


@torch.inference_mode()
def logits_from_x(model: torch.nn.Module, x: torch.Tensor, input_device: torch.device, batch_size: int = 16) -> torch.Tensor:
    logits: list[torch.Tensor] = []
    lm_head = model.get_output_embeddings()
    head_dtype = lm_head.weight.dtype
    for start in range(0, x.shape[0], batch_size):
        batch = x[start : start + batch_size].to(device=input_device, dtype=head_dtype)
        out = lm_head(batch).detach().float().cpu()
        logits.append(out)
    return torch.cat(logits, dim=0)


def svd_replace(drop_x: torch.Tensor, full_x: torch.Tensor, rank_start: int | None, rank_end: int | None) -> tuple[torch.Tensor, dict[str, Any]]:
    u_drop, s_drop, vh_drop = torch.linalg.svd(drop_x, full_matrices=False)
    s_full = torch.linalg.svdvals(full_x)
    s_new = s_drop.clone()
    if rank_start is None or rank_end is None:
        count = min(s_new.numel(), s_full.numel())
        s_new[:count] = s_full[:count]
        label = "all"
    else:
        start = max(rank_start - 1, 0)
        end = min(rank_end, s_new.numel(), s_full.numel())
        if start >= end:
            raise ValueError(f"Invalid rank interval: {rank_start}-{rank_end}")
        s_new[start:end] = s_full[start:end]
        label = f"{rank_start}_{rank_end}"
    corrected = (u_drop * s_new.unsqueeze(0)) @ vh_drop
    meta = {
        "replacement": label,
        "drop_singular_sum": float(s_drop.sum()),
        "full_singular_sum": float(s_full.sum()),
        "new_singular_sum": float(s_new.sum()),
        "drop_energy": float((s_drop * s_drop).sum()),
        "full_energy": float((s_full * s_full).sum()),
        "new_energy": float((s_new * s_new).sum()),
    }
    return corrected.contiguous(), meta


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for mode in sorted({row["mode"] for row in rows}):
        rs = [row for row in rows if row["mode"] == mode]
        mean_nll = sum(float(row["nll"]) for row in rs) / len(rs)
        result.append(
            {
                "mode": mode,
                "token_count": len(rs),
                "token_accuracy": sum(row["is_correct"] == "True" for row in rs) / len(rs),
                "mean_nll": mean_nll,
                "mean_ppl": math.exp(min(mean_nll, 80.0)),
                "mean_gold_prob": sum(float(row["gold_prob"]) for row in rs) / len(rs),
                "median_gold_prob": sorted(float(row["gold_prob"]) for row in rs)[len(rs) // 2],
                "mean_pred_prob": sum(float(row["pred_prob"]) for row in rs) / len(rs),
                "mean_gold_rank": sum(float(row["gold_rank"]) for row in rs) / len(rs),
                "mean_margin": sum(float(row["gold_minus_best_wrong_logit"]) for row in rs) / len(rs),
                "mean_entropy": sum(float(row["entropy"]) for row in rs) / len(rs),
                "very_bad_token_count_nll_gt_10": sum(float(row["nll"]) > 10.0 for row in rs),
                "very_bad_token_fraction_nll_gt_10": sum(float(row["nll"]) > 10.0 for row in rs) / len(rs),
            }
        )
    return result


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exp.install_qwen3_attention_patch()
    requested_device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dtype = exp.resolve_dtype(args.dtype, requested_device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
    load_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "trust_remote_code": args.trust_remote_code,
        "attn_implementation": args.attn_implementation,
    }
    if requested_device.type != "cpu" and args.device_map.lower() != "none":
        load_kwargs["device_map"] = args.device_map
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **load_kwargs)
    if requested_device.type == "cpu" or args.device_map.lower() == "none":
        model.to(requested_device)
    model.eval()
    input_device = exp.pick_input_device(model, requested_device)

    samples = exp.load_samples(Path(args.data_path), args.max_samples, args.max_context_chars)
    full_chunks: list[torch.Tensor] = []
    drop_chunks: list[torch.Tensor] = []
    token_specs: list[dict[str, Any]] = []
    answer_text_by_sample: dict[str, str] = {}

    for sample in samples:
        prompt = exp.build_prompt(sample)
        payload = exp.prompt_answer_payload(tokenizer, prompt, str(sample["answer"]))
        token_ids = payload["token_ids"]
        input_ids = torch.tensor([token_ids], dtype=torch.long)
        prompt_token_count = int(payload["prompt_token_count"])
        answer_token_count = int(payload["answer_token_count"])
        answer_ids = token_ids[prompt_token_count : prompt_token_count + answer_token_count]
        answer_indices = exp.answer_span_token_indices(prompt, str(sample["answer"]), payload["offsets"], prompt_token_count)
        suffix_start_index = exp.answer_suffix_start_token(prompt, payload["offsets"], prompt_token_count)
        query_indices = set(range(prompt_token_count - 1, prompt_token_count + answer_token_count - 1))
        drop_context = exp.AblationContext("drop_answer", args.top_ratio, answer_indices, query_indices, suffix_start_index)

        full_x = hidden_for_answer(model, input_ids, prompt_token_count, answer_token_count, input_device, None)
        drop_x = hidden_for_answer(model, input_ids, prompt_token_count, answer_token_count, input_device, drop_context)
        full_chunks.append(full_x)
        drop_chunks.append(drop_x)
        answer_text_by_sample[str(sample["sample_id"])] = str(sample["answer"])
        for pos, gold_id in enumerate(answer_ids):
            token_specs.append({"sample": sample, "answer_token_pos": pos, "gold_id": gold_id})
        print(f"collected {sample['sample_id']}", flush=True)
        if input_device.type == "cuda":
            torch.cuda.empty_cache()

    full_x_all = torch.cat(full_chunks, dim=0)
    drop_x_all = torch.cat(drop_chunks, dim=0)
    lab1_x, lab1_meta = svd_replace(drop_x_all, full_x_all, None, None)
    lab2_x, lab2_meta = svd_replace(drop_x_all, full_x_all, args.lab2_start_rank, args.lab2_end_rank)

    mode_to_x = {
        "full_attention_lmhead": full_x_all,
        "drop_answer_lmhead": drop_x_all,
        "lab1_drop_answer_s_full_all": lab1_x,
        f"lab2_drop_answer_s_full_{args.lab2_start_rank}_{args.lab2_end_rank}": lab2_x,
    }

    all_rows: list[dict[str, Any]] = []
    sequence_rows: list[dict[str, Any]] = []
    for mode, x in mode_to_x.items():
        logits = logits_from_x(model, x, input_device)
        cursor = 0
        for sample in samples:
            sample_specs = [spec for spec in token_specs if spec["sample"]["sample_id"] == sample["sample_id"]]
            answer_ids = [int(spec["gold_id"]) for spec in sample_specs]
            sample_logits = logits[cursor : cursor + len(answer_ids)]
            cursor += len(answer_ids)
            rows = token_rows(tokenizer, sample, mode, sample_logits, answer_ids)
            all_rows.extend(rows)
            greedy = "".join(row["pred_token_text"] for row in rows).strip()
            sequence_rows.append(
                {
                    "sample_id": sample["sample_id"],
                    "mode": mode,
                    "answer": sample["answer"],
                    "greedy_answer": greedy,
                    "keyword_recall": exp.keyword_recall(greedy, str(sample["answer"]))[2],
                    "token_accuracy": sum(row["is_correct"] == "True" for row in rows) / len(rows),
                    "sequence_logprob": -sum(float(row["nll"]) for row in rows),
                    "sequence_prob": math.exp(-sum(float(row["nll"]) for row in rows)),
                }
            )
        print(f"evaluated {mode}", flush=True)

    token_fields = [
        "sample_id",
        "mode",
        "answer_token_pos",
        "gold_token_id",
        "gold_token_text",
        "pred_token_id",
        "pred_token_text",
        "is_correct",
        "gold_rank",
        "gold_logit",
        "pred_logit",
        "gold_minus_best_wrong_logit",
        "gold_prob",
        "pred_prob",
        "nll",
        "entropy",
    ]
    write_csv(output_dir / "answer_token_logits.csv", all_rows, token_fields)
    write_csv(
        output_dir / "summary.csv",
        summarize(all_rows),
        [
            "mode",
            "token_count",
            "token_accuracy",
            "mean_nll",
            "mean_ppl",
            "mean_gold_prob",
            "median_gold_prob",
            "mean_pred_prob",
            "mean_gold_rank",
            "mean_margin",
            "mean_entropy",
            "very_bad_token_count_nll_gt_10",
            "very_bad_token_fraction_nll_gt_10",
        ],
    )
    write_csv(
        output_dir / "answer_sequence_summary.csv",
        sequence_rows,
        ["sample_id", "mode", "answer", "greedy_answer", "keyword_recall", "token_accuracy", "sequence_logprob", "sequence_prob"],
    )

    svd_rows = []
    for mode, x in mode_to_x.items():
        s = torch.linalg.svdvals(x)
        total = float((s * s).sum())
        running = 0.0
        for rank, value in enumerate(s.tolist(), start=1):
            running += float(value) * float(value)
            svd_rows.append(
                {
                    "mode": mode,
                    "rank": rank,
                    "singular_value": float(value),
                    "energy_ratio": (float(value) * float(value)) / total if total else 0.0,
                    "cumulative_energy_ratio": running / total if total else 0.0,
                }
            )
    write_csv(output_dir / "surgery_singular_values.csv", svd_rows, ["mode", "rank", "singular_value", "energy_ratio", "cumulative_energy_ratio"])

    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "args": vars(args),
                "lab1": lab1_meta,
                "lab2": lab2_meta,
                "x_shape": list(full_x_all.shape),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    print(f"wrote outputs to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
