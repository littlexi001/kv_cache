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

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    from transformers import AutoModelWithLMHead as AutoModelForCausalLM
    from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze answer logits under redefined top1 category ablations.")
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
            / "local_8sample_4k_redefined_logit_analysis"
        ),
    )
    parser.add_argument("--max_samples", type=int, default=8)
    parser.add_argument("--max_context_chars", type=int, default=4000)
    parser.add_argument("--top_ratio", type=float, default=0.01)
    parser.add_argument("--modes", default="full_attention,top1_all,drop_other,drop_answer,drop_end")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--trust_remote_code", type=exp.str2bool, default=True)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def logits_for_answer(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    prompt_token_count: int,
    answer_token_count: int,
    input_device: torch.device,
    context: exp.AblationContext | None,
) -> torch.Tensor:
    input_ids = input_ids.to(input_device)
    with exp.active_context(context):
        outputs = model(input_ids=input_ids, return_dict=True, use_cache=False, output_hidden_states=False)
    logits = outputs.logits[:, :-1, :].float()
    answer_start = prompt_token_count - 1
    answer_logits = logits[:, answer_start : answer_start + answer_token_count, :].detach().cpu()
    del outputs, logits
    return answer_logits[0]


def token_rows(
    tokenizer: Any,
    sample: dict[str, Any],
    mode: str,
    answer_logits: torch.Tensor,
    answer_ids: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    log_probs = F.log_softmax(answer_logits, dim=-1)
    probs = log_probs.exp()
    top_probs, top_ids = probs.max(dim=-1)
    sorted_logits, sorted_ids = torch.sort(answer_logits, dim=-1, descending=True)
    for pos, gold_id in enumerate(answer_ids):
        gold_logit = float(answer_logits[pos, gold_id])
        gold_log_prob = float(log_probs[pos, gold_id])
        gold_prob = float(probs[pos, gold_id])
        pred_id = int(top_ids[pos])
        pred_prob = float(top_probs[pos])
        rank_tensor = (sorted_ids[pos] == gold_id).nonzero(as_tuple=False)
        gold_rank = int(rank_tensor[0, 0]) + 1 if rank_tensor.numel() else -1
        best_wrong_logit = float(sorted_logits[pos, 1] if pred_id == gold_id else sorted_logits[pos, 0])
        entropy = float(-(probs[pos] * log_probs[pos]).sum())
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "mode": mode,
                "answer_token_pos": pos,
                "gold_token_id": gold_id,
                "gold_token_text": tokenizer.decode([gold_id], skip_special_tokens=False),
                "pred_token_id": pred_id,
                "pred_token_text": tokenizer.decode([pred_id], skip_special_tokens=False),
                "is_correct": str(pred_id == gold_id),
                "gold_rank": gold_rank,
                "gold_logit": gold_logit,
                "pred_logit": float(answer_logits[pos, pred_id]),
                "gold_minus_best_wrong_logit": gold_logit - best_wrong_logit,
                "gold_prob": gold_prob,
                "pred_prob": pred_prob,
                "nll": -gold_log_prob,
                "entropy": entropy,
            }
        )
    return rows


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
    modes = exp.parse_modes(args.modes)

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
    all_rows: list[dict[str, Any]] = []
    answer_text_rows: list[dict[str, Any]] = []
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
        for mode in modes:
            context = (
                None
                if mode == "full_attention"
                else exp.AblationContext(mode, args.top_ratio, answer_indices, query_indices, suffix_start_index)
            )
            answer_logits = logits_for_answer(
                model,
                input_ids,
                prompt_token_count,
                answer_token_count,
                input_device,
                context,
            )
            rows = token_rows(tokenizer, sample, mode, answer_logits, answer_ids)
            all_rows.extend(rows)
            greedy = "".join(row["pred_token_text"] for row in rows).strip()
            answer_text_rows.append(
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
            print(f"finished {sample['sample_id']} {mode}", flush=True)
            del answer_logits
        if input_device.type == "cuda":
            torch.cuda.empty_cache()

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
        output_dir / "logit_summary.csv",
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
        answer_text_rows,
        ["sample_id", "mode", "answer", "greedy_answer", "keyword_recall", "token_accuracy", "sequence_logprob", "sequence_prob"],
    )
    (output_dir / "config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote outputs to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
