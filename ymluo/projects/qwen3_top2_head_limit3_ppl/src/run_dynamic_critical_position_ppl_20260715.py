from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_controlled_public_kv_benchmark_v1 as lb  # noqa: E402
from critical_position_router import (  # noqa: E402
    build_feature_vector,
    load_router_artifact,
    predict_risk,
)
from evaluate_qwen3_top2_head_limit3_ppl import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    pick_input_device,
    resolve_dtype,
)
from run_critical_position_budget_probe_20260715 import (  # noqa: E402
    causal_logit_features,
    run_one_token,
)
from run_head_top2_targeted_ppl_20260714 import (  # noqa: E402
    head_top_fraction_mode,
    install_llama_head_top_fraction_patch,
    parse_int_list,
    set_attention_implementation,
)
from run_multitopic_lpcm_ppl_20260714 import (  # noqa: E402
    TOPICS,
    encode_topic_stream,
    make_bundle,
    topic_names,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a causal two-stage critical-position attention budget router.")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--router_path", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--topics", default="sports,medicine")
    parser.add_argument("--window_indices", default="2")
    parser.add_argument("--history_tokens", type=int, default=32_000)
    parser.add_argument("--query_tokens", type=int, default=256)
    parser.add_argument("--eval_tokens", type=int, default=256)
    parser.add_argument("--window_stride_tokens", type=int, default=32_512)
    parser.add_argument("--prefill_chunk_tokens", type=int, default=2048)
    parser.add_argument("--dataset_cache_dir", default="/home/fdong/ymluo/datasets/sklearn")
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    return parser.parse_args()


def crop_cache(past_key_values: Any, length: int) -> Any:
    crop = getattr(past_key_values, "crop", None)
    if not callable(crop):
        raise RuntimeError(f"cache type {type(past_key_values).__name__} does not support rollback crop()")
    crop(int(length))
    if lb.cache_sequence_length(past_key_values) != int(length):
        raise RuntimeError("cache rollback did not restore the requested length")
    return past_key_values


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_model(args: argparse.Namespace) -> tuple[Any, torch.nn.Module, torch.device]:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = resolve_dtype(args.dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
        "attn_implementation": "sdpa",
    }
    if args.device_map:
        kwargs["device_map"] = args.device_map
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **kwargs)
    model.eval()
    model.config.use_cache = True
    return tokenizer, model, pick_input_device(model, device)


@torch.inference_mode()
def routed_step(
    model: torch.nn.Module,
    tokenizer: Any,
    artifact: dict[str, Any],
    cache: Any,
    input_id: int,
    position: int,
    prediction_index: int,
    topic: str,
    history_counts: Counter[int],
    input_device: torch.device,
) -> tuple[Any, torch.Tensor, dict[str, Any]]:
    low_fraction = float(artifact["low_fraction"])
    high_fraction = float(artifact["high_fraction"])
    previous_length = lb.cache_sequence_length(cache)
    started = time.perf_counter()
    with head_top_fraction_mode(low_fraction):
        low_cache, low_logits, low_seconds, attention_features = run_one_token(
            model,
            input_id,
            cache,
            position,
            input_device,
            collect_attention_stats=True,
        )
    causal_features = causal_logit_features(low_logits)
    top1_id = int(causal_features["top1_id"])
    top1_text = tokenizer.decode(
        [top1_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    input_text = tokenizer.decode(
        [int(input_id)], skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    feature_vector = build_feature_vector(
        logit_features=causal_features,
        attention_features=attention_features,
        top1_text=top1_text,
        top1_history_frequency=int(history_counts[top1_id]),
        prediction_index=prediction_index,
        prediction_horizon=int(artifact["prediction_horizon"]),
        topic=topic,
        input_text=input_text,
        input_history_frequency=int(history_counts[int(input_id)]),
    )
    risk = predict_risk(artifact, feature_vector)
    use_high = risk >= float(artifact["threshold"])
    high_seconds = 0.0
    if use_high:
        crop_cache(low_cache, previous_length)
        with head_top_fraction_mode(high_fraction):
            final_cache, final_logits, high_seconds, _ = run_one_token(
                model,
                input_id,
                low_cache,
                position,
                input_device,
                collect_attention_stats=False,
            )
    else:
        final_cache, final_logits = low_cache, low_logits
    return final_cache, final_logits, {
        "risk_score": risk,
        "use_high": int(use_high),
        "committed_fraction": high_fraction if use_high else low_fraction,
        "executed_fraction": low_fraction + (high_fraction if use_high else 0.0),
        "low_seconds": low_seconds,
        "high_seconds": high_seconds,
        "step_seconds": time.perf_counter() - started,
        "provisional_top1_id": top1_id,
        "provisional_top1_probability": float(causal_features["top1_probability"]),
        **attention_features,
    }


def main() -> None:
    args = parse_args()
    artifact = load_router_artifact(args.router_path)
    topics = topic_names(args.topics)
    windows = parse_int_list(args.window_indices)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(
        json.dumps({**vars(args), "router_path": str(args.router_path)}, indent=2, default=str),
        encoding="utf-8",
    )

    install_llama_head_top_fraction_patch()
    tokenizer, model, input_device = load_model(args)
    required_tokens = max(windows) * args.window_stride_tokens + args.history_tokens + args.eval_tokens
    token_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for topic in topics:
        stream = encode_topic_stream(
            tokenizer, TOPICS[topic], required_tokens, args.dataset_cache_dir, args.seed
        )
        for window in windows:
            start = window * args.window_stride_tokens
            history = stream[start : start + args.history_tokens]
            target_ids = stream[start + args.history_tokens : start + args.history_tokens + args.eval_tokens]
            remote_ids = history[: -args.query_tokens]
            query_ids = history[-args.query_tokens :]
            history_counts = Counter(history)
            bundle, _ = make_bundle(tokenizer, remote_ids, page_tokens=16)

            set_attention_implementation(model, "sdpa")
            with head_top_fraction_mode(None):
                cache, prefill_seconds = lb.prefill_prefix(
                    model, bundle, input_device, args.prefill_chunk_tokens
                )
            set_attention_implementation(model, "eager")
            previous_logits: torch.Tensor | None = None
            previous_decision: dict[str, Any] | None = None
            query_high = 0
            for offset, input_id in enumerate(query_ids):
                cache, previous_logits, previous_decision = routed_step(
                    model,
                    tokenizer,
                    artifact,
                    cache,
                    int(input_id),
                    len(remote_ids) + offset,
                    min(args.eval_tokens - 1, offset),
                    topic,
                    history_counts,
                    input_device,
                )
                query_high += int(previous_decision["use_high"])
            if previous_logits is None or previous_decision is None:
                raise RuntimeError("query must be non-empty")

            loss_sum = 0.0
            target_high = 0
            committed_sum = 0.0
            executed_sum = 0.0
            online_seconds = 0.0
            for target_index in range(args.eval_tokens):
                label_id = int(target_ids[target_index])
                nll = float(
                    F.cross_entropy(
                        previous_logits.float(),
                        torch.tensor([label_id], dtype=torch.long, device=previous_logits.device),
                        reduction="sum",
                    ).item()
                )
                loss_sum += nll
                target_high += int(previous_decision["use_high"])
                committed_sum += float(previous_decision["committed_fraction"])
                executed_sum += float(previous_decision["executed_fraction"])
                online_seconds += float(previous_decision["step_seconds"])
                token_text = tokenizer.decode(
                    [label_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
                )
                token_rows.append(
                    {
                        "topic": topic,
                        "window": window,
                        "target_index": target_index,
                        "token_id": label_id,
                        "token_text": token_text.replace("\n", "\\n").replace("\r", "\\r"),
                        "nll": nll,
                        **previous_decision,
                    }
                )
                if target_index + 1 < args.eval_tokens:
                    cache, previous_logits, previous_decision = routed_step(
                        model,
                        tokenizer,
                        artifact,
                        cache,
                        label_id,
                        len(remote_ids) + len(query_ids) + target_index,
                        target_index + 1,
                        topic,
                        history_counts,
                        input_device,
                    )

            mean_nll = loss_sum / args.eval_tokens
            summary = {
                "topic": topic,
                "window": window,
                "tokens": args.eval_tokens,
                "nll": mean_nll,
                "ppl": math.exp(min(20.0, mean_nll)),
                "target_high_rate": target_high / args.eval_tokens,
                "query_high_rate": query_high / len(query_ids),
                "mean_committed_fraction": committed_sum / args.eval_tokens,
                "mean_executed_fraction": executed_sum / args.eval_tokens,
                "prefill_seconds": prefill_seconds,
                "target_online_seconds": online_seconds,
            }
            summary_rows.append(summary)
            write_csv(args.output_dir / "token_results.csv", token_rows)
            write_csv(args.output_dir / "summary.csv", summary_rows)
            print(json.dumps(summary, indent=2), flush=True)
            del cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    write_csv(args.output_dir / "token_results.csv", token_rows)
    write_csv(args.output_dir / "summary.csv", summary_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary_rows, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
