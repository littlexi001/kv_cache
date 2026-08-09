from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_controlled_public_kv_benchmark_v1 as lb  # noqa: E402
from run_adaptive_mass_budget_ppl_20260715 import write_csv  # noqa: E402
from run_critical_position_budget_probe_20260715 import (  # noqa: E402
    load_model,
    run_one_token,
)
from run_head_top2_targeted_ppl_20260714 import (  # noqa: E402
    head_adaptive_mass_mode,
    head_qabs_sampled_mass_mode,
    head_top_fraction_mode,
    install_llama_head_top_fraction_patch,
    parse_int_list,
    current_qabs_retrieval_features,
    qabs_runtime_budget_fraction,
    set_attention_implementation,
)
from run_multitopic_lpcm_ppl_20260714 import (  # noqa: E402
    TOPICS,
    encode_topic_stream,
    make_bundle,
    topic_names,
)


ACTION_LATENCY_MS = {"0.005": 0.4544256, "0.01": 0.6304461, "0.02": 0.9554893}
RETRIEVAL_FEATURE_LATENCY_MS = 0.0184269
MLP_ROUTER_TOKEN_LATENCY_MS = 0.0035939
FULL_LATENCY_MS = 2.628


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--router_path", type=Path, required=True)
    parser.add_argument("--full_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--topics", default="sports,medicine")
    parser.add_argument("--window_indices", default="2")
    parser.add_argument("--history_tokens", type=int, default=32_000)
    parser.add_argument("--query_tokens", type=int, default=256)
    parser.add_argument("--eval_tokens", type=int, default=256)
    parser.add_argument("--window_stride_tokens", type=int, default=32_512)
    parser.add_argument("--projection_dim", type=int, default=32)
    parser.add_argument(
        "--gqa_candidate_mode",
        choices=["independent", "shared_max", "shared_zmax", "shared_mean"],
        default="independent",
    )
    parser.add_argument(
        "--collect_gqa_union_stats",
        action="store_true",
        help="Collect final-action KV-head candidate unions; adds diagnostic overhead.",
    )
    parser.add_argument("--prefill_chunk_tokens", type=int, default=2048)
    parser.add_argument("--dataset_cache_dir", default="/home/fdong/ymluo/datasets/sklearn")
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument(
        "--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="float16"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    return parser.parse_args()


def provisional_features(
    logits: torch.Tensor,
    target_index: int,
    retrieval_features: dict[str, float],
    feature_names: tuple[str, ...],
) -> np.ndarray:
    log_probs = F.log_softmax(logits[0].float(), dim=-1)
    probabilities = log_probs.exp()
    top_probabilities = torch.topk(probabilities, k=5, dim=-1).values
    entropy = float((-(probabilities * log_probs)).sum().item())
    vocabulary_size = int(probabilities.numel())
    values = {
        "target_position_fraction": target_index / 255.0,
        "logit_entropy": entropy,
        "logit_entropy_normalized": entropy / math.log(vocabulary_size),
        "logit_top1_probability": float(top_probabilities[0].item()),
        "logit_top2_probability": float(top_probabilities[1].item()),
        "logit_top1_top2_margin": float(
            (top_probabilities[0] - top_probabilities[1]).item()
        ),
        "logit_top5_mass": float(top_probabilities.sum().item()),
        **retrieval_features,
    }
    return np.asarray([values[name] for name in feature_names], dtype=np.float64)


def select_action(router: dict[str, Any], features: np.ndarray) -> tuple[str, dict[str, float]]:
    if "mlp_action_router" in router:
        mlp = router["mlp_action_router"]
        normalized = (features - mlp["feature_mean"]) / mlp["feature_scale"]
        hidden = np.maximum(0.0, normalized @ mlp["input_weight"] + mlp["input_bias"])
        logits = hidden @ mlp["output_weight"] + mlp["output_bias"]
        class_position = int(np.argmax(logits))
        action_index = int(mlp["classes"][class_position])
        return router["actions"][action_index], {
            router["actions"][int(class_index)]: float(logit)
            for class_index, logit in zip(mlp["classes"], logits, strict=True)
        }
    upper_bounds = {}
    linear_predictions = None
    if "linear_router" in router:
        linear_predictions = (
            router["linear_router"]["weights"] @ features
            + router["linear_router"]["biases"]
        )
    for action_index, action in enumerate(router["actions"]):
        prediction = (
            float(linear_predictions[action_index])
            if linear_predictions is not None
            else float(router["models"][action].predict(features[None, :])[0])
        )
        upper_bounds[action] = prediction + float(router["offsets"][action])
    selected = "0.02"
    for action in router["actions"][:-1]:
        if upper_bounds[action] <= float(router["safe_delta_nll"]):
            selected = action
            break
    return selected, upper_bounds


def nll_from_logits(logits: torch.Tensor, label: int) -> float:
    return -float(F.log_softmax(logits[0].float(), dim=-1)[int(label)].item())


def full_reference_nlls(
    root: Path, topic: str, window: int
) -> dict[int, float]:
    path = root / f"{topic}_w{window}" / "token_results.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {
            int(row["target_index"]): float(row["nll"])
            for row in csv.DictReader(handle)
            if row["method"] == "full_attention"
        }


@torch.inference_mode()
def run_case(
    model: torch.nn.Module,
    prefix_cache: Any,
    remote_count: int,
    query_ids: list[int],
    target_ids: list[int],
    router: dict[str, Any],
    input_device: torch.device,
    full_nll_by_index: dict[int, float],
    projection_dim: int,
    gqa_candidate_mode: str = "independent",
    collect_gqa_union_stats: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cache = prefix_cache
    online_seconds = 0.0
    token_rows = []

    def fixed_step(token_id: int, position: int, fraction: float) -> torch.Tensor:
        nonlocal cache, online_seconds
        with qabs_runtime_budget_fraction(fraction):
            cache, logits, seconds, _ = run_one_token(
                model, token_id, cache, position, input_device, collect_attention_stats=False
            )
        online_seconds += seconds
        return logits

    def progressive_step(
        token_id: int, position: int, target_index: int
    ) -> tuple[torch.Tensor, str, dict[str, float], np.ndarray, float, dict[str, float]]:
        nonlocal cache, online_seconds
        with qabs_runtime_budget_fraction(0.005):
            cache, provisional_logits, seconds, provisional_stats = run_one_token(
                model,
                token_id,
                cache,
                position,
                input_device,
                collect_attention_stats=collect_gqa_union_stats,
            )
        online_seconds += seconds
        features = provisional_features(
            provisional_logits,
            target_index,
            current_qabs_retrieval_features(),
            tuple(router["feature_names"]),
        )
        action, upper_bounds = select_action(router, features)
        step_seconds = seconds
        if action == "0.005":
            return (
                provisional_logits,
                action,
                upper_bounds,
                features,
                step_seconds,
                provisional_stats,
            )
        if not hasattr(cache, "crop"):
            raise RuntimeError("progressive rerun requires a cache with crop()")
        cache.crop(position)
        with qabs_runtime_budget_fraction(float(action)):
            cache, final_logits, rescue_seconds, final_stats = run_one_token(
                model,
                token_id,
                cache,
                position,
                input_device,
                collect_attention_stats=collect_gqa_union_stats,
            )
        online_seconds += rescue_seconds
        return (
            final_logits,
            action,
            upper_bounds,
            features,
            step_seconds + rescue_seconds,
            final_stats,
        )

    set_attention_implementation(model, "eager")
    with head_qabs_sampled_mass_mode(
        1.0e-6,
        (0.005,),
        sample_fraction=0.0025,
        qabs_dim_count=16,
        candidate_fraction=0.005,
        use_cuda_kernels=True,
        skip_candidate_rerank=True,
        score_mode="pca_int8",
        projection_dim=projection_dim,
        gqa_candidate_mode=gqa_candidate_mode,
    ):
        for offset, token_id in enumerate(query_ids[:-1]):
            fixed_step(int(token_id), remote_count + offset, 0.02)

        final_logits, action, upper_bounds, features, seconds, final_stats = progressive_step(
            int(query_ids[-1]), remote_count + len(query_ids) - 1, 0
        )
        token_rows.append(
            {
                "target_index": 0,
                "input_token_id": int(query_ids[-1]),
                "token_id": int(target_ids[0]),
                "nll": nll_from_logits(final_logits, int(target_ids[0])),
                "full_nll": full_nll_by_index[0],
                "selected_action": action,
                "step_seconds": seconds,
                **(
                    {
                        "candidate_union_fraction_mean": float(
                            final_stats["candidate_union_fraction_mean"]
                        )
                    }
                    if collect_gqa_union_stats
                    else {}
                ),
                **{f"router_feature_{i}": float(value) for i, value in enumerate(features)},
                **{f"upper_bound_{name}": value for name, value in upper_bounds.items()},
            }
        )
        for target_index in range(1, len(target_ids)):
            input_id = int(target_ids[target_index - 1])
            final_logits, action, upper_bounds, features, seconds, final_stats = progressive_step(
                input_id,
                remote_count + len(query_ids) + target_index - 1,
                target_index,
            )
            token_rows.append(
                {
                    "target_index": target_index,
                    "input_token_id": input_id,
                    "token_id": int(target_ids[target_index]),
                    "nll": nll_from_logits(final_logits, int(target_ids[target_index])),
                    "full_nll": full_nll_by_index[target_index],
                    "selected_action": action,
                    "step_seconds": seconds,
                    **(
                        {
                            "candidate_union_fraction_mean": float(
                                final_stats["candidate_union_fraction_mean"]
                            )
                        }
                        if collect_gqa_union_stats
                        else {}
                    ),
                    **{f"router_feature_{i}": float(value) for i, value in enumerate(features)},
                    **{f"upper_bound_{name}": value for name, value in upper_bounds.items()},
                }
            )

    mean_nll = sum(float(row["nll"]) for row in token_rows) / len(token_rows)
    mean_full_nll = sum(float(row["full_nll"]) for row in token_rows) / len(token_rows)
    action_counts = Counter(str(row["selected_action"]) for row in token_rows)
    mean_fraction = sum(float(row["selected_action"]) for row in token_rows) / len(token_rows)
    estimated_latency = sum(
        ACTION_LATENCY_MS["0.005"]
        + RETRIEVAL_FEATURE_LATENCY_MS
        + (0.0 if row["selected_action"] == "0.005" else ACTION_LATENCY_MS[row["selected_action"]])
        for row in token_rows
    ) / len(token_rows)
    if "mlp_action_router" in router:
        estimated_latency += MLP_ROUTER_TOKEN_LATENCY_MS / 32.0
    summary = {
        "tokens": len(token_rows),
        "nll": mean_nll,
        "ppl": math.exp(mean_nll),
        "full_nll": mean_full_nll,
        "full_ppl": math.exp(mean_full_nll),
        "ppl_ratio_vs_full": math.exp(mean_nll - mean_full_nll),
        "action_rates": {
            action: action_counts[action] / len(token_rows) for action in router["actions"]
        },
        "mean_final_attention_fraction": mean_fraction,
        "index_plus_per_query_link_lower_bound": 0.06640625 + mean_fraction,
        "estimated_attention_latency_ms_128k": estimated_latency,
        "estimated_attention_speedup_128k": FULL_LATENCY_MS / estimated_latency,
        "online_seconds_including_query_and_rescues": online_seconds,
    }
    if gqa_candidate_mode != "independent":
        summary["logical_pca_index_plus_shared_gqa_kv_fraction"] = (
            0.06640625 + mean_fraction
        )
    if collect_gqa_union_stats:
        mean_union_fraction = sum(
            float(row["candidate_union_fraction_mean"]) for row in token_rows
        ) / len(token_rows)
        summary.update(
            {
                "mean_gqa_kv_union_fraction": mean_union_fraction,
                "logical_pca_index_plus_gqa_kv_union_fraction": (
                    0.06640625 + mean_union_fraction
                ),
            }
        )
    return token_rows, summary


def main() -> None:
    args = parse_args()
    topics = topic_names(args.topics)
    windows = parse_int_list(args.window_indices)
    router = joblib.load(args.router_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2, default=str), encoding="utf-8"
    )
    install_llama_head_top_fraction_patch()
    tokenizer, model, input_device = load_model(args)
    required_tokens = max(windows) * args.window_stride_tokens + args.history_tokens + args.eval_tokens
    all_rows = []
    summaries = []
    for topic in topics:
        stream = encode_topic_stream(
            tokenizer, TOPICS[topic], required_tokens, args.dataset_cache_dir, args.seed
        )
        for window in windows:
            start = window * args.window_stride_tokens
            history = stream[start : start + args.history_tokens]
            target_ids = stream[
                start + args.history_tokens : start + args.history_tokens + args.eval_tokens
            ]
            remote_ids = history[: -args.query_tokens]
            query_ids = history[-args.query_tokens :]
            bundle, _ = make_bundle(tokenizer, remote_ids, page_tokens=16)
            set_attention_implementation(model, "sdpa")
            with head_top_fraction_mode(None), head_adaptive_mass_mode(None):
                prefix_cache, prefill_seconds = lb.prefill_prefix(
                    model, bundle, input_device, args.prefill_chunk_tokens
                )
            rows, summary = run_case(
                model,
                prefix_cache,
                len(remote_ids),
                query_ids,
                target_ids,
                router,
                input_device,
                full_reference_nlls(args.full_root, topic, window),
                args.projection_dim,
                args.gqa_candidate_mode,
                args.collect_gqa_union_stats,
            )
            summary.update(
                {"topic": topic, "window": window, "prefill_seconds": prefill_seconds}
            )
            for row in rows:
                row.update({"topic": topic, "window": window})
            all_rows.extend(rows)
            summaries.append(summary)
            write_csv(args.output_dir / "token_results.csv", all_rows)
            write_csv(args.output_dir / "summary.csv", summaries)
            print(json.dumps(summary, sort_keys=True), flush=True)
            del prefix_cache
            torch.cuda.empty_cache()
    (args.output_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
