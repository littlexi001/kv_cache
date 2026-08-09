from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

import run_sample_calibrated_longbench_20260717 as runner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect per-layer CountCap temporal-reuse quality labels."
    )
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--longbench_data_dir", required=True, type=Path)
    parser.add_argument("--output_path", required=True, type=Path)
    parser.add_argument("--task", default="gov_report")
    parser.add_argument("--sample_offset", type=int, default=115)
    parser.add_argument("--max_prompt_tokens", type=int, default=16000)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--prefill_chunk_tokens", type=int, default=2048)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--max_memory_per_gpu_gib", type=float, default=0.0)
    parser.add_argument(
        "--raw_trace_path",
        type=Path,
        default=None,
        help="Optional torch file containing per-step, per-layer temporal tensors.",
    )
    return parser.parse_args()


def make_runner_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        model_name_or_path=args.model_name_or_path,
        longbench_data_dir=args.longbench_data_dir,
        tasks=args.task,
        sample_offset_per_task=args.sample_offset,
        max_samples_per_task=1,
        num_shards=1,
        shard_index=0,
        max_new_tokens_override=args.max_new_tokens,
        max_prompt_tokens=args.max_prompt_tokens,
        max_context_tokens=0,
        prompt_wrapper="llama3",
        prefill_chunk_tokens=args.prefill_chunk_tokens,
        mass_threshold=0.75,
        sample_fraction=0.0025,
        candidate_fraction=0.08,
        projection_dim=64,
        value_mass_threshold=1.0,
        partition_ucb_z=0.0,
        partition_overfetch_factor=2,
        collect_attention_stats=True,
        dtype=args.dtype,
        device=args.device,
        device_map=args.device_map,
        max_memory_per_gpu_gib=args.max_memory_per_gpu_gib,
    )


def tensor_mean(
    records: list[dict[str, Any]],
    name: str,
) -> float:
    values = [
        torch.as_tensor(record[name]).float().reshape(-1)
        for record in records
        if name in record
    ]
    if not values:
        return 0.0
    return float(torch.cat(values).mean().item())


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    run_args = make_runner_args(args)
    runner.install_llama_head_top_fraction_patch()
    tokenizer, model, input_device = runner.load_model(run_args)
    example = runner.load_examples(run_args)[0]
    bundle = runner.build_bundle(tokenizer, example, run_args)
    config = runner.countcap_config(
        bundle.query_start,
        score_mode=(
            runner.COUNTCAP_KEYPCA_DIRECT_QKV_FUSED_QPROJSCAN_SCORE_MODE
        ),
    )
    direct_fraction = float(config["candidate_fraction"])
    records: list[dict[str, Any]] = []
    result = runner.generate_global_partition(
        model,
        tokenizer,
        input_device,
        bundle,
        example.max_new_tokens,
        args.prefill_chunk_tokens,
        (direct_fraction,),
        run_args,
        config["score_mode"],
        candidate_fraction=direct_fraction,
        projection_dim=config["projection_dim"],
        dense_suffix=True,
        incremental_prefill_index=True,
        attention_record_sink=records,
        eos_token_ids=runner.longbench_stop_token_ids(tokenizer, example.task),
    )

    layer_rows: list[dict[str, Any]] = []
    for layer in sorted({int(record["layer"]) for record in records}):
        layer_records = [
            record
            for record in records
            if int(record["layer"]) == layer
            and tensor_mean([record], "temporal_reuse_output_trace_available")
            > 0.0
        ]
        layer_rows.append(
            {
                "layer": layer,
                "steps": len(layer_records),
                "candidate_jaccard": tensor_mean(
                    layer_records,
                    "temporal_candidate_jaccard",
                ),
                "candidate_recall": tensor_mean(
                    layer_records,
                    "temporal_candidate_recall_from_previous",
                ),
                "candidate_recall_with_new_tokens": tensor_mean(
                    layer_records,
                    "temporal_candidate_recall_from_previous_with_new_tokens",
                ),
                "query_delta_norm": tensor_mean(
                    layer_records,
                    "temporal_query_delta_norm",
                ),
                "key_norm_bound": tensor_mean(
                    layer_records,
                    "temporal_key_norm_bound",
                ),
                "score_change_bound": tensor_mean(
                    layer_records,
                    "temporal_score_change_bound",
                ),
                "boundary_margin": tensor_mean(
                    layer_records,
                    "temporal_boundary_margin",
                ),
                "threshold_slack": tensor_mean(
                    layer_records,
                    "temporal_threshold_slack",
                ),
                "certificate_head_rate": tensor_mean(
                    layer_records,
                    "temporal_certificate_safe",
                ),
                "certificate_layer_rate": tensor_mean(
                    layer_records,
                    "temporal_certificate_layer_safe",
                ),
                "certificate_margin_ratio": tensor_mean(
                    layer_records,
                    "temporal_certificate_margin_ratio",
                ),
                "core_recall_top25": tensor_mean(
                    layer_records,
                    "temporal_core_recall_from_previous_top25",
                ),
                "fresh_attention_mass": tensor_mean(
                    layer_records,
                    "temporal_reuse_fresh_attention_mass",
                ),
                "output_relative_error": tensor_mean(
                    layer_records,
                    "temporal_reuse_output_relative_error",
                ),
                "output_cosine": tensor_mean(
                    layer_records,
                    "temporal_reuse_output_cosine",
                ),
                "gqa_union_candidate_fraction": tensor_mean(
                    layer_records,
                    "temporal_gqa_union_candidate_fraction",
                ),
                "gqa_union_fresh_attention_mass": tensor_mean(
                    layer_records,
                    "temporal_gqa_union_fresh_attention_mass",
                ),
                "gqa_union_output_relative_error": tensor_mean(
                    layer_records,
                    "temporal_gqa_union_output_relative_error",
                ),
            }
        )
    layer_rows.sort(key=lambda row: row["output_relative_error"])
    output = {
        "task": args.task,
        "sample_id": example.sample_id,
        "prompt_tokens": int(bundle.input_ids.shape[-1]),
        "prediction": result["prediction"],
        "online_seconds": result["query_seconds"] + result["decode_seconds"],
        "global_temporal_metrics": {
            "candidate_jaccard": tensor_mean(
                records,
                "temporal_candidate_jaccard",
            ),
            "candidate_recall": tensor_mean(
                records,
                "temporal_candidate_recall_from_previous",
            ),
            "candidate_recall_with_new_tokens": tensor_mean(
                records,
                "temporal_candidate_recall_from_previous_with_new_tokens",
            ),
            "query_delta_norm": tensor_mean(
                records,
                "temporal_query_delta_norm",
            ),
            "score_change_bound": tensor_mean(
                records,
                "temporal_score_change_bound",
            ),
            "boundary_margin": tensor_mean(
                records,
                "temporal_boundary_margin",
            ),
            "threshold_slack": tensor_mean(
                records,
                "temporal_threshold_slack",
            ),
            "certificate_head_rate": tensor_mean(
                records,
                "temporal_certificate_safe",
            ),
            "certificate_layer_rate": tensor_mean(
                records,
                "temporal_certificate_layer_safe",
            ),
        },
        "layers_by_reuse_error": layer_rows,
        "stable_layers_error_le_2pct": [
            row["layer"]
            for row in layer_rows
            if row["output_relative_error"] <= 0.02
        ],
        "stable_layers_error_le_5pct": [
            row["layer"]
            for row in layer_rows
            if row["output_relative_error"] <= 0.05
        ],
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if args.raw_trace_path is not None:
        args.raw_trace_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(records, args.raw_trace_path)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
