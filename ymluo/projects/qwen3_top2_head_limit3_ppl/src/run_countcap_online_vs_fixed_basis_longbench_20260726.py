from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

import run_controlled_public_kv_benchmark_v1 as lb
import run_sample_calibrated_longbench_20260717 as runner
from run_fixed_pca_basis_cross_prompt_probe_20260724 import (
    combine_pca_templates,
)


FINAL_SCORE_MODE = (
    "pca_int4_chunked_logscale16_sampleq_direct_qkvfused_"
    "qprojscan_qkvsplitauto"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Matched online versus cross-prompt fixed PCA LongBench."
    )
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--longbench_data_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--max_samples_per_task", type=int, default=20)
    parser.add_argument("--sample_offset_per_task", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--max_context_tokens", type=int, default=7500)
    parser.add_argument("--max_prompt_tokens", type=int, default=0)
    parser.add_argument("--max_new_tokens_override", type=int, default=0)
    parser.add_argument("--prefill_chunk_tokens", type=int, default=2048)
    parser.add_argument(
        "--prompt_wrapper",
        choices=("llama3", "qwen3", "none"),
        required=True,
    )
    parser.add_argument(
        "--calibration_specs",
        default=(
            "gov_report:150,narrativeqa:150,"
            "qasper:150,repobench-p:150"
        ),
    )
    parser.add_argument("--calibration_max_new_tokens", type=int, default=2)
    parser.add_argument("--mass_threshold", type=float, default=0.75)
    parser.add_argument("--sample_fraction", type=float, default=0.0025)
    parser.add_argument("--candidate_fraction", type=float, default=0.06)
    parser.add_argument("--projection_dim", type=int, default=48)
    parser.add_argument("--partition_ucb_z", type=float, default=0.0)
    parser.add_argument("--partition_overfetch_factor", type=int, default=2)
    parser.add_argument("--value_mass_threshold", type=float, default=1.0)
    parser.add_argument("--collect_attention_stats", action="store_true")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--max_memory_per_gpu_gib", type=float, default=0.0)
    return parser.parse_args()


def clone_args(args: argparse.Namespace, **updates: Any) -> SimpleNamespace:
    values = vars(args).copy()
    values.update(updates)
    return SimpleNamespace(**values)


def parse_calibration_specs(specification: str) -> list[tuple[str, int]]:
    output = []
    for item in specification.split(","):
        task, separator, offset = item.strip().partition(":")
        if not task or not separator:
            raise ValueError("calibration specs must use task:offset")
        output.append((task, int(offset)))
    if not output:
        raise ValueError("at least one calibration prompt is required")
    return output


def run_direct(
    model: torch.nn.Module,
    tokenizer: Any,
    input_device: torch.device,
    example: lb.Example,
    bundle: lb.PromptBundle,
    args: argparse.Namespace,
    *,
    fixed_templates: dict[int, dict[str, torch.Tensor]] | None = None,
    capture_templates: dict[int, dict[str, torch.Tensor]] | None = None,
    max_new_tokens: int | None = None,
) -> dict[str, Any]:
    attention_tokens, fraction = runner.countcap_direct_budget(
        bundle.query_start
    )
    generation_limit = (
        example.max_new_tokens
        if max_new_tokens is None
        else max_new_tokens
    )
    result = runner.generate_global_partition(
        model,
        tokenizer,
        input_device,
        bundle,
        generation_limit,
        args.prefill_chunk_tokens,
        (fraction,),
        args,
        FINAL_SCORE_MODE,
        candidate_fraction=fraction,
        projection_dim=48,
        dense_suffix=True,
        incremental_prefill_index=True,
        fixed_pca_basis_templates=fixed_templates,
        captured_pca_basis_templates=capture_templates,
        eos_token_ids=runner.longbench_stop_token_ids(
            tokenizer,
            example.task,
        ),
        use_preallocated_cache=(
            int(bundle.input_ids.shape[-1])
            >= runner.COUNTCAP_CACHE_AUTO_DEFAULT_MIN_TOKENS
        ),
    )
    result["configured_attention_tokens"] = attention_tokens
    result["configured_attention_fraction"] = fraction
    result["score"] = lb.score_prediction(
        example.metric,
        result["prediction"],
        example.answers,
        example.all_classes,
        task=example.task,
    )
    result["online_seconds"] = (
        result["query_seconds"] + result["decode_seconds"]
    )
    result["total_seconds"] = (
        result["prefill_seconds"] + result["online_seconds"]
    )
    return result


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.calibration_max_new_tokens < 2:
        raise ValueError("calibration needs at least two generated tokens")
    if args.projection_dim != 48:
        raise ValueError("the frozen method requires projection_dim=48")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runner.install_llama_head_top_fraction_patch()
    tokenizer, model, input_device = runner.load_model(args)

    calibration_records = []
    template_sets = []
    for task, offset in parse_calibration_specs(args.calibration_specs):
        calibration_args = clone_args(
            args,
            tasks=task,
            sample_offset_per_task=offset,
            max_samples_per_task=1,
            num_shards=1,
            shard_index=0,
        )
        example = runner.load_examples(calibration_args)[0]
        bundle = runner.build_bundle(tokenizer, example, calibration_args)
        templates: dict[int, dict[str, torch.Tensor]] = {}
        result = run_direct(
            model,
            tokenizer,
            input_device,
            example,
            bundle,
            calibration_args,
            capture_templates=templates,
            max_new_tokens=args.calibration_max_new_tokens,
        )
        if not templates:
            raise RuntimeError(f"calibration {task}:{offset} produced no basis")
        template_sets.append(templates)
        calibration_records.append(
            {
                "task": task,
                "offset": offset,
                "sample_id": example.sample_id,
                "prompt_tokens": int(bundle.input_ids.shape[-1]),
                "template_layers": len(templates),
                "score": result["score"],
            }
        )
        runner.empty_cuda_caches()
    fixed_templates = combine_pca_templates(template_sets)
    (args.output_dir / "calibration.json").write_text(
        json.dumps(calibration_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    output_path = args.output_dir / "sample_results.csv"
    existing = runner.read_csv(output_path)
    completed = {
        (row["task"], row["sample_id"], row["method"])
        for row in existing
    }
    examples = runner.load_examples(args)
    for index, example in enumerate(examples):
        bundle = runner.build_bundle(tokenizer, example, args)
        order = ("online", "fixed") if index % 2 == 0 else ("fixed", "online")
        for method in order:
            key = (example.task, example.sample_id, method)
            if key in completed:
                continue
            result = run_direct(
                model,
                tokenizer,
                input_device,
                example,
                bundle,
                args,
                fixed_templates=(
                    fixed_templates if method == "fixed" else None
                ),
            )
            row = {
                "task": example.task,
                "sample_id": example.sample_id,
                "method": method,
                "metric": example.metric,
                "score": result["score"],
                "prediction": result["prediction"],
                "prompt_tokens": int(bundle.input_ids.shape[-1]),
                "prefix_tokens": bundle.query_start,
                "suffix_tokens": bundle.suffix_token_count,
                "generated_tokens": len(result["generated_ids"]),
                "configured_attention_tokens": result[
                    "configured_attention_tokens"
                ],
                "configured_attention_fraction": result[
                    "configured_attention_fraction"
                ],
                "prefill_seconds": result["prefill_seconds"],
                "index_build_seconds": result["index_build_seconds"],
                "online_seconds": result["online_seconds"],
                "total_seconds": result["total_seconds"],
            }
            runner.append_csv_row(output_path, row)
            completed.add(key)
            print(
                f"[{index + 1}/{len(examples)}] {example.task} "
                f"{example.sample_id} {method} score={result['score']:.6f}",
                flush=True,
            )
            runner.empty_cuda_caches()


if __name__ == "__main__":
    main()
