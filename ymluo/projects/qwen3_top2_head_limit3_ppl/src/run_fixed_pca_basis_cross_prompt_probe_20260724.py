from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

import run_controlled_public_kv_benchmark_v1 as lb
import run_sample_calibrated_longbench_20260717 as runner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-prompt fixed Key-PCA basis quality and timing probe."
    )
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--longbench_data_dir", required=True, type=Path)
    parser.add_argument("--output_path", required=True, type=Path)
    parser.add_argument("--task", default="gov_report")
    parser.add_argument("--calibration_task", default="gov_report")
    parser.add_argument("--calibration_offset", type=int, default=114)
    parser.add_argument(
        "--calibration_specs",
        default="",
        help="Optional comma-separated task:offset list for a shared PCA basis.",
    )
    parser.add_argument("--calibration_max_new_tokens", type=int, default=1)
    parser.add_argument("--test_offset", type=int, default=115)
    parser.add_argument("--max_prompt_tokens", type=int, default=16000)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--prefill_chunk_tokens", type=int, default=2048)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--max_memory_per_gpu_gib", type=float, default=0.0)
    parser.add_argument(
        "--score_mode",
        default=runner.COUNTCAP_KEYPCA_DIRECT_QKV_FUSED_QPROJSCAN_SCORE_MODE,
    )
    return parser.parse_args()


def runner_args(
    args: argparse.Namespace,
    task: str,
    offset: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        model_name_or_path=args.model_name_or_path,
        longbench_data_dir=args.longbench_data_dir,
        tasks=task,
        sample_offset_per_task=offset,
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
        collect_attention_stats=False,
        dtype=args.dtype,
        device=args.device,
        device_map=args.device_map,
        max_memory_per_gpu_gib=args.max_memory_per_gpu_gib,
        score_mode=args.score_mode,
    )


def run_sparse(
    model: torch.nn.Module,
    tokenizer: Any,
    input_device: torch.device,
    example: lb.Example,
    bundle: lb.PromptBundle,
    args: SimpleNamespace,
    *,
    fixed_templates: dict[int, dict[str, torch.Tensor]] | None = None,
    capture_templates: dict[int, dict[str, torch.Tensor]] | None = None,
    generation_tokens: int | None = None,
) -> dict[str, Any]:
    config = runner.countcap_config(
        bundle.query_start,
        score_mode=args.score_mode,
    )
    direct_fraction = float(config["candidate_fraction"])
    result = runner.generate_global_partition(
        model,
        tokenizer,
        input_device,
        bundle,
        (
            example.max_new_tokens
            if generation_tokens is None
            else generation_tokens
        ),
        args.prefill_chunk_tokens,
        (direct_fraction,),
        args,
        config["score_mode"],
        candidate_fraction=direct_fraction,
        projection_dim=config["projection_dim"],
        dense_suffix=True,
        incremental_prefill_index=True,
        fixed_pca_basis_templates=fixed_templates,
        captured_pca_basis_templates=capture_templates,
        eos_token_ids=runner.longbench_stop_token_ids(tokenizer, example.task),
    )
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


def parse_calibration_specs(
    args: argparse.Namespace,
) -> list[tuple[str, int]]:
    if not args.calibration_specs.strip():
        return [(args.calibration_task, args.calibration_offset)]
    specs: list[tuple[str, int]] = []
    for item in args.calibration_specs.split(","):
        task, separator, offset_text = item.strip().partition(":")
        if not task or not separator:
            raise ValueError(
                "calibration_specs must contain comma-separated task:offset values"
            )
        specs.append((task, int(offset_text)))
    return specs


def combine_pca_templates(
    template_sets: list[dict[int, dict[str, torch.Tensor]]],
) -> dict[int, dict[str, torch.Tensor]]:
    """Average rank-r key second moments, then recover one shared top-r basis."""
    if not template_sets:
        raise ValueError("at least one PCA template set is required")
    common_layers = set(template_sets[0])
    for templates in template_sets[1:]:
        common_layers.intersection_update(templates)
    if not common_layers:
        raise RuntimeError("calibration prompts produced no common PCA layers")

    combined: dict[int, dict[str, torch.Tensor]] = {}
    for layer_index in sorted(common_layers):
        layer_templates = [
            templates[layer_index] for templates in template_sets
        ]
        reference_basis = layer_templates[0]["basis"]
        projection_dim = int(reference_basis.shape[-1])
        second_moment = None
        for template in layer_templates:
            basis = template["basis"].float()
            weights = template["spectral_weights"].float()
            reconstructed = torch.einsum(
                "bhdr,bhr,bher->bhde",
                basis,
                weights,
                basis,
            )
            second_moment = (
                reconstructed
                if second_moment is None
                else second_moment + reconstructed
            )
        if second_moment is None:
            raise RuntimeError("failed to reconstruct PCA second moments")
        second_moment.div_(len(layer_templates))
        eigenvalues, eigenvectors = torch.linalg.eigh(second_moment)
        combined[layer_index] = {
            "basis": eigenvectors[..., -projection_dim:].to(
                reference_basis.dtype
            ),
            "spectral_weights": eigenvalues[
                ..., -projection_dim:
            ].clamp_min(1.0e-12),
        }
    return combined


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.repeats <= 0:
        raise ValueError("repeats must be positive")
    calibration_specs = parse_calibration_specs(args)
    load_args = runner_args(args, *calibration_specs[0])
    runner.install_llama_head_top_fraction_patch()
    tokenizer, model, input_device = runner.load_model(load_args)

    calibration_records: list[dict[str, Any]] = []
    template_sets: list[dict[int, dict[str, torch.Tensor]]] = []
    for calibration_task, calibration_offset in calibration_specs:
        calibration_args = runner_args(
            args,
            calibration_task,
            calibration_offset,
        )
        calibration_example = runner.load_examples(calibration_args)[0]
        calibration_bundle = runner.build_bundle(
            tokenizer,
            calibration_example,
            calibration_args,
        )
        captured_templates: dict[int, dict[str, torch.Tensor]] = {}
        calibration_result = run_sparse(
            model,
            tokenizer,
            input_device,
            calibration_example,
            calibration_bundle,
            calibration_args,
            capture_templates=captured_templates,
            generation_tokens=args.calibration_max_new_tokens,
        )
        if not captured_templates:
            raise RuntimeError("calibration produced no PCA basis templates")
        template_sets.append(captured_templates)
        calibration_records.append(
            {
                "task": calibration_task,
                "offset": calibration_offset,
                "sample_id": calibration_example.sample_id,
                "score": calibration_result["score"],
                "template_layers": len(captured_templates),
            }
        )
        runner.empty_cuda_caches()
    templates = (
        template_sets[0]
        if len(template_sets) == 1
        else combine_pca_templates(template_sets)
    )

    test_args = runner_args(args, args.task, args.test_offset)
    test_example = runner.load_examples(test_args)[0]
    test_bundle = runner.build_bundle(tokenizer, test_example, test_args)
    rows: list[dict[str, Any]] = []
    for repeat in range(args.repeats):
        order = ("adaptive", "fixed") if repeat % 2 == 0 else (
            "fixed",
            "adaptive",
        )
        for method in order:
            result = run_sparse(
                model,
                tokenizer,
                input_device,
                test_example,
                test_bundle,
                test_args,
                fixed_templates=templates if method == "fixed" else None,
            )
            rows.append(
                {
                    "repeat": repeat + 1,
                    "method": method,
                    "score": result["score"],
                    "prediction": result["prediction"],
                    "generated_ids": result["generated_ids"],
                    "prefill_seconds": result["prefill_seconds"],
                    "index_build_seconds": result["index_build_seconds"],
                    "online_seconds": result["online_seconds"],
                    "total_seconds": result["total_seconds"],
                }
            )
            runner.empty_cuda_caches()

    summary: dict[str, Any] = {
        "task": args.task,
        "calibration_specs": calibration_records,
        "basis_mode": (
            "single_prompt"
            if len(template_sets) == 1
            else "multiprompt_rank_moment"
        ),
        "max_prompt_tokens": args.max_prompt_tokens,
        "score_mode": args.score_mode,
        "test_sample_id": test_example.sample_id,
        "template_layers": len(templates),
        "rows": rows,
    }
    for method in ("adaptive", "fixed"):
        method_rows = [row for row in rows if row["method"] == method]
        summary[method] = {
            "score": statistics.mean(row["score"] for row in method_rows),
            "prefill_seconds_median": statistics.median(
                row["prefill_seconds"] for row in method_rows
            ),
            "index_build_seconds_median": statistics.median(
                row["index_build_seconds"] for row in method_rows
            ),
            "online_seconds_median": statistics.median(
                row["online_seconds"] for row in method_rows
            ),
            "total_seconds_median": statistics.median(
                row["total_seconds"] for row in method_rows
            ),
            "prediction_agreement": len(
                {tuple(row["generated_ids"]) for row in method_rows}
            )
            == 1,
        }
    summary["fixed_vs_adaptive"] = {
        metric: summary["adaptive"][metric] / summary["fixed"][metric]
        for metric in (
            "prefill_seconds_median",
            "index_build_seconds_median",
            "online_seconds_median",
            "total_seconds_median",
        )
    }
    summary["cross_method_prediction_match"] = (
        rows[0]["generated_ids"]
        == next(row for row in rows if row["method"] != rows[0]["method"])[
            "generated_ids"
        ]
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
