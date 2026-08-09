from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable


Validator = Callable[[Any], str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the queued ICLR evidence artifacts.")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def overall_validator(method_samples: dict[str, int]) -> Validator:
    def validate(payload: dict[str, Any]) -> str:
        observed = {
            str(row["method"]): int(row["samples"])
            for row in payload["overall"]
        }
        if observed != method_samples:
            raise ValueError(f"overall methods/samples {observed} != {method_samples}")
        return f"methods={observed}"

    return validate


def ruler_validator(samples: int, gated: bool) -> Validator:
    def validate(payload: dict[str, Any]) -> str:
        observed = {
            str(row["method"]): int(row["samples"])
            for row in payload["overall"]
        }
        expected = {"full_kv": samples, "hierarchical_pca_perhead": samples}
        if observed != expected:
            raise ValueError(f"RULER methods/samples {observed} != {expected}")
        if gated:
            gated_rows = payload.get("gated_overall", [])
            if len(gated_rows) != 1 or int(gated_rows[0]["samples"]) != samples:
                raise ValueError("missing complete RULER length-gate result")
        return f"paired_samples={samples}, gated={gated}"

    return validate


def rows_validator(expected: int) -> Validator:
    def validate(payload: Any) -> str:
        rows = payload["rows"] if isinstance(payload, dict) else payload
        if len(rows) != expected:
            raise ValueError(f"expected {expected} rows, found {len(rows)}")
        return f"rows={len(rows)}"

    return validate


def field_validator(**expected: Any) -> Validator:
    def validate(payload: dict[str, Any]) -> str:
        for key, value in expected.items():
            if payload.get(key) != value:
                raise ValueError(f"{key}={payload.get(key)!r}, expected {value!r}")
        return ", ".join(f"{key}={value}" for key, value in expected.items())

    return validate


def audit(root: Path) -> list[dict[str, str]]:
    artifacts: list[tuple[str, str, Validator]] = [
        (
            "longbench_full_16task",
            "outputs/20260716_hierarchical_longbench_full_v1_merged/summary.json",
            overall_validator({"full_kv": 3750, "hierarchical_pca_perhead": 3750}),
        ),
        (
            "longbench_fullprompt_and_gate",
            "outputs/20260716_longbench_fullprompt_speed_m10_merged/summary.json",
            overall_validator(
                {
                    "auto_length_gate": 160,
                    "full_kv": 160,
                    "hierarchical_pca_perhead": 160,
                }
            ),
        ),
        (
            "temporal_refresh_i2",
            "outputs/20260716_longbench_temporal_refresh_i2_m20_merged/summary.json",
            overall_validator({"full_kv": 80, "hierarchical_pca_perhead": 80}),
        ),
        (
            "temporal_refresh_i4",
            "outputs/20260716_longbench_temporal_refresh_i4_m20_merged/summary.json",
            overall_validator({"full_kv": 80, "hierarchical_pca_perhead": 80}),
        ),
        (
            "speed_pareto",
            "results/20260716_128k_speed_pareto_summary/summary.json",
            rows_validator(10),
        ),
        (
            "attention_host_paths",
            "outputs/20260716_128k_attention_bottleneck/summary/summary.json",
            rows_validator(6),
        ),
        (
            "candidate_overlap_trace",
            "results/20260716_128k_candidate_overlap_trace/summary.json",
            field_validator(cases=2),
        ),
        (
            "ruler_4k_32k",
            "outputs/20260716_hierarchical_ruler_4k32k_m10_merged/summary.json",
            ruler_validator(360, True),
        ),
        (
            "ruler_64k_128k",
            "outputs/20260716_hierarchical_ruler_64k128k_m5_merged/summary.json",
            ruler_validator(90, True),
        ),
        (
            "longicl_calibration",
            "outputs/20260716_longicl_physical_calibration_m14_merged/summary.json",
            overall_validator({"full_kv": 14, "hierarchical_pca_perhead": 14}),
        ),
        (
            "matched_ablation",
            "results/20260716_32k_matched_ablation_m64_summary/summary.json",
            rows_validator(21),
        ),
        (
            "multitopic_windows",
            "results/20260716_128k_multitopic_windows_w3_summary/summary.json",
            field_validator(cases=18, target_tokens_per_case=[256]),
        ),
        (
            "low_peak",
            "results/20260716_128k_low_peak_ablation/summary/summary.json",
            rows_validator(3),
        ),
        (
            "strict_router",
            "results/20260716_router_128k_split_summary/summary.json",
            field_validator(cases=6, ablation_cases=12),
        ),
        (
            "long_generation",
            "results/20260716_128k_long_generation_religion_w3_m2048/summary.json",
            field_validator(),
        ),
        (
            "numa_pcie",
            "results/20260716_128k_numa_ablation/summary.json",
            field_validator(),
        ),
        (
            "batch_throughput",
            "results/20260716_batch_throughput/summary/summary.json",
            rows_validator(6),
        ),
        (
            "qwen_longbench",
            "outputs/20260716_qwen3_4b_longbench_m10_merged/summary.json",
            overall_validator(
                {
                    "auto_length_gate": 40,
                    "full_kv": 40,
                    "hierarchical_pca_perhead": 40,
                }
            ),
        ),
        (
            "qwen_ruler",
            "outputs/20260716_qwen3_4b_ruler_m5_merged/summary.json",
            ruler_validator(60, True),
        ),
        (
            "qwen_physical_full",
            "results/20260716_qwen3_4b_physical_32k_full.json",
            field_validator(eval_tokens=256),
        ),
        (
            "qwen_physical_sparse",
            "results/20260716_qwen3_4b_physical_32k_sparse.json",
            field_validator(eval_tokens=256),
        ),
    ]
    results: list[dict[str, str]] = []
    failures: list[str] = []
    for name, relative, validator in artifacts:
        path = root / relative
        try:
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            detail = validator(payload)
            results.append({"name": name, "status": "complete", "path": relative, "detail": detail})
        except Exception as error:  # The audit must report every missing artifact at once.
            failures.append(f"{name}: {error}")
            results.append({"name": name, "status": "failed", "path": relative, "detail": str(error)})
    if failures:
        raise RuntimeError("experiment completion audit failed:\n" + "\n".join(failures))
    return results


def main() -> None:
    args = parse_args()
    report = {"status": "complete", "checks": audit(args.root)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
