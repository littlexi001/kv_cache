#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


METHODS = [
    "full_kv_cache",
    "naive_kv_gather_absolute_query_pos",
    "naive_kv_gather_compact_query_pos",
    "rope_delta_repack_compact_query_pos",
    "rope_delta_repack_shifted_query_pos",
    "prompt_rebuild_selected_pages",
    "variable_budget_kv_planner",
    "output_level_risk_kv_planner",
]

RULER_TASKS = [
    "niah_single_1",
    "niah_single_2",
    "niah_multiquery",
    "niah_multivalue",
    "niah_multikey_1",
    "vt",
    "cwe",
    "fwe",
]


def ruler16k_variable_paths(tag: str, case_ids: tuple[int, ...]) -> list[str]:
    return [
        f"outputs/variable_budget_runtime_qwen8b_ruler_16k_{task}_case{case_id}_{tag}_20260707/results.csv"
        for case_id in case_ids
        for task in RULER_TASKS
    ]


EXPERIMENTS = {
    "qwen8b_mixed13_m1": [
        "outputs/output_verifier_runtime_qwen8b_13tasks_m1_tau07_prefix_20260707/results.csv",
    ],
    "qwen8b_longbench_m2": [
        "outputs/output_verifier_runtime_qwen8b_longbench_m2_tau07_prefix_20260707/results.csv",
    ],
    "qwen8b_longbench_m4": [
        "outputs/output_verifier_runtime_qwen8b_longbench_m4_tau07_prefix_20260707/results.csv",
    ],
    "qwen8b_longbench_m4_floor2": [
        "outputs/output_verifier_runtime_qwen8b_longbench_m4_tau07_prefix_floor2_20260707/results.csv",
    ],
    "qwen8b_longbench_m4_floor3": [
        "outputs/output_verifier_runtime_qwen8b_longbench_m4_tau07_prefix_floor3_20260707/results.csv",
    ],
    "qwen8b_longbench_m4_var_bestcal_tail035": [
        "outputs/variable_budget_runtime_qwen8b_longbench_m4_bestcal_tail035_20260707/results.csv",
    ],
    "qwen8b_longbench_m4_var_minsafe_tail035": [
        "outputs/variable_budget_runtime_qwen8b_longbench_m4_minsafe_tail035_20260707/results.csv",
    ],
    "qwen8b_longbench_m4_conformal_auto": [
        "outputs/variable_budget_runtime_qwen8b_longbench_m4_conformal_auto_20260707/results.csv",
    ],
    "qwen8b_longbench_m8_var_bestcal_tail035": [
        "outputs/variable_budget_runtime_qwen8b_longbench_m8_bestcal_tail035_20260707/results.csv",
    ],
    "qwen8b_longbench_m8_var_minsafe_tail035": [
        "outputs/variable_budget_runtime_qwen8b_longbench_m8_minsafe_tail035_20260707/results.csv",
    ],
    "qwen8b_longbench_m8_conformal_auto": [
        "outputs/variable_budget_runtime_qwen8b_longbench_m8_conformal_auto_20260707/results.csv",
    ],
    "qwen8b_mixed13_m1_floor2": [
        "outputs/output_verifier_runtime_qwen8b_13tasks_m1_tau07_prefix_floor2_20260707/results.csv",
    ],
    "qwen8b_mixed13_m1_var_bestcal_tail035": [
        "outputs/variable_budget_runtime_qwen8b_mixed13_m1_bestcal_tail035_20260707/results.csv",
    ],
    "qwen8b_mixed13_m1_var_minsafe_tail035": [
        "outputs/variable_budget_runtime_qwen8b_mixed13_m1_minsafe_tail035_20260707/results.csv",
    ],
    "qwen8b_mixed13_m1_conformal_auto": [
        "outputs/variable_budget_runtime_qwen8b_mixed13_m1_conformal_auto_20260707/results.csv",
    ],
    "qwen8b_mixed13_m2_var_minsafe_tail035": [
        "outputs/variable_budget_runtime_qwen8b_mixed13_m2_minsafe_tail035_20260707/results.csv",
    ],
    "qwen8b_mixed13_m2_conformal_auto": [
        "outputs/variable_budget_runtime_qwen8b_mixed13_m2_conformal_auto_20260707/results.csv",
    ],
    "qwen8b_ruler4k_m1": [
        "outputs/output_verifier_runtime_qwen8b_ruler_4k_m1_tau07_prefix_20260707/results.csv",
    ],
    "qwen8b_ruler4k_m1_floor2": [
        "outputs/output_verifier_runtime_qwen8b_ruler_4k_m1_tau07_prefix_floor2_20260707/results.csv",
    ],
    "qwen8b_ruler4k_m3_floor2": [
        "outputs/output_verifier_runtime_qwen8b_ruler_4096_m3_tau07_prefix_floor2_20260707/results.csv",
    ],
    "qwen8b_ruler8k_m1_floor2": [
        "outputs/output_verifier_runtime_qwen8b_ruler_8k_m1_tau07_prefix_floor2_v2_20260707/results.csv",
    ],
    "qwen8b_ruler8k_m3_floor2": [
        "outputs/output_verifier_runtime_qwen8b_ruler_8192_m3_tau07_prefix_floor2_20260707/results.csv",
    ],
    "qwen8b_ruler4k_m3_var_bestcal_tail035": [
        "outputs/variable_budget_runtime_qwen8b_ruler4096_m3_bestcal_tail035_20260707/results.csv",
    ],
    "qwen8b_ruler8k_m3_var_bestcal_tail035": [
        "outputs/variable_budget_runtime_qwen8b_ruler8192_m3_bestcal_tail035_20260707/results.csv",
    ],
    "qwen8b_ruler4k_m3_conformal_auto": [
        "outputs/variable_budget_runtime_qwen8b_ruler4096_m3_conformal_auto_20260707/results.csv",
    ],
    "qwen8b_ruler8k_m3_conformal_auto": [
        "outputs/variable_budget_runtime_qwen8b_ruler8192_m3_conformal_auto_20260707/results.csv",
    ],
    "qwen8b_ruler4k_m5_var_bestcal_tail035": [
        "outputs/variable_budget_runtime_qwen8b_ruler4096_m5_bestcal_tail035_20260707/results.csv",
    ],
    "qwen8b_ruler8k_m5_var_bestcal_tail035": [
        "outputs/variable_budget_runtime_qwen8b_ruler8192_m5_bestcal_tail035_20260707/results.csv",
    ],
    "qwen8b_ruler4k_m5_var_minsafe_tail035": [
        "outputs/variable_budget_runtime_qwen8b_ruler4096_m5_minsafe_tail035_20260707/results.csv",
    ],
    "qwen8b_ruler8k_m5_var_minsafe_tail035": [
        "outputs/variable_budget_runtime_qwen8b_ruler8192_m5_minsafe_tail035_20260707/results.csv",
    ],
    "qwen8b_ruler4k_m5_conformal_auto": [
        "outputs/variable_budget_runtime_qwen8b_ruler4096_m5_conformal_auto_20260707/results.csv",
    ],
    "qwen8b_ruler8k_m5_conformal_auto": [
        "outputs/variable_budget_runtime_qwen8b_ruler8192_m5_conformal_auto_20260707/results.csv",
    ],
    "qwen8b_ruler4k_m5_conformal_floor2": [
        "outputs/variable_budget_runtime_qwen8b_ruler4096_m5_conformal_floor2_20260707/results.csv",
    ],
    "qwen8b_ruler8k_m5_conformal_floor2": [
        "outputs/variable_budget_runtime_qwen8b_ruler8192_m5_conformal_floor2_20260707/results.csv",
    ],
    "qwen8b_ruler8k_m1_var_bestcal_tail035": [
        "outputs/variable_budget_runtime_qwen8b_ruler8k_m1_bestcal_tail035_20260707/results.csv",
    ],
    "qwen8b_ruler8k_m1_conformal_auto": [
        "outputs/variable_budget_runtime_qwen8b_ruler8k_m1_conformal_auto_20260707/results.csv",
    ],
    "qwen8b_ruler16k_m1_floor2": [
        "outputs/output_verifier_runtime_qwen8b_ruler_16k_niah_single_1_tau07_prefix_floor2_20260707/results.csv",
        "outputs/output_verifier_runtime_qwen8b_ruler_16k_niah_single_2_tau07_prefix_floor2_20260707/results.csv",
        "outputs/output_verifier_runtime_qwen8b_ruler_16k_niah_multiquery_tau07_prefix_floor2_20260707/results.csv",
        "outputs/output_verifier_runtime_qwen8b_ruler_16k_niah_multivalue_tau07_prefix_floor2_20260707/results.csv",
        "outputs/output_verifier_runtime_qwen8b_ruler_16k_niah_multikey_1_tau07_prefix_floor2_20260707/results.csv",
        "outputs/output_verifier_runtime_qwen8b_ruler_16k_vt_tau07_prefix_floor2_20260707/results.csv",
        "outputs/output_verifier_runtime_qwen8b_ruler_16k_cwe_tau07_prefix_floor2_20260707/results.csv",
        "outputs/output_verifier_runtime_qwen8b_ruler_16k_fwe_tau07_prefix_floor2_20260707/results.csv",
    ],
    "qwen8b_ruler16k_m2_floor2_sharded": [
        "outputs/output_verifier_runtime_qwen8b_ruler_16k_niah_single_1_tau07_prefix_floor2_20260707/results.csv",
        "outputs/output_verifier_runtime_qwen8b_ruler_16k_niah_single_2_tau07_prefix_floor2_20260707/results.csv",
        "outputs/output_verifier_runtime_qwen8b_ruler_16k_niah_multiquery_tau07_prefix_floor2_20260707/results.csv",
        "outputs/output_verifier_runtime_qwen8b_ruler_16k_niah_multivalue_tau07_prefix_floor2_20260707/results.csv",
        "outputs/output_verifier_runtime_qwen8b_ruler_16k_niah_multikey_1_tau07_prefix_floor2_20260707/results.csv",
        "outputs/output_verifier_runtime_qwen8b_ruler_16k_vt_tau07_prefix_floor2_20260707/results.csv",
        "outputs/output_verifier_runtime_qwen8b_ruler_16k_cwe_tau07_prefix_floor2_20260707/results.csv",
        "outputs/output_verifier_runtime_qwen8b_ruler_16k_fwe_tau07_prefix_floor2_20260707/results.csv",
        "outputs/output_verifier_runtime_qwen8b_ruler_16k_niah_single_1_case1_tau07_prefix_floor2_20260707/results.csv",
        "outputs/output_verifier_runtime_qwen8b_ruler_16k_niah_single_2_case1_tau07_prefix_floor2_20260707/results.csv",
        "outputs/output_verifier_runtime_qwen8b_ruler_16k_niah_multiquery_case1_tau07_prefix_floor2_20260707/results.csv",
        "outputs/output_verifier_runtime_qwen8b_ruler_16k_niah_multivalue_case1_tau07_prefix_floor2_20260707/results.csv",
        "outputs/output_verifier_runtime_qwen8b_ruler_16k_niah_multikey_1_case1_tau07_prefix_floor2_20260707/results.csv",
        "outputs/output_verifier_runtime_qwen8b_ruler_16k_vt_case1_tau07_prefix_floor2_20260707/results.csv",
        "outputs/output_verifier_runtime_qwen8b_ruler_16k_cwe_case1_tau07_prefix_floor2_20260707/results.csv",
        "outputs/output_verifier_runtime_qwen8b_ruler_16k_fwe_case1_tau07_prefix_floor2_20260707/results.csv",
    ],
    "qwen8b_ruler16k_m2_var_bestcal_tail035_sharded": [
        "outputs/variable_budget_runtime_qwen8b_ruler_16k_niah_single_1_case0_bestcal_tail035_20260707/results.csv",
        "outputs/variable_budget_runtime_qwen8b_ruler_16k_niah_single_2_case0_bestcal_tail035_20260707/results.csv",
        "outputs/variable_budget_runtime_qwen8b_ruler_16k_niah_multiquery_case0_bestcal_tail035_20260707/results.csv",
        "outputs/variable_budget_runtime_qwen8b_ruler_16k_niah_multivalue_case0_bestcal_tail035_20260707/results.csv",
        "outputs/variable_budget_runtime_qwen8b_ruler_16k_niah_multikey_1_case0_bestcal_tail035_20260707/results.csv",
        "outputs/variable_budget_runtime_qwen8b_ruler_16k_vt_case0_bestcal_tail035_20260707/results.csv",
        "outputs/variable_budget_runtime_qwen8b_ruler_16k_cwe_case0_bestcal_tail035_20260707/results.csv",
        "outputs/variable_budget_runtime_qwen8b_ruler_16k_fwe_case0_bestcal_tail035_20260707/results.csv",
        "outputs/variable_budget_runtime_qwen8b_ruler_16k_niah_single_1_case1_bestcal_tail035_20260707/results.csv",
        "outputs/variable_budget_runtime_qwen8b_ruler_16k_niah_single_2_case1_bestcal_tail035_20260707/results.csv",
        "outputs/variable_budget_runtime_qwen8b_ruler_16k_niah_multiquery_case1_bestcal_tail035_20260707/results.csv",
        "outputs/variable_budget_runtime_qwen8b_ruler_16k_niah_multivalue_case1_bestcal_tail035_20260707/results.csv",
        "outputs/variable_budget_runtime_qwen8b_ruler_16k_niah_multikey_1_case1_bestcal_tail035_20260707/results.csv",
        "outputs/variable_budget_runtime_qwen8b_ruler_16k_vt_case1_bestcal_tail035_20260707/results.csv",
        "outputs/variable_budget_runtime_qwen8b_ruler_16k_cwe_case1_bestcal_tail035_20260707/results.csv",
        "outputs/variable_budget_runtime_qwen8b_ruler_16k_fwe_case1_bestcal_tail035_20260707/results.csv",
    ],
    "qwen8b_ruler16k_m2_conformal_auto_sharded": [
        "outputs/variable_budget_runtime_qwen8b_ruler_16k_niah_single_1_case0_conformal_auto_20260707/results.csv",
        "outputs/variable_budget_runtime_qwen8b_ruler_16k_niah_single_2_case0_conformal_auto_20260707/results.csv",
        "outputs/variable_budget_runtime_qwen8b_ruler_16k_niah_multiquery_case0_conformal_auto_20260707/results.csv",
        "outputs/variable_budget_runtime_qwen8b_ruler_16k_niah_multivalue_case0_conformal_auto_20260707/results.csv",
        "outputs/variable_budget_runtime_qwen8b_ruler_16k_niah_multikey_1_case0_conformal_auto_20260707/results.csv",
        "outputs/variable_budget_runtime_qwen8b_ruler_16k_vt_case0_conformal_auto_20260707/results.csv",
        "outputs/variable_budget_runtime_qwen8b_ruler_16k_cwe_case0_conformal_auto_20260707/results.csv",
        "outputs/variable_budget_runtime_qwen8b_ruler_16k_fwe_case0_conformal_auto_20260707/results.csv",
        "outputs/variable_budget_runtime_qwen8b_ruler_16k_niah_single_1_case1_conformal_auto_20260707/results.csv",
        "outputs/variable_budget_runtime_qwen8b_ruler_16k_niah_single_2_case1_conformal_auto_20260707/results.csv",
        "outputs/variable_budget_runtime_qwen8b_ruler_16k_niah_multiquery_case1_conformal_auto_20260707/results.csv",
        "outputs/variable_budget_runtime_qwen8b_ruler_16k_niah_multivalue_case1_conformal_auto_20260707/results.csv",
        "outputs/variable_budget_runtime_qwen8b_ruler_16k_niah_multikey_1_case1_conformal_auto_20260707/results.csv",
        "outputs/variable_budget_runtime_qwen8b_ruler_16k_vt_case1_conformal_auto_20260707/results.csv",
        "outputs/variable_budget_runtime_qwen8b_ruler_16k_cwe_case1_conformal_auto_20260707/results.csv",
        "outputs/variable_budget_runtime_qwen8b_ruler_16k_fwe_case1_conformal_auto_20260707/results.csv",
    ],
    "qwen8b_ruler16k_m3_var_bestcal_tail035_sharded": ruler16k_variable_paths("bestcal_tail035", (0, 1, 2)),
    "qwen8b_ruler16k_m3_conformal_auto_sharded": ruler16k_variable_paths("conformal_auto", (0, 1, 2)),
}


def to_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def load_rows(root: Path, paths: list[str]) -> tuple[list[dict[str, str]], list[str]]:
    rows: list[dict[str, str]] = []
    missing: list[str] = []
    for rel_path in paths:
        path = root / rel_path
        if not path.exists():
            missing.append(rel_path)
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    return rows, missing


def aggregate(name: str, rows: list[dict[str, str]]) -> list[dict[str, object]]:
    by_case: dict[tuple[str, str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        key = (row["benchmark"], row["task"], row["case_id"])
        by_case[key][row["method"]] = row

    output: list[dict[str, object]] = []
    for method in METHODS:
        selected: list[tuple[dict[str, str], dict[str, str]]] = []
        for case_rows in by_case.values():
            if "full_kv_cache" in case_rows and method in case_rows:
                selected.append((case_rows[method], case_rows["full_kv_cache"]))
        if not selected:
            continue

        samples = len(selected)
        sum_score = sum(to_float(row["score"]) for row, _ in selected)
        sum_exact = sum(1.0 if row["exact_correct"].lower() in {"1", "true"} else 0.0 for row, _ in selected)
        sum_active = sum(to_float(row["active_kv_tokens"]) for row, _ in selected)
        sum_context = sum(max(1.0, to_float(row["context_tokens"])) for row, _ in selected)
        sum_full_online = sum(to_float(full["total_online_seconds"]) for _, full in selected)
        sum_online = sum(to_float(row["total_online_seconds"]) for row, _ in selected)
        sum_full_e2e = sum(to_float(full["end_to_end_seconds"]) for _, full in selected)
        sum_e2e = sum(to_float(row["end_to_end_seconds"]) for row, _ in selected)
        sum_repack = sum(to_float(row["repack_seconds"]) for row, _ in selected)
        sum_planner = sum(to_float(row["planner_seconds"]) for row, _ in selected)
        sum_query = sum(to_float(row["query_seconds"]) for row, _ in selected)
        sum_decode = sum(to_float(row["decode_seconds"]) for row, _ in selected)

        output.append(
            {
                "experiment": name,
                "method": method,
                "samples": samples,
                "score_pct": 100.0 * sum_score / samples,
                "exact_pct": 100.0 * sum_exact / samples,
                "kv_ratio_pct": 100.0 * sum_active / sum_context,
                "active_kv_tokens": sum_active / samples,
                "online_speedup_sum": sum_full_online / max(sum_online, 1e-9),
                "e2e_speedup_sum": sum_full_e2e / max(sum_e2e, 1e-9),
                "avg_repack_ms": 1000.0 * sum_repack / samples,
                "avg_planner_ms": 1000.0 * sum_planner / samples,
                "avg_query_ms": 1000.0 * sum_query / samples,
                "avg_decode_ms": 1000.0 * sum_decode / samples,
            }
        )
    return output


def write_markdown(path: Path, rows: list[dict[str, object]], missing: dict[str, list[str]]) -> None:
    lines = [
        "# Runtime scaling summary 20260707",
        "",
        "| Experiment | Method | N | Score | KV | Online speed | E2E speed |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {experiment} | {method} | {samples} | {score_pct:.2f}% | {kv_ratio_pct:.2f}% | "
            "{online_speedup_sum:.3f}x | {e2e_speedup_sum:.3f}x |".format(**row)
        )
    if missing:
        lines += ["", "## Missing inputs", ""]
        for experiment, paths in missing.items():
            for rel_path in paths:
                lines.append(f"- {experiment}: `{rel_path}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/runtime_scaling_summary_20260707"))
    args = parser.parse_args()

    root = args.root.resolve()
    output_dir = (root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, object]] = []
    missing: dict[str, list[str]] = {}
    for name, paths in EXPERIMENTS.items():
        rows, missing_paths = load_rows(root, paths)
        if missing_paths:
            missing[name] = missing_paths
        all_rows.extend(aggregate(name, rows))

    csv_path = output_dir / "runtime_scaling_summary.csv"
    fieldnames = [
        "experiment",
        "method",
        "samples",
        "score_pct",
        "exact_pct",
        "kv_ratio_pct",
        "active_kv_tokens",
        "online_speedup_sum",
        "e2e_speedup_sum",
        "avg_repack_ms",
        "avg_planner_ms",
        "avg_query_ms",
        "avg_decode_ms",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    md_path = output_dir / "runtime_scaling_summary.md"
    write_markdown(md_path, all_rows, missing)
    (output_dir / "runtime_scaling_summary.json").write_text(
        json.dumps({"rows": all_rows, "missing": missing}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(md_path)


if __name__ == "__main__":
    main()
