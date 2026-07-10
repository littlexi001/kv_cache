#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


METHOD_LABELS = {
    "full_kv_cache": "Full KV",
    "variable_budget_kv_planner": "RiskKV",
    "output_level_risk_kv_planner": "RiskKV verifier",
    "naive_kv_gather_absolute_query_pos": "Naive gather + absolute query",
    "naive_kv_gather_compact_query_pos": "Naive gather + compact query",
    "rope_delta_repack_compact_query_pos": "RoPE repack + compact query",
    "rope_delta_repack_shifted_query_pos": "RoPE repack + shifted query",
    "prompt_rebuild_selected_pages": "Prompt rebuild",
}

MAIN_ROWS = [
    ("LongBench m8", "qwen8b_longbench_m8_conformal_auto", "full_kv_cache"),
    ("LongBench m8", "qwen8b_longbench_m8_conformal_auto", "variable_budget_kv_planner"),
    ("Mixed13 m2", "qwen8b_mixed13_m2_var_minsafe_tail035", "full_kv_cache"),
    ("Mixed13 m2", "qwen8b_mixed13_m2_var_minsafe_tail035", "variable_budget_kv_planner"),
    ("RULER 4k m5", "qwen8b_ruler4k_m5_conformal_floor2", "full_kv_cache"),
    ("RULER 4k m5", "qwen8b_ruler4k_m5_conformal_floor2", "variable_budget_kv_planner"),
    ("RULER 8k m5", "qwen8b_ruler8k_m5_conformal_floor2", "full_kv_cache"),
    ("RULER 8k m5", "qwen8b_ruler8k_m5_conformal_floor2", "variable_budget_kv_planner"),
    ("RULER 16k m3", "qwen8b_ruler16k_m3_conformal_auto_sharded", "full_kv_cache"),
    ("RULER 16k m3", "qwen8b_ruler16k_m3_conformal_auto_sharded", "variable_budget_kv_planner"),
]

FLOOR_ABLATION_ROWS = [
    ("4k m5 full", "qwen8b_ruler4k_m5_conformal_floor2", "full_kv_cache"),
    ("4k bestcal", "qwen8b_ruler4k_m5_var_bestcal_tail035", "variable_budget_kv_planner"),
    ("4k min-safe", "qwen8b_ruler4k_m5_var_minsafe_tail035", "variable_budget_kv_planner"),
    ("4k conformal", "qwen8b_ruler4k_m5_conformal_auto", "variable_budget_kv_planner"),
    ("4k conformal floor2", "qwen8b_ruler4k_m5_conformal_floor2", "variable_budget_kv_planner"),
    ("8k m5 full", "qwen8b_ruler8k_m5_conformal_floor2", "full_kv_cache"),
    ("8k bestcal", "qwen8b_ruler8k_m5_var_bestcal_tail035", "variable_budget_kv_planner"),
    ("8k min-safe", "qwen8b_ruler8k_m5_var_minsafe_tail035", "variable_budget_kv_planner"),
    ("8k conformal", "qwen8b_ruler8k_m5_conformal_auto", "variable_budget_kv_planner"),
    ("8k conformal floor2", "qwen8b_ruler8k_m5_conformal_floor2", "variable_budget_kv_planner"),
]

ROPE_ABLATION_ROWS = [
    ("Mixed13 m1", "qwen8b_mixed13_m1_floor2", "full_kv_cache"),
    ("Mixed13 m1", "qwen8b_mixed13_m1_floor2", "naive_kv_gather_absolute_query_pos"),
    ("Mixed13 m1", "qwen8b_mixed13_m1_floor2", "naive_kv_gather_compact_query_pos"),
    ("Mixed13 m1", "qwen8b_mixed13_m1_floor2", "rope_delta_repack_compact_query_pos"),
    ("Mixed13 m1", "qwen8b_mixed13_m1_floor2", "rope_delta_repack_shifted_query_pos"),
    ("Mixed13 m1", "qwen8b_mixed13_m1_floor2", "prompt_rebuild_selected_pages"),
    ("RULER 4k m3", "qwen8b_ruler4k_m3_floor2", "full_kv_cache"),
    ("RULER 4k m3", "qwen8b_ruler4k_m3_floor2", "naive_kv_gather_absolute_query_pos"),
    ("RULER 4k m3", "qwen8b_ruler4k_m3_floor2", "naive_kv_gather_compact_query_pos"),
    ("RULER 4k m3", "qwen8b_ruler4k_m3_floor2", "rope_delta_repack_compact_query_pos"),
    ("RULER 4k m3", "qwen8b_ruler4k_m3_floor2", "rope_delta_repack_shifted_query_pos"),
    ("RULER 4k m3", "qwen8b_ruler4k_m3_floor2", "prompt_rebuild_selected_pages"),
    ("RULER 8k m3", "qwen8b_ruler8k_m3_floor2", "full_kv_cache"),
    ("RULER 8k m3", "qwen8b_ruler8k_m3_floor2", "naive_kv_gather_absolute_query_pos"),
    ("RULER 8k m3", "qwen8b_ruler8k_m3_floor2", "naive_kv_gather_compact_query_pos"),
    ("RULER 8k m3", "qwen8b_ruler8k_m3_floor2", "rope_delta_repack_compact_query_pos"),
    ("RULER 8k m3", "qwen8b_ruler8k_m3_floor2", "rope_delta_repack_shifted_query_pos"),
    ("RULER 8k m3", "qwen8b_ruler8k_m3_floor2", "prompt_rebuild_selected_pages"),
    ("RULER 16k m2", "qwen8b_ruler16k_m2_floor2_sharded", "full_kv_cache"),
    ("RULER 16k m2", "qwen8b_ruler16k_m2_floor2_sharded", "naive_kv_gather_absolute_query_pos"),
    ("RULER 16k m2", "qwen8b_ruler16k_m2_floor2_sharded", "naive_kv_gather_compact_query_pos"),
    ("RULER 16k m2", "qwen8b_ruler16k_m2_floor2_sharded", "rope_delta_repack_compact_query_pos"),
    ("RULER 16k m2", "qwen8b_ruler16k_m2_floor2_sharded", "rope_delta_repack_shifted_query_pos"),
    ("RULER 16k m2", "qwen8b_ruler16k_m2_floor2_sharded", "prompt_rebuild_selected_pages"),
]


def load_rows(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return {(row["experiment"], row["method"]): row for row in rows}


def fnum(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return 0.0


def markdown_table(index: dict[tuple[str, str], dict[str, str]], spec: list[tuple[str, str, str]]) -> str:
    lines = [
        "| Setting | Method | N | Score | KV | Online | E2E |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for setting, experiment, method in spec:
        row = index.get((experiment, method))
        if row is None:
            lines.append(f"| {setting} | missing: `{experiment}/{method}` |  |  |  |  |  |")
            continue
        lines.append(
            "| {setting} | {method} | {samples} | {score:.2f}% | {kv:.2f}% | {online:.3f}x | {e2e:.3f}x |".format(
                setting=setting,
                method=METHOD_LABELS.get(method, method),
                samples=row["samples"],
                score=fnum(row, "score_pct"),
                kv=fnum(row, "kv_ratio_pct"),
                online=fnum(row, "online_speedup_sum"),
                e2e=fnum(row, "e2e_speedup_sum"),
            )
        )
    return "\n".join(lines) + "\n"


def latex_table(index: dict[tuple[str, str], dict[str, str]], spec: list[tuple[str, str, str]]) -> str:
    lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Setting & Method & N & Score & KV & Online & E2E \\",
        r"\midrule",
    ]
    for setting, experiment, method in spec:
        row = index.get((experiment, method))
        if row is None:
            continue
        lines.append(
            r"{setting} & {method} & {samples} & {score:.2f} & {kv:.2f} & {online:.3f}x & {e2e:.3f}x \\".format(
                setting=setting.replace("_", r"\_"),
                method=METHOD_LABELS.get(method, method).replace("_", r"\_"),
                samples=row["samples"],
                score=fnum(row, "score_pct"),
                kv=fnum(row, "kv_ratio_pct"),
                online=fnum(row, "online_speedup_sum"),
                e2e=fnum(row, "e2e_speedup_sum"),
            )
        )
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary_csv",
        type=Path,
        default=Path("outputs/runtime_scaling_summary_20260707/runtime_scaling_summary.csv"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/runtime_scaling_summary_20260707/icml_paper_tables"),
    )
    args = parser.parse_args()

    index = load_rows(args.summary_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "paper_main_table.md").write_text(markdown_table(index, MAIN_ROWS), encoding="utf-8")
    (args.output_dir / "paper_main_table.tex").write_text(latex_table(index, MAIN_ROWS), encoding="utf-8")
    (args.output_dir / "paper_floor_ablation.md").write_text(markdown_table(index, FLOOR_ABLATION_ROWS), encoding="utf-8")
    (args.output_dir / "paper_floor_ablation.tex").write_text(latex_table(index, FLOOR_ABLATION_ROWS), encoding="utf-8")
    (args.output_dir / "paper_rope_ablation.md").write_text(markdown_table(index, ROPE_ABLATION_ROWS), encoding="utf-8")
    (args.output_dir / "paper_rope_ablation.tex").write_text(latex_table(index, ROPE_ABLATION_ROWS), encoding="utf-8")
    print(args.output_dir)


if __name__ == "__main__":
    main()
