#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


MAIN_EXPERIMENTS = [
    "qwen8b_longbench_m4",
    "qwen8b_longbench_m4_floor2",
    "qwen8b_longbench_m4_floor3",
    "qwen8b_longbench_m4_var_bestcal_tail035",
    "qwen8b_longbench_m4_var_minsafe_tail035",
    "qwen8b_longbench_m4_conformal_auto",
    "qwen8b_longbench_m8_var_bestcal_tail035",
    "qwen8b_longbench_m8_var_minsafe_tail035",
    "qwen8b_longbench_m8_conformal_auto",
    "qwen8b_mixed13_m1",
    "qwen8b_mixed13_m1_floor2",
    "qwen8b_mixed13_m1_var_bestcal_tail035",
    "qwen8b_mixed13_m1_var_minsafe_tail035",
    "qwen8b_mixed13_m1_conformal_auto",
    "qwen8b_mixed13_m2_var_minsafe_tail035",
    "qwen8b_mixed13_m2_conformal_auto",
    "qwen8b_ruler8k_m1_floor2",
    "qwen8b_ruler4k_m3_var_bestcal_tail035",
    "qwen8b_ruler8k_m3_var_bestcal_tail035",
    "qwen8b_ruler4k_m3_conformal_auto",
    "qwen8b_ruler8k_m3_conformal_auto",
    "qwen8b_ruler4k_m5_var_bestcal_tail035",
    "qwen8b_ruler8k_m5_var_bestcal_tail035",
    "qwen8b_ruler4k_m5_var_minsafe_tail035",
    "qwen8b_ruler8k_m5_var_minsafe_tail035",
    "qwen8b_ruler4k_m5_conformal_auto",
    "qwen8b_ruler8k_m5_conformal_auto",
    "qwen8b_ruler4k_m5_conformal_floor2",
    "qwen8b_ruler8k_m5_conformal_floor2",
    "qwen8b_ruler8k_m1_var_bestcal_tail035",
    "qwen8b_ruler8k_m1_conformal_auto",
    "qwen8b_ruler16k_m1_floor2",
    "qwen8b_ruler16k_m2_var_bestcal_tail035_sharded",
    "qwen8b_ruler16k_m2_conformal_auto_sharded",
    "qwen8b_ruler16k_m3_var_bestcal_tail035_sharded",
    "qwen8b_ruler16k_m3_conformal_auto_sharded",
]

METHOD_LABELS = {
    "full_kv_cache": "Full KV",
    "naive_kv_gather_absolute_query_pos": "Naive gather + absolute query",
    "naive_kv_gather_compact_query_pos": "Naive gather + compact query",
    "rope_delta_repack_compact_query_pos": "RoPE repack + compact query",
    "rope_delta_repack_shifted_query_pos": "RoPE repack + shifted query",
    "prompt_rebuild_selected_pages": "Prompt rebuild",
    "variable_budget_kv_planner": "RiskKV input planner",
    "output_level_risk_kv_planner": "RiskKV output verifier",
}


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def fnum(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, ValueError):
        return 0.0


def markdown_table(rows: list[dict[str, str]], experiments: list[str]) -> str:
    lines = [
        "| Setting | Method | N | Score | KV | Online | E2E |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["experiment"] not in experiments:
            continue
        method = METHOD_LABELS.get(row["method"], row["method"])
        lines.append(
            "| {experiment} | {method} | {samples} | {score:.2f}% | {kv:.2f}% | {online:.3f}x | {e2e:.3f}x |".format(
                experiment=row["experiment"],
                method=method,
                samples=row["samples"],
                score=fnum(row, "score_pct"),
                kv=fnum(row, "kv_ratio_pct"),
                online=fnum(row, "online_speedup_sum"),
                e2e=fnum(row, "e2e_speedup_sum"),
            )
        )
    return "\n".join(lines) + "\n"


def latex_table(rows: list[dict[str, str]], experiments: list[str]) -> str:
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Setting & Method & N & Score & KV & Online \\",
        r"\midrule",
    ]
    for row in rows:
        if row["experiment"] not in experiments:
            continue
        method = METHOD_LABELS.get(row["method"], row["method"]).replace("_", r"\_")
        setting = row["experiment"].replace("_", r"\_")
        lines.append(
            r"{setting} & {method} & {samples} & {score:.2f} & {kv:.2f} & {online:.3f}x \\".format(
                setting=setting,
                method=method,
                samples=row["samples"],
                score=fnum(row, "score_pct"),
                kv=fnum(row, "kv_ratio_pct"),
                online=fnum(row, "online_speedup_sum"),
            )
        )
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def best_by_setting(rows: list[dict[str, str]]) -> str:
    by_exp: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_exp[row["experiment"]].append(row)
    lines = ["# Best Runtime Rows", ""]
    for experiment in MAIN_EXPERIMENTS:
        candidates = [
            row for row in by_exp.get(experiment, [])
            if row["method"] in {"variable_budget_kv_planner", "output_level_risk_kv_planner", "rope_delta_repack_compact_query_pos"}
        ]
        if not candidates:
            continue
        best = max(candidates, key=lambda row: (fnum(row, "score_pct"), fnum(row, "online_speedup_sum"), -fnum(row, "kv_ratio_pct")))
        lines.append(
            "- {experiment}: {method}, score={score:.2f}%, kv={kv:.2f}%, online={online:.3f}x".format(
                experiment=experiment,
                method=METHOD_LABELS.get(best["method"], best["method"]),
                score=fnum(best, "score_pct"),
                kv=fnum(best, "kv_ratio_pct"),
                online=fnum(best, "online_speedup_sum"),
            )
        )
    return "\n".join(lines) + "\n"


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
        default=Path("outputs/runtime_scaling_summary_20260707/icml_tables"),
    )
    args = parser.parse_args()

    rows = load_rows(args.summary_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "main_runtime_table.md").write_text(markdown_table(rows, MAIN_EXPERIMENTS), encoding="utf-8")
    (args.output_dir / "main_runtime_table.tex").write_text(latex_table(rows, MAIN_EXPERIMENTS), encoding="utf-8")
    (args.output_dir / "best_runtime_rows.md").write_text(best_by_setting(rows), encoding="utf-8")
    print(args.output_dir)


if __name__ == "__main__":
    main()
