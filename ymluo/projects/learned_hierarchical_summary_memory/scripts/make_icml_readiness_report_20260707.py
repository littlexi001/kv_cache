#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


INPUT_PLANNER_EXPERIMENTS = [
    "qwen8b_longbench_m4_var_bestcal_tail035",
    "qwen8b_longbench_m4_var_minsafe_tail035",
    "qwen8b_longbench_m4_conformal_auto",
    "qwen8b_mixed13_m1_var_bestcal_tail035",
    "qwen8b_mixed13_m1_var_minsafe_tail035",
    "qwen8b_mixed13_m1_conformal_auto",
    "qwen8b_ruler8k_m1_var_bestcal_tail035",
    "qwen8b_ruler8k_m1_conformal_auto",
]

LONG_BENCH_INPUTS = [
    "qwen8b_longbench_m8_conformal_auto",
    "qwen8b_longbench_m8_var_minsafe_tail035",
    "qwen8b_longbench_m8_var_bestcal_tail035",
    "qwen8b_longbench_m4_var_bestcal_tail035",
    "qwen8b_longbench_m4_var_minsafe_tail035",
    "qwen8b_longbench_m4_conformal_auto",
]

MIXED_INPUTS = [
    "qwen8b_mixed13_m2_conformal_auto",
    "qwen8b_mixed13_m2_var_minsafe_tail035",
    "qwen8b_mixed13_m1_var_bestcal_tail035",
    "qwen8b_mixed13_m1_var_minsafe_tail035",
    "qwen8b_mixed13_m1_conformal_auto",
]

RULER8K_INPUTS = [
    "qwen8b_ruler8k_m5_conformal_floor2",
    "qwen8b_ruler8k_m5_conformal_auto",
    "qwen8b_ruler8k_m5_var_minsafe_tail035",
    "qwen8b_ruler8k_m5_var_bestcal_tail035",
    "qwen8b_ruler8k_m3_conformal_auto",
    "qwen8b_ruler8k_m3_var_bestcal_tail035",
    "qwen8b_ruler8k_m1_var_bestcal_tail035",
    "qwen8b_ruler8k_m1_conformal_auto",
]

OUTPUT_VERIFIER_LONG_BENCH = [
    "qwen8b_longbench_m4_floor2",
    "qwen8b_longbench_m4_floor3",
]

SCALING_GROUPS = [
    {
        "label": "RULER 4k expanded",
        "primary": [
            "qwen8b_ruler4k_m5_conformal_floor2",
            "qwen8b_ruler4k_m5_conformal_auto",
            "qwen8b_ruler4k_m5_var_minsafe_tail035",
            "qwen8b_ruler4k_m5_var_bestcal_tail035",
            "qwen8b_ruler4k_m3_conformal_auto",
            "qwen8b_ruler4k_m3_var_bestcal_tail035",
        ],
        "fallback": ["qwen8b_ruler4k_m3_floor2"],
        "min_samples": 12,
        "requires_speedup": False,
    },
    {
        "label": "RULER 8k expanded",
        "primary": [
            "qwen8b_ruler8k_m5_conformal_floor2",
            "qwen8b_ruler8k_m5_conformal_auto",
            "qwen8b_ruler8k_m5_var_minsafe_tail035",
            "qwen8b_ruler8k_m5_var_bestcal_tail035",
            "qwen8b_ruler8k_m3_conformal_auto",
            "qwen8b_ruler8k_m3_var_bestcal_tail035",
        ],
        "fallback": ["qwen8b_ruler8k_m3_floor2"],
        "min_samples": 12,
        "requires_speedup": True,
    },
    {
        "label": "RULER 16k expanded",
        "primary": [
            "qwen8b_ruler16k_m3_conformal_auto_sharded",
            "qwen8b_ruler16k_m3_var_bestcal_tail035_sharded",
            "qwen8b_ruler16k_m2_conformal_auto_sharded",
            "qwen8b_ruler16k_m2_var_bestcal_tail035_sharded",
        ],
        "fallback": ["qwen8b_ruler16k_m2_floor2_sharded"],
        "min_samples": 16,
        "requires_speedup": True,
    },
]

METHOD_LABELS = {
    "full_kv_cache": "Full KV",
    "variable_budget_kv_planner": "RiskKV input planner",
    "output_level_risk_kv_planner": "RiskKV output verifier",
    "rope_delta_repack_compact_query_pos": "RoPE compact",
    "prompt_rebuild_selected_pages": "Prompt rebuild",
}


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def fnum(row: dict[str, str] | None, key: str) -> float:
    if row is None:
        return 0.0
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return 0.0


def inum(row: dict[str, str] | None, key: str) -> int:
    if row is None:
        return 0
    try:
        return int(float(row[key]))
    except (KeyError, TypeError, ValueError):
        return 0


def index_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {(row["experiment"], row["method"]): row for row in rows}


def row_for(index: dict[tuple[str, str], dict[str, str]], experiment: str, method: str) -> dict[str, str] | None:
    return index.get((experiment, method))


def quality_match(candidate: dict[str, str] | None, full: dict[str, str] | None, tolerance_pct: float = 0.05) -> bool:
    return candidate is not None and full is not None and fnum(candidate, "score_pct") + tolerance_pct >= fnum(full, "score_pct")


def format_row(row: dict[str, str] | None) -> str:
    if row is None:
        return "missing"
    return "score={:.2f}%, kv={:.2f}%, online={:.3f}x, e2e={:.3f}x, n={}".format(
        fnum(row, "score_pct"),
        fnum(row, "kv_ratio_pct"),
        fnum(row, "online_speedup_sum"),
        fnum(row, "e2e_speedup_sum"),
        inum(row, "samples"),
    )


def best_candidate(
    index: dict[tuple[str, str], dict[str, str]],
    experiments: list[str],
    method: str,
) -> tuple[str | None, dict[str, str] | None, dict[str, str] | None]:
    candidates: list[tuple[str, dict[str, str], dict[str, str]]] = []
    for experiment in experiments:
        row = row_for(index, experiment, method)
        full = row_for(index, experiment, "full_kv_cache")
        if row is not None and full is not None:
            candidates.append((experiment, row, full))
    if not candidates:
        return None, None, None
    return max(
        candidates,
        key=lambda item: (
            quality_match(item[1], item[2]),
            inum(item[1], "samples"),
            fnum(item[1], "score_pct"),
            fnum(item[1], "online_speedup_sum"),
            -fnum(item[1], "kv_ratio_pct"),
        ),
    )


def max_verifier_online(index: dict[tuple[str, str], dict[str, str]]) -> tuple[str | None, float]:
    best_name: str | None = None
    best_online = 0.0
    for experiment in OUTPUT_VERIFIER_LONG_BENCH:
        row = row_for(index, experiment, "output_level_risk_kv_planner")
        if row is None:
            continue
        online = fnum(row, "online_speedup_sum")
        if online > best_online:
            best_name = experiment
            best_online = online
    return best_name, best_online


def best_scaling_row(
    index: dict[tuple[str, str], dict[str, str]],
    experiments: list[str],
    method: str,
) -> tuple[str | None, dict[str, str] | None, dict[str, str] | None]:
    candidates: list[tuple[str, dict[str, str], dict[str, str]]] = []
    for experiment in experiments:
        row = row_for(index, experiment, method)
        full = row_for(index, experiment, "full_kv_cache")
        if row is not None and full is not None:
            candidates.append((experiment, row, full))
    if not candidates:
        return None, None, None
    return max(
        candidates,
        key=lambda item: (
            quality_match(item[1], item[2]),
            inum(item[1], "samples"),
            fnum(item[1], "online_speedup_sum"),
            fnum(item[1], "score_pct"),
            -fnum(item[1], "kv_ratio_pct"),
        ),
    )


def make_gate(name: str, status: str, evidence: str, requirement: str) -> dict[str, str]:
    return {
        "name": name,
        "status": status,
        "requirement": requirement,
        "evidence": evidence,
    }


def evaluate(index: dict[tuple[str, str], dict[str, str]]) -> tuple[list[dict[str, str]], dict[str, object]]:
    gates: list[dict[str, str]] = []
    lb_exp, lb_row, lb_full = best_candidate(index, LONG_BENCH_INPUTS, "variable_budget_kv_planner")
    verifier_exp, verifier_online = max_verifier_online(index)
    if lb_row is None:
        gates.append(make_gate("LongBench input planner", "PENDING", "No input planner result found.", "Match full score and beat output verifier speed."))
    else:
        lb_match = quality_match(lb_row, lb_full)
        lb_online = fnum(lb_row, "online_speedup_sum")
        required_online = max(1.0, verifier_online * 1.05) if verifier_online > 0 else 1.0
        if lb_match and lb_online >= required_online:
            status = "PASS"
        elif lb_match and lb_online > max(0.95, verifier_online):
            status = "BORDERLINE"
        else:
            status = "FAIL"
        gates.append(
            make_gate(
                "LongBench input planner",
                status,
                f"{lb_exp}: {format_row(lb_row)}; full={format_row(lb_full)}; best_verifier={verifier_exp or 'missing'} online={verifier_online:.3f}x.",
                f"Score should match full and online speed should be >= {required_online:.3f}x or clearly better than verifier.",
            )
        )

    mixed_exp, mixed_row, mixed_full = best_candidate(index, MIXED_INPUTS, "variable_budget_kv_planner")
    if mixed_row is None:
        gates.append(make_gate("Mixed13 robustness", "PENDING", "No Mixed13 input planner result found.", "Match full score and keep online speed near or above 1.0x."))
    else:
        mixed_match = quality_match(mixed_row, mixed_full)
        mixed_online = fnum(mixed_row, "online_speedup_sum")
        status = "PASS" if mixed_match and mixed_online >= 0.98 else "BORDERLINE" if mixed_match and mixed_online >= 0.90 else "FAIL"
        gates.append(
            make_gate(
                "Mixed13 robustness",
                status,
                f"{mixed_exp}: {format_row(mixed_row)}; full={format_row(mixed_full)}.",
                "Score should match full and online speed should be near or above 1.0x.",
            )
        )

    ruler_exp, ruler_row, ruler_full = best_candidate(index, RULER8K_INPUTS, "variable_budget_kv_planner")
    if ruler_row is None:
        gates.append(make_gate("RULER 8k input planner", "PENDING", "No RULER 8k input planner result found.", "Match full score and online speed > 1.0x."))
    else:
        ruler_match = quality_match(ruler_row, ruler_full)
        ruler_online = fnum(ruler_row, "online_speedup_sum")
        status = "PASS" if ruler_match and ruler_online > 1.0 else "FAIL"
        gates.append(
            make_gate(
                "RULER 8k input planner",
                status,
                f"{ruler_exp}: {format_row(ruler_row)}; full={format_row(ruler_full)}.",
                "Score should match full and online speed should be above 1.0x.",
            )
        )

    scaling_statuses: list[str] = []
    scaling_evidence: list[str] = []
    for group in SCALING_GROUPS:
        experiment, row, full = best_scaling_row(index, group["primary"], "variable_budget_kv_planner")
        if row is None or full is None:
            fallback_exp, fallback_row, fallback_full = best_scaling_row(index, group["fallback"], "output_level_risk_kv_planner")
            scaling_statuses.append("PENDING")
            if fallback_row is None:
                scaling_evidence.append(f"{group['label']}: missing input-side planner result")
            else:
                scaling_evidence.append(
                    f"{group['label']}: pending input-side planner; fallback {fallback_exp}: "
                    f"{format_row(fallback_row)}; full={format_row(fallback_full)}"
                )
            continue
        match = quality_match(row, full)
        enough_samples = inum(row, "samples") >= int(group["min_samples"])
        online_ok = fnum(row, "online_speedup_sum") > 1.0 if bool(group["requires_speedup"]) else True
        scaling_statuses.append("PASS" if match and enough_samples and online_ok else "FAIL")
        scaling_evidence.append(f"{group['label']} {experiment}: {format_row(row)}; full={format_row(full)}")
    if all(status == "PASS" for status in scaling_statuses):
        scaling_status = "PASS"
    elif any(status == "FAIL" for status in scaling_statuses):
        scaling_status = "FAIL"
    else:
        scaling_status = "PENDING"
    gates.append(
        make_gate(
            "RULER scaling coverage",
            scaling_status,
            " | ".join(scaling_evidence),
            "RULER 4k/8k expanded and 16k sharded results should be present, quality-matched, and long-context settings should show speedup.",
        )
    )

    counts = {status: sum(1 for gate in gates if gate["status"] == status) for status in ["PASS", "BORDERLINE", "FAIL", "PENDING"]}
    if counts["PENDING"] > 0:
        overall = "PENDING_RESULTS"
        recommendation = "实验结果还不完整，暂时不能声称足够支撑 ICML。优先等 input-side planner 和 RULER 扩样返回。"
    elif counts["FAIL"] == 0 and counts["BORDERLINE"] <= 1:
        overall = "ICML_CANDIDATE"
        recommendation = "证据链已经可以作为 ICML 主线候选：主方法应写成 RiskKV input planner，output verifier 作为安全闭环和蒸馏来源。"
    elif counts["FAIL"] <= 1 and counts["PASS"] >= 2:
        overall = "BORDERLINE"
        recommendation = "可以写成强 workshop/CCFB 版本，但 ICML 主会还需要补强失败门槛，尤其是 LongBench 质量或端到端速度。"
    else:
        overall = "NOT_READY"
        recommendation = "当前证据不足以支撑 ICML 主会，需要继续优化 planner 或收窄为 long-context serving scaling 论文。"
    return gates, {"overall": overall, "recommendation": recommendation, "counts": counts}


def write_markdown(path: Path, gates: list[dict[str, str]], summary: dict[str, object]) -> None:
    def cell(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", "<br>")

    lines = [
        "# RiskKV ICML Readiness Report",
        "",
        f"总体状态：**{summary['overall']}**",
        "",
        str(summary["recommendation"]),
        "",
        "## Gate Summary",
        "",
        "| Gate | Status | Requirement | Evidence |",
        "|---|---|---|---|",
    ]
    for gate in gates:
        lines.append(
            "| {name} | {status} | {requirement} | {evidence} |".format(
                name=cell(gate["name"]),
                status=cell(gate["status"]),
                requirement=cell(gate["requirement"]),
                evidence=cell(gate["evidence"]),
            )
        )
    lines += [
        "",
        "## 判读规则",
        "",
        "- `ICML_CANDIDATE`：主实验和扩样实验基本通过，可以开始按主会论文组织主结果。",
        "- `BORDERLINE`：有可写价值，但需要补强失败项或把 claim 收窄。",
        "- `PENDING_RESULTS`：关键远端实验未返回，不应过早判断。",
        "- `NOT_READY`：质量或速度门槛明显不够，需要继续做方法改进。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
        default=Path("outputs/runtime_scaling_summary_20260707/icml_readiness"),
    )
    args = parser.parse_args()

    rows = load_rows(args.summary_csv)
    index = index_rows(rows)
    gates, summary = evaluate(index)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_markdown(args.output_dir / "icml_readiness_report.md", gates, summary)
    (args.output_dir / "icml_readiness_report.json").write_text(
        json.dumps({"summary": summary, "gates": gates}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(args.output_dir)


if __name__ == "__main__":
    main()
