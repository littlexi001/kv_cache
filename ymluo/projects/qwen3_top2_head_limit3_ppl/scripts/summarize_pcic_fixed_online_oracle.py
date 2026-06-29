from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
DOCS = ROOT / "docs"


@dataclass(frozen=True)
class Case:
    case_id: str
    label: str
    combos: tuple[str, ...]
    fixed_pattern: str
    online_output: str


CASES: tuple[Case, ...] = (
    Case(
        case_id="hardtopic_eval64",
        label="Hard-topic eval64",
        combos=("0,6", "0,7", "0,13", "7,6", "2,0", "2,7", "2,0,7,12", "7,13"),
        fixed_pattern="server_pcic_hardtopic_static_b4_eval64_{combo_tag}_eager",
        online_output="server_pcic_hardtopic_b4_horizongate_top2_timingv2_eval64_seed64_eager",
    ),
    Case(
        case_id="hardtopic_eval128",
        label="Hard-topic eval128",
        combos=("0,6", "0,7", "0,13", "7,6", "2,0", "2,7", "2,0,7,12", "7,13"),
        fixed_pattern="server_pcic_hardtopic_static_b4_eval128_{combo_tag}_eager",
        online_output="server_pcic_hardtopic_b4_horizongate_top2_timingv2_eval128_seed64_eager",
    ),
    Case(
        case_id="war",
        label="War and Peace",
        combos=("7,6", "0,13", "0,7", "0,6"),
        fixed_pattern="server_pcic_war_static_b2_eval64_{combo_tag}_eager",
        online_output="server_pcic_war_b2_horizongate_top2_timingv2_seed64_eager",
    ),
    Case(
        case_id="monte",
        label="Count of Monte Cristo",
        combos=("2,0,7,12", "7,13", "2,7", "2,0"),
        fixed_pattern="server_pcic_monte_static_b2_eval64_{combo_tag}_eager",
        online_output="server_pcic_monte_b2_horizongate_top2_timingv2_seed64_eager",
    ),
)


def combo_tag(combo: str) -> str:
    return combo.replace(",", "_")


def result_path(output_name: str) -> Path:
    return OUTPUTS / output_name / "pcic_r_blockwise_results.csv"


def read_eval_rows(output_name: str) -> list[dict[str, str]]:
    path = result_path(output_name)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [row for row in csv.DictReader(handle) if row.get("kind") == "pcic_r_eval"]


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("empty values")
    return sum(values) / len(values)


def summarize_rows(rows: list[dict[str, str]]) -> dict[str, float]:
    baseline_seconds = sum(float(row.get("baseline_seconds") or 0.0) for row in rows)
    method_seconds = sum(float(row.get("method_seconds") or row.get("seconds") or 0.0) for row in rows)
    selected_seconds = sum(float(row.get("seconds") or 0.0) for row in rows)
    gate_seconds = sum(float(row.get("gate_seconds") or 0.0) for row in rows)
    return {
        "avg_delta_ppl": mean([float(row["delta_ppl"]) for row in rows]),
        "selected_ratio": selected_seconds / max(baseline_seconds, 1e-9),
        "method_ratio": method_seconds / max(baseline_seconds, 1e-9),
        "gate_s": gate_seconds,
    }


def fixed_output_name(case: Case, combo: str) -> str:
    return case.fixed_pattern.format(combo_tag=combo_tag(combo))


def block_key(row: dict[str, str]) -> int:
    return int(row["block"])


def summarize_case(case: Case) -> tuple[dict[str, object] | None, list[str]]:
    missing: list[str] = []
    fixed_by_combo: dict[str, list[dict[str, str]]] = {}
    for combo in case.combos:
        output_name = fixed_output_name(case, combo)
        try:
            fixed_by_combo[combo] = read_eval_rows(output_name)
        except FileNotFoundError:
            missing.append(str(result_path(output_name)))

    online_rows: list[dict[str, str]] | None = None
    try:
        online_rows = read_eval_rows(case.online_output)
    except FileNotFoundError:
        missing.append(str(result_path(case.online_output)))

    if missing:
        return None, missing

    assert online_rows is not None
    fixed_summaries = {
        combo: summarize_rows(rows)
        for combo, rows in fixed_by_combo.items()
    }
    best_fixed_combo, best_fixed = min(
        fixed_summaries.items(),
        key=lambda item: float(item[1]["avg_delta_ppl"]),
    )

    block_ids = sorted({block_key(row) for rows in fixed_by_combo.values() for row in rows})
    oracle_rows: list[dict[str, str]] = []
    oracle_combos: list[str] = []
    for block_id in block_ids:
        candidates = [
            (combo, next(row for row in rows if block_key(row) == block_id))
            for combo, rows in fixed_by_combo.items()
            if any(block_key(row) == block_id for row in rows)
        ]
        chosen_combo, chosen_row = min(candidates, key=lambda item: float(item[1]["delta_ppl"]))
        oracle_combos.append(chosen_combo)
        oracle_rows.append(chosen_row)
    oracle = summarize_rows(oracle_rows)

    online = summarize_rows(online_rows)
    online_combos = ";".join(row.get("combo", "") for row in online_rows)
    online_rules = [json.loads(row.get("rescue_rule") or "{}") for row in online_rows]
    cascade_extended = sum(int(rule.get("sentinel_cascade_extended", 0) or 0) for rule in online_rules)
    cascade_early = sum(int(rule.get("sentinel_cascade_accepted_early", 0) or 0) for rule in online_rules)

    return (
        {
            "case_id": case.case_id,
            "label": case.label,
            "blocks": len(block_ids),
            "best_fixed_combo": best_fixed_combo,
            "best_fixed_delta_ppl": best_fixed["avg_delta_ppl"],
            "best_fixed_ratio": best_fixed["method_ratio"],
            "online_delta_ppl": online["avg_delta_ppl"],
            "online_selected_ratio": online["selected_ratio"],
            "online_method_ratio": online["method_ratio"],
            "online_gate_s": online["gate_s"],
            "online_cascade_extended": cascade_extended,
            "online_cascade_early": cascade_early,
            "online_combos": online_combos,
            "oracle_delta_ppl": oracle["avg_delta_ppl"],
            "oracle_ratio": oracle["method_ratio"],
            "oracle_combos": ";".join(oracle_combos),
            "online_vs_fixed_delta": online["avg_delta_ppl"] - best_fixed["avg_delta_ppl"],
            "online_vs_oracle_gap": online["avg_delta_ppl"] - oracle["avg_delta_ppl"],
        },
        [],
    )


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, object]], missing: dict[str, list[str]], path: Path) -> None:
    lines: list[str] = []
    lines.append("# Fixed / Online PCIC / Blockwise Oracle 对比（2026-06-29）")
    lines.append("")
    lines.append("目的：验证 Horizon-PCIC 不是一个固定 combo 可以替代的小改，而是接近 blockwise oracle 的在线策略选择器。")
    lines.append("")
    lines.append("## 已完成结果")
    lines.append("")
    if rows:
        lines.append("| dataset | blocks | best fixed | fixed ΔPPL | online ΔPPL | oracle ΔPPL | online-fixed | online-oracle gap | online/base | gate_s | online combos | oracle combos |")
        lines.append("| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |")
        for row in rows:
            lines.append(
                f"| {row['label']} | {int(row['blocks'])} | `{row['best_fixed_combo']}` | "
                f"{float(row['best_fixed_delta_ppl']):.6f} | {float(row['online_delta_ppl']):.6f} | "
                f"{float(row['oracle_delta_ppl']):.6f} | {float(row['online_vs_fixed_delta']):.6f} | "
                f"{float(row['online_vs_oracle_gap']):.6f} | {float(row['online_method_ratio']):.3f} | "
                f"{float(row['online_gate_s']):.3f} | `{row['online_combos']}` | `{row['oracle_combos']}` |"
            )
    else:
        lines.append("当前本地没有可汇总的完整结果；脚本会在服务器 outputs 存在后自动生成表格。")
    lines.append("")
    lines.append("## 缺失结果")
    lines.append("")
    if missing:
        for case_id, paths in missing.items():
            lines.append(f"### {case_id}")
            for item in paths:
                lines.append(f"- `{item}`")
            lines.append("")
    else:
        lines.append("- 无。")
        lines.append("")
    lines.append("## 判据")
    lines.append("")
    lines.append("- 若 online 明显优于 best fixed，说明动态选择不是可有可无。")
    lines.append("- 若 online 接近 blockwise oracle，说明 Pairwise-CIC + rescue gate 的选择信号有效。")
    lines.append("- 若 online 与 best fixed 接近，需要增加非平稳文本、更多 blocks 或更强候选集来证明策略切换价值。")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_csv", default=str(DOCS / "pcic_fixed_online_oracle_2026_06_29.csv"))
    parser.add_argument("--output_md", default=str(DOCS / "pcic_fixed_online_oracle_2026_06_29.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    DOCS.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    missing: dict[str, list[str]] = {}
    for case in CASES:
        row, case_missing = summarize_case(case)
        if row is None:
            missing[case.case_id] = case_missing
        else:
            rows.append(row)
    write_csv(rows, Path(args.output_csv))
    write_markdown(rows, missing, Path(args.output_md))
    print(args.output_md)
    if missing:
        print(f"missing_cases={len(missing)}")
    else:
        print("missing_cases=0")


if __name__ == "__main__":
    main()
