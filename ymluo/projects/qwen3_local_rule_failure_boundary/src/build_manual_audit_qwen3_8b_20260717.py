from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


CONDITION_NAMES = {
    "gold_only": "仅正确证据链",
    "gold_plus_conflict": "正确链 + 冲突链",
    "filler_plus_gold": "Filler 中埋藏正确链",
    "filler_plus_gold_plus_conflict": "Filler 中埋藏正确链 + 冲突链",
}

# These six labels come from line-by-line semantic review of the complete 256-token
# generations.  All other 250 rows clearly state the gold code as the direct answer,
# the second VERIFIED-rule result, or the final active code.
INCOMPLETE_GOLD_CONFLICT = {
    5: "截断在 'According to VERIFIED RULE T1,' 之后，尚未生成 T1 的结果",
    7: "截断在 'According to VERIFIED RULE T1,' 之后，尚未生成 T1 的结果",
    25: "长篇复述四条规则后截断在 Step 1 的 T0 条件中，尚未完成两步推理",
    37: "完成 T0 后截断在 'we apply VERIFIED RULE T1'，尚未生成第二步结果",
    43: "已选择并应用 VERIFIED T1，但最终编号截断为 'GC25-8'，答案不完整",
    53: "完成 T0 后截断在 'apply VERIFIED RULE T1'，尚未生成第二步结果",
}

AUTO_DECOY_FALSE_FAILURES = {9, 12, 19, 47, 49, 52, 54, 62}
AUTO_INTERMEDIATE_FALSE_FAILURES = {15}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def audit_reason(row: dict[str, str]) -> tuple[str, str]:
    seed = int(row["seed"])
    condition = row["condition"]
    auto_class = row["generation_256_final_class"]
    if condition == "gold_plus_conflict" and seed in INCOMPLETE_GOLD_CONFLICT:
        return "incomplete", INCOMPLETE_GOLD_CONFLICT[seed]
    if condition == "gold_plus_conflict" and seed in AUTO_DECOY_FALSE_FAILURES:
        return (
            "correct",
            "明确沿 VERIFIED T0→T1 得到 gold；随后在说明应忽略 DECOY 时再次提到 decoy，旧抽取器误取最后一次 decoy 提及",
        )
    if condition == "gold_plus_conflict" and seed in AUTO_INTERMEDIATE_FALSE_FAILURES:
        return (
            "correct",
            "明确说明 VERIFIED T1 激活完整 gold code；随后总结句截断在中间编号，旧抽取器误取 intermediate",
        )
    if auto_class == "gold":
        return "correct", "逐条复核：明确把 gold 作为第二步结果、final active code 或直接答案"
    if condition == "gold_only":
        return "correct", "旧抽取器取到后续复核中的中间编号；此前已明确完成 T1 并给出 gold"
    if condition == "filler_plus_gold":
        return "correct", "生成开头已直接给出 gold；随后解释被截断，旧抽取器误取后文中间编号"
    if condition == "filler_plus_gold_plus_conflict":
        return "correct", "生成开头/正文已明确给出 gold；后续解释截断不改变已给出的最终答案"
    return "incorrect", "逐条复核未找到支持 gold 的完整结论"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    with args.results_csv.open(encoding="utf-8", newline="") as handle:
        source = list(csv.DictReader(handle))
    if len(source) != 256:
        raise ValueError(f"Expected 256 rows, got {len(source)}")

    audited: list[dict[str, Any]] = []
    for row in sorted(source, key=lambda item: (int(item["seed"]), item["condition"])):
        label, reason = audit_reason(row)
        audited.append(
            {
                "seed": int(row["seed"]),
                "condition": row["condition"],
                "condition_zh": CONDITION_NAMES[row["condition"]],
                "gold_answer": row["gold_answer"],
                "conflict_answer": row["conflict_answer"],
                "automatic_final_class": row["generation_256_final_class"],
                "automatic_final_answer": row["generation_256_final_answer"],
                "automatic_correct": int(row["generation_256_final_correct"]),
                "manual_label": label,
                "manual_strict_correct": int(label == "correct"),
                "manual_incomplete": int(label == "incomplete"),
                "manual_incorrect": int(label == "incorrect"),
                "manual_reason": reason,
                "generated_text_256": row["generation_256_text"],
            }
        )

    summary: list[dict[str, Any]] = []
    for condition in CONDITION_NAMES:
        selected = [row for row in audited if row["condition"] == condition]
        counts = Counter(row["manual_label"] for row in selected)
        auto_correct = sum(int(row["automatic_correct"]) for row in selected)
        summary.append(
            {
                "condition": condition,
                "condition_zh": CONDITION_NAMES[condition],
                "sample_count": len(selected),
                "automatic_correct_count": auto_correct,
                "automatic_accuracy": auto_correct / len(selected),
                "manual_correct_count": counts["correct"],
                "manual_strict_accuracy": counts["correct"] / len(selected),
                "manual_incomplete_count": counts["incomplete"],
                "manual_incorrect_count": counts["incorrect"],
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "manual_audit_rows.csv", audited)
    write_csv(args.output_dir / "manual_audit_summary.csv", summary)
    (args.output_dir / "manual_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
