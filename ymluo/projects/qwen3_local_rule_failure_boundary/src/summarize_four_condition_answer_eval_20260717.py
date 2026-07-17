from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from run_four_condition_answer_eval_20260717 import CONDITIONS, summarize_rows, write_csv


ZH_NAMES = {
    "gold_only": "仅正确证据链",
    "gold_plus_conflict": "正确链 + 冲突链",
    "filler_plus_gold": "filler 中埋藏正确链",
    "filler_plus_gold_plus_conflict": "filler 中埋藏正确链 + 冲突链",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for directory in args.input_dirs:
        with (directory / "results.csv").open(encoding="utf-8", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    keys = [(int(row["seed"]), row["condition"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate seed/condition rows across shards")
    for seed in sorted({int(row["seed"]) for row in rows}):
        present = {row["condition"] for row in rows if int(row["seed"]) == seed}
        if present != set(CONDITIONS):
            raise ValueError(f"seed {seed} is missing conditions: {set(CONDITIONS) - present}")
    summary = summarize_rows(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "results.csv", rows)
    write_csv(args.output_dir / "summary.csv", summary)
    lines = [
        "# Qwen3-0.6B 合成证据四条件准确率与答案 PPL",
        "",
        "每个条件使用同一批两步规则问题；candidate accuracy 在固定 8 个候选中按答案 mean NLL 选最优；generation final accuracy 从自由生成文本中抽取最后一个已知 code；答案 PPL = exp(所有 gold-answer token 的平均 NLL)。",
        "",
        "| 条件 | n | 平均 prompt tokens | candidate accuracy | generation final accuracy | gold-answer PPL | best-wrong PPL | margin |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {ZH_NAMES[row['condition']]} | {row['sample_count']} | "
            f"{float(row['mean_prompt_tokens']):.1f} | "
            f"{100 * float(row['candidate_accuracy']):.2f}% | "
            f"{100 * float(row['generation_final_accuracy']):.2f}% | "
            f"{float(row['gold_answer_ppl']):.4f} | "
            f"{float(row['best_wrong_ppl']):.4f} | "
            f"{float(row['mean_candidate_margin']):+.4f} |"
        )
    lines.extend(
        [
            "",
            "说明：冲突链使用既有数据协议中的 DECOY RULE，与 gold chain 共用起点并导向错误终点；filler body 为 8192 token，gold chain 位于中间。",
        ]
    )
    (args.output_dir / "report_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
