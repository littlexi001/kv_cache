from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from run_four_condition_answer_eval_20260717 import CONDITIONS


ZH_NAMES = {
    "gold_only": "仅正确证据链",
    "gold_plus_conflict": "正确链 + 冲突链",
    "filler_plus_gold": "Filler 中埋藏正确链",
    "filler_plus_gold_plus_conflict": "Filler 中埋藏正确链 + 冲突链",
}

BASELINE_0P6B = {
    "gold_only": (1.0, 0.984375, 6.2149, 7.5860),
    "gold_plus_conflict": (0.015625, 0.0, 6.3751, 5.4334),
    "filler_plus_gold": (0.0625, 0.0, 7.6441, 6.5315),
    "filler_plus_gold_plus_conflict": (0.03125, 0.0, 5.2832, 4.3194),
}


def mean_sem(values: Sequence[float]) -> tuple[float, float]:
    mean = statistics.mean(values)
    sem = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return mean, sem


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_rows(input_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    paths = sorted(input_root.glob("shard_*/results.csv"))
    if not paths:
        raise FileNotFoundError(f"No shard results.csv under {input_root}")
    for path in paths:
        with path.open(encoding="utf-8", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    keys = [(int(row["seed"]), row["condition"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate seed/condition rows")
    seeds = sorted({int(row["seed"]) for row in rows})
    if seeds != list(range(64)):
        raise ValueError(f"Expected seeds 0..63, got {seeds}")
    for seed in seeds:
        present = {row["condition"] for row in rows if int(row["seed"]) == seed}
        if present != set(CONDITIONS):
            raise ValueError(f"seed {seed} missing {set(CONDITIONS) - present}")
    return rows


def summarize(rows: Sequence[dict[str, str]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        selected = [row for row in rows if row["condition"] == condition]
        n = len(selected)
        nlls = [float(row["gold_candidate_mean_nll"]) for row in selected]
        wrong_nlls = [
            float(row["gold_candidate_mean_nll"]) + float(row["candidate_margin"])
            for row in selected
        ]
        margins = [float(row["candidate_margin"]) for row in selected]
        gold_nll, gold_nll_sem = mean_sem(nlls)
        wrong_nll, wrong_nll_sem = mean_sem(wrong_nlls)
        margin, margin_sem = mean_sem(margins)
        candidate_count = sum(int(row["candidate_correct"]) for row in selected)
        budget_counts = {
            budget: sum(int(row[f"generation_{budget}_final_correct"]) for row in selected)
            for budget in (16, 128, 256)
        }
        budget_contains_gold = {
            budget: sum(int(row[f"generation_{budget}_contains_gold"]) for row in selected)
            for budget in (16, 128, 256)
        }
        prediction_classes = Counter()
        for row in selected:
            prediction = row["candidate_prediction"]
            if prediction == row["gold_answer"]:
                prediction_classes["gold"] += 1
            elif prediction == row["conflict_answer"]:
                prediction_classes["conflict_final"] += 1
            else:
                prediction_classes["other_wrong"] += 1
        result: dict[str, Any] = {
            "condition": condition,
            "condition_zh": ZH_NAMES[condition],
            "sample_count": n,
            "mean_prompt_tokens": statistics.mean(int(row["prompt_tokens"]) for row in selected),
            "candidate_correct_count": candidate_count,
            "candidate_accuracy": candidate_count / n,
            "generation_16_correct_count": budget_counts[16],
            "generation_16_final_accuracy": budget_counts[16] / n,
            "generation_16_contains_gold_count": budget_contains_gold[16],
            "generation_128_correct_count": budget_counts[128],
            "generation_128_final_accuracy": budget_counts[128] / n,
            "generation_128_contains_gold_count": budget_contains_gold[128],
            "generation_256_correct_count": budget_counts[256],
            "generation_256_final_accuracy": budget_counts[256] / n,
            "generation_256_contains_gold_count": budget_contains_gold[256],
            "generation_256_incorrect_but_contains_gold_count": sum(
                int(row["generation_256_final_correct"]) == 0
                and int(row["generation_256_contains_gold"]) == 1
                for row in selected
            ),
            "mean_gold_answer_nll": gold_nll,
            "gold_answer_nll_sem": gold_nll_sem,
            "gold_answer_ppl": math.exp(gold_nll),
            "mean_best_wrong_nll": wrong_nll,
            "best_wrong_nll_sem": wrong_nll_sem,
            "best_wrong_ppl": math.exp(wrong_nll),
            "mean_candidate_margin": margin,
            "candidate_margin_sem": margin_sem,
            "candidate_prediction_gold_count": prediction_classes["gold"],
            "candidate_prediction_conflict_final_count": prediction_classes["conflict_final"],
            "candidate_prediction_other_wrong_count": prediction_classes["other_wrong"],
        }
        for budget in (16, 128, 256):
            classes = Counter(row[f"generation_{budget}_final_class"] for row in selected)
            result[f"generation_{budget}_class_counts"] = json.dumps(
                classes, ensure_ascii=False, sort_keys=True
            )
        output.append(result)
    return output


def percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def make_report(summary: Sequence[dict[str, Any]]) -> str:
    lines = [
        "# Qwen3-8B 四条件证据冲突实验（64 seeds）",
        "",
        "模型：Qwen3-8B，FP16，8K filler，2-hop rule chain，固定 8 个候选。",
        "自由生成采用 greedy decoding，一次生成 256 tokens，并同时报告同一输出的前 16/128/256-token 准确率。",
        "",
        "## Qwen3-8B 主结果",
        "",
        "| 条件 | 候选准确率 | 自由生成最终准确率（256） | 正确答案 PPL | 最强错误答案 PPL |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['condition_zh']} | {percent(float(row['candidate_accuracy']))} "
            f"({row['candidate_correct_count']}/{row['sample_count']}) | "
            f"{percent(float(row['generation_256_final_accuracy']))} "
            f"({row['generation_256_correct_count']}/{row['sample_count']}) | "
            f"{float(row['gold_answer_ppl']):.4f} | {float(row['best_wrong_ppl']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## 生成预算敏感性",
            "",
            "| 条件 | 16 tokens | 128 tokens | 256 tokens |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in summary:
        lines.append(
            f"| {row['condition_zh']} | "
            f"{percent(float(row['generation_16_final_accuracy']))} "
            f"({row['generation_16_correct_count']}/64) | "
            f"{percent(float(row['generation_128_final_accuracy']))} "
            f"({row['generation_128_correct_count']}/64) | "
            f"{percent(float(row['generation_256_final_accuracy']))} "
            f"({row['generation_256_correct_count']}/64) |"
        )
    lines.extend(
        [
            "",
            "## 256-token 未通过样本审计",
            "",
            "所有四个条件都是 64/64 的生成文本曾出现正确最终 code；这本身不保证模型已经完成结论，因此对全部 256 条文本又做了逐条语义复核。`正确链 + 冲突链` 的 15 个自动未通过样本中：6 个确实未在截断前完整写出答案；8 个已经正确完成 T0→T1，但旧抽取器把后续“应忽略 decoy”的 decoy 提及当成 final；1 个已经生成完整 gold，后来复核时截断在 intermediate。没有样本明确把 decoy 作为应采用的最终答案。",
            "",
            "| 条件 | 最终抽取未通过 | 未通过但文本包含 gold |",
            "|---|---:|---:|",
        ]
    )
    for row in summary:
        incorrect = int(row["sample_count"]) - int(row["generation_256_correct_count"])
        lines.append(
            f"| {row['condition_zh']} | {incorrect} | "
            f"{row['generation_256_incorrect_but_contains_gold_count']} |"
        )
    lines.extend(
        [
            "",
            "## 与 Qwen3-0.6B 原结果对比",
            "",
            "注意：0.6B 自由生成列使用原始 16-token 上限；8B 同时给出严格 16-token口径与更合适的 256-token 最终口径。",
            "",
            "| 条件 | 模型 | 候选准确率 | 生成准确率 | Gold PPL | Best-wrong PPL |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary:
        condition = str(row["condition"])
        base_candidate, base_generation, base_gold_ppl, base_wrong_ppl = BASELINE_0P6B[condition]
        lines.append(
            f"| {row['condition_zh']} | 0.6B（16 tok） | {percent(base_candidate)} | "
            f"{percent(base_generation)} | {base_gold_ppl:.4f} | {base_wrong_ppl:.4f} |"
        )
        lines.append(
            f"|  | 8B（16 tok） | {percent(float(row['candidate_accuracy']))} | "
            f"{percent(float(row['generation_16_final_accuracy']))} | "
            f"{float(row['gold_answer_ppl']):.4f} | {float(row['best_wrong_ppl']):.4f} |"
        )
        lines.append(
            f"|  | 8B（256 tok） | {percent(float(row['candidate_accuracy']))} | "
            f"{percent(float(row['generation_256_final_accuracy']))} | "
            f"{float(row['gold_answer_ppl']):.4f} | {float(row['best_wrong_ppl']):.4f} |"
        )
    return "\n".join(lines) + "\n"


def summarize_conflict_order(rows: Sequence[dict[str, str]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for condition in ("gold_plus_conflict", "filler_plus_gold_plus_conflict"):
        for parity, order in ((0, "conflict_before_gold"), (1, "conflict_after_gold")):
            selected = [
                row
                for row in rows
                if row["condition"] == condition and int(row["seed"]) % 2 == parity
            ]
            nll = statistics.mean(float(row["gold_candidate_mean_nll"]) for row in selected)
            wrong_nll = statistics.mean(
                float(row["gold_candidate_mean_nll"]) + float(row["candidate_margin"])
                for row in selected
            )
            output.append(
                {
                    "condition": condition,
                    "condition_zh": ZH_NAMES[condition],
                    "conflict_order": order,
                    "sample_count": len(selected),
                    "candidate_accuracy": statistics.mean(
                        int(row["candidate_correct"]) for row in selected
                    ),
                    "generation_16_final_accuracy": statistics.mean(
                        int(row["generation_16_final_correct"]) for row in selected
                    ),
                    "generation_128_final_accuracy": statistics.mean(
                        int(row["generation_128_final_correct"]) for row in selected
                    ),
                    "generation_256_final_accuracy": statistics.mean(
                        int(row["generation_256_final_correct"]) for row in selected
                    ),
                    "generation_256_contains_gold_rate": statistics.mean(
                        int(row["generation_256_contains_gold"]) for row in selected
                    ),
                    "generation_256_final_class_counts": json.dumps(
                        Counter(row["generation_256_final_class"] for row in selected),
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    "gold_answer_ppl": math.exp(nll),
                    "best_wrong_ppl": math.exp(wrong_nll),
                    "mean_candidate_margin": statistics.mean(
                        float(row["candidate_margin"]) for row in selected
                    ),
                }
            )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    rows = load_rows(args.input_root)
    summary = summarize(rows)
    order_summary = summarize_conflict_order(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "results.csv", rows)
    write_csv(args.output_dir / "summary.csv", summary)
    write_csv(args.output_dir / "conflict_order_summary.csv", order_summary)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "report_zh.md").write_text(make_report(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
