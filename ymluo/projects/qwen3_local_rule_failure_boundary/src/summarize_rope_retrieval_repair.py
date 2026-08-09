from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Sequence


VARIANTS = (
    "full_attention",
    "post_top2",
    "post_top2_repair",
    "pre_top2",
    "pre_top2_repair",
    "envelope_top2",
    "envelope_top2_repair",
)

LABELS = {
    "full_attention": "Full attention",
    "post_top2": "post-RoPE Top-2%",
    "post_top2_repair": "post-RoPE Top-2% + repair",
    "pre_top2": "pre-RoPE QK Top-2%",
    "pre_top2_repair": "pre-RoPE QK Top-2% + repair",
    "envelope_top2": "phase-envelope Top-2%",
    "envelope_top2_repair": "phase-envelope Top-2% + repair",
}


def read_rows(paths: Sequence[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    rows.sort(key=lambda row: (int(row["seed"]), VARIANTS.index(str(row["variant"]))))
    keys = [(int(row["seed"]), str(row["variant"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate seed/variant rows found")
    return rows


def summarize(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for variant in VARIANTS:
        selected = sorted(
            (row for row in rows if row["variant"] == variant),
            key=lambda row: int(row["seed"]),
        )
        if not selected:
            continue
        mean_nll = statistics.fmean(float(row["gold_nll"]) for row in selected)
        correct = sum(int(row["generation_final_correct"]) for row in selected)
        output.append(
            {
                "variant": variant,
                "sample_count": len(selected),
                "gold_evidence_token_recall": statistics.fmean(
                    float(row["gold_evidence_token_recall"]) for row in selected
                ),
                "gold_evidence_line_hit_rate": statistics.fmean(
                    float(row["gold_evidence_line_hit_rate"]) for row in selected
                ),
                "gold_chain_complete_rate": statistics.fmean(
                    float(row["gold_chain_complete_rate"]) for row in selected
                ),
                "gold_evidence_attention_mass": statistics.fmean(
                    float(row["gold_evidence_attention_mass"]) for row in selected
                ),
                "mean_gold_nll": mean_nll,
                "gold_answer_ppl": math.exp(mean_nll),
                "correct_count": correct,
                "final_answer_accuracy": correct / len(selected),
            }
        )
    return output


def paired(rows: Sequence[dict[str, Any]], stem: str) -> dict[str, Any]:
    original = {
        int(row["seed"]): row for row in rows if row["variant"] == stem
    }
    repaired = {
        int(row["seed"]): row for row in rows if row["variant"] == f"{stem}_repair"
    }
    seeds = sorted(set(original) & set(repaired))
    deltas = [
        float(repaired[seed]["gold_nll"]) - float(original[seed]["gold_nll"])
        for seed in seeds
    ]
    if not all(
        float(original[seed]["gold_evidence_token_recall"])
        == float(repaired[seed]["gold_evidence_token_recall"])
        for seed in seeds
    ):
        raise ValueError(f"repair changed the candidate set for {stem}")
    return {
        "retrieval_method": stem,
        "sample_count": len(seeds),
        "mean_delta_nll_repair_minus_original": statistics.fmean(deltas),
        "median_delta_nll_repair_minus_original": statistics.median(deltas),
        "ppl_improved_count": sum(delta < 0 for delta in deltas),
        "ppl_worsened_count": sum(delta > 0 for delta in deltas),
        "answers_rescued": sum(
            not int(original[seed]["generation_final_correct"])
            and int(repaired[seed]["generation_final_correct"])
            for seed in seeds
        ),
        "answers_broken": sum(
            int(original[seed]["generation_final_correct"])
            and not int(repaired[seed]["generation_final_correct"])
            for seed in seeds
        ),
        "mean_attention_mass_delta": statistics.fmean(
            float(repaired[seed]["gold_evidence_attention_mass"])
            - float(original[seed]["gold_evidence_attention_mass"])
            for seed in seeds
        ),
        "candidate_recall_identical": True,
    }


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report(summary: Sequence[dict[str, Any]], pairs: Sequence[dict[str, Any]]) -> str:
    lines = [
        "# Qwen3-8B 64K：RoPE-free retrieval 与位置修复",
        "",
        "设置：16 个英文单-token clean 两跳样本；证据位于约第 256 token，Query 位于约 64K；每层每个 query head 保留 2%；chat-concise 单答案槽；RoPE YaRN factor=4。",
        "",
        "## 主结果",
        "",
        "| 方法 | Gold token recall | 两条链均命中 | 证据 attention mass | Gold PPL | 最终准确率 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {LABELS[str(row['variant'])]} | "
            f"{100 * float(row['gold_evidence_token_recall']):.2f}% | "
            f"{100 * float(row['gold_chain_complete_rate']):.2f}% | "
            f"{100 * float(row['gold_evidence_attention_mass']):.3f}% | "
            f"{float(row['gold_answer_ppl']):.4g} | "
            f"{100 * float(row['final_answer_accuracy']):.2f}% "
            f"({int(row['correct_count'])}/{int(row['sample_count'])}) |"
        )
    lines.extend(
        [
            "",
            "## 固定候选的位置修复配对结果",
            "",
            "| 候选方法 | 平均 ΔNLL（修复−原始） | PPL改善/恶化 | 救回/破坏答案 | attention mass变化 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in pairs:
        lines.append(
            f"| {LABELS[str(row['retrieval_method'])]} | "
            f"{float(row['mean_delta_nll_repair_minus_original']):+.4f} | "
            f"{int(row['ppl_improved_count'])}/{int(row['ppl_worsened_count'])} | "
            f"{int(row['answers_rescued'])}/{int(row['answers_broken'])} | "
            f"{100 * float(row['mean_attention_mass_delta']):+.3f} pp |"
        )
    lines.extend(
        [
            "",
            "## 结论",
            "",
            "1. pre-RoPE QK 将 evidence token recall 从 39.74% 提高到 48.77%，两条链均命中率从 57.28% 提高到 77.62%，但 PPL 与准确率显著变差。只提高显式证据召回，不足以近似原始 Top-2% 的下游效用。",
            "2. 原始 post-RoPE Top-2% 的 Gold PPL 为 1.202，优于 Full attention 的 1.317；准确率为 100%，Full 为 93.75%，复现了 Top-2% 可优于 Full 的观察。",
            "3. 将全部选中 Key 密集搬到 Query 前方并不稳定。对 pre-RoPE 候选，平均 NLL 明显下降，但只救回 1 个答案、破坏 2 个答案；对 post-Top2 候选则整体变差。",
            "4. phase-envelope 会偏向高幅值而非任务证据，证据召回仅 4.37%，不适合作为独立检索分数。",
            "5. 下一步位置修复应采用证据块整体平移：只移动远程证据块，保留块内相对距离，并让 recent/格式/局部功能 token 保持原位置；不应压缩所有选中 token。",
            "",
            "注：attention mass 是答案位置、逐层逐 head 的证据 softmax mass 均值；PPL 为先平均 NLL 再取指数。位置修复组与对应未修复组的首答案位置候选集合完全相同。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, nargs="+", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    rows = read_rows(args.rows)
    summary = summarize(rows)
    pairs = [paired(rows, stem) for stem in ("post_top2", "pre_top2", "envelope_top2")]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_csv(args.output_dir / "rows.csv", rows)
    write_csv(args.output_dir / "summary.csv", summary)
    (args.output_dir / "summary.json").write_text(
        json.dumps({"summary": summary, "paired_repair": pairs}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "analysis_report.md").write_text(
        report(summary, pairs), encoding="utf-8"
    )
    print(f"wrote {len(rows)} rows to {args.output_dir}")


if __name__ == "__main__":
    main()
