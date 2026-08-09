from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FOCUS_VARIANTS = (
    "full_rope",
    "rope_top2",
    "local_global_blend25",
    "dual_max_blend25",
)

LABELS = {
    "full_rope": "Full RoPE",
    "rope_top2": "post-RoPE Top-2%",
    "local_global_blend25": "SAGE pre-only",
    "dual_max_blend25": "SAGE dual-max",
}

COLORS = {
    "full_rope": "#64748b",
    "rope_top2": "#2563eb",
    "local_global_blend25": "#14b8a6",
    "dual_max_blend25": "#0f766e",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    return parser.parse_args()


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def bootstrap_mean_interval(
    values: np.ndarray,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    if values.size == 1:
        value = float(values[0])
        return value, value
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(samples, values.size))
    means = values[indices].mean(axis=1)
    return percentile_interval(means)


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    z = 1.959963984540054
    probability = successes / total
    denominator = 1.0 + z * z / total
    center = (probability + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            probability * (1.0 - probability) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return center - radius, center + radius


def aggregate(
    rows: pd.DataFrame,
    bootstrap_samples: int,
) -> pd.DataFrame:
    output: list[dict[str, float | int | str]] = []
    for (length, variant), frame in rows.groupby(
        ["target_context_tokens", "variant"],
        sort=True,
    ):
        nll = frame.gold_nll.to_numpy(dtype=float)
        nll_low, nll_high = bootstrap_mean_interval(
            nll,
            bootstrap_samples,
            seed=20260730 + int(length) + FOCUS_VARIANTS.index(str(variant)),
        )
        correct = int(frame.next_token_correct.sum())
        accuracy_low, accuracy_high = wilson_interval(correct, len(frame))
        output.append(
            {
                "target_context_tokens": int(length),
                "variant": str(variant),
                "sample_count": int(len(frame)),
                "mean_gold_nll": float(nll.mean()),
                "gold_ppl": math.exp(float(nll.mean())),
                "gold_ppl_ci_low": math.exp(nll_low),
                "gold_ppl_ci_high": math.exp(nll_high),
                "next_token_accuracy": correct / len(frame),
                "accuracy_ci_low": accuracy_low,
                "accuracy_ci_high": accuracy_high,
                "gold_evidence_token_recall": float(
                    frame.gold_evidence_token_recall.mean()
                ),
                "gold_chain_complete_rate": float(
                    frame.gold_chain_complete_rate.mean()
                ),
                "gold_evidence_attention_mass": float(
                    frame.gold_evidence_attention_mass.mean()
                ),
            }
        )
    result = pd.DataFrame(output)
    order = {variant: index for index, variant in enumerate(FOCUS_VARIANTS)}
    result["_variant_order"] = result.variant.map(order)
    return (
        result.sort_values(["target_context_tokens", "_variant_order"])
        .drop(columns="_variant_order")
        .reset_index(drop=True)
    )


def paired_vs_top2(
    rows: pd.DataFrame,
    bootstrap_samples: int,
) -> pd.DataFrame:
    baseline = rows[rows.variant == "rope_top2"][
        ["target_context_tokens", "seed", "gold_nll"]
    ].rename(columns={"gold_nll": "baseline_nll"})
    output: list[dict[str, float | int | str]] = []
    for variant in ("local_global_blend25", "dual_max_blend25"):
        selected = rows[rows.variant == variant].merge(
            baseline,
            on=["target_context_tokens", "seed"],
            how="inner",
        )
        selected["delta_nll"] = selected.gold_nll - selected.baseline_nll
        for length, frame in selected.groupby("target_context_tokens", sort=True):
            delta = frame.delta_nll.to_numpy(dtype=float)
            low, high = bootstrap_mean_interval(
                delta,
                bootstrap_samples,
                seed=20260731 + int(length) + FOCUS_VARIANTS.index(variant),
            )
            output.append(
                {
                    "target_context_tokens": int(length),
                    "variant": variant,
                    "sample_count": int(len(frame)),
                    "mean_delta_nll_vs_top2": float(delta.mean()),
                    "delta_nll_ci_low": low,
                    "delta_nll_ci_high": high,
                    "improved_sample_fraction": float((delta < 0).mean()),
                    "tied_sample_fraction": float((delta == 0).mean()),
                }
            )
    result = pd.DataFrame(output)
    order = {
        "local_global_blend25": 0,
        "dual_max_blend25": 1,
    }
    result["_variant_order"] = result.variant.map(order)
    return (
        result.sort_values(["target_context_tokens", "_variant_order"])
        .drop(columns="_variant_order")
        .reset_index(drop=True)
    )


def percent(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def write_report(
    output_path: Path,
    aggregate_rows: pd.DataFrame,
    paired_rows: pd.DataFrame,
) -> None:
    lines = [
        "# SAGE-RoPE：24 个全新 seeds 的扩展验证",
        "",
        "Qwen3-8B，seeds 8–31；所有方法共享同一条样例的公共 prefill；"
        "每层每个 head 的候选预算均约为上下文的 2%。",
        "",
        "## 主要结果",
        "",
        "| 长度 | 方法 | Gold PPL（95% bootstrap CI） | 首 token 准确率 | 证据 recall | 两链均命中 | 证据 mass |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows.itertuples(index=False):
        lines.append(
            f"| {int(row.target_context_tokens / 1024)}K | "
            f"{LABELS[row.variant]} | "
            f"{row.gold_ppl:.3f} "
            f"[{row.gold_ppl_ci_low:.3f}, {row.gold_ppl_ci_high:.3f}] | "
            f"{percent(row.next_token_accuracy)} | "
            f"{percent(row.gold_evidence_token_recall)} | "
            f"{percent(row.gold_chain_complete_rate)} | "
            f"{percent(row.gold_evidence_attention_mass)} |"
        )
    lines.extend(
        [
            "",
            "## 与 exact post-RoPE Top-2% 的逐样例比较",
            "",
            "负的 ΔNLL 表示 SAGE 的正确答案 PPL 更低。",
            "",
            "| 长度 | 方法 | 平均 ΔNLL（95% bootstrap CI） | 改善样例比例 |",
            "|---:|---|---:|---:|",
        ]
    )
    for row in paired_rows.itertuples(index=False):
        lines.append(
            f"| {int(row.target_context_tokens / 1024)}K | "
            f"{LABELS[row.variant]} | "
            f"{row.mean_delta_nll_vs_top2:+.3f} "
            f"[{row.delta_nll_ci_low:+.3f}, {row.delta_nll_ci_high:+.3f}] | "
            f"{percent(row.improved_sample_fraction)} |"
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- 这是独立于 seeds 0–7 的新测试集，没有据此重新调融合比例。",
            "- Gold PPL 是先对 NLL 求均值再取指数，因此是几何平均 PPL。",
            "- 当前实现显式计算全历史 pre-RoPE QK，只用于验证质量，不代表端到端加速。",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def plot_results(output_path: Path, rows: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    for variant in FOCUS_VARIANTS:
        frame = rows[rows.variant == variant].sort_values(
            "target_context_tokens"
        )
        x = frame.target_context_tokens / 1024
        label = LABELS[variant]
        color = COLORS[variant]
        axes[0, 0].plot(x, frame.gold_ppl, marker="o", label=label, color=color)
        axes[0, 1].plot(
            x,
            frame.next_token_accuracy,
            marker="o",
            label=label,
            color=color,
        )
        axes[1, 0].plot(
            x,
            frame.gold_evidence_token_recall,
            marker="o",
            label=label,
            color=color,
        )
        axes[1, 1].plot(
            x,
            100.0 * frame.gold_evidence_attention_mass,
            marker="o",
            label=label,
            color=color,
        )
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylabel("Gold PPL")
    axes[0, 1].set_ylabel("First-token accuracy")
    axes[1, 0].set_ylabel("Gold evidence recall")
    axes[1, 1].set_ylabel("Gold evidence mass (%)")
    for axis in axes.reshape(-1):
        axis.set_xlabel("Context length (K)")
        axis.grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.96),
        ncol=4,
        frameon=False,
    )
    fig.suptitle("Qwen3-8B SAGE-RoPE held-out evaluation (24 seeds)", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(args.rows_csv)
    rows = rows[rows.variant.isin(FOCUS_VARIANTS)].copy()
    expected = set(FOCUS_VARIANTS)
    actual = set(rows.variant.unique())
    if actual != expected:
        raise RuntimeError(f"missing variants: {sorted(expected - actual)}")
    counts = rows.groupby(["target_context_tokens", "variant"]).size()
    if counts.nunique() != 1:
        raise RuntimeError(f"unbalanced sample counts: {counts.to_dict()}")

    aggregate_rows = aggregate(rows, args.bootstrap_samples)
    paired_rows = paired_vs_top2(rows, args.bootstrap_samples)
    aggregate_rows.to_csv(output_dir / "heldout_aggregate.csv", index=False)
    paired_rows.to_csv(output_dir / "heldout_paired_vs_top2.csv", index=False)
    write_report(
        output_dir / "heldout_report.md",
        aggregate_rows,
        paired_rows,
    )
    plot_results(output_dir / "heldout_comparison.png", aggregate_rows)
    manifest = {
        "row_count": int(len(rows)),
        "sample_count_per_length_variant": int(counts.iloc[0]),
        "lengths": sorted(
            int(value) for value in rows.target_context_tokens.unique()
        ),
        "variants": list(FOCUS_VARIANTS),
        "bootstrap_samples": int(args.bootstrap_samples),
        "files": [
            "heldout_aggregate.csv",
            "heldout_paired_vs_top2.csv",
            "heldout_report.md",
            "heldout_comparison.png",
        ],
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
