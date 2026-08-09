from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument(
        "--with-distractors-csv",
        default=None,
    )
    return parser.parse_args()


def token_name(value: str) -> str:
    replacements = {
        "␠": "[space]",
        "↵": "\\n",
    }
    output = str(value)
    for source, target in replacements.items():
        output = output.replace(source, target)
    return output


def dominant_competitor(
    frame: pd.DataFrame,
) -> tuple[str, int, float]:
    counts = Counter(
        frame["strongest_competitor_token_label"].astype(str)
    )
    label, count = counts.most_common(1)[0]
    return label, count, count / len(frame)


def exact_metrics(
    exact: pd.DataFrame,
    boundaries: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in boundaries.items():
        result[key] = value
    frequent = boundaries[
        "first_512_token_window_majority_failure"
    ]
    boundary_end = (
        int(frequent["end_total_tokens"])
        if frequent is not None
        else int(exact.total_tokens.iloc[-1])
    )
    after = exact[
        (exact.total_tokens >= boundary_end)
        & (exact.total_tokens < boundary_end + 1024)
    ]
    if after.empty:
        after = exact[exact.total_tokens >= boundary_end]
    result["next_1k_failure_rate"] = float(
        (~after.top_is_gold.astype(bool)).mean()
    )
    result["next_1k_mean_gold_probability"] = float(
        after.gold_exact_probability.mean()
    )
    result["next_1k_mean_margin"] = float(
        after.gold_exact_vs_competitor_margin.mean()
    )
    label, count, share = dominant_competitor(after)
    result["next_1k_dominant_competitor"] = label
    result["next_1k_dominant_competitor_count"] = count
    result["next_1k_dominant_competitor_share"] = share

    first = boundaries["first_failure_total_tokens"]
    if first is not None:
        row = exact[exact.total_tokens == int(first)].iloc[0]
        result["first_failure_point"] = {
            "total_tokens": int(row.total_tokens),
            "gold_probability": float(row.gold_exact_probability),
            "margin": float(
                row.gold_exact_vs_competitor_margin
            ),
            "winner": str(row.top_token_label),
        }
    return result


def probability_competition_metrics(
    exact: pd.DataFrame,
    boundaries: dict[str, Any],
) -> dict[str, Any]:
    frequent = boundaries[
        "first_512_token_window_majority_failure"
    ]
    window_start = int(frequent["start_total_tokens"])
    window_end = int(frequent["end_total_tokens"])

    def summarize(
        name: str,
        selected: pd.DataFrame,
    ) -> dict[str, Any]:
        failures = ~selected.top_is_gold.astype(bool)
        label, count, share = dominant_competitor(selected)
        conditional: dict[str, Any] = {}
        for outcome, mask in (
            ("correct_points", ~failures),
            ("wrong_points", failures),
        ):
            subset = selected[mask]
            conditional[outcome] = {
                "points": int(len(subset)),
                "mean_gold_probability": float(
                    subset.gold_exact_probability.mean()
                ),
                "mean_strongest_competitor_probability": float(
                    subset.strongest_competitor_probability.mean()
                ),
                "mean_other_vocabulary_probability": float(
                    (
                        1.0
                        - subset.gold_exact_probability
                        - subset.strongest_competitor_probability
                    ).mean()
                ),
                "mean_gold_vs_competitor_margin": float(
                    subset.gold_exact_vs_competitor_margin.mean()
                ),
            }
        return {
            "name": name,
            "start_total_tokens": int(selected.total_tokens.iloc[0]),
            "end_total_tokens": int(selected.total_tokens.iloc[-1]),
            "points": int(len(selected)),
            "failure_rate": float(failures.mean()),
            "mean_gold_probability": float(
                selected.gold_exact_probability.mean()
            ),
            "mean_strongest_competitor_probability": float(
                selected.strongest_competitor_probability.mean()
            ),
            "mean_gold_vs_competitor_margin": float(
                selected.gold_exact_vs_competitor_margin.mean()
            ),
            "dominant_competitor": label,
            "dominant_competitor_count": int(count),
            "dominant_competitor_share": float(share),
            "conditional_on_output": conditional,
        }

    before = exact[exact.total_tokens < window_start]
    transition = exact[
        (exact.total_tokens >= window_start)
        & (exact.total_tokens <= window_end)
    ]
    after = exact[exact.total_tokens > window_end]
    all_competitors = Counter(
        exact.strongest_competitor_token_label.astype(str)
    )
    first_loss = exact[
        exact.strongest_competitor_probability
        > exact.gold_exact_probability
    ].iloc[0]
    return {
        "first_competitor_probability_crossing": {
            "total_tokens": int(first_loss.total_tokens),
            "gold_probability": float(
                first_loss.gold_exact_probability
            ),
            "strongest_competitor_probability": float(
                first_loss.strongest_competitor_probability
            ),
            "competitor": str(
                first_loss.strongest_competitor_token_label
            ),
        },
        "regions": [
            summarize("before_frequent_failure", before),
            summarize("majority_failure_transition", transition),
            summarize("after_frequent_failure_boundary", after),
        ],
        "competitor_counts_over_exact_scan": dict(
            all_competitors.most_common()
        ),
    }


def make_probability_competition_plot(
    exact: pd.DataFrame,
    boundaries: dict[str, Any],
    output: Path,
) -> None:
    frequent = boundaries[
        "first_512_token_window_majority_failure"
    ]
    window_start = int(frequent["start_total_tokens"])
    window_end = int(frequent["end_total_tokens"])
    x = exact.total_tokens.to_numpy(dtype=float) / 1024.0
    gold = 100 * exact.gold_exact_probability.to_numpy(dtype=float)
    competitor = (
        100
        * exact.strongest_competitor_probability.to_numpy(dtype=float)
    )
    gold_rolling = (
        pd.Series(gold).rolling(64, center=True, min_periods=1).mean()
    )
    competitor_rolling = (
        pd.Series(competitor)
        .rolling(64, center=True, min_periods=1)
        .mean()
    )
    is_gold = exact.top_is_gold.astype(bool).to_numpy()

    fig, (state_ax, probability_ax) = plt.subplots(
        2,
        1,
        figsize=(16, 7.5),
        sharex=True,
        gridspec_kw={"height_ratios": [0.55, 5.0]},
        constrained_layout=True,
    )
    state_ax.scatter(
        x,
        np.zeros_like(x),
        c=np.where(is_gold, "#56c7b1", "#ef6f6c"),
        marker="|",
        s=18,
        linewidths=1.0,
    )
    state_ax.set_yticks([])
    state_ax.set_ylabel("Top-1")
    state_ax.set_ylim(-0.8, 0.8)

    probability_ax.plot(
        x,
        gold,
        color="#4797e5",
        linewidth=0.75,
        alpha=0.35,
        label="P(nine), every token",
    )
    probability_ax.plot(
        x,
        competitor,
        color="#f2ad49",
        linewidth=0.75,
        alpha=0.35,
        label="P(strongest competitor: [space]), every token",
    )
    probability_ax.plot(
        x,
        gold_rolling,
        color="#2563a8",
        linewidth=2.1,
        label="P(nine), 64-token mean",
    )
    probability_ax.plot(
        x,
        competitor_rolling,
        color="#d47d13",
        linewidth=2.1,
        label="P([space]), 64-token mean",
    )
    probability_ax.fill_between(
        [window_start / 1024, window_end / 1024],
        0,
        100,
        color="#ef6f6c",
        alpha=0.09,
        label="first 512-token majority-failure window",
    )
    probability_ax.axvline(
        window_end / 1024,
        color="#b91c1c",
        linestyle="--",
        linewidth=1.2,
        label=f"frequent-failure boundary: {window_end:,}",
    )
    probability_ax.set_xlim(x[0], x[-1])
    probability_ax.set_ylim(0, 100)
    probability_ax.set_ylabel("First-token probability (%)")
    probability_ax.set_xlabel("Total sequence length (Ki tokens)")
    probability_ax.set_title(
        "No age distractors: nine loses probability mass to [space]"
    )
    probability_ax.grid(
        axis="y",
        color="#d1d5db",
        linewidth=0.6,
        alpha=0.7,
    )
    probability_ax.legend(
        loc="upper right",
        frameon=False,
        ncol=2,
    )
    fig.savefig(output, dpi=210)
    plt.close(fig)


def make_plot(
    coarse: pd.DataFrame,
    exact: pd.DataFrame,
    boundaries: dict[str, Any],
    output: Path,
) -> None:
    frequent = boundaries[
        "first_512_token_window_majority_failure"
    ]
    boundary_end = (
        int(frequent["end_total_tokens"])
        if frequent is not None
        else None
    )
    failures = (~exact.top_is_gold.astype(bool)).astype(float)
    rolling = failures.rolling(512).mean()

    fig, axes = plt.subplots(3, 1, figsize=(16, 13))
    axes[0].plot(
        coarse.total_tokens / 1024,
        100 * coarse.gold_exact_probability,
        linewidth=1.2,
        label="P(nine)",
    )
    axes[0].scatter(
        coarse.loc[~coarse.top_is_gold.astype(bool), "total_tokens"]
        / 1024,
        100
        * coarse.loc[
            ~coarse.top_is_gold.astype(bool),
            "gold_exact_probability",
        ],
        s=12,
        color="#d62728",
        label="wrong top token",
    )
    axes[0].set_ylabel("P(nine) (%)")
    axes[0].set_xlabel("Total length (Ki tokens)")
    axes[0].set_title("Coarse scan: 32 tokens to 136 Ki")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(
        exact.total_tokens,
        100 * exact.gold_exact_probability,
        label="P(nine)",
    )
    second = axes[1].twinx()
    second.plot(
        exact.total_tokens,
        exact.gold_exact_vs_competitor_margin,
        color="#ff7f0e",
        alpha=0.8,
        label="gold margin",
    )
    second.axhline(0.0, color="black", linewidth=1)
    if boundary_end is not None:
        axes[1].axvline(
            boundary_end,
            color="#d62728",
            linestyle="--",
            label="512-token majority-failure boundary",
        )
    axes[1].set_ylabel("P(nine) (%)")
    second.set_ylabel("Gold vs strongest competitor margin")
    axes[1].set_xlabel("Total tokens")
    axes[1].set_title("Exact tokenwise boundary region")
    axes[1].grid(alpha=0.25)
    handles1, labels1 = axes[1].get_legend_handles_labels()
    handles2, labels2 = second.get_legend_handles_labels()
    axes[1].legend(handles1 + handles2, labels1 + labels2)

    axes[2].plot(
        exact.total_tokens,
        100 * rolling,
        color="#9467bd",
        label="512-token rolling failure rate",
    )
    axes[2].axhline(
        50.0,
        color="black",
        linestyle="--",
        label="50% threshold",
    )
    if boundary_end is not None:
        axes[2].axvline(
            boundary_end,
            color="#d62728",
            linestyle="--",
        )
    axes[2].set_ylabel("Failure rate (%)")
    axes[2].set_xlabel("Total tokens")
    axes[2].set_title("Definition of frequent failure")
    axes[2].grid(alpha=0.25)
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(output, dpi=190)
    plt.close(fig)


def write_report(
    path: Path,
    summary: dict[str, Any],
    metrics: dict[str, Any],
    competition: dict[str, Any],
    with_distractors: dict[str, Any] | None,
) -> None:
    exact = summary["exact"]
    boundaries = exact["boundaries"]
    window64 = boundaries[
        "first_64_token_window_majority_failure"
    ]
    window512 = boundaries[
        "first_512_token_window_majority_failure"
    ]
    lines = [
        "# 无年龄干扰信息：频繁失败长度边界",
        "",
        "## 结论",
        "",
        (
            f"- 首次瞬时失败："
            f"**{boundaries['first_failure_total_tokens']:,} token**。"
        ),
        (
            f"- 首次连续 5 点失败结束位置："
            f"**{boundaries['first_5_consecutive_failure_end']:,} "
            f"token**。"
        ),
        (
            f"- 首个 64-token 窗口失败率超过 50%："
            f"**{window64['start_total_tokens']:,}–"
            f"{window64['end_total_tokens']:,} token**，"
            f"失败率 {100 * window64['failure_rate']:.2f}%。"
        ),
        (
            f"- 首个 512-token 窗口失败率超过 50%："
            f"**{window512['start_total_tokens']:,}–"
            f"{window512['end_total_tokens']:,} token**，"
            f"失败率 {100 * window512['failure_rate']:.2f}%。"
        ),
        "",
        (
            "本文把最后一项定义为“频繁失败边界”，因为它比一次"
            "瞬时错误或短暂连续错误更稳定。"
        ),
        "",
        "## 边界后的行为",
        "",
        (
            f"- 边界后最多 1K token 的失败率："
            f"{100 * metrics['next_1k_failure_rate']:.2f}%。"
        ),
        (
            f"- 平均 P(nine)："
            f"{100 * metrics['next_1k_mean_gold_probability']:.2f}%。"
        ),
        (
            f"- 平均 gold margin："
            f"{metrics['next_1k_mean_margin']:.3f}。"
        ),
        (
            f"- 最常见竞争 token："
            f"`{token_name(metrics['next_1k_dominant_competitor'])}`，"
            f"占 {100 * metrics['next_1k_dominant_competitor_share']:.2f}%。"
        ),
        "",
        "## 正确 token 与最强竞争 token 的概率交接",
        "",
        (
            "精扫区间的最强竞争 token 始终是 `[space]`，"
            "不是多个竞争 token 轮流接管。"
        ),
        "",
        "| 区间 | 失败率 | 平均 P(nine) | 平均 P([space]) | 平均 margin |",
        "|---|---:|---:|---:|---:|",
    ]
    for region in competition["regions"]:
        region_names = {
            "before_frequent_failure": "频繁失败窗口前",
            "majority_failure_transition": "首个 512-token 多数失败窗口",
            "after_frequent_failure_boundary": "频繁失败边界后",
        }
        lines.append(
            f"| {region_names[region['name']]} "
            f"({region['start_total_tokens']:,}–"
            f"{region['end_total_tokens']:,}) | "
            f"{100 * region['failure_rate']:.2f}% | "
            f"{100 * region['mean_gold_probability']:.2f}% | "
            f"{100 * region['mean_strongest_competitor_probability']:.2f}% | "
            f"{region['mean_gold_vs_competitor_margin']:.3f} |"
        )
    crossing = competition[
        "first_competitor_probability_crossing"
    ]
    after = competition["regions"][2]
    after_correct = after["conditional_on_output"]["correct_points"]
    after_wrong = after["conditional_on_output"]["wrong_points"]
    lines.extend(
        [
            "",
            (
                f"- 首次概率交叉发生在 **{crossing['total_tokens']:,} "
                f"tokens**：P(nine)="
                f"{100 * crossing['gold_probability']:.2f}%，"
                f"P([space])="
                f"{100 * crossing['strongest_competitor_probability']:.2f}%。"
            ),
            (
                "- 在首个多数失败窗口中，两者平均概率几乎持平；"
                "越过频繁失败边界后，差距扩大为 `[space]` "
                "约 59.42% 对 `nine` 约 34.66%。"
            ),
            (
                "- 只看边界后的错误点：平均 P(nine)="
                f"{100 * after_wrong['mean_gold_probability']:.2f}%，"
                "P([space])="
                f"{100 * after_wrong['mean_strongest_competitor_probability']:.2f}%，"
                "其余整个词表合计仅 "
                f"{100 * after_wrong['mean_other_vocabulary_probability']:.2f}%。"
            ),
            (
                "- 同一区间仍然答对的点：平均 P(nine)="
                f"{100 * after_correct['mean_gold_probability']:.2f}%，"
                "P([space])="
                f"{100 * after_correct['mean_strongest_competitor_probability']:.2f}%。"
                "这说明错误主要是 `nine` 与单个空格之间的概率质量交换，"
                "不是许多无关 token 一起上涨。"
            ),
            "",
            "![正确 token 与竞争 token 概率](./probability_competition.png)",
            "",
            "## 计算方式",
            "",
            "- 粗扫：32 token–136 Ki，间隔 256 token。",
            (
                f"- 中扫：{summary['medium']['start_total']:,}–"
                f"{summary['medium']['end_total']:,}，间隔 16 token。"
            ),
            (
                f"- 精扫：{exact['start_total']:,}–"
                f"{exact['end_total']:,}，逐 token。"
            ),
            (
                "- 每个阶段只 prefill 一次公共前缀；测试某长度时临时"
                "接上同一个问题，得到 logits 后裁掉问题 KV，再继续延长"
                "同一份 filler cache。"
            ),
            "",
            "## 解释限制",
            "",
            (
                "无干扰条件通过更多重复句号保持总长度，因此该边界描述的"
                "是“纯重复句号 filler”退化，不等同于一般自然文本中删除"
                "语义干扰后的边界。"
            ),
        ]
    )
    if with_distractors is not None:
        lines.extend(
            [
                "",
                "## 与含年龄干扰句条件比较",
                "",
                (
                    f"- 无年龄干扰、重复句号：约 "
                    f"**{window512['end_total_tokens']:,} token** "
                    "进入频繁失败。"
                ),
                (
                    f"- 含年龄干扰句的既有扫描只覆盖 136–144 Ki；"
                    f"在该窗口内，约 **{with_distractors['boundary']:,} "
                    "token** 首次进入多数失败。它不是对 32 token–"
                    "136 Ki 的全区间边界搜索。"
                ),
                "",
                (
                    "这不表示年龄干扰信息有益，而是说明纯重复句号会"
                    "更早激活空格/格式续写通道。"
                ),
            ]
        )
    lines.extend(
        [
            "",
            "![边界扫描](./failure_boundary.png)",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = Path(args.input_dir)
    summary = json.loads(
        (root / "summary.json").read_text(encoding="utf-8")
    )
    coarse = pd.read_csv(root / "coarse" / "points.csv")
    exact = pd.read_csv(root / "exact" / "points.csv")
    boundaries = summary["exact"]["boundaries"]
    metrics = exact_metrics(exact, boundaries)
    competition = probability_competition_metrics(
        exact,
        boundaries,
    )
    with_distractors = None
    if args.with_distractors_csv:
        other = pd.read_csv(args.with_distractors_csv)
        failure = (~other.top_is_gold.astype(bool)).astype(float)
        rolling = failure.rolling(64).mean()
        candidates = rolling[rolling > 0.5]
        if not candidates.empty:
            index = int(candidates.index[0])
            with_distractors = {
                "boundary": int(other.loc[index, "total_tokens"])
            }

    analysis = {
        "summary": summary,
        "exact_metrics": metrics,
        "probability_competition": competition,
        "with_distractors": with_distractors,
    }
    (root / "analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    make_plot(
        coarse,
        exact,
        boundaries,
        root / "failure_boundary.png",
    )
    make_probability_competition_plot(
        exact,
        boundaries,
        root / "probability_competition.png",
    )
    write_report(
        root / "report.md",
        summary,
        metrics,
        competition,
        with_distractors,
    )


if __name__ == "__main__":
    main()
