"""Generate Chinese-labelled companion figures from the frozen CSV data."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from make_figures import (
    BLUE,
    DATA,
    FIGURES,
    GRAY,
    GREEN,
    INK,
    ORANGE,
    RED,
    read_csv,
)


def setup_style() -> None:
    windows = Path(os.environ.get("WINDIR", "C:/Windows"))
    font_path = windows / "Fonts" / "msyh.ttc"
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        family = font_manager.FontProperties(fname=str(font_path)).get_name()
    else:
        family = "Noto Sans CJK SC"
    mpl.rcParams.update(
        {
            "font.family": family,
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def rounded_box(ax, xy, width, height, text, *, fc="white", ec=INK,
                fontsize=8, weight="normal", align="center", pad=0.02):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad={pad},rounding_size=0.025",
        linewidth=1.1,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    x = xy[0] + (width / 2 if align == "center" else 0.03)
    ax.text(
        x,
        xy[1] + height / 2,
        text,
        ha=align,
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        linespacing=1.15,
    )
    return patch


def arrow(ax, start, end, *, color=GRAY, connectionstyle="arc3", lw=1.15):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=lw,
            color=color,
            connectionstyle=connectionstyle,
        )
    )


def make_overview() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.15))
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    ax = axes[0]
    ax.text(0.01, 0.96, "A   相同证据，不同相位", fontsize=10,
            fontweight="bold", va="top")
    rounded_box(ax, (0.03, 0.68), 0.26, 0.13, "真实证据", fc="#E6F4F1",
                ec=GREEN, fontsize=8, weight="bold")
    ax.plot([0.31, 0.67], [0.745, 0.745], color=GRAY, lw=2)
    ax.scatter(np.linspace(0.34, 0.64, 8), np.full(8, 0.745), s=8, color=GRAY)
    rounded_box(ax, (0.70, 0.68), 0.25, 0.13, "查询", fc="#EEF3FB",
                ec=BLUE, fontsize=8, weight="bold")
    ax.text(0.49, 0.84, "143,424 tokens", ha="center", fontsize=7.3, color=GRAY)
    ax.text(0.49, 0.59, r"$P(\mathrm{gold})=42.62\%$", ha="center",
            fontsize=9, color=GREEN, fontweight="bold")

    rounded_box(ax, (0.03, 0.28), 0.26, 0.13, "相同证据", fc="#E6F4F1",
                ec=GREEN, fontsize=8, weight="bold")
    ax.plot([0.31, 0.67], [0.345, 0.345], color=GRAY, lw=2)
    ax.scatter(np.linspace(0.33, 0.65, 12), np.full(12, 0.345), s=8, color=GRAY)
    rounded_box(ax, (0.70, 0.28), 0.25, 0.13, "相同查询", fc="#EEF3FB",
                ec=BLUE, fontsize=8, weight="bold")
    ax.text(0.49, 0.445, "+64 个填充 token", ha="center", fontsize=7.3,
            color=ORANGE, fontweight="bold")
    ax.text(0.49, 0.18, r"$P(\mathrm{gold})=1.75\%$", ha="center",
            fontsize=9, color=RED, fontweight="bold")
    ax.text(0.49, 0.06, "内容匹配保持不变", ha="center", fontsize=7.3, color=GRAY)

    ax = axes[1]
    ax.text(0.01, 0.96, "B   作用机制", fontsize=10, fontweight="bold", va="top")
    x = np.linspace(0, 2.4 * np.pi, 300)
    y = 0.64 + 0.15 * np.cos(x - 0.5)
    xx = np.linspace(0.08, 0.92, len(x))
    ax.plot(xx, y, color=ORANGE, lw=2)
    ax.fill_between(xx, 0.64, y, where=y >= 0.64, color=ORANGE, alpha=0.16)
    ax.fill_between(xx, 0.64, y, where=y < 0.64, color=RED, alpha=0.12)
    ax.axhline(0.64, color=GRAY, lw=0.8, ls="--")
    ax.text(0.16, 0.84, "相长", color=ORANGE, fontsize=7.3)
    ax.text(0.60, 0.43, "相消", color=RED, fontsize=7.3)
    ax.text(0.50, 0.53, r"$s_i(\Delta)=\rho_i\cos(\Delta\omega_i-\psi_i)$",
            ha="center", fontsize=8.6, fontweight="bold")
    boxes = [
        (0.02, "证据 QK\n下降"),
        (0.27, "证据注意力\n占比下降"),
        (0.52, "残差状态\n发生分叉"),
        (0.77, "margin\n穿过零点"),
    ]
    for pos, label in boxes:
        rounded_box(ax, (pos, 0.12), 0.20, 0.16, label,
                    fc="#F7F8FA", ec=GRAY, fontsize=7.2, weight="bold")
    for left in (0.22, 0.47, 0.72):
        arrow(ax, (left, 0.20), (left + 0.045, 0.20), color=GRAY)

    ax = axes[2]
    ax.text(0.01, 0.96, "C   由机制推导的修复", fontsize=10,
            fontweight="bold", va="top")
    rounded_box(ax, (0.04, 0.68), 0.25, 0.16, "sink + 局部\n标准 RoPE",
                fc="#EEF3FB", ec=BLUE, fontsize=8, weight="bold")
    rounded_box(ax, (0.04, 0.38), 0.25, 0.16, "远程历史\npre-RoPE 召回",
                fc="#E6F4F1", ec=GREEN, fontsize=8, weight="bold")
    rounded_box(ax, (0.38, 0.51), 0.24, 0.18, "2% 候选\n并集",
                fc="#FFF5E9", ec=ORANGE, fontsize=8.5, weight="bold")
    rounded_box(ax, (0.68, 0.51), 0.29, 0.18, "精确 post-RoPE\nsoftmax + V",
                fc="#E6F4F1", ec=GREEN, fontsize=8.2, weight="bold")
    arrow(ax, (0.29, 0.76), (0.38, 0.63), color=BLUE)
    arrow(ax, (0.29, 0.46), (0.38, 0.57), color=GREEN)
    arrow(ax, (0.62, 0.60), (0.68, 0.60), color=ORANGE)
    ax.text(0.50, 0.27, "召回阶段避免远程相位干扰；", ha="center", fontsize=7.5)
    ax.text(0.50, 0.18, "使用阶段保留预训练几何结构。", ha="center",
            fontsize=7.5, fontweight="bold")
    ax.text(0.50, 0.06, "SAGE-Post：保守且无需训练的干预", ha="center",
            fontsize=7.4, color=GREEN)

    fig.subplots_adjust(left=0.01, right=0.995, top=0.98, bottom=0.02, wspace=0.08)
    fig.savefig(FIGURES / "overview_zh.pdf", bbox_inches="tight",
                metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(FIGURES / "overview_zh.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_mechanism_evidence() -> None:
    qk = read_csv("first_layer_qk.csv")
    layers = read_csv("layerwise_exact_reconstruction.csv")
    patch = [r for r in read_csv("activation_patch.csv") if r["patch_kind"] == "residual_in"]
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.35))

    ax = axes[0]
    distance = np.array([float(r["relative_distance"]) / 1024 for r in qk])
    mean_qk = np.array([float(r["mean_gold_qk"]) for r in qk])
    ax.plot(distance, mean_qk, marker="o", color=GREEN, lw=1.8, ms=4)
    ax.axhline(0, color=GRAY, lw=0.8, ls="--")
    ax.set_xlabel("证据--查询距离 (K)")
    ax.set_ylabel("平均证据 QK")
    ax.set_title("固定 pre-RoPE Q/K")
    ax.grid(alpha=0.2)

    ax = axes[1]
    layer = np.array([int(r["layer"]) for r in layers])
    delta = np.array([float(r["delta_output_l2"]) for r in layers])
    ax.plot(layer, delta, color=BLUE, lw=1.8)
    ax.scatter([0, 16, 20, 24, 35], delta[[0, 16, 20, 24, 35]],
               color=BLUE, s=14, zorder=3)
    ax.set_yscale("log")
    ax.set_xlabel("层")
    ax.set_ylabel(r"$\|\delta h_{l+1}\|_2$")
    ax.set_title("跨层状态分叉")
    ax.grid(alpha=0.2, which="both")

    ax = axes[2]
    patch_layer = np.array([int(r["layer"]) for r in patch])
    margin = np.array([float(r["gold_vs_fixed_competitor_margin"]) for r in patch])
    ax.plot(patch_layer, margin, color=ORANGE, lw=1.8)
    ax.axhline(0, color=RED, lw=1, ls="--")
    ax.scatter([16, 20], [margin[16], margin[20]], color=GREEN, s=18, zorder=4)
    ax.annotate("L16: +0.625", (16, margin[16]), xytext=(10, 1.35),
                arrowprops={"arrowstyle": "->", "color": GREEN, "lw": 0.8},
                fontsize=6.7, color=GREEN)
    ax.set_xlabel("修补的残差层")
    ax.set_ylabel("正确答案 margin")
    ax.set_title("因果状态修补")
    ax.grid(alpha=0.2)

    for label, ax in zip(("a", "b", "c"), axes):
        ax.text(-0.18, 1.05, label, transform=ax.transAxes, fontsize=9,
                fontweight="bold", va="top")
    fig.tight_layout(w_pad=1.15)
    fig.savefig(FIGURES / "mechanism_evidence_zh.pdf", bbox_inches="tight",
                metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(FIGURES / "mechanism_evidence_zh.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_heldout_results() -> None:
    rows = read_csv("heldout_summary.csv")
    variants = ["full_rope", "rope_top2", "local_global_postscore"]
    names = {
        "full_rope": "全 RoPE",
        "rope_top2": "post-RoPE Top-2%",
        "local_global_postscore": "SAGE-Post",
    }
    colors = {"full_rope": GRAY, "rope_top2": BLUE, "local_global_postscore": GREEN}
    markers = {"full_rope": "o", "rope_top2": "s", "local_global_postscore": "D"}
    by_variant = {
        variant: sorted(
            [r for r in rows if r["variant"] == variant],
            key=lambda r: int(r["target_context_tokens"]),
        )
        for variant in variants
    }

    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.35))
    fields = [
        ("gold_ppl", "Gold PPL", True),
        ("next_token_correct", "首 token 准确率 (%)", False),
        ("gold_evidence_attention_mass", "证据注意力占比 (%)", False),
    ]
    for ax, (field, ylabel, log_scale) in zip(axes, fields):
        for variant in variants:
            selected = by_variant[variant]
            x = np.array([int(r["target_context_tokens"]) / 1024 for r in selected])
            y = np.array([float(r[field]) for r in selected])
            if field != "gold_ppl":
                y *= 100
            ax.plot(x, y, marker=markers[variant], color=colors[variant],
                    lw=1.6, ms=3.8, label=names[variant])
        if log_scale:
            ax.set_yscale("log")
        ax.set_xlabel("上下文长度 (K)")
        ax.set_ylabel(ylabel)
        ax.set_xticks([8, 16, 32, 64])
        ax.grid(alpha=0.2, which="both")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.035))
    for label, ax in zip(("a", "b", "c"), axes):
        ax.text(-0.18, 1.05, label, transform=ax.transAxes, fontsize=9,
                fontweight="bold", va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.90), w_pad=1.0)
    fig.savefig(FIGURES / "heldout_results_zh.pdf", bbox_inches="tight",
                metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(FIGURES / "heldout_results_zh.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    setup_style()
    make_overview()
    make_mechanism_evidence()
    make_heldout_results()


if __name__ == "__main__":
    main()
