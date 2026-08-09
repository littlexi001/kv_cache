from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT_PDF = ROOT / "figures" / "qksieve_overview.pdf"
OUT_PNG = ROOT / "figures" / "qksieve_overview.png"


def box(ax, xy, width, height, text, face, edge, fontsize=8.2, lw=1.2):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=face,
        edgecolor=edge,
        linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#17202a",
        linespacing=1.18,
    )


def arrow(ax, start, end, color="#4d5656", lw=1.3, style="-|>"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=10,
            color=color,
            linewidth=lw,
            connectionstyle="arc3,rad=0",
        )
    )


fig, ax = plt.subplots(figsize=(7.05, 3.75))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

navy = "#215a6d"
blue_fill = "#e7f2f5"
orange = "#b85c24"
orange_fill = "#fff0e5"
green = "#287a5d"
green_fill = "#e8f5ef"
gray = "#5f6b6d"
gray_fill = "#f1f3f3"
red = "#a33f3f"

ax.text(0.02, 0.955, "Dense prompt prefill and one-time index construction", fontsize=9.2, fontweight="bold", color=navy)
box(ax, (0.02, 0.69), 0.14, 0.17, "Prompt\n$K,V$", blue_fill, navy)
box(ax, (0.205, 0.69), 0.18, 0.17, "Dense prefill\n(full attention)", blue_fill, navy)
box(ax, (0.43, 0.77), 0.20, 0.14, "Keys: stride 32\nQueries: final 8", orange_fill, orange)
box(ax, (0.43, 0.59), 0.20, 0.14, "Per layer/KV head\nQK-balanced basis", orange_fill, orange)
box(ax, (0.69, 0.69), 0.26, 0.17, "Auto 0/1/2/4/8-bit bands\npacked 30 B/token/KV head", orange_fill, orange)
arrow(ax, (0.16, 0.775), (0.205, 0.775))
arrow(ax, (0.385, 0.80), (0.43, 0.84))
arrow(ax, (0.53, 0.77), (0.53, 0.73))
arrow(ax, (0.63, 0.66), (0.69, 0.75))

ax.plot([0.02, 0.97], [0.515, 0.515], color="#c7cdcf", linewidth=0.9)
ax.text(0.02, 0.475, "Each decode step", fontsize=9.2, fontweight="bold", color=green)

box(ax, (0.02, 0.25), 0.12, 0.14, "Query\n$q_t$", green_fill, green)
box(ax, (0.18, 0.25), 0.17, 0.14, "QK-balanced\nprojection + INT8", green_fill, green)
box(ax, (0.39, 0.25), 0.20, 0.14, "Full packed-index scan\n+ exact proxy top-$B$", green_fill, green)
box(ax, (0.63, 0.25), 0.16, 0.14, "Candidate IDs\n$\\widehat{S}_t$", green_fill, green)
box(ax, (0.83, 0.25), 0.14, 0.14, "Exact sparse\nattention", green_fill, green)
arrow(ax, (0.14, 0.32), (0.18, 0.32))
arrow(ax, (0.35, 0.32), (0.39, 0.32))
arrow(ax, (0.59, 0.32), (0.63, 0.32))
arrow(ax, (0.79, 0.32), (0.83, 0.32))
arrow(ax, (0.82, 0.69), (0.49, 0.39), color=orange, lw=1.1)

box(ax, (0.65, 0.035), 0.32, 0.105, "Original FP16 $K,V$ remain resident\nand are gathered only for $\\widehat{S}_t$", gray_fill, gray, fontsize=7.6)
arrow(ax, (0.81, 0.14), (0.89, 0.25), color=gray, lw=1.1)

ax.text(
    0.02,
    0.075,
    r"$B(N)=\min\{N,1280,\max(256,\lceil0.06N\rceil)\}$",
    fontsize=7.8,
    color="#17202a",
)
ax.text(
    0.43,
    0.175,
    "Direct path: no sampled threshold, exact-QK rerank, task router, or Full fallback",
    fontsize=7.0,
    color=red,
    fontweight="bold",
    ha="center",
)

fig.subplots_adjust(left=0.015, right=0.99, bottom=0.02, top=0.99)
fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.02)
fig.savefig(OUT_PNG, dpi=240, bbox_inches="tight", pad_inches=0.02)
plt.close(fig)

print(OUT_PDF)
print(OUT_PNG)
