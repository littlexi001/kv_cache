from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT_PDF = ROOT / "figures" / "qksieve_overview_zh.pdf"
OUT_PNG = ROOT / "figures" / "qksieve_overview_zh.png"
FONT_PATH = Path(r"C:\Windows\Fonts\msyh.ttc")
CN_FONT = font_manager.FontProperties(fname=str(FONT_PATH))


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
        fontproperties=CN_FONT,
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

ax.text(
    0.02,
    0.955,
    "稠密 prompt prefill 与一次性索引构建",
    fontsize=9.2,
    fontweight="bold",
    color=navy,
    fontproperties=CN_FONT,
)
box(ax, (0.02, 0.69), 0.14, 0.17, "Prompt\n$K,V$", blue_fill, navy)
box(ax, (0.205, 0.69), 0.18, 0.17, "稠密 prefill\n（完整注意力）", blue_fill, navy)
box(ax, (0.43, 0.77), 0.20, 0.14, "Key：stride 32\nQuery：最后 8 个", orange_fill, orange)
box(ax, (0.43, 0.59), 0.20, 0.14, "逐 layer/KV head\nQK-balanced basis", orange_fill, orange)
box(ax, (0.69, 0.69), 0.26, 0.17, "自动 0/1/2/4/8-bit band\n30 B/token/KV head", orange_fill, orange)
arrow(ax, (0.16, 0.775), (0.205, 0.775))
arrow(ax, (0.385, 0.80), (0.43, 0.84))
arrow(ax, (0.53, 0.77), (0.53, 0.73))
arrow(ax, (0.63, 0.66), (0.69, 0.75))

ax.plot([0.02, 0.97], [0.515, 0.515], color="#c7cdcf", linewidth=0.9)
ax.text(
    0.02,
    0.475,
    "每个 decode step",
    fontsize=9.2,
    fontweight="bold",
    color=green,
    fontproperties=CN_FONT,
)

box(ax, (0.02, 0.25), 0.12, 0.14, "Query\n$q_t$", green_fill, green)
box(ax, (0.18, 0.25), 0.17, 0.14, "QK-balanced\n投影 + INT8", green_fill, green)
box(ax, (0.39, 0.25), 0.20, 0.14, "扫描完整 packed index\n+ proxy top-$B$", green_fill, green)
box(ax, (0.63, 0.25), 0.16, 0.14, "候选位置\n$\\widehat{S}_t$", green_fill, green)
box(ax, (0.83, 0.25), 0.14, 0.14, "精确稀疏\n注意力", green_fill, green)
arrow(ax, (0.14, 0.32), (0.18, 0.32))
arrow(ax, (0.35, 0.32), (0.39, 0.32))
arrow(ax, (0.59, 0.32), (0.63, 0.32))
arrow(ax, (0.79, 0.32), (0.83, 0.32))
arrow(ax, (0.82, 0.69), (0.49, 0.39), color=orange, lw=1.1)

box(
    ax,
    (0.65, 0.035),
    0.32,
    0.105,
    "原始 FP16 $K,V$ 始终驻留 GPU\n仅为 $\\widehat{S}_t$ gather",
    gray_fill,
    gray,
    fontsize=7.6,
)
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
    "直接路径：无 sampled threshold、精确 QK 重排、task router 或 Full 回退",
    fontsize=7.0,
    color=red,
    fontweight="bold",
    ha="center",
    fontproperties=CN_FONT,
)

fig.subplots_adjust(left=0.015, right=0.99, bottom=0.02, top=0.99)
fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.02)
fig.savefig(OUT_PNG, dpi=240, bbox_inches="tight", pad_inches=0.02)
plt.close(fig)

print(OUT_PDF)
print(OUT_PNG)
