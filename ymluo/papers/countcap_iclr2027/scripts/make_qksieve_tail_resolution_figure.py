from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_PDF = ROOT / "figures" / "qksieve_tail_resolution.pdf"
OUT_PNG = ROOT / "figures" / "qksieve_tail_resolution.png"

lengths = np.asarray(
    [8192, 16384, 32768, 65536, 131072, 262080, 524256],
    dtype=float,
)
budget = np.minimum(
    lengths,
    np.minimum(1280.0, np.maximum(256.0, np.ceil(0.06 * lengths))),
)
rate = budget / lengths
fixed = np.minimum(lengths, 256.0)
old = np.minimum(
    lengths,
    np.minimum(2048.0, np.maximum(256.0, np.ceil(16.0 / rate))),
)
new = np.minimum(
    lengths,
    np.minimum(
        8192.0,
        np.maximum(
            256.0,
            256.0 * np.ceil((64.0 / rate) / 256.0),
        ),
    ),
)

plt.rcParams.update(
    {
        "font.size": 8.2,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9.0,
        "legend.fontsize": 7.3,
        "xtick.labelsize": 7.8,
        "ytick.labelsize": 7.8,
    }
)

fig, (ax_count, ax_tail) = plt.subplots(1, 2, figsize=(6.7, 2.55))
profiles = [
    ("Fixed 256", fixed, "#777d80", "--", "s"),
    ("Old cap 2,048", old, "#b05a32", "-.", "^"),
    ("Current c=64, aligned", new, "#176b7a", "-", "o"),
]
for label, values, color, style, marker in profiles:
    ax_count.plot(
        lengths / 1024.0,
        values,
        label=label,
        color=color,
        linestyle=style,
        marker=marker,
        linewidth=1.8,
        markersize=4.2,
    )

ax_count.set_xscale("log", base=2)
ax_count.set_xticks(lengths / 1024.0)
ax_count.set_xticklabels(
    ["8K", "16K", "32K", "64K", "128K", "256K", "512K"]
)
ax_count.grid(axis="y", color="#d7dadd", linewidth=0.6)
ax_count.spines["top"].set_visible(False)
ax_count.spines["right"].set_visible(False)

ax_count.set_ylabel("Quantile sample count")
ax_count.set_xlabel("Historical tokens")
ax_count.set_title("(a) Fixed-1,280 stress profile")
ax_count.legend(frameon=False, loc="upper left")
ax_count.annotate(
    "8,192 cap",
    (262080 / 1024.0, new[-2]),
    xytext=(-7, 6),
    textcoords="offset points",
    ha="right",
    color="#155967",
)

c_values = np.geomspace(4.0, 512.0, 200)
relative_std = 100.0 / np.sqrt(c_values)
ax_tail.plot(
    c_values,
    relative_std,
    color="#176b7a",
    linewidth=2.0,
)
marker_c = np.asarray([16.0, 64.0, 256.0])
marker_std = 100.0 / np.sqrt(marker_c)
ax_tail.scatter(
    marker_c,
    marker_std,
    color=["#b05a32", "#d17a22", "#176b7a"],
    s=28,
    zorder=3,
)
for c_value, std_value in zip(marker_c, marker_std):
    ax_tail.annotate(
        f"c={int(c_value)}: {std_value:.1f}%",
        (c_value, std_value),
        xytext=(5, 4),
        textcoords="offset points",
        fontsize=7.2,
        color="#34393b",
    )
ax_tail.set_xscale("log", base=2)
ax_tail.set_xticks(marker_c)
ax_tail.set_xticklabels(["16", "64", "256"])
ax_tail.set_xlabel("Expected target-tail samples, c")
ax_tail.set_ylabel("Candidate-count relative std.")
ax_tail.set_title("(b) Finite-sample quantile variation")
ax_tail.set_ylim(0.0, 30.0)
ax_tail.grid(axis="y", color="#d7dadd", linewidth=0.6)
ax_tail.spines["top"].set_visible(False)
ax_tail.spines["right"].set_visible(False)

fig.subplots_adjust(
    left=0.085,
    right=0.99,
    top=0.90,
    bottom=0.20,
    wspace=0.31,
)
fig.savefig(OUT_PDF, bbox_inches="tight")
fig.savefig(OUT_PNG, dpi=240, bbox_inches="tight")
