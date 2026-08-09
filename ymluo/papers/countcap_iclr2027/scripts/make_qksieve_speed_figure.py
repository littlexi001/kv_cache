from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT_PDF = ROOT / "figures" / "qksieve_subsystem_scaling.pdf"
OUT_PNG = ROOT / "figures" / "qksieve_subsystem_scaling.png"

lengths = [8, 16, 32, 64, 128]
retrieval_attention = [0.665, 1.214, 2.288, 3.502, 5.026]
with_index_append = [0.483, 0.888, 1.694, 2.730, 4.162]

fig, ax = plt.subplots(figsize=(6.5, 3.45))
ax.plot(
    lengths,
    retrieval_attention,
    color="#1f6f78",
    marker="o",
    linewidth=2.0,
    markersize=5.5,
    label="Retrieval + exact sparse attention",
)
ax.plot(
    lengths,
    with_index_append,
    color="#c15f2a",
    marker="s",
    linewidth=2.0,
    markersize=5.2,
    label="+ per-token index append",
)
ax.axhline(1.0, color="#60686b", linewidth=1.0, linestyle="--")
ax.set_xscale("log", base=2)
ax.set_xticks(lengths, [f"{length}K" for length in lengths])
ax.set_xlabel("Historical tokens")
ax.set_ylabel("Speedup over dense SDPA")
ax.set_ylim(0.0, 5.5)
ax.grid(axis="y", color="#d5d9da", linewidth=0.7)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(frameon=False, loc="upper left", fontsize=8.5)
fig.tight_layout()
fig.savefig(OUT_PDF, bbox_inches="tight")
fig.savefig(OUT_PNG, dpi=220, bbox_inches="tight")
