from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PAPER_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = (
    PAPER_ROOT.parents[1]
    / "projects"
    / "qwen3_top2_head_limit3_ppl"
    / "results"
    / "20260801_overnight"
)
OUT_PDF = PAPER_ROOT / "figures" / "qksieve_system_diagnostics.pdf"
OUT_PNG = PAPER_ROOT / "figures" / "qksieve_system_diagnostics.png"


def load_json(name: str) -> dict:
    return json.loads((RESULT_ROOT / name).read_text(encoding="utf-8"))


qksieve = load_json("qksieve_general_direct_stages.json")["rows"]
binarypc = load_json("binarypc_official_direct_summary.json")["rows"]
whole_model = load_json("direct_length_summary.json")["rows"]

lengths = np.asarray([row["history_tokens"] / 1024 for row in qksieve])
qksieve_speed = np.asarray(
    [row["general_profile_complete_speedup"] for row in qksieve]
)
binarypc_speed = np.asarray(
    [row["attention_speedup_vs_full_preexpanded_sdpa"] for row in binarypc]
)

whole_lengths = np.asarray(
    [row["history_tokens"] / 1024 for row in whole_model]
)
steady_speed = np.asarray(
    [row["steady_decode_speedup_direct"] for row in whole_model]
)
horizon_speed = np.asarray(
    [row["measured_decode_horizon_speedup"] for row in whole_model]
)
request_speed = np.asarray(
    [row["measured_request_speedup"] for row in whole_model]
)

plt.rcParams.update(
    {
        "font.size": 8.2,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9.0,
        "legend.fontsize": 7.2,
        "xtick.labelsize": 7.8,
        "ytick.labelsize": 7.8,
        "axes.grid": False,
    }
)

fig, (ax_attention, ax_model) = plt.subplots(
    1, 2, figsize=(6.7, 2.65), gridspec_kw={"width_ratios": [1, 1]}
)

ax_attention.plot(
    lengths,
    qksieve_speed,
    color="#176b7a",
    marker="o",
    linewidth=2.0,
    markersize=4.8,
    label="QKSieve General",
)
ax_attention.plot(
    lengths,
    binarypc_speed,
    color="#b05a32",
    marker="s",
    linewidth=1.8,
    markersize=4.6,
    label="BinaryPC official selector",
)
ax_attention.axhline(1.0, color="#6d7375", linewidth=0.9, linestyle="--")
ax_attention.set_xscale("log", base=2)
ax_attention.set_xticks(lengths, [f"{int(value)}K" for value in lengths])
ax_attention.set_xlabel("Historical tokens")
ax_attention.set_ylabel("Complete attention speedup")
ax_attention.set_ylim(0.4, 12.1)
ax_attention.set_title("(a) Direct BF16 CUDA paths")
ax_attention.grid(axis="y", color="#d7dadd", linewidth=0.6)
ax_attention.spines["top"].set_visible(False)
ax_attention.spines["right"].set_visible(False)
ax_attention.legend(frameon=False, loc="upper left")

ax_model.plot(
    whole_lengths,
    steady_speed,
    color="#176b7a",
    marker="o",
    linewidth=2.0,
    markersize=4.8,
    label="Fast-path steady forward",
)
ax_model.plot(
    whole_lengths,
    horizon_speed,
    color="#7a5c9e",
    marker="^",
    linewidth=1.7,
    markersize=4.7,
    label="Fast-path 32-token decode",
)
ax_model.plot(
    whole_lengths,
    request_speed,
    color="#69995d",
    marker="D",
    linewidth=1.7,
    markersize=4.2,
    label="Fast-path prefill + decode",
)
ax_model.axhline(1.0, color="#6d7375", linewidth=0.9, linestyle="--")
ax_model.set_xscale("log", base=2)
ax_model.set_xticks(
    whole_lengths, [f"{int(round(value))}K" for value in whole_lengths]
)
ax_model.set_xlabel("Historical tokens")
ax_model.set_ylabel("Direct whole-model speedup")
ax_model.set_ylim(0.0, 5.25)
ax_model.set_title("(b) Direct Llama-3.1-8B timings")
ax_model.grid(axis="y", color="#d7dadd", linewidth=0.6)
ax_model.spines["top"].set_visible(False)
ax_model.spines["right"].set_visible(False)
ax_model.legend(frameon=False, loc="upper left")

fig.subplots_adjust(left=0.08, right=0.99, top=0.88, bottom=0.20, wspace=0.31)
fig.savefig(OUT_PDF, bbox_inches="tight")
fig.savefig(OUT_PNG, dpi=240, bbox_inches="tight")
