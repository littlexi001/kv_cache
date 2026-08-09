from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


COLORS = {
    "attention": "#00897B",
    "decode": "#3366CC",
    "ceiling": "#D1495B",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def derive_speedup_point(
    *,
    history_tokens: int,
    full_ms_per_token: float,
    countcap_ms_per_token: float,
    non_attention_floor_ms: float,
    source: str,
) -> dict[str, float | int | str]:
    full_attention_ms = full_ms_per_token - non_attention_floor_ms
    countcap_attention_path_ms = countcap_ms_per_token - non_attention_floor_ms
    if full_attention_ms <= 0.0:
        raise ValueError(
            f"Full attention residual is non-positive at N={history_tokens}: "
            f"{full_attention_ms}"
        )
    if countcap_attention_path_ms <= 0.0:
        raise ValueError(
            f"CountCap attention-path residual is non-positive at "
            f"N={history_tokens}: {countcap_attention_path_ms}"
        )

    return {
        "history_tokens": int(history_tokens),
        "full_ms_per_token": float(full_ms_per_token),
        "countcap_ms_per_token": float(countcap_ms_per_token),
        "non_attention_floor_ms": float(non_attention_floor_ms),
        "full_attention_residual_ms": float(full_attention_ms),
        "countcap_attention_path_residual_ms": float(
            countcap_attention_path_ms
        ),
        "attention_subsystem_speedup": float(
            full_attention_ms / countcap_attention_path_ms
        ),
        "decode_speedup": float(full_ms_per_token / countcap_ms_per_token),
        "zero_attention_ceiling": float(
            full_ms_per_token / non_attention_floor_ms
        ),
        "source": source,
    }


def build_curve(
    cost_summary: dict[str, Any],
    long_summary: list[dict[str, Any]],
) -> dict[str, Any]:
    floor_ms = float(
        cost_summary["full_model"]["coefficients"]["intercept_ms"]
    )
    points: list[dict[str, float | int | str]] = []

    for row in cost_summary["measurements"]:
        points.append(
            derive_speedup_point(
                history_tokens=int(row["history_tokens"]),
                full_ms_per_token=float(row["full_steady_ms_per_token"]),
                countcap_ms_per_token=float(
                    row["countcap_steady_ms_per_token"]
                ),
                non_attention_floor_ms=floor_ms,
                source="same_protocol_steady_2k_32k",
            )
        )

    for row in long_summary:
        if row["model"] != "llama31_8b":
            continue
        points.append(
            derive_speedup_point(
                history_tokens=int(row["history_tokens"]),
                full_ms_per_token=float(row["full_milliseconds_per_step"]),
                countcap_ms_per_token=float(
                    row["direct_milliseconds_per_step"]
                ),
                non_attention_floor_ms=floor_ms,
                source="independent_long_context_64k_128k",
            )
        )

    points.sort(key=lambda row: int(row["history_tokens"]))
    if [int(row["history_tokens"]) for row in points] != [
        2048,
        4096,
        8192,
        16000,
        24000,
        32000,
        64000,
        128000,
    ]:
        raise ValueError("Unexpected history-length grid")

    return {
        "definition": {
            "non_attention_floor": (
                "Intercept of the Full steady-decode fit over 2K-32K"
            ),
            "attention_subsystem_speedup": (
                "(T_full - T_base) / (T_countcap - T_base)"
            ),
            "decode_speedup": "T_full / T_countcap",
            "zero_attention_ceiling": "T_full / T_base",
        },
        "non_attention_floor_ms": floor_ms,
        "points": points,
        "scope": [
            (
                "The attention-subsystem curve is a residual decomposition, "
                "not an independently timed CUDA-kernel benchmark."
            ),
            (
                "The zero-attention curve is a conservative Amdahl ceiling: "
                "the fitted intercept still contains fixed launch and "
                "projection costs that do not scale with history length."
            ),
            (
                "2K-32K points use the same steady-decode protocol. "
                "64K/128K are independent Llama long-context measurements "
                "and are shown with open markers."
            ),
        ],
    }


def write_curve(payload: dict[str, Any], output_json: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output_csv = output_json.with_suffix(".csv")
    rows = payload["points"]
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_segmented_curve(
    ax: plt.Axes,
    points: list[dict[str, Any]],
    *,
    field: str,
    label: str,
    color: str,
    marker: str,
    linestyle: str = "-",
) -> None:
    core = [
        row
        for row in points
        if row["source"] == "same_protocol_steady_2k_32k"
    ]
    independent = [
        row
        for row in points
        if row["source"] == "independent_long_context_64k_128k"
    ]

    x_core = [float(row["history_tokens"]) / 1000.0 for row in core]
    y_core = [float(row[field]) for row in core]
    ax.plot(
        x_core,
        y_core,
        color=color,
        marker=marker,
        markersize=6,
        linewidth=2.2,
        linestyle=linestyle,
        label=label,
    )

    bridge = [core[-1], *independent]
    x_bridge = [float(row["history_tokens"]) / 1000.0 for row in bridge]
    y_bridge = [float(row[field]) for row in bridge]
    ax.plot(
        x_bridge,
        y_bridge,
        color=color,
        marker=marker,
        markerfacecolor="white",
        markeredgewidth=1.5,
        markersize=6,
        linewidth=1.7,
        linestyle=":",
    )


def plot_curve(payload: dict[str, Any], output: Path) -> None:
    points = payload["points"]
    figure, ax = plt.subplots(figsize=(9.2, 5.5))

    plot_segmented_curve(
        ax,
        points,
        field="attention_subsystem_speedup",
        label="Attention subsystem (decomposed)",
        color=COLORS["attention"],
        marker="o",
    )
    plot_segmented_curve(
        ax,
        points,
        field="decode_speedup",
        label="Decode token (measured)",
        color=COLORS["decode"],
        marker="s",
    )
    plot_segmented_curve(
        ax,
        points,
        field="zero_attention_ceiling",
        label="Zero-attention Amdahl ceiling",
        color=COLORS["ceiling"],
        marker="^",
        linestyle="-.",
    )

    ax.axhline(1.0, color="#4A4A4A", linewidth=1.0, linestyle="--")
    ax.axvspan(48.0, 140.0, color="#777777", alpha=0.055, linewidth=0)
    ax.text(
        66.0,
        0.42,
        "independent long-context protocol",
        color="#555555",
        fontsize=8,
    )

    final = points[-1]
    annotations = [
        ("attention_subsystem_speedup", 0.25, COLORS["attention"]),
        ("decode_speedup", -0.45, COLORS["decode"]),
        ("zero_attention_ceiling", 0.25, COLORS["ceiling"]),
    ]
    for field, offset, color in annotations:
        value = float(final[field])
        ax.annotate(
            f"{value:.2f}x",
            xy=(128.0, value),
            xytext=(6, offset * 18),
            textcoords="offset points",
            color=color,
            fontsize=9,
            fontweight="bold",
        )

    floor_ms = float(payload["non_attention_floor_ms"])
    ax.text(
        0.018,
        0.965,
        f"Non-attention floor from Full fit: {floor_ms:.2f} ms/token",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color="#333333",
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": "white",
            "edgecolor": "#BBBBBB",
            "alpha": 0.92,
        },
    )

    ax.set_xscale("log", base=2)
    ticks = [2.048, 4.096, 8.192, 16.0, 24.0, 32.0, 64.0, 128.0]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["2K", "4K", "8K", "16K", "24K", "32K", "64K", "128K"])
    ax.set_xlim(1.75, 146.0)
    ymax = max(float(row["zero_attention_ceiling"]) for row in points)
    ax.set_ylim(0.0, math.ceil(ymax + 0.8))
    ax.set_xlabel("History length (tokens)")
    ax.set_ylabel("Speedup over Full KV (x)")
    ax.set_title(
        "QKSieve speedup decomposition across context length",
        fontsize=13,
    )
    ax.grid(alpha=0.2, which="both")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.885), fontsize=8.5)

    figure.text(
        0.5,
        0.012,
        (
            "Solid markers: same-protocol steady decode (2K-32K). "
            "Open markers: independent Llama measurements (64K-128K). "
            "Attention speedup and ceiling are latency-decomposition estimates."
        ),
        ha="center",
        va="bottom",
        fontsize=7.7,
        color="#444444",
    )
    figure.tight_layout(rect=(0.0, 0.045, 1.0, 1.0))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=240, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cost_summary", type=Path, required=True)
    parser.add_argument("--long_summary", type=Path, required=True)
    parser.add_argument("--output_data", type=Path, required=True)
    parser.add_argument("--output_figure", type=Path, required=True)
    args = parser.parse_args()

    payload = build_curve(
        load_json(args.cost_summary),
        load_json(args.long_summary),
    )
    write_curve(payload, args.output_data)
    plot_curve(payload, args.output_figure)


if __name__ == "__main__":
    main()
