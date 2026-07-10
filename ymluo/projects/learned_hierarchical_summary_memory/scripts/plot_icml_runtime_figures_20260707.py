#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import re
from pathlib import Path


METHOD_STYLES = {
    "full_kv_cache": ("Full KV", "#3b3b3b"),
    "rope_delta_repack_compact_query_pos": ("RoPE compact", "#2f80ed"),
    "prompt_rebuild_selected_pages": ("Prompt rebuild", "#9b51e0"),
    "variable_budget_kv_planner": ("RiskKV input", "#219653"),
    "output_level_risk_kv_planner": ("RiskKV verifier", "#eb5757"),
}


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def fnum(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return 0.0


def ruler_length(experiment: str) -> int | None:
    patterns = [
        r"ruler(\d+)k",
        r"ruler_(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, experiment)
        if match is None:
            continue
        value = int(match.group(1))
        return value * 1024 if value in {4, 8, 16, 32} else value
    return None


def scale(value: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
    if in_max <= in_min:
        return (out_min + out_max) / 2.0
    return out_min + (value - in_min) * (out_max - out_min) / (in_max - in_min)


def svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;font-size:13px;fill:#222}.label{font-size:12px;fill:#555}.title{font-size:18px;font-weight:700}.tick{font-size:11px;fill:#666}.grid{stroke:#e7e7e7;stroke-width:1}.axis{stroke:#333;stroke-width:1.2}.legend{font-size:12px}</style>',
    ]


def write_speed_scaling(rows: list[dict[str, str]], path: Path) -> None:
    points: dict[str, list[tuple[int, float]]] = {}
    for row in rows:
        length = ruler_length(row["experiment"])
        if length is None:
            continue
        method = row["method"]
        if method not in METHOD_STYLES:
            continue
        points.setdefault(method, []).append((length, fnum(row, "online_speedup_sum")))
    for method in list(points):
        dedup: dict[int, float] = {}
        for length, speed in points[method]:
            dedup[length] = max(speed, dedup.get(length, 0.0))
        points[method] = sorted(dedup.items())

    width, height = 760, 430
    left, right, top, bottom = 72, 24, 46, 70
    plot_w, plot_h = width - left - right, height - top - bottom
    lengths = sorted({length for series in points.values() for length, _ in series}) or [4096, 8192, 16384]
    speeds = [speed for series in points.values() for _, speed in series] or [1.0]
    x_min, x_max = min(lengths), max(lengths)
    y_min, y_max = 0.0, max(1.2, max(speeds) * 1.12)

    svg = svg_header(width, height)
    svg.append('<text x="24" y="28" class="title">Online Speed Scaling on RULER</text>')
    for tick in [0.0, 0.5, 1.0, 1.5, 2.0]:
        if tick > y_max:
            continue
        y = top + plot_h - scale(tick, y_min, y_max, 0, plot_h)
        svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" class="grid"/>')
        svg.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" class="tick">{tick:.1f}x</text>')
    for length in lengths:
        x = left + scale(length, x_min, x_max, 0, plot_w)
        svg.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" class="grid"/>')
        svg.append(f'<text x="{x:.1f}" y="{top + plot_h + 22}" text-anchor="middle" class="tick">{length // 1024}k</text>')
    svg.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" class="axis"/>')
    svg.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis"/>')
    svg.append(f'<text x="{left + plot_w / 2:.1f}" y="{height - 24}" text-anchor="middle" class="label">Context length</text>')
    svg.append(f'<text x="20" y="{top + plot_h / 2:.1f}" transform="rotate(-90 20 {top + plot_h / 2:.1f})" text-anchor="middle" class="label">Online speedup vs full</text>')

    legend_y = 58
    for idx, (method, (label, color)) in enumerate(METHOD_STYLES.items()):
        series = points.get(method)
        if not series:
            continue
        coords = []
        for length, speed in series:
            x = left + scale(length, x_min, x_max, 0, plot_w)
            y = top + plot_h - scale(speed, y_min, y_max, 0, plot_h)
            coords.append((x, y))
        polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
        svg.append(f'<polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="2.4"/>')
        for x, y in coords:
            svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.2" fill="{color}"/>')
        lx = width - 190
        ly = legend_y + idx * 20
        svg.append(f'<rect x="{lx}" y="{ly - 9}" width="12" height="12" fill="{color}"/>')
        svg.append(f'<text x="{lx + 18}" y="{ly + 2}" class="legend">{html.escape(label)}</text>')
    svg.append("</svg>")
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def write_pareto(rows: list[dict[str, str]], path: Path) -> None:
    filtered = [row for row in rows if row["method"] in METHOD_STYLES]
    width, height = 760, 430
    left, right, top, bottom = 72, 24, 46, 70
    plot_w, plot_h = width - left - right, height - top - bottom
    x_min, x_max = 0.0, max(105.0, max((fnum(row, "kv_ratio_pct") for row in filtered), default=100.0) * 1.05)
    y_min, y_max = 0.0, max(105.0, max((fnum(row, "score_pct") for row in filtered), default=100.0) * 1.05)

    svg = svg_header(width, height)
    svg.append('<text x="24" y="28" class="title">Accuracy-KV Pareto</text>')
    for tick in [0, 25, 50, 75, 100]:
        x = left + scale(tick, x_min, x_max, 0, plot_w)
        y = top + plot_h - scale(tick, y_min, y_max, 0, plot_h)
        svg.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" class="grid"/>')
        svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" class="grid"/>')
        svg.append(f'<text x="{x:.1f}" y="{top + plot_h + 22}" text-anchor="middle" class="tick">{tick}%</text>')
        svg.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" class="tick">{tick}%</text>')
    svg.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" class="axis"/>')
    svg.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis"/>')
    svg.append(f'<text x="{left + plot_w / 2:.1f}" y="{height - 24}" text-anchor="middle" class="label">Active KV ratio</text>')
    svg.append(f'<text x="20" y="{top + plot_h / 2:.1f}" transform="rotate(-90 20 {top + plot_h / 2:.1f})" text-anchor="middle" class="label">Score</text>')

    for row in filtered:
        label, color = METHOD_STYLES[row["method"]]
        x = left + scale(fnum(row, "kv_ratio_pct"), x_min, x_max, 0, plot_w)
        y = top + plot_h - scale(fnum(row, "score_pct"), y_min, y_max, 0, plot_h)
        svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.0" fill="{color}" opacity="0.78"/>')
    legend_x, legend_y = width - 190, 58
    for idx, (_, (label, color)) in enumerate(METHOD_STYLES.items()):
        ly = legend_y + idx * 20
        svg.append(f'<rect x="{legend_x}" y="{ly - 9}" width="12" height="12" fill="{color}"/>')
        svg.append(f'<text x="{legend_x + 18}" y="{ly + 2}" class="legend">{html.escape(label)}</text>')
    svg.append("</svg>")
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary_csv",
        type=Path,
        default=Path("outputs/runtime_scaling_summary_20260707/runtime_scaling_summary.csv"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/runtime_scaling_summary_20260707/icml_figures"),
    )
    args = parser.parse_args()
    rows = load_rows(args.summary_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_speed_scaling(rows, args.output_dir / "speed_scaling.svg")
    write_pareto(rows, args.output_dir / "accuracy_kv_pareto.svg")
    print(args.output_dir)


if __name__ == "__main__":
    main()
