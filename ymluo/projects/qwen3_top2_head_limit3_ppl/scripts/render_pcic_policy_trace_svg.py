#!/usr/bin/env python3
from __future__ import annotations

import csv
import html
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRACE_CSV = ROOT / "docs" / "pcic_blockwise_policy_trace_2026_06_29.csv"
FIG_DIR = ROOT / "figures"
OUT_SVG = FIG_DIR / "pcic_policy_trace_2026_06_29.svg"
OUT_MD = ROOT / "docs" / "pcic_policy_trace_figure_2026_06_29.md"


COLORS = [
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#E45756",
    "#72B7B2",
    "#B279A2",
    "#FF9DA6",
    "#9D755D",
    "#BAB0AC",
    "#8CD17D",
]


SELECTED_CASES = [
    "b8_top2",
    "b8_cond_auto_anchor",
    "hard",
    "monte",
    "ruler_variable",
    "ruler_topicswitch",
]


def split_trace(value: str) -> list[str]:
    return [part.strip() for part in value.split("/") if part.strip()]


def svg_text(x: int, y: int, text: str, size: int = 12, weight: str = "400", fill: str = "#222") -> str:
    return (
        f'<text x="{x}" y="{y}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}">{html.escape(text)}</text>'
    )


def main() -> None:
    rows = list(csv.DictReader(TRACE_CSV.open(newline="", encoding="utf-8-sig")))
    selected = [row for row in rows if row.get("case_id") in SELECTED_CASES]
    if not selected:
        selected = rows[:8]

    combo_order: list[str] = []
    for row in selected:
        for combo in split_trace(row.get("final_trace", "")):
            if combo not in combo_order:
                combo_order.append(combo)
    color_by_combo = {combo: COLORS[idx % len(COLORS)] for idx, combo in enumerate(combo_order)}

    left = 260
    top = 78
    cell_w = 48
    cell_h = 26
    row_gap = 18
    max_blocks = max(int(row.get("blocks") or 0) for row in selected)
    width = left + max_blocks * cell_w + 300
    height = top + len(selected) * (cell_h + row_gap) + 150

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(24, 32, "Horizon-PCIC blockwise policy trace", 20, "700"),
        svg_text(24, 55, "Each cell is the selected compression policy for one block; outlined cells changed from the initial short-horizon choice.", 12, "400", "#555"),
    ]

    for block_idx in range(max_blocks):
        x = left + block_idx * cell_w + cell_w // 2
        parts.append(svg_text(x - 8, top - 18, f"b{block_idx}", 11, "700", "#555"))

    for row_idx, row in enumerate(selected):
        y = top + row_idx * (cell_h + row_gap)
        final_trace = split_trace(row.get("final_trace", ""))
        initial_trace = split_trace(row.get("initial_trace", ""))
        label = row.get("label", row.get("case_id", ""))
        meta = (
            f"ΔPPL={float(row.get('avg_delta_ppl') or 0.0):.4f}, "
            f"switch={row.get('combo_switches')}, ext={row.get('extended')}"
        )
        parts.append(svg_text(24, y + 18, label[:34], 12, "700"))
        parts.append(svg_text(24, y + 34, meta, 10, "400", "#666"))
        for block_idx in range(max_blocks):
            x = left + block_idx * cell_w
            parts.append(f'<rect x="{x}" y="{y}" width="{cell_w - 5}" height="{cell_h}" rx="4" fill="#f4f4f4" stroke="#dddddd"/>')
            if block_idx >= len(final_trace):
                continue
            combo = final_trace[block_idx]
            initial = initial_trace[block_idx] if block_idx < len(initial_trace) else ""
            changed = bool(initial and initial != combo)
            fill = color_by_combo.get(combo, "#999999")
            stroke = "#111111" if changed else "#ffffff"
            stroke_width = 2 if changed else 1
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell_w - 5}" height="{cell_h}" rx="4" '
                f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}">'
                f'<title>{html.escape(label)} block {block_idx}: final {combo}'
                f'{f" (initial {initial})" if initial else ""}</title></rect>'
            )
            parts.append(svg_text(x + 5, y + 17, combo, 9, "700", "#ffffff"))

    legend_x = left + max_blocks * cell_w + 28
    legend_y = top
    parts.append(svg_text(legend_x, legend_y - 18, "Policy combo legend", 12, "700"))
    for idx, combo in enumerate(combo_order):
        y = legend_y + idx * 22
        parts.append(f'<rect x="{legend_x}" y="{y}" width="16" height="16" rx="3" fill="{color_by_combo[combo]}"/>')
        parts.append(svg_text(legend_x + 24, y + 12, combo, 11))
    parts.append(svg_text(legend_x, legend_y + len(combo_order) * 22 + 20, "Black border: rescue changed initial choice", 11, "400", "#555"))

    parts.append("</svg>")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_SVG.write_text("\n".join(parts), encoding="utf-8")

    OUT_MD.write_text(
        "\n".join(
            [
                "# PCIC Policy Trace Figure（2026-06-29）",
                "",
                f"SVG：`figures/{OUT_SVG.name}`",
                "",
                "该图由 `scripts/render_pcic_policy_trace_svg.py` 生成，读取 `docs/pcic_blockwise_policy_trace_2026_06_29.csv`。",
                "",
                "图中每一行是一个任务/消融 case，每个方块是一个 block 的最终 selected combo。",
                "黑色描边表示最终 combo 与 initial short-horizon choice 不同，即 rescue gate 改写了早期选择。",
                "",
                "论文用途：",
                "",
                "- 支撑 `online blockwise policy selection` 的动态性；",
                "- 展示 Hard-topic / RULER-style variable/topic 中的非平凡 policy switch；",
                "- 作为 Figure 2 的草图，后续可替换成更正式的 matplotlib / LaTeX 图。",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(OUT_SVG)
    print(OUT_MD)


if __name__ == "__main__":
    main()
