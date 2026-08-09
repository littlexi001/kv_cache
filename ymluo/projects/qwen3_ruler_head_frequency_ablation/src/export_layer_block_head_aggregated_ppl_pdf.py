from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from export_layer_block_ppl_pdf import BLOCKS
from export_single_layer_ppl_pdf import BANDS, cell_color, footer, register_fonts


def aggregate(payload: dict[str, Any]) -> tuple[float, list[dict[str, Any]]]:
    cells = payload["cells"]
    native_ppl = mean(
        float(cell["gold_ppl"]) / (1.0 + float(cell["gold_ppl_relative_change_percent"]) / 100.0)
        for cell in cells
    )
    grouped: dict[tuple[int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        layer = int(cell["layer"])
        for block_start, block_end in BLOCKS:
            if block_start <= layer <= block_end:
                grouped[(block_start, block_end, str(cell["frequency_band"]))].append(cell)
                break

    result: list[dict[str, Any]] = []
    for (block_start, block_end, band), values in sorted(grouped.items()):
        if len(values) != 48:
            raise ValueError(f"expected 48 cells for L{block_start}-L{block_end}/{band}, got {len(values)}")
        mean_improvement = mean(float(value["gold_nll_improvement"]) for value in values)
        aggregate_ppl = native_ppl * math.exp(-mean_improvement)
        result.append(
            {
                "block_start": block_start,
                "block_end": block_end,
                "frequency_band": band,
                "gold_nll_improvement": mean_improvement,
                "gold_ppl": aggregate_ppl,
                "gold_ppl_relative_change_percent": 100.0 * (aggregate_ppl / native_ppl - 1.0),
            }
        )
    if len(result) != 24:
        raise ValueError(f"expected 24 aggregate cells, got {len(result)}")
    return native_ppl, result


def build_pdf(payload: dict[str, Any], output: Path) -> None:
    regular_font, bold_font = register_fonts()
    native_ppl, cells = aggregate(payload)
    page_size = landscape(A4)
    document = SimpleDocTemplate(
        str(output),
        pagesize=page_size,
        leftMargin=14 * mm,
        rightMargin=14 * mm,
        topMargin=12 * mm,
        bottomMargin=14 * mm,
        title="层块与 Head 聚合 RoPE Frequency：Gold PPL",
        author="Codex",
    )
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "HeadAggregateTitle",
        parent=styles["Title"],
        fontName=bold_font,
        fontSize=22,
        leading=28,
        alignment=TA_LEFT,
        textColor=colors.HexColor("#202124"),
        spaceAfter=6 * mm,
    )
    body = ParagraphStyle(
        "HeadAggregateBody",
        parent=styles["BodyText"],
        fontName=regular_font,
        fontSize=10.5,
        leading=17,
        textColor=colors.HexColor("#303134"),
        spaceAfter=3 * mm,
    )
    centered = ParagraphStyle(
        "HeadAggregateCentered",
        parent=body,
        fontName=regular_font,
        fontSize=9.5,
        leading=12,
        alignment=TA_CENTER,
        spaceAfter=0,
    )
    centered_bold = ParagraphStyle(
        "HeadAggregateCenteredBold",
        parent=centered,
        fontName=bold_font,
    )

    indexed = {
        (int(cell["block_start"]), int(cell["block_end"]), str(cell["frequency_band"])): cell
        for cell in cells
    }
    table_data: list[list[Any]] = [
        [Paragraph("层块 / 频带", centered_bold)]
        + [Paragraph(band, centered_bold) for band in BANDS]
    ]
    cell_styles: list[tuple[Any, ...]] = []
    for row_index, (block_start, block_end) in enumerate(BLOCKS, start=1):
        row: list[Any] = [Paragraph(f"L{block_start}-L{block_end}", centered_bold)]
        for column_index, band in enumerate(BANDS, start=1):
            cell = indexed[(block_start, block_end, band)]
            relative = float(cell["gold_ppl_relative_change_percent"])
            sign = "+" if relative > 0 else ""
            row.append(
                Paragraph(
                    f"<b>{cell['gold_ppl']:.2f}</b><br/><font size='8'>{sign}{relative:.2f}%</font>",
                    centered,
                )
            )
            cell_styles.append(("BACKGROUND", (column_index, row_index), (column_index, row_index), cell_color(relative)))
        table_data.append(row)

    min_cell = min(cells, key=lambda cell: cell["gold_ppl"])
    max_cell = max(cells, key=lambda cell: cell["gold_ppl"])
    story: list[Any] = [
        Paragraph("层块与 Head 聚合 RoPE Frequency：Gold PPL", title),
        Paragraph(
            "聚合定义：对同一层块和频带下的 6 层 × 8 KV Head 组，共 48 个单层干预的 Gold NLL 求均值，再取指数得到几何平均 PPL。该表描述平均单层趋势，不是一次性联合干预 48 个位置。",
            body,
        ),
        Paragraph(
            f"原生 RoPE PPL 为 {native_ppl:.3f}。最低聚合 PPL 为 {min_cell['gold_ppl']:.3f}（L{min_cell['block_start']}-L{min_cell['block_end']} / {min_cell['frequency_band']}）；最高为 {max_cell['gold_ppl']:.3f}（L{max_cell['block_start']}-L{max_cell['block_end']} / {max_cell['frequency_band']}）。",
            body,
        ),
        Paragraph(
            "每格第一行是聚合 Gold PPL，第二行是相对原生 RoPE 的变化率。蓝色表示 PPL 下降，橙色表示 PPL 上升，灰色表示变化小于 0.1%。",
            body,
        ),
        Spacer(1, 5 * mm),
    ]
    available_width = page_size[0] - document.leftMargin - document.rightMargin
    table = Table(
        table_data,
        colWidths=[28 * mm] + [(available_width - 28 * mm) / 8] * 8,
        rowHeights=[15 * mm] + [18 * mm] * 3,
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EAED")),
                ("BACKGROUND", (0, 1), (0, -1), colors.HexColor("#F1F3F4")),
                ("BOX", (0, 0), (-1, -1), 0.7, colors.HexColor("#9AA0A6")),
                ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#C8CCD0")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                *cell_styles,
            ]
        )
    )
    story.extend(
        [
            table,
            Spacer(1, 5 * mm),
            Paragraph(
                "蓝色：PPL 下降（改善）　　灰色：|变化| &lt; 0.1%　　橙色：PPL 上升（退化）",
                centered,
            ),
        ]
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    document.build(story, onFirstPage=lambda canvas, doc: footer(canvas, doc, regular_font))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.data.read_text(encoding="utf-8"))
    build_pdf(payload, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
