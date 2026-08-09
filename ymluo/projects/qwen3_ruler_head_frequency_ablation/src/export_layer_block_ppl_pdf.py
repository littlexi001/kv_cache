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
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from export_single_layer_ppl_pdf import BANDS, GROUPS, cell_color, footer, register_fonts


BLOCKS = ((18, 23), (24, 29), (30, 35))


def aggregate(payload: dict[str, Any]) -> tuple[float, list[dict[str, Any]]]:
    cells = payload["cells"]
    native_estimates = [
        float(cell["gold_ppl"]) / (1.0 + float(cell["gold_ppl_relative_change_percent"]) / 100.0)
        for cell in cells
    ]
    native_ppl = mean(native_estimates)
    grouped: dict[tuple[int, int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        layer = int(cell["layer"])
        for block_start, block_end in BLOCKS:
            if block_start <= layer <= block_end:
                grouped[(block_start, block_end, int(cell["head_group"]), str(cell["frequency_band"]))].append(cell)
                break

    result: list[dict[str, Any]] = []
    for (block_start, block_end, group, band), values in sorted(grouped.items()):
        if len(values) != 6:
            raise ValueError(f"expected six layers for {block_start}-{block_end}/G{group}/{band}, got {len(values)}")
        mean_improvement = mean(float(value["gold_nll_improvement"]) for value in values)
        aggregate_ppl = native_ppl * math.exp(-mean_improvement)
        relative_change = 100.0 * (aggregate_ppl / native_ppl - 1.0)
        result.append(
            {
                "block_start": block_start,
                "block_end": block_end,
                "head_group": group,
                "frequency_band": band,
                "gold_nll_improvement": mean_improvement,
                "gold_ppl": aggregate_ppl,
                "gold_ppl_relative_change_percent": relative_change,
            }
        )
    if len(result) != 192:
        raise ValueError(f"expected 192 aggregate cells, got {len(result)}")
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
        title="层块聚合 RoPE Head × Frequency：Gold PPL",
        author="Codex",
    )
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "AggregateTitle",
        parent=styles["Title"],
        fontName=bold_font,
        fontSize=22,
        leading=28,
        alignment=TA_LEFT,
        textColor=colors.HexColor("#202124"),
        spaceAfter=8 * mm,
    )
    heading = ParagraphStyle(
        "AggregateHeading",
        parent=styles["Heading2"],
        fontName=bold_font,
        fontSize=16,
        leading=20,
        textColor=colors.HexColor("#202124"),
        spaceAfter=4 * mm,
    )
    body = ParagraphStyle(
        "AggregateBody",
        parent=styles["BodyText"],
        fontName=regular_font,
        fontSize=10.5,
        leading=17,
        textColor=colors.HexColor("#303134"),
        spaceAfter=3 * mm,
    )
    centered = ParagraphStyle(
        "AggregateCentered",
        parent=body,
        fontName=regular_font,
        fontSize=9.5,
        leading=12,
        alignment=TA_CENTER,
        spaceAfter=0,
    )
    centered_bold = ParagraphStyle(
        "AggregateCenteredBold",
        parent=centered,
        fontName=bold_font,
    )

    min_cell = min(cells, key=lambda cell: cell["gold_ppl"])
    max_cell = max(cells, key=lambda cell: cell["gold_ppl"])
    story: list[Any] = [
        Paragraph("层块聚合 RoPE Head × Frequency：Gold PPL", title),
        Paragraph(
            "设置：Qwen3-8B，6 条固定 RULER-32K 发现样本。L18-L35 被分成三个连续的六层区域；每个单层配置只在一个层、一个 KV Head 组和一个连续 8 频率对的频带中移除 RoPE 旋转。",
            body,
        ),
        Paragraph(
            "聚合定义：先对同一层块内六个单层干预的 Gold NLL 求均值，再取指数得到几何平均 PPL。它反映该 Head-频带在一个层区域内的平均单层趋势，不是同时干预六层的联合实验。",
            body,
        ),
        Paragraph(
            "每格第一行是聚合 Gold PPL，第二行是相对原生 RoPE 的变化率。PPL 越低越好：蓝色表示改善，橙色表示退化，灰色表示变化小于 0.1%。",
            body,
        ),
        Spacer(1, 4 * mm),
    ]

    def label(cell: dict[str, Any]) -> str:
        return f"L{cell['block_start']}-{cell['block_end']} / G{cell['head_group']} / {cell['frequency_band']}"

    summary_data = [
        [Paragraph("原生 RoPE PPL", centered_bold), Paragraph("聚合最小 PPL", centered_bold), Paragraph("聚合最大 PPL", centered_bold), Paragraph("聚合单元", centered_bold)],
        [
            Paragraph(f"{native_ppl:.3f}", centered),
            Paragraph(f"{min_cell['gold_ppl']:.3f}<br/><font size='8'>{label(min_cell)}</font>", centered),
            Paragraph(f"{max_cell['gold_ppl']:.3f}<br/><font size='8'>{label(max_cell)}</font>", centered),
            Paragraph(f"{len(cells)}", centered),
        ],
    ]
    summary_table = Table(summary_data, colWidths=[47 * mm] * 4, rowHeights=[10 * mm, 19 * mm])
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EAED")),
                ("BOX", (0, 0), (-1, -1), 0.7, colors.HexColor("#B7BBC1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D0D3D7")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    story.extend([summary_table, Spacer(1, 7 * mm)])
    story.append(Paragraph("页面结构", heading))
    story.append(
        Paragraph(
            "后续三页分别对应 L18-L23、L24-L29 和 L30-L35。横轴为 KV Head 组 G0-G7，纵轴为八个连续 RoPE 频带。",
            body,
        )
    )

    indexed = {
        (int(cell["block_start"]), int(cell["block_end"]), str(cell["frequency_band"]), int(cell["head_group"])): cell
        for cell in cells
    }
    for block_start, block_end in BLOCKS:
        story.append(PageBreak())
        story.append(Paragraph(f"L{block_start}-L{block_end} · 聚合 Gold PPL", heading))
        table_data: list[list[Any]] = [
            [Paragraph("频带 / Head", centered_bold)]
            + [Paragraph(f"{group}<br/><font size='7'>Q{4 * index}-{4 * index + 3}</font>", centered_bold) for index, group in enumerate(GROUPS)]
        ]
        cell_styles: list[tuple[Any, ...]] = []
        for row_index, band in enumerate(BANDS, start=1):
            row: list[Any] = [Paragraph(band, centered_bold)]
            for group in range(8):
                cell = indexed[(block_start, block_end, band, group)]
                relative = float(cell["gold_ppl_relative_change_percent"])
                sign = "+" if relative > 0 else ""
                row.append(
                    Paragraph(
                        f"<b>{cell['gold_ppl']:.1f}</b><br/><font size='8'>{sign}{relative:.2f}%</font>",
                        centered,
                    )
                )
                cell_styles.append(("BACKGROUND", (group + 1, row_index), (group + 1, row_index), cell_color(relative)))
            table_data.append(row)

        available_width = page_size[0] - document.leftMargin - document.rightMargin
        table = Table(
            table_data,
            colWidths=[26 * mm] + [(available_width - 26 * mm) / 8] * 8,
            rowHeights=[16 * mm] + [12.5 * mm] * 8,
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
                Spacer(1, 4 * mm),
                Paragraph(
                    "蓝色：PPL 下降（改善）　　灰色：|变化| &lt; 0.1%　　橙色：PPL 上升（退化）",
                    centered,
                ),
            ]
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    document.build(
        story,
        onFirstPage=lambda canvas, doc: footer(canvas, doc, regular_font),
        onLaterPages=lambda canvas, doc: footer(canvas, doc, regular_font),
    )


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
