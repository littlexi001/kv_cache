from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


BANDS = [f"F{start}-{start + 7}" for start in range(0, 64, 8)]
GROUPS = [f"G{group}" for group in range(8)]


def register_fonts() -> tuple[str, str]:
    candidates = [
        (Path(r"C:\Windows\Fonts\Deng.ttf"), Path(r"C:\Windows\Fonts\Dengb.ttf")),
        (Path(r"C:\Windows\Fonts\simhei.ttf"), Path(r"C:\Windows\Fonts\simhei.ttf")),
    ]
    for regular, bold in candidates:
        if regular.exists() and bold.exists():
            pdfmetrics.registerFont(TTFont("CJK", str(regular)))
            pdfmetrics.registerFont(TTFont("CJK-Bold", str(bold)))
            return "CJK", "CJK-Bold"
    return "Helvetica", "Helvetica-Bold"


def blend(base: colors.Color, strength: float) -> colors.Color:
    strength = max(0.0, min(1.0, strength))
    return colors.Color(
        1.0 - (1.0 - base.red) * strength,
        1.0 - (1.0 - base.green) * strength,
        1.0 - (1.0 - base.blue) * strength,
    )


def cell_color(relative_change: float) -> colors.Color:
    magnitude = abs(relative_change)
    if magnitude < 0.1:
        return colors.HexColor("#E8E8E8")
    strength = 0.25 if magnitude < 1.0 else 0.48 if magnitude < 5.0 else 0.75
    base = colors.HexColor("#5DAAF2") if relative_change < 0 else colors.HexColor("#EE8A4B")
    return blend(base, strength)


def footer(canvas: Any, document: Any, regular_font: str) -> None:
    canvas.saveState()
    canvas.setFont(regular_font, 8)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawString(14 * mm, 8 * mm, "Qwen3-8B · RULER-32K · single-layer RoPE frequency ablation")
    canvas.drawRightString(landscape(A4)[0] - 14 * mm, 8 * mm, f"{document.page}")
    canvas.restoreState()


def build_pdf(payload: dict[str, Any], output: Path) -> None:
    regular_font, bold_font = register_fonts()
    page_size = landscape(A4)
    document = SimpleDocTemplate(
        str(output),
        pagesize=page_size,
        leftMargin=14 * mm,
        rightMargin=14 * mm,
        topMargin=12 * mm,
        bottomMargin=14 * mm,
        title="逐层 RoPE Head × Frequency：Gold PPL",
        author="Codex",
    )
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "CJKTitle",
        parent=styles["Title"],
        fontName=bold_font,
        fontSize=22,
        leading=28,
        alignment=TA_LEFT,
        textColor=colors.HexColor("#202124"),
        spaceAfter=8 * mm,
    )
    heading = ParagraphStyle(
        "CJKHeading",
        parent=styles["Heading2"],
        fontName=bold_font,
        fontSize=16,
        leading=20,
        textColor=colors.HexColor("#202124"),
        spaceAfter=4 * mm,
    )
    body = ParagraphStyle(
        "CJKBody",
        parent=styles["BodyText"],
        fontName=regular_font,
        fontSize=10.5,
        leading=17,
        textColor=colors.HexColor("#303134"),
        spaceAfter=3 * mm,
    )
    centered = ParagraphStyle(
        "CJKCentered",
        parent=body,
        fontName=regular_font,
        fontSize=9.5,
        leading=12,
        alignment=TA_CENTER,
        spaceAfter=0,
    )
    centered_bold = ParagraphStyle(
        "CJKCenteredBold",
        parent=centered,
        fontName=bold_font,
    )

    cells = payload["cells"]
    native_ppls = [cell["gold_ppl"] / (1.0 + cell["gold_ppl_relative_change_percent"] / 100.0) for cell in cells]
    native_ppl = sum(native_ppls) / len(native_ppls)
    min_cell = min(cells, key=lambda cell: cell["gold_ppl"])
    max_cell = max(cells, key=lambda cell: cell["gold_ppl"])

    story: list[Any] = [
        Paragraph("逐层 RoPE Head × Frequency：Gold PPL", title),
        Paragraph(
            "设置：Qwen3-8B，6 条固定 RULER-32K 发现样本。每次只在一个层、一个 KV Head 组和一个连续 8 频率对的频带中移除 RoPE 旋转，其余位置保持原生 RoPE。",
            body,
        ),
        Paragraph(
            "每格第一行是干预后的 Gold answer PPL；第二行是相对原生 RoPE 的变化率。PPL 越低越好：蓝色表示改善，橙色表示退化，灰色表示变化小于 0.1%。",
            body,
        ),
        Spacer(1, 4 * mm),
    ]

    summary_data = [
        [Paragraph("原生 RoPE PPL", centered_bold), Paragraph("扫描最小 PPL", centered_bold), Paragraph("扫描最大 PPL", centered_bold), Paragraph("配置数", centered_bold)],
        [
            Paragraph(f"{native_ppl:.3f}", centered),
            Paragraph(f"{min_cell['gold_ppl']:.3f}<br/><font size='8'>L{min_cell['layer']} / G{min_cell['head_group']} / {min_cell['frequency_band']}</font>", centered),
            Paragraph(f"{max_cell['gold_ppl']:.3f}<br/><font size='8'>L{max_cell['layer']} / G{max_cell['head_group']} / {max_cell['frequency_band']}</font>", centered),
            Paragraph(f"{len(cells):,}", centered),
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
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.extend([summary_table, Spacer(1, 7 * mm)])
    story.append(Paragraph("页面结构", heading))
    story.append(
        Paragraph(
            "后续每页对应一个具体层 L18-L35。横轴是 KV Head 组 G0-G7，分别对应 Query Heads Q0-3、Q4-7、…、Q28-31；纵轴是八个连续 RoPE 频带。",
            body,
        )
    )

    by_layer: dict[int, dict[tuple[str, int], dict[str, Any]]] = defaultdict(dict)
    for cell in cells:
        by_layer[int(cell["layer"])][(cell["frequency_band"], int(cell["head_group"]))] = cell

    for layer in sorted(by_layer):
        story.append(PageBreak())
        story.append(Paragraph(f"L{layer} · Gold PPL", heading))
        table_data: list[list[Any]] = [
            [Paragraph("频带 / Head", centered_bold)]
            + [Paragraph(f"{group}<br/><font size='7'>Q{4 * index}-{4 * index + 3}</font>", centered_bold) for index, group in enumerate(GROUPS)]
        ]
        cell_styles: list[tuple[Any, ...]] = []
        for row_index, band in enumerate(BANDS, start=1):
            row: list[Any] = [Paragraph(band, centered_bold)]
            for group in range(8):
                cell = by_layer[layer][(band, group)]
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
