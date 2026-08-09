from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


PROJECT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT / "outputs" / "remote_extract"
OUTPUT = PROJECT / "output" / "pdf" / "rope_frequency_deletion_comparison.pdf"


def register_fonts() -> tuple[str, str]:
    candidates = [
        (Path(r"C:\Windows\Fonts\Deng.ttf"), Path(r"C:\Windows\Fonts\Dengb.ttf")),
        (Path(r"C:\Windows\Fonts\msyh.ttc"), Path(r"C:\Windows\Fonts\msyhbd.ttc")),
        (Path(r"C:\Windows\Fonts\simhei.ttf"), Path(r"C:\Windows\Fonts\simhei.ttf")),
    ]
    for regular, bold in candidates:
        if regular.exists() and bold.exists():
            pdfmetrics.registerFont(TTFont("CJK", str(regular)))
            pdfmetrics.registerFont(TTFont("CJK-Bold", str(bold)))
            return "CJK", "CJK-Bold"
    return "Helvetica", "Helvetica-Bold"


def load_rows() -> list[dict[str, Any]]:
    validation = json.loads((RESULTS / "validation_26_summary.json").read_text(encoding="utf-8"))
    ridge = json.loads((RESULTS / "ridge_validation_summary.json").read_text(encoding="utf-8"))
    discovery = json.loads((RESULTS / "discovery_6_summary.json").read_text(encoding="utf-8"))

    sources = {
        "validation": {row["variant"]: row for row in validation},
        "ridge": {row["variant"]: row for row in ridge},
        "discovery": {row["variant"]: row for row in discovery},
    }
    native_26 = next(row for row in validation if row["variant"] == "native_rope")
    native_6 = next(row for row in discovery if row["variant"] == "native_rope")

    specs = [
        {
            "label": "全层删除 F0-F7",
            "type": "高频",
            "source": "validation",
            "variant": "global_f00_07_delete",
            "native": native_26,
            "evidence": "26 样本\n验证集",
            "short": "未跑 WikiText\nRULER 严重退化",
        },
        {
            "label": "L24-L35 删除 F0-F11",
            "type": "高频扩展",
            "source": "ridge",
            "variant": "late_l24_f00_11_delete",
            "native": native_26,
            "evidence": "26 样本\n验证集",
            "short": "2K -0.95% / 4K -0.55%\n8K -0.11%",
        },
        {
            "label": "L30-L35 删除 F0-F15",
            "type": "高频扩展",
            "source": "ridge",
            "variant": "late_l30_f00_15_delete",
            "native": native_26,
            "evidence": "26 样本\n验证集",
            "short": "2K -0.05% / 4K +0.37%\n8K +0.73%",
        },
        {
            "label": "L18-L35 删除 F8-F15",
            "type": "中高频",
            "source": "validation",
            "variant": "deep_f08_15_delete",
            "native": native_26,
            "evidence": "26 样本\n验证集",
            "short": "未测",
        },
        {
            "label": "L18-L35 删除 F16-F23",
            "type": "中频",
            "source": "discovery",
            "variant": "deep_f16_23_delete",
            "native": native_6,
            "evidence": "6 样本\n发现集",
            "short": "未测",
        },
        {
            "label": "L18-L35 删除 F56-F63",
            "type": "最低频",
            "source": "discovery",
            "variant": "deep_f56_63_delete",
            "native": native_6,
            "evidence": "6 样本\n发现集",
            "short": "未测",
        },
    ]

    rows: list[dict[str, Any]] = []
    for spec in specs:
        row = sources[spec["source"]][spec["variant"]]
        rows.append({**spec, "result": row})
    return rows


def sign(value: float, digits: int = 2) -> str:
    return f"{value:+.{digits}f}"


def metric_fill(value: float, positive_is_good: bool = True) -> colors.Color:
    adjusted = value if positive_is_good else -value
    if abs(adjusted) < 1e-12:
        return colors.HexColor("#F1F3F4")
    if adjusted > 0:
        return colors.HexColor("#E5F4EC")
    return colors.HexColor("#FCE8E6")


def build_pdf() -> Path:
    regular_font, bold_font = register_fonts()
    rows = load_rows()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    page_size = landscape(A4)
    document = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=page_size,
        leftMargin=11 * mm,
        rightMargin=11 * mm,
        topMargin=9 * mm,
        bottomMargin=10 * mm,
        title="Qwen3-8B RoPE 频带删除对照",
        author="Codex",
    )

    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "TitleCJK",
        parent=styles["Title"],
        fontName=bold_font,
        fontSize=21,
        leading=26,
        alignment=TA_LEFT,
        textColor=colors.HexColor("#172B4D"),
        spaceAfter=2.5 * mm,
    )
    subtitle = ParagraphStyle(
        "SubtitleCJK",
        parent=styles["BodyText"],
        fontName=regular_font,
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#4A5568"),
        spaceAfter=4 * mm,
    )
    cell = ParagraphStyle(
        "CellCJK",
        parent=styles["BodyText"],
        fontName=regular_font,
        fontSize=8.6,
        leading=11.5,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#1F2937"),
        spaceAfter=0,
    )
    cell_left = ParagraphStyle("CellLeftCJK", parent=cell, alignment=TA_LEFT)
    cell_bold = ParagraphStyle("CellBoldCJK", parent=cell, fontName=bold_font)
    header = ParagraphStyle(
        "HeaderCJK",
        parent=cell_bold,
        fontSize=8.7,
        leading=11,
        textColor=colors.white,
    )
    note = ParagraphStyle(
        "NoteCJK",
        parent=styles["BodyText"],
        fontName=regular_font,
        fontSize=8.5,
        leading=12.5,
        textColor=colors.HexColor("#4A5568"),
        spaceAfter=0,
    )
    conclusion_title = ParagraphStyle(
        "ConclusionTitleCJK",
        parent=styles["Heading2"],
        fontName=bold_font,
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#0F766E"),
        spaceAfter=1.5 * mm,
    )

    story: list[Any] = [
        Paragraph("Qwen3-8B RoPE 频带删除：高频、中频与低频放在一起比较", title),
        Paragraph(
            "所有配置都作用于全部 8 个 KV Head 组（即 32 个 Query Heads），并把所选二维频率对的 RoPE 旋转完全删除。"
            "RULER 与 Gold 指标均相对同一证据子集上的原生 RoPE 计算。",
            subtitle,
        ),
    ]

    headings = [
        "配置",
        "频率类别",
        "证据",
        "RULER 分数变化",
        "Gold NLL 改善",
        "Gold PPL\n原生 -> 干预",
        "首 token 准确率\n原生 -> 干预",
        "短上下文影响",
    ]
    table_data: list[list[Any]] = [[Paragraph(h.replace("\n", "<br/>"), header) for h in headings]]
    table_styles: list[tuple[Any, ...]] = []

    for index, item in enumerate(rows, start=1):
        result = item["result"]
        native = item["native"]
        ruler_delta_pp = float(result["paired_official_delta"]) * 100
        nll_improvement = float(result["mean_nll_improvement"])
        table_data.append(
            [
                Paragraph(item["label"], cell_left),
                Paragraph(item["type"], cell_bold),
                Paragraph(item["evidence"].replace("\n", "<br/>"), cell),
                Paragraph(f"<b>{sign(ruler_delta_pp)}</b> pp", cell),
                Paragraph(f"<b>{sign(nll_improvement, 3)}</b>", cell),
                Paragraph(
                    f"{native['gold_answer_ppl_from_mean_nll']:.2f} -> <b>{result['gold_answer_ppl_from_mean_nll']:.2f}</b>",
                    cell,
                ),
                Paragraph(
                    f"{native['first_token_accuracy'] * 100:.2f}% -> <b>{result['first_token_accuracy'] * 100:.2f}%</b>",
                    cell,
                ),
                Paragraph(item["short"].replace("\n", "<br/>"), cell),
            ]
        )
        table_styles.extend(
            [
                ("BACKGROUND", (3, index), (3, index), metric_fill(ruler_delta_pp)),
                ("BACKGROUND", (4, index), (4, index), metric_fill(nll_improvement)),
            ]
        )
        if item["evidence"].startswith("6 样本"):
            table_styles.append(("BACKGROUND", (2, index), (2, index), colors.HexColor("#FFF3CD")))
        else:
            table_styles.append(("BACKGROUND", (2, index), (2, index), colors.HexColor("#E8F1FA")))

    available_width = page_size[0] - document.leftMargin - document.rightMargin
    widths_mm = [42, 19, 20, 27, 25, 38, 38, 64]
    scale = available_width / sum(widths_mm) / mm
    col_widths = [width * scale * mm for width in widths_mm]
    table = Table(table_data, colWidths=col_widths, rowHeights=[12 * mm] + [15.5 * mm] * len(rows), repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#243B53")),
                ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#9AA5B1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD2D9")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("ROWBACKGROUNDS", (0, 1), (-1, 4), [colors.white, colors.HexColor("#F8FAFC")]),
                *table_styles,
            ]
        )
    )
    story.extend([table, Spacer(1, 3.2 * mm)])

    conclusion = KeepTogether(
        [
            Paragraph("读表结论", conclusion_title),
            Paragraph(
                "<b>长程分数最高：</b>L30-L35 删除 F0-F15（RULER +6.54 pp）。  "
                "<b>长程与短程最均衡：</b>L24-L35 删除 F0-F11（Gold NLL +0.611，2K/4K/8K PPL 均未退化）。  "
                "<b>频带对照：</b>单独删除 F8-F15 仍有小幅收益；F16-F23 的分数与 NLL 方向不一致；F56-F63 暂无收益。",
                note,
            ),
            Spacer(1, 1.3 * mm),
            Paragraph(
                "注：黄色证据格对应 6 样本发现集，证据强度低于蓝色证据格对应的 26 样本验证集。两组使用的样本不同，因此绝对 PPL 不应跨组直接比较；可以比较每行相对其配对原生 RoPE 的变化。短上下文中的负百分比表示 PPL 下降（改善），正百分比表示 PPL 上升（退化）。",
                note,
            ),
        ]
    )
    story.append(conclusion)

    def footer(canvas: Any, doc: Any) -> None:
        canvas.saveState()
        canvas.setFont(regular_font, 7.5)
        canvas.setFillColor(colors.HexColor("#7B8794"))
        canvas.drawString(11 * mm, 6 * mm, "Qwen3-8B · RULER long-context RoPE frequency deletion")
        canvas.drawRightString(page_size[0] - 11 * mm, 6 * mm, f"{doc.page}")
        canvas.restoreState()

    document.build(story, onFirstPage=footer, onLaterPages=footer)
    return OUTPUT


if __name__ == "__main__":
    print(build_pdf())
