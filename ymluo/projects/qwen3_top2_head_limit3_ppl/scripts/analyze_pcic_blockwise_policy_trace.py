#!/usr/bin/env python3
from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
OUT_CSV = DOCS / "pcic_blockwise_policy_trace_2026_06_29.csv"
OUT_MD = DOCS / "pcic_blockwise_policy_trace_2026_06_29.md"


def read_csv_auto(path: Path) -> list[dict[str, str]]:
    data = path.read_bytes()
    encoding = "utf-16" if data[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8-sig"
    return list(csv.DictReader(io.StringIO(data.decode(encoding))))


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def inum(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def fmt(value: float) -> str:
    if abs(value) < 0.5e-6:
        value = 0.0
    return f"{value:.6f}"


def summarize_rows(source: str, case_id: str, label: str, rows: list[dict[str, str]]) -> dict[str, Any]:
    rows = sorted(rows, key=lambda row: inum(row.get("block")))
    combos = [str(row.get("combo") or row.get("final_combo") or "") for row in rows]
    initial = [str(row.get("early_selected") or row.get("initial_combo") or "") for row in rows]
    deltas = [fnum(row.get("delta_ppl")) for row in rows]
    extended = sum(inum(row.get("extended")) for row in rows)
    early = sum(inum(row.get("early")) for row in rows)
    changed = sum(1 for a, b in zip(initial, combos) if a and b and a != b)
    no_change_ext = sum(
        1
        for row in rows
        if inum(row.get("extended")) and inum(row.get("no_change_after_extension"))
    )
    extension_seconds = sum(fnum(row.get("extension_seconds")) for row in rows)
    avoidable_seconds = sum(
        fnum(row.get("extension_seconds"))
        for row in rows
        if inum(row.get("extended")) and inum(row.get("no_change_after_extension"))
    )
    switches = sum(1 for prev, cur in zip(combos, combos[1:]) if prev != cur)
    return {
        "source": source,
        "case_id": case_id,
        "label": label,
        "blocks": len(rows),
        "avg_delta_ppl": sum(deltas) / len(deltas) if deltas else 0.0,
        "unique_final_combos": len(set(combos)),
        "combo_switches": switches,
        "extended": extended,
        "early": early,
        "initial_final_changes": changed,
        "no_change_extensions": no_change_ext,
        "extension_seconds": extension_seconds,
        "avoidable_extension_seconds": avoidable_seconds,
        "avoidable_extension_fraction": (avoidable_seconds / extension_seconds) if extension_seconds else 0.0,
        "final_trace": "/".join(combos),
        "initial_trace": "/".join(x for x in initial if x),
    }


def main() -> None:
    summaries: list[dict[str, Any]] = []

    hard_rows = read_csv_auto(DOCS / "pcic_hardtopic_b8_condautoanchor_blocks_2026_06_29.csv")
    hard_labels = {
        "b4_top2": "Hard-topic b4 top2",
        "b4_cond_auto_anchor": "Hard-topic b4 conditional rescue",
        "b8_top2": "Hard-topic b8 top2",
        "b8_cond_auto_anchor": "Hard-topic b8 conditional rescue",
    }
    for case_id, label in hard_labels.items():
        rows = [row for row in hard_rows if row.get("case_id") == case_id]
        if rows:
            summaries.append(summarize_rows("hardtopic_b8_blocks", case_id, label, rows))

    needle_rows = read_csv_auto(DOCS / "pcic_needle_smoke_blocks_2026_06_29.csv")
    needle_labels = {
        "needle_top2": "Needle-style top2",
        "needle_cond_auto_anchor": "Needle-style conditional rescue",
    }
    for case_id, label in needle_labels.items():
        rows = [row for row in needle_rows if row.get("case_id") == case_id]
        if rows:
            summaries.append(summarize_rows("needle_blocks", case_id, label, rows))

    waste_rows = read_csv_auto(DOCS / "pcic_extension_waste_blocks_2026_06_29.csv")
    waste_labels = {
        "hard": "Hard-topic conditional rescue",
        "war": "War easy regime",
        "monte": "Monte online selection",
        "needle": "Needle-style conditional rescue",
        "ruler_multineedle": "RULER-style multi-needle",
        "ruler_variable": "RULER-style variable binding",
        "ruler_topicswitch": "RULER-style topic switch",
    }
    for case_id, label in waste_labels.items():
        rows = [row for row in waste_rows if row.get("case_id") == case_id]
        if rows:
            summaries.append(summarize_rows("extension_waste_blocks", case_id, label, rows))

    fieldnames = list(summaries[0].keys())
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)

    lines: list[str] = []
    lines.append("# PCIC Blockwise Policy Trace Evidence（2026-06-29）")
    lines.append("")
    lines.append("目的：补强 paper 主线中的动态性证据，证明方法不是固定 sparse operator，而是在 block 级别做 online policy selection；同时量化 rescue gate 的必要性和浪费空间。")
    lines.append("")
    lines.append("本分析只读取已有本地 CSV，不跑模型、不访问服务器、不下载数据。")
    lines.append("")
    lines.append("## Trace 总表")
    lines.append("")
    lines.append("| case | blocks | avg ΔPPL | unique combos | switches | extended | initial→final changes | avoidable ext frac | final trace |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for row in summaries:
        lines.append(
            "| {label} | {blocks} | {avg} | {uniq} | {switches} | {extended} | {changed} | {avoidable:.3f} | `{trace}` |".format(
                label=row["label"],
                blocks=row["blocks"],
                avg=fmt(row["avg_delta_ppl"]),
                uniq=row["unique_final_combos"],
                switches=row["combo_switches"],
                extended=row["extended"],
                changed=row["initial_final_changes"],
                avoidable=row["avoidable_extension_fraction"],
                trace=row["final_trace"],
            )
        )

    lines.append("")
    lines.append("## 对创新性的直接支撑")
    lines.append("")
    lines.append("- **动态策略选择存在**：Hard-topic b8 conditional rescue 使用 4 个不同 final combos，发生 6 次 block-to-block switch；RULER-style variable/topic 也从 `0,13` 切到 `2,0`。")
    lines.append("- **rescue gate 不是装饰项**：Hard-topic conditional rescue 中 initial→final changes 为 2，说明 early top choice 会被更长 horizon rescue 改写。")
    lines.append("- **不同任务选择不同策略**：War 固定为 `0,7`，Monte 为 `2,7/2,0`，RULER multi-needle 固定为 `2,0`，variable/topic 出现 `0,13→2,0`。这比固定 qabs/SparQ-like operator 更像 online policy selection。")
    lines.append("- **速度瓶颈有明确方向**：extension waste 表明部分 extension 后 final combo 不变，存在 calibrated skip-gate 空间；但当前主速度路线仍应是 fused/sparse candidate probe。")
    lines.append("")
    lines.append("## 仍然不足")
    lines.append("")
    lines.append("- 这些 trace 主要来自 hard-topic、needle-style、RULER-style synthetic/offline smoke，还不能替代正式 LongBench/RULER。")
    lines.append("- 当前 trace 证明“会切换”，但还需要把切换和文本结构、retrieval/variable binding failure 对齐，形成 paper figure。")
    lines.append("- corrected gate 成本仍高；创新性主张可以推进，速度主张必须等 fused probe 或真实 kernel 结果。")
    lines.append("")
    lines.append("## 下一步建议")
    lines.append("")
    lines.append("1. 用同一脚本接入正式 benchmark 的 blockwise CSV，生成 standard trace table。")
    lines.append("2. 增加每个 block 的文本摘要/任务位置，画 policy trace figure。")
    lines.append("3. 对 `initial→final changes` 的 block 做 case study，解释 rescue gate 修复了什么短视错误。")
    lines.append("")
    lines.append(f"CSV：`docs/{OUT_CSV.name}`")
    lines.append("")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
