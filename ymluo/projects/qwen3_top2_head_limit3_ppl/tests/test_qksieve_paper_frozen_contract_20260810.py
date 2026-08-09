from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PAPER = ROOT / "papers" / "countcap_iclr2027"


def paper_text(chinese: bool) -> str:
    section_root = PAPER / ("sections_zh" if chinese else "sections")
    paths = [PAPER / ("main_zh.tex" if chinese else "main.tex")]
    paths.extend(sorted(section_root.glob("*.tex")))
    return "\n".join(path.read_text(encoding="utf-8") for path in paths)


def test_english_and_chinese_papers_preserve_frozen_contract() -> None:
    required = (
        r"B(N)=\min\{N,1280,\max(256,\lceil.06N\rceil)\}",
        "at most 512 effective samples",
        "306 bits/token/KV-head",
        "Rank 16, block 256, INT4",
        r"fixed $\alpha=.5$",
        "qksieve\\_qmse\\_oas\\_requestlocal\\_valuesketch16\\_sorted\\_c64",
    )
    english = paper_text(chinese=False)
    for phrase in required:
        assert phrase in english

    chinese = paper_text(chinese=True)
    chinese_required = (
        required[0],
        "有效样本最多 512 个",
        "306 bit/token/KV-head",
        "Rank 16、block 256、INT4",
        r"固定 $\alpha=.5$",
        required[5],
    )
    for phrase in chinese_required:
        assert phrase in chinese


def test_submission_sources_have_no_placeholder_tokens() -> None:
    for chinese in (False, True):
        text = paper_text(chinese=chinese)
        assert not re.search(r"\b(?:TBD|TODO|placeholder)\b", text, re.I)


def test_quality_queue_finishes_all_preregistered_quality_runs() -> None:
    queue = (
        ROOT
        / "projects"
        / "qwen3_top2_head_limit3_ppl"
        / "scripts"
        / "launch_qksieve_quality_evidence_queue_20260810.sh"
    ).read_text(encoding="utf-8")
    ruler_call = queue.index('bash "${RULER}"')
    multimodel_call = queue.index('bash "${MULTIMODEL}"')
    full_longbench_call = queue.index('bash "${FULL_LONGBENCH}"')
    assert ruler_call < multimodel_call < full_longbench_call
    assert "minimum_quality_retention" not in queue
