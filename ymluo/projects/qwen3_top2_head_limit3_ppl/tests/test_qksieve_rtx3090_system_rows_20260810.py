from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    ROOT
    / "papers"
    / "countcap_iclr2027"
    / "scripts"
    / "make_qksieve_rtx3090_system_rows.py"
)
SPEC = importlib.util.spec_from_file_location("qksieve_rtx3090_rows", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_raw_rtx3090_evidence_generates_frozen_paper_rows() -> None:
    attention, attention_paths = MODULE.attention_evidence()
    decode, decode_paths = MODULE.decode_evidence()
    persistent, persistent_path = MODULE.persistent_evidence()

    assert len(attention_paths) == 3
    assert len(decode_paths) == 9
    assert persistent_path.exists()
    assert [int(row["history_tokens"]) for row in attention] == list(MODULE.LENGTHS)
    assert [int(row["history_tokens"]) for row in decode] == list(
        MODULE.DECODE_LENGTHS
    )

    row128 = attention[-1]
    assert round(row128["fast_speedup"], 2) == 6.37
    assert round(row128["robust_speedup"], 2) == 4.12
    assert round(row128["robust_vs_fier"], 2) == 2.31

    decode128 = decode[-1]
    assert round(decode128["fast_speedup"], 2) == 4.84
    assert round(decode128["robust_speedup"], 2) == 3.98
    assert int(decode128["robust_break_even_tokens"]) == 10
    assert round(decode128["robust_online64_speedup"], 2) == 2.15

    text = MODULE.render(
        attention,
        decode,
        persistent,
        provenance="test-sha",
    )
    assert "128K & 2.3708 & 0.3722 & 0.5757" in text
    assert "128K & 0.743 & 1.839 & 4 & 10" in text
    assert "64K & 1.125$\\times$ & 2.221$\\times$" in text
    assert "Generated from audited RTX 3090 evidence: test-sha" in text


def test_bilingual_paper_consumes_generated_system_rows() -> None:
    paper = ROOT / "papers" / "countcap_iclr2027"
    for section in (
        paper / "sections" / "05_experiments.tex",
        paper / "sections_zh" / "05_experiments.tex",
    ):
        text = section.read_text(encoding="utf-8")
        assert "\\input{data/generated/qksieve_rtx3090_system_rows.tex}" in text
        for command in (
            "\\QKSieveMhaAttentionRows",
            "\\QKSieveMhaFierRows",
            "\\QKSieveMhaDecodeRows",
            "\\QKSievePersistentRows",
        ):
            assert text.count(command) == 1

    for appendix in (
        paper / "sections" / "appendix_system_diagnostics.tex",
        paper / "sections_zh" / "appendix_system_diagnostics.tex",
    ):
        assert (
            appendix.read_text(encoding="utf-8").count(
                "\\QKSieveMhaBuildBreakEvenRows"
            )
            == 1
        )

    for build in (paper / "build.ps1", paper / "build_zh.ps1"):
        assert "make_qksieve_rtx3090_system_rows.py" in build.read_text(
            encoding="utf-8"
        )
