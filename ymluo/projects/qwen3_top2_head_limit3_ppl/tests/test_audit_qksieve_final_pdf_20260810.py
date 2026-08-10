from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "papers" / "countcap_iclr2027" / "scripts" / "audit_qksieve_final_pdf.py"
SPEC = importlib.util.spec_from_file_location("qksieve_final_pdf_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def english_pages(*, author: bool = False, reference_page: int = 10) -> list[str]:
    pages = [f"Body page {index}" for index in range(1, 10)] + ["Appendix"]
    pages[reference_page - 1] = "References\nPaper A"
    if author:
        pages[0] += "\nYiming Luo\nFudan University"
    return pages


def test_final_pdf_audit_accepts_page_10_references_and_anonymity() -> None:
    anonymous = MODULE.audit_english_pages(english_pages(), anonymous=True)
    author = MODULE.audit_english_pages(
        english_pages(author=True), anonymous=False
    )

    assert anonymous["references_start_page"] == 10
    assert anonymous["identity_markers"] == []
    assert author["identity_markers"] == ["Yiming Luo", "Fudan University"]


def test_final_pdf_audit_rejects_body_overflow() -> None:
    with pytest.raises(AssertionError, match="page 10"):
        MODULE.audit_english_pages(
            english_pages(reference_page=9), anonymous=True
        )


def test_final_pdf_audit_rejects_anonymous_identity_and_placeholders() -> None:
    leaked = english_pages(author=True)
    with pytest.raises(AssertionError, match="leaks author"):
        MODULE.audit_english_pages(leaked, anonymous=True)

    placeholder = english_pages()
    placeholder[3] += "\nTBD"
    with pytest.raises(AssertionError, match="placeholder"):
        MODULE.audit_english_pages(placeholder, anonymous=True)
