#!/usr/bin/env python
"""Audit final QKSieve PDFs for page budget, anonymity, and placeholders."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from pypdf import PdfReader


PLACEHOLDER_PATTERN = re.compile(r"\b(?:TBD|TODO|PLACEHOLDER)\b", re.IGNORECASE)
REFERENCE_PATTERN = re.compile(r"(?im)^\s*references\s*$")
AUTHOR_MARKERS = ("Yiming Luo", "Fudan University")
STALE_ENGLISH_MARKERS = (
    "Required Final Experiments",
    "Complete reference-profile LongBench result",
    "Matched H100 measurements remain required",
)
STALE_CHINESE_MARKERS = (
    "正式投稿仍需补齐的实验",
    "完整 reference-profile LongBench 结果",
    "仍需补充 matched H100 测量",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anonymous", required=True, type=Path)
    parser.add_argument("--author", required=True, type=Path)
    parser.add_argument("--chinese", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def extract_pages(path: Path) -> list[str]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise AssertionError(f"missing or empty PDF: {path}")
    reader = PdfReader(str(path))
    if not reader.pages:
        raise AssertionError(f"PDF has no pages: {path}")
    return [(page.extract_text() or "") for page in reader.pages]


def placeholder_pages(pages: list[str]) -> list[int]:
    return [
        index + 1
        for index, text in enumerate(pages)
        if PLACEHOLDER_PATTERN.search(text)
    ]


def reference_page(pages: list[str]) -> int:
    matches = [
        index + 1
        for index, text in enumerate(pages)
        if REFERENCE_PATTERN.search(text)
    ]
    if not matches:
        raise AssertionError("English PDF does not contain a References heading")
    return matches[0]


def stale_markers(pages: list[str], markers: tuple[str, ...]) -> list[str]:
    full_text = "\n".join(pages)
    return [marker for marker in markers if marker in full_text]


def audit_english_pages(
    pages: list[str], *, anonymous: bool
) -> dict[str, Any]:
    if len(pages) < 10:
        raise AssertionError("English paper has fewer than ten total pages")
    references = reference_page(pages)
    if references != 10:
        raise AssertionError(
            f"References must begin on page 10, observed page {references}"
        )
    placeholders = placeholder_pages(pages)
    if placeholders:
        raise AssertionError(f"placeholder text remains on pages {placeholders}")
    stale = stale_markers(pages, STALE_ENGLISH_MARKERS)
    if stale:
        raise AssertionError(f"stale draft claims remain: {', '.join(stale)}")
    full_text = "\n".join(pages)
    identities = [marker for marker in AUTHOR_MARKERS if marker in full_text]
    if anonymous and identities:
        raise AssertionError(
            f"anonymous PDF leaks author identity: {', '.join(identities)}"
        )
    if not anonymous and len(identities) != len(AUTHOR_MARKERS):
        raise AssertionError("author PDF lacks the registered author affiliation")
    return {
        "pages": len(pages),
        "references_start_page": references,
        "anonymous": anonymous,
        "identity_markers": identities,
        "placeholder_pages": placeholders,
        "stale_markers": stale,
    }


def audit_chinese_pages(pages: list[str]) -> dict[str, Any]:
    placeholders = placeholder_pages(pages)
    if placeholders:
        raise AssertionError(
            f"Chinese PDF contains placeholder text on pages {placeholders}"
        )
    stale = stale_markers(pages, STALE_CHINESE_MARKERS)
    if stale:
        raise AssertionError(f"Chinese PDF contains stale draft claims: {stale}")
    return {
        "pages": len(pages),
        "placeholder_pages": placeholders,
        "stale_markers": stale,
    }


def audit(
    anonymous_path: Path,
    author_path: Path,
    chinese_path: Path,
) -> dict[str, Any]:
    return {
        "schema": "qksieve_final_pdf_audit_v1",
        "anonymous": audit_english_pages(
            extract_pages(anonymous_path), anonymous=True
        ),
        "author": audit_english_pages(extract_pages(author_path), anonymous=False),
        "chinese": audit_chinese_pages(extract_pages(chinese_path)),
    }


def main() -> None:
    args = parse_args()
    report = audit(args.anonymous, args.author, args.chinese)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
