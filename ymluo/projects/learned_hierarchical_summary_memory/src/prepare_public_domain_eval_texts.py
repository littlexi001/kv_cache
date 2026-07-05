from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path
from typing import Any


BOOKS = [
    {"name": "moby_dick", "title": "Moby-Dick", "gutenberg_id": 2701},
    {"name": "pride_prejudice", "title": "Pride and Prejudice", "gutenberg_id": 1342},
    {"name": "tale_two_cities", "title": "A Tale of Two Cities", "gutenberg_id": 98},
    {"name": "sherlock_holmes", "title": "The Adventures of Sherlock Holmes", "gutenberg_id": 1661},
    {"name": "dracula", "title": "Dracula", "gutenberg_id": 345},
    {"name": "frankenstein", "title": "Frankenstein", "gutenberg_id": 84},
    {"name": "origin_species", "title": "On the Origin of Species", "gutenberg_id": 1228},
    {"name": "republic", "title": "The Republic", "gutenberg_id": 1497},
    {"name": "walden", "title": "Walden", "gutenberg_id": 205},
    {"name": "time_machine", "title": "The Time Machine", "gutenberg_id": 35},
]


def candidate_urls(gutenberg_id: int) -> list[str]:
    return [
        f"https://www.gutenberg.org/cache/epub/{gutenberg_id}/pg{gutenberg_id}.txt",
        f"https://www.gutenberg.org/files/{gutenberg_id}/{gutenberg_id}-0.txt",
        f"https://www.gutenberg.org/files/{gutenberg_id}/{gutenberg_id}.txt",
    ]


def download_text(gutenberg_id: int, timeout: int) -> tuple[str, str]:
    errors: list[str] = []
    for url in candidate_urls(gutenberg_id):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                raw = response.read()
            text = raw.decode("utf-8", errors="ignore")
            if len(text.split()) > 1000:
                return url, text
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{url}: {exc}")
    raise RuntimeError("\n".join(errors))


def clean_gutenberg_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    start_match = re.search(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", text, re.IGNORECASE | re.DOTALL)
    if start_match:
        text = text[start_match.end() :]
    end_match = re.search(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*", text, re.IGNORECASE | re.DOTALL)
    if end_match:
        text = text[: end_match.start()]
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and clean public-domain long-text eval files.")
    parser.add_argument("--output_dir", default="/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/data/public_domain_eval")
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    for book in BOOKS:
        path = output_dir / f"{book['name']}.txt"
        if path.exists() and path.stat().st_size > 10_000:
            text = path.read_text(encoding="utf-8", errors="ignore")
            url = "existing"
        else:
            url, raw = download_text(book["gutenberg_id"], args.timeout)
            text = clean_gutenberg_text(raw)
            path.write_text(text, encoding="utf-8")
        row = {
            **book,
            "path": str(path),
            "source_url": url,
            "words": len(text.split()),
            "characters": len(text),
        }
        manifest.append(row)
        print(f"{book['name']}: words={row['words']} path={path}")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote manifest to {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
