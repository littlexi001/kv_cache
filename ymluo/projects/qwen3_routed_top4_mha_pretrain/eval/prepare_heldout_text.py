from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_dataset_checked(*args: Any, **kwargs: Any) -> Any:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets is required. Install with: pip install datasets") from exc
    return load_dataset(*args, **kwargs)


DATASETS = {
    "wikitext103_validation": ("wikitext", "wikitext-103-raw-v1", "validation", "text"),
    "wikitext103_test": ("wikitext", "wikitext-103-raw-v1", "test", "text"),
    "wikitext2_validation": ("wikitext", "wikitext-2-raw-v1", "validation", "text"),
    "wikitext2_test": ("wikitext", "wikitext-2-raw-v1", "test", "text"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="wikitext103_validation", choices=sorted(DATASETS))
    parser.add_argument("--output_dir", default="output/heldout_text")
    parser.add_argument("--max_chars", type=int, default=5_000_000)
    parser.add_argument("--min_line_chars", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_name, config_name, split, text_key = DATASETS[args.dataset]
    dataset = load_dataset_checked(dataset_name, config_name, split=split)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    text_path = output_dir / f"{args.dataset}.txt"
    meta_path = output_dir / f"{args.dataset}.meta.json"

    total_chars = 0
    kept_rows = 0
    skipped_rows = 0
    with text_path.open("w", encoding="utf-8") as handle:
        for item in dataset:
            text = str(item.get(text_key, "")).strip()
            if len(text) < args.min_line_chars:
                skipped_rows += 1
                continue
            if args.max_chars > 0 and total_chars >= args.max_chars:
                break
            remaining = args.max_chars - total_chars if args.max_chars > 0 else len(text)
            if args.max_chars > 0 and len(text) > remaining:
                text = text[:remaining]
            handle.write(text)
            handle.write("\n\n")
            total_chars += len(text) + 2
            kept_rows += 1

    meta = {
        "dataset": args.dataset,
        "hf_dataset": dataset_name,
        "hf_config": config_name,
        "split": split,
        "text_key": text_key,
        "max_chars": args.max_chars,
        "min_line_chars": args.min_line_chars,
        "kept_rows": kept_rows,
        "skipped_rows": skipped_rows,
        "total_chars": total_chars,
        "text_path": str(text_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
