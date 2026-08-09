#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_key(sample_id: str, seed: str) -> str:
    return hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--design-manifest", nargs="+", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", default="riskkv-lbv2-20260714")
    parser.add_argument("--train-fraction", type=float, default=0.50)
    parser.add_argument("--calibration-fraction", type=float, default=0.20)
    args = parser.parse_args()

    dataset = load_json(args.dataset)
    if not isinstance(dataset, list):
        raise ValueError("LongBench v2 dataset must be a JSON list")
    design_ids: set[str] = set()
    for path in args.design_manifest:
        manifest = load_json(path)
        if not isinstance(manifest, list):
            raise ValueError(f"Design manifest must be a list: {path}")
        design_ids.update(str(item.get("sample_id", item.get("_id", ""))) for item in manifest)
    design_ids.discard("")

    design_rows: list[dict[str, Any]] = []
    remaining_by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in dataset:
        if not isinstance(row, dict):
            continue
        sample_id = str(row.get("_id", ""))
        if sample_id in design_ids:
            design_rows.append(row)
        else:
            remaining_by_domain[str(row.get("domain", ""))].append(row)

    splits: dict[str, list[dict[str, Any]]] = {
        "design_dev": design_rows,
        "router_train": [],
        "router_calibration": [],
        "paper_test": [],
    }
    train_fraction = min(1.0, max(0.0, args.train_fraction))
    calibration_fraction = min(1.0 - train_fraction, max(0.0, args.calibration_fraction))
    for domain, rows in sorted(remaining_by_domain.items()):
        ordered = sorted(rows, key=lambda row: stable_key(str(row.get("_id", "")), args.seed))
        train_end = round(len(ordered) * train_fraction)
        calibration_end = train_end + round(len(ordered) * calibration_fraction)
        splits["router_train"].extend(ordered[:train_end])
        splits["router_calibration"].extend(ordered[train_end:calibration_end])
        splits["paper_test"].extend(ordered[calibration_end:])

    all_ids = [str(row.get("_id", "")) for rows in splits.values() for row in rows]
    if len(all_ids) != len(set(all_ids)):
        raise RuntimeError("Frozen split construction produced duplicate sample IDs")
    source_ids = {str(row.get("_id", "")) for row in dataset if isinstance(row, dict)}
    if set(all_ids) != source_ids:
        missing = source_ids - set(all_ids)
        extra = set(all_ids) - source_ids
        raise RuntimeError(f"Frozen splits do not cover the source dataset: missing={len(missing)} extra={len(extra)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "seed": args.seed,
        "train_fraction": train_fraction,
        "calibration_fraction": calibration_fraction,
        "test_fraction": 1.0 - train_fraction - calibration_fraction,
        "source_samples": len(dataset),
        "splits": {},
    }
    for name, rows in splits.items():
        rows.sort(key=lambda row: (str(row.get("domain", "")), stable_key(str(row.get("_id", "")), args.seed)))
        (args.output_dir / f"{name}.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        summary["splits"][name] = {
            "samples": len(rows),
            "domains": dict(sorted(Counter(str(row.get("domain", "")) for row in rows).items())),
        }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
