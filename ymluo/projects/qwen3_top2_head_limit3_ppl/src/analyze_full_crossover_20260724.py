from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import statistics
from collections import defaultdict
from pathlib import Path


METHOD_LABELS = {
    "full_kv": "full_kv",
    "countcap_fullprompt_keypca_direct_qkvfused_qprojscan_prefillindex": "base",
    "countcap_fullprompt_keypca_direct_qkvfused_qprojscan_qkvsplitauto_prefillindex": "auto",
    "countcap_fullprompt_keypca_direct_qkvfused_qprojscan_qkvsplitauto_inplacecache_prefillindex": "combo",
}
TIME_FIELDS = ("query_seconds", "decode_seconds", "online_seconds", "total_seconds")


def median(rows: list[dict[str, str]], field: str) -> float:
    return statistics.median(float(row[field]) for row in rows)


def analyze(root: Path) -> dict[str, object]:
    grouped: dict[int, dict[str, list[dict[str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for csv_path in glob.glob(str(root / "*k_*" / "sample_results.csv")):
        run_name = os.path.basename(os.path.dirname(csv_path))
        match = re.match(r"(\d+)k_", run_name)
        if match is None:
            continue
        length_k = int(match.group(1))
        with open(csv_path, encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                label = METHOD_LABELS.get(row["method"])
                if label is not None:
                    grouped[length_k][label].append(row)

    result: dict[str, object] = {"root": str(root), "lengths": {}}
    for length_k in sorted(grouped):
        by_method = grouped[length_k]
        missing = sorted(set(METHOD_LABELS.values()) - set(by_method))
        if missing:
            raise RuntimeError(f"{length_k}K is missing methods: {missing}")
        if any(len(rows) != 4 for rows in by_method.values()):
            counts = {label: len(rows) for label, rows in by_method.items()}
            raise RuntimeError(f"{length_k}K expected four rotations: {counts}")

        methods = {}
        for label, rows in by_method.items():
            methods[label] = {
                "runs": len(rows),
                "prompt_tokens": statistics.median(
                    int(row["prompt_tokens"]) for row in rows
                ),
                "generated_tokens": statistics.median(
                    int(row["generated_tokens"]) for row in rows
                ),
                "score": statistics.median(float(row["score"]) for row in rows),
                **{field: median(rows, field) for field in TIME_FIELDS},
            }

        full_online = methods["full_kv"]["online_seconds"]
        base_online = methods["base"]["online_seconds"]
        auto_online = methods["auto"]["online_seconds"]
        combo_online = methods["combo"]["online_seconds"]
        sparse_predictions = {
            row["prediction"]
            for label in ("base", "auto", "combo")
            for row in by_method[label]
        }
        sparse_scores = {
            row["score"]
            for label in ("base", "auto", "combo")
            for row in by_method[label]
        }
        result["lengths"][str(length_k)] = {
            "methods": methods,
            "speedups": {
                "combo_vs_full": full_online / combo_online,
                "combo_vs_base": base_online / combo_online,
                "combo_vs_auto": auto_online / combo_online,
            },
            "sparse_prediction_exact": len(sparse_predictions) == 1,
            "sparse_score_exact": len(sparse_scores) == 1,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output_json", type=Path)
    args = parser.parse_args()
    result = analyze(args.root)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
