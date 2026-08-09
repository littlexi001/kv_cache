#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--top_k", type=int, default=80)
    args = parser.parse_args()

    with Path(args.model).open("rb") as handle:
        model = pickle.load(handle)
    preprocess = model.named_steps["preprocess"]
    clf = model.named_steps["clf"]
    feature_names = list(preprocess.get_feature_names_out())
    importances = getattr(clf, "feature_importances_", None)
    if importances is None:
        raise SystemExit("Classifier does not expose feature_importances_.")
    rows = [
        {
            "rank": rank,
            "feature": feature,
            "importance": float(importance),
        }
        for rank, (feature, importance) in enumerate(
            sorted(zip(feature_names, importances), key=lambda item: item[1], reverse=True),
            start=1,
        )
    ][: max(1, args.top_k)]
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["rank", "feature", "importance"])
        writer.writeheader()
        writer.writerows(rows)
    print(str(output))


if __name__ == "__main__":
    main()
