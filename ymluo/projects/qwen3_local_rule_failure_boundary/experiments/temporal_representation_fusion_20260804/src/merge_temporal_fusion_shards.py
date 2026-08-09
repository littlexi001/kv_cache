from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"empty rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def number(row: dict[str, Any], key: str) -> float:
    return float(row[key])


def summarize(
    rows: list[dict[str, Any]],
    target_baseline: dict[str, Any],
) -> dict[str, Any]:
    best_by_mode: dict[str, dict[str, Any]] = {}
    best_by_layer_mode: dict[str, dict[str, Any]] = {}
    for mode in sorted({row["mode"] for row in rows}):
        subset = [row for row in rows if row["mode"] == mode]
        best_by_mode[mode] = max(
            subset, key=lambda row: number(row, "gold_vs_fixed_competitor_margin")
        )
    for layer in sorted({int(row["layer"]) for row in rows}):
        for mode in sorted({row["mode"] for row in rows if int(row["layer"]) == layer}):
            subset = [
                row
                for row in rows
                if int(row["layer"]) == layer and row["mode"] == mode
            ]
            best_by_layer_mode[f"L{layer}:{mode}"] = max(
                subset,
                key=lambda row: number(row, "gold_vs_fixed_competitor_margin"),
            )
    pre_lookup = {
        (int(row["layer"]), float(row["alpha"]), row["anchor_offsets"]): row
        for row in rows
        if row["mode"] == "q_pre_current_phase"
    }
    phase_pairs = []
    for native in rows:
        if native["mode"] != "q_native_phase":
            continue
        key = (
            int(native["layer"]),
            float(native["alpha"]),
            native["anchor_offsets"],
        )
        if key not in pre_lookup:
            continue
        current = pre_lookup[key]
        phase_pairs.append(
            {
                "layer": key[0],
                "alpha": key[1],
                "anchor_offsets": key[2],
                "native_minus_current_phase_margin": (
                    number(native, "gold_vs_fixed_competitor_margin")
                    - number(current, "gold_vs_fixed_competitor_margin")
                ),
                "native_minus_current_phase_qk": (
                    number(native, "critical_qk") - number(current, "critical_qk")
                ),
                "native_minus_current_phase_attention": (
                    number(native, "critical_evidence_attention_weighted")
                    - number(current, "critical_evidence_attention_weighted")
                ),
            }
        )
    jointly_better = [
        row
        for row in phase_pairs
        if row["native_minus_current_phase_margin"] > 0
        and row["native_minus_current_phase_qk"] > 0
        and row["native_minus_current_phase_attention"] > 0
    ]
    baseline_margin = float(target_baseline["gold_vs_fixed_competitor_margin"])
    hard_recovered = [
        row
        for row in rows
        if number(row, "gold_vs_fixed_competitor_margin") > 0
        and row["top_token_id"] == "11627"
    ]
    return {
        "target_baseline": target_baseline,
        "intervention_count": len(rows),
        "hard_recovery_count": len(hard_recovered),
        "hard_recovery_fraction": len(hard_recovered) / len(rows),
        "baseline_margin": baseline_margin,
        "best_by_mode": best_by_mode,
        "best_by_layer_mode": best_by_layer_mode,
        "native_vs_current_phase_pair_count": len(phase_pairs),
        "native_jointly_better_count": len(jointly_better),
        "native_jointly_better_fraction": (
            len(jointly_better) / len(phase_pairs) if phase_pairs else None
        ),
        "best_native_phase_advantage": (
            max(
                phase_pairs,
                key=lambda row: row["native_minus_current_phase_margin"],
            )
            if phase_pairs
            else None
        ),
        "phase_pairs": phase_pairs,
    }


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    shard_dirs = sorted(
        path for path in input_root.iterdir() if path.is_dir() and (path / "manifest.json").exists()
    )
    if not shard_dirs:
        raise RuntimeError(f"no complete shard directories under {input_root}")

    merged: dict[str, list[dict[str, Any]]] = {
        "interventions.csv": [],
        "intervention_head_metrics.csv": [],
        "anchor_selection.csv": [],
    }
    manifests = []
    baselines_by_offset: dict[str, dict[str, Any]] = {}
    baseline_heads_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    target_baselines = []
    for shard_dir in shard_dirs:
        with (shard_dir / "manifest.json").open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if not manifest.get("complete"):
            raise RuntimeError(f"incomplete shard: {shard_dir}")
        manifest["shard"] = shard_dir.name
        manifests.append(manifest)
        for filename in merged:
            for row in read_csv(shard_dir / filename):
                row["shard"] = shard_dir.name
                merged[filename].append(row)
        for row in read_csv(shard_dir / "history_baselines.csv"):
            baselines_by_offset.setdefault(row["offset_from_target"], row)
        for row in read_csv(shard_dir / "baseline_head_metrics.csv"):
            key = (row["offset_from_target"], row["layer"], row["head"])
            baseline_heads_by_key.setdefault(key, row)
        with (shard_dir / "summary.json").open("r", encoding="utf-8") as handle:
            target_baselines.append(json.load(handle)["target_baseline"])

    for filename, rows in merged.items():
        write_csv(output_dir / filename, rows)
    write_csv(
        output_dir / "history_baselines.csv",
        sorted(baselines_by_offset.values(), key=lambda row: int(row["offset_from_target"])),
    )
    write_csv(
        output_dir / "baseline_head_metrics.csv",
        sorted(
            baseline_heads_by_key.values(),
            key=lambda row: (
                int(row["offset_from_target"]),
                int(row["layer"]),
                int(row["head"]),
            ),
        ),
    )
    reference = target_baselines[0]
    for candidate in target_baselines[1:]:
        for field in (
            "gold_probability",
            "gold_vs_fixed_competitor_margin",
            "critical_qk",
            "critical_evidence_attention_weighted",
        ):
            if abs(float(candidate[field]) - float(reference[field])) > 1e-5:
                raise RuntimeError(f"shard baseline mismatch for {field}")
    summary = summarize(merged["interventions.csv"], reference)
    write_json(output_dir / "summary.json", summary)
    write_json(
        output_dir / "manifest.json",
        {
            "schema_version": 1,
            "experiment": "temporal_representation_fusion_merged",
            "source_root": str(input_root),
            "shard_count": len(shard_dirs),
            "shards": manifests,
            "intervention_count": len(merged["interventions.csv"]),
            "complete": True,
        },
    )


if __name__ == "__main__":
    main()
