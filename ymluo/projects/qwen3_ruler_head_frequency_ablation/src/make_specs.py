from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


LAYER_BLOCKS = (tuple(range(18, 24)), tuple(range(24, 30)), tuple(range(30, 36)))
FREQUENCY_BANDS = tuple(tuple(range(start, start + 8)) for start in range(0, 64, 8))


def write(path: Path, specs: list[dict[str, Any]], stage: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"stage": stage, "specs": specs}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def native_spec() -> dict[str, Any]:
    return {"name": "native_rope", "stage": "native", "atoms": []}


def smoke_specs() -> list[dict[str, Any]]:
    return [
        {
            "name": "original_native",
            "stage": "smoke",
            "bypass_original": True,
            "reference_original": True,
            "atoms": [],
        },
        {
            "name": "patched_native_replay",
            "stage": "smoke",
            "compare_to_original": True,
            "atoms": [],
        },
        {
            "name": "smoke_l30_g0_f0_7",
            "stage": "smoke",
            "atoms": [{"layers": [30], "head_groups": [0], "frequency_pairs": list(range(8))}],
        },
    ]


def coarse_specs() -> list[dict[str, Any]]:
    specs = [native_spec()]
    for block_index, layers in enumerate(LAYER_BLOCKS):
        for group in range(8):
            for band_index, frequencies in enumerate(FREQUENCY_BANDS):
                specs.append({
                    "name": f"coarse_b{block_index}_g{group}_f{frequencies[0]:02d}_{frequencies[-1]:02d}",
                    "stage": "coarse",
                    "region": {
                        "block_index": block_index,
                        "layers": list(layers),
                        "head_group": group,
                        "band_index": band_index,
                        "frequency_pairs": list(frequencies),
                    },
                    "atoms": [{
                        "layers": list(layers),
                        "head_groups": [group],
                        "frequency_pairs": list(frequencies),
                    }],
                })
    return specs


def dense_layer_band_specs() -> list[dict[str, Any]]:
    """Complete deep-layer x KV-group x 8-pair frequency-band grid."""
    specs = [native_spec()]
    for layer in range(18, 36):
        for group in range(8):
            for frequencies in FREQUENCY_BANDS:
                specs.append({
                    "name": f"dense_l{layer:02d}_g{group}_f{frequencies[0]:02d}_{frequencies[-1]:02d}",
                    "stage": "dense_layer_band",
                    "region": {
                        "layer": layer,
                        "head_group": group,
                        "frequency_pairs": list(frequencies),
                    },
                    "atoms": [{
                        "layers": [layer],
                        "head_groups": [group],
                        "frequency_pairs": list(frequencies),
                    }],
                })
    return specs


def read_top_regions(summary_path: Path, limit: int) -> list[dict[str, Any]]:
    rows = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = [
        row for row in rows
        if row["variant"] != "native_rope" and row.get("utility", float("-inf")) > 0
    ]
    rows.sort(key=lambda row: (row["utility"], row["official_score_mean"]), reverse=True)
    return [row["spec"]["region"] for row in rows[:limit]]


def refinement_specs(summary_path: Path, limit: int) -> list[dict[str, Any]]:
    specs = [native_spec()]
    for region_index, region in enumerate(read_top_regions(summary_path, limit)):
        layers = list(map(int, region["layers"]))
        group = int(region["head_group"])
        frequencies = list(map(int, region["frequency_pairs"]))
        for layer in layers:
            specs.append({
                "name": f"ref_r{region_index}_layer_l{layer}_g{group}_f{frequencies[0]:02d}_{frequencies[-1]:02d}",
                "stage": "refine_layer",
                "region_index": region_index,
                "region": region,
                "selected_layer": layer,
                "atoms": [{"layers": [layer], "head_groups": [group], "frequency_pairs": frequencies}],
            })
        for frequency in frequencies:
            specs.append({
                "name": f"ref_r{region_index}_freq_b{region['block_index']}_g{group}_f{frequency:02d}",
                "stage": "refine_frequency",
                "region_index": region_index,
                "region": region,
                "selected_frequency": frequency,
                "atoms": [{"layers": layers, "head_groups": [group], "frequency_pairs": [frequency]}],
            })
    return specs


def cross_specs(refinement_summary: Path) -> list[dict[str, Any]]:
    rows = json.loads(refinement_summary.read_text(encoding="utf-8"))
    by_region: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        spec = row.get("spec", {})
        if "region_index" in spec:
            by_region.setdefault(int(spec["region_index"]), []).append(row)
    specs = [native_spec()]
    for region_index, selected in sorted(by_region.items()):
        layers = [row for row in selected if row["spec"].get("stage") == "refine_layer"]
        frequencies = [row for row in selected if row["spec"].get("stage") == "refine_frequency"]
        if not layers or not frequencies:
            continue
        best_layer = max(layers, key=lambda row: row["utility"])
        best_frequency = max(frequencies, key=lambda row: row["utility"])
        layer = int(best_layer["spec"]["selected_layer"])
        frequency = int(best_frequency["spec"]["selected_frequency"])
        group = int(best_layer["spec"]["region"]["head_group"])
        specs.append({
            "name": f"cross_r{region_index}_l{layer}_g{group}_f{frequency:02d}",
            "stage": "cross",
            "region_index": region_index,
            "source_layer_utility": best_layer["utility"],
            "source_frequency_utility": best_frequency["utility"],
            "atoms": [{"layers": [layer], "head_groups": [group], "frequency_pairs": [frequency]}],
        })
    return specs


def finalists(summary_paths: list[Path], limit: int) -> list[dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    for path in summary_paths:
        for row in json.loads(path.read_text(encoding="utf-8")):
            if row["variant"] == "native_rope" or row.get("utility", 0.0) <= 0:
                continue
            name = json.dumps(row["spec"].get("atoms", []), sort_keys=True)
            if name not in candidates or row["utility"] > candidates[name]["utility"]:
                candidates[name] = row
    ranked = sorted(candidates.values(), key=lambda row: row["utility"], reverse=True)[:limit]
    specs = [native_spec()]
    for index, row in enumerate(ranked):
        specs.append({
            "name": f"finalist_{index}_{row['variant']}",
            "stage": "finalist",
            "screen_utility": row["utility"],
            "source_variant": row["variant"],
            "atoms": row["spec"]["atoms"],
        })
    return specs


def combinations(final_summary: Path, limit: int) -> list[dict[str, Any]]:
    rows = json.loads(final_summary.read_text(encoding="utf-8"))
    ranked = [row for row in rows if row["variant"] != "native_rope"]
    ranked.sort(
        key=lambda row: (row["paired_official_delta"], row["mean_clipped_nll_improvement"]),
        reverse=True,
    )
    specs = [native_spec()]
    atoms: list[dict[str, Any]] = []
    for index, row in enumerate(ranked[:limit], start=1):
        atoms.extend(row["spec"]["atoms"])
        specs.append({
            "name": f"combo_top{index}",
            "stage": "combination",
            "members": [value["variant"] for value in ranked[:index]],
            "atoms": list(atoms),
        })
    return specs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("smoke", "coarse", "dense_layer_band", "refine", "cross", "finalists", "combinations"),
        required=True,
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", action="append", type=Path, default=[])
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()
    if args.stage == "smoke":
        specs = smoke_specs()
    elif args.stage == "coarse":
        specs = coarse_specs()
    elif args.stage == "dense_layer_band":
        specs = dense_layer_band_specs()
    elif args.stage == "refine":
        specs = refinement_specs(args.summary[0], args.limit)
    elif args.stage == "cross":
        specs = cross_specs(args.summary[0])
    elif args.stage == "finalists":
        specs = finalists(args.summary, args.limit)
    else:
        specs = combinations(args.summary[0], args.limit)
    write(args.output, specs, args.stage)
    print(f"wrote {len(specs)} specs to {args.output}")


if __name__ == "__main__":
    main()
