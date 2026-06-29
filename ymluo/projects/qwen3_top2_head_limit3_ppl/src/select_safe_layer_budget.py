from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LayerResult:
    layer: int
    ppl: float
    seconds: float
    delta_ppl: float
    delta_seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize single-layer fallback ablations and generate full_layer_map JSON files "
            "for specialized fulll{N}landmark/recent modes."
        )
    )
    parser.add_argument("--single_layer_dir", required=True, help="Directory containing layerXX/ppl_by_mode.csv files.")
    parser.add_argument("--baseline_csv", required=True, help="Baseline ppl_by_mode.csv used for delta calculations.")
    parser.add_argument("--output_dir", required=True, help="Directory for summary CSV and candidate map JSON files.")
    parser.add_argument("--layer_count", type=int, default=28)
    parser.add_argument("--max_delta_ppl", type=float, default=0.2)
    parser.add_argument("--max_candidates", type=int, default=8)
    parser.add_argument(
        "--force_layers",
        default="",
        help="Optional comma-separated compressed layer list. When set, also writes force_layers_last.json.",
    )
    return parser.parse_args()


def read_first_row(csv_path: Path) -> dict[str, str]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty csv: {csv_path}")
    return rows[0]


def read_metric(csv_path: Path) -> tuple[float, float]:
    row = read_first_row(csv_path)
    return float(row["ppl"]), float(row["seconds"])


def parse_layer_from_name(path: Path) -> int:
    digits = "".join(ch for ch in path.name if ch.isdigit())
    if not digits:
        raise ValueError(f"cannot parse layer id from {path}")
    return int(digits)


def collect_results(single_layer_dir: Path, baseline_csv: Path) -> tuple[float, float, list[LayerResult]]:
    baseline_ppl, baseline_seconds = read_metric(baseline_csv)
    results: list[LayerResult] = []
    for csv_path in sorted(single_layer_dir.glob("layer*/ppl_by_mode.csv")):
        layer = parse_layer_from_name(csv_path.parent)
        ppl, seconds = read_metric(csv_path)
        results.append(
            LayerResult(
                layer=layer,
                ppl=ppl,
                seconds=seconds,
                delta_ppl=ppl - baseline_ppl,
                delta_seconds=seconds - baseline_seconds,
            )
        )
    if not results:
        raise FileNotFoundError(f"no layer*/ppl_by_mode.csv under {single_layer_dir}")
    return baseline_ppl, baseline_seconds, sorted(results, key=lambda item: item.layer)


def make_full_layer_order(layer_count: int, compressed_layers: list[int]) -> list[int]:
    compressed_set = set(compressed_layers)
    full_layers = [layer for layer in range(layer_count) if layer not in compressed_set]
    return full_layers + list(compressed_layers)


def write_map(output_path: Path, layer_count: int, compressed_layers: list[int], metadata: dict[str, object]) -> None:
    output = {
        "layer_count": layer_count,
        "compressed_layers": compressed_layers,
        "top_layers": make_full_layer_order(layer_count, compressed_layers),
        "metadata": metadata,
    }
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_force_layers(raw: str) -> list[int]:
    if not raw.strip():
        return []
    layers = [int(part.strip()) for part in raw.split(",") if part.strip()]
    seen: set[int] = set()
    unique_layers: list[int] = []
    for layer in layers:
        if layer not in seen:
            unique_layers.append(layer)
            seen.add(layer)
    return unique_layers


def main() -> None:
    args = parse_args()
    single_layer_dir = Path(args.single_layer_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_ppl, baseline_seconds, results = collect_results(single_layer_dir, Path(args.baseline_csv))
    ranked = sorted(results, key=lambda item: (item.delta_ppl, item.delta_seconds, item.layer))
    safe_ranked = [item for item in ranked if item.delta_ppl <= args.max_delta_ppl]
    candidate_limit = min(args.max_candidates, len(safe_ranked))

    summary_path = output_dir / "single_layer_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["rank", "layer", "ppl", "delta_ppl", "seconds", "delta_seconds", "safe"],
        )
        writer.writeheader()
        rank_by_layer = {item.layer: rank for rank, item in enumerate(ranked, start=1)}
        for item in sorted(results, key=lambda value: value.layer):
            writer.writerow(
                {
                    "rank": rank_by_layer[item.layer],
                    "layer": item.layer,
                    "ppl": f"{item.ppl:.8f}",
                    "delta_ppl": f"{item.delta_ppl:.8f}",
                    "seconds": f"{item.seconds:.8f}",
                    "delta_seconds": f"{item.delta_seconds:.8f}",
                    "safe": str(item.delta_ppl <= args.max_delta_ppl).lower(),
                }
            )

    metadata_base = {
        "baseline_ppl": baseline_ppl,
        "baseline_seconds": baseline_seconds,
        "source_single_layer_dir": str(single_layer_dir),
        "max_delta_ppl": args.max_delta_ppl,
    }
    for candidate_size in range(1, candidate_limit + 1):
        compressed_layers = [item.layer for item in safe_ranked[:candidate_size]]
        write_map(
            output_dir / f"safe_top{candidate_size}_layers_last.json",
            args.layer_count,
            compressed_layers,
            {
                **metadata_base,
                "candidate_size": candidate_size,
                "selection": [
                    {"layer": item.layer, "delta_ppl": item.delta_ppl, "delta_seconds": item.delta_seconds}
                    for item in safe_ranked[:candidate_size]
                ],
            },
        )

    forced_layers = parse_force_layers(args.force_layers)
    if forced_layers:
        write_map(
            output_dir / "force_layers_last.json",
            args.layer_count,
            forced_layers,
            {**metadata_base, "forced": True},
        )

    print(f"baseline_ppl={baseline_ppl:.8f} baseline_seconds={baseline_seconds:.8f}")
    print(f"wrote {summary_path}")
    if candidate_limit == 0:
        print("no safe candidates under --max_delta_ppl")
    else:
        best_layers = [item.layer for item in safe_ranked[:candidate_limit]]
        print(f"safe_ranked_layers={best_layers}")
        for candidate_size in range(1, candidate_limit + 1):
            full_count = args.layer_count - candidate_size
            print(
                "candidate "
                f"k={candidate_size} mode=fulll{full_count}landmarkr4096s64attn "
                f"map={output_dir / f'safe_top{candidate_size}_layers_last.json'}"
            )


if __name__ == "__main__":
    main()
