from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def mean(rows: list[dict[str, Any]], key: str) -> float:
    return statistics.fmean(float(row[key]) for row in rows)


def fraction(rows: list[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool]) -> float:
    return sum(bool(predicate(row)) for row in rows) / max(1, len(rows))


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "samples": len({row["sample_id"] for row in rows}),
        "mean_high_phase_contribution": mean(rows, "high_phase_contribution_mean"),
        "negative_high_phase_fraction": fraction(rows, lambda row: row["high_phase_contribution_mean"] < 0),
        "mean_evidence_mass_delta_nope_minus_native": mean(rows, "evidence_mass_delta_nope_minus_native"),
        "mass_improved_fraction": fraction(rows, lambda row: row["evidence_mass_delta_nope_minus_native"] > 0),
        "mean_evidence_rank_delta_nope_minus_native": mean(rows, "evidence_rank_delta_nope_minus_native"),
        "rank_improved_fraction": fraction(rows, lambda row: row["evidence_rank_delta_nope_minus_native"] < 0),
        "pre_rope_cosine_max_ge_0p9_fraction": fraction(rows, lambda row: row["pre_rope_cosine_max"] >= 0.9),
        "mean_pre_rope_cosine_max": mean(rows, "pre_rope_cosine_max"),
        "mean_q_pair_energy_l1_uniform": mean(rows, "q_pair_energy_l1_uniform"),
        "mean_q_high8_energy_mass": mean(rows, "q_high8_energy_mass"),
        "mean_native_evidence_mass": mean(rows, "native_evidence_mass"),
        "mean_nope_high_evidence_mass": mean(rows, "nope_high_evidence_mass"),
    }


def top_native_mass(rows: list[dict[str, Any]], fraction_value: float = 0.1) -> list[dict[str, Any]]:
    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_sample[str(row["sample_id"])].append(row)
    output: list[dict[str, Any]] = []
    for selected in by_sample.values():
        selected.sort(key=lambda row: float(row["native_evidence_mass"]), reverse=True)
        keep = max(1, round(len(selected) * fraction_value))
        output.extend(selected[:keep])
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    paths = sorted(args.run_dir.glob("sample*/head_rows.jsonl"))
    if not paths:
        raise FileNotFoundError("no sample*/head_rows.jsonl files")
    rows = [row for path in paths for row in read_jsonl(path)]
    deep = [row for row in rows if int(row["layer"]) >= 18]
    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in deep:
        by_sample[str(row["sample_id"])].append(row)
    summary = {
        "definition": {
            "high_frequency_pairs": "F0-F7",
            "counterfactual": "same native hidden state; replace only F0-F7 relative RoPE phase with zero relative phase",
            "positive_mass_delta": "NoPE-high counterfactual assigns more softmax mass to gold-token occurrences",
            "negative_rank_delta": "NoPE-high counterfactual improves the best gold-token rank",
            "top_native_mass_10pct": "post-hoc diagnostic only; top 10% deep-layer head rows per sample by native evidence mass",
        },
        "all_layers_all_heads": aggregate(rows),
        "deep_layers_all_heads": aggregate(deep),
        "deep_layers_top_native_mass_10pct": aggregate(top_native_mass(deep)),
        "per_sample_deep": {
            sample_id: aggregate(selected) for sample_id, selected in sorted(by_sample.items())
        },
    }
    output = args.run_dir / "summary.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
