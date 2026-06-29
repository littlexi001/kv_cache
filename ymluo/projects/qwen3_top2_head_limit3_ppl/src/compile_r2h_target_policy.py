from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TEMPLATE_KEEP_FRACTION = {
    "FULL": 1.00,
    "HYBRID_SAFE_B4": 0.25,
    "QSKETCH_PAGE_B4": 0.125,
    "QSKETCH_PAGE_B2": 0.0625,
    "SYNTH_M8": 0.03125,
    "LOCAL_R512": 0.25,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile an aggressive target-compression R2H-KV head policy.")
    parser.add_argument("--profile_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target_keep_fraction", type=float, default=0.10)
    parser.add_argument("--head_group_size", type=int, default=4)
    parser.add_argument("--recent_tokens", type=int, default=512)
    parser.add_argument("--full_reserve_fraction", type=float, default=0.08)
    parser.add_argument("--hybrid_reserve_fraction", type=float, default=0.12)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            row = dict(row)
            row["layer"] = int(row["layer"])
            row["head"] = int(row["head"])
            for key in [
                "remote_mass_mean",
                "qabs_candidate_energy",
                "qabs_true_top_overlap",
                "reuse_previous_energy",
                "reuse_jaccard",
            ]:
                row[key] = float(row.get(key, 0.0) or 0.0)
            rows.append(row)
    return rows


def risk_score(row: dict[str, Any]) -> float:
    remote = row["remote_mass_mean"]
    qabs_gap = 1.0 - row["qabs_candidate_energy"]
    overlap_gap = 1.0 - row["qabs_true_top_overlap"]
    reuse_gap = 1.0 - max(row["reuse_previous_energy"], row["reuse_jaccard"])
    return 0.45 * remote + 0.30 * qabs_gap + 0.15 * overlap_gap + 0.10 * reuse_gap


def assign_target_templates(
    rows: list[dict[str, Any]],
    target_keep_fraction: float,
    full_reserve_fraction: float,
    hybrid_reserve_fraction: float,
) -> None:
    total = len(rows)
    for row in rows:
        row["risk_score"] = risk_score(row)

    ordered = sorted(rows, key=lambda r: r["risk_score"], reverse=True)
    full_count = max(0, min(total, round(total * full_reserve_fraction)))
    hybrid_count = max(0, min(total - full_count, round(total * hybrid_reserve_fraction)))

    for idx, row in enumerate(ordered):
        if idx < full_count:
            row["target_template"] = "FULL"
            row["target_reason"] = "highest_risk_reserved_full"
        elif idx < full_count + hybrid_count:
            row["target_template"] = "HYBRID_SAFE_B4"
            row["target_reason"] = "high_risk_hybrid_safe"
        else:
            # Low-risk heads are assigned by proxy quality. The most stable q-sketch
            # heads get B2; weaker but still compressible heads get B4; very local
            # heads get LOCAL; the remaining lowest-risk heads get SYNTH_M8 as the
            # 90%-target placeholder for future synthetic/page runtime.
            if row["remote_mass_mean"] < 0.03:
                row["target_template"] = "LOCAL_R512"
                row["target_reason"] = "low_remote_mass"
            elif row["qabs_candidate_energy"] >= 0.95 and row["reuse_previous_energy"] >= 0.80:
                row["target_template"] = "QSKETCH_PAGE_B2"
                row["target_reason"] = "strong_qsketch_reuse"
            elif row["qabs_candidate_energy"] >= 0.88:
                row["target_template"] = "QSKETCH_PAGE_B4"
                row["target_reason"] = "qsketch_energy_ok"
            else:
                row["target_template"] = "SYNTH_M8"
                row["target_reason"] = "aggressive_synthetic_placeholder"

    # If keep fraction is still above target, promote lowest-risk non-FULL heads
    # to SYNTH_M8 until the target is met.
    def mean_keep() -> float:
        return sum(TEMPLATE_KEEP_FRACTION[r["target_template"]] for r in rows) / max(1, len(rows))

    if mean_keep() > target_keep_fraction:
        candidates = sorted(
            [r for r in rows if r["target_template"] not in {"FULL", "SYNTH_M8"}],
            key=lambda r: r["risk_score"],
        )
        for row in candidates:
            if mean_keep() <= target_keep_fraction:
                break
            row["target_template"] = "SYNTH_M8"
            row["target_reason"] = "promoted_to_meet_target_keep"


def group_heads(rows: list[dict[str, Any]], group_size: int) -> dict[str, Any]:
    by_layer: dict[int, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_layer[row["layer"]][row["target_template"]].append(row["head"])

    layers: dict[str, Any] = {}
    for layer, template_heads in sorted(by_layer.items()):
        groups: list[dict[str, Any]] = []
        for template, heads in sorted(template_heads.items()):
            heads = sorted(heads)
            chunks = [heads] if group_size <= 0 else [heads[i : i + group_size] for i in range(0, len(heads), group_size)]
            for chunk in chunks:
                groups.append({"heads": chunk, "template": template})
        layers[str(layer)] = {"head_groups": groups, "template_counts": dict(Counter(g["template"] for g in groups))}
    return layers


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = read_rows(Path(args.profile_csv))
    assign_target_templates(rows, args.target_keep_fraction, args.full_reserve_fraction, args.hybrid_reserve_fraction)

    keep_sum = sum(TEMPLATE_KEEP_FRACTION[r["target_template"]] for r in rows)
    keep_fraction = keep_sum / max(1, len(rows))
    template_counts = Counter(r["target_template"] for r in rows)
    layers = group_heads(rows, args.head_group_size)

    policy = {
        "method": "R2H-KV",
        "version": "target-90-v1",
        "target_keep_fraction": args.target_keep_fraction,
        "estimated_keep_fraction": keep_fraction,
        "estimated_compression_fraction": 1.0 - keep_fraction,
        "template_keep_fraction": TEMPLATE_KEEP_FRACTION,
        "template_counts": dict(template_counts),
        "gpu_friendly": {
            "head_group_size": args.head_group_size,
            "templates": {
                "FULL": {"backend": "full"},
                "HYBRID_SAFE_B4": {"backend": "hybrid_safe", "selected_pages": 4, "page_size": 64},
                "QSKETCH_PAGE_B4": {"backend": "qsketch_page", "selected_pages": 4, "page_size": 64},
                "QSKETCH_PAGE_B2": {"backend": "qsketch_page", "selected_pages": 2, "page_size": 64},
                "SYNTH_M8": {"backend": "synthetic_page", "prototypes": 8},
                "LOCAL_R512": {"backend": "recent", "recent": args.recent_tokens},
            },
        },
        "layers": layers,
    }

    write_csv(out / "target_profile_by_layer_head.csv", rows)
    (out / "recommended_hybrid_policy_target90.json").write_text(
        json.dumps(policy, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "heads": len(rows),
        "target_keep_fraction": args.target_keep_fraction,
        "estimated_keep_fraction": keep_fraction,
        "estimated_compression_fraction": 1.0 - keep_fraction,
        "template_counts": dict(template_counts),
        "outputs": {
            "target_profile_by_layer_head": str(out / "target_profile_by_layer_head.csv"),
            "recommended_hybrid_policy_target90": str(out / "recommended_hybrid_policy_target90.json"),
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
