from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Thresholds:
    local_remote_mass: float = 0.05
    qabs_energy_good: float = 0.92
    qabs_energy_ok: float = 0.85
    qabs_overlap_good: float = 0.80
    reuse_energy_good: float = 0.85
    reuse_jaccard_good: float = 0.35
    fragile_remote_mass: float = 0.20
    fragile_qabs_energy: float = 0.80


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile R2H-KV layer/head profile into GPU-friendly hybrid policy.")
    parser.add_argument("--remote_mass_json", required=True)
    parser.add_argument("--qabs_overlap_csv", required=True)
    parser.add_argument("--adjacent_top2_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mode", default="qabs8cand3rerank")
    parser.add_argument("--qabs_runtime_type", default="qabs8cand3reuse")
    parser.add_argument("--recent_tokens", type=int, default=512)
    parser.add_argument("--head_group_size", type=int, default=4)
    parser.add_argument("--max_templates_per_layer", type=int, default=3)
    parser.add_argument("--local_remote_mass", type=float, default=0.05)
    parser.add_argument("--qabs_energy_good", type=float, default=0.92)
    parser.add_argument("--qabs_energy_ok", type=float, default=0.85)
    parser.add_argument("--qabs_overlap_good", type=float, default=0.80)
    parser.add_argument("--reuse_energy_good", type=float, default=0.85)
    parser.add_argument("--reuse_jaccard_good", type=float, default=0.35)
    parser.add_argument("--fragile_remote_mass", type=float, default=0.20)
    parser.add_argument("--fragile_qabs_energy", type=float, default=0.80)
    return parser.parse_args()


def read_csv_by_layer_head(path: Path, mode: str | None = None) -> dict[tuple[int, int], dict[str, Any]]:
    rows: dict[tuple[int, int], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if mode is not None and row.get("mode") != mode:
                continue
            key = (int(row["layer"]), int(row["head"]))
            rows[key] = row
    return rows


def to_float(row: dict[str, Any] | None, key: str, default: float = 0.0) -> float:
    if not row:
        return default
    value = row.get(key, "")
    if value in {"", None}:
        return default
    return float(value)


def classify_head(
    *,
    remote_mass: float,
    qabs_energy: float,
    qabs_overlap: float,
    reuse_energy: float,
    reuse_jaccard: float,
    thresholds: Thresholds,
) -> tuple[str, str, float]:
    if remote_mass <= thresholds.local_remote_mass:
        return "LOCAL_R512", "remote_mass_low", 0.10

    if (
        qabs_energy >= thresholds.qabs_energy_good
        and qabs_overlap >= thresholds.qabs_overlap_good
        and (reuse_energy >= thresholds.reuse_energy_good or reuse_jaccard >= thresholds.reuse_jaccard_good)
    ):
        score = 1.0 - max(0.0, 1.0 - qabs_energy) - 0.25 * max(0.0, thresholds.qabs_overlap_good - qabs_overlap)
        return "QSKETCH_PAGE_B2", "qabs_and_reuse_stable", score

    if qabs_energy >= thresholds.qabs_energy_ok:
        score = 0.70 + 0.30 * min(1.0, max(0.0, (qabs_energy - thresholds.qabs_energy_ok) / 0.15))
        return "QSKETCH_PAGE_B4", "qabs_energy_ok", score

    if remote_mass >= thresholds.fragile_remote_mass and qabs_energy < thresholds.fragile_qabs_energy:
        return "FULL", "fragile_remote_high_qabs_low", 0.95

    return "HYBRID_SAFE_B4", "mixed_uncertain", 0.55


def group_heads(head_templates: dict[int, str], group_size: int) -> list[dict[str, Any]]:
    template_to_heads: dict[str, list[int]] = defaultdict(list)
    for head, template in sorted(head_templates.items()):
        template_to_heads[template].append(head)

    groups: list[dict[str, Any]] = []
    for template, heads in sorted(template_to_heads.items()):
        if group_size <= 0:
            chunks = [heads]
        else:
            chunks = [heads[i : i + group_size] for i in range(0, len(heads), group_size)]
        for chunk in chunks:
            groups.append({"heads": chunk, "template": template})
    return groups


def layer_budget_from_templates(templates: list[str], qabs_runtime_type: str, recent_tokens: int) -> dict[str, Any]:
    counts = Counter(templates)
    total = max(1, sum(counts.values()))
    if counts.get("LOCAL_R512", 0) / total >= 0.75:
        return {"type": "recent", "recent": recent_tokens}
    if counts.get("FULL", 0) / total >= 0.75:
        return {"type": "full"}
    qsketch_count = counts.get("QSKETCH_PAGE_B2", 0) + counts.get("QSKETCH_PAGE_B4", 0)
    if qsketch_count / total >= 0.60:
        return {"type": qabs_runtime_type, "dims": 8, "candidate_fraction": 0.03, "top_fraction": 0.03}
    if qsketch_count > 0:
        full_heads = min(15, max(1, counts.get("FULL", 0) + counts.get("HYBRID_SAFE_B4", 0)))
        return {
            "type": "headmix_qabs_reuse",
            "full_heads": full_heads,
            "dims": 8,
            "candidate_fraction": 0.03,
            "top_fraction": 0.03,
        }
    if counts.get("HYBRID_SAFE_B4", 0) > 0:
        return {"type": "landmark", "recent": recent_tokens, "stride": 64}
    return {"type": "full"}


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    thresholds = Thresholds(
        local_remote_mass=args.local_remote_mass,
        qabs_energy_good=args.qabs_energy_good,
        qabs_energy_ok=args.qabs_energy_ok,
        qabs_overlap_good=args.qabs_overlap_good,
        reuse_energy_good=args.reuse_energy_good,
        reuse_jaccard_good=args.reuse_jaccard_good,
        fragile_remote_mass=args.fragile_remote_mass,
        fragile_qabs_energy=args.fragile_qabs_energy,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    remote = json.loads(Path(args.remote_mass_json).read_text(encoding="utf-8"))
    layer_count = int(remote["layer_count"])
    head_count = int(remote["head_count"])
    mean_remote_mass = remote["mean_remote_mass"]
    mean_recent_mass = remote.get("mean_recent_mass", [[0.0 for _ in range(head_count)] for _ in range(layer_count)])

    qabs_rows = read_csv_by_layer_head(Path(args.qabs_overlap_csv), args.mode)
    adjacent_rows = read_csv_by_layer_head(Path(args.adjacent_top2_csv), None)

    profile_rows: list[dict[str, Any]] = []
    layer_head_templates: dict[int, dict[int, str]] = defaultdict(dict)
    policy_layers: dict[str, Any] = {}

    for layer in range(layer_count):
        for head in range(head_count):
            key = (layer, head)
            qrow = qabs_rows.get(key)
            arow = adjacent_rows.get(key)
            remote_mass = float(mean_remote_mass[layer][head])
            recent_mass = float(mean_recent_mass[layer][head])
            qabs_energy = to_float(qrow, "candidate_attention_mass_mean", 0.0)
            qabs_selected_energy = to_float(qrow, "selected_attention_mass_mean", 0.0)
            qabs_overlap = to_float(qrow, "true_top_overlap", 0.0)
            reuse_energy = to_float(arow, "previous_set_current_attention_mass_mean", 0.0)
            reuse_intersection_energy = to_float(arow, "intersection_current_attention_mass_mean", 0.0)
            reuse_jaccard = to_float(arow, "jaccard", 0.0)

            template, reason, confidence = classify_head(
                remote_mass=remote_mass,
                qabs_energy=qabs_energy,
                qabs_overlap=qabs_overlap,
                reuse_energy=reuse_energy,
                reuse_jaccard=reuse_jaccard,
                thresholds=thresholds,
            )
            layer_head_templates[layer][head] = template
            profile_rows.append(
                {
                    "layer": layer,
                    "head": head,
                    "remote_mass_mean": remote_mass,
                    "recent_mass_mean": recent_mass,
                    "qabs_candidate_energy": qabs_energy,
                    "qabs_selected_energy": qabs_selected_energy,
                    "qabs_true_top_overlap": qabs_overlap,
                    "reuse_previous_energy": reuse_energy,
                    "reuse_intersection_energy": reuse_intersection_energy,
                    "reuse_jaccard": reuse_jaccard,
                    "recommended_template": template,
                    "recommendation_reason": reason,
                    "confidence": confidence,
                }
            )

    for layer in range(layer_count):
        templates = [layer_head_templates[layer][head] for head in range(head_count)]
        groups = group_heads(layer_head_templates[layer], args.head_group_size)
        policy_layers[str(layer)] = {
            "head_groups": groups,
            "template_counts": dict(Counter(templates)),
            "fallback_layerbudgetattn": layer_budget_from_templates(
                templates, args.qabs_runtime_type, args.recent_tokens
            ),
        }

    layer_budget_map = {
        "default": {"type": "full"},
        "layers": {
            layer: spec["fallback_layerbudgetattn"]
            for layer, spec in policy_layers.items()
            if spec["fallback_layerbudgetattn"].get("type") != "full"
        },
        "metadata": {
            "source": "compile_r2h_hybrid_policy.py",
            "note": "Fallback map for current layerbudgetattn runtime; recommended_hybrid_policy.json contains head groups.",
        },
    }

    policy = {
        "method": "R2H-KV",
        "version": 1,
        "gpu_friendly": {
            "head_group_size": args.head_group_size,
            "max_templates_per_layer": args.max_templates_per_layer,
            "templates": {
                "FULL": {"backend": "full"},
                "LOCAL_R512": {"backend": "recent", "recent": args.recent_tokens},
                "QSKETCH_PAGE_B2": {
                    "backend": "qsketch_page",
                    "runtime_fallback": args.qabs_runtime_type,
                    "selected_pages": 2,
                    "page_size": 64,
                },
                "QSKETCH_PAGE_B4": {
                    "backend": "qsketch_page",
                    "runtime_fallback": args.qabs_runtime_type,
                    "selected_pages": 4,
                    "page_size": 64,
                },
                "HYBRID_SAFE_B4": {
                    "backend": "hybrid_safe",
                    "runtime_fallback": "headmix_qabs_reuse",
                    "selected_pages": 4,
                    "page_size": 64,
                },
            },
        },
        "thresholds": asdict(thresholds),
        "layers": policy_layers,
        "artifacts": {
            "profile_by_layer_head": str(output_dir / "profile_by_layer_head.csv"),
            "fallback_layerbudgetattn_map": str(output_dir / "recommended_layerbudget_map.json"),
        },
    }

    fields = list(profile_rows[0].keys()) if profile_rows else []
    write_csv(output_dir / "profile_by_layer_head.csv", profile_rows, fields)
    (output_dir / "recommended_hybrid_policy.json").write_text(
        json.dumps(policy, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "recommended_layerbudget_map.json").write_text(
        json.dumps(layer_budget_map, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "profile_rows": len(profile_rows),
                "layer_count": layer_count,
                "head_count": head_count,
                "template_counts": dict(Counter(row["recommended_template"] for row in profile_rows)),
                "outputs": {
                    "profile_by_layer_head": str(output_dir / "profile_by_layer_head.csv"),
                    "recommended_hybrid_policy": str(output_dir / "recommended_hybrid_policy.json"),
                    "recommended_layerbudget_map": str(output_dir / "recommended_layerbudget_map.json"),
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(json.loads((output_dir / "summary.json").read_text(encoding="utf-8")), indent=2))


if __name__ == "__main__":
    main()
