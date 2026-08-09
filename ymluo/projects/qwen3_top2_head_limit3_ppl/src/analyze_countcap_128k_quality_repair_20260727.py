from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze strictly paired CountCap 128K repair variants."
    )
    parser.add_argument("--baseline_root", type=Path, required=True)
    parser.add_argument("--repair_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"expected a list in {path}")
    return [dict(row) for row in payload]


def case_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["topic"]), int(row["window"])


def collect_baseline(root: Path) -> tuple[
    dict[tuple[str, int], dict[str, Any]],
    dict[tuple[str, int], dict[str, Any]],
]:
    full: dict[tuple[str, int], dict[str, Any]] = {}
    current: dict[tuple[str, int], dict[str, Any]] = {}
    for path in sorted(root.glob("length128000_*/case_summary.json")):
        for row in load_rows(path):
            key = case_key(row)
            if row["method"] == "full_attention":
                full[key] = row
            elif row["method"] == "direct_countcap":
                current[key] = row
    if not full:
        raise ValueError(f"no Full rows found under {root}")
    if set(current) != set(full):
        raise ValueError("current CountCap rows are not paired with Full")
    return full, current


def collect_variants(
    root: Path,
) -> dict[str, dict[tuple[str, int], dict[str, Any]]]:
    variants: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    for path in sorted(root.glob("*/case_summary.json")):
        rows = {
            case_key(row): row
            for row in load_rows(path)
            if row["method"] != "full_attention"
        }
        if rows:
            variants[path.parent.name] = rows
    return variants


def weighted_mean(rows: list[dict[str, Any]], field: str) -> float:
    total_weight = sum(int(row["tokens"]) for row in rows)
    return (
        sum(float(row[field]) * int(row["tokens"]) for row in rows)
        / total_weight
    )


def summarize_variant(
    label: str,
    full_by_case: dict[tuple[str, int], dict[str, Any]],
    sparse_by_case: dict[tuple[str, int], dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if set(sparse_by_case) != set(full_by_case):
        missing = sorted(set(full_by_case) - set(sparse_by_case))
        extra = sorted(set(sparse_by_case) - set(full_by_case))
        raise ValueError(f"{label}: unpaired cases; missing={missing}, extra={extra}")
    keys = sorted(full_by_case)
    full_rows = [full_by_case[key] for key in keys]
    sparse_rows = [sparse_by_case[key] for key in keys]
    full_nll = weighted_mean(full_rows, "nll")
    sparse_nll = weighted_mean(sparse_rows, "nll")
    full_steps = sum(max(0, int(row["tokens"]) - 1) for row in full_rows)
    sparse_steps = sum(max(0, int(row["tokens"]) - 1) for row in sparse_rows)
    full_seconds = sum(float(row["sparse_decode_seconds"]) for row in full_rows)
    sparse_seconds = sum(
        float(row["sparse_decode_seconds"]) for row in sparse_rows
    )
    actual_weight = sum(max(0, int(row["tokens"]) - 1) for row in sparse_rows)
    actual_mean = (
        sum(
            float(row["actual_attention_tokens_mean"])
            * max(0, int(row["tokens"]) - 1)
            for row in sparse_rows
        )
        / actual_weight
    )
    summary = {
        "variant": label,
        "paired_cases": len(keys),
        "tokens": sum(int(row["tokens"]) for row in sparse_rows),
        "full_nll": full_nll,
        "sparse_nll": sparse_nll,
        "delta_nll": sparse_nll - full_nll,
        "full_ppl": math.exp(full_nll),
        "sparse_ppl": math.exp(sparse_nll),
        "quality_retention": math.exp(full_nll - sparse_nll),
        "configured_tokens_per_head": sum(
            float(row["configured_attention_tokens_mean"])
            for row in sparse_rows
        )
        / len(sparse_rows),
        "actual_tokens_per_head": actual_mean,
        "actual_tokens_min": min(
            float(row["actual_attention_tokens_min"]) for row in sparse_rows
        ),
        "actual_tokens_max": max(
            float(row["actual_attention_tokens_max"]) for row in sparse_rows
        ),
        "full_ms_per_step": 1000.0 * full_seconds / full_steps,
        "sparse_ms_per_step": 1000.0 * sparse_seconds / sparse_steps,
        "decode_speedup": (
            (full_seconds / full_steps) / (sparse_seconds / sparse_steps)
        ),
        "score_mode": str(sparse_rows[0]["score_mode"]),
        "method": str(sparse_rows[0]["method"]),
        "projection_dim": int(sparse_rows[0].get("projection_dim", 0)),
        "protected_recent_tokens": int(
            sparse_rows[0].get("protected_recent_tokens", 0)
        ),
        "candidate_overfetch": float(
            sparse_rows[0].get("candidate_overfetch", 1.0)
        ),
        "pca_basis_source_history_count": int(
            sparse_rows[0].get("pca_basis_source_history_count", 0)
        ),
        "pca_basis_sample_count": int(
            sparse_rows[0].get("pca_basis_sample_count", 0)
        ),
    }
    cases = []
    for key in keys:
        full = full_by_case[key]
        sparse = sparse_by_case[key]
        cases.append(
            {
                "variant": label,
                "topic": key[0],
                "window": key[1],
                "full_ppl": float(full["ppl"]),
                "sparse_ppl": float(sparse["ppl"]),
                "delta_nll": float(sparse["nll"]) - float(full["nll"]),
                "quality_retention": math.exp(
                    float(full["nll"]) - float(sparse["nll"])
                ),
                "actual_tokens_per_head": float(
                    sparse["actual_attention_tokens_mean"]
                ),
                "actual_tokens_min": float(
                    sparse["actual_attention_tokens_min"]
                ),
                "actual_tokens_max": float(
                    sparse["actual_attention_tokens_max"]
                ),
            }
        )
    return summary, cases


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summaries: list[dict[str, Any]]) -> None:
    lines = [
        "# CountCap 128K quality repair",
        "",
        "| Variant | Rank | Recent | PPL | Retention | Actual tok/head | Range | Decode |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {variant} | {projection_dim} | {protected_recent_tokens} | "
            "{sparse_ppl:.4f} | {quality_retention:.2%} | "
            "{actual_tokens_per_head:.1f} | {actual_tokens_min:.0f}-"
            "{actual_tokens_max:.0f} | {decode_speedup:.3f}x |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    full, current = collect_baseline(args.baseline_root)
    variants = {"sampled_k1280_current": current}
    variants.update(collect_variants(args.repair_root))
    summaries = []
    cases = []
    for label, rows in sorted(variants.items()):
        if set(rows) != set(full):
            continue
        summary, case_rows = summarize_variant(label, full, rows)
        summaries.append(summary)
        cases.extend(case_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2),
        encoding="utf-8",
    )
    write_csv(args.output_dir / "summary.csv", summaries)
    write_csv(args.output_dir / "case_breakdown.csv", cases)
    write_markdown(args.output_dir / "summary.md", summaries)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
