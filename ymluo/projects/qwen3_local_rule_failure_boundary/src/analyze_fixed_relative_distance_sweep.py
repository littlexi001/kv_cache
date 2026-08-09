from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import linregress, spearmanr


TARGET_ROLE = "hop2_result"


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or np.all(left == left.flat[0]) or np.all(right == right.flat[0]):
        return 0.0
    value = float(spearmanr(left, right).statistic)
    return value if math.isfinite(value) else 0.0


def residualize(values: np.ndarray, design: np.ndarray) -> np.ndarray:
    return values - design @ np.linalg.lstsq(design, values, rcond=None)[0]


def correlations(values: np.ndarray, log_ppl: np.ndarray, design: np.ndarray) -> dict[str, float]:
    return {
        "raw_spearman": safe_spearman(values, log_ppl),
        "length_residual_spearman": safe_spearman(
            residualize(values, design), residualize(log_ppl, design)
        ),
        "adjacent_delta_spearman": safe_spearman(np.diff(values), np.diff(log_ppl)),
    }


def read_rows(data_dir: Path, *, fixed_body_overhead: int | None) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for path in data_dir.glob("length_*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        attention = payload["attention"]
        role_index = attention["role_order"].index(TARGET_ROLE)
        target = int(payload["target_context_tokens"])
        filler = target if fixed_body_overhead is None else target - fixed_body_overhead
        role_position = int(payload["spans"][TARGET_ROLE][0][0])
        query_position = int(attention["key_length"]) - 1
        role_mass = np.asarray(attention["head_role_mass"], dtype=float)[:, :, role_index]
        role_logit = np.asarray(attention["head_role_logit_mean"], dtype=float)[:, :, role_index]
        role_cosine = np.asarray(attention["head_role_cosine_mean"], dtype=float)[:, :, role_index]
        role_key_norm = np.asarray(attention["head_role_key_norm_mean"], dtype=float)[:, :, role_index]
        role_rank = np.asarray(attention["head_role_best_rank"], dtype=float)[:, :, role_index]
        query_norm = np.asarray(attention["head_query_norm"], dtype=float)
        head_lse = np.asarray(attention["head_logsumexp"], dtype=float)
        head_max = np.asarray(attention["head_max_logit"], dtype=float)
        top2 = np.asarray(attention.get("head_top2pct_role_mass", []), dtype=float)
        top2_fraction = float("nan")
        if top2.size:
            top2_fraction = float((top2[:, :, role_index] > 0).mean())
        rows.append(
            {
                "filler_tokens": float(filler),
                "target_context_tokens": float(target),
                "prompt_tokens": float(payload["prompt_tokens"]),
                "key_length": float(attention["key_length"]),
                "evidence_position": float(role_position),
                "query_position": float(query_position),
                "relative_distance": float(query_position - role_position),
                "gold_ppl": float(payload["answer"]["gold_ppl"]),
                "gold_probability": 1.0 / float(payload["answer"]["gold_ppl"]),
                "mean_evidence_mass": float(role_mass.mean()),
                "mean_evidence_logit": float(role_logit.mean()),
                "mean_evidence_cosine": float(role_cosine.mean()),
                "mean_evidence_key_norm": float(role_key_norm.mean()),
                "mean_query_norm": float(query_norm.mean()),
                "mean_evidence_rank": float(role_rank.mean()),
                "mean_head_logsumexp": float(head_lse.mean()),
                "mean_head_max_logit": float(head_max.mean()),
                "target_top2pct_head_fraction": top2_fraction,
            }
        )
    rows.sort(key=lambda row: row["filler_tokens"])
    return rows


def group_summary(rows: list[dict[str, float]], low: float, high: float) -> dict[str, float]:
    selected = [row for row in rows if low <= row["filler_tokens"] <= high]
    output: dict[str, float] = {"sample_count": float(len(selected))}
    for metric in (
        "gold_ppl",
        "gold_probability",
        "mean_evidence_mass",
        "mean_evidence_logit",
        "mean_evidence_cosine",
        "mean_evidence_key_norm",
        "mean_query_norm",
        "mean_evidence_rank",
        "mean_head_logsumexp",
    ):
        values = np.asarray([row[metric] for row in selected], dtype=float)
        output[f"{metric}_mean"] = float(values.mean())
        output[f"{metric}_median"] = float(np.median(values))
    return output


def summarize(rows: list[dict[str, float]]) -> dict[str, Any]:
    filler = np.asarray([row["filler_tokens"] for row in rows], dtype=float)
    ppl = np.asarray([row["gold_ppl"] for row in rows], dtype=float)
    log_ppl = np.log(ppl)
    normalized = np.log1p(filler) / np.log1p(max(filler.max(), 1.0))
    design = np.column_stack([np.ones(len(rows)), normalized, normalized**2, normalized**3])
    metrics = {
        metric: np.asarray([row[metric] for row in rows], dtype=float)
        for metric in (
            "mean_evidence_mass",
            "mean_evidence_logit",
            "mean_evidence_cosine",
            "mean_evidence_key_norm",
            "mean_query_norm",
            "mean_evidence_rank",
            "mean_head_logsumexp",
        )
    }
    short = group_summary(rows, 0, 8000)
    long = group_summary(rows, 120000, 128000)
    delta_logit = long["mean_evidence_logit_mean"] - short["mean_evidence_logit_mean"]
    delta_lse = long["mean_head_logsumexp_mean"] - short["mean_head_logsumexp_mean"]
    long_mask = filler >= 8000
    lse_regression = linregress(
        np.log(np.asarray([row["key_length"] for row in rows], dtype=float)[long_mask]),
        metrics["mean_head_logsumexp"][long_mask],
    )
    bins = []
    for label, low, high in (
        ("0-8K", 0, 8000),
        ("8-32K", 8001, 32000),
        ("32-64K", 32001, 64000),
        ("64-96K", 64001, 96000),
        ("96-120K", 96001, 120000),
        ("120-128K", 120001, 128000),
    ):
        selected = ppl[(filler >= low) & (filler <= high)]
        bins.append(
            {
                "label": label,
                "count": int(selected.size),
                "gold_ppl_median": float(np.median(selected)),
                "gold_ppl_mean": float(selected.mean()),
            }
        )
    return {
        "length_count": len(rows),
        "filler_min": int(filler.min()),
        "filler_max": int(filler.max()),
        "relative_distance_unique": sorted({int(row["relative_distance"]) for row in rows}),
        "short_definition": "filler_tokens <= 8000",
        "long_definition": "filler_tokens >= 120000",
        "short": short,
        "long": long,
        "median_ppl_factor_long_over_short": (
            long["gold_ppl_median"] / short["gold_ppl_median"]
        ),
        "correlations_with_log_ppl": {
            metric: correlations(values, log_ppl, design) for metric, values in metrics.items()
        },
        "attention_log_mass_decomposition": {
            "delta_evidence_logit": delta_logit,
            "delta_logsumexp": delta_lse,
            "delta_geometric_log_mass": delta_logit - delta_lse,
            "numerator_factor": math.exp(delta_logit),
            "competition_factor": math.exp(-delta_lse),
            "combined_factor": math.exp(delta_logit - delta_lse),
        },
        "logsumexp_vs_log_key_length_ge_8k": {
            "slope": float(lse_regression.slope),
            "r_squared": float(lse_regression.rvalue**2),
            "p_value": float(lse_regression.pvalue),
        },
        "ppl_bins": bins,
    }


def comparison_rows(fixed_rows: list[dict[str, float]], middle_rows: list[dict[str, float]]) -> list[dict[str, float]]:
    middle_by_length = {int(row["filler_tokens"]): row for row in middle_rows}
    output = []
    for fixed in fixed_rows:
        filler = int(fixed["filler_tokens"])
        middle = middle_by_length.get(filler)
        if middle is None:
            continue
        output.append(
            {
                "filler_tokens": float(filler),
                "fixed_gold_ppl": fixed["gold_ppl"],
                "middle_gold_ppl": middle["gold_ppl"],
                "fixed_over_middle_ppl": fixed["gold_ppl"] / middle["gold_ppl"],
                "fixed_relative_distance": fixed["relative_distance"],
                "middle_relative_distance": middle["relative_distance"],
                "fixed_evidence_logit": fixed["mean_evidence_logit"],
                "middle_evidence_logit": middle["mean_evidence_logit"],
                "logit_fixed_minus_middle": fixed["mean_evidence_logit"] - middle["mean_evidence_logit"],
                "fixed_evidence_cosine": fixed["mean_evidence_cosine"],
                "middle_evidence_cosine": middle["mean_evidence_cosine"],
                "cosine_fixed_minus_middle": fixed["mean_evidence_cosine"] - middle["mean_evidence_cosine"],
                "fixed_evidence_mass": fixed["mean_evidence_mass"],
                "middle_evidence_mass": middle["mean_evidence_mass"],
                "mass_fixed_over_middle": fixed["mean_evidence_mass"] / max(middle["mean_evidence_mass"], 1e-30),
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a fixed-evidence-query-distance 0-128K filler sweep.")
    parser.add_argument("--fixed_data_dir", required=True)
    parser.add_argument("--middle_data_dir", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--fixed_body_overhead", type=int, default=290)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fixed_rows = read_rows(Path(args.fixed_data_dir), fixed_body_overhead=args.fixed_body_overhead)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "model": "Qwen3-8B",
        "condition": "clean two-hop English single-token full2; fixed evidence-query distance",
        "fixed_body_overhead": args.fixed_body_overhead,
        "fixed": summarize(fixed_rows),
    }
    write_csv(output_dir / "fixed_relative_rows.csv", fixed_rows)
    if args.middle_data_dir:
        middle_rows = read_rows(Path(args.middle_data_dir), fixed_body_overhead=None)
        aligned = comparison_rows(fixed_rows, middle_rows)
        write_csv(output_dir / "fixed_vs_middle_rows.csv", aligned)
        summary["middle"] = summarize(middle_rows)
        summary["fixed_vs_middle"] = {
            "aligned_count": len(aligned),
            "long_fixed_over_middle_ppl_median": float(
                np.median(
                    [row["fixed_over_middle_ppl"] for row in aligned if row["filler_tokens"] >= 120000]
                )
            ),
            "long_logit_fixed_minus_middle_mean": float(
                np.mean(
                    [row["logit_fixed_minus_middle"] for row in aligned if row["filler_tokens"] >= 120000]
                )
            ),
            "long_cosine_fixed_minus_middle_mean": float(
                np.mean(
                    [row["cosine_fixed_minus_middle"] for row in aligned if row["filler_tokens"] >= 120000]
                )
            ),
            "long_mass_fixed_over_middle_median": float(
                np.median(
                    [row["mass_fixed_over_middle"] for row in aligned if row["filler_tokens"] >= 120000]
                )
            ),
        }
    (output_dir / "fixed_relative_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output_dir), **summary["fixed"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
