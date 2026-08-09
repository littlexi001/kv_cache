from __future__ import annotations

import argparse
import base64
import csv
import gzip
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import median_filter
from scipy.stats import linregress, spearmanr


EVIDENCE_ROLES = ("hop1_result", "hop2_input", "hop2_result")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_gzip_json(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def decode_array(value: str, dtype: str, shape: tuple[int, ...]) -> np.ndarray:
    output = np.frombuffer(base64.b64decode(value), dtype=dtype)
    expected = int(np.prod(shape))
    if output.size != expected:
        raise ValueError(f"decoded {output.size} values, expected {expected} for {shape}")
    return output.reshape(shape)


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or right.size < 2:
        return 0.0
    if np.all(left == left.flat[0]) or np.all(right == right.flat[0]):
        return 0.0
    result = float(spearmanr(left, right).statistic)
    return result if math.isfinite(result) else 0.0


def residualize(values: np.ndarray, design: np.ndarray) -> np.ndarray:
    return values - design @ np.linalg.lstsq(design, values, rcond=None)[0]


def correlations(
    values: np.ndarray,
    log_ppl: np.ndarray,
    length_design: np.ndarray,
) -> dict[str, float]:
    return {
        "raw_spearman": safe_spearman(values, log_ppl),
        "length_residual_spearman": safe_spearman(
            residualize(values, length_design),
            residualize(log_ppl, length_design),
        ),
        "adjacent_delta_spearman": safe_spearman(np.diff(values), np.diff(log_ppl)),
    }


def mean_or_median(values: np.ndarray) -> dict[str, float]:
    return {"mean": float(np.mean(values)), "median": float(np.median(values))}


def piecewise_changepoint(lengths: np.ndarray, values: np.ndarray, window: int = 9) -> dict[str, float]:
    smoothed = median_filter(values, size=window, mode="nearest")
    best: tuple[float, int, np.ndarray] | None = None
    for index in range(32, len(lengths) - 32):
        hinge = np.maximum(0.0, lengths - lengths[index])
        design = np.column_stack([np.ones(len(lengths)), lengths, hinge])
        coefficients = np.linalg.lstsq(design, smoothed, rcond=None)[0]
        residual_sum = float(np.square(smoothed - design @ coefficients).sum())
        if best is None or residual_sum < best[0]:
            best = (residual_sum, index, coefficients)
    if best is None:
        raise RuntimeError("not enough points for changepoint analysis")
    _, index, coefficients = best
    return {
        "length": int(lengths[index]),
        "slope_before_per_10k": float(coefficients[1] * 10_000),
        "slope_after_per_10k": float((coefficients[1] + coefficients[2]) * 10_000),
        "median_filter_points": window,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def load_token_type(
    data_root: Path,
    manifest: dict[str, Any],
    token: str,
) -> dict[str, np.ndarray]:
    entry = manifest["tokens"][token]
    payload = read_gzip_json(data_root / "token_type_all_lengths" / entry["file"])
    shape = tuple(int(value) for value in payload["shape"])
    return {
        "counts": decode_array(payload["occurrence_counts_u32_b64"], "<u4", (shape[0],)),
        "logits": decode_array(payload["mean_logits_f16_b64"], "<f2", shape).astype(np.float64),
        "mass": decode_array(payload["probability_mass_f32_b64"], "<f4", shape).astype(np.float64),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose why gold-answer PPL degrades with context length.")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = read_json(data_root / "manifest.json")
    pre = read_json(data_root / "pre_softmax_summary.json")
    token_manifest = read_json(data_root / "token_type_all_lengths" / "manifest.json")
    pre_rows = pre["rows"]
    summaries = {int(row["length"]): row for row in manifest["summaries"]}
    lengths = np.asarray([int(row["length"]) for row in pre_rows], dtype=np.int64)
    if lengths.tolist() != token_manifest["lengths"]:
        raise ValueError("token-type and pre-softmax length grids do not match")
    log_lengths = np.log1p(lengths.astype(np.float64))
    ppl = np.asarray([float(row["gold_ppl"]) for row in pre_rows])
    log_ppl = np.log(ppl)
    length_design = np.column_stack(
        [np.ones(len(lengths)), log_lengths, log_lengths**2, log_lengths**3]
    )
    role_order = manifest["role_order"]

    token_curves = {
        token: load_token_type(data_root, token_manifest, token)
        for token in ("river", "window", "basket", "office", "the", "and")
        if token in token_manifest["tokens"]
    }

    metric_arrays: dict[str, np.ndarray] = {
        "length": lengths.astype(np.float64),
        "mean_head_logsumexp": np.asarray([row["mean_head_logsumexp"] for row in pre_rows]),
        "mean_head_max_logit": np.asarray([row["mean_head_max_logit"] for row in pre_rows]),
        "mean_query_norm": np.asarray([row["mean_query_norm"] for row in pre_rows]),
        "overall_entropy": np.asarray([summaries[int(length)]["overall_entropy"] for length in lengths]),
        "overall_effective_tokens": np.asarray(
            [summaries[int(length)]["overall_effective_tokens"] for length in lengths]
        ),
    }
    for role in EVIDENCE_ROLES:
        for key in (
            "mean_logit",
            "mean_cosine",
            "mean_rank",
            "mean_rank_percentile",
            "top2pct_head_fraction",
            "top100_head_fraction",
            "mean_max_logit_gap",
        ):
            metric_arrays[f"{role}.{key}"] = np.asarray(
                [row["roles"][role][key] for row in pre_rows]
            )
    for role in role_order:
        role_index = role_order.index(role)
        metric_arrays[f"mass.{role}"] = np.asarray(
            [summaries[int(length)]["overall_role_mass"][role_index] for length in lengths]
        )
    for token, curves in token_curves.items():
        metric_arrays[f"token_mass.{token}"] = curves["mass"].mean(axis=(1, 2))
        metric_arrays[f"token_logit.{token}"] = curves["logits"].mean(axis=(1, 2))

    length_rows: list[dict[str, Any]] = []
    for index, length in enumerate(lengths):
        row: dict[str, Any] = {
            "length": int(length),
            "prompt_tokens": int(pre_rows[index]["prompt_tokens"]),
            "gold_ppl": float(ppl[index]),
            "gold_probability": float(1.0 / ppl[index]),
            "log_gold_ppl": float(log_ppl[index]),
        }
        for name, values in metric_arrays.items():
            if name != "length":
                row[name] = float(values[index])
        length_rows.append(row)
    write_csv(output_dir / "length_mechanism_rows.csv", length_rows)

    short = lengths <= 8_000
    long = lengths >= 120_000
    bins = [0, 1_000, 8_000, 16_000, 32_000, 64_000, 96_000, 120_000, 128_001]
    binned: list[dict[str, Any]] = []
    for start, stop in zip(bins[:-1], bins[1:]):
        selected = (lengths >= start) & (lengths < stop)
        binned.append(
            {
                "start": start,
                "stop_exclusive": stop,
                "count": int(selected.sum()),
                "ppl_median": float(np.median(ppl[selected])),
                "ppl_p90": float(np.percentile(ppl[selected], 90)),
                "gold_probability_median": float(np.median(1.0 / ppl[selected])),
                "logsumexp_mean": float(metric_arrays["mean_head_logsumexp"][selected].mean()),
                "hop2_result_logit_mean": float(
                    metric_arrays["hop2_result.mean_logit"][selected].mean()
                ),
                "hop2_result_cosine_mean": float(
                    metric_arrays["hop2_result.mean_cosine"][selected].mean()
                ),
                "hop2_result_mass_mean": float(metric_arrays["mass.hop2_result"][selected].mean()),
                "effective_tokens_mean": float(
                    metric_arrays["overall_effective_tokens"][selected].mean()
                ),
            }
        )

    correlation_summary = {
        name: correlations(values, log_ppl, length_design)
        for name, values in metric_arrays.items()
    }

    head_payload = read_gzip_json(data_root / "pre_softmax_head_length_summary.json.gz")
    head_shape = tuple(int(value) for value in head_payload["shape"])
    role_logits = decode_array(
        head_payload["role_logits_f16_b64"], "<f2", head_shape
    ).astype(np.float64)
    role_mass = decode_array(
        head_payload["role_mass_f32_b64"], "<f4", head_shape
    ).astype(np.float64)
    head_shape_without_role = (head_shape[0], head_shape[2], head_shape[3])
    head_logsumexp = decode_array(
        head_payload["head_logsumexp_f16_b64"], "<f2", head_shape_without_role
    ).astype(np.float64)

    decompositions: dict[str, Any] = {}
    for role in EVIDENCE_ROLES:
        role_index = head_payload["role_order"].index(role)
        short_logit = float(role_logits[short, role_index].mean())
        long_logit = float(role_logits[long, role_index].mean())
        short_lse = float(head_logsumexp[short].mean())
        long_lse = float(head_logsumexp[long].mean())
        delta_logit = long_logit - short_logit
        delta_lse = long_lse - short_lse
        delta_log_mass = delta_logit - delta_lse
        total_attenuation = -delta_log_mass
        decompositions[role] = {
            "short_mean_logit": short_logit,
            "long_mean_logit": long_logit,
            "delta_logit": delta_logit,
            "short_mean_logsumexp": short_lse,
            "long_mean_logsumexp": long_lse,
            "delta_logsumexp": delta_lse,
            "delta_geometric_log_mass": delta_log_mass,
            "raw_numerator_factor": float(math.exp(delta_logit)),
            "competition_factor": float(math.exp(-delta_lse)),
            "combined_geometric_mass_factor": float(math.exp(delta_log_mass)),
            "direction_share_of_log_attenuation": float((-delta_logit) / total_attenuation),
            "competition_share_of_log_attenuation": float(delta_lse / total_attenuation),
            "arithmetic_mass_short": float(metric_arrays[f"mass.{role}"][short].mean()),
            "arithmetic_mass_long": float(metric_arrays[f"mass.{role}"][long].mean()),
            "arithmetic_mass_factor": float(
                metric_arrays[f"mass.{role}"][long].mean()
                / metric_arrays[f"mass.{role}"][short].mean()
            ),
        }

    hop2_index = head_payload["role_order"].index("hop2_result")
    layer_rows: list[dict[str, Any]] = []
    for layer in range(head_shape[2]):
        layer_mass = role_mass[:, hop2_index, layer, :].mean(axis=1)
        layer_logit = role_logits[:, hop2_index, layer, :].mean(axis=1)
        short_mass = float(layer_mass[short].mean())
        long_mass = float(layer_mass[long].mean())
        layer_rows.append(
            {
                "layer": layer,
                "mass_logppl_spearman": safe_spearman(layer_mass, log_ppl),
                "logit_logppl_spearman": safe_spearman(layer_logit, log_ppl),
                "mass_short": short_mass,
                "mass_long": long_mass,
                "mass_retention_factor": long_mass / short_mass,
                "delta_mean_logit": float(layer_logit[long].mean() - layer_logit[short].mean()),
            }
        )
    write_csv(output_dir / "hop2_result_layer_diagnostics.csv", layer_rows)

    concentration: dict[str, Any] = {}
    for label, selected in {
        "short_le_8k": short,
        "mid_64k_to_96k": (lengths >= 64_000) & (lengths < 96_000),
        "long_ge_120k": long,
    }.items():
        values = role_mass[selected, hop2_index].reshape(int(selected.sum()), -1)
        ordered = np.sort(values, axis=1)[:, ::-1]
        totals = ordered.sum(axis=1)
        concentration[label] = {
            "top_1pct_head_mass_fraction": float(np.mean(ordered[:, :12].sum(axis=1) / totals)),
            "top_5pct_head_mass_fraction": float(np.mean(ordered[:, :58].sum(axis=1) / totals)),
            "top_10pct_head_mass_fraction": float(np.mean(ordered[:, :115].sum(axis=1) / totals)),
            "heads_above_1e_3_mean": float(np.mean((values > 1e-3).sum(axis=1))),
        }

    key_lengths = np.asarray([row["key_length"] for row in pre_rows], dtype=np.float64)
    lse_regression_mask = lengths >= 8_000
    lse_regression = linregress(
        np.log(key_lengths[lse_regression_mask]),
        metric_arrays["mean_head_logsumexp"][lse_regression_mask],
    )

    concrete_lengths = [0, 8_000, 32_000, 56_000, 64_000, 72_000, 96_000, 120_000, 128_000]
    concrete = [length_rows[int(np.flatnonzero(lengths == length)[0])] for length in concrete_lengths]

    summary = {
        "schema_version": 1,
        "model": manifest["model"],
        "condition": "clean two-hop, English single-token evidence, middle placement, one seed",
        "length_count": len(lengths),
        "length_min": int(lengths.min()),
        "length_max": int(lengths.max()),
        "target_answer_token": "basket",
        "ppl_is_inverse_gold_probability": True,
        "short_definition": "length <= 8000",
        "long_definition": "length >= 120000",
        "short_ppl": mean_or_median(ppl[short]),
        "long_ppl": mean_or_median(ppl[long]),
        "median_ppl_factor_long_over_short": float(np.median(ppl[long]) / np.median(ppl[short])),
        "length_logppl_spearman": safe_spearman(lengths.astype(float), log_ppl),
        "binned": binned,
        "correlations_with_log_ppl": correlation_summary,
        "attention_decomposition": decompositions,
        "hop2_result_head_concentration": concentration,
        "hop2_result_layer_diagnostics": layer_rows,
        "hop2_result_direction_changepoint": piecewise_changepoint(
            lengths.astype(float), metric_arrays["hop2_result.mean_logit"]
        ),
        "hop2_result_cosine_changepoint": piecewise_changepoint(
            lengths.astype(float), metric_arrays["hop2_result.mean_cosine"]
        ),
        "logsumexp_vs_log_key_length_ge_8k": {
            "slope": float(lse_regression.slope),
            "r_squared": float(lse_regression.rvalue**2),
            "p_value": float(lse_regression.pvalue),
        },
        "concrete_length_rows": concrete,
        "caveats": [
            "Only one synthetic instance and one seed were swept; adjacent 500-token points are highly variable.",
            "Observed associations do not replace a causal attention intervention.",
            "Token-type mass sums all matching positions, whereas marked evidence-role mass isolates the true chain position.",
        ],
    }
    (output_dir / "mechanism_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "output_dir": str(output_dir),
        "length_count": len(lengths),
        "median_ppl_factor": summary["median_ppl_factor_long_over_short"],
        "hop2_mass_spearman": correlation_summary["mass.hop2_result"],
        "hop2_decomposition": decompositions["hop2_result"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
