from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


def target_budget(history_tokens: int) -> int:
    return min(
        history_tokens,
        1280,
        max(256, math.ceil(0.06 * history_tokens)),
    )


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def quantile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot take a quantile of an empty list")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def small_nnls(
    design: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Solve a tiny NNLS problem by enumerating active coefficient sets."""

    column_count = int(design.shape[1])
    best_coefficients: torch.Tensor | None = None
    best_error = float("inf")
    for active_count in range(1, column_count + 1):
        for active in itertools.combinations(range(column_count), active_count):
            active_index = torch.tensor(active, dtype=torch.long)
            sub_design = design.index_select(1, active_index)
            solution = torch.linalg.lstsq(sub_design, target).solution
            if torch.any(solution < -1.0e-9):
                continue
            coefficients = torch.zeros(
                column_count,
                dtype=design.dtype,
            )
            coefficients[active_index] = solution.clamp_min(0.0)
            error = float(
                (design @ coefficients - target).square().sum().item()
            )
            if error < best_error:
                best_error = error
                best_coefficients = coefficients
    if best_coefficients is None:
        return torch.zeros(column_count, dtype=design.dtype)
    return best_coefficients


def fit_model(
    predictors: list[list[float]],
    targets: list[float],
    names: tuple[str, ...],
) -> dict[str, Any]:
    design = torch.tensor(predictors, dtype=torch.float64)
    target = torch.tensor(targets, dtype=torch.float64)
    coefficients = small_nnls(design, target)
    prediction = design @ coefficients
    residual = target - prediction
    total = (target - target.mean()).square().sum().clamp_min(1.0e-30)
    r_squared = 1.0 - residual.square().sum() / total
    return {
        "coefficients": {
            name: float(value.item())
            for name, value in zip(names, coefficients)
        },
        "prediction": [float(value.item()) for value in prediction],
        "rmse_ms": float(residual.square().mean().sqrt().item()),
        "r_squared": float(r_squared.item()),
    }


def load_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("length*/case_summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"{path} must contain a JSON list")
        rows.extend(payload)
    if not rows:
        raise ValueError(f"no length*/case_summary.json files under {root}")
    return rows


def load_token_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("length*/token_results.csv")):
        match = re.fullmatch(r"length(\d+)", path.parent.name)
        if match is None:
            continue
        history_tokens = int(match.group(1))
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append(
                    {
                        **row,
                        "history_tokens": history_tokens,
                    }
                )
    if not rows:
        raise ValueError(f"no length*/token_results.csv files under {root}")
    return rows


def load_budget_probe_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    default_gpu = {
        "budget320": "gpu2",
        "budget960": "gpu2",
        "budget640": "gpu3",
        "budget1280": "gpu3",
    }
    for path in sorted(root.glob("*/token_results.csv")):
        run_name = path.parent.name
        if run_name.startswith("cross_gpu2_"):
            gpu = "gpu2"
        elif run_name.startswith("cross_gpu3_"):
            gpu = "gpu3"
        else:
            gpu = default_gpu.get(run_name, run_name)
        budget_match = re.search(r"budget(\d+)", run_name)
        if budget_match is None:
            continue
        configured_budget = int(budget_match.group(1))
        grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("method") != "direct_countcap":
                    continue
                if int(row["target_index"]) <= 0:
                    continue
                grouped[(str(row["topic"]), int(row["window"]))].append(row)
        for (topic, window), items in grouped.items():
            ordered = sorted(items, key=lambda row: int(row["target_index"]))
            steady = ordered[1:]
            if not steady:
                continue
            rows.append(
                {
                    "probe_run": run_name,
                    "probe_gpu": gpu,
                    "topic": topic,
                    "window": window,
                    "configured_attention_tokens_mean": configured_budget,
                    "actual_attention_tokens_mean": mean(
                        [
                            float(row["actual_attention_tokens_mean"])
                            for row in steady
                        ]
                    ),
                    "sparse_seconds_per_step": mean(
                        [
                            float(row["sparse_forward_seconds"])
                            for row in steady
                        ]
                    ),
                }
            )
    if not rows:
        raise ValueError(
            f"no direct CountCap steady-state budget probes under {root}"
        )
    return rows


def fit_fixed_effect_budget_slope(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Estimate dT/dB after removing GPU/topic/window fixed effects."""

    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["probe_gpu"]),
                str(row["topic"]),
                int(row["window"]),
            )
        ].append(row)
    centered_budget: list[float] = []
    centered_time: list[float] = []
    usable_groups = 0
    for items in grouped.values():
        budgets = {
            round(float(row["configured_attention_tokens_mean"]))
            for row in items
        }
        if len(budgets) < 3:
            continue
        usable_groups += 1
        budget_values = [
            float(row["actual_attention_tokens_mean"]) for row in items
        ]
        time_values = [
            float(row["sparse_seconds_per_step"]) * 1000.0
            for row in items
        ]
        budget_center = mean(budget_values)
        time_center = mean(time_values)
        centered_budget.extend(value - budget_center for value in budget_values)
        centered_time.extend(value - time_center for value in time_values)
    if usable_groups == 0:
        raise ValueError("budget probes do not contain complete fixed-effect groups")

    x = torch.tensor(centered_budget, dtype=torch.float64)
    y = torch.tensor(centered_time, dtype=torch.float64)
    sum_squares = x.square().sum().clamp_min(1.0e-30)
    raw_slope = (x * y).sum() / sum_squares
    residual = y - raw_slope * x
    degrees_of_freedom = max(1, len(centered_budget) - usable_groups - 1)
    residual_variance = residual.square().sum() / degrees_of_freedom
    standard_error = torch.sqrt(residual_variance / sum_squares)
    nonnegative_slope = raw_slope.clamp_min(0.0)
    return {
        "observations": len(centered_budget),
        "fixed_effect_groups": usable_groups,
        "raw_ms_per_attention_token": float(raw_slope.item()),
        "nonnegative_ms_per_attention_token": float(
            nonnegative_slope.item()
        ),
        "standard_error": float(standard_error.item()),
        "raw_95_percent_ci": [
            float((raw_slope - 1.96 * standard_error).item()),
            float((raw_slope + 1.96 * standard_error).item()),
        ],
        "within_group_r_squared": float(
            (
                1.0
                - residual.square().sum()
                / y.square().sum().clamp_min(1.0e-30)
            ).item()
        ),
    }


def sequence_timings(
    token_rows: list[dict[str, Any]],
) -> dict[tuple[int, str, str, int], dict[str, float]]:
    grouped: dict[
        tuple[int, str, str, int],
        list[dict[str, Any]],
    ] = defaultdict(list)
    for row in token_rows:
        if int(row["target_index"]) <= 0:
            continue
        grouped[
            (
                int(row["history_tokens"]),
                str(row["method"]),
                str(row["topic"]),
                int(row["window"]),
            )
        ].append(row)

    output: dict[tuple[int, str, str, int], dict[str, float]] = {}
    for key, items in grouped.items():
        ordered = sorted(items, key=lambda row: int(row["target_index"]))
        if len(ordered) < 2:
            raise ValueError(f"{key} needs at least two timed decode steps")
        first_ms = float(ordered[0]["sparse_forward_seconds"]) * 1000.0
        steady_ms = mean(
            [
                float(row["sparse_forward_seconds"]) * 1000.0
                for row in ordered[1:]
            ]
        )
        output[key] = {
            "timed_steps": float(len(ordered)),
            "first_step_ms": first_ms,
            "steady_ms_per_token": steady_ms,
            "first_step_excess_ms": first_ms - steady_ms,
        }
    return output


def aggregate_lengths(
    rows: list[dict[str, Any]],
    token_rows: list[dict[str, Any]],
) -> list[dict[str, float]]:
    grouped: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        grouped[int(row["history_tokens"])][str(row["method"])].append(row)

    timing = sequence_timings(token_rows)
    output: list[dict[str, float]] = []
    for history_tokens, methods in sorted(grouped.items()):
        if "full_attention" not in methods or "direct_countcap" not in methods:
            continue
        full_rows = methods["full_attention"]
        direct_rows = methods["direct_countcap"]
        full_ms = mean(
            [float(row["sparse_seconds_per_step"]) * 1000.0 for row in full_rows]
        )
        direct_ms = mean(
            [
                float(row["sparse_seconds_per_step"]) * 1000.0
                for row in direct_rows
            ]
        )
        full_prefill = mean(
            [float(row["dense_prompt_seconds"]) for row in full_rows]
        )
        direct_prefill = mean(
            [float(row["dense_prompt_seconds"]) for row in direct_rows]
        )
        actual_budget = mean(
            [float(row["actual_attention_tokens_mean"]) for row in direct_rows]
        )
        configured_budget = mean(
            [
                float(row["configured_attention_tokens_mean"])
                for row in direct_rows
            ]
        )
        pair_ids = sorted(
            {
                (str(row["topic"]), int(row["window"]))
                for row in direct_rows
            }
            & {
                (str(row["topic"]), int(row["window"]))
                for row in full_rows
            }
        )
        full_sequence = [
            timing[
                (
                    history_tokens,
                    "full_attention",
                    topic,
                    window,
                )
            ]
            for topic, window in pair_ids
        ]
        direct_sequence = [
            timing[
                (
                    history_tokens,
                    "direct_countcap",
                    topic,
                    window,
                )
            ]
            for topic, window in pair_ids
        ]
        full_steady_ms = mean(
            [row["steady_ms_per_token"] for row in full_sequence]
        )
        direct_steady_ms = mean(
            [row["steady_ms_per_token"] for row in direct_sequence]
        )
        lazy_index_overheads = [
            direct_item["first_step_excess_ms"]
            - full_item["first_step_excess_ms"]
            for direct_item, full_item in zip(
                direct_sequence,
                full_sequence,
            )
        ]
        lazy_index_ms = max(0.0, median(lazy_index_overheads))
        per_step_saving = (full_steady_ms - direct_steady_ms) / 1000.0
        break_even = (
            (lazy_index_ms / 1000.0) / per_step_saving
            if per_step_saving > 0.0
            else float("inf")
        )
        timed_steps = int(full_sequence[0]["timed_steps"])
        output.append(
            {
                "history_tokens": float(history_tokens),
                "configured_budget": configured_budget,
                "actual_budget": actual_budget,
                "budget_inflation": (
                    actual_budget / max(configured_budget, 1.0)
                ),
                "full_ms_per_token": full_ms,
                "countcap_ms_per_token": direct_ms,
                "decode_speedup": full_ms / direct_ms,
                "timed_decode_steps": float(timed_steps),
                "full_steady_ms_per_token": full_steady_ms,
                "countcap_steady_ms_per_token": direct_steady_ms,
                "steady_decode_speedup": (
                    full_steady_ms / direct_steady_ms
                ),
                "countcap_first_step_ms": mean(
                    [row["first_step_ms"] for row in direct_sequence]
                ),
                "lazy_index_fixed_overhead_ms": lazy_index_ms,
                "lazy_index_fixed_overhead_mean_ms": mean(
                    lazy_index_overheads
                ),
                "lazy_index_fixed_overhead_p10_ms": quantile(
                    lazy_index_overheads,
                    0.10,
                ),
                "lazy_index_fixed_overhead_p90_ms": quantile(
                    lazy_index_overheads,
                    0.90,
                ),
                "countcap_reconstructed_amortized_ms": (
                    direct_steady_ms + lazy_index_ms / timed_steps
                ),
                "full_prefill_seconds": full_prefill,
                "countcap_prefill_seconds": direct_prefill,
                "dense_prefill_pair_delta_seconds": (
                    direct_prefill - full_prefill
                ),
                "per_token_saving_seconds": per_step_saving,
                "break_even_generation_tokens": break_even,
            }
        )
    return output


def find_crossover(
    full_coefficients: dict[str, float],
    sparse_coefficients: dict[str, float],
    budget_inflation: float,
    minimum: int = 512,
    maximum: int = 262144,
) -> int | None:
    previous_difference: float | None = None
    for history_tokens in range(minimum, maximum + 1):
        n_k = history_tokens / 1000.0
        budget_k = (
            target_budget(history_tokens) * budget_inflation / 1000.0
        )
        full = (
            full_coefficients["intercept_ms"]
            + full_coefficients["history_ktokens_ms"] * n_k
        )
        sparse = (
            sparse_coefficients["intercept_ms"]
            + sparse_coefficients["history_ktokens_ms"] * n_k
            + sparse_coefficients["attention_ktokens_ms"] * budget_k
        )
        difference = full - sparse
        if (
            previous_difference is not None
            and previous_difference < 0.0
            and difference >= 0.0
        ):
            return history_tokens
        previous_difference = difference
    return None


def predict_curve(
    full_coefficients: dict[str, float],
    sparse_coefficients: dict[str, float],
    fixed_coefficients: dict[str, float],
    budget_inflation: float,
    lengths: tuple[int, ...],
) -> list[dict[str, float]]:
    rows = []
    for history_tokens in lengths:
        n_k = history_tokens / 1000.0
        target = target_budget(history_tokens)
        actual_budget = target * budget_inflation
        full = (
            full_coefficients["intercept_ms"]
            + full_coefficients["history_ktokens_ms"] * n_k
        )
        sparse = (
            sparse_coefficients["intercept_ms"]
            + sparse_coefficients["history_ktokens_ms"] * n_k
            + sparse_coefficients["attention_ktokens_ms"]
            * (actual_budget / 1000.0)
        )
        fixed = (
            fixed_coefficients["intercept_ms"]
            + fixed_coefficients["history_ktokens_ms"] * n_k
        )
        saving = full - sparse
        break_even = (
            math.ceil(max(0.0, fixed) / saving)
            if saving > 0.0
            else None
        )
        rows.append(
            {
                "history_tokens": float(history_tokens),
                "target_budget": float(target),
                "predicted_actual_budget": float(actual_budget),
                "predicted_full_ms_per_token": float(full),
                "predicted_countcap_ms_per_token": float(sparse),
                "predicted_decode_speedup": float(full / sparse),
                "predicted_lazy_index_fixed_ms": float(max(0.0, fixed)),
                "predicted_break_even_generation_tokens": break_even,
                "is_length_extrapolation": float(
                    history_tokens > 32000
                ),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fit an auditable CountCap decode cost model from the frozen "
            "2K-32K same-hardware length sweep."
        )
    )
    parser.add_argument("--input_root", type=Path, required=True)
    parser.add_argument("--budget_probe_root", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    measurements = aggregate_lengths(
        load_rows(args.input_root),
        load_token_rows(args.input_root),
    )
    if len(measurements) < 4:
        raise ValueError("at least four paired lengths are required")
    full_predictors = [
        [1.0, row["history_tokens"] / 1000.0]
        for row in measurements
    ]
    sparse_predictors = [
        [
            1.0,
            row["history_tokens"] / 1000.0,
            row["actual_budget"] / 1000.0,
        ]
        for row in measurements
    ]
    full_fit = fit_model(
        full_predictors,
        [row["full_steady_ms_per_token"] for row in measurements],
        ("intercept_ms", "history_ktokens_ms"),
    )
    fixed_fit = fit_model(
        [
            [1.0, row["history_tokens"] / 1000.0]
            for row in measurements
        ],
        [
            row["lazy_index_fixed_overhead_ms"]
            for row in measurements
        ],
        ("intercept_ms", "history_ktokens_ms"),
    )
    budget_probe_fit = None
    if args.budget_probe_root is not None:
        budget_probe_fit = fit_fixed_effect_budget_slope(
            load_budget_probe_rows(args.budget_probe_root)
        )
        attention_coefficient = (
            budget_probe_fit["nonnegative_ms_per_attention_token"] * 1000.0
        )
        base_fit = fit_model(
            [
                [1.0, row["history_tokens"] / 1000.0]
                for row in measurements
            ],
            [
                row["countcap_steady_ms_per_token"]
                - attention_coefficient * row["actual_budget"] / 1000.0
                for row in measurements
            ],
            ("intercept_ms", "history_ktokens_ms"),
        )
        sparse_coefficients = {
            **base_fit["coefficients"],
            "attention_ktokens_ms": attention_coefficient,
        }
        sparse_prediction = [
            sparse_coefficients["intercept_ms"]
            + sparse_coefficients["history_ktokens_ms"]
            * row["history_tokens"]
            / 1000.0
            + sparse_coefficients["attention_ktokens_ms"]
            * row["actual_budget"]
            / 1000.0
            for row in measurements
        ]
        sparse_target = torch.tensor(
            [row["countcap_steady_ms_per_token"] for row in measurements],
            dtype=torch.float64,
        )
        sparse_prediction_tensor = torch.tensor(
            sparse_prediction,
            dtype=torch.float64,
        )
        sparse_residual = sparse_target - sparse_prediction_tensor
        sparse_total = (
            sparse_target - sparse_target.mean()
        ).square().sum().clamp_min(1.0e-30)
        sparse_fit = {
            "coefficients": sparse_coefficients,
            "prediction": sparse_prediction,
            "rmse_ms": float(
                sparse_residual.square().mean().sqrt().item()
            ),
            "r_squared": float(
                (1.0 - sparse_residual.square().sum() / sparse_total).item()
            ),
            "coefficient_identification": (
                "B coefficient from fixed-N paired budget probes; "
                "intercept and N coefficient from the length sweep"
            ),
        }
    else:
        sparse_fit = fit_model(
            sparse_predictors,
            [row["countcap_steady_ms_per_token"] for row in measurements],
            (
                "intercept_ms",
                "history_ktokens_ms",
                "attention_ktokens_ms",
            ),
        )
        sparse_fit["coefficient_identification"] = (
            "N and B co-vary in the length sweep; separate coefficients "
            "are not causally identified"
        )
    inflation = float(
        torch.tensor(
            [row["budget_inflation"] for row in measurements],
            dtype=torch.float64,
        ).median().item()
    )
    crossover = find_crossover(
        full_fit["coefficients"],
        sparse_fit["coefficients"],
        inflation,
    )
    curve = predict_curve(
        full_fit["coefficients"],
        sparse_fit["coefficients"],
        fixed_fit["coefficients"],
        inflation,
        (2048, 4096, 8192, 12000, 16000, 24000, 32000, 64000, 128000),
    )
    for row, full_prediction, sparse_prediction in zip(
        measurements,
        full_fit["prediction"],
        sparse_fit["prediction"],
    ):
        row["fitted_full_ms_per_token"] = full_prediction
        row["fitted_countcap_ms_per_token"] = sparse_prediction

    report = {
        "config": {
            "input_root": str(args.input_root),
            "fit_lengths": [
                int(row["history_tokens"]) for row in measurements
            ],
            "same_hardware_fit_only": True,
            "budget": "min(N, 1280, max(256, ceil(0.06*N)))",
        },
        "full_model": {
            "equation": (
                "t_full_steady_ms = a_f + b_f * (N / 1000)"
            ),
            **full_fit,
        },
        "countcap_model": {
            "equation": (
                "t_cc_steady_ms = a_c + b_scan * (N / 1000) "
                "+ b_sparse * (B_actual / 1000)"
            ),
            **sparse_fit,
        },
        "lazy_index_model": {
            "equation": (
                "I_lazy_ms = a_i + b_i * (N / 1000)"
            ),
            **fixed_fit,
            "definition": (
                "median across paired cases of "
                "(CountCap first-step excess - Full first-step excess)"
            ),
        },
        "fixed_n_budget_probe": budget_probe_fit,
        "median_actual_to_target_budget_ratio": inflation,
        "predicted_decode_crossover_tokens": crossover,
        "measurements": measurements,
        "predicted_curve": curve,
        "operation_model": {
            "full_attention_per_layer": (
                "Theta(Hq * N * d) for QK plus Theta(Hq * N * d) for AV"
            ),
            "countcap_scan_per_layer": "Theta(Hq * N * r_lowbit)",
            "countcap_exact_sparse_attention_per_layer": (
                "Theta(Hq * B * d)"
            ),
            "countcap_index_build_per_layer": (
                "Theta(Hkv * m_prefix * d^2 + Hkv * d^3 "
                "+ Hkv * N * d * r)"
            ),
            "physical_kv_storage": "Theta(2 * Hkv * N * d * fp16_bytes)",
            "lowbit_key_index_storage": (
                "Hkv*N*(r/2 + 2-byte base scale "
                "+ ceil(ceil(r/16)/2) exponent bytes)"
            ),
            "pca48_index_to_full_fp16_kv_ratio_excluding_basis": (
                (48.0 / 2.0 + 2.0 + math.ceil(math.ceil(48 / 16) / 2))
                / (2.0 * 128.0 * 2.0)
            ),
        },
        "scope": [
            "The fitted coefficients are empirical kernel costs, not FLOP constants.",
            "The 64K/128K multi-GPU measurements must not be mixed into this fit.",
            "B_actual is variable under the sampled threshold; the extrapolated curve uses the observed median inflation only.",
            "Steady-state fits exclude the first timed decode step.",
            "The lazy index is built in the first CountCap decode step in this PPL harness; dense_prompt_seconds does not contain that index.",
            "Production prefill-index placement moves this one-time cost into prefill and may overlap it, but does not change the steady decode equation.",
            "Rows above 32K in predicted_curve are labeled extrapolations and are not measurements.",
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "cost_measurements.csv", measurements)
    write_csv(args.output_dir / "predicted_curve.csv", curve)
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "fit_lengths": report["config"]["fit_lengths"],
                "predicted_decode_crossover_tokens": crossover,
                "full_r_squared": full_fit["r_squared"],
                "countcap_r_squared": sparse_fit["r_squared"],
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
