from __future__ import annotations

"""Reproducible seed-cluster audit for the value-mediated no-op probe.

The independent unit is always the prompt seed.  Layer/head/token rows are
first averaged within a seed and are never treated as independent samples.
"""

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


EVIDENCE_CLASSES = ("gold_evidence", "conflict_evidence")
NON_EVIDENCE_CLASSES = ("lexical_format_distractor", "filler")
CLASS_ORDER = (*EVIDENCE_CLASSES, *NON_EVIDENCE_CLASSES)
PREDICTED = "predicted_first_order_delta_gold_conflict_margin"
ACTUAL = "actual_delta_gold_conflict_margin"
SAMPLE_FIELDS = (
    "dm_dscore",
    "direct_ov_centered_margin_derivative",
    "attention_probability",
    "suppression_gap",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def finite(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def mean(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    return statistics.fmean(items) if items else float("nan")


def percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return float("nan")
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * float(probability)
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def average_ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(enumerate(map(float, values)), key=lambda item: item[1])
    ranks = [0.0] * len(ordered)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        rank = ((index + 1) + end) / 2.0
        for original, _ in ordered[index:end]:
            ranks[original] = rank
        index = end
    return ranks


def correlation(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    x_mean, y_mean = mean(x), mean(y)
    dx = [float(value) - x_mean for value in x]
    dy = [float(value) - y_mean for value in y]
    denominator = math.sqrt(
        math.fsum(value * value for value in dx)
        * math.fsum(value * value for value in dy)
    )
    if denominator == 0.0:
        return float("nan")
    return math.fsum(a * b for a, b in zip(dx, dy)) / denominator


def regression(x: Sequence[float], y: Sequence[float]) -> tuple[float, float, float]:
    x_mean, y_mean = mean(x), mean(y)
    denominator = math.fsum((float(value) - x_mean) ** 2 for value in x)
    if denominator == 0.0:
        return float("nan"), float("nan"), float("nan")
    slope = math.fsum(
        (float(a) - x_mean) * (float(b) - y_mean) for a, b in zip(x, y)
    ) / denominator
    intercept = y_mean - slope * x_mean
    predicted = [intercept + slope * float(value) for value in x]
    residual = math.fsum((float(a) - float(b)) ** 2 for a, b in zip(y, predicted))
    total = math.fsum((float(value) - y_mean) ** 2 for value in y)
    r_squared = float("nan") if total == 0.0 else 1.0 - residual / total
    return intercept, slope, r_squared


def intervention_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    predicted = [float(row[PREDICTED]) for row in rows]
    actual = [float(row[ACTUAL]) for row in rows]
    intercept, slope, r_squared = regression(predicted, actual)
    return {
        "pearson": correlation(predicted, actual),
        "spearman": correlation(average_ranks(predicted), average_ranks(actual)),
        "sign_accuracy": mean(float(row["first_order_sign_match"]) for row in rows),
        "mean_absolute_closure_error": mean(abs(a - b) for a, b in zip(predicted, actual)),
        "mean_symmetric_closure_error": mean(
            float(row["first_order_symmetric_closure_error"]) for row in rows
        ),
        "mean_actual_delta_margin": mean(actual),
        "mean_delta_gold_nll": mean(float(row["delta_gold_nll"]) for row in rows),
        "regression_intercept": intercept,
        "regression_slope": slope,
        "regression_r_squared": r_squared,
    }


def bootstrap_cluster_rows(
    rows: Sequence[dict[str, Any]],
    metric: Callable[[Sequence[dict[str, Any]]], dict[str, float]],
    *,
    resamples: int,
    rng_seed: int,
) -> tuple[dict[str, float], dict[str, list[float | None]]]:
    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_seed[int(row["seed"])].append(row)
    seeds = sorted(by_seed)
    if len(seeds) < 2:
        raise ValueError("seed-cluster bootstrap requires at least two seeds")
    point = metric(rows)
    draws: dict[str, list[float]] = {name: [] for name in point}
    rng = random.Random(int(rng_seed))
    for _ in range(int(resamples)):
        sampled: list[dict[str, Any]] = []
        for _ in seeds:
            sampled.extend(by_seed[rng.choice(seeds)])
        values = metric(sampled)
        for name, value in values.items():
            if math.isfinite(float(value)):
                draws[name].append(float(value))
    intervals = {
        name: [finite(percentile(values, 0.025)), finite(percentile(values, 0.975))]
        for name, values in draws.items()
    }
    return point, intervals


def stable_seed(base: int, name: str) -> int:
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return int(base) + int.from_bytes(digest[:4], "little")


def summarize_intervention_subset(
    rows: Sequence[dict[str, Any]],
    *,
    name: str,
    resamples: int,
    rng_seed: int,
) -> dict[str, Any]:
    point, intervals = bootstrap_cluster_rows(
        rows,
        intervention_metrics,
        resamples=resamples,
        rng_seed=stable_seed(rng_seed, name),
    )
    return {
        "n_seeds": len({int(row["seed"]) for row in rows}),
        "n_events": len(rows),
        "point": {key: finite(value) for key, value in point.items()},
        "seed_cluster_percentile_95_ci": intervals,
    }


def paired_target_random(
    rows: Sequence[dict[str, Any]],
    classes: Sequence[str],
    *,
    name: str,
    resamples: int,
    rng_seed: int,
) -> dict[str, Any]:
    lookup = {
        (int(row["seed"]), str(row["intervention_class"]), str(row["plan_kind"])): row
        for row in rows
        if row["intervention_scope"] == "singleton"
    }
    seeds = sorted({seed for seed, _, _ in lookup})
    metric_fields: dict[str, Callable[[dict[str, Any]], float]] = {
        "sign_accuracy": lambda row: float(row["first_order_sign_match"]),
        "symmetric_closure_error": lambda row: float(
            row["first_order_symmetric_closure_error"]
        ),
        "absolute_closure_error": lambda row: float(
            row["first_order_absolute_closure_error"]
        ),
        "absolute_actual_delta_margin": lambda row: abs(float(row[ACTUAL])),
        "delta_gold_nll": lambda row: float(row["delta_gold_nll"]),
    }
    per_seed: dict[str, list[float]] = {field: [] for field in metric_fields}
    for seed in seeds:
        for category in classes:
            for kind in ("target", "random"):
                if (seed, category, kind) not in lookup:
                    raise RuntimeError(f"missing paired row: {seed}/{category}/{kind}")
        for field, extractor in metric_fields.items():
            target = mean(extractor(lookup[(seed, category, "target")]) for category in classes)
            random_control = mean(
                extractor(lookup[(seed, category, "random")]) for category in classes
            )
            per_seed[field].append(target - random_control)

    rng = random.Random(stable_seed(rng_seed, "paired-" + name))
    intervals: dict[str, list[float | None]] = {}
    for field, differences in per_seed.items():
        draws = [
            mean(rng.choice(differences) for _ in differences)
            for _ in range(int(resamples))
        ]
        intervals[field] = [
            finite(percentile(draws, 0.025)),
            finite(percentile(draws, 0.975)),
        ]
    return {
        "classes": list(classes),
        "n_seed_pairs": len(seeds),
        "contrast_definition": "target_minus_matched_random; averaged over classes within seed",
        "point": {field: finite(mean(values)) for field, values in per_seed.items()},
        "seed_cluster_percentile_95_ci": intervals,
        "per_seed_differences": {
            field: [finite(value) for value in values] for field, values in per_seed.items()
        },
    }


def bootstrap_seed_values(
    values: Sequence[float], *, resamples: int, rng_seed: int
) -> list[float | None]:
    rng = random.Random(int(rng_seed))
    draws = [mean(rng.choice(values) for _ in values) for _ in range(int(resamples))]
    return [finite(percentile(draws, 0.025)), finite(percentile(draws, 0.975))]


def sample_seed_macro(
    rows: Sequence[dict[str, Any]], *, resamples: int, rng_seed: int
) -> dict[str, Any]:
    cells: dict[tuple[int, str, str], list[float]] = defaultdict(list)
    for row in rows:
        seed, category = int(row["seed"]), str(row["class"])
        for field in SAMPLE_FIELDS:
            cells[(seed, category, field)].append(float(row[field]))
    seeds = sorted({key[0] for key in cells})
    per_seed = {
        (seed, category, field): mean(cells[(seed, category, field)])
        for seed in seeds
        for category in CLASS_ORDER
        for field in SAMPLE_FIELDS
    }
    classes: dict[str, Any] = {}
    for category in CLASS_ORDER:
        classes[category] = {}
        for field in SAMPLE_FIELDS:
            values = [per_seed[(seed, category, field)] for seed in seeds]
            classes[category][field] = {
                "seed_macro_mean": finite(mean(values)),
                "seed_cluster_percentile_95_ci": bootstrap_seed_values(
                    values,
                    resamples=resamples,
                    rng_seed=stable_seed(rng_seed, f"sample-{category}-{field}"),
                ),
                "per_seed_means": [finite(value) for value in values],
            }
    contrast: dict[str, Any] = {}
    for field in SAMPLE_FIELDS:
        differences = [
            per_seed[(seed, "gold_evidence", field)]
            - per_seed[(seed, "conflict_evidence", field)]
            for seed in seeds
        ]
        contrast[field] = {
            "gold_minus_conflict_seed_macro_mean": finite(mean(differences)),
            "seed_cluster_percentile_95_ci": bootstrap_seed_values(
                differences,
                resamples=resamples,
                rng_seed=stable_seed(rng_seed, f"sample-contrast-{field}"),
            ),
            "per_seed_differences": [finite(value) for value in differences],
        }
    return {
        "aggregation": "mean layer/head/token rows within each seed and class, then bootstrap seeds",
        "n_seeds": len(seeds),
        "classes": classes,
        "gold_minus_conflict": contrast,
    }


def raw_results(input_dir: Path) -> list[dict[str, Any]]:
    results = []
    for path in sorted(input_dir.glob("shard_gpu*/raw/*_result.json")):
        results.append(json.loads(path.read_text(encoding="utf-8")))
    return results


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_integer_csv(value: Any) -> list[int]:
    if isinstance(value, (list, tuple)):
        return sorted({int(item) for item in value})
    return sorted(
        {int(item.strip()) for item in str(value or "").split(",") if item.strip()}
    )


def inspect_merge_provenance(
    input_dir: Path,
    merge_config: dict[str, Any],
    *,
    actual_lengths: Sequence[int],
    actual_num_seeds: int,
    actual_class_sample_counts: Sequence[int],
    actual_top_n: Sequence[int],
) -> dict[str, Any]:
    schema_version = merge_config.get("merge_schema_version")
    shard_derived = bool(
        isinstance(schema_version, int)
        and int(schema_version) >= 2
        and isinstance(merge_config.get("shared_config"), dict)
        and isinstance(merge_config.get("shards"), list)
        and bool(merge_config.get("shards"))
    )
    legacy_files = sorted(
        path.name
        for path in (input_dir / "merged").glob("merge_config_legacy*.json")
    )
    if shard_derived:
        shared = merge_config["shared_config"]
        shard_configs = [
            item.get("config", {})
            for item in merge_config["shards"]
            if isinstance(item, dict)
        ]
        reported_lengths = sorted(
            {
                length
                for config in shard_configs
                for length in parse_integer_csv(
                    config.get("resolved_lengths", config.get("lengths", ""))
                )
            }
        )
        reported_num_seeds = sum(
            int(config.get("num_seeds", 0)) for config in shard_configs
        )
        reported_class_samples = shared.get("class_sample_count")
        reported_top_n = shared.get("singleton_top_n")
        status = "shard_derived_schema"
    else:
        reported_lengths = parse_integer_csv(merge_config.get("lengths", ""))
        reported_num_seeds = merge_config.get("num_seeds")
        reported_class_samples = merge_config.get("class_sample_count")
        reported_top_n = merge_config.get("singleton_top_n")
        status = "legacy_merge_invocation_defaults"

    comparisons = {
        "context_lengths": {
            "actual": list(map(int, actual_lengths)),
            "reported": reported_lengths,
        },
        "num_seeds": {
            "actual": int(actual_num_seeds),
            "reported": reported_num_seeds,
        },
        "class_sample_count": {
            "actual_inferred": list(map(int, actual_class_sample_counts)),
            "reported": reported_class_samples,
        },
        "singleton_top_n": {
            "actual": list(map(int, actual_top_n)),
            "reported": reported_top_n,
        },
    }
    discrepancies: dict[str, Any] = {}
    if comparisons["context_lengths"]["actual"] != comparisons["context_lengths"][
        "reported"
    ]:
        discrepancies["context_lengths"] = comparisons["context_lengths"]
    if comparisons["num_seeds"]["actual"] != comparisons["num_seeds"]["reported"]:
        discrepancies["num_seeds"] = comparisons["num_seeds"]
    if comparisons["class_sample_count"]["actual_inferred"] != [
        comparisons["class_sample_count"]["reported"]
    ]:
        discrepancies["class_sample_count"] = comparisons["class_sample_count"]
    if comparisons["singleton_top_n"]["actual"] != [
        comparisons["singleton_top_n"]["reported"]
    ]:
        discrepancies["singleton_top_n"] = comparisons["singleton_top_n"]

    warning_required = (not shard_derived) or bool(discrepancies)
    return {
        "status": status,
        "merge_schema_version": schema_version,
        "current_config_is_shard_derived": shard_derived,
        "legacy_incorrect_files_present": legacy_files,
        "comparisons": comparisons,
        "discrepancies": discrepancies,
        "warning_required": warning_required,
        "warning": (
            "Current merge_config is a legacy merge invocation/default snapshot, "
            "not shard-derived run provenance."
            if not shard_derived
            else (
                "Shard-derived merge_config disagrees with actual CSV/raw content."
                if discrepancies
                else None
            )
        ),
    }


def corrected_provenance(
    input_dir: Path,
    case_rows: Sequence[dict[str, Any]],
    sample_rows: Sequence[dict[str, Any]],
    results: Sequence[dict[str, Any]],
    launcher: Path | None,
) -> dict[str, Any]:
    launcher_text = launcher.read_text(encoding="utf-8") if launcher else ""
    launcher_requests_unquantized_bf16 = bool(
        launcher
        and "--dtype bfloat16" in launcher_text
        and "--load-in-4bit" not in launcher_text
    )
    seeds = sorted({int(row["seed"]) for row in case_rows})
    lengths = sorted({int(row["target_context_tokens"]) for row in case_rows})
    sample_counts = Counter(
        (
            int(row["seed"]),
            int(row["layer"]),
            int(row["head"]),
            str(row["class"]),
        )
        for row in sample_rows
    )
    inferred_samples = sorted(set(sample_counts.values()))
    merge_config_path = input_dir / "merged" / "merge_config.json"
    merge_config = (
        json.loads(merge_config_path.read_text(encoding="utf-8"))
        if merge_config_path.exists()
        else {}
    )
    actual_top_n = sorted(
        {int(result["singleton_top_n_per_class"]) for result in results}
    )
    merge_provenance = inspect_merge_provenance(
        input_dir,
        merge_config,
        actual_lengths=lengths,
        actual_num_seeds=len(seeds),
        actual_class_sample_counts=inferred_samples,
        actual_top_n=actual_top_n,
    )
    score_lifts = sorted(
        {
            float(row["uniform_score_lift"])
            for row in case_rows
            if row.get("intervention_scope") == "singleton"
            and row.get("uniform_score_lift") not in (None, "")
        }
    )
    provenance: dict[str, Any] = {
        "schema_version": 1,
        "derived_from_actual_raw_and_csv_content": True,
        "original_merge_config_modified": False,
        "merge_provenance": merge_provenance,
        "experiment": sorted({str(result.get("experiment")) for result in results}),
        "raw_schema_versions": sorted(
            {int(result.get("schema_version", -1)) for result in results}
        ),
        "context_lengths": lengths,
        "seeds": seeds,
        "num_seeds": len(seeds),
        "case_row_count": len(case_rows),
        "value_sample_row_count": len(sample_rows),
        "singleton_top_n_per_class": actual_top_n,
        "class_sample_count_per_layer_head_class_inferred": inferred_samples,
        "layers": sorted({int(row["layer"]) for row in sample_rows}),
        "heads": sorted({int(row["head"]) for row in sample_rows}),
        "classes": sorted({str(row["class"]) for row in sample_rows}),
        "score_lifts": score_lifts,
        "candidate_ranking_metrics": sorted(
            {
                str(result.get("singleton_candidate_ranking_metric"))
                for result in results
            }
        ),
        "final_hidden_source_dtypes": sorted(
            {
                str(row["final_hidden_source_dtype"])
                for row in case_rows
                if row.get("final_hidden_source_dtype")
            }
        ),
        "all_case_replay_audits_passed": all(
            bool(result.get("case_replay_audit", {}).get("passed"))
            for result in results
        ),
        "all_prefix_caches_immutable": all(
            bool(result.get("prefix_cache_immutable")) for result in results
        ),
        "weight_mode": {
            "requested": (
                "unquantized_bfloat16"
                if launcher_requests_unquantized_bf16
                else "not_verifiable"
            ),
            "raw_artifact_directly_records_weight_quantization": False,
            "caveat": (
                "BF16 hidden dtype is recorded in raw results; weight quantization "
                "is established by the launcher, not by the synced raw schema."
            ),
        },
        "merge_config_discrepancies": merge_provenance["discrepancies"],
    }
    if launcher is not None:
        provenance["launcher"] = {
            "path": str(launcher),
            "sha256": sha256(launcher),
            "contains_load_in_4bit_flag": "--load-in-4bit" in launcher_text,
            "contains_dtype_bfloat16": "--dtype bfloat16" in launcher_text,
            "contains_only_physical_gpu6_gpu7": (
                "CUDA_VISIBLE_DEVICES=6" in launcher_text
                and "CUDA_VISIBLE_DEVICES=7" in launcher_text
                and all(
                    f"CUDA_VISIBLE_DEVICES={index}" not in launcher_text
                    for index in range(6)
                )
            ),
        }
    return provenance


def baseline_audit(
    case_rows: Sequence[dict[str, Any]], results: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    noop_native_margin = [
        float(result["custom_noop_delta_from_native"]["delta_gold_conflict_margin"])
        for result in results
    ]
    noop_instrumented_margin = [
        float(
            result["custom_noop_delta_from_instrumented"][
                "delta_gold_conflict_margin"
            ]
        )
        for result in results
    ]
    noop_instrumented_nll = [
        float(result["custom_noop_delta_from_instrumented"]["delta_gold_nll"])
        for result in results
    ]
    native_by_seed = {
        int(row["seed"]): row
        for row in case_rows
        if row["intervention_class"] == "native_baseline"
    }
    noop_by_seed = {
        int(row["seed"]): row
        for row in case_rows
        if row["intervention_class"] == "custom_noop_baseline"
    }
    return {
        "all_noop_minus_instrumented_margin_exact_zero": all(
            value == 0.0 for value in noop_instrumented_margin
        ),
        "all_noop_minus_instrumented_nll_exact_zero": all(
            value == 0.0 for value in noop_instrumented_nll
        ),
        "noop_minus_native_margin_mean": finite(mean(noop_native_margin)),
        "noop_minus_native_margin_mean_absolute": finite(
            mean(abs(value) for value in noop_native_margin)
        ),
        "noop_minus_native_margin_max_absolute": finite(
            max(abs(value) for value in noop_native_margin)
        ),
        "native_noop_top1_decision_agreement": finite(
            mean(
                native_by_seed[seed]["prediction_token_id"]
                == noop_by_seed[seed]["prediction_token_id"]
                for seed in sorted(native_by_seed)
            )
        ),
    }


def markdown_report(audit: dict[str, Any], provenance: dict[str, Any]) -> str:
    def fmt(value: float | None) -> str:
        return "NA" if value is None else f"{float(value):.6g}"

    length_summary = "、".join(
        f"{int(length):,} tokens" for length in provenance["context_lengths"]
    )
    merge_provenance = provenance["merge_provenance"]
    if merge_provenance["warning_required"]:
        if merge_provenance["current_config_is_shard_derived"]:
            merge_config_note = (
                "- 当前 shard-derived `merged/merge_config.json` 与实际 CSV/raw "
                "内容不一致；具体差异见同目录 `corrected_provenance.json`。"
            )
        else:
            merge_config_note = (
                "- 当前 `merged/merge_config.json` 是旧 merge 调用/默认参数快照，"
                "且与实际 CSV/raw 内容不一致；本报告使用实际内容，修正后的运行信息见"
                "同目录 `corrected_provenance.json`。"
            )
    elif merge_provenance["legacy_incorrect_files_present"]:
        legacy_names = "、".join(
            f"`{name}`"
            for name in merge_provenance["legacy_incorrect_files_present"]
        )
        merge_config_note = (
            "- 当前 `merged/merge_config.json` 是 shard-derived schema，且与实际 "
            f"CSV/raw 内容一致；{legacy_names} 仅为旧错误配置的归档，不参与本报告统计。"
        )
    else:
        merge_config_note = (
            "- 当前 `merged/merge_config.json` 是 shard-derived schema，且与实际 "
            "CSV/raw 内容一致。"
        )

    lines = [
        "# Value-mediated no-op：独立 seed-cluster 审计",
        "",
        "独立统计单位是 **seed**。Layer、head、token 行仅在 seed 内聚合，绝不作为独立样本 bootstrap。",
        "本报告只读取 `merged/case_rows.csv`、`merged/value_samples.csv` 和 raw result；",
        "没有拼接重复的 `first_order_prediction_summary` / `singleton_prediction_summary`，也没有使用 `value_sample_summary` 中重复的 `all` alias。",
        "",
        f"- Bootstrap：percentile 95% CI，{audit['methodology']['bootstrap_resamples']} 次，固定 RNG seed {audit['methodology']['rng_seed']}。",
        f"- Seeds：{provenance['seeds']}；实际长度：{provenance['context_lengths']}。",
        f"- Raw/case 审计全部通过：{provenance['all_case_replay_audits_passed']}。",
        "",
        "## 一阶预测闭合",
        "",
        "| 范围 | events | Pearson | 95% CI | Spearman | 95% CI | sign accuracy | 95% CI |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in ("all_target", "evidence_only_target", "gold_target", "conflict_target", "non_evidence_target", "random_evidence"):
        row = audit["intervention_summaries"][key]
        point, ci = row["point"], row["seed_cluster_percentile_95_ci"]
        lines.append(
            f"| {key} | {row['n_events']} | {fmt(point['pearson'])} | "
            f"[{fmt(ci['pearson'][0])}, {fmt(ci['pearson'][1])}] | "
            f"{fmt(point['spearman'])} | [{fmt(ci['spearman'][0])}, {fmt(ci['spearman'][1])}] | "
            f"{fmt(point['sign_accuracy'])} | [{fmt(ci['sign_accuracy'][0])}, {fmt(ci['sign_accuracy'][1])}] |"
        )
    evidence = audit["intervention_summaries"]["evidence_only_target"]
    lines.extend(
        [
            "",
            "Evidence-only 回归：",
            "",
            f"- slope {fmt(evidence['point']['regression_slope'])}, CI [{fmt(evidence['seed_cluster_percentile_95_ci']['regression_slope'][0])}, {fmt(evidence['seed_cluster_percentile_95_ci']['regression_slope'][1])}]；",
            f"- intercept {fmt(evidence['point']['regression_intercept'])}, CI [{fmt(evidence['seed_cluster_percentile_95_ci']['regression_intercept'][0])}, {fmt(evidence['seed_cluster_percentile_95_ci']['regression_intercept'][1])}]；",
            f"- R² {fmt(evidence['point']['regression_r_squared'])}, CI [{fmt(evidence['seed_cluster_percentile_95_ci']['regression_r_squared'][0])}, {fmt(evidence['seed_cluster_percentile_95_ci']['regression_r_squared'][1])}]。",
            "",
            "## Seed-macro Gold − Conflict",
            "",
            "| 指标 | 差值 | seed-cluster 95% CI |",
            "|---|---:|---:|",
        ]
    )
    for field in SAMPLE_FIELDS:
        item = audit["sample_seed_macro"]["gold_minus_conflict"][field]
        ci = item["seed_cluster_percentile_95_ci"]
        lines.append(
            f"| {field} | {fmt(item['gold_minus_conflict_seed_macro_mean'])} | "
            f"[{fmt(ci[0])}, {fmt(ci[1])}] |"
        )
    paired = audit["paired_target_vs_random"]["evidence"]
    lines.extend(
        [
            "",
            "## Evidence target − matched random",
            "",
            "| 指标 | 配对差 | seed-cluster 95% CI |",
            "|---|---:|---:|",
        ]
    )
    for field, value in paired["point"].items():
        ci = paired["seed_cluster_percentile_95_ci"][field]
        lines.append(f"| {field} | {fmt(value)} | [{fmt(ci[0])}, {fmt(ci[1])}] |")
    base = audit["baseline_replay_audit"]
    lines.extend(
        [
            "",
            "## 审计限制",
            "",
            f"- Instrumented 与 no-op margin/NLL 严格一致：{base['all_noop_minus_instrumented_margin_exact_zero']} / {base['all_noop_minus_instrumented_nll_exact_zero']}。",
            f"- Custom no-op 相对 native 的最大绝对 pair-margin drift：{fmt(base['noop_minus_native_margin_max_absolute'])}；top-1 agreement：{fmt(base['native_noop_top1_decision_agreement'])}。",
            "- Candidate ranking 使用 oracle answer gradient；这里只能作为机制诊断，不能称可部署 selector。",
            "- Evidence target 与 random 没有匹配 decisive-token 身份；相关差异不能解释为公平检索基线优势。",
            f"- 本批实验只有 Qwen3-8B、上下文长度 {length_summary}、score lift {provenance['score_lifts']} 和 {provenance['num_seeds']} 个合成 seeds，不能外推到其他长度、模型或真实任务。",
            merge_config_note,
            "",
        ]
    )
    return "\n".join(lines)


def run_analysis(
    input_dir: Path,
    *,
    resamples: int,
    rng_seed: int,
    launcher: Path | None,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    merged = input_dir / "merged"
    case_rows: list[dict[str, Any]] = read_csv(merged / "case_rows.csv")
    sample_rows: list[dict[str, Any]] = read_csv(merged / "value_samples.csv")
    results = raw_results(input_dir)
    if not results:
        raise FileNotFoundError("no shard raw result JSON files found")
    singleton = [row for row in case_rows if row["intervention_scope"] == "singleton"]
    filters = {
        "all_target": lambda row: row["plan_kind"] == "target",
        "evidence_only_target": lambda row: row["plan_kind"] == "target"
        and row["intervention_class"] in EVIDENCE_CLASSES,
        "gold_target": lambda row: row["plan_kind"] == "target"
        and row["intervention_class"] == "gold_evidence",
        "conflict_target": lambda row: row["plan_kind"] == "target"
        and row["intervention_class"] == "conflict_evidence",
        "non_evidence_target": lambda row: row["plan_kind"] == "target"
        and row["intervention_class"] in NON_EVIDENCE_CLASSES,
        "random_evidence": lambda row: row["plan_kind"] == "random"
        and row["intervention_class"] in EVIDENCE_CLASSES,
    }
    summaries = {
        name: summarize_intervention_subset(
            [row for row in singleton if predicate(row)],
            name=name,
            resamples=resamples,
            rng_seed=rng_seed,
        )
        for name, predicate in filters.items()
    }
    audit: dict[str, Any] = {
        "schema_version": 1,
        "methodology": {
            "independent_unit": "seed",
            "bootstrap": "seed-cluster percentile bootstrap",
            "bootstrap_resamples": int(resamples),
            "rng_seed": int(rng_seed),
            "sample_row_aggregation": "within-seed mean before bootstrap",
            "excluded_duplicate_aliases": [
                "merged/first_order_prediction_summary.csv",
                "merged/singleton_prediction_summary.csv",
                "merged/value_sample_summary.csv target_context_tokens=all",
            ],
        },
        "integrity": {
            "case_row_count": len(case_rows),
            "case_unique_key_count": len(
                {
                    (
                        row["seed"],
                        row["intervention_class"],
                        row["plan_kind"],
                        row["intervention_scope"],
                        row.get("pair_id", ""),
                    )
                    for row in case_rows
                }
            ),
            "value_sample_row_count": len(sample_rows),
            "value_sample_unique_key_count": len(
                {
                    (
                        row["seed"],
                        row["layer"],
                        row["head"],
                        row["class"],
                        row["sample_index"],
                        row["token_position"],
                    )
                    for row in sample_rows
                }
            ),
        },
        "intervention_summaries": summaries,
        "paired_target_vs_random": {
            "all": paired_target_random(
                singleton,
                CLASS_ORDER,
                name="all",
                resamples=resamples,
                rng_seed=rng_seed,
            ),
            "evidence": paired_target_random(
                singleton,
                EVIDENCE_CLASSES,
                name="evidence",
                resamples=resamples,
                rng_seed=rng_seed,
            ),
            "gold": paired_target_random(
                singleton,
                ("gold_evidence",),
                name="gold",
                resamples=resamples,
                rng_seed=rng_seed,
            ),
            "conflict": paired_target_random(
                singleton,
                ("conflict_evidence",),
                name="conflict",
                resamples=resamples,
                rng_seed=rng_seed,
            ),
        },
        "sample_seed_macro": sample_seed_macro(
            sample_rows, resamples=resamples, rng_seed=rng_seed
        ),
        "baseline_replay_audit": baseline_audit(case_rows, results),
    }
    provenance = corrected_provenance(
        input_dir, case_rows, sample_rows, results, launcher
    )
    # The decisive flag is stored in frozen candidates, not case_rows in schema 3.
    decisive: dict[str, list[int]] = {"target": [], "random": []}
    for result in results:
        for candidate in result["frozen_singleton_candidates"]:
            if candidate["class"] not in EVIDENCE_CLASSES:
                continue
            for kind in decisive:
                decisive[kind].append(int(candidate[kind]["is_decisive_token"]))
    audit["evidence_decisive_token_balance"] = {
        kind: {"n": len(values), "decisive_fraction": finite(mean(values))}
        for kind, values in decisive.items()
    }
    report = markdown_report(audit, provenance)
    return audit, provenance, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--bootstrap-resamples", type=int, default=50000)
    parser.add_argument("--rng-seed", type=int, default=20260801)
    parser.add_argument("--launcher", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if int(args.bootstrap_resamples) < 100:
        raise ValueError("bootstrap resamples must be at least 100")
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    launcher = Path(args.launcher) if args.launcher else None
    audit, provenance, report = run_analysis(
        input_dir,
        resamples=int(args.bootstrap_resamples),
        rng_seed=int(args.rng_seed),
        launcher=launcher,
    )
    write_json(output_dir / "independent_seed_cluster_audit.json", audit)
    write_json(output_dir / "corrected_provenance.json", provenance)
    (output_dir / "independent_seed_cluster_audit.md").write_text(
        report, encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "independent_unit": "seed",
                "bootstrap_resamples": int(args.bootstrap_resamples),
                "rng_seed": int(args.rng_seed),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
