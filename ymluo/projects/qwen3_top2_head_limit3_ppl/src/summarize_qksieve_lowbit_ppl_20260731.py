from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean


LABELS = {
    "full_attention": "full_fp16",
    "qksieve_keymse_requestlocal_fixedalloc_b12_fulltopk_k1280": (
        "int_fixed441_b192"
    ),
    "qksieve_keymse_requestlocal_fixedalloc_b10_fulltopk_k1280": (
        "int_fixed440_b160"
    ),
    "qksieve_keymse_requestlocal_fixedalloc_b8_fulltopk_k1280": (
        "int_fixed420_b128"
    ),
    "qksieve_keymse_requestlocal_fixedalloc_i112_41_fulltopk_k1280": (
        "int_fixed410_b112"
    ),
    "qksieve_keymse_requestlocal_fixedalloc_i112_211_fulltopk_k1280": (
        "int_fixed211_b112"
    ),
    "qksieve_keymse_requestlocal_fixedalloc_i96_22_fulltopk_k1280": (
        "int_fixed220_b96"
    ),
    (
        "qksieve_keymse_requestlocal_fixedalloc_"
        "post2xprererank_b8_fulltopk_k1280"
    ): "int_fixed420_b128_post2x_prerope_rerank",
    (
        "qksieve_keymse_requestlocal_fixedalloc_"
        "post2xprererank_i112_41_fulltopk_k1280"
    ): "int_fixed410_b112_post2x_prerope_rerank",
    (
        "qksieve_keymse_requestlocal_fixedalloc_"
        "post2xprererank_i112_41_l00to08_fulltopk_k1280"
    ): "int_fixed410_b112_post2x_prerope_rerank_layers00_08",
    (
        "qksieve_keymse_requestlocal_fixedalloc_"
        "post2xprererank_i112_41_l00to08_fulltopk_k2560"
    ): "int_fixed410_b112_post2x_prerope_rerank_layers00_08_k2560",
    (
        "qksieve_keymse_requestlocal_fixedalloc_"
        "post2xdualmass_i112_41_l00to08_fulltopk_k1280"
    ): "int_fixed410_b112_post2x_dual_mass_layers00_08",
    (
        "qksieve_keymse_requestlocal_fixedalloc_"
        "prerope32int2_b8_fulltopk_k1280"
    ): "prerope32_int2_candidates_exact_attention",
    (
        "qksieve_keymse_requestlocal_fixedalloc_"
        "prerope32int4_b8_fulltopk_k1280"
    ): "prerope32_int4_candidates_exact_attention",
    (
        "qksieve_keymse_requestlocal_fixedalloc_"
        "prerope32adaptive_b8_fulltopk_k1280"
    ): "prerope32_adaptive_candidates_exact_attention",
    "qksieve_keymse_requestlocal_fixedalloc_b5_fulltopk_k1280": (
        "int_fixed400_b80"
    ),
    "qksieve_keymse_requestlocal_fixedalloc_minifloat_b5_fulltopk_k1280": (
        "minifloat_fixed400_b80_quality_reference"
    ),
    "qksieve_keymse_requestlocal_fixedalloc_b3_fulltopk_k1280": (
        "int_fixed200_b48"
    ),
    (
        "qksieve_keymse_requestlocal_fixedalloc_minifloat_"
        "b8_fulltopk_k1280"
    ): "minifloat_fixed420_b128_quality_reference",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--kernel-json", type=Path)
    parser.add_argument("--trace-root", type=Path, action="append", default=[])
    parser.add_argument(
        "--exclude-topic",
        action="append",
        default=[],
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260731)
    return parser.parse_args()


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def cluster_bootstrap_quality(
    case_deltas: list[float],
    repetitions: int,
    rng: random.Random,
) -> tuple[float, float]:
    samples = []
    for _ in range(repetitions):
        delta = mean(rng.choice(case_deltas) for _ in case_deltas)
        samples.append(100.0 * math.exp(-delta))
    return percentile(samples, 0.025), percentile(samples, 0.975)


def token_bootstrap_quality(
    token_deltas: list[float],
    repetitions: int,
    rng: random.Random,
) -> tuple[float, float]:
    samples = []
    for _ in range(repetitions):
        delta = mean(rng.choice(token_deltas) for _ in token_deltas)
        samples.append(100.0 * math.exp(-delta))
    return percentile(samples, 0.025), percentile(samples, 0.975)


def load_ppl_rows(
    input_roots: list[Path],
    excluded_topics: set[str] | None = None,
) -> tuple[list[dict], dict[str, list[float]]]:
    summaries = sorted(
        path
        for input_root in input_roots
        for path in input_root.glob("*/summary.json")
    )
    if not summaries:
        raise FileNotFoundError(
            f"no summary.json files under {input_roots}"
        )

    rows_by_case: dict[tuple[str, str], dict] = {}
    token_deltas_by_case: dict[tuple[str, str], list[float]] = {}
    for path in summaries:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("schema")
            != "qksieve_coldskip_longcontext_quality_v1"
        ):
            continue
        topic = str(payload["topic"])
        if topic in (excluded_topics or set()):
            continue
        by_variant = {str(row["variant"]): row for row in payload["rows"]}
        token_rows = payload["token_rows"]
        full_tokens = token_rows["full_attention"]
        full_by_index = {
            int(row["target_index"]): float(row["nll"]) for row in full_tokens
        }
        for variant, row in by_variant.items():
            if variant not in LABELS:
                continue
            record = dict(row)
            record["topic"] = topic
            record["source"] = str(path)
            case_key = (topic, variant)
            existing = rows_by_case.get(case_key)
            if existing is not None:
                if not math.isclose(
                    float(existing["nll"]),
                    float(record["nll"]),
                    rel_tol=0.0,
                    abs_tol=1.0e-9,
                ):
                    raise ValueError(f"conflicting duplicate row: {case_key}")
            else:
                rows_by_case[case_key] = record
            if variant == "full_attention":
                continue
            sparse_by_index = {
                int(item["target_index"]): float(item["nll"])
                for item in token_rows[variant]
            }
            if sparse_by_index.keys() != full_by_index.keys():
                raise ValueError(f"unpaired token rows in {path}: {variant}")
            token_deltas_by_case[case_key] = [
                sparse_by_index[index] - full_by_index[index]
                for index in sorted(full_by_index)
            ]
    token_deltas: dict[str, list[float]] = defaultdict(list)
    for (_, variant), values in sorted(token_deltas_by_case.items()):
        token_deltas[variant].extend(values)
    return list(rows_by_case.values()), token_deltas


def summarize_ppl(
    rows: list[dict],
    token_deltas: dict[str, list[float]],
    repetitions: int,
    rng: random.Random,
) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["variant"])].append(row)
    full_rows = grouped["full_attention"]
    full_nll = mean(float(row["nll"]) for row in full_rows)
    full_steady = mean(
        float(row["steady_sparse_seconds_per_step"]) for row in full_rows
    )
    full_by_topic = {
        str(row["topic"]): float(row["nll"]) for row in full_rows
    }

    output = []
    for variant, group in sorted(grouped.items()):
        nll = mean(float(row["nll"]) for row in group)
        steady = mean(
            float(row["steady_sparse_seconds_per_step"]) for row in group
        )
        case_deltas = [
            float(row["nll"]) - full_by_topic[str(row["topic"])]
            for row in group
        ]
        if variant == "full_attention":
            cluster_ci = (100.0, 100.0)
            token_ci = (100.0, 100.0)
        else:
            cluster_ci = cluster_bootstrap_quality(
                case_deltas, repetitions, rng
            )
            token_ci = token_bootstrap_quality(
                token_deltas[variant], repetitions, rng
            )
        output.append(
            {
                "variant": variant,
                "label": LABELS[variant],
                "cases": len(group),
                "tokens": sum(int(row["tokens"]) for row in group),
                "mean_nll": nll,
                "ppl": math.exp(nll),
                "quality_retention_pct": 100.0 * math.exp(full_nll - nll),
                "cluster_bootstrap_quality_ci95_low": cluster_ci[0],
                "cluster_bootstrap_quality_ci95_high": cluster_ci[1],
                "token_bootstrap_quality_ci95_low": token_ci[0],
                "token_bootstrap_quality_ci95_high": token_ci[1],
                "top1_agreement_pct": 100.0
                * mean(float(row.get("top1_agreement", 1.0)) for row in group),
                "mean_kl_full_to_sparse": mean(
                    float(row.get("kl_full_to_sparse_mean", 0.0))
                    for row in group
                ),
                "index_ratio_of_full_kv_pct": 100.0
                * mean(
                    float(row.get("packed_index_ratio_of_full_kv", 0.0))
                    for row in group
                ),
                "steady_ms_per_token": 1000.0 * steady,
                "steady_speedup_vs_full": full_steady / steady,
                "fixed_index_seconds": mean(
                    float(row.get("fixed_sparse_overhead_seconds", 0.0))
                    for row in group
                ),
            }
        )
    return output


def paired_variant_contrasts(
    rows: list[dict],
    repetitions: int,
    rng: random.Random,
) -> list[dict]:
    comparisons = (
        (
            "qksieve_keymse_requestlocal_fixedalloc_b8_fulltopk_k1280",
            (
                "qksieve_keymse_requestlocal_fixedalloc_minifloat_"
                "b8_fulltopk_k1280"
            ),
        ),
        (
            "qksieve_keymse_requestlocal_fixedalloc_b5_fulltopk_k1280",
            (
                "qksieve_keymse_requestlocal_fixedalloc_minifloat_"
                "b5_fulltopk_k1280"
            ),
        ),
        (
            "qksieve_keymse_requestlocal_fixedalloc_b8_fulltopk_k1280",
            (
                "qksieve_keymse_requestlocal_fixedalloc_"
                "prerope32int2_b8_fulltopk_k1280"
            ),
        ),
        (
            "qksieve_keymse_requestlocal_fixedalloc_b8_fulltopk_k1280",
            (
                "qksieve_keymse_requestlocal_fixedalloc_"
                "prerope32int4_b8_fulltopk_k1280"
            ),
        ),
        (
            (
                "qksieve_keymse_requestlocal_fixedalloc_"
                "prerope32int2_b8_fulltopk_k1280"
            ),
            (
                "qksieve_keymse_requestlocal_fixedalloc_"
                "prerope32int4_b8_fulltopk_k1280"
            ),
        ),
    )
    by_case = {
        (str(row["topic"]), str(row["variant"])): row for row in rows
    }
    output = []
    for baseline, variant in comparisons:
        topics = sorted(
            topic
            for topic, candidate in by_case
            if candidate == baseline and (topic, variant) in by_case
        )
        if not topics:
            continue
        deltas = [
            float(by_case[(topic, variant)]["nll"])
            - float(by_case[(topic, baseline)]["nll"])
            for topic in topics
        ]
        low, high = cluster_bootstrap_quality(deltas, repetitions, rng)
        output.append(
            {
                "baseline": baseline,
                "variant": variant,
                "paired_topics": len(topics),
                "mean_delta_nll": mean(deltas),
                "quality_ratio_pct": 100.0 * math.exp(-mean(deltas)),
                "cluster_bootstrap_quality_ci95_low": low,
                "cluster_bootstrap_quality_ci95_high": high,
                "improved_topic_fraction": mean(
                    float(delta < 0.0) for delta in deltas
                ),
            }
        )
    return output


def summarize_kernel(path: Path | None) -> list[dict]:
    if path is None:
        return []
    rows = json.loads(path.read_text(encoding="utf-8"))
    references = {
        int(row["history_count"]): float(row["plain_ms"])
        for row in rows
        if row["profile"] == "auto240_reference"
    }
    output = []
    for row in rows:
        item = dict(row)
        item["correctness_safe_timing"] = "plain_ms"
        item["plain_speedup_vs_auto240"] = (
            references[int(row["history_count"])] / float(row["plain_ms"])
        )
        item["bandmajor_timing_valid"] = bool(
            row.get(
                "untruncated_candidate_sets_equal",
                row["candidate_sets_equal"],
            )
        )
        if not item["bandmajor_timing_valid"]:
            item["bandmajor_speedup_vs_auto240"] = None
        output.append(item)
    return output


def summarize_traces(roots: list[Path]) -> list[dict]:
    grouped: dict[tuple[str, int, float], list[dict]] = defaultdict(list)
    for root in roots:
        paths = sorted(root.glob("*/summary.csv"))
        if not paths:
            raise FileNotFoundError(f"no trace summary.csv files under {root}")
        for path in paths:
            with path.open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    key = (
                        row["family"],
                        int(row["budget_bits"]),
                        float(row["selected_fraction_target"]),
                    )
                    grouped[key].append(row)

    metrics = [
        "physical_bits_mean",
        "index_ratio_of_full_kv",
        "top2_recall_mean",
        "top2_recall_p10",
        "selected_attention_mass_mean",
        "selected_attention_mass_p10",
        "score_pearson_mean",
    ]
    output = []
    for (family, budget, fraction), group in sorted(grouped.items()):
        total_cases = sum(int(row["cases"]) for row in group)
        item: dict[str, float | int | str] = {
            "family": family,
            "budget_bits": budget,
            "selected_fraction": fraction,
            "traces": len(group),
            "cases": total_cases,
        }
        for metric in metrics:
            item[metric] = sum(
                float(row[metric]) * int(row["cases"]) for row in group
            ) / total_cases
        output.append(item)
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    rows, token_deltas = load_ppl_rows(
        args.input_root,
        excluded_topics=set(args.exclude_topic),
    )
    ppl_summary = summarize_ppl(
        rows,
        token_deltas,
        args.bootstrap_repetitions,
        rng,
    )
    contrasts = paired_variant_contrasts(
        rows,
        args.bootstrap_repetitions,
        rng,
    )
    kernel_summary = summarize_kernel(args.kernel_json)
    trace_summary = summarize_traces(args.trace_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "ppl_summary.csv", ppl_summary)
    write_csv(args.output_dir / "ppl_contrasts.csv", contrasts)
    write_csv(args.output_dir / "kernel_summary.csv", kernel_summary)
    write_csv(args.output_dir / "trace_summary.csv", trace_summary)
    payload = {
        "schema": "qksieve_lowbit_ppl_summary_v1",
        "input_roots": [str(path) for path in args.input_root],
        "excluded_topics": sorted(set(args.exclude_topic)),
        "kernel_json": str(args.kernel_json) if args.kernel_json else None,
        "bootstrap_unit_primary": "topic/document",
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "ppl": ppl_summary,
        "ppl_contrasts": contrasts,
        "kernel": kernel_summary,
        "trace": trace_summary,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
