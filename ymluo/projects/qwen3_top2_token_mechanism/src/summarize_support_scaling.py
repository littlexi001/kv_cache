from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[1]
ARTIFACT = PROJECT / "artifacts" / "20260722_support_scaling"
MATCHED = (
    PROJECT / "artifacts" / "20260722_fused_sampleq" / "matched_raw"
)
TOPIC_RUNS = {
    "medicine": "matched_isolated_g01_full_first",
    "politics": "matched_politics_g23",
    "computer": "matched_computer_g45",
    "space": "matched_space_g67",
}
LONG512_RUNS = {
    "medicine": "medicine_first",
    "politics": "politics",
    "computer": "computer",
    "space": "space_c4",
}
LONGER_RUNS = {
    "medicine_180k": (180_000, "medicine_180k"),
    "religion_192k": (192_000, "religion_192k"),
    "space_192k": (192_000, "space_192k"),
    "politics_256k": (256_000, "politics_256k_4gpu"),
}


def load_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload[0] if isinstance(payload, list) else payload


def mean_first_n_csv(path: Path, field: str, count: int) -> float:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))[:count]
    if len(rows) != count:
        raise ValueError(f"{path} has only {len(rows)} rows")
    return sum(float(row[field]) for row in rows) / count


def geometric_ppl(rows: dict[str, dict[str, float]]) -> float:
    return math.exp(sum(row["nll"] for row in rows.values()) / len(rows))


def pipeline_rows(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["results"][0]["rows"]
    return {str(row["method"]): row for row in rows}


def sparse_metrics(row: dict[str, Any]) -> dict[str, float]:
    metrics = {
        "nll": float(row["nll"]),
        "ppl": float(row["ppl"]),
        "online_seconds": float(row["online_seconds"]),
        "attention_link_ratio": float(row["attention_link_ratio"]),
        "candidate_fraction_mean": float(row["candidate_fraction_mean"]),
    }
    if "prefill_seconds" in row:
        metrics["prefill_seconds"] = float(row["prefill_seconds"])
    return metrics


def main() -> None:
    full: dict[str, dict[str, float]] = {}
    fixed2: dict[str, dict[str, float]] = {}
    for topic, run_name in TOPIC_RUNS.items():
        full_payload = json.loads(
            (MATCHED / run_name / "full" / "result.json").read_text(
                encoding="utf-8"
            )
        )
        full_nll = sum(full_payload["token_nll"][:128]) / 128
        fixed2_nll = mean_first_n_csv(
            MATCHED / run_name / "sparse" / "token_results.csv",
            "nll",
            128,
        )
        full[topic] = {"nll": full_nll, "ppl": math.exp(full_nll)}
        fixed2[topic] = {"nll": fixed2_nll, "ppl": math.exp(fixed2_nll)}

    medicine_frontier: dict[str, dict[str, float]] = {}
    for fraction in ("f005", "f010", "f015", "f020"):
        row = load_summary(ARTIFACT / "frontier" / fraction / "summary.json")
        medicine_frontier[fraction] = {
            "nll": float(row["nll"]),
            "ppl": float(row["ppl"]),
            "online_seconds": float(row["online_seconds"]),
            "attention_link_ratio": float(row["attention_link_ratio"]),
            "candidate_fraction_mean": float(row["candidate_fraction_mean"]),
        }

    fixed1: dict[str, dict[str, float]] = {
        "medicine": medicine_frontier["f010"]
    }
    for topic in ("politics", "computer", "space"):
        row = load_summary(
            ARTIFACT / "cross_topic_1pct" / f"{topic}_f010" / "summary.json"
        )
        fixed1[topic] = {
            "nll": float(row["nll"]),
            "ppl": float(row["ppl"]),
            "online_seconds": float(row["online_seconds"]),
            "attention_link_ratio": float(row["attention_link_ratio"]),
            "candidate_fraction_mean": float(row["candidate_fraction_mean"]),
        }

    candidate4: dict[str, dict[str, float]] = {}
    candidate4_root = ARTIFACT / "candidate4_1pct"
    if candidate4_root.exists():
        for topic in TOPIC_RUNS:
            row = load_summary(candidate4_root / topic / "summary.json")
            candidate4[topic] = {
                "nll": float(row["nll"]),
                "ppl": float(row["ppl"]),
                "online_seconds": float(row["online_seconds"]),
                "attention_link_ratio": float(row["attention_link_ratio"]),
                "candidate_fraction_mean": float(row["candidate_fraction_mean"]),
            }

    full_geomean = geometric_ppl(full)
    fixed1_geomean = geometric_ppl(fixed1)
    fixed2_geomean = geometric_ppl(fixed2)
    aggregate: dict[str, Any] = {
        "full_geometric_ppl": full_geomean,
        "fixed1_geometric_ppl": fixed1_geomean,
        "fixed1_quality_retention_percent": 100.0
        * full_geomean
        / fixed1_geomean,
        "fixed2_geometric_ppl": fixed2_geomean,
        "fixed2_quality_retention_percent": 100.0
        * full_geomean
        / fixed2_geomean,
    }
    if candidate4:
        candidate4_geomean = geometric_ppl(candidate4)
        aggregate |= {
            "candidate4_fixed1_geometric_ppl": candidate4_geomean,
            "candidate4_fixed1_quality_retention_percent": 100.0
            * full_geomean
            / candidate4_geomean,
            "candidate4_speedup_vs_candidate6": sum(
                fixed1[topic]["online_seconds"] for topic in TOPIC_RUNS
            )
            / sum(candidate4[topic]["online_seconds"] for topic in TOPIC_RUNS),
        }

    long512_full: dict[str, dict[str, float]] = {}
    long512_countcap: dict[str, dict[str, float]] = {}
    for topic, run_name in TOPIC_RUNS.items():
        full_payload = json.loads(
            (MATCHED / run_name / "full" / "result.json").read_text(
                encoding="utf-8"
            )
        )
        long512_full[topic] = {
            "nll": float(full_payload["nll"]),
            "ppl": float(full_payload["ppl"]),
            "online_seconds": float(
                full_payload["synchronized_model_forward_seconds"]
            ),
        }
        sparse_row = load_summary(
            ARTIFACT
            / "long512"
            / LONG512_RUNS[topic]
            / "summary.json"
        )
        long512_countcap[topic] = sparse_metrics(sparse_row)

    long512_full_geomean = geometric_ppl(long512_full)
    long512_countcap_geomean = geometric_ppl(long512_countcap)
    long512_aggregate = {
        "full_geometric_ppl": long512_full_geomean,
        "countcap_geometric_ppl": long512_countcap_geomean,
        "quality_retention_percent": 100.0
        * long512_full_geomean
        / long512_countcap_geomean,
        "online_speedup": sum(
            row["online_seconds"] for row in long512_full.values()
        )
        / sum(row["online_seconds"] for row in long512_countcap.values()),
        "candidate_fraction_mean": sum(
            row["candidate_fraction_mean"]
            for row in long512_countcap.values()
        )
        / len(long512_countcap),
        "attention_link_ratio_mean": sum(
            row["attention_link_ratio"]
            for row in long512_countcap.values()
        )
        / len(long512_countcap),
    }
    medicine_repeat = sparse_metrics(
        load_summary(
            ARTIFACT / "long512" / "medicine_repeat" / "summary.json"
        )
    )
    space_candidate6 = sparse_metrics(
        load_summary(
            ARTIFACT / "long512" / "space_c6" / "summary.json"
        )
    )

    longer_root = ARTIFACT / "longer"
    longer_full: dict[str, dict[str, float]] = {}
    longer_countcap: dict[str, dict[str, float]] = {}
    for label, (history_tokens, directory) in LONGER_RUNS.items():
        full_payload = json.loads(
            (longer_root / directory / "full" / "result.json").read_text(
                encoding="utf-8"
            )
        )
        longer_full[label] = {
            "history_tokens": history_tokens,
            "nll": float(full_payload["nll"]),
            "ppl": float(full_payload["ppl"]),
            "online_seconds": float(
                full_payload["synchronized_model_forward_seconds"]
            ),
            "prefill_seconds": float(full_payload["prefill_seconds"]),
        }
        longer_countcap[label] = sparse_metrics(
            load_summary(
                longer_root / directory / "countcap" / "summary.json"
            )
        )

    longer_full_geomean = geometric_ppl(longer_full)
    longer_countcap_geomean = geometric_ppl(longer_countcap)
    longer_aggregate = {
        "full_geometric_ppl": longer_full_geomean,
        "countcap_geometric_ppl": longer_countcap_geomean,
        "quality_retention_percent": 100.0
        * longer_full_geomean
        / longer_countcap_geomean,
        "online_speedup": sum(
            row["online_seconds"] for row in longer_full.values()
        )
        / sum(row["online_seconds"] for row in longer_countcap.values()),
        "prefill_speedup": sum(
            row["prefill_seconds"] for row in longer_full.values()
        )
        / sum(row["prefill_seconds"] for row in longer_countcap.values()),
    }
    politics_full = longer_full["politics_256k"]
    politics_candidate_sweep: dict[str, dict[str, float]] = {}
    for candidate, directory in (
        ("candidate3", "politics_256k_4gpu/countcap"),
        ("candidate4", "politics_256k_candidate4"),
        ("candidate6", "politics_256k_candidate6"),
    ):
        row = sparse_metrics(
            load_summary(longer_root / directory / "summary.json")
        )
        row["quality_retention_percent"] = (
            100.0 * politics_full["ppl"] / row["ppl"]
        )
        row["online_speedup"] = (
            politics_full["online_seconds"] / row["online_seconds"]
        )
        politics_candidate_sweep[candidate] = row

    attention_accounting_runs: list[dict[str, float]] = []
    for filename in (
        "pipeline_hf_accounting_qwen128.json",
        "pipeline_hf_accounting_qwen128_repeat.json",
    ):
        rows = pipeline_rows(ARTIFACT / filename)
        preexpanded = rows["full_sdpa"]
        hf_full = rows["full_hf_sdpa_with_repeat_kv"]
        countcap = rows["qkmetric48_sampleq256_candidate4_top1"]
        attention_accounting_runs.append(
            {
                "preexpanded_full_ms": float(preexpanded["pipeline_ms"]),
                "hf_full_ms": float(hf_full["pipeline_ms"]),
                "countcap_ms": float(countcap["pipeline_ms"]),
                "countcap_speedup_vs_preexpanded": float(
                    preexpanded["pipeline_ms"] / countcap["pipeline_ms"]
                ),
                "countcap_speedup_vs_hf_full": float(
                    hf_full["pipeline_ms"] / countcap["pipeline_ms"]
                ),
            }
        )
    conservative_attention_speedup = min(
        row["countcap_speedup_vs_hf_full"]
        for row in attention_accounting_runs
    )
    countcap_e2e_speedup = long512_aggregate["online_speedup"]
    full_attention_fraction = (1.0 - 1.0 / countcap_e2e_speedup) / (
        1.0 - 1.0 / conservative_attention_speedup
    )
    attention_accounting = {
        "runs": attention_accounting_runs,
        "conservative_hf_attention_speedup": conservative_attention_speedup,
        "countcap_e2e_speedup": countcap_e2e_speedup,
        "amdahl_full_attention_fraction": full_attention_fraction,
        "amdahl_other_fraction": 1.0 - full_attention_fraction,
        "amdahl_zero_attention_e2e_limit": 1.0
        / (1.0 - full_attention_fraction),
    }

    report = {
        "protocol": {
            "model": "Qwen3-4B-Instruct",
            "history_tokens": 128000,
            "evaluated_tokens": 128,
            "candidate_index": "QK-Metric48 logscale16 INT4",
            "candidate_generation": "sampleq256 threshold compaction",
        },
        "full": full,
        "fixed2_reference": fixed2,
        "medicine_support_frontier": medicine_frontier,
        "fixed1_cross_topic": fixed1,
        "candidate4_fixed1_cross_topic": candidate4,
        "aggregate": aggregate,
        "long512": {
            "full": long512_full,
            "countcap_candidate4_top1": long512_countcap,
            "aggregate": long512_aggregate,
            "medicine_repeat": medicine_repeat,
            "space_candidate6_top1": space_candidate6,
        },
        "longer_countcap": {
            "full": longer_full,
            "countcap": longer_countcap,
            "aggregate": longer_aggregate,
            "politics_256k_candidate_sweep": politics_candidate_sweep,
        },
        "attention_accounting": attention_accounting,
    }
    output = ARTIFACT / "summary.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
