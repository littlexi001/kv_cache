from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Callable, Iterable, Sequence

import numpy as np


BINS = ("common", "medium", "tail")


def load_rows(output_dir: Path) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in sorted(output_dir.glob("rows_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row["case_id"]] = row
    return sorted(rows.values(), key=lambda row: row["case_id"])


def avg(values: Iterable[float]) -> float:
    values = list(values)
    return float(mean(values)) if values else math.nan


def se(values: Iterable[float]) -> float:
    values = list(values)
    return float(stdev(values) / math.sqrt(len(values))) if len(values) > 1 else math.nan


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def f4(value: float) -> str:
    return "NA" if not math.isfinite(value) else f"{value:.4f}"


def group_summary(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["bin"], int(row["target_context_tokens"]))].append(row)
    output = []
    for (label, length), group in sorted(groups.items(), key=lambda item: (item[0][1], BINS.index(item[0][0]))):
        metric_getters: dict[str, Callable[[dict[str, Any]], float]] = {
            "candidate_accuracy": lambda row: float(row["evidence"]["candidate_correct"]),
            "greedy_accuracy": lambda row: float(row["evidence"]["greedy_correct"]),
            "mean_gold_nll": lambda row: -float(row["evidence"]["gold_logprob"]),
            "control_candidate_accuracy": lambda row: float(row["control"]["candidate_correct"]),
            "control_mean_gold_nll": lambda row: -float(row["control"]["gold_logprob"]),
            "evidence_lift_nats": lambda row: float(row["evidence_lift_nats"]),
            "candidate_margin": lambda row: float(row["evidence"]["candidate_margin"]),
        }
        attention_keys = (
            "rule_mass", "code_mass", "final_mass", "both_rules_top2", "code_recall_top2",
            "final_rank_fraction", "final_logit", "final_cosine", "background_log_partition",
            "background_max_logit", "needle_log_odds", "residual_after_top20_mass",
        )
        for key in attention_keys:
            metric_getters[key] = lambda row, key=key: float(row["attention"]["model_mean"][key])
        summary: dict[str, Any] = {"bin": label, "length": length, "n": len(group)}
        for metric, getter in metric_getters.items():
            values = [getter(row) for row in group]
            summary[metric] = avg(values)
            summary[metric + "_se"] = se(values)
        summary["geometric_mean_gold_ppl"] = math.exp(summary["mean_gold_nll"])
        summary["median_gold_ppl"] = float(median(row["evidence"]["gold_ppl"] for row in group))
        output.append(summary)
    return output


def failure_boundaries(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["bin"], int(row["sample_index"]))].append(row)
    result = []
    for (label, sample_index), group in sorted(groups.items()):
        group.sort(key=lambda row: int(row["target_context_tokens"]))
        states = [bool(row["evidence"]["candidate_correct"]) for row in group]
        lengths = [int(row["target_context_tokens"]) for row in group]
        failure_lengths = [length for length, state in zip(lengths, states) if not state]
        persistent = None
        for index, length in enumerate(lengths):
            if not any(states[index:]):
                persistent = length
                break
        recovery_count = sum((not states[index - 1]) and states[index] for index in range(1, len(states)))
        result.append(
            {
                "bin": label,
                "sample_index": sample_index,
                "gold_word": group[0]["gold_word"],
                "first_failure": min(failure_lengths) if failure_lengths else None,
                "persistent_failure": persistent,
                "last_success": max((length for length, state in zip(lengths, states) if state), default=None),
                "recovery_count": recovery_count,
                "states": [{"length": length, "correct": state} for length, state in zip(lengths, states)],
            }
        )
    return result


def regression(rows: Sequence[dict[str, Any]], outcome: Callable[[dict[str, Any]], float]) -> dict[str, Any]:
    # Exploratory OLS. Paired word/filler effects are summarized separately; this regression
    # is used only to expose the frequency x length direction after controlling output prior.
    x_rows = []
    y_rows = []
    for row in rows:
        zipf = float(row["chain"][2]["zipf"])
        log_length = math.log(int(row["target_context_tokens"]))
        prior = float(row["control"]["gold_logprob"])
        x_rows.append([1.0, zipf, log_length, zipf * log_length, prior])
        y_rows.append(outcome(row))
    x = np.asarray(x_rows, dtype=np.float64)
    y = np.asarray(y_rows, dtype=np.float64)
    coefficients, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    prediction = x @ coefficients
    ss_total = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - float(np.sum((y - prediction) ** 2)) / ss_total if ss_total > 0 else math.nan
    names = ("intercept", "zipf", "log_length", "zipf_x_log_length", "control_gold_logprob")
    return {"coefficients": dict(zip(names, coefficients.tolist())), "r2": r2, "n": len(y_rows)}


def paired_differences(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {
        (row["bin"], int(row["sample_index"]), int(row["target_context_tokens"])): row
        for row in rows
    }
    lengths = sorted({int(row["target_context_tokens"]) for row in rows})
    samples = sorted({int(row["sample_index"]) for row in rows})
    result = []
    for length in lengths:
        pairs = []
        for sample in samples:
            common = indexed.get(("common", sample, length))
            tail = indexed.get(("tail", sample, length))
            if common is None or tail is None:
                continue
            pairs.append(
                {
                    "accuracy": float(common["evidence"]["candidate_correct"])
                    - float(tail["evidence"]["candidate_correct"]),
                    "lift": float(common["evidence_lift_nats"]) - float(tail["evidence_lift_nats"]),
                    "needle_log_odds": float(common["attention"]["model_mean"]["needle_log_odds"])
                    - float(tail["attention"]["model_mean"]["needle_log_odds"]),
                    "rule_mass": float(common["attention"]["model_mean"]["rule_mass"])
                    - float(tail["attention"]["model_mean"]["rule_mass"]),
                }
            )
        result.append(
            {
                "length": length,
                "n": len(pairs),
                **(
                    {f"common_minus_tail_{key}": avg(pair[key] for pair in pairs) for key in pairs[0]}
                    if pairs
                    else {}
                ),
            }
        )
    return result


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_report(
    rows: Sequence[dict[str, Any]], groups: Sequence[dict[str, Any]], boundaries: Sequence[dict[str, Any]],
    paired: Sequence[dict[str, Any]], regressions: dict[str, Any],
) -> str:
    lengths = sorted({int(row["target_context_tokens"]) for row in rows})
    placement = rows[0].get("placement", "unknown") if rows else "unknown"
    expected = len(lengths) * len(BINS) * len({row["sample_index"] for row in rows}) if rows else 0
    lines = [
        "# Common vs long-tail needle failure boundary",
        "",
        f"- Placement: `{placement}`",
        f"- Completed paired cases: {len(rows)}/{expected}",
        "- Commonness: English wordfreq Zipf frequency; WordNet concrete nouns only; every code is one stable Qwen token.",
        "- Primary decision: within-bin 8-way candidate accuracy. Evidence lift subtracts the matched no-evidence output prior.",
        "",
        "## Results by length",
        "",
        "| Tokens | Bin | Candidate acc. | Greedy acc. | Geomean PPL | Evidence lift (nat) | Rule mass | Needle log-odds | Rest mass after Top-20 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in groups:
        lines.append(
            f"| {row['length']} | {row['bin']} | {pct(row['candidate_accuracy'])} | "
            f"{pct(row['greedy_accuracy'])} | {f4(row['geometric_mean_gold_ppl'])} | "
            f"{f4(row['evidence_lift_nats'])} | {pct(row['rule_mass'])} | "
            f"{f4(row['needle_log_odds'])} | {pct(row['residual_after_top20_mass'])} |"
        )
    lines.extend(
        [
            "",
            "## Paired common minus tail",
            "",
            "Positive values favor common needles. Accuracy is percentage-point difference; other metrics are raw differences.",
            "",
            "| Tokens | Accuracy | Evidence lift | Needle log-odds | Rule mass |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in paired:
        lines.append(
            f"| {row['length']} | {100 * row.get('common_minus_tail_accuracy', math.nan):.1f} pp | "
            f"{f4(row.get('common_minus_tail_lift', math.nan))} | "
            f"{f4(row.get('common_minus_tail_needle_log_odds', math.nan))} | "
            f"{f4(row.get('common_minus_tail_rule_mass', math.nan))} |"
        )
    lines.extend(
        [
            "",
            "## Failure-boundary diagnostic",
            "",
            "`persistent failure` is the first tested length after which that needle never recovers. Recoveries are reported because RoPE or prompt effects can make correctness non-monotone.",
            "",
            "| Bin | Median first failure | Median persistent failure | Total recoveries |",
            "|---|---:|---:|---:|",
        ]
    )
    for label in BINS:
        subset = [row for row in boundaries if row["bin"] == label]
        first = [row["first_failure"] for row in subset if row["first_failure"] is not None]
        persistent = [row["persistent_failure"] for row in subset if row["persistent_failure"] is not None]
        lines.append(
            f"| {label} | {int(median(first)) if first else '> max'} | "
            f"{int(median(persistent)) if persistent else '> max'} | "
            f"{sum(row['recovery_count'] for row in subset)} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation rule",
            "",
            "- Common has lower raw PPL but no advantage in evidence lift / needle log-odds: output prior, not easier retrieval.",
            "- Common has higher evidence lift and needle log-odds after controlling prior: common semantics are genuinely easier to retrieve.",
            "- Tail has higher rule mass or needle log-odds: rarity provides distinctiveness that can outweigh weaker learned directions.",
            "- The softmax variable to track is background `logsumexp`, not token count alone; harder distractors can raise it faster at the same length.",
            "",
            "## Exploratory OLS",
            "",
            "Model: outcome ~ Zipf + log(length) + Zipf*log(length) + matched control gold log-probability.",
            "",
            "```json",
            json.dumps(regressions, ensure_ascii=False, indent=2),
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    rows = load_rows(output_dir)
    if not rows:
        raise SystemExit(f"no rows_*.jsonl in {output_dir}")
    groups = group_summary(rows)
    boundaries = failure_boundaries(rows)
    paired = paired_differences(rows)
    regressions = {
        "evidence_lift": regression(rows, lambda row: float(row["evidence_lift_nats"])),
        "needle_log_odds": regression(
            rows, lambda row: float(row["attention"]["model_mean"]["needle_log_odds"])
        ),
    }
    summary = {
        "completed_cases": len(rows),
        "groups": groups,
        "failure_boundaries": boundaries,
        "paired_common_minus_tail": paired,
        "regressions": regressions,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(output_dir / "group_summary.csv", groups)
    write_csv(output_dir / "failure_boundaries.csv", boundaries)
    (output_dir / "report.md").write_text(
        build_report(rows, groups, boundaries, paired, regressions), encoding="utf-8"
    )
    print(json.dumps({"rows": len(rows), "report": str(output_dir / "report.md")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
