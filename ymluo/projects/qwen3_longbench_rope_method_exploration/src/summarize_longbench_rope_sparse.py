from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Sequence


VARIANTS = (
    "native_full",
    "full_rope_replay",
    "rope_top2",
    "semantic_top2_postscore",
    "local_global_postscore",
    "local_global_blend25",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260803)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def paired_bootstrap(
    pairs: Sequence[tuple[float, float]], replicates: int, seed: int
) -> tuple[float, float, float]:
    deltas = [left - right for left, right in pairs]
    rng = random.Random(seed)
    boot = []
    for _ in range(replicates):
        boot.append(mean([deltas[rng.randrange(len(deltas))] for _ in deltas]))
    return mean(deltas), quantile(boot, 0.025), quantile(boot, 0.975)


def aggregate(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for variant in VARIANTS:
        selected = [row for row in rows if row["variant"] == variant]
        if not selected:
            continue
        nll = mean([float(row["gold_answer_mean_nll"]) for row in selected])
        def optional(key: str) -> float | None:
            values = [float(row[key]) for row in selected if row.get(key) is not None]
            return mean(values) if values else None
        output.append(
            {
                "variant": variant,
                "sample_count": len(selected),
                "qa_f1_percent": 100.0 * optional("official_qa_f1"),
                "em_percent": 100.0 * optional("normalized_exact_match"),
                "gold_answer_mean_nll": nll,
                "gold_answer_ppl": math.exp(min(nll, 30.0)),
                "first_token_accuracy_percent": 100.0 * optional("first_token_correct"),
                "gold_evidence_token_recall_percent": (
                    100.0 * optional("gold_evidence_token_recall")
                    if optional("gold_evidence_token_recall") is not None else None
                ),
                "gold_chain_complete_percent": (
                    100.0 * optional("gold_chain_complete_rate")
                    if optional("gold_chain_complete_rate") is not None else None
                ),
                "gold_evidence_attention_mass_percent": (
                    100.0 * optional("gold_evidence_attention_mass")
                    if optional("gold_evidence_attention_mass") is not None else None
                ),
                "mean_query_seconds": optional("query_seconds"),
                "mean_generation_seconds": optional("generation_seconds"),
            }
        )
    return output


def paired_rows(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    output: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        sample = str(row["sample_id"])
        variant = str(row["variant"])
        if variant in output[sample]:
            raise RuntimeError(f"duplicate sample/variant row: {sample}/{variant}")
        output[sample][variant] = row
    return output


def comparison(
    pairs: dict[str, dict[str, dict[str, Any]]],
    left: str,
    right: str,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    sample_ids = sorted(pairs)
    for sample_id in sample_ids:
        if left not in pairs[sample_id] or right not in pairs[sample_id]:
            raise RuntimeError(f"missing paired arm for {sample_id}: {left}/{right}")
    metrics: dict[str, Callable[[dict[str, Any]], float]] = {
        "gold_nll": lambda row: float(row["gold_answer_mean_nll"]),
        "qa_f1": lambda row: float(row["official_qa_f1"]),
        "em": lambda row: float(row["normalized_exact_match"]),
        "first_token_correct": lambda row: float(row["first_token_correct"]),
    }
    results: dict[str, Any] = {"left": left, "right": right, "sample_count": len(sample_ids)}
    for offset, (name, function) in enumerate(metrics.items()):
        values = [(function(pairs[sample][left]), function(pairs[sample][right])) for sample in sample_ids]
        estimate, low, high = paired_bootstrap(values, replicates, seed + offset * 1009)
        scale = 100.0 if name in {"qa_f1", "em", "first_token_correct"} else 1.0
        results[f"delta_{name}"] = estimate * scale
        results[f"delta_{name}_ci_low"] = low * scale
        results[f"delta_{name}_ci_high"] = high * scale
    results["em_rescues"] = sum(
        int(not pairs[sample][right]["normalized_exact_match"] and pairs[sample][left]["normalized_exact_match"])
        for sample in sample_ids
    )
    results["em_harms"] = sum(
        int(pairs[sample][right]["normalized_exact_match"] and not pairs[sample][left]["normalized_exact_match"])
        for sample in sample_ids
    )
    return results


def audit(rows: Sequence[dict[str, Any]], pairs: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    expected = set(VARIANTS)
    incomplete = {sample: sorted(expected - set(arms)) for sample, arms in pairs.items() if set(arms) != expected}
    prompt_hash_mismatch = 0
    for arms in pairs.values():
        if len({row["prompt_sha256"] for row in arms.values()}) != 1:
            prompt_hash_mismatch += 1
    replay_errors = [
        float(row["dense_replay_max_logit_error"])
        for row in rows
        if row.get("dense_replay_max_logit_error") is not None
    ]
    budget = [float(row.get("support_budget_violation_fraction", 0.0) or 0.0) for row in rows]
    duplicates = [float(row.get("duplicate_support_violation_fraction", 0.0) or 0.0) for row in rows]
    return {
        "sample_count": len(pairs),
        "row_count": len(rows),
        "expected_row_count": len(pairs) * len(VARIANTS),
        "incomplete_samples": incomplete,
        "prompt_hash_mismatch_count": prompt_hash_mismatch,
        "dense_replay_max_logit_error": max(replay_errors, default=float("nan")),
        "support_budget_violation_max": max(budget, default=0.0),
        "duplicate_support_violation_max": max(duplicates, default=0.0),
        "passed": (
            len(pairs) == 18
            and len(rows) == 18 * len(VARIANTS)
            and not incomplete
            and prompt_hash_mismatch == 0
            and max(replay_errors, default=0.0) == 0.0
            and max(budget, default=0.0) == 0.0
            and max(duplicates, default=0.0) == 0.0
        ),
    }


def per_sample_deltas(
    pairs: dict[str, dict[str, dict[str, Any]]], left: str, right: str
) -> list[dict[str, Any]]:
    rows = []
    for sample_id in sorted(pairs):
        a, b = pairs[sample_id][left], pairs[sample_id][right]
        rows.append(
            {
                "sample_id": sample_id,
                "evidence_position_bin": a["evidence_position_bin"],
                "hotpot_type": a["hotpot_type"],
                "delta_gold_nll": float(a["gold_answer_mean_nll"]) - float(b["gold_answer_mean_nll"]),
                "delta_qa_f1": float(a["official_qa_f1"]) - float(b["official_qa_f1"]),
                "delta_em": int(a["normalized_exact_match"]) - int(b["normalized_exact_match"]),
                "delta_evidence_recall": float(a["gold_evidence_token_recall"]) - float(b["gold_evidence_token_recall"]),
                "delta_evidence_mass": float(a["gold_evidence_attention_mass"]) - float(b["gold_evidence_attention_mass"]),
                "left_prediction": a["prediction"],
                "right_prediction": b["prediction"],
                "answers": a["answers"],
            }
        )
    return rows


def make_plots(summary: Sequence[dict[str, Any]], deltas: Sequence[dict[str, Any]], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    methods = [row["variant"] for row in summary]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].bar(methods, [row["qa_f1_percent"] for row in summary], color="#2a9d8f")
    axes[0].set_ylabel("QA-F1 (%) ↑")
    axes[1].bar(methods, [row["em_percent"] for row in summary], color="#457b9d")
    axes[1].set_ylabel("Exact match (%) ↑")
    axes[2].bar(methods, [row["gold_answer_mean_nll"] for row in summary], color="#e9c46a")
    axes[2].set_ylabel("Gold mean token NLL ↓")
    for axis in axes:
        axis.tick_params(axis="x", rotation=55)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Qwen3-8B · LongBench HotpotQA strict aligned cohort · n=18")
    fig.tight_layout()
    fig.savefig(output_dir / "quality_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 5))
    colors = {"early": "#2a9d8f", "middle": "#e9c46a", "late": "#e76f51"}
    for row in deltas:
        axis.scatter(
            100.0 * row["delta_evidence_recall"],
            row["delta_gold_nll"],
            color=colors.get(row["evidence_position_bin"], "#666666"),
            alpha=0.85,
        )
    axis.axhline(0.0, color="black", linewidth=1)
    axis.axvline(0.0, color="black", linewidth=1)
    axis.set_xlabel("LS − post2 gold-evidence recall (percentage points)")
    axis.set_ylabel("LS − post2 gold NLL (lower is better)")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "retrieval_answer_scatter.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    ordered = sorted(deltas, key=lambda row: row["delta_gold_nll"])
    fig, axis = plt.subplots(figsize=(12, 4.5))
    labels = [row["sample_id"][:8] for row in ordered]
    bars = axis.bar(labels, [row["delta_gold_nll"] for row in ordered])
    for bar, row in zip(bars, ordered):
        bar.set_color(colors.get(row["evidence_position_bin"], "#666666"))
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set_ylabel("LS − post2 gold NLL (lower is better)")
    axis.tick_params(axis="x", rotation=55)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "paired_nll_deltas.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    shard_rows = []
    for index, shard in enumerate(args.shards):
        current = read_jsonl(shard / "rows.jsonl")
        rows.extend(current)
        shard_rows.append({"shard": index, "row_count": len(current), "sample_count": len({r['sample_id'] for r in current})})
    rows.sort(key=lambda row: (str(row["sample_id"]), VARIANTS.index(str(row["variant"]))))
    pairs = paired_rows(rows)
    summary = aggregate(rows)
    comparisons = [
        comparison(pairs, "local_global_postscore", "rope_top2", args.bootstrap_replicates, args.seed),
        comparison(pairs, "local_global_blend25", "rope_top2", args.bootstrap_replicates, args.seed + 5000),
        comparison(pairs, "local_global_postscore", "native_full", args.bootstrap_replicates, args.seed + 10000),
    ]
    primary_deltas = per_sample_deltas(pairs, "local_global_postscore", "rope_top2")
    integrity = audit(rows, pairs)
    if not integrity["passed"]:
        write_json(args.output_dir / "integrity_failure.json", integrity)
        raise RuntimeError(f"integrity audit failed: {integrity}")
    write_jsonl(args.output_dir / "rows.jsonl", rows)
    write_csv(args.output_dir / "rows.csv", rows)
    write_csv(args.output_dir / "summary.csv", summary)
    write_csv(args.output_dir / "comparisons.csv", comparisons)
    write_csv(args.output_dir / "paired_ls_vs_post2.csv", primary_deltas)
    write_json(
        args.output_dir / "summary.json",
        {"integrity": integrity, "shards": shard_rows, "summary": summary, "comparisons": comparisons},
    )
    make_plots(summary, primary_deltas, args.output_dir)
    primary = comparisons[0]
    decision = (
        "PASS"
        if primary["delta_gold_nll_ci_high"] < 0.0
        else "FAIL"
        if primary["delta_gold_nll_ci_low"] > 0.0
        else "INSUFFICIENT"
    )
    report = [
        "# LongBench RoPE sparse retrieval result",
        "",
        f"- Integrity: **PASS** ({integrity['sample_count']} samples, {integrity['row_count']} rows).",
        f"- Primary decision: **{decision}**.",
        f"- LS − exact post-RoPE Top-2% gold NLL: {primary['delta_gold_nll']:+.4f} "
        f"(95% paired bootstrap CI [{primary['delta_gold_nll_ci_low']:+.4f}, {primary['delta_gold_nll_ci_high']:+.4f}]).",
        f"- QA-F1 delta: {primary['delta_qa_f1']:+.2f} points; EM delta: {primary['delta_em']:+.2f} points.",
        f"- EM rescues / harms: {primary['em_rescues']} / {primary['em_harms']}.",
        "",
        "The method is paper-supporting only if the NLL interval is wholly below zero and the retrieval/answer evidence is directionally coherent. See `summary.csv`, `comparisons.csv`, and the per-sample delta table for the claim boundary.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

