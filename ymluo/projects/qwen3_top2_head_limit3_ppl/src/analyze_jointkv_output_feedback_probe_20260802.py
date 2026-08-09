#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def distribution(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": float(tensor.mean()),
        "p50": float(torch.quantile(tensor, 0.50)),
        "p90": float(torch.quantile(tensor, 0.90)),
        "max": float(tensor.max()),
    }


def quality_from_losses(
    full_losses: list[float], sparse_losses: list[float], start: int
) -> dict[str, float]:
    full = full_losses[start:]
    sparse = sparse_losses[start:]
    if not full:
        raise ValueError("quality window is empty")
    full_nll = sum(full) / len(full)
    sparse_nll = sum(sparse) / len(sparse)
    return {
        "tokens": len(full),
        "full_nll": full_nll,
        "sparse_nll": sparse_nll,
        "full_ppl": math.exp(full_nll),
        "sparse_ppl": math.exp(sparse_nll),
        "quality_ratio": math.exp(full_nll - sparse_nll),
    }


def summarize_records(
    records: list[dict[str, float | int]],
    contract: dict[str, Any],
) -> dict[str, Any]:
    recent_tokens = int(contract["recent_tokens"])
    sink_tokens = int(contract["sink_tokens"])
    binary_bits = int(contract["binary_bits"])
    residual_id_bits = int(contract["residual_vq_bits"])
    risk_bits = int(contract.get("risk_error_bits", 4))
    residual_binary_bits = int(contract["residual_binary_bits"])
    base_scan_bits = binary_bits + residual_id_bits + 2 * risk_bits
    refinement_scan_bits = residual_binary_bits + risk_bits
    active_fractions = []
    index_scan_bits = []
    modeled_read_ratios = []
    for record in records:
        history = int(record["history_tokens"])
        recent = min(recent_tokens, history)
        sink = min(sink_tokens, history - recent)
        active = int(record["selected_remote_tokens"]) + recent + sink
        active_fraction = active / history
        active_fractions.append(active_fraction)
        remote_fraction = int(record["remote_tokens"]) / history
        current_index_bits = remote_fraction * (
            base_scan_bits
            + refinement_scan_bits * float(record["refined_remote_fraction"])
        )
        index_scan_bits.append(current_index_bits)
        modeled_read_ratios.append(
            (current_index_bits + 4096.0 * active_fraction) / 4096.0
        )
    return {
        "record_count": len(records),
        "active_kv_fraction": distribution(active_fractions),
        "index_scan_bits_per_history_token": distribution(index_scan_bits),
        "modeled_read_ratio_vs_full_kv": distribution(modeled_read_ratios),
        "selected_remote_fraction": distribution(
            [float(record["selected_remote_fraction"]) for record in records]
        ),
        "selected_remote_tokens": distribution(
            [float(record["selected_remote_tokens"]) for record in records]
        ),
        "refined_remote_fraction": distribution(
            [float(record["refined_remote_fraction"]) for record in records]
        ),
        "refined_remote_tokens": distribution(
            [float(record["refined_remote_tokens"]) for record in records]
        ),
        "required_remote_fraction": distribution(
            [float(record["required_remote_fraction"]) for record in records]
        ),
        "local_attention_relative_l2": distribution(
            [
                float(record["local_attention_relative_l2"])
                for record in records
                if "local_attention_relative_l2" in record
            ]
        ),
        "local_attention_l2": distribution(
            [
                float(record["local_attention_l2"])
                for record in records
                if "local_attention_l2" in record
            ]
        ),
        "selected_attention_mass": distribution(
            [
                float(record["selected_attention_mass"])
                for record in records
                if "selected_attention_mass" in record
            ]
        ),
        "output_error_multiplier": distribution(
            [
                float(record["output_error_multiplier"])
                for record in records
                if "output_error_multiplier" in record
            ]
        ),
        "projected_attention_error_l2": distribution(
            [
                float(record["projected_attention_error_l2"])
                for record in records
                if "projected_attention_error_l2" in record
            ]
        ),
        "output_error_normalizer_l2": distribution(
            [
                float(record["output_error_normalizer_l2"])
                for record in records
                if "output_error_normalizer_l2" in record
            ]
        ),
        "projected_layer_relative_l2": distribution(
            [
                float(record["projected_layer_relative_l2"])
                for record in records
                if "projected_layer_relative_l2" in record
            ]
        ),
    }


def summarize_file(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if len(payload["rows"]) != 1:
        raise ValueError(f"expected one text in {path}")
    row = payload["rows"][0]
    full_losses = [float(value) for value in row["full_token_nll"]]
    sparse_losses = [float(value) for value in row["sparse_token_nll"]]
    feedback = bool(payload["contract"]["one_shot_output_error_feedback"])
    feedback_source = payload["contract"].get(
        "one_shot_feedback_source", "first_decode"
    )
    probe_loss_included = feedback and feedback_source == "first_decode"
    token_count = len(full_losses)
    aggregate = dict(payload["aggregate"])
    if probe_loss_included:
        if token_count <= 1:
            raise ValueError("feedback evaluation needs a post-probe token")
        steady_top1 = (
            float(aggregate["top1_agreement"]) * token_count - 1.0
        ) / (token_count - 1)
        steady_kl = (
            float(aggregate["full_to_sparse_kl"]) * token_count
        ) / (token_count - 1)
    else:
        steady_top1 = float(aggregate["top1_agreement"])
        steady_kl = float(aggregate["full_to_sparse_kl"])
    records = row.get("budget_records") or []
    probe_records = [
        record for record in records if int(record.get("one_shot_full_probe", 0))
    ]
    steady_records = [
        record for record in records if not int(record.get("one_shot_full_probe", 0))
    ]
    if not feedback:
        steady_records = records
    first_history = min(
        (int(record["history_tokens"]) for record in records),
        default=-1,
    )
    aligned_records = [
        record
        for record in records
        if int(record["history_tokens"]) != first_history
    ]
    return {
        "file": path.name,
        "feedback": feedback,
        "feedback_source": feedback_source,
        "all_tokens": quality_from_losses(full_losses, sparse_losses, 0),
        "steady_after_probe": {
            **quality_from_losses(
                full_losses, sparse_losses, 1 if probe_loss_included else 0
            ),
            "top1_agreement": steady_top1,
            "full_to_sparse_kl": steady_kl,
        },
        "aligned_after_first_quality": quality_from_losses(
            full_losses, sparse_losses, 1 if probe_loss_included else 0
        ),
        "probe_budget": summarize_records(
            probe_records,
            payload["contract"],
        ),
        "steady_budget": summarize_records(
            steady_records,
            payload["contract"],
        ),
        "aligned_budget_after_first": summarize_records(
            aligned_records,
            payload["contract"],
        ),
    }


def main() -> None:
    args = parse_args()
    files = sorted(
        path
        for path in args.input_dir.glob("*.json")
        if not path.name.endswith("summary.json")
    )
    if not files:
        raise ValueError(f"no JSON results in {args.input_dir}")
    runs = [summarize_file(path) for path in files]
    by_name = {run["file"].removesuffix(".json"): run for run in runs}
    pairs = []
    for name, feedback in sorted(by_name.items()):
        if not name.endswith("_feedback"):
            continue
        control_name = name.removesuffix("_feedback") + "_control"
        control = by_name.get(control_name)
        if control is None:
            raise ValueError(f"missing paired control for {name}")
        pairs.append(
            {
                "name": name.removesuffix("_feedback"),
                "control": control,
                "feedback": feedback,
                "steady_quality_delta_points": 100.0
                * (
                    feedback["aligned_after_first_quality"]["quality_ratio"]
                    - control["aligned_after_first_quality"]["quality_ratio"]
                ),
                "steady_active_kv_delta_points": 100.0
                * (
                    feedback["aligned_budget_after_first"]["active_kv_fraction"]["mean"]
                    - control["aligned_budget_after_first"]["active_kv_fraction"]["mean"]
                ),
            }
        )
    output = {
        "schema": "jointkv-output-feedback-probe-v1",
        "runs": runs,
        "pairs": pairs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    for pair in pairs:
        control = pair["control"]
        feedback = pair["feedback"]
        print(
            pair["name"],
            f"control_q={100 * control['aligned_after_first_quality']['quality_ratio']:.3f}%",
            f"feedback_q={100 * feedback['aligned_after_first_quality']['quality_ratio']:.3f}%",
            f"control_kv={100 * control['aligned_budget_after_first']['active_kv_fraction']['mean']:.2f}%",
            f"feedback_kv={100 * feedback['aligned_budget_after_first']['active_kv_fraction']['mean']:.2f}%",
            f"feedback_kl={feedback['steady_after_probe']['full_to_sparse_kl']:.6f}",
        )


if __name__ == "__main__":
    main()
