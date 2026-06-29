from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_qwen3_top2_head_limit3_ppl import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    attention_mode,
    clone_past_key_values,
    install_qwen3_attention_patch,
    model_forward,
    pick_input_device,
    prefill_cache,
    read_text_prefix,
    resolve_dtype,
)


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "PCIC-R prototype: blockwise Pairwise-CIC selection plus token-level margin rescue. "
            "Each block uses a short calibration prefix to choose a compressed layer combo, then "
            "evaluates the following tokens with rescue-to-full when next-token margin is low."
        )
    )
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--text_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prefill_tokens", type=int, default=4096)
    parser.add_argument(
        "--start_token_offset",
        type=int,
        default=0,
        help="Token offset into the tokenized text before prefill/calibration/eval slicing.",
    )
    parser.add_argument("--num_blocks", type=int, default=2)
    parser.add_argument("--calibration_tokens", type=int, default=16)
    parser.add_argument("--eval_tokens_per_block", type=int, default=64)
    parser.add_argument("--chunk_size", type=int, default=16)
    parser.add_argument("--max_chars", type=int, default=4_000_000)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--recent_tokens", type=int, default=512)
    parser.add_argument("--landmark_stride", type=int, default=64)
    parser.add_argument(
        "--budget_type",
        choices=["landmark", "head_recent"],
        default="landmark",
        help="Compression budget used for selected layers. head_recent compresses only non-full heads in those layers.",
    )
    parser.add_argument(
        "--full_heads",
        type=int,
        default=8,
        help="For --budget_type head_recent, number of full-attention heads to keep in each selected layer.",
    )
    parser.add_argument(
        "--combos",
        required=True,
        help="Semicolon-separated layer combos, e.g. '7,6;0,13;0,7'.",
    )
    parser.add_argument(
        "--rescue_margin",
        type=float,
        default=0.0,
        help="If previous next-token logit top1-top2 margin is below this threshold, use full attention.",
    )
    parser.add_argument(
        "--rescue_strategy",
        choices=[
            "none",
            "margin",
            "calib_margin",
            "calib_entropy",
            "calib_margin_entropy",
            "calib_disagreement",
            "block_fallback",
            "adaptive_block_fallback",
            "sentinel_block_fallback",
            "calib_meta_fallback",
        ],
        default="margin",
        help=(
            "Rescue gate type. margin keeps the old fixed margin rule; calib_* builds a rule from "
            "calibration compressed-vs-full token loss gaps."
        ),
    )
    parser.add_argument(
        "--risk_quantile",
        type=float,
        default=0.8,
        help="Calibration loss-gap quantile used to define high-risk compressed tokens.",
    )
    parser.add_argument(
        "--risk_positive_gap",
        type=float,
        default=0.0,
        help="Minimum compressed-vs-full loss gap for a calibration token to be treated as risky.",
    )
    parser.add_argument(
        "--risk_rescue_fraction",
        type=float,
        default=0.25,
        help="Target fraction of tokens to rescue under a calibrated risk gate.",
    )
    parser.add_argument(
        "--disagreement_metric",
        choices=["js", "l2"],
        default="js",
        help="Metric for compressed-policy disagreement rescue.",
    )
    parser.add_argument(
        "--safe_delta_loss",
        type=float,
        default=0.02,
        help="Calibration combo is safe if loss <= baseline_loss + this value; choose fastest among safe combos.",
    )
    parser.add_argument(
        "--combo_select_policy",
        choices=[
            "fastest_safe",
            "min_loss",
            "risk_pareto",
            "risk_budget",
            "risk_memory",
            "risk_memory_sentinel",
            "risk_memory_sentinel_all",
            "risk_memory_confidence_routed",
            "risk_memory_confidence_fast",
            "risk_memory_confidence_lazy",
            "risk_memory_horizon_gate",
        ],
        default="fastest_safe",
        help="Blockwise combo selection policy from calibration rows.",
    )
    parser.add_argument(
        "--risk_memory_loss_slack",
        type=float,
        default=0.2,
        help="Risk-memory selector considers combos within this loss slack from current block min-loss.",
    )
    parser.add_argument(
        "--risk_memory_seed_tokens",
        type=int,
        default=0,
        help="Optional prefill-tail calibration tokens used to seed risk-memory before block 0.",
    )
    parser.add_argument(
        "--risk_memory_use_history",
        type=str2bool,
        default=True,
        help="If false, do not seed or update cross-block risk memory; current-block risk scoring is still available.",
    )
    parser.add_argument(
        "--block_risk_max_gap",
        type=float,
        default=0.2,
        help="Block fallback triggers if the chosen combo calibration max loss gap is above this value.",
    )
    parser.add_argument(
        "--block_risk_positive_ratio",
        type=float,
        default=0.5,
        help="Block fallback triggers if the chosen combo positive loss-gap ratio is above this value.",
    )
    parser.add_argument(
        "--adaptive_loss_slack",
        type=float,
        default=0.02,
        help="Adaptive fallback allows candidate calibration delta loss to be this much worse than the chosen combo.",
    )
    parser.add_argument(
        "--adaptive_max_gap_improvement",
        type=float,
        default=0.05,
        help="Minimum max loss-gap reduction required for adaptive block fallback.",
    )
    parser.add_argument(
        "--adaptive_positive_ratio_improvement",
        type=float,
        default=0.125,
        help="Minimum positive loss-gap ratio reduction required for adaptive block fallback.",
    )
    parser.add_argument(
        "--adaptive_require_degraded",
        type=str2bool,
        default=True,
        help="If true, adaptive fallback only triggers when the chosen combo has positive calibration delta loss.",
    )
    parser.add_argument(
        "--sentinel_tokens",
        type=int,
        default=8,
        help="Number of eval-prefix tokens used to accept or reject sentinel_block_fallback proposals.",
    )
    parser.add_argument(
        "--sentinel_all_tokens",
        type=int,
        default=0,
        help="Optional eval-prefix tokens for all-candidate sentinel routes; 0 reuses --sentinel_tokens.",
    )
    parser.add_argument(
        "--sentinel_cascade_initial_tokens",
        type=int,
        default=0,
        help="If >0 and smaller than the route sentinel length, run this many tokens first and extend only when the early sentinel is not confident.",
    )
    parser.add_argument(
        "--sentinel_cascade_accept_margin",
        type=float,
        default=0.15,
        help="Minimum early sentinel loss margin for accepting a cascade decision without extending to the full sentinel length.",
    )
    parser.add_argument(
        "--sentinel_cascade_extend_topk",
        type=int,
        default=0,
        help="If >0, extend only the top-k early sentinel candidates plus memory/min-loss anchors; 0 extends all candidates.",
    )
    parser.add_argument(
        "--sentinel_cascade_anchor_combos",
        default="",
        help="Semicolon-separated combo anchors that are always included in cascade extension when present in candidates.",
    )
    parser.add_argument(
        "--sentinel_cascade_anchor_accept_on_match",
        type=str2bool,
        default=False,
        help="If true, accept the early cascade decision when it already selects a validation-prior anchor combo.",
    )
    parser.add_argument(
        "--sentinel_cascade_accept_low_spread",
        type=float,
        default=0.0,
        help="If >0, accept early when the early sentinel best-vs-runner-up loss spread is at most this value.",
    )
    parser.add_argument(
        "--sentinel_cascade_skip_anchor_nonpositive_gain",
        type=str2bool,
        default=False,
        help="If true, skip cascade extension when the early selected combo is a validation-prior anchor and horizon_gain_ratio <= 0.",
    )
    parser.add_argument(
        "--sentinel_batched_candidates",
        type=str2bool,
        default=False,
        help="If true, evaluate all-candidate sentinel stages with batch-row layer budget maps.",
    )
    parser.add_argument(
        "--sentinel_loss_slack",
        type=float,
        default=0.0,
        help="Sentinel fallback is accepted if proposed loss <= original loss + this slack.",
    )
    parser.add_argument(
        "--sentinel_all_min_margin",
        type=float,
        default=0.0,
        help="For risk_memory_sentinel_all, only trust the best sentinel candidate if it beats the runner-up by this loss margin; otherwise fall back to memory anchor.",
    )
    parser.add_argument(
        "--sentinel_pairwise_min_margin",
        type=float,
        default=0.05,
        help="For risk_memory_confidence_routed, trust min-loss over memory in low-confidence all-candidate cases only if pairwise sentinel loss improves by at least this margin.",
    )
    parser.add_argument(
        "--pairwise_candidate_probe",
        type=str2bool,
        default=True,
        help="If false, disable pairwise/horizon candidate probing and keep only the memory anchor candidate.",
    )
    parser.add_argument(
        "--horizon_gate_min_gain",
        type=float,
        default=0.0,
        help="For risk_memory_horizon_gate, trust the best horizon sentinel candidate only if it improves memory loss by at least this amount.",
    )
    parser.add_argument(
        "--horizon_gate_min_ratio",
        type=float,
        default=0.0,
        help="For risk_memory_horizon_gate, require (memory_loss - best_loss) / max(best_margin, floor) to exceed this value.",
    )
    parser.add_argument(
        "--horizon_gate_uncertainty_floor",
        type=float,
        default=0.005,
        help="For risk_memory_horizon_gate, denominator floor for normalized horizon gain.",
    )
    parser.add_argument(
        "--confidence_fast_all_min_delta_loss",
        type=float,
        default=-0.05,
        help="For risk_memory_confidence_fast, run all-candidate sentinel only when min-loss and memory agree and current calibration delta loss is at most this value.",
    )
    parser.add_argument(
        "--confidence_lazy_pairwise_min_delta_loss",
        type=float,
        default=-0.025,
        help="For risk_memory_confidence_lazy, skip pairwise sentinel when min-loss calibration delta loss is at least this value.",
    )
    parser.add_argument(
        "--confidence_lazy_pairwise_max_calib_gap",
        type=float,
        default=0.08,
        help="For risk_memory_confidence_lazy, skip pairwise sentinel when memory_delta_loss - min_loss_delta_loss is at most this value.",
    )
    parser.add_argument(
        "--confidence_lazy_pairwise_max_memory_delta_loss",
        type=float,
        default=0.08,
        help="For risk_memory_confidence_lazy, skip pairwise sentinel when memory calibration delta loss is at most this value.",
    )
    parser.add_argument(
        "--sentinel_min_original_max_gap",
        type=float,
        default=0.3,
        help="Skip sentinel probe unless the original combo calibration max loss gap is at least this value.",
    )
    parser.add_argument(
        "--sentinel_min_original_positive_ratio",
        type=float,
        default=0.0,
        help="Skip sentinel probe unless the original combo positive loss-gap ratio is at least this value.",
    )
    parser.add_argument(
        "--meta_min_original_max_gap",
        type=float,
        default=0.5,
        help="Calibration-only meta fallback accepts a proposal only if original max loss gap is at least this value.",
    )
    parser.add_argument(
        "--meta_selected_loss_slack",
        type=float,
        default=0.1,
        help="Calibration-only meta fallback accepts selected combo if selected delta loss <= original delta loss + this slack.",
    )
    parser.add_argument(
        "--meta_max_positive_ratio_increase",
        type=float,
        default=1.0,
        help="Calibration-only meta fallback rejects proposals whose positive loss-gap ratio increases by more than this value.",
    )
    parser.add_argument(
        "--meta_max_gap_increase",
        type=float,
        default=0.0,
        help="Calibration-only meta fallback rejects proposals whose max loss-gap increases by more than this value.",
    )
    parser.add_argument(
        "--meta_min_original_positive_ratio_if_increase",
        type=float,
        default=0.0,
        help="If selected positive ratio increases, require original positive ratio to be at least this value.",
    )
    parser.add_argument("--include_no_rescue", type=str2bool, default=True)
    parser.add_argument("--log_every", type=int, default=1000)
    return parser.parse_args()


def parse_layer_list(raw: str) -> list[int]:
    result: list[int] = []
    seen: set[int] = set()
    for part in raw.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        layer = int(stripped)
        if layer not in seen:
            result.append(layer)
            seen.add(layer)
    return result


def parse_combos(raw: str) -> list[list[int]]:
    combos: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for chunk in raw.split(";"):
        combo = parse_layer_list(chunk)
        if not combo:
            continue
        key = tuple(combo)
        if key not in seen:
            combos.append(combo)
            seen.add(key)
    if not combos:
        raise ValueError("--combos produced no valid layer combos")
    return combos


def format_combo_name(combo: list[int] | tuple[int, ...]) -> str:
    return ",".join(str(layer) for layer in combo)


def write_layer_budget_map(
    path: Path,
    compressed_layers: list[int],
    recent_tokens: int,
    stride: int,
    budget_type: str = "landmark",
    full_heads: int = 8,
) -> None:
    if budget_type == "landmark":
        layer_budget = {
            str(layer): {
                "type": "landmark",
                "recent": recent_tokens,
                "stride": stride,
            }
            for layer in compressed_layers
        }
    elif budget_type == "head_recent":
        layer_budget = {
            str(layer): {
                "type": "head_recent",
                "full_heads": full_heads,
                "recent": recent_tokens,
            }
            for layer in compressed_layers
        }
    else:
        raise ValueError(f"unsupported budget_type: {budget_type}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "default": {"type": "full"},
                "layers": layer_budget,
                "compressed_layers": compressed_layers,
                "metadata": {
                    "kind": "pcic_r_blockwise_layer_budget",
                    "budget_type": budget_type,
                    "full_heads": full_heads if budget_type == "head_recent" else None,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def write_batch_layer_budget_map(
    path: Path,
    compressed_layer_sets: list[list[int]],
    recent_tokens: int,
    stride: int,
    budget_type: str = "landmark",
    full_heads: int = 8,
) -> None:
    row_maps: list[dict[str, Any]] = []
    for compressed_layers in compressed_layer_sets:
        if budget_type == "landmark":
            layer_budget = {
                str(layer): {
                    "type": "landmark",
                    "recent": recent_tokens,
                    "stride": stride,
                }
                for layer in compressed_layers
            }
        elif budget_type == "head_recent":
            layer_budget = {
                str(layer): {
                    "type": "head_recent",
                    "full_heads": full_heads,
                    "recent": recent_tokens,
                }
                for layer in compressed_layers
            }
        else:
            raise ValueError(f"unsupported budget_type: {budget_type}")
        row_maps.append(
            {
                "default": {"type": "full"},
                "layers": layer_budget,
                "compressed_layers": compressed_layers,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "default": {"type": "full"},
                "layers": {},
                "batch_maps": row_maps,
                "metadata": {
                    "kind": "pcic_r_batch_row_layer_budget",
                    "budget_type": budget_type,
                    "full_heads": full_heads if budget_type == "head_recent" else None,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def mode_context(mode: str, layer_budget_map_path: str = ""):
    return attention_mode(
        mode,
        0.02,
        3,
        True,
        0,
        0,
        None,
        layer_budget_map_path=layer_budget_map_path,
    )


def quantile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("quantile requires at least one value")
    clipped_q = min(1.0, max(0.0, q))
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = clipped_q * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def logit_disagreement_score(first_logits: torch.Tensor, second_logits: torch.Tensor, metric: str) -> float:
    first = first_logits.float()
    second = second_logits.float()
    if metric == "l2":
        first_centered = first - first.mean(dim=-1, keepdim=True)
        second_centered = second - second.mean(dim=-1, keepdim=True)
        return float(torch.sqrt(torch.mean((first_centered - second_centered) ** 2)).item())
    if metric == "js":
        first_log_probs = F.log_softmax(first, dim=-1)
        second_log_probs = F.log_softmax(second, dim=-1)
        first_probs = first_log_probs.exp()
        second_probs = second_log_probs.exp()
        mixture_probs = 0.5 * (first_probs + second_probs)
        mixture_log_probs = torch.log(mixture_probs.clamp_min(1e-30))
        first_kl = (first_probs * (first_log_probs - mixture_log_probs)).sum(dim=-1)
        second_kl = (second_probs * (second_log_probs - mixture_log_probs)).sum(dim=-1)
        return float((0.5 * (first_kl + second_kl)).mean().item())
    raise ValueError(f"unknown disagreement metric: {metric}")


def build_disagreement_rescue_rule(
    *,
    chosen_rows: list[dict[str, Any]],
    probe_rows: list[dict[str, Any]],
    risk_quantile: float,
    risk_positive_gap: float,
    risk_rescue_fraction: float,
    metric: str,
    probe_combo: str,
) -> dict[str, Any]:
    probe_by_step = {int(row["step"]): row for row in probe_rows}
    disagreement_rows: list[dict[str, Any]] = []
    for row in chosen_rows:
        probe_row = probe_by_step.get(int(row["step"]))
        if probe_row is None or row.get("logits") is None or probe_row.get("logits") is None:
            continue
        disagreement_rows.append(
            {
                "step": int(row["step"]),
                "loss_gap": float(row["loss_gap"]),
                "disagreement": logit_disagreement_score(row["logits"], probe_row["logits"], metric),
            }
        )
    if not disagreement_rows:
        return {"kind": "none", "reason": "empty_disagreement_calibration", "probe_combo": probe_combo}

    gaps = [float(row["loss_gap"]) for row in disagreement_rows]
    gap_cutoff = max(float(risk_positive_gap), quantile(gaps, risk_quantile))
    risky_rows = [row for row in disagreement_rows if float(row["loss_gap"]) >= gap_cutoff]
    positive_risky_rows = [row for row in risky_rows if float(row["loss_gap"]) > risk_positive_gap]
    if positive_risky_rows:
        risky_rows = positive_risky_rows
    if not risky_rows:
        return {
            "kind": "none",
            "reason": "no_positive_loss_gap",
            "probe_combo": probe_combo,
            "gap_cutoff": gap_cutoff,
            "max_loss_gap": max(gaps),
        }

    safe_rows = [row for row in disagreement_rows if row not in risky_rows]
    risky_mean = sum(float(row["disagreement"]) for row in risky_rows) / max(1, len(risky_rows))
    safe_mean = sum(float(row["disagreement"]) for row in safe_rows) / max(1, len(safe_rows))
    disagreements = [float(row["disagreement"]) for row in disagreement_rows]
    if risky_mean <= safe_mean:
        return {
            "kind": "none",
            "reason": "disagreement_not_aligned",
            "probe_combo": probe_combo,
            "gap_cutoff": gap_cutoff,
            "risky_disagreement_mean": risky_mean,
            "safe_disagreement_mean": safe_mean,
            "max_disagreement": max(disagreements),
            "mean_loss_gap": sum(gaps) / max(1, len(gaps)),
            "max_loss_gap": max(gaps),
        }

    rescue_fraction = min(1.0, max(0.0, risk_rescue_fraction))
    return {
        "kind": "calib_disagreement",
        "metric": metric,
        "probe_combo": probe_combo,
        "disagreement_threshold": quantile(disagreements, 1.0 - rescue_fraction),
        "target_rescue_fraction": rescue_fraction,
        "gap_cutoff": gap_cutoff,
        "risk_token_count": len(risky_rows),
        "calibration_token_count": len(disagreement_rows),
        "risk_fraction": len(risky_rows) / max(1, len(disagreement_rows)),
        "risky_disagreement_mean": risky_mean,
        "safe_disagreement_mean": safe_mean,
        "mean_disagreement": sum(disagreements) / max(1, len(disagreements)),
        "max_disagreement": max(disagreements),
        "mean_loss_gap": sum(gaps) / max(1, len(gaps)),
        "max_loss_gap": max(gaps),
    }


def summarize_token_risk(token_rows: list[dict[str, Any]]) -> dict[str, float]:
    if not token_rows:
        return {
            "risk_mean_loss_gap": 0.0,
            "risk_max_loss_gap": 0.0,
            "risk_positive_ratio": 0.0,
            "risk_positive_mean_gap": 0.0,
        }
    gaps = [float(row["loss_gap"]) for row in token_rows]
    positive_gaps = [gap for gap in gaps if gap > 0.0]
    return {
        "risk_mean_loss_gap": sum(gaps) / len(gaps),
        "risk_max_loss_gap": max(gaps),
        "risk_positive_ratio": len(positive_gaps) / len(gaps),
        "risk_positive_mean_gap": sum(positive_gaps) / max(1, len(positive_gaps)),
    }


def block_risk_score(row: dict[str, Any]) -> tuple[float, float, float, float]:
    max_gap = float(row.get("risk_max_loss_gap", 0.0))
    positive_ratio = float(row.get("risk_positive_ratio", 0.0))
    mean_gap = float(row.get("risk_mean_loss_gap", 0.0))
    loss = float(row.get("loss", 0.0))
    return (max_gap, positive_ratio, max(0.0, mean_gap), loss)


def choose_block_fallback_combo(
    *,
    chosen: dict[str, Any],
    block_rows: list[dict[str, Any]],
    baseline_loss: float,
    safe_delta_loss: float,
    max_gap_threshold: float,
    positive_ratio_threshold: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    chosen_max_gap = float(chosen.get("risk_max_loss_gap", 0.0))
    chosen_positive_ratio = float(chosen.get("risk_positive_ratio", 0.0))
    should_fallback = chosen_max_gap > max_gap_threshold or chosen_positive_ratio > positive_ratio_threshold
    rule: dict[str, Any] = {
        "kind": "block_fallback",
        "triggered": int(should_fallback),
        "original_combo": str(chosen.get("combo", "")),
        "original_risk_max_loss_gap": chosen_max_gap,
        "original_risk_positive_ratio": chosen_positive_ratio,
        "block_risk_max_gap": max_gap_threshold,
        "block_risk_positive_ratio": positive_ratio_threshold,
    }
    if not should_fallback:
        rule["selected_combo"] = str(chosen.get("combo", ""))
        rule["reason"] = "chosen_within_risk_budget"
        return chosen, rule

    candidates = [
        row
        for row in block_rows
        if row["kind"] == "calibration_combo" and str(row.get("combo", "")) != str(chosen.get("combo", ""))
    ]
    safe_candidates = [
        row
        for row in candidates
        if float(row["loss"]) <= baseline_loss + safe_delta_loss
        and float(row.get("risk_max_loss_gap", 0.0)) <= max_gap_threshold
        and float(row.get("risk_positive_ratio", 0.0)) <= positive_ratio_threshold
    ]
    if safe_candidates:
        selected = min(safe_candidates, key=block_risk_score)
        rule["reason"] = "safe_low_risk_candidate"
    elif candidates:
        selected = min(candidates, key=block_risk_score)
        rule["reason"] = "best_available_risk_candidate"
    else:
        selected = chosen
        rule["reason"] = "no_alternative_candidate"
    rule["selected_combo"] = str(selected.get("combo", ""))
    rule["selected_risk_max_loss_gap"] = float(selected.get("risk_max_loss_gap", 0.0))
    rule["selected_risk_positive_ratio"] = float(selected.get("risk_positive_ratio", 0.0))
    rule["selected_delta_loss"] = float(selected.get("delta_loss", 0.0))
    return selected, rule


def choose_adaptive_block_fallback_combo(
    *,
    chosen: dict[str, Any],
    block_rows: list[dict[str, Any]],
    baseline_loss: float,
    loss_slack: float,
    max_gap_improvement: float,
    positive_ratio_improvement: float,
    require_degraded: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    chosen_combo = str(chosen.get("combo", ""))
    chosen_loss = float(chosen.get("loss", 0.0))
    chosen_delta_loss = chosen_loss - baseline_loss
    chosen_max_gap = float(chosen.get("risk_max_loss_gap", 0.0))
    chosen_positive_ratio = float(chosen.get("risk_positive_ratio", 0.0))
    chosen_mean_gap = float(chosen.get("risk_mean_loss_gap", 0.0))
    rule: dict[str, Any] = {
        "kind": "adaptive_block_fallback",
        "triggered": 0,
        "original_combo": chosen_combo,
        "original_delta_loss": chosen_delta_loss,
        "original_risk_max_loss_gap": chosen_max_gap,
        "original_risk_positive_ratio": chosen_positive_ratio,
        "original_risk_mean_loss_gap": chosen_mean_gap,
        "adaptive_loss_slack": loss_slack,
        "adaptive_max_gap_improvement": max_gap_improvement,
        "adaptive_positive_ratio_improvement": positive_ratio_improvement,
        "adaptive_require_degraded": int(require_degraded),
    }
    if require_degraded and chosen_delta_loss <= 0.0:
        rule["selected_combo"] = chosen_combo
        rule["reason"] = "chosen_calibration_not_degraded"
        return chosen, rule

    candidates = [
        row
        for row in block_rows
        if row["kind"] == "calibration_combo" and str(row.get("combo", "")) != chosen_combo
    ]
    eligible: list[dict[str, Any]] = []
    for row in candidates:
        candidate_loss = float(row.get("loss", 0.0))
        candidate_delta_loss = candidate_loss - baseline_loss
        candidate_max_gap = float(row.get("risk_max_loss_gap", 0.0))
        candidate_positive_ratio = float(row.get("risk_positive_ratio", 0.0))
        candidate_mean_gap = float(row.get("risk_mean_loss_gap", 0.0))
        max_gap_gain = chosen_max_gap - candidate_max_gap
        positive_ratio_gain = chosen_positive_ratio - candidate_positive_ratio
        improves_tail = (
            max_gap_gain >= max_gap_improvement
            or positive_ratio_gain >= positive_ratio_improvement
            or (candidate_mean_gap < chosen_mean_gap and candidate_delta_loss < chosen_delta_loss)
        )
        comparable_loss = candidate_delta_loss <= chosen_delta_loss + loss_slack
        if improves_tail and comparable_loss:
            eligible.append(row)

    if not eligible:
        rule["selected_combo"] = chosen_combo
        rule["reason"] = "no_comparable_lower_risk_candidate"
        return chosen, rule

    selected = min(
        eligible,
        key=lambda row: (
            max(0.0, float(row.get("loss", 0.0)) - baseline_loss),
            block_risk_score(row),
        ),
    )
    rule["triggered"] = 1
    rule["reason"] = "comparable_loss_lower_tail_risk"
    rule["selected_combo"] = str(selected.get("combo", ""))
    rule["selected_delta_loss"] = float(selected.get("loss", 0.0)) - baseline_loss
    rule["selected_risk_max_loss_gap"] = float(selected.get("risk_max_loss_gap", 0.0))
    rule["selected_risk_positive_ratio"] = float(selected.get("risk_positive_ratio", 0.0))
    rule["selected_risk_mean_loss_gap"] = float(selected.get("risk_mean_loss_gap", 0.0))
    return selected, rule


def build_rescue_rule(
    *,
    strategy: str,
    token_rows: list[dict[str, Any]],
    risk_quantile: float,
    risk_positive_gap: float,
    risk_rescue_fraction: float,
    fixed_margin: float,
) -> dict[str, Any]:
    if strategy == "none":
        return {"kind": "none"}
    if strategy == "margin":
        return {"kind": "margin", "margin_threshold": fixed_margin}
    if not token_rows:
        return {"kind": "none", "reason": "empty_calibration"}

    gaps = [float(row["loss_gap"]) for row in token_rows]
    gap_cutoff = max(float(risk_positive_gap), quantile(gaps, risk_quantile))
    risky_rows = [row for row in token_rows if float(row["loss_gap"]) >= gap_cutoff]
    positive_risky_rows = [row for row in risky_rows if float(row["loss_gap"]) > risk_positive_gap]
    if positive_risky_rows:
        risky_rows = positive_risky_rows
    if not risky_rows:
        return {
            "kind": "none",
            "reason": "no_positive_loss_gap",
            "gap_cutoff": gap_cutoff,
            "max_loss_gap": max(gaps),
        }

    safe_rows = [row for row in token_rows if row not in risky_rows]
    rescue_fraction = min(1.0, max(0.0, risk_rescue_fraction))
    margins = [float(row["margin"]) for row in token_rows]
    entropies = [float(row["entropy"]) for row in token_rows]
    risky_margin_mean = sum(float(row["margin"]) for row in risky_rows) / max(1, len(risky_rows))
    safe_margin_mean = sum(float(row["margin"]) for row in safe_rows) / max(1, len(safe_rows))
    risky_entropy_mean = sum(float(row["entropy"]) for row in risky_rows) / max(1, len(risky_rows))
    safe_entropy_mean = sum(float(row["entropy"]) for row in safe_rows) / max(1, len(safe_rows))
    margin_aligned = risky_margin_mean < safe_margin_mean
    entropy_aligned = risky_entropy_mean > safe_entropy_mean
    rule: dict[str, Any] = {
        "kind": strategy,
        "gap_cutoff": gap_cutoff,
        "risk_token_count": len(risky_rows),
        "calibration_token_count": len(token_rows),
        "risk_fraction": len(risky_rows) / max(1, len(token_rows)),
        "target_rescue_fraction": rescue_fraction,
        "mean_loss_gap": sum(gaps) / max(1, len(gaps)),
        "max_loss_gap": max(gaps),
        "risky_margin_mean": risky_margin_mean,
        "safe_margin_mean": safe_margin_mean,
        "risky_entropy_mean": risky_entropy_mean,
        "safe_entropy_mean": safe_entropy_mean,
    }
    if strategy in {"calib_margin", "calib_margin_entropy"}:
        if margin_aligned:
            rule["margin_threshold"] = quantile(margins, rescue_fraction)
        else:
            rule["margin_not_aligned"] = True
    if strategy in {"calib_entropy", "calib_margin_entropy"}:
        if entropy_aligned:
            rule["entropy_threshold"] = quantile(entropies, 1.0 - rescue_fraction)
        else:
            rule["entropy_not_aligned"] = True
    if "margin_threshold" not in rule and "entropy_threshold" not in rule:
        rule["kind"] = "none"
        rule["reason"] = "risk_features_not_aligned"
    return rule


def should_rescue_token(rule: dict[str, Any] | None, margin: float, entropy: float | None) -> bool:
    if not rule:
        return False
    kind = str(rule.get("kind", "none"))
    if kind in {
        "none",
        "block_fallback",
        "adaptive_block_fallback",
        "sentinel_block_fallback",
        "calib_meta_fallback",
        "risk_memory_sentinel",
        "risk_memory_sentinel_all",
        "risk_memory_confidence_routed",
        "risk_memory_confidence_fast",
        "risk_memory_confidence_lazy",
        "risk_memory_horizon_gate",
    }:
        return False
    if kind == "margin":
        threshold = rule.get("margin_threshold")
        return threshold is not None and margin < float(threshold)
    if kind == "calib_margin":
        threshold = rule.get("margin_threshold")
        return threshold is not None and margin <= float(threshold)
    if kind == "calib_entropy":
        threshold = rule.get("entropy_threshold")
        return entropy is not None and threshold is not None and entropy >= float(threshold)
    if kind == "calib_margin_entropy":
        margin_threshold = rule.get("margin_threshold")
        entropy_threshold = rule.get("entropy_threshold")
        margin_risky = margin_threshold is not None and margin <= float(margin_threshold)
        entropy_risky = entropy is not None and entropy_threshold is not None and entropy >= float(entropy_threshold)
        return bool(margin_risky or entropy_risky)
    raise ValueError(f"unknown rescue rule kind: {kind}")


def choose_probe_combo(rows: list[dict[str, Any]], chosen_combo: str) -> dict[str, Any] | None:
    candidates = [
        row
        for row in rows
        if row["kind"] == "calibration_combo" and str(row["combo"]) != chosen_combo
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda row: (float(row["loss"]), float(row["seconds"])))


def repeat_batch_value(value: Any, batch_count: int) -> Any:
    if value is None:
        return None
    if torch.is_tensor(value):
        cloned = value.detach().clone()
        if cloned.ndim > 0 and cloned.shape[0] == 1:
            return cloned.repeat_interleave(batch_count, dim=0)
        return cloned
    if isinstance(value, tuple):
        return tuple(repeat_batch_value(item, batch_count) for item in value)
    if isinstance(value, list):
        return [repeat_batch_value(item, batch_count) for item in value]
    if isinstance(value, dict):
        return {key: repeat_batch_value(item, batch_count) for key, item in value.items()}
    to_legacy_cache = getattr(value, "to_legacy_cache", None)
    from_legacy_cache = getattr(type(value), "from_legacy_cache", None)
    if callable(to_legacy_cache) and callable(from_legacy_cache):
        return from_legacy_cache(repeat_batch_value(to_legacy_cache(), batch_count))
    return clone_past_key_values(value)


def slice_batch_value(value: Any, row_idx: int) -> Any:
    if value is None:
        return None
    if torch.is_tensor(value):
        cloned = value.detach()
        if cloned.ndim > 0 and cloned.shape[0] > row_idx:
            return cloned[row_idx : row_idx + 1].clone()
        return cloned.clone()
    if isinstance(value, tuple):
        return tuple(slice_batch_value(item, row_idx) for item in value)
    if isinstance(value, list):
        return [slice_batch_value(item, row_idx) for item in value]
    if isinstance(value, dict):
        return {key: slice_batch_value(item, row_idx) for key, item in value.items()}
    to_legacy_cache = getattr(value, "to_legacy_cache", None)
    from_legacy_cache = getattr(type(value), "from_legacy_cache", None)
    if callable(to_legacy_cache) and callable(from_legacy_cache):
        return from_legacy_cache(slice_batch_value(to_legacy_cache(), row_idx))
    return clone_past_key_values(value)


def concat_batch_values(values: list[Any]) -> Any:
    if not values:
        return None
    first = values[0]
    if first is None:
        return None
    if torch.is_tensor(first):
        return torch.cat([value.detach().clone() for value in values], dim=0)
    if isinstance(first, tuple):
        return tuple(concat_batch_values([value[idx] for value in values]) for idx in range(len(first)))
    if isinstance(first, list):
        return [concat_batch_values([value[idx] for value in values]) for idx in range(len(first))]
    if isinstance(first, dict):
        return {key: concat_batch_values([value[key] for value in values]) for key in first.keys()}
    to_legacy_cache = getattr(first, "to_legacy_cache", None)
    from_legacy_cache = getattr(type(first), "from_legacy_cache", None)
    if callable(to_legacy_cache) and callable(from_legacy_cache):
        return from_legacy_cache(concat_batch_values([value.to_legacy_cache() for value in values]))
    raise TypeError(f"Cannot concatenate batched cache values of type {type(first)!r}")


@torch.inference_mode()
def eval_segment(
    *,
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    start_token: int,
    token_count: int,
    input_device: torch.device,
    initial_past_key_values: Any,
    initial_prev_logits: torch.Tensor,
    mode: str,
    layer_budget_map_path: str = "",
    rescue_margin: float | None = None,
    rescue_rule: dict[str, Any] | None = None,
    return_token_records: bool = False,
    return_logits_records: bool = False,
    log_prefix: str = "",
    log_every: int = 1000,
) -> dict[str, Any]:
    past_key_values = clone_past_key_values(initial_past_key_values)
    prev_logits = initial_prev_logits.detach().clone()
    total_loss = 0.0
    total_count = 0
    rescue_count = 0
    compressed_count = 0
    token_records: list[dict[str, Any]] = []
    started = time.perf_counter()
    end_token = start_token + token_count
    for step, token_index in enumerate(range(start_token, end_token), start=1):
        prev_logits_float = prev_logits.float()
        top2 = torch.topk(prev_logits_float, k=2, dim=-1).values
        margin = float((top2[:, 0] - top2[:, 1]).mean())
        log_probs = F.log_softmax(prev_logits_float, dim=-1)
        probs = log_probs.exp()
        entropy = float((-(probs * log_probs).sum(dim=-1)).mean())
        top1_prob = float(probs.max(dim=-1).values.mean())
        active_rule = rescue_rule
        if active_rule is None and rescue_margin is not None:
            active_rule = {"kind": "margin", "margin_threshold": rescue_margin}
        use_rescue = should_rescue_token(active_rule, margin, entropy)
        active_mode = "baseline" if use_rescue else mode
        active_map = "" if use_rescue else layer_budget_map_path
        if use_rescue:
            rescue_count += 1
        else:
            compressed_count += 1
        if log_every <= 1 or step == 1 or step == token_count or step % log_every == 0:
            print(
                f"{log_prefix} step {step}/{token_count}: token {token_index}, "
                f"mode={active_mode}, margin={margin:.4f}",
                flush=True,
            )
        chunk = input_ids[:, token_index : token_index + 1].to(input_device)
        kwargs: dict[str, Any] = {
            "input_ids": chunk,
            "use_cache": True,
            "return_dict": True,
            "output_attentions": False,
            "output_hidden_states": False,
            "cache_position": torch.arange(token_index, token_index + 1, device=input_device),
        }
        if past_key_values is not None:
            kwargs["past_key_values"] = past_key_values
        with mode_context(active_mode, active_map):
            outputs = model_forward(model, kwargs)
        logits = outputs.logits
        labels = input_ids[:, token_index : token_index + 1].to(input_device)
        shifted_logits = prev_logits.unsqueeze(1)
        loss = F.cross_entropy(
            shifted_logits.reshape(-1, shifted_logits.shape[-1]).float(),
            labels.reshape(-1),
            reduction="sum",
        )
        total_loss += float(loss.item())
        total_count += 1
        if return_token_records:
            token_record = {
                "step": step,
                "token_index": token_index,
                "loss": float(loss.item()),
                "margin": margin,
                "entropy": entropy,
                "top1_prob": top1_prob,
                "active_mode": active_mode,
                "rescued": int(use_rescue),
            }
            if return_logits_records:
                token_record["logits"] = prev_logits.detach().float().cpu()
            token_records.append(token_record)
        prev_logits = logits[:, -1, :].detach()
        past_key_values = outputs.past_key_values
        del outputs, logits, labels, shifted_logits, loss, chunk, prev_logits_float, top2, log_probs, probs
    seconds = time.perf_counter() - started
    mean_loss = total_loss / max(1, total_count)
    return {
        "loss": mean_loss,
        "ppl": math.exp(mean_loss),
        "token_count": total_count,
        "seconds": seconds,
        "rescue_tokens": rescue_count,
        "compressed_tokens": compressed_count,
        "token_records": token_records,
        "final_past_key_values": past_key_values,
        "final_prev_logits": prev_logits,
    }


@torch.inference_mode()
def eval_segment_batched_candidates(
    *,
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    start_token: int,
    token_count: int,
    input_device: torch.device,
    initial_past_key_values: Any,
    initial_prev_logits: torch.Tensor,
    candidate_names: list[str],
    batch_layer_budget_map_path: str,
    log_prefix: str = "",
    log_every: int = 1000,
) -> dict[str, dict[str, Any]]:
    if not candidate_names:
        return {}
    batch_count = len(candidate_names)
    initial_batch = int(initial_prev_logits.shape[0])
    if initial_batch == 1:
        past_key_values = repeat_batch_value(initial_past_key_values, batch_count)
        prev_logits = initial_prev_logits.detach().clone().repeat_interleave(batch_count, dim=0)
    elif initial_batch == batch_count:
        past_key_values = clone_past_key_values(initial_past_key_values)
        prev_logits = initial_prev_logits.detach().clone()
    else:
        raise ValueError(
            "eval_segment_batched_candidates expects initial_prev_logits batch size "
            f"1 or {batch_count}, got {initial_batch}"
        )
    total_losses = torch.zeros((batch_count,), dtype=torch.float64)
    total_count = 0
    started = time.perf_counter()
    end_token = start_token + token_count
    for step, token_index in enumerate(range(start_token, end_token), start=1):
        if log_every <= 1 or step == 1 or step == token_count or step % log_every == 0:
            print(
                f"{log_prefix} batched step {step}/{token_count}: token {token_index}, "
                f"candidates={batch_count}",
                flush=True,
            )
        chunk = input_ids[:, token_index : token_index + 1].to(input_device)
        chunk = chunk.repeat_interleave(batch_count, dim=0)
        kwargs: dict[str, Any] = {
            "input_ids": chunk,
            "use_cache": True,
            "return_dict": True,
            "output_attentions": False,
            "output_hidden_states": False,
            "cache_position": torch.arange(token_index, token_index + 1, device=input_device),
        }
        if past_key_values is not None:
            kwargs["past_key_values"] = past_key_values
        with mode_context("layerbudgetattn", batch_layer_budget_map_path):
            outputs = model_forward(model, kwargs)
        labels = input_ids[:, token_index : token_index + 1].to(input_device)
        labels = labels.repeat_interleave(batch_count, dim=0)
        shifted_logits = prev_logits.unsqueeze(1)
        loss_rows = F.cross_entropy(
            shifted_logits.reshape(-1, shifted_logits.shape[-1]).float(),
            labels.reshape(-1),
            reduction="none",
        ).detach().double().cpu()
        total_losses += loss_rows
        total_count += 1
        prev_logits = outputs.logits[:, -1, :].detach()
        past_key_values = outputs.past_key_values
        del outputs, labels, shifted_logits, loss_rows, chunk
    seconds = time.perf_counter() - started
    results: dict[str, dict[str, Any]] = {}
    for row_idx, candidate_name in enumerate(candidate_names):
        mean_loss = float(total_losses[row_idx].item() / max(1, total_count))
        results[candidate_name] = {
            "loss": mean_loss,
            "ppl": math.exp(mean_loss),
            "token_count": total_count,
            "seconds": seconds,
            "rescue_tokens": 0,
            "compressed_tokens": total_count,
            "token_records": [],
            "final_past_key_values": slice_batch_value(past_key_values, row_idx),
            "final_prev_logits": prev_logits[row_idx : row_idx + 1].detach().clone(),
            "batched_candidate_count": batch_count,
            "batched_total_seconds": seconds,
        }
    return results


@torch.inference_mode()
def eval_segment_disagreement(
    *,
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    start_token: int,
    token_count: int,
    input_device: torch.device,
    initial_past_key_values: Any,
    initial_prev_logits: torch.Tensor,
    primary_map_path: str,
    probe_map_path: str,
    rescue_rule: dict[str, Any],
    log_prefix: str = "",
    log_every: int = 1000,
) -> dict[str, Any]:
    active_past = clone_past_key_values(initial_past_key_values)
    primary_past = clone_past_key_values(initial_past_key_values)
    probe_past = clone_past_key_values(initial_past_key_values)
    active_prev_logits = initial_prev_logits.detach().clone()
    primary_prev_logits = initial_prev_logits.detach().clone()
    probe_prev_logits = initial_prev_logits.detach().clone()
    total_loss = 0.0
    total_count = 0
    rescue_count = 0
    compressed_count = 0
    token_records: list[dict[str, Any]] = []
    started = time.perf_counter()
    metric = str(rescue_rule.get("metric", "js"))
    threshold = rescue_rule.get("disagreement_threshold")
    end_token = start_token + token_count
    for step, token_index in enumerate(range(start_token, end_token), start=1):
        disagreement = logit_disagreement_score(primary_prev_logits, probe_prev_logits, metric)
        use_rescue = threshold is not None and disagreement >= float(threshold)
        active_mode = "baseline" if use_rescue else "layerbudgetattn"
        active_map = "" if use_rescue else primary_map_path
        if use_rescue:
            rescue_count += 1
        else:
            compressed_count += 1

        active_logits_float = active_prev_logits.float()
        top2 = torch.topk(active_logits_float, k=2, dim=-1).values
        margin = float((top2[:, 0] - top2[:, 1]).mean())
        log_probs = F.log_softmax(active_logits_float, dim=-1)
        probs = log_probs.exp()
        entropy = float((-(probs * log_probs).sum(dim=-1)).mean())
        top1_prob = float(probs.max(dim=-1).values.mean())
        if log_every <= 1 or step == 1 or step == token_count or step % log_every == 0:
            print(
                f"{log_prefix} step {step}/{token_count}: token {token_index}, "
                f"mode={active_mode}, disagreement={disagreement:.6f}, threshold={threshold}",
                flush=True,
            )

        chunk = input_ids[:, token_index : token_index + 1].to(input_device)
        labels = input_ids[:, token_index : token_index + 1].to(input_device)
        shifted_logits = active_prev_logits.unsqueeze(1)
        loss = F.cross_entropy(
            shifted_logits.reshape(-1, shifted_logits.shape[-1]).float(),
            labels.reshape(-1),
            reduction="sum",
        )
        total_loss += float(loss.item())
        total_count += 1

        def run_one(past_key_values: Any, mode: str, map_path: str):
            kwargs: dict[str, Any] = {
                "input_ids": chunk,
                "use_cache": True,
                "return_dict": True,
                "output_attentions": False,
                "output_hidden_states": False,
                "cache_position": torch.arange(token_index, token_index + 1, device=input_device),
            }
            if past_key_values is not None:
                kwargs["past_key_values"] = past_key_values
            with mode_context(mode, map_path):
                return model_forward(model, kwargs)

        primary_outputs = run_one(primary_past, "layerbudgetattn", primary_map_path)
        probe_outputs = run_one(probe_past, "layerbudgetattn", probe_map_path)
        active_outputs = run_one(active_past, active_mode, active_map)

        if token_records is not None:
            token_records.append(
                {
                    "step": step,
                    "token_index": token_index,
                    "loss": float(loss.item()),
                    "margin": margin,
                    "entropy": entropy,
                    "top1_prob": top1_prob,
                    "disagreement": disagreement,
                    "active_mode": active_mode,
                    "rescued": int(use_rescue),
                }
            )

        primary_prev_logits = primary_outputs.logits[:, -1, :].detach()
        probe_prev_logits = probe_outputs.logits[:, -1, :].detach()
        active_prev_logits = active_outputs.logits[:, -1, :].detach()
        primary_past = primary_outputs.past_key_values
        probe_past = probe_outputs.past_key_values
        active_past = active_outputs.past_key_values
        del (
            primary_outputs,
            probe_outputs,
            active_outputs,
            labels,
            shifted_logits,
            loss,
            chunk,
            active_logits_float,
            top2,
            log_probs,
            probs,
        )
    seconds = time.perf_counter() - started
    mean_loss = total_loss / max(1, total_count)
    return {
        "loss": mean_loss,
        "ppl": math.exp(mean_loss),
        "token_count": total_count,
        "seconds": seconds,
        "rescue_tokens": rescue_count,
        "compressed_tokens": compressed_count,
        "token_records": token_records,
        "final_past_key_values": active_past,
        "final_prev_logits": active_prev_logits,
    }


def choose_combo(
    rows: list[dict[str, Any]],
    baseline_loss: float,
    safe_delta_loss: float,
    policy: str,
    max_gap_threshold: float = float("inf"),
    positive_ratio_threshold: float = 1.0,
    risk_memory: dict[str, list[dict[str, Any]]] | None = None,
    risk_memory_loss_slack: float = 0.2,
) -> dict[str, Any]:
    candidates = [row for row in rows if row["kind"] == "calibration_combo"]
    def combo_size(row: dict[str, Any]) -> int:
        return len(parse_layer_list(str(row.get("combo", ""))))
    if policy == "min_loss":
        return min(candidates, key=lambda row: (float(row["loss"]), float(row["seconds"])))
    if policy == "risk_pareto":
        safe = [row for row in candidates if float(row["loss"]) <= baseline_loss + safe_delta_loss]
        if safe:
            return min(safe, key=block_risk_score)
        return min(candidates, key=lambda row: (block_risk_score(row), float(row["loss"])))
    if policy == "risk_budget":
        safe = [
            row
            for row in candidates
            if float(row["loss"]) <= baseline_loss + safe_delta_loss
            and float(row.get("risk_max_loss_gap", 0.0)) <= max_gap_threshold
            and float(row.get("risk_positive_ratio", 0.0)) <= positive_ratio_threshold
        ]
        if safe:
            return max(
                safe,
                key=lambda row: (
                    combo_size(row),
                    -float(row.get("risk_max_loss_gap", 0.0)),
                    -float(row.get("risk_positive_ratio", 0.0)),
                    -max(0.0, float(row["loss"]) - baseline_loss),
                ),
            )
        return min(
            candidates,
            key=lambda row: (
                max(0.0, float(row["loss"]) - baseline_loss),
                block_risk_score(row),
                -combo_size(row),
            ),
        )
    if policy == "risk_memory":
        min_loss = min(float(row["loss"]) for row in candidates)
        eligible = [
            row
            for row in candidates
            if float(row["loss"]) <= min_loss + risk_memory_loss_slack
        ]
        memory = risk_memory or {}

        def memory_score(row: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
            combo = str(row.get("combo", ""))
            records = list(memory.get(combo, [])) + [row]
            avg_max_gap = sum(float(record.get("risk_max_loss_gap", 0.0)) for record in records) / len(records)
            worst_max_gap = max(float(record.get("risk_max_loss_gap", 0.0)) for record in records)
            avg_positive_ratio = (
                sum(float(record.get("risk_positive_ratio", 0.0)) for record in records) / len(records)
            )
            avg_delta_loss = sum(float(record.get("delta_loss", 0.0)) for record in records) / len(records)
            return (
                avg_max_gap,
                worst_max_gap,
                avg_positive_ratio,
                avg_delta_loss,
                float(row["loss"]),
                float(row["seconds"]),
            )

        return min(eligible, key=memory_score)
    if policy != "fastest_safe":
        raise ValueError(f"unknown combo selection policy: {policy}")
    safe = [row for row in candidates if float(row["loss"]) <= baseline_loss + safe_delta_loss]
    if safe:
        return min(safe, key=lambda row: (float(row["seconds"]), float(row["loss"])))
    return min(candidates, key=lambda row: (float(row["loss"]), float(row["seconds"])))


def main() -> None:
    args = parse_args()
    if args.attn_implementation and args.attn_implementation != "eager":
        raise ValueError(
            "PCIC-R layerbudget attention requires --attn_implementation eager; "
            f"got {args.attn_implementation!r}. Non-eager kernels bypass the custom attention patch "
            "and produce invalid zero-delta results."
        )
    if (
        args.combo_select_policy
        in {
            "risk_memory_sentinel",
            "risk_memory_sentinel_all",
            "risk_memory_confidence_routed",
            "risk_memory_confidence_fast",
            "risk_memory_confidence_lazy",
            "risk_memory_horizon_gate",
        }
        and args.rescue_strategy != "none"
    ):
        raise ValueError(
            f"{args.combo_select_policy} is a selector-level rescue gate; use --rescue_strategy none "
            "to avoid mixing it with token/block fallback gates."
        )
    combos = parse_combos(args.combos)
    cascade_anchor_combo_names = (
        [format_combo_name(combo) for combo in parse_combos(args.sentinel_cascade_anchor_combos)]
        if args.sentinel_cascade_anchor_combos.strip()
        else []
    )
    output_dir = Path(args.output_dir)
    map_dir = output_dir / "maps"
    batch_map_dir = output_dir / "batch_maps"
    output_dir.mkdir(parents=True, exist_ok=True)
    map_dir.mkdir(parents=True, exist_ok=True)
    batch_map_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    text = read_text_prefix(Path(args.text_path), args.max_chars)
    input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
    required = args.prefill_tokens + args.num_blocks * (args.calibration_tokens + args.eval_tokens_per_block)
    start_offset = max(0, int(args.start_token_offset))
    if input_ids.shape[-1] < start_offset + required:
        raise ValueError(
            f"not enough tokens: need offset {start_offset} + {required}, got {input_ids.shape[-1]}"
        )
    input_ids = input_ids[:, start_offset : start_offset + required]

    load_kwargs: dict[str, Any] = {"trust_remote_code": True, "torch_dtype": dtype}
    if args.device_map:
        load_kwargs["device_map"] = args.device_map
    if args.attn_implementation:
        load_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **load_kwargs)
    model.eval()
    model.config.use_cache = True
    install_qwen3_attention_patch()
    input_device = pick_input_device(model, device)

    budget_maps: dict[tuple[int, ...], Path] = {}
    for combo in combos:
        map_path = map_dir / f"combo_{'_'.join(str(layer) for layer in combo)}_landmark.json"
        write_layer_budget_map(
            map_path,
            combo,
            args.recent_tokens,
            args.landmark_stride,
            args.budget_type,
            args.full_heads,
        )
        budget_maps[tuple(combo)] = map_path

    rows: list[dict[str, Any]] = []
    token_risk_rows: list[dict[str, Any]] = []
    risk_memory_by_combo: dict[str, list[dict[str, Any]]] = {}

    if (
        args.combo_select_policy
        in {
            "risk_memory",
            "risk_memory_sentinel",
            "risk_memory_sentinel_all",
            "risk_memory_confidence_routed",
            "risk_memory_confidence_fast",
            "risk_memory_confidence_lazy",
            "risk_memory_horizon_gate",
        }
        and args.risk_memory_seed_tokens > 0
        and args.risk_memory_use_history
    ):
        seed_tokens = min(int(args.risk_memory_seed_tokens), max(0, args.prefill_tokens - 1))
        if seed_tokens > 0:
            seed_start = args.prefill_tokens - seed_tokens
            print(f"starting risk-memory seed cache at token {seed_start}", flush=True)
            seed_past, seed_logits = prefill_cache(
                model,
                input_ids,
                seed_start,
                args.chunk_size,
                input_device,
            )
            baseline_seed = eval_segment(
                model=model,
                input_ids=input_ids,
                start_token=seed_start,
                token_count=seed_tokens,
                input_device=input_device,
                initial_past_key_values=seed_past,
                initial_prev_logits=seed_logits,
                mode="baseline",
                return_token_records=True,
                log_prefix="risk-memory seed baseline",
                log_every=args.log_every,
            )
            baseline_seed_by_step = {
                int(row["step"]): row for row in baseline_seed["token_records"]
            }
            seed_baseline_row = {
                "kind": "risk_memory_seed_baseline",
                "block": -1,
                "combo": "",
                "loss": baseline_seed["loss"],
                "ppl": baseline_seed["ppl"],
                "seconds": baseline_seed["seconds"],
                "delta_loss": 0.0,
                "delta_ppl": 0.0,
                "rescue_tokens": 0,
                "compressed_tokens": seed_tokens,
            }
            rows.append(seed_baseline_row)
            for combo in combos:
                combo_key = tuple(combo)
                combo_name = ",".join(str(layer) for layer in combo)
                seed_cal = eval_segment(
                    model=model,
                    input_ids=input_ids,
                    start_token=seed_start,
                    token_count=seed_tokens,
                    input_device=input_device,
                    initial_past_key_values=seed_past,
                    initial_prev_logits=seed_logits,
                    mode="layerbudgetattn",
                    layer_budget_map_path=str(budget_maps[combo_key]),
                    return_token_records=True,
                    return_logits_records=False,
                    log_prefix=f"risk-memory seed combo {combo_key}",
                    log_every=args.log_every,
                )
                seed_combo_token_rows: list[dict[str, Any]] = []
                for token_row in seed_cal["token_records"]:
                    baseline_token = baseline_seed_by_step[int(token_row["step"])]
                    risk_row = {
                        "block": -1,
                        "combo": combo_name,
                        "step": int(token_row["step"]),
                        "token_index": int(token_row["token_index"]),
                        "baseline_loss": float(baseline_token["loss"]),
                        "compressed_loss": float(token_row["loss"]),
                        "loss_gap": float(token_row["loss"]) - float(baseline_token["loss"]),
                        "compressed_margin": float(token_row["margin"]),
                        "compressed_entropy": float(token_row["entropy"]),
                        "compressed_top1_prob": float(token_row["top1_prob"]),
                        "baseline_margin": float(baseline_token["margin"]),
                        "baseline_entropy": float(baseline_token["entropy"]),
                        "baseline_top1_prob": float(baseline_token["top1_prob"]),
                    }
                    token_risk_rows.append(risk_row)
                    seed_combo_token_rows.append(
                        {
                            "step": risk_row["step"],
                            "loss_gap": risk_row["loss_gap"],
                            "margin": risk_row["compressed_margin"],
                            "entropy": risk_row["compressed_entropy"],
                            "top1_prob": risk_row["compressed_top1_prob"],
                            "logits": token_row.get("logits"),
                        }
                    )
                risk_summary = summarize_token_risk(seed_combo_token_rows)
                seed_row = {
                    "kind": "risk_memory_seed_combo",
                    "block": -1,
                    "combo": combo_name,
                    "loss": seed_cal["loss"],
                    "ppl": seed_cal["ppl"],
                    "seconds": seed_cal["seconds"],
                    "delta_loss": seed_cal["loss"] - baseline_seed["loss"],
                    "delta_ppl": seed_cal["ppl"] - baseline_seed["ppl"],
                    "rescue_tokens": 0,
                    "compressed_tokens": seed_tokens,
                    **risk_summary,
                }
                rows.append(seed_row)
                risk_memory_by_combo.setdefault(combo_name, []).append(seed_row)

    print("starting shared prefill cache", flush=True)
    initial_past, initial_logits = prefill_cache(
        model,
        input_ids,
        args.prefill_tokens,
        args.chunk_size,
        input_device,
    )

    block_past = initial_past
    block_logits = initial_logits
    cursor = args.prefill_tokens
    total_eval_loss = 0.0
    total_eval_tokens = 0
    total_eval_seconds = 0.0
    total_gate_seconds = 0.0
    total_rescue_tokens = 0
    total_compressed_tokens = 0
    for block_idx in range(args.num_blocks):
        block_rows: list[dict[str, Any]] = []
        calibration_records_by_combo: dict[str, list[dict[str, Any]]] = {}
        baseline_cal = eval_segment(
            model=model,
            input_ids=input_ids,
            start_token=cursor,
            token_count=args.calibration_tokens,
            input_device=input_device,
            initial_past_key_values=block_past,
            initial_prev_logits=block_logits,
            mode="baseline",
            return_token_records=True,
            log_prefix=f"block {block_idx} calibration baseline",
            log_every=args.log_every,
        )
        baseline_token_by_step = {int(row["step"]): row for row in baseline_cal["token_records"]}
        baseline_row = {
            "kind": "calibration_baseline",
            "block": block_idx,
            "combo": "",
            "loss": baseline_cal["loss"],
            "ppl": baseline_cal["ppl"],
            "seconds": baseline_cal["seconds"],
            "delta_loss": 0.0,
            "delta_ppl": 0.0,
            "rescue_tokens": 0,
            "compressed_tokens": 0,
        }
        rows.append(baseline_row)
        block_rows.append(baseline_row)
        for combo in combos:
            combo_key = tuple(combo)
            cal = eval_segment(
                model=model,
                input_ids=input_ids,
                start_token=cursor,
                token_count=args.calibration_tokens,
                input_device=input_device,
                initial_past_key_values=block_past,
                initial_prev_logits=block_logits,
                mode="layerbudgetattn",
                layer_budget_map_path=str(budget_maps[combo_key]),
                return_token_records=True,
                return_logits_records=args.rescue_strategy == "calib_disagreement",
                log_prefix=f"block {block_idx} calibration combo {combo_key}",
                log_every=args.log_every,
            )
            combo_token_rows: list[dict[str, Any]] = []
            combo_name = ",".join(str(layer) for layer in combo)
            for token_row in cal["token_records"]:
                baseline_token = baseline_token_by_step[int(token_row["step"])]
                risk_row = {
                    "block": block_idx,
                    "combo": combo_name,
                    "step": int(token_row["step"]),
                    "token_index": int(token_row["token_index"]),
                    "baseline_loss": float(baseline_token["loss"]),
                    "compressed_loss": float(token_row["loss"]),
                    "loss_gap": float(token_row["loss"]) - float(baseline_token["loss"]),
                    "compressed_margin": float(token_row["margin"]),
                    "compressed_entropy": float(token_row["entropy"]),
                    "compressed_top1_prob": float(token_row["top1_prob"]),
                    "baseline_margin": float(baseline_token["margin"]),
                    "baseline_entropy": float(baseline_token["entropy"]),
                    "baseline_top1_prob": float(baseline_token["top1_prob"]),
                }
                token_risk_rows.append(risk_row)
                combo_token_rows.append(
                    {
                        "step": risk_row["step"],
                        "loss_gap": risk_row["loss_gap"],
                        "margin": risk_row["compressed_margin"],
                        "entropy": risk_row["compressed_entropy"],
                        "top1_prob": risk_row["compressed_top1_prob"],
                        "logits": token_row.get("logits"),
                    }
                )
            calibration_records_by_combo[combo_name] = combo_token_rows
            risk_summary = summarize_token_risk(combo_token_rows)
            row = {
                "kind": "calibration_combo",
                "block": block_idx,
                "combo": combo_name,
                "loss": cal["loss"],
                "ppl": cal["ppl"],
                "seconds": cal["seconds"],
                "delta_loss": cal["loss"] - baseline_cal["loss"],
                "delta_ppl": cal["ppl"] - baseline_cal["ppl"],
                "rescue_tokens": 0,
                "compressed_tokens": args.calibration_tokens,
                **risk_summary,
            }
            rows.append(row)
            block_rows.append(row)

        selector_rule: dict[str, Any] | None = None
        if args.combo_select_policy in {
            "risk_memory_sentinel",
            "risk_memory_sentinel_all",
            "risk_memory_confidence_routed",
            "risk_memory_confidence_fast",
            "risk_memory_confidence_lazy",
            "risk_memory_horizon_gate",
        }:
            min_loss_chosen = choose_combo(
                block_rows,
                float(baseline_cal["loss"]),
                args.safe_delta_loss,
                "min_loss",
                args.block_risk_max_gap,
                args.block_risk_positive_ratio,
                risk_memory_by_combo,
                args.risk_memory_loss_slack,
            )
            memory_chosen = choose_combo(
                block_rows,
                float(baseline_cal["loss"]),
                args.safe_delta_loss,
                "risk_memory",
                args.block_risk_max_gap,
                args.block_risk_positive_ratio,
                risk_memory_by_combo if args.risk_memory_use_history else {},
                args.risk_memory_loss_slack,
            )
            min_loss_combo_name = str(min_loss_chosen.get("combo", ""))
            memory_combo_name = str(memory_chosen.get("combo", ""))
            chosen = memory_chosen
            candidate_rows = [row for row in block_rows if row["kind"] == "calibration_combo"]
            all_candidate_combo_names = [str(row.get("combo", "")) for row in candidate_rows]
            if args.combo_select_policy == "risk_memory_horizon_gate":
                candidate_combo_names = all_candidate_combo_names
                fast_route = "horizon_all_candidates"
            elif args.combo_select_policy in {"risk_memory_confidence_fast", "risk_memory_confidence_lazy"}:
                if min_loss_combo_name != memory_combo_name:
                    min_loss_delta_loss = float(min_loss_chosen.get("delta_loss", 0.0))
                    memory_delta_loss = float(memory_chosen.get("delta_loss", 0.0))
                    lazy_calib_gap = memory_delta_loss - min_loss_delta_loss
                    lazy_skip_pairwise = (
                        args.combo_select_policy == "risk_memory_confidence_lazy"
                        and min_loss_delta_loss >= args.confidence_lazy_pairwise_min_delta_loss
                        and lazy_calib_gap <= args.confidence_lazy_pairwise_max_calib_gap
                        and memory_delta_loss <= args.confidence_lazy_pairwise_max_memory_delta_loss
                    )
                    if lazy_skip_pairwise:
                        candidate_combo_names = [memory_combo_name]
                        fast_route = "lazy_skip_pairwise_memory_anchor"
                    else:
                        candidate_combo_names = list(dict.fromkeys([min_loss_combo_name, memory_combo_name]))
                        fast_route = "pairwise_min_vs_memory"
                elif float(min_loss_chosen.get("delta_loss", 0.0)) <= args.confidence_fast_all_min_delta_loss:
                    candidate_combo_names = all_candidate_combo_names
                    fast_route = "all_candidates_strong_calibration"
                else:
                    candidate_combo_names = [memory_combo_name]
                    fast_route = "skip_sentinel_memory_anchor"
            else:
                candidate_combo_names = all_candidate_combo_names
                fast_route = ""
            if not args.pairwise_candidate_probe:
                candidate_combo_names = [memory_combo_name] if memory_combo_name else []
                fast_route = "no_pairwise_memory_anchor"
            selector_rule = {
                "kind": args.combo_select_policy,
                "triggered": int(
                    (
                        args.combo_select_policy in {
                            "risk_memory_confidence_fast",
                            "risk_memory_confidence_lazy",
                        }
                        and len(candidate_combo_names) > 1
                    )
                    or (
                        args.combo_select_policy
                        not in {"risk_memory_confidence_fast", "risk_memory_confidence_lazy"}
                        and min_loss_combo_name != memory_combo_name
                    )
                    or (
                        args.combo_select_policy
                        in {"risk_memory_sentinel_all", "risk_memory_confidence_routed"}
                        and len(candidate_combo_names) > 1
                    )
                    or (
                        args.combo_select_policy == "risk_memory_horizon_gate"
                        and len(candidate_combo_names) > 1
                    )
                ),
                "reason": "memory_anchor_matches_min_loss"
                if min_loss_combo_name == memory_combo_name
                else "memory_anchor_disagrees_with_min_loss",
                "min_loss_combo": min_loss_combo_name,
                "memory_combo": memory_combo_name,
                "candidate_combos": candidate_combo_names,
                "all_candidate_combos": all_candidate_combo_names,
                "sentinel_cascade_anchor_combos": cascade_anchor_combo_names,
                "fast_route": fast_route,
                "pairwise_candidate_probe": int(args.pairwise_candidate_probe),
                "confidence_fast_all_min_delta_loss": args.confidence_fast_all_min_delta_loss,
                "confidence_lazy_pairwise_min_delta_loss": args.confidence_lazy_pairwise_min_delta_loss,
                "confidence_lazy_pairwise_max_calib_gap": args.confidence_lazy_pairwise_max_calib_gap,
                "confidence_lazy_pairwise_max_memory_delta_loss": args.confidence_lazy_pairwise_max_memory_delta_loss,
                "horizon_gate_min_gain": args.horizon_gate_min_gain,
                "horizon_gate_min_ratio": args.horizon_gate_min_ratio,
                "horizon_gate_uncertainty_floor": args.horizon_gate_uncertainty_floor,
                "risk_memory_use_history": int(args.risk_memory_use_history),
                "selected_combo": memory_combo_name,
                "min_loss_delta_loss": float(min_loss_chosen.get("delta_loss", 0.0)),
                "memory_delta_loss": float(memory_chosen.get("delta_loss", 0.0)),
                "lazy_pairwise_calib_gap": float(memory_chosen.get("delta_loss", 0.0))
                - float(min_loss_chosen.get("delta_loss", 0.0)),
                "min_loss_risk_max_loss_gap": float(min_loss_chosen.get("risk_max_loss_gap", 0.0)),
                "memory_risk_max_loss_gap": float(memory_chosen.get("risk_max_loss_gap", 0.0)),
                "min_loss_risk_positive_ratio": float(min_loss_chosen.get("risk_positive_ratio", 0.0)),
                "memory_risk_positive_ratio": float(memory_chosen.get("risk_positive_ratio", 0.0)),
            }
        else:
            chosen = choose_combo(
                block_rows,
                float(baseline_cal["loss"]),
                args.safe_delta_loss,
                args.combo_select_policy,
                args.block_risk_max_gap,
                args.block_risk_positive_ratio,
                risk_memory_by_combo if args.risk_memory_use_history else {},
                args.risk_memory_loss_slack,
            )
        if args.risk_memory_use_history:
            for row in block_rows:
                if row["kind"] == "calibration_combo":
                    risk_memory_by_combo.setdefault(str(row.get("combo", "")), []).append(row)
        original_chosen = chosen
        block_fallback_rule: dict[str, Any] | None = None
        if args.rescue_strategy in {"block_fallback", "sentinel_block_fallback", "calib_meta_fallback"}:
            chosen, block_fallback_rule = choose_block_fallback_combo(
                chosen=chosen,
                block_rows=block_rows,
                baseline_loss=float(baseline_cal["loss"]),
                safe_delta_loss=args.safe_delta_loss,
                max_gap_threshold=args.block_risk_max_gap,
                positive_ratio_threshold=args.block_risk_positive_ratio,
            )
        elif args.rescue_strategy == "adaptive_block_fallback":
            chosen, block_fallback_rule = choose_adaptive_block_fallback_combo(
                chosen=chosen,
                block_rows=block_rows,
                baseline_loss=float(baseline_cal["loss"]),
                loss_slack=args.adaptive_loss_slack,
                max_gap_improvement=args.adaptive_max_gap_improvement,
                positive_ratio_improvement=args.adaptive_positive_ratio_improvement,
                require_degraded=args.adaptive_require_degraded,
            )
        if args.rescue_strategy == "calib_meta_fallback" and block_fallback_rule is not None:
            original_combo_name = str(block_fallback_rule.get("original_combo", str(original_chosen.get("combo", ""))))
            selected_combo_name = str(block_fallback_rule.get("selected_combo", str(chosen.get("combo", ""))))
            original_row = next(
                (
                    row
                    for row in block_rows
                    if row["kind"] == "calibration_combo" and str(row.get("combo", "")) == original_combo_name
                ),
                original_chosen,
            )
            selected_row = next(
                (
                    row
                    for row in block_rows
                    if row["kind"] == "calibration_combo" and str(row.get("combo", "")) == selected_combo_name
                ),
                chosen,
            )
            original_max_gap = float(original_row.get("risk_max_loss_gap", 0.0))
            original_positive_ratio = float(original_row.get("risk_positive_ratio", 0.0))
            original_delta_loss = float(original_row.get("delta_loss", 0.0))
            selected_max_gap = float(selected_row.get("risk_max_loss_gap", 0.0))
            selected_positive_ratio = float(selected_row.get("risk_positive_ratio", 0.0))
            selected_delta_loss = float(selected_row.get("delta_loss", 0.0))
            max_gap_increase = selected_max_gap - original_max_gap
            max_gap_allowed = max_gap_increase <= args.meta_max_gap_increase
            positive_ratio_increase = selected_positive_ratio - original_positive_ratio
            positive_ratio_allowed = (
                positive_ratio_increase <= args.meta_max_positive_ratio_increase
                and (
                    positive_ratio_increase <= 0.0
                    or original_positive_ratio >= args.meta_min_original_positive_ratio_if_increase
                )
            )
            meta_accept = (
                int(block_fallback_rule.get("triggered", 0)) == 1
                and original_max_gap >= args.meta_min_original_max_gap
                and selected_delta_loss <= original_delta_loss + args.meta_selected_loss_slack
                and max_gap_allowed
                and positive_ratio_allowed
            )
            block_fallback_rule.update(
                {
                    "kind": "calib_meta_fallback",
                    "proposal_triggered": int(block_fallback_rule.get("triggered", 0)),
                    "meta_min_original_max_gap": args.meta_min_original_max_gap,
                    "meta_selected_loss_slack": args.meta_selected_loss_slack,
                    "meta_max_positive_ratio_increase": args.meta_max_positive_ratio_increase,
                    "meta_max_gap_increase": args.meta_max_gap_increase,
                    "meta_min_original_positive_ratio_if_increase": args.meta_min_original_positive_ratio_if_increase,
                    "meta_original_delta_loss": original_delta_loss,
                    "meta_selected_delta_loss": selected_delta_loss,
                    "meta_original_max_gap": original_max_gap,
                    "meta_selected_max_gap": selected_max_gap,
                    "meta_max_gap_increase_observed": max_gap_increase,
                    "meta_max_gap_allowed": int(max_gap_allowed),
                    "meta_original_positive_ratio": original_positive_ratio,
                    "meta_selected_positive_ratio": selected_positive_ratio,
                    "meta_positive_ratio_increase": positive_ratio_increase,
                    "meta_positive_ratio_allowed": int(positive_ratio_allowed),
                    "meta_accepted": int(meta_accept),
                    "triggered": int(meta_accept),
                }
            )
            if not meta_accept:
                chosen = original_chosen
                block_fallback_rule["selected_combo"] = original_combo_name
                block_fallback_rule["reason"] = "calib_meta_rejected_proposal"
            else:
                block_fallback_rule["reason"] = f"{block_fallback_rule.get('reason', '')}+calib_meta_accepted"
        chosen_combo = parse_layer_list(str(chosen["combo"]))
        chosen_map = budget_maps[tuple(chosen_combo)]
        chosen_combo_name = ",".join(str(layer) for layer in chosen_combo)
        probe_combo: list[int] | None = None
        probe_map: Path | None = None
        chosen_token_rows = calibration_records_by_combo.get(chosen_combo_name, [])
        if args.rescue_strategy in {"block_fallback", "adaptive_block_fallback", "sentinel_block_fallback", "calib_meta_fallback"}:
            rescue_rule = block_fallback_rule or {
                "kind": args.rescue_strategy,
                "triggered": 0,
                "reason": "missing_rule",
                "selected_combo": chosen_combo_name,
            }
            if args.rescue_strategy == "sentinel_block_fallback":
                rescue_rule["kind"] = "sentinel_block_fallback"
            if args.rescue_strategy == "calib_meta_fallback":
                rescue_rule["kind"] = "calib_meta_fallback"
        elif args.rescue_strategy == "calib_disagreement":
            probe = choose_probe_combo(block_rows, chosen_combo_name)
            if probe is None:
                rescue_rule = {"kind": "none", "reason": "no_probe_combo"}
            else:
                probe_combo = parse_layer_list(str(probe["combo"]))
                probe_combo_name = ",".join(str(layer) for layer in probe_combo)
                probe_map = budget_maps[tuple(probe_combo)]
                rescue_rule = build_disagreement_rescue_rule(
                    chosen_rows=chosen_token_rows,
                    probe_rows=calibration_records_by_combo.get(probe_combo_name, []),
                    risk_quantile=args.risk_quantile,
                    risk_positive_gap=args.risk_positive_gap,
                    risk_rescue_fraction=args.risk_rescue_fraction,
                    metric=args.disagreement_metric,
                    probe_combo=probe_combo_name,
                )
        else:
            rescue_rule = build_rescue_rule(
                strategy=args.rescue_strategy,
                token_rows=chosen_token_rows,
                risk_quantile=args.risk_quantile,
                risk_positive_gap=args.risk_positive_gap,
                risk_rescue_fraction=args.risk_rescue_fraction,
                fixed_margin=args.rescue_margin,
            )
        if selector_rule is not None:
            rescue_rule = selector_rule
        print(
            f"block {block_idx} chosen combo {tuple(chosen_combo)} rescue_rule={json.dumps(rescue_rule, sort_keys=True)}",
            flush=True,
        )

        advanced = eval_segment(
            model=model,
            input_ids=input_ids,
            start_token=cursor,
            token_count=args.calibration_tokens,
            input_device=input_device,
            initial_past_key_values=block_past,
            initial_prev_logits=block_logits,
            mode="baseline",
            log_prefix=f"block {block_idx} advance calibration",
            log_every=args.log_every,
        )
        eval_start = cursor + args.calibration_tokens
        baseline_eval = eval_segment(
            model=model,
            input_ids=input_ids,
            start_token=eval_start,
            token_count=args.eval_tokens_per_block,
            input_device=input_device,
            initial_past_key_values=advanced["final_past_key_values"],
            initial_prev_logits=advanced["final_prev_logits"],
            mode="baseline",
            log_prefix=f"block {block_idx} eval baseline",
            log_every=args.log_every,
        )
        gate_seconds = 0.0
        pcic_eval: dict[str, Any] | None = None
        if rescue_rule.get("kind") in {
            "risk_memory_sentinel_all",
            "risk_memory_confidence_routed",
            "risk_memory_confidence_fast",
            "risk_memory_confidence_lazy",
            "risk_memory_horizon_gate",
        } and int(rescue_rule.get("triggered", 0)):
            candidate_names = list(dict.fromkeys(str(name) for name in rescue_rule.get("candidate_combos", [])))
            sentinel_tokens = args.sentinel_tokens
            if len(candidate_names) > 2 and args.sentinel_all_tokens > 0:
                sentinel_tokens = args.sentinel_all_tokens
            sentinel_count = min(max(1, sentinel_tokens), args.eval_tokens_per_block)
            initial_sentinel_count = sentinel_count
            if 0 < args.sentinel_cascade_initial_tokens < sentinel_count:
                initial_sentinel_count = min(
                    max(1, args.sentinel_cascade_initial_tokens),
                    args.eval_tokens_per_block,
                )
            sentinel_results: dict[str, dict[str, Any]] = {}
            if args.sentinel_batched_candidates and len(candidate_names) > 1:
                initial_candidate_combos = [parse_layer_list(name) for name in candidate_names]
                batch_map = batch_map_dir / f"block_{block_idx}_initial_{len(candidate_names)}.json"
                write_batch_layer_budget_map(
                    batch_map,
                    initial_candidate_combos,
                    args.recent_tokens,
                    args.landmark_stride,
                    args.budget_type,
                    args.full_heads,
                )
                sentinel_results = eval_segment_batched_candidates(
                    model=model,
                    input_ids=input_ids,
                    start_token=eval_start,
                    token_count=initial_sentinel_count,
                    input_device=input_device,
                    initial_past_key_values=advanced["final_past_key_values"],
                    initial_prev_logits=advanced["final_prev_logits"],
                    candidate_names=candidate_names,
                    batch_layer_budget_map_path=str(batch_map),
                    log_prefix=f"block {block_idx} rm-sentinel-all-batched",
                    log_every=args.log_every,
                )
            else:
                for candidate_name in candidate_names:
                    candidate_combo = parse_layer_list(candidate_name)
                    candidate_map = budget_maps[tuple(candidate_combo)]
                    sentinel_results[candidate_name] = eval_segment(
                        model=model,
                        input_ids=input_ids,
                        start_token=eval_start,
                        token_count=initial_sentinel_count,
                        input_device=input_device,
                        initial_past_key_values=advanced["final_past_key_values"],
                        initial_prev_logits=advanced["final_prev_logits"],
                        mode="layerbudgetattn",
                        layer_budget_map_path=str(candidate_map),
                        rescue_rule={"kind": "none"},
                        log_prefix=f"block {block_idx} rm-sentinel-all {tuple(candidate_combo)}",
                        log_every=args.log_every,
                    )
            memory_combo_name = str(rescue_rule.get("memory_combo", ""))
            min_loss_combo_name = str(rescue_rule.get("min_loss_combo", ""))

            def choose_sentinel_candidate(
                results: dict[str, dict[str, Any]],
            ) -> dict[str, Any]:
                best = min(float(result["loss"]) for result in results.values())
                best_name = min(
                    results,
                    key=lambda name: (
                        float(results[name]["loss"]),
                        0 if name == min_loss_combo_name else 1,
                    ),
                )
                mem_loss = (
                    float(results[memory_combo_name]["loss"])
                    if memory_combo_name in results
                    else float("inf")
                )
                min_loss = (
                    float(results[min_loss_combo_name]["loss"])
                    if min_loss_combo_name in results
                    else float("inf")
                )
                sorted_losses = sorted(float(result["loss"]) for result in results.values())
                margin = (
                    sorted_losses[1] - sorted_losses[0]
                    if len(sorted_losses) > 1
                    else float("inf")
                )
                horizon_gain = mem_loss - best
                horizon_uncertainty = max(
                    float(args.horizon_gate_uncertainty_floor),
                    margin if math.isfinite(margin) else 0.0,
                )
                horizon_gain_ratio = horizon_gain / horizon_uncertainty
                horizon_gate_accept = (
                    horizon_gain >= args.horizon_gate_min_gain
                    and horizon_gain_ratio >= args.horizon_gate_min_ratio
                    and margin >= args.sentinel_all_min_margin
                )
                selected_name: str
                selected_route = "all_confident"
                if rescue_rule.get("kind") == "risk_memory_horizon_gate":
                    if memory_combo_name not in results:
                        selected_name = best_name
                        selected_route = "horizon_gate_no_memory_best"
                    elif horizon_gate_accept:
                        selected_name = best_name
                        selected_route = "horizon_gate_best"
                    else:
                        selected_name = memory_combo_name
                        selected_route = "horizon_gate_memory_anchor"
                elif rescue_rule.get("kind") in {
                    "risk_memory_confidence_routed",
                    "risk_memory_confidence_fast",
                    "risk_memory_confidence_lazy",
                }:
                    if margin >= args.sentinel_all_min_margin:
                        if mem_loss <= best + args.sentinel_loss_slack:
                            selected_name = memory_combo_name
                            selected_route = "all_confident_memory_slack"
                        else:
                            selected_name = best_name
                    else:
                        pairwise_delta = mem_loss - min_loss
                        if pairwise_delta >= args.sentinel_pairwise_min_margin:
                            selected_name = min_loss_combo_name
                            selected_route = "low_conf_pairwise_min_loss"
                        else:
                            selected_name = memory_combo_name
                            selected_route = "low_conf_memory_anchor"
                elif (
                    margin < args.sentinel_all_min_margin
                    and memory_combo_name in results
                ):
                    selected_name = memory_combo_name
                    selected_route = "all_low_conf_memory_anchor"
                elif mem_loss <= best + args.sentinel_loss_slack:
                    selected_name = memory_combo_name
                    selected_route = "all_memory_slack"
                else:
                    selected_name = best_name
                    selected_route = "all_best_candidate"
                return {
                    "selected_combo_name": selected_name,
                    "route": selected_route,
                    "best_loss": best,
                    "memory_loss": mem_loss,
                    "min_loss_loss": min_loss,
                    "best_margin": margin,
                    "pairwise_delta": mem_loss - min_loss,
                    "horizon_best_combo": best_name,
                    "horizon_gain": horizon_gain,
                    "horizon_uncertainty": horizon_uncertainty,
                    "horizon_gain_ratio": horizon_gain_ratio,
                    "horizon_gate_accept": int(horizon_gate_accept),
                }

            selection = choose_sentinel_candidate(sentinel_results)
            cascade_initial_losses = {
                name: float(result["loss"]) for name, result in sentinel_results.items()
            }
            cascade_initial_seconds = {
                name: float(result["seconds"]) for name, result in sentinel_results.items()
            }
            cascade_extended = 0
            cascade_accepted_early = 0
            cascade_accepted_by_anchor_match = 0
            cascade_accepted_by_low_spread = 0
            cascade_skipped_by_anchor_nonpositive_gain = 0
            cascade_initial_route = str(selection["route"])
            cascade_initial_selected_combo = str(selection["selected_combo_name"])
            cascade_initial_best_margin = float(selection["best_margin"])
            cascade_initial_pairwise_delta = float(selection["pairwise_delta"])
            cascade_initial_horizon_gain_ratio = float(selection["horizon_gain_ratio"])
            cascade_extended_candidates: list[str] = []
            cascade_extension_seconds: dict[str, float] = {}
            if initial_sentinel_count < sentinel_count:
                early_margin = (
                    abs(cascade_initial_pairwise_delta)
                    if len(candidate_names) == 2
                    else cascade_initial_best_margin
                )
                anchor_match_accept = (
                    args.sentinel_cascade_anchor_accept_on_match
                    and cascade_initial_selected_combo in cascade_anchor_combo_names
                )
                low_spread_accept = (
                    args.sentinel_cascade_accept_low_spread > 0.0
                    and early_margin <= args.sentinel_cascade_accept_low_spread
                )
                anchor_nonpositive_gain_skip = (
                    args.sentinel_cascade_skip_anchor_nonpositive_gain
                    and cascade_initial_selected_combo in cascade_anchor_combo_names
                    and cascade_initial_horizon_gain_ratio <= 0.0
                )
                if (
                    early_margin >= args.sentinel_cascade_accept_margin
                    or anchor_match_accept
                    or low_spread_accept
                    or anchor_nonpositive_gain_skip
                ):
                    sentinel_count = initial_sentinel_count
                    cascade_accepted_early = 1
                    cascade_accepted_by_anchor_match = int(
                        anchor_match_accept and early_margin < args.sentinel_cascade_accept_margin
                    )
                    cascade_accepted_by_low_spread = int(
                        low_spread_accept
                        and early_margin < args.sentinel_cascade_accept_margin
                        and not anchor_match_accept
                    )
                    cascade_skipped_by_anchor_nonpositive_gain = int(
                        anchor_nonpositive_gain_skip
                        and early_margin < args.sentinel_cascade_accept_margin
                        and not anchor_match_accept
                        and not cascade_accepted_by_low_spread
                    )
                else:
                    extension_tokens = sentinel_count - initial_sentinel_count
                    extended_results: dict[str, dict[str, Any]] = {}
                    if args.sentinel_cascade_extend_topk > 0:
                        extension_candidate_names = [
                            name
                            for name, _ in sorted(
                                sentinel_results.items(),
                                key=lambda item: (
                                    float(item[1]["loss"]),
                                    0 if item[0] == min_loss_combo_name else 1,
                                ),
                            )[: args.sentinel_cascade_extend_topk]
                        ]
                        for anchor_name in (memory_combo_name, min_loss_combo_name):
                            if anchor_name in sentinel_results and anchor_name not in extension_candidate_names:
                                extension_candidate_names.append(anchor_name)
                        for anchor_name in cascade_anchor_combo_names:
                            if anchor_name in sentinel_results and anchor_name not in extension_candidate_names:
                                extension_candidate_names.append(anchor_name)
                    else:
                        extension_candidate_names = list(candidate_names)
                    cascade_extended_candidates = list(extension_candidate_names)
                    if args.sentinel_batched_candidates and len(extension_candidate_names) > 1:
                        extension_candidate_combos = [parse_layer_list(name) for name in extension_candidate_names]
                        batch_map = batch_map_dir / f"block_{block_idx}_extension_{len(extension_candidate_names)}.json"
                        write_batch_layer_budget_map(
                            batch_map,
                            extension_candidate_combos,
                            args.recent_tokens,
                            args.landmark_stride,
                            args.budget_type,
                            args.full_heads,
                        )
                        batched_extension_results = eval_segment_batched_candidates(
                            model=model,
                            input_ids=input_ids,
                            start_token=eval_start + initial_sentinel_count,
                            token_count=extension_tokens,
                            input_device=input_device,
                            initial_past_key_values=concat_batch_values(
                                [
                                    sentinel_results[name]["final_past_key_values"]
                                    for name in extension_candidate_names
                                ]
                            ),
                            initial_prev_logits=torch.cat(
                                [
                                    sentinel_results[name]["final_prev_logits"]
                                    for name in extension_candidate_names
                                ],
                                dim=0,
                            ),
                            candidate_names=extension_candidate_names,
                            batch_layer_budget_map_path=str(batch_map),
                            log_prefix=f"block {block_idx} rm-sentinel-cascade-batched",
                            log_every=args.log_every,
                        )
                        for candidate_name, extension_result in batched_extension_results.items():
                            prefix_result = sentinel_results[candidate_name]
                            cascade_extension_seconds[candidate_name] = float(extension_result["seconds"])
                            total_count = prefix_result["token_count"] + extension_result["token_count"]
                            total_loss = (
                                prefix_result["loss"] * prefix_result["token_count"]
                                + extension_result["loss"] * extension_result["token_count"]
                            ) / total_count
                            combined = dict(extension_result)
                            combined.update(
                                {
                                    "loss": total_loss,
                                    "ppl": math.exp(total_loss),
                                    "token_count": total_count,
                                    "seconds": float(prefix_result["seconds"]) + float(extension_result["seconds"]),
                                }
                            )
                            extended_results[candidate_name] = combined
                    else:
                        for candidate_name in extension_candidate_names:
                            candidate_combo = parse_layer_list(candidate_name)
                            candidate_map = budget_maps[tuple(candidate_combo)]
                            prefix_result = sentinel_results[candidate_name]
                            extension_result = eval_segment(
                                model=model,
                                input_ids=input_ids,
                                start_token=eval_start + initial_sentinel_count,
                                token_count=extension_tokens,
                                input_device=input_device,
                                initial_past_key_values=prefix_result["final_past_key_values"],
                                initial_prev_logits=prefix_result["final_prev_logits"],
                                mode="layerbudgetattn",
                                layer_budget_map_path=str(candidate_map),
                                rescue_rule={"kind": "none"},
                                log_prefix=f"block {block_idx} rm-sentinel-cascade {tuple(candidate_combo)}",
                                log_every=args.log_every,
                            )
                            cascade_extension_seconds[candidate_name] = float(extension_result["seconds"])
                            total_count = prefix_result["token_count"] + extension_result["token_count"]
                            total_loss = (
                                prefix_result["loss"] * prefix_result["token_count"]
                                + extension_result["loss"] * extension_result["token_count"]
                            ) / total_count
                            combined = dict(extension_result)
                            combined.update(
                                {
                                    "loss": total_loss,
                                    "ppl": math.exp(total_loss),
                                    "token_count": total_count,
                                    "seconds": float(prefix_result["seconds"]) + float(extension_result["seconds"]),
                                }
                            )
                            extended_results[candidate_name] = combined
                    sentinel_results = extended_results
                    selection = choose_sentinel_candidate(sentinel_results)
                    cascade_extended = 1
            best_loss = float(selection["best_loss"])
            memory_loss = float(selection["memory_loss"])
            min_loss_loss = float(selection["min_loss_loss"])
            best_margin = float(selection["best_margin"])
            horizon_gain = float(selection["horizon_gain"])
            horizon_uncertainty = float(selection["horizon_uncertainty"])
            horizon_gain_ratio = float(selection["horizon_gain_ratio"])
            route = str(selection["route"])
            selected_combo_name = str(selection["selected_combo_name"])
            selected_combo = parse_layer_list(selected_combo_name)
            selected_map = budget_maps[tuple(selected_combo)]
            selected_sentinel = sentinel_results[selected_combo_name]
            cascade_initial_gate_seconds = sum(
                float(seconds)
                for name, seconds in cascade_initial_seconds.items()
                if name != selected_combo_name
            )
            cascade_extension_gate_seconds = sum(
                float(seconds)
                for name, seconds in cascade_extension_seconds.items()
                if name != selected_combo_name
            )
            gate_seconds = cascade_initial_gate_seconds + cascade_extension_gate_seconds
            chosen_combo = selected_combo
            chosen_map = selected_map
            chosen_combo_name = selected_combo_name
            rescue_rule.update(
                {
                    "sentinel_tokens": sentinel_count,
                    "sentinel_base_tokens": args.sentinel_tokens,
                    "sentinel_all_tokens": args.sentinel_all_tokens,
                    "sentinel_cascade_initial_tokens": args.sentinel_cascade_initial_tokens,
                    "sentinel_cascade_accept_margin": args.sentinel_cascade_accept_margin,
                    "sentinel_cascade_extend_topk": args.sentinel_cascade_extend_topk,
                    "sentinel_cascade_anchor_combos": cascade_anchor_combo_names,
                    "sentinel_cascade_anchor_accept_on_match": int(
                        args.sentinel_cascade_anchor_accept_on_match
                    ),
                    "sentinel_cascade_accept_low_spread": args.sentinel_cascade_accept_low_spread,
                    "sentinel_cascade_skip_anchor_nonpositive_gain": int(
                        args.sentinel_cascade_skip_anchor_nonpositive_gain
                    ),
                    "sentinel_batched_candidates": int(args.sentinel_batched_candidates),
                    "sentinel_cascade_extended_candidates": cascade_extended_candidates,
                    "sentinel_cascade_accepted_early": cascade_accepted_early,
                    "sentinel_cascade_accepted_by_anchor_match": cascade_accepted_by_anchor_match,
                    "sentinel_cascade_accepted_by_low_spread": cascade_accepted_by_low_spread,
                    "sentinel_cascade_skipped_by_anchor_nonpositive_gain": (
                        cascade_skipped_by_anchor_nonpositive_gain
                    ),
                    "sentinel_cascade_extended": cascade_extended,
                    "sentinel_cascade_initial_route": cascade_initial_route,
                    "sentinel_cascade_initial_selected_combo": cascade_initial_selected_combo,
                    "sentinel_cascade_initial_best_margin": cascade_initial_best_margin,
                    "sentinel_cascade_initial_pairwise_delta": cascade_initial_pairwise_delta,
                    "sentinel_cascade_initial_horizon_gain_ratio": cascade_initial_horizon_gain_ratio,
                    "sentinel_cascade_initial_losses": cascade_initial_losses,
                    "sentinel_cascade_initial_seconds": cascade_initial_seconds,
                    "sentinel_loss_slack": args.sentinel_loss_slack,
                    "sentinel_all_losses": {
                        name: float(result["loss"]) for name, result in sentinel_results.items()
                    },
                    "sentinel_all_seconds": {
                        name: float(result["seconds"]) for name, result in sentinel_results.items()
                    },
                    "sentinel_best_loss": best_loss,
                    "sentinel_best_margin": best_margin,
                    "sentinel_horizon_best_combo": selection["horizon_best_combo"],
                    "sentinel_horizon_gain": horizon_gain,
                    "sentinel_horizon_uncertainty": horizon_uncertainty,
                    "sentinel_horizon_gain_ratio": horizon_gain_ratio,
                    "sentinel_horizon_gate_accept": int(selection["horizon_gate_accept"]),
                    "horizon_gate_min_gain": args.horizon_gate_min_gain,
                    "horizon_gate_min_ratio": args.horizon_gate_min_ratio,
                    "horizon_gate_uncertainty_floor": args.horizon_gate_uncertainty_floor,
                    "sentinel_all_min_margin": args.sentinel_all_min_margin,
                    "sentinel_memory_loss": memory_loss,
                    "sentinel_min_loss_loss": min_loss_loss,
                    "sentinel_pairwise_delta_loss": memory_loss - min_loss_loss,
                    "sentinel_pairwise_min_margin": args.sentinel_pairwise_min_margin,
                    "sentinel_route": route,
                    "sentinel_gate_seconds": gate_seconds,
                    "sentinel_cascade_initial_gate_seconds": cascade_initial_gate_seconds,
                    "sentinel_cascade_extension_gate_seconds": cascade_extension_gate_seconds,
                    "sentinel_cascade_extension_seconds": cascade_extension_seconds,
                    "sentinel_selected_prefix_seconds": selected_sentinel["seconds"],
                    "sentinel_reused_prefix": 1,
                    "sentinel_memory_selected": int(selected_combo_name == memory_combo_name),
                    "selected_combo": selected_combo_name,
                    "reason": "sentinel_all_selected_candidate",
                }
            )
            remainder_tokens = args.eval_tokens_per_block - sentinel_count
            if remainder_tokens > 0:
                remainder_eval = eval_segment(
                    model=model,
                    input_ids=input_ids,
                    start_token=eval_start + sentinel_count,
                    token_count=remainder_tokens,
                    input_device=input_device,
                    initial_past_key_values=selected_sentinel["final_past_key_values"],
                    initial_prev_logits=selected_sentinel["final_prev_logits"],
                    mode="layerbudgetattn",
                    layer_budget_map_path=str(chosen_map),
                    rescue_rule={"kind": "none"},
                    log_prefix=f"block {block_idx} eval rm-sentinel-all remainder {tuple(chosen_combo)}",
                    log_every=args.log_every,
                )
                total_loss_sum = (
                    selected_sentinel["loss"] * selected_sentinel["token_count"]
                    + remainder_eval["loss"] * remainder_eval["token_count"]
                )
                total_count = selected_sentinel["token_count"] + remainder_eval["token_count"]
                pcic_eval = {
                    "loss": total_loss_sum / max(1, total_count),
                    "ppl": math.exp(total_loss_sum / max(1, total_count)),
                    "token_count": total_count,
                    "seconds": float(selected_sentinel["seconds"]) + float(remainder_eval["seconds"]),
                    "rescue_tokens": selected_sentinel["rescue_tokens"] + remainder_eval["rescue_tokens"],
                    "compressed_tokens": selected_sentinel["compressed_tokens"] + remainder_eval["compressed_tokens"],
                    "token_records": [],
                    "final_past_key_values": remainder_eval["final_past_key_values"],
                    "final_prev_logits": remainder_eval["final_prev_logits"],
                }
            else:
                pcic_eval = selected_sentinel
        if rescue_rule.get("kind") == "risk_memory_sentinel" and int(rescue_rule.get("triggered", 0)):
            sentinel_count = min(max(1, args.sentinel_tokens), args.eval_tokens_per_block)
            min_loss_combo = parse_layer_list(str(rescue_rule["min_loss_combo"]))
            memory_combo = parse_layer_list(str(rescue_rule["memory_combo"]))
            min_loss_map = budget_maps[tuple(min_loss_combo)]
            memory_map = budget_maps[tuple(memory_combo)]
            min_loss_sentinel = eval_segment(
                model=model,
                input_ids=input_ids,
                start_token=eval_start,
                token_count=sentinel_count,
                input_device=input_device,
                initial_past_key_values=advanced["final_past_key_values"],
                initial_prev_logits=advanced["final_prev_logits"],
                mode="layerbudgetattn",
                layer_budget_map_path=str(min_loss_map),
                rescue_rule={"kind": "none"},
                log_prefix=f"block {block_idx} rm-sentinel min-loss {tuple(min_loss_combo)}",
                log_every=args.log_every,
            )
            memory_sentinel = eval_segment(
                model=model,
                input_ids=input_ids,
                start_token=eval_start,
                token_count=sentinel_count,
                input_device=input_device,
                initial_past_key_values=advanced["final_past_key_values"],
                initial_prev_logits=advanced["final_prev_logits"],
                mode="layerbudgetattn",
                layer_budget_map_path=str(memory_map),
                rescue_rule={"kind": "none"},
                log_prefix=f"block {block_idx} rm-sentinel memory {tuple(memory_combo)}",
                log_every=args.log_every,
            )
            memory_accept = memory_sentinel["loss"] <= min_loss_sentinel["loss"] + args.sentinel_loss_slack
            selected_sentinel = memory_sentinel if memory_accept else min_loss_sentinel
            rejected_sentinel = min_loss_sentinel if memory_accept else memory_sentinel
            selected_combo = memory_combo if memory_accept else min_loss_combo
            selected_map = memory_map if memory_accept else min_loss_map
            selected_combo_name = ",".join(str(layer) for layer in selected_combo)
            gate_seconds = float(rejected_sentinel["seconds"])
            chosen_combo = selected_combo
            chosen_map = selected_map
            chosen_combo_name = selected_combo_name
            rescue_rule.update(
                {
                    "sentinel_tokens": sentinel_count,
                    "sentinel_loss_slack": args.sentinel_loss_slack,
                    "sentinel_min_loss_loss": min_loss_sentinel["loss"],
                    "sentinel_memory_loss": memory_sentinel["loss"],
                    "sentinel_min_loss_seconds": min_loss_sentinel["seconds"],
                    "sentinel_memory_seconds": memory_sentinel["seconds"],
                    "sentinel_gate_seconds": gate_seconds,
                    "sentinel_selected_prefix_seconds": selected_sentinel["seconds"],
                    "sentinel_rejected_probe_seconds": rejected_sentinel["seconds"],
                    "sentinel_reused_prefix": 1,
                    "sentinel_memory_delta_loss": memory_sentinel["loss"] - min_loss_sentinel["loss"],
                    "sentinel_memory_selected": int(memory_accept),
                    "selected_combo": selected_combo_name,
                    "reason": "sentinel_selected_memory_anchor" if memory_accept else "sentinel_selected_min_loss",
                }
            )
            remainder_tokens = args.eval_tokens_per_block - sentinel_count
            if remainder_tokens > 0:
                remainder_eval = eval_segment(
                    model=model,
                    input_ids=input_ids,
                    start_token=eval_start + sentinel_count,
                    token_count=remainder_tokens,
                    input_device=input_device,
                    initial_past_key_values=selected_sentinel["final_past_key_values"],
                    initial_prev_logits=selected_sentinel["final_prev_logits"],
                    mode="layerbudgetattn",
                    layer_budget_map_path=str(chosen_map),
                    rescue_rule={"kind": "none"},
                    log_prefix=f"block {block_idx} eval rm-sentinel remainder {tuple(chosen_combo)}",
                    log_every=args.log_every,
                )
                total_loss_sum = (
                    selected_sentinel["loss"] * selected_sentinel["token_count"]
                    + remainder_eval["loss"] * remainder_eval["token_count"]
                )
                total_count = selected_sentinel["token_count"] + remainder_eval["token_count"]
                pcic_eval = {
                    "loss": total_loss_sum / max(1, total_count),
                    "ppl": math.exp(total_loss_sum / max(1, total_count)),
                    "token_count": total_count,
                    "seconds": float(selected_sentinel["seconds"]) + float(remainder_eval["seconds"]),
                    "rescue_tokens": selected_sentinel["rescue_tokens"] + remainder_eval["rescue_tokens"],
                    "compressed_tokens": selected_sentinel["compressed_tokens"] + remainder_eval["compressed_tokens"],
                    "token_records": [],
                    "final_past_key_values": remainder_eval["final_past_key_values"],
                    "final_prev_logits": remainder_eval["final_prev_logits"],
                }
            else:
                pcic_eval = selected_sentinel
        if args.rescue_strategy == "sentinel_block_fallback" and int(rescue_rule.get("triggered", 0)):
            original_max_gap = float(rescue_rule.get("original_risk_max_loss_gap", 0.0))
            original_positive_ratio = float(rescue_rule.get("original_risk_positive_ratio", 0.0))
            sentinel_should_probe = (
                original_max_gap >= args.sentinel_min_original_max_gap
                and original_positive_ratio >= args.sentinel_min_original_positive_ratio
            )
            rescue_rule.update(
                {
                    "sentinel_min_original_max_gap": args.sentinel_min_original_max_gap,
                    "sentinel_min_original_positive_ratio": args.sentinel_min_original_positive_ratio,
                    "sentinel_probe_allowed": int(sentinel_should_probe),
                }
            )
            if not sentinel_should_probe:
                original_combo = parse_layer_list(str(original_chosen["combo"]))
                chosen_combo = original_combo
                chosen_map = budget_maps[tuple(original_combo)]
                chosen_combo_name = ",".join(str(layer) for layer in original_combo)
                rescue_rule["selected_combo"] = chosen_combo_name
                rescue_rule["triggered"] = 0
                rescue_rule["reason"] = "sentinel_skipped_low_original_risk"
        if (
            args.rescue_strategy == "sentinel_block_fallback"
            and int(rescue_rule.get("triggered", 0))
            and int(rescue_rule.get("sentinel_probe_allowed", 1))
        ):
            sentinel_count = min(max(1, args.sentinel_tokens), args.eval_tokens_per_block)
            original_combo = parse_layer_list(str(original_chosen["combo"]))
            original_combo_name = ",".join(str(layer) for layer in original_combo)
            original_map = budget_maps[tuple(original_combo)]
            proposed_combo = chosen_combo
            proposed_map = chosen_map
            original_sentinel = eval_segment(
                model=model,
                input_ids=input_ids,
                start_token=eval_start,
                token_count=sentinel_count,
                input_device=input_device,
                initial_past_key_values=advanced["final_past_key_values"],
                initial_prev_logits=advanced["final_prev_logits"],
                mode="layerbudgetattn",
                layer_budget_map_path=str(original_map),
                rescue_rule={"kind": "none"},
                log_prefix=f"block {block_idx} sentinel original {tuple(original_combo)}",
                log_every=args.log_every,
            )
            proposed_sentinel = eval_segment(
                model=model,
                input_ids=input_ids,
                start_token=eval_start,
                token_count=sentinel_count,
                input_device=input_device,
                initial_past_key_values=advanced["final_past_key_values"],
                initial_prev_logits=advanced["final_prev_logits"],
                mode="layerbudgetattn",
                layer_budget_map_path=str(proposed_map),
                rescue_rule={"kind": "none"},
                log_prefix=f"block {block_idx} sentinel proposed {tuple(proposed_combo)}",
                log_every=args.log_every,
            )
            sentinel_delta_loss = proposed_sentinel["loss"] - original_sentinel["loss"]
            sentinel_accept = sentinel_delta_loss <= args.sentinel_loss_slack
            selected_sentinel = proposed_sentinel if sentinel_accept else original_sentinel
            rejected_sentinel = original_sentinel if sentinel_accept else proposed_sentinel
            gate_seconds = float(rejected_sentinel["seconds"])
            rescue_rule.update(
                {
                    "proposal_triggered": 1,
                    "sentinel_tokens": sentinel_count,
                    "sentinel_loss_slack": args.sentinel_loss_slack,
                    "sentinel_original_combo": original_combo_name,
                    "sentinel_proposed_combo": ",".join(str(layer) for layer in proposed_combo),
                    "sentinel_original_loss": original_sentinel["loss"],
                    "sentinel_proposed_loss": proposed_sentinel["loss"],
                    "sentinel_original_seconds": original_sentinel["seconds"],
                    "sentinel_proposed_seconds": proposed_sentinel["seconds"],
                    "sentinel_gate_seconds": gate_seconds,
                    "sentinel_selected_prefix_seconds": selected_sentinel["seconds"],
                    "sentinel_rejected_probe_seconds": rejected_sentinel["seconds"],
                    "sentinel_reused_prefix": 1,
                    "sentinel_delta_loss": sentinel_delta_loss,
                    "sentinel_accepted": int(sentinel_accept),
                    "triggered": int(sentinel_accept),
                }
            )
            if not sentinel_accept:
                chosen_combo = original_combo
                chosen_map = original_map
                chosen_combo_name = original_combo_name
                rescue_rule["selected_combo"] = original_combo_name
                rescue_rule["reason"] = "sentinel_rejected_proposal"
            else:
                rescue_rule["reason"] = f"{rescue_rule.get('reason', '')}+sentinel_accepted"
            remainder_tokens = args.eval_tokens_per_block - sentinel_count
            if remainder_tokens > 0:
                remainder_eval = eval_segment(
                    model=model,
                    input_ids=input_ids,
                    start_token=eval_start + sentinel_count,
                    token_count=remainder_tokens,
                    input_device=input_device,
                    initial_past_key_values=selected_sentinel["final_past_key_values"],
                    initial_prev_logits=selected_sentinel["final_prev_logits"],
                    mode="layerbudgetattn",
                    layer_budget_map_path=str(chosen_map),
                    rescue_rule={"kind": "none"},
                    log_prefix=f"block {block_idx} eval pcic-r remainder {tuple(chosen_combo)}",
                    log_every=args.log_every,
                )
                total_loss_sum = (
                    selected_sentinel["loss"] * selected_sentinel["token_count"]
                    + remainder_eval["loss"] * remainder_eval["token_count"]
                )
                total_count = selected_sentinel["token_count"] + remainder_eval["token_count"]
                pcic_eval = {
                    "loss": total_loss_sum / max(1, total_count),
                    "ppl": math.exp(total_loss_sum / max(1, total_count)),
                    "token_count": total_count,
                    "seconds": float(selected_sentinel["seconds"]) + float(remainder_eval["seconds"]),
                    "rescue_tokens": selected_sentinel["rescue_tokens"] + remainder_eval["rescue_tokens"],
                    "compressed_tokens": selected_sentinel["compressed_tokens"] + remainder_eval["compressed_tokens"],
                    "token_records": [],
                    "final_past_key_values": remainder_eval["final_past_key_values"],
                    "final_prev_logits": remainder_eval["final_prev_logits"],
                }
            else:
                pcic_eval = selected_sentinel
        if pcic_eval is None and rescue_rule.get("kind") == "calib_disagreement" and probe_map is not None:
            pcic_eval = eval_segment_disagreement(
                model=model,
                input_ids=input_ids,
                start_token=eval_start,
                token_count=args.eval_tokens_per_block,
                input_device=input_device,
                initial_past_key_values=advanced["final_past_key_values"],
                initial_prev_logits=advanced["final_prev_logits"],
                primary_map_path=str(chosen_map),
                probe_map_path=str(probe_map),
                rescue_rule=rescue_rule,
                log_prefix=f"block {block_idx} eval pcic-r2 {tuple(chosen_combo)} probe {tuple(probe_combo or [])}",
                log_every=args.log_every,
            )
        elif pcic_eval is None:
            pcic_eval = eval_segment(
                model=model,
                input_ids=input_ids,
                start_token=eval_start,
                token_count=args.eval_tokens_per_block,
                input_device=input_device,
                initial_past_key_values=advanced["final_past_key_values"],
                initial_prev_logits=advanced["final_prev_logits"],
                mode="layerbudgetattn",
                layer_budget_map_path=str(chosen_map),
                rescue_rule=rescue_rule,
                log_prefix=f"block {block_idx} eval pcic-r {tuple(chosen_combo)}",
                log_every=args.log_every,
            )
        row = {
            "kind": "pcic_r_eval",
            "block": block_idx,
            "start_token_offset": start_offset,
            "combo": ",".join(str(layer) for layer in chosen_combo),
            "loss": pcic_eval["loss"],
            "ppl": pcic_eval["ppl"],
            "seconds": pcic_eval["seconds"],
            "delta_loss": pcic_eval["loss"] - baseline_eval["loss"],
            "delta_ppl": pcic_eval["ppl"] - baseline_eval["ppl"],
            "baseline_loss": baseline_eval["loss"],
            "baseline_ppl": baseline_eval["ppl"],
            "baseline_seconds": baseline_eval["seconds"],
            "gate_seconds": gate_seconds,
            "method_seconds": pcic_eval["seconds"] + gate_seconds,
            "rescue_tokens": pcic_eval["rescue_tokens"],
            "compressed_tokens": pcic_eval["compressed_tokens"],
            "rescue_strategy": args.rescue_strategy,
            "risk_quantile": args.risk_quantile,
            "risk_positive_gap": args.risk_positive_gap,
            "risk_rescue_fraction": args.risk_rescue_fraction,
            "disagreement_metric": args.disagreement_metric,
            "combo_select_policy": args.combo_select_policy,
            "risk_memory_loss_slack": args.risk_memory_loss_slack,
            "risk_memory_seed_tokens": args.risk_memory_seed_tokens,
            "block_risk_max_gap": args.block_risk_max_gap,
            "block_risk_positive_ratio": args.block_risk_positive_ratio,
            "adaptive_loss_slack": args.adaptive_loss_slack,
            "adaptive_max_gap_improvement": args.adaptive_max_gap_improvement,
            "adaptive_positive_ratio_improvement": args.adaptive_positive_ratio_improvement,
            "adaptive_require_degraded": int(args.adaptive_require_degraded),
            "sentinel_tokens": args.sentinel_tokens,
            "sentinel_all_tokens": args.sentinel_all_tokens,
            "sentinel_cascade_initial_tokens": args.sentinel_cascade_initial_tokens,
            "sentinel_cascade_accept_margin": args.sentinel_cascade_accept_margin,
            "sentinel_cascade_extend_topk": args.sentinel_cascade_extend_topk,
            "sentinel_cascade_anchor_combos": args.sentinel_cascade_anchor_combos,
            "sentinel_cascade_anchor_accept_on_match": int(
                args.sentinel_cascade_anchor_accept_on_match
            ),
            "sentinel_cascade_accept_low_spread": args.sentinel_cascade_accept_low_spread,
            "sentinel_cascade_skip_anchor_nonpositive_gain": int(
                args.sentinel_cascade_skip_anchor_nonpositive_gain
            ),
            "sentinel_batched_candidates": int(args.sentinel_batched_candidates),
            "sentinel_loss_slack": args.sentinel_loss_slack,
            "sentinel_all_min_margin": args.sentinel_all_min_margin,
            "sentinel_pairwise_min_margin": args.sentinel_pairwise_min_margin,
            "horizon_gate_min_gain": args.horizon_gate_min_gain,
            "horizon_gate_min_ratio": args.horizon_gate_min_ratio,
            "horizon_gate_uncertainty_floor": args.horizon_gate_uncertainty_floor,
            "confidence_fast_all_min_delta_loss": args.confidence_fast_all_min_delta_loss,
            "confidence_lazy_pairwise_min_delta_loss": args.confidence_lazy_pairwise_min_delta_loss,
            "confidence_lazy_pairwise_max_calib_gap": args.confidence_lazy_pairwise_max_calib_gap,
            "confidence_lazy_pairwise_max_memory_delta_loss": args.confidence_lazy_pairwise_max_memory_delta_loss,
            "sentinel_min_original_max_gap": args.sentinel_min_original_max_gap,
            "sentinel_min_original_positive_ratio": args.sentinel_min_original_positive_ratio,
            "meta_min_original_max_gap": args.meta_min_original_max_gap,
            "meta_selected_loss_slack": args.meta_selected_loss_slack,
            "meta_max_positive_ratio_increase": args.meta_max_positive_ratio_increase,
            "meta_max_gap_increase": args.meta_max_gap_increase,
            "meta_min_original_positive_ratio_if_increase": args.meta_min_original_positive_ratio_if_increase,
            "probe_combo": ",".join(str(layer) for layer in (probe_combo or [])),
            "rescue_rule": json.dumps(rescue_rule, sort_keys=True),
        }
        rows.append(row)
        total_eval_loss += pcic_eval["loss"] * pcic_eval["token_count"]
        total_eval_tokens += pcic_eval["token_count"]
        total_eval_seconds += pcic_eval["seconds"]
        total_gate_seconds += gate_seconds
        total_rescue_tokens += pcic_eval["rescue_tokens"]
        total_compressed_tokens += pcic_eval["compressed_tokens"]

        block_past = baseline_eval["final_past_key_values"]
        block_logits = baseline_eval["final_prev_logits"]
        cursor += args.calibration_tokens + args.eval_tokens_per_block

    csv_path = output_dir / "pcic_r_blockwise_results.csv"
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    risk_csv_path = output_dir / "pcic_r_calibration_token_risk.csv"
    if token_risk_rows:
        risk_fieldnames = sorted({key for row in token_risk_rows for key in row.keys()})
        with risk_csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=risk_fieldnames)
            writer.writeheader()
            writer.writerows(token_risk_rows)

    summary_path = output_dir / "summary.md"
    mean_loss = total_eval_loss / max(1, total_eval_tokens)
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("# PCIC-R blockwise experiment\n\n")
        handle.write(f"- text: `{args.text_path}`\n")
        handle.write(f"- start_token_offset: `{start_offset}`\n")
        handle.write(f"- prefill: `{args.prefill_tokens}`\n")
        handle.write(f"- blocks: `{args.num_blocks}`\n")
        handle.write(f"- calibration/eval per block: `{args.calibration_tokens}/{args.eval_tokens_per_block}`\n")
        handle.write(f"- budget_type: `{args.budget_type}`\n")
        handle.write(f"- full_heads: `{args.full_heads}`\n")
        handle.write(f"- recent_tokens: `{args.recent_tokens}`\n")
        handle.write(f"- landmark_stride: `{args.landmark_stride}`\n")
        handle.write(f"- rescue_margin: `{args.rescue_margin}`\n")
        handle.write(f"- rescue_strategy: `{args.rescue_strategy}`\n")
        handle.write(f"- risk_quantile: `{args.risk_quantile}`\n")
        handle.write(f"- risk_positive_gap: `{args.risk_positive_gap}`\n")
        handle.write(f"- risk_rescue_fraction: `{args.risk_rescue_fraction}`\n")
        handle.write(f"- disagreement_metric: `{args.disagreement_metric}`\n")
        handle.write(f"- combo_select_policy: `{args.combo_select_policy}`\n")
        handle.write(f"- risk_memory_loss_slack: `{args.risk_memory_loss_slack}`\n")
        handle.write(f"- risk_memory_seed_tokens: `{args.risk_memory_seed_tokens}`\n")
        handle.write(f"- block_risk_max_gap: `{args.block_risk_max_gap}`\n")
        handle.write(f"- block_risk_positive_ratio: `{args.block_risk_positive_ratio}`\n")
        handle.write(f"- adaptive_loss_slack: `{args.adaptive_loss_slack}`\n")
        handle.write(f"- adaptive_max_gap_improvement: `{args.adaptive_max_gap_improvement}`\n")
        handle.write(f"- adaptive_positive_ratio_improvement: `{args.adaptive_positive_ratio_improvement}`\n")
        handle.write(f"- adaptive_require_degraded: `{args.adaptive_require_degraded}`\n")
        handle.write(f"- sentinel_tokens: `{args.sentinel_tokens}`\n")
        handle.write(f"- sentinel_all_tokens: `{args.sentinel_all_tokens}`\n")
        handle.write(f"- sentinel_cascade_initial_tokens: `{args.sentinel_cascade_initial_tokens}`\n")
        handle.write(f"- sentinel_cascade_accept_margin: `{args.sentinel_cascade_accept_margin}`\n")
        handle.write(f"- sentinel_cascade_extend_topk: `{args.sentinel_cascade_extend_topk}`\n")
        handle.write(f"- sentinel_cascade_anchor_combos: `{args.sentinel_cascade_anchor_combos}`\n")
        handle.write(
            f"- sentinel_cascade_anchor_accept_on_match: `{int(args.sentinel_cascade_anchor_accept_on_match)}`\n"
        )
        handle.write(f"- sentinel_cascade_accept_low_spread: `{args.sentinel_cascade_accept_low_spread}`\n")
        handle.write(
            f"- sentinel_cascade_skip_anchor_nonpositive_gain: `{int(args.sentinel_cascade_skip_anchor_nonpositive_gain)}`\n"
        )
        handle.write(f"- sentinel_batched_candidates: `{int(args.sentinel_batched_candidates)}`\n")
        handle.write(f"- sentinel_loss_slack: `{args.sentinel_loss_slack}`\n")
        handle.write(f"- sentinel_all_min_margin: `{args.sentinel_all_min_margin}`\n")
        handle.write(f"- sentinel_pairwise_min_margin: `{args.sentinel_pairwise_min_margin}`\n")
        handle.write(f"- horizon_gate_min_gain: `{args.horizon_gate_min_gain}`\n")
        handle.write(f"- horizon_gate_min_ratio: `{args.horizon_gate_min_ratio}`\n")
        handle.write(f"- horizon_gate_uncertainty_floor: `{args.horizon_gate_uncertainty_floor}`\n")
        handle.write(f"- confidence_fast_all_min_delta_loss: `{args.confidence_fast_all_min_delta_loss}`\n")
        handle.write(f"- confidence_lazy_pairwise_min_delta_loss: `{args.confidence_lazy_pairwise_min_delta_loss}`\n")
        handle.write(f"- confidence_lazy_pairwise_max_calib_gap: `{args.confidence_lazy_pairwise_max_calib_gap}`\n")
        handle.write(f"- confidence_lazy_pairwise_max_memory_delta_loss: `{args.confidence_lazy_pairwise_max_memory_delta_loss}`\n")
        handle.write(f"- sentinel_min_original_max_gap: `{args.sentinel_min_original_max_gap}`\n")
        handle.write(f"- sentinel_min_original_positive_ratio: `{args.sentinel_min_original_positive_ratio}`\n")
        handle.write(f"- meta_min_original_max_gap: `{args.meta_min_original_max_gap}`\n")
        handle.write(f"- meta_selected_loss_slack: `{args.meta_selected_loss_slack}`\n")
        handle.write(f"- meta_max_gap_increase: `{args.meta_max_gap_increase}`\n")
        handle.write(f"- meta_max_positive_ratio_increase: `{args.meta_max_positive_ratio_increase}`\n")
        handle.write(
            f"- meta_min_original_positive_ratio_if_increase: `{args.meta_min_original_positive_ratio_if_increase}`\n"
        )
        handle.write(f"- aggregate PCIC-R PPL: `{math.exp(mean_loss):.6f}`\n")
        handle.write(f"- aggregate PCIC-R seconds: `{total_eval_seconds:.4f}`\n")
        handle.write(f"- aggregate gate seconds: `{total_gate_seconds:.4f}`\n")
        handle.write(f"- aggregate method seconds: `{total_eval_seconds + total_gate_seconds:.4f}`\n")
        handle.write(
            f"- rescue/compressed tokens: `{total_rescue_tokens}/{total_compressed_tokens}` "
            f"(rescue fraction {total_rescue_tokens / max(1, total_eval_tokens):.4f})\n\n"
        )
        handle.write("| block | chosen_combo | delta_loss | delta_ppl | rescue | compressed | rule |\n")
        handle.write("| ---: | --- | ---: | ---: | ---: | ---: | --- |\n")
        for row in rows:
            if row["kind"] != "pcic_r_eval":
                continue
            handle.write(
                f"| {row['block']} | `{row['combo']}` | {float(row['delta_loss']):.6f} | "
                f"{float(row['delta_ppl']):.6f} | {row['rescue_tokens']} | {row['compressed_tokens']} | "
                f"`{row['rescue_rule']}` |\n"
            )
    print(f"wrote {csv_path}")
    if token_risk_rows:
        print(f"wrote {risk_csv_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
