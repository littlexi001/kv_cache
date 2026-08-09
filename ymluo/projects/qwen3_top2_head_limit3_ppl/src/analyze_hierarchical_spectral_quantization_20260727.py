from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch


HEAD_DIM = 128
FULL_KV_BITS_PER_TOKEN = 2 * HEAD_DIM * 16


def parse_float_list(value: str) -> list[float]:
    values = sorted({float(part) for part in value.split(",") if part.strip()})
    if not values:
        raise ValueError("expected at least one floating-point value")
    return values


def covariance_basis(sampled_key: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    working = sampled_key.float()
    second_moment = working.transpose(0, 1) @ working
    second_moment /= float(working.shape[0])
    eigenvalues, eigenvectors = torch.linalg.eigh(second_moment)
    return eigenvectors.flip(-1).contiguous(), eigenvalues.flip(-1).contiguous()


def quantize_groupwise(
    values: torch.Tensor,
    bits: int,
    group_size: int = 16,
) -> torch.Tensor:
    if bits not in {2, 4, 8}:
        raise ValueError("groupwise quantization supports 2, 4, or 8 bits")
    if values.shape[-1] % group_size != 0:
        raise ValueError("the final dimension must be divisible by group_size")
    shape = (*values.shape[:-1], values.shape[-1] // group_size, group_size)
    grouped = values.float().reshape(shape)
    maximum_code = (1 << (bits - 1)) - 1
    scale = grouped.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-8)
    scale /= float(maximum_code)
    codes = torch.round(grouped / scale).clamp(-maximum_code, maximum_code)
    return (codes * scale).reshape_as(values)


def sign_reconstruct(values: torch.Tensor, group_size: int = 80) -> torch.Tensor:
    if values.shape[-1] % group_size != 0:
        raise ValueError("the final dimension must be divisible by group_size")
    shape = (*values.shape[:-1], values.shape[-1] // group_size, group_size)
    grouped = values.float().reshape(shape)
    scale = grouped.abs().mean(dim=-1, keepdim=True)
    signs = torch.where(grouped >= 0.0, 1.0, -1.0)
    return (signs * scale).reshape_as(values)


def query_int8(values: torch.Tensor) -> torch.Tensor:
    return quantize_groupwise(values.view(1, -1), 8, 16).view(-1)


def selection_metrics(
    exact_scores: torch.Tensor,
    attention: torch.Tensor,
    approximate_scores: torch.Tensor,
    true_top_indices: torch.Tensor,
    selected_fraction: float,
) -> dict[str, float]:
    selected_count = min(
        exact_scores.numel(),
        max(1, math.ceil(selected_fraction * exact_scores.numel())),
    )
    selected = torch.topk(approximate_scores, k=selected_count).indices
    selected_mask = torch.zeros_like(exact_scores, dtype=torch.bool)
    selected_mask[selected] = True
    hits = int(selected_mask[true_top_indices].sum().item())
    selected_mass = float(attention[selected].sum().item())
    true_mass = float(attention[true_top_indices].sum().item())
    finite = torch.isfinite(approximate_scores)
    metric_exact = exact_scores[finite]
    metric_approximate = approximate_scores[finite]
    centered_exact = metric_exact - metric_exact.mean()
    centered_approximate = metric_approximate - metric_approximate.mean()
    denominator = (
        torch.linalg.vector_norm(centered_exact)
        * torch.linalg.vector_norm(centered_approximate)
    )
    pearson = (
        float((centered_exact @ centered_approximate / denominator).item())
        if float(denominator.item()) > 0.0
        else 0.0
    )
    rmse = float(
        torch.mean((metric_exact - metric_approximate).square()).sqrt().item()
    )
    return {
        "selected_fraction": selected_count / exact_scores.numel(),
        "selected_count": selected_count,
        "top2_recall": hits / max(1, int(true_top_indices.numel())),
        "selected_attention_mass": selected_mass,
        "oracle_top2_attention_mass": true_mass,
        "top2_attention_mass_recall": (
            selected_mass / true_mass if true_mass > 0.0 else 0.0
        ),
        "score_pearson": pearson,
        "score_rmse": rmse,
    }


def progressive_scores(
    coarse_scores: torch.Tensor,
    tail_scores: torch.Tensor,
    candidate_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    candidate_count = min(
        coarse_scores.numel(),
        max(1, math.ceil(candidate_fraction * coarse_scores.numel())),
    )
    candidates = torch.topk(coarse_scores, k=candidate_count).indices
    refined = torch.full_like(coarse_scores, -torch.inf)
    refined[candidates] = coarse_scores[candidates] + tail_scores[candidates]
    return refined, candidates


def summarize(values: Iterable[float]) -> dict[str, float]:
    tensor = torch.tensor(list(values), dtype=torch.float64)
    return {
        "mean": float(tensor.mean().item()),
        "p10": float(torch.quantile(tensor, 0.10).item()),
        "p50": float(torch.quantile(tensor, 0.50).item()),
        "p90": float(torch.quantile(tensor, 0.90).item()),
        "minimum": float(tensor.min().item()),
        "maximum": float(tensor.max().item()),
    }


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metric_fields = (
        "top2_recall",
        "selected_attention_mass",
        "oracle_top2_attention_mass",
        "top2_attention_mass_recall",
        "score_pearson",
        "score_rmse",
        "candidate_top2_recall",
        "candidate_ratio",
    )
    groups: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["method"]), float(row["selected_fraction_target"]))].append(
            row
        )
    output: list[dict[str, Any]] = []
    for (method, selected_fraction), items in sorted(groups.items()):
        result: dict[str, Any] = {
            "method": method,
            "selected_fraction_target": selected_fraction,
            "cases": len(items),
            "logical_index_bits_per_token": float(
                items[0]["logical_index_bits_per_token"]
            ),
            "logical_index_ratio_of_full_kv": float(
                items[0]["logical_index_bits_per_token"]
            )
            / FULL_KV_BITS_PER_TOKEN,
            "mean_scan_bits_per_history_token": float(
                items[0]["mean_scan_bits_per_history_token"]
            ),
            "mean_scan_ratio_of_full_kv": float(
                items[0]["mean_scan_bits_per_history_token"]
            )
            / FULL_KV_BITS_PER_TOKEN,
        }
        for field in metric_fields:
            stats = summarize(float(item[field]) for item in items)
            result.update({f"{field}_{name}": value for name, value in stats.items()})
        output.append(result)
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay real per-head Q/K traces to evaluate spectrally hierarchical "
            "mixed-precision and sub-one-bit tail retrieval."
        )
    )
    parser.add_argument("--trace_path", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample_stride", type=int, default=32)
    parser.add_argument("--top_fraction", type=float, default=0.02)
    parser.add_argument("--selected_fractions", default="0.02,0.03,0.04")
    parser.add_argument("--candidate_fractions", default="0.03,0.04,0.06,0.08")
    parser.add_argument(
        "--tail_resident_fractions",
        default="0.1,0.25,0.5",
        help=(
            "Fractions of high-tail-norm tokens that receive a resident 1-bit "
            "tail code. The average tail code rate equals this fraction."
        ),
    )
    parser.add_argument("--max_records", type=int, default=0)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.sample_stride <= 0:
        raise ValueError("sample_stride must be positive")
    if not 0.0 < args.top_fraction < 1.0:
        raise ValueError("top_fraction must be in (0, 1)")
    selected_fractions = parse_float_list(args.selected_fractions)
    candidate_fractions = parse_float_list(args.candidate_fractions)
    tail_resident_fractions = parse_float_list(args.tail_resident_fractions)
    if any(value < args.top_fraction for value in selected_fractions):
        raise ValueError("selected fractions cannot be below top_fraction")
    if max(candidate_fractions) < max(selected_fractions):
        raise ValueError(
            "at least one candidate fraction must cover the largest selected fraction"
        )
    if any(not 0.0 < value <= 1.0 for value in tail_resident_fractions):
        raise ValueError("tail resident fractions must be in (0, 1]")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    payload = torch.load(args.trace_path, map_location="cpu", weights_only=False)
    records = list(payload.get("records", []))
    if args.max_records > 0:
        records = records[: args.max_records]
    if not records:
        raise ValueError("trace contains no records")
    head_dim = int(records[0]["query"].shape[-1])
    if head_dim != HEAD_DIM:
        raise ValueError(f"expected head dimension {HEAD_DIM}, got {head_dim}")

    rows: list[dict[str, Any]] = []
    band_rows: list[dict[str, Any]] = []
    layer_states: dict[int, dict[str, Any]] = {}

    for record_index, record in enumerate(records):
        layer = int(record["layer"])
        query = record["query"].to(device).float()[0, :, 0, :]
        scaling = float(record["scaling"])
        query_heads = int(query.shape[0])
        raw_key = record.get("key")
        if layer not in layer_states:
            if raw_key is None:
                raise ValueError(
                    f"layer {layer} has no initial key state before query replay"
                )
            all_key = raw_key.to(device).float()[0]
            history_count = int(all_key.shape[1]) - 1
            key = all_key[:, :history_count]
            kv_heads = int(key.shape[0])
            prepared_heads: list[dict[str, Any]] = []
            for kv_head in range(kv_heads):
                head_key = key[kv_head]
                basis, eigenvalues = covariance_basis(
                    head_key[:: args.sample_stride]
                )
                coefficients = head_key @ basis
                pca48_int4_key = quantize_groupwise(
                    coefficients[:, :48], 4
                )
                pca128_int4_key = quantize_groupwise(coefficients, 4)
                mixed_core_key = torch.cat(
                    (
                        quantize_groupwise(coefficients[:, :16], 8),
                        quantize_groupwise(coefficients[:, 16:48], 4),
                    ),
                    dim=-1,
                )
                tail = coefficients[:, 48:]
                tail_sign_key = sign_reconstruct(tail)
                tail_int2_key = quantize_groupwise(tail, 2, 16)
                tail_norm = torch.linalg.vector_norm(tail, dim=-1)
                resident_masks: dict[float, torch.Tensor] = {}
                for resident_fraction in tail_resident_fractions:
                    resident_count = min(
                        history_count,
                        max(
                            1,
                            math.ceil(
                                resident_fraction * history_count
                            ),
                        ),
                    )
                    mask = torch.zeros(
                        history_count, dtype=torch.bool, device=device
                    )
                    mask[
                        torch.topk(tail_norm, k=resident_count).indices
                    ] = True
                    resident_masks[resident_fraction] = mask
                prepared_heads.append(
                    {
                        "head_key": head_key,
                        "basis": basis,
                        "eigenvalues": eigenvalues,
                        "coefficients": coefficients,
                        "pca48_int4_key": pca48_int4_key,
                        "pca128_int4_key": pca128_int4_key,
                        "mixed_core_key": mixed_core_key,
                        "tail": tail,
                        "tail_sign_key": tail_sign_key,
                        "tail_int2_key": tail_int2_key,
                        "resident_masks": resident_masks,
                    }
                )
            layer_states[layer] = {
                "history_count": history_count,
                "kv_heads": kv_heads,
                "prepared_heads": prepared_heads,
            }
        layer_state = layer_states[layer]
        history_count = int(layer_state["history_count"])
        kv_heads = int(layer_state["kv_heads"])
        prepared_heads = layer_state["prepared_heads"]
        if query_heads % kv_heads != 0:
            raise ValueError("query heads must be divisible by KV heads")
        groups = query_heads // kv_heads
        top_count = max(1, math.ceil(args.top_fraction * history_count))

        for kv_head in range(kv_heads):
            prepared = prepared_heads[kv_head]
            head_key = prepared["head_key"]
            basis = prepared["basis"]
            eigenvalues = prepared["eigenvalues"]
            coefficients = prepared["coefficients"]
            pca48_int4_key = prepared["pca48_int4_key"]
            pca128_int4_key = prepared["pca128_int4_key"]
            mixed_core_key = prepared["mixed_core_key"]
            tail = prepared["tail"]
            tail_sign_key = prepared["tail_sign_key"]
            tail_int2_key = prepared["tail_int2_key"]
            resident_masks = prepared["resident_masks"]

            for group in range(groups):
                query_head = kv_head * groups + group
                head_query = query[query_head]
                projected_query = head_query @ basis
                projected_query_q8 = query_int8(projected_query)
                exact_scores = (head_key @ head_query) * scaling
                attention = torch.softmax(exact_scores, dim=-1)
                true_top_indices = torch.topk(
                    exact_scores, k=top_count
                ).indices

                pca48_scores = (
                    pca48_int4_key @ projected_query_q8[:48]
                ) * scaling
                pca128_scores = (
                    pca128_int4_key @ projected_query_q8
                ) * scaling
                mixed_core_scores = (
                    mixed_core_key @ projected_query_q8[:48]
                ) * scaling
                tail_exact_scores = (
                    tail @ projected_query[48:]
                ) * scaling
                tail_sign_scores = (
                    tail_sign_key @ projected_query_q8[48:]
                ) * scaling
                tail_int2_scores = (
                    tail_int2_key @ projected_query_q8[48:]
                ) * scaling

                exact_core_scores = (
                    coefficients[:, :48] @ projected_query[:48]
                ) * scaling
                exact_tail_centered = tail_exact_scores - tail_exact_scores.mean()
                sign_tail_centered = tail_sign_scores - tail_sign_scores.mean()
                tail_denominator = (
                    torch.linalg.vector_norm(exact_tail_centered)
                    * torch.linalg.vector_norm(sign_tail_centered)
                )
                tail_pearson = (
                    float(
                        (
                            exact_tail_centered
                            @ sign_tail_centered
                            / tail_denominator
                        ).item()
                    )
                    if float(tail_denominator.item()) > 0.0
                    else 0.0
                )
                core_selected = torch.topk(
                    mixed_core_scores, k=top_count
                ).indices
                core_mask = torch.zeros_like(exact_scores, dtype=torch.bool)
                core_mask[core_selected] = True
                missed = true_top_indices[~core_mask[true_top_indices]]
                band_rows.append(
                    {
                        "label": args.label,
                        "record": record_index,
                        "layer": layer,
                        "kv_head": kv_head,
                        "query_head": query_head,
                        "history_count": history_count,
                        "top_count": top_count,
                        "eigenvalue_top16_fraction": float(
                            eigenvalues[:16].sum().item()
                            / eigenvalues.sum().clamp_min(1.0e-12).item()
                        ),
                        "eigenvalue_top48_fraction": float(
                            eigenvalues[:48].sum().item()
                            / eigenvalues.sum().clamp_min(1.0e-12).item()
                        ),
                        "query_energy_top16_fraction": float(
                            projected_query[:16].square().sum().item()
                            / projected_query.square().sum().clamp_min(1.0e-12).item()
                        ),
                        "query_energy_top48_fraction": float(
                            projected_query[:48].square().sum().item()
                            / projected_query.square().sum().clamp_min(1.0e-12).item()
                        ),
                        "exact_tail_score_std": float(
                            tail_exact_scores.std().item()
                        ),
                        "exact_core_score_std": float(
                            exact_core_scores.std().item()
                        ),
                        "tail_to_core_score_std_ratio": float(
                            tail_exact_scores.std().item()
                            / exact_core_scores.std().clamp_min(1.0e-12).item()
                        ),
                        "tail_sign_score_pearson": tail_pearson,
                        "core_top2_missed_count": int(missed.numel()),
                        "core_top2_missed_fraction": (
                            float(missed.numel()) / top_count
                        ),
                        "missed_tail_score_advantage": (
                            float(
                                (
                                    tail_exact_scores[missed].mean()
                                    - tail_exact_scores[core_selected].mean()
                                ).item()
                            )
                            if missed.numel()
                            else 0.0
                        ),
                    }
                )

                methods: dict[str, tuple[torch.Tensor, float, float, float, float]] = {
                    "pca48_int4": (
                        pca48_scores,
                        48 * 4,
                        48 * 4,
                        1.0,
                        1.0,
                    ),
                    "pca128_int4": (
                        pca128_scores,
                        128 * 4,
                        128 * 4,
                        1.0,
                        1.0,
                    ),
                    "hier16i8_32i4_tail0": (
                        mixed_core_scores,
                        16 * 8 + 32 * 4,
                        16 * 8 + 32 * 4,
                        1.0,
                        1.0,
                    ),
                    "hier16i8_32i4_tail1_all": (
                        mixed_core_scores + tail_sign_scores,
                        16 * 8 + 32 * 4 + 80,
                        16 * 8 + 32 * 4 + 80,
                        1.0,
                        1.0,
                    ),
                    "hier16i8_32i4_tail2_all": (
                        mixed_core_scores + tail_int2_scores,
                        16 * 8 + 32 * 4 + 80 * 2,
                        16 * 8 + 32 * 4 + 80 * 2,
                        1.0,
                        1.0,
                    ),
                    "pca48i4_tail1_all": (
                        pca48_scores + tail_sign_scores,
                        48 * 4 + 80,
                        48 * 4 + 80,
                        1.0,
                        1.0,
                    ),
                }

                for resident_fraction, resident_mask in resident_masks.items():
                    sparse_tail = torch.where(
                        resident_mask, tail_sign_scores, 0.0
                    )
                    methods[
                        f"hier16i8_32i4_tail1_normresident{resident_fraction:g}"
                    ] = (
                        mixed_core_scores + sparse_tail,
                        16 * 8 + 32 * 4 + 80 * resident_fraction,
                        16 * 8 + 32 * 4 + 80 * resident_fraction,
                        1.0,
                        1.0,
                    )
                    methods[
                        f"pca48i4_tail1_normresident{resident_fraction:g}"
                    ] = (
                        pca48_scores + sparse_tail,
                        48 * 4 + 80 * resident_fraction,
                        48 * 4 + 80 * resident_fraction,
                        1.0,
                        1.0,
                    )

                for candidate_fraction in candidate_fractions:
                    refined_hier_sign, hier_sign_candidates = progressive_scores(
                        mixed_core_scores,
                        tail_sign_scores,
                        candidate_fraction,
                    )
                    hier_sign_candidate_mask = torch.zeros_like(
                        exact_scores, dtype=torch.bool
                    )
                    hier_sign_candidate_mask[hier_sign_candidates] = True
                    hier_sign_candidate_recall = float(
                        hier_sign_candidate_mask[true_top_indices]
                        .float()
                        .mean()
                        .item()
                    )
                    methods[
                        f"progressive_hier841_c{candidate_fraction:g}"
                    ] = (
                        refined_hier_sign,
                        16 * 8 + 32 * 4 + 80,
                        16 * 8 + 32 * 4 + 80 * candidate_fraction,
                        hier_sign_candidate_recall,
                        candidate_fraction,
                    )

                    refined_hier_int2, hier_int2_candidates = progressive_scores(
                        mixed_core_scores,
                        tail_int2_scores,
                        candidate_fraction,
                    )
                    hier_int2_candidate_mask = torch.zeros_like(
                        exact_scores, dtype=torch.bool
                    )
                    hier_int2_candidate_mask[hier_int2_candidates] = True
                    hier_int2_candidate_recall = float(
                        hier_int2_candidate_mask[true_top_indices]
                        .float()
                        .mean()
                        .item()
                    )
                    methods[
                        f"progressive_hier842_c{candidate_fraction:g}"
                    ] = (
                        refined_hier_int2,
                        16 * 8 + 32 * 4 + 80 * 2,
                        16 * 8 + 32 * 4 + 80 * 2 * candidate_fraction,
                        hier_int2_candidate_recall,
                        candidate_fraction,
                    )

                    refined_sign, sign_candidates = progressive_scores(
                        pca48_scores, tail_sign_scores, candidate_fraction
                    )
                    sign_candidate_mask = torch.zeros_like(
                        exact_scores, dtype=torch.bool
                    )
                    sign_candidate_mask[sign_candidates] = True
                    sign_candidate_recall = float(
                        sign_candidate_mask[true_top_indices].float().mean().item()
                    )
                    methods[
                        f"progressive_pca48i4_tail1_c{candidate_fraction:g}"
                    ] = (
                        refined_sign,
                        48 * 4 + 80,
                        48 * 4 + 80 * candidate_fraction,
                        sign_candidate_recall,
                        candidate_fraction,
                    )

                    refined_int2, int2_candidates = progressive_scores(
                        pca48_scores, tail_int2_scores, candidate_fraction
                    )
                    int2_candidate_mask = torch.zeros_like(
                        exact_scores, dtype=torch.bool
                    )
                    int2_candidate_mask[int2_candidates] = True
                    int2_candidate_recall = float(
                        int2_candidate_mask[true_top_indices].float().mean().item()
                    )
                    methods[
                        f"progressive_pca48i4_tail2_c{candidate_fraction:g}"
                    ] = (
                        refined_int2,
                        48 * 4 + 80 * 2,
                        48 * 4 + 80 * 2 * candidate_fraction,
                        int2_candidate_recall,
                        candidate_fraction,
                    )

                    refined_exact, exact_candidates = progressive_scores(
                        pca48_scores, tail_exact_scores, candidate_fraction
                    )
                    exact_candidate_mask = torch.zeros_like(
                        exact_scores, dtype=torch.bool
                    )
                    exact_candidate_mask[exact_candidates] = True
                    exact_candidate_recall = float(
                        exact_candidate_mask[true_top_indices].float().mean().item()
                    )
                    methods[
                        f"oracle_progressive_pca48i4_tailexact_c{candidate_fraction:g}"
                    ] = (
                        refined_exact,
                        48 * 4 + 80 * 16,
                        48 * 4 + 80 * 16 * candidate_fraction,
                        exact_candidate_recall,
                        candidate_fraction,
                    )

                for method, (
                    approximate_scores,
                    index_bits,
                    scan_bits,
                    candidate_recall,
                    candidate_ratio,
                ) in methods.items():
                    for selected_fraction in selected_fractions:
                        if (
                            candidate_ratio < 1.0
                            and selected_fraction > candidate_ratio
                        ):
                            continue
                        metrics = selection_metrics(
                            exact_scores,
                            attention,
                            approximate_scores,
                            true_top_indices,
                            selected_fraction,
                        )
                        rows.append(
                            {
                                "label": args.label,
                                "record": record_index,
                                "layer": layer,
                                "kv_head": kv_head,
                                "query_head": query_head,
                                "history_count": history_count,
                                "method": method,
                                "selected_fraction_target": selected_fraction,
                                "logical_index_bits_per_token": index_bits,
                                "mean_scan_bits_per_history_token": scan_bits,
                                "candidate_top2_recall": candidate_recall,
                                "candidate_ratio": candidate_ratio,
                                **metrics,
                            }
                        )

        print(
            json.dumps(
                {
                    "label": args.label,
                    "record": record_index + 1,
                    "records": len(records),
                    "layer": layer,
                    "rows": len(rows),
                }
            ),
            flush=True,
        )

    aggregate_rows = aggregate(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_head.csv", rows)
    write_csv(args.output_dir / "band_diagnostics.csv", band_rows)
    write_csv(args.output_dir / "summary.csv", aggregate_rows)
    report = {
        "config": {
            **vars(args),
            "trace_path": str(args.trace_path),
            "output_dir": str(args.output_dir),
            "selected_fractions": selected_fractions,
            "candidate_fractions": candidate_fractions,
            "tail_resident_fractions": tail_resident_fractions,
        },
        "records": len(records),
        "per_head_cases": len(band_rows),
        "methods": aggregate_rows,
        "band_diagnostics": {
            field: summarize(float(row[field]) for row in band_rows)
            for field in (
                "eigenvalue_top16_fraction",
                "eigenvalue_top48_fraction",
                "query_energy_top16_fraction",
                "query_energy_top48_fraction",
                "tail_to_core_score_std_ratio",
                "tail_sign_score_pearson",
                "core_top2_missed_fraction",
                "missed_tail_score_advantage",
            )
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
