#!/usr/bin/env python
"""Real layer-0 QK-balanced selector plus bounded block-mean tail probe."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from analyze_hierarchical_spectral_quantization_20260727 import query_int8
from analyze_qk_balanced_spectral_rate_20260727 import qk_balanced_factors
from analyze_qk_progressive_refinement_20260727 import (
    allocation_rate,
    interval_candidates,
    quantized_bands,
    reconstruct,
)
from analyze_qk_norm_certified_refinement_20260727 import residual_norm_bound
from analyze_qksieve_block_coreset_20260802 import (
    block_coreset_tail_statistics,
    fit_block_coreset,
)
from analyze_qksieve_conditional_value_moments_20260802 import (
    combine_selected_and_tail,
)
from analyze_qksieve_control_variate_layer0_probe_20260802 import (
    load_layer0_activations,
    output_metrics,
)


GROUP_SIZE = 16
PROFILES: dict[str, tuple[int, ...]] = {
    "fixed400_b80": (4, 0, 0, 0, 0, 0, 0, 0),
    "fixed410_b112": (4, 1, 0, 0, 0, 0, 0, 0),
    "fixed4221_b208": (4, 2, 2, 1, 0, 0, 0, 0),
    "fixed4421_b240": (4, 4, 2, 1, 0, 0, 0, 0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--texts", type=Path, nargs="+", required=True)
    parser.add_argument("--history_tokens", type=int, default=8192)
    parser.add_argument("--calibration_tokens", type=int, default=16)
    parser.add_argument("--query_tokens", type=int, default=4)
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--fractions", default="0.01,0.02,0.06")
    parser.add_argument("--profiles", default=",".join(PROFILES))
    parser.add_argument("--sample_stride", type=int, default=16)
    parser.add_argument("--query_shrinkage", type=float, default=0.75)
    parser.add_argument("--key_mean_bits", type=int, default=8)
    parser.add_argument("--value_mean_bits", type=int, default=4)
    parser.add_argument("--certified_full_profile", default="fixed4421_b240")
    parser.add_argument("--norm_bits", type=int, default=4)
    parser.add_argument(
        "--norm_mode", choices=("global", "per_band"), default="per_band"
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def evaluate_text(
    text_path: Path,
    token_ids: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    query, key, value, metadata = load_layer0_activations(args.model, token_ids)
    history_count = args.history_tokens
    calibration_stop = history_count + args.calibration_tokens
    history_key = key[:history_count]
    history_value = value[:history_count]
    calibration = query[history_count:calibration_stop]
    heldout = query[calibration_stop : calibration_stop + args.query_tokens]
    fractions = [float(item) for item in args.fractions.split(",")]
    profile_names = [item for item in args.profiles.split(",") if item]
    scale = float(metadata["head_dim"] ** -0.5)
    group_size = int(metadata["gqa_groups"])
    rows: list[dict[str, Any]] = []

    for kv_head in range(int(metadata["kv_heads"])):
        head_key = history_key[:, kv_head].contiguous()
        head_value = history_value[:, kv_head].contiguous()
        head_calibration = calibration[
            :, kv_head * group_size : (kv_head + 1) * group_size
        ].reshape(-1, int(metadata["head_dim"]))
        query_factor, key_factor, _ = qk_balanced_factors(
            head_key[:: args.sample_stride],
            head_calibration,
            args.query_shrinkage,
        )
        raw_coordinates = head_key @ key_factor
        projected_calibration = head_calibration @ query_factor
        bands = quantized_bands(raw_coordinates, projected_calibration)
        coreset = fit_block_coreset(
            raw_coordinates,
            head_value,
            args.block_size,
            cluster_count=1,
            moment_bits=2,
            iterations=1,
            full_score_coordinates=head_key,
            value_moment_bits=args.value_mean_bits,
            full_score_moment_bits=args.key_mean_bits,
        )

        profile_states = {}
        for profile_name in profile_names:
            allocation = PROFILES[profile_name]
            profile_states[profile_name] = {
                "allocation": allocation,
                "coordinates": reconstruct(bands, allocation).float(),
                "key_bits": GROUP_SIZE * allocation_rate(allocation),
            }
        if args.certified_full_profile not in profile_states:
            raise ValueError("certified full profile must be included in profiles")
        full_state = profile_states[args.certified_full_profile]
        for state in profile_states.values():
            state["residual"] = full_state["coordinates"] - state["coordinates"]
            state["differing_bands"] = tuple(
                index
                for index, (full_bits, base_bits) in enumerate(
                    zip(full_state["allocation"], state["allocation"])
                )
                if full_bits != base_bits
            )

        for query_offset in range(args.query_tokens):
            for group_offset in range(group_size):
                query_head = kv_head * group_size + group_offset
                current_query = heldout[query_offset, query_head]
                exact_scores = head_key @ current_query * scale
                full_weights = torch.softmax(exact_scores, dim=0)
                full_output = full_weights @ head_value
                oracle_order = torch.argsort(exact_scores, descending=True)
                projected_query = query_int8(current_query @ query_factor).float()
                full_proxy_scores = (
                    full_state["coordinates"] @ projected_query * scale
                )

                for profile_name, state in profile_states.items():
                    proxy_scores = state["coordinates"] @ projected_query * scale
                    for fraction in fractions:
                        keep = max(1, math.ceil(history_count * fraction))
                        selected = torch.topk(
                            proxy_scores, keep, sorted=False
                        ).indices
                        oracle = oracle_order[:keep]
                        sparse_output = torch.softmax(
                            exact_scores.index_select(0, selected), dim=0
                        ) @ head_value.index_select(0, selected)
                        oracle_output = torch.softmax(
                            exact_scores.index_select(0, oracle), dim=0
                        ) @ head_value.index_select(0, oracle)

                        method_outputs = {
                            "selected_only": sparse_output,
                            "oracle_topk": oracle_output,
                        }
                        method_indices = {
                            "selected_only": selected,
                            "oracle_topk": oracle,
                        }
                        method_scan_bits = {
                            "selected_only": float(state["key_bits"]),
                            "oracle_topk": float(state["key_bits"]),
                        }
                        method_refinement_ratio = {
                            "selected_only": 0.0,
                            "oracle_topk": 0.0,
                        }
                        for method, indices in (
                            ("blockmean_tail", selected),
                            ("oracle_topk_blockmean_tail", oracle),
                        ):
                            reference = exact_scores.index_select(0, indices).amin()
                            tail_z, tail_y, diagnostics = (
                                block_coreset_tail_statistics(
                                    state["coordinates"],
                                    head_value,
                                    projected_query * scale,
                                    indices,
                                    reference,
                                    coreset,
                                    selected_conditioned=False,
                                    full_score_coordinates=head_key,
                                    full_score_direction=current_query * scale,
                                )
                            )
                            method_outputs[method] = combine_selected_and_tail(
                                exact_scores,
                                exact_scores,
                                head_value,
                                indices,
                                tail_y,
                                tail_z,
                                1.0,
                            )
                            method_indices[method] = indices
                            method_scan_bits[method] = float(state["key_bits"])
                            method_refinement_ratio[method] = 0.0

                        if state["differing_bands"]:
                            bound, code_bits, metadata_bits = residual_norm_bound(
                                state["residual"],
                                projected_query,
                                state["differing_bands"],
                                args.norm_bits,
                                args.norm_mode,
                            )
                            candidate, _ = interval_candidates(
                                proxy_scores,
                                bound * scale,
                                keep,
                            )
                            candidate_indices = torch.nonzero(
                                candidate, as_tuple=False
                            ).flatten()
                            selected_local = torch.topk(
                                full_proxy_scores.index_select(0, candidate_indices),
                                keep,
                                sorted=False,
                            ).indices
                            certified = candidate_indices.index_select(
                                0, selected_local
                            )
                            certified_output = torch.softmax(
                                exact_scores.index_select(0, certified), dim=0
                            ) @ head_value.index_select(0, certified)
                            certified_reference = exact_scores.index_select(
                                0, certified
                            ).amin()
                            certified_z, certified_y, certified_diagnostics = (
                                block_coreset_tail_statistics(
                                    state["coordinates"],
                                    head_value,
                                    projected_query * scale,
                                    certified,
                                    certified_reference,
                                    coreset,
                                    selected_conditioned=False,
                                    full_score_coordinates=head_key,
                                    full_score_direction=current_query * scale,
                                )
                            )
                            certified_tail_output = combine_selected_and_tail(
                                exact_scores,
                                exact_scores,
                                head_value,
                                certified,
                                certified_y,
                                certified_z,
                                1.0,
                            )
                            refinement_ratio = float(candidate.float().mean())
                            missing_bits = (
                                float(full_state["key_bits"])
                                - float(state["key_bits"])
                            )
                            effective_scan_bits = (
                                float(state["key_bits"])
                                + float(code_bits)
                                + float(metadata_bits) / history_count
                                + missing_bits * refinement_ratio
                            )
                            for method, output in (
                                ("certified_selected_only", certified_output),
                                (
                                    "certified_blockmean_tail",
                                    certified_tail_output,
                                ),
                            ):
                                method_outputs[method] = output
                                method_indices[method] = certified
                                method_scan_bits[method] = effective_scan_bits
                                method_refinement_ratio[method] = refinement_ratio
                            coreset_bits_by_method = {
                                "certified_blockmean_tail": float(
                                    certified_diagnostics["bits_per_token"]
                                )
                            }
                        else:
                            coreset_bits_by_method = {}

                        for method, output in method_outputs.items():
                            selected_indices = method_indices[method]
                            rows.append(
                                {
                                    "text": text_path.stem,
                                    "kv_head": kv_head,
                                    "query_head": query_head,
                                    "query_offset": query_offset,
                                    "profile": profile_name,
                                    "allocation": "-".join(
                                        map(str, state["allocation"])
                                    ),
                                    "selector_bits_per_token": state["key_bits"],
                                    "tail_bits_per_token": (
                                        coreset_bits_by_method.get(
                                            method,
                                            float(diagnostics["bits_per_token"]),
                                        )
                                        if method.endswith("blockmean_tail")
                                        else 0.0
                                    ),
                                    "effective_scan_bits_per_token": (
                                        method_scan_bits[method]
                                    ),
                                    "refinement_ratio": (
                                        method_refinement_ratio[method]
                                    ),
                                    "fraction": fraction,
                                    "selected_tokens": keep,
                                    "method": method,
                                    **output_metrics(output, full_output),
                                    "selected_mass": float(
                                        full_weights.index_select(
                                            0, selected_indices
                                        ).sum()
                                    ),
                                    "top1_recall": float(
                                        torch.isin(
                                            torch.argmax(exact_scores)[None],
                                            selected_indices,
                                        )[0]
                                    ),
                                }
                            )
    return rows, metadata


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["profile"], row["fraction"], row["method"])].append(row)
    summaries = []
    for (profile, fraction, method), group in sorted(groups.items()):
        errors = torch.tensor([row["relative_l2"] for row in group])
        summaries.append(
            {
                "profile": profile,
                "fraction": fraction,
                "method": method,
                "conditions": len(group),
                "relative_l2_mean": float(errors.mean()),
                "relative_l2_p90": float(torch.quantile(errors, 0.9)),
                "relative_l2_worst": float(errors.max()),
                "cosine_mean": sum(row["cosine"] for row in group) / len(group),
                "selected_mass_mean": sum(
                    row["selected_mass"] for row in group
                )
                / len(group),
                "top1_recall_mean": sum(row["top1_recall"] for row in group)
                / len(group),
                "selector_bits_per_token": group[0]["selector_bits_per_token"],
                "tail_bits_per_token": group[0]["tail_bits_per_token"],
                "effective_scan_bits_per_token": sum(
                    row["effective_scan_bits_per_token"] for row in group
                )
                / len(group),
                "refinement_ratio_mean": sum(
                    row["refinement_ratio"] for row in group
                )
                / len(group),
            }
        )
    return summaries


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False
    )
    needed = args.history_tokens + args.calibration_tokens + args.query_tokens
    rows: list[dict[str, Any]] = []
    metadata: dict[str, Any] | None = None
    for text_path in args.texts:
        text = text_path.read_text(encoding="utf-8", errors="ignore")
        token_ids = tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt",
            truncation=True,
            max_length=needed,
        ).input_ids[0, :needed]
        if token_ids.numel() < needed:
            raise ValueError(f"{text_path} contains fewer than {needed} tokens")
        current_rows, metadata = evaluate_text(text_path, token_ids, args)
        rows.extend(current_rows)
    payload = {
        "schema": "qksieve_qkbalanced_blockmean_layer0_v1",
        "contract": {
            "scope": "real Qwen3 layer-0 mechanism audit; not end-to-end quality",
            "model": str(args.model),
            "history_tokens": args.history_tokens,
            "calibration_tokens": args.calibration_tokens,
            "query_tokens": args.query_tokens,
            "block_size": args.block_size,
            "fractions": args.fractions,
            "profiles": args.profiles,
            "model_metadata": metadata,
        },
        "aggregate": summarize(rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["aggregate"], indent=2))


if __name__ == "__main__":
    main()
