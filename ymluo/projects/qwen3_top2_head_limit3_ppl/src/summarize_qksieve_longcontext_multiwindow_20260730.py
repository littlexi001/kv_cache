#!/usr/bin/env python
"""Pool paired long-context windows without averaging PPL values."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", required=True, type=Path)
    parser.add_argument("--output_json", required=True, type=Path)
    parser.add_argument("--output_md", required=True, type=Path)
    parser.add_argument("--bootstrap_samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260730)
    return parser.parse_args()


def percentile_interval(values: np.ndarray) -> list[float]:
    return [
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
    ]


def main() -> None:
    args = parse_args()
    paths = sorted(args.run_root.glob("*/summary.json"))
    if not paths:
        raise FileNotFoundError(f"no window summaries under {args.run_root}")

    windows: list[dict[str, Any]] = []
    paired_rows: dict[str, list[dict[str, float]]] = defaultdict(list)
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = {row["variant"]: row for row in payload["rows"]}
        full = rows["full_attention"]
        full_tokens = payload["token_rows"]["full_attention"]
        window_id = path.parent.name
        windows.append(
            {
                "window": window_id,
                "topic": payload["topic"],
                "history_tokens": payload["history_tokens"],
                "eval_tokens": payload["eval_tokens"],
                "full_ppl": full["ppl"],
                "shared_prefill_seconds": payload["shared_prefill_seconds"],
            }
        )
        for variant, token_rows in payload["token_rows"].items():
            if variant == "full_attention":
                continue
            if len(token_rows) != len(full_tokens):
                raise ValueError(f"unpaired token rows in {path}: {variant}")
            row = rows[variant]
            for full_token, sparse_token in zip(full_tokens, token_rows):
                if full_token["target_index"] != sparse_token["target_index"]:
                    raise ValueError(f"target mismatch in {path}: {variant}")
                paired_rows[variant].append(
                    {
                        "window": window_id,
                        "full_nll": float(full_token["nll"]),
                        "sparse_nll": float(sparse_token["nll"]),
                        "kl": float(sparse_token["kl_full_to_sparse"]),
                        "top1": float(sparse_token["top1_agreement"]),
                    }
                )
            paired_rows[variant].append(
                {
                    "_window_marker": 1.0,
                    "window": window_id,
                    "steady_seconds": float(
                        row["steady_sparse_seconds_per_step"]
                    ),
                    "online_seconds": float(row["sparse_seconds_per_step"]),
                    "full_seconds": float(
                        full["steady_sparse_seconds_per_step"]
                    ),
                    "active_tokens": float(
                        row["actual_attention_tokens_mean"]
                    ),
                    "active_ratio": float(
                        row["actual_attention_tokens_mean"]
                        / payload["history_tokens"]
                    ),
                    "oracle_retained_mass": (
                        float(row["oracle_retained_mass_mean"])
                        if "oracle_retained_mass_mean" in row
                        else None
                    ),
                }
            )

    rng = np.random.default_rng(args.seed)
    summaries: list[dict[str, Any]] = []
    for variant, raw_rows in sorted(paired_rows.items()):
        token_rows = [row for row in raw_rows if "_window_marker" not in row]
        markers = [row for row in raw_rows if "_window_marker" in row]
        full_nll = np.asarray([row["full_nll"] for row in token_rows])
        sparse_nll = np.asarray([row["sparse_nll"] for row in token_rows])
        delta = sparse_nll - full_nll
        window_names = sorted({str(row["window"]) for row in token_rows})
        window_delta = np.asarray(
            [
                np.mean(
                    [
                        row["sparse_nll"] - row["full_nll"]
                        for row in token_rows
                        if row["window"] == window
                    ]
                )
                for window in window_names
            ]
        )
        sample_indices = rng.integers(
            0,
            window_delta.size,
            size=(args.bootstrap_samples, window_delta.size),
        )
        bootstrap_retention = np.exp(
            -window_delta[sample_indices].mean(axis=1)
        )
        per_window = []
        for window in window_names:
            selected = [
                row for row in token_rows if row["window"] == window
            ]
            marker = next(
                row for row in markers if row["window"] == window
            )
            selected_full_nll = np.asarray(
                [row["full_nll"] for row in selected]
            )
            selected_sparse_nll = np.asarray(
                [row["sparse_nll"] for row in selected]
            )
            per_window.append(
                {
                    "window": window,
                    "tokens": len(selected),
                    "quality_retention": float(
                        np.exp(
                            selected_full_nll.mean()
                            - selected_sparse_nll.mean()
                        )
                    ),
                    "top1_agreement": float(
                        np.mean([row["top1"] for row in selected])
                    ),
                    "kl_full_to_sparse": float(
                        np.mean([row["kl"] for row in selected])
                    ),
                    "active_ratio": float(marker["active_ratio"]),
                    "steady_speedup": float(
                        marker["full_seconds"] / marker["steady_seconds"]
                    ),
                    "online_decode_speedup": float(
                        marker["full_seconds"] / marker["online_seconds"]
                    ),
                }
            )
        full_ppl = math.exp(float(full_nll.mean()))
        sparse_ppl = math.exp(float(sparse_nll.mean()))
        full_seconds = np.mean([row["full_seconds"] for row in markers])
        steady_seconds = np.mean(
            [row["steady_seconds"] for row in markers]
        )
        online_seconds = np.mean(
            [row["online_seconds"] for row in markers]
        )
        oracle_mass_values = [
            float(row["oracle_retained_mass"])
            for row in markers
            if row.get("oracle_retained_mass") is not None
        ]
        summaries.append(
            {
                "variant": variant,
                "windows": len(markers),
                "tokens": int(delta.size),
                "full_ppl": full_ppl,
                "sparse_ppl": sparse_ppl,
                "quality_retention": full_ppl / sparse_ppl,
                "quality_retention_95ci": percentile_interval(
                    bootstrap_retention
                ),
                "top1_agreement": float(
                    np.mean([row["top1"] for row in token_rows])
                ),
                "kl_full_to_sparse": float(
                    np.mean([row["kl"] for row in token_rows])
                ),
                "active_tokens_mean": float(
                    np.mean([row["active_tokens"] for row in markers])
                ),
                "active_ratio_mean": float(
                    np.mean([row["active_ratio"] for row in markers])
                ),
                "steady_speedup": float(full_seconds / steady_seconds),
                "online_decode_speedup": float(
                    full_seconds / online_seconds
                ),
                "oracle_retained_attention_mass": (
                    float(np.mean(oracle_mass_values))
                    if oracle_mass_values
                    else None
                ),
                "per_window": per_window,
            }
        )

    result = {
        "schema": "qksieve_longcontext_multiwindow_v1",
        "windows": windows,
        "summaries": summaries,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    history_lengths = {
        int(window["history_tokens"]) for window in windows
    }
    history_label = (
        f"{next(iter(history_lengths)):,}-token"
        if len(history_lengths) == 1
        else "mixed-length"
    )
    lines = [
        f"# QKSieve {history_label} multi-window summary",
        "",
        "| Variant | Windows | Tokens | Active | Oracle mass | PPL retention (95% CI) | Top-1 | KL | Steady | Online |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lo, hi = row["quality_retention_95ci"]
        lines.append(
            f"| {row['variant']} | {row['windows']} | {row['tokens']} | "
            f"{100 * row['active_ratio_mean']:.3f}% | "
            + (
                f"{100 * row['oracle_retained_attention_mass']:.3f}% | "
                if row["oracle_retained_attention_mass"] is not None
                else "-- | "
            )
            +
            f"{100 * row['quality_retention']:.3f}% "
            f"([{100 * lo:.2f}, {100 * hi:.2f}]) | "
            f"{100 * row['top1_agreement']:.3f}% | "
            f"{row['kl_full_to_sparse']:.5f} | "
            f"{row['steady_speedup']:.3f}x | "
            f"{row['online_decode_speedup']:.3f}x |"
        )
    for row in summaries:
        lines.extend(
            [
                "",
                f"## {row['variant']}",
                "",
                "| Window | Tokens | Active | PPL retention | Top-1 | KL | Steady | Online |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for window in row["per_window"]:
            lines.append(
                f"| {window['window']} | {window['tokens']} | "
                f"{100 * window['active_ratio']:.3f}% | "
                f"{100 * window['quality_retention']:.3f}% | "
                f"{100 * window['top1_agreement']:.3f}% | "
                f"{window['kl_full_to_sparse']:.5f} | "
                f"{window['steady_speedup']:.3f}x | "
                f"{window['online_decode_speedup']:.3f}x |"
            )
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output_md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
