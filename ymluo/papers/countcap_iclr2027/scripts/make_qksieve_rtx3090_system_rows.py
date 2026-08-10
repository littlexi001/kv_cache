#!/usr/bin/env python
"""Generate auditable RTX 3090 system-table rows from raw QKSieve evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any


PAPER_ROOT = Path(__file__).resolve().parents[1]
YMLUO_ROOT = PAPER_ROOT.parents[1]
PROJECT_ROOT = YMLUO_ROOT / "projects" / "qwen3_top2_head_limit3_ppl"
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from verify_qksieve_robust_paper_evidence_20260810 import (  # noqa: E402
    validate_persistent,
)


LENGTHS = (8192, 16384, 32768, 65536, 131072)
DECODE_LENGTHS = (32768, 65536, 131072)
ATTENTION_ROOT = (
    PROJECT_ROOT
    / "docs"
    / "mha_valuesketch_speed_ab_20260809"
    / "artifacts"
    / "attention"
)
DECODE_ROOT = ATTENTION_ROOT.parent
PERSISTENT_SUMMARY = (
    PROJECT_ROOT
    / "docs"
    / "qksieve_persistent_kv_20260810"
    / "raw_results"
    / "20260810_qksieve_persistent_kv_v3_multiseed"
    / "independent_summary.json"
)
DECODE_PATHS = {
    32768: {
        "full": DECODE_ROOT / "decode_strict/n32768/seed20260809/full.json",
        "fast": (
            DECODE_ROOT
            / "decode_clean32/n32768/seed20260809/qksieve_no_value_top1280.json"
        ),
        "robust": (
            DECODE_ROOT
            / "decode_clean32/n32768/seed20260809/qksieve_valuesketch_top1280.json"
        ),
    },
    65536: {
        "full": DECODE_ROOT / "decode_strict/n65536/seed20260809/full.json",
        "fast": (
            DECODE_ROOT
            / "decode_clean64/n65536/seed20260809/qksieve_no_value_top1280.json"
        ),
        "robust": (
            DECODE_ROOT
            / "decode_strict/n65536/seed20260809/qksieve_valuesketch_top1280.json"
        ),
    },
    131072: {
        "full": DECODE_ROOT / "decode_strict/n131072/seed20260809/full.json",
        "fast": (
            DECODE_ROOT
            / "decode_strict/n131072/seed20260809/qksieve_no_value_top1280.json"
        ),
        "robust": (
            DECODE_ROOT
            / "decode_strict/n131072/seed20260809/qksieve_valuesketch_top1280.json"
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PAPER_ROOT / "data" / "generated" / "qksieve_rtx3090_system_rows.tex"
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=(
            PAPER_ROOT
            / "data"
            / "generated"
            / "qksieve_rtx3090_system_manifest.json"
        ),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(YMLUO_ROOT.parent.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def attention_evidence() -> tuple[list[dict[str, float]], list[Path]]:
    paths = sorted(ATTENTION_ROOT.glob("seed*.json"))
    if len(paths) != 3:
        raise AssertionError("RTX 3090 attention evidence requires exactly three seeds")
    by_length: dict[int, list[dict[str, Any]]] = {length: [] for length in LENGTHS}
    for path in paths:
        payload = read_json(path)
        if payload.get("gpu") != "NVIDIA GeForce RTX 3090":
            raise AssertionError(f"non-RTX-3090 attention evidence: {path}")
        if int(payload.get("warmup", -1)) != 10 or int(
            payload.get("iterations", -1)
        ) != 40:
            raise AssertionError(f"attention timing protocol drifted: {path}")
        rows = payload.get("rows")
        if not isinstance(rows, list) or len(rows) != len(LENGTHS):
            raise AssertionError(f"attention length grid is incomplete: {path}")
        for row in rows:
            length = int(row["history_count"])
            if length not in by_length or row.get("layout") != "MHA_32Q_32KV_D128":
                raise AssertionError(f"attention shape or length drifted: {path}")
            if not (
                row.get("qksieve_valuesketch_candidate_counts_equal") is True
                and row.get("qksieve_valuesketch_candidate_sets_equal") is True
                and int(row.get("qksieve_valuesketch_count_max_abs_diff", -1)) == 0
                and float(row.get("qksieve_valuesketch_threshold_max_abs_diff", 1.0))
                == 0.0
            ):
                raise AssertionError(f"Fast/Robust candidate mismatch: {path}")
            by_length[length].append(row)

    fields = {
        "full_ms": "full_mha_sdpa_ms",
        "fast_ms": "qksieve_complete_ms",
        "robust_ms": "qksieve_valuesketch_complete_ms",
        "fier_ms": "fier_complete_ms",
    }
    result: list[dict[str, float]] = []
    for length in LENGTHS:
        source = by_length[length]
        row = {"history_tokens": float(length)}
        for target, field in fields.items():
            row[target] = statistics.median(float(item[field]) for item in source)
        row["fast_speedup"] = row["full_ms"] / row["fast_ms"]
        row["robust_speedup"] = row["full_ms"] / row["robust_ms"]
        row["fast_vs_fier"] = row["fier_ms"] / row["fast_ms"]
        row["robust_vs_fier"] = row["fier_ms"] / row["robust_ms"]
        result.append(row)
    return result, paths


def validate_decode(payload: dict[str, Any], *, length: int, profile: str) -> None:
    if (
        int(payload.get("history_tokens", -1)) != length
        or int(payload.get("generation_steps", -1)) != 64
        or int(payload.get("steady_start", -1)) != 16
        or payload.get("gpu_name") != "NVIDIA GeForce RTX 3090"
        or int(payload.get("num_attention_heads", -1)) != 32
        or int(payload.get("num_key_value_heads", -1)) != 32
        or payload.get("dtype") != "float16"
    ):
        raise AssertionError(f"decode protocol drifted for {profile}@{length}")
    expected_disabled = profile != "robust"
    if bool(payload.get("value_sketch_disabled")) != expected_disabled:
        raise AssertionError(f"ValueSketch state drifted for {profile}@{length}")
    if profile == "full" and payload.get("method") != "full":
        raise AssertionError(f"Full decode method drifted at {length}")
    if profile != "full" and not str(payload.get("method", "")).startswith(
        "qksieve_"
    ):
        raise AssertionError(f"QKSieve decode method drifted for {profile}@{length}")


def decode_evidence() -> tuple[list[dict[str, float]], list[Path]]:
    paths: list[Path] = []
    rows: list[dict[str, float]] = []
    for length in DECODE_LENGTHS:
        payloads: dict[str, dict[str, Any]] = {}
        for profile, path in DECODE_PATHS[length].items():
            payload = read_json(path)
            validate_decode(payload, length=length, profile=profile)
            payloads[profile] = payload
            paths.append(path)
        full_ms = float(payloads["full"]["steady_mean_ms_per_token"])
        row: dict[str, float] = {
            "history_tokens": float(length),
            "full_ms": full_ms,
        }
        full_horizon = float(
            payloads["full"]["horizons"]["64"][
                "ms_per_generated_token_including_prebuild"
            ]
        )
        for profile in ("fast", "robust"):
            payload = payloads[profile]
            method_ms = float(payload["steady_mean_ms_per_token"])
            build = float(payload["prebuild_wall_seconds"])
            saved_seconds = (full_ms - method_ms) / 1000.0
            if saved_seconds <= 0.0:
                raise AssertionError(f"no decode saving for {profile}@{length}")
            method_horizon = float(
                payload["horizons"]["64"][
                    "ms_per_generated_token_including_prebuild"
                ]
            )
            row[f"{profile}_ms"] = method_ms
            row[f"{profile}_speedup"] = full_ms / method_ms
            row[f"{profile}_build_seconds"] = build
            row[f"{profile}_break_even_tokens"] = float(
                math.ceil(build / saved_seconds)
            )
            row[f"{profile}_online64_speedup"] = full_horizon / method_horizon
        rows.append(row)
    return rows, paths


def persistent_evidence() -> tuple[list[dict[str, Any]], Path]:
    payload = read_json(PERSISTENT_SUMMARY)
    validate_persistent(payload)
    rows = sorted(
        payload["aggregate_rows"], key=lambda row: int(row["history_tokens"])
    )
    for row in rows:
        saved_seconds = (
            float(row["full_warm_ms_per_token"])
            - float(row["qksieve_warm_ms_per_token"])
        ) / 1000.0
        if saved_seconds <= 0.0:
            raise AssertionError("persistent warm path has no per-token saving")
        row["break_even_tokens"] = math.ceil(
            float(row["qksieve_prebuild_seconds"]) / saved_seconds
        )
    return rows, PERSISTENT_SUMMARY


def command(name: str, rows: list[str]) -> str:
    body = "%\n".join(rows)
    return f"\\newcommand{{\\{name}}}{{%\n{body}%\n}}\n"


def scalar_command(name: str, value: str) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}\n"


def speed_with_interval(row: dict[str, Any], field: str) -> str:
    return (
        "{:.3f}$\\times$ {{\\scriptsize[{:.3f},{:.3f}]}}".format(
            float(row[field]),
            float(row[f"{field}_bootstrap_ci95_low"]),
            float(row[f"{field}_bootstrap_ci95_high"]),
        )
    )


def render(
    attention: list[dict[str, float]],
    decode: list[dict[str, float]],
    persistent: list[dict[str, Any]],
    *,
    provenance: str,
) -> str:
    attention_rows = [
        "{}K & {:.4f} & {:.4f} & {:.4f} & {:.2f}$\\times$ & {:.2f}$\\times$ \\\\".format(
            int(row["history_tokens"]) // 1024,
            row["full_ms"],
            row["fast_ms"],
            row["robust_ms"],
            row["fast_speedup"],
            row["robust_speedup"],
        )
        for row in attention
    ]
    fier_rows = [
        "{}K & {:.4f} & {:.4f} & {:.4f} & {:.2f}$\\times$ & {:.2f}$\\times$ \\\\".format(
            int(row["history_tokens"]) // 1024,
            row["fier_ms"],
            row["fast_ms"],
            row["robust_ms"],
            row["fast_vs_fier"],
            row["robust_vs_fier"],
        )
        for row in attention
    ]
    decode_rows = [
        "{}K & {:.2f} & {:.2f} & {:.2f} & {:.2f}$\\times$ & {:.2f}$\\times$ \\\\".format(
            int(row["history_tokens"]) // 1024,
            row["full_ms"],
            row["fast_ms"],
            row["robust_ms"],
            row["fast_speedup"],
            row["robust_speedup"],
        )
        for row in decode
    ]
    build_rows = [
        "{}K & {:.3f} & {:.3f} & {} & {} & {:.2f}$\\times$ & {:.2f}$\\times$ \\\\".format(
            int(row["history_tokens"]) // 1024,
            row["fast_build_seconds"],
            row["robust_build_seconds"],
            int(row["fast_break_even_tokens"]),
            int(row["robust_break_even_tokens"]),
            row["fast_online64_speedup"],
            row["robust_online64_speedup"],
        )
        for row in decode
    ]
    persistent_rows = []
    for row in persistent:
        build = (
            "{:.3f} {{\\scriptsize[{:.3f},{:.3f}]}}".format(
                float(row["qksieve_prebuild_seconds"]),
                float(row["qksieve_prebuild_seconds_bootstrap_ci95_low"]),
                float(row["qksieve_prebuild_seconds_bootstrap_ci95_high"]),
            )
        )
        persistent_rows.append(
            "{}K & {} & {} & {} & {} & {} & {} \\\\".format(
                int(row["history_tokens"]) // 1024,
                speed_with_interval(row, "cold_speedup"),
                speed_with_interval(row, "cold_end_to_end_speedup"),
                speed_with_interval(row, "warm_speedup"),
                speed_with_interval(row, "amortized_speedup"),
                speed_with_interval(row, "append_only_speedup"),
                build,
            )
        )
    persistent_by_length = {
        int(row["history_tokens"]): row for row in persistent
    }
    persistent_32 = persistent_by_length[32768]
    persistent_64 = persistent_by_length[65536]
    return "\n".join(
        [
            f"% Generated from audited RTX 3090 evidence: {provenance}",
            command("QKSieveMhaAttentionRows", attention_rows),
            command("QKSieveMhaFierRows", fier_rows),
            command("QKSieveMhaDecodeRows", decode_rows),
            command("QKSieveMhaBuildBreakEvenRows", build_rows),
            command("QKSievePersistentRows", persistent_rows),
            scalar_command(
                "QKSievePersistentWarmLatencyText",
                "{:.3f}/{:.3f}; {:.3f}/{:.3f}".format(
                    float(persistent_32["full_warm_ms_per_token"]),
                    float(persistent_32["qksieve_warm_ms_per_token"]),
                    float(persistent_64["full_warm_ms_per_token"]),
                    float(persistent_64["qksieve_warm_ms_per_token"]),
                ),
            ),
            scalar_command(
                "QKSievePersistentBuildText",
                "{:.3f}/{:.3f}".format(
                    float(persistent_32["qksieve_prebuild_seconds"]),
                    float(persistent_64["qksieve_prebuild_seconds"]),
                ),
            ),
            scalar_command(
                "QKSievePersistentBreakEvenText",
                "{}/{}".format(
                    int(persistent_32["break_even_tokens"]),
                    int(persistent_64["break_even_tokens"]),
                ),
            ),
        ]
    )


def main() -> None:
    args = parse_args()
    attention, attention_paths = attention_evidence()
    decode, decode_paths = decode_evidence()
    persistent, persistent_path = persistent_evidence()
    paths = attention_paths + decode_paths + [persistent_path]
    hashes = {relative(path): sha256(path) for path in paths}
    aggregate = hashlib.sha256(
        "\n".join(f"{path}:{digest}" for path, digest in sorted(hashes.items())).encode(
            "utf-8"
        )
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        render(attention, decode, persistent, provenance=aggregate),
        encoding="utf-8",
    )
    manifest = {
        "schema": "qksieve_rtx3090_system_table_manifest_v1",
        "aggregate_sha256": aggregate,
        "inputs": hashes,
        "output": relative(args.output),
        "output_sha256": sha256(args.output),
        "attention_rows": attention,
        "decode_rows": decode,
        "persistent_rows": persistent,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
