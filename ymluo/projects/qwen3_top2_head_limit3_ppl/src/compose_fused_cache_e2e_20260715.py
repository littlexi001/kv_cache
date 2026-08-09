from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROTOCOL_FIELDS = (
    "topic",
    "history_tokens",
    "remote_tokens",
    "query_tokens",
    "eval_tokens",
    "final_cache_length",
)


def compose_fused_cache_e2e(
    physical: dict[str, Any],
    full: dict[str, Any],
    *,
    gpu: str,
) -> dict[str, Any]:
    for field in PROTOCOL_FIELDS:
        if physical[field] != full[field]:
            raise ValueError(
                f"protocol mismatch for {field}: {physical[field]} != {full[field]}"
            )
    if physical.get("directory_backend") != "fused":
        raise ValueError("physical result must use the fused directory backend")
    if not physical.get("timing_is_synchronized_per_token"):
        raise ValueError("physical timing must be synchronized per token")
    if full.get("attention_implementation") != "sdpa":
        raise ValueError("full baseline must use SDPA")

    decode_steps = int(physical["query_tokens"]) + int(physical["eval_tokens"]) - 1
    full_decode = float(full["synchronized_model_forward_seconds"])
    physical_decode = float(physical["synchronized_model_forward_seconds"])
    full_setup = float(full["prefill_seconds"])
    physical_setup = float(physical["prefill_plus_conversion_seconds"])
    full_total = full_setup + full_decode
    physical_total = physical_setup + physical_decode
    saved_per_step = (full_decode - physical_decode) / decode_steps
    extra_setup = physical_setup - full_setup
    break_even_steps = extra_setup / saved_per_step if saved_per_step > 0 else None

    full_ppl = float(full["ppl"])
    physical_ppl = float(physical["ppl"])
    final_process_full = int(full["process_gpu_allocated_after_decode"])
    final_process_physical = int(physical["process_gpu_allocated_after_conversion"])
    checks = {
        "quality_retention_at_least_95_percent": full_ppl / physical_ppl >= 0.95,
        "persistent_gpu_kv_below_10_percent": float(
            physical["hierarchical_over_final_length_full_kv"]
        )
        < 0.10,
        "decode_is_faster": full_decode > physical_decode,
        "protocol_total_is_faster": full_total > physical_total,
    }
    status = (
        "fused_hf_cache_quality_storage_and_e2e_validated"
        if all(checks.values())
        else "fused_hf_cache_measured_with_failed_target"
    )
    return {
        "status": status,
        "gpu": gpu,
        "protocol": {
            field: physical[field] for field in PROTOCOL_FIELDS
        }
        | {"synchronized_decode_steps": decode_steps},
        "quality": {
            "full_ppl": full_ppl,
            "physical_ppl": physical_ppl,
            "physical_over_full": physical_ppl / full_ppl,
            "retention_vs_full": full_ppl / physical_ppl,
        },
        "kv_storage": {
            "full_final_gpu_kv_bytes": int(full["final_gpu_kv_bytes"]),
            "physical_persistent_gpu_kv_bytes": int(
                physical["hierarchical_persistent_gpu_bytes"]
            ),
            "physical_over_full": float(
                physical["hierarchical_over_final_length_full_kv"]
            ),
            "pinned_host_bytes": int(physical["pinned_host_bytes"]),
        },
        "latency_seconds": {
            "full_prefill": full_setup,
            "physical_prefill": float(physical["prefill_seconds"]),
            "physical_cache_conversion": float(
                physical["cache_conversion_seconds"]
            ),
            "full_decode": full_decode,
            "physical_decode": physical_decode,
            "full_total_for_protocol": full_total,
            "physical_total_for_protocol": physical_total,
        },
        "speedup": {
            "decode": full_decode / physical_decode,
            "total_for_protocol": full_total / physical_total,
            "full_ms_per_decode_step": 1000.0 * full_decode / decode_steps,
            "physical_ms_per_decode_step": 1000.0
            * physical_decode
            / decode_steps,
            "amortized_break_even_decode_steps": break_even_steps,
        },
        "process_memory": {
            "full_allocated_after_decode": final_process_full,
            "physical_allocated_after_conversion": final_process_physical,
            "allocated_reduction_fraction": 1.0
            - final_process_physical / final_process_full,
            "full_peak_allocated": int(
                full["process_peak_gpu_allocated_during_prefill_decode"]
            ),
            "physical_peak_allocated": int(
                physical["process_peak_gpu_allocated_during_prefill_conversion"]
            ),
        },
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical", type=Path, required=True)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--gpu", default="RTX 3090")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = compose_fused_cache_e2e(
        json.loads(args.physical.read_text(encoding="utf-8")),
        json.loads(args.full.read_text(encoding="utf-8")),
        gpu=args.gpu,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
