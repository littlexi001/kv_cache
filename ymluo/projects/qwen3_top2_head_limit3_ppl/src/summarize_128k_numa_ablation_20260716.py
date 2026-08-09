from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, type=Path)
    args = parser.parse_args()
    local = json.loads(
        (args.input_dir / "local_gpu0_mem0.json").read_text(encoding="utf-8")
    )
    remote = json.loads(
        (args.input_dir / "remote_gpu4_mem0.json").read_text(encoding="utf-8")
    )
    micro_local = json.loads(
        (args.input_dir / "micro_local_gpu0_mem0.json").read_text(encoding="utf-8")
    )
    micro_remote = json.loads(
        (args.input_dir / "micro_remote_gpu4_mem0.json").read_text(encoding="utf-8")
    )
    if local["target_token_ids"] != remote["target_token_ids"]:
        raise ValueError("NUMA comparison target tokens do not match")
    output = {
        "protocol": (
            "CPU and pinned memory fixed on NUMA node 0; compare local GPU0-3 "
            "against remote GPU4-7"
        ),
        "local_online_seconds": local["online_seconds"],
        "remote_online_seconds": remote["online_seconds"],
        "remote_over_local_online": remote["online_seconds"]
        / local["online_seconds"],
        "local_conversion_seconds": local["cache_conversion_seconds"],
        "remote_conversion_seconds": remote["cache_conversion_seconds"],
        "remote_over_local_conversion": remote["cache_conversion_seconds"]
        / local["cache_conversion_seconds"],
        "local_ppl": local["ppl"],
        "remote_ppl": remote["ppl"],
    }
    for key in (
        "mapped_gather_ms_per_layer",
        "mapped_host_fill_cache_ms_per_layer",
        "mapped_host_fill_cache_attention_ms_per_layer",
        "gqa_hybrid_mapped_attention_cache_update_ms_per_layer",
        "gather_resident_cache_sdpa_ms_per_layer",
    ):
        local_value = float(micro_local[key])
        remote_value = float(micro_remote[key])
        output[f"micro_local_{key}"] = local_value
        output[f"micro_remote_{key}"] = remote_value
        output[f"micro_remote_over_local_{key}"] = remote_value / local_value
    device_only_key = "packed_final_attention_ms_per_layer"
    device_only_ratio = float(micro_remote[device_only_key]) / float(
        micro_local[device_only_key]
    )
    output["micro_remote_over_local_device_only_attention"] = device_only_ratio
    mapped_key = "mapped_host_fill_cache_attention_ms_per_layer"
    output["topology_penalty_normalized_by_device_attention"] = (
        float(micro_remote[mapped_key]) / float(micro_local[mapped_key])
    ) / device_only_ratio
    (args.input_dir / "summary.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output))


if __name__ == "__main__":
    main()
