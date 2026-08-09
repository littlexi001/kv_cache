from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rewrite only the per-head bit allocation of a frozen "
            "QKSieve template while preserving its QK-balanced bases."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allocation",
        default="4,4,1,0,0,0,0,0",
        help="Eight comma-separated bit widths, one per 16-D band.",
    )
    return parser.parse_args()


def parse_allocation(value: str) -> torch.Tensor:
    bits = [int(item.strip()) for item in value.split(",")]
    if len(bits) != 8:
        raise ValueError("allocation must contain exactly eight bands")
    if any(bit not in {0, 1, 2, 4, 8} for bit in bits):
        raise ValueError("band widths must be one of 0, 1, 2, 4, or 8")
    return torch.tensor(bits, dtype=torch.int8)


def main() -> None:
    args = parse_args()
    allocation = parse_allocation(args.allocation)
    template = torch.load(
        args.input,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(template, dict):
        raise TypeError("template must be a layer dictionary")

    output: dict[int, dict[str, Any]] = {}
    head_count = 0
    for raw_layer_index, raw_layer in template.items():
        layer_index = int(raw_layer_index)
        if not isinstance(raw_layer, dict):
            raise TypeError(f"layer {layer_index} is not a dictionary")
        old_allocation = raw_layer.get("allocation")
        if not isinstance(old_allocation, torch.Tensor):
            raise ValueError(f"layer {layer_index} has no allocation tensor")
        if old_allocation.ndim < 1 or old_allocation.shape[-1] != 8:
            raise ValueError(
                f"layer {layer_index} allocation must end in eight bands"
            )
        rewritten = dict(raw_layer)
        rewritten["allocation"] = (
            allocation.view(
                (1,) * (old_allocation.ndim - 1)
                + (allocation.shape[0],)
            )
            .expand(*old_allocation.shape)
            .contiguous()
        )
        rewritten["allocation_override"] = allocation.tolist()
        output[layer_index] = rewritten
        head_count += int(old_allocation.numel() // allocation.numel())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    mean_bits = float(allocation.sum().item())
    active_bands = int((allocation > 0).sum().item())
    logical_index_ratio = (
        2.0 * mean_bits + 2.0 * active_bands
    ) / 512.0
    print(
        f"saved {len(output)} layers / {head_count} KV heads to "
        f"{args.output}; mean_bits={mean_bits:.1f}, "
        f"ideal_index_ratio={logical_index_ratio:.6f}"
    )


if __name__ == "__main__":
    main()
