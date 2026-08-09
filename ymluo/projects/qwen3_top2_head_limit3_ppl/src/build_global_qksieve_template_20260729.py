from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from run_head_top2_targeted_ppl_20260714 import (
    _qk_metric_projection_factors,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a model-level QKSieve template from disjoint "
            "calibration-request second moments."
        )
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--query_shrinkage",
        type=float,
        default=0.75,
    )
    parser.add_argument(
        "--keep_moments",
        action="store_true",
        help="Retain calibration moments for future template merging.",
    )
    return parser.parse_args()


def load_template(path: Path) -> dict[int, dict[str, Any]]:
    template = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(template, dict):
        raise TypeError(f"{path} is not a layer-template dictionary")
    return {int(layer): values for layer, values in template.items()}


def require_tensor(
    template: dict[str, Any],
    name: str,
    layer_index: int,
) -> torch.Tensor:
    value = template.get(name)
    if not isinstance(value, torch.Tensor):
        raise ValueError(
            f"layer {layer_index} does not contain tensor {name!r}"
        )
    return value.float()


def median_allocation(
    layers: list[dict[str, Any]],
    layer_index: int,
) -> torch.Tensor:
    allocations = torch.stack(
        [
            require_tensor(layer, "allocation", layer_index).to(torch.int16)
            for layer in layers
        ],
        dim=0,
    )
    return allocations.median(dim=0).values.to(torch.int8).contiguous()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.query_shrinkage <= 1.0:
        raise ValueError("query_shrinkage must be in [0, 1]")
    sources = [load_template(path) for path in args.inputs]
    layer_sets = [set(source) for source in sources]
    if any(layer_set != layer_sets[0] for layer_set in layer_sets[1:]):
        raise ValueError("all calibration templates must cover the same layers")

    output: dict[int, dict[str, Any]] = {}
    for layer_index in sorted(layer_sets[0]):
        source_layers = [source[layer_index] for source in sources]
        transforms = {str(layer["transform"]) for layer in source_layers}
        objectives = {
            str(layer["allocation_objective"]) for layer in source_layers
        }
        if transforms != {"qk_metric"}:
            raise ValueError(
                f"layer {layer_index} is not consistently QK-metric"
            )
        if len(objectives) != 1:
            raise ValueError(
                f"layer {layer_index} allocation objectives differ"
            )
        key_moment = torch.stack(
            [
                require_tensor(layer, "key_second_moment", layer_index)
                for layer in source_layers
            ],
            dim=0,
        ).mean(dim=0)
        query_moment = torch.stack(
            [
                require_tensor(layer, "query_second_moment", layer_index)
                for layer in source_layers
            ],
            dim=0,
        ).mean(dim=0)
        query_basis, key_basis = _qk_metric_projection_factors(
            key_moment,
            query_moment,
            projection_dim=128,
            query_shrinkage=args.query_shrinkage,
        )
        spectral_weights = torch.linalg.eigvalsh(
            key_moment
        ).flip(-1).contiguous()
        reference = source_layers[0]
        layer_output = {
            "basis": key_basis.to(torch.float16).contiguous(),
            "query_basis": query_basis.to(torch.float16).contiguous(),
            "allocation": median_allocation(
                source_layers,
                layer_index,
            ),
            "spectral_weights": spectral_weights,
            "key_mean": None,
            "scale_metrics": None,
            "exact_query_mean": None,
            "proxy_query_mean": None,
            "sharedtail_coordinate_rms": None,
            "sharedtail_coordinate_amplitude": None,
            "transform": "qk_metric",
            "allocation_objective": objectives.pop(),
            "key_centered": bool(reference.get("key_centered", False)),
            "sharedtail": bool(reference.get("sharedtail", False)),
            "metric_scale": bool(reference.get("metric_scale", False)),
            "metric_scale_shrinkage": str(
                reference.get("metric_scale_shrinkage", "none")
            ),
            "metric_scale_oas_alpha": None,
            "mean_score_bias": bool(
                reference.get("mean_score_bias", False)
            ),
            "qk_metric_shrinkage": float(args.query_shrinkage),
            "calibration_source_count": len(sources),
        }
        if args.keep_moments:
            layer_output.update(
                key_second_moment=key_moment,
                query_second_moment=query_moment,
            )
        output[layer_index] = layer_output

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(
        f"saved {len(output)} layers from {len(sources)} sources "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
