from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate head- and layer-level temporal reuse gates."
    )
    parser.add_argument("--trace_path", required=True, type=Path)
    parser.add_argument("--output_path", required=True, type=Path)
    return parser.parse_args()


def tensor(record: dict[str, Any], name: str) -> torch.Tensor:
    return torch.as_tensor(record[name]).float().reshape(-1)


def gate_summary(
    records: list[dict[str, Any]],
    gate_name: str,
    gate_fn: Any,
) -> dict[str, float | str]:
    accepted_heads = 0
    total_heads = 0
    bad_heads = 0
    low_mass_heads = 0
    accepted_layer_steps = 0
    total_layer_steps = 0
    bad_accepted_layer_steps = 0
    for record in records:
        available = tensor(
            record, "temporal_reuse_output_trace_available"
        ) > 0.5
        if not bool(available.any()):
            continue
        gate = gate_fn(record).bool() & available
        error = tensor(record, "temporal_reuse_output_relative_error")
        mass = tensor(record, "temporal_reuse_fresh_attention_mass")
        accepted_heads += int(gate.sum().item())
        total_heads += int(available.sum().item())
        bad_heads += int((gate & (error > 0.02)).sum().item())
        low_mass_heads += int((gate & (mass < 0.95)).sum().item())
        total_layer_steps += 1
        layer_accepted = bool(gate.all())
        accepted_layer_steps += int(layer_accepted)
        bad_accepted_layer_steps += int(
            layer_accepted and bool((error > 0.02).any())
        )
    accepted = max(1, accepted_heads)
    accepted_layers = max(1, accepted_layer_steps)
    return {
        "gate": gate_name,
        "head_accept_rate": accepted_heads / max(1, total_heads),
        "bad_output_rate_among_accepted_heads": bad_heads / accepted,
        "low_mass_rate_among_accepted_heads": low_mass_heads / accepted,
        "layer_all_heads_accept_rate": (
            accepted_layer_steps / max(1, total_layer_steps)
        ),
        "bad_layer_rate_among_accepted_layers": (
            bad_accepted_layer_steps / accepted_layers
        ),
    }


def main() -> None:
    args = parse_args()
    records = torch.load(
        args.trace_path,
        map_location="cpu",
        weights_only=False,
    )
    required = {
        "temporal_reuse_output_trace_available",
        "temporal_reuse_output_relative_error",
        "temporal_reuse_fresh_attention_mass",
        "temporal_expected_core_crossing_fraction",
        "temporal_core_margin_ratio_top25",
        "temporal_sampled_reuse_mass_estimate",
    }
    records = [
        record for record in records if required.issubset(record)
    ]
    if not records:
        raise RuntimeError("trace contains no complete temporal records")

    rows: list[dict[str, float | str]] = []
    for threshold in (
        0.25,
        0.5,
        1.0,
        2.0,
        4.0,
        8.0,
        16.0,
        32.0,
        64.0,
        128.0,
    ):
        rows.append(
            gate_summary(
                records,
                f"crossing_fraction<={threshold:g}",
                lambda record, value=threshold: tensor(
                    record,
                    "temporal_expected_core_crossing_fraction",
                )
                <= value,
            )
        )
    for threshold in (0.25, 0.5, 1.0, 2.0, 4.0):
        rows.append(
            gate_summary(
                records,
                f"core_margin_ratio>={threshold:g}",
                lambda record, value=threshold: tensor(
                    record,
                    "temporal_core_margin_ratio_top25",
                )
                >= value,
            )
        )
    for threshold in (0.90, 0.95, 0.97, 0.99, 0.995, 0.999):
        rows.append(
            gate_summary(
                records,
                f"sampled_mass>={threshold:g}",
                lambda record, value=threshold: tensor(
                    record,
                    "temporal_sampled_reuse_mass_estimate",
                )
                >= value,
            )
        )
    for mass_threshold in (0.90, 0.95, 0.97, 0.99):
        for risk_threshold in (1.0, 2.0, 4.0, 8.0, 16.0):
            rows.append(
                gate_summary(
                    records,
                    (
                        f"sampled_mass>={mass_threshold:g}"
                        f"&crossing_fraction<={risk_threshold:g}"
                    ),
                    lambda record, mass=mass_threshold, risk=risk_threshold: (
                        tensor(
                            record,
                            "temporal_sampled_reuse_mass_estimate",
                        )
                        >= mass
                    )
                    & (
                        tensor(
                            record,
                            "temporal_expected_core_crossing_fraction",
                        )
                        <= risk
                    ),
                )
            )

    pareto = sorted(
        (
            row
            for row in rows
            if row["bad_output_rate_among_accepted_heads"] <= 0.01
        ),
        key=lambda row: float(row["head_accept_rate"]),
        reverse=True,
    )
    output = {
        "trace_path": str(args.trace_path),
        "records": len(records),
        "rows": rows,
        "best_head_accept_with_bad_rate_le_1pct": (
            pareto[0] if pareto else None
        ),
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
