from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def analyze_trace(
    trace_path: Path,
    top_fractions: tuple[float, ...],
    enclosure_fractions: tuple[float, ...],
    device: torch.device,
) -> dict[str, object]:
    trace = torch.load(trace_path, map_location="cpu", weights_only=False)
    by_layer: dict[int, list[dict[str, object]]] = defaultdict(list)
    for record in trace["records"]:
        by_layer[int(record["layer"])].append(record)

    metrics: dict[str, list[float]] = defaultdict(list)
    pair_count = 0
    for layer, records in sorted(by_layer.items()):
        records.sort(key=lambda item: int(item["step"]))
        key = records[0]["key"][0].to(device=device, dtype=torch.float16)
        query = torch.stack(
            [record["query"][0, :, 0] for record in records]
        ).to(device=device, dtype=torch.float16)
        scaling = float(records[0]["scaling"])
        kv_head_count = key.shape[0]
        group_count = query.shape[1] // kv_head_count
        query = query.reshape(query.shape[0], kv_head_count, group_count, -1)
        scores = torch.einsum("thgd,hnd->thgn", query, key).float() * scaling
        key_norm = key.float().square().sum(dim=-1).sqrt()
        key_count = key.shape[1]

        top_counts = {
            fraction: max(1, math.ceil(fraction * key_count))
            for fraction in top_fractions
        }
        enclosure_counts = {
            fraction: max(1, math.ceil(fraction * key_count))
            for fraction in enclosure_fractions
        }
        max_top_count = max(top_counts.values())
        max_enclosure_count = max(enclosure_counts.values())

        for step in range(1, query.shape[0]):
            delta_norm = (
                query[step].float() - query[step - 1].float()
            ).square().sum(dim=-1).sqrt()
            for kv_head in range(kv_head_count):
                norm = key_norm[kv_head]
                for group in range(group_count):
                    previous = scores[step - 1, kv_head, group]
                    current = scores[step, kv_head, group]
                    bound = scaling * delta_norm[kv_head, group] * norm
                    previous_order = torch.topk(
                        previous, k=max_enclosure_count, sorted=True
                    ).indices
                    current_order = torch.topk(
                        current, k=max_top_count, sorted=True
                    ).indices
                    log_partition = torch.logsumexp(current, dim=0)

                    for top_fraction, top_count in top_counts.items():
                        current_top = current_order[:top_count]
                        for enclosure_fraction, enclosure_count in enclosure_counts.items():
                            if enclosure_count < top_count:
                                continue
                            previous_enclosure = previous_order[:enclosure_count]
                            inside_mask = torch.zeros(
                                key_count, dtype=torch.bool, device=device
                            )
                            inside_mask[previous_enclosure] = True
                            containment = inside_mask[current_top].float().mean().item()
                            full_containment = float(containment == 1.0)

                            lower_inside = (
                                previous[previous_enclosure]
                                - bound[previous_enclosure]
                            )
                            kth_inside_lower = torch.topk(
                                lower_inside, k=top_count, sorted=False
                            ).values.min()
                            outside_upper = (previous + bound).masked_fill(
                                inside_mask, -torch.inf
                            ).max()
                            certificate_margin = (
                                kth_inside_lower - outside_upper
                            ).item()
                            certified = float(certificate_margin >= 0.0)
                            mass = torch.exp(
                                torch.logsumexp(current[previous_enclosure], dim=0)
                                - log_partition
                            ).item()

                            prefix = (
                                f"top{100 * top_fraction:g}_in_"
                                f"prev{100 * enclosure_fraction:g}"
                            )
                            metrics[f"{prefix}_recall"].append(containment)
                            metrics[f"{prefix}_full_containment"].append(
                                full_containment
                            )
                            metrics[f"{prefix}_attention_mass"].append(mass)
                            metrics[f"{prefix}_certificate_rate"].append(certified)
                            metrics[f"{prefix}_certificate_margin"].append(
                                certificate_margin
                            )
                    pair_count += 1

        del scores, query, key, key_norm
        torch.cuda.empty_cache()

    return {
        "trace": str(trace_path),
        "layers": len(by_layer),
        "query_pairs": pair_count,
        "metrics": {name: _mean(values) for name, values in sorted(metrics.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", required=True)
    parser.add_argument("--top_fractions", default="0.005,0.01,0.02")
    parser.add_argument("--enclosure_fractions", default="0.01,0.02,0.04,0.08")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_path", type=Path, required=True)
    args = parser.parse_args()

    top_fractions = tuple(float(item) for item in args.top_fractions.split(","))
    enclosure_fractions = tuple(
        float(item) for item in args.enclosure_fractions.split(",")
    )
    reports = [
        analyze_trace(
            Path(path),
            top_fractions,
            enclosure_fractions,
            torch.device(args.device),
        )
        for path in args.traces.split(",")
    ]
    aggregate: dict[str, list[float]] = defaultdict(list)
    for report in reports:
        for name, value in report["metrics"].items():
            aggregate[name].append(float(value))
    output = {
        "protocol": {
            "top_fractions": top_fractions,
            "enclosure_fractions": enclosure_fractions,
            "certificate": (
                "k-th largest lower score bound inside the previous enclosure "
                "exceeds every outside upper score bound"
            ),
        },
        "traces": reports,
        "aggregate": {
            name: _mean(values) for name, values in sorted(aggregate.items())
        },
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
