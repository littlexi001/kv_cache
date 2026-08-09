from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert paired leakage-free step-state Q profiles into a two-state oracle "
            "trajectory. This isolates retrieval-state conditioning from bridge-generation "
            "errors and source-context effects."
        )
    )
    parser.add_argument("--step_profile", required=True)
    parser.add_argument("--output_profile", required=True)
    parser.add_argument("--summary_path", required=True)
    parser.add_argument(
        "--mode",
        choices=["mean", "token_ensemble"],
        default="mean",
        help="Represent each reasoning state by one mean Q or by all content-token Q vectors.",
    )
    return parser.parse_args()


def aggregate_tokens(q: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not bool(mask.any()):
        raise ValueError("step profile has no valid query token")
    value = q[mask].float().mean(dim=0)
    return F.normalize(value, dim=-1).to(torch.float16)


def main() -> None:
    args = parse_args()
    source: dict[str, Any] = torch.load(
        args.step_profile, map_location="cpu", weights_only=False
    )
    steps = source["steps"]
    grouped: dict[int, list[int]] = defaultdict(list)
    for index, step in enumerate(steps):
        grouped[int(step["query_id"])].append(index)

    pairs = []
    for query_id, indices in sorted(grouped.items()):
        indices.sort(key=lambda index: int(steps[index]["step_index"]))
        if len(indices) != 2 or [int(steps[index]["step_index"]) for index in indices] != [0, 1]:
            raise ValueError(f"query {query_id} does not have exactly steps 0 and 1")
        pairs.append(indices)

    token_counts = [
        (
            int(source["mask"][hop1_index].sum().item()),
            int(source["mask"][hop2_index].sum().item()),
        )
        for hop1_index, hop2_index in pairs
    ]
    states_per_trajectory = (
        2 if args.mode == "mean" else max(left + right for left, right in token_counts)
    )
    shape = (len(pairs), states_per_trajectory, *source["svd_q"].shape[2:])
    trajectory_q = torch.zeros(shape, dtype=torch.float16)
    trajectory_mask = torch.zeros(len(pairs), states_per_trajectory, dtype=torch.bool)
    trajectories = []
    for trajectory_index, (hop1_index, hop2_index) in enumerate(pairs):
        hop1 = steps[hop1_index]
        hop2 = steps[hop2_index]
        bridge = str(hop1["target_output"])
        hop1_gold = [int(item) for item in hop1["target_block_ids"]]
        hop2_gold = [int(item) for item in hop2["target_block_ids"]]
        state_metadata = []
        if args.mode == "mean":
            trajectory_q[trajectory_index, 0].copy_(
                aggregate_tokens(source["svd_q"][hop1_index], source["mask"][hop1_index])
            )
            trajectory_q[trajectory_index, 1].copy_(
                aggregate_tokens(source["svd_q"][hop2_index], source["mask"][hop2_index])
            )
            trajectory_mask[trajectory_index, :2] = True
            state_metadata = [
                {
                    "state_index": 0,
                    "oracle_step_index": 0,
                    "state_token_index": None,
                    "generated_text": "",
                    "bridge_progress": 0.0,
                    "bridge_complete": False,
                },
                {
                    "state_index": 1,
                    "oracle_step_index": 1,
                    "state_token_index": None,
                    "generated_text": bridge,
                    "bridge_progress": 1.0,
                    "bridge_complete": True,
                },
            ]
            first_complete_state = 1
        else:
            output_state = 0
            first_complete_state = int(source["mask"][hop1_index].sum().item())
            for oracle_step_index, step_index in enumerate((hop1_index, hop2_index)):
                valid_q = source["svd_q"][step_index][source["mask"][step_index]]
                for token_index, token_q in enumerate(valid_q):
                    trajectory_q[trajectory_index, output_state].copy_(token_q)
                    trajectory_mask[trajectory_index, output_state] = True
                    state_metadata.append(
                        {
                            "state_index": output_state,
                            "oracle_step_index": oracle_step_index,
                            "state_token_index": token_index,
                            "generated_text": "" if oracle_step_index == 0 else bridge,
                            "bridge_progress": float(oracle_step_index),
                            "bridge_complete": oracle_step_index == 1,
                        }
                    )
                    output_state += 1
        trajectories.append(
            {
                "trajectory_index": trajectory_index,
                "query_id": int(hop1["query_id"]),
                "split": str(hop1["split"]),
                "question": str(hop1["question"]),
                "step_question": str(hop1["step_question"]),
                "bridge_target": bridge,
                "hop1_gold_block_ids": hop1_gold,
                "hop2_gold_block_ids": hop2_gold,
                "state_metadata": state_metadata,
                "generation": {
                    "text": bridge,
                    "raw_target_hit": True,
                    "parsed_target_hit": True,
                    "first_complete_state": first_complete_state,
                    "is_oracle_state": True,
                },
            }
        )

    output = {
        "svd_q": trajectory_q,
        "mask": trajectory_mask,
        "trajectories": trajectories,
        "layers": source["layers"],
        "num_query_heads": source["num_query_heads"],
        "num_kv_heads": source["num_kv_heads"],
        "profile_space": "oracle two-step pre-RoPE Q projected into frozen K-SVD basis",
        "query_vector_mode": (
            "mean of leakage-free step-state content-token Q"
            if args.mode == "mean"
            else "all leakage-free step-state content-token Q as a multi-vector query"
        ),
        "contains_source_context": False,
        "contains_synthetic_vectors": False,
        "uses_oracle_bridge_state": True,
        "oracle_trajectory_mode": args.mode,
    }
    output_path = Path(args.output_profile)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)
    valid = trajectory_q[trajectory_mask].float()
    summary = {
        "source": "paired natural 2WikiMQA oracle step-state Q trajectory",
        "input_step_profile": str(args.step_profile),
        "output_profile": str(output_path),
        "contains_source_context": False,
        "contains_synthetic_vectors": False,
        "uses_oracle_bridge_state": True,
        "oracle_trajectory_mode": args.mode,
        "trajectories": len(trajectories),
        "states": int(trajectory_mask.sum().item()),
        "splits": sorted({str(item["split"]) for item in trajectories}),
        "q_norm_mean": float(valid.norm(dim=-1).mean().item()),
        "q_finite": bool(torch.isfinite(valid).all()),
    }
    summary_path = Path(args.summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
