from __future__ import annotations

import argparse
import json
import math


EXPECTED = {
    "initialization": "from_scratch",
    "sequence_length": 8192,
    "micro_batch": 1,
    "gradient_accumulation": 32,
    "global_batch_size": 256,
    "target_tokens": 100_000_000_000,
    "learning_rate": 1e-4,
}


def validate(args: argparse.Namespace) -> dict[str, int | float | str]:
    gpu_ids = [value.strip() for value in args.gpu_list.split(",") if value.strip()]
    if len(gpu_ids) != 8 or len(set(gpu_ids)) != 8 or any(not value.isdigit() for value in gpu_ids):
        raise ValueError(f"the 16-way protocol requires eight unique GPU IDs, found {gpu_ids}")
    actual = {
        "initialization": args.initialization,
        "sequence_length": args.sequence_length,
        "micro_batch": args.micro_batch,
        "gradient_accumulation": args.gradient_accumulation,
        "global_batch_size": args.global_batch_size,
        "target_tokens": args.target_tokens,
        "learning_rate": args.learning_rate,
    }
    allow_target_override = bool(getattr(args, "allow_target_token_override", False))
    if allow_target_override and args.target_tokens <= 0:
        raise ValueError("an overridden target token count must remain positive")
    mismatches = {
        key: {"expected": value, "actual": actual[key]}
        for key, value in EXPECTED.items()
        if actual[key] != value
        and not (key == "target_tokens" and allow_target_override)
    }
    if mismatches:
        raise ValueError(f"pretraining protocol mismatch: {json.dumps(mismatches, sort_keys=True)}")
    derived_global_batch = args.micro_batch * args.gradient_accumulation * len(gpu_ids)
    if derived_global_batch != args.global_batch_size:
        raise ValueError(
            f"derived global batch {derived_global_batch} != declared {args.global_batch_size}"
        )
    tokens_per_step = args.sequence_length * args.global_batch_size
    steps = math.ceil(args.target_tokens / tokens_per_step)
    return {
        **actual,
        "gpu_count": len(gpu_ids),
        "tokens_per_step": tokens_per_step,
        "optimizer_steps": steps,
        "actual_final_tokens": steps * tokens_per_step,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-list", required=True)
    parser.add_argument("--initialization", required=True)
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--micro-batch", type=int, required=True)
    parser.add_argument("--gradient-accumulation", type=int, required=True)
    parser.add_argument("--global-batch-size", type=int, required=True)
    parser.add_argument("--target-tokens", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument(
        "--allow-target-token-override",
        action="store_true",
        help="Permit a shorter target only for an explicitly separated smoke-test run.",
    )
    args = parser.parse_args()
    print(json.dumps({"ok": True, **validate(args)}, sort_keys=True))


if __name__ == "__main__":
    main()
