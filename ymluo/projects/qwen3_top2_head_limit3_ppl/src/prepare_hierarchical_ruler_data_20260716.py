from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from run_hierarchical_ruler_probe_20260716 import load_examples, write_examples_jsonl


def parse_list(spec: str) -> list[str]:
    return [item.strip() for item in spec.split(",") if item.strip()]


def build_manifest(args: argparse.Namespace) -> dict[str, object]:
    tasks = parse_list(args.ruler_tasks)
    lengths = [int(value) for value in parse_list(args.ruler_lengths)]
    return {
        "schema": "hierarchical_ruler_examples_v1",
        "model_name_or_path": str(args.model_name_or_path),
        "lm_eval_path": str(args.lm_eval_path),
        "ruler_tasks": tasks,
        "ruler_lengths": lengths,
        "max_samples_per_task": int(args.max_samples_per_task),
        "max_new_tokens_override": int(args.max_new_tokens_override),
        "seed": int(args.seed),
        "ruler_hotpot_parquet": str(args.ruler_hotpot_parquet),
        "expected_examples": len(tasks) * len(lengths) * int(args.max_samples_per_task),
    }


def manifest_path(output: Path) -> Path:
    return output.with_suffix(output.suffix + ".manifest.json")


def validate_existing_cache(output: Path, expected_manifest: dict[str, object]) -> None:
    sidecar = manifest_path(output)
    if not sidecar.is_file():
        raise RuntimeError(f"existing RULER cache lacks manifest: {sidecar}")
    observed_manifest = json.loads(sidecar.read_text(encoding="utf-8"))
    if observed_manifest != expected_manifest:
        raise RuntimeError(
            "existing RULER cache manifest differs from requested protocol: "
            f"{sidecar}"
        )
    with output.open("r", encoding="utf-8") as handle:
        existing = sum(bool(line.strip()) for line in handle)
    expected = int(expected_manifest["expected_examples"])
    if existing != expected:
        raise RuntimeError(
            f"existing RULER cache has {existing} rows, expected {expected}: {output}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--lm_eval_path", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ruler_tasks", required=True)
    parser.add_argument("--ruler_lengths", required=True)
    parser.add_argument("--max_samples_per_task", required=True, type=int)
    parser.add_argument("--max_new_tokens_override", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ruler_hotpot_parquet",
        default=(
            "/home/fdong/ymluo/datasets/ruler_sources/hotpotqa/distractor/"
            "validation-00000-of-00001.parquet"
        ),
    )
    args = parser.parse_args()
    manifest = build_manifest(args)
    expected = int(manifest["expected_examples"])
    if args.output.is_file() and args.output.stat().st_size > 0:
        validate_existing_cache(args.output, manifest)
        print(f"using existing RULER examples: {args.output}")
        return
    examples = load_examples(
        SimpleNamespace(
            **vars(args),
            examples_jsonl=None,
            num_shards=1,
            shard_index=0,
        )
    )
    if len(examples) != expected:
        raise RuntimeError(f"expected {expected} RULER examples, generated {len(examples)}")
    write_examples_jsonl(args.output, examples)
    sidecar = manifest_path(args.output)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    temporary = sidecar.with_suffix(sidecar.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(sidecar)
    print(f"wrote {len(examples)} RULER examples to {args.output}")


if __name__ == "__main__":
    main()
