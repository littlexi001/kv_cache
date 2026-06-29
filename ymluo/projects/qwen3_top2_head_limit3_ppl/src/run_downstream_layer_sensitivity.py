from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_qwen3_top2_head_limit3_ppl import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    clone_past_key_values,
    install_qwen3_attention_patch,
    pick_input_device,
    prefill_cache,
    resolve_dtype,
)
from run_qabs_downstream_kv_retrieval import Config, build_task, eval_task  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Layer sensitivity scan for retrieval-preserving hybrid KV.")
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tasks", type=int, default=12)
    parser.add_argument("--records_per_task", type=int, default=64)
    parser.add_argument("--seed", type=int, default=202606294)
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--top_fraction", type=float, default=0.05)
    parser.add_argument("--candidate_fraction", type=float, default=0.05)
    parser.add_argument("--protect_sink_tokens", type=int, default=10)
    parser.add_argument("--protect_recent_tokens", type=int, default=10)
    parser.add_argument("--log_every", type=int, default=4)
    return parser.parse_args()


def write_map(path: Path, layer: int, candidate_fraction: float, top_fraction: float) -> None:
    payload = {
        "default": {
            "type": "qabs8cand3reuse",
            "dims": 8,
            "candidate_fraction": candidate_fraction,
            "top_fraction": top_fraction,
        },
        "layers": {str(layer): {"type": "full"}},
        "metadata": {"label": f"default_qabs_layer_{layer}_full"},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    map_dir = output_dir / "maps"
    output_dir.mkdir(parents=True, exist_ok=True)
    map_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    rng = random.Random(args.seed)
    tasks = [build_task(rng, idx, args.records_per_task) for idx in range(args.tasks)]

    device = torch.device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    load_kwargs: dict[str, Any] = {"trust_remote_code": True, "torch_dtype": dtype}
    if args.device_map:
        load_kwargs["device_map"] = args.device_map
    if args.attn_implementation:
        load_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **load_kwargs)
    model.eval()
    model.config.use_cache = True
    install_qwen3_attention_patch()
    input_device = pick_input_device(model, device)
    layer_count = int(getattr(model.config, "num_hidden_layers"))

    base_config = Config(
        model_name_or_path=args.model_name_or_path,
        output_dir=args.output_dir,
        tasks=args.tasks,
        records_per_task=args.records_per_task,
        seed=args.seed,
        chunk_size=args.chunk_size,
        dtype=args.dtype,
        device=args.device,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
        top_fraction=args.top_fraction,
        protect_sink_tokens=args.protect_sink_tokens,
        protect_recent_tokens=args.protect_recent_tokens,
        modes="",
        layer_budget_map_path="",
        log_every=args.log_every,
    )

    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    modes = ["baseline", "qabs8cand5reuse"] + [f"layer_{layer}_full" for layer in range(layer_count)]
    layer_configs: dict[str, Config] = {"baseline": base_config, "qabs8cand5reuse": base_config}
    for layer in range(layer_count):
        map_path = map_dir / f"layer_{layer}_full.json"
        write_map(map_path, layer, args.candidate_fraction, args.top_fraction)
        layer_configs[f"layer_{layer}_full"] = replace(base_config, layer_budget_map_path=str(map_path))

    for task_index, task in enumerate(tasks, start=1):
        if task_index == 1 or task_index == len(tasks) or task_index % args.log_every == 0:
            print(f"task {task_index}/{len(tasks)}", flush=True)
        context_ids = tokenizer(task["context"], return_tensors="pt", add_special_tokens=False)["input_ids"]
        context_cache, context_prev = prefill_cache(
            model,
            context_ids,
            context_ids.shape[-1],
            args.chunk_size,
            input_device,
        )
        for mode in modes:
            eval_mode = "layerbudgetattn" if mode.startswith("layer_") else mode
            result = eval_task(
                model,
                tokenizer,
                task,
                layer_configs[mode],
                input_device,
                eval_mode,
                clone_past_key_values(context_cache),
                context_prev.detach().clone(),
            )
            result["mode"] = mode
            rows.append(result)

    fields = ["task_id", "mode", "target_key", "target_index", "target_label", "pred_label", "correct"] + [
        f"score_{label}" for label in ["A", "B", "C", "D"]
    ]
    with (output_dir / "downstream_layer_sensitivity.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []
    for mode in modes:
        mode_rows = [row for row in rows if row["mode"] == mode]
        correct = sum(int(row["correct"]) for row in mode_rows)
        summary_rows.append({"mode": mode, "correct": correct, "total": len(mode_rows), "accuracy": correct / max(1, len(mode_rows))})
    summary_rows.sort(key=lambda row: (-row["accuracy"], row["mode"]))
    with (output_dir / "summary_by_mode.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["mode", "correct", "total", "accuracy"])
        writer.writeheader()
        writer.writerows(summary_rows)

    summary = {"seconds": time.perf_counter() - started, "layer_count": layer_count, "summary": summary_rows}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary_rows[:8], indent=2), flush=True)


if __name__ == "__main__":
    main()
