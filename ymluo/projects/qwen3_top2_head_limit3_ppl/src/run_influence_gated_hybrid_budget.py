from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_qwen3_top2_head_limit3_ppl import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    compute_eval_loss,
    install_qwen3_attention_patch,
    pick_input_device,
    prefill_cache,
    read_text_prefix,
    resolve_dtype,
)


@dataclass(frozen=True)
class RunConfig:
    model_name_or_path: str
    text_path: str
    output_dir: str
    prefill_tokens: int
    eval_tokens: int
    chunk_size: int
    eval_chunk_size: int
    max_chars: int
    dtype: str
    device: str
    device_map: str
    attn_implementation: str
    top_fraction: float
    candidate_fraction: float
    qabs_dims: int
    protect_sink_tokens: int
    protect_recent_tokens: int
    landmark_recent: int
    landmark_stride: int
    full_heads: int
    log_every: int


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description="Influence-gated layer/head hybrid KV budget experiment.")
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--text_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prefill_tokens", type=int, default=1024)
    parser.add_argument("--eval_tokens", type=int, default=128)
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--eval_chunk_size", type=int, default=1)
    parser.add_argument("--max_chars", type=int, default=2_000_000)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--top_fraction", type=float, default=0.03)
    parser.add_argument("--candidate_fraction", type=float, default=0.03)
    parser.add_argument("--qabs_dims", type=int, default=8)
    parser.add_argument("--protect_sink_tokens", type=int, default=10)
    parser.add_argument("--protect_recent_tokens", type=int, default=10)
    parser.add_argument("--landmark_recent", type=int, default=512)
    parser.add_argument("--landmark_stride", type=int, default=64)
    parser.add_argument("--full_heads", type=int, default=8)
    parser.add_argument("--log_every", type=int, default=64)
    return RunConfig(**vars(parser.parse_args()))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def qabs_budget(config: RunConfig) -> dict[str, Any]:
    return {
        "type": "qabs8cand3reuse",
        "dims": config.qabs_dims,
        "candidate_fraction": config.candidate_fraction,
        "top_fraction": config.top_fraction,
    }


def headmix_budget(config: RunConfig) -> dict[str, Any]:
    return {
        "type": "headmix_qabs_reuse",
        "full_heads": config.full_heads,
        "dims": config.qabs_dims,
        "candidate_fraction": config.candidate_fraction,
        "top_fraction": config.top_fraction,
    }


def landmark_budget(config: RunConfig) -> dict[str, Any]:
    return {
        "type": "landmark",
        "recent": config.landmark_recent,
        "stride": config.landmark_stride,
        "sink": config.protect_sink_tokens,
    }


def write_budget_map(path: Path, layer_count: int, layers: list[int], budget: dict[str, Any], label: str) -> None:
    layer_set = [int(layer) for layer in layers if 0 <= int(layer) < layer_count]
    path.write_text(
        json.dumps(
            {
                "default": {"type": "full"},
                "layers": {str(layer): budget for layer in layer_set},
                "metadata": {"label": label, "compressed_layers": layer_set, "layer_count": layer_count},
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def run_mode(
    *,
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    config: RunConfig,
    input_device: torch.device,
    mode: str,
    layer_budget_map_path: str,
    shared_past_key_values: Any | None,
    shared_prev_logits: torch.Tensor | None,
) -> tuple[float, float, int, float]:
    return compute_eval_loss(
        model=model,
        input_ids=input_ids,
        prefill_tokens=config.prefill_tokens,
        eval_tokens=config.eval_tokens,
        prefill_chunk_size=config.chunk_size,
        eval_chunk_size=config.eval_chunk_size,
        input_device=input_device,
        mode=mode,
        top_fraction=config.top_fraction,
        max_heads_per_token=3,
        always_keep_self=True,
        protect_sink_tokens=config.protect_sink_tokens,
        protect_recent_tokens=config.protect_recent_tokens,
        load_stats=None,
        qabs_fast_path=False,
        qabs_cuda_final_kernel=True,
        qabs_cuda_candidate_kernel=False,
        qabs_cuda_reuse_select_kernel=False,
        qabs_candidate_selection="topk",
        qabs_threshold_sample_size=256,
        layer_budget_map_path=layer_budget_map_path,
        initial_past_key_values=shared_past_key_values,
        initial_prev_logits=shared_prev_logits,
        clone_initial_cache=True,
        log_every=config.log_every,
    )


def main() -> None:
    config = parse_args()
    output_dir = Path(config.output_dir)
    map_dir = output_dir / "maps"
    output_dir.mkdir(parents=True, exist_ok=True)
    map_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    device = torch.device(config.device)
    dtype = resolve_dtype(config.dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=True)
    text = read_text_prefix(Path(config.text_path), config.max_chars)
    input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
    required_tokens = config.prefill_tokens + config.eval_tokens
    if input_ids.shape[-1] < required_tokens:
        raise ValueError(f"not enough tokens: need {required_tokens}, got {input_ids.shape[-1]}")
    input_ids = input_ids[:, :required_tokens]

    load_kwargs: dict[str, Any] = {"trust_remote_code": True, "torch_dtype": dtype}
    if config.device_map:
        load_kwargs["device_map"] = config.device_map
    if config.attn_implementation:
        load_kwargs["attn_implementation"] = config.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path, **load_kwargs)
    model.eval()
    model.config.use_cache = True
    install_qwen3_attention_patch()
    input_device = pick_input_device(model, device)
    layer_count = int(getattr(model.config, "num_hidden_layers"))

    started = time.perf_counter()
    shared_past_key_values, shared_prev_logits = prefill_cache(
        model, input_ids, config.prefill_tokens, config.chunk_size, input_device
    )
    prefill_seconds = time.perf_counter() - started

    layer_sets: list[tuple[str, list[int]]] = [
        ("pcic_0_6", [0, 6]),
        ("pcic_0_13", [0, 13]),
        ("synthetic_safe_4_5", [4, 5]),
        ("auto_mse_1_2_5", [1, 2, 5]),
        ("mid_7_14", list(range(7, 15))),
    ]
    map_specs: list[tuple[str, list[int], dict[str, Any]]] = []
    for label, layers in layer_sets:
        map_specs.append((f"{label}_landmark", layers, landmark_budget(config)))
        map_specs.append((f"{label}_qabs3set", layers, qabs_budget(config)))
        map_specs.append((f"{label}_headmix{config.full_heads}", layers, headmix_budget(config)))

    rows: list[dict[str, Any]] = []
    baseline_loss, baseline_ppl, baseline_tokens, baseline_seconds = run_mode(
        model=model,
        input_ids=input_ids,
        config=config,
        input_device=input_device,
        mode="baseline",
        layer_budget_map_path="",
        shared_past_key_values=shared_past_key_values,
        shared_prev_logits=shared_prev_logits,
    )
    rows.append(
        {
            "mode": "baseline",
            "compressed_layers": "",
            "budget_type": "full",
            "loss": baseline_loss,
            "ppl": baseline_ppl,
            "token_count": baseline_tokens,
            "seconds": baseline_seconds,
            "ppl_ratio": 1.0,
            "time_ratio": 1.0,
            "prefill_seconds": prefill_seconds,
            "map_path": "",
        }
    )

    for label, layers, budget in map_specs:
        map_path = map_dir / f"{label}.json"
        write_budget_map(map_path, layer_count, layers, budget, label)
        loss, ppl, token_count, seconds = run_mode(
            model=model,
            input_ids=input_ids,
            config=config,
            input_device=input_device,
            mode="layerbudgetattn",
            layer_budget_map_path=str(map_path),
            shared_past_key_values=shared_past_key_values,
            shared_prev_logits=shared_prev_logits,
        )
        rows.append(
            {
                "mode": label,
                "compressed_layers": ",".join(str(layer) for layer in layers),
                "budget_type": budget["type"],
                "loss": loss,
                "ppl": ppl,
                "token_count": token_count,
                "seconds": seconds,
                "ppl_ratio": ppl / baseline_ppl,
                "time_ratio": seconds / baseline_seconds,
                "prefill_seconds": prefill_seconds,
                "map_path": str(map_path),
            }
        )
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)

    write_csv(
        output_dir / "hybrid_budget_results.csv",
        rows,
        [
            "mode",
            "compressed_layers",
            "budget_type",
            "loss",
            "ppl",
            "token_count",
            "seconds",
            "ppl_ratio",
            "time_ratio",
            "prefill_seconds",
            "map_path",
        ],
    )
    (output_dir / "summary.json").write_text(json.dumps({"config": asdict(config), "rows": rows}, indent=2), encoding="utf-8")
    print(json.dumps({"config": asdict(config), "rows": rows}, indent=2), flush=True)


if __name__ == "__main__":
    main()
