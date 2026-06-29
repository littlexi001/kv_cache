from __future__ import annotations

import argparse
import csv
import json
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
class Config:
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
    local_recent: int
    log_every: int


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="R2H-KV prior smoke experiment with fixed layer-budget maps.")
    parser.add_argument("--model_name_or_path", default="/home/fdong/hrj/prove/Qwen3-0.6B")
    parser.add_argument("--text_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prefill_tokens", type=int, default=2048)
    parser.add_argument("--eval_tokens", type=int, default=64)
    parser.add_argument("--chunk_size", type=int, default=64)
    parser.add_argument("--eval_chunk_size", type=int, default=1)
    parser.add_argument("--max_chars", type=int, default=2_000_000)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--top_fraction", type=float, default=0.03)
    parser.add_argument("--candidate_fraction", type=float, default=0.03)
    parser.add_argument("--qabs_dims", type=int, default=8)
    parser.add_argument("--protect_sink_tokens", type=int, default=10)
    parser.add_argument("--protect_recent_tokens", type=int, default=10)
    parser.add_argument("--local_recent", type=int, default=512)
    parser.add_argument("--log_every", type=int, default=32)
    return Config(**vars(parser.parse_args()))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def qabs_budget(config: Config) -> dict[str, Any]:
    return {
        "type": "qabs8cand3reuse",
        "dims": config.qabs_dims,
        "candidate_fraction": config.candidate_fraction,
        "top_fraction": config.top_fraction,
    }


def recent_budget(config: Config) -> dict[str, Any]:
    return {"type": "recent", "recent": config.local_recent}


def headmix_budget(config: Config) -> dict[str, Any]:
    return {
        "type": "headmix_qabs_reuse",
        "full_heads": 8,
        "dims": config.qabs_dims,
        "candidate_fraction": config.candidate_fraction,
        "top_fraction": config.top_fraction,
    }


def write_map(path: Path, layer_count: int, layer_budgets: dict[int, dict[str, Any]], label: str) -> None:
    layers = {
        str(layer): budget
        for layer, budget in sorted(layer_budgets.items())
        if 0 <= int(layer) < layer_count
    }
    path.write_text(
        json.dumps(
            {
                "default": {"type": "full"},
                "layers": layers,
                "metadata": {"label": label, "layer_count": layer_count},
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def run_mode(
    *,
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    config: Config,
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

    local = recent_budget(config)
    qabs = qabs_budget(config)
    headmix = headmix_budget(config)
    policies: dict[str, dict[int, dict[str, Any]]] = {
        "r2h_local_shallow_tail": {
            **{layer: local for layer in range(0, 6)},
            **{layer: local for layer in range(23, 28)},
        },
        "r2h_mid_qabs": {layer: qabs for layer in range(6, 14)},
        "r2h_prior_local_midqabs": {
            **{layer: local for layer in range(0, 6)},
            **{layer: qabs for layer in range(6, 14)},
            **{layer: local for layer in range(23, 28)},
        },
        "r2h_refined_safe": {
            0: qabs,
            13: qabs,
        },
        "r2h_refined_mid_headmix": {
            **{layer: headmix for layer in range(7, 15)},
        },
        "r2h_refined_hybrid": {
            0: qabs,
            **{layer: headmix for layer in [4, 5]},
            **{layer: headmix for layer in range(7, 15)},
        },
        "uniform_local_all": {layer: local for layer in range(layer_count)},
        "uniform_qabs_all": {layer: qabs for layer in range(layer_count)},
    }

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
            "budget_summary": "full",
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

    for label, layer_budgets in policies.items():
        map_path = map_dir / f"{label}.json"
        write_map(map_path, layer_count, layer_budgets, label)
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
        row = {
            "mode": label,
            "compressed_layers": ",".join(str(layer) for layer in sorted(layer_budgets)),
            "budget_summary": "+".join(sorted({budget["type"] for budget in layer_budgets.values()})),
            "loss": loss,
            "ppl": ppl,
            "token_count": token_count,
            "seconds": seconds,
            "ppl_ratio": ppl / baseline_ppl,
            "time_ratio": seconds / baseline_seconds,
            "prefill_seconds": prefill_seconds,
            "map_path": str(map_path),
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    fields = [
        "mode",
        "compressed_layers",
        "budget_summary",
        "loss",
        "ppl",
        "token_count",
        "seconds",
        "ppl_ratio",
        "time_ratio",
        "prefill_seconds",
        "map_path",
    ]
    write_csv(output_dir / "r2h_prior_results.csv", rows, fields)
    (output_dir / "summary.json").write_text(json.dumps({"config": asdict(config), "rows": rows}, indent=2), encoding="utf-8")
    print(json.dumps({"config": asdict(config), "rows": rows}, indent=2), flush=True)


if __name__ == "__main__":
    main()
