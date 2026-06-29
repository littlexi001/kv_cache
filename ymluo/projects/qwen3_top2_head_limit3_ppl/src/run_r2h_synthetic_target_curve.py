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
    profile_csv: str
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
    protect_sink_tokens: int
    synthetic_recent: int
    synthetic_prototypes: int
    synthetic_method: str
    log_every: int


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Evaluate staged R2H synthetic target-compression curve.")
    parser.add_argument("--model_name_or_path", default="/home/fdong/hrj/prove/Qwen3-0.6B")
    parser.add_argument("--text_path", required=True)
    parser.add_argument("--profile_csv", required=True)
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
    parser.add_argument("--protect_sink_tokens", type=int, default=10)
    parser.add_argument("--synthetic_recent", type=int, default=64)
    parser.add_argument("--synthetic_prototypes", type=int, default=8)
    parser.add_argument("--synthetic_method", choices=["mean", "mass"], default="mass")
    parser.add_argument("--log_every", type=int, default=32)
    return Config(**vars(parser.parse_args()))


def read_profile_layer_risk(path: Path) -> list[tuple[int, float]]:
    by_layer: dict[int, list[float]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            layer = int(row["layer"])
            remote = float(row.get("remote_mass_mean", 0.0) or 0.0)
            qabs_energy = float(row.get("qabs_candidate_energy", 0.0) or 0.0)
            reuse = float(row.get("reuse_previous_energy", 0.0) or 0.0)
            risk = 0.45 * remote + 0.40 * (1.0 - qabs_energy) + 0.15 * (1.0 - reuse)
            by_layer.setdefault(layer, []).append(risk)
    layer_risks = []
    for layer, risks in by_layer.items():
        # Tail-sensitive layer score: average plus a small worst-head penalty.
        layer_risks.append((layer, sum(risks) / len(risks) + 0.25 * max(risks)))
    return sorted(layer_risks, key=lambda item: item[1])


def write_map(path: Path, layers: list[int], config: Config, label: str) -> None:
    budget = {
        "type": "synthetic",
        "recent": config.synthetic_recent,
        "prototypes": config.synthetic_prototypes,
        "method": config.synthetic_method,
        "sink": config.protect_sink_tokens,
    }
    path.write_text(
        json.dumps(
            {
                "default": {"type": "full"},
                "layers": {str(layer): budget for layer in layers},
                "metadata": {"label": label, "synthetic_layers": layers},
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_mode(
    *,
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    config: Config,
    input_device: torch.device,
    mode: str,
    map_path: str,
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
        top_fraction=0.03,
        max_heads_per_token=3,
        always_keep_self=True,
        protect_sink_tokens=config.protect_sink_tokens,
        protect_recent_tokens=config.synthetic_recent,
        load_stats=None,
        qabs_fast_path=False,
        qabs_cuda_final_kernel=False,
        qabs_cuda_candidate_kernel=False,
        qabs_cuda_reuse_select_kernel=False,
        qabs_candidate_selection="topk",
        qabs_threshold_sample_size=256,
        layer_budget_map_path=map_path,
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

    layer_risks = read_profile_layer_risk(Path(config.profile_csv))
    (output_dir / "layer_risk_order.json").write_text(json.dumps(layer_risks, indent=2), encoding="utf-8")

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

    rows: list[dict[str, Any]] = []
    baseline_loss, baseline_ppl, baseline_tokens, baseline_seconds = run_mode(
        model=model,
        input_ids=input_ids,
        config=config,
        input_device=input_device,
        mode="baseline",
        map_path="",
        shared_past_key_values=shared_past_key_values,
        shared_prev_logits=shared_prev_logits,
    )
    rows.append(
        {
            "mode": "baseline",
            "target_compression": 0.0,
            "synthetic_layers": "",
            "synthetic_layer_count": 0,
            "estimated_keep_fraction": 1.0,
            "estimated_compression": 0.0,
            "loss": baseline_loss,
            "ppl": baseline_ppl,
            "ppl_ratio": 1.0,
            "seconds": baseline_seconds,
            "time_ratio": 1.0,
            "prefill_seconds": prefill_seconds,
            "token_count": baseline_tokens,
            "map_path": "",
        }
    )

    compressed_keep = (config.protect_sink_tokens + config.synthetic_recent + config.synthetic_prototypes) / max(1, config.prefill_tokens)
    stages = [(0.50, 15), (0.70, 21), (0.80, 24), (0.90, 26)]
    for target, layer_budget in stages:
        selected_layers = [layer for layer, _risk in layer_risks[: min(layer_budget, layer_count)]]
        label = f"target{int(target * 100)}_synth{len(selected_layers)}"
        map_path = map_dir / f"{label}.json"
        write_map(map_path, selected_layers, config, label)
        loss, ppl, token_count, seconds = run_mode(
            model=model,
            input_ids=input_ids,
            config=config,
            input_device=input_device,
            mode="layerbudgetattn",
            map_path=str(map_path),
            shared_past_key_values=shared_past_key_values,
            shared_prev_logits=shared_prev_logits,
        )
        keep_fraction = (layer_count - len(selected_layers)) / layer_count + len(selected_layers) / layer_count * compressed_keep
        row = {
            "mode": label,
            "target_compression": target,
            "synthetic_layers": ",".join(str(layer) for layer in selected_layers),
            "synthetic_layer_count": len(selected_layers),
            "estimated_keep_fraction": keep_fraction,
            "estimated_compression": 1.0 - keep_fraction,
            "loss": loss,
            "ppl": ppl,
            "ppl_ratio": ppl / baseline_ppl,
            "seconds": seconds,
            "time_ratio": seconds / baseline_seconds,
            "prefill_seconds": prefill_seconds,
            "token_count": token_count,
            "map_path": str(map_path),
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    fields = list(rows[0].keys())
    write_csv(output_dir / "synthetic_target_curve.csv", rows, fields)
    (output_dir / "summary.json").write_text(json.dumps({"config": asdict(config), "rows": rows}, indent=2), encoding="utf-8")
    print(json.dumps({"rows": rows}, indent=2), flush=True)


if __name__ == "__main__":
    main()
