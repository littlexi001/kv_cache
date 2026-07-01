from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from evaluate_qwen3_top2_head_limit3_ppl import (
    AutoModelForCausalLM,
    AutoTokenizer,
    model_forward,
    pick_input_device,
    read_text_prefix,
    resolve_dtype,
)


_ORIGINAL_EAGER_ATTENTION_FORWARD: Any | None = None
_ACTIVE_COLLECTOR: "SyntheticFitCollector | None" = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit synthetic K/V prototypes and evaluate heldout attention-output reconstruction.")
    parser.add_argument("--model_name_or_path", default="/home/fdong/hrj/prove/Qwen3-0.6B")
    parser.add_argument("--text_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prefill_tokens", type=int, default=2048)
    parser.add_argument("--calib_tokens", type=int, default=32)
    parser.add_argument("--heldout_tokens", type=int, default=32)
    parser.add_argument("--chunk_size", type=int, default=64)
    parser.add_argument("--recent_tokens", type=int, default=64)
    parser.add_argument("--sink_tokens", type=int, default=10)
    parser.add_argument("--prototype_counts", default="4,8,16,32")
    parser.add_argument("--layers", default="0,4,5,7,8,13,14,16,20,27")
    parser.add_argument("--max_chars", type=int, default=2_000_000)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--ridge_lambda", type=float, default=1e-3)
    return parser.parse_args()


def parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class SyntheticFitCollector:
    def __init__(
        self,
        layers: set[int],
        prefill_tokens: int,
        calib_tokens: int,
        heldout_tokens: int,
        sink_tokens: int,
        recent_tokens: int,
    ) -> None:
        self.layers = layers
        self.prefill_tokens = prefill_tokens
        self.calib_start = prefill_tokens
        self.calib_end = prefill_tokens + calib_tokens
        self.heldout_end = self.calib_end + heldout_tokens
        self.sink_tokens = sink_tokens
        self.recent_tokens = recent_tokens
        self.remote_k: dict[int, torch.Tensor] = {}
        self.remote_v: dict[int, torch.Tensor] = {}
        self.calib_q: dict[int, list[torch.Tensor]] = defaultdict(list)
        self.heldout_q: dict[int, list[torch.Tensor]] = defaultdict(list)

    def observe(self, layer: int, query_token: int, query_states: torch.Tensor, key_states: torch.Tensor, value_states: torch.Tensor, query_index: int) -> None:
        if layer not in self.layers:
            return
        if not (self.calib_start <= query_token < self.heldout_end):
            return
        if key_states.shape[1] != query_states.shape[1]:
            repeat_groups = query_states.shape[1] // key_states.shape[1]
            key_states = key_states.repeat_interleave(repeat_groups, dim=1)
            value_states = value_states.repeat_interleave(repeat_groups, dim=1)

        remote_start = min(max(0, self.sink_tokens), self.prefill_tokens)
        remote_end = max(remote_start, self.prefill_tokens - max(0, self.recent_tokens))
        if layer not in self.remote_k:
            self.remote_k[layer] = key_states[0, :, remote_start:remote_end, :].detach().float().cpu()
            self.remote_v[layer] = value_states[0, :, remote_start:remote_end, :].detach().float().cpu()

        q = query_states[0, :, query_index, :].detach().float().cpu()
        if query_token < self.calib_end:
            self.calib_q[layer].append(q)
        else:
            self.heldout_q[layer].append(q)


def _patched_eager_attention_forward(
    module: torch.nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float | None = None,
    dropout: float = 0.0,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scaling is None:
        scaling = float(getattr(module, "scaling", 1.0 / math.sqrt(query_states.shape[-1])))
    if key_states.shape[1] != query_states.shape[1]:
        repeat_groups = query_states.shape[1] // key_states.shape[1]
        key_states = key_states.repeat_interleave(repeat_groups, dim=1)
        value_states = value_states.repeat_interleave(repeat_groups, dim=1)
    scores = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask[:, :, :, : scores.shape[-1]]
    if _ACTIVE_COLLECTOR is not None:
        layer = int(getattr(module, "layer_idx", 0))
        query_count = scores.shape[-2]
        key_count = scores.shape[-1]
        chunk_query_start = key_count - query_count
        for query_index in range(query_count):
            _ACTIVE_COLLECTOR.observe(layer, chunk_query_start + query_index, query_states, key_states, value_states, query_index)
    attention_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attention_output = torch.matmul(attention_weights, value_states)
    return attention_output.transpose(1, 2).contiguous(), attention_weights


@contextmanager
def patched_attention(collector: SyntheticFitCollector):
    global _ORIGINAL_EAGER_ATTENTION_FORWARD, _ACTIVE_COLLECTOR
    import transformers.models.qwen3.modeling_qwen3 as modeling_qwen3

    _ORIGINAL_EAGER_ATTENTION_FORWARD = getattr(modeling_qwen3, "eager_attention_forward")
    _ACTIVE_COLLECTOR = collector
    setattr(modeling_qwen3, "eager_attention_forward", _patched_eager_attention_forward)
    if hasattr(modeling_qwen3, "ALL_ATTENTION_FUNCTIONS"):
        modeling_qwen3.ALL_ATTENTION_FUNCTIONS["eager"] = _patched_eager_attention_forward
    try:
        yield
    finally:
        if _ORIGINAL_EAGER_ATTENTION_FORWARD is not None:
            setattr(modeling_qwen3, "eager_attention_forward", _ORIGINAL_EAGER_ATTENTION_FORWARD)
            if hasattr(modeling_qwen3, "ALL_ATTENTION_FUNCTIONS"):
                modeling_qwen3.ALL_ATTENTION_FUNCTIONS["eager"] = _ORIGINAL_EAGER_ATTENTION_FORWARD
        _ACTIVE_COLLECTOR = None


def remote_output(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    scores = torch.matmul(q, k.transpose(0, 1)) / math.sqrt(q.shape[-1])
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, v)


def chunk_mean(k: torch.Tensor, v: torch.Tensor, m: int) -> tuple[torch.Tensor, torch.Tensor]:
    n = k.shape[0]
    if n <= m:
        return k.clone(), v.clone()
    boundaries = torch.linspace(0, n, steps=m + 1).round().long()
    ks: list[torch.Tensor] = []
    vs: list[torch.Tensor] = []
    prev = 0
    for i in range(m):
        start = max(prev, int(boundaries[i].item()))
        end = min(n, max(start + 1, int(boundaries[i + 1].item())))
        ks.append(k[start:end].mean(dim=0))
        vs.append(v[start:end].mean(dim=0))
        prev = end
    return torch.stack(ks), torch.stack(vs)


def solve_v_ridge(q: torch.Tensor, k_syn: torch.Tensor, y_target: torch.Tensor, ridge_lambda: float) -> torch.Tensor:
    scores = torch.matmul(q, k_syn.transpose(0, 1)) / math.sqrt(q.shape[-1])
    a = F.softmax(scores, dim=-1)
    ata = torch.matmul(a.transpose(0, 1), a)
    rhs = torch.matmul(a.transpose(0, 1), y_target)
    eye = torch.eye(ata.shape[0], dtype=ata.dtype, device=ata.device)
    return torch.linalg.solve(ata + ridge_lambda * eye, rhs)


def topmass_k(q: torch.Tensor, k: torch.Tensor, m: int) -> torch.Tensor:
    scores = torch.matmul(q, k.transpose(0, 1)) / math.sqrt(q.shape[-1])
    mass = F.softmax(scores, dim=-1).mean(dim=0)
    top = torch.topk(mass, k=min(m, k.shape[0]), largest=True).indices
    return k.index_select(dim=0, index=top)


def metrics(y_hat: torch.Tensor, y: torch.Tensor) -> tuple[float, float, float]:
    mse = float(F.mse_loss(y_hat, y).item())
    denom = float(y.float().square().mean().item()) + 1e-12
    nmse = mse / denom
    cosine = float(F.cosine_similarity(y_hat, y, dim=-1).mean().item())
    return mse, nmse, cosine


def evaluate_methods(
    *,
    q_calib: torch.Tensor,
    q_heldout: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    prototype_counts: list[int],
    ridge_lambda: float,
    device: torch.device,
) -> list[dict[str, Any]]:
    q_calib = q_calib.to(device)
    q_heldout = q_heldout.to(device)
    k = k.to(device)
    v = v.to(device)
    y_calib = remote_output(q_calib, k, v)
    y_heldout = remote_output(q_heldout, k, v)

    rows: list[dict[str, Any]] = []
    for m in prototype_counts:
        for method in ["chunk_mean", "chunk_mean_ridge", "topmass_ridge"]:
            if method in {"chunk_mean", "chunk_mean_ridge"}:
                k_syn, v_syn = chunk_mean(k, v, m)
                if method == "chunk_mean_ridge":
                    v_syn = solve_v_ridge(q_calib, k_syn, y_calib, ridge_lambda)
            else:
                k_syn = topmass_k(q_calib, k, m)
                v_syn = solve_v_ridge(q_calib, k_syn, y_calib, ridge_lambda)
            y_calib_hat = remote_output(q_calib, k_syn, v_syn)
            y_heldout_hat = remote_output(q_heldout, k_syn, v_syn)
            calib_mse, calib_nmse, calib_cos = metrics(y_calib_hat, y_calib)
            held_mse, held_nmse, held_cos = metrics(y_heldout_hat, y_heldout)
            rows.append(
                {
                    "method": method,
                    "prototypes": m,
                    "remote_tokens": int(k.shape[0]),
                    "keep_fraction": float(m / max(1, k.shape[0])),
                    "calib_mse": calib_mse,
                    "calib_nmse": calib_nmse,
                    "calib_cosine": calib_cos,
                    "heldout_mse": held_mse,
                    "heldout_nmse": held_nmse,
                    "heldout_cosine": held_cos,
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prototype_counts = parse_int_list(args.prototype_counts)
    layers = set(parse_int_list(args.layers))

    device = torch.device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    text = read_text_prefix(Path(args.text_path), args.max_chars)
    input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
    total_tokens = args.prefill_tokens + args.calib_tokens + args.heldout_tokens
    if input_ids.shape[-1] < total_tokens:
        raise ValueError(f"not enough tokens: need {total_tokens}, got {input_ids.shape[-1]}")
    input_ids = input_ids[:, :total_tokens]

    load_kwargs: dict[str, Any] = {"trust_remote_code": True, "torch_dtype": dtype}
    if args.device_map:
        load_kwargs["device_map"] = args.device_map
    if args.attn_implementation:
        load_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **load_kwargs)
    model.eval()
    input_device = pick_input_device(model, device)

    collector = SyntheticFitCollector(layers, args.prefill_tokens, args.calib_tokens, args.heldout_tokens, args.sink_tokens, args.recent_tokens)
    with torch.inference_mode(), patched_attention(collector):
        past = None
        for start in range(0, total_tokens, args.chunk_size):
            end = min(total_tokens, start + args.chunk_size)
            chunk = input_ids[:, start:end].to(input_device)
            kwargs = {
                "input_ids": chunk,
                "past_key_values": past,
                "use_cache": True,
                "return_dict": True,
                "output_attentions": False,
                "cache_position": torch.arange(start, end, device=input_device),
            }
            outputs = model_forward(model, kwargs)
            past = outputs.past_key_values
            print(f"forward chunk {start}-{end - 1}", flush=True)

    rows: list[dict[str, Any]] = []
    fit_device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    for layer in sorted(collector.remote_k):
        q_calib_layer = torch.stack(collector.calib_q[layer], dim=0)  # [C,H,D]
        q_held_layer = torch.stack(collector.heldout_q[layer], dim=0)  # [Hld,H,D]
        k_layer = collector.remote_k[layer]
        v_layer = collector.remote_v[layer]
        head_count = k_layer.shape[0]
        for head in range(head_count):
            method_rows = evaluate_methods(
                q_calib=q_calib_layer[:, head, :],
                q_heldout=q_held_layer[:, head, :],
                k=k_layer[head],
                v=v_layer[head],
                prototype_counts=prototype_counts,
                ridge_lambda=args.ridge_lambda,
                device=fit_device,
            )
            for row in method_rows:
                row.update({"layer": layer, "head": head})
                rows.append(row)

    fields = [
        "layer",
        "head",
        "method",
        "prototypes",
        "remote_tokens",
        "keep_fraction",
        "calib_mse",
        "calib_nmse",
        "calib_cosine",
        "heldout_mse",
        "heldout_nmse",
        "heldout_cosine",
    ]
    write_csv(output_dir / "fitted_synthetic_reconstruction_by_head.csv", rows, fields)

    summary: dict[str, Any] = {"args": vars(args), "method_summary": []}
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], int(row["prototypes"]))].append(row)
    for (method, prototypes), group in sorted(grouped.items()):
        summary["method_summary"].append(
            {
                "method": method,
                "prototypes": prototypes,
                "heads": len(group),
                "heldout_nmse_mean": sum(float(r["heldout_nmse"]) for r in group) / len(group),
                "heldout_cosine_mean": sum(float(r["heldout_cosine"]) for r in group) / len(group),
                "calib_nmse_mean": sum(float(r["calib_nmse"]) for r in group) / len(group),
            }
        )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["method_summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
