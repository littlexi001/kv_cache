from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from geometry_metrics import select_indices  # noqa: E402
from model_loader import load_model_and_tokenizer  # noqa: E402
from text_loader import decode_tokens, load_tokenized_text  # noqa: E402


EPS = 1e-12


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(values: List[float], prefix: str) -> Dict[str, float]:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return {
            f"{prefix}_count": 0,
            f"{prefix}_mean": float("nan"),
            f"{prefix}_std": float("nan"),
            f"{prefix}_p05": float("nan"),
            f"{prefix}_p50": float("nan"),
            f"{prefix}_p95": float("nan"),
            f"{prefix}_max": float("nan"),
        }
    x = torch.tensor(finite, dtype=torch.float64)
    q = torch.quantile(x, torch.tensor([0.05, 0.50, 0.95], dtype=torch.float64))
    return {
        f"{prefix}_count": int(x.numel()),
        f"{prefix}_mean": float(x.mean().item()),
        f"{prefix}_std": float(x.std(unbiased=False).item()) if x.numel() > 1 else 0.0,
        f"{prefix}_p05": float(q[0].item()),
        f"{prefix}_p50": float(q[1].item()),
        f"{prefix}_p95": float(q[2].item()),
        f"{prefix}_max": float(x.max().item()),
    }


def js_divergence(p: torch.Tensor, q: torch.Tensor) -> float:
    p = p.to(dtype=torch.float64).clamp_min(EPS)
    q = q.to(dtype=torch.float64).clamp_min(EPS)
    p = p / p.sum().clamp_min(EPS)
    q = q / q.sum().clamp_min(EPS)
    m = 0.5 * (p + q)
    kl_pm = (p * (torch.log(p) - torch.log(m))).sum()
    kl_qm = (q * (torch.log(q) - torch.log(m))).sum()
    return float((0.5 * (kl_pm + kl_qm)).item())


def pearson_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.numel() < 2:
        return float("nan")
    x = x.to(dtype=torch.float64)
    y = y.to(dtype=torch.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    if float(denom.item()) <= EPS:
        return float("nan")
    return float(((x * y).sum() / denom).item())


def topk_overlap(x: torch.Tensor, y: torch.Tensor, top_k: int) -> float:
    if x.numel() == 0:
        return float("nan")
    k = min(top_k, x.numel())
    x_idx = set(torch.topk(x, k=k).indices.tolist())
    y_idx = set(torch.topk(y, k=k).indices.tolist())
    return len(x_idx & y_idx) / float(k)


def attention_entropy(probs: torch.Tensor) -> Tuple[float, float]:
    probs = probs.to(dtype=torch.float64).clamp_min(EPS)
    probs = probs / probs.sum().clamp_min(EPS)
    entropy = float((-(probs * torch.log(probs)).sum()).item())
    max_entropy = math.log(max(1, probs.numel()))
    normalized = entropy / max_entropy if max_entropy > 0 else float("nan")
    return entropy, normalized


def run_forward(model, input_ids: torch.Tensor, device: torch.device):
    with torch.no_grad():
        return model.model(
            input_ids=input_ids[None, :].to(device),
            use_cache=False,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )


def compute_layer_qk(model, hidden_states: torch.Tensor, layer_idx: int, position_embeddings) -> Tuple[torch.Tensor, torch.Tensor]:
    layer = model.model.layers[layer_idx]
    attn = layer.self_attn
    with torch.no_grad():
        x = layer.input_layernorm(hidden_states)
        input_shape = x.shape[:-1]
        hidden_shape = (*input_shape, -1, attn.head_dim)
        q = attn.q_norm(attn.q_proj(x).view(hidden_shape)).transpose(1, 2)
        k = attn.k_norm(attn.k_proj(x).view(hidden_shape)).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, position_embeddings[0], position_embeddings[1])
    return q[0], k[0]


def query_indices(seq_len: int, min_query_index: int, stride: int) -> List[int]:
    start = max(1, min_query_index)
    return list(range(start, seq_len, max(1, stride)))


def analyze_q_head(
    q: torch.Tensor,
    k: torch.Tensor,
    scaling: float,
    layer_idx: int,
    q_head_idx: int,
    kv_head_idx: int,
    query_ids: Iterable[int],
    top_k: int,
) -> Dict:
    qh = q[q_head_idx].detach().cpu().to(dtype=torch.float64)
    kh = k[kv_head_idx].detach().cpu().to(dtype=torch.float64)
    seq_len = kh.shape[0]
    global_mean_k = kh.mean(dim=0)
    global_mean_k_norm = float(torch.linalg.vector_norm(global_mean_k).item())

    q_common_cos = []
    q_common_abs_cos = []
    k_common_cos = (
        F.normalize(kh, p=2, dim=1, eps=EPS)
        @ F.normalize(global_mean_k[None, :], p=2, dim=1, eps=EPS).T
    ).squeeze(1)

    raw_std = []
    centered_std = []
    std_ratio = []
    raw_centered_corr = []
    topk_overlaps = []
    js_values = []
    max_shift_errors = []
    attn_top1 = []
    attn_top5_mass = []
    attn_entropy_norm = []

    for t in query_ids:
        if t <= 0 or t >= seq_len:
            continue
        q_t = qh[t]
        k_past = kh[:t]
        c = k_past.mean(dim=0)
        centered = k_past - c

        cos_q_c = float(
            (
                F.normalize(q_t[None, :], p=2, dim=1, eps=EPS)
                @ F.normalize(c[None, :], p=2, dim=1, eps=EPS).T
            ).item()
        )
        q_common_cos.append(cos_q_c)
        q_common_abs_cos.append(abs(cos_q_c))

        raw_scores = (q_t @ k_past.T) * scaling
        centered_scores = (q_t @ centered.T) * scaling
        common_scalar = (q_t @ c) * scaling
        shift_error = (raw_scores - centered_scores - common_scalar).abs().max()
        max_shift_errors.append(float(shift_error.item()))

        raw_s = float(raw_scores.std(unbiased=False).item()) if raw_scores.numel() > 1 else 0.0
        centered_s = float(centered_scores.std(unbiased=False).item()) if centered_scores.numel() > 1 else 0.0
        raw_std.append(raw_s)
        centered_std.append(centered_s)
        std_ratio.append(centered_s / raw_s if raw_s > EPS else float("nan"))
        raw_centered_corr.append(pearson_corr(raw_scores, centered_scores))
        topk_overlaps.append(topk_overlap(raw_scores, centered_scores, top_k))

        raw_attn = torch.softmax(raw_scores, dim=-1)
        centered_attn = torch.softmax(centered_scores, dim=-1)
        js_values.append(js_divergence(raw_attn, centered_attn))
        attn_top1.append(float(raw_attn.max().item()))
        k5 = min(5, raw_attn.numel())
        attn_top5_mass.append(float(torch.topk(raw_attn, k=k5).values.sum().item()))
        _, entropy_norm = attention_entropy(raw_attn)
        attn_entropy_norm.append(entropy_norm)

    row = {
        "layer": layer_idx,
        "q_head": q_head_idx,
        "kv_head": kv_head_idx,
        "seq_len": seq_len,
        "num_queries": len(list(query_ids)) if not isinstance(query_ids, list) else len(query_ids),
        "global_mean_k_norm": global_mean_k_norm,
        "global_k_common_cos_mean": float(k_common_cos.mean().item()),
        "global_k_common_cos_p50": float(torch.quantile(k_common_cos, 0.50).item()),
        "global_k_common_cos_p95": float(torch.quantile(k_common_cos, 0.95).item()),
    }
    row.update(summarize(q_common_cos, "q_common_cos"))
    row.update(summarize(q_common_abs_cos, "q_common_abs_cos"))
    row.update(summarize(raw_std, "raw_score_std"))
    row.update(summarize(centered_std, "centered_score_std"))
    row.update(summarize(std_ratio, "centered_raw_std_ratio"))
    row.update(summarize(raw_centered_corr, "raw_centered_score_corr"))
    row.update(summarize(topk_overlaps, f"top{top_k}_overlap"))
    row.update(summarize(js_values, "attention_js_raw_vs_centered"))
    row.update(summarize(max_shift_errors, "raw_centered_shift_error"))
    row.update(summarize(attn_top1, "attention_top1_mass"))
    row.update(summarize(attn_top5_mass, "attention_top5_mass"))
    row.update(summarize(attn_entropy_norm, "attention_entropy_norm"))
    return row


def aggregate_rows(rows: List[Dict]) -> Dict:
    keys = [
        "q_common_cos_mean",
        "q_common_abs_cos_mean",
        "global_k_common_cos_mean",
        "raw_score_std_mean",
        "centered_score_std_mean",
        "centered_raw_std_ratio_mean",
        "raw_centered_score_corr_mean",
        "attention_js_raw_vs_centered_mean",
        "raw_centered_shift_error_max",
        "attention_top1_mass_mean",
        "attention_top5_mass_mean",
        "attention_entropy_norm_mean",
    ]
    result = {}
    for key in keys:
        vals = [float(row[key]) for row in rows if key in row and math.isfinite(float(row[key]))]
        result.update(summarize(vals, key))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe why raw K-K similarity can be high while qK attention remains selective.")
    parser.add_argument("--model-path", default="fdong/Qwen3-0.6B")
    parser.add_argument("--text-path", default="fdong_seq_compress/data/synthetic_texts/long_english_12000_words.txt")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--max-tokens", type=int, default=1000)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--q-heads", default="all")
    parser.add_argument("--query-stride", type=int, default=8)
    parser.add_argument("--min-query-index", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--allow-longer-than-model-max", action="store_true")
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or f"fdong_seq_compress/outputs/qk_common_direction_probe_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, model, device = load_model_and_tokenizer(
        args.model_path,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    _, input_ids = load_tokenized_text(tokenizer, args.text_path, args.max_tokens)
    seq_len = int(input_ids.numel())
    model_max_position_embeddings = getattr(model.config, "max_position_embeddings", None)
    if model_max_position_embeddings is not None and seq_len > int(model_max_position_embeddings):
        message = (
            f"Tokenized sequence length ({seq_len}) exceeds model.config.max_position_embeddings "
            f"({model_max_position_embeddings}). This can make Q/K geometry unreliable due to "
            "position/RoPE handling. Use a shorter sequence or pass --allow-longer-than-model-max "
            "only if you intentionally want to test extrapolation."
        )
        if not args.allow_longer_than_model_max:
            raise ValueError(message)
        print(f"WARNING: {message}", flush=True)
    write_csv(output_dir / "tokens.csv", decode_tokens(tokenizer, input_ids))

    outputs = run_forward(model, input_ids, device)
    hidden_states = outputs.hidden_states
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    position_embeddings = model.model.rotary_emb(hidden_states[0], position_ids)

    num_layers = int(getattr(model.config, "num_hidden_layers"))
    layer_indices = select_indices(num_layers, args.layers)
    num_q_heads = int(getattr(model.config, "num_attention_heads"))
    num_kv_heads = int(getattr(model.config, "num_key_value_heads"))
    num_groups = num_q_heads // num_kv_heads
    q_head_indices = select_indices(num_q_heads, args.q_heads)
    q_ids = query_indices(seq_len, args.min_query_index, args.query_stride)

    rows: List[Dict] = []
    for layer_idx in layer_indices:
        q, k = compute_layer_qk(model, hidden_states[layer_idx], layer_idx, position_embeddings)
        scaling = float(model.model.layers[layer_idx].self_attn.scaling)
        for q_head_idx in q_head_indices:
            kv_head_idx = q_head_idx // num_groups
            row = analyze_q_head(q, k, scaling, layer_idx, q_head_idx, kv_head_idx, q_ids, args.top_k)
            rows.append(row)
            print(
                f"layer={layer_idx} q_head={q_head_idx} kv_head={kv_head_idx} "
                f"q_common_cos={row['q_common_cos_mean']:.4f} "
                f"std_ratio={row['centered_raw_std_ratio_mean']:.4f} "
                f"corr={row['raw_centered_score_corr_mean']:.6f} "
                f"js={row['attention_js_raw_vs_centered_mean']:.3e}",
                flush=True,
            )

    write_csv(output_dir / "qk_common_direction_by_layer_head.csv", rows)
    summary = {
        "model_path": args.model_path,
        "text_path": args.text_path,
        "output_dir": str(output_dir),
        "device": str(device),
        "dtype": args.dtype,
        "seq_len": seq_len,
        "model_max_position_embeddings": model_max_position_embeddings,
        "seq_len_within_model_max_position_embeddings": (
            None if model_max_position_embeddings is None else seq_len <= int(model_max_position_embeddings)
        ),
        "layers": layer_indices,
        "q_heads": q_head_indices,
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "query_stride": args.query_stride,
        "min_query_index": args.min_query_index,
        "num_sampled_queries": len(q_ids),
        "top_k": args.top_k,
        "global": aggregate_rows(rows),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
