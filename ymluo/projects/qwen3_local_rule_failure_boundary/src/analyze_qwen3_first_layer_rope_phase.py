from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence

import torch

import run_local_global_rope_probe_8b as probe
import run_rope_retrieval_repair_8b as rope_repair


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract Qwen3-8B layer-0 pre-RoPE Q/K for a gold evidence token "
            "and verify the exact per-frequency phase decomposition."
        )
    )
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lengths", default="8192,16384,32768,65536")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--original-max-position-embeddings", type=int, default=40960)
    parser.add_argument("--global-max-position", type=int, default=70000)
    return parser.parse_args()


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rounded(value: float) -> float:
    return round(float(value), 10)


def find_gold_position(
    tokenizer: Any,
    prompt: torch.Tensor,
    evidence_span: tuple[int, int],
    gold: str,
) -> int:
    ids = tokenizer(f" {gold}", add_special_tokens=False)["input_ids"]
    if len(ids) != 1:
        raise RuntimeError(f"gold is not one token: {gold!r} -> {ids}")
    gold_id = int(ids[0])
    start, end = evidence_span
    matches = (
        prompt[0, start:end] == gold_id
    ).nonzero(as_tuple=False).view(-1)
    if matches.numel() != 1:
        raise RuntimeError(
            f"expected one gold token in span {evidence_span}, got {matches.numel()}"
        )
    return start + int(matches[0].item())


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    lengths = sorted(
        {int(item.strip()) for item in args.lengths.split(",") if item.strip()}
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = probe.load_model(args)
    layer = model.model.layers[0]
    embedding = model.model.embed_tokens
    inv_freq = model.model.rotary_emb.inv_freq.detach().float()
    attention_scaling = float(model.model.rotary_emb.attention_scaling)
    head_dim = int(model.config.head_dim)
    heads = int(model.config.num_attention_heads)
    kv_heads = int(model.config.num_key_value_heads)
    groups = heads // kv_heads
    pair_count = head_dim // 2
    scale = 1.0 / math.sqrt(head_dim)

    rows: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    reference_q: torch.Tensor | None = None
    reference_k: torch.Tensor | None = None
    for length in lengths:
        case = probe.seeded_case(tokenizer, length, args.seed)
        prompt = case["prompt"]
        query_position = int(prompt.shape[1]) - 1
        evidence_position = find_gold_position(
            tokenizer,
            prompt,
            case["evidence_spans"][-1],
            case["codes"][-1],
        )
        query_id = prompt[:, query_position : query_position + 1].to(
            probe.base.input_device(model)
        )
        evidence_id = prompt[:, evidence_position : evidence_position + 1].to(
            probe.base.input_device(model)
        )
        query_hidden = layer.input_layernorm(embedding(query_id))
        key_hidden = layer.input_layernorm(embedding(evidence_id))
        q_pre = layer.self_attn.q_norm(
            layer.self_attn.q_proj(query_hidden).view(
                1,
                1,
                heads,
                head_dim,
            )
        )[0, 0].float()
        k_pre = layer.self_attn.k_norm(
            layer.self_attn.k_proj(key_hidden).view(
                1,
                1,
                kv_heads,
                head_dim,
            )
        )[0, 0].float().repeat_interleave(groups, dim=0)
        if reference_q is None:
            reference_q = q_pre.detach().cpu()
            reference_k = k_pre.detach().cpu()
        else:
            if not torch.equal(reference_q, q_pre.detach().cpu()):
                raise RuntimeError("layer-0 pre-RoPE Query changed across lengths")
            if not torch.equal(reference_k, k_pre.detach().cpu()):
                raise RuntimeError("layer-0 pre-RoPE Key changed across lengths")

        delta = query_position - evidence_position
        phase = delta * inv_freq.to(q_pre.device)
        cos = phase.cos() * attention_scaling
        sin = phase.sin() * attention_scaling
        qx, qy = q_pre[:, :pair_count], q_pre[:, pair_count:]
        kx, ky = k_pre[:, :pair_count], k_pre[:, pair_count:]
        a = qx * kx + qy * ky
        b = qx * ky - qy * kx
        pair_contribution = (
            a * cos.view(1, -1) + b * sin.view(1, -1)
        ) * attention_scaling * scale

        q_cos, q_sin = rope_repair.rope_angles(
            torch.tensor([query_position], device=q_pre.device),
            inv_freq.to(q_pre.device),
            head_dim,
            q_pre.dtype,
        )
        k_cos, k_sin = rope_repair.rope_angles(
            torch.tensor([evidence_position], device=q_pre.device),
            inv_freq.to(q_pre.device),
            head_dim,
            q_pre.dtype,
        )
        q_cos = q_cos[0] * attention_scaling
        q_sin = q_sin[0] * attention_scaling
        k_cos = k_cos[0] * attention_scaling
        k_sin = k_sin[0] * attention_scaling
        q_post = q_pre * q_cos + rope_repair.rotate_half(q_pre) * q_sin
        k_post = k_pre * k_cos + rope_repair.rotate_half(k_pre) * k_sin
        explicit = (q_post * k_post).sum(dim=-1) * scale
        reconstructed = pair_contribution.sum(dim=-1)
        max_error = float((explicit - reconstructed).abs().max().item())
        if max_error > 2e-3:
            raise RuntimeError(f"phase reconstruction error too large: {max_error}")

        for head in range(heads):
            for pair in range(pair_count):
                rows.append(
                    {
                        "target_context_tokens": length,
                        "prompt_tokens": int(prompt.shape[1]),
                        "query_position": query_position,
                        "evidence_position": evidence_position,
                        "relative_distance": delta,
                        "head": head,
                        "pair": pair,
                        "inv_frequency": rounded(inv_freq[pair].item()),
                        "period_tokens": rounded(
                            2.0 * math.pi / float(inv_freq[pair].item())
                        ),
                        "phase": rounded(phase[pair].item()),
                        "A": rounded(a[head, pair].item()),
                        "B": rounded(b[head, pair].item()),
                        "pair_qk_contribution": rounded(
                            pair_contribution[head, pair].item()
                        ),
                    }
                )
        summary.append(
            {
                "target_context_tokens": length,
                "prompt_tokens": int(prompt.shape[1]),
                "query_position": query_position,
                "evidence_position": evidence_position,
                "relative_distance": delta,
                "mean_gold_qk": rounded(explicit.mean().item()),
                "min_gold_qk": rounded(explicit.min().item()),
                "max_gold_qk": rounded(explicit.max().item()),
                "max_reconstruction_error": rounded(max_error),
                "pre_rope_q_norm": rounded(q_pre.norm().item()),
                "pre_rope_k_norm": rounded(k_pre.norm().item()),
                "attention_scaling": rounded(attention_scaling),
            }
        )

    write_csv(output_dir / "first_layer_pair_contributions.csv", rows)
    write_csv(output_dir / "first_layer_summary.csv", summary)
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "model_name_or_path": args.model_name_or_path,
                "seed": args.seed,
                "lengths": lengths,
                "head_dim": head_dim,
                "heads": heads,
                "kv_heads": kv_heads,
                "pair_count": pair_count,
                "rope_theta": float(model.config.rope_theta),
                "rope_scaling": model.config.rope_scaling,
                "attention_scaling": attention_scaling,
                "cuda_visible_devices": __import__("os").environ.get(
                    "CUDA_VISIBLE_DEVICES",
                    "",
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
