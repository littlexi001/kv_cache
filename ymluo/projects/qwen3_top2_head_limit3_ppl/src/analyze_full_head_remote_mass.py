from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from evaluate_qwen3_top2_head_limit3_ppl import (
    AutoModelForCausalLM,
    AutoTokenizer,
    model_forward,
    pick_input_device,
    prefill_cache,
    read_text_prefix,
    resolve_dtype,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank layer/head importance by baseline attention mass outside a recent window.")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--text_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--prefill_tokens", type=int, default=20000)
    parser.add_argument("--eval_tokens", type=int, default=32)
    parser.add_argument("--chunk_size", type=int, default=128)
    parser.add_argument("--recent_tokens", type=int, default=512)
    parser.add_argument("--max_chars", type=int, default=8_000_000)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--add_special_tokens", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    text = read_text_prefix(Path(args.text_path), args.max_chars)
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=args.add_special_tokens)
    input_ids = encoded["input_ids"]
    required = args.prefill_tokens + args.eval_tokens
    if input_ids.shape[-1] < required:
        raise ValueError(f"not enough tokens: need {required}, got {input_ids.shape[-1]}")

    load_kwargs: dict[str, object] = {"trust_remote_code": True, "torch_dtype": dtype}
    if args.device_map:
        load_kwargs["device_map"] = args.device_map
    if args.attn_implementation:
        load_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **load_kwargs)
    model.eval()
    input_device = pick_input_device(model, device)
    input_ids = input_ids[:, :required]

    past_key_values, _ = prefill_cache(model, input_ids, args.prefill_tokens, args.chunk_size, input_device)
    layer_count = int(getattr(model.config, "num_hidden_layers"))
    head_count = int(getattr(model.config, "num_attention_heads"))
    remote_mass = torch.zeros((layer_count, head_count), dtype=torch.float64)
    recent_mass = torch.zeros((layer_count, head_count), dtype=torch.float64)

    with torch.inference_mode():
        for token_idx in range(args.prefill_tokens, args.prefill_tokens + args.eval_tokens):
            chunk = input_ids[:, token_idx : token_idx + 1].to(input_device)
            outputs = model_forward(
                model,
                {
                    "input_ids": chunk,
                    "past_key_values": past_key_values,
                    "use_cache": True,
                    "return_dict": True,
                    "output_attentions": True,
                    "output_hidden_states": False,
                    "cache_position": torch.arange(token_idx, token_idx + 1, device=input_device),
                },
            )
            attentions = outputs.attentions
            if attentions is None:
                raise RuntimeError("model did not return attentions; use --attn_implementation eager")
            for layer_idx, attention in enumerate(attentions):
                weights = attention.detach().float()[0, :, 0, :]
                key_count = int(weights.shape[-1])
                history_count = max(0, key_count - 1)
                remote_end = max(0, history_count - args.recent_tokens)
                if remote_end > 0:
                    remote_mass[layer_idx] += weights[:, :remote_end].sum(dim=-1).cpu().double()
                if remote_end < key_count:
                    recent_mass[layer_idx] += weights[:, remote_end:key_count].sum(dim=-1).cpu().double()
            past_key_values = outputs.past_key_values
            del outputs, chunk
            if input_device.type == "cuda":
                torch.cuda.empty_cache()

    top_heads_by_layer: dict[str, list[int]] = {}
    remote_mass_rows: list[list[float]] = []
    for layer_idx in range(layer_count):
        order = torch.argsort(remote_mass[layer_idx], descending=True).tolist()
        top_heads_by_layer[str(layer_idx)] = [int(head) for head in order]
        remote_mass_rows.append([float(remote_mass[layer_idx, head]) / max(1, args.eval_tokens) for head in range(head_count)])

    output = {
        "prefill_tokens": args.prefill_tokens,
        "eval_tokens": args.eval_tokens,
        "recent_tokens": args.recent_tokens,
        "layer_count": layer_count,
        "head_count": head_count,
        "top_heads_by_layer": top_heads_by_layer,
        "mean_remote_mass": remote_mass_rows,
        "mean_recent_mass": [
            [float(recent_mass[layer_idx, head]) / max(1, args.eval_tokens) for head in range(head_count)]
            for layer_idx in range(layer_count)
        ],
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
