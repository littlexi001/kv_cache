from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch

THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model_loader import load_model_and_tokenizer  # noqa: E402
from text_loader import load_tokenized_text  # noqa: E402


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


def sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    if device.type == "mps":
        torch.mps.synchronize()


def write_decode_svg(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    xs = [int(row["absolute_token_index"]) for row in rows]
    ys = [float(row["decode_forward_seconds"]) * 1000.0 for row in rows]
    max_y = max(max(ys), 1e-6)
    width, height = 980, 420
    left, right, top, bottom = 68, 24, 24, 54
    plot_w = width - left - right
    plot_h = height - top - bottom
    min_x, max_x = min(xs), max(xs)

    def px(x: int) -> float:
        if max_x == min_x:
            return left + plot_w / 2
        return left + (x - min_x) * plot_w / (max_x - min_x)

    def py(y: float) -> float:
        return top + plot_h - (y / max_y) * plot_h

    points = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in zip(xs, ys))
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="18" font-family="monospace" font-size="13" fill="#111827">Full KV-cache decode time by token</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#111827"/>',
        f'<polyline points="{points}" fill="none" stroke="#2563eb" stroke-width="1.7"/>',
        f'<text x="{left + plot_w / 2 - 65}" y="{height - 12}" font-family="monospace" font-size="12" fill="#374151">absolute token index</text>',
        f'<text x="8" y="{top + 16}" font-family="monospace" font-size="12" fill="#374151">ms</text>',
    ]
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = top + plot_h - frac * plot_h
        value = frac * max_y
        pieces.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        pieces.append(f'<text x="12" y="{y + 4:.1f}" font-family="monospace" font-size="10" fill="#6b7280">{value:.1f}</text>')
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        x = min_x + int(frac * (max_x - min_x))
        pieces.append(f'<text x="{px(x) - 18:.1f}" y="{top + plot_h + 18}" font-family="monospace" font-size="10" fill="#6b7280">{x}</text>')
    pieces.append("</svg>")
    path.write_text("\n".join(pieces), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark ordinary full-attention KV-cache decode timing.")
    parser.add_argument("--model-path", default="fdong/Qwen3-0.6B")
    parser.add_argument("--text-path", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--prefill-tokens", type=int, default=2500)
    parser.add_argument("--decode-tokens", type=int, default=512)
    parser.add_argument("--warmup-tokens", type=int, default=8)
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or f"fdong_seq_compress/outputs/full_decode_timing_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, model, device = load_model_and_tokenizer(
        args.model_path,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    total_needed = args.prefill_tokens + args.decode_tokens + 1
    _, input_ids = load_tokenized_text(tokenizer, args.text_path, total_needed)
    input_ids = input_ids.to(device)
    prefill_ids = input_ids[: args.prefill_tokens][None, :]

    with torch.no_grad():
        sync_if_needed(device)
        prefill_start = time.perf_counter()
        output = model(prefill_ids, use_cache=True, return_dict=True)
        sync_if_needed(device)
        prefill_seconds = time.perf_counter() - prefill_start
        past_key_values = output.past_key_values

        rows: List[Dict] = []
        for i in range(args.decode_tokens):
            token_index = args.prefill_tokens + i
            step_input = input_ids[token_index : token_index + 1][None, :]
            sync_if_needed(device)
            start = time.perf_counter()
            output = model(step_input, past_key_values=past_key_values, use_cache=True, return_dict=True)
            sync_if_needed(device)
            elapsed = time.perf_counter() - start
            past_key_values = output.past_key_values
            if i >= args.warmup_tokens:
                rows.append(
                    {
                        "decode_step": i,
                        "absolute_token_index": token_index,
                        "decode_forward_seconds": elapsed,
                        "decode_forward_ms": elapsed * 1000.0,
                    }
                )
            print(f"decode_step={i} token_index={token_index} time_ms={elapsed * 1000.0:.3f}", flush=True)

    write_csv(output_dir / "decode_timing.csv", rows)
    write_decode_svg(output_dir / "decode_time_by_token.svg", rows)
    summary = {
        "model_path": args.model_path,
        "text_path": args.text_path,
        "device": str(device),
        "dtype": args.dtype,
        "prefill_tokens": args.prefill_tokens,
        "decode_tokens": args.decode_tokens,
        "warmup_tokens": args.warmup_tokens,
        "prefill_seconds": prefill_seconds,
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
