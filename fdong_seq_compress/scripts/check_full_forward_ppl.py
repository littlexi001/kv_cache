from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F

THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model_loader import load_model_and_tokenizer  # noqa: E402
from text_loader import load_tokenized_text  # noqa: E402


def parse_text_paths(value: str) -> List[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


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


def ppl(loss: torch.Tensor) -> float:
    value = float(loss.detach().cpu().item())
    return float(math.exp(value))


def main() -> None:
    parser = argparse.ArgumentParser(description="Check ordinary full-attention next-token loss/PPL.")
    parser.add_argument("--model-path", default="fdong/Qwen3-0.6B")
    parser.add_argument("--text-paths", required=True, help="Comma-separated text paths.")
    parser.add_argument("--output-csv", default="fdong_seq_compress/outputs/full_forward_ppl_check.csv")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--max-tokens", type=int, default=3000)
    parser.add_argument("--decode-start", type=int, default=1000)
    args = parser.parse_args()

    tokenizer, model, device = load_model_and_tokenizer(
        args.model_path,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )

    rows: List[Dict] = []
    for text_path in parse_text_paths(args.text_paths):
        print(f"Checking {text_path}", flush=True)
        text, input_ids = load_tokenized_text(tokenizer, text_path, args.max_tokens)
        input_ids = input_ids.to(device)
        with torch.no_grad():
            output = model(input_ids[None, :], labels=input_ids[None, :], use_cache=False, return_dict=True)
            logits = output.logits[0]

        shift_logits = logits[:-1].float()
        shift_labels = input_ids[1:]
        positions = torch.arange(1, input_ids.numel(), device=device)
        decode_mask = positions >= max(1, min(args.decode_start, input_ids.numel() - 1))
        prefix_mask = ~decode_mask

        all_loss = F.cross_entropy(shift_logits, shift_labels, reduction="mean")
        prefix_loss = F.cross_entropy(shift_logits[prefix_mask], shift_labels[prefix_mask], reduction="mean")
        decode_loss = F.cross_entropy(shift_logits[decode_mask], shift_labels[decode_mask], reduction="mean")
        row = {
            "text_path": text_path,
            "chars": len(text),
            "seq_len": int(input_ids.numel()),
            "max_tokens": args.max_tokens,
            "decode_start": args.decode_start,
            "hf_labels_loss": float(output.loss.detach().cpu().item()),
            "hf_labels_ppl": ppl(output.loss),
            "all_loss": float(all_loss.detach().cpu().item()),
            "all_ppl": ppl(all_loss),
            "prefix_loss": float(prefix_loss.detach().cpu().item()),
            "prefix_ppl": ppl(prefix_loss),
            "decode_loss": float(decode_loss.detach().cpu().item()),
            "decode_ppl": ppl(decode_loss),
            "num_prefix_targets": int(prefix_mask.sum().item()),
            "num_decode_targets": int(decode_mask.sum().item()),
        }
        rows.append(row)
        print(json.dumps(row, indent=2), flush=True)

    output_csv = Path(args.output_csv)
    write_csv(output_csv, rows)
    print(f"Wrote {output_csv}", flush=True)


if __name__ == "__main__":
    main()
