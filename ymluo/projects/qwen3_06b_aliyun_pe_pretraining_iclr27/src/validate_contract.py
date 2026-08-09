from __future__ import annotations

import argparse
import json
from pathlib import Path

from pe_strategies import load_strategy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--dclm-root", type=Path, required=True)
    parser.add_argument("--strategy", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, required=True)
    args = parser.parse_args()
    required_model_files = ["config.json", "model.safetensors", "tokenizer.json"]
    missing = [name for name in required_model_files if not (args.model_root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"model files missing under {args.model_root}: {missing}")
    if not args.dclm_root.is_dir():
        raise FileNotFoundError(args.dclm_root)
    strategy = load_strategy(args.strategy)
    config = json.loads((args.model_root / "config.json").read_text(encoding="utf-8"))
    max_positions = int(config.get("max_position_embeddings", 0) or 0)
    if max_positions and args.sequence_length > max_positions:
        raise ValueError(
            f"SEQ_LEN={args.sequence_length} exceeds model max_position_embeddings={max_positions}; "
            "this package does not silently rewrite the checkpoint config"
        )
    print(
        json.dumps(
            {
                "ok": True,
                "strategy": strategy.name,
                "model_type": config.get("model_type"),
                "num_hidden_layers": config.get("num_hidden_layers"),
                "max_position_embeddings": max_positions,
                "sequence_length": args.sequence_length,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

