from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import torch


def read_text(text_path: str) -> str:
    path = Path(text_path)
    if not path.exists():
        raise FileNotFoundError(f"Text path does not exist: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Text path is empty: {path}")
    return text


def tokenize_text(tokenizer, text: str, max_tokens: int) -> torch.Tensor:
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_tokens)
    input_ids = encoded.input_ids[0]
    if input_ids.numel() < 2:
        raise ValueError("Need at least two tokens for prefix geometry analysis.")
    return input_ids


def decode_tokens(tokenizer, input_ids: torch.Tensor) -> List[Dict[str, str]]:
    rows = []
    for idx, token_id in enumerate(input_ids.tolist()):
        piece = tokenizer.convert_ids_to_tokens(int(token_id))
        text = tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)
        rows.append(
            {
                "token_index": idx,
                "token_id": int(token_id),
                "token_piece": piece,
                "token_text": text.replace("\n", "\\n"),
            }
        )
    return rows


def load_tokenized_text(tokenizer, text_path: str, max_tokens: int) -> Tuple[str, torch.Tensor]:
    text = read_text(text_path)
    return text, tokenize_text(tokenizer, text, max_tokens)

