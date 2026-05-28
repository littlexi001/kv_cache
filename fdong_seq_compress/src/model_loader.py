from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(dtype: str, device: torch.device) -> torch.dtype:
    if dtype == "float32":
        return torch.float32
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype != "auto":
        raise ValueError(f"Unsupported dtype: {dtype}")

    if device.type == "cuda":
        return torch.bfloat16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def load_model_and_tokenizer(
    model_path: str,
    device: str = "auto",
    dtype: str = "auto",
    attn_implementation: str = "eager",
) -> Tuple[AutoTokenizer, AutoModelForCausalLM, torch.device]:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model path does not exist: {path}")

    torch_device = resolve_device(device)
    torch_dtype = resolve_dtype(dtype, torch_device)

    tokenizer = AutoTokenizer.from_pretrained(str(path), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(path),
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        attn_implementation=attn_implementation,
    )
    model.eval()
    model.to(torch_device)
    return tokenizer, model, torch_device

