from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from pe_strategies import Strategy, patch_model


def resolve_dtype(name: str) -> torch.dtype:
    normalized = name.lower().replace("torch.", "")
    values = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if normalized not in values:
        raise ValueError(f"unsupported dtype {name!r}")
    return values[normalized]


def load_tokenizer(path: str | Path) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(path), trust_remote_code=True, use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(
    model_path: str | Path,
    strategy: Strategy,
    dtype_name: str,
    attention_implementation: str,
    for_training: bool,
    initialization: str = "checkpoint",
) -> tuple[Any, str]:
    from transformers import AutoConfig, AutoModelForCausalLM

    dtype = resolve_dtype(dtype_name)
    if initialization not in {"checkpoint", "from_scratch"}:
        raise ValueError(
            f"initialization must be checkpoint or from_scratch, found {initialization!r}"
        )
    kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
    }
    if initialization == "checkpoint":
        kwargs["low_cpu_mem_usage"] = True
    chosen = attention_implementation
    kwargs["attn_implementation"] = chosen

    def construct() -> Any:
        if initialization == "from_scratch":
            config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
            config.torch_dtype = dtype
            return AutoModelForCausalLM.from_config(config, **kwargs)
        return AutoModelForCausalLM.from_pretrained(str(model_path), **kwargs)

    try:
        model = construct()
    except (ImportError, ValueError) as error:
        if chosen == "sdpa":
            raise
        chosen = "sdpa"
        kwargs["attn_implementation"] = chosen
        model = construct()
        print(f"attention implementation fallback to sdpa: {error}", flush=True)
    model.config.use_cache = not for_training
    patched = patch_model(model, strategy)
    print(
        f"loaded model={model_path} initialization={initialization} strategy={strategy.name} "
        f"patched_layers={patched} attention={chosen}",
        flush=True,
    )
    return model, chosen


def input_device(model: Any) -> torch.device:
    return next(model.parameters()).device
