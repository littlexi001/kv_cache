#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}
PY=${PY:-/home/fdong/miniconda3/envs/moe/bin/python}
MODEL=${MODEL:-/home/fdong/hrj/prove/Qwen3-0.6B}
TEXT=${TEXT:-data/war_and_peace_pg2600.txt}
cd "$ROOT"

export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" - <<'PY'
from pathlib import Path

import torch

from src.evaluate_qwen3_top2_head_limit3_ppl import (
    AutoModelForCausalLM,
    AutoTokenizer,
    install_qwen3_attention_patch,
    pick_input_device,
    prefill_cache,
    read_text_prefix,
    resolve_dtype,
)
from src.run_pcic_rescue_blockwise_local import (
    eval_segment,
    eval_segment_batched_candidates,
    write_batch_layer_budget_map,
    write_layer_budget_map,
)

root = Path("/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl")
model_path = Path("/home/fdong/hrj/prove/Qwen3-0.6B")
text_path = root / "data/war_and_peace_pg2600.txt"
out = root / "outputs/batched_candidate_smoke"
maps = out / "maps"
maps.mkdir(parents=True, exist_ok=True)

dtype = resolve_dtype("float16", torch.device("cuda:0"))
tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
input_ids = tokenizer(
    read_text_prefix(text_path, 80_000),
    return_tensors="pt",
    add_special_tokens=False,
    truncation=True,
    max_length=400,
)["input_ids"]
input_ids = input_ids[:, :400]
model = AutoModelForCausalLM.from_pretrained(
    str(model_path),
    trust_remote_code=True,
    torch_dtype=dtype,
    attn_implementation="eager",
)
model.eval()
model.config.use_cache = True
install_qwen3_attention_patch()
input_device = pick_input_device(model, torch.device("cuda:0"))

prefill_tokens = 256
past, prev_logits = prefill_cache(
    model,
    input_ids,
    prefill_tokens,
    128,
    input_device,
)

candidate_combos = {"0,7": [0, 7], "2,0": [2, 0]}
serial = {}
for name, combo in candidate_combos.items():
    path = maps / f"combo_{name.replace(',', '_')}.json"
    write_layer_budget_map(path, combo, 512, 64)
    serial[name] = eval_segment(
        model=model,
        input_ids=input_ids,
        start_token=prefill_tokens,
        token_count=2,
        input_device=input_device,
        initial_past_key_values=past,
        initial_prev_logits=prev_logits,
        mode="layerbudgetattn",
        layer_budget_map_path=str(path),
        rescue_rule={"kind": "none"},
        log_prefix=f"serial {name}",
        log_every=1000,
    )

batch_path = maps / "batch_0_7__2_0.json"
write_batch_layer_budget_map(batch_path, list(candidate_combos.values()), 512, 64)
batched = eval_segment_batched_candidates(
    model=model,
    input_ids=input_ids,
    start_token=prefill_tokens,
    token_count=2,
    input_device=input_device,
    initial_past_key_values=past,
    initial_prev_logits=prev_logits,
    candidate_names=list(candidate_combos.keys()),
    batch_layer_budget_map_path=str(batch_path),
    log_prefix="batched",
    log_every=1000,
)

print("| combo | serial_loss | batched_loss | abs_diff | serial_s | batched_s |")
print("|---|---:|---:|---:|---:|---:|")
max_diff = 0.0
for name in candidate_combos:
    diff = abs(float(serial[name]["loss"]) - float(batched[name]["loss"]))
    max_diff = max(max_diff, diff)
    print(
        f"| {name} | {float(serial[name]['loss']):.8f} | {float(batched[name]['loss']):.8f} | "
        f"{diff:.8g} | {float(serial[name]['seconds']):.4f} | {float(batched[name]['seconds']):.4f} |"
    )
print(f"MAX_DIFF={max_diff:.8g}")
if max_diff > 1e-5:
    raise SystemExit(f"batched candidate smoke failed: max diff {max_diff}")
PY
