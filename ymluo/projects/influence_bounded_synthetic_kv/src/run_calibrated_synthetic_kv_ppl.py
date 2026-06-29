from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[4]
QABS_SRC = REPO_ROOT / "ymluo/projects/qwen3_top2_head_limit3_ppl/src"
if str(QABS_SRC) not in sys.path:
    sys.path.insert(0, str(QABS_SRC))

import evaluate_qwen3_top2_head_limit3_ppl as qabs_eval  # noqa: E402


try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    from transformers import AutoModelWithLMHead as AutoModelForCausalLM
    from transformers import AutoTokenizer


@dataclass(frozen=True)
class Config:
    model_name_or_path: str
    text_path: str
    output_dir: str
    prefill_tokens: int
    calib_tokens: int
    eval_tokens: int
    prototypes: int
    protect_sink_tokens: int
    protect_recent_tokens: int
    ridge: float
    joint_steps: int
    joint_lr: float
    layer_sets: str
    auto_mse_thresholds: str
    auto_topk_layers: str
    chunk_size: int
    dtype: str
    device: str
    device_map: str
    attn_implementation: str
    max_chars: int
    add_special_tokens: bool
    append_eos: bool
    require_total_tokens: bool
    log_every: int


@dataclass
class LayerCalibration:
    queries: list[torch.Tensor]
    final_k: torch.Tensor | None = None
    final_v: torch.Tensor | None = None
    scaling: float = 1.0


@dataclass
class LayerSyntheticKV:
    k_syn: torch.Tensor
    v_syn: torch.Tensor
    bias: torch.Tensor
    sink_end: int
    recent_start: int


_PHASE = "dense"
_CALIBRATION: dict[int, LayerCalibration] = {}
_SYNTHETIC: dict[int, LayerSyntheticKV] = {}
_ORIGINAL_EAGER_ATTENTION_FORWARD: Any | None = None
_STAGE_LOG: Path | None = None


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Calibrated synthetic KV PPL experiment.")
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--text_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prefill_tokens", type=int, default=512)
    parser.add_argument("--calib_tokens", type=int, default=16)
    parser.add_argument("--eval_tokens", type=int, default=32)
    parser.add_argument("--prototypes", type=int, default=16)
    parser.add_argument("--protect_sink_tokens", type=int, default=10)
    parser.add_argument("--protect_recent_tokens", type=int, default=10)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--joint_steps", type=int, default=0)
    parser.add_argument("--joint_lr", type=float, default=3e-2)
    parser.add_argument(
        "--layer_sets",
        default="all",
        help=(
            "Semicolon-separated layer sets to evaluate after fitting, e.g. "
            "'all;0-6;7-13;14-20;21-27;0;27'. Use 'none' for dense."
        ),
    )
    parser.add_argument(
        "--auto_mse_thresholds",
        default="",
        help="Comma-separated calibration output-MSE thresholds for automatic synthetic layer selection.",
    )
    parser.add_argument(
        "--auto_topk_layers",
        default="",
        help="Comma-separated K values. For each K, evaluate the K layers with lowest calibration output MSE.",
    )
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--max_chars", type=int, default=8_000_000)
    parser.add_argument("--add_special_tokens", type=str2bool, default=False)
    parser.add_argument("--append_eos", type=str2bool, default=False)
    parser.add_argument("--require_total_tokens", type=str2bool, default=False)
    parser.add_argument("--log_every", type=int, default=8)
    return Config(**vars(parser.parse_args()))


def torch_dtype(name: str) -> torch.dtype | str:
    if name == "auto":
        return "auto"
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(name)


def repeat_kv_if_needed(query_states: torch.Tensor, key_states: torch.Tensor, value_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if key_states.shape[1] != query_states.shape[1]:
        repeat_groups = query_states.shape[1] // key_states.shape[1]
        key_states = key_states.repeat_interleave(repeat_groups, dim=1)
        value_states = value_states.repeat_interleave(repeat_groups, dim=1)
    return key_states, value_states


def dense_attention(
    module: torch.nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask[:, :, :, : scores.shape[-1]]
    weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
    if dropout and module.training:
        weights = F.dropout(weights, p=dropout, training=True)
    output = torch.matmul(weights, value_states)
    return output.transpose(1, 2).contiguous(), weights


def synthetic_attention(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    scaling: float,
    layer_state: LayerSyntheticKV,
) -> torch.Tensor:
    score_parts: list[torch.Tensor] = []
    value_parts: list[torch.Tensor] = []
    k_syn = layer_state.k_syn.to(device=query_states.device, dtype=query_states.dtype)
    v_syn = layer_state.v_syn.to(device=query_states.device, dtype=query_states.dtype)
    bias = layer_state.bias.to(device=query_states.device, dtype=query_states.dtype)
    if layer_state.sink_end > 0:
        sink_k = key_states[:, :, : layer_state.sink_end, :]
        sink_v = value_states[:, :, : layer_state.sink_end, :]
        score_parts.append(torch.matmul(query_states, sink_k.transpose(2, 3)) * scaling)
        value_parts.append(sink_v[:, :, None, :, :])
    synth_scores = torch.einsum("bhqd,hpd->bhqp", query_states, k_syn) * scaling
    synth_scores = synth_scores + bias[None, :, None, :]
    score_parts.append(synth_scores)
    value_parts.append(v_syn[None, :, None, :, :].expand(query_states.shape[0], -1, query_states.shape[2], -1, -1))
    if layer_state.recent_start < key_states.shape[-2]:
        recent_k = key_states[:, :, layer_state.recent_start :, :]
        recent_v = value_states[:, :, layer_state.recent_start :, :]
        score_parts.append(torch.matmul(query_states, recent_k.transpose(2, 3)) * scaling)
        value_parts.append(recent_v[:, :, None, :, :])
    scores = torch.cat(score_parts, dim=-1)
    values = torch.cat(value_parts, dim=-2)
    weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
    output = torch.sum(weights[..., None] * values, dim=-2)
    return output.to(query_states.dtype).transpose(1, 2).contiguous()


def calibrated_attention_forward(
    module: torch.nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float | None = None,
    dropout: float = 0.0,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if scaling is None:
        scaling = float(getattr(module, "scaling", 1.0 / math.sqrt(query_states.shape[-1])))
    layer_idx = int(getattr(module, "layer_idx", 0))
    if _PHASE == "synthetic" and layer_idx in _SYNTHETIC and query_states.shape[-2] == 1:
        expanded_k, expanded_v = repeat_kv_if_needed(query_states, key_states, value_states)
        return synthetic_attention(query_states, expanded_k, expanded_v, scaling, _SYNTHETIC[layer_idx]), None
    if _ORIGINAL_EAGER_ATTENTION_FORWARD is None:
        expanded_k, expanded_v = repeat_kv_if_needed(query_states, key_states, value_states)
        output, weights = dense_attention(module, query_states, expanded_k, expanded_v, attention_mask, scaling, dropout)
    else:
        output, weights = _ORIGINAL_EAGER_ATTENTION_FORWARD(
            module,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling=scaling,
            dropout=dropout,
            **kwargs,
        )
    if _PHASE == "collect" and query_states.shape[-2] == 1:
        expanded_k, expanded_v = repeat_kv_if_needed(query_states, key_states, value_states)
        state = _CALIBRATION.setdefault(layer_idx, LayerCalibration(queries=[], scaling=scaling))
        state.queries.append(query_states.detach().squeeze(0).squeeze(1).float().cpu())
        state.final_k = expanded_k.detach().squeeze(0).float().cpu()
        state.final_v = expanded_v.detach().squeeze(0).float().cpu()
        state.scaling = scaling
    if bool(kwargs.get("output_attentions", False)):
        return output, weights
    return output, None


def install_attention_patch() -> None:
    global _ORIGINAL_EAGER_ATTENTION_FORWARD
    import transformers.models.qwen3.modeling_qwen3 as modeling_qwen3

    if _ORIGINAL_EAGER_ATTENTION_FORWARD is None:
        _ORIGINAL_EAGER_ATTENTION_FORWARD = getattr(modeling_qwen3, "eager_attention_forward")
        setattr(modeling_qwen3, "eager_attention_forward", calibrated_attention_forward)
        if hasattr(modeling_qwen3, "ALL_ATTENTION_FUNCTIONS"):
            modeling_qwen3.ALL_ATTENTION_FUNCTIONS["eager"] = calibrated_attention_forward


def set_phase(phase: str) -> None:
    global _PHASE
    _PHASE = phase


def parse_layer_set(spec: str, available_layers: set[int]) -> tuple[str, set[int]]:
    normalized = spec.strip().lower()
    if not normalized or normalized == "none":
        return "none", set()
    if normalized == "all":
        return "all", set(available_layers)
    selected: set[int] = set()
    for part in normalized.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            start = int(left)
            end = int(right)
            if end < start:
                start, end = end, start
            selected.update(range(start, end + 1))
        else:
            selected.add(int(part))
    selected &= available_layers
    label = "layers" + "_".join(str(value) for value in sorted(selected)) if selected else "none"
    if len(selected) > 3:
        sorted_layers = sorted(selected)
        label = f"layers{sorted_layers[0]}-{sorted_layers[-1]}n{len(sorted_layers)}"
    return label, selected


def parse_float_list(spec: str) -> list[float]:
    values: list[float] = []
    for part in spec.split(","):
        part = part.strip()
        if part:
            values.append(float(part))
    return values


def parse_int_list(spec: str) -> list[int]:
    values: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    return values


def safe_number_label(value: float) -> str:
    return f"{value:.4g}".replace("-", "m").replace(".", "p")


def mark(message: str) -> None:
    text = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(text, flush=True)
    if _STAGE_LOG is not None:
        with _STAGE_LOG.open("a", encoding="utf-8") as handle:
            handle.write(text + "\n")


@torch.inference_mode()
def run_tokens(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    start: int,
    count: int,
    past_key_values: Any,
    prev_logits: torch.Tensor,
    input_device: torch.device,
    score: bool,
    log_prefix: str,
    log_every: int,
) -> tuple[Any, torch.Tensor, float, int]:
    total_loss = 0.0
    total_count = 0
    for index, pos in enumerate(range(start, start + count), start=1):
        chunk = input_ids[:, pos : pos + 1].to(input_device)
        if log_every <= 1 or index == 1 or index == count or index % log_every == 0:
            mark(f"{log_prefix} token {index}/{count}: position {pos}")
        kwargs: dict[str, Any] = {
            "input_ids": chunk,
            "past_key_values": past_key_values,
            "use_cache": True,
            "return_dict": True,
            "output_attentions": False,
            "output_hidden_states": False,
            "cache_position": torch.arange(pos, pos + 1, device=input_device),
        }
        outputs = qabs_eval.model_forward(model, kwargs)
        logits = outputs.logits
        if score:
            loss = F.cross_entropy(prev_logits.float(), chunk.reshape(-1), reduction="sum")
            total_loss += float(loss)
            total_count += int(chunk.numel())
        past_key_values = outputs.past_key_values
        prev_logits = logits[:, -1, :].detach()
        del outputs, chunk, logits
        if input_device.type == "cuda":
            torch.cuda.empty_cache()
    return past_key_values, prev_logits, total_loss, total_count


def fit_synthetic(config: Config, device: torch.device) -> list[dict[str, Any]]:
    global _SYNTHETIC
    _SYNTHETIC = {}
    rows: list[dict[str, Any]] = []
    for layer_idx, calib in sorted(_CALIBRATION.items()):
        if not calib.queries or calib.final_k is None or calib.final_v is None:
            continue
        q = torch.stack(calib.queries, dim=0).to(device=device, dtype=torch.float32)  # [C, H, D]
        k = calib.final_k.to(device=device, dtype=torch.float32)  # [H, T, D]
        v = calib.final_v.to(device=device, dtype=torch.float32)  # [H, T, Dv]
        head_count, key_count, head_dim = k.shape
        value_dim = v.shape[-1]
        sink_end = min(max(0, config.protect_sink_tokens), key_count)
        recent_start = max(sink_end, key_count - max(0, config.protect_recent_tokens))
        remote_start = sink_end
        remote_end = recent_start
        remote_len = remote_end - remote_start
        proto_count = min(config.prototypes, max(remote_len, 1))
        k_syn = torch.zeros(head_count, proto_count, head_dim, device=device, dtype=torch.float32)
        v_syn = torch.zeros(head_count, proto_count, value_dim, device=device, dtype=torch.float32)
        bias = torch.zeros(head_count, proto_count, device=device, dtype=torch.float32)
        if remote_len <= 0:
            _SYNTHETIC[layer_idx] = LayerSyntheticKV(k_syn.to(device), v_syn.to(device), bias.to(device), sink_end, recent_start)
            continue
        boundaries = torch.linspace(0, remote_len, steps=proto_count + 1, device=device).round().to(torch.long)
        eye = torch.eye(head_dim + 1, device=device, dtype=torch.float32)
        for head in range(head_count):
            qh = q[:, head, :]
            q_aug = torch.cat([qh, torch.ones(qh.shape[0], 1, device=device)], dim=-1)
            lhs = q_aug.T @ q_aug + config.ridge * eye
            for proto in range(proto_count):
                start = int(boundaries[proto].item())
                end = int(boundaries[proto + 1].item())
                end = max(end, start + 1)
                end = min(end, remote_len)
                key_chunk = k[head, remote_start + start : remote_start + end, :]
                value_chunk = v[head, remote_start + start : remote_start + end, :]
                chunk_scores = qh @ key_chunk.T * calib.scaling
                target_logit = torch.logsumexp(chunk_scores, dim=-1)
                rhs = q_aug.T @ target_logit
                beta = torch.linalg.solve(lhs, rhs)
                k_syn[head, proto, :] = beta[:-1] / calib.scaling
                bias[head, proto] = beta[-1]
                weights = F.softmax(chunk_scores, dim=-1, dtype=torch.float32)
                chunk_values = weights @ value_chunk
                calib_mass = torch.softmax(target_logit, dim=0)[:, None]
                v_syn[head, proto, :] = torch.sum(calib_mass * chunk_values, dim=0)
        target = full_attention_targets(q, k, v, calib.scaling)
        initial_mse = synthetic_output_mse(
            q,
            target,
            k,
            v,
            k_syn,
            v_syn,
            bias,
            calib.scaling,
            sink_end,
            recent_start,
        )
        final_mse = initial_mse
        if config.joint_steps > 0:
            k_syn, v_syn, bias, final_mse = optimize_synthetic_layer(
                q,
                target,
                k,
                v,
                k_syn,
                v_syn,
                bias,
                calib.scaling,
                sink_end,
                recent_start,
                config.joint_steps,
                config.joint_lr,
            )
        _SYNTHETIC[layer_idx] = LayerSyntheticKV(
            k_syn.to(device=device),
            v_syn.to(device=device),
            bias.to(device=device),
            sink_end=sink_end,
            recent_start=recent_start,
        )
        rows.append(
            {
                "layer": layer_idx,
                "calib_queries": len(calib.queries),
                "key_count": key_count,
                "remote_tokens": remote_len,
                "prototypes": proto_count,
                "sink_tokens": sink_end,
                "recent_tokens": key_count - recent_start,
                "mean_k_norm": float(torch.linalg.vector_norm(k_syn, dim=-1).mean().item()),
                "max_k_norm": float(torch.linalg.vector_norm(k_syn, dim=-1).max().item()),
                "mean_v_norm": float(torch.linalg.vector_norm(v_syn, dim=-1).mean().item()),
                "max_abs_bias": float(torch.max(torch.abs(bias)).item()),
                "initial_output_mse": initial_mse,
                "final_output_mse": final_mse,
            }
        )
    return rows


def full_attention_targets(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    scores = torch.einsum("chd,htd->cht", q, k) * scaling
    weights = F.softmax(scores, dim=-1, dtype=torch.float32)
    return torch.einsum("cht,htv->chv", weights, v)


def synthetic_outputs(
    q: torch.Tensor,
    k_full: torch.Tensor,
    v_full: torch.Tensor,
    k_syn: torch.Tensor,
    v_syn: torch.Tensor,
    bias: torch.Tensor,
    scaling: float,
    sink_end: int,
    recent_start: int,
) -> torch.Tensor:
    score_parts: list[torch.Tensor] = []
    value_parts: list[torch.Tensor] = []
    if sink_end > 0:
        sink_k = k_full[:, :sink_end, :]
        sink_v = v_full[:, :sink_end, :]
        score_parts.append(torch.einsum("chd,htd->cht", q, sink_k) * scaling)
        value_parts.append(sink_v[None, :, :, :].expand(q.shape[0], -1, -1, -1))
    synth_scores = torch.einsum("chd,hpd->chp", q, k_syn) * scaling + bias[None, :, :]
    score_parts.append(synth_scores)
    value_parts.append(v_syn[None, :, :, :].expand(q.shape[0], -1, -1, -1))
    if recent_start < k_full.shape[1]:
        recent_k = k_full[:, recent_start:, :]
        recent_v = v_full[:, recent_start:, :]
        score_parts.append(torch.einsum("chd,htd->cht", q, recent_k) * scaling)
        value_parts.append(recent_v[None, :, :, :].expand(q.shape[0], -1, -1, -1))
    scores = torch.cat(score_parts, dim=-1)
    values = torch.cat(value_parts, dim=-2)
    weights = F.softmax(scores, dim=-1, dtype=torch.float32)
    return torch.sum(weights[..., None] * values, dim=-2)


def synthetic_output_mse(
    q: torch.Tensor,
    target: torch.Tensor,
    k_full: torch.Tensor,
    v_full: torch.Tensor,
    k_syn: torch.Tensor,
    v_syn: torch.Tensor,
    bias: torch.Tensor,
    scaling: float,
    sink_end: int,
    recent_start: int,
) -> float:
    with torch.no_grad():
        pred = synthetic_outputs(q, k_full, v_full, k_syn, v_syn, bias, scaling, sink_end, recent_start)
        return float(torch.mean((pred - target) ** 2).item())


def optimize_synthetic_layer(
    q: torch.Tensor,
    target: torch.Tensor,
    k_full: torch.Tensor,
    v_full: torch.Tensor,
    k_init: torch.Tensor,
    v_init: torch.Tensor,
    bias_init: torch.Tensor,
    scaling: float,
    sink_end: int,
    recent_start: int,
    steps: int,
    lr: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    k_param = torch.nn.Parameter(k_init.detach().clone())
    v_param = torch.nn.Parameter(v_init.detach().clone())
    bias_param = torch.nn.Parameter(bias_init.detach().clone())
    optimizer = torch.optim.Adam([k_param, v_param, bias_param], lr=lr)
    for _ in range(steps):
        pred = synthetic_outputs(q, k_full, v_full, k_param, v_param, bias_param, scaling, sink_end, recent_start)
        loss = torch.mean((pred - target) ** 2)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    final_mse = synthetic_output_mse(q, target, k_full, v_full, k_param, v_param, bias_param, scaling, sink_end, recent_start)
    return k_param.detach(), v_param.detach(), bias_param.detach(), final_mse


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_tokens(config: Config, tokenizer: Any) -> torch.Tensor:
    text = Path(config.text_path).read_text(encoding="utf-8", errors="ignore")[: config.max_chars]
    token_ids = tokenizer(text, add_special_tokens=config.add_special_tokens)["input_ids"]
    if config.append_eos and tokenizer.eos_token_id is not None:
        token_ids.append(tokenizer.eos_token_id)
    need = config.prefill_tokens + config.calib_tokens + config.eval_tokens
    if len(token_ids) < need:
        if config.require_total_tokens:
            raise RuntimeError(f"Need {need} tokens but got {len(token_ids)} from {config.text_path}.")
        repeats = math.ceil(need / max(len(token_ids), 1))
        token_ids = (token_ids * repeats)[:need]
    else:
        token_ids = token_ids[:need]
    return torch.tensor([token_ids], dtype=torch.long)


def main() -> None:
    global _CALIBRATION, _SYNTHETIC, _STAGE_LOG
    config = parse_args()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _STAGE_LOG = output_dir / "stage.log"
    _STAGE_LOG.write_text("", encoding="utf-8")
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=True)
    input_ids = load_tokens(config, tokenizer)
    input_device = torch.device(config.device)
    load_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "attn_implementation": config.attn_implementation,
        "torch_dtype": torch_dtype(config.dtype),
    }
    if config.device_map != "none":
        load_kwargs["device_map"] = config.device_map
    mark("loading model")
    model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path, **load_kwargs)
    model.eval()
    if config.device_map == "none":
        model.to(input_device)
    install_attention_patch()

    mark("dense prefill")
    set_phase("dense")
    prefill_started = time.perf_counter()
    base_cache, base_prev_logits = qabs_eval.prefill_cache(
        model,
        input_ids,
        config.prefill_tokens,
        config.chunk_size,
        input_device,
    )
    prefill_seconds = time.perf_counter() - prefill_started

    mark("baseline calibration skip")
    baseline_cache = qabs_eval.clone_past_key_values(base_cache)
    baseline_prev = base_prev_logits.detach().clone()
    baseline_cache, baseline_prev, _, _ = run_tokens(
        model,
        input_ids,
        config.prefill_tokens,
        config.calib_tokens,
        baseline_cache,
        baseline_prev,
        input_device,
        score=False,
        log_prefix="baseline calib",
        log_every=config.log_every,
    )
    mark("baseline eval")
    baseline_started = time.perf_counter()
    baseline_cache, baseline_prev, baseline_loss_sum, baseline_count = run_tokens(
        model,
        input_ids,
        config.prefill_tokens + config.calib_tokens,
        config.eval_tokens,
        baseline_cache,
        baseline_prev,
        input_device,
        score=True,
        log_prefix="baseline eval",
        log_every=config.log_every,
    )
    baseline_seconds = time.perf_counter() - baseline_started
    del baseline_cache, baseline_prev
    if input_device.type == "cuda":
        torch.cuda.empty_cache()

    mark("synthetic calibration collect")
    _CALIBRATION = {}
    synth_cache = qabs_eval.clone_past_key_values(base_cache)
    synth_prev = base_prev_logits.detach().clone()
    set_phase("collect")
    synth_cache, synth_prev, _, _ = run_tokens(
        model,
        input_ids,
        config.prefill_tokens,
        config.calib_tokens,
        synth_cache,
        synth_prev,
        input_device,
        score=False,
        log_prefix="synth calib",
        log_every=config.log_every,
    )
    del synth_cache, synth_prev
    if input_device.type == "cuda":
        torch.cuda.empty_cache()
    fit_started = time.perf_counter()
    mark("fit synthetic start")
    fit_rows = fit_synthetic(config, input_device)
    fit_seconds = time.perf_counter() - fit_started
    mark(f"fit synthetic done: layers={len(fit_rows)} seconds={fit_seconds:.3f}")
    write_csv(
        output_dir / "synthetic_fit_by_layer.csv",
        fit_rows,
        [
            "layer",
            "calib_queries",
            "key_count",
            "remote_tokens",
            "prototypes",
            "sink_tokens",
            "recent_tokens",
            "mean_k_norm",
            "max_k_norm",
            "mean_v_norm",
            "max_abs_bias",
            "initial_output_mse",
            "final_output_mse",
        ],
    )

    rows = []
    baseline_loss = baseline_loss_sum / max(baseline_count, 1)
    rows.append(
        {
            "mode": "baseline",
            "synthetic_layers": "",
            "synthetic_layer_count": "",
            "loss": baseline_loss,
            "ppl": math.exp(min(baseline_loss, 80.0)),
            "token_count": baseline_count,
            "seconds": baseline_seconds,
            "prefill_seconds": prefill_seconds,
            "fit_seconds": "",
            "ppl_ratio_vs_baseline": "",
            "time_ratio_vs_baseline": "",
        }
    )
    all_synthetic = dict(_SYNTHETIC)
    available_layers = set(all_synthetic)
    layer_set_specs = [part.strip() for part in config.layer_sets.split(";") if part.strip()]
    if not layer_set_specs:
        layer_set_specs = ["all"]
    eval_layer_sets: list[tuple[str, str, set[int]]] = []
    for layer_spec in layer_set_specs:
        layer_label, selected_layers = parse_layer_set(layer_spec, available_layers)
        eval_layer_sets.append((layer_spec, layer_label, selected_layers))
    mse_by_layer = {
        int(row["layer"]): float(row["final_output_mse"])
        for row in fit_rows
        if int(row["layer"]) in available_layers
    }
    for threshold in parse_float_list(config.auto_mse_thresholds):
        selected_layers = {layer for layer, mse in mse_by_layer.items() if mse <= threshold}
        label = f"automse{safe_number_label(threshold)}n{len(selected_layers)}"
        eval_layer_sets.append((f"auto_mse<={threshold:g}", label, selected_layers))
    ordered_by_mse = sorted(mse_by_layer, key=lambda layer: (mse_by_layer[layer], layer))
    for topk in parse_int_list(config.auto_topk_layers):
        k = max(0, min(topk, len(ordered_by_mse)))
        selected_layers = set(ordered_by_mse[:k])
        label = f"autotop{k}mse"
        eval_layer_sets.append((f"auto_topk={k}", label, selected_layers))

    for layer_spec, layer_label, selected_layers in eval_layer_sets:
        _SYNTHETIC = {layer: all_synthetic[layer] for layer in selected_layers}
        eval_cache = qabs_eval.clone_past_key_values(base_cache)
        eval_prev = base_prev_logits.detach().clone()
        set_phase("dense")
        eval_cache, eval_prev, _, _ = run_tokens(
            model,
            input_ids,
            config.prefill_tokens,
            config.calib_tokens,
            eval_cache,
            eval_prev,
            input_device,
            score=False,
            log_prefix=f"synth {layer_label} calib replay",
            log_every=config.log_every,
        )
        mark(f"synthetic eval layer_set={layer_spec} selected={sorted(selected_layers)}")
        set_phase("synthetic")
        synth_started = time.perf_counter()
        eval_cache, eval_prev, synth_loss_sum, synth_count = run_tokens(
            model,
            input_ids,
            config.prefill_tokens + config.calib_tokens,
            config.eval_tokens,
            eval_cache,
            eval_prev,
            input_device,
            score=True,
            log_prefix=f"synth {layer_label} eval",
            log_every=config.log_every,
        )
        synth_seconds = time.perf_counter() - synth_started
        mark(f"synthetic eval done layer_set={layer_spec} seconds={synth_seconds:.3f}")
        del eval_cache, eval_prev
        if input_device.type == "cuda":
            torch.cuda.empty_cache()
        synth_loss = synth_loss_sum / max(synth_count, 1)
        rows.append(
            {
                "mode": f"synthkv_calibrated_{layer_label}",
                "synthetic_layers": ",".join(str(layer) for layer in sorted(selected_layers)),
                "synthetic_layer_count": len(selected_layers),
                "loss": synth_loss,
                "ppl": math.exp(min(synth_loss, 80.0)),
                "token_count": synth_count,
                "seconds": synth_seconds,
                "prefill_seconds": prefill_seconds,
                "fit_seconds": fit_seconds,
                "ppl_ratio_vs_baseline": "",
                "time_ratio_vs_baseline": "",
            }
        )
    set_phase("dense")
    baseline_ppl = float(rows[0]["ppl"])
    baseline_time = float(rows[0]["seconds"])
    for row in rows:
        row["ppl_ratio_vs_baseline"] = float(row["ppl"]) / baseline_ppl
        row["time_ratio_vs_baseline"] = float(row["seconds"]) / baseline_time
    write_csv(
        output_dir / "ppl_by_mode.csv",
        rows,
        [
            "mode",
            "synthetic_layers",
            "synthetic_layer_count",
            "loss",
            "ppl",
            "token_count",
            "seconds",
            "prefill_seconds",
            "fit_seconds",
            "ppl_ratio_vs_baseline",
            "time_ratio_vs_baseline",
        ],
    )
    summary = {"config": asdict(config), "rows": rows, "fit_layer_count": len(fit_rows)}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
