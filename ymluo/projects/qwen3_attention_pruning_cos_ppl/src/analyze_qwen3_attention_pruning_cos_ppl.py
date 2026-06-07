from __future__ import annotations

import argparse
import csv
import json
import math
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    from transformers import AutoModelWithLMHead as AutoModelForCausalLM
    from transformers import AutoTokenizer


DEFAULT_MODEL_PATH = "/mnt/workspace/Qwen3-0.6B"
DEFAULT_TEXT_PATH = (
    "/mnt/workspace/dclm/global-shard_01_of_10/local-shard_0_of_10/part-00000.txt"
)
DEFAULT_RATIOS = "0.001,0.005,0.01,0.02,0.04,0.06,0.08,0.10,0.15,0.20"

_ACTIVE_PRUNE_RATIO: float | None = None
_PATCHED_QWEN3_MODULE: Any | None = None
_ORIGINAL_EAGER_ATTENTION_FORWARD: Any | None = None


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare full attention softmax with top-ratio pruned attention and evaluate PPL."
    )
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--text_path", default=DEFAULT_TEXT_PATH)
    parser.add_argument("--output_dir", default="outputs/attention_pruning_cos_ppl")
    parser.add_argument("--prefill_tokens", type=int, default=5000)
    parser.add_argument("--eval_tokens", type=int, default=5000)
    parser.add_argument(
        "--eval_last_tokens_only",
        type=str2bool,
        default=False,
        help="Use the whole tokenized text as context and evaluate only the last eval_tokens tokens.",
    )
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument("--max_chars", type=int, default=8_000_000)
    parser.add_argument("--add_special_tokens", type=str2bool, default=False)
    parser.add_argument("--append_eos", type=str2bool, default=False)
    parser.add_argument("--require_total_tokens", type=str2bool, default=True)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--layers", default="all")
    parser.add_argument("--heads", default="all")
    parser.add_argument("--ratios", default=DEFAULT_RATIOS)
    parser.add_argument("--save_cos_per_token", type=str2bool, default=True)
    parser.add_argument("--cos_csv_flush_rows", type=int, default=200_000)
    parser.add_argument("--compute_cos", type=str2bool, default=True)
    parser.add_argument("--compute_ppl", type=str2bool, default=True)
    parser.add_argument("--make_plots", type=str2bool, default=True)
    parser.add_argument("--plot_dpi", type=int, default=180)
    return parser.parse_args()


def parse_float_list(spec: str) -> list[float]:
    values = [float(part.strip()) for part in spec.split(",") if part.strip()]
    if not values or any(value <= 0.0 or value > 1.0 for value in values):
        raise ValueError("--ratios must contain floats in (0, 1].")
    return values


def parse_index_spec(spec: str, max_count: int, name: str) -> list[int]:
    normalized = spec.strip().lower()
    if normalized == "all":
        return list(range(max_count))
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
                raise ValueError(f"Invalid {name} range: {part}")
            selected.update(range(start, end + 1))
        else:
            selected.add(int(part))
    invalid = sorted(index for index in selected if index < 0 or index >= max_count)
    if invalid:
        raise ValueError(f"{name} out of range 0..{max_count - 1}: {invalid}")
    return sorted(selected)


def resolve_dtype(dtype_name: str, device: torch.device) -> torch.dtype | str:
    if dtype_name == "auto":
        return "auto"
    if device.type == "cpu":
        return torch.float32
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def read_text_prefix(path: Path, max_chars: int) -> str:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        if max_chars > 0:
            return handle.read(max_chars)
        return handle.read()


def pick_input_device(model: torch.nn.Module, fallback_device: torch.device) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return fallback_device


def model_forward(model: torch.nn.Module, kwargs: dict[str, Any]) -> Any:
    try:
        return model(**kwargs)
    except TypeError as exc:
        if "cache_position" in kwargs and "cache_position" in str(exc):
            kwargs = dict(kwargs)
            kwargs.pop("cache_position")
            return model(**kwargs)
        raise


def append_rows(path: Path, rows: list[dict[str, Any]], fields: list[str], append: bool) -> None:
    with path.open("a" if append else "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not append:
            writer.writeheader()
        writer.writerows(rows)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def cos_per_token_fields() -> list[str]:
    return ["layer", "head", "query_token", "key_len", "ratio", "keep_count", "cosine"]


def cos_summary_fields() -> list[str]:
    return ["layer", "head", "ratio", "query_count", "mean_cosine"]


def ppl_fields() -> list[str]:
    return ["mode", "ratio", "kept_percent", "loss", "ppl", "token_count"]


class CosAccumulator:
    def __init__(self, layer_indices: list[int], head_indices: list[int], ratios: list[float]) -> None:
        self.layers = layer_indices
        self.heads = head_indices
        self.ratios = ratios
        self.sums: dict[tuple[int, int, float], float] = {
            (layer, head, ratio): 0.0 for layer in layer_indices for head in head_indices for ratio in ratios
        }
        self.counts: dict[tuple[int, int, float], int] = {
            (layer, head, ratio): 0 for layer in layer_indices for head in head_indices for ratio in ratios
        }

    def update(self, layer: int, head: int, ratio: float, values: torch.Tensor) -> None:
        key = (layer, head, ratio)
        self.sums[key] += float(values.sum())
        self.counts[key] += int(values.numel())

    def rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for layer in self.layers:
            for head in self.heads:
                for ratio in self.ratios:
                    key = (layer, head, ratio)
                    count = self.counts[key]
                    rows.append(
                        {
                            "layer": layer,
                            "head": head,
                            "ratio": ratio,
                            "query_count": count,
                            "mean_cosine": self.sums[key] / count if count else "",
                        }
                    )
        return rows


@torch.inference_mode()
def prefill_cache(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    prefill_tokens: int,
    chunk_size: int,
    input_device: torch.device,
    return_last_logits: bool,
) -> tuple[Any, torch.Tensor | None]:
    past_key_values = None
    last_logits: torch.Tensor | None = None
    total_chunks = math.ceil(prefill_tokens / chunk_size)
    for chunk_idx, start in enumerate(range(0, prefill_tokens, chunk_size), start=1):
        end = min(start + chunk_size, prefill_tokens)
        chunk = input_ids[:, start:end].to(input_device)
        kwargs: dict[str, Any] = {
            "input_ids": chunk,
            "use_cache": True,
            "return_dict": True,
            "output_attentions": False,
            "output_hidden_states": False,
            "cache_position": torch.arange(start, end, device=input_device),
        }
        if past_key_values is not None:
            kwargs["past_key_values"] = past_key_values
        print(f"prefill chunk {chunk_idx}/{total_chunks}: tokens {start}-{end - 1}", flush=True)
        outputs = model_forward(model, kwargs)
        past_key_values = outputs.past_key_values
        if return_last_logits:
            last_logits = outputs.logits[:, -1, :].detach()
        del outputs, chunk
        if input_device.type == "cuda":
            torch.cuda.empty_cache()
    return past_key_values, last_logits


def cosine_rows_for_attention(
    attention: torch.Tensor,
    layer: int,
    head_indices: list[int],
    query_start: int,
    ratios: list[float],
    accumulator: CosAccumulator,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    # attention: [heads, query_tokens, key_tokens], already post-softmax.
    _, query_count, key_count = attention.shape
    max_ratio = max(ratios)
    max_keep = max(1, math.ceil(max_ratio * key_count))
    for local_query in range(query_count):
        query_token = query_start + local_query
        current_key_len = query_token + 1
        max_keep_for_query = min(max_keep, current_key_len)
        full = attention[head_indices, local_query, :current_key_len].float()
        full_square_sum = (full * full).sum(dim=-1).clamp_min(1e-30)
        top_values = torch.topk(full, k=max_keep_for_query, dim=-1, largest=True).values
        top_square_cumsum = torch.cumsum(top_values * top_values, dim=-1)
        for ratio in ratios:
            keep_count = max(1, math.ceil(ratio * current_key_len))
            keep_count = min(keep_count, current_key_len)
            top_square_sum = top_square_cumsum[:, keep_count - 1]
            cosine = torch.sqrt(top_square_sum / full_square_sum)
            for head, value in zip(head_indices, cosine.tolist()):
                rows.append(
                    {
                        "layer": layer,
                        "head": head,
                        "query_token": query_token,
                        "key_len": current_key_len,
                        "ratio": ratio,
                        "keep_count": keep_count,
                        "cosine": float(value),
                    }
                )
            for head_pos, head in enumerate(head_indices):
                accumulator.update(layer, head, ratio, cosine[head_pos : head_pos + 1])
    return rows


@torch.inference_mode()
def compute_cos_metrics(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    prefill_tokens: int,
    eval_tokens: int,
    chunk_size: int,
    input_device: torch.device,
    layer_indices: list[int],
    head_indices: list[int],
    ratios: list[float],
    output_dir: Path,
    save_cos_per_token: bool,
    flush_rows: int,
) -> list[dict[str, Any]]:
    print("building full prefill cache for cosine metrics", flush=True)
    past_key_values, _ = prefill_cache(model, input_ids, prefill_tokens, chunk_size, input_device, False)
    accumulator = CosAccumulator(layer_indices, head_indices, ratios)
    cos_csv_path = output_dir / "cos_per_token.csv"
    wrote = False
    pending_rows: list[dict[str, Any]] = []
    total_chunks = math.ceil(eval_tokens / chunk_size)
    eval_end = prefill_tokens + eval_tokens

    for chunk_idx, start in enumerate(range(prefill_tokens, eval_end, chunk_size), start=1):
        end = min(start + chunk_size, eval_end)
        chunk = input_ids[:, start:end].to(input_device)
        kwargs: dict[str, Any] = {
            "input_ids": chunk,
            "use_cache": True,
            "return_dict": True,
            "output_attentions": True,
            "output_hidden_states": False,
            "cache_position": torch.arange(start, end, device=input_device),
        }
        if past_key_values is not None:
            kwargs["past_key_values"] = past_key_values
        print(f"cos attention chunk {chunk_idx}/{total_chunks}: tokens {start}-{end - 1}", flush=True)
        outputs = model_forward(model, kwargs)
        if outputs.attentions is None:
            raise RuntimeError("Model did not return attentions. Use ATTN_IMPLEMENTATION=eager.")
        past_key_values = outputs.past_key_values
        for layer in layer_indices:
            attention_layer = outputs.attentions[layer][0].detach().cpu()
            pending_rows.extend(
                cosine_rows_for_attention(
                    attention_layer,
                    layer,
                    head_indices,
                    start,
                    ratios,
                    accumulator,
                )
            )
            if save_cos_per_token and len(pending_rows) >= flush_rows:
                append_rows(cos_csv_path, pending_rows, cos_per_token_fields(), append=wrote)
                wrote = True
                pending_rows.clear()
        del outputs, chunk
        if input_device.type == "cuda":
            torch.cuda.empty_cache()

    if save_cos_per_token and pending_rows:
        append_rows(cos_csv_path, pending_rows, cos_per_token_fields(), append=wrote)
    return accumulator.rows()


def _pruned_eager_attention_forward(
    module: torch.nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float | None = None,
    dropout: float = 0.0,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    ratio = _ACTIVE_PRUNE_RATIO
    if scaling is None:
        scaling = float(getattr(module, "scaling", 1.0 / math.sqrt(query_states.shape[-1])))
    if key_states.shape[1] != query_states.shape[1]:
        repeat_groups = query_states.shape[1] // key_states.shape[1]
        key_states = key_states.repeat_interleave(repeat_groups, dim=1)
        value_states = value_states.repeat_interleave(repeat_groups, dim=1)
    scores = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask[:, :, :, : scores.shape[-1]]
    if ratio is not None:
        keep = torch.zeros_like(scores, dtype=torch.bool)
        query_count = scores.shape[-2]
        for query_index in range(query_count):
            row = scores[:, :, query_index, :]
            valid_count = int(torch.isfinite(row[0, 0]).sum().item())
            keep_count = min(valid_count, max(1, math.ceil(ratio * valid_count))) if valid_count > 0 else 1
            _, top_indices = torch.topk(row, k=keep_count, dim=-1, largest=True)
            keep[:, :, query_index, :].scatter_(-1, top_indices, True)
        scores = scores.masked_fill(~keep, torch.finfo(scores.dtype).min)
    attention_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
    if dropout and module.training:
        attention_weights = F.dropout(attention_weights, p=dropout, training=True)
    attention_output = torch.matmul(attention_weights, value_states)
    attention_output = attention_output.transpose(1, 2).contiguous()
    return attention_output, attention_weights


def install_qwen3_attention_patch() -> None:
    global _PATCHED_QWEN3_MODULE, _ORIGINAL_EAGER_ATTENTION_FORWARD
    try:
        import transformers.models.qwen3.modeling_qwen3 as modeling_qwen3
    except Exception as exc:
        raise RuntimeError("Could not import transformers.models.qwen3.modeling_qwen3 for PPL pruning.") from exc
    if _ORIGINAL_EAGER_ATTENTION_FORWARD is None:
        _ORIGINAL_EAGER_ATTENTION_FORWARD = getattr(modeling_qwen3, "eager_attention_forward")
        setattr(modeling_qwen3, "eager_attention_forward", _pruned_eager_attention_forward)
        if hasattr(modeling_qwen3, "ALL_ATTENTION_FUNCTIONS"):
            modeling_qwen3.ALL_ATTENTION_FUNCTIONS["eager"] = _pruned_eager_attention_forward
    _PATCHED_QWEN3_MODULE = modeling_qwen3


@contextmanager
def pruning_ratio(ratio: float | None):
    global _ACTIVE_PRUNE_RATIO
    previous = _ACTIVE_PRUNE_RATIO
    _ACTIVE_PRUNE_RATIO = ratio
    try:
        yield
    finally:
        _ACTIVE_PRUNE_RATIO = previous


@torch.inference_mode()
def compute_eval_loss(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    prefill_tokens: int,
    eval_tokens: int,
    chunk_size: int,
    input_device: torch.device,
    prune_ratio: float | None,
) -> tuple[float, float, int]:
    past_key_values, prev_logits = prefill_cache(model, input_ids, prefill_tokens, chunk_size, input_device, True)
    if prev_logits is None:
        raise RuntimeError("Prefill did not return last logits.")
    total_loss = 0.0
    total_count = 0
    eval_end = prefill_tokens + eval_tokens
    total_chunks = math.ceil(eval_tokens / chunk_size)
    for chunk_idx, start in enumerate(range(prefill_tokens, eval_end, chunk_size), start=1):
        end = min(start + chunk_size, eval_end)
        chunk = input_ids[:, start:end].to(input_device)
        kwargs: dict[str, Any] = {
            "input_ids": chunk,
            "use_cache": True,
            "return_dict": True,
            "output_attentions": False,
            "output_hidden_states": False,
            "cache_position": torch.arange(start, end, device=input_device),
        }
        if past_key_values is not None:
            kwargs["past_key_values"] = past_key_values
        label = "baseline" if prune_ratio is None else f"ratio {prune_ratio:g}"
        print(f"ppl {label} chunk {chunk_idx}/{total_chunks}: tokens {start}-{end - 1}", flush=True)
        with pruning_ratio(prune_ratio):
            outputs = model_forward(model, kwargs)
        logits = outputs.logits
        shifted_logits = torch.cat([prev_logits.unsqueeze(1), logits[:, :-1, :]], dim=1)
        loss = F.cross_entropy(
            shifted_logits.reshape(-1, shifted_logits.shape[-1]).float(),
            chunk.reshape(-1),
            reduction="sum",
        )
        total_loss += float(loss)
        total_count += int(chunk.numel())
        prev_logits = logits[:, -1, :].detach()
        past_key_values = outputs.past_key_values
        del outputs, chunk, logits, shifted_logits, loss
        if input_device.type == "cuda":
            torch.cuda.empty_cache()
    mean_loss = total_loss / max(1, total_count)
    return mean_loss, math.exp(mean_loss), total_count


def compute_ppl_rows(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    prefill_tokens: int,
    eval_tokens: int,
    chunk_size: int,
    input_device: torch.device,
    ratios: list[float],
) -> list[dict[str, Any]]:
    install_qwen3_attention_patch()
    rows: list[dict[str, Any]] = []
    baseline_loss, baseline_ppl, baseline_count = compute_eval_loss(
        model, input_ids, prefill_tokens, eval_tokens, chunk_size, input_device, None
    )
    rows.append(
        {
            "mode": "baseline",
            "ratio": "",
            "kept_percent": 100.0,
            "loss": baseline_loss,
            "ppl": baseline_ppl,
            "token_count": baseline_count,
        }
    )
    for ratio in ratios:
        loss, ppl, count = compute_eval_loss(
            model, input_ids, prefill_tokens, eval_tokens, chunk_size, input_device, ratio
        )
        rows.append(
            {
                "mode": "top_ratio",
                "ratio": ratio,
                "kept_percent": 100.0 * ratio,
                "loss": loss,
                "ppl": ppl,
                "token_count": count,
            }
        )
    return rows


def plot_outputs(
    output_dir: Path,
    cos_rows: list[dict[str, Any]],
    ppl_rows: list[dict[str, Any]],
    layer_indices: list[int],
    head_indices: list[int],
    ratios: list[float],
    dpi: int,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: list[str] = []
    cos_by_head = {
        (int(row["layer"]), int(row["head"]), float(row["ratio"])): float(row["mean_cosine"])
        for row in cos_rows
        if row["mean_cosine"] != ""
    }
    if cos_rows:
        for layer in layer_indices:
            for head in head_indices:
                values = [cos_by_head.get((layer, head, ratio), float("nan")) for ratio in ratios]
                plot_dir = output_dir / "plots" / f"layer_{layer:02d}" / f"head_{head:02d}"
                plot_dir.mkdir(parents=True, exist_ok=True)
                fig, ax = plt.subplots(figsize=(7, 4), dpi=dpi)
                ax.plot([100.0 * ratio for ratio in ratios], values, marker="o", linewidth=1.5)
                ax.set_xlabel("Kept top attention tokens (%)")
                ax.set_ylabel("Mean cosine")
                ax.set_ylim(0.0, 1.02)
                ax.grid(True, alpha=0.25)
                ax.set_title(f"Attention pruning cosine L{layer} H{head}")
                fig.tight_layout()
                path = plot_dir / "mean_cosine_by_keep_ratio.png"
                fig.savefig(path)
                plt.close(fig)
                paths.append(str(path))

                fig, ax = plt.subplots(figsize=(7, 4), dpi=dpi)
                ax.plot([100.0 * ratio for ratio in ratios], values, marker="o", linewidth=1.5)
                ax.set_xscale("log")
                ax.set_xlabel("Kept top attention tokens (%)")
                ax.set_ylabel("Mean cosine")
                ax.set_ylim(0.0, 1.02)
                ax.grid(True, which="both", alpha=0.25)
                ax.set_title(f"Attention pruning cosine L{layer} H{head} (log x)")
                fig.tight_layout()
                path = plot_dir / "mean_cosine_by_keep_ratio_logx.png"
                fig.savefig(path)
                plt.close(fig)
                paths.append(str(path))

    if ppl_rows:
        sorted_ppl = sorted(
            ppl_rows,
            key=lambda row: 101.0 if row["mode"] == "baseline" else float(row["kept_percent"]),
        )
        x_values = [float(row["kept_percent"]) for row in sorted_ppl]
        y_values = [float(row["ppl"]) for row in sorted_ppl]
        labels = ["baseline" if row["mode"] == "baseline" else f"{float(row['kept_percent']):g}%" for row in sorted_ppl]
        fig, ax = plt.subplots(figsize=(8, 4), dpi=dpi)
        ax.plot(x_values, y_values, marker="o", linewidth=1.5)
        ax.set_xlabel("Kept top attention tokens (%)")
        ax.set_ylabel("PPL")
        ax.grid(True, alpha=0.25)
        ax.set_title("Attention pruning PPL")
        ax.set_xticks(x_values)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        fig.tight_layout()
        path = output_dir / "plots" / "ppl_by_keep_ratio.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
        plt.close(fig)
        paths.append(str(path))

        fig, ax = plt.subplots(figsize=(8, 4), dpi=dpi)
        ax.plot(x_values, y_values, marker="o", linewidth=1.5)
        ax.set_xscale("log")
        ax.set_xlabel("Kept top attention tokens (%)")
        ax.set_ylabel("PPL")
        ax.grid(True, which="both", alpha=0.25)
        ax.set_title("Attention pruning PPL (log x)")
        ax.set_xticks(x_values)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        fig.tight_layout()
        path = output_dir / "plots" / "ppl_by_keep_ratio_logx.png"
        fig.savefig(path)
        plt.close(fig)
        paths.append(str(path))
    return paths


def main() -> None:
    args = parse_args()
    ratios = parse_float_list(args.ratios)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    text = read_text_prefix(Path(args.text_path), args.max_chars)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    token_ids = tokenizer(text, add_special_tokens=args.add_special_tokens)["input_ids"]
    if args.append_eos and tokenizer.eos_token_id is not None:
        token_ids.append(tokenizer.eos_token_id)

    eval_tokens = args.eval_tokens
    if args.eval_last_tokens_only:
        if len(token_ids) <= eval_tokens:
            raise ValueError(
                f"Tokenization produced {len(token_ids)} tokens, "
                f"not enough to evaluate the last {eval_tokens} tokens with a non-empty prefill."
            )
        prefill_tokens = len(token_ids) - eval_tokens
    else:
        prefill_tokens = args.prefill_tokens
        total_tokens_needed = prefill_tokens + eval_tokens
        if args.require_total_tokens and len(token_ids) < total_tokens_needed:
            raise ValueError(f"Tokenization produced {len(token_ids)} tokens, fewer than {total_tokens_needed}.")
        token_ids = token_ids[:total_tokens_needed]
    input_ids = torch.tensor(token_ids, dtype=torch.long).view(1, -1)

    requested_device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model_dtype = resolve_dtype(args.dtype, requested_device)
    load_kwargs: dict[str, Any] = {"trust_remote_code": True, "torch_dtype": model_dtype}
    if args.device_map.lower() != "none":
        load_kwargs["device_map"] = args.device_map
    if args.attn_implementation.lower() != "auto":
        load_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **load_kwargs)
    if args.device_map.lower() == "none":
        model = model.to(requested_device)
    model.eval()
    model.config.use_cache = True

    layer_count = int(getattr(model.config, "num_hidden_layers"))
    head_count = int(getattr(model.config, "num_attention_heads"))
    layer_indices = parse_index_spec(args.layers, layer_count, "layers")
    head_indices = parse_index_spec(args.heads, head_count, "heads")
    input_device = pick_input_device(model, requested_device)

    cos_rows: list[dict[str, Any]] = []
    if args.compute_cos:
        cos_rows = compute_cos_metrics(
            model,
            input_ids,
            prefill_tokens,
            eval_tokens,
            args.chunk_size,
            input_device,
            layer_indices,
            head_indices,
            ratios,
            output_dir,
            args.save_cos_per_token,
            args.cos_csv_flush_rows,
        )
        write_csv(output_dir / "cos_summary_by_head.csv", cos_rows, cos_summary_fields())

    ppl_rows: list[dict[str, Any]] = []
    if args.compute_ppl:
        ppl_rows = compute_ppl_rows(
            model,
            input_ids,
            prefill_tokens,
            eval_tokens,
            args.chunk_size,
            input_device,
            ratios,
        )
        write_csv(output_dir / "ppl_by_ratio.csv", ppl_rows, ppl_fields())

    plot_paths = (
        plot_outputs(output_dir, cos_rows, ppl_rows, layer_indices, head_indices, ratios, args.plot_dpi)
        if args.make_plots
        else []
    )
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "resolved": {
                    "prefill_tokens": prefill_tokens,
                    "eval_tokens": eval_tokens,
                    "eval_last_tokens_only": args.eval_last_tokens_only,
                    "total_tokenized_tokens_used": int(input_ids.numel()),
                    "layers": layer_indices,
                    "heads": head_indices,
                    "ratios": ratios,
                },
                "paths": {
                    "cos_per_token": str(output_dir / "cos_per_token.csv") if args.compute_cos and args.save_cos_per_token else None,
                    "cos_summary_by_head": str(output_dir / "cos_summary_by_head.csv") if args.compute_cos else None,
                    "ppl_by_ratio": str(output_dir / "ppl_by_ratio.csv") if args.compute_ppl else None,
                    "plots": plot_paths,
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"wrote outputs to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
