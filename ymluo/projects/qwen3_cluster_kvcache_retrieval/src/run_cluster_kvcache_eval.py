from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    from transformers import AutoModelWithLMHead as AutoModelForCausalLM
    from transformers import AutoTokenizer

from cluster_kvcache_attention import ClusterKVConfig, PROFILER, install_qwen3_cluster_attention_patch


DEFAULT_MODEL_PATH = "/mnt/workspace/Qwen3-0.6B"
DEFAULT_TEXT_PATH = "/mnt/workspace/dclm/global-shard_01_of_10/local-shard_0_of_10/part-00000.txt"


class ModuleProfiler:
    def __init__(self, enabled: bool, device: torch.device) -> None:
        self.enabled = enabled and device.type == "cuda"
        self.device = device
        self.handles: list[Any] = []
        self.pending: list[tuple[str, str, torch.cuda.Event, torch.cuda.Event]] = []
        self.records: dict[tuple[str, str], dict[str, float | int]] = {}

    def _category(self, name: str, module: torch.nn.Module) -> str | None:
        class_name = module.__class__.__name__.lower()
        leaf_name = name.rsplit(".", 1)[-1]
        if class_name == "qwen3attention" or leaf_name == "self_attn":
            return "attention_module"
        if class_name == "qwen3mlp" or leaf_name == "mlp":
            return "mlp_module"
        if "rmsnorm" in class_name or leaf_name in {"input_layernorm", "post_attention_layernorm", "norm"}:
            return "norm"
        if leaf_name in {"q_proj", "k_proj", "v_proj", "o_proj"}:
            return f"attention_{leaf_name}"
        if leaf_name in {"gate_proj", "up_proj", "down_proj"}:
            return f"mlp_{leaf_name}"
        if leaf_name in {"embed_tokens", "lm_head"}:
            return leaf_name
        return None

    def install(self, model: torch.nn.Module) -> None:
        if not self.enabled:
            return
        for name, module in model.named_modules():
            category = self._category(name, module)
            if category is None:
                continue

            def pre_hook(_module, _inputs, module_name=name):
                if not self.enabled:
                    return
                event = torch.cuda.Event(enable_timing=True)
                event.record()
                _module._module_profile_start = (module_name, event)

            def post_hook(_module, _inputs, _outputs, module_name=name, module_category=category):
                if not self.enabled:
                    return
                start_info = getattr(_module, "_module_profile_start", None)
                if start_info is None:
                    return
                _, start_event = start_info
                end_event = torch.cuda.Event(enable_timing=True)
                end_event.record()
                self.pending.append((module_category, module_name, start_event, end_event))

            self.handles.append(module.register_forward_pre_hook(pre_hook))
            self.handles.append(module.register_forward_hook(post_hook))

    def flush(self) -> None:
        if not self.enabled or not self.pending:
            return
        torch.cuda.synchronize(self.device)
        for category, name, start, end in self.pending:
            key = (category, name)
            record = self.records.setdefault(key, {"calls": 0, "elapsed_ms": 0.0})
            record["calls"] = int(record["calls"]) + 1
            record["elapsed_ms"] = float(record["elapsed_ms"]) + float(start.elapsed_time(end))
        self.pending.clear()

    def rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for (category, name), record in sorted(self.records.items()):
            calls = int(record["calls"])
            elapsed_ms = float(record["elapsed_ms"])
            rows.append(
                {
                    "category": category,
                    "module": name,
                    "calls": calls,
                    "elapsed_ms": elapsed_ms,
                    "mean_ms": elapsed_ms / calls if calls else 0.0,
                }
            )
        return rows

    def summary(self) -> dict[str, dict[str, float | int]]:
        grouped: dict[str, dict[str, float | int]] = {}
        for row in self.rows():
            category = str(row["category"])
            bucket = grouped.setdefault(category, {"calls": 0, "elapsed_ms": 0.0})
            bucket["calls"] = int(bucket["calls"]) + int(row["calls"])
            bucket["elapsed_ms"] = float(bucket["elapsed_ms"]) + float(row["elapsed_ms"])
        for bucket in grouped.values():
            calls = int(bucket["calls"])
            bucket["mean_ms"] = float(bucket["elapsed_ms"]) / calls if calls else 0.0
        return dict(sorted(grouped.items()))

    def reset(self) -> None:
        self.pending.clear()
        self.records.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.pending.clear()


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate baseline vs cluster-selected KV-cache decoding.")
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--text_path", default=DEFAULT_TEXT_PATH)
    parser.add_argument("--output_dir", default="outputs/qwen3_cluster_kvcache_retrieval")
    parser.add_argument("--modes", default="baseline,cluster")
    parser.add_argument("--prefill_tokens", type=int, default=100_000)
    parser.add_argument("--eval_tokens", type=int, default=512)
    parser.add_argument("--prefill_chunk_size", type=int, default=512)
    parser.add_argument("--cluster_size", type=int, default=50)
    parser.add_argument("--keep_ratio", type=float, default=0.02)
    parser.add_argument("--edge_ratio", type=float, default=0.01)
    parser.add_argument("--force_endpoints", type=str2bool, default=True)
    parser.add_argument("--endpoints_count_in_budget", type=str2bool, default=True)
    parser.add_argument("--max_chars", type=int, default=160_000_000)
    parser.add_argument("--add_special_tokens", type=str2bool, default=False)
    parser.add_argument("--append_eos", type=str2bool, default=False)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--profile_attention", type=str2bool, default=True)
    parser.add_argument("--profile_modules", type=str2bool, default=False)
    parser.add_argument("--warmup_eval_tokens", type=int, default=8)
    parser.add_argument("--save_token_timings", type=str2bool, default=True)
    parser.add_argument("--require_total_tokens", type=str2bool, default=True)
    return parser.parse_args()


def resolve_dtype(dtype_name: str, device: torch.device) -> torch.dtype | str:
    if dtype_name == "auto":
        return "auto"
    if device.type == "cpu":
        return torch.float32
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype_name]


def read_text_prefix(path: Path, max_chars: int) -> str:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        if max_chars > 0:
            return handle.read(max_chars)
        return handle.read()


def pick_input_device(model: torch.nn.Module, fallback: torch.device) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return fallback


def model_forward(model: torch.nn.Module, kwargs: dict[str, Any]) -> Any:
    try:
        return model(**kwargs)
    except TypeError as exc:
        if "cache_position" in kwargs and "cache_position" in str(exc):
            kwargs = dict(kwargs)
            kwargs.pop("cache_position")
            return model(**kwargs)
        raise


def cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_forward(model: torch.nn.Module, kwargs: dict[str, Any], device: torch.device) -> tuple[Any, float]:
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        outputs = model_forward(model, kwargs)
        end.record()
        torch.cuda.synchronize(device)
        return outputs, float(start.elapsed_time(end))
    start_time = time.perf_counter()
    outputs = model_forward(model, kwargs)
    return outputs, (time.perf_counter() - start_time) * 1000.0


@torch.inference_mode()
def prefill_cache(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    prefill_tokens: int,
    chunk_size: int,
    input_device: torch.device,
) -> tuple[Any, torch.Tensor, float]:
    past_key_values = None
    last_logits = None
    total_ms = 0.0
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
        outputs, elapsed_ms = timed_forward(model, kwargs, input_device)
        total_ms += elapsed_ms
        past_key_values = outputs.past_key_values
        last_logits = outputs.logits[:, -1, :].detach()
        del outputs, chunk
    if last_logits is None:
        raise RuntimeError("Prefill produced no logits.")
    return past_key_values, last_logits, total_ms


@torch.inference_mode()
def evaluate_decode(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    prefill_tokens: int,
    eval_tokens: int,
    prefill_chunk_size: int,
    input_device: torch.device,
    warmup_eval_tokens: int,
    module_profiler: ModuleProfiler | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    profile_enabled = PROFILER.enabled
    PROFILER.enabled = False
    PROFILER.reset()
    module_profile_enabled = module_profiler.enabled if module_profiler is not None else False
    if module_profiler is not None:
        module_profiler.enabled = False
        module_profiler.reset()
    past_key_values, prev_logits, prefill_ms = prefill_cache(
        model, input_ids, prefill_tokens, prefill_chunk_size, input_device
    )
    PROFILER.enabled = profile_enabled
    PROFILER.reset()
    if module_profiler is not None:
        module_profiler.enabled = module_profile_enabled
        module_profiler.reset()

    total_loss = 0.0
    total_count = 0
    measured_decode_ms = 0.0
    all_decode_ms = 0.0
    token_rows: list[dict[str, Any]] = []
    eval_end = prefill_tokens + eval_tokens

    for local_idx, pos in enumerate(range(prefill_tokens, eval_end)):
        token = input_ids[:, pos : pos + 1].to(input_device)
        loss = F.cross_entropy(prev_logits.float(), token.reshape(-1), reduction="sum")
        total_loss += float(loss)
        total_count += 1

        kwargs: dict[str, Any] = {
            "input_ids": token,
            "use_cache": True,
            "return_dict": True,
            "output_attentions": False,
            "output_hidden_states": False,
            "past_key_values": past_key_values,
            "cache_position": torch.tensor([pos], device=input_device),
        }
        outputs, elapsed_ms = timed_forward(model, kwargs, input_device)
        PROFILER.synchronize_and_flush()
        if module_profiler is not None:
            module_profiler.flush()
        past_key_values = outputs.past_key_values
        prev_logits = outputs.logits[:, -1, :].detach()
        all_decode_ms += elapsed_ms
        if local_idx >= warmup_eval_tokens:
            measured_decode_ms += elapsed_ms
        token_rows.append(
            {
                "token_index": local_idx,
                "absolute_position": pos,
                "key_len": pos + 1,
                "decode_ms": elapsed_ms,
                "measured": local_idx >= warmup_eval_tokens,
            }
        )
        del outputs, token, loss

    measured_tokens = max(0, eval_tokens - warmup_eval_tokens)
    mean_loss = total_loss / max(1, total_count)
    summary = {
        "loss": mean_loss,
        "ppl": math.exp(mean_loss),
        "token_count": total_count,
        "prefill_ms": prefill_ms,
        "decode_ms_all": all_decode_ms,
        "decode_ms_measured": measured_decode_ms,
        "decode_tokens_measured": measured_tokens,
        "decode_ms_per_token": measured_decode_ms / measured_tokens if measured_tokens else float("nan"),
        "tokens_per_second": 1000.0 * measured_tokens / measured_decode_ms if measured_decode_ms > 0 else float("nan"),
        "attention_profile": PROFILER.snapshot(),
        "module_profile": module_profiler.summary() if module_profiler is not None else {},
    }
    return summary, token_rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_inputs(args: argparse.Namespace) -> tuple[torch.Tensor, Any]:
    text = read_text_prefix(Path(args.text_path), args.max_chars)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    token_ids = tokenizer(text, add_special_tokens=args.add_special_tokens)["input_ids"]
    if args.append_eos and tokenizer.eos_token_id is not None:
        token_ids.append(tokenizer.eos_token_id)
    needed = args.prefill_tokens + args.eval_tokens
    if args.require_total_tokens and len(token_ids) < needed:
        raise ValueError(f"Tokenization produced {len(token_ids)} tokens, fewer than required {needed}.")
    token_ids = token_ids[:needed]
    return torch.tensor(token_ids, dtype=torch.long).view(1, -1), tokenizer


def load_model(args: argparse.Namespace, requested_device: torch.device) -> torch.nn.Module:
    dtype = resolve_dtype(args.dtype, requested_device)
    kwargs: dict[str, Any] = {"trust_remote_code": True, "torch_dtype": dtype}
    if args.device_map.lower() != "none":
        kwargs["device_map"] = args.device_map
    if args.attn_implementation.lower() != "auto":
        kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **kwargs)
    if args.device_map.lower() == "none":
        model = model.to(requested_device)
    model.eval()
    model.config.use_cache = True
    return model


def main() -> None:
    args = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_ids, tokenizer = load_inputs(args)
    requested_device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    summary_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {"args": vars(args), "modes": {}}

    for mode in modes:
        if mode not in {"baseline", "cluster", "edges"}:
            raise ValueError(f"Unsupported mode: {mode}")
        print(f"loading model for mode={mode}", flush=True)
        model = load_model(args, requested_device)
        input_device = pick_input_device(model, requested_device)
        cfg = ClusterKVConfig(
            mode=mode,
            cluster_size=args.cluster_size,
            keep_ratio=args.keep_ratio,
            edge_ratio=args.edge_ratio,
            force_endpoints=args.force_endpoints,
            endpoints_count_in_budget=args.endpoints_count_in_budget,
            profile=args.profile_attention,
        )
        install_qwen3_cluster_attention_patch(model, cfg)
        module_profiler = ModuleProfiler(args.profile_modules, input_device)
        module_profiler.install(model)
        print(f"evaluating mode={mode}", flush=True)
        summary, token_rows = evaluate_decode(
            model,
            input_ids,
            args.prefill_tokens,
            args.eval_tokens,
            args.prefill_chunk_size,
            input_device,
            args.warmup_eval_tokens,
            module_profiler,
        )
        cluster_count = math.ceil(args.prefill_tokens / args.cluster_size)
        keep_clusters = max(1, math.ceil(args.keep_ratio * cluster_count))
        approx_edge_tokens = 2 * max(1, math.ceil(args.edge_ratio * args.prefill_tokens))
        summary.update(
            {
                "mode": mode,
                "cluster_size": args.cluster_size,
                "keep_ratio": args.keep_ratio,
                "edge_ratio": args.edge_ratio,
                "approx_prefill_plus_one_clusters": cluster_count,
                "approx_keep_clusters": keep_clusters,
                "approx_edge_tokens": approx_edge_tokens,
            }
        )
        summaries["modes"][mode] = summary
        summary_rows.append(
            {
                "mode": mode,
                "loss": summary["loss"],
                "ppl": summary["ppl"],
                "token_count": summary["token_count"],
                "prefill_ms": summary["prefill_ms"],
                "decode_ms_all": summary["decode_ms_all"],
                "decode_ms_measured": summary["decode_ms_measured"],
                "decode_tokens_measured": summary["decode_tokens_measured"],
                "decode_ms_per_token": summary["decode_ms_per_token"],
                "tokens_per_second": summary["tokens_per_second"],
                "cluster_size": args.cluster_size,
                "keep_ratio": args.keep_ratio,
                "edge_ratio": args.edge_ratio,
                "approx_keep_clusters": keep_clusters,
                "approx_edge_tokens": approx_edge_tokens,
            }
        )
        if args.save_token_timings:
            write_csv(
                output_dir / f"token_timings_{mode}.csv",
                token_rows,
                ["token_index", "absolute_position", "key_len", "decode_ms", "measured"],
            )
        if args.profile_modules:
            write_csv(
                output_dir / f"module_profile_{mode}.csv",
                module_profiler.rows(),
                ["category", "module", "calls", "elapsed_ms", "mean_ms"],
            )
        module_profiler.close()
        del model
        if requested_device.type == "cuda":
            torch.cuda.empty_cache()

    comparisons: dict[str, Any] = {}
    if "baseline" in summaries["modes"] and "cluster" in summaries["modes"]:
        baseline = summaries["modes"]["baseline"]
        cluster = summaries["modes"]["cluster"]
        comparisons["cluster_vs_baseline"] = {
            "ppl_delta_cluster_minus_baseline": cluster["ppl"] - baseline["ppl"],
            "loss_delta_cluster_minus_baseline": cluster["loss"] - baseline["loss"],
            "decode_speedup_vs_baseline": baseline["decode_ms_per_token"] / cluster["decode_ms_per_token"],
        }
    if "baseline" in summaries["modes"] and "edges" in summaries["modes"]:
        baseline = summaries["modes"]["baseline"]
        edges = summaries["modes"]["edges"]
        comparisons["edges_vs_baseline"] = {
            "ppl_delta_edges_minus_baseline": edges["ppl"] - baseline["ppl"],
            "loss_delta_edges_minus_baseline": edges["loss"] - baseline["loss"],
            "decode_speedup_vs_baseline": baseline["decode_ms_per_token"] / edges["decode_ms_per_token"],
        }
    if comparisons:
        summaries["comparison"] = comparisons

    write_csv(
        output_dir / "summary.csv",
        summary_rows,
        [
            "mode",
            "loss",
            "ppl",
            "token_count",
            "prefill_ms",
            "decode_ms_all",
            "decode_ms_measured",
            "decode_tokens_measured",
            "decode_ms_per_token",
            "tokens_per_second",
            "cluster_size",
            "keep_ratio",
            "edge_ratio",
            "approx_keep_clusters",
            "approx_edge_tokens",
        ],
    )
    (output_dir / "summary.json").write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote outputs to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
