from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_controlled_public_kv_benchmark_v1 as lb  # noqa: E402
from evaluate_qwen3_top2_head_limit3_ppl import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    pick_input_device,
    resolve_dtype,
)
from run_critical_position_budget_probe_20260715 import (  # noqa: E402
    causal_logit_features,
    summarize_attention_records,
)
from run_head_top2_targeted_ppl_20260714 import (  # noqa: E402
    collect_head_top_fraction_stats,
    head_qabs_sampled_mass_mode,
    head_top_fraction_mode,
    install_llama_head_top_fraction_patch,
    parse_int_list,
    set_attention_implementation,
)
from run_multitopic_lpcm_ppl_20260714 import (  # noqa: E402
    TOPICS,
    encode_topic_stream,
    make_bundle,
    topic_names,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect causal low-budget hidden states for budget routing.")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--topics", default="sports,medicine")
    parser.add_argument("--window_indices", default="0,1,2")
    parser.add_argument("--history_tokens", type=int, default=32_000)
    parser.add_argument("--query_tokens", type=int, default=256)
    parser.add_argument("--eval_tokens", type=int, default=256)
    parser.add_argument("--window_stride_tokens", type=int, default=32_512)
    parser.add_argument("--top_fraction", type=float, default=0.01)
    parser.add_argument("--attention_mode", choices=["top_fraction", "qabs_scan"], default="top_fraction")
    parser.add_argument("--hidden_layers", default="-1")
    parser.add_argument("--qabs_dim_count", type=int, default=16)
    parser.add_argument("--qabs_use_cuda_kernels", action="store_true")
    parser.add_argument("--prefill_chunk_tokens", type=int, default=2048)
    parser.add_argument("--dataset_cache_dir", default="/home/fdong/ymluo/datasets/sklearn")
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_model(args: argparse.Namespace) -> tuple[Any, torch.nn.Module, torch.device]:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = resolve_dtype(args.dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
        "attn_implementation": "sdpa",
    }
    if args.device_map:
        kwargs["device_map"] = args.device_map
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **kwargs)
    model.eval()
    model.config.use_cache = True
    return tokenizer, model, pick_input_device(model, device)


@torch.inference_mode()
def run_hidden_token(
    model: torch.nn.Module,
    token_id: int,
    cache: Any,
    position: int,
    input_device: torch.device,
    hidden_layers: tuple[int, ...],
) -> tuple[Any, torch.Tensor, torch.Tensor, dict[str, float], float]:
    attention_records: list[dict[str, Any]] = []
    started = time.perf_counter()
    with collect_head_top_fraction_stats(attention_records):
        outputs = lb.model_forward(
            model,
            {
                "input_ids": torch.tensor([[int(token_id)]], dtype=torch.long, device=input_device),
                "past_key_values": cache,
                "use_cache": True,
                "return_dict": True,
                "output_attentions": False,
                "output_hidden_states": True,
                "cache_position": torch.tensor([position], dtype=torch.long, device=input_device),
            },
        )
    resolved_layers = tuple(
        layer if layer >= 0 else len(outputs.hidden_states) + layer for layer in hidden_layers
    )
    hidden = torch.stack(
        [
            outputs.hidden_states[layer][:, -1, :]
            .detach()
            .to("cpu", dtype=torch.float16)
            .squeeze(0)
            for layer in resolved_layers
        ],
        dim=0,
    )
    if len(resolved_layers) == 1:
        hidden = hidden.squeeze(0)
    return (
        outputs.past_key_values,
        outputs.logits[:, -1, :].detach(),
        hidden,
        summarize_attention_records(attention_records),
        time.perf_counter() - started,
    )


def metadata_row(
    tokenizer: Any,
    topic: str,
    window: int,
    target_index: int,
    label_id: int,
    logits: torch.Tensor,
    attention_features: dict[str, float],
    history_counts: Counter[int],
    input_id: int,
) -> dict[str, Any]:
    causal = causal_logit_features(logits)
    top1_id = int(causal["top1_id"])
    row: dict[str, Any] = {
        "topic": topic,
        "window": window,
        "target_index": target_index,
        "label_id": int(label_id),
        "label_text": tokenizer.decode(
            [int(label_id)], skip_special_tokens=False, clean_up_tokenization_spaces=False
        ).replace("\n", "\\n").replace("\r", "\\r"),
        "input_id": int(input_id),
        "input_text": tokenizer.decode(
            [int(input_id)], skip_special_tokens=False, clean_up_tokenization_spaces=False
        ).replace("\n", "\\n").replace("\r", "\\r"),
        "input_history_frequency": int(history_counts[int(input_id)]),
        "top1_text": tokenizer.decode(
            [top1_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
        ).replace("\n", "\\n").replace("\r", "\\r"),
        "top1_history_frequency": int(history_counts[top1_id]),
    }
    row.update(causal)
    row.update(attention_features)
    return row


def main() -> None:
    args = parse_args()
    topics = topic_names(args.topics)
    windows = parse_int_list(args.window_indices)
    hidden_layers = tuple(parse_int_list(args.hidden_layers))
    if not hidden_layers:
        raise ValueError("hidden_layers must be non-empty")
    if not (0.0 < args.top_fraction <= 1.0):
        raise ValueError("top_fraction must be in (0, 1]")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2, default=str), encoding="utf-8"
    )

    install_llama_head_top_fraction_patch()
    tokenizer, model, input_device = load_model(args)
    required_tokens = max(windows) * args.window_stride_tokens + args.history_tokens + args.eval_tokens
    all_metadata: list[dict[str, Any]] = []
    all_hidden: list[torch.Tensor] = []

    for topic in topics:
        stream = encode_topic_stream(
            tokenizer, TOPICS[topic], required_tokens, args.dataset_cache_dir, args.seed
        )
        for window in windows:
            start = window * args.window_stride_tokens
            history = stream[start : start + args.history_tokens]
            target_ids = stream[start + args.history_tokens : start + args.history_tokens + args.eval_tokens]
            remote_ids = history[: -args.query_tokens]
            query_ids = history[-args.query_tokens :]
            history_counts = Counter(history)
            bundle, _ = make_bundle(tokenizer, remote_ids, page_tokens=16)

            set_attention_implementation(model, "sdpa")
            with head_top_fraction_mode(None):
                cache, _ = lb.prefill_prefix(model, bundle, input_device, args.prefill_chunk_tokens)
            set_attention_implementation(model, "eager")
            previous_logits: torch.Tensor | None = None
            previous_hidden: torch.Tensor | None = None
            previous_attention: dict[str, float] = {}
            previous_input_id = int(query_ids[-1])
            if args.attention_mode == "qabs_scan":
                attention_context = head_qabs_sampled_mass_mode(
                    1.0e-6,
                    (args.top_fraction,),
                    0.0025,
                    args.qabs_dim_count,
                    args.top_fraction,
                    args.qabs_use_cuda_kernels,
                    True,
                )
            else:
                attention_context = head_top_fraction_mode(args.top_fraction)
            with attention_context:
                for offset, input_id in enumerate(query_ids):
                    previous_input_id = int(input_id)
                    cache, previous_logits, previous_hidden, previous_attention, _ = run_hidden_token(
                        model,
                        previous_input_id,
                        cache,
                        len(remote_ids) + offset,
                        input_device,
                        hidden_layers,
                    )
                if previous_logits is None or previous_hidden is None:
                    raise RuntimeError("query must be non-empty")

                all_metadata.append(
                    metadata_row(
                        tokenizer,
                        topic,
                        window,
                        0,
                        int(target_ids[0]),
                        previous_logits,
                        previous_attention,
                        history_counts,
                        previous_input_id,
                    )
                )
                all_hidden.append(previous_hidden)
                for target_index in range(args.eval_tokens - 1):
                    previous_input_id = int(target_ids[target_index])
                    cache, previous_logits, previous_hidden, previous_attention, _ = run_hidden_token(
                        model,
                        previous_input_id,
                        cache,
                        len(remote_ids) + len(query_ids) + target_index,
                        input_device,
                        hidden_layers,
                    )
                    all_metadata.append(
                        metadata_row(
                            tokenizer,
                            topic,
                            window,
                            target_index + 1,
                            int(target_ids[target_index + 1]),
                            previous_logits,
                            previous_attention,
                            history_counts,
                            previous_input_id,
                        )
                    )
                    all_hidden.append(previous_hidden)

            write_csv(args.output_dir / "metadata.csv", all_metadata)
            torch.save(
                {"metadata": all_metadata, "hidden_states": torch.stack(all_hidden, dim=0)},
                args.output_dir / "hidden_states.pt",
            )
            print(
                f"[saved] topic={topic} window={window} rows={len(all_metadata)} "
                f"shape={tuple(torch.stack(all_hidden, dim=0).shape)}",
                flush=True,
            )
            del cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
