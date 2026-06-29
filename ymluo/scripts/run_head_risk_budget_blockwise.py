from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
SERVER_SRC = Path("/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/src")
if SERVER_SRC.exists() and str(SERVER_SRC) not in sys.path:
    sys.path.insert(0, str(SERVER_SRC))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_qwen3_top2_head_limit3_ppl import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    attention_mode,
    clone_past_key_values,
    install_qwen3_attention_patch,
    model_forward,
    pick_input_device,
    prefill_cache,
    read_text_prefix,
    resolve_dtype,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Head-granular counterfactual risk-budgeted KV compression. "
            "Each block calibrates several full-head budgets, then selects the most compressed "
            "candidate whose counterfactual loss tail stays inside a risk budget."
        )
    )
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--text_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prefill_tokens", type=int, default=8192)
    parser.add_argument("--num_blocks", type=int, default=4)
    parser.add_argument("--calibration_tokens", type=int, default=16)
    parser.add_argument("--eval_tokens_per_block", type=int, default=128)
    parser.add_argument("--full_head_candidates", default="16,14,12,10,8")
    parser.add_argument("--head_orders", default="identity,reverse,evenfirst,oddfirst,rotated")
    parser.add_argument("--recent_tokens", type=int, default=512)
    parser.add_argument("--safe_delta_loss", type=float, default=0.01)
    parser.add_argument("--risk_max_gap", type=float, default=0.20)
    parser.add_argument("--risk_positive_ratio", type=float, default=0.65)
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument("--max_chars", type=int, default=4_000_000)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--log_every", type=int, default=100000)
    return parser.parse_args()


def parse_int_list(raw: str) -> list[int]:
    values: list[int] = []
    seen: set[int] = set()
    for part in raw.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        value = int(stripped)
        if value not in seen:
            values.append(value)
            seen.add(value)
    return values


def parse_str_list(raw: str) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for part in raw.split(","):
        value = part.strip().lower()
        if value and value not in seen:
            values.append(value)
            seen.add(value)
    return values


def head_order(name: str, head_count: int, layer_idx: int) -> list[int]:
    heads = list(range(head_count))
    if name == "identity":
        return heads
    if name == "reverse":
        return list(reversed(heads))
    if name == "evenfirst":
        return list(range(0, head_count, 2)) + list(range(1, head_count, 2))
    if name == "oddfirst":
        return list(range(1, head_count, 2)) + list(range(0, head_count, 2))
    if name == "rotated":
        shift = layer_idx % head_count
        return heads[shift:] + heads[:shift]
    raise ValueError(f"unknown head order: {name}")


def write_full_head_map(path: Path, order_name: str, layer_count: int, head_count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "kind": "head_risk_budget_order",
            "order": order_name,
            "head_count": head_count,
            "layer_count": layer_count,
        },
        "top_heads_by_layer": {
            str(layer_idx): head_order(order_name, head_count, layer_idx) for layer_idx in range(layer_count)
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def mode_context(mode: str, full_head_map_path: str = ""):
    return attention_mode(
        mode,
        0.02,
        3,
        True,
        0,
        0,
        None,
        full_head_map_path=full_head_map_path,
    )


def summarize_token_risk(token_rows: list[dict[str, Any]]) -> dict[str, float]:
    if not token_rows:
        return {
            "risk_mean_loss_gap": 0.0,
            "risk_max_loss_gap": 0.0,
            "risk_positive_ratio": 0.0,
            "risk_positive_mean_gap": 0.0,
        }
    gaps = [float(row["loss_gap"]) for row in token_rows]
    positive = [gap for gap in gaps if gap > 0.0]
    return {
        "risk_mean_loss_gap": sum(gaps) / len(gaps),
        "risk_max_loss_gap": max(gaps),
        "risk_positive_ratio": len(positive) / len(gaps),
        "risk_positive_mean_gap": sum(positive) / max(1, len(positive)),
    }


def candidate_saved_fraction(full_heads: int, head_count: int) -> float:
    return max(0, head_count - full_heads) / max(1, head_count)


def choose_candidate(
    rows: list[dict[str, Any]],
    baseline_loss: float,
    head_count: int,
    safe_delta_loss: float,
    risk_max_gap: float,
    risk_positive_ratio: float,
) -> tuple[dict[str, Any], str]:
    candidates = [row for row in rows if row["kind"] == "calibration_candidate"]
    safe = [
        row
        for row in candidates
        if float(row["loss"]) <= baseline_loss + safe_delta_loss
        and float(row.get("risk_max_loss_gap", 0.0)) <= risk_max_gap
        and float(row.get("risk_positive_ratio", 0.0)) <= risk_positive_ratio
    ]
    if safe:
        selected = max(
            safe,
            key=lambda row: (
                candidate_saved_fraction(int(row["full_heads"]), head_count),
                -float(row.get("risk_max_loss_gap", 0.0)),
                -max(0.0, float(row["loss"]) - baseline_loss),
            ),
        )
        return selected, "max_compression_within_risk_budget"
    selected = min(
        candidates,
        key=lambda row: (
            max(0.0, float(row["loss"]) - baseline_loss),
            float(row.get("risk_max_loss_gap", 0.0)),
            float(row.get("risk_positive_ratio", 0.0)),
            -candidate_saved_fraction(int(row["full_heads"]), head_count),
        ),
    )
    return selected, "no_safe_candidate_min_loss_risk"


@torch.inference_mode()
def eval_segment(
    *,
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    start_token: int,
    token_count: int,
    input_device: torch.device,
    initial_past_key_values: Any,
    initial_prev_logits: torch.Tensor,
    mode: str,
    full_head_map_path: str = "",
    return_token_records: bool = False,
    log_prefix: str = "",
    log_every: int = 100000,
) -> dict[str, Any]:
    past_key_values = clone_past_key_values(initial_past_key_values)
    prev_logits = initial_prev_logits.detach().clone()
    total_loss = 0.0
    total_count = 0
    token_records: list[dict[str, Any]] = []
    started = time.perf_counter()
    end_token = start_token + token_count
    for step, token_index in enumerate(range(start_token, end_token), start=1):
        prev_logits_float = prev_logits.float()
        top2 = torch.topk(prev_logits_float, k=2, dim=-1).values
        margin = float((top2[:, 0] - top2[:, 1]).mean())
        log_probs = F.log_softmax(prev_logits_float, dim=-1)
        probs = log_probs.exp()
        entropy = float((-(probs * log_probs).sum(dim=-1)).mean())
        top1_prob = float(probs.max(dim=-1).values.mean())
        if log_every <= 1 or step == 1 or step == token_count or step % log_every == 0:
            print(f"{log_prefix} step {step}/{token_count}: token {token_index}, mode={mode}", flush=True)
        chunk = input_ids[:, token_index : token_index + 1].to(input_device)
        kwargs: dict[str, Any] = {
            "input_ids": chunk,
            "use_cache": True,
            "return_dict": True,
            "output_attentions": False,
            "output_hidden_states": False,
            "cache_position": torch.arange(token_index, token_index + 1, device=input_device),
        }
        if past_key_values is not None:
            kwargs["past_key_values"] = past_key_values
        with mode_context(mode, full_head_map_path):
            outputs = model_forward(model, kwargs)
        logits = outputs.logits
        labels = input_ids[:, token_index : token_index + 1].to(input_device)
        shifted_logits = prev_logits.unsqueeze(1)
        loss = F.cross_entropy(
            shifted_logits.reshape(-1, shifted_logits.shape[-1]).float(),
            labels.reshape(-1),
            reduction="sum",
        )
        total_loss += float(loss.item())
        total_count += 1
        if return_token_records:
            token_records.append(
                {
                    "step": step,
                    "token_index": token_index,
                    "loss": float(loss.item()),
                    "margin": margin,
                    "entropy": entropy,
                    "top1_prob": top1_prob,
                }
            )
        prev_logits = logits[:, -1, :].detach()
        past_key_values = outputs.past_key_values
        del outputs, logits, labels, shifted_logits, loss, chunk, prev_logits_float, top2, log_probs, probs
    seconds = time.perf_counter() - started
    mean_loss = total_loss / max(1, total_count)
    return {
        "loss": mean_loss,
        "ppl": math.exp(mean_loss),
        "token_count": total_count,
        "seconds": seconds,
        "token_records": token_records,
        "final_past_key_values": past_key_values,
        "final_prev_logits": prev_logits,
    }


def main() -> None:
    args = parse_args()
    if args.attn_implementation and args.attn_implementation != "eager":
        raise ValueError("head risk budget fullh modes require --attn_implementation eager")
    output_dir = Path(args.output_dir)
    map_dir = output_dir / "head_maps"
    output_dir.mkdir(parents=True, exist_ok=True)
    map_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    text = read_text_prefix(Path(args.text_path), args.max_chars)
    input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
    required = args.prefill_tokens + args.num_blocks * (args.calibration_tokens + args.eval_tokens_per_block)
    if input_ids.shape[-1] < required:
        raise ValueError(f"not enough tokens: need {required}, got {input_ids.shape[-1]}")
    input_ids = input_ids[:, :required]

    load_kwargs: dict[str, Any] = {"trust_remote_code": True, "torch_dtype": dtype}
    if args.device_map:
        load_kwargs["device_map"] = args.device_map
    if args.attn_implementation:
        load_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **load_kwargs)
    model.eval()
    model.config.use_cache = True
    install_qwen3_attention_patch()
    input_device = pick_input_device(model, device)
    layer_count = int(getattr(model.config, "num_hidden_layers"))
    head_count = int(getattr(model.config, "num_attention_heads"))

    full_head_candidates = sorted(
        {min(max(0, value), head_count) for value in parse_int_list(args.full_head_candidates)},
        reverse=True,
    )
    if head_count not in full_head_candidates:
        full_head_candidates.insert(0, head_count)
    orders = parse_str_list(args.head_orders)
    order_maps: dict[str, Path] = {}
    for order in orders:
        map_path = map_dir / f"{order}_heads.json"
        write_full_head_map(map_path, order, layer_count, head_count)
        order_maps[order] = map_path

    candidate_specs: list[dict[str, Any]] = []
    for full_heads in full_head_candidates:
        use_orders = [orders[0]] if full_heads == head_count else orders
        for order in use_orders:
            mode = f"fullh{full_heads}recent{args.recent_tokens}attn"
            candidate_specs.append(
                {
                    "full_heads": full_heads,
                    "compressed_heads": head_count - full_heads,
                    "order": order,
                    "mode": mode,
                    "map_path": str(order_maps[order]),
                }
            )

    print(f"head_count={head_count} layer_count={layer_count} candidates={len(candidate_specs)}", flush=True)
    print("starting shared prefill cache", flush=True)
    initial_past, initial_logits = prefill_cache(
        model,
        input_ids,
        args.prefill_tokens,
        args.chunk_size,
        input_device,
    )

    rows: list[dict[str, Any]] = []
    token_risk_rows: list[dict[str, Any]] = []
    block_past = initial_past
    block_logits = initial_logits
    cursor = args.prefill_tokens
    for block_idx in range(args.num_blocks):
        block_rows: list[dict[str, Any]] = []
        baseline_cal = eval_segment(
            model=model,
            input_ids=input_ids,
            start_token=cursor,
            token_count=args.calibration_tokens,
            input_device=input_device,
            initial_past_key_values=block_past,
            initial_prev_logits=block_logits,
            mode="baseline",
            return_token_records=True,
            log_prefix=f"block {block_idx} cal baseline",
            log_every=args.log_every,
        )
        baseline_by_step = {int(row["step"]): row for row in baseline_cal["token_records"]}
        baseline_row = {
            "kind": "calibration_baseline",
            "block": block_idx,
            "loss": baseline_cal["loss"],
            "ppl": baseline_cal["ppl"],
            "seconds": baseline_cal["seconds"],
            "full_heads": head_count,
            "compressed_heads": 0,
            "order": "baseline",
        }
        rows.append(baseline_row)
        block_rows.append(baseline_row)

        for spec in candidate_specs:
            cal = eval_segment(
                model=model,
                input_ids=input_ids,
                start_token=cursor,
                token_count=args.calibration_tokens,
                input_device=input_device,
                initial_past_key_values=block_past,
                initial_prev_logits=block_logits,
                mode=spec["mode"],
                full_head_map_path=spec["map_path"],
                return_token_records=True,
                log_prefix=f"block {block_idx} cal {spec['order']} K{spec['full_heads']}",
                log_every=args.log_every,
            )
            risk_rows: list[dict[str, Any]] = []
            for token_row in cal["token_records"]:
                baseline_token = baseline_by_step[int(token_row["step"])]
                risk_row = {
                    "block": block_idx,
                    "order": spec["order"],
                    "full_heads": spec["full_heads"],
                    "compressed_heads": spec["compressed_heads"],
                    "step": int(token_row["step"]),
                    "token_index": int(token_row["token_index"]),
                    "baseline_loss": float(baseline_token["loss"]),
                    "compressed_loss": float(token_row["loss"]),
                    "loss_gap": float(token_row["loss"]) - float(baseline_token["loss"]),
                    "compressed_margin": float(token_row["margin"]),
                    "compressed_entropy": float(token_row["entropy"]),
                    "compressed_top1_prob": float(token_row["top1_prob"]),
                }
                token_risk_rows.append(risk_row)
                risk_rows.append({"loss_gap": risk_row["loss_gap"]})
            risk_summary = summarize_token_risk(risk_rows)
            row = {
                "kind": "calibration_candidate",
                "block": block_idx,
                "candidate": f"{spec['order']}:K{spec['full_heads']}",
                "order": spec["order"],
                "mode": spec["mode"],
                "map_path": spec["map_path"],
                "full_heads": spec["full_heads"],
                "compressed_heads": spec["compressed_heads"],
                "head_compression_fraction": candidate_saved_fraction(spec["full_heads"], head_count),
                "loss": cal["loss"],
                "ppl": cal["ppl"],
                "seconds": cal["seconds"],
                "delta_loss": cal["loss"] - baseline_cal["loss"],
                "delta_ppl": cal["ppl"] - baseline_cal["ppl"],
                **risk_summary,
            }
            rows.append(row)
            block_rows.append(row)

        chosen, reason = choose_candidate(
            block_rows,
            float(baseline_cal["loss"]),
            head_count,
            args.safe_delta_loss,
            args.risk_max_gap,
            args.risk_positive_ratio,
        )
        print(
            "block {} chosen {} reason={} delta_loss={:+.6f} max_gap={:.6f} pos_ratio={:.4f}".format(
                block_idx,
                chosen["candidate"],
                reason,
                float(chosen["loss"]) - float(baseline_cal["loss"]),
                float(chosen.get("risk_max_loss_gap", 0.0)),
                float(chosen.get("risk_positive_ratio", 0.0)),
            ),
            flush=True,
        )

        advanced = eval_segment(
            model=model,
            input_ids=input_ids,
            start_token=cursor,
            token_count=args.calibration_tokens,
            input_device=input_device,
            initial_past_key_values=block_past,
            initial_prev_logits=block_logits,
            mode="baseline",
            log_prefix=f"block {block_idx} advance calibration",
            log_every=args.log_every,
        )
        eval_start = cursor + args.calibration_tokens
        baseline_eval = eval_segment(
            model=model,
            input_ids=input_ids,
            start_token=eval_start,
            token_count=args.eval_tokens_per_block,
            input_device=input_device,
            initial_past_key_values=advanced["final_past_key_values"],
            initial_prev_logits=advanced["final_prev_logits"],
            mode="baseline",
            log_prefix=f"block {block_idx} eval baseline",
            log_every=args.log_every,
        )
        head_eval = eval_segment(
            model=model,
            input_ids=input_ids,
            start_token=eval_start,
            token_count=args.eval_tokens_per_block,
            input_device=input_device,
            initial_past_key_values=advanced["final_past_key_values"],
            initial_prev_logits=advanced["final_prev_logits"],
            mode=str(chosen["mode"]),
            full_head_map_path=str(chosen["map_path"]),
            log_prefix=f"block {block_idx} eval {chosen['candidate']}",
            log_every=args.log_every,
        )
        eval_row = {
            "kind": "head_risk_eval",
            "block": block_idx,
            "candidate": chosen["candidate"],
            "order": chosen["order"],
            "mode": chosen["mode"],
            "full_heads": chosen["full_heads"],
            "compressed_heads": chosen["compressed_heads"],
            "head_compression_fraction": chosen["head_compression_fraction"],
            "selection_reason": reason,
            "loss": head_eval["loss"],
            "ppl": head_eval["ppl"],
            "seconds": head_eval["seconds"],
            "baseline_loss": baseline_eval["loss"],
            "baseline_ppl": baseline_eval["ppl"],
            "baseline_seconds": baseline_eval["seconds"],
            "delta_loss": head_eval["loss"] - baseline_eval["loss"],
            "delta_ppl": head_eval["ppl"] - baseline_eval["ppl"],
            "token_count": head_eval["token_count"],
            "calibration_delta_loss": float(chosen["loss"]) - float(baseline_cal["loss"]),
            "calibration_risk_max_loss_gap": chosen.get("risk_max_loss_gap", 0.0),
            "calibration_risk_positive_ratio": chosen.get("risk_positive_ratio", 0.0),
        }
        rows.append(eval_row)
        block_past = baseline_eval["final_past_key_values"]
        block_logits = baseline_eval["final_prev_logits"]
        cursor += args.calibration_tokens + args.eval_tokens_per_block

    csv_path = output_dir / "head_risk_blockwise_results.csv"
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    risk_csv_path = output_dir / "head_risk_calibration_token_risk.csv"
    if token_risk_rows:
        risk_fieldnames = sorted({key for row in token_risk_rows for key in row.keys()})
        with risk_csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=risk_fieldnames)
            writer.writeheader()
            writer.writerows(token_risk_rows)

    eval_rows = [row for row in rows if row["kind"] == "head_risk_eval"]
    total_tokens = sum(int(row["token_count"]) for row in eval_rows)
    total_loss = sum(float(row["loss"]) * int(row["token_count"]) for row in eval_rows)
    total_baseline_loss = sum(float(row["baseline_loss"]) * int(row["token_count"]) for row in eval_rows)
    total_seconds = sum(float(row["seconds"]) for row in eval_rows)
    total_baseline_seconds = sum(float(row["baseline_seconds"]) for row in eval_rows)
    summary_path = output_dir / "summary.md"
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("# Head risk-budgeted blockwise experiment\n\n")
        handle.write(f"- text: `{args.text_path}`\n")
        handle.write(f"- prefill: `{args.prefill_tokens}`\n")
        handle.write(f"- blocks: `{args.num_blocks}`\n")
        handle.write(f"- calibration/eval per block: `{args.calibration_tokens}/{args.eval_tokens_per_block}`\n")
        handle.write(f"- full_head_candidates: `{args.full_head_candidates}`\n")
        handle.write(f"- head_orders: `{args.head_orders}`\n")
        handle.write(f"- recent_tokens: `{args.recent_tokens}`\n")
        handle.write(f"- safe_delta_loss: `{args.safe_delta_loss}`\n")
        handle.write(f"- risk_max_gap: `{args.risk_max_gap}`\n")
        handle.write(f"- risk_positive_ratio: `{args.risk_positive_ratio}`\n")
        handle.write(f"- aggregate baseline PPL: `{math.exp(total_baseline_loss / max(1, total_tokens)):.6f}`\n")
        handle.write(f"- aggregate method PPL: `{math.exp(total_loss / max(1, total_tokens)):.6f}`\n")
        handle.write(f"- aggregate PPL ratio: `{math.exp((total_loss - total_baseline_loss) / max(1, total_tokens)):.6f}`\n")
        handle.write(f"- aggregate seconds ratio: `{total_seconds / max(1e-9, total_baseline_seconds):.6f}`\n\n")
        handle.write("| block | candidate | delta_loss | full_heads | compressed_heads | seconds_ratio | reason |\n")
        handle.write("| ---: | --- | ---: | ---: | ---: | ---: | --- |\n")
        for row in eval_rows:
            handle.write(
                f"| {row['block']} | `{row['candidate']}` | {float(row['delta_loss']):.6f} | "
                f"{row['full_heads']} | {row['compressed_heads']} | "
                f"{float(row['seconds']) / max(1e-9, float(row['baseline_seconds'])):.6f} | "
                f"{row['selection_reason']} |\n"
            )
    print(f"wrote {csv_path}")
    if token_risk_rows:
        print(f"wrote {risk_csv_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
