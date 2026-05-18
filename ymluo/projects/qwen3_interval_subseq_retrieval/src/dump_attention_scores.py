from __future__ import annotations

import argparse
import csv
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[4]
FDONG_SCRIPTS_DIR = REPO_ROOT / "fdong" / "scripts"
PROJECT_SRC_DIR = Path(__file__).resolve().parent
for path in (FDONG_SCRIPTS_DIR, PROJECT_SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import models.myqwen as myqwen  # noqa: E402
from train_interval_subseq_retrieval import (  # noqa: E402
    generate_batch,
    parse_int_list,
    prepare_model,
    str2bool,
)


def parse_query_positions(value: str, seq_len: int) -> list[int] | None:
    value = value.strip().lower()
    if value in {"", "all"}:
        return None
    if value == "last":
        return [seq_len - 1]

    positions = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        pos = int(item)
        if pos < 0:
            pos += seq_len
        if pos < 0 or pos >= seq_len:
            raise ValueError(f"query position {item!r} is outside sequence length {seq_len}")
        positions.append(pos)
    return sorted(set(positions))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", default="")
    parser.add_argument("--ckpt_file", default="")
    parser.add_argument("--ckpt_step", type=int, default=2000)
    parser.add_argument("--config_dir", default="/mnt/workspace/Qwen3-0.6B")
    parser.add_argument("--output_dir", default="")

    parser.add_argument("--total_token", type=int, default=None)
    parser.add_argument("--subseq_len", type=int, default=None)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--intervals", type=parse_int_list, default=None)
    parser.add_argument("--interval_group_mode", choices=["scaled", "bounded"], default=None)
    parser.add_argument("--sample_with_replacement", type=str2bool, default=None)
    parser.add_argument("--num_hidden_layers", type=int, default=None)
    parser.add_argument("--attention_stride_pattern", type=parse_int_list, default=None)
    parser.add_argument("--residual_source_pattern", type=parse_int_list, default=None)
    parser.add_argument("--auto_resize_vocab", type=str2bool, default=None)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use_bf16", type=str2bool, default=True)
    parser.add_argument("--query_positions", default="all", help="all, last, or comma-separated 0-based positions.")
    parser.add_argument("--save_raw_scores", type=str2bool, default=True)
    parser.add_argument("--save_probabilities", type=str2bool, default=True)
    parser.add_argument("--save_format", choices=["pt", "npy"], default="pt")
    parser.add_argument("--topk", type=int, default=16)
    args = parser.parse_args()

    if not args.run_dir and not args.ckpt_file:
        raise ValueError("Provide --run_dir or --ckpt_file.")
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")
    if not args.save_raw_scores and not args.save_probabilities:
        raise ValueError("At least one of --save_raw_scores/--save_probabilities must be true.")
    return args


def load_train_config(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "train_config.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_checkpoint(args: argparse.Namespace) -> Path:
    if args.ckpt_file:
        return Path(args.ckpt_file)
    return Path(args.run_dir) / "checkpoints" / f"{args.ckpt_step}.pth"


def build_model_args(args: argparse.Namespace) -> argparse.Namespace:
    train_config = load_train_config(Path(args.run_dir)) if args.run_dir else {}

    def pick(name: str, default: Any) -> Any:
        cli_value = getattr(args, name)
        if cli_value is not None:
            return cli_value
        return train_config.get(name, default)

    model_args = argparse.Namespace(
        config_dir=args.config_dir or train_config.get("config_dir", "/mnt/workspace/Qwen3-0.6B"),
        total_token=pick("total_token", 10_000),
        subseq_len=pick("subseq_len", 4),
        seq_len=pick("seq_len", 1024),
        intervals=pick("intervals", [1]),
        interval_group_mode=pick("interval_group_mode", "scaled"),
        sample_with_replacement=pick("sample_with_replacement", True),
        num_hidden_layers=pick("num_hidden_layers", 8),
        attention_stride_pattern=pick("attention_stride_pattern", [1, 1, 4, 4, 4, 4, 1, 1]),
        residual_source_pattern=pick("residual_source_pattern", None),
        init_checkpoint=str(resolve_checkpoint(args)),
        auto_resize_vocab=pick("auto_resize_vocab", True),
        attn_implementation="eager",
    )
    return model_args


def save_tensor(path: Path, tensor: torch.Tensor, save_format: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cpu_tensor = tensor.detach().cpu().contiguous()
    if save_format == "pt":
        torch.save(cpu_tensor, path.with_suffix(".pt"))
    else:
        import numpy as np

        np.save(path.with_suffix(".npy"), cpu_tensor.float().numpy())


def summarize_head(prob: torch.Tensor, raw: torch.Tensor | None, topk: int) -> dict[str, Any]:
    prob_float = prob.float()
    entropy = -(prob_float.clamp_min(1e-30) * prob_float.clamp_min(1e-30).log()).sum(dim=-1)
    max_prob = prob_float.max(dim=-1).values
    summary = {
        "mean_entropy": float(entropy.mean().item()),
        "mean_max_probability": float(max_prob.mean().item()),
    }
    if topk > 0:
        k = min(topk, prob_float.shape[-1])
        top_values, top_indices = torch.topk(prob_float, k=k, dim=-1)
        summary["topk_mean_probability_sum"] = float(top_values.sum(dim=-1).mean().item())
        summary["last_query_topk_indices"] = top_indices[0, -1].detach().cpu().tolist()
        summary["last_query_topk_probabilities"] = top_values[0, -1].detach().cpu().tolist()
    if raw is not None:
        finite = raw.float()
        finite = finite[torch.isfinite(finite)]
        if finite.numel() > 0:
            summary["raw_finite_mean"] = float(finite.mean().item())
            summary["raw_finite_std"] = float(finite.std(unbiased=False).item())
    return summary


@contextmanager
def capture_attention_scores(
    storage: dict[int, dict[str, torch.Tensor]],
    query_positions: list[int] | None,
    save_raw_scores: bool,
    save_probabilities: bool,
):
    original = myqwen.eager_attention_forward

    def capturing_eager_attention_forward(
        module,
        query,
        key,
        value,
        attention_mask,
        scaling,
        dropout=0.0,
        **kwargs,
    ):
        key_states = myqwen.repeat_kv(key, module.num_key_value_groups)
        value_states = myqwen.repeat_kv(value, module.num_key_value_groups)

        raw_scores = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            raw_scores = raw_scores + causal_mask

        probs = F.softmax(raw_scores, dim=-1, dtype=torch.float32).to(query.dtype)
        probs = F.dropout(probs, p=dropout, training=module.training)
        attn_output = torch.matmul(probs, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()

        q_slice = query_positions if query_positions is not None else slice(None)
        captured: dict[str, torch.Tensor] = {}
        if save_raw_scores:
            captured["raw_scores"] = raw_scores[:, :, q_slice, :].detach().to(torch.float16).cpu()
        if save_probabilities:
            captured["probabilities"] = probs[:, :, q_slice, :].detach().to(torch.float16).cpu()
        storage[int(module.layer_idx)] = captured
        return attn_output, probs

    myqwen.eager_attention_forward = capturing_eager_attention_forward
    try:
        yield
    finally:
        myqwen.eager_attention_forward = original


def write_outputs(
    output_dir: Path,
    captures: dict[int, dict[str, torch.Tensor]],
    batch,
    model_args: argparse.Namespace,
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_tensor(output_dir / "input_ids", batch.source, args.save_format)
    save_tensor(output_dir / "intervals", batch.interval, args.save_format)

    metadata = {
        "ckpt_file": str(resolve_checkpoint(args)),
        "config_dir": model_args.config_dir,
        "batch_size": args.batch_size,
        "seq_len": model_args.seq_len,
        "query_positions": args.query_positions,
        "attention_stride_pattern": model_args.attention_stride_pattern,
        "num_hidden_layers": model_args.num_hidden_layers,
        "save_raw_scores": args.save_raw_scores,
        "save_probabilities": args.save_probabilities,
        "save_format": args.save_format,
        "tensor_shape_per_head": "[batch, selected_query_positions, key_positions]",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary_rows = []
    for layer_idx in sorted(captures):
        layer_dir = output_dir / f"layer_{layer_idx:02d}"
        raw_scores = captures[layer_idx].get("raw_scores")
        probabilities = captures[layer_idx].get("probabilities")
        tensor_for_heads = probabilities if probabilities is not None else raw_scores
        if tensor_for_heads is None:
            continue
        num_heads = tensor_for_heads.shape[1]
        for head_idx in range(num_heads):
            head_dir = layer_dir / f"head_{head_idx:02d}"
            raw_head = raw_scores[:, head_idx] if raw_scores is not None else None
            prob_head = probabilities[:, head_idx] if probabilities is not None else None
            if raw_head is not None:
                save_tensor(head_dir / "raw_scores", raw_head, args.save_format)
            if prob_head is not None:
                save_tensor(head_dir / "probabilities", prob_head, args.save_format)
            if prob_head is not None:
                row = {
                    "layer": layer_idx,
                    "head": head_idx,
                    **summarize_head(prob_head, raw_head, args.topk),
                }
                summary_rows.append(row)

    if summary_rows:
        fields = sorted({key for row in summary_rows for key in row})
        with (output_dir / "head_summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in summary_rows:
                writer.writerow(row)
        (output_dir / "head_summary.json").write_text(
            json.dumps(summary_rows, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def main() -> None:
    args = parse_args()
    model_args = build_model_args(args)
    ckpt_file = resolve_checkpoint(args)
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0 if device.index is None else device.index)

    output_dir = Path(args.output_dir) if args.output_dir else (
        (Path(args.run_dir) if args.run_dir else ckpt_file.parent.parent)
        / "attention_scores"
        / f"step_{args.ckpt_step}"
    )
    print(f"loading checkpoint: {ckpt_file}", flush=True)
    print(f"writing attention scores to: {output_dir}", flush=True)

    model = prepare_model(model_args, device)
    model.eval()

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    batch = generate_batch(model_args, args.batch_size, generator, device)
    query_positions = parse_query_positions(args.query_positions, model_args.seq_len)

    captures: dict[int, dict[str, torch.Tensor]] = {}
    with torch.inference_mode():
        autocast_enabled = args.use_bf16 and device.type == "cuda"
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
            with capture_attention_scores(
                captures,
                query_positions=query_positions,
                save_raw_scores=args.save_raw_scores,
                save_probabilities=args.save_probabilities,
            ):
                _ = model(input_ids=batch.source, use_cache=False, output_hidden_states=False)

    write_outputs(output_dir, captures, batch, model_args, args)
    print(f"captured layers: {sorted(captures)}", flush=True)
    print(f"done: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
