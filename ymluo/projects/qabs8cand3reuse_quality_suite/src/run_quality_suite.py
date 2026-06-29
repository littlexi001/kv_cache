from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
PROJECT_DIR = Path(__file__).resolve().parents[1]
EVALUATOR = REPO_ROOT / "ymluo/projects/qwen3_top2_head_limit3_ppl/src/evaluate_qwen3_top2_head_limit3_ppl.py"


@dataclass(frozen=True)
class EvalItem:
    group: str
    name: str
    path: Path
    prefill_tokens: int
    eval_tokens: int


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def read_needle_items(
    config: dict,
    max_needle_cases: int,
    needle_prefill_tokens: int,
    needle_eval_tokens: int,
) -> list[EvalItem]:
    needle_config = config.get("needle_prompts", {})
    manifest_path = REPO_ROOT / needle_config.get("manifest", "")
    if not manifest_path.exists():
        print(f"warning: needle manifest not found: {manifest_path}", file=sys.stderr)
        return []

    allowed_lengths = {int(value) for value in needle_config.get("lengths", [])}
    allowed_depths = {float(value) for value in needle_config.get("depths", [])}
    items: list[EvalItem] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            length = int(row["context_length_words_target"])
            depth = float(row["target_depth_percent"])
            if allowed_lengths and length not in allowed_lengths:
                continue
            if allowed_depths and depth not in allowed_depths:
                continue
            prompt_path = REPO_ROOT / row["prompt_path"]
            if not prompt_path.exists():
                continue
            prefill_tokens = (
                needle_prefill_tokens
                if needle_prefill_tokens > 0
                else max(256, min(12000, int(length * 1.25)))
            )
            eval_tokens = (
                needle_eval_tokens
                if needle_eval_tokens > 0
                else max(64, min(512, prefill_tokens // 8))
            )
            items.append(
                EvalItem(
                    group="needle_ppl",
                    name=row["sample_id"],
                    path=prompt_path,
                    prefill_tokens=prefill_tokens,
                    eval_tokens=eval_tokens,
                )
            )
            if max_needle_cases > 0 and len(items) >= max_needle_cases:
                break
    return items


def build_eval_items(
    config: dict,
    prefill_tokens: int,
    eval_tokens: int,
    max_needle_cases: int,
    needle_prefill_tokens: int,
    needle_eval_tokens: int,
) -> list[EvalItem]:
    items: list[EvalItem] = []
    for row in config.get("topic_texts", []):
        path = REPO_ROOT / row["path"]
        if not path.exists():
            print(f"warning: topic text not found: {path}", file=sys.stderr)
            continue
        items.append(
            EvalItem(
                group="topic_ppl",
                name=row["name"],
                path=path,
                prefill_tokens=prefill_tokens,
                eval_tokens=eval_tokens,
            )
        )
    items.extend(read_needle_items(config, max_needle_cases, needle_prefill_tokens, needle_eval_tokens))
    return items


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def run_one(args: argparse.Namespace, item: EvalItem) -> Path:
    output_dir = Path(args.output_root) / item.group / safe_name(item.name)
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        args.python_bin,
        str(EVALUATOR),
        "--model_name_or_path",
        args.model_name_or_path,
        "--text_path",
        str(item.path),
        "--output_dir",
        str(output_dir),
        "--prefill_tokens",
        str(item.prefill_tokens),
        "--eval_tokens",
        str(item.eval_tokens),
        "--chunk_size",
        str(args.chunk_size),
        "--eval_chunk_size",
        str(args.eval_chunk_size),
        "--max_chars",
        str(args.max_chars),
        "--add_special_tokens",
        "false",
        "--append_eos",
        "false",
        "--require_total_tokens",
        str(args.require_total_tokens).lower(),
        "--dtype",
        args.dtype,
        "--device",
        args.device,
        "--device_map",
        args.device_map,
        "--attn_implementation",
        args.attn_implementation,
        "--top_fraction",
        str(args.top_fraction),
        "--protect_sink_tokens",
        str(args.protect_sink_tokens),
        "--protect_recent_tokens",
        str(args.protect_recent_tokens),
        "--always_keep_self",
        "true",
        "--modes",
        args.modes,
        "--qabs_fast_path",
        "true",
        "--qabs_cuda_final_kernel",
        str(args.qabs_cuda_final_kernel).lower(),
        "--qabs_cuda_candidate_kernel",
        str(args.qabs_cuda_candidate_kernel).lower(),
        "--qabs_cuda_reuse_select_kernel",
        str(args.qabs_cuda_reuse_select_kernel).lower(),
        "--reuse_prefill_cache",
        "true",
        "--baseline_last",
        "true",
        "--disable_sparse_stats",
        "true",
        "--log_every",
        str(args.log_every),
        "--make_plots",
        str(args.make_plots).lower(),
    ]
    print(f"=== {item.group}/{item.name}: {' '.join(command)} ===", flush=True)
    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)
    return output_dir


def combine_results(output_root: Path) -> Path:
    rows: list[dict[str, str]] = []
    for csv_path in sorted(output_root.glob("*/*/ppl_by_mode.csv")):
        group = csv_path.parents[1].name
        dataset = csv_path.parent.name
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            mode_rows = list(csv.DictReader(handle))
        baseline = next((row for row in mode_rows if row["mode"] == "baseline"), None)
        baseline_ppl = float(baseline["ppl"]) if baseline else None
        baseline_seconds = float(baseline["seconds"]) if baseline else None
        for row in mode_rows:
            ppl = float(row["ppl"])
            seconds = float(row["seconds"])
            rows.append(
                {
                    "group": group,
                    "dataset": dataset,
                    "mode": row["mode"],
                    "loss": row["loss"],
                    "ppl": row["ppl"],
                    "token_count": row["token_count"],
                    "seconds": row["seconds"],
                    "ppl_delta_vs_baseline": "" if baseline_ppl is None else f"{ppl - baseline_ppl:.6f}",
                    "ppl_ratio_vs_baseline": "" if baseline_ppl is None else f"{ppl / baseline_ppl:.6f}",
                    "time_ratio_vs_baseline": "" if baseline_seconds is None else f"{seconds / baseline_seconds:.6f}",
                    "source_csv": str(csv_path),
                }
            )
    combined_path = output_root / "quality_suite_combined.csv"
    fields = [
        "group",
        "dataset",
        "mode",
        "loss",
        "ppl",
        "token_count",
        "seconds",
        "ppl_delta_vs_baseline",
        "ppl_ratio_vs_baseline",
        "time_ratio_vs_baseline",
        "source_csv",
    ]
    with combined_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    write_markdown_summary(output_root / "quality_suite_combined.md", rows)
    return combined_path


def write_markdown_summary(path: Path, rows: list[dict[str, str]]) -> None:
    lines = [
        "# QABS8Cand3Reuse quality suite summary",
        "",
        "| group | dataset | mode | ppl | ppl ratio vs baseline | time ratio vs baseline |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['group']} | {row['dataset']} | {row['mode']} | {float(row['ppl']):.4f} | "
            f"{row['ppl_ratio_vs_baseline']} | {row['time_ratio_vs_baseline']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run topic and needle PPL quality comparisons.")
    parser.add_argument("--config", default=str(PROJECT_DIR / "configs/default_eval_sets.json"))
    parser.add_argument("--output_root", default=str(PROJECT_DIR / "outputs"))
    parser.add_argument("--python_bin", default=sys.executable)
    parser.add_argument("--model_name_or_path", default="/mnt/workspace/Qwen3-0.6B")
    parser.add_argument("--modes", default="baseline,qabs8cand3reuse,sparqfast8cand3")
    parser.add_argument("--prefill_tokens", type=int, default=4096)
    parser.add_argument("--eval_tokens", type=int, default=512)
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--eval_chunk_size", type=int, default=1)
    parser.add_argument("--max_chars", type=int, default=80_000_000)
    parser.add_argument("--require_total_tokens", type=str2bool, default=False)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--top_fraction", type=float, default=0.02)
    parser.add_argument("--protect_sink_tokens", type=int, default=10)
    parser.add_argument("--protect_recent_tokens", type=int, default=10)
    parser.add_argument("--qabs_cuda_final_kernel", type=str2bool, default=True)
    parser.add_argument("--qabs_cuda_candidate_kernel", type=str2bool, default=False)
    parser.add_argument("--qabs_cuda_reuse_select_kernel", type=str2bool, default=False)
    parser.add_argument("--max_needle_cases", type=int, default=12)
    parser.add_argument("--needle_prefill_tokens", type=int, default=0)
    parser.add_argument("--needle_eval_tokens", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--make_plots", type=str2bool, default=False)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    config = load_config(Path(args.config))
    items = build_eval_items(
        config,
        args.prefill_tokens,
        args.eval_tokens,
        args.max_needle_cases,
        args.needle_prefill_tokens,
        args.needle_eval_tokens,
    )
    if not items:
        raise RuntimeError("No evaluation items found.")
    for item in items:
        run_one(args, item)
    combined_path = combine_results(output_root)
    print(f"combined results: {combined_path}")


if __name__ == "__main__":
    main()
