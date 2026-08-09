from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from data_pipeline import ensure_manifests
from io_utils import append_jsonl, parse_int_list, read_json, utc_timestamp, write_json
from pe_strategies import load_strategy


HERE = Path(__file__).resolve().parent


def event(event_path: Path, stage: str, status: str, **values: Any) -> None:
    append_jsonl(
        event_path,
        {"timestamp": utc_timestamp(), "stage": stage, "status": status, **values},
    )


def run(
    command: list[str],
    events: Path,
    stage: str,
    environment: dict[str, str] | None = None,
    required: bool = True,
) -> bool:
    event(events, stage, "started", command=command)
    started = time.time()
    completed = subprocess.run(command, env=environment, check=False)
    event(
        events,
        stage,
        "complete" if completed.returncode == 0 else "failed",
        returncode=completed.returncode,
        elapsed_seconds=time.time() - started,
    )
    if completed.returncode != 0:
        if required:
            raise RuntimeError(f"stage {stage} failed with exit code {completed.returncode}")
        return False
    return True


@contextlib.contextmanager
def manifest_lock(path: Path):
    """Serialize manifest creation when conditions share one Linux workspace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            yield
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except ImportError:
            yield


def latest_checkpoint(checkpoint_root: Path, maximum_step: int) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    for path in checkpoint_root.glob("checkpoint-*"):
        try:
            step = int(path.name.split("-")[-1])
        except ValueError:
            continue
        if step <= maximum_step and (path / "trainer_state.json").is_file():
            candidates.append((step, path))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def evaluation_complete(path: Path, require_longbench: bool = False) -> bool:
    status = path / "status.json"
    if not (status.exists() and read_json(status).get("complete", False)):
        return False
    if require_longbench:
        summary = path / "summary.json"
        return bool(summary.exists() and read_json(summary).get("longbench_status") == "complete")
    return True


def log_evaluation_to_tensorboard(run_dir: Path, evaluation_dir: Path, step: int) -> None:
    summary_path = evaluation_dir / "summary.json"
    if not summary_path.is_file():
        return
    from torch.utils.tensorboard import SummaryWriter

    summary = read_json(summary_path)
    writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))
    ppl = summary.get("validation_ppl", {}).get("ppl")
    if ppl is not None:
        writer.add_scalar("eval/dclm_ppl", float(ppl), step)
    for prefix, rows in [
        ("controlled", summary.get("controlled", [])),
        ("longbench", summary.get("longbench", [])),
    ]:
        if rows:
            for key in ["qa_f1_percent", "exact_match_percent", "gold_answer_mean_nll"]:
                values = [float(row[key]) for row in rows if row.get(key) is not None]
                if values:
                    writer.add_scalar(f"eval/{prefix}_{key}", sum(values) / len(values), step)
    writer.flush()
    writer.close()


def eval_command(
    args: argparse.Namespace,
    model_path: Path,
    strategy_path: Path,
    output_dir: Path,
    label: str,
    step: int,
    validation_manifest: Path,
) -> list[str]:
    return [
        sys.executable,
        str(HERE / "evaluate_checkpoint.py"),
        "--model-path", str(model_path),
        "--tokenizer-path", str(args.model_root),
        "--strategy", str(strategy_path),
        "--validation-manifest", str(validation_manifest),
        "--output-dir", str(output_dir),
        "--label", label,
        "--step", str(step),
        "--eval-lengths", args.eval_lengths,
        "--ruler-samples-per-task", str(args.ruler_samples_per_task),
        "--ppl-blocks", str(args.ppl_blocks),
        "--run-longbench", str(args.run_longbench),
        "--longbench-tasks", args.longbench_tasks,
        "--longbench-samples-per-task", str(args.longbench_samples_per_task),
        "--max-new-tokens", str(args.max_new_tokens),
        "--dtype", args.dtype,
        "--attention-implementation", args.attention_implementation,
        "--seed", str(args.seed),
        "--initialization", (
            args.initialization if step == 0 else "checkpoint"
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--dclm-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--strategy", type=Path, required=True)
    parser.add_argument("--native-strategy", type=Path, required=True)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--sequence-length", type=int, default=8192)
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=32)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--target-tokens", type=int, default=100_000_000_000)
    parser.add_argument(
        "--milestone-tokens",
        default="100000000,1000000000,10000000000,25000000000,50000000000,75000000000,100000000000",
    )
    parser.add_argument("--total-steps", type=int, default=47_684)
    parser.add_argument("--milestones", default="48,477,4769,11921,23842,35763,47684")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--data-seed", type=int, default=1701)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--train-files", type=int, default=200_000)
    parser.add_argument("--validation-files", type=int, default=1_024)
    parser.add_argument("--eval-lengths", default="2048,4096,8192")
    parser.add_argument("--ruler-samples-per-task", type=int, default=4)
    parser.add_argument("--ppl-blocks", type=int, default=16)
    parser.add_argument("--run-longbench", type=int, choices=[0, 1], default=1)
    parser.add_argument("--longbench-tasks", default="hotpotqa,2wikimqa,multifieldqa_en")
    parser.add_argument("--longbench-samples-per-task", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attention-implementation", default="sdpa")
    parser.add_argument(
        "--initialization", choices=["checkpoint", "from_scratch"], default="from_scratch"
    )
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--tensorboard", type=int, choices=[0, 1], default=1)
    args = parser.parse_args()

    strategy = load_strategy(args.strategy)
    native = load_strategy(args.native_strategy)
    if native.kind != "native":
        raise ValueError("--native-strategy must have kind=native")
    run_dir = args.run_root / ("base_eval" if args.eval_only else strategy.name)
    run_dir.mkdir(parents=True, exist_ok=True)
    events = run_dir / "controller_events.jsonl"
    shutil.copy2(args.strategy, run_dir / "strategy.json")
    if not (run_dir / "environment.json").exists():
        run(
            [sys.executable, str(HERE / "capture_environment.py"), "--output", str(run_dir / "environment.json")],
            events,
            "capture_environment",
        )
    shared_manifest = args.run_root / "_shared" / f"manifests_seed_{args.data_seed}"
    event(events, "prepare_manifest", "started", path=str(shared_manifest))
    with manifest_lock(shared_manifest / ".manifest.lock"):
        metadata = ensure_manifests(
            args.dclm_root,
            shared_manifest,
            args.train_files,
            args.validation_files,
            args.data_seed,
        )
    write_json(run_dir / "manifest_metadata.json", metadata)
    event(events, "prepare_manifest", "complete", **metadata)
    validation_manifest = shared_manifest / "validation_manifest.txt"
    train_manifest = shared_manifest / "train_manifest.txt"

    world_size = 0
    if not args.eval_only:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        world_size = len([item for item in visible.split(",") if item.strip()])
        if world_size < 1:
            raise ValueError("CUDA_VISIBLE_DEVICES exposes no GPUs")
        actual_global_batch = args.micro_batch * args.gradient_accumulation * world_size
        if args.global_batch_size and actual_global_batch != args.global_batch_size:
            raise ValueError(
                "global batch mismatch: "
                f"{args.micro_batch} * {args.gradient_accumulation} * {world_size} "
                f"= {actual_global_batch}, expected {args.global_batch_size}"
            )

    base_eval = run_dir / "evaluations" / "base_step0"
    if not evaluation_complete(base_eval, require_longbench=bool(args.run_longbench)):
        run(
            eval_command(
                args,
                args.model_root,
                args.native_strategy,
                base_eval,
                "base",
                0,
                validation_manifest,
            ),
            events,
            "evaluate_base",
            required=False,
        )
    else:
        event(events, "evaluate_base", "skipped_complete")
    if args.tensorboard:
        log_evaluation_to_tensorboard(run_dir, base_eval, 0)

    if not args.eval_only:
        actual_global_batch = args.micro_batch * args.gradient_accumulation * world_size
        global_batch = args.global_batch_size or actual_global_batch
        tokens_per_step = args.sequence_length * global_batch
        if args.target_tokens:
            total_steps = math.ceil(args.target_tokens / tokens_per_step)
            token_targets = sorted(set(parse_int_list(args.milestone_tokens)))
            if not token_targets:
                raise ValueError("--milestone-tokens is required when --target-tokens is set")
            if token_targets[-1] > args.target_tokens:
                raise ValueError("a milestone token target exceeds target-tokens")
            if token_targets[-1] != args.target_tokens:
                token_targets.append(args.target_tokens)
            milestones = sorted({math.ceil(value / tokens_per_step) for value in token_targets})
        else:
            total_steps = args.total_steps
            token_targets = []
            milestones = sorted(set(parse_int_list(args.milestones)))
            if milestones[-1] > total_steps:
                raise ValueError("a milestone exceeds total-steps")
            if milestones[-1] != total_steps:
                milestones.append(total_steps)
        write_json(
            run_dir / "token_schedule.json",
            {
                "initialization": args.initialization,
                "target_tokens": args.target_tokens or total_steps * tokens_per_step,
                "tokens_per_step": tokens_per_step,
                "global_batch_size_sequences": global_batch,
                "total_steps": total_steps,
                "milestone_token_targets": token_targets,
                "milestone_steps": milestones,
                "actual_final_tokens": total_steps * tokens_per_step,
            },
        )
        for target in milestones:
            checkpoint = run_dir / "checkpoints" / f"checkpoint-{target}"
            if not (checkpoint / "trainer_state.json").is_file():
                resume = latest_checkpoint(run_dir / "checkpoints", target - 1)
                command = [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    "--nproc_per_node",
                    str(world_size),
                    str(HERE / "train_segment.py"),
                    "--model-root", str(args.model_root),
                    "--strategy", str(args.strategy),
                    "--train-manifest", str(train_manifest),
                    "--output-dir", str(run_dir),
                    "--sequence-length", str(args.sequence_length),
                    "--micro-batch", str(args.micro_batch),
                    "--gradient-accumulation", str(args.gradient_accumulation),
                    "--total-steps", str(total_steps),
                    "--stop-after-step", str(target),
                    "--learning-rate", str(args.learning_rate),
                    "--warmup-steps", str(args.warmup_steps),
                    "--weight-decay", str(args.weight_decay),
                    "--seed", str(args.seed),
                    "--data-seed", str(args.data_seed),
                    "--num-workers", str(args.num_workers),
                    "--dtype", args.dtype,
                    "--attention-implementation", args.attention_implementation,
                    "--initialization", args.initialization,
                    "--global-batch-size", str(global_batch),
                    "--target-tokens", str(args.target_tokens),
                    "--adam-beta1", str(args.adam_beta1),
                    "--adam-beta2", str(args.adam_beta2),
                    "--adam-epsilon", str(args.adam_epsilon),
                    "--logging-steps", str(args.logging_steps),
                    "--tensorboard", str(args.tensorboard),
                ]
                if resume:
                    command.extend(["--resume-from", str(resume)])
                run(command, events, f"train_to_step_{target}")
            else:
                event(events, f"train_to_step_{target}", "skipped_complete")
            evaluation = run_dir / "evaluations" / f"step_{target:06d}"
            if not evaluation_complete(evaluation, require_longbench=bool(args.run_longbench)):
                run(
                    eval_command(
                        args,
                        checkpoint,
                        args.strategy,
                        evaluation,
                        "trained",
                        target,
                        validation_manifest,
                    ),
                    events,
                    f"evaluate_step_{target}",
                    required=False,
                )
            else:
                event(events, f"evaluate_step_{target}", "skipped_complete")
            if args.tensorboard:
                log_evaluation_to_tensorboard(run_dir, evaluation, target)

    run(
        [
            sys.executable,
            str(HERE / "collect_results.py"),
            "--run-dir", str(run_dir),
            "--strategy-name", ("base_eval" if args.eval_only else strategy.name),
        ],
        events,
        "collect_results",
    )
    run(
        [
            sys.executable,
            str(HERE / "package_results.py"),
            "--run-dir", str(run_dir),
            "--strategy-name", ("base_eval" if args.eval_only else strategy.name),
        ],
        events,
        "package_results",
    )
    (run_dir / "controller.done").write_text("ok\n", encoding="utf-8")
    event(events, "controller", "complete")


if __name__ == "__main__":
    main()
