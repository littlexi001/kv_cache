from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from statistics import mean
from typing import Any


MODEL_REPO = "Qwen/Qwen3-4B-Instruct-2507"
LONGBENCH_REPO = "THUDM/LongBench"
HOT_POT_REPO = "hotpotqa/hotpot_qa"
LM_EVAL_REPO = "https://github.com/EleutherAI/lm-evaluation-harness.git"
LM_EVAL_COMMIT = "8c05cfe04fafcdd41dd64019f2b3797ef54dcd81"
LONG_TASKS = (
    "narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,qmsum,"
    "trec,triviaqa,samsum,passage_retrieval_en,passage_count,gov_report,"
    "multi_news,lcc,repobench-p"
)
RULER_TASKS = (
    "niah_single_1,niah_multikey_1,niah_multivalue,niah_multiquery,vt,cwe,"
    "fwe,qa_squad,qa_hotpot"
)


def run(command: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def clone_lm_eval(path: Path) -> None:
    if not (path / ".git").is_dir():
        path.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", LM_EVAL_REPO, str(path)])
    run(["git", "fetch", "origin", LM_EVAL_COMMIT], cwd=path)
    run(["git", "checkout", "--detach", LM_EVAL_COMMIT], cwd=path)


def extract_longbench(zip_path: Path, output_root: Path) -> Path:
    target = output_root / "data"
    required = [target / f"{task}.jsonl" for task in LONG_TASKS.split(",")]
    if not all(path.is_file() for path in required):
        output_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(output_root)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"LongBench archive is incomplete: {missing}")
    return target


def prepare(args: argparse.Namespace) -> None:
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    work_root = args.work_root.resolve()
    model_dir = work_root / "models" / "Qwen3-4B-Instruct-2507"
    longbench_root = work_root / "data" / "longbench"
    lm_eval_dir = work_root / "third_party" / "lm-evaluation-harness"
    ruler_source_root = work_root / "data" / "ruler_sources" / "hotpotqa"
    ruler_dir = work_root / "data" / "ruler"
    ruler_jsonl = ruler_dir / (
        f"qwen3_4b_{args.ruler_lengths.replace(',', '_')}_m{args.ruler_samples}_"
        f"seed{args.seed}.jsonl"
    )

    work_root.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {MODEL_REPO} to {model_dir}", flush=True)
    snapshot_download(
        repo_id=MODEL_REPO,
        revision=args.model_revision,
        local_dir=model_dir,
        token=os.environ.get("HF_TOKEN"),
    )
    model_info = HfApi().model_info(
        MODEL_REPO,
        revision=args.model_revision,
        token=os.environ.get("HF_TOKEN"),
    )

    zip_path = Path(
        hf_hub_download(
            repo_id=LONGBENCH_REPO,
            repo_type="dataset",
            filename="data.zip",
            revision=args.longbench_revision,
            local_dir=longbench_root,
            token=os.environ.get("HF_TOKEN"),
        )
    )
    longbench_data = extract_longbench(zip_path, longbench_root)

    hotpot_path = Path(
        hf_hub_download(
            repo_id=HOT_POT_REPO,
            repo_type="dataset",
            filename="distractor/validation-00000-of-00001.parquet",
            revision=args.hotpot_revision,
            local_dir=ruler_source_root,
            token=os.environ.get("HF_TOKEN"),
        )
    )
    clone_lm_eval(lm_eval_dir)
    if args.install_lm_eval:
        run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-e",
                f"{lm_eval_dir}[ruler]",
            ]
        )

    prepare_script = (
        args.project_root.resolve()
        / "src"
        / "prepare_hierarchical_ruler_data_20260716.py"
    )
    run(
        [
            sys.executable,
            str(prepare_script),
            "--model_name_or_path",
            str(model_dir),
            "--lm_eval_path",
            str(lm_eval_dir),
            "--output",
            str(ruler_jsonl),
            "--ruler_tasks",
            RULER_TASKS,
            "--ruler_lengths",
            args.ruler_lengths,
            "--max_samples_per_task",
            str(args.ruler_samples),
            "--seed",
            str(args.seed),
            "--ruler_hotpot_parquet",
            str(hotpot_path),
        ]
    )

    manifest = {
        "model_repo": MODEL_REPO,
        "model_revision_requested": args.model_revision,
        "model_commit": model_info.sha,
        "model_dir": str(model_dir),
        "longbench_repo": LONGBENCH_REPO,
        "longbench_revision_requested": args.longbench_revision,
        "longbench_data_dir": str(longbench_data),
        "lm_eval_repo": LM_EVAL_REPO,
        "lm_eval_commit": LM_EVAL_COMMIT,
        "lm_eval_dir": str(lm_eval_dir),
        "ruler_tasks": RULER_TASKS.split(","),
        "ruler_lengths": [int(value) for value in args.ruler_lengths.split(",")],
        "ruler_samples_per_task_length": args.ruler_samples,
        "ruler_jsonl": str(ruler_jsonl),
        "hotpot_parquet": str(hotpot_path),
        "seed": args.seed,
    }
    (work_root / "assets.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


def doctor(args: argparse.Namespace) -> None:
    import accelerate
    import huggingface_hub
    import torch
    import transformers
    from torch.utils.cpp_extension import CUDA_HOME

    failures: list[str] = []
    if not torch.cuda.is_available():
        failures.append("PyTorch cannot access CUDA")
    if torch.cuda.device_count() < args.expected_gpus:
        failures.append(
            f"expected at least {args.expected_gpus} GPUs, found {torch.cuda.device_count()}"
        )
    nvcc_path = shutil.which("nvcc")
    if nvcc_path is None and CUDA_HOME is not None:
        cuda_home_nvcc = Path(CUDA_HOME) / "bin" / "nvcc"
        if cuda_home_nvcc.is_file():
            nvcc_path = str(cuda_home_nvcc)
    if CUDA_HOME is None or nvcc_path is None:
        failures.append("CUDA toolkit/nvcc is required to compile CountCap kernels")
    if shutil.which("g++") is None:
        failures.append("g++ is required to compile CountCap kernels")
    required = [
        "run_head_top2_targeted_ppl_20260714.py",
        "qabs_cuda_kernels.py",
        "run_sample_calibrated_longbench_20260717.py",
        "run_sample_calibrated_ruler_20260717.py",
        "prepare_hierarchical_ruler_data_20260716.py",
    ]
    missing = [
        name for name in required if not (args.project_root.resolve() / "src" / name).is_file()
    ]
    if missing:
        failures.append(f"missing project source files: {missing}")

    payload = {
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "accelerate": accelerate.__version__,
        "huggingface_hub": huggingface_hub.__version__,
        "cuda_home": CUDA_HOME,
        "nvcc": nvcc_path,
        "gpu_count": torch.cuda.device_count(),
        "gpus": [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "total_memory_gib": round(
                    torch.cuda.get_device_properties(index).total_memory / 2**30, 3
                ),
            }
            for index in range(torch.cuda.device_count())
        ],
        "failures": failures,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    if failures:
        raise SystemExit(2)


def read_rows(pattern: str) -> tuple[list[Path], list[dict[str, str]]]:
    paths = sorted(Path(path) for path in glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(pattern)
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"no rows found in {pattern}")
    return paths, rows


def validate_pairs(rows: list[dict[str, str]], benchmark: str) -> dict[str, Any]:
    methods = {row["method"] for row in rows}
    if methods != {"full_kv", "countcap"}:
        raise RuntimeError(f"{benchmark}: unexpected methods {sorted(methods)}")
    sample_sets = {
        method: {
            (row["task"], row["sample_id"])
            for row in rows
            if row["method"] == method
        }
        for method in methods
    }
    if sample_sets["full_kv"] != sample_sets["countcap"]:
        raise RuntimeError(f"{benchmark}: Full KV and CountCap sample sets differ")
    return {
        "benchmark": benchmark,
        "paired_samples": len(sample_sets["full_kv"]),
        "rows": len(rows),
    }


def find_summary(
    summary: list[dict[str, Any]], task: str, method: str
) -> dict[str, Any]:
    return next(
        row
        for row in summary
        if str(row["task"]) == task and str(row["method"]) == method
    )


def ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator != 0.0 else None


def format_ratio(value: float | None) -> str:
    return "N/A" if value is None else f"{100.0 * value:.2f}%"


def finalize(args: argparse.Namespace) -> None:
    project_src = args.project_root.resolve() / "src"
    if str(project_src) not in sys.path:
        sys.path.insert(0, str(project_src))
    import run_sample_calibrated_longbench_20260717 as longbench
    import run_sample_calibrated_ruler_20260717 as ruler

    long_paths, long_rows = read_rows(args.longbench_glob)
    ruler_paths, ruler_rows = read_rows(args.ruler_glob)
    audits = [
        validate_pairs(long_rows, "LongBench"),
        validate_pairs(ruler_rows, "RULER"),
    ]
    long_summary = longbench.summarize(long_rows)
    ruler_summary = ruler.summarize(ruler_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    longbench.write_csv(args.output_dir / "longbench_sample_results.csv", long_rows)
    longbench.write_csv(args.output_dir / "longbench_summary.csv", long_summary)
    longbench.write_csv(args.output_dir / "ruler_sample_results.csv", ruler_rows)
    longbench.write_csv(args.output_dir / "ruler_summary.csv", ruler_summary)

    long_full = find_summary(long_summary, "ALL", "full_kv")
    long_ours = find_summary(long_summary, "ALL", "countcap")
    ruler_full = find_summary(ruler_summary, "ALL", "full_kv")
    ruler_ours = find_summary(ruler_summary, "ALL", "countcap")
    payload = {
        "audit": audits,
        "input_shards": {
            "longbench": [str(path) for path in long_paths],
            "ruler": [str(path) for path in ruler_paths],
        },
        "longbench": long_summary,
        "ruler": ruler_summary,
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    ruler_lengths = sorted(
        {
            int(row["requested_length"])
            for row in ruler_rows
            if row.get("requested_length")
        }
    )
    lines = [
        "# CountCap LongBench / RULER 实验结果",
        "",
        "## 方法与协议",
        "",
        "- 模型：Qwen3-4B-Instruct-2507。",
        "- CountCap：QK-Metric48 + logscale16 INT4 K 索引 + 256 点分位候选估计 + 原始 K 精确重排。",
        "- 最终 attention token 数：`min(round(0.02 * N), 1280)`。",
        "- 候选比例：`clip(4 * attention_fraction, 0.03, 0.06)`。",
        "- Full KV 与 CountCap 使用相同 prompt、样本、greedy decoding 和最大生成长度。",
        "",
        "## 总结果",
        "",
        "| Benchmark | 方法 | Macro score | 质量保持率 | Attention ratio | Candidate ratio | Online speedup |",
        "|---|---|---:|---:|---:|---:|---:|",
        (
            f"| LongBench | Full KV | {float(long_full['score']):.6f} | 100.00% | "
            f"100.00% | 100.00% | 1.000x |"
        ),
        (
            f"| LongBench | CountCap | {float(long_ours['score']):.6f} | "
            f"{format_ratio(ratio(float(long_ours['score']), float(long_full['score'])))} | "
            f"{100.0 * float(long_ours['mean_configured_attention_fraction']):.2f}% | "
            f"{100.0 * float(long_ours['mean_configured_candidate_fraction']):.2f}% | "
            f"{float(long_ours['paired_online_speedup']):.3f}x |"
        ),
        (
            f"| RULER | Full KV | {float(ruler_full['score']):.6f} | 100.00% | "
            f"100.00% | 100.00% | 1.000x |"
        ),
        (
            f"| RULER | CountCap | {float(ruler_ours['score']):.6f} | "
            f"{format_ratio(ratio(float(ruler_ours['score']), float(ruler_full['score'])))} | "
            f"{100.0 * float(ruler_ours['mean_configured_attention_fraction']):.2f}% | "
            f"{100.0 * float(ruler_ours['mean_configured_candidate_fraction']):.2f}% | "
            f"{float(ruler_ours['paired_online_speedup']):.3f}x |"
        ),
        "",
        "## RULER 分长度",
        "",
        "| 长度 | Full score | CountCap score | 质量保持率 | Online speedup |",
        "|---:|---:|---:|---:|---:|",
    ]
    for length in ruler_lengths:
        full = find_summary(ruler_summary, f"LENGTH_{length}", "full_kv")
        ours = find_summary(ruler_summary, f"LENGTH_{length}", "countcap")
        full_seconds = mean(
            float(row["online_seconds"])
            for row in ruler_rows
            if row["method"] == "full_kv"
            and int(row["requested_length"]) == length
        )
        ours_seconds = mean(
            float(row["online_seconds"])
            for row in ruler_rows
            if row["method"] == "countcap"
            and int(row["requested_length"]) == length
        )
        lines.append(
            f"| {length} | {float(full['score']):.6f} | {float(ours['score']):.6f} | "
            f"{format_ratio(ratio(float(ours['score']), float(full['score'])))} | "
            f"{full_seconds / ours_seconds:.3f}x |"
        )
    lines.extend(
        [
            "",
            "## 审计",
            "",
            *[
                f"- {row['benchmark']}：{row['paired_samples']} 个严格配对样本，"
                f"共 {row['rows']} 条方法结果。"
                for row in audits
            ],
            "",
            "## 结果解释边界",
            "",
            "- `Attention ratio` 是每个 query head 最终参与精确 attention 的历史 token 比例。",
            "- `Candidate ratio` 是低维 INT4 索引送入原始 K 精确重排的候选比例。",
            "- 本质量 runner 保留完整 DynamicCache，因此不能把这里的比例表述为物理 GPU KV 存储率。",
            "- 并行 shard 的 online timing 适合 Full/CountCap 配对比较；论文最终单流延迟仍应在独占 GPU 上复测。",
            "",
        ]
    )
    (args.output_dir / "RESULTS.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print("\n".join(lines), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Portable CountCap benchmark utility")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--project_root", required=True, type=Path)
    prepare_parser.add_argument("--work_root", required=True, type=Path)
    prepare_parser.add_argument("--ruler_lengths", default="65536,131072")
    prepare_parser.add_argument("--ruler_samples", type=int, default=5)
    prepare_parser.add_argument("--seed", type=int, default=42)
    prepare_parser.add_argument("--model_revision", default="main")
    prepare_parser.add_argument("--longbench_revision", default="main")
    prepare_parser.add_argument("--hotpot_revision", default="main")
    prepare_parser.add_argument("--install_lm_eval", action="store_true")
    prepare_parser.set_defaults(func=prepare)

    doctor_parser = subparsers.add_parser("doctor")
    doctor_parser.add_argument("--project_root", required=True, type=Path)
    doctor_parser.add_argument("--expected_gpus", type=int, default=8)
    doctor_parser.add_argument("--output", type=Path)
    doctor_parser.set_defaults(func=doctor)

    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--project_root", required=True, type=Path)
    finalize_parser.add_argument("--longbench_glob", required=True)
    finalize_parser.add_argument("--ruler_glob", required=True)
    finalize_parser.add_argument("--output_dir", required=True, type=Path)
    finalize_parser.set_defaults(func=finalize)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
