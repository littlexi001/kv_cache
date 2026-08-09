from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def parse_csv(spec: str) -> list[str]:
    values = list(dict.fromkeys(item.strip() for item in spec.split(",")))
    values = [value for value in values if value]
    if not values:
        raise ValueError("matrix axis must not be empty")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the fused QKSieve Query correctness/latency validator over "
            "the production FP16/BF16 and GQA-group matrix."
        )
    )
    parser.add_argument("--lengths", default="4096,32768")
    parser.add_argument("--group_counts", default="4,8")
    parser.add_argument("--dtypes", default="float16,bfloat16")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--timing_repeats", type=int, default=5)
    parser.add_argument("--min_code_match", type=float, default=0.90)
    parser.add_argument("--max_scale_relative_p99", type=float, default=0.01)
    parser.add_argument("--max_score_nrmse", type=float, default=0.01)
    parser.add_argument("--min_topk_recall", type=float, default=0.995)
    parser.add_argument("--min_output_cosine", type=float, default=0.999)
    parser.add_argument("--max_output_rmse", type=float, default=0.01)
    parser.add_argument(
        "--min_query_prepare_speedup",
        type=float,
        default=1.05,
    )
    parser.add_argument(
        "--min_selection_speedup",
        type=float,
        default=1.00,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    group_counts = [int(value) for value in parse_csv(args.group_counts)]
    if any(not 1 <= value <= 16 for value in group_counts):
        raise ValueError("group_counts must be in [1, 16]")
    dtypes = parse_csv(args.dtypes)
    source_root = Path(__file__).resolve().parent
    validator = source_root / "validate_qksieve_qfused_cuda_20260728.py"
    output_root = args.output.parent
    output_root.mkdir(parents=True, exist_ok=True)

    threshold_args = [
        "--lengths",
        args.lengths,
        "--trials",
        str(args.trials),
        "--warmup",
        str(args.warmup),
        "--iterations",
        str(args.iterations),
        "--timing_repeats",
        str(args.timing_repeats),
        "--min_code_match",
        str(args.min_code_match),
        "--max_scale_relative_p99",
        str(args.max_scale_relative_p99),
        "--max_score_nrmse",
        str(args.max_score_nrmse),
        "--min_topk_recall",
        str(args.min_topk_recall),
        "--min_output_cosine",
        str(args.min_output_cosine),
        "--max_output_rmse",
        str(args.max_output_rmse),
        "--min_query_prepare_speedup",
        str(args.min_query_prepare_speedup),
        "--min_selection_speedup",
        str(args.min_selection_speedup),
    ]
    configurations: list[dict[str, object]] = []
    for dtype in dtypes:
        for group_count in group_counts:
            name = f"{dtype}_g{group_count}"
            child_root = output_root / name
            child_root.mkdir(parents=True, exist_ok=True)
            child_output = child_root / "correctness_and_latency.json"
            child_log = child_root / "run.log"
            command = [
                sys.executable,
                str(validator),
                *threshold_args,
                "--dtype",
                dtype,
                "--group_count",
                str(group_count),
                "--output",
                str(child_output),
            ]
            completed = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
            child_log.write_text(completed.stdout, encoding="utf-8")
            child_report = None
            parse_error = None
            if child_output.exists():
                try:
                    child_report = json.loads(
                        child_output.read_text(encoding="utf-8")
                    )
                except (OSError, json.JSONDecodeError) as error:
                    parse_error = repr(error)
            passed = bool(
                completed.returncode == 0
                and isinstance(child_report, dict)
                and child_report.get("all_passed") is True
            )
            configurations.append(
                {
                    "name": name,
                    "dtype": dtype,
                    "group_count": group_count,
                    "returncode": completed.returncode,
                    "passed": passed,
                    "result_path": str(child_output.resolve()),
                    "log_path": str(child_log.resolve()),
                    "parse_error": parse_error,
                    "report": child_report,
                }
            )

    report = {
        "schema": "qksieve_qfused_validation_matrix_v1",
        "visible_cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "lengths": [int(value) for value in parse_csv(args.lengths)],
        "group_counts": group_counts,
        "dtypes": dtypes,
        "configurations": configurations,
        "all_passed": all(
            bool(configuration["passed"])
            for configuration in configurations
        ),
        "source_sha256": {
            path.name: sha256(path)
            for path in (
                Path(__file__),
                validator,
                source_root / "qksieve_query_cuda_20260728.py",
                source_root / "variablebit_spectral_cuda_20260727.py",
                source_root / "qabs_cuda_kernels.py",
            )
        },
    }
    args.output.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)
    if not report["all_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
