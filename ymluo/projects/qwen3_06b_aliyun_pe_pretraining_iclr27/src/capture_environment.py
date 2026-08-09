from __future__ import annotations

import argparse
import os
import platform
import sys
from pathlib import Path

from io_utils import run_capture, utc_timestamp, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = {
        "captured_at": utc_timestamp(),
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "cwd": os.getcwd(),
        "environment": {
            key: os.environ.get(key, "")
            for key in [
                "CUDA_VISIBLE_DEVICES",
                "HF_ENDPOINT",
                "NCCL_DEBUG",
                "TOKENIZERS_PARALLELISM",
            ]
        },
        "nvidia_smi": run_capture(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,driver_version",
                "--format=csv,noheader",
            ]
        ),
        "pip_freeze": run_capture([sys.executable, "-m", "pip", "freeze"], timeout=120),
    }
    write_json(args.output, payload)


if __name__ == "__main__":
    main()

