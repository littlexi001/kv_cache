#!/usr/bin/env python
"""Capture immutable host, software, model, and run provenance for QKSieve."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy
import torch
import transformers


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(32 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_files(model_root: Path) -> list[Path]:
    index = model_root / "model.safetensors.index.json"
    if not index.is_file():
        raise FileNotFoundError(f"model index is missing: {index}")
    payload = json.loads(index.read_text(encoding="utf-8"))
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise AssertionError("model index has no weight map")
    names = {
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        *{str(name) for name in weight_map.values()},
    }
    paths = [model_root / name for name in sorted(names)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"model files are missing: {missing}")
    return paths


def gpu_inventory() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    text = subprocess.check_output(command, text=True)
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        index, name, uuid, memory_mib, driver = [
            item.strip() for item in line.split(",", 4)
        ]
        rows.append(
            {
                "index": int(index),
                "name": name,
                "uuid": uuid,
                "memory_mib": int(memory_mib),
                "driver": driver,
            }
        )
    return rows


def capture(
    *,
    role: str,
    model_root: Path,
    run_manifest: Path,
    audited_implementation_sha: str,
    numerical_freeze_sha: str,
    visible_cuda_devices: str,
) -> dict[str, Any]:
    files = model_files(model_root)
    download_manifest = model_root / "qksieve_download_manifest.json"
    return {
        "schema": "qksieve_runtime_provenance_v1",
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "role": role,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "audited_implementation_commit_sha": audited_implementation_sha,
        "numerical_freeze_commit_sha": numerical_freeze_sha,
        "visible_cuda_devices": visible_cuda_devices,
        "software": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": numpy.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
        },
        "gpus": gpu_inventory(),
        "run_manifest": {
            "path": str(run_manifest.resolve()),
            "sha256": sha256(run_manifest),
        },
        "model": {
            "path": str(model_root.resolve()),
            "download_manifest": (
                {
                    "sha256": sha256(download_manifest),
                    "payload": json.loads(
                        download_manifest.read_text(encoding="utf-8")
                    ),
                }
                if download_manifest.is_file()
                else None
            ),
            "files": {
                path.name: {
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
                for path in files
            },
        },
        "environment": {
            "python_executable": os.path.realpath(os.sys.executable),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", required=True)
    parser.add_argument("--model_root", required=True, type=Path)
    parser.add_argument("--run_manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audited_implementation_sha", required=True)
    parser.add_argument("--numerical_freeze_sha", required=True)
    parser.add_argument("--visible_cuda_devices", default="0,1,2,3,4,5,6,7")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = capture(
        role=args.role,
        model_root=args.model_root,
        run_manifest=args.run_manifest,
        audited_implementation_sha=args.audited_implementation_sha,
        numerical_freeze_sha=args.numerical_freeze_sha,
        visible_cuda_devices=args.visible_cuda_devices,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
