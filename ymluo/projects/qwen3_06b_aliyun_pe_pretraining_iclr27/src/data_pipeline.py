from __future__ import annotations

import hashlib
import heapq
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.utils.data import IterableDataset, get_worker_info

from io_utils import read_json, write_json


def _score(seed: int, path: Path) -> int:
    payload = f"{seed}\0{path.as_posix()}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def iter_text_files(root: Path) -> Iterator[Path]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    for current, directories, files in os.walk(root):
        directories.sort()
        for filename in sorted(files):
            if filename.lower().endswith(".txt"):
                yield Path(current) / filename


def build_manifests(
    root: Path,
    output_dir: Path,
    train_files: int,
    validation_files: int,
    seed: int,
) -> dict[str, Any]:
    required = train_files + validation_files
    if required <= 0:
        raise ValueError("manifest must contain at least one file")
    output_dir.mkdir(parents=True, exist_ok=True)
    heap: list[tuple[int, str]] = []
    scanned = 0
    for path in iter_text_files(root):
        scanned += 1
        score = _score(seed, path)
        item = (-score, str(path))
        if len(heap) < required:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)
    if len(heap) < required:
        raise RuntimeError(f"found {len(heap)} text files, need {required}")
    selected = sorted([(-neg_score, path) for neg_score, path in heap])
    paths = [path for _, path in selected]
    validation = paths[:validation_files]
    train = paths[validation_files:]
    train_path = output_dir / "train_manifest.txt"
    validation_path = output_dir / "validation_manifest.txt"
    train_path.write_text("\n".join(train) + "\n", encoding="utf-8")
    validation_path.write_text("\n".join(validation) + "\n", encoding="utf-8")
    manifest_hash = hashlib.sha256("\n".join(paths).encode("utf-8")).hexdigest()
    metadata = {
        "root": str(root),
        "seed": seed,
        "files_scanned": scanned,
        "train_files": len(train),
        "validation_files": len(validation),
        "manifest_sha256": manifest_hash,
        "selection": "lowest deterministic SHA256-derived scores over all recursively discovered .txt files",
    }
    write_json(output_dir / "manifest_metadata.json", metadata)
    return metadata


def ensure_manifests(
    root: Path,
    output_dir: Path,
    train_files: int,
    validation_files: int,
    seed: int,
) -> dict[str, Any]:
    metadata_path = output_dir / "manifest_metadata.json"
    train_path = output_dir / "train_manifest.txt"
    validation_path = output_dir / "validation_manifest.txt"
    if metadata_path.exists() and train_path.exists() and validation_path.exists():
        metadata = read_json(metadata_path)
        expected = (str(root), seed, train_files, validation_files)
        actual = (
            str(metadata.get("root")),
            int(metadata.get("seed", -1)),
            int(metadata.get("train_files", -1)),
            int(metadata.get("validation_files", -1)),
        )
        if actual != expected:
            raise RuntimeError(
                f"existing manifest contract {actual} does not match requested {expected}; "
                "use a different MANIFEST_ROOT or remove the old manifest explicitly"
            )
        return metadata
    return build_manifests(root, output_dir, train_files, validation_files, seed)


def read_manifest(path: Path) -> list[Path]:
    paths = [Path(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not paths:
        raise RuntimeError(f"empty manifest: {path}")
    missing = [str(path) for path in paths[:32] if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"manifest paths are missing, examples={missing[:3]}")
    return paths


class PackedTextDataset(IterableDataset):
    def __init__(
        self,
        manifest_path: Path,
        tokenizer: Any,
        sequence_length: int,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
        infinite: bool = True,
    ) -> None:
        super().__init__()
        self.paths = read_manifest(manifest_path)
        self.tokenizer = tokenizer
        self.sequence_length = int(sequence_length)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.infinite = bool(infinite)
        if self.sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")

    def _assigned_paths(self, epoch: int, worker_id: int, worker_count: int) -> list[Path]:
        paths = list(self.paths)
        random.Random(self.seed + epoch).shuffle(paths)
        global_worker = self.rank * worker_count + worker_id
        global_workers = self.world_size * worker_count
        assigned = paths[global_worker::global_workers]
        if not assigned:
            raise RuntimeError(
                f"no DCLM files assigned to rank={self.rank} worker={worker_id}; "
                "increase manifest size or reduce workers"
            )
        return assigned

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        worker_count = 1 if worker is None else worker.num_workers
        eos = self.tokenizer.eos_token_id
        if eos is None:
            raise RuntimeError("tokenizer has no eos_token_id")
        epoch = 0
        buffer: list[int] = []
        while True:
            for path in self._assigned_paths(epoch, worker_id, worker_count):
                try:
                    with path.open("r", encoding="utf-8", errors="replace") as handle:
                        for line in handle:
                            text = line.strip()
                            if not text:
                                continue
                            buffer.extend(self.tokenizer.encode(text, add_special_tokens=False))
                            buffer.append(eos)
                            while len(buffer) >= self.sequence_length:
                                block = torch.tensor(buffer[: self.sequence_length], dtype=torch.long)
                                del buffer[: self.sequence_length]
                                yield {"input_ids": block, "labels": block.clone()}
                except OSError:
                    continue
            if not self.infinite:
                break
            epoch += 1


def take_validation_blocks(
    manifest_path: Path,
    tokenizer: Any,
    sequence_length: int,
    blocks: int,
    seed: int,
) -> list[torch.Tensor]:
    dataset = PackedTextDataset(
        manifest_path=manifest_path,
        tokenizer=tokenizer,
        sequence_length=sequence_length,
        seed=seed,
        infinite=True,
    )
    iterator = iter(dataset)
    return [next(iterator)["input_ids"] for _ in range(blocks)]

