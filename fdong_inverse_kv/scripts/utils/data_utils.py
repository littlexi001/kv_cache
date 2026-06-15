"""Data adapters used by the server training scripts."""

from __future__ import annotations

import json
import os
import random
from typing import Any

import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info


def _extract_text(record: Any) -> str:
    if isinstance(record, str):
        return record
    if isinstance(record, dict):
        for key in ("text", "content", "document", "raw_content"):
            value = record.get(key)
            if isinstance(value, str):
                return value
    raise ValueError("Each DCLM line must be a JSON string or contain a text/content field")


class TokenizedJSONLData(IterableDataset):
    """Distributed streaming reader for the existing DCLM directory layout.

    The remote path remains unchanged. Files may use ``.txt`` or ``.jsonl``;
    each non-empty line is parsed as JSON when possible and otherwise treated
    as plain text. Files are sharded across distributed ranks and DataLoader
    workers without scanning the full corpus at startup.
    """

    def __init__(self, dataset_dir, max_seq_len, tokenizer, padding=True, seed=0):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.max_seq_len = int(max_seq_len)
        self.tokenizer = tokenizer
        self.padding = bool(padding)
        self.seed = int(seed)
        self.rank = 0
        self.world_size = 1
        self.epoch = 0
        self.files = []
        for root, _, filenames in os.walk(dataset_dir):
            for filename in filenames:
                if filename.endswith((".txt", ".jsonl")):
                    self.files.append(os.path.join(root, filename))
        self.files.sort()
        if not self.files:
            raise FileNotFoundError(f"No .txt or .jsonl files found under {dataset_dir}")

    def set_distributed(self, rank, world_size):
        self.rank = int(rank)
        self.world_size = int(world_size)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _encode(self, raw_line):
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError:
            record = raw_line
        text = _extract_text(record)

        tokenized = self.tokenizer(
            text,
            padding="max_length" if self.padding else False,
            max_length=self.max_seq_len + 1,
            truncation=True,
            return_tensors="pt",
        )
        token_ids = tokenized.input_ids[0]
        if hasattr(tokenized, "attention_mask"):
            real_len = int(tokenized.attention_mask[0].sum().item())
        else:
            real_len = len(token_ids)
        return token_ids[:-1], token_ids[1:], real_len

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1
        shard_id = self.rank * num_workers + worker_id
        num_shards = self.world_size * num_workers
        files = list(self.files)
        random.Random(self.seed + self.epoch).shuffle(files)
        files = files[shard_id::num_shards]
        if not files:
            raise RuntimeError(
                f"No input files assigned to rank={self.rank}, worker={worker_id}; "
                f"files={len(self.files)}, shards={num_shards}. Reduce num_workers."
            )
        for path in files:
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        yield self._encode(line)


class RandomTokenDataset(Dataset):
    """Local smoke-test data. It is never selected by the server shell script."""

    def __init__(self, num_samples: int, seq_len: int, vocab_size: int, seed: int = 0):
        generator = torch.Generator().manual_seed(seed)
        self.tokens = torch.randint(
            low=1,
            high=vocab_size,
            size=(num_samples, seq_len + 1),
            generator=generator,
        )

    def __len__(self):
        return self.tokens.shape[0]

    def __getitem__(self, index):
        tokens = self.tokens[index]
        return tokens[:-1], tokens[1:], len(tokens)
