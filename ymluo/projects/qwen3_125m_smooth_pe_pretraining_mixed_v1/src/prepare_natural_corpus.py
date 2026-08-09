from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from array import array
from pathlib import Path
from typing import Iterable, Iterator

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


KEY_COUNT = 1024
VALUE_COUNT = 1024
BASE_SPECIAL_TOKENS = [
    "<pad>",
    "<bos>",
    "<eos>",
    "<unk>",
    "<fact>",
    "<key>",
    "<value>",
    "<query>",
    "<answer>",
    "<sep>",
]


def special_tokens() -> list[str]:
    return (
        BASE_SPECIAL_TOKENS
        + [f"<k{index:04d}>" for index in range(KEY_COUNT)]
        + [f"<v{index:04d}>" for index in range(VALUE_COUNT)]
    )


def numeric_suffix(path: Path) -> int:
    match = re.search(r"(\d+)$", path.stem)
    return int(match.group(1)) if match else sys.maxsize


def discover_files(root: Path, pattern: str, limit: int | None = None) -> list[Path]:
    paths = sorted(root.glob(pattern), key=lambda path: (numeric_suffix(path), path.name))
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(f"No files match {root / pattern}")
    return paths


def iter_documents(paths: Iterable[Path], max_documents: int | None = None) -> Iterator[str]:
    emitted = 0
    for path in paths:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                yield text
                emitted += 1
                if max_documents is not None and emitted >= max_documents:
                    return


def source_fingerprint(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        stat = path.stat()
        digest.update(f"{path.name}\t{stat.st_size}\n".encode())
    return digest.hexdigest()


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def train_tokenizer(args: argparse.Namespace) -> None:
    files = discover_files(args.input_root, args.pattern, args.file_limit)
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        show_progress=True,
        special_tokens=special_tokens(),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    started = time.time()
    tokenizer.train_from_iterator(
        iter_documents(files, args.max_documents),
        trainer=trainer,
        length=args.max_documents,
    )
    actual_size = tokenizer.get_vocab_size()
    if actual_size != args.vocab_size:
        raise RuntimeError(f"Tokenizer size {actual_size} != requested {args.vocab_size}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(args.output))
    mapping = {token: tokenizer.token_to_id(token) for token in special_tokens()}
    if any(value is None for value in mapping.values()):
        raise RuntimeError("A reserved token is missing from the trained tokenizer")
    metadata = {
        "vocab_size": actual_size,
        "tokenizer_sha256": sha256_file(args.output),
        "source_fingerprint": source_fingerprint(files),
        "source_file_count": len(files),
        "max_documents": args.max_documents,
        "elapsed_seconds": time.time() - started,
        "special_token_ids": mapping,
    }
    args.output.with_suffix(".meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False))


def flush_buffer(handle, buffer: array) -> None:
    if sys.byteorder != "little":
        buffer.byteswap()
    buffer.tofile(handle)
    del buffer[:]


def build_binary(args: argparse.Namespace) -> None:
    files = discover_files(args.input_root, args.pattern, args.file_limit)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    eos_id = tokenizer.token_to_id("<eos>")
    if eos_id is None:
        raise RuntimeError("Tokenizer has no <eos> token")
    if tokenizer.get_vocab_size() > 65535:
        raise RuntimeError("Vocabulary does not fit uint16")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    buffer = array("H")
    token_count = 0
    document_count = 0
    batch: list[str] = []
    started = time.time()

    def consume(texts: list[str], handle) -> bool:
        nonlocal token_count, document_count
        encodings = tokenizer.encode_batch(texts, add_special_tokens=False)
        for encoding in encodings:
            ids = encoding.ids + [eos_id]
            remaining = args.target_tokens - token_count
            if remaining <= 0:
                return True
            if len(ids) > remaining:
                ids = ids[:remaining]
            buffer.extend(ids)
            token_count += len(ids)
            document_count += 1
            if len(buffer) >= args.flush_tokens:
                flush_buffer(handle, buffer)
            if token_count >= args.target_tokens:
                return True
        return False

    with temporary.open("wb") as handle:
        finished = False
        for document in iter_documents(files):
            batch.append(document)
            if len(batch) < args.encode_batch_size:
                continue
            finished = consume(batch, handle)
            batch.clear()
            if finished:
                break
        if not finished and batch:
            finished = consume(batch, handle)
        if buffer:
            flush_buffer(handle, buffer)
    if token_count != args.target_tokens:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Corpus ended at {token_count} tokens, expected {args.target_tokens}")
    temporary.replace(args.output)
    metadata = {
        "dtype": "uint16-le",
        "token_count": token_count,
        "document_count": document_count,
        "byte_count": args.output.stat().st_size,
        "sha256": sha256_file(args.output),
        "tokenizer_sha256": sha256_file(args.tokenizer),
        "source_fingerprint": source_fingerprint(files),
        "source_file_count": len(files),
        "elapsed_seconds": time.time() - started,
    }
    args.output.with_suffix(args.output.suffix + ".meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train-tokenizer")
    train.add_argument("--input-root", type=Path, required=True)
    train.add_argument("--pattern", default="openweb_every_4096_*.txt")
    train.add_argument("--file-limit", type=int, default=64)
    train.add_argument("--max-documents", type=int, default=250_000)
    train.add_argument("--vocab-size", type=int, default=32_000)
    train.add_argument("--min-frequency", type=int, default=2)
    train.add_argument("--output", type=Path, required=True)
    train.set_defaults(function=train_tokenizer)

    build = subparsers.add_parser("build-bin")
    build.add_argument("--input-root", type=Path, required=True)
    build.add_argument("--pattern", required=True)
    build.add_argument("--file-limit", type=int)
    build.add_argument("--tokenizer", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--target-tokens", type=int, required=True)
    build.add_argument("--encode-batch-size", type=int, default=256)
    build.add_argument("--flush-tokens", type=int, default=1_000_000)
    build.set_defaults(function=build_binary)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
