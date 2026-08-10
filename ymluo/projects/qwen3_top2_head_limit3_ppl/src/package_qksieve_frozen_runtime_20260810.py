#!/usr/bin/env python
"""Create a byte-exact, deterministic QKSieve runtime archive."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path
from typing import Any, Iterable


SOURCE_MANIFEST = "configs/qksieve_robust_source_manifest_20260810.json"
DEFAULT_INCLUDES = ("src", "scripts", "configs")
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_source_manifest(project_root: Path) -> dict[str, Any]:
    path = project_root / SOURCE_MANIFEST
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "qksieve_frozen_source_manifest_v1":
        raise AssertionError("frozen source manifest schema mismatch")
    files = payload.get("files")
    if not isinstance(files, dict) or not files:
        raise AssertionError("frozen source manifest is empty")
    return payload


def validate_frozen_files(
    project_root: Path, source_manifest: dict[str, Any]
) -> dict[str, str]:
    root = project_root.resolve()
    observed: dict[str, str] = {}
    for relative, expected in sorted(source_manifest["files"].items()):
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise AssertionError(f"unsafe frozen source path: {relative}")
        path = (root / relative_path).resolve()
        if root not in path.parents or not path.is_file():
            raise AssertionError(f"frozen source is missing: {relative}")
        digest = sha256_file(path)
        if digest != expected:
            raise AssertionError(
                f"frozen source drifted: {relative}; expected {expected}, "
                f"observed {digest}"
            )
        observed[relative_path.as_posix()] = digest
    return observed


def iter_runtime_files(
    project_root: Path, includes: Iterable[str]
) -> list[tuple[str, Path]]:
    root = project_root.resolve()
    selected: dict[str, Path] = {}
    for include in includes:
        relative = Path(include)
        if relative.is_absolute() or ".." in relative.parts:
            raise AssertionError(f"unsafe include path: {include}")
        source = (root / relative).resolve()
        if root not in source.parents or not source.is_dir():
            raise AssertionError(f"runtime include directory is missing: {include}")
        for path in source.rglob("*"):
            if not path.is_file() or path.is_symlink():
                continue
            rel = path.relative_to(root)
            if "__pycache__" in rel.parts or path.suffix in EXCLUDED_SUFFIXES:
                continue
            selected[rel.as_posix()] = path
    if SOURCE_MANIFEST not in selected:
        raise AssertionError("runtime package does not include the source manifest")
    return sorted(selected.items())


def write_archive(
    output: Path, members: list[tuple[str, Path]]
) -> dict[str, str]:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    hashes: dict[str, str] = {}
    with temporary.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as zipped:
            with tarfile.open(fileobj=zipped, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for relative, path in members:
                    payload = path.read_bytes()
                    info = tarfile.TarInfo(relative)
                    info.size = len(payload)
                    info.mode = 0o755 if path.stat().st_mode & 0o111 else 0o644
                    info.mtime = 0
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    archive.addfile(info, io.BytesIO(payload))
                    hashes[relative] = sha256_bytes(payload)
    temporary.replace(output)
    return hashes


def verify_archive(
    archive_path: Path,
    expected_members: dict[str, str],
    frozen_files: dict[str, str],
) -> None:
    observed: dict[str, str] = {}
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            path = Path(member.name)
            if path.is_absolute() or ".." in path.parts or not member.isfile():
                raise AssertionError(f"unsafe or non-file archive member: {member.name}")
            extracted = archive.extractfile(member)
            if extracted is None:
                raise AssertionError(f"archive member has no bytes: {member.name}")
            observed[path.as_posix()] = sha256_bytes(extracted.read())
    if observed != expected_members:
        missing = sorted(set(expected_members) - set(observed))
        extra = sorted(set(observed) - set(expected_members))
        drifted = sorted(
            name
            for name in set(observed).intersection(expected_members)
            if observed[name] != expected_members[name]
        )
        raise AssertionError(
            f"archive verification failed: missing={missing}, extra={extra}, "
            f"drifted={drifted}"
        )
    for relative, expected in frozen_files.items():
        if observed.get(relative) != expected:
            raise AssertionError(f"archive changed frozen bytes: {relative}")


def package_runtime(
    project_root: Path,
    output: Path,
    *,
    includes: tuple[str, ...] = DEFAULT_INCLUDES,
) -> dict[str, Any]:
    root = project_root.resolve()
    output = output.resolve()
    source_manifest = read_source_manifest(root)
    frozen_files = validate_frozen_files(root, source_manifest)
    members = iter_runtime_files(root, includes)
    member_hashes = write_archive(output, members)
    verify_archive(output, member_hashes, frozen_files)
    report = {
        "schema": "qksieve_frozen_runtime_package_v1",
        "archive": str(output),
        "archive_sha256": sha256_file(output),
        "archive_bytes": output.stat().st_size,
        "member_count": len(member_hashes),
        "includes": list(includes),
        "source_manifest": SOURCE_MANIFEST,
        "source_manifest_sha256": sha256_file(root / SOURCE_MANIFEST),
        "recorded_from_commit": source_manifest.get("recorded_from_commit"),
        "audited_implementation_commit_sha": source_manifest.get(
            "audited_implementation_commit_sha"
        ),
        "frozen_files": frozen_files,
        "members": member_hashes,
    }
    sidecar = Path(str(output) + ".manifest.json")
    sidecar.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--include", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = package_runtime(
        args.project_root,
        args.output,
        includes=tuple(args.include) or DEFAULT_INCLUDES,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
