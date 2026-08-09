from __future__ import annotations

import argparse
import hashlib
import tarfile
import time
from pathlib import Path

from io_utils import write_json


EXCLUDED_PARTS = {"checkpoints", "bundles", "tokenizer"}
EXCLUDED_NAMES = {"controller.pid"}


def included_files(run_dir: Path) -> list[Path]:
    files = []
    for path in run_dir.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(run_dir)
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        if path.name in EXCLUDED_NAMES:
            continue
        files.append(path)
    return sorted(files)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--strategy-name", required=True)
    args = parser.parse_args()
    bundle_dir = args.run_dir / "bundles"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    files = included_files(args.run_dir)
    manifest = {
        "strategy": args.strategy_name,
        "run_dir": str(args.run_dir),
        "checkpoints_included": False,
        "files": [
            {
                "path": str(path.relative_to(args.run_dir)),
                "bytes": path.stat().st_size,
                "sha256": digest(path),
            }
            for path in files
        ],
    }
    manifest_path = args.run_dir / "bundle_manifest.json"
    write_json(manifest_path, manifest)
    files = included_files(args.run_dir)
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    destination = bundle_dir / f"{args.strategy_name}_{stamp}.tar.gz"
    with tarfile.open(destination, "w:gz") as archive:
        for path in files:
            archive.add(path, arcname=f"{args.strategy_name}/{path.relative_to(args.run_dir)}")
    print(destination)


if __name__ == "__main__":
    main()

