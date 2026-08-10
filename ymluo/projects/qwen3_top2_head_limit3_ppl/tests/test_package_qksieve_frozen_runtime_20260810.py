from __future__ import annotations

import hashlib
import json
import sys
import tarfile
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from package_qksieve_frozen_runtime_20260810 import package_runtime  # noqa: E402


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def project(tmp_path: Path) -> tuple[Path, bytes]:
    root = tmp_path / "project"
    source = b"frozen = True\n"
    files = {
        "src/core.py": source,
        "src/helper.py": b"helper = 1\n",
        "scripts/launch.sh": b"#!/usr/bin/env bash\necho run\n",
    }
    for relative, payload in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    config = root / "configs"
    config.mkdir()
    (config / "qksieve_robust_source_manifest_20260810.json").write_text(
        json.dumps(
            {
                "schema": "qksieve_frozen_source_manifest_v1",
                "recorded_from_commit": "test-commit",
                "audited_implementation_commit_sha": "implementation-commit",
                "files": {"src/core.py": digest(source)},
            }
        ),
        encoding="utf-8",
    )
    return root, source


def test_runtime_package_preserves_frozen_bytes_and_is_deterministic(
    tmp_path: Path,
) -> None:
    root, source = project(tmp_path)
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"
    report = package_runtime(root, first)
    package_runtime(root, second)

    assert report["frozen_files"]["src/core.py"] == digest(source)
    assert report["archive_sha256"] == digest(first.read_bytes())
    assert first.read_bytes() == second.read_bytes()
    assert Path(str(first) + ".manifest.json").is_file()
    with tarfile.open(first, "r:gz") as archive:
        assert archive.extractfile("src/core.py").read() == source
        assert "configs/qksieve_robust_source_manifest_20260810.json" in {
            member.name for member in archive.getmembers()
        }


def test_runtime_package_rejects_line_ending_drift(tmp_path: Path) -> None:
    root, _ = project(tmp_path)
    (root / "src/core.py").write_bytes(b"frozen = True\r\n")

    with pytest.raises(AssertionError, match="frozen source drifted"):
        package_runtime(root, tmp_path / "drifted.tar.gz")
