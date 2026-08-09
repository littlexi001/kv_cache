from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


SMALL_MODEL_FILES = {
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
    "params.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer.model.v3",
    "tokenizer_config.json",
}


def make_session() -> requests.Session:
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        status=5,
        backoff_factor=2.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def normalize_endpoints(endpoints: Iterable[str]) -> list[str]:
    normalized = []
    for endpoint in endpoints:
        value = endpoint.rstrip("/")
        if value and value not in normalized:
            normalized.append(value)
    return normalized


def fetch_repository_metadata(
    session: requests.Session,
    repo_id: str,
    endpoints: list[str],
) -> tuple[dict[str, Any], str]:
    errors = []
    for endpoint in endpoints:
        url = f"{endpoint}/api/models/{repo_id}?blobs=true"
        try:
            response = session.get(url, timeout=(20, 120))
            response.raise_for_status()
            return response.json(), endpoint
        except Exception as error:  # pragma: no cover - network-specific branch
            errors.append(f"{url}: {error}")
    raise RuntimeError("repository metadata requests failed:\n" + "\n".join(errors))


def sibling_map(metadata: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item["rfilename"]): item
        for item in metadata.get("siblings", [])
        if item.get("rfilename")
    }


def metadata_files(siblings: dict[str, dict[str, Any]]) -> list[str]:
    selected = {
        name
        for name in siblings
        if name in SMALL_MODEL_FILES
        or name.startswith("tokenizer_")
        or name == "added_tokens.json"
    }
    if "config.json" not in selected:
        raise RuntimeError("repository has no config.json")
    return sorted(selected)


def weight_files(
    local_dir: Path,
    siblings: dict[str, dict[str, Any]],
) -> list[str]:
    for index_name in (
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    ):
        index_path = local_dir / index_name
        if index_path.is_file():
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            weights = sorted(set(payload.get("weight_map", {}).values()))
            if not weights:
                raise RuntimeError(f"weight index is empty: {index_path}")
            missing = [name for name in weights if name not in siblings]
            if missing:
                raise RuntimeError(
                    f"weight index references absent repository files: {missing}"
                )
            return weights

    for candidate in (
        "model.safetensors",
        "pytorch_model.bin",
        "consolidated.safetensors",
    ):
        if candidate in siblings:
            return [candidate]
    raise RuntimeError("repository exposes no supported model weights")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_file_metadata(item: dict[str, Any]) -> tuple[int, str]:
    lfs = item.get("lfs") or {}
    size = int(lfs.get("size") or item.get("size") or 0)
    digest = str(lfs.get("sha256") or "")
    return size, digest


def valid_existing_file(path: Path, expected_size: int, expected_sha: str) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    if expected_size and path.stat().st_size != expected_size:
        return False
    return not expected_sha or sha256_file(path) == expected_sha


def download_file(
    session: requests.Session,
    repo_id: str,
    revision: str,
    filename: str,
    item: dict[str, Any],
    local_dir: Path,
    endpoints: list[str],
) -> dict[str, Any]:
    destination = local_dir / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected_size, expected_sha = expected_file_metadata(item)
    if valid_existing_file(destination, expected_size, expected_sha):
        return {
            "path": filename,
            "bytes": destination.stat().st_size,
            "sha256": expected_sha or "not_provided",
            "status": "existing_valid",
        }

    partial = destination.with_suffix(destination.suffix + ".part")
    if expected_size and partial.is_file() and partial.stat().st_size > expected_size:
        partial.unlink()
    if valid_existing_file(partial, expected_size, expected_sha):
        partial.replace(destination)
        return {
            "path": filename,
            "bytes": destination.stat().st_size,
            "sha256": expected_sha or "not_provided",
            "status": "resumed_complete",
        }
    errors = []
    for endpoint in endpoints:
        encoded = quote(filename, safe="/")
        url = f"{endpoint}/{repo_id}/resolve/{revision}/{encoded}?download=true"
        try:
            offset = partial.stat().st_size if partial.is_file() else 0
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            with session.get(
                url,
                headers=headers,
                stream=True,
                timeout=(20, 600),
                allow_redirects=True,
            ) as response:
                response.raise_for_status()
                append = offset > 0 and response.status_code == 206
                mode = "ab" if append else "wb"
                with partial.open(mode) as stream:
                    for chunk in response.iter_content(chunk_size=8 << 20):
                        if chunk:
                            stream.write(chunk)
            if expected_size and partial.stat().st_size != expected_size:
                raise RuntimeError(
                    f"size mismatch: got {partial.stat().st_size}, "
                    f"expected {expected_size}"
                )
            observed_sha = sha256_file(partial) if expected_sha else "not_provided"
            if expected_sha and observed_sha != expected_sha:
                raise RuntimeError(
                    f"SHA256 mismatch: got {observed_sha}, expected {expected_sha}"
                )
            partial.replace(destination)
            return {
                "path": filename,
                "bytes": destination.stat().st_size,
                "sha256": observed_sha,
                "status": "downloaded",
                "endpoint": endpoint,
            }
        except Exception as error:  # pragma: no cover - network-specific branch
            errors.append(f"{url}: {error}")
    raise RuntimeError(
        f"all direct downloads failed for {filename}:\n" + "\n".join(errors)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download a public Hugging Face model snapshot without metadata HEAD "
            "requests; validates LFS size and SHA256 and resumes .part files."
        )
    )
    parser.add_argument("--repo_id", required=True)
    parser.add_argument("--local_dir", required=True, type=Path)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--endpoint", action="append", default=[])
    args = parser.parse_args()

    endpoints = normalize_endpoints(
        args.endpoint
        or [
            os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"),
            "https://huggingface.co",
        ]
    )
    session = make_session()
    metadata, metadata_endpoint = fetch_repository_metadata(
        session, args.repo_id, endpoints
    )
    revision = str(metadata.get("sha") or args.revision)
    siblings = sibling_map(metadata)
    args.local_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for filename in metadata_files(siblings):
        records.append(
            download_file(
                session,
                args.repo_id,
                revision,
                filename,
                siblings[filename],
                args.local_dir,
                endpoints,
            )
        )
    for filename in weight_files(args.local_dir, siblings):
        records.append(
            download_file(
                session,
                args.repo_id,
                revision,
                filename,
                siblings[filename],
                args.local_dir,
                endpoints,
            )
        )

    manifest = {
        "schema": "hf_direct_snapshot_v1",
        "repo_id": args.repo_id,
        "revision": revision,
        "metadata_endpoint": metadata_endpoint,
        "files": records,
    }
    manifest_path = args.local_dir / "direct_download_manifest.json"
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(manifest_path)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
