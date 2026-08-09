from __future__ import annotations

import argparse
import json
import tarfile
import tempfile
from pathlib import Path
from typing import Any

from io_utils import read_json, write_csv, write_json


def safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.getmembers():
        target = (destination / member.name).resolve()
        if root not in target.parents and target != root:
            raise RuntimeError(f"unsafe archive path: {member.name}")
    try:
        archive.extractall(destination, filter="data")
    except TypeError:  # Python < 3.12 after explicit path validation above
        archive.extractall(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    archives = sorted(args.bundle_dir.glob("*.tar.gz"))
    if not archives:
        raise FileNotFoundError(f"no tar.gz bundles in {args.bundle_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    sources = []
    manifest_hashes: dict[str, str] = {}
    with tempfile.TemporaryDirectory() as temporary:
        extraction = Path(temporary)
        for index, path in enumerate(archives):
            target = extraction / str(index)
            target.mkdir()
            with tarfile.open(path, "r:gz") as archive:
                safe_extract(archive, target)
            summaries = list(target.rglob("strategy_summary.json"))
            if len(summaries) != 1:
                raise RuntimeError(f"expected one strategy_summary.json in {path}, found {len(summaries)}")
            payload = read_json(summaries[0])
            for row in payload["rows"]:
                rows.append({**row, "source_bundle": path.name})
            metadata_files = list(target.rglob("manifest_metadata.json"))
            if len(metadata_files) > 1:
                raise RuntimeError(
                    f"expected at most one manifest_metadata.json in {path}, "
                    f"found {len(metadata_files)}"
                )
            if metadata_files:
                metadata = read_json(metadata_files[0])
                manifest_hash = metadata.get("manifest_sha256")
                if manifest_hash:
                    manifest_hashes[path.name] = str(manifest_hash)
            sources.append(str(path))
    unique_manifest_hashes = sorted(set(manifest_hashes.values()))
    if len(unique_manifest_hashes) > 1:
        details = ", ".join(
            f"{bundle}={digest}" for bundle, digest in sorted(manifest_hashes.items())
        )
        raise RuntimeError(
            "result bundles use different deterministic DCLM manifests; "
            f"refusing an unmatched comparison: {details}"
        )
    # Keep one base row; repeated base evaluations remain available in source bundles.
    deduplicated: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["strategy"]), int(row["step"]))
        if key not in deduplicated or bool(row.get("critical_complete")):
            deduplicated[key] = row
    rows = sorted(deduplicated.values(), key=lambda row: (int(row["step"]), str(row["strategy"])))
    native = {
        int(row["step"]): row
        for row in rows
        if row["strategy"] == "native_rope" and bool(row.get("critical_complete"))
    }
    for row in rows:
        step = int(row["step"])
        control = native.get(step)
        if control and row["strategy"] not in {"native_rope", "base_checkpoint"}:
            row_tokens = row.get("tokens_seen_nominal")
            control_tokens = control.get("tokens_seen_nominal")
            if (
                row_tokens is not None
                and control_tokens is not None
                and int(row_tokens) != int(control_tokens)
            ):
                row["native_control_status"] = "token_mismatch"
                row["native_control_tokens"] = control_tokens
                continue
            row["native_control_status"] = "matched"
            if row.get("validation_ppl") is not None and control.get("validation_ppl") is not None:
                row["validation_ppl_change_vs_native_percent"] = 100.0 * (
                    float(row["validation_ppl"]) / float(control["validation_ppl"]) - 1.0
                )
                row["ppl_guardrail_pass"] = row["validation_ppl_change_vs_native_percent"] <= 2.0
            for metric in ["controlled_qa_f1_percent", "longbench_qa_f1_percent"]:
                if row.get(metric) is not None and control.get(metric) is not None:
                    row[f"{metric}_change_vs_native_pp"] = float(row[metric]) - float(control[metric])
    write_csv(args.output_dir / "combined_summary.csv", rows)
    write_json(
        args.output_dir / "combined_summary.json",
        {
            "source_bundles": sources,
            "manifest_sha256": unique_manifest_hashes[0] if unique_manifest_hashes else None,
            "rows": rows,
            "claim_boundary": "Matched-token native deltas are screening evidence; multi-seed full evaluation is still required.",
        },
    )
    print(args.output_dir / "combined_summary.csv")


if __name__ == "__main__":
    main()
