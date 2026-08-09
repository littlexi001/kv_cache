#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import run_controlled_public_kv_benchmark_v1 as runner  # noqa: E402


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild the exact blocks selected by a sparse-KV run into a compact prompt."
    )
    parser.add_argument("--selected-results", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    config_payload = json.loads(Path(args.config).read_text(encoding="utf-8"))
    config = runner.Config(**config_payload)
    selected_rows = [row for row in read_csv(Path(args.selected_results)) if row["method"] == "ours_page_gather"]
    if args.limit > 0:
        selected_rows = selected_rows[: args.limit]
    selected_index = {row["sample_id"]: row for row in selected_rows}

    loader_config = replace(config, max_samples_per_task=503)
    examples = runner.load_longbench_v2_examples(loader_config)
    example_index = {example.sample_id: example for example in examples}

    runner.install_llama_layerwise_attention_mask_patch()
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    dtype = runner.resolve_dtype(config.dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=True)
    load_kwargs: dict[str, Any] = {"trust_remote_code": True, "torch_dtype": dtype}
    if config.device_map:
        load_kwargs["device_map"] = config.device_map
    if config.attn_implementation:
        load_kwargs["attn_implementation"] = config.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path, **load_kwargs)
    model.eval()
    model.config.use_cache = True
    input_device = runner.pick_input_device(model, device)

    output_rows: list[dict[str, Any]] = []
    for index, selected_row in enumerate(selected_rows):
        sample_id = selected_row["sample_id"]
        example = example_index[sample_id]
        example_config = runner.config_for_example(config, example)
        original_bundle, pages, _, _, _ = runner.build_bundle(tokenizer, example, example_config)
        selected_ids = {
            int(item)
            for item in selected_row.get("selected_pages", "").split(",")
            if item.strip()
        }
        selected_pages = [page for page in pages if page.page_id in selected_ids]
        compact_context = "\n\n".join(page.text for page in selected_pages)
        compact_example = replace(example, context=compact_context)
        compact_config = replace(
            example_config,
            max_context_tokens=0,
            constrained_choice_decode=True,
            ours_task_policy_json="",
        )
        compact_bundle, _, _, _, _ = runner.build_bundle(tokenizer, compact_example, compact_config)
        compact_cache, prefill_seconds = runner.prefill_prefix(
            model,
            compact_bundle,
            input_device,
            compact_config.prefill_chunk_tokens,
        )
        prediction, generated_ids, query_seconds, decode_seconds = runner.generate_for_example(
            model,
            tokenizer,
            compact_bundle,
            compact_cache,
            compact_example.max_new_tokens,
            input_device,
            compact_example,
            compact_config,
        )
        score = runner.score_prediction(
            compact_example.metric,
            prediction,
            compact_example.answers,
            compact_example.all_classes,
        )
        output_rows.append(
            {
                "sample_id": sample_id,
                "domain": example.domain,
                "answer": example.answers[0],
                "prediction": prediction,
                "score": score,
                "selected_pages": len(selected_pages),
                "original_prefix_tokens": original_bundle.query_start,
                "compact_prefix_tokens": compact_bundle.query_start,
                "token_ratio": compact_bundle.query_start / max(1, original_bundle.query_start),
                "generated_tokens": len(generated_ids),
                "prefill_seconds": prefill_seconds,
                "query_seconds": query_seconds,
                "decode_seconds": decode_seconds,
                "total_seconds": prefill_seconds + query_seconds + decode_seconds,
            }
        )
        print(
            f"[{index + 1}/{len(selected_rows)}] {sample_id} score={score:.0f} "
            f"tokens={compact_bundle.query_start}/{original_bundle.query_start} pred={prediction}",
            flush=True,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "task_results.csv", output_rows)
    summary = {
        "samples": len(output_rows),
        "score": sum(float(row["score"]) for row in output_rows) / max(1, len(output_rows)),
        "mean_token_ratio": sum(float(row["token_ratio"]) for row in output_rows) / max(1, len(output_rows)),
        "mean_total_seconds": sum(float(row["total_seconds"]) for row in output_rows) / max(1, len(output_rows)),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
