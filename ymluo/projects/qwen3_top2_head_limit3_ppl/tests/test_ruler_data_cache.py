import json
from types import SimpleNamespace

import pytest

from run_controlled_public_kv_benchmark_v1 import (
    Example,
    hotpot_rows_to_ruler_source,
)
from run_hierarchical_ruler_probe_20260716 import (
    load_examples,
    write_examples_jsonl,
)
from prepare_hierarchical_ruler_data_20260716 import (
    build_manifest,
    manifest_path,
    validate_existing_cache,
)


def make_example(index: int) -> Example:
    return Example(
        benchmark="ruler",
        task="niah_single_1_4096",
        sample_id=f"sample-{index}",
        context=f"context-{index}",
        query="question",
        answers=["answer"],
        prefix_template="",
        suffix_template="question",
        metric="ruler_string_match",
        max_new_tokens=8,
        length=4096,
        all_classes=[],
    )


def test_cached_ruler_examples_are_stream_sharded(tmp_path) -> None:
    path = tmp_path / "examples.jsonl"
    write_examples_jsonl(path, [make_example(index) for index in range(4)])
    rows = load_examples(
        SimpleNamespace(
            max_samples_per_task=1,
            examples_jsonl=path,
            num_shards=2,
            shard_index=1,
        )
    )
    assert [row.sample_id for row in rows] == ["sample-1", "sample-3"]


def test_hotpot_parquet_rows_match_ruler_qas_docs_schema() -> None:
    shared = {"title": ["A"], "sentences": [["Sentence one."]]}
    rows = [
        {"question": "Q1?", "answer": "A1", "context": shared},
        {"question": "Q2?", "answer": "A2", "context": shared},
    ]
    qas, docs = hotpot_rows_to_ruler_source(rows)
    assert docs == ["A\nSentence one."]
    assert qas[0] == {"query": "Q1?", "outputs": ["A1"], "context": [0]}
    assert qas[1]["context"] == [0]


def test_ruler_cache_manifest_prevents_protocol_reuse(tmp_path) -> None:
    args = SimpleNamespace(
        model_name_or_path="model",
        lm_eval_path="lm_eval",
        ruler_tasks="niah_single_1",
        ruler_lengths="4096",
        max_samples_per_task=1,
        max_new_tokens_override=0,
        seed=42,
        ruler_hotpot_parquet="hotpot.parquet",
    )
    manifest = build_manifest(args)
    output = tmp_path / "examples.jsonl"
    output.write_text("{}\n", encoding="utf-8")
    manifest_path(output).write_text(json.dumps(manifest), encoding="utf-8")
    validate_existing_cache(output, manifest)
    changed = dict(manifest)
    changed["seed"] = 43
    with pytest.raises(RuntimeError, match="manifest differs"):
        validate_existing_cache(output, changed)
