from __future__ import annotations

import json
from argparse import Namespace

from run_hierarchical_longicl_probe_20260716 import LONG_ICL_DOMAIN, load_examples


def test_load_examples_filters_domain_then_shards(tmp_path) -> None:
    rows = []
    for index in range(5):
        rows.append(
            {
                "_id": str(index),
                "domain": LONG_ICL_DOMAIN if index < 4 else "Other",
                "question": "question",
                "choice_A": "a",
                "choice_B": "b",
                "choice_C": "c",
                "choice_D": "d",
                "answer": "B",
                "context": "context",
            }
        )
    path = tmp_path / "split.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    args = Namespace(
        longbench_v2_json=path,
        max_samples=0,
        max_new_tokens=32,
        num_shards=2,
        shard_index=1,
    )
    examples = load_examples(args)
    assert [example.sample_id for example in examples] == ["1", "3"]
    assert all(example.metric == "longbench_v2_mc" for example in examples)
