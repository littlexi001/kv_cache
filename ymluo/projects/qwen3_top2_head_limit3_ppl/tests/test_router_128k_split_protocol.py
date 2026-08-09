import json
import sys

from summarize_router_128k_split_20260716 import FOLDS, TOPICS, main


def test_each_topic_is_held_out_exactly_once() -> None:
    held_out = [topic for split in FOLDS.values() for topic in split["test"]]
    assert sorted(held_out) == sorted(TOPICS)
    assert len(held_out) == len(set(held_out))


def test_each_fold_has_strict_topic_isolation() -> None:
    for split in FOLDS.values():
        train = set(split["train"])
        calibration = set(split["calibration"])
        test = set(split["test"])
        assert train.isdisjoint(calibration)
        assert train.isdisjoint(test)
        assert calibration.isdisjoint(test)
        assert train | calibration | test == set(TOPICS)


def test_summary_keeps_refresh_two_as_a_matched_ablation(
    tmp_path, monkeypatch
) -> None:
    paired = tmp_path / "paired"
    router = tmp_path / "router"
    output = tmp_path / "summary"
    paired.mkdir()
    router.mkdir()
    topic_to_fold = {
        topic: fold
        for fold, split in FOLDS.items()
        for topic in split["test"]
    }
    for topic in TOPICS:
        full = {
            "target_token_ids": [1, 2],
            "ppl": 2.0,
            "nll": 1.0,
            "synchronized_model_forward_seconds": 2.0,
            "prefill_seconds": 3.0,
        }
        fixed = {**full, "ppl": 2.1, "nll": 1.05}
        (paired / f"{topic}_w2_full.json").write_text(json.dumps(full))
        (paired / f"{topic}_w2_sparse.json").write_text(json.dumps(fixed))
        fold = topic_to_fold[topic]
        for refresh, suffix in ((1, ""), (2, "_refresh2")):
            dynamic = {
                "target_token_ids": [1, 2],
                "candidate_refresh_interval": refresh,
                "ppl": 2.05,
                "nll": 1.025,
                "online_seconds": 1.0,
                "prefill_seconds": 3.0,
                "cache_conversion_seconds": 0.5,
                "action_fractions": [0.01, 0.015],
                "hierarchical_over_final_length_full_kv": 0.1,
                "router_seconds": 0.01,
            }
            (router / f"test_{fold}_{topic}_w2{suffix}.json").write_text(
                json.dumps(dynamic)
            )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summary",
            "--router_dir",
            str(router),
            "--paired_dir",
            str(paired),
            "--output_dir",
            str(output),
            "--bootstrap_samples",
            "20",
        ],
    )

    main()

    payload = json.loads((output / "summary.json").read_text())
    assert payload["cases"] == 6
    assert payload["ablation_cases"] == 12
    assert payload["by_candidate_refresh_interval"]["1"]["cases"] == 6
    assert payload["by_candidate_refresh_interval"]["2"]["cases"] == 6
