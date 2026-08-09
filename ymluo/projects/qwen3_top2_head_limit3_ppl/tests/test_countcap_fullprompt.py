from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import csv
import json
import sys

import pytest
import torch


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import run_sample_calibrated_longbench_20260717 as runner
import run_direct_countcap_denseprompt_ppl_20260725 as countcap_ppl
import run_head_top2_targeted_ppl_20260714 as sparse_attention


class FakeTokenizer:
    eos_token_id = 2
    _vocab = {
        "<|end_of_text|>": 5,
        "<|eom_id|>": 6,
        "<|eot_id|>": 7,
    }

    @staticmethod
    def decode(token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return " ".join(str(token_id) for token_id in token_ids)

    @classmethod
    def get_vocab(cls):
        return dict(cls._vocab)

    @staticmethod
    def encode(text, add_special_tokens=False):
        del add_special_tokens
        return [4] if text == "\n" else [3]


def logits_for(token_id: int) -> torch.Tensor:
    logits = torch.zeros((1, 8), dtype=torch.float32)
    logits[0, token_id] = 1.0
    return logits


def test_longbench_stop_tokens_match_llama_and_samsum_policy():
    tokenizer = FakeTokenizer()

    assert runner.longbench_stop_token_ids(tokenizer, "qasper") == {
        2,
        5,
        6,
        7,
    }
    assert runner.longbench_stop_token_ids(tokenizer, "samsum") == {
        2,
        4,
        5,
        6,
        7,
    }


@pytest.mark.parametrize(
    "method",
    [
        runner.COUNTCAP_QK_BALANCED_PACKED_METHOD,
        runner.QKSIEVE_FULLTOPK_METHOD,
        runner.QKSIEVE_QFUSED_FULLTOPK_METHOD,
        runner.QKSIEVE_FIXED410_FULLTOPK_METHOD,
        runner.QKSIEVE_FIXED410_PRERERANK_L00TO08_FULLTOPK_METHOD,
        runner.FIER_RTN1_G32_FULLTOPK_METHOD,
        runner.FIER_RTN1_G32_PACKED_FULLTOPK_METHOD,
        "countcap_massadaptive_fullprompt",
    ],
)
def test_dense_prompt_suffix_protocol_covers_frozen_and_fair_methods(method):
    assert runner.uses_dense_prompt_suffix(method)


@pytest.mark.parametrize(
    "method",
    ["full_kv", "global_partition", "qgate_partition"],
)
def test_dense_prompt_suffix_protocol_excludes_non_fullprompt_methods(method):
    assert not runner.uses_dense_prompt_suffix(method)


def test_cache_auto_uses_measured_length_crossover(monkeypatch):
    method = runner.COUNTCAP_CACHE_AUTO_METHOD

    monkeypatch.delenv("COUNTCAP_INPLACE_CACHE_MIN_TOKENS", raising=False)
    assert not runner.should_use_preallocated_cache(method, 8_000)
    assert runner.should_use_preallocated_cache(method, 16_000)

    monkeypatch.setenv("COUNTCAP_INPLACE_CACHE_MIN_TOKENS", "20000")
    assert not runner.should_use_preallocated_cache(method, 16_000)
    assert runner.should_use_preallocated_cache(method, 24_000)


@pytest.mark.parametrize(
    ("history_count", "expected"),
    (
        (128, 128),
        (2_048, 256),
        (4_096, 256),
        (8_192, 492),
        (16_000, 960),
        (24_000, 1_280),
        (32_000, 1_280),
        (64_000, 1_280),
        (128_000, 1_280),
    ),
)
def test_direct_countcap_target_count(history_count, expected):
    assert (
        sparse_attention.direct_countcap_target_count(history_count)
        == expected
    )


@pytest.mark.parametrize(
    ("history_tokens", "window", "stride", "anchor", "expected"),
    [
        (2048, 0, 128512, 128000, (125952, 128000)),
        (32000, 1, 128512, 128000, (224512, 256512)),
        (128000, 2, 128512, 128000, (257024, 385024)),
        (32000, 2, 32512, 0, (65024, 97024)),
    ],
)
def test_denseprompt_window_bounds(
    history_tokens,
    window,
    stride,
    anchor,
    expected,
):
    assert countcap_ppl.window_bounds(
        history_tokens,
        window,
        stride,
        anchor,
    ) == expected


@pytest.mark.parametrize(
    ("mode", "history_tokens", "expected"),
    (
        ("dynamic", 128_000, False),
        ("preallocated", 2_000, True),
        ("auto", 8_000, False),
        ("auto", 16_000, True),
    ),
)
def test_denseprompt_cache_policy(mode, history_tokens, expected):
    assert (
        countcap_ppl.should_use_preallocated_cache(
            mode,
            history_tokens,
            preallocated_cache_min_tokens=14_000,
        )
        is expected
    )


def test_denseprompt_preallocated_cache_has_decode_headroom():
    cache = countcap_ppl.build_initial_cache(
        "auto",
        history_tokens=16_000,
        max_new_tokens=256,
        preallocated_cache_min_tokens=14_000,
    )

    assert isinstance(cache, countcap_ppl.PreallocatedDynamicCache)
    assert cache.max_cache_len == 16_264
    assert (
        countcap_ppl.build_initial_cache(
            "auto",
            history_tokens=8_000,
            max_new_tokens=256,
            preallocated_cache_min_tokens=14_000,
        )
        is None
    )


def test_explicit_inplace_cache_method_is_not_length_gated():
    method = (
        "countcap_fullprompt_keypca_direct_qkvfused_qprojscan_"
        "qkvsplitauto_inplacecache_prefillindex"
    )
    assert runner.should_use_preallocated_cache(method, 8_000)


@pytest.mark.parametrize("interval", (2, 4, 8))
def test_qprojscan_periodic_reuse_uses_cache_crossover(interval):
    method = (
        "countcap_fullprompt_keypca_direct_qkvfused_qprojscan_"
        f"qkvsplitauto_cacheauto_reuse{interval}_prefillindex"
    )
    assert not runner.should_use_preallocated_cache(method, 8_000)
    assert runner.should_use_preallocated_cache(method, 16_000)


def test_prompt_limit_keeps_asymmetric_document_ends(monkeypatch):
    token_map = {
        "prefix": [10],
        "context": list(range(20, 30)),
        "suffix question": [40, 41],
    }
    monkeypatch.setattr(
        runner.lb,
        "token_ids",
        lambda tokenizer, text: list(token_map[text]),
    )
    tokenizer = SimpleNamespace(bos_token_id=1)
    example = SimpleNamespace(
        prefix_template="prefix",
        context="context",
        suffix_template="suffix {input}",
        query="question",
        no_chat=True,
    )

    bundle = runner.build_prompt_limited_bundle(
        tokenizer,
        example,
        max_prompt_tokens=8,
        prompt_wrapper="llama3",
    )

    # BOS + prefix consume two slots in the first half, while the two-token
    # suffix consumes two slots in the second half.
    assert bundle.input_ids.tolist() == [[1, 10, 20, 21, 28, 29, 40, 41]]
    assert bundle.query_start == 6
    assert bundle.suffix_token_count == 2


def test_prompt_limit_reclaims_front_budget_for_oversized_suffix(
    monkeypatch,
):
    token_map = {
        "prefix": [10],
        "context": list(range(20, 30)),
        "suffix question": [40, 41, 42, 43, 44],
    }
    monkeypatch.setattr(
        runner.lb,
        "token_ids",
        lambda tokenizer, text: list(token_map[text]),
    )
    tokenizer = SimpleNamespace(bos_token_id=1)
    example = SimpleNamespace(
        prefix_template="prefix",
        context="context",
        suffix_template="suffix {input}",
        query="question",
        no_chat=True,
    )

    bundle = runner.build_prompt_limited_bundle(
        tokenizer,
        example,
        max_prompt_tokens=8,
        prompt_wrapper="llama3",
    )

    assert bundle.input_ids.tolist() == [
        [1, 10, 20, 40, 41, 42, 43, 44]
    ]
    assert bundle.query_start == 3
    assert bundle.suffix_token_count == 5


def test_prompt_limit_reclaims_back_budget_for_oversized_prefix(
    monkeypatch,
):
    token_map = {
        "prefix": [10, 11, 12, 13, 14],
        "context": list(range(20, 30)),
        "suffix question": [40],
    }
    monkeypatch.setattr(
        runner.lb,
        "token_ids",
        lambda tokenizer, text: list(token_map[text]),
    )
    tokenizer = SimpleNamespace(bos_token_id=None)
    example = SimpleNamespace(
        prefix_template="prefix",
        context="context",
        suffix_template="suffix {input}",
        query="question",
        no_chat=True,
    )

    bundle = runner.build_prompt_limited_bundle(
        tokenizer,
        example,
        max_prompt_tokens=8,
        prompt_wrapper="llama3",
    )

    assert bundle.input_ids.tolist() == [
        [10, 11, 12, 13, 14, 28, 29, 40]
    ]
    assert bundle.query_start == 7
    assert bundle.suffix_token_count == 1


class CharacterTokenizer:
    bos_token_id = 0

    def __call__(
        self,
        text,
        *,
        add_special_tokens,
        truncation,
    ):
        assert not truncation
        values = [ord(character) for character in text]
        if add_special_tokens:
            values.insert(0, self.bos_token_id)
        return {"input_ids": values}

    def decode(self, values, *, skip_special_tokens):
        assert skip_special_tokens
        return "".join(
            chr(int(value))
            for value in values
            if int(value) != self.bos_token_id
        )


class ChatTemplateCharacterTokenizer(CharacterTokenizer):
    def __init__(self):
        self.calls = []

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
    ):
        self.calls.append(
            {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
            }
        )
        content = messages[0]["content"]
        return [900, *[ord(character) for character in content], 901]


def test_official_middle_prompt_matches_whole_prompt_halves():
    tokenizer = CharacterTokenizer()
    example = SimpleNamespace(
        prefix_template="P",
        context="abcdefghij",
        suffix_template="{input}",
        query="S",
        no_chat=True,
    )

    prompt_ids = runner.official_middle_prompt_ids(
        tokenizer,
        example,
        max_prompt_tokens=8,
        prompt_wrapper="none",
    )

    assert prompt_ids == [
        0,
        ord("P"),
        ord("a"),
        ord("b"),
        ord("h"),
        ord("i"),
        ord("j"),
        ord("S"),
    ]


def test_official_middle_supports_native_tokenizer_chat_template():
    tokenizer = ChatTemplateCharacterTokenizer()
    example = SimpleNamespace(
        prefix_template="P",
        context="abcdefghij",
        suffix_template="{input}",
        query="S",
        no_chat=False,
    )

    prompt_ids = runner.official_middle_prompt_ids(
        tokenizer,
        example,
        max_prompt_tokens=8,
        prompt_wrapper="tokenizer_chat",
    )

    assert prompt_ids == [
        900,
        ord("P"),
        ord("a"),
        ord("b"),
        ord("h"),
        ord("i"),
        ord("j"),
        ord("S"),
        901,
    ]
    assert tokenizer.calls == [
        {
            "messages": [{"role": "user", "content": "PabhijS"}],
            "tokenize": True,
            "add_generation_prompt": True,
        }
    ]


def test_preserve_suffix_rejects_native_chat_template():
    tokenizer = ChatTemplateCharacterTokenizer()
    example = SimpleNamespace(
        prefix_template="P",
        context="abc",
        suffix_template="{input}",
        query="S",
        no_chat=False,
    )

    with pytest.raises(ValueError, match="official_middle"):
        runner.build_prompt_limited_bundle(
            tokenizer,
            example,
            max_prompt_tokens=8,
            prompt_wrapper="tokenizer_chat",
        )


def test_official_middle_bundle_reserves_only_frozen_query_tail():
    tokenizer = CharacterTokenizer()
    example = SimpleNamespace(
        prefix_template="P",
        context="abcdefghij",
        suffix_template="{input}",
        query="S",
        no_chat=True,
    )

    bundle = runner.build_official_middle_bundle(
        tokenizer,
        example,
        max_prompt_tokens=8,
        prompt_wrapper="none",
        query_tail_tokens=3,
    )

    assert bundle.input_ids.shape[-1] == 8
    assert bundle.query_start == 5
    assert bundle.suffix_token_count == 3


def test_load_examples_applies_sample_offset_before_limit(tmp_path):
    rows = [
        {
            "_id": f"sample-{index}",
            "context": f"context-{index}",
            "input": f"question-{index}",
            "answers": [f"answer-{index}"],
            "length": index + 1,
            "all_classes": [],
        }
        for index in range(3)
    ]
    (tmp_path / "qasper.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        tasks="qasper",
        longbench_data_dir=tmp_path,
        sample_offset_per_task=1,
        max_samples_per_task=1,
        num_shards=1,
        shard_index=0,
        max_new_tokens_override=8,
    )

    examples = runner.load_examples(args)

    assert len(examples) == 1
    assert examples[0].sample_id == "sample-1"
    assert examples[0].context == "context-1"


def test_qk_trace_config_is_explicit_and_requires_step_zero(tmp_path):
    args = SimpleNamespace(
        qk_trace_output_dir=tmp_path,
        qk_trace_method=runner.QKSIEVE_FULLTOPK_METHOD,
        qk_trace_layers="0,8,31",
        qk_trace_steps="0,7,31,127",
    )
    config = runner.resolve_qk_trace_config(
        args,
        [runner.QKSIEVE_FULLTOPK_METHOD],
    )

    assert config == {
        "output_dir": tmp_path,
        "method": runner.QKSIEVE_FULLTOPK_METHOD,
        "layers": (0, 8, 31),
        "steps": (0, 7, 31, 127),
        "prefill_query_tail_tokens": 32,
    }

    args.qk_trace_steps = "1,7"
    with pytest.raises(ValueError, match="step 0"):
        runner.resolve_qk_trace_config(
            args,
            [runner.QKSIEVE_FULLTOPK_METHOD],
        )


def test_qk_trace_config_rejects_non_qksieve_or_unlisted_method(tmp_path):
    args = SimpleNamespace(
        qk_trace_output_dir=tmp_path,
        qk_trace_method="full_kv",
        qk_trace_layers="0",
        qk_trace_steps="0",
    )
    with pytest.raises(ValueError, match="QKSieve"):
        runner.resolve_qk_trace_config(args, ["full_kv"])

    args.qk_trace_method = runner.QKSIEVE_FULLTOPK_METHOD
    with pytest.raises(ValueError, match="included"):
        runner.resolve_qk_trace_config(args, ["full_kv"])


def test_qk_trace_filename_is_sanitized(tmp_path):
    path = runner.qk_trace_path(
        tmp_path,
        "qa/task",
        "sample id:1",
        runner.QKSIEVE_FULLTOPK_METHOD,
    )
    assert path.parent == tmp_path
    assert path.name == (
        "qa_task__sample_id_1__"
        "qksieve_fullprompt_auto_plain_fulltopk.pt"
    )


def test_fullprompt_batches_suffix_before_sparse_decode(monkeypatch):
    events = []
    bundle = SimpleNamespace(
        input_ids=torch.tensor([[10, 11, 12, 13]], dtype=torch.long),
        query_start=2,
    )
    args = SimpleNamespace(
        mass_threshold=0.75,
        sample_fraction=0.0025,
        candidate_fraction=0.06,
        projection_dim=48,
        value_mass_threshold=1.0,
        partition_ucb_z=0.0,
        partition_overfetch_factor=2,
        collect_attention_stats=False,
    )

    monkeypatch.setattr(runner, "set_attention_implementation", lambda model, name: events.append(("attention", name)))
    monkeypatch.setattr(runner, "head_qabs_sampled_mass_mode", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(runner, "synchronize_cuda_devices", lambda: None)
    monkeypatch.setattr(runner.lb, "prefill_prefix", lambda *args, **kwargs: ("prefix-cache", 1.0))

    def fake_run_token_segment(model, input_ids, cache, position_start, input_device):
        del model, input_device
        events.append(("dense_suffix", input_ids.tolist(), cache, position_start))
        return "prompt-cache", logits_for(3), 0.2

    monkeypatch.setattr(runner.lb, "run_token_segment", fake_run_token_segment)

    def fake_run_one_token(model, token_id, cache, position, input_device, collect_attention_stats=False):
        del model, input_device, collect_attention_stats
        events.append(("sparse_token", token_id, cache, position))
        return "decode-cache", logits_for(2), 0.1, {}

    monkeypatch.setattr(runner, "run_one_token", fake_run_one_token)

    result = runner.generate_global_partition(
        model=object(),
        tokenizer=FakeTokenizer(),
        input_device=torch.device("cpu"),
        bundle=bundle,
        max_new_tokens=4,
        prefill_chunk_tokens=2048,
        budget_fractions=(0.02,),
        args=args,
        score_mode=runner.COUNTCAP_SCORE_MODE,
        candidate_fraction=0.06,
        projection_dim=48,
        dense_suffix=True,
    )

    dense_events = [event for event in events if event[0] == "dense_suffix"]
    sparse_events = [event for event in events if event[0] == "sparse_token"]
    assert dense_events == [("dense_suffix", [[12, 13]], "prefix-cache", 2)]
    assert sparse_events == [("sparse_token", 3, "prompt-cache", 4)]
    assert result["generated_ids"] == [3]
    assert result["query_seconds"] == 0.2


@pytest.mark.parametrize(
    "score_mode",
    [
        runner.QKSIEVE_FULLTOPK_SCORE_MODE,
        runner.QKSIEVE_FIXED410_FULLTOPK_SCORE_MODE,
        runner.QKSIEVE_FIXED410_PRERERANK_L00TO08_FULLTOPK_SCORE_MODE,
    ],
)
def test_qksieve_uses_requested_query_tail_and_exports_it(
    monkeypatch,
    score_mode,
):
    events = []
    bundle = SimpleNamespace(
        input_ids=torch.tensor([[10, 11, 12, 13]], dtype=torch.long),
        query_start=2,
    )
    args = SimpleNamespace(
        mass_threshold=0.75,
        sample_fraction=0.0025,
        candidate_fraction=0.06,
        projection_dim=128,
        value_mass_threshold=1.0,
        partition_ucb_z=0.0,
        partition_overfetch_factor=0,
        collect_attention_stats=False,
        qk_metric_query_shrinkage=0.75,
    )
    captured = {3: torch.randn(1, 4, 32, 128)}
    exported = {}

    monkeypatch.setattr(
        runner,
        "set_attention_implementation",
        lambda model, name: None,
    )
    monkeypatch.setattr(
        runner,
        "head_qabs_sampled_mass_mode",
        lambda *args, **kwargs: nullcontext(),
    )
    monkeypatch.setattr(runner, "synchronize_cuda_devices", lambda: None)
    monkeypatch.setattr(
        runner.lb,
        "prefill_prefix",
        lambda *args, **kwargs: ("prefix-cache", 1.0),
    )
    monkeypatch.setattr(
        runner,
        "prefill_query_tail_mode",
        lambda tail_tokens: (
            events.append(("query_tail_tokens", tail_tokens))
            or nullcontext(captured)
        ),
    )
    monkeypatch.setattr(
        runner,
        "seed_packed_qmse_prefill_queries",
        lambda records: events.append(("seeded", records)),
    )
    monkeypatch.setattr(
        runner.lb,
        "run_token_segment",
        lambda *args, **kwargs: ("prompt-cache", logits_for(3), 0.2),
    )

    result = runner.generate_global_partition(
        model=object(),
        tokenizer=FakeTokenizer(),
        input_device=torch.device("cpu"),
        bundle=bundle,
        max_new_tokens=1,
        prefill_chunk_tokens=2048,
        budget_fractions=(0.06,),
        args=args,
        score_mode=score_mode,
        candidate_fraction=0.06,
        projection_dim=128,
        dense_suffix=True,
        query_calibration_tokens=16,
        prefill_query_record_sink=exported,
        prefill_query_record_tokens=32,
    )

    assert events[0] == ("query_tail_tokens", 32)
    seeded = events[1][1]
    assert tuple(seeded) == (3,)
    assert torch.equal(seeded[3], captured[3][..., -16:, :])
    assert torch.equal(exported[3], captured[3])
    assert exported[3].device.type == "cpu"
    assert result["generated_ids"] == [3]


def test_countcap_fullprompt_uses_frozen_countcap_configuration():
    args = SimpleNamespace(candidate_fraction=0.08, projection_dim=64)
    config = runner.sparse_method_config(
        "countcap_fullprompt", 7500, (0.01,), args
    )
    assert config == runner.countcap_config(7500)


def test_keypca_variant_avoids_qk_metric_rebuild():
    args = SimpleNamespace(candidate_fraction=0.08, projection_dim=64)
    config = runner.sparse_method_config(
        "countcap_fullprompt_keypca", 7500, (0.01,), args
    )
    assert config["score_mode"] == runner.COUNTCAP_KEYPCA_SCORE_MODE
    assert "qkmetric" not in config["score_mode"]


def test_keypca_direct_variant_uses_candidate_fraction_as_attention_budget():
    args = SimpleNamespace(candidate_fraction=0.08, projection_dim=64)
    config = runner.sparse_method_config(
        "countcap_fullprompt_keypca_direct", 7500, (0.01,), args
    )
    assert config["score_mode"] == runner.COUNTCAP_KEYPCA_DIRECT_SCORE_MODE
    assert config["candidate_fraction"] == 0.06
    assert config["budget_fractions"] == (0.06,)
    assert config["attention_tokens"] == 450


def test_keypca_direct_fused_variant_preserves_direct_budget():
    args = SimpleNamespace(candidate_fraction=0.08, projection_dim=64)
    config = runner.sparse_method_config(
        "countcap_fullprompt_keypca_direct_fused", 7500, (0.01,), args
    )
    assert config["score_mode"] == runner.COUNTCAP_KEYPCA_DIRECT_FUSED_SCORE_MODE
    assert config["candidate_fraction"] == 0.06
    assert config["budget_fractions"] == (0.06,)
    assert config["attention_tokens"] == 450


def test_qkbalanced_fixed4421_variant_uses_the_same_direct_budget():
    args = SimpleNamespace(candidate_fraction=0.08, projection_dim=64)
    config = runner.sparse_method_config(
        runner.COUNTCAP_QK_BALANCED_FIXED4421_PACKED_METHOD,
        7500,
        (0.01,),
        args,
    )
    assert (
        config["score_mode"]
        == runner.COUNTCAP_QK_BALANCED_FIXED4421_PACKED_SCORE_MODE
    )
    assert config["candidate_fraction"] == 0.06
    assert config["budget_fractions"] == (0.06,)
    assert config["attention_tokens"] == 450


@pytest.mark.parametrize(
    ("method", "score_mode"),
    (
        (
            runner.QKSIEVE_FIXED410_FULLTOPK_METHOD,
            runner.QKSIEVE_FIXED410_FULLTOPK_SCORE_MODE,
        ),
        (
            runner.QKSIEVE_FIXED410_PRERERANK_L00TO08_FULLTOPK_METHOD,
            runner.QKSIEVE_FIXED410_PRERERANK_L00TO08_FULLTOPK_SCORE_MODE,
        ),
    ),
)
def test_fixed410_prerope_probe_actions_are_strictly_budget_matched(
    method,
    score_mode,
):
    args = SimpleNamespace(
        candidate_fraction=0.08,
        projection_dim=64,
        sampled_quantile_sample_count=256,
    )

    assert runner.parse_methods(method) == [method]
    config = runner.sparse_method_config(method, 32_000, (0.01,), args)

    assert config["score_mode"] == score_mode
    assert config["candidate_fraction"] == 0.04
    assert config["budget_fractions"] == (0.04,)
    assert config["attention_tokens"] == 1_280
    assert runner.configured_index_bits_per_token(score_mode) == 112.0


def test_packed_fier_variant_is_selectable_with_the_same_direct_budget():
    args = SimpleNamespace(candidate_fraction=0.08, projection_dim=64)
    method = runner.FIER_RTN1_G32_PACKED_FULLTOPK_METHOD

    assert runner.parse_methods(method) == [method]
    config = runner.sparse_method_config(
        method,
        7500,
        (0.01,),
        args,
    )

    assert (
        config["score_mode"]
        == runner.FIER_RTN1_G32_PACKED_FULLTOPK_SCORE_MODE
    )
    assert config["candidate_fraction"] == 0.06
    assert config["budget_fractions"] == (0.06,)
    assert config["attention_tokens"] == 450


@pytest.mark.parametrize(
    ("method", "score_mode"),
    (
        (
            runner.QKSIEVE_KEYPCA_UNIFORM1_FULLTOPK_METHOD,
            runner.QKSIEVE_KEYPCA_UNIFORM1_FULLTOPK_SCORE_MODE,
        ),
        (
            runner.QKSIEVE_QKBALANCED_UNIFORM1_FULLTOPK_METHOD,
            runner.QKSIEVE_QKBALANCED_UNIFORM1_FULLTOPK_SCORE_MODE,
        ),
        (
            runner.QKSIEVE_RANDOM_UNIFORM1_FULLTOPK_METHOD,
            runner.QKSIEVE_RANDOM_UNIFORM1_FULLTOPK_SCORE_MODE,
        ),
    ),
)
def test_uniform1_ablation_variants_preserve_the_frozen_token_budget(
    method,
    score_mode,
):
    args = SimpleNamespace(candidate_fraction=0.08, projection_dim=64)

    assert runner.parse_methods(method) == [method]
    config = runner.sparse_method_config(
        method,
        7500,
        (0.01,),
        args,
    )

    assert config["score_mode"] == score_mode
    assert config["candidate_fraction"] == 0.06
    assert config["budget_fractions"] == (0.06,)
    assert config["attention_tokens"] == 450


def test_qfused_fulltopk_is_a_separate_experimental_method() -> None:
    args = SimpleNamespace(candidate_fraction=0.08, projection_dim=64)
    method = runner.QKSIEVE_QFUSED_FULLTOPK_METHOD

    assert method != runner.QKSIEVE_FULLTOPK_METHOD
    assert runner.parse_methods(method) == [method]
    config = runner.sparse_method_config(method, 7500, (0.01,), args)

    assert config["score_mode"] == runner.QKSIEVE_QFUSED_FULLTOPK_SCORE_MODE
    assert config["score_mode"] != runner.QKSIEVE_FULLTOPK_SCORE_MODE
    assert config["candidate_fraction"] == 0.06
    assert config["attention_tokens"] == 450


@pytest.mark.parametrize(
    ("method", "score_mode"),
    (
        (
            runner.QKSIEVE_KEYPCA_AUTOKEY_FULLTOPK_METHOD,
            runner.QKSIEVE_KEYPCA_AUTOKEY_FULLTOPK_SCORE_MODE,
        ),
        (
            runner.QKSIEVE_QKBALANCED_AUTOKEY_FULLTOPK_METHOD,
            runner.QKSIEVE_QKBALANCED_AUTOKEY_FULLTOPK_SCORE_MODE,
        ),
    ),
)
def test_without_query_covariance_ablation_preserves_frozen_token_budget(
    method,
    score_mode,
):
    args = SimpleNamespace(candidate_fraction=0.08, projection_dim=64)

    assert runner.parse_methods(method) == [method]
    config = runner.sparse_method_config(
        method,
        7500,
        (0.01,),
        args,
    )

    assert config["score_mode"] == score_mode
    assert config["candidate_fraction"] == 0.06
    assert config["budget_fractions"] == (0.06,)
    assert config["attention_tokens"] == 450


def test_prefill_query_capture_is_enabled_only_when_the_mode_uses_queries():
    assert (
        countcap_ppl.PACKED_QMSE_QKMETRIC_FULLTOPK_SCORE_MODE
        in countcap_ppl.PACKED_PREFILL_QUERY_SCORE_MODES
    )
    assert (
        countcap_ppl.PACKED_QMSE_QKMETRIC_QFUSED_FULLTOPK_SCORE_MODE
        in countcap_ppl.PACKED_PREFILL_QUERY_SCORE_MODES
    )
    assert (
        countcap_ppl.PACKED_QKBALANCED_AUTOKEY_FULLTOPK_SCORE_MODE
        in countcap_ppl.PACKED_PREFILL_QUERY_SCORE_MODES
    )
    assert (
        countcap_ppl.PACKED_RANDOM_UNIFORM1_FULLTOPK_SCORE_MODE
        not in countcap_ppl.PACKED_PREFILL_QUERY_SCORE_MODES
    )
    assert (
        countcap_ppl.PACKED_KEYPCA_AUTOKEY_FULLTOPK_SCORE_MODE
        not in countcap_ppl.PACKED_PREFILL_QUERY_SCORE_MODES
    )


@pytest.mark.parametrize(
    ("method", "score_mode"),
    (
        (
            runner.COUNTCAP_QK_BALANCED_QSCALE_PACKED_METHOD,
            runner.COUNTCAP_QK_BALANCED_QSCALE_PACKED_SCORE_MODE,
        ),
        (
            runner.COUNTCAP_QK_BALANCED_FIXED4421_QSCALE_PACKED_METHOD,
            runner.COUNTCAP_QK_BALANCED_FIXED4421_QSCALE_PACKED_SCORE_MODE,
        ),
        (
            runner.COUNTCAP_QK_BALANCED_QSCALE_OAS_PACKED_METHOD,
            runner.COUNTCAP_QK_BALANCED_QSCALE_OAS_PACKED_SCORE_MODE,
        ),
        (
            runner.COUNTCAP_QK_BALANCED_FIXED4421_QSCALE_OAS_PACKED_METHOD,
            runner.COUNTCAP_QK_BALANCED_FIXED4421_QSCALE_OAS_PACKED_SCORE_MODE,
        ),
    ),
)
def test_qkbalanced_qscale_factorial_variants_preserve_direct_budget(
    method,
    score_mode,
):
    args = SimpleNamespace(candidate_fraction=0.08, projection_dim=64)
    config = runner.sparse_method_config(method, 7500, (0.01,), args)
    assert config["score_mode"] == score_mode
    assert config["candidate_fraction"] == 0.06
    assert config["budget_fractions"] == (0.06,)
    assert config["attention_tokens"] == 450


def test_keypca_direct_qkv_fused_variant_preserves_direct_budget():
    args = SimpleNamespace(candidate_fraction=0.08, projection_dim=64)
    config = runner.sparse_method_config(
        "countcap_fullprompt_keypca_direct_qkvfused", 7500, (0.01,), args
    )
    assert (
        config["score_mode"]
        == runner.COUNTCAP_KEYPCA_DIRECT_QKV_FUSED_SCORE_MODE
    )
    assert config["candidate_fraction"] == 0.06
    assert config["budget_fractions"] == (0.06,)
    assert config["attention_tokens"] == 450


def test_keypca_direct_qkv_fused_prefill_index_variant_preserves_direct_budget():
    args = SimpleNamespace(candidate_fraction=0.08, projection_dim=64)
    config = runner.sparse_method_config(
        "countcap_fullprompt_keypca_direct_qkvfused_prefillindex",
        7500,
        (0.01,),
        args,
    )
    assert (
        config["score_mode"]
        == runner.COUNTCAP_KEYPCA_DIRECT_QKV_FUSED_SCORE_MODE
    )
    assert config["candidate_fraction"] == 0.06
    assert config["budget_fractions"] == (0.06,)
    assert config["attention_tokens"] == 450


def test_keypca_direct_qkv_fused_async_prefill_variant_preserves_direct_budget():
    args = SimpleNamespace(candidate_fraction=0.08, projection_dim=64)
    config = runner.sparse_method_config(
        "countcap_fullprompt_keypca_direct_qkvfused_asyncprefillindex",
        7500,
        (0.01,),
        args,
    )
    assert (
        config["score_mode"]
        == runner.COUNTCAP_KEYPCA_DIRECT_QKV_FUSED_SCORE_MODE
    )
    assert config["candidate_fraction"] == 0.06
    assert config["budget_fractions"] == (0.06,)
    assert config["attention_tokens"] == 450


def test_keypca_direct_qkv_fused_dp4a_prefill_variant_preserves_direct_budget():
    args = SimpleNamespace(candidate_fraction=0.08, projection_dim=64)
    config = runner.sparse_method_config(
        "countcap_fullprompt_keypca_direct_qkvfused_dp4a_prefillindex",
        7500,
        (0.01,),
        args,
    )
    assert (
        config["score_mode"]
        == runner.COUNTCAP_KEYPCA_DIRECT_QKV_FUSED_DP4A_SCORE_MODE
    )
    assert config["candidate_fraction"] == 0.06
    assert config["budget_fractions"] == (0.06,)
    assert config["attention_tokens"] == 450


@pytest.mark.parametrize(
    ("suffix", "threshold"),
    (
        ("massgate90", 0.90),
        ("massgate94", 0.94),
        ("massgate95", 0.95),
    ),
)
def test_qprojscan_temporal_mass_gate_variants(suffix, threshold):
    args = SimpleNamespace(candidate_fraction=0.08, projection_dim=64)
    method = (
        "countcap_fullprompt_keypca_direct_qkvfused_qprojscan_"
        f"{suffix}_prefillindex"
    )
    config = runner.sparse_method_config(
        method,
        7500,
        (0.01,),
        args,
    )

    assert (
        config["score_mode"]
        == runner.COUNTCAP_KEYPCA_DIRECT_QKV_FUSED_QPROJSCAN_SCORE_MODE
    )
    assert config["temporal_mass_gate_threshold"] == threshold
    assert config["temporal_mass_gate_gqa_union"] is False
    assert config["candidate_fraction"] == 0.06
    assert config["budget_fractions"] == (0.06,)


@pytest.mark.parametrize(
    ("suffix", "threshold"),
    (("gqamassgate94", 0.94), ("gqamassgate95", 0.95)),
)
def test_qprojscan_temporal_gqa_mass_gate_variants(suffix, threshold):
    args = SimpleNamespace(candidate_fraction=0.08, projection_dim=64)
    method = (
        "countcap_fullprompt_keypca_direct_qkvfused_qprojscan_"
        f"{suffix}_prefillindex"
    )
    config = runner.sparse_method_config(
        method,
        7500,
        (0.01,),
        args,
    )

    assert (
        config["score_mode"]
        == runner.COUNTCAP_KEYPCA_DIRECT_QKV_FUSED_QPROJSCAN_SCORE_MODE
    )
    assert config["temporal_mass_gate_threshold"] == threshold
    assert config["temporal_mass_gate_gqa_union"] is True


def test_keypca_direct_qkv_split4_prefill_variant_preserves_direct_budget():
    args = SimpleNamespace(candidate_fraction=0.08, projection_dim=64)
    config = runner.sparse_method_config(
        "countcap_fullprompt_keypca_direct_qkvsplit4_prefillindex",
        7500,
        (0.01,),
        args,
    )
    assert (
        config["score_mode"]
        == runner.COUNTCAP_KEYPCA_DIRECT_QKV_SPLIT4_SCORE_MODE
    )
    assert config["candidate_fraction"] == 0.06
    assert config["budget_fractions"] == (0.06,)
    assert config["attention_tokens"] == 450


def test_keypca_direct_proxy_av_prefill_variant_preserves_direct_budget():
    args = SimpleNamespace(candidate_fraction=0.08, projection_dim=64)
    config = runner.sparse_method_config(
        "countcap_fullprompt_keypca_direct_proxyav_prefillindex",
        7500,
        (0.01,),
        args,
    )
    assert (
        config["score_mode"]
        == runner.COUNTCAP_KEYPCA_DIRECT_PROXY_AV_SCORE_MODE
    )
    assert config["candidate_fraction"] == 0.06
    assert config["budget_fractions"] == (0.06,)
    assert config["attention_tokens"] == 450


@pytest.mark.parametrize("interval", (2, 4, 8))
def test_keypca_periodic_reuse_variants_preserve_direct_budget(interval):
    args = SimpleNamespace(candidate_fraction=0.08, projection_dim=64)
    config = runner.sparse_method_config(
        (
            "countcap_fullprompt_keypca_direct_qkvfused_"
            f"reuse{interval}_prefillindex"
        ),
        7500,
        (0.01,),
        args,
    )
    assert (
        config["score_mode"]
        == runner.COUNTCAP_KEYPCA_DIRECT_QKV_FUSED_REUSE_SCORE_MODES[
            interval
        ]
    )
    assert config["candidate_fraction"] == 0.06
    assert config["budget_fractions"] == (0.06,)
    assert config["attention_tokens"] == 450


@pytest.mark.parametrize("interval", (2, 4, 8))
def test_qprojscan_periodic_reuse_variants_preserve_current_path(interval):
    args = SimpleNamespace(candidate_fraction=0.08, projection_dim=64)
    method = (
        "countcap_fullprompt_keypca_direct_qkvfused_qprojscan_"
        f"qkvsplitauto_cacheauto_reuse{interval}_prefillindex"
    )
    config = runner.sparse_method_config(
        method,
        7500,
        (0.01,),
        args,
    )

    assert config["score_mode"].endswith(
        f"qprojscan_qkvsplitauto_reuse{interval}"
    )
    assert config["candidate_fraction"] == 0.06
    assert config["budget_fractions"] == (0.06,)
    assert config["attention_tokens"] == 450


def test_keypca_direct_qkv_warp_variant_preserves_direct_budget():
    args = SimpleNamespace(candidate_fraction=0.08, projection_dim=64)
    config = runner.sparse_method_config(
        "countcap_fullprompt_keypca_direct_qkvwarp", 7500, (0.01,), args
    )
    assert (
        config["score_mode"]
        == runner.COUNTCAP_KEYPCA_DIRECT_QKV_WARP_SCORE_MODE
    )
    assert config["candidate_fraction"] == 0.06
    assert config["budget_fractions"] == (0.06,)
    assert config["attention_tokens"] == 450


def test_direct_candidate_attention_appends_self_at_each_ragged_boundary():
    indices = torch.tensor([[[4, 7, 0], [3, 5, 8]]], dtype=torch.long)
    scores = torch.tensor([[[0.4, 0.7, -torch.inf], [0.3, 0.5, 0.8]]])
    counts = torch.tensor([[2, 3]], dtype=torch.long)
    self_indices = torch.tensor([[[9], [9]]], dtype=torch.long)
    self_scores = torch.tensor([[[0.9], [1.0]]])

    packed_indices, packed_scores, selected_counts = (
        sparse_attention._append_self_to_ragged_candidates(
            indices, scores, counts, self_indices, self_scores
        )
    )

    assert selected_counts.tolist() == [[3, 4]]
    assert packed_indices[0, 0, :3].tolist() == [4, 7, 9]
    assert packed_indices[0, 1, :4].tolist() == [3, 5, 8, 9]
    assert packed_scores[0, 0, :3].tolist() == [
        pytest.approx(0.4),
        pytest.approx(0.7),
        pytest.approx(0.9),
    ]


def test_periodic_candidate_reuse_appends_new_tokens_at_ragged_boundaries():
    state = {
        "periodic_candidate_step": 1,
        "periodic_candidate_indices": torch.tensor(
            [[[4, 7, 0], [3, 5, 8]]],
            dtype=torch.long,
        ),
        "periodic_candidate_counts": torch.tensor(
            [[2, 3]],
            dtype=torch.long,
        ),
        "periodic_candidate_history_count": 9,
    }

    result = sparse_attention._periodic_reuse_candidates(
        state,
        history_count=11,
        interval=4,
    )

    assert result is not None
    indices, counts = result
    assert counts.tolist() == [[4, 5]]
    assert indices[0, 0, :4].tolist() == [4, 7, 9, 10]
    assert indices[0, 1, :5].tolist() == [3, 5, 8, 9, 10]
    assert state["periodic_candidate_step"] == 2


def temporal_trace_state(current_query_code: int) -> dict:
    return {
        "packed_chunked": torch.zeros(
            (1, 1, 1, 8, 8),
            dtype=torch.uint8,
        ),
        "scales": torch.ones((1, 1, 8, 1), dtype=torch.float16),
        "logscale_exponents": torch.zeros(
            (1, 1, 8, 1),
            dtype=torch.uint8,
        ),
        "last_projected_query_codes": torch.full(
            (1, 1, 2, 16),
            current_query_code,
            dtype=torch.int8,
        ),
        "last_projected_query_scale": torch.ones(
            (1, 1, 2, 1),
            dtype=torch.float32,
        ),
        "temporal_trace_candidate_indices": torch.tensor(
            [[[0, 1], [0, 1]]],
            dtype=torch.long,
        ),
        "temporal_trace_candidate_counts": torch.tensor(
            [[2, 2]],
            dtype=torch.long,
        ),
        "temporal_trace_effective_query": torch.zeros(
            (1, 2, 16),
            dtype=torch.float32,
        ),
        "temporal_trace_boundary_margin": torch.ones(
            (1, 2),
            dtype=torch.float32,
        ),
        "temporal_trace_candidate_overflow": torch.zeros(
            (1, 2),
            dtype=torch.bool,
        ),
        "temporal_trace_history_count": 4,
    }


def test_certified_temporal_trace_accepts_zero_query_delta():
    state = temporal_trace_state(0)
    metrics = sparse_attention._certified_temporal_trace_metrics(
        state,
        torch.tensor([[[0, 1], [0, 1]]], dtype=torch.long),
        torch.tensor([[2, 2]], dtype=torch.long),
        torch.tensor([[[10.0, 9.0], [10.0, 9.0]]]),
        torch.tensor([[8.0, 8.0]]),
        torch.zeros((1, 2), dtype=torch.bool),
        history_count=5,
    )

    assert metrics["temporal_candidate_jaccard"].tolist() == [[1.0, 1.0]]
    assert metrics["temporal_key_norm_bound"].tolist() == [[28.0, 28.0]]
    assert metrics["temporal_certificate_safe"].tolist() == [[1.0, 1.0]]
    assert metrics["temporal_certificate_layer_safe"].tolist() == [[1.0]]


def test_certified_temporal_trace_rejects_large_query_delta():
    state = temporal_trace_state(1)
    metrics = sparse_attention._certified_temporal_trace_metrics(
        state,
        torch.tensor([[[0, 1], [0, 1]]], dtype=torch.long),
        torch.tensor([[2, 2]], dtype=torch.long),
        torch.tensor([[[10.0, 9.0], [10.0, 9.0]]]),
        torch.tensor([[8.0, 8.0]]),
        torch.zeros((1, 2), dtype=torch.bool),
        history_count=5,
    )

    assert metrics["temporal_score_change_bound"].min().item() > 100.0
    assert metrics["temporal_certificate_safe"].tolist() == [[0.0, 0.0]]
    assert metrics["temporal_certificate_layer_safe"].tolist() == [[0.0]]


def test_gqa_union_candidates_deduplicates_and_replicates():
    indices = torch.tensor(
        [[[0, 1, 0], [1, 2, 0], [2, 3, 0], [3, 4, 0]]],
        dtype=torch.long,
    )
    counts = torch.full((1, 4), 2, dtype=torch.long)
    union_indices, union_counts = sparse_attention._gqa_union_candidates(
        indices,
        counts,
        kv_head_count=1,
        history_count=6,
    )

    assert union_counts.tolist() == [[5, 5, 5, 5]]
    for row in union_indices[0]:
        assert set(row.tolist()) == {0, 1, 2, 3, 4}


def test_quality_diagnostic_method_configurations():
    args = SimpleNamespace(candidate_fraction=0.08, projection_dim=64)
    exact = runner.sparse_method_config(
        "exact_top2_fullprompt", 7500, (0.01,), args
    )
    exact_adaptive = runner.sparse_method_config(
        "exact_massadaptive_fullprompt", 7500, (0.01,), args
    )
    approximate_adaptive = runner.sparse_method_config(
        "countcap_massadaptive_fullprompt", 7500, (0.01,), args
    )
    gqa_union_reuse = runner.sparse_method_config(
        "countcap_fullprompt_keypca_direct_qkvfused_"
        "gqaunion_reuse2_prefillindex",
        7500,
        (0.01,),
        args,
    )

    assert exact["budget_fractions"] == (0.02,)
    assert exact["score_mode"] == "exact_qk_top2"
    assert gqa_union_reuse["score_mode"].endswith(
        "qkvfused_gqaunion_reuse2"
    )
    assert exact_adaptive["budget_fractions"] == (0.02, 0.03, 0.04, 0.06)
    assert exact_adaptive["projection_dim"] == 0
    assert approximate_adaptive["budget_fractions"] == exact_adaptive["budget_fractions"]
    assert approximate_adaptive["candidate_fraction"] == 0.06
    assert approximate_adaptive["score_mode"] == runner.COUNTCAP_ADAPTIVE_SCORE_MODE
    assert "sampleq" not in approximate_adaptive["score_mode"]


def test_exact_qk_diagnostic_accepts_explicit_fraction_override():
    args = SimpleNamespace(
        candidate_fraction=0.08,
        projection_dim=64,
        countcap_direct_fraction_override=0.12,
    )
    exact = runner.sparse_method_config(
        "exact_top2_fullprompt", 7500, (0.01,), args
    )

    assert exact["attention_tokens"] == 900
    assert exact["budget_fractions"] == (0.12,)
    assert exact["candidate_fraction"] == 1.0
    assert exact["projection_dim"] == 0


def test_exact_top2_reports_configured_link_ratio_with_diagnostics(monkeypatch):
    bundle = SimpleNamespace(
        input_ids=torch.tensor([[10, 11, 12, 13]], dtype=torch.long),
        query_start=2,
    )
    args = SimpleNamespace(mass_threshold=0.95, collect_attention_stats=True)
    monkeypatch.setattr(runner, "set_attention_implementation", lambda *args: None)
    monkeypatch.setattr(runner, "head_top_fraction_mode", lambda *args: nullcontext())
    monkeypatch.setattr(runner, "head_adaptive_mass_mode", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(runner, "head_qabs_sampled_mass_mode", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(runner, "synchronize_cuda_devices", lambda: None)
    monkeypatch.setattr(runner.lb, "prefill_prefix", lambda *args, **kwargs: ("prefix", 1.0))
    monkeypatch.setattr(
        runner.lb,
        "run_token_segment",
        lambda *args, **kwargs: ("prompt", logits_for(3), 0.2),
    )
    monkeypatch.setattr(
        runner,
        "run_one_token",
        lambda *args, **kwargs: (
            "decode",
            logits_for(2),
            0.1,
            {"selected_history_links": 0.0, "possible_history_links": 1.0},
        ),
    )

    result = runner.generate_exact_sparse(
        model=object(),
        tokenizer=FakeTokenizer(),
        input_device=torch.device("cpu"),
        bundle=bundle,
        max_new_tokens=4,
        prefill_chunk_tokens=2048,
        budget_fractions=(0.02,),
        args=args,
        adaptive_mass=False,
    )

    assert result["attention_link_ratio"] == 0.02


def test_auto_plan_routes_short_context_to_full_kv():
    args = SimpleNamespace(
        candidate_fraction=0.08,
        projection_dim=64,
        cost_quality_floor=0.95,
        cost_speed_margin=1.03,
    )
    model = {"fixed_seconds": 0.0, "step_seconds": 0.04}
    slower = {"fixed_seconds": 1.0, "step_seconds": 0.06}
    profile = {
        "lengths": [
            {
                "mean_prompt_tokens": 7500,
                "full_decode_cost_model": model,
                "methods": {
                    "countcap_fullprompt": {
                        "quality_retention": 0.98,
                        "decode_cost_model": slower,
                    },
                    "countcap_fullprompt_keypca": {
                        "quality_retention": 0.98,
                        "decode_cost_model": slower,
                    },
                },
            }
        ]
    }
    executed, config, decision = runner.resolve_method_plan(
        "countcap_auto",
        history_tokens=7500,
        expected_generated_tokens=128,
        budget_fractions=(0.02,),
        args=args,
        cost_profile=profile,
    )
    assert executed == "full_kv"
    assert config["budget_fractions"] == (1.0,)
    assert decision is not None and decision["selected_path"] == "full_kv"


def test_append_csv_row_preserves_existing_resume_header(tmp_path):
    path = tmp_path / "sample_results.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task", "method"])
        writer.writeheader()
        writer.writerow({"task": "qasper", "method": "full_kv"})

    runner.append_csv_row(
        path,
        {
            "task": "qasper",
            "method": "countcap",
            "executed_path": "countcap",
        },
    )

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert reader.fieldnames == ["task", "method"]
    assert rows == [
        {"task": "qasper", "method": "full_kv"},
        {"task": "qasper", "method": "countcap"},
    ]
