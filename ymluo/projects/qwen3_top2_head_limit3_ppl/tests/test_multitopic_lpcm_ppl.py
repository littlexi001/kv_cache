from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import LlamaConfig, LlamaForCausalLM

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import run_controlled_public_kv_benchmark_v1 as runner  # noqa: E402
from run_causal_echo_ppl_20260714 import (  # noqa: E402
    build_echo_index,
    evaluate_causal_echo_ppl,
    evaluate_universal_controller_ppl,
    find_echo_match_with_backward_span,
    update_aligned_match_run,
)
import run_multitopic_lpcm_ppl_20260714 as multitopic  # noqa: E402
from run_multitopic_lpcm_ppl_20260714 import (  # noqa: E402
    encode_topic_stream,
    evaluate_target_ppl,
)


class MultitopicLpcmPplTest(unittest.TestCase):
    def test_topic_stream_can_repeat_deterministically_when_opted_in(
        self,
    ) -> None:
        class Tokenizer:
            def __call__(
                self,
                text: str,
                add_special_tokens: bool,
            ) -> dict[str, list[int]]:
                del add_special_tokens
                return {"input_ids": [ord(text[-1])] * 3}

        dataset = SimpleNamespace(data=["a" * 200, "b" * 200])
        original_fetch = multitopic.fetch_20newsgroups
        multitopic.fetch_20newsgroups = lambda **_: dataset
        try:
            with self.assertRaisesRegex(RuntimeError, "only 6 usable"):
                encode_topic_stream(
                    Tokenizer(),
                    "unused",
                    required_tokens=7,
                    cache_dir="unused",
                    seed=17,
                )
            first = encode_topic_stream(
                Tokenizer(),
                "unused",
                required_tokens=7,
                cache_dir="unused",
                seed=17,
                repeat_documents=True,
            )
            second = encode_topic_stream(
                Tokenizer(),
                "unused",
                required_tokens=7,
                cache_dir="unused",
                seed=17,
                repeat_documents=True,
            )
        finally:
            multitopic.fetch_20newsgroups = original_fetch

        self.assertEqual(first, second)
        self.assertEqual(len(first), 7)

    def test_aligned_match_run_rejects_source_jumps(self) -> None:
        run = update_aligned_match_run(None, 100, 0)
        run = update_aligned_match_run(100, 101, run)
        self.assertEqual(run, 2)
        self.assertEqual(update_aligned_match_run(101, 900, run), 1)
        self.assertEqual(update_aligned_match_run(900, None, 1), 0)

    def test_backward_span_selects_the_continuous_source(self) -> None:
        remote = [9, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8]
        index = build_echo_index(remote, 8)
        start, span = find_echo_match_with_backward_span(
            [9, 1, 2, 3, 4, 5, 6, 7, 8], remote, index, 8
        )
        self.assertEqual(start, 1)
        self.assertEqual(span, 9)

    def test_chunked_ppl_matches_tokenwise_sparse_reference(self) -> None:
        torch.manual_seed(17)
        config = LlamaConfig(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=256,
        )
        config._attn_implementation = "sdpa"
        model = LlamaForCausalLM(config).eval()
        runner.install_llama_layerwise_attention_mask_patch()

        remote = torch.randint(0, config.vocab_size, (1, 32))
        bundle = runner.PromptBundle(
            input_ids=remote,
            prefix_token_count=0,
            context_token_start=0,
            query_start=32,
            suffix_token_count=0,
            page_spans={},
        )
        full_cache, _ = runner.prefill_prefix(model, bundle, torch.device("cpu"), 0)
        keep_indices = [0, 1, 4, 9, 15, 23, 31]
        tokenwise_cache = runner.gather_past_key_values(full_cache, keep_indices)
        chunked_cache = runner.gather_past_key_values(full_cache, keep_indices)
        query_ids = torch.randint(0, config.vocab_size, (7,)).tolist()
        target_ids = torch.randint(0, config.vocab_size, (9,)).tolist()

        tokenwise = evaluate_target_ppl(
            model,
            tokenwise_cache,
            query_ids,
            target_ids,
            logical_query_start=32,
            input_device=torch.device("cpu"),
            chunk_tokens=1,
            physical_mask=True,
        )
        chunked = evaluate_target_ppl(
            model,
            chunked_cache,
            query_ids,
            target_ids,
            logical_query_start=32,
            input_device=torch.device("cpu"),
            chunk_tokens=4,
            physical_mask=True,
        )

        self.assertEqual(tokenwise[3], len(target_ids))
        self.assertEqual(chunked[3], len(target_ids))
        self.assertAlmostEqual(tokenwise[0], chunked[0], places=5)

    def test_causal_echo_without_match_equals_static_sparse_ppl(self) -> None:
        torch.manual_seed(23)
        config = LlamaConfig(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=256,
        )
        config._attn_implementation = "sdpa"
        model = LlamaForCausalLM(config).eval()
        runner.install_llama_layerwise_attention_mask_patch()

        remote_ids = list(range(32))
        remote = torch.tensor([remote_ids], dtype=torch.long)
        bundle = runner.PromptBundle(
            input_ids=remote,
            prefix_token_count=0,
            context_token_start=0,
            query_start=32,
            suffix_token_count=0,
            page_spans={},
        )
        full_cache, _ = runner.prefill_prefix(model, bundle, torch.device("cpu"), 0)
        base_keep = [0, 1, 4, 9, 15, 23, 31]
        static_cache = runner.gather_past_key_values(full_cache, base_keep)
        query_ids = list(range(40, 47))
        target_ids = list(range(80, 89))

        static = evaluate_target_ppl(
            model,
            static_cache,
            query_ids,
            target_ids,
            logical_query_start=32,
            input_device=torch.device("cpu"),
            chunk_tokens=4,
            physical_mask=True,
        )
        dynamic = evaluate_causal_echo_ppl(
            model,
            full_cache,
            bundle,
            None,  # type: ignore[arg-type]
            base_keep,
            query_ids,
            target_ids,
            torch.device("cpu"),
            match_tokens=4,
            refresh_tokens=4,
            replay_chunk_tokens=4,
        )
        controlled = evaluate_universal_controller_ppl(
            model,
            full_cache,
            bundle,
            SimpleNamespace(budget_tokens=7, sink_tokens=1, recent_tokens=4),
            base_keep,
            query_ids,
            target_ids,
            torch.device("cpu"),
            match_tokens=4,
            stability_matches=3,
            confirmation_tokens=4,
            replay_chunk_tokens=4,
        )

        self.assertEqual(dynamic[3], len(target_ids))
        self.assertEqual(dynamic[4], [])
        self.assertAlmostEqual(static[0], dynamic[0], places=5)
        self.assertEqual(controlled[3], len(target_ids))
        self.assertEqual(controlled[4], [])
        self.assertAlmostEqual(static[0], controlled[0], places=5)

        rebuild_stats: dict[str, float] = {}
        rebuilding = evaluate_universal_controller_ppl(
            model,
            full_cache,
            bundle,
            SimpleNamespace(budget_tokens=7, sink_tokens=1, recent_tokens=4),
            base_keep,
            [20, 21, 22, 23],
            [24, 25, 26],
            torch.device("cpu"),
            match_tokens=2,
            stability_matches=2,
            confirmation_tokens=2,
            replay_chunk_tokens=2,
            timing_stats=rebuild_stats,
        )
        self.assertEqual(rebuilding[3], 3)
        self.assertGreaterEqual(rebuild_stats["cache_rebuilds"], 1)
        self.assertTrue(any(trace["cache_rebuilt"] for trace in rebuilding[4]))


if __name__ == "__main__":
    unittest.main()
