from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from transformers import LlamaConfig, LlamaForCausalLM

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import run_controlled_public_kv_benchmark_v1 as runner  # noqa: E402


class SparsePhysicalCausalMaskTest(unittest.TestCase):
    def test_parallel_mask_matches_tokenwise_reference(self) -> None:
        torch.manual_seed(7)
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

        prefix = torch.randint(0, config.vocab_size, (1, 32))
        suffix = torch.randint(0, config.vocab_size, (1, 7))
        bundle = runner.PromptBundle(
            input_ids=torch.cat([prefix, suffix], dim=1),
            prefix_token_count=1,
            context_token_start=1,
            query_start=32,
            suffix_token_count=7,
            page_spans={},
        )
        full_cache, _ = runner.prefill_prefix(model, bundle, torch.device("cpu"), 0)
        keep_indices = [0, 1, 4, 9, 15, 23, 31]
        tokenwise_cache = runner.gather_past_key_values(full_cache, keep_indices)
        physical_mask_cache = runner.gather_past_key_values(full_cache, keep_indices)
        automatic_mask_cache = runner.gather_past_key_values(full_cache, keep_indices)

        _, tokenwise_logits, _ = runner.run_token_segment(
            model,
            suffix,
            tokenwise_cache,
            32,
            torch.device("cpu"),
            chunk_tokens=1,
        )
        _, physical_mask_logits, _ = runner.run_token_segment(
            model,
            suffix,
            physical_mask_cache,
            32,
            torch.device("cpu"),
            physical_causal_mask=True,
        )
        _, automatic_mask_logits, _ = runner.run_token_segment(
            model,
            suffix,
            automatic_mask_cache,
            32,
            torch.device("cpu"),
        )

        torch.testing.assert_close(physical_mask_logits, tokenwise_logits, atol=3e-5, rtol=3e-5)
        self.assertGreater(float((automatic_mask_logits - tokenwise_logits).abs().max()), 1e-3)


if __name__ == "__main__":
    unittest.main()
