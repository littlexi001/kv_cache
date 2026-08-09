# Design: natural95_synth5_v1

## Research question

When a small Qwen-style language model is pretrained on an ordinary natural-text
distribution, do the four positional-encoding variants differ in long-range
retrieval without losing short-context language modeling quality?

## What is changed

The earlier experiment used a low-entropy synthetic template and applied loss to
only a small subset of positions. The new experiment instead uses:

- 95% ordinary OpenWebText chunks;
- 5% OpenWebText chunks with key-value retrieval records injected;
- standard next-token cross entropy on every token;
- an additional weight on synthetic answer tokens only;
- the same tokenizer, packed token stream, initialization seed, sample order,
  optimizer and evaluation examples for every PE condition.

The model is still initialized from scratch. Completed old checkpoints are kept
untouched and are not used for initialization.

## Compared PE conditions

1. `native`: standard RoPE in every layer and frequency pair.
2. `deep_highfreq_taper`: replace the old deep-layer hard deletion with a
   continuous layer-by-frequency taper whose minimum angular scale is 0.25.
3. `layerwise_slow_rope`: replace uniform 0.5x slowing with a smooth transition
   from nearly native shallow layers to 0.5x deep layers.
4. `complementary_smooth`: keep one GQA branch at native RoPE and smoothly slow
   fast/mid-frequency pairs in the other branch, preserving local and remote
   channels simultaneously.

## Primary measurements

- held-out natural-text NLL and PPL;
- retrieval accuracy, Gold NLL and Gold margin at 512/1K/2K/4K/8K;
- training loss, gradient norm and throughput;
- exact hashes of tokenizer, train token stream and validation token stream.

## Interpretation boundary

Long-range comparisons are considered meaningful only if the native model first
learns the 512-token retrieval task. A low training loss alone is not evidence of
retrieval learning.
