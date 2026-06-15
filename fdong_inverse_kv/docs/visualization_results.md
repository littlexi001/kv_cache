# Current Verification Results

## 1. What Was Tested

The current evidence is a local CPU micro-test, not a DCLM training result. It tests the implementation contract before using remote compute:

1. detached exclusive causal mean does not send gradient into historical router states；
2. changing future tokens does not change earlier logits；
3. `layer_input/q/k/v` router inputs all support forward and backward；
4. NTP-path gradients reach router parameters；
5. checkpoint, runtime config, JSONL metrics, and summary generation complete end to end；
6. the full Qwen3-0.6B configuration remains below the 2B parameter limit。

## 2. Observed Results

For a two-layer, four-head debug model:

| Check | Result |
|---|---|
| output shape | passed |
| causal prefix invariance | passed |
| historical center gradient | exactly zero |
| current-token router gradient | nonzero |
| all four router input choices | passed |
| two-step optimizer/checkpoint loop | passed |
| metrics parser | passed |
| Qwen tokenizer + streamed JSONL/DCLM adapter + one training step | passed |

The random-initialized debug model produced a candidate ratio around `0.64` with local window 4, one sink token, and four buckets. This number is only a mask sanity check; it is not evidence of useful retrieval.

The following table records the old concatenated-head implementation:

| Expert width | Total parameters |
|---:|---:|
| 256 | 0.420B |
| 1024 | 0.684B |
| 3072 | 1.389B |

These downloaded runs used the old `64 -> 3072 -> 64` expert and are historical evidence only. The current implementation uses `64 -> 512 -> 1024`, sums the 16 head outputs with `1/sqrt(16)` scaling, and still has about `1.389B` total parameters. New training results have not yet been collected, so the research state for the revised structure is incomplete.

Current equal-budget parameter check:

| Current architecture | Total parameters |
|---|---:|
| ordinary top-1 MoE, `1024 -> 3072 -> 1024` | 1.388888B |
| shared full-output head MoE, `64 -> 512 -> 1024` | 1.389003B |

## 3. What This Proves

The code path is internally consistent enough to start the first remote training run. In particular, the router is not disconnected from the loss, and causal centering does not leak gradients into prior positions.

## 4. What This Does Not Prove

The micro-test does not show that:

1. NTP learns semantically useful buckets；
2. routing avoids collapse over long training；
3. shared buckets preserve DCLM validation loss；
4. logical candidate reduction becomes physical KV-memory or latency reduction。

Those claims require the remote DCLM experiments in `experiment_design.md`, followed by a separate bucketed decode implementation.
