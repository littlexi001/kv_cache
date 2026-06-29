# Retrieval-Preserving Hybrid KV Trial

Date: 2026-06-29

## Hypothesis

PPL-preserving compression and retrieval-preserving compression are different objectives. The tested direction was:

> Influence-gated hybrid KV compression allocates budget by layer/head/task sensitivity and uses block-aware retention for remote evidence.

## Baseline downstream setup

Task: synthetic long-context key-value retrieval.

- Model: `/home/fdong/hrj/prove/Qwen3-0.6B`
- Tasks: 32
- Records per task: 64
- Context length: about 3.7k tokens
- Labels: A/B/C/D scored by next-token likelihood
- Dense baseline accuracy: `17/32 = 53.1%`
- Original `qabs8cand5reuse`, `top_fraction=0.05`: `14/32 = 43.8%`

The baseline model itself is not strong on this task, but the compression drop is real: 3 fewer correct tasks than dense baseline.

## Tried variants

| Variant | Intended retained KV | Accuracy | Result |
|---|---:|---:|---|
| `qabs8cand5reuse`, `top_fraction=0.05` | about 6% | 14/32 = 43.8% | current reference |
| `qabs8cand5reuse`, `top_fraction=0.08` | about 9% | 14/32 = 43.8% | uniform larger budget did not help |
| `qabs8cand5reuseblk8`, `top_fraction=0.01` | about 8% rough target | 13/32 = 40.6% | block expansion with too few seed tokens hurt |
| `qabs8cand5reuseblk4`, `top_fraction=0.02` | about 8% rough target | 13/32 = 40.6% | smaller block still hurt |
| default qabs5 + layer 13 full | about 9-10% | 12/32 = 37.5% | hand-picked middle full layer hurt |
| default qabs5 + layer 0 full | about 9-10% | 14/32 = 43.8% | no gain over qabs |
| default qabs5 + layers 0/11/14/15 headmix4 | about 9% rough target | 14/32 = 43.8% | no gain over qabs |

## Layer sensitivity scan

I also ran an 8-task single-full-layer scan:

- Baseline: 6/8
- `qabs8cand5reuse`: 7/8
- Many single-full-layer variants: 7/8
- No single-full-layer variant exceeded qabs on this small screen.

This screen shows the first 8 tasks are not representative of the full 32-task set. It is useful only for eliminating clearly bad layer choices, not for claiming improvement.

## Interpretation

The broad PPL result remains promising: about 2.7% average relative PPL increase at roughly 6% retained KV. However, retrieval-preserving behavior is not fixed by the naive variants above.

The negative results are informative:

- Block expansion after final token selection can add softmax noise without recovering missing evidence.
- Increasing the uniform token budget from 5% to 8% did not recover retrieval accuracy.
- Hand-picked layer/head full attention does not reliably identify retrieval-sensitive computation.
- The current failure is likely earlier than final attention: candidate generation often misses the key/value evidence span, or the chosen heads are not the heads that route exact lookup information.

## Next useful experiment

Do not keep manually choosing layers. The next method should measure evidence coverage and derive gates from calibration:

1. Tokenize each synthetic record and save target key/value token spans.
2. During qabs decode, log whether the selected candidate/final masks cover the target key span and answer-label span.
3. Run an oracle variant that forces the target record span to be retained. If oracle span retention restores accuracy, the problem is token/span selection.
4. Use calibration tasks to rank layer/head sensitivity by the accuracy drop caused by compressing that layer/head.
5. Build the real influence-gated hybrid from measured sensitivity, not from fixed layer ids.

The paper direction is still coherent, but the current contribution should be reframed:

> PPL-preserving qabs is strong for language modeling, but retrieval requires calibrated evidence coverage. Influence-gated hybrid KV should allocate its limited full/block budget to heads/layers that demonstrably preserve target evidence spans.

