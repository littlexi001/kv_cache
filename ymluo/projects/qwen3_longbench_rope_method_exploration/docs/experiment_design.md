# Experiment design

## Frozen data and execution

- Task: LongBench v1 HotpotQA.
- Cohort: all 18 strictly current-passage-aligned examples frozen in
  `qwen3_longbench_oracle_evidence/outputs/hotpot_semantic_aligned_18_20260802`.
- Model: Qwen3-8B, BF16, SDPA, thinking disabled.
- Hardware: server physical GPUs 6 and 7 only; nine examples per shard.
- Prompt: byte-identical Full-context chat prompt from the Oracle experiment.
- Budget: 2% per layer and Query head; local window 128; sink 16.
- Decoding: greedy, at most 32 new tokens.

## Stages

1. **Preflight:** reconstruct every prompt from LongBench, verify its SHA-256
   against the frozen Full prediction, and align every accepted current support
   span to prompt token offsets. Any mismatch stops the sample.
2. **Prefill:** run the common prompt prefix once and cache native post-RoPE K/V.
3. **First answer query:** replay the last prompt token under every arm; record
   support, evidence recall, evidence attention mass, logits, and query time.
4. **Generation:** greedily decode with the same arm active at every new token;
   score official QA-F1 and normalized EM.
5. **Gold likelihood:** reset to the common prefix, teacher-force the first gold
   surface with the same arm, and record mean token NLL/PPL.
6. **Aggregation:** pair by sample, bootstrap over examples, audit both shards,
   and inspect all LS-vs-post2 rescue/harm cases.

## Smoke then expansion

The smoke uses one example on GPU 7 and only `native_full`,
`full_rope_replay`, `rope_top2`, and `local_global_postscore`. It passes only if
prompt/evidence audits pass, dense replay agrees with native Full, all outputs
are finite, and each sparse query has the expected support size. After that,
the two nine-example shards run all frozen arms.

## Primary and secondary metrics

Primary: paired gold-answer mean token NLL, LS minus exact post-RoPE Top-2%.

Secondary: QA-F1, EM, first-answer-token correctness, gold evidence token
recall, both-support-span hit rate, evidence attention mass, query/generation
time, and rescue/harm counts. Full attention is a reference rather than the
matched-budget baseline.

## Pass/fail/insufficient rules

- **Pass:** LS has lower paired NLL with a wholly negative 95% bootstrap CI,
  plus non-decreasing QA-F1 or EM and higher evidence recall on average.
- **Fail:** CI is wholly positive, or fidelity audits fail.
- **Insufficient:** CI crosses zero; apparent aggregate gain is caused by one
  example; shard directions disagree; or evidence recall rises but answer
  metrics do not.

Every output row must include the frozen sample ID, prompt hash, exact variant,
token budget, aligned evidence spans, prediction, official scores, NLL, recall,
mass, and timing. Logs, configs, rows, summaries, and plots remain under one
versioned output directory.

## Iteration-2 screen after a failed retrieval explanation

If the frozen LS arm improves QA-F1 without improving gold-evidence recall or
gold NLL, run a first-token-only screen on the same 18 prompts with: untouched
Full, final-Query pre-RoPE selection, question-span token-max pre-RoPE
selection, and question-span block selection. The question anchors are selected
deterministically from the literal question substring; evidence labels remain
measurement-only. Advance to full generation only if a question-span arm
improves both evidence recall and paired first-gold-token NLL.
