# Frozen pilot protocol

## Data and sampling

- Task: LongBench v1 `hotpotqa`.
- Evidence source: original HotpotQA distractor validation `supporting_facts`.
- Model: Qwen3-8B, BF16, SDPA, greedy decoding, thinking disabled.
- Pilot size: 8 eligible examples.
- Sample seed: `20260802`.
- Eligibility:
  - unique normalized-question join;
  - all supporting facts resolve to valid document/sentence indices;
  - all supporting sentences occur in the LongBench context;
  - full Qwen3 chat prompt is between 6,000 and 16,384 tokens;
  - at least two supporting documents.
- Sampling is frozen before inference. Evidence positions are split into early,
  middle, and late thirds, and examples are selected round-robin across bins,
  preferring longer eligible contexts within a bin.

## Paired conditions

| Condition | Context supplied to the same reader prompt | Purpose |
|---|---|---|
| `full` | Complete LongBench context | Baseline |
| `oracle_sentence` | Gold supporting sentences only | Tightest annotation-faithful condition |
| `oracle_document` | Full documents containing gold sentences | Main evidence-compression ceiling |
| `random_document_seed{0,1,2}` | Same document count, closest token budget, no support title | Short-context control |
| `bm25_document` | Same document count, selected from question only | Realizable lexical baseline |
| `query_only` | Empty evidence context | Parametric-memory / guessing control |

All selected documents are serialized in their original order. Random document
selection excludes gold-support titles and excludes documents containing the
normalized gold answer, so it cannot accidentally become an answer-bearing
oracle.

## Metrics

Primary:

- official LongBench QA-F1;
- normalized exact match;
- paired QA-F1 gain over `full`;
- rescue rate among samples where `full` exact match is false;
- harm rate among samples where `full` exact match is true.

Auxiliary:

- first-gold-answer mean token NLL and PPL;
- context and prompt token counts;
- compression ratio;
- evidence join and occurrence audit;
- answer-containing rate of each selected context.

Random controls are averaged within sample before paired comparisons. A paired
bootstrap over examples is reported, but with eight samples every confidence
interval is descriptive rather than submission-grade evidence.

## Pilot interpretation

- Oracle above Full and Random: evidence selection plus re-encoding can recover
  some failures.
- Oracle near Random: shortening, rather than evidence identity, explains the
  apparent gain.
- Oracle sentence below Oracle document: local document context is needed in
  addition to annotated sentences.
- Oracle still poor: the remaining bottleneck is reader reasoning, instruction
  following, annotation sufficiency, or generation rather than evidence access.
- Query-only high: parametric memory or contamination weakens a retrieval claim.

## Frozen expansion protocol

This section specifies the larger run before model inference.

### Inputs and fixed parameters

- LongBench HotpotQA: all 200 released rows.
- Original labels: HotpotQA distractor validation `supporting_facts`.
- Alignment mode: exact normalized title followed by fail-closed current-span
  semantic alignment with protected factual anchors.
- Pre-inference target: 20. Strict preflight yielded 18 eligible rows, so all 18
  are frozen without inspecting model outputs.
- Full prompt range: 6,000–16,384 Qwen3 tokens, matching the first pilot and
  avoiding condition-specific truncation on a 24 GB GPU.
- Sample seed: `20260802`.
- Model and decoding: unchanged from the 8-sample pilot.
- Conditions: Full, Oracle current supporting spans, Oracle matched LongBench
  support documents, BM25 documents, three token-budget-matched random
  distractor spans, and Question-only.

### Stage-by-stage algorithm

1. **Question join.** Match a LongBench row uniquely to original HotpotQA by
   normalized question. Reject as `question_join_not_unique` otherwise.
2. **Passage parsing.** Parse every actual LongBench block beginning with
   `Passage N:` into one title and one body. Reject malformed blocks.
3. **Gold-title mapping.** Read every original supporting-fact title. Normalize
   Unicode, case, underscores, and whitespace. Require exactly one LongBench
   passage with each normalized title. Reject missing or duplicate titles; do
   not use the answer to repair the mapping.
4. **Current-span alignment.** In the uniquely matched passage, align every old
   supporting sentence to a one- or two-sentence current span. Pass canonical
   token-subsequence matches directly. Otherwise require sequence similarity,
   bag-token F1, and source recall of at least 0.95, length ratio in
   `[0.75, 1.33]`, best-candidate margin at least 0.05, and exact preservation
   of protected numeric, temporal, negation, and comparison anchors. Reject the
   whole row if any fact fails.
5. **Length gate.** Build the unchanged Qwen3 chat prompt and retain only
   6,000–16,384-token rows.
6. **Evidence position.** Use the mean character center of the matched support
   passages in the actual LongBench context; classify it as early, middle, or
   late thirds.
7. **Frozen sampling.** Form six strata from HotpotQA type
   (`bridge`/`comparison`) × position (`early`/`middle`/`late`). Within each
   stratum, order rows by a seed-dependent SHA-256 key and select round-robin
   across strata until all 18 eligible rows are frozen. No model output enters
   selection.
8. **Random control.** Pool the natural Wikipedia passages from the entire
   frozen cohort, remove the current sample's support titles, and for each of
   three fixed seeds deterministically shuffle this pool and take a token-prefix
   span matching the Oracle-document budget within 5%. A with-replacement cycle
   remains a fail-safe and its count is recorded, but the global pool should
   make it unnecessary. Gold answers are not used for selection; accidental
   answer-string occurrences are recorded for sensitivity analysis.
9. **Inference and scoring.** Run every condition with the same instruction,
   question, checkpoint, precision, and greedy decoding; save prompt hash,
   selected titles, token counts, output, QA-F1, EM, and gold-answer NLL.

### Pass, fail, and insufficient-evidence rules

- Alignment passes only if every gold title maps uniquely and every annotated
  fact passes current-span and protected-anchor checks.
- The run passes integrity only if all 18 samples have every condition exactly
  once, the two GPU manifests have the same hash, and every random replicate is
  within 5% of its Oracle-document token budget.
- The expanded experiment supports a recoverable evidence-access gap if Oracle
  document exceeds both Full and within-sample random mean in QA-F1, with a
  positive paired-bootstrap 95% CI against random.
- It does not support a stable Full-to-Oracle improvement if that paired CI
  includes zero.
- Fewer than 18 valid samples, any condition-specific truncation, a non-unique
  title mapping, a protected-anchor change, a random budget error above 5%, or a
  manifest mismatch stops the run before interpretation.
