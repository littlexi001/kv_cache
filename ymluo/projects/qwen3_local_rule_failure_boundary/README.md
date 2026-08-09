# Qwen3 Local Rule Failure Boundary

This project tests subquestion 3:

```text
Why does inference fail when each local dependency is simple?
```

The workload keeps every rule local:

```text
VERIFIED RULE: IF AX12 IS ACTIVE THEN BY34 BECOMES ACTIVE.
```

The experiment varies context length, distractor count, distractor similarity,
distance between relevant rules, relevant-rule position, chain length, and
competing chains. It records:

- candidate logprob accuracy and margin;
- greedy generation class;
- attention mass on relevant rule spans vs distractor spans;
- failure-boundary summaries.

## Documents

- `doc/experiment_design_20260709.md`: experiment design.
- `doc/results_summary_0p6b_20260709.md`: Qwen3-0.6B result summary and interpretation.
- `doc/four_condition_answer_results_20260717.md`: paired conflict × filler accuracy and gold-answer PPL experiment.
- `doc/long_context_retrieval_degradation_concise_summary_20260719.md`: concise Qwen3-8B RoPE, softmax-competition, and PPL mechanism synthesis.
- `doc/why_long_context_hurts_needle_retrieval_softmax_rope_20260720.md`: one-page Chinese explanation of why longer contexts hurt needle retrieval, separated into Softmax and RoPE mechanisms.

## Smoke

```bash
bash scripts/run_question3_boundary_smoke_server.sh
```

## Main Qwen3-0.6B sweep

```bash
bash scripts/run_question3_boundary_phase1_qwen06_server.sh
```

## Optional Qwen3-8B comparison

```bash
bash scripts/run_question3_boundary_qwen8b_compare_server.sh
```

## Per-head gold-evidence attention study

This paired study asks which layer/head combinations attend to the two gold
`VERIFIED RULE` spans at the answer query.  Each conflict prompt is paired with
an equal-token-length nonconflict prompt.  The paired prompt keeps every rule
position, wrong consequent, candidate answer, and filler token fixed; only each
`DECOY RULE` antecedent is changed so that it no longer matches the gold chain.

The runner records gold/decoy/competitor attention mass, span-length-corrected
enrichment, gold-rule selectivity, per-step attention, and gold-token coverage
inside each head's top 2% logits.  It also scores the answer candidates.

```bash
CUDA_VISIBLE_DEVICES=0 \
  bash scripts/run_head_evidence_attention_server.sh
```

Raw and summarized outputs include `head_attention.csv`,
`head_event_attention.csv`, `paired_conflict_effect_by_head.csv`, `top_heads.csv`,
and layer-by-head heatmaps.

## Qwen3-8B clean two-hop confidence/attention sweep

This single-case diagnostic keeps the same two verified rules and seed while
scanning the context body from 0 to 64K in 500-token increments. At the final
two-hop answer query it records the true post-softmax distribution for all 36
layers and 32 query heads. The first-hop result, its repeated second-hop input,
and the final second-hop result use separate exact token spans.

```bash
bash scripts/run_attention_confidence_qwen3_8b_server.sh
```

The eight-GPU launcher is resumable: already completed length files are not
recomputed. It produces raw JSON, correlation/head-trend tables, a Markdown
analysis, and a gzip browser bundle. The interactive frontend is under
`attention_confidence_dashboard/`.

### Single-token evidence-code control

The matched control replaces each multi-piece `GA89-987` identifier with an
opaque Han character that Qwen3-8B maps to exactly one stable tokenizer token
in the rule and query templates. Seed, clean two-hop chain, filler placement,
length grid, attention collector, and model remain unchanged.

```bash
bash scripts/run_attention_confidence_qwen3_8b_single_token_server.sh
```

The launcher uses eight one-GPU workers through 49.5K and four exact two-GPU
workers for 50K-64K. The dashboard can switch between the legacy multi-token
condition and this single-token control.

### Ordinary-English single-token 128K mechanism study

The 128K control uses the fixed chain `river → window → basket`. Each word is
one tokenizer token in every rule/query occurrence and is excluded from the
neutral filler corpus. The main sweep evaluates one nested filler sequence at
500-token intervals from 0 through 128K with the evidence chain in the middle.

```bash
bash scripts/run_attention_confidence_qwen3_8b_english_128k_server.sh
```

In addition to post-softmax attention mass, every layer/head records the target
evidence's pre-softmax QK logit, Q/K cosine, key rank, query/key norms, the
strongest competing logit, and log-sum-exp. This separates three mechanisms:

- a growing softmax denominator and irrelevant-token extreme values;
- degradation of the target Q/K match or evidence rank;
- downstream value transport or two-hop composition failure despite retrieval.

After the dense sweep, the watcher runs causal controls automatically:

```bash
bash scripts/watch_english_main_then_probes_20260718.sh
```

The controls move evidence to the prefix or near the query, split the two-hop
task into hop-1 and oracle hop-2 lookups, and compare native RoPE factor 1 with
YaRN factors 2 and 4. Exact-rule cloze probes distinguish whether the local
`river → window` / `window → basket` association is unavailable or merely not
reliably invoked by the ordinary QA prompt. Outputs include `analysis_summary.csv`,
`length_bin_summary.csv`, `retrieval_head_retention.csv`, `head_trends.csv`,
`analysis_report.md`, and a combined intervention report.

The completed interpretation is in
`doc/english_single_token_128k_length_failure_mechanism_20260718.md`; compact
tables are mirrored under
`artifacts/20260718_attention_confidence_qwen3_8b_english_single_token/`.

## Local dry run

```bash
python src/run_local_rule_failure_boundary.py \
  --output_dir outputs/dry_run \
  --lengths 512 \
  --depths 50 \
  --seeds 0 \
  --distractor_counts 2 \
  --distractor_similarities high \
  --chain_lengths 2 \
  --competitor_counts 1 \
  --dry_run_cases true
```
