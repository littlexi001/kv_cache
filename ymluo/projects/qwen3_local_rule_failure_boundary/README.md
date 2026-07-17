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
