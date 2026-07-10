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
