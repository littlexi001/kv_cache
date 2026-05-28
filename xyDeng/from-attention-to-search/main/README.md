# from-attention-to-search Main

Project-level research notes for `from-attention-to-search`.

## Read Order

Default project entry is one level up:

```text
../graph.yaml
```

Then read only the linked files needed for the active question:

1. `problem_anchors/` - living small-problem anchors.
2. `roadmap.md` - gated branch route from specialization diagnosis to later effects.
3. `hypotheses/README.md` - legacy hypothesis cards and closures only when reading old evidence.
4. `experiments/README.md` - execution plans, result summaries, and detailed analyses.

## Current Anchor-First Rule

```text
problem_anchors/<anchor_id>.md
../graph.yaml
  -> experiments/<experiment_id>/execution_plan.md
  -> experiments/<experiment_id>/summary.md
  -> experiments/<experiment_id>/detailed.md
  -> update problem_anchors/<anchor_id>.md and ../graph.yaml if needed
  -> copy updated anchor to root sync
  -> write XingyuD/sync/MMDD_topic/report.md when meeting-facing
```

## Current Active Anchor

```text
problem_anchors/gating_induced_expert_specialization.md
```

Current active decision:

```text
Current anchor has been reframed from Reverse KV utility to gating-induced
expert specialization. Next sync should focus on a reused-token diagnostic:
when token identity and slot/context feature conflict, what does gating route
by, and does the routed expert have causal utility for the intended feature?
```

Current cleanup rule:

```text
KV retrieval material is archived as downstream-effect evidence. Do not use it
as the default entry point for the current anchor.
```

## Archived Reporting Pack

```text
hypotheses/archive/router_specialization_and_retrieval_0518/router_specialization_and_retrieval_report_ZH.md
hypotheses/archive/router_specialization_and_retrieval_0518/router_specialization_and_retrieval_report_EN.md
```

## Execution Docs

```text
experiments/archive/E0_E3_synthetic_data_understanding/execution_plan.md
../XingyuD/Attention_Search_Experiments/active/synthetic_data_understanding/README.md
```

## Folder Roles

- `problem_anchors/`: living documents for small research problems. They should
  be updated as experiments clarify physical priors, mathematical modeling,
  computational realization, claim boundaries, and next decisions.
- `hypotheses/`: legacy hypothesis purpose cards and closures; do not add new
  purpose cards here unless explicitly maintaining old evidence.
- `experiments/`: execution plans, result summaries, detailed analyses, and figures; multi-stage lines may use recursive `stages/<stage_id>/` folders.

New hypotheses belong in the active anchor. New closure-style reporting belongs
in `../XingyuD/sync/MMDD_topic/report.md`, not in a new
`hypothesis_closure.md`.

Retired from default workflow:

```text
current_mainline.md
consensus.md
status.md
```

Do not add new project dependencies on these files. Put route structure in
`../graph.yaml`, compact gate logic in `roadmap.md`, and reasoning details in
`problem_anchors/`.

## Reporting Rule

Every report must be self-contained and should name the most important metric
first. Explain what the metric decides before listing supporting metrics.
