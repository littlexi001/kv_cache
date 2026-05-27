# xyDeng Mainline Sync

This folder is a lightweight sync from:

```text
Research_System/Projects/from-attention-to-search
branch: Xingyu
```

Only the project `main/` materials are included: anchors, experiment summaries,
detailed reports, curated figures, and small tables.

The `XingyuD/` working surface and runnable experiment workspace are
intentionally not included here.

## First Read

The most important document is:

```text
from-attention-to-search/main/problem_anchors/gated_main_causes/slot_context_dominance_router_specialization_anchor.md
```

It states the current anchor:

```text
Slot information is visible and useful, but learned top-1 routing does not
reliably bind the slot axis to utility-best experts under many B identities.
```

## Current Main Evidence

Read these reports in order:

```text
from-attention-to-search/main/experiments/slot_context_dominance_router_specialization/summary.md
from-attention-to-search/main/experiments/slot_context_dominance_router_specialization/detailed.md
from-attention-to-search/main/experiments/slot_context_bridge_abcd_context_length/summary.md
from-attention-to-search/main/experiments/slot_context_bridge_abcd_context_length/detailed.md
```

These experiments test whether stronger slot context plus slot-centroid router
initialization can make standard top-1 MoE route by slot and form slot-specific
expert utility.

## Additional Mainline Context

Supporting context copied from `main/`:

```text
from-attention-to-search/main/README.md
from-attention-to-search/main/roadmap.md
from-attention-to-search/main/problem_anchors/README.md
from-attention-to-search/main/experiments/README.md
```

Older but still relevant experiment folders included:

```text
from-attention-to-search/main/experiments/H0529a_zipfian_frequency_shortcut/
from-attention-to-search/main/experiments/H0530a_hierarchical_common_sense_moe/
```

## Excluded

```text
XingyuD/
experiment code workspaces
raw outputs
checkpoints
logs
generated datasets
run directories
```
