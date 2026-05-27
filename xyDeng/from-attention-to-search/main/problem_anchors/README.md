# Problem Anchors

This folder stores living documents for each small research problem in
`from-attention-to-search`.

Use one anchor when a question needs stable reasoning across multiple
experiments:

```text
problem definition
-> physical priors
-> mathematical modeling
-> computational realization
-> minimal falsifiable test
-> evidence update
-> claim boundary and next decision
```

Anchors are not static proposals. Update the relevant anchor when an experiment
changes the physical prior, mathematical object, computational test, claim
boundary, or next decision.

## Active Anchor

- `gated_main_causes/slot_context_dominance_router_specialization_anchor.md`:
  current living anchor for the slot-context dominance diagnostic. This is the
  document to improve before the next experiment.

## Shared Definitions

Expert specialization:

```text
A route, expert, or representation partition is useful only if it aligns with a
feature/function-relevant factor and has causal utility. Bucket purity or high
normalized mutual information is not sufficient.
```

Archived downstream effect:

```text
Reverse KV retrieval is not the current success definition. It can be reopened
only after the anchor establishes stable feature-level expert specialization.
```

Current measurement principle:

```text
Assignment diagnostics show where tokens go. Expert ablation utility shows what
the expert causally supports. The current anchor must judge whether assignment
and utility agree under a setup where token identity, frequency, and
slot/context feature can disagree.
```
