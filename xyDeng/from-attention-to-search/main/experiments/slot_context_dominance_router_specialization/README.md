# Slot-Context Dominance Router Specialization

Anchor:

```text
../../problem_anchors/gated_main_causes/slot_context_dominance_router_specialization_anchor.md
```

Story:

```text
This experiment tests whether stronger slot visibility in the B-position hidden state is sufficient for a standard top-1 MoE router to form slot-aligned routing with slot-specific expert utility.
```

Current status:

```text
completed
```

Primary decision metric:

```text
route NMI trajectory + route/utility heatmaps + assignment-utility agreement
```

Documents:

```text
execution_plan.md
summary.md
detailed.md
figures/
tables/
```

Current conclusion:

```text
single-B supports slot-controlled routing; multi-B trajectory shows slot-centroid init starts non-diagonal and training only partially binds routing to utility, weakening the claim that stronger slot context is sufficient for robust functional specialization.
```
