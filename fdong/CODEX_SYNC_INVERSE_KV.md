# Codex Sync: Inverse KV

Before discussing the inverse KV project, read these two files:

1. `main_inverse_kv/inverse_kv_overview.md`
   - compact report for advisor
   - contains the current conclusion and TODO

2. `fdong/inverse_kv_experiment_notes.md`
   - full experiment history
   - contains synthetic data design, model variants, analysis metrics, and detailed observations

After reading, summarize:

- core hypothesis;
- current conclusion;
- what current experiments support;
- what current experiments deny or revise;
- why current MoE routing is insufficient;
- why Attention is now considered the bottleneck;
- next TODO.

Important current conclusion:

The project is not simply "use existing MoE for KV cache compression." The intended direction is feature-based memory organization. Experiments show Attention learns high-level relational features, but not cleanly enough. `attention output routing + head-level MoE` improves alignment with Attention, but does not yet produce reliable KV buckets. Next step: design constrained Attention so feature relations are cleaner, then use Attention-derived relations as routing/index signals.
