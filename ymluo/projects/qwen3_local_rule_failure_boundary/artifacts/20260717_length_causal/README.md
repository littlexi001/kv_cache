# 2026-07-17 length-causal experiment artifacts

The main aggregate was validated against the full Cartesian product:

- 16 seeds;
- 5 lengths;
- 6 interference conditions;
- 1 middle placement;
- 3 queries per body.

See `main/validation.json` for the exact counts. The `main` CSV files contain:

- `core_summary.csv`: full two-hop generation and candidate metrics;
- `probe_summary.csv`: first-hop and oracle-second-hop cloze metrics;
- `final_only_summary.csv`: ranking after removing start and intermediate-state candidates;
- `latent_mechanism_summary.csv`: access/binding vs second-rule vs composition diagnosis;
- `paired_effects.csv`: paired interference-minus-clean effects.

The full per-sample results and candidate scores remain on the server at:

```text
/home/fdong/ymluo/projects/qwen3_local_rule_failure_boundary/outputs/length_causal_main_20260717
/home/fdong/ymluo/projects/qwen3_local_rule_failure_boundary/outputs/length_causal_placement_20260717
```

The Chinese research synthesis is in `../../doc/length_causal_data_design_results_20260717.md`.
