# Data provenance

These files are frozen copies or concise exports of the experiment artifacts
used by the paper figures and tables.

- `heldout_summary.csv` and `heldout_paired_deltas.csv`:
  `ymluo/projects/qwen3_local_rule_failure_boundary/artifacts/`
  `20260731_sage_prerope_heldout24_8gpu/analysis/`
- `layerwise_exact_reconstruction.csv`:
  `ymluo/projects/qwen3_local_rule_failure_boundary/outputs/`
  `onehop_layerwise_amplification_patch_20260728/exact_reconstruction/`
- `activation_patch.csv`: same layerwise experiment, parent output directory.
- `first_layer_frequency_bands.csv`:
  `ymluo/projects/qwen3_local_rule_failure_boundary/artifacts/`
  `20260730_first_layer_rope_phase_gpu7/analysis/`
- `first_layer_qk.csv`: concise export of the exact table in
  `doc/rope_long_retrieval_derivation_and_design_20260730.md`.

Do not edit numerical values in place. Regenerate the upstream artifact,
freeze a new copy, and record the change in the paper notes.

