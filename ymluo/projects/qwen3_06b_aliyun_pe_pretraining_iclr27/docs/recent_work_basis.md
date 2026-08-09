# Recent PE work used only to choose experimental axes

This package does not copy a published training recipe. The following primary
sources were used to identify which variables are worth testing:

- [Rope to Nope and Back Again (2025)](https://arxiv.org/abs/2501.18795):
  motivates separating position-sensitive and position-free layers and treating
  training stability as a first-class metric.
- [Layer-Specific Scaling of Positional Encodings (2025)](https://arxiv.org/abs/2503.04355):
  motivates measuring different phase scales by layer instead of assuming one
  scale is optimal everywhere.
- [Context-aware RoPE (2025)](https://arxiv.org/abs/2507.23083) and
  [LaMPE (2025)](https://arxiv.org/abs/2508.02308): motivate conditional or
  length-aware phase maps rather than one fixed extrapolation rule.
- [Fractional Rotation, Full Potential? (2026)](https://arxiv.org/abs/2603.11611):
  motivates treating rotary dimensions as an experimental axis and retaining a
  small amount of rotation for stability rather than assuming all dimensions
  are equally necessary.
- [RoPE Distinguishes Neither Positions Nor Tokens in Long Contexts, Provably
  (2026)](https://arxiv.org/abs/2605.15514): motivates testing whether a fixed
  phase function loses reliable relevance ordering as length grows.
- [RULER](https://github.com/NVIDIA/RULER) and
  [LongBench](https://github.com/THUDM/LongBench): motivate separating
  controlled retrieval/tracing diagnostics from realistic long-context QA.

The package's `smooth_layer_frequency` candidate uses a product of continuous
layer and frequency gates. The `smooth_remote_warp` candidate additionally uses
a local-identity logarithmic remote map with matched value and derivative at the
boundary. These exact formulas and parameters are the package's own testable
operationalizations; novelty and value remain contingent on matched-control
results and a later literature/claim audit.

