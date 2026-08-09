# Position encoding design for Qwen3-0.6B pretraining

## Falsifiable conjecture

When Qwen3-0.6B is trained from the same random initialization on the same 100B
DCLM tokens, reducing remote RoPE phase interference selectively by layer and
frequency can improve long-context retrieval over native RoPE without increasing
held-out language-model PPL by more than 2%.

The claim is false for a tested function if it loses to native RoPE at matched
tokens, improves only one hand-built probe, omits more evaluation cases, or
breaks the PPL guardrail. One seed ranks candidates but cannot establish the
final claim.

## Inputs and output

- Architecture and tokenizer: `/mnt/workspace/Qwen3-0.6B`.
- Initialization: random weights from seed 20260808; checkpoint weights are not
  read under the default `INITIALIZATION=from_scratch`.
- Data: deterministic DCLM training/validation manifests under
  `/mnt/workspace/dclm`.
- Changed variable: only the position-to-phase function used to rotate Q and K.
- Outputs: resolved PE profile, TensorBoard events, checkpoints, PPL, controlled
  retrieval rows, LongBench rows, failures, environment metadata, and bundles.

## Physical priors and mathematical model

Let \(p\) be token position, \(\ell\) layer, \(i\) rotary pair, and \(\omega_i\)
the model's native inverse frequency. Every condition implements

$$
\phi_{\ell i}(p)=f_{\ell i}(p)\omega_i.
$$

Native RoPE has \(f_{\ell i}(p)=p\). The experiment tests four priors:

1. **Local order remains useful.** Strong NoPE conditions are controls; leading
   candidates preserve shallow or local position information.
2. **Deep representations may need less fast position rotation.** Deep-layer
   and high-frequency interventions test whether contextualized states benefit
   from more content-oriented matching.
3. **Hard boundaries may be harder to optimize.** Smooth layer, frequency, and
   remote-position functions are compared with hard deletion.
4. **One common phase rate can create shared aliases.** A phase-diverse candidate
   gives consecutive deep layers different high-frequency rates.

The layer-mixing and layer-specific axes are motivated by recent primary work
on [RoPE/NoPE hybrid attention](https://arxiv.org/abs/2501.18795),
[layer-specific scaling](https://arxiv.org/abs/2503.04355), and
[long-distance rotary dimension inefficiency](https://arxiv.org/abs/2502.11276).
The package does not copy their implementations. Exact functions for all 16
conditions are in `docs/methods/00_*.md` through `15_*.md`.

## Condition structure

- Tasks 0–1: native RoPE and all-NoPE controls.
- Tasks 2–3: layer placement of NoPE.
- Tasks 4–8: high-, middle-, and low-frequency deep interventions.
- Tasks 9–10: uniform linear scaling controls.
- Tasks 11–13: smooth layer, factorized layer-frequency, and remote warp.
- Tasks 14–15: period-aware and phase-diverse project candidates.

This structure lets a failure identify which assumption failed. For example,
method 12 beating method 11 indicates that frequency selection matters beyond
depth; method 05 beating method 04 indicates that soft attenuation matters
beyond high-frequency removal.

## Implementation contract

1. Read the Qwen config and tokenizer. Under from-scratch initialization, create
   model weights only after setting the common seed.
2. Read the live rotary `inv_freq`; do not hard-code the RoPE base or head size.
3. Compute a `[sequence, rotary_pairs]` effective-position matrix for every
   layer from the selected JSON method.
4. Convert it to phase, cosine, and sine in float64/float32, then cast to the
   reference model dtype.
5. Replace only the `(cos, sin)` tensors entering the original Qwen attention.
   Q/K/V projections, causal masking, softmax, residual blocks, and MLP remain
   unchanged.
6. Reject unknown kinds, invalid bands, scales outside `[0,1]`, non-finite
   phases, wrong tensor shapes, missing configs, or a global batch other than
   256.
7. Save `strategy_profile.json` so every layer/pair scale and sample effective
   position can be inspected rather than inferred from a condition name.

## Stage evidence

| Stage | Required artifact | Pass condition | Named failure |
|---|---|---|---|
| Initialization | environment and run contract | same architecture/seed across tasks | initialization mismatch |
| Data | `manifest_metadata.json` | identical SHA256 and disjoint validation | data mismatch |
| PE construction | `strategy_profile.json` | finite expected shape and declared profile | invalid PE |
| Training | TensorBoard + JSONL | finite loss/gradient and exact token step | divergence or batch mismatch |
| Checkpoint | integrity hashes | complete local checkpoint | incomplete/untrusted checkpoint |
| Evaluation | per-sample rows | equal completed sample counts | insufficient evidence |
| Merge | matched native deltas | same manifest and token count | unmatched comparison |

## Claim boundary

The period-aware and phase-diverse functions are explicit research hypotheses,
not established innovations. Their novelty requires a broader literature audit,
and their value requires matched multi-seed full-benchmark evidence after the
16-way screening run.
