# P0 — Offline-optimized cross-layer phase complementarity

## Falsifiable claim

For remote offsets from 8K to 128K, assigning different RoPE rates to the first
16 rotary pairs of the deep half of the network can increase the best available
phase response across deep layers. After matched pretraining, this should reduce
retrieval position variance relative to native RoPE without increasing held-out
DCLM PPL by more than 2%.

Improving the offline phase objective alone does not support the downstream
claim. The claim fails for this implementation if matched RULER/LongBench and
gold-answer NLL do not improve, or if the PPL guardrail fails.

## Mathematical objective

Let \(\alpha_{\ell i}\in[0.25,1]\) multiply the native inverse frequency
\(\omega_i\) in deep layer \(\ell\) and rotary pair \(i\). For remote distance
\(\Delta\) and an unknown content-preferred phase \(\psi\), define

$$
r_{\ell i}(\Delta,\psi)
=
\cos\!\left(\alpha_{\ell i}\omega_i\Delta-\psi\right).
$$

The implementation uses a normalized smooth maximum over deep layers:

$$
C_i(\Delta,\psi)
=
\tau\log\!\left(
\frac{1}{|\mathcal D|}
\sum_{\ell\in\mathcal D}
\exp\frac{r_{\ell i}(\Delta,\psi)}{\tau}
\right),
$$

and minimizes

$$
\mathcal L_{\mathrm{phase}}
=
-\mathbb E_{i,\Delta,\psi}[C_i(\Delta,\psi)].
$$

The normalized smooth maximum prevents the objective from improving merely
because more layers are included. Averaging over 32 values of \(\psi\) avoids
assuming that every semantic Q/K pair prefers phase zero.

## Exact algorithm

1. Read Qwen's layer count, head dimension, rotary dimension, and `rope_theta`
   from `/mnt/workspace/Qwen3-0.6B/config.json`; do not load checkpoint weights.
2. Reconstruct the native `inv_freq` and reject non-default RoPE configs.
3. Modify layers 14–27 for a 28-layer model and pairs F0–F15. All other
   layer/pair entries remain exactly 1.
4. Parameterize every modified scale with a sigmoid in `[0.25, 1]` and initialize
   deep layers with the repeating sequence `[1, 0.75, 0.5, 0.25]`.
5. Optimize the phase-coverage objective with Adam for 2,000 CPU steps at LR
   0.03. Use 33 log-spaced distances from 8,192 to 131,072 tokens, 32 uniformly
   spaced content phases, and temperature 0.08.
6. Save the full `[layer, rotary_pair]` scale matrix in the strategy JSON and
   save the optimization trace beside it.
7. During LM pretraining, the matrix is fixed. Only normal model weights are
   optimized. The only architectural change is the layer/pair-specific RoPE
   rate.

## Parameters and failure modes

| Parameter | Value | Meaning | Too small | Too large |
|---|---:|---|---|---|
| Modified depth | final 50% | layers allowed to use complementary rates | weak remote effect | local/early processing disturbed |
| Modified pairs | F0–F15 | shortest-period pairs | insufficient phase coverage | too much positional structure changed |
| Scale range | 0.25–1.0 | allowed fraction of native rate | approaches NoPE | insufficient diversity if lower bound rises |
| Remote range | 8K–128K | distances optimized offline | misses target context | spends capacity outside evaluation range |
| Content phases | 32 | possible semantic preferred phases | objective aliases | more CPU cost with diminishing benefit |
| Temperature | 0.08 | smooth-max sharpness | unstable near-hard max | rewards average layers rather than best coverage |

Named failures are: invalid model geometry, non-finite objective, scale outside
`[0,1]`, offline coverage no better than native, training divergence, unmatched
data/token count, downstream retrieval failure, and DCLM PPL guardrail failure.

## Outputs and debug artifacts

- `configs/strategies/optimized_phase_complementary.json`
- `configs/strategies/optimized_phase_complementary.optimization.json`
- run-local `strategy_profile.json`
- training/checkpoint/evaluation artifacts under the configured run root

