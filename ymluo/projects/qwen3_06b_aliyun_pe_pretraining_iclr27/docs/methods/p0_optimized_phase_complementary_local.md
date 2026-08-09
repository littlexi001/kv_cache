# P0 — Phase complementarity with local-phase preservation

## Difference from the unconstrained method

This condition uses the same variables, initialization, remote distance grid,
and phase-coverage objective as `optimized_phase_complementary`. It adds an
explicit penalty for changing the native phase at distances up to 2,048 tokens.
It tests whether preserving local order is necessary for the long-context gain
to survive ordinary language modeling.

For local distance \(\delta\), the phase error relative to native RoPE is

$$
e_{\ell i}(\delta)
=
(\alpha_{\ell i}-1)\omega_i\delta.
$$

The local penalty is

$$
\mathcal L_{\mathrm{local}}
=
\mathbb E_{\ell,i,\delta}
\left[1-\cos e_{\ell i}(\delta)\right].
$$

The offline optimization minimizes

$$
\mathcal L
=
\mathcal L_{\mathrm{phase}}
+0.5\,\mathcal L_{\mathrm{local}}.
$$

The 17 local distances are log-spaced from 1 to 2,048 tokens. The weight 0.5 is
a declared first-round value, not an established optimum. If it is too small,
the result should resemble the unconstrained method; if it is too large, scales
collapse toward native RoPE and remote phase coverage disappears.

## Pass, fail, and insufficient evidence

- **Pass:** compared at identical tokens and data, it improves long-context
  retrieval over native, meets the 2% DCLM PPL guardrail, and preserves short
  metrics better than the unconstrained method.
- **Fail:** it does not improve remote metrics, violates the PPL guardrail, or
  gives no measurable local-retention advantage over the unconstrained method.
- **Insufficient:** checkpoint token counts, manifest hashes, completed sample
  counts, or evaluation tasks do not match.

Outputs are `optimized_phase_complementary_local.json`, its adjacent
`.optimization.json` trace, the run-local strategy profile, checkpoints, and
per-example evaluations.

