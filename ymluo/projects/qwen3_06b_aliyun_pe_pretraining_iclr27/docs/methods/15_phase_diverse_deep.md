# 15 — Phase-diverse deep layers

**New candidate.** If every deep layer uses the same RoPE rate, the same remote
offset can place important fast pairs near an unfavorable phase repeatedly.
This method gives consecutive deep layers different rates so at least some
layers see a different alias pattern.

Shallow layers and F16 onward remain native. Starting at layer
\(\lfloor0.5L\rfloor\), F0–F15 cycle through

$$
\alpha_\ell\in[1.0,\ 0.75,\ 0.5,\ 0.25],\qquad
\phi_{\ell i}(p)=\alpha_\ell p\omega_i.
$$

The test is not whether one checkpoint improves by chance, but whether retrieval
position variance falls while PPL remains competitive. If average retrieval
improves but layer outputs become unstable or PPL worsens, phase diversity is
not yet a useful implementation.

Config: `configs/strategies/phase_diverse_deep.json`.
