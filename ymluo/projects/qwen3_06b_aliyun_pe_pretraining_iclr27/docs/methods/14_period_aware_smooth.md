# 14 — Period-aware deep phase attenuation

**New candidate.** A fixed pair number means different things when head
dimension or RoPE base changes. This method reads the live inverse frequency and
uses the actual native period

$$
T_i=\frac{2\pi}{|\omega_i|}.
$$

Define

$$
g_i=\sigma\!\left(\frac{\log 2048-\log T_i}{0.75}\right),\qquad
g_\ell=\sigma\!\left(\frac{\ell-0.65(L-1)}{0.08L}\right).
$$

Then

$$
\phi_{\ell i}(p)=[1-0.75g_\ell g_i]p\omega_i.
$$

Short-period pairs in deep layers are slowed most, while long-period and shallow
pairs remain near native. It passes only if the advantage persists across
context lengths; otherwise period 2048 is merely an overfit threshold.

Config: `configs/strategies/period_aware_smooth.json`.
