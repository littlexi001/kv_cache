# 11 — Smooth layer-only phase schedule

**Prior.** Deep layers should become less position-sensitive, but a hard layer
boundary can make optimization brittle.

Let

$$
g_\ell=\sigma\!\left(\frac{\ell-0.65(L-1)}{0.08L}\right),\qquad
\alpha_\ell=1-0.5g_\ell.
$$

All frequency pairs in layer \(\ell\) use

$$
\phi_{\ell i}(p)=\alpha_\ell p\omega_i.
$$

This tests layer dependence without frequency selection. If method 12 wins and
this does not, frequency selectivity—not only depth—is necessary.

Config: `configs/strategies/deep_allfreq_smooth.json`.
