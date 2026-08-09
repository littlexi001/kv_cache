# 12 — Smooth layer × frequency attenuation

This factorized candidate changes fast pairs most strongly in deep layers:

$$
g_{\ell i}=
\sigma\!\left(\frac{\ell-0.65(L-1)}{0.08L}\right)
\sigma\!\left(\frac{11-i}{2}\right),
$$

$$
\phi_{\ell i}(p)=\left[1-0.75g_{\ell i}\right]p\omega_i.
$$

The function has no hard layer or frequency boundary. It tests the combined
claim that shallow/local order and slow-frequency broad position should remain
near native, while deep fast rotations should weaken. A win over methods 05 and
11 indicates both smoothness and factorization matter.

Config: `configs/strategies/smooth_layer_frequency.json`.
