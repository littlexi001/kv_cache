# 09 — Uniform 0.75× RoPE

Every layer and frequency pair uses

$$
\phi_{\ell i}(p)=0.75p\omega_i.
$$

This is a conservative global interpolation baseline. It preserves translation
equivariance because the position map remains linear. Compare it with 0.5×
RoPE: if both improve similarly, exact scale is not decisive; if 0.75× retains
PPL but 0.5× hurts it, excessive global compression is the failure source.

Config: `configs/strategies/uniform_slow075.json`.
