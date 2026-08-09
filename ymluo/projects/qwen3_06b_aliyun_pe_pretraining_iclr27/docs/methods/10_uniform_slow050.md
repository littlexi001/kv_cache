# 10 — Uniform 0.5× RoPE

Every phase accumulates at half speed:

$$
\phi_{\ell i}(p)=0.5p\omega_i.
$$

It doubles every rotary period and preserves a linear relative-position map,
but changes local offsets in all layers. It is therefore a strong test of
global slowing rather than a locality-preserving proposal. Method 09 is the
weaker-dose comparison; methods 11–14 test more selective alternatives.

Config: `configs/strategies/uniform_slow_rope.json`.
