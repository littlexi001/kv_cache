# 05 — Deep high-frequency 0.25× phase

This is the soft counterpart of method 04. In the deepest third, pairs F0–F11
use

$$
\phi_{\ell i}(p)=0.25p\omega_i;
$$

all other layer–pair combinations remain native. The method preserves a weak
local offset signal while increasing the distance required for a complete
rotation by four times.

It supports the soft-attenuation hypothesis if it improves retrieval and PPL
relative to both hard deletion and native RoPE. Similar results to deletion mean
the residual 0.25× phase was not important under this budget.

Config: `configs/strategies/deep_highfreq_soft025.json`.
