# 03 — NoPE in the deepest quarter

**Prior.** Shallow layers need position to build local syntax, while the deepest
layers receive contextual states and may benefit from content-only matching.

With \(L\) layers and \(\ell_0=\lceil0.75L\rceil\):

$$
\phi_{\ell i}(p)=
\begin{cases}
0,&\ell\ge\ell_0,\\
p\omega_i,&\ell<\ell_0.
\end{cases}
$$

This isolates *where* position is removed. Comparing it with all-NoPE and
every-fourth-layer NoPE separates a deep-layer effect from position removal in
general. PPL damage falsifies the claim that the deepest quarter no longer needs
order.

Config: `configs/strategies/deep_nope_quarter.json`.
