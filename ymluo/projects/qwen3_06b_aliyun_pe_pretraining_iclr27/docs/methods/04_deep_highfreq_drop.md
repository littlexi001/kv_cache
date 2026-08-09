# 04 — Delete deep high-frequency RoPE pairs

**Prior.** Fast-rotating pairs encode useful local offsets in shallow layers but
can introduce many remote phase aliases in deep semantic matching.

Let \(K=12\) and \(\ell_0=\lceil2L/3\rceil\):

$$
\phi_{\ell i}(p)=
\begin{cases}
0,&\ell\ge\ell_0\ \text{and}\ i<K,\\
p\omega_i,&\text{otherwise}.
\end{cases}
$$

Compare directly with methods 05 and 06. If soft attenuation wins, deletion was
too abrupt; if all three fail, the high-frequency prior or the selected layer
range is unsupported.

Config: `configs/strategies/deep_highfreq_drop.json`.
