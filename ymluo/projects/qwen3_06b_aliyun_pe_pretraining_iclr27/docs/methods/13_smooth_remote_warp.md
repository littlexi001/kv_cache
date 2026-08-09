# 13 — Local-identity, remote-log position warp

Positions at or below \(W=2048\) are unchanged. Beyond the window:

$$
c(p)=W+4096\log\!\left(1+\frac{p-W}{4096}\right).
$$

A smooth deep/high-frequency gate \(g_{\ell i}\), capped at 0.85, produces

$$
\tilde p_{\ell i}=p-0.85g_{\ell i}[p-c(p)],\qquad
\phi_{\ell i}(p)=\tilde p_{\ell i}\omega_i.
$$

The value and slope match at the local boundary, so the intervention turns on
smoothly. Unlike linear scaling, this nonlinear absolute map is not strictly
translation-equivariant; position-sweep variance is therefore a required
failure check.

Config: `configs/strategies/smooth_remote_warp.json`.
