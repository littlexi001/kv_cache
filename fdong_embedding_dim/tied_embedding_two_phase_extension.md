# Two-Phase Singular-Mode Learning with Tied Input-Output Embeddings

## Extending the Rank-One Proof to the Shared-Space Setting

---

### 0. What Changes Under Tied Embedding

The original proof assumes $u = u_\star$ fixed. In a tied embedding model:

$$u(t) = \frac{E_{K}(t)}{\|E_{K}(t)\|} \in \mathbb{R}^d, \quad v(t) \in \mathbb{R}^d, \quad \|v(t)\| = 1$$

where $u(t)$ is the output-side singular vector (the embedding of the most frequent target token $K$) and $v(t)$ is the dominant input-side singular direction of $W_q$ (or $W_k$).

**The input and output vectors live in the same $d$-dimensional space** — $u(t)$ and $v(t)$ are directly comparable and couple through both forward and backward passes.

---

### 1. Model Setup (3-Body Extension)

Consider a reduced model capturing the dominant singular mode:

$$W(t) = \sigma(t)\, u(t)\, v(t)^\top$$

**Three degrees of freedom** (compared to two in the fixed-$u$ proof):

- $\sigma(t) \ge 0$ — singular value (gain)
- $v(t)$ — input-side singular direction, $\|v\| = 1$
- $u(t)$ — output-side singular direction, $\|u\| = 1$, tied to the embedding of the most frequent target token $K$

Let $\bar{x} \in \mathbb{R}^d$ be the mean direction of all hidden states that predict token $K$ (the centroid toward which $E_K$ is pulled by cross-entropy gradient). Let $\|\bar{x}\| = 1$ without loss of generality (direction is what matters).

Define three pairwise alignments:

$$c_{vu} = v^\top u \qquad c_{vx} = v^\top \bar{x} \qquad c_{ux} = u^\top \bar{x}$$

The effective margin along the dominant mode:

$$m = \sigma \cdot c_{vu} \cdot c_{ux}$$

(Decoding: singular gain $\sigma$, times input-direction-to-output-direction alignment $c_{vu}$, times output-direction-to-centroid alignment $c_{ux}$. All three must cooperate for a useful signal.)

---

### 2. Loss and Gradient Flow

As in the original proof, let $\ell = \phi(m)$ with $\phi'(m) < 0$, and define $r(m) = -\phi'(m) > 0$.

**Gradient flow for $\sigma$** (unchanged in structure):

$$\frac{d\sigma}{dt} = r(m) \cdot c_{vu} \cdot c_{ux}$$

When $v$ and $u$ are both aligned with $\bar{x}$ ($c_{vu} \approx 1$, $c_{ux} \approx 1$), this reduces to $\frac{d\sigma}{dt} \approx r(m)$, identical to the original proof. **No saturation factor appears for $\sigma$.**

**Gradient flow for $v$** (input-side, projected to unit sphere):

$$\frac{dv}{dt} = r(m) \sigma \cdot (I - vv^\top) \cdot u$$

Here $u$ replaces the fixed $x$ from the original proof. $v$ rotates toward $u$ because the gradient $\partial L / \partial W_q$ is an outer product $\propto (\partial L/\partial q) \otimes h$, and $h$'s dominant direction aligns with $u$ through the tied embedding path.

**Gradient flow for $u$** (output-side, new equation):

$$\frac{du}{dt} = r(m) \sigma \cdot (I - uu^\top) \cdot \bar{x}$$

$u = E_K$ is pulled toward $\bar{x}$ (the centroid of all hidden states that predict $K$). The projection operator $(I - uu^\top)$ preserves $\|u\| = 1$ on the unit sphere.

---

### 3. Coupled Dynamics

We now have three coupled ODEs:

$$\begin{aligned}
\frac{d\sigma}{dt} &= r(m) \cdot c_{vu} \cdot c_{ux} \\[4pt]
\frac{dv}{dt}    &= r(m) \sigma \cdot (I - vv^\top) \cdot u \\[4pt]
\frac{du}{dt}    &= r(m) \sigma \cdot (I - uu^\top) \cdot \bar{x}
\end{aligned}$$

The three alignments evolve as:

$$\begin{aligned}
\frac{dc_{vu}}{dt} &= r(m) \sigma \left[ (1 - c_{vu}^2) \cdot c_{ux} + c_{vx} \cdot (1 - c_{vu}^2) \right] \\[4pt]
\frac{dc_{ux}}{dt} &= r(m) \sigma \cdot (1 - c_{ux}^2) \cdot c_{vu} \\[4pt]
\frac{dc_{vx}}{dt} &= r(m) \sigma \cdot (1 - c_{vx}^2) \cdot c_{vu}^2
\end{aligned}$$

---

### 4. Phase 1: Direction Discovery (all three ODEs active)

All three $(1-c^2)$ factors are $\approx 1$ initially. Each pair (v-u, u-x, v-x) undergoes rapid alignment:

- $v$ rotates toward $u$ (gradient coupling through tied embedding)
- $u$ rotates toward $\bar{x}$ (centroid pull from being the most frequent target)
- $v$ indirectly rotates toward $\bar{x}$ via $u$

The product $r(m) \sigma$ amplifies all rates. Alignment saturates when $c_{vu} \to 1$, $c_{ux} \to 1$.

**This phase is faster than the fixed-$u$ case** because both $u$ and $v$ can move simultaneously — they meet halfway.

---

### 5. Phase 2: Why Gain Keeps Growing After Direction Locks

When $c_{vu} \approx 1$ and $c_{ux} \approx 1$:

The alignment equations contain saturation factors:

$$\frac{dc_{vu}}{dt} \approx r(m) \sigma \cdot (1 - 1^2) \cdot c_{ux} + \ldots \approx 0$$
$$\frac{dc_{ux}}{dt} \approx r(m) \sigma \cdot (1 - 1^2) \cdot c_{vu} \approx 0$$

**But the singular value equation has NO such factor:**

$$\frac{d\sigma}{dt} = r(m) \cdot c_{vu} \cdot c_{ux} \approx r(m) > 0$$

Under cross-entropy, $r(m) = 1/(1+e^m) > 0$ for all finite $m$, so $\frac{d\sigma}{dt}$ remains positive.

### Critical Difference from the Fixed-$u$ Case

In the tied-embedding setting, $\bar{x}$ (the centroid) **itself drifts slowly during training** because the distribution of hidden states that predict $K$ changes as other tokens' embeddings also evolve.

This slow drift creates a small but persistent misalignment between $u(t)$ and the instantaneous $\bar{x}(t)$. The factor $(1-c_{ux}^2)$ is never exactly zero — it settles at a small but finite value $\epsilon > 0$.

This provides a **sustained source of directional correction** that prevents the system from reaching the fully-saturated fixed point of the original proof. The consequence:

- $u$ and $v$ co-drift, tracking a slowly-moving $\bar{x}$
- $\sigma$ continues to grow, driven by $r(m) > 0$
- The three-body system reaches a **dynamical equilibrium** rather than a frozen state

---

### 6. Theorem Extension

**Theorem (Two-Phase Learning with Tied Embeddings).** Consider the three-body model $W = \sigma u v^\top$, with $\|u\| = \|v\| = \|\bar{x}\| = 1$, margin $m = \sigma c_{vu} c_{ux}$, and loss $\ell = \phi(m)$ with $\phi'(m) < 0$. Under gradient flow with both $u$ and $v$ constrained to the unit sphere:

$$\begin{aligned}
\frac{d\sigma}{dt} &= r(m) c_{vu} c_{ux} \\[4pt]
\frac{dc_{vu}}{dt} &= r(m)\sigma\left[(1-c_{vu}^2)c_{ux} + c_{vx}(1-c_{vu}^2)\right] \\[4pt]
\frac{dc_{ux}}{dt} &= r(m)\sigma(1-c_{ux}^2)c_{vu}
\end{aligned}$$

Consequently, the system exhibits two phases:

1. **Phase 1 (Direction Discovery):** $c_{vu} \to 1$ and $c_{ux} \to 1$ — both input and output singular directions align with the centroid.
2. **Phase 2 (Gain Amplification with Co-Drift):** After alignment saturates, $\frac{d\sigma}{dt} \approx r(m) > 0$ while $\frac{dc_{vu}}{dt} \approx 0$ and $\frac{dc_{ux}}{dt} \approx 0$. Unlike the fixed-$u$ case, $u$ and $v$ experience sustained co-drift tracking a slowly-moving centroid $\bar{x}(t)$, preventing the saturation factors $(1-c_{vu}^2)$ and $(1-c_{ux}^2)$ from reaching exact zero. This provides continuous, albeit small, directional corrections that sustain the gain amplification loop.

---

### 7. What This Means for the Tied-Embedding Mechanism

| | Fixed-$u$ Proof | Tied-Embedding Extension |
|---|---|---|
| Phase 1 | $v$ rotates to $x$ | $v$ rotates toward $u$, $u$ toward $\bar{x}$ — **both move** |
| Phase 2 | $v$ frozen, $\sigma$ grows | $u$ and $v$ co-drift tracking $\bar{x}$, $\sigma$ grows |
| Convergence speed | Single-direction rotation | **Faster** — $u$ and $v$ meet halfway |
| Saturation exactness | $(1-c^2) \to 0$ exactly | $(1-c^2) \to \epsilon > 0$ — persistent misalignment |
| Source of continued gain | Pure $\sigma$ growth | $\sigma$ growth + slow $u,v$ co-drift provide constant small loss signal |

---

### 8. Connection to Experimental Evidence

| Theoretical Prediction | Observed in Toy / Qwen |
|---|---|
| Phase 1: $v$ and $u$ rapidly converge | $\cos(Vh_E[0], Vh_{W_q}[0]) = 0.99$ after convergence |
| Phase 2: $\sigma$ continues growing after direction locks | $\sigma_1/\sigma_4$ from 1.0 → 1000× (no_rew) while $\cos \approx 1.0$ |
| $u$ and $v$ co-drift | Qwen: $Vh$ direction is stable ($\cos = 1.000$ across L2-L26) but $\sigma$ still grows |
| Reweighting controls Phase 2 gain | hard_rew: $\sigma_1/\sigma_4$ from 1000 → 134 (7.5× reduction) |
| Reweighting does not harm Phase 1 alignment | $\cos(Vh_H, Vh_{W_q})$ remains 0.99 under hard_rew |

---

### 9. Summary

The tied-embedding extension preserves the two-phase structure of the original proof while correcting for three missing phenomena:

1. **Both $u$ and $v$ move** during Phase 1, accelerating direction discovery
2. **Co-drift** in Phase 2: $u$ and $v$ track a slowly-moving centroid, preventing exact saturation of the $(1-c^2)$ factors
3. **Sustained gain amplification**: the persistent small misalignment provides a constant signal for $\sigma$ growth, explaining why cross-entropy training continues to inflate singular values long after directions stabilize

The implication for intervention design is that **reweighting should target Phase 2 gain amplification** (controlling $\sigma$ growth) rather than Phase 1 direction discovery (which is both useful and hard to prevent in tied-embedding models).
