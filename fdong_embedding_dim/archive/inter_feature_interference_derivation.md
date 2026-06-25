# Inter-Feature Interference: Assumptions, Mechanisms, and Validation Plan

This note is a compact working model for rare-feature learning in a finite-dimensional latent space.

It is not a full Transformer theory. It is a set of falsifiable assumptions and measurements.

## 1. Core Question

We want to understand when a rare feature, such as

$$
X \to Y,
$$

fails to learn in the presence of a frequent feature, such as

$$
A \to B.
$$

The current hypothesis is:

$$
\text{rare-feature failure is controlled by effective feature-subspace dimension, not raw latent dimension alone.}
$$

Raw latent dimension is

$$
d.
$$

But the model may only use a smaller effective number of directions:

$$
d_{\mathrm{eff}} \ll d.
$$

## 2. Local Latent Space and Feature Directions

Assume that, around a fixed training state, semantic features are represented inside a local latent space

$$
\mathbb{R}^d.
$$

Let there be \(K\) semantic features, with

$$
K>d.
$$

Feature \(i\) is represented by a unit direction

$$
u_i\in\mathbb{R}^d,
\qquad
\|u_i\|_2=1.
$$

If the local basis is given by orthonormal singular-vector directions

$$
e_1,e_2,\dots,e_d,
$$

then feature \(i\) can be written as

$$
u_i=\sum_{a\in S_i}c_{ia}e_a,
$$

where \(S_i\) is the subset of basis directions used by the feature.

Because the basis is orthonormal,

$$
\langle e_a,e_b\rangle=
\begin{cases}
1, & a=b,\\
0, & a\neq b,
\end{cases}
$$

we have

$$
\begin{aligned}
\|u_i\|_2^2
&=
\left\langle
\sum_{a\in S_i}c_{ia}e_a,
\sum_{b\in S_i}c_{ib}e_b
\right\rangle\\
&=
\sum_{a\in S_i}c_{ia}^2.
\end{aligned}
$$

Therefore the unit-vector normalization is

$$
\sum_{a\in S_i}c_{ia}^2=1.
$$

This normalization separates:

- feature direction, \(u_i\),
- feature frequency, \(\pi_i\),
- and feature gradient strength, \(\alpha_i(t)\).

This separation matters because a feature can be rare but geometrically well separated, or frequent but geometrically entangled with other features.

## 3. Feature Overlap and the Correct Use of the Welch Bound

Define the pairwise overlap between features \(i\) and \(j\) as

$$
\rho_{ij}=\langle u_i,u_j\rangle.
$$

If

$$
\rho_{ij}=0,
$$

then the two feature directions are orthogonal.

If

$$
|\rho_{ij}|>0,
$$

then the two features share latent directions.

Because \(K>d\), the \(K\) unit directions cannot all be mutually orthogonal. The Welch bound gives

$$
\frac{1}{K(K-1)}
\sum_{i\neq j}\rho_{ij}^2
\geq
\frac{K-d}{d(K-1)}.
$$

When

$$
K\gg d,
$$

we have

$$
K-d\approx K,
\qquad
K-1\approx K.
$$

Therefore

$$
\frac{K-d}{d(K-1)}
\approx
\frac{K}{dK}
=
\frac{1}{d}.
$$

So the Welch bound implies

$$
\frac{1}{K(K-1)}
\sum_{i\neq j}\rho_{ij}^2
\gtrsim
\frac{1}{d}.
$$

Define the RMS pairwise overlap:

$$
\rho_{\mathrm{rms}}
=
\left(
\frac{1}{K(K-1)}
\sum_{i\neq j}\rho_{ij}^2
\right)^{1/2}.
$$

Then

$$
\rho_{\mathrm{rms}}
\gtrsim
\frac{1}{\sqrt d}.
$$

Important caveat:

- Welch gives a lower bound on RMS overlap.
- It does not prove that a typical pair has overlap \(1/\sqrt d\).
- A small number of highly aligned pairs can raise the RMS.
- Typical-pair claims require a distributional assumption.

## 4. Objects and Mechanisms

The main measured objects are:

- feature frequency:

$$
\pi_i;
$$

- gradient feature direction:

$$
u_i^{(g)}(t);
$$

- signed gradient coefficient:

$$
a_i(t)\in\mathbb{R};
$$

- gradient magnitude:

$$
\alpha_i(t)=|a_i(t)|.
$$

This document focuses on one mechanism:

$$
\text{finite-dimensional feature overlap creates gradient interference.}
$$

### 4.1 Gradient Interference

Model the average gradient as

$$
g(t)
=
\sum_i \pi_i a_i(t)u_i^{(g)}(t)
+\xi(t),
$$

where $\xi(t)$ is minibatch noise.

For rare feature $r$, project onto its gradient direction:

$$
\langle g(t),u_r^{(g)}(t)\rangle
=
\pi_r a_r(t)
+
\sum_{i\neq r}\pi_i a_i(t)
\langle u_i^{(g)}(t),u_r^{(g)}(t)\rangle
+
\langle \xi(t),u_r^{(g)}(t)\rangle.
$$

Define signal:

$$
S_r^{(g)}(t)=\pi_r a_r(t).
$$

Define interference:

$$
I_r^{(g)}(t)
=
\sum_{i\neq r}\pi_i a_i(t)
\langle u_i^{(g)}(t),u_r^{(g)}(t)\rangle.
$$

This gives:

$$
\langle g(t),u_r^{(g)}(t)\rangle
=
S_r^{(g)}(t)+I_r^{(g)}(t)+N_r^{(g)}(t).
$$

## 5. Isotropic Null Model

The clean null model assumes feature directions are isotropic random unit vectors.

For fixed rare feature $r$:

$$
\rho_{ir}=\langle u_i,u_r\rangle.
$$

Under isotropy:

$$
\mathbb{E}[\rho_{ir}]=0,
\qquad
\mathbb{E}[\rho_{ir}^2]=\frac{1}{d}.
$$

Then typical overlap is:

$$
|\rho_{ir}|_{\mathrm{typical}}
\approx
\frac{1}{\sqrt d}.
$$

If overlaps are also approximately uncorrelated, then random gradient interference has scale:

$$
|I_r^{(g)}(t)|_{\mathrm{random}}
\approx
\frac{1}{\sqrt d}
\left(
\sum_{i\neq r}\pi_i^2\alpha_i(t)^2
\right)^{1/2}.
$$

The corresponding SIR proxy is:

$$
\operatorname{SIR}_r^{(g)}(t)
\approx
\sqrt d\,
\frac{
\pi_r\alpha_r(t)
}{
\left(
\sum_{i\neq r}\pi_i^2\alpha_i(t)^2
\right)^{1/2}
}.
$$

Interpretation:

- This is a null baseline.
- It is not assumed true for trained LLMs.
- It isolates the best-case $1/\sqrt d$ random-overlap scaling.

## 6. Zipf Frequencies and Effective Dimension

Zipf frequency alone does not contradict isotropic directions.

Zipf says:

$$
\pi_i=Ci^{-\beta}.
$$

Isotropy says:

$$
\mathbb{E}[u_i u_i^\top]\approx \frac{1}{d}I.
$$

Both can be true only if frequency and direction are approximately independent:

$$
\pi_i \perp u_i.
$$

This is a strong assumption.

In trained models, high-frequency features may:

- be learned earlier,
- receive more updates,
- reuse shared directions,
- occupy high-gain subspaces,
- concentrate feature and gradient energy.

So instead of isotropy, expect anisotropy:

$$
\Sigma\neq \frac{1}{d}I.
$$

For gradient interference, define the frequency-weighted gradient covariance:

$$
\Sigma_g(t)
=
\sum_i
\pi_i^2 a_i(t)^2
u_i^{(g)}(t){u_i^{(g)}(t)}^\top.
$$

Then the interference variance seen by rare feature $r$ is better modeled as:

$$
u_r^{(g)}(t)^\top
\Sigma_g(t)
u_r^{(g)}(t).
$$

The isotropic model replaces this matrix by:

$$
\Sigma_g(t)
\approx
\frac{\operatorname{tr}\Sigma_g(t)}{d}I.
$$

That replacement is the strong assumption.

Define effective dimension:

$$
d_{\mathrm{eff}}(\Sigma)
=
\frac{
\left(\operatorname{tr}\Sigma\right)^2
}{
\operatorname{tr}(\Sigma^2)
}.
$$

If the spectrum is flat:

$$
d_{\mathrm{eff}}\approx d.
$$

If energy concentrates into a few directions:

$$
d_{\mathrm{eff}}\ll d.
$$

Therefore the realistic scaling is closer to:

$$
\frac{1}{\sqrt{d_{\mathrm{eff}}}},
$$

not necessarily:

$$
\frac{1}{\sqrt d}.
$$

Main revised conjecture:

$$
\text{long-tail feature learning is controlled by the effective dimension of the frequency-weighted feature subspace.}
$$

## 7. Assumptions to Validate

### Assumption A: Isotropic Feature Directions

Claim:

$$
\mathbb{E}[u_i u_i^\top]\approx \frac{1}{d}I.
$$

Validation:

- Estimate the feature covariance and gradient covariance.
- Compute their eigenvalue spectra.
- Compute $d_{\mathrm{eff}}$.
- Check whether $d_{\mathrm{eff}}\approx d$ or $d_{\mathrm{eff}}\ll d$.

Failure condition:

$$
d_{\mathrm{eff}}\ll d.
$$

Interpretation:

- Isotropy is false.
- Interference is governed by effective dimension, not raw dimension.

### Assumption B: Frequency-Direction Independence

Claim:

$$
\pi_i \perp u_i.
$$

Validation:

- Compare high-frequency and low-frequency feature directions.
- Check whether high-frequency features concentrate in leading eigenvectors of the feature or gradient covariance.
- Measure correlation between feature frequency $\pi_i$ and projection onto top covariance eigenvectors.

Failure condition:

$$
\operatorname{corr}
\left(
\pi_i,
\|P_{\mathrm{top}}u_i\|_2^2
\right)
\gg 0.
$$

Interpretation:

- Head features occupy a preferred subspace.
- Zipf plus semantic reuse breaks the isotropic null model.

### Assumption C: Rare Feature Has Enough Gradient Signal

Claim:

$$
\pi_r\alpha_r(t)
$$

is large enough relative to interference.

Validation:

- Estimate conditional gradient:

$$
q_i^{(g)}(t)
=
\mathbb{E}[\nabla_\theta \mathcal{L}\mid i,t].
$$

- Define:

$$
u_i^{(g)}(t)
=
\frac{q_i^{(g)}(t)}{\|q_i^{(g)}(t)\|_2}.
$$

- Estimate signal:

$$
\widehat S_r^{(g)}(t)
=
\pi_r\|q_r^{(g)}(t)\|_2.
$$

- Estimate interference:

$$
\widehat I_r^{(g)}(t)
=
\sum_{i\neq r}
\pi_i
\langle q_i^{(g)}(t),u_r^{(g)}(t)\rangle.
$$

- Compute:

$$
\widehat{\operatorname{SIR}}_r^{(g)}(t)
=
\frac{
\widehat S_r^{(g)}(t)
}{
|\widehat I_r^{(g)}(t)|+\varepsilon
}.
$$

Failure condition:

$$
\widehat{\operatorname{SIR}}_r^{(g)}(t)\ll 1.
$$

Interpretation:

- Rare-feature update is dominated by other feature updates.
- This supports the gradient-interference hypothesis.

## 8. Minimal Validation Experiments

### Experiment 1: Isotropy and Effective Dimension

Goal:

- Test whether feature directions are isotropic or concentrated.

Construct:

$$
\Sigma_g(t)
=
\sum_i
\pi_i^2a_i(t)^2
u_i^{(g)}(t){u_i^{(g)}(t)}^\top.
$$

Measure:

- $d_{\mathrm{eff}}(\Sigma_g)$.
- eigenvalue spectrum of $\Sigma_g$.
- top-eigenvector concentration.

Evidence against isotropy:

$$
d_{\mathrm{eff}}(\Sigma_g)\ll d.
$$

### Experiment 2: Frequency-Direction Coupling

Goal:

- Test whether high-frequency features occupy preferred directions.

Measure:

- projection of each feature direction onto the top covariance subspace:

$$
\|P_{\mathrm{top}}u_i^{(g)}\|_2^2.
$$

- correlation with feature frequency:

$$
\operatorname{corr}
\left(
\pi_i,
\|P_{\mathrm{top}}u_i^{(g)}\|_2^2
\right).
$$

Evidence for coupling:

$$
\operatorname{corr}
\left(
\pi_i,
\|P_{\mathrm{top}}u_i^{(g)}\|_2^2
\right)
\gg 0.
$$

### Experiment 3: Rare-Feature Gradient SIR

Goal:

- Test whether rare features fail when gradient interference dominates their signal.

For rare feature \(r\), estimate:

$$
\widehat S_r^{(g)}(t)
=
\pi_r\|q_r^{(g)}(t)\|_2,
$$

and

$$
\widehat I_r^{(g)}(t)
=
\sum_{i\neq r}
\pi_i
\langle q_i^{(g)}(t),u_r^{(g)}(t)\rangle.
$$

Then compute:

$$
\widehat{\operatorname{SIR}}_r^{(g)}(t)
=
\frac{
\widehat S_r^{(g)}(t)
}{
|\widehat I_r^{(g)}(t)|+\varepsilon
}.
$$

Evidence for gradient interference:

$$
\widehat{\operatorname{SIR}}_r^{(g)}(t)\ll 1
$$

when the rare feature fails.

### Experiment 4: Dimension Sweep

Sweep:

$$
d\in\{2,3,4,5,10,20,50,100\}.
$$

Measure:

- $d_{\mathrm{eff}}(\Sigma_g)$.
- rare-feature accuracy.
- rare-feature gradient SIR.

Key test:

- Does rare-feature learning correlate better with raw dimension \(d\) or effective dimension \(d_{\mathrm{eff}}\)?

Expected result under the revised conjecture:

$$
\text{rare-feature learning correlates better with }d_{\mathrm{eff}}\text{ than raw }d.
$$

## 9. Decision Tree for This Document

For a failed rare feature \(r\):

1. Estimate the feature or gradient covariance.
   - Compute eigenvalues.
   - Compute \(d_{\mathrm{eff}}\).

2. Check isotropy.
   - If \(d_{\mathrm{eff}}\approx d\), the isotropic null model is plausible.
   - If \(d_{\mathrm{eff}}\ll d\), raw dimension is misleading.

3. Check frequency-direction coupling.
   - If high-frequency features project strongly onto top covariance directions, the Zipf-frequency distribution is coupled to geometry.

4. Check rare-feature gradient SIR.
   - If \(\widehat{\operatorname{SIR}}_r^{(g)}\ll 1\), the rare feature is gradient-interference limited.

5. Sweep raw dimension.
   - If learning tracks \(d_{\mathrm{eff}}\) better than \(d\), the effective-dimension conjecture is supported.

## 10. Final Working Claim

The isotropic model gives a useful null baseline:

$$
|I_r|_{\mathrm{random}}
\sim
\frac{1}{\sqrt d}.
$$

But trained models may have concentrated feature covariance:

$$
d_{\mathrm{eff}}\ll d.
$$

Therefore a better working claim is:

$$
\text{rare-feature gradient interference depends on effective feature-subspace dimension, not raw dimension alone.}
$$

The next empirical goal is to determine whether rare-feature learning correlates better with

$$
d_{\mathrm{eff}}
$$

than with

$$
d.
$$
