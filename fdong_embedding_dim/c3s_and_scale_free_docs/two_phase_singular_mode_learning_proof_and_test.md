# A Two-Phase Theory of Singular-Mode Learning

## A Self-Contained Formal Proof and a Minimal Test

### Abstract

This note gives a minimal formal explanation for a two-phase phenomenon observed during training: a useful singular direction first rotates toward a feature direction, and after this direction becomes aligned, training mostly increases the corresponding singular value. The goal is to distinguish two objects that are often mixed together in discussion. A singular vector specifies **which feature direction a matrix reads or writes**. A singular value specifies **how strongly that direction is amplified**.

The proof uses the simplest possible model: a rank-one linear map trained with a margin-based loss. Even in this toy setting, the dynamics separate naturally. The singular-vector alignment error decays with a saturation factor, while the singular value can continue to grow as long as the loss still rewards a larger margin. This gives a clean mathematical explanation for why cross-entropy training tends to produce stable directions but continuously growing gains.

The central conclusion is:

**$1-c(t)^2 = (1-c(0)^2)\exp\left(-\left(\sigma(t)^2-\sigma(0)^2\right)\right)$.**

Here, $c(t)$ is the alignment between the learned singular vector and the true feature direction, and $\sigma(t)$ is the singular value. This equation says that alignment error decays exponentially as singular gain grows. Therefore, direction learning saturates early, while gain amplification can continue later.

---

## 1. Motivation

In a trained neural network matrix, a large singular value often indicates that the model has created a high-gain channel. If a feature aligns with the corresponding singular vector, then that feature can strongly affect logits, attention scores, or downstream hidden states.

However, there are two distinct learning processes.

The first process is **direction discovery**. The model must rotate a singular vector toward a useful feature direction.

The second process is **gain amplification**. Once the useful direction is found, the model can keep increasing the singular value to increase the margin or confidence.

The question is whether this two-phase picture can be derived rather than merely described.

This note proves that it can be derived in a minimal linear model.

---

## 2. Model Setup

Consider a rank-one linear model:

**$W(t)=\sigma(t)u(t)v(t)^\top$.**

Here, $W(t)$ is the trainable matrix at time $t$, $\sigma(t)\geq 0$ is its singular value, $u(t)$ is the output-side singular vector, and $v(t)$ is the input-side singular vector. Both $u(t)$ and $v(t)$ have unit norm.

To isolate the input-side effect, assume the output direction is already fixed to the correct output direction:

**$u(t)=u_\star$.**

Here, $u_\star$ is the target output direction. This assumption removes one degree of freedom so that we can focus on how the model learns the input-side singular vector and the singular value.

Let the input feature be a unit vector:

**$\|x\|=1$.**

The model output is:

**$Wx=\sigma u_\star(v^\top x)$.**

Define the alignment between the learned singular vector and the input feature:

**$c=v^\top x$.**

Here, $c=\cos\theta$, where $\theta$ is the angle between $v$ and $x$. If $c=1$, then the singular vector exactly matches the feature direction. If $c=0$, then the singular vector is orthogonal to the feature.

The scalar margin along the correct output direction is:

**$m=\sigma c$.**

Here, $m$ is the useful output strength. It is the product of a gain term $\sigma$ and an alignment term $c$.

This equation is the core decomposition:

**$m=\sigma c$.**

It means the model can improve the margin either by rotating $v$ toward $x$, which increases $c$, or by increasing the singular value $\sigma$.

---

## 3. Loss Assumption

Assume the loss is a decreasing function of the margin:

**$\ell=\phi(m)$, with $\phi'(m)<0$.**

Here, $\phi(m)$ is any smooth loss that rewards larger margins.

Define the residual learning pressure:

**$r(m)=-\phi'(m)>0$.**

Here, $r(m)$ measures how strongly the loss still wants to increase the margin.

For binary logistic loss, which is the simplest cross-entropy-like loss:

**$\phi(m)=\log(1+\exp(-m))$.**

Then:

**$r(m)=\frac{1}{1+\exp(m)}$.**

This pressure decreases as the margin grows, but it remains positive for every finite $m$. Therefore, cross-entropy-type losses continue to reward larger margins even after the classification direction is already correct.

---

## 4. Gradient Flow for the Singular Value

Because $m=\sigma c$, the derivative of the margin with respect to the singular value is:

**$\frac{\partial m}{\partial \sigma}=c$.**

By gradient flow, the singular value evolves as:

**$\frac{d\sigma}{dt}=-\frac{\partial \ell}{\partial \sigma}$.**

Using the chain rule:

**$\frac{\partial \ell}{\partial \sigma}=\phi'(m)\frac{\partial m}{\partial \sigma}=\phi'(m)c$.**

Since $r(m)=-\phi'(m)$, we get:

**$\frac{d\sigma}{dt}=r(m)c$.**

This equation says that the singular value increases whenever the current singular vector has positive alignment with the feature direction. If $c>0$, then $d\sigma/dt>0$.

---

## 5. Gradient Flow for the Singular Vector

The singular vector $v$ must remain on the unit sphere:

**$\|v\|=1$.**

Therefore, its update cannot move in an arbitrary Euclidean direction. It must lie in the tangent space of the unit sphere at $v$.

The alignment is:

**$c=v^\top x$.**

The Euclidean gradient of $c$ with respect to $v$ is $x$. But only the component of $x$ orthogonal to $v$ can rotate $v$ while preserving $\|v\|=1$.

The tangent projection is:

**$(I-vv^\top)x$.**

Here, $I$ is the identity matrix, and $vv^\top x$ is the component of $x$ parallel to $v$. Therefore, $(I-vv^\top)x$ is the part of $x$ that can rotate $v$ toward the feature direction.

Since $m=\sigma v^\top x$, the gradient flow for $v$ on the unit sphere is:

**$\frac{dv}{dt}=r(m)\sigma(I-vv^\top)x$.**

This equation says that $v$ rotates toward $x$. The rotation speed is proportional to the residual pressure $r(m)$, the singular value $\sigma$, and the remaining tangent misalignment $(I-vv^\top)x$.

Now compute the alignment dynamics:

**$\frac{dc}{dt}=\frac{d}{dt}(v^\top x)=x^\top\frac{dv}{dt}$.**

Substitute the expression for $dv/dt$:

**$\frac{dc}{dt}=r(m)\sigma x^\top(I-vv^\top)x$.**

Because $\|x\|=1$ and $v^\top x=c$, we have:

**$x^\top(I-vv^\top)x=1-c^2$.**

Therefore:

**$\frac{dc}{dt}=r(m)\sigma(1-c^2)$.**

This is the key direction-learning equation. It contains the saturation factor $1-c^2$. As $c$ approaches $1$, this factor approaches $0$, so the rotation of $v$ must slow down.

---

## 6. Coupled Dynamics and the Two-Phase Mechanism

We have derived two coupled differential equations:

**$\frac{d\sigma}{dt}=r(m)c$.**

**$\frac{dc}{dt}=r(m)\sigma(1-c^2)$.**

The singular-value equation does not contain the saturation factor $1-c^2$. The alignment equation does.

This difference is the mathematical source of the two-phase behavior.

To see this more precisely, divide the alignment equation by the singular-value equation:

**$\frac{dc}{d\sigma}=\frac{\sigma(1-c^2)}{c}$.**

The residual pressure $r(m)$ cancels out. This means the geometric relation between alignment and singular-value growth is independent of the exact decreasing margin loss, as long as the loss depends on $m=\sigma c$.

Rearrange:

**$\frac{c}{1-c^2}dc=\sigma d\sigma$.**

Integrate both sides:

**$-\frac{1}{2}\log(1-c^2)=\frac{1}{2}\sigma^2+\mathrm{constant}$.**

Using the initial values $c(0)=c_0$ and $\sigma(0)=\sigma_0$, we obtain:

**$1-c(t)^2=(1-c_0^2)\exp\left(-\left(\sigma(t)^2-\sigma_0^2\right)\right)$.**

This proves the central claim.

The alignment error $1-c(t)^2$ decays exponentially in the increase of $\sigma(t)^2$. Therefore, moderate singular-value growth can rapidly make the singular vector nearly aligned with the feature direction.

Once $c(t)\approx 1$, the alignment dynamics become:

**$\frac{dc}{dt}=r(m)\sigma(1-c^2)\approx 0$.**

The direction no longer changes much.

But the singular-value dynamics become:

**$\frac{d\sigma}{dt}=r(m)c\approx r(m)$.**

So if the loss still has positive residual pressure, the singular value can keep growing.

This proves the two-phase mechanism in the minimal model.

---

## 7. Theorem Statement

**Theorem: two-phase singular-mode learning in a rank-one linear model.**

Consider the rank-one model $W=\sigma u_\star v^\top$, where $\sigma\geq 0$, $\|v\|=1$, $\|x\|=1$, and $u_\star$ is fixed. Let the margin be $m=\sigma v^\top x$, and let the loss be $\ell=\phi(m)$ with $\phi'(m)<0$. Under gradient flow with $v$ constrained to the unit sphere, define $c=v^\top x$ and $r(m)=-\phi'(m)>0$. Then:

**$\frac{d\sigma}{dt}=r(m)c$.**

**$\frac{dc}{dt}=r(m)\sigma(1-c^2)$.**

Consequently:

**$1-c(t)^2=(1-c_0^2)\exp\left(-\left(\sigma(t)^2-\sigma_0^2\right)\right)$.**

Therefore, singular-vector alignment error decays exponentially as singular gain grows. After alignment saturates, the singular vector changes little, while the singular value can continue to increase whenever the loss still rewards a larger margin.

---

## 8. Interpretation of Singular Vectors and Singular Values

The singular vector $v$ answers the question:

**Which input feature direction does the matrix read?**

If $v$ aligns with $x$, then the matrix reads the feature $x$ strongly.

The singular value $\sigma$ answers the question:

**How strongly does the matrix amplify that feature direction?**

If $v=x$, then:

**$Wx=\sigma u_\star$.**

Here, increasing $\sigma$ directly increases the output magnitude along the correct output direction.

Therefore, the two-phase picture has a precise meaning.

In Phase 1, training discovers the feature direction by rotating $v$ toward $x$.

In Phase 2, training increases the gain $\sigma$ after the direction has already been discovered.

---

## 9. Why Cross-Entropy Produces Late Singular-Value Growth

For logistic loss:

**$\phi(m)=\log(1+\exp(-m))$.**

The residual pressure is:

**$r(m)=\frac{1}{1+\exp(m)}$.**

This is always positive for finite $m$. Therefore, even if $c\approx 1$, the singular-value dynamics satisfy:

**$\frac{d\sigma}{dt}\approx \frac{1}{1+\exp(\sigma)} > 0$.**

The growth becomes slow, but it continues.

This explains why cross-entropy can keep increasing singular values after the singular direction is already aligned. The model is no longer primarily learning the direction. It is increasing the margin.

This also explains why confidence-capped loss can reduce late-stage singular-value amplification. If the loss is capped once $p_y\geq q$, then the residual pressure becomes zero or negligible after sufficient confidence. In this model, that means $r(m)$ becomes zero or small, so both $d\sigma/dt$ and $dc/dt$ vanish. Since $dc/dt$ was already small after alignment, the main practical effect is to stop further singular-value growth.

---

## 10. Relation to C3S-AdamW

The proof changes how we should interpret C3S-AdamW.

C3S-AdamW should not be understood as a method that necessarily prevents features from aligning with top singular vectors. Useful feature alignment is part of Phase 1 and should be allowed.

Instead, the method should be understood as controlling Phase 2:

**C3S-AdamW preserves useful direction discovery while reducing unnecessary singular-value amplification.**

This matches the observed pattern in which final feature alignment with top singular vectors can remain high, while the corresponding singular values are significantly reduced.

The correct functional quantity is not only the unweighted alignment:

**$\langle f,v_i\rangle^2$.**

The more important gain-weighted contribution is:

**$\sigma_i^2\langle f,v_i\rangle^2$.**

Here, $f$ is a learned feature direction, $v_i$ is a singular vector, and $\sigma_i$ is the corresponding singular value. C3S can reduce high-gain interference even if the alignment term remains large, provided that the singular value is reduced.

---

## 11. Minimal Test Suggested

The simplest test should directly measure the two predicted phases.

Train a small linear or transformer model and save checkpoints every fixed number of steps. For a selected matrix $W(t)$, compute the top-$k$ right singular subspace:

**$P_k(t)=V_k(t)V_k(t)^\top$.**

Here, $V_k(t)$ contains the top $k$ right singular vectors of $W(t)$.

Measure subspace drift:

**$\delta_k(t)=\|P_k(t)-P_k(t-\Delta)\|_2$.**

Here, $\delta_k(t)$ measures how much the top-$k$ singular subspace changes over a time window $\Delta$.

Measure singular-gain growth:

**$g_k(t)=\sum_{i=1}^k\sigma_i(t)^2-\sum_{i=1}^k\sigma_i(t-\Delta)^2$.**

Here, $g_k(t)$ measures how much the top singular values grow over the same window.

Measure feature alignment, if a known feature direction $f(t)$ is available:

**$a_k(t)=\|P_k(t)f(t)\|^2$.**

Here, $a_k(t)$ measures how much the feature lies in the top singular subspace.

Measure gain-weighted feature effect:

**$e_k(t)=\sum_{i=1}^k\sigma_i(t)^2\langle f(t),v_i(t)\rangle^2$.**

The two-phase hypothesis predicts the following pattern.

Early training should have high subspace drift and rapidly increasing feature alignment:

**$\delta_k(t)$ large, and $a_k(t)$ increasing.**

Later training should have low subspace drift but continuing singular-gain growth under cross-entropy:

**$\delta_k(t)\approx 0$, while $g_k(t)>0$.**

Under confidence-capped loss or C3S-AdamW, the late-stage singular-gain growth should be smaller:

**$g_k^{\mathrm{C3S}}(t)<g_k^{\mathrm{CE}}(t)$ during the late phase.**

However, feature alignment may remain similar:

**$a_k^{\mathrm{C3S}}(t)\approx a_k^{\mathrm{CE}}(t)$.**

The decisive signature is therefore:

**similar alignment, lower singular gain, and lower gain-weighted effect under C3S.**

---

## 12. Minimal Experimental Protocol

A minimal experiment can be run as follows.

Train two models from the same initialization.

The first model uses standard cross-entropy with AdamW.

The second model uses confidence-capped C3S-AdamW after warmup.

At checkpoints $t=0,\Delta,2\Delta,\ldots,T$, record the selected matrix $W(t)$.

For each checkpoint, compute:

**$V_k(t)$, $\sigma_1(t),\ldots,\sigma_k(t)$, and $P_k(t)=V_k(t)V_k(t)^\top$.**

Then plot four curves.

First, plot subspace drift:

**$\delta_k(t)=\|P_k(t)-P_k(t-\Delta)\|_2$.**

Second, plot top-$k$ singular energy:

**$S_k(t)=\sum_{i=1}^k\sigma_i(t)^2$.**

Third, plot feature alignment:

**$a_k(t)=\|P_k(t)f(t)\|^2$.**

Fourth, plot gain-weighted feature effect:

**$e_k(t)=\sum_{i=1}^k\sigma_i(t)^2\langle f(t),v_i(t)\rangle^2$.**

The expected result is that standard CE and C3S may both discover similar top singular directions, so $a_k(t)$ may be comparable. But C3S should reduce $S_k(t)$ and $e_k(t)$ in the late phase.

This would support the refined claim:

**The proposed method does not necessarily prevent feature-subspace sharing; it reduces excessive gain amplification on shared singular directions.**

---

## 13. Claim Boundary

This proof is exact only for a rank-one linear model with a fixed output direction and a margin-based loss. It does not prove that every neural network matrix must behave exactly this way.

However, the proof identifies a local mechanism that remains relevant in larger models. For any matrix block, a singular vector specifies a direction of feature reading or writing, and the singular value specifies gain. Once a useful direction is found, cross-entropy-style losses can keep increasing gain because they continue to reward larger margins.

Therefore, the proof gives a principled explanation for the experimentally observed pattern:

**feature alignment can remain high while singular values decrease under the proposed method.**

The correct conclusion is not that top singular vectors are always bad. The correct conclusion is that excessive singular gain can create interference, and controlling late-stage gain growth is a valid optimizer objective.

---

## 14. Summary

The rank-one proof gives a precise mathematical basis for the two-phase hypothesis.

The model margin is:

**$m=\sigma c$.**

Here, $\sigma$ is singular gain and $c$ is feature-direction alignment.

Gradient flow gives:

**$\frac{d\sigma}{dt}=r(m)c$.**

**$\frac{dc}{dt}=r(m)\sigma(1-c^2)$.**

The alignment equation contains the saturation factor $1-c^2$, but the singular-value equation does not. Therefore, direction learning naturally saturates, while singular-value growth can continue.

The exact relationship is:

**$1-c(t)^2=(1-c_0^2)\exp\left(-\left(\sigma(t)^2-\sigma_0^2\right)\right)$.**

This proves that the singular vector can align early, while the singular value continues to grow later.

The suggested test is to track top singular subspace drift, top singular energy, feature alignment, and gain-weighted feature effect over training. The expected signature of C3S-AdamW is not necessarily lower alignment with top singular vectors, but lower singular values and lower gain-weighted feature effect after alignment has already stabilized.
