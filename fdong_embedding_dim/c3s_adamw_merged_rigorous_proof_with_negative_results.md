# From Compositional Singular Concentration to C3S-AdamW

## A Rigorous Problem Statement, Step-by-Step Proof, and Optimizer Design

## Abstract

Language learning has a nested structure. A simple feature can be reused by a more specific feature, and the more specific feature can then be reused by an even more specific feature. In a controlled symbolic setting, this looks like

$$
A \rightarrow AB \rightarrow ABC \rightarrow ABCD.
$$

In natural language, the analogous structure is that a phrase such as `city` can be extended to `beautiful city`, then to `NYC is a beautiful city`, and then to `NYC is a beautiful city in winter`. Each longer expression preserves part of the earlier meaning and adds a new constraint.

This document develops one coherent theory and method. The problem is that frequent common features are learned early and repeatedly. Their repeated gradients can create a high-gain singular subspace in matrices such as $W_{\mathrm{out}}$, $W_Q$, and $W_K$. Later long-tail features are then attracted to that subspace because moving along a large-singular-value direction gives a larger first-order loss decrease when the direction is aligned with the current prediction error. This reuse can be efficient, but it also creates interference: many distinct features become coupled through the same high-gain directions.

The document also proves why two tempting alternatives, spherical hidden states with AdamW and Poincaré-bounded hidden states with AdamW, do not solve the identified mechanism: they constrain representation geometry but do not guarantee suppression of common-subspace components in the actual AdamW update direction.

The proposed solution is **C3S-AdamW**, short for **Confidence-Capped Frozen Common-Subspace Suppressed AdamW**. The method first trains normally so that common prerequisite features can form a stable backbone. It then estimates and freezes the common-feature subspace. After that, it uses a confidence-capped loss to stop already-learned common examples from producing unnecessary gradients, and it suppresses the frozen common-subspace component of active update directions so that underlearned long-tail features are forced to use complementary dimensions.

The central statement is:

$$
\boxed{
\text{common-feature learning creates a high-gain backbone, and C3S-AdamW prevents this backbone from monopolizing later long-tail learning.}
}
$$

This document separates what is proved, what is assumed, and what must be tested. The proofs are given in a local linearized setting, which is the right level for understanding gradient-based training. The extension to transformers follows by applying the same argument to each selected parameter block and its local Jacobian.

---

## 1. What Is Proved, What Is Assumed, and What Is Empirical

The document uses three kinds of statements.

A **proved statement** follows from algebra, first-order Taylor expansion, or the definition of the proposed optimizer. For example, if an update vector $u$ is replaced by $(I-\rho P_C)u$, then its component inside $C$ is multiplied by $1-\rho$ and its component outside $C$ is preserved exactly.

An **assumption** is a condition required by the theorem. For example, the frozen-projector theorem assumes that the common subspace stabilizes after warmup:

$$
\|P_C(t)-P_C(T_0)\|_2 \leq \epsilon_C.
$$

Here, $P_C(t)$ is the common-feature projector at step $t$, $P_C(T_0)$ is the projector frozen after warmup, and $\epsilon_C$ is the allowed drift.

An **empirical claim** is something that must be measured. For example, the claim that $W_Q$, $W_K$, and $W_{\mathrm{out}}$ show stronger concentration than $W_V$ is supported by the nested-prefix experiment, but it is not a theorem about all transformers or all natural language.

This distinction matters because the optimizer guarantee is geometric. It guarantees that the active update is redirected away from the frozen common subspace. It does not guarantee by itself that every rare feature will become learnable. That also requires enough data, capacity, and identifiability.

---

## 2. Nested Features Create a Temporal Learning Order

The synthetic structure is

$$
A \rightarrow AB \rightarrow ABC \rightarrow ABCD.
$$

Here, $A$ is a common prerequisite feature, $AB$ is a composed feature that reuses $A$, and $ABC$ or $ABCD$ are more specific nested features. The arrow means that the later feature depends on the earlier one.

The core modeling assumption is a separation of learning timescales:

$$
T_C \ll T_T.
$$

Here, $T_C$ is the timescale on which common prerequisite features stabilize, and $T_T$ is the timescale on which long-tail nested features are learned. This assumption is justified by frequency and prerequisite structure: common features appear more often and are needed before later composed features can be represented.

This assumption does not say that common features stop appearing later. They do appear later. It says their dominant subspace stabilizes early. That is why we can estimate the common-feature subspace once after warmup.

The diagnostic for this assumption is

$$
\delta(t)=\|P_C(t)-P_C(T_0)\|_2.
$$

Here, $\delta(t)$ measures how much the common-feature projector has drifted after warmup. If $\delta(t)$ is small, the frozen-projector method is justified. If $\delta(t)$ is large, the warmup point $T_0$ is too early or the common-feature backbone is still changing.

---

## 3. Singular Values Create Descent-Efficient Directions

Consider one local linear map from hidden representation to logits:

$$
z = Wh.
$$

Here, $h\in\mathbb{R}^d$ is a hidden representation, $W$ is a trainable matrix, and $z$ is the logit vector.

Let the singular value decomposition of $W$ be

$$
W=U\Sigma V^\top=\sum_i \sigma_i u_i v_i^\top.
$$

Here, $u_i$ is the output-side singular vector, $v_i$ is the hidden-side singular vector, and $\sigma_i$ is the singular value. The singular value is the gain of direction $v_i$: a unit movement along $v_i$ produces an output movement of size $\sigma_i$ along $u_i$.

Expand the hidden representation as

$$
h=\sum_i c_i v_i.
$$

Here, $c_i$ is the coefficient of $h$ along $v_i$. Then

$$
Wh=\sum_i \sigma_i c_i u_i.
$$

This proves the forward-gain fact: for the same hidden coefficient $c_i$, the output effect is proportional to $\sigma_i$.

Now use softmax cross entropy:

$$
L(z,y)=-\log p_y,
\qquad
p=\operatorname{softmax}(z).
$$

Here, $p_y$ is the probability assigned to the correct label $y$.

The gradient with respect to logits is

$$
\nabla_z L=p-e_y.
$$

Here, $e_y$ is the one-hot vector for the correct label. By the chain rule,

$$
\nabla_h L=W^\top\nabla_z L.
$$

Substituting the SVD gives

$$
\nabla_h L=V\Sigma U^\top(p-e_y).
$$

Define

$$
\beta_i=u_i^\top(p-e_y).
$$

Here, $\beta_i$ is the alignment between the output error and the output-side singular vector $u_i$. Therefore,

$$
\nabla_h L=\sum_i \sigma_i\beta_i v_i.
$$

A gradient descent update to the hidden representation is

$$
\Delta h=-\eta\nabla_h L=-\eta\sum_i\sigma_i\beta_i v_i.
$$

Therefore, the update coefficient along $v_i$ is

$$
\Delta c_i=-\eta\sigma_i\beta_i.
$$

Taking magnitude gives

$$
|\Delta c_i|=\eta\sigma_i|\beta_i|.
$$

This proves the local update result: for the same error alignment $|\beta_i|$, a larger singular value $\sigma_i$ gives a larger feature update along $v_i$.

However, the stronger statement is about loss decrease. Consider a perturbation along one singular direction:

$$
\delta h=\epsilon v_i.
$$

Here, $\epsilon$ is the perturbation size. The logit change is

$$
\delta z=W\delta h=\epsilon\sigma_i u_i.
$$

First-order Taylor expansion gives

$$
\delta L \approx \langle \nabla_z L,\delta z\rangle.
$$

Substitute $\nabla_z L=p-e_y$ and $\delta z=\epsilon\sigma_i u_i$:

$$
\delta L\approx \epsilon\sigma_i\langle p-e_y,u_i\rangle.
$$

For a fixed perturbation norm $\epsilon$, the best direction inside $\operatorname{span}(v_i)$ chooses the sign that decreases the loss. Therefore,

$$
\max_{\delta h\in\operatorname{span}(v_i),\ \|\delta h\|=\epsilon}(-\delta L)
\approx
\epsilon\sigma_i|\langle p-e_y,u_i\rangle|.
$$

This proves the descent-efficiency statement:

$$
\boxed{
\text{a large singular value is useful only when its output-side singular vector aligns with the prediction error.}
}
$$

A large $\sigma_i$ alone is not sufficient. If $\langle p-e_y,u_i\rangle=0$, then movement along $v_i$ gives no first-order loss decrease.

Now substitute the actual gradient-step component along $v_i$:

$$
\delta h_i=-\eta\sigma_i\beta_i v_i.
$$

The induced logit change is

$$
\delta z_i=W\delta h_i=-\eta\sigma_i^2\beta_i u_i.
$$

The first-order loss change from this component is

$$
\Delta L_i\approx \langle p-e_y,\delta z_i\rangle
=-\eta\sigma_i^2\beta_i^2.
$$

Equivalently,

$$
\boxed{
\Delta L_i\approx -\eta\sigma_i^2\langle u_i,p-e_y\rangle^2.
}
$$

This proves why high-gain directions attract later features. If a later feature produces error aligned with a direction that already has large $\sigma_i$, then gradient descent obtains larger first-order loss decrease by reusing that direction.

---

## 4. Attention Q/K Directions Follow the Same Descent Principle

Attention computes

$$
q_t=W_Qh_t,
\qquad
k_i=W_Kh_i.
$$

Here, $q_t$ is the query at position $t$, and $k_i$ is the key at position $i$. The attention score is

$$
s_{ti}=\frac{q_t^\top k_i}{\sqrt d}.
$$

The attention weight is

$$
\alpha_{ti}=\frac{\exp(s_{ti})}{\sum_{j\leq t}\exp(s_{tj})}.
$$

For two positions $i$ and $j$,

$$
\frac{\alpha_{ti}}{\alpha_{tj}}=\exp(s_{ti}-s_{tj}).
$$

Thus, once a useful score margin exists, softmax converts it into a multiplicative routing difference. But this does not by itself prove why $W_Q$ or $W_K$ develop high singular values. The proof again comes from first-order loss descent.

Let the attention output be

$$
o_t=\sum_{i\leq t}\alpha_{ti}r_i,
\qquad
r_i=W_Vh_i.
$$

Here, $r_i$ is the value vector. Let

$$
g_{ti}=\frac{\partial L}{\partial s_{ti}}.
$$

For softmax attention, differentiating $o_t$ with respect to $s_{ti}$ gives

$$
\frac{\partial o_t}{\partial s_{ti}}=\alpha_{ti}(r_i-o_t).
$$

Therefore, by the chain rule,

$$
g_{ti}=\alpha_{ti}\left\langle \frac{\partial L}{\partial o_t},r_i-o_t\right\rangle.
$$

This equation identifies whether increasing the score $s_{ti}$ helps or hurts the downstream loss.

Now decompose the query matrix:

$$
W_Q=U_Q\Sigma_QV_Q^\top=\sum_a \sigma_a^Q u_a^Q(v_a^Q)^\top.
$$

Perturb the hidden state at the query position along $v_a^Q$:

$$
\delta h_t=\epsilon v_a^Q.
$$

Then

$$
\delta q_t=W_Q\delta h_t=\epsilon\sigma_a^Q u_a^Q.
$$

The induced score change is

$$
\delta s_{ti}=\frac{(\delta q_t)^\top k_i}{\sqrt d}
=\frac{\epsilon\sigma_a^Q}{\sqrt d}\langle u_a^Q,k_i\rangle.
$$

The first-order loss change is

$$
\delta L\approx \sum_{i\leq t} g_{ti}\delta s_{ti}
=\epsilon\sigma_a^Q\left\langle u_a^Q,\frac{1}{\sqrt d}\sum_{i\leq t}g_{ti}k_i\right\rangle.
$$

Define

$$
\gamma_t^Q=\frac{1}{\sqrt d}\sum_{i\leq t}g_{ti}k_i.
$$

This is exactly $\partial L/\partial q_t$. Therefore,

$$
\delta L\approx \epsilon\sigma_a^Q\langle u_a^Q,\gamma_t^Q\rangle.
$$

The gradient-step contribution along $v_a^Q$ gives

$$
\Delta L_a^Q\approx -\eta(\sigma_a^Q)^2\langle u_a^Q,\gamma_t^Q\rangle^2.
$$

Thus, large query singular values are favored only when their output-side query singular vectors align with useful score-gradient directions.

The key matrix has the same structure. If

$$
W_K=U_K\Sigma_KV_K^\top=\sum_b \sigma_b^K u_b^K(v_b^K)^\top,
$$

then define

$$
\gamma_i^K=\frac{1}{\sqrt d}\sum_t g_{ti}q_t.
$$

This is $\partial L/\partial k_i$. The corresponding key-side loss decrease is

$$
\Delta L_b^K\approx -\eta(\sigma_b^K)^2\langle u_b^K,\gamma_i^K\rangle^2.
$$

Therefore,

$$
\boxed{
\text{Q/K high-gain directions are selected because they implement useful score changes efficiently, and softmax turns those score changes into sharp routing.}
}
$$

This also explains why $W_V$ can behave differently. $W_Q$ and $W_K$ decide which states interact; $W_V$ carries content after routing. There is therefore less reason for all value content to collapse into the same small high-gain routing subspace.

---

## 5. Feature Interference from Shared High-Gain Directions

Let two feature representations be

$$
h_f=\sum_i c_{f,i}v_i,
\qquad
h_g=\sum_i c_{g,i}v_i.
$$

Their raw representation overlap is

$$
\langle h_f,h_g\rangle=\sum_i c_{f,i}c_{g,i}.
$$

The output effects are

$$
Wh_f=\sum_i\sigma_i c_{f,i}u_i,
\qquad
Wh_g=\sum_i\sigma_i c_{g,i}u_i.
$$

Because the $u_i$ are orthonormal, their output-effect overlap is

$$
\langle Wh_f,Wh_g\rangle=\sum_i\sigma_i^2c_{f,i}c_{g,i}.
$$

This is the precise interference amplification formula. Even if the raw representation overlap is moderate, overlap inside high-$\sigma_i$ directions is amplified by $\sigma_i^2$ in output space.

Therefore, if two features share the same high-gain top-$k$ subspace, then changes to one feature can strongly affect the other. This is useful when the features genuinely share structure. It is harmful when a rare feature needs specificity that is not well represented by the common backbone.

The same effect appears in gradients:

$$
\nabla_h L=W^\top\nabla_z L=\sum_i\sigma_i\langle u_i,\nabla_z L\rangle v_i.
$$

So high-gain singular directions are also high-gradient-flow directions. This is why common-feature subspaces become attractors for later learning.

---

## 6. Why Projection Alone Is Not Enough

A natural first solution is to freeze the common-feature subspace $C$ and suppress later gradients inside $C$. This is necessary but not sufficient.

The reason is that common examples continue to appear later in training. Under ordinary cross entropy,

$$
\ell_{\mathrm{CE}}=-\log p_y.
$$

Its derivative with respect to the correct-label log probability is nonzero unless $p_y=1$. In softmax models, exact $p_y=1$ requires infinite logit separation. Thus, ordinary cross entropy keeps pushing already-learned common examples through the sequence

$$
0.9\rightarrow0.99\rightarrow0.999\rightarrow0.9999.
$$

If we suppress only the component inside $C$, then residual common-feature gradients may move into $P_{\perp C}$. That would let common features colonize the complementary directions that we want to reserve for long-tail features.

Therefore, the method needs two mechanisms:

$$
\boxed{
\text{confidence capping stops overtraining of learned common examples, and subspace suppression redirects active tail updates.}
}
$$

---


## 7. Why Spherical and Poincaré-Bounded Hidden States with AdamW Do Not Solve This Failure Mechanism

This section gives a negative result. The claim is not that spherical or Poincaré-bounded hidden states are useless. They may reduce activation-norm growth or improve some representation geometry. The precise claim is narrower and stronger:

$$
\boxed{
\text{spherical or Poincaré-bounded hidden states with ordinary AdamW do not, by themselves, guarantee suppression of common-subspace reuse.}
}
$$

The reason is that the proven failure mechanism lives in the **update direction through a learned high-gain subspace**. A hidden-state constraint changes the representation map, but it does not necessarily remove the component of the AdamW update lying inside the frozen common-feature subspace.

### 7.1 The Required Guarantee

Let $C$ be the frozen common-feature subspace, and let $P_C$ be its projector. Let $u_t$ denote the actual AdamW preconditioned update direction before weight decay for a selected parameter block. The failure metric is the common-subspace update fraction:

$$
\alpha_t^{\mathrm{upd}}
=
\frac{\|P_Cu_t\|^2}{\|u_t\|^2+\epsilon}.
$$

Here, $\alpha_t^{\mathrm{upd}}$ measures how much of the actual update direction lies inside the common-feature subspace. A method that directly solves the proven failure mechanism should guarantee that this quantity is reduced for active long-tail updates.

C3S-AdamW does this by defining

$$
\tilde u_t=r_t(I-\rho_tP_C)u_t.
$$

When $r_t>0$, the common-subspace component becomes

$$
P_C\tilde u_t=r_t(1-\rho_t)P_Cu_t,
$$

and the complementary component becomes

$$
P_{\perp C}\tilde u_t=r_tP_{\perp C}u_t.
$$

Therefore, the common-to-complementary update-energy ratio is multiplied by $(1-\rho_t)^2$. This is an algebraic guarantee.

A hidden-state constraint must be judged against this same target. It is not enough to bound $\|h\|$. It must reduce $\|P_Cu_t\|$ relative to $\|P_{\perp C}u_t\|$ in the actual optimizer update.

### 7.2 Spherical Hidden States Bound Activation Norm but Not Parameter-Space Reuse

A spherical hidden state replaces a hidden representation $h$ by

$$
\bar h=\frac{h}{\|h\|}.
$$

Here, $\bar h$ lies on the unit sphere, so $\|\bar h\|=1$. This removes hidden-state norm as a degree of freedom.

However, the output logit can still be

$$
z_y=w_y^\top \bar h.
$$

Here, $w_y$ is the output weight for label $y$. Since $\|\bar h\|=1$, the model can still increase the logit by increasing $\|w_y\|$ or by increasing singular values in earlier parameter matrices. Thus spherical normalization blocks one route to large logits, but it does not block parameter-space gain growth.

A minimal counterexample proves the point. Consider binary logistic regression with normalized feature $\bar h$ and label $y=1$:

$$
\ell(w)=\log(1+\exp(-w^\top \bar h)).
$$

The gradient is

$$
\nabla_w\ell(w)=-\sigma(-w^\top\bar h)\bar h.
$$

Here, $\sigma$ is the logistic sigmoid. The gradient direction is exactly $\bar h$. If $\bar h\in C$, then

$$
P_C\nabla_w\ell(w)=\nabla_w\ell(w),
\qquad
P_{\perp C}\nabla_w\ell(w)=0.
$$

Therefore,

$$
\alpha^{\mathrm{upd}}=1
$$

for an AdamW step whose preconditioner does not rotate the update out of this one-dimensional direction. The hidden state is perfectly spherical, but the update is still completely inside the common-feature subspace.

This gives the formal negative result:

$$
\boxed{
\text{spherical normalization does not imply }\|P_Cu_t\|<\|u_t\|\text{ or reduce }\alpha_t^{\mathrm{upd}}.
}
$$

The proof is by construction. We exhibited a valid spherical-hidden-state model where the entire update direction lies inside $C$. Therefore, spherical hidden states with AdamW cannot provide the same guarantee as C3S-AdamW.

### 7.3 Spherical Hidden States Can Still Permit Singular-Value Amplification

The same counterexample also shows why cross entropy can still amplify parameter norms. The loss derivative magnitude is

$$
\|\nabla_w\ell(w)\|=\sigma(-w^\top\bar h)\|\bar h\|=\sigma(-w^\top\bar h).
$$

As long as $w^\top\bar h$ is finite, this quantity is positive. Therefore, ordinary cross entropy keeps increasing $w^\top\bar h$ toward larger margins. Since $\|\bar h\|=1$, this pressure must be absorbed by $w$ or by earlier matrices that influence the direction of $\bar h$.

Thus, spherical hidden states do not solve the overtraining problem of common features. They remove hidden-state radial growth, but they leave the cross-entropy pressure toward larger parameter-space margins intact.

### 7.4 Poincaré-Bounded Hidden States Bound Euclidean Radius but Not AdamW Update Geometry

A Poincaré-bounded hidden state maps representations into the open unit ball:

$$
h_B\in\mathbb{B}^d=\{x\in\mathbb{R}^d:\|x\|<1\}.
$$

This imposes a Euclidean norm bound on the hidden representation and may give a hyperbolic interpretation to radius. The hyperbolic distance from the origin is

$$
d_{\mathbb{B}}(0,h_B)=2\operatorname{arctanh}(\|h_B\|).
$$

Here, $d_{\mathbb{B}}(0,h_B)$ is the Poincaré-ball distance from the origin. This distance grows rapidly as $\|h_B\|$ approaches $1$.

However, with ordinary AdamW, the trainable parameters are still updated in Euclidean parameter coordinates. The hidden representation may live inside a hyperbolic ball, but the optimizer does not automatically project updates away from the common-feature subspace $C$.

The same counterexample applies. Let the output logit be

$$
z_y=w_y^\top h_B,
\qquad
\|h_B\|<1.
$$

For binary logistic loss with label $y=1$,

$$
\ell(w)=\log(1+\exp(-w^\top h_B)),
$$

and

$$
\nabla_w\ell(w)=-\sigma(-w^\top h_B)h_B.
$$

If $h_B\in C$, then the entire parameter update lies inside the common-feature subspace:

$$
P_C\nabla_w\ell(w)=\nabla_w\ell(w),
\qquad
P_{\perp C}\nabla_w\ell(w)=0.
$$

Therefore,

$$
\alpha^{\mathrm{upd}}=1.
$$

The hidden state is bounded inside the Poincaré ball, but the AdamW update is still fully aligned with the common-feature subspace. Hence Poincaré-bounded hidden states with AdamW do not guarantee reduction of common-subspace update dominance.

### 7.5 The Hyperbolic Metric Does Not Remove the Need for Update-Space Suppression

One might hope that the nonlinear Poincaré geometry changes the gradient enough to solve the problem. But with AdamW, unless the optimizer itself is Riemannian and explicitly designed to control the relevant subspaces, the parameter update is still based on Euclidean gradients of trainable matrices.

Suppose the bounded hidden state is produced by a differentiable map

$$
h_B=\phi(a),
$$

where $a$ is an unconstrained internal activation. Then the gradient with respect to $a$ is

$$
\nabla_a\ell=J_\phi(a)^\top\nabla_{h_B}\ell.
$$

Here, $J_\phi(a)$ is the Jacobian of the map $\phi$. This Jacobian can rescale or rotate gradients, but it does not by definition remove the common-subspace component. In general there is no identity of the form

$$
P_CJ_\phi(a)^\top\nabla_{h_B}\ell=0.
$$

Therefore, Poincaré bounding may change gradient magnitudes and local geometry, but it does not provide the required algebraic guarantee:

$$
\|P_Cu_t\|^2 \mapsto (1-\rho_t)^2\|P_Cu_t\|^2.
$$

Only an explicit update-space suppression operator provides that guarantee.

### 7.6 Negative Theorem

**Theorem.** Spherical hidden states with AdamW and Poincaré-bounded hidden states with AdamW do not, by themselves, guarantee reduction of common-subspace update dominance. More precisely, for each method there exists a valid model state and a valid example such that the hidden representation satisfies the method's geometric constraint, but the AdamW update direction satisfies $P_Cu_t=u_t$ and therefore $\alpha_t^{\mathrm{upd}}=1$.

**Proof.** For the spherical case, choose a normalized hidden state $\bar h$ with $\|\bar h\|=1$ and $\bar h\in C$. In binary logistic regression, the gradient with respect to the output weight is $-\sigma(-w^\top\bar h)\bar h$, which lies entirely in $C$. Therefore the update direction is entirely inside $C$, so $P_Cu_t=u_t$ and $\alpha_t^{\mathrm{upd}}=1$.

For the Poincaré-bounded case, choose a bounded hidden state $h_B$ with $\|h_B\|<1$ and $h_B\in C$. The same logistic-loss calculation gives gradient $-\sigma(-w^\top h_B)h_B$, again lying entirely in $C$. Therefore the update direction is entirely inside $C$, so $P_Cu_t=u_t$ and $\alpha_t^{\mathrm{upd}}=1$.

Both constructions satisfy the proposed hidden-state geometric constraints. Yet neither reduces the common-subspace update fraction. Therefore neither method can prove the guarantee required to solve the singular-subspace reuse mechanism. This proves the theorem.

### 7.7 Consequence for the Paper

The correct conclusion is not that spherical or Poincaré hidden states are bad methods in general. The correct conclusion is:

$$
\boxed{
\text{they address representation norm or representation geometry, while our failure mechanism is an update-space subspace-reuse problem.}
}
$$

Therefore, they are not the right primary solution for this paper. They may be useful auxiliary controls, but the rigorous solution must act on the actual update direction. This is exactly what C3S-AdamW does.

---

## 8. C3S-AdamW: Corrected Optimizer Design

C3S-AdamW has three stages.

Stage 1 is warmup. Train normally with AdamW and cross entropy until common prerequisite features have formed a stable backbone.

Stage 2 is freezing. Estimate a common-feature subspace $C$ and freeze its projector $P_C$.

Stage 3 is corrected training. Replace ordinary cross entropy with confidence-capped loss, and suppress the common-subspace component of the active AdamW update direction.

The update-space detail is important. AdamW uses a preconditioned update direction, not the raw gradient alone. If we project the raw gradient before Adam’s coordinate-wise preconditioner, the diagonal preconditioner may distort the subspace unless it commutes with $P_C$. Therefore, the rigorous version suppresses the **preconditioned Adam update direction**.

Let

$$
u_t=A_tg_t.
$$

Here, $g_t$ is the raw gradient for a selected parameter block, $A_t$ is Adam’s positive diagonal preconditioner including moment normalization, and $u_t$ is the preconditioned Adam direction before weight decay. In standard AdamW notation, $A_tg_t$ represents the bias-corrected first moment divided coordinate-wise by the square root of the bias-corrected second moment. For the proof, we only need $u_t$ to be the vector that AdamW would use before weight decay.

The ordinary AdamW update is

$$
\Delta\theta_t^{\mathrm{AdamW}}=-\eta u_t-\eta\lambda\theta_t.
$$

C3S-AdamW modifies $u_t$, not merely $g_t$.

---

## 9. Confidence-Capped Loss and Residual Learning Gate

Choose a confidence threshold $q<1$, for example $q=0.9$. The hard confidence-capped loss is

$$
\ell_{\mathrm{cap}}(x,y)=\max(-\log p_y+\log q,0).
$$

If $p_y<q$, then $-\log p_y+\log q>0$, so the example is still active. If $p_y\geq q$, then $-\log p_y+\log q\leq0$, so the loss is zero.

A smooth version is

$$
\ell_{\mathrm{smooth}}(x,y)=\tau\log\left(1+\exp\left(\frac{-\log p_y+\log q}{\tau}\right)\right).
$$

Here, $\tau>0$ is a smoothing temperature. The derivative of the smooth cap with respect to $-
\log p_y$ is

$$
r_t=\sigma\left(\frac{\log q-\log p_{y,t}}{\tau}\right).
$$

Here, $r_t$ is a residual learning gate. If $p_{y,t}\ll q$, then $r_t\approx1$. If $p_{y,t}\gg q$, then $r_t\approx0$.

Thus $r_t$ measures whether this example should still create meaningful learning pressure.

---

## 10. Frozen Common-Subspace Suppression in Update Space

Let $U_C$ be an orthonormal basis for the frozen common-feature subspace. The projector is

$$
P_C=U_CU_C^\top.
$$

The complementary projector is

$$
P_{\perp C}=I-P_C.
$$

Given the preconditioned Adam direction $u_t$, define the alignment with the frozen common subspace:

$$
\alpha_t=\frac{\|P_Cu_t\|^2}{\|u_t\|^2+\epsilon}.
$$

Here, $\epsilon>0$ prevents division by zero. Set

$$
\rho_t=\rho_{\max}\alpha_t.
$$

Here, $\rho_{\max}\in[0,1]$ is the largest allowed suppression strength. The corrected update direction is

$$
\tilde u_t=r_t(I-\rho_tP_C)u_t.
$$

The final C3S-AdamW update is

$$
\Delta\theta_t=-\eta\tilde u_t-\eta\lambda\theta_t.
$$

This is the rigorous optimizer. It differs from the earlier informal version in one important way: suppression is applied after Adam preconditioning, so the proof applies to the actual parameter update direction.

---

## 11. Matrix Form for Transformer Blocks

For a matrix block $W\in\mathbb{R}^{m\times d}$, the common subspace may live on the input side. Suppose the common hidden-side basis is $V_C\in\mathbb{R}^{d\times k}$, with projector

$$
P_C^h=V_CV_C^\top.
$$

If $U_t$ is the preconditioned Adam update matrix for $W$, then the input-side common component is

$$
U_tP_C^h.
$$

The corrected update is

$$
\tilde U_t=r_t\left(U_t-\rho_tU_tP_C^h\right).
$$

This is the matrix form of $\tilde u_t=r_t(I-\rho_tP_C)u_t$ after vectorization. If the relevant common subspace lives on the output side, use left multiplication:

$$
\tilde U_t=r_t\left(U_t-\rho_tP_C^oU_t\right).
$$

Here, $P_C^o$ is the output-side projector. This distinction is important for implementation. The proof is the same after vectorizing the matrix block.

---

## 12. Algorithm

For $t<T_0$, train normally:

$$
\Delta\theta_t=-\eta u_t-\eta\lambda\theta_t.
$$

At $t=T_0$, estimate the common-feature subspace. One option is to compute the SVD of a selected matrix:

$$
W(T_0)=U\Sigma V^\top.
$$

If the hidden-side directions are relevant, choose the top $k$ columns of $V$ as $V_C$. If output-side directions are relevant, choose the top $k$ columns of $U$ as $U_C$.

Another option is to use common-example update covariance:

$$
G_C=\mathbb{E}_{x\in\mathcal{D}_C}[u(x)u(x)^\top].
$$

Here, $u(x)$ is the preconditioned update direction produced by a common or already-learned example. Then choose

$$
U_C=\operatorname{TopEig}_k(G_C).
$$

After freezing $P_C$, for each later step compute $p_{y,t}$ and

$$
r_t=\sigma\left(\frac{\log q-\log p_{y,t}}{\tau}\right).
$$

Compute Adam’s preconditioned update direction $u_t$, then compute

$$
\alpha_t=\frac{\|P_Cu_t\|^2}{\|u_t\|^2+\epsilon},
\qquad
\rho_t=\rho_{\max}\alpha_t.
$$

Correct the update direction:

$$
\tilde u_t=r_t(I-\rho_tP_C)u_t.
$$

Finally update:

$$
\Delta\theta_t=-\eta\tilde u_t-\eta\lambda\theta_t.
$$

---

## 13. Proof I: Frozen Projector Approximation

This proof justifies computing $P_C$ once after warmup.

Let $P_C(t)$ be the ideal common-feature projector at step $t$, and let $P_C(T_0)$ be the frozen projector. Assume

$$
\|P_C(t)-P_C(T_0)\|_2\leq\epsilon_C
\qquad
\text{for all }t\geq T_0.
$$

Online suppression would use

$$
\tilde u_t^{\mathrm{online}}=(I-\rho P_C(t))u_t.
$$

Frozen suppression uses

$$
\tilde u_t^{\mathrm{frozen}}=(I-\rho P_C(T_0))u_t.
$$

Subtract the two:

$$
\tilde u_t^{\mathrm{online}}-\tilde u_t^{\mathrm{frozen}}
= -\rho(P_C(t)-P_C(T_0))u_t.
$$

Take norms:

$$
\|\tilde u_t^{\mathrm{online}}-\tilde u_t^{\mathrm{frozen}}\|
\leq
\rho\|P_C(t)-P_C(T_0)\|_2\|u_t\|.
$$

Use the stability assumption:

$$
\|\tilde u_t^{\mathrm{online}}-\tilde u_t^{\mathrm{frozen}}\|
\leq
\rho\epsilon_C\|u_t\|.
$$

Because $0\leq\rho\leq1$, the frozen method approximates the online method whenever $\epsilon_C$ is small. This proves the validity of one-time decomposition under measurable subspace stability.

---

## 14. Proof II: Confidence Capping Stops Common-Feature Leakage

Consider an already-learned example with

$$
p_y\geq q.
$$

The hard capped loss is

$$
\ell_{\mathrm{cap}}(x,y)=\max(-\log p_y+\log q,0).
$$

Since $p_y\geq q$, we have

$$
-\log p_y+\log q\leq0.
$$

Therefore

$$
\ell_{\mathrm{cap}}(x,y)=0.
$$

In the region $p_y>q$, the loss is locally constant. Therefore its gradient is zero:

$$
\nabla_\theta \ell_{\mathrm{cap}}(x,y)=0.
$$

So the update direction is zero:

$$
u_t=0,
\qquad
\tilde u_t=0.
$$

Thus an already-learned common example cannot create new motion inside $C$ or outside $C$. In particular, it cannot leak into $P_{\perp C}$.

At the boundary $p_y=q$, the hard cap has a subgradient. This single boundary has measure zero in continuous training, and one can choose the zero subgradient or use the smooth cap.

For the smooth cap, the gradient is multiplied by

$$
r_t=\sigma\left(\frac{\log q-\log p_y}{\tau}\right).
$$

When $p_y\gg q$, $r_t\approx0$, so the update is not exactly zero but becomes negligible. This proves the smooth version of the leakage control.

---

## 15. Proof III: Subspace Suppression Preserves the Complementary Component

Let $P_C$ be an orthogonal projector. Therefore

$$
P_C^2=P_C,
\qquad
P_{\perp C}=I-P_C,
\qquad
P_CP_{\perp C}=0.
$$

Every update direction $u_t$ decomposes as

$$
u_t=P_Cu_t+P_{\perp C}u_t.
$$

Suppress the common component:

$$
\hat u_t=(I-\rho P_C)u_t.
$$

Substitute the decomposition:

$$
\hat u_t=(I-\rho P_C)(P_Cu_t+P_{\perp C}u_t).
$$

Distribute the operator:

$$
\hat u_t=P_Cu_t+P_{\perp C}u_t-\rho P_C^2u_t-\rho P_CP_{\perp C}u_t.
$$

Use $P_C^2=P_C$ and $P_CP_{\perp C}=0$:

$$
\hat u_t=(1-\rho)P_Cu_t+P_{\perp C}u_t.
$$

This proves two exact identities:

$$
P_C\hat u_t=(1-\rho)P_Cu_t,
$$

and

$$
P_{\perp C}\hat u_t=P_{\perp C}u_t.
$$

Thus suppression shrinks the common-subspace component and preserves the complementary component exactly.

With the residual gate, $\tilde u_t=r_t\hat u_t$, so both components are scaled by $r_t$. The gate controls whether the example should learn at all. The projection controls where the active update goes.

---

## 16. Proof IV: Suppression Reduces Common-Subspace Energy Ratio

Assume $P_{\perp C}u_t\neq0$. Define the common-to-complementary energy ratio before suppression:

$$
R_{\mathrm{raw}}=\frac{\|P_Cu_t\|^2}{\|P_{\perp C}u_t\|^2}.
$$

After suppression,

$$
\hat u_t=(1-\rho)P_Cu_t+P_{\perp C}u_t.
$$

Therefore, the new ratio is

$$
R_{\mathrm{sup}}=\frac{\|(1-\rho)P_Cu_t\|^2}{\|P_{\perp C}u_t\|^2}.
$$

Because $\|(1-\rho)v\|^2=(1-\rho)^2\|v\|^2$,

$$
R_{\mathrm{sup}}=(1-\rho)^2R_{\mathrm{raw}}.
$$

Since $0\leq\rho\leq1$,

$$
R_{\mathrm{sup}}\leq R_{\mathrm{raw}}.
$$

If $\rho>0$ and $P_Cu_t\neq0$, the inequality is strict. This proves that C3S-AdamW reduces the relative dominance of the common subspace in the active update direction.

---

## 17. Proof V: Suppression Reduces High-Gain Functional Attraction

Let $J_t$ be the local Jacobian mapping a parameter update direction to a first-order change in the relevant model output:

$$
\delta z_t\approx J_t\delta\theta_t.
$$

Here, $J_t$ is fixed for the first-order approximation at the current parameters.

The common-subspace contribution of the raw update is

$$
\delta z_C=J_tP_Cu_t.
$$

The complementary contribution is

$$
\delta z_\perp=J_tP_{\perp C}u_t.
$$

After suppression, the common contribution becomes

$$
\delta z_C^{\mathrm{sup}}=J_t((1-\rho)P_Cu_t)=(1-\rho)J_tP_Cu_t=(1-\rho)\delta z_C.
$$

The complementary contribution becomes

$$
\delta z_\perp^{\mathrm{sup}}=J_tP_{\perp C}u_t=\delta z_\perp.
$$

Therefore, if $\delta z_\perp\neq0$, the ratio of common functional effect to complementary functional effect satisfies

$$
Q_{\mathrm{sup}}
=\frac{\|\delta z_C^{\mathrm{sup}}\|}{\|\delta z_\perp^{\mathrm{sup}}\|}
=(1-\rho)\frac{\|\delta z_C\|}{\|\delta z_\perp\|}
=(1-\rho)Q_{\mathrm{raw}}.
$$

Thus, C3S-AdamW reduces high-gain common-subspace functional dominance by the factor $1-\rho$.

This proof does not require us to know the exact gains $a$ and $b$. If the common subspace is high-gain, then $Q_{\mathrm{raw}}$ may be large. Suppression reduces it directly.

---

## 18. Main Theorem

**Theorem.** Assume the following conditions hold for a selected parameter block after warmup.

First, the common-feature projector is stable:

$$
\|P_C(t)-P_C(T_0)\|_2\leq\epsilon_C.
$$

Second, already-learned common examples satisfy $p_y\geq q$.

Third, active long-tail examples have nonzero complementary update component:

$$
P_{\perp C}u_t\neq0.
$$

Fourth, the optimizer uses the C3S-AdamW update

$$
\tilde u_t=r_t(I-\rho_tP_C)u_t,
\qquad
\Delta\theta_t=-\eta\tilde u_t-\eta\lambda\theta_t.
$$

Then the following statements hold.

For already-learned common examples, the hard confidence-capped loss gives zero update. Therefore these examples cannot create new motion inside $C$ or inside $P_{\perp C}$. For the smooth cap, the update is multiplied by a small residual gate $r_t$ once $p_y>q$.

For active examples, the common-subspace component of the preconditioned update is multiplied by $1-\rho_t$, while the complementary component is preserved before residual gating:

$$
(I-\rho_tP_C)u_t=(1-\rho_t)P_Cu_t+P_{\perp C}u_t.
$$

Consequently, the common-to-complementary update-energy ratio is multiplied by $(1-\rho_t)^2$:

$$
R_{\mathrm{sup}}=(1-\rho_t)^2R_{\mathrm{raw}}.
$$

Under local linearization, the common-to-complementary functional-effect ratio is multiplied by $1-\rho_t$:

$$
Q_{\mathrm{sup}}=(1-\rho_t)Q_{\mathrm{raw}}.
$$

Therefore, C3S-AdamW simultaneously stops overtraining of learned common features and reduces the high-gain common-subspace attraction experienced by active long-tail updates.

**Proof.** The frozen-projector approximation follows from Proof I. The zero-update claim for learned common examples follows from Proof II. The exact decomposition of the suppressed update follows from Proof III. The energy-ratio reduction follows from Proof IV. The functional-attraction reduction follows from Proof V. Combining these results proves the theorem.

---

## 19. What the Theorem Does Not Claim

The theorem does not claim that C3S-AdamW always improves validation loss. It claims that C3S-AdamW changes the geometry of active updates in a precise way.

The theorem does not claim that the common-feature subspace is automatically stable. Stability must be measured using $\delta(t)$.

The theorem does not claim that the complementary space has enough capacity to learn all long-tail features. This must be tested by long-tail accuracy, rare-pattern recovery, and effective-rank diagnostics.

The theorem does not claim that all natural-language semantics are low-rank or nested in the same way as the synthetic prefix task. The synthetic task isolates one mechanism: nested compositional continuation.

---

## 20. Diagnostics and Falsification Criteria

The first diagnostic is subspace stability:

$$
\delta(t)=\|P_C(t)-P_C(T_0)\|_2.
$$

If $\delta(t)$ is large, the frozen-projector assumption fails.

The second diagnostic is update alignment:

$$
\alpha_t=\frac{\|P_Cu_t\|^2}{\|u_t\|^2+\epsilon}.
$$

This should decrease effectively after C3S suppression for active tail examples.

The third diagnostic is feature top-$k$ mass. For a feature direction $u$ and singular vectors $v_i$,

$$
M_k(u)=\sum_{i=1}^k\langle u,v_i\rangle^2.
$$

Long-tail extra features should show less forced concentration in the frozen common subspace.

The fourth diagnostic is effective rank:

$$
d_{\mathrm{eff}}=\frac{(\operatorname{tr}\Sigma)^2}{\operatorname{tr}(\Sigma^2)}.
$$

Here, $\Sigma$ is a covariance spectrum or singular-value spectrum. If the method works, effective rank should increase or at least avoid further collapse in the representation space used for long-tail learning.

The fifth diagnostic is common-feature preservation. Common-feature accuracy should remain high after confidence capping.

The sixth diagnostic is long-tail improvement. Tail completion accuracy, rare feature recovery, or long-tail loss should improve.

The method is falsified if the frozen common subspace is unstable, if confidence capping harms common-feature retention, or if suppressing $P_Cu_t$ fails to reduce tail-update concentration and fails to improve long-tail recovery.

---

## 21. Practical Efficiency

The expensive operation is not the projection itself. Once $U_C$ is frozen, projection is low-rank:

$$
P_Cu=U_C(U_C^\top u).
$$

If the parameter block has dimension $p$ and the frozen rank is $k$, this costs $O(pk)$ rather than $O(p^2)$.

For a matrix block with hidden-side projector $V_CV_C^\top$, the update correction is

$$
\tilde U=r_t(U-\rho UV_CV_C^\top).
$$

This costs one multiplication by $V_C$ and one multiplication by $V_C^\top$. If $k\ll d$, this is cheap compared with the full training step.

The rank $k$ can be chosen by explained spectral energy:

$$
\frac{\sum_{i=1}^k\sigma_i^2}{\sum_i\sigma_i^2}\geq\gamma.
$$

Here, $\gamma$ can start at $0.8$ or $0.9$.

A reasonable first configuration is:

- warmup until common-prefix accuracy saturates;
- $q=0.9$;
- $\rho_{\max}\in[0.3,0.7]$;
- small but nonzero smoothing temperature $\tau$;
- apply projection first to $W_{\mathrm{out}}$, then to $W_Q$ and $W_K$.

---

## 22. Experimental Plan

The first experiment should use the existing nested-prefix benchmark. Compare standard AdamW with C3S-AdamW using the same architecture, data distribution, and training budget.

The expected result is not only lower loss. The expected result is a change in learning geometry:

$$
\text{common features remain accurate, while tail features become less dependent on the frozen common subspace.}
$$

The core measurements are:

- common-prefix accuracy;
- tail-prefix accuracy;
- subspace drift $\delta(t)$;
- update alignment $\alpha_t$;
- top-$k$ feature mass $M_k(u)$;
- effective rank $d_{\mathrm{eff}}$;
- singular spectra of $W_{\mathrm{out}}$, $W_Q$, $W_K$, and $W_V$.

A stronger second experiment should vary frequency while keeping compositional structure fixed. For example, use common and rare factual patterns with the same template. The prediction is that C3S-AdamW helps rare compositions more than common compositions, because it specifically targets long-tail absorption into the common backbone.

---

## 23. Final Working Theory

The complete theory is:

1. Nested compositional learning creates a temporal order: common prerequisite features are learned before long-tail composed features.
2. Repeated gradients from common features create high-gain singular directions.
3. Later features are attracted to those directions because they give larger first-order loss decrease when aligned with the prediction error.
4. This creates efficient reuse but also feature interference.
5. Projection alone is insufficient because common examples continue to produce cross-entropy gradients and can leak into complementary directions.
6. Confidence capping stops already-learned common examples from continuing to train.
7. Frozen common-subspace suppression redirects active long-tail updates away from the common high-gain backbone.
8. Applying suppression after Adam preconditioning gives a rigorous guarantee on the actual update direction.

In one sentence:

$$
\boxed{
\text{C3S-AdamW preserves the useful common-feature backbone while preventing it from monopolizing later long-tail learning.}
}
$$

