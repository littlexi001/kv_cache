# Common-Residual Split Attention: Method, Symbols, and First Results

## 0. Conclusion

In the conflict-free synthetic causal attention task, the common-residual split
method works: it reduces long-tail stable convergence from about `1256` steps in
the dense baseline to about `152-168` steps.

The main useful operation is the hard split:

$$
y_k = \alpha W_c P_{c,k}h_k + W_r P_{\perp,k}h_k
$$

where the common direction for position $k$ is estimated only from prefix tokens
$1,\ldots,k-1$. This avoids future-token leakage.

The current result supports the mechanism-level claim:

> splitting the input representation into a common component and a residual
> component, and sending the two components through separate parameters, can
> prevent the long-tail path from being dominated by the common high-gain
> direction.

This is not yet a real-LLM result. It is a first positive result in a
single-layer residual causal attention toy model.

## 1. Two clarification questions

### 1.1 What happens if the data has label conflict?

Label conflict means that the same causal prefix has multiple next-token labels.
For example, in the first draft of the synthetic data, the prefix containing
only token $K$ had different valid next tokens:

$$
K \rightarrow A,\quad K \rightarrow B,\quad K \rightarrow C,\quad K \rightarrow D
$$

In that case, a deterministic causal language model cannot reach 100% next-token
accuracy for that prefix, because the input is identical but the target differs.

This has two consequences:

1. `tail_accuracy == 1.0` is no longer a valid convergence criterion.
2. Failure to reach full tail accuracy does not prove the model failed to learn
   long-tail structure. It may only reflect an impossible label mapping.

That is what happened in the first CRS run. All variants saturated around
`0.91-0.93` tail accuracy. The reason was not necessarily long-tail failure; the
dataset contained a causal prefix conflict.

Therefore the final reported experiment uses a conflict-free causal cycle. In
that setting, full tail accuracy is a valid convergence metric.

### 1.2 Where does $u_{c,k}$ come from?

$u_{c,k}$ is not a learned router and not an oracle label. It is a causal
estimate of the common direction in the input representation at position $k$.

Given the hidden states in one sequence:

$$
h_1,h_2,\ldots,h_L
$$

the common direction for position $k$ is:

$$
u_{c,k}
=
\frac{\sum_{i<k}h_i}
{\left\|\sum_{i<k}h_i\right\|}
$$

Only prefix states $h_i$ with $i<k$ are used. The current token $h_k$ and all
future tokens $h_{k+1},\ldots,h_L$ are not used. This is required in causal LM
training and inference.

For the first position, there is no prefix. In the implementation the prefix
sum is zero, so the common projection is zero and the token goes through the
residual path.

In code, the direction is computed with stop-gradient:

$$
u_{c,k}=\mathrm{stopgrad}\left(
\frac{\sum_{i<k}h_i}
{\left\|\sum_{i<k}h_i\right\|}
\right)
$$

This prevents the model from learning to manipulate the direction-estimation
procedure itself.

## 2. Method definition

### 2.1 Original dense linear map

For a normal linear layer:

$$
y_k = Wh_k
$$

where:

- $h_k\in\mathbb{R}^d$ is the input representation at position $k$;
- $W\in\mathbb{R}^{d_{\mathrm{out}}\times d}$ is the parameter matrix;
- $y_k$ is the output representation.

### 2.2 Common-residual split

For each position $k$, estimate a causal common direction:

$$
u_{c,k}
=
\frac{\sum_{i<k}h_i}
{\left\|\sum_{i<k}h_i\right\|}
$$

Then define the common projection:

$$
P_{c,k}=u_{c,k}u_{c,k}^{\top}
$$

and the residual projection:

$$
P_{\perp,k}=I-P_{c,k}
$$

The representation is split as:

$$
h_{c,k}=P_{c,k}h_k
$$

$$
h_{r,k}=P_{\perp,k}h_k
$$

The original parameter matrix $W$ is replaced by two parameter matrices:

- $W_c$: common branch parameter;
- $W_r$: residual / long-tail branch parameter.

The new linear operation is:

$$
y_k = \alpha W_c h_{c,k} + W_r h_{r,k}
$$

equivalently:

$$
y_k = \alpha W_c P_{c,k}h_k + W_r P_{\perp,k}h_k
$$

where $\alpha$ is a fixed common-branch scale. The experiment tests:

- $\alpha=1$: hard split without common suppression;
- $\alpha=0.5$: hard split with common branch suppression.

### 2.3 Why this should help long-tail learning

In the dense model, the long-tail margin can be written in the simplified form:

$$
m_T=b\sigma_t\tau-a\sigma_c\rho
$$

where:

- $b\sigma_t\tau$ is the useful long-tail direction contribution;
- $a\sigma_c\rho$ is the wrong common-logit contribution on a tail sample;
- $\rho$ is the common projection inside the tail representation;
- $\sigma_c$ is the common direction gain.

The split method targets both terms in the harmful product:

1. representation split reduces the common projection seen by the residual
   branch:

$$
\rho \rightarrow \rho'
$$

2. parameter split prevents the residual branch from sharing the common
   high-gain parameter channel:

$$
\sigma_c \rightarrow \sigma_c^{(r)}
$$

The desired tail margin becomes:

$$
m_T^{\mathrm{CRS}}
=
b\sigma_t^{(r)}\tau
-
a\sigma_c^{(r)}\rho'
$$

If the split works, then:

$$
\rho'\ll \rho
$$

and:

$$
\sigma_c^{(r)}\ll \sigma_c
$$

so the common direction hurts tail margin less.

## 3. Experiment design

### 3.1 Data

The final reported experiment uses a conflict-free synthetic causal sequence
task.

There is one shared common token:

$$
K
$$

There are four groups:

$$
A,B,C,D
$$

Group $A$ is high-frequency:

$$
p_A=0.70
$$

Groups $B,C,D$ are tails:

$$
p_B=p_C=p_D=0.10
$$

Each group has three group-specific tokens:

$$
i_0,i_1,i_2
$$

Each group follows the same causal cycle:

$$
[i_0,i_1,K,i_2,i_0,i_1,K,i_2,i_0]
$$

This makes $K$ both a frequent target and a frequent input, while preserving a
unique next token for every causal prefix. Therefore full tail accuracy is a
valid convergence metric.

### 3.2 Model

The model is tied-embedding single-layer residual causal attention.

Input embedding:

$$
x=E[\mathrm{tokens}]
$$

RMSNorm:

$$
\tilde{x}=\mathrm{RMSNorm}(x)
$$

Causal attention:

$$
q=W_q\tilde{x},\quad k=W_k\tilde{x},\quad v=W_v\tilde{x}
$$

$$
a=\mathrm{CausalAttention}(q,k,v)
$$

Output projection:

$$
o=W_oa
$$

Residual stream:

$$
h=x+o
$$

Tied output logits:

$$
z=hE^\top
$$

For CRS variants, the Q/K/V/O projections are replaced by:

$$
W_{\{\cdot\}}h_k
\quad\rightarrow\quad
\alpha W_{\{\cdot\},c}P_{c,k}h_k
+W_{\{\cdot\},r}P_{\perp,k}h_k
$$

There is no softmax router and no learned MoE gate in this experiment.

### 3.3 Loss

The main dense and CRS variants use weighted plain cross-entropy:

$$
L=\sum_i w_i\mathrm{CE}(z_i,y_i)
$$

The sequence weights implement the Zipf group distribution.

There is also a dense reweighting reference. It uses inverse-square-root
target-frequency reweighting and is included only as a known useful baseline.

### 3.4 Compared variants

1. `dense`: standard single-layer residual causal attention.
2. `dense_reweight`: dense model with loss reweighting reference.
3. `crs_alpha1`: CRS split with $\alpha=1$.
4. `crs_alpha05`: CRS split with $\alpha=0.5$.

The experiment uses five seeds:

$$
0,1,2,3,4
$$

Training length:

$$
5000\ \mathrm{steps}
$$

## 4. Results

Artifacts:

- `fdong_embedding_dim/common_direction_experiments/run_crs_split_attention.py`
- `fdong_embedding_dim/outputs/crs_split_attention_cycle/summary.json`
- `fdong_embedding_dim/outputs/crs_split_attention_cycle/aggregate.csv`
- `fdong_embedding_dim/outputs/crs_split_attention_cycle/crs_split_attention.png`

### 4.1 Final summary

| variant | mean first stable tail-accuracy step | final tail loss | final tail accuracy | final Bqk $\sigma_1/\sigma_4$ |
|---|---:|---:|---:|---:|
| dense | 1256 | 0.00897 | 1.000 | 492 |
| dense_reweight | 1100 | 0.0442 | 0.978 | 1059 |
| crs_alpha1 | 152 | 0.0000156 | 1.000 | 359 |
| crs_alpha05 | 168 | 0.0000082 | 1.000 | 241 |

Both CRS variants substantially accelerate tail convergence.

### 4.2 Early training behavior

At step 100:

| variant | tail loss | tail accuracy | common loss | Bqk $\sigma_1/\sigma_4$ |
|---|---:|---:|---:|---:|
| dense | 0.955 | 0.578 | 0.384 | 3104 |
| dense_reweight | 0.915 | 0.600 | 0.419 | 4348 |
| crs_alpha1 | 0.290 | 0.933 | 0.106 | 218 |
| crs_alpha05 | 0.383 | 0.911 | 0.0638 | 5125 |

At step 500:

| variant | tail loss | tail accuracy | common loss | Bqk $\sigma_1/\sigma_4$ |
|---|---:|---:|---:|---:|
| dense | 0.261 | 0.889 | 0.0556 | 4781 |
| dense_reweight | 0.352 | 0.844 | 0.143 | 1239 |
| crs_alpha1 | 0.00109 | 1.000 | 0.00066 | 104 |
| crs_alpha05 | 0.00195 | 1.000 | 0.00030 | 173 |

The CRS improvement appears early, not only at the final checkpoint.

### 4.3 Interpretation

The result supports the proposed mechanism:

1. CRS estimates a causal common direction from the prefix.
2. CRS projects each input representation into common and residual parts.
3. Common and residual parts go through separate parameter matrices.
4. The residual branch can learn tail transitions without sharing the same
   high-gain common parameter channel.

This is why tail accuracy reaches 100% much earlier.

The common-branch scale $\alpha$ is not the main reason for the speedup:

- $\alpha=1$ reaches stable tail accuracy at about `152` steps;
- $\alpha=0.5$ reaches stable tail accuracy at about `168` steps.

However, $\alpha=0.5$ gives a flatter final Bqk spectrum:

$$
\sigma_1/\sigma_4 = 241
$$

compared with:

$$
\sigma_1/\sigma_4 = 359
$$

for $\alpha=1$.

Therefore the current interpretation is:

> hard common-residual split is the main cause of faster tail convergence;
> common branch suppression may help spectral flattening, but is not required
> for the observed speedup in this toy task.

## 5. Claim boundary

This experiment supports CRS in a controlled synthetic setting. It does not yet
prove that the same method improves a real LLM.

The result depends on the following choices:

1. conflict-free causal labels;
2. low-dimensional tied embedding;
3. single-layer residual causal attention;
4. Q/K/V/O all replaced by CRS projections;
5. causal prefix-mean estimate of the common direction.

The next checks should be:

1. ablate which projections need CRS: Q only, K only, V only, O only, QK only;
2. compare prefix mean versus EMA prefix mean;
3. test rank-$k$ common subspace instead of rank-1;
4. test whether CRS still helps when the task has richer tail structure;
5. test whether CRS can be implemented inside a real Transformer block without
   harming normal language modeling loss.

