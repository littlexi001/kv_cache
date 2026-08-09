# Blockwise coherent phase transport

## Goal

Retrieve remote semantic evidence at an approximately 2% token budget while
preserving the order geometry inside every retrieved block.

This first runner is a frozen-Qwen3 causal screen: all prefix states and K/V
caches are produced by native RoPE, and only the final query-token attention is
changed. The consumer always uses the original cached Value vectors.

## 1. Exact pre-RoPE block selector

Split the remote history into complete blocks of size (B\in\{16,32\}). For
head (h), define the exact position-free semantic score

$$
u_{h,j}=\frac{q_h^\top k_{h,j}}{\sqrt d}.
$$

The score and semantic anchor of block (b) are

$$
U_{h,b}=\max_{j\in b}u_{h,j},
\qquad
j^*_{h,b}=\arg\max_{j\in b}u_{h,j}.
$$

After reserving the current token, the recent local window, and sink tokens,
the selector purchases as many complete highest-scoring blocks as fit in

$$
K=\lceil 0.02N\rceil.
$$

The realized budget never exceeds (K) and differs from it by fewer than
(B) tokens whenever at least one further complete block is affordable.

## 2. Native score and local-anchor counterfactual

For a cached token at evidence--Query distance \(\delta_j\), define

$$
s_{h,j}(\delta_j)
=
\frac{q_h^\top R(-\delta_j\Omega)k_{h,j}}{\sqrt d}.
$$

Let (D_{\mathrm{loc}}) be the local-window boundary. A selected block is
considered position-suppressed when its semantic anchor would score higher at
that local distance:

$$
g_{h,b}
=
s_{h,j^*}(D_{\mathrm{loc}})
-
s_{h,j^*}(\delta_{j^*})
>0.
$$

## 3. Shared block transport

For a triggered head/block pair, use one shared distance translation

$$
\tau_{h,b}
=
\bigl(\delta_{j^*}-D_{\mathrm{loc}}\bigr)
\mathbf 1[g_{h,b}>0].
$$

Every token (j\in b) is consumed at

$$
\widetilde\delta_{h,j}
=
\delta_j-\tau_{h,b},
\qquad
\widetilde s_{h,j}
=
s_{h,j}(\widetilde\delta_{h,j}).
$$

Because the same \(\tau_{h,b}\) is applied to the entire block,

$$
\widetilde\delta_{h,j}-\widetilde\delta_{h,k}
=
\delta_j-\delta_k,
\qquad j,k\in b.
$$

Thus block-internal relative distances and order are preserved exactly. The
unit tests verify a maximum invariant error of zero in integer distance space.

## 4. Ablations

| Variant | Selector | Consumer |
|---|---|---|
| `selector_only` | Exact pre-RoPE block max | Native post-RoPE score |
| `clipped_consumer` | Same selected blocks | Independently use \(\min(\delta_j,D_{\mathrm{loc}})\); intentionally loses within-block offsets |
| `transport` | Same selected blocks | Triggered shared \(\tau_{h,b}\) |
| `transport_masspreserve` | Same selected blocks | Transport, then match native selected-remote log partition |
| `random_matched` | Same selected blocks | Randomly transport exactly the same number of blocks per layer/head |

The clipped consumer is an **InfLLM-like diagnostic**, not a claim of exact
InfLLM reproduction.

For mass preservation, corrected remote logits use

$$
\widehat s'_{h,j}
=
s'_{h,j}
-
\left[
\log\sum_{r\in\mathcal R_h}e^{s'_{h,r}}
-
\log\sum_{r\in\mathcal R_h}e^{s_{h,r}}
\right].
$$

Consequently, transport redistributes attention inside the selected remote
set without changing its total softmax partition relative to selected local
tokens.

## 5. Outputs

In addition to evidence recall, both-line hit rate, attention mass, Gold PPL,
accuracy, and query time, the runner records:

- realized selected-token fraction;
- selected remote blocks per head;
- targeted and random trigger fractions;
- mean positive anchor suppression;
- mean absolute transport distance;
- maximum block-relative-distance invariant error.

## 6. Run command

```bash
cd /home/fdong/ymluo/projects/qwen3_local_rule_failure_boundary
bash scripts/run_blockwise_phase_transport_gpu67_20260801.sh
```

The launcher uses only server GPUs 6 and 7 and must be started explicitly; it
is not launched by the implementation task.

## Interpretation limit

A positive result would show that semantic block selection plus coherent
phase transport improves this controlled final-query retrieval intervention.
It would not yet establish an end-to-end positional encoding: training-time
integration, multi-token decoding, local-order controls, and natural long-
context benchmarks remain necessary.
