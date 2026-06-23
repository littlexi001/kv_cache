# Common-Direction Causal Experiments: Design

## Objective

Test the proposed causal chain in order:

```text
frequency/shared target/input
-> aligned gradient mode
-> parameter singular mode
-> nested feature occupation
-> representation-parameter positive feedback
-> saturation and long-tail harm
```

The experiments use small controlled models so that every update path can be
measured exactly. A result in this suite supports the mechanism in the stated toy
model; it does not by itself establish that the same mechanism dominates natural
language pretraining.

## Falsifiable conjecture

A high-frequency shared token first creates an aligned gradient mode through its
target and input roles. Once a parameter singular mode forms, nested features
whose errors align with that mode increasingly occupy it. This increases later
updates to the same mode until the relevant errors saturate. Before saturation,
the dominant mode reduces the effective residual space and slows tail learning.

## Physical priors

1. Repeated prediction of one target aligns the output-side gradient.
2. Repeated use of one input token injects a shared hidden component.
3. A larger singular gain amplifies only error components aligned with its output
   singular vector.
4. Cross-entropy gradients decay as the relevant prediction becomes confident.
5. Useful common structure should be retained; the pathology is excessive global
   occupation and interference.

## Mathematical objects mapped to code

- `E`: tied token embedding and output classifier.
- `M = [M1, M2]`: linear map from two concatenated token embeddings to one hidden
  state.
- `h`: context hidden state.
- `G_output`: embedding gradient from the output-classifier role only.
- `G_input`: embedding gradient from the input role only.
- `G_M`: gradient of the context map.
- `sigma_ratio`: first divided by second singular value.
- `top1_energy`: fraction of matrix energy in the first singular value.
- `k_context_cosine`: cosine between `E[K]` and the mean hidden state of examples
  whose target is K.
- `tail_residual_rank`: effective rank after removing the common direction.

## Claim boundary

Stage 1 can establish nucleation in the tied linear-context model. Stage 2 can
establish attraction under a controlled nested topology. Stage 3 can establish
feedback, saturation, and tail harm under explicit interventions. Natural-language
and large-Transformer claims remain out of scope until the same temporal metrics
are collected in a larger model.
