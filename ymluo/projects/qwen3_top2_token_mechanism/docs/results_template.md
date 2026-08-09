# Results Template

## Run identity

- model:
- model revision/hash:
- text/dataset:
- text hash:
- prefill/eval tokens:
- chunk size:
- dtype/device:
- git commit:

## Sanity checks

- [ ] Top-attention 100% delta NLL vs full is approximately zero.
- [ ] `sink_recent_s0` and `recent` are token-wise identical.
- [ ] Scored token indices match for every mode.
- [ ] Actual keep ratios match the nominal budget except explicit drop ablations.

## Question 1: Why is 2% good?

- best ratio by PPL:
- delta NLL of 2% vs full, 95% CI:
- mean/median Top-2% attention mass:
- effective support fraction:
- cutoff gap:
- layer/head exceptions:
- does random or bottom attention reproduce the gain?:

## Question 2: What are the Top-2% tokens?

- sink/recent/remote event shares:
- top lexical enrichments after exposure correction:
- top tokens by selection event count:
- top tokens by attention mass:
- temporal persistence / head sharing notes:
- evidence-span overlap, if task labels exist:

## Question 3: Does equal-budget sink + recent match Top-2%?

- best sink allocation:
- PPL ratio vs Top-2%:
- paired delta NLL and 95% CI:
- equivalence decision at ±0.01 nat/token:
- oracle position recall:
- oracle mass recall:
- pruned-distribution cosine:
- drop-sink/recent/remote ablation interpretation:

## Generalization

| Model | Dataset | Context | Best ratio | sink+recent equivalent? | Notes |
| --- | --- | ---: | ---: | --- | --- |
|  |  |  |  |  |  |

## Final claim

State the narrowest claim supported by all runs. Separate:

1. behavior (PPL/task score),
2. selector overlap,
3. mechanism inference,
4. deployability.

