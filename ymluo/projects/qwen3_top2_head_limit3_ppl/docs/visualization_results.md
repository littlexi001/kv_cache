# Visualization Results

This file is updated after the run finishes.

## Plot Contracts

### PPL by attention selection mode

- Question: which attention selection mode has the lowest next-token PPL?
- Metric: perplexity, `exp(mean cross entropy)`.
- Unit: PPL.
- Data source: `ppl_by_mode.csv`.
- X-axis: attention selection mode.
- Y-axis: PPL on evaluation tokens.
- Allowed conclusion: compare behavior of baseline, top2, and top2limit3 on
  this text prefix.
- Limitation: PPL does not explain which token/head decisions caused a change.

### Top2limit3 historical-token load per layer and head

- Question: how many historical token links remain for each layer/head after
  the max-3 rule?
- Metric: final kept historical tokens per query.
- Unit: token links per query.
- Data source: `top2limit3_load_by_head.csv`.
- X-axis: attention head index.
- Y-axis: layer index.
- Color: mean kept historical tokens per query.
- Allowed conclusion: identify load imbalance across layer/head.
- Limitation: does not show whether kept tokens are semantically useful.

### Fraction of original top2 selections kept after limit3

- Question: how much of each head's original top2 load survives the max-3 rule?
- Metric: final kept historical links divided by original top2 historical links.
- Unit: fraction in `[0, 1]`.
- Data source: `top2limit3_load_by_head.csv`.
- X-axis: attention head index.
- Y-axis: layer index.
- Color: kept fraction.
- Allowed conclusion: identify heads/layers where the max-3 rule removes more
  of the top2 choices.
- Limitation: this is a load metric, not a PPL contribution metric.

## Actual Results

Run completed on 2026-06-18.

Setup:

- Model: `ymluo/models/Qwen3-0.6B`.
- Input text:
  `external/needle-in-a-haystack/needlehaystack/PaulGrahamEssays/worked.txt`.
- Prefill tokens: `1024`.
- Evaluation tokens: `512`.
- Chunk size: `128`.
- Top fraction: `0.02`.
- Max heads per historical token: `3`.
- Self-token rule: kept unconditionally.
- Random seed for limit3 head selection: `1234`.

## PPL Results

`ppl_by_mode.csv`:

| mode | loss | PPL | runtime seconds |
|---|---:|---:|---:|
| baseline | 3.490 | 32.793 | 2.012 |
| top2 | 3.568 | 35.448 | 17.216 |
| top2limit3 | 9.828 | 18548.679 | 296.093 |

Read the PPL plots as: baseline is the original attention result; top2 keeps
only each head's top 2% historical tokens plus the current token; top2limit3
starts from top2 and then caps every historical token at 3 selecting heads.

Observed result: top2 is close to baseline. PPL increases from `32.793` to
`35.448`, a `1.081x` ratio. Top2limit3 is much worse: PPL increases to
`18548.679`, which is `523.3x` larger than top2 and `565.6x` larger than
baseline.

This means: on this text prefix, plain top2 historical attention is tolerated
reasonably well, but the random max-3-per-token rule removes important
head-token links. The specific top2limit3 operationalization is falsified as a
drop-in PPL-preserving replacement.

What this does not prove: it does not prove every max-3 rule must fail. This
run used random head choice among the heads that selected the same token.
A score-aware or head-group-aware choice could behave differently.

## Load Results

`top2limit3_load_by_head.csv`:

- Original top2 load is exactly `27.5` historical tokens per query for every
  layer/head on average. This comes from the evaluation range: historical
  context length runs from 1024 to 1535, and 2% gives about 21 to 31 tokens.
- After limit3, the mean final load is `16.60` historical tokens per query.
- The mean removed load is `10.90` historical token links per query.
- The mean kept fraction of original top2 selections is `0.604`.
- Across layer/head rows, kept fraction ranges from `0.407` to `0.924`.
- Lowest-load layer by average head load: layer 17, mean `13.59`.
- Highest-load layer by average head load: layer 20, mean `19.10`.

Read the load heatmap as: each cell is one layer/head. Brighter cells keep more
historical tokens per query after the max-3 rule.

Observed result: load is not uniform. Some layer/head cells keep near `25`
historical tokens per query, while the lowest cells keep near `11`. The same
pattern appears in the kept-fraction heatmap: some heads retain more than `90%`
of their original top2 choices, while some retain only about `41%`.

This means: random limit3 does not merely reduce a uniform amount of duplicate
work. It creates uneven per-head load, and the heads that lose the most links
are plausible contributors to the large PPL degradation.

What this does not prove: load imbalance alone is not a causal explanation of
the PPL increase. The removed links may be important because of token identity,
head role, layer role, or the random selection rule.

## Conclusion

The experiment supports keeping the plain top2 baseline as a meaningful sparse
attention condition, because it only modestly increases PPL on this run. The
experiment does not support the random top2limit3 rule: it removes about `39.6%`
of top2 historical links and makes PPL explode.

The next concrete test is to replace random choice with a deterministic rule.
The most direct next rule is: when a token is selected by more than 3 heads,
keep the 3 heads with the largest pre-softmax scores for that token instead of
choosing randomly.

## Iteration 2: score-based head choice and automatic max-head sweep

The random top2limit3 rule failed, so the next operationalization changed only
the conflict resolution rule:

```text
top2limit3score:
  if one historical token is selected by more than 3 heads,
  keep the 3 selecting heads with the largest pre-softmax scores for that token.
```

This also failed:

| mode | PPL | PPL / top2 |
|---|---:|---:|
| top2 | 35.448 | 1.000 |
| top2limit3score | 21146.067 | 596.538 |

Failure interpretation: the PPL collapse is not mainly caused by random head
choice. Even when the highest-score three heads are kept, limiting a shared
historical token to only 3 heads removes too many important head-token links.

Because score-based max-3 still failed, I swept larger max-head limits:

| mode | mean kept fraction of original top2 links | PPL | PPL / top2 |
|---|---:|---:|---:|
| top2limit3score | 0.565 | 21146.067 | 596.538 |
| top2limit4score | 0.674 | 4902.904 | 138.313 |
| top2limit6score | 0.808 | 1463.280 | 41.280 |
| top2limit8score | 0.899 | 668.211 | 18.850 |
| top2limit12score | 0.978 | 80.424 | 2.269 |
| top2limit13score | 0.986 | 61.285 | 1.729 |
| top2limit14score | 0.993 | 48.604 | 1.371 |
| top2limit15score | 0.997 | 38.869 | 1.097 |
| top2limit16score | 1.000 | 35.448 | 1.000 |

Plot:
`outputs/score_sweep_combined/ppl_vs_max_heads_score_logy.png`

Read this plot as: each point is a score-based cap. The x-axis is the maximum
number of heads allowed to keep the same historical token. The y-axis is PPL on
a log scale.

Observed result: PPL drops smoothly as the cap increases. Caps of 3, 4, 6, and
8 all collapse. Cap 12 is still more than 2x worse than top2. Cap 15 is the
first tested cap that is close to top2: PPL `38.869`, which is `1.097x` top2.

This means: the model is very sensitive to removing even a small number of
duplicate high-score head-token links. To avoid collapse under this family of
rules, the cap must be weak, near 15 out of 16 heads.

## Iteration 3: score-based cap with per-head fill

One possible failure explanation was that max3 hurt PPL because it reduced each
head's number of kept historical tokens. To test that, I added:

```text
top2limit3scorefill:
  first keep the highest-score 3 heads for each over-selected token;
  then, for each head, fill removed slots with the next highest-score historical
  tokens that are still below the max-3 cap.
```

This keeps the per-head historical-token load equal to original top2:

| metric | value |
|---|---:|
| original top2 kept per query/head | 27.5 |
| final kept per query/head | 27.5 |
| kept fraction of original top2 count | 1.0 |
| PPL | 4589.929 |
| PPL / top2 | 129.483 |

Failure interpretation: preserving each head's load is not enough. The identity
of the original shared head-token links matters. Replacing removed links with
next-best available tokens still destroys PPL.

## Updated Conclusion

The best tested non-collapsing point is `top2limit15score`, not max3. It removes
only `0.28%` of original top2 links on average and raises PPL from `35.448` to
`38.869`.

This updates the conjecture:

```text
Old conjecture:
  cross-head duplicate token selections are mostly redundant.

Updated conjecture:
  many duplicate selections are not redundant for PPL. The model can tolerate
  plain top2 pruning, but it cannot tolerate forcing a highly shared historical
  token down to only 3 selecting heads, even with score-aware selection and
  per-head load refill.
```

The next concrete direction should not be a hard per-token cap of 3. A safer
next rule would be token-specific: only cap low-attention-energy duplicate links,
or apply the cap only when the removed heads have very small score gaps from
their replacement tokens.

## Failure Decomposition: why max3 collapses

I added an offline diagnostic that does not change model outputs. It takes the
original full-attention weights, reconstructs the top2 historical selections,
then asks what a score-based cap would remove.

Diagnostic files:

- `outputs/removal_diagnostics/removed_weight_summary_by_cap.csv`
- `outputs/removal_diagnostics/removed_weight_by_layer_cap.csv`
- `outputs/removal_diagnostics/removed_link_vs_weight_fraction_by_cap.png`

Main result:

| cap | removed top2 link fraction | removed top2 attention-weight fraction |
|---:|---:|---:|
| 3 | 0.355 | 0.482 |
| 8 | 0.102 | 0.176 |
| 12 | 0.025 | 0.051 |
| 15 | 0.003 | 0.006 |

Read this as: cap3 removes `35.5%` of original top2 head-token links, but those
removed links carry `48.2%` of the original top2 attention weight. The removed
links are not a low-weight tail. They are disproportionately high-weight
connections.

Layer-level evidence also supports this interpretation. Under cap3, layer 27
loses `69.1%` of its original top2 attention weight, while layer 2 loses only
`17.1%`. The failure is therefore not uniform; some layers depend heavily on
tokens selected by many heads.

Updated causal path:

```text
PPL collapse
-> not caused mainly by random selection, because score top3 also collapses
-> not caused mainly by fewer tokens per head, because scorefill restores
   per-head load but still collapses
-> verified bottleneck: hard caps delete high-attention-weight duplicate
   head-token links
```

## Iteration 4: score-gap rule

The next tested rule avoids deleting near-tied high-score heads:

```text
top2limit3gapM:
  for each over-selected historical token,
  always keep the 3 highest-score selected heads;
  also keep any additional selected head whose score is within M of the
  3rd-highest selected head score.
```

This rule still has a floor of 3 heads, but it avoids deleting heads that are
close to the kept heads by pre-softmax score.

Results:

| mode | kept fraction of original top2 links | removed links per head/query | PPL | PPL / top2 |
|---|---:|---:|---:|---:|
| top2limit3gap0p5 | 0.621 | 10.434 | 7873.961 | 222.127 |
| top2limit3gap1p0 | 0.680 | 8.805 | 4548.596 | 128.317 |
| top2limit3gap2p0 | 0.772 | 6.276 | 2488.050 | 70.189 |
| top2limit3gap4p0 | 0.903 | 2.662 | 213.291 | 6.017 |
| top2limit3gap5p0 | 0.943 | 1.558 | 96.550 | 2.724 |
| top2limit3gap6p0 | 0.967 | 0.902 | 60.987 | 1.720 |
| top2limit3gap7p0 | 0.981 | 0.528 | 45.329 | 1.279 |
| top2limit3gap8p0 | 0.989 | 0.294 | 37.011 | 1.044 |

Plot:

- `outputs/final_fine_rules/gap_rule_ppl_vs_margin_logy.png`
- `outputs/final_fine_rules/fine_rules_ppl_comparison_logy.png`

Read this plot as: the x-axis is the allowed score margin below the third
selected head. Larger margin means fewer links are removed. The y-axis is PPL on
a log scale.

Observed result: the margin has a sharp effect. Margins `0.5`, `1`, and `2`
still collapse. Margin `4` is improved but still unusable. Margin `8` is the
best tested point: PPL `37.011`, only `1.044x` top2, while deleting more links
than top2limit15score.

This means: a useful rule must preserve near-tied high-score duplicate links.
The model can tolerate deleting only the clearly lower-score duplicate links.

## Iteration 5: protect sink and recent tokens before top3-head limiting

New conjecture: the hard top3-head cap fails because it deletes cross-head
copies of attention sink tokens and very recent tokens. The tested
operationalization is:

```text
top2limit3protectsSrR:
  start from each head's top2 historical tokens;
  protect the first S historical token positions for every query;
  protect the most recent R% historical token positions for every query;
  protected tokens keep all originally selected heads;
  every other over-selected token keeps only the 3 selected heads with largest
  pre-softmax score.
```

Here `top2limit3protects64r16p0` means: protect the first 64 historical token
positions and the most recent 16% historical token positions; all other shared
tokens are limited to the best 3 heads.

Main result table:

| mode | PPL | PPL / top2 | kept fraction of original top2 links | removed links per head/query |
|---|---:|---:|---:|---:|
| top2 | 35.448 | 1.000 |  |  |
| top2limit3score | 21146.067 | 596.538 | 0.565 | 11.965 |
| top2limit3protects0r1p0 | 10981.795 | 309.800 | 0.597 | 11.094 |
| top2limit3protects32r1p0 | 4756.548 | 134.184 | 0.629 | 10.197 |
| top2limit3protects64r8p0 | 64.375 | 1.816 | 0.881 | 3.286 |
| top2limit3protects64r10p0 | 42.351 | 1.195 | 0.935 | 1.781 |
| top2limit3protects64r12p0 | 36.964 | 1.043 | 0.952 | 1.322 |
| top2limit3protects64r14p0 | 36.768 | 1.037 | 0.959 | 1.132 |
| top2limit3protects64r16p0 | 36.462 | 1.029 | 0.961 | 1.064 |
| top2limit3protects32r16p0 | 36.605 | 1.033 | 0.961 | 1.067 |
| top2limit3gap8p0 | 37.011 | 1.044 | 0.989 | 0.294 |

Plots:

- `outputs/protect_sink_recent_combined/recent_percent_threshold_ppl_logy.png`
- `outputs/protect_sink_recent_combined/combined_kept_fraction_vs_ppl_logy.png`

Read the threshold plot as: each line fixes the number of protected sink
positions and varies the protected recent-token fraction. The y-axis is PPL on a
log scale; lower is better. The dashed line is top2 PPL, and the dotted line is
the previous score-gap rule `top2limit3gap8p0`.

Observed result: protecting only the most recent `1%` is not enough. It still
leaves PPL in the thousands. The useful transition happens near `10%` to `12%`
recent protection. With sink `64`, recent `10%` gives PPL `42.351`; recent
`12%` gives PPL `36.964`; recent `16%` gives PPL `36.462`.

Interpretation: the sink/recent prior is partly correct, but the important
"recent" band is much wider than `1%` on this run. The model is not only using
the immediately previous few tokens; it is sensitive to a broader local
history band when top2 links are shared by many heads.

## Current Best Rule

The best tested rule is now:

```text
top2limit3protects64r16p0
```

Compared with top2:

- PPL changes from `35.448` to `36.462`.
- PPL ratio is `1.029`.
- It keeps `96.13%` of original top2 links.
- It removes `1.064` top2 links per head/query on average.

This is better than the previous best `top2limit3gap8p0` on this one text
window: `36.462` versus `37.011`. It also removes more links than gap8, so it is
a better compression point in this run.

## Next Uncertainty

This experiment supports protecting a broad recent band, but the `16%` value is
still tied to this sequence length and text. The next concrete test should use
an adaptive definition:

```text
protect sink tokens;
protect recent tokens until the remaining removable top2 links have low
full-attention weight or large score gap from the third selected head.
```

That would combine the sink/recent prior with the previous score-gap evidence,
instead of using a fixed raw score margin or a fixed recent-token percentage.

## Iteration 6: position distribution by number of selecting heads

Question: when a historical token is selected by exactly `c` heads in the
original top2 rule, where is that token located?

Operational definition:

```text
For each layer and query:
  each head selects top ceil(0.02 * history_count) historical tokens;
  for each historical token j, count how many heads selected j;
  group token j by selected_head_count = 1..16;
  record relative position = j / (history_count - 1).
```

`relative position = 0` means sequence start / sink side. `relative position =
1` means close to the current query. The experiment used the same 1024-token
prefill and 512-token evaluation window as the PPL tests.

Data and plots:

- `outputs/head_count_position_distribution/head_count_position_summary.csv`
- `outputs/head_count_position_distribution/relative_position_distribution_by_head_count.png`
- `outputs/head_count_position_distribution/sink_recent_fraction_by_head_count.png`
- `outputs/head_count_position_distribution/relative_position_quantiles_by_head_count.png`
- `outputs/head_count_position_distribution/selected_token_cases_by_head_count.png`

Main numbers:

| selected heads | token cases | median relative position | median distance to query | key < 64 | recent 1% | recent 8% | recent 16% |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1212248 | 0.705 | 385 | 0.039 | 0.006 | 0.185 | 0.333 |
| 2 | 388868 | 0.817 | 237 | 0.026 | 0.022 | 0.316 | 0.471 |
| 4 | 99393 | 0.946 | 70 | 0.008 | 0.105 | 0.583 | 0.713 |
| 8 | 34429 | 0.987 | 18 | 0.001 | 0.398 | 0.858 | 0.909 |
| 12 | 19242 | 0.995 | 8 | 0.000 | 0.743 | 0.977 | 0.987 |
| 14 | 13374 | 0.997 | 5 | 0.012 | 0.851 | 0.983 | 0.985 |
| 15 | 11331 | 0.996 | 6 | 0.157 | 0.761 | 0.841 | 0.842 |
| 16 | 17690 | 0.000 | 1123 | 0.613 | 0.367 | 0.387 | 0.387 |

Read the count plot as: most selected historical token cases are selected by
only one head. Exact numbers: `54.5%` of selected token cases are selected by
one head, `17.5%` by two heads, and `0.8%` by all 16 heads.

Read the relative-position heatmap as: for each selected-head-count group, the
color column sums to 1. It shows where tokens in that group are located. For
counts `1..14`, higher head count moves strongly toward the query-near end. For
count `16`, the distribution becomes bimodal: a large sink component at the
start and a smaller query-near component.

Observed result:

- Low overlap (`1` or `2` heads) is broad and not especially recent.
- Medium/high overlap (`8..14` heads) is mostly recent. For `12` heads, `98.7%`
  of cases are within the most recent `16%` of history.
- All-head overlap (`16` heads) is special. It is not simply "even more recent":
  `61.3%` of those cases are in the first 64 token positions, while `38.7%` are
  in the most recent `16%`.

This supports the previous protect-rule result. The top2 duplicate links that
many heads share are concentrated in two regions: broad recent context and true
sink positions. The exact all-16 case is mostly sink, while counts `8..14` are
mostly recent.

What this does not prove: the plot does not prove that every sink/recent link
is necessary for PPL. It only shows where the shared top2 selections are. The
PPL experiments show that deleting many of those shared links is harmful.
