# Viewer Audit

Independent review: not used. The available sub-agent tool may only be used
when the user explicitly asks for delegation, so this run uses manual audit.

## PPL by attention selection mode

- Status: pass.
- Metric definition: perplexity, `exp(mean next-token cross entropy)`.
- Unit: PPL.
- Axis check: pass. X-axis is attention selection mode; y-axis is PPL. A
  log-scale PPL plot and a loss plot were added because linear PPL compresses
  baseline and top2 when top2limit3 is very large.
- Legend check: pass. No legend is needed because modes are x-axis labels.
- Unit compatibility check: pass. Each plot has one y-axis unit.
- Visible-number check: pass. Bars are labeled with PPL or loss values, and
  `visualization_results.md` records the table.
- Reader-confusion risk: low after adding log PPL and loss plots.
- Fix applied: added `ppl_by_mode_logy.png` and `loss_by_mode.png`.
- Allowed conclusion: top2 is close to baseline on this run, while top2limit3
  has much worse PPL.
- Remaining uncertainty: PPL does not identify which removed links caused the
  degradation.

## Top2limit3 historical-token load per layer and head

- Status: pass.
- Metric definition: mean final kept historical tokens per query after the
  max-3 rule.
- Unit: token links per query.
- Axis check: pass. X-axis is attention head index, y-axis is layer index, and
  colorbar is mean kept historical tokens per query.
- Legend check: pass. The colorbar gives the metric and unit.
- Unit compatibility check: pass. The heatmap uses one unit.
- Visible-number check: pass. The result text records mean `16.60`, min cell
  near `11.18`, and max cell near `25.41`.
- Reader-confusion risk: low. The title says top2limit3 and historical-token
  load.
- Fix applied: none after visual inspection.
- Allowed conclusion: the random max-3 rule creates uneven layer/head load.
- Remaining uncertainty: the heatmap does not say whether high- or low-load
  heads are more important for PPL.

## Fraction of original top2 selections kept after limit3

- Status: pass.
- Metric definition: final kept historical links divided by original top2
  historical links.
- Unit: fraction.
- Axis check: pass. X-axis is attention head index, y-axis is layer index, and
  colorbar is final kept divided by original top2 kept.
- Legend check: pass. The colorbar states the fraction.
- Unit compatibility check: pass. One fraction unit is used.
- Visible-number check: pass. The result text records mean `0.604`, minimum
  `0.407`, and maximum `0.924`.
- Reader-confusion risk: low. The title and colorbar both state that the metric
  is a fraction of original top2 selections.
- Fix applied: none after visual inspection.
- Allowed conclusion: the max-3 rule removes very different fractions from
  different heads/layers.
- Remaining uncertainty: this plot does not show whether removed links are
  high-score or low-score inside the over-selected token group.

## Score-based limit rule: PPL versus max heads per token

- Status: pass.
- Metric definition: PPL for score-based caps where each historical token may
  be kept by at most `n` heads, and the kept heads are the highest-score
  selecting heads.
- Unit: PPL on a log scale.
- Axis check: pass. X-axis is max heads per historical token; y-axis is PPL on
  evaluation tokens.
- Legend check: pass. Dashed reference lines identify baseline and top2 PPL.
- Unit compatibility check: pass. One PPL unit is shown.
- Visible-number check: pass. Each point is labeled with its PPL value.
- Reader-confusion risk: low. The title states score-based limit rule, and the
  x-axis states max heads per token.
- Fix applied: used log y-axis because low caps produce PPL values orders of
  magnitude larger than top2.
- Allowed conclusion: score-based caps below 12 collapse; cap 15 is close to
  top2 on this run.
- Remaining uncertainty: this plot does not explain which token/head links
  cause the PPL increase.

## PPL versus retained top2-link fraction

- Status: pass.
- Metric definition: x-axis is the mean fraction of original top2 head-token
  links kept after the score-based cap; y-axis is PPL.
- Unit: x-axis fraction, y-axis PPL on a log scale.
- Axis check: pass. Both axes state the unit and meaning.
- Legend check: pass. The dashed reference line identifies top2 PPL; point
  labels give the max-head cap.
- Unit compatibility check: pass. Separate axes use separate units.
- Visible-number check: pass. The result table in `visualization_results.md`
  records the exact kept fractions and PPL values.
- Reader-confusion risk: low.
- Fix applied: point labels are max-head caps so the plot can be read without
  source code.
- Allowed conclusion: PPL is highly sensitive to deleting even a small fraction
  of high-overlap top2 links.
- Remaining uncertainty: retained fraction alone does not identify whether a
  removed link is semantically important.

## Iterated methods: PPL comparison

- Status: pass.
- Metric definition: PPL for selected methods from the random, score, and
  score-fill iterations.
- Unit: PPL on a log scale.
- Axis check: pass. X-axis lists methods; y-axis is PPL.
- Legend check: pass. Dashed lines identify baseline and top2.
- Unit compatibility check: pass. One PPL unit is used.
- Visible-number check: pass. Each bar is labeled with its PPL value.
- Reader-confusion risk: medium. Abbreviated labels such as `limit15` require
  the nearby result text to define them as score-based caps.
- Fix applied: result text defines the methods; log scale prevents large
  failure cases from hiding the non-collapsing points.
- Allowed conclusion: max3 score and max3 score-fill both fail; max15 score is
  the best tested non-collapsing cap.
- Remaining uncertainty: it does not test token-specific or score-gap-specific
  rules.

## Removed link fraction versus removed attention-weight fraction

- Status: pass.
- Metric definition: for each score-based cap, x-axis is the fraction of
  original top2 historical head-token links removed; y-axis is the fraction of
  original full-attention probability mass, restricted to those original top2
  links, removed by the cap.
- Unit: fraction on both axes.
- Axis check: pass. Both axes explicitly state removed fraction and distinguish
  links from top2 attention weight.
- Legend check: pass. Point labels identify the max-head cap.
- Unit compatibility check: pass. Both axes are fractions but represent
  different denominators, and the labels make this distinction visible.
- Visible-number check: pass. The exact cap-level values are recorded in
  `visualization_results.md`.
- Reader-confusion risk: low after documenting that this diagnostic is offline
  and does not change model outputs.
- Fix applied: none after visual inspection.
- Allowed conclusion: small caps remove a larger share of top2 attention weight
  than their link fraction, so they are deleting important overlapping links,
  not just low-weight duplicates.
- Remaining uncertainty: attention weight is measured under original full
  attention, not after the pruned model renormalizes attention.

## Score-gap rule: PPL versus margin

- Status: pass.
- Metric definition: PPL for `top2limit3gapM`, where each over-selected token
  keeps the top 3 selected heads and any additional selected heads within
  margin `M` of the third-best score.
- Unit: PPL on a log scale.
- Axis check: pass. X-axis is score margin; y-axis is PPL.
- Legend check: pass. Dashed reference lines identify baseline and top2 PPL.
- Unit compatibility check: pass. One PPL unit is shown.
- Visible-number check: pass. Each point is labeled with its PPL value.
- Reader-confusion risk: medium. Raw pre-softmax score margins are model-scale
  dependent, so the result text records the exact operational definition.
- Fix applied: used log y-axis because small margins still produce very large
  PPL.
- Allowed conclusion: larger score-gap margins reduce collapse; margin `8.0`
  is close to top2 on this run.
- Remaining uncertainty: a raw score margin may not transfer across layers,
  prompts, or models without normalization.

## Fine rules: PPL comparison

- Status: pass.
- Metric definition: PPL for baseline, top2, hard score caps, score-fill, and
  score-gap variants.
- Unit: PPL on a log scale.
- Axis check: pass. X-axis lists methods; y-axis is PPL.
- Legend check: pass. Dashed lines identify baseline and top2.
- Unit compatibility check: pass. One PPL unit is used.
- Visible-number check: pass. Each bar is labeled with its PPL value, and
  `fine_rule_summary.csv` records exact values.
- Reader-confusion risk: low. Method labels are defined in
  `visualization_results.md` and README.
- Fix applied: log scale keeps both collapse and near-top2 methods visible.
- Allowed conclusion: `top2limit3gap8p0` is the best tested max-3-derived rule;
  it is much safer than hard max3 and score-fill, while `top2limit15score` is a
  separate loose-cap comparison point.
- Remaining uncertainty: this plot is one text/model-window setting and should
  be validated on more prompts before claiming generality.

## Sink/recent protection threshold plot

- Status: pass.
- Metric definition: PPL for `top2limit3protectsSrR`, where protected sink and
  recent historical tokens keep all original top2-selected heads, and all other
  over-selected historical tokens keep only the 3 highest-score selected heads.
- Unit: PPL on a log scale.
- Axis check: pass. X-axis is the protected recent historical-token percentage;
  y-axis is PPL on evaluation tokens.
- Legend check: pass. Each line identifies a fixed sink-token count, and
  horizontal reference lines identify top2 and score-gap PPL.
- Unit compatibility check: pass. One y-axis unit is used.
- Visible-number check: pass. Exact values are recorded in
  `combined_protect_sink_recent_summary.csv` and summarized in
  `visualization_results.md`.
- Reader-confusion risk: low. The mode definition is documented before the
  result table.
- Fix applied: used log y-axis because low recent protection gives PPL in the
  thousands while the best points are near top2.
- Allowed conclusion: protecting only recent `1%` is insufficient; the useful
  transition is around `10%` to `12%`, and recent `16%` with sink protection is
  close to top2 on this run.
- Remaining uncertainty: the threshold may depend on sequence length, text, and
  model.

## Combined kept-fraction versus PPL for protect rules

- Status: pass.
- Metric definition: x-axis is mean final kept top2 links divided by original
  top2 links; y-axis is PPL for each protect or gap rule.
- Unit: x-axis fraction, y-axis PPL on a log scale.
- Axis check: pass. Both axes state the metric and unit.
- Legend check: pass. Horizontal lines identify top2 and gap8 reference PPL.
- Unit compatibility check: pass. Separate axes use separate units.
- Visible-number check: pass. Exact mode values are in
  `combined_protect_sink_recent_summary.csv`.
- Reader-confusion risk: medium. Some labels are close together near the best
  region, so the CSV and result table are the authoritative numeric source.
- Fix applied: kept the plot because it clearly shows the broad trend and the
  table gives exact numbers.
- Allowed conclusion: PPL becomes close to top2 only after the rule keeps about
  `95%` or more of original top2 links; the best protect rule keeps less than
  gap8 but has slightly lower PPL on this run.
- Remaining uncertainty: kept-link fraction alone does not prove why the kept
  links matter; it should be read with the sink/recent threshold plot.

## Selected historical-token cases by head-count

- Status: pass.
- Metric definition: fraction or count of layer-query-token cases where a
  historical token is selected by exactly `c` heads under the original top2
  rule.
- Unit: fraction in `selected_token_cases_by_head_count.png`; number of cases
  in `selected_token_case_count_by_head_count_logy.png`.
- Axis check: pass. X-axis is the number of selecting heads; y-axis states
  fraction or count.
- Legend check: pass. No legend is needed because one metric is shown.
- Unit compatibility check: pass. Each plot has one y-axis unit.
- Visible-number check: pass. Each bar is labeled.
- Reader-confusion risk: low.
- Fix applied: added a log-count version so rare high-head-count groups remain
  visible.
- Allowed conclusion: most selected token cases are selected by one or two
  heads; all-16-head cases are rare but nonzero.
- Remaining uncertainty: count size alone does not show whether a group is
  important for PPL.

## Relative position distribution by selected-head count

- Status: pass.
- Metric definition: for each selected-head-count group, distribution of
  relative key position `key_index / (history_count - 1)`, where `0` is sequence
  start and `1` is query-near.
- Unit: fraction within each selected-head-count group.
- Axis check: pass. X-axis is selected-head count; y-axis is relative
  historical position.
- Legend check: pass. Colorbar states that each column is a within-group
  fraction.
- Unit compatibility check: pass. One color metric is shown.
- Visible-number check: pass. Exact quantiles and fractions are recorded in
  `head_count_position_summary.csv` and summarized in `visualization_results.md`.
- Reader-confusion risk: medium. A reader might mistake the heatmap for global
  counts, so the result text states that columns are normalized within each
  head-count group.
- Fix applied: kept count plots next to this heatmap to show group size.
- Allowed conclusion: counts `1..14` move increasingly query-near, while count
  `16` has a strong sequence-start component plus a query-near component.
- Remaining uncertainty: position distribution does not prove causal importance.

## Sink and recent-token fraction by selected-head count

- Status: pass.
- Metric definition: fraction of selected token cases in each head-count group
  that fall in `key position < 64`, recent `1%`, recent `8%`, or recent `16%`.
- Unit: fraction.
- Axis check: pass. X-axis is selected-head count; y-axis is fraction of token
  cases.
- Legend check: pass. Each line identifies one sink/recent region.
- Unit compatibility check: pass. All lines are fractions.
- Visible-number check: pass. Exact values are in
  `head_count_position_summary.csv`.
- Reader-confusion risk: low.
- Fix applied: none after visual inspection.
- Allowed conclusion: high overlap up to 14 heads is mostly recent; all-16-head
  overlap is mostly sink with a smaller recent component.
- Remaining uncertainty: the fixed thresholds 64, 1%, 8%, and 16% are probes,
  not learned boundaries.

## Relative position quantiles by selected-head count

- Status: pass.
- Metric definition: median and 25%-75% interval of relative key position for
  each selected-head-count group.
- Unit: relative position in `[0, 1]`.
- Axis check: pass. X-axis is selected-head count; y-axis defines `0` as
  sequence start and `1` as query-near.
- Legend check: pass. Median line and 25%-75% band are labeled.
- Unit compatibility check: pass.
- Visible-number check: pass. Exact quantiles are in
  `head_count_position_summary.csv`.
- Reader-confusion risk: medium for count `16`, because the group is bimodal
  and the median alone jumps to `0`. The heatmap and sink/recent fraction plot
  are required to interpret that group.
- Fix applied: the quantile plot is presented as a summary, not the only
  evidence.
- Allowed conclusion: the typical position shifts toward query-near as
  selected-head count rises, except the all-16 group where sink dominates the
  median.
- Remaining uncertainty: quantiles hide multimodal structure.
