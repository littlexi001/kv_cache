# Execution Plan: H0530a Hierarchical Common/Sense MoE

Hypothesis purpose card:

```text
Projects/from-attention-to-search/main/hypotheses/H0530a_hierarchical_common_sense_moe/purpose_card.md
```

## 1. Run Goal

在 H0529a 的同一批 full run 中，比较 `flat_zipfian` 与 `hierarchical_zipfian`，判断 two-level common/sense MoE 是否改善 tail context-sense functional specialization。

## 2. Data Construction

Use the same Zipfian hierarchical-sense dataset as H0529a:

- $P(b_i)\propto i^{-1}$；
- $P(r\mid b_i)=P(A\mid b_i)=P(C\mid b_i)=1/3$；
- uniform evaluation over token and context。

## 3. Model / Training / Evaluation Setup

Baseline:

```text
flat_zipfian
```

Hierarchy:

```text
hierarchical_zipfian
```

Hierarchy has:

- coarse common router over the current hidden state；
- fine router over a family-reference residual；
- output as common path plus fine path；
- no load balance。

Primary judgment uses fine expert ablation, not routing heatmap.

## 4. Configs And Commands

Shared with H0529a:

```text
runner: XingyuD/Attention_Search_Experiments/active/synthetic_data_understanding/scripts/run_h0529a_zipfian_frequency_shortcut.py
config: XingyuD/Attention_Search_Experiments/active/synthetic_data_understanding/configs/h0529a_zipfian_frequency_shortcut.json
submit: XingyuD/Attention_Search_Experiments/active/synthetic_data_understanding/scripts/submit_h0529a_zipfian_frequency_shortcut_4gpu_acp.sh
```

Full ACP command:

```bash
H0529A_ZIPFIAN_ALLOW_REAL_SUBMIT=1 \
RUN_NAME=h0529a_zipfian_hierarchical_4gpu_20260525 \
JOB_NAME=ats-h0529a-zipfian-hier-4gpu \
bash scripts/submit_h0529a_zipfian_frequency_shortcut_4gpu_acp.sh
```

## 5. Metrics And Artifacts

Primary:

- tail $S_{\mathrm{sense}}$ for `hierarchical_zipfian` versus `flat_zipfian`；
- tail route-function alignment。

Supporting:

- uniform-eval CE / accuracy by bucket；
- route NMI with token id, family id, frequency bucket, context；
- sparse gradient audit；
- selected gate probability。

## 6. Pass / Fail Criteria

Hierarchy supports the hypothesis only if it improves ablation-based functional metrics, especially tail $S_{\mathrm{sense}}$ and tail route-function alignment.

If it improves routing NMI but not ablation deltas, it does not count as functional specialization success.

## 7. Failure Modes To Check

- interpreting extra capacity as proven hierarchy benefit；
- reading routing heatmap as causal function；
- comparing against Zipfian evaluation instead of uniform evaluation；
- ignoring H0529a if Zipfian frequency does not actually worsen flat router behavior。

## 8. Result Location

Summary:

```text
Projects/from-attention-to-search/main/experiments/H0530a_hierarchical_common_sense_moe/summary.md
```

Detailed:

```text
Projects/from-attention-to-search/main/experiments/H0530a_hierarchical_common_sense_moe/detailed.md
```

Result status:

```text
TBD.
```
