# Execution Plan: H0529a Zipfian Frequency Shortcut

Hypothesis purpose card:

```text
Projects/from-attention-to-search/main/hypotheses/archive/H0529a_zipfian_frequency_shortcut/purpose_card.md
```

## 1. Run Goal

判断 Zipfian token frequency 是否会放大 flat top-1 selected-gate sparse MoE 的 route-function mismatch，尤其是否削弱 tail bucket 的 context-sense functional specialization。

主判断不看 routing heatmap，而看 ablation-based tail $S_{\mathrm{sense}}$ 和 tail route-function alignment。

## 2. Data Construction

Synthetic hierarchical-sense setup:

- tokens: $b_1,\dots,b_{64}$；
- contexts: $r,A,C$；
- families: $g(b_i)\in\{1,\dots,8\}$；
- targets: $(r,b_i)\mapsto y_{\mathrm{common}}[g(b_i)]$，$(A,b_i)\mapsto y_A[g(b_i)]$，$(C,b_i)\mapsto y_C[g(b_i)]$。

Training conditions:

- `flat_uniform`: $b_i$ uniform, context balanced conditional on token；
- `flat_zipfian`: $P(b_i)\propto i^{-1}$, context balanced conditional on token。

Evaluation:

- uniform over token and context；
- report head / middle / tail buckets separately。

## 3. Model / Training / Evaluation Setup

Model:

- corrected top-1 selected-gate sparse MoE；
- no load balance；
- selected gate is not renormalized to 1；
- one expert active per token。

Training:

- seeds: `20260521, 20260522, 20260523, 20260524`；
- steps: `1600` full, `40` smoke；
- batch size: `384`；
- optimizer: AdamW。

## 4. Configs And Commands

Runner:

```text
XingyuD/Attention_Search_Experiments/active/synthetic_data_understanding/scripts/run_h0529a_zipfian_frequency_shortcut.py
```

Config:

```text
XingyuD/Attention_Search_Experiments/active/synthetic_data_understanding/configs/h0529a_zipfian_frequency_shortcut.json
```

4GPU ACP submit wrapper:

```text
XingyuD/Attention_Search_Experiments/active/synthetic_data_understanding/scripts/submit_h0529a_zipfian_frequency_shortcut_4gpu_acp.sh
```

Smoke command:

```bash
python scripts/run_h0529a_zipfian_frequency_shortcut.py \
  --config configs/h0529a_zipfian_frequency_shortcut.json \
  --run-name h0529a_zipfian_hierarchical_smoke_20260525 \
  --run-stage smoke \
  --parallel \
  --max-parallel 2
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

- tail bucket $S_{\mathrm{sense}}$ from class-specific expert ablation delta；
- tail route-function alignment。

Supporting:

- uniform-eval CE / accuracy by bucket；
- normalized mutual information between route and token id, family id, frequency bucket, and context；
- selected gate probability；
- sparse gradient audit。

Artifacts:

```text
results/h0529a_zipfian_frequency_shortcut/<run_name>/
figures/h0529a_zipfian_frequency_shortcut/<run_name>/
```

## 6. Pass / Fail Criteria

Sparse implementation must pass:

- router task-loss gradient is nonzero；
- router parameters change after optimizer step；
- active experts per token min/max are both 1；
- `full_soft_mixture = False`；
- `selected_gate_renormalized = False`。

Hypothesis support:

- `flat_zipfian` tail $S_{\mathrm{sense}}$ lower than `flat_uniform`；
- `flat_zipfian` tail route-function alignment lower than `flat_uniform`；
- routing diagnostics show stronger token/family/frequency association without matching functional alignment。

Hypothesis weakened:

- `flat_zipfian` behaves like `flat_uniform` on tail $S_{\mathrm{sense}}$ and route-function alignment。

## 7. Failure Modes To Check

- Zipfian train distribution accidentally used for evaluation；
- context distribution not balanced conditional on token；
- routing NMI mistaken for functional evidence；
- hierarchical comparison interpreted before H0529a flat diagnosis is closed；
- local smoke result overread as full result。

## 8. Result Location

Summary:

```text
Projects/from-attention-to-search/main/experiments/H0529a_zipfian_frequency_shortcut/summary.md
```

Detailed:

```text
Projects/from-attention-to-search/main/experiments/H0529a_zipfian_frequency_shortcut/detailed.md
```

Result status:

```text
TBD.
```
