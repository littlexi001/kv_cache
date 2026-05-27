# Detailed Report: H0529a Zipfian Frequency Shortcut

## 0. 摘要

H0529a 问的是：

> 当 $P(b_i)$ 从 uniform 变成 Zipfian、但 $P(c\mid b_i)$ 仍 balanced 时，flat top-1 selected-gate sparse MoE 是否更难形成 tail-stable context-sense functional specialization？

答案是：

```text
没有观察到 tail functional specialization 被 Zipfian 削弱。
Zipfian 改变了 load / optimization structure，但不是当前 setup 中 tail S_sense failure 的充分放大器。
```

## 1. 实验设置

Runner:

```text
XingyuD/Attention_Search_Experiments/active/synthetic_data_understanding/scripts/run_h0529a_zipfian_frequency_shortcut.py
```

Config:

```text
XingyuD/Attention_Search_Experiments/active/synthetic_data_understanding/configs/h0529a_zipfian_frequency_shortcut.json
```

ACP job:

```text
pt-1d236anf
```

Run name:

```text
h0529a_zipfian_hierarchical_4gpu_20260525
```

Flat conditions used for H0529a:

```text
flat_uniform
flat_zipfian
```

Seeds:

```text
20260521, 20260522, 20260523, 20260524
```

Dataset:

- $64$ B tokens；
- $8$ families；
- contexts: $r,A,C$；
- Zipfian training uses $P(b_i)\propto i^{-1}$；
- context stays balanced conditional on token；
- evaluation is uniform over token and context。

## 2. Implementation Audit

Sparse audit passed for the smoke and full-run entry path:

| Condition | Router grad | Router delta | Active experts / token | Dispatch |
|---|---:|---:|---:|---|
| `flat_uniform` | nonzero | nonzero | `1` | `top1_selected_gate_sparse` |
| `flat_zipfian` | nonzero | nonzero | `1` | `top1_selected_gate_sparse` |

This means H0529a is testing the corrected selected-gate sparse router, not the old hard argmax no-gradient path.

## 3. Primary Metric

Primary metric:

$$
S_{\mathrm{sense}}^{B}(k)=\max_e\left[\Delta L_{k,e}-\max_{k'\neq k}\Delta L_{k',e}\right]
$$

It is computed from class-specific expert ablation delta and summarized by head / middle / tail token buckets.

| Condition | Head $S_{\mathrm{sense}}$ | Middle $S_{\mathrm{sense}}$ | Tail $S_{\mathrm{sense}}$ |
|---|---:|---:|---:|
| `flat_uniform` | `0.0411` | `0.0399` | `0.0451` |
| `flat_zipfian` | `0.0595` | `0.0528` | `0.0612` |

Interpretation:

```text
Zipfian did not reduce tail S_sense.
Therefore the strong H0529a hypothesis is weakened.
```

## 4. Supporting Metrics

### 4.1 Route-Function Alignment

| Condition | Head alignment | Middle alignment | Tail alignment |
|---|---:|---:|---:|
| `flat_uniform` | `0.7714` | `0.8166` | `0.7777` |
| `flat_zipfian` | `0.8542` | `0.8128` | `0.7930` |

Tail alignment is not lower under Zipfian.

### 4.2 Uniform-Eval CE / Accuracy

| Condition | Head CE | Middle CE | Tail CE | Head acc | Middle acc | Tail acc |
|---|---:|---:|---:|---:|---:|---:|
| `flat_uniform` | `0.0005` | `0.0057` | `0.0005` | `1.0000` | `0.9974` | `1.0000` |
| `flat_zipfian` | `0.0005` | `0.0219` | `0.0055` | `1.0000` | `0.9902` | `0.9978` |

Zipfian has slightly worse middle/tail CE under uniform evaluation, but not enough to show functional specialization collapse.

### 4.3 Routing Diagnostics

| Condition | NMI(route, token) | NMI(route, family) | NMI(route, bucket) | NMI(route, context) |
|---|---:|---:|---:|---:|
| `flat_uniform` | `0.1077` | `0.0469` | `0.0071` | `0.1100` |
| `flat_zipfian` | `0.1102` | `0.0609` | `0.0054` | `0.1002` |

Routing diagnostics do not show a strong frequency-bucket routing shortcut.

### 4.4 Train Dynamics

| Condition | Final train loss | Final load entropy | Selected gate prob |
|---|---:|---:|---:|
| `flat_uniform` | `0.0021` | `0.8143` | `0.6731` |
| `flat_zipfian` | `0.0039` | `0.5218` | `0.7734` |

Zipfian clearly concentrates expert load more than uniform. This is the strongest positive signal for a frequency effect, but it is not yet a functional-specialization failure.

## 5. Figure

![H0529a flat Zipfian vs uniform](figures/h0529a_flat_zipfian_vs_uniform.png)

The figure shows the two key facts together: tail $S_{\mathrm{sense}}$ is not lower under Zipfian, while middle/tail CE is modestly worse.

## 6. Interpretation

Result:

```text
Zipfian token frequency changes load and optimization pressure.
```

Interpretation:

```text
That pressure did not translate into worse tail context-sense functional specialization in this controlled setup.
```

Claim:

```text
In H0529a, token-frequency Zipfianity alone is not a sufficient explanation for flat top-1 router functional-specialization failure.
```

Speculation:

```text
The balanced context conditional distribution may make each token's contexts sufficiently learnable despite frequency skew.
Failure may require context/sense frequency skew, stronger capacity bottleneck, or an explicit objective mismatch rather than token-frequency skew alone.
```

## 7. Claim Boundary

Can claim:

```text
In the current Zipfian-token / balanced-context synthetic setup, flat top-1 selected-gate sparse MoE does not show worse tail ablation-based context-sense specialization than the uniform-token control.
```

Cannot claim:

```text
Zipfian distributions are harmless.
Real language long-tail sense distributions behave the same.
Hierarchical MoE is unnecessary.
Frequency is irrelevant to router specialization.
Routing heatmap is enough to judge function.
```

## 8. Artifact Map

Curated tables:

```text
Projects/from-attention-to-search/main/experiments/H0529a_zipfian_frequency_shortcut/tables/
```

Curated figure:

```text
Projects/from-attention-to-search/main/experiments/H0529a_zipfian_frequency_shortcut/figures/h0529a_flat_zipfian_vs_uniform.png
```

Raw result dir:

```text
XingyuD/Attention_Search_Experiments/active/synthetic_data_understanding/results/h0529a_zipfian_frequency_shortcut/h0529a_zipfian_hierarchical_4gpu_20260525
```

Runtime log:

```text
XingyuD/Attention_Search_Experiments/active/synthetic_data_understanding/logs/acp/h0529a_zipfian_hierarchical_4gpu_20260525_runtime_20260525_034916.log
```

Repro command:

```bash
H0529A_ZIPFIAN_ALLOW_REAL_SUBMIT=1 \
RUN_NAME=h0529a_zipfian_hierarchical_4gpu_20260525 \
JOB_NAME=ats-h0529a-zipfian-hier-4gpu \
RUN_STAGE=full \
MAX_PARALLEL=4 \
bash scripts/submit_h0529a_zipfian_frequency_shortcut_4gpu_acp.sh
```
