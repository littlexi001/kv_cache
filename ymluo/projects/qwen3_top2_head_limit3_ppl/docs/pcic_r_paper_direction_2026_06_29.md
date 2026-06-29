# PCIC-R 论文主线进展（2026-06-29）

## 方法定位

当前建议主线：

```text
PCIC-R: Pairwise Counterfactual Influence Cache with Online Blockwise Selection and Rescue
```

核心思想：

1. 不再固定一组压缩层长期使用；
2. 每个 block 用很短 calibration window 做 Pairwise-CIC 组合选择；
3. 后续 token 默认使用选出的压缩层组合；
4. 如果 token 风险高，则通过 rescue gate 回退 full attention。

这个方向比 `Pairwise-CIC + landmark` 更像论文方法，因为它解决了之前实验暴露的两个核心问题：

- 静态 layer list 迁移不稳；
- 单层/组合选择存在上下文依赖。

## 已实现原型

新增脚本：

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/src/run_pcic_rescue_blockwise_local.py
```

实现内容：

- 加载模型并 prefill；
- 每个 block 先跑 baseline calibration；
- 对候选 layer combo 跑 calibration；
- 从 calibration 结果中选安全组合；
- 用选中的 combo 评估后续 tokens；
- 支持 token-level margin rescue：

```text
if previous logits top1-top2 margin < threshold:
    use full attention
else:
    use selected Pairwise-CIC landmark compression
```

当前 fallback 仍使用：

```text
recent=512 + landmark stride=64
```

## 远端实验设置

服务器：

```text
ssh fdong@10.176.37.31
```

项目路径：

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
```

环境：

```text
conda env: moe
GPU: RTX 3090
model: Qwen/Qwen3-0.6B
dtype: bfloat16
```

实验配置：

```text
prefill_tokens: 4096
num_blocks: 2
calibration_tokens: 16
eval_tokens_per_block: 64
recent_tokens: 512
landmark_stride: 64
```

War 候选组合：

```text
7,6; 0,13; 0,7; 0,6
```

Monte 候选组合：

```text
2,0,7,12; 7,13; 2,7; 2,0
```

## 远端实验结果

### War and Peace

| rescue margin | avg delta_loss | avg delta_ppl | rescue tokens | compressed tokens |
| ---: | ---: | ---: | ---: | ---: |
| 0.0 | -0.026085 | -0.687288 | 0 | 128 |
| 0.5 | -0.014369 | -0.391132 | 34 | 94 |
| 2.0 | -0.015531 | -0.416840 | 99 | 29 |

详细结果：

```text
outputs/server_pcic_r_war_b2_m0
outputs/server_pcic_r_war_b2_m0p5
outputs/server_pcic_r_war_b2_m2
```

观察：

- blockwise calibration 选择了 `0,6` 或 `0,13`；
- 无 rescue 时整体 PPL 改善最大；
- margin rescue 越强，压缩 token 越少，PPL 改善反而变弱。

### Monte Cristo

| rescue margin | avg delta_loss | avg delta_ppl | rescue tokens | compressed tokens |
| ---: | ---: | ---: | ---: | ---: |
| 0.0 | -0.003223 | -0.108427 | 0 | 128 |
| 0.5 | -0.001238 | +0.016823 | 31 | 97 |
| 2.0 | +0.010839 | +0.142788 | 84 | 44 |

详细结果：

```text
outputs/server_pcic_r_monte_b2_m0
outputs/server_pcic_r_monte_b2_m0p5
outputs/server_pcic_r_monte_b2_m2
```

观察：

- blockwise calibration 选择了 `2,0`；
- 无 rescue 时略优于 baseline；
- margin rescue 在 Monte 上没有收益。

## 当前结论

### 成立的部分

1. **Online blockwise selection 可行**：脚本已经能在每个 block 重新 calibration 并选择组合。
2. **Pairwise-CIC 仍有效**：War 和 Monte 的无 rescue blockwise 结果均没有崩，War 明显改善。
3. **动态选择有必要**：不同 block 可能选出不同组合，例如 War 从 `0,6` 变到 `0,13`。

### 不成立的部分

当前 naive margin rescue 不成立：

```text
top1-top2 logit margin low -> full attention
```

这个规则太粗，原因是：

- 低 margin 不一定意味着压缩层会出错；
- 高 margin 也不一定意味着压缩安全；
- 直接 rescue 到 full 会减少压缩 token，降低潜在收益；
- 当前 gating 信号和“压缩误差”没有直接对齐。

因此，下一步不能继续简单调 margin threshold，而要把 rescue gate 改成“压缩风险预测”。

## 下一步方法升级

建议把 rescue gate 从 margin gate 改成：

```text
counterfactual-risk rescue gate
```

具体做法：

1. 在 calibration window 中同时收集：
   - baseline loss；
   - compressed loss；
   - margin；
   - entropy；
   - chosen combo；
   - token position；
   - compressed-vs-full loss gap。
2. 用这些特征拟合一个轻量风险规则：

```text
rescue if predicted compressed_loss_gap > threshold
```

3. 最小实现可以先不用训练模型，只做规则搜索：
   - margin threshold；
   - entropy threshold；
   - margin × entropy；
   - calibration 中最危险 token 的邻域传播；
   - block-level high-risk ratio。
4. 如果规则 gate 有效，再考虑学习一个 tiny logistic regressor。

## 论文创新性表述

当前可形成的创新主张：

```text
We propose a counterfactual-influence-driven online cache budget allocation method.
Unlike prior static layer/head/token pruning rules, PCIC-R estimates compression risk in a short calibration window,
models pairwise layer interactions, and applies blockwise cache budgets with token-level rescue.
```

需要避免的弱表述：

```text
We compress selected layers with landmark attention.
```

因为 landmark 和 layer-wise compression 都容易被认为不新。

更强表述应是：

```text
compression budget is selected by measured counterfactual impact, not by attention mass, layer index, or static heuristics.
```

## 下一轮实验计划

1. 实现 calibration risk logger，记录每个 calibration token 的 compressed-vs-full loss gap。
2. 对 margin、entropy、loss gap 做相关性分析。
3. 实现 risk-based rescue gate。
4. 用同样 War/Monte 两个文本跑：
   - no rescue；
   - margin rescue；
   - entropy rescue；
   - counterfactual-risk rescue。
5. 如果 risk gate 显著优于 no rescue 和 margin rescue，再扩展到更多文本和更长 block。

当前状态：

```text
Pairwise-CIC + online blockwise selection 已经可跑；
rescue gate 框架已实现；
但 naive margin rescue 不够好，下一步必须做 counterfactual-risk rescue。
```

## 2026-06-29 继续实验：Counterfactual-Risk Rescue Gate

### 新增实现

本轮把 `run_pcic_rescue_blockwise_local.py` 从固定 margin rescue 扩展为 calibration loss-gap 风险门控：

```text
calibration 阶段：同时跑 full baseline 与 compressed combo
逐 token 记录：compressed_loss - baseline_loss、margin、entropy、top1_prob
block 内选出 combo 后：用该 combo 的 calibration loss-gap 定义 high-risk token
rescue 阶段：根据 calibration 得到的风险规则决定当前 token 是否回退 full attention
```

新增输出：

```text
pcic_r_calibration_token_risk.csv
```

该文件用于证明方法主线不是只靠静态层号，而是显式测量 counterfactual compression risk。

### 新增参数

```text
--rescue_strategy none|margin|calib_margin|calib_entropy|calib_margin_entropy
--risk_quantile 0.8
--risk_positive_gap 0.0
--risk_rescue_fraction 0.25
```

其中 `calib_margin` / `calib_entropy` 加入两个限制：

1. high-risk token 的特征方向必须和规则一致；例如 margin gate 要求 high-risk token 的平均 margin 更低。
2. 阈值按 `risk_rescue_fraction` 控制目标 rescue 比例，避免第一版规则过度保守。

### 远端实验配置

```text
model: Qwen/Qwen3-0.6B
prefill_tokens: 4096
num_blocks: 2
calibration_tokens: 16
eval_tokens_per_block: 64
recent_tokens: 512
landmark_stride: 64
server: fdong@10.176.37.31
```

War combos：

```text
7,6;0,13;0,7;0,6
```

Monte combos：

```text
2,0,7,12;7,13;2,7;2,0
```

### 结果汇总

| run | avg_delta_loss | avg_delta_ppl | pcic_seconds | baseline_seconds | rescue | compressed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| War no rescue | -0.026085 | -0.687288 | 4.3972 | 4.2337 | 0 | 128 |
| War margin0.5 | -0.014369 | -0.391132 | 4.3843 | 4.2512 | 34 | 94 |
| War margin2 | -0.015531 | -0.416840 | 4.1987 | 4.1785 | 99 | 29 |
| War calib_margin old | -0.007273 | -0.201957 | 4.2450 | 4.2314 | 85 | 43 |
| War calib_entropy old | 0.000717 | -0.003362 | 4.2965 | 4.2363 | 88 | 40 |
| War calib_margin frac25 | -0.001263 | -0.044693 | 4.9086 | 4.7139 | 26 | 102 |
| War calib_entropy frac25 | -0.029309 | -0.764574 | 5.7776 | 5.6627 | 44 | 84 |
| Monte no rescue | -0.003223 | -0.108427 | 4.3420 | 4.1941 | 0 | 128 |
| Monte margin0.5 | -0.001238 | 0.016823 | 4.2789 | 4.1642 | 31 | 97 |
| Monte margin2 | 0.010839 | 0.142788 | 4.2943 | 4.2435 | 84 | 44 |
| Monte calib_margin old | -0.001138 | 0.002415 | 4.2136 | 4.2376 | 103 | 25 |
| Monte calib_entropy old | 0.007926 | 0.129759 | 4.2109 | 4.1846 | 96 | 32 |
| Monte calib_margin frac25 | -0.002992 | -0.105658 | 4.4704 | 4.3864 | 28 | 100 |
| Monte calib_entropy frac25 | -0.001541 | -0.088247 | 4.3670 | 4.2576 | 28 | 100 |

### 风险特征诊断

在 calibration token risk CSV 上计算 `loss_gap` 与候选特征的 Pearson 相关性：

| split | n | corr(loss_gap, margin) | corr(loss_gap, entropy) | corr(loss_gap, top1_prob) | positive_gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| War | 128 | -0.0513 | -0.0476 | 0.0405 | 76 |
| Monte | 128 | 0.0279 | 0.0051 | -0.0049 | 68 |

结论：普通 uncertainty proxy（margin、entropy、top1 probability）和真实 compressed-vs-full loss gap 的线性相关性很弱。这解释了为什么 naive margin rescue 和第一版 risk gate 都不稳定。

### 当前判断

1. `Pairwise-CIC + online blockwise selection` 仍然成立：no-rescue 在 War/Monte 都没有崩，War 明显优于 baseline。
2. `rescue gate` 的创新方向仍然值得保留，但不能继续依赖普通 confidence signal。
3. 本轮最重要的证据是：已经有逐 token 的 counterfactual risk 数据，可以支撑下一步做真正的 learned/estimated risk，而不是手写 margin 规则。
4. 从 paper 创新性看，主张应从“uncertainty rescue”改成“counterfactual risk estimation for cache compression”。

### 下一步方法建议

下一轮不再继续调 margin/entropy 阈值，而是做更直接的风险预测信号：

```text
PCIC-R2: Counterfactual Risk Predictor
```

候选做法：

1. 用 calibration 中的 `(margin, entropy, top1_prob, position, combo id, delta logits proxy)` 拟合一个 tiny logistic/ridge regressor，预测 `loss_gap > 0` 或 `loss_gap > quantile`。
2. 加入 cheap disagreement proxy：同一个 token 上不跑 full，而是跑两个低成本 compressed policies，若二者 logits 分歧大则 rescue。
3. 做 block-level fallback：如果 calibration 发现当前 combo 的 positive loss-gap 比例过高，则整块改用更保守 combo，而不是逐 token rescue。
4. 把 rescue 的目标改为“保持压缩收益下的尾部风险控制”，即只救 top-risk 10% token，而不是追求全部 token 更低 PPL。

当前最可能成为 paper 亮点的是第 2 点：**counterfactual disagreement without full attention**。它比 margin/entropy 更接近压缩误差本身，同时比每个 token 都跑 full 便宜。

## 2026-06-29 继续实验：PCIC-R2 Disagreement Gate

### 新增实现

本轮实现了 `PCIC-R2: compressed-policy disagreement rescue`。核心动机是：margin/entropy/top1 probability 与真实 `compressed_loss - full_loss` 相关性很弱，因此 rescue gate 不应该只看模型自身不确定性，而应该看“压缩策略之间是否互相不一致”。

实现方式：

```text
1. block calibration 中照常评估多个 Pairwise-CIC layer combo。
2. 选择 primary combo 后，选择另一个 compressed combo 作为 probe。
3. calibration 中计算 primary/probe 的 logits disagreement。
4. 如果 high-risk token 的 disagreement 明显高于 safe token，则建立阈值规则。
5. eval 时同时维护 primary/probe 两条 compressed stream，用 disagreement 决定是否 rescue 到 full。
```

新增参数：

```text
--rescue_strategy calib_disagreement
--disagreement_metric js|l2
--combo_select_policy fastest_safe|min_loss
```

当前实现为了验证信号，eval 阶段显式跑 active / primary / probe 三条 stream。因此 `seconds` 不能代表最终速度，只能代表原型验证开销。后续如果该方向成立，需要把 disagreement proxy 做成低成本近似，例如只比较少量 logits、head summary、或者复用已计算的 compressed logits。

### 远端实验结果

| run | avg_delta_loss | avg_delta_ppl | pcic_seconds | baseline_seconds | rescue | compressed | chosen/probe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| War no rescue | -0.026085 | -0.687288 | 4.3972 | 4.2337 | 0 | 128 | 0,6; 0,6 |
| War calib_entropy frac25 | -0.029309 | -0.764574 | 5.7776 | 5.6627 | 44 | 84 | 0,6; 0,6 |
| War R2 fastest_safe | 0.009848 | 0.247110 | 8.3711 | 4.0880 | 45 | 83 | 0,13/0,6; 0,13/0,6 |
| War R2 min_loss | -0.004669 | -0.140112 | 8.5419 | 4.1695 | 45 | 83 | 0,6/0,13; 0,13/0,6 |
| Monte no rescue | -0.003223 | -0.108427 | 4.3420 | 4.1941 | 0 | 128 | 2,0; 2,0 |
| Monte calib_margin frac25 | -0.002992 | -0.105658 | 4.4704 | 4.3864 | 28 | 100 | 2,0; 2,0 |
| Monte R2 fastest_safe | -0.004011 | -0.091509 | 12.7387 | 4.1489 | 45 | 83 | 2,0/2,7; 2,0/7,13 |
| Monte R2 min_loss | -0.004011 | -0.091509 | 12.9403 | 4.2082 | 45 | 83 | 2,0/2,7; 2,0/7,13 |

### 观察

1. Monte 上 `calib_disagreement` 的质量略好于 no-rescue 的 loss，但 PPL 改善不如 no-rescue；它至少说明 disagreement 有机会捕捉部分压缩风险。
2. War 上 `fastest_safe` 选择到了 `0,13`，明显劣于此前稳定的 `0,6`；改成 `min_loss` 后恢复到负 delta，但仍不如 no-rescue / entropy gate。
3. 当前 R2 rescue token 为 45/128，说明阈值仍偏激进；需要改成 top-risk 10% 或 block-level fallback，而不是 25% token rescue。
4. 三路 forward 让速度明显变差，所以 R2 当前只是“信号验证原型”，不能作为最终速度结果。

### 方法判断

`calib_disagreement` 的创新性比 margin/entropy 更强，因为它直接比较两个低成本压缩策略的预测差异，逻辑上更接近“压缩误差估计”。但当前实验还没有证明它显著优于 no-rescue，因此不能直接作为最终 paper 主结果。

更合理的下一步是：

```text
PCIC-R3: block-level risk fallback + sparse disagreement
```

建议具体改法：

1. block-level：如果 calibration 中某个 combo 的 positive loss-gap ratio 或 max loss-gap 过高，整块换成更保守 combo，而不是逐 token rescue。
2. sparse disagreement：只在每个 block 的少数 sentinel token 上计算 probe disagreement，避免每个 token 三路 forward。
3. rescue fraction 从 `0.25` 降到 `0.1`，目标是控制尾部风险，而不是大量回退 full。
4. combo 选择默认应从 `fastest_safe` 改成质量优先的 `min_loss` 或 Pareto 选择，否则速度噪声会导致选错 combo。

当前 paper 主线仍然建议保持：

```text
Pairwise-CIC + online blockwise selection + counterfactual-risk rescue
```

但 rescue 的最终实现应更偏向 block-level risk fallback / sparse disagreement，而不是 dense token-level full rescue。

## 2026-06-29 继续实验：PCIC-R3 Block-Level Risk Fallback

### 新增实现

本轮实现了 `PCIC-R3: block-level risk fallback`。它不再对每个 eval token 做 full rescue，也不需要 R2 那种三路 forward，而是在 calibration 阶段直接判断当前 block 选择出来的 combo 是否风险过高：

```text
对每个候选 combo 记录 calibration token 的 compressed_loss - baseline_loss
计算每个 combo 的：
  risk_max_loss_gap
  risk_positive_ratio
  risk_mean_loss_gap
如果原始选中的 combo 超过风险阈值：
  在候选 combo 中切换到 calibration 风险更低的 combo
后续 eval 整个 block 都用这个新 combo，不做逐 token rescue
```

新增参数：

```text
--rescue_strategy block_fallback
--block_risk_max_gap 0.2
--block_risk_positive_ratio 0.5
```

代码位置：

```text
run_pcic_rescue_blockwise_local.py
  summarize_token_risk(...)
  choose_block_fallback_combo(...)
```

### 2-block 结果

| run | blocks | avg_delta_loss | avg_delta_ppl | pcic_seconds | baseline_seconds | triggered | combos |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| War b2 no rescue | 2 | -0.026085 | -0.687288 | 4.3972 | 4.2337 | 0 | 0,6; 0,6 |
| War b2 R3 fast | 2 | -0.029522 | -0.777443 | 4.2423 | 4.0696 | 2 | 0,7; 0,13 |
| War b2 R3 minloss | 2 | -0.030843 | -0.806873 | 4.3438 | 4.1802 | 2 | 0,7; 0,6 |
| Monte b2 no rescue | 2 | -0.003223 | -0.108427 | 4.3420 | 4.1941 | 0 | 2,0; 2,0 |
| Monte b2 R3 fast | 2 | -0.015614 | -0.255059 | 4.2524 | 4.1132 | 1 | 2,0; 2,7 |
| Monte b2 R3 minloss | 2 | -0.015614 | -0.255059 | 4.5008 | 4.3203 | 1 | 2,0; 2,7 |

### 4-block 鲁棒性结果

| run | blocks | avg_delta_loss | avg_delta_ppl | pcic_seconds | baseline_seconds | triggered | combos |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| War b4 none minloss | 4 | 0.000781 | 0.432030 | 8.5850 | 8.2910 | 0 | 0,6; 0,13; 0,13; 0,7 |
| War b4 R3 minloss | 4 | -0.045202 | -1.504012 | 8.6360 | 8.3213 | 3 | 0,7; 0,6; 0,13; 0,13 |
| Monte b4 none minloss | 4 | 0.016328 | 0.219077 | 8.7471 | 8.3569 | 0 | 2,0; 2,0; 2,0,7,12; 2,0,7,12 |
| Monte b4 R3 minloss | 4 | -0.001649 | -0.069722 | 8.6592 | 8.3228 | 2 | 2,0; 2,7; 2,0,7,12; 2,0 |

### risk-Pareto 选择策略对照

本轮额外测试了 `combo_select_policy=risk_pareto`。它的设计目标是：先筛出 calibration loss 不明显差于 baseline 的候选，再优先选择 counterfactual tail-risk 更低的 combo。

| run | blocks | avg_delta_loss | avg_delta_ppl | pcic_seconds | baseline_seconds | triggered | combos |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| War b4 none riskpareto | 4 | -0.013288 | -0.761415 | 8.5207 | 8.2334 | 0 | 0,13; 0,13; 0,7; 0,13 |
| War b4 R3 riskpareto | 4 | 0.008079 | 0.539733 | 8.6803 | 8.3571 | 3 | 0,7; 0,6; 0,7; 0,7 |
| Monte b4 none riskpareto | 4 | 0.005847 | 0.017709 | 8.7734 | 8.4724 | 0 | 2,0; 2,0; 2,0; 2,0 |
| Monte b4 R3 riskpareto | 4 | -0.000348 | -0.055607 | 8.7087 | 8.4034 | 1 | 2,0; 2,7; 2,0; 2,0 |

结论：`risk_pareto` 不是当前最优默认策略。它在 War 的 no-rescue 下比 `min_loss` 更稳，但叠加 R3 fallback 后反而变差；Monte 上则只带来很小改善。当前主线应保留 `min_loss + block_fallback` 作为最强版本，把 `risk_pareto` 作为消融负例，用来说明“只做风险排序不够，风险统计必须和平均校准质量联合设计”。

### 8-block 稳定性验证与实现陷阱

追加 8-block 实验时发现一个重要实现陷阱：如果手动传入 `--attn_implementation sdpa`，Transformers 会绕过当前 eager attention patch，导致 `layerbudgetattn` 实际等价于 baseline，所有 `delta_loss/delta_ppl` 都变成 0。这个结果是无效结果，不能写入论文。已在 `run_pcic_rescue_blockwise_local.py` 中加入硬性检查：PCIC-R 必须使用 `--attn_implementation eager`。

有效的 eager 8-block 结果如下：

| run | blocks | avg_delta_loss | avg_delta_ppl | pcic_seconds | baseline_seconds | triggered | combos |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| War b8 none minloss eager | 8 | 0.017480 | 1.586619 | 17.2557 | 16.6521 | 0 | 0,6; 0,6; 0,7; 0,7; 0,7; 0,13; 0,7; 0,6 |
| War b8 R3 minloss eager | 8 | -0.011840 | -0.100134 | 17.4651 | 16.8319 | 6 | 0,7; 0,13; 0,7; 0,13; 0,13; 0,13; 0,13; 0,13 |
| Monte b8 none minloss eager | 8 | 0.014333 | 0.108444 | 17.4990 | 16.7898 | 0 | 2,0; 2,0; 2,0,7,12; 2,0,7,12; 7,13; 2,7; 2,0; 2,0 |
| Monte b8 R3 minloss eager, gap0.2 ratio0.5 | 8 | 0.017366 | 0.423648 | 17.6253 | 16.9446 | 6 | 2,0; 2,0; 2,0; 2,0; 2,0; 2,0; 2,0,7,12; 7,13 |

结论：8-block 比 4-block 更能暴露在线选择漂移。War 上默认 R3 明显有效，把 no-rescue 的正 delta 拉回负 delta；Monte 上默认阈值过度 fallback，说明 rescue gate 不能使用一个固定激进阈值覆盖所有文本。

### Monte 8-block 阈值扫描

为验证 Monte 失败是否来自方法本身，追加扫描 `block_risk_max_gap / block_risk_positive_ratio`：

| run | blocks | avg_delta_loss | avg_delta_ppl | triggered | combos |
| --- | ---: | ---: | ---: | ---: | --- |
| Monte b8 none | 8 | 0.014333 | 0.108444 | 0 | 2,0; 2,0; 2,0,7,12; 2,0,7,12; 7,13; 2,7; 2,0; 2,0 |
| Monte b8 R3 gap0.1 ratio0.5 | 8 | 0.012271 | 0.343046 | 8 | 7,13; 2,7; 2,0; 2,0; 2,0; 2,0; 2,0,7,12; 7,13 |
| Monte b8 R3 gap0.2 ratio0.4 | 8 | 0.014762 | 0.372236 | 7 | 7,13; 2,0; 2,0; 2,0; 2,0; 2,0; 2,0,7,12; 7,13 |
| Monte b8 R3 gap0.2 ratio0.5 | 8 | 0.017366 | 0.423648 | 6 | 2,0; 2,0; 2,0; 2,0; 2,0; 2,0; 2,0,7,12; 7,13 |
| Monte b8 R3 gap0.4 ratio0.6 | 8 | 0.014163 | 0.288510 | 3 | 2,0; 2,0; 2,0,7,12; 2,0; 2,0; 2,7; 2,0; 7,13 |
| Monte b8 R3 gap0.6 ratio0.7 | 8 | 0.001515 | -0.056621 | 2 | 2,0; 2,0; 2,0,7,12; 2,0; 2,0; 2,7; 2,0; 2,0 |

结论：Monte 不是 R3 主线失败，而是默认 gate 过激。更宽松的 `gap0.6/ratio0.7` 只触发 2 个 block，把 no-rescue 的 `avg_delta_ppl=0.108444` 改成 `-0.056621`。这说明 rescue gate 的核心应该是 **calibrated risk budget**，而不是固定阈值；后续需要让阈值根据 calibration loss-gap 分布自适应。

### eval_tokens_per_block=128 验证

为了确认 8-block 结论不是短 eval window 的偶然性，又追加了每块 128 eval token 的验证：

| run | blocks | avg_delta_loss | avg_delta_ppl | pcic_seconds | baseline_seconds | triggered | combos |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| War b4 eval128 none | 4 | -0.001551 | -0.009529 | 17.2340 | 16.6056 | 0 | 0,6; 0,6; 0,7; 0,7 |
| War b4 eval128 R3 gap0.2 ratio0.5 | 4 | 0.014290 | 0.365961 | 17.1614 | 16.5625 | 3 | 0,7; 0,6; 0,13; 7,6 |
| Monte b4 eval128 none | 4 | -0.002673 | -0.073396 | 17.4809 | 16.8178 | 0 | 2,0; 2,0; 2,0; 2,0 |
| Monte b4 eval128 R3 gap0.6 ratio0.7 | 4 | -0.002673 | -0.073396 | 17.3366 | 16.6990 | 0 | 2,0; 2,0; 2,0; 2,0 |

这个结果对固定阈值 R3 是负面证据：War 在 128-token eval 下 no-rescue 本身已经略好，而固定 R3 触发 3 次后变差；Monte 的宽松阈值没有触发，等价于 no-rescue。结论不是放弃主线，而是明确下一步必须从 **fixed threshold rescue** 转向 **adaptive risk gate**：

```text
如果 no-rescue 当前 block 已经低风险，不应该强行 fallback；
如果短窗口 calibration 中 tail risk 高但平均 loss 已经足够好，需要更谨慎判断 fallback 是否会引入分布漂移；
gate 应该学习/估计“是否需要换策略”，而不是只看 max_gap/positive_ratio 是否超过固定阈值。
```

### adaptive_block_fallback 第一轮

本轮实现了 `adaptive_block_fallback`，目的是替代固定阈值：

```text
只有当候选 combo 的 calibration loss 不比原 combo 差太多，
并且 tail risk 明确降低时，才允许 block-level fallback。
```

实现中加入了以下参数：

```text
--rescue_strategy adaptive_block_fallback
--adaptive_loss_slack 0.02
--adaptive_max_gap_improvement 0.05
--adaptive_positive_ratio_improvement 0.125
--adaptive_require_degraded true|false
```

对照结果：

| run | blocks | avg_delta_loss | avg_delta_ppl | triggered | combos |
| --- | ---: | ---: | ---: | ---: | --- |
| War b8 none | 8 | 0.017480 | 1.586619 | 0 | 0,6; 0,6; 0,7; 0,7; 0,7; 0,13; 0,7; 0,6 |
| War b8 fixed R3 | 8 | -0.011840 | -0.100134 | 6 | 0,7; 0,13; 0,7; 0,13; 0,13; 0,13; 0,13; 0,13 |
| War b8 adaptive require_degraded=true | 8 | 0.017480 | 1.586619 | 0 | 0,6; 0,6; 0,7; 0,7; 0,7; 0,13; 0,7; 0,6 |
| War b8 adaptive require_degraded=false | 8 | 0.024316 | 1.794991 | 3 | 0,13; 0,13; 0,13; 0,7; 0,7; 0,13; 0,7; 0,6 |
| Monte b8 none | 8 | 0.014333 | 0.108444 | 0 | 2,0; 2,0; 2,0,7,12; 2,0,7,12; 7,13; 2,7; 2,0; 2,0 |
| Monte b8 fixed R3 gap0.6 ratio0.7 | 8 | 0.001515 | -0.056621 | 2 | 2,0; 2,0; 2,0,7,12; 2,0; 2,0; 2,7; 2,0; 2,0 |
| Monte b8 adaptive require_degraded=true | 8 | 0.005935 | 0.023042 | 1 | 2,0; 2,0; 2,0,7,12; 2,0,7,12; 2,0; 2,7; 2,0; 2,0 |
| Monte b8 adaptive require_degraded=false | 8 | 0.001632 | -0.038932 | 2 | 2,0; 2,0; 2,0,7,12; 2,0,7,12; 2,0; 2,0; 2,0; 2,0 |
| War eval128 none | 4 | -0.001551 | -0.009529 | 0 | 0,6; 0,6; 0,7; 0,7 |
| War eval128 fixed R3 | 4 | 0.014290 | 0.365961 | 3 | 0,7; 0,6; 0,13; 7,6 |
| War eval128 adaptive require_degraded=true | 4 | -0.001551 | -0.009529 | 0 | 0,6; 0,6; 0,7; 0,7 |
| War eval128 adaptive require_degraded=false | 4 | 0.005769 | 0.159155 | 2 | 0,13; 0,6; 0,13; 0,7 |
| Monte eval128 none | 4 | -0.002673 | -0.073396 | 0 | 2,0; 2,0; 2,0; 2,0 |
| Monte eval128 adaptive require_degraded=false | 4 | -0.002673 | -0.073396 | 0 | 2,0; 2,0; 2,0; 2,0 |

结论：第一版 adaptive gate 还没有成功。`require_degraded=true` 足够保守，能避免 eval128 误触发，但救不了 War b8；`require_degraded=false` 能在 Monte b8 上接近固定 R3，但会误伤 War b8 和 War eval128。这个负结果说明：只用 `mean/max/positive_ratio` 三个统计量不足以稳定判断“是否应该换 combo”。

### calibration_tokens=32 验证

进一步测试了更长 calibration window，想验证是否只是 16-token calibration 太短：

| run | blocks | avg_delta_loss | avg_delta_ppl | triggered | combos |
| --- | ---: | ---: | ---: | ---: | --- |
| War b8 cal16 none | 8 | 0.017480 | 1.586619 | 0 | 0,6; 0,6; 0,7; 0,7; 0,7; 0,13; 0,7; 0,6 |
| War b8 cal16 fixed | 8 | -0.011840 | -0.100134 | 6 | 0,7; 0,13; 0,7; 0,13; 0,13; 0,13; 0,13; 0,13 |
| War b8 cal32 none | 8 | 0.030360 | 0.742060 | 0 | 7,6; 0,13; 0,7; 0,13; 0,7; 0,7; 0,7; 0,6 |
| War b8 cal32 fixed | 8 | 0.016661 | 0.591699 | 7 | 0,7; 0,6; 0,13; 0,13; 0,13; 0,13; 7,6; 0,13 |
| Monte b8 cal16 none | 8 | 0.014333 | 0.108444 | 0 | 2,0; 2,0; 2,0,7,12; 2,0,7,12; 7,13; 2,7; 2,0; 2,0 |
| Monte b8 cal16 fixed gap0.6 ratio0.7 | 8 | 0.001515 | -0.056621 | 2 | 2,0; 2,0; 2,0,7,12; 2,0; 2,0; 2,7; 2,0; 2,0 |
| Monte b8 cal32 none | 8 | 0.013250 | 0.221356 | 0 | 2,0; 7,13; 2,0; 2,0; 2,0; 2,0,7,12; 2,7; 7,13 |
| Monte b8 cal32 fixed gap0.6 ratio0.7 | 8 | 0.004719 | -0.006763 | 2 | 2,0; 7,13; 2,0; 2,0; 2,0; 2,0; 2,0; 7,13 |

结论：简单加长 calibration 也不是稳定解。War 上 cal32 反而弱于 cal16 fixed；Monte 上 cal32 fixed 只略好于 baseline，不如 cal16 fixed。下一步不应继续盲目增加 calibration，而应引入更强的 gate 目标。

### 下一版 gate 目标

现在更清晰的目标不是“找到一个固定阈值”，而是：

```text
学习或构造一个 no-harm gate：
1. 对 unsafe block：允许 fallback，降低正 delta；
2. 对 safe block：必须保持 no-rescue，不误触发；
3. fallback 候选不仅要 tail risk 低，还要通过跨候选一致性/历史稳定性检查。
```

候选下一步：

1. **stability-aware fallback**：只有候选 combo 在当前 block calibration 和最近几个 block calibration 中都稳定，才允许替换。
2. **two-signal gate**：tail-risk 只是必要条件，还需要 compressed-policy disagreement 或 calibration loss-rank agreement 作为第二信号。
3. **offline meta-gate**：先用已有 block 结果训练/拟合一个 tiny gate，目标是预测“fallback 是否会改善 eval delta”，再检查它能否泛化到 War/Monte 的 held-out block。

### sentinel_block_fallback 实现

基于上面的负结果，本轮实现了更直接的 no-harm gate：

```text
sentinel_block_fallback

1. 先按固定 R3 规则生成 fallback proposal；
2. 如果 proposal 不触发，保持原 combo；
3. 如果 proposal 触发，在 eval block 开头取少量 sentinel tokens；
4. 只用两个 compressed policies 比较：
   - original combo
   - proposed fallback combo
5. 如果 proposed 在 sentinel 上的 loss <= original loss + slack，则接受 fallback；
6. 否则拒绝 proposal，整个 eval block 仍使用 original combo。
```

这个 gate 的关键创新点是：它不依赖 full attention rescue，也不使用 eval block 的完整答案；只用 block 开头少量 token 做在线反事实验证。它比固定阈值更接近 “no-harm rescue gate”：

```text
fallback is allowed only after an online counterfactual check shows it is not worse than the original compressed policy.
```

新增参数：

```text
--rescue_strategy sentinel_block_fallback
--sentinel_tokens 8
--sentinel_loss_slack 0.0
```

代码位置：

```text
run_pcic_rescue_blockwise_local.py
  --rescue_strategy sentinel_block_fallback
  --sentinel_tokens
  --sentinel_loss_slack
```

当前本地验证：

```text
python -m py_compile src/evaluate_qwen3_top2_head_limit3_ppl.py src/run_pcic_rescue_blockwise_local.py
python src/run_pcic_rescue_blockwise_local.py --help
```

均已通过。远端服务器 `fdong@10.176.37.31` 在本轮末尾出现 SSH timeout，因此 sentinel 实验尚未启动。

待服务器恢复后优先运行四组对照：

```text
War b8: none / fixed R3 / sentinel R3
Monte b8: none / fixed R3 / sentinel R3
War eval128: none / fixed R3 / sentinel R3
Monte eval128: none / sentinel R3
```

预期判断标准：

1. 如果 sentinel 在 War b8 上接近 fixed R3，同时在 War eval128 上拒绝 bad fallback，则它就是当前最强 no-harm gate。
2. 如果 sentinel 过于保守，则增加 `sentinel_loss_slack` 或 `sentinel_tokens`。
3. 如果 sentinel 仍误触发，说明 block 开头 token 不能代表整个 block，需要改成 sparse sentinel（例如 block 开头、中间位置各取少量 token）。

### sentinel_block_fallback 实验结果

服务器网络带宽较小，因此本轮只通过 SSH 返回聚合表，不拉取日志、不传输输出目录。所有结果来自远端本地 CSV 聚合。

`sentinel_tokens=8` 的关键结果：

| run | blocks | avg_delta_loss | avg_delta_ppl | method_seconds | baseline_seconds | triggered | sentinel_accept | combos |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| War b8 none | 8 | 0.017480 | 1.586619 | 17.256 | 16.652 | 0 | 0 | 0,6; 0,6; 0,7; 0,7; 0,7; 0,13; 0,7; 0,6 |
| War b8 fixed R3 | 8 | -0.011840 | -0.100134 | 17.465 | 16.832 | 6 | 0 | 0,7; 0,13; 0,7; 0,13; 0,13; 0,13; 0,13; 0,13 |
| War b8 sentinel s8 | 8 | -0.009581 | -0.001981 | 20.725 | 16.836 | 5 | 5 | 0,7; 0,13; 0,7; 0,13; 0,13; 0,13; 0,7; 0,13 |
| Monte b8 none | 8 | 0.014333 | 0.108444 | 17.499 | 16.790 | 0 | 0 | 2,0; 2,0; 2,0,7,12; 2,0,7,12; 7,13; 2,7; 2,0; 2,0 |
| Monte b8 fixed R3 | 8 | 0.001515 | -0.056621 | 17.443 | 16.765 | 2 | 0 | 2,0; 2,0; 2,0,7,12; 2,0; 2,0; 2,7; 2,0; 2,0 |
| Monte b8 sentinel s8 | 8 | 0.001515 | -0.056621 | 18.330 | 16.626 | 2 | 2 | 2,0; 2,0; 2,0,7,12; 2,0; 2,0; 2,7; 2,0; 2,0 |
| War eval128 none | 4 | -0.001551 | -0.009529 | 17.234 | 16.606 | 0 | 0 | 0,6; 0,6; 0,7; 0,7 |
| War eval128 fixed R3 | 4 | 0.014290 | 0.365961 | 17.161 | 16.563 | 3 | 0 | 0,7; 0,6; 0,13; 7,6 |
| War eval128 sentinel s8 | 4 | -0.002501 | -0.079488 | 19.029 | 16.774 | 2 | 2 | 0,7; 0,6; 0,13; 0,7 |
| Monte eval128 none | 4 | -0.002673 | -0.073396 | 17.481 | 16.818 | 0 | 0 | 2,0; 2,0; 2,0; 2,0 |
| Monte eval128 sentinel s8 | 4 | -0.002673 | -0.073396 | 18.829 | 17.597 | 0 | 0 | 2,0; 2,0; 2,0; 2,0 |

`sentinel_tokens=4` 的低开销结果：

| run | blocks | avg_delta_loss | avg_delta_ppl | method_seconds | baseline_seconds | triggered | sentinel_accept | combos |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| War b8 sentinel s4 | 8 | -0.009951 | -0.007080 | 19.350 | 17.002 | 4 | 4 | 0,7; 0,13; 0,7; 0,13; 0,7; 0,13; 0,7; 0,13 |
| Monte b8 sentinel s4 | 8 | 0.001515 | -0.056621 | 18.447 | 17.185 | 2 | 2 | 2,0; 2,0; 2,0,7,12; 2,0; 2,0; 2,7; 2,0; 2,0 |
| War eval128 sentinel s4 | 4 | -0.002501 | -0.079488 | 18.397 | 16.970 | 2 | 2 | 0,7; 0,6; 0,13; 0,7 |
| Monte eval128 sentinel s4 | 4 | -0.002673 | -0.073396 | 17.516 | 16.933 | 0 | 0 | 2,0; 2,0; 2,0; 2,0 |

结论：

1. `sentinel_block_fallback` 是目前最强的 no-harm gate 方向：War b8 从 no-rescue 的 `avg_delta_ppl=+1.586619` 拉到负 delta；War eval128 避免了 fixed R3 的明显误伤。
2. `sentinel s4` 基本保留 `s8` 的质量，但 gate 开销更低，应作为下一轮默认设置。
3. Monte b8 上 sentinel 与宽松 fixed R3 质量相同，说明 sentinel 没有破坏已有有效 fallback。
4. 当前主要问题是速度：sentinel gate 额外跑 original/proposed 两条短 stream，`method_seconds` 仍高于 baseline。论文方法需要把 sentinel probe 做成低频、批处理或复用式实现。
5. 创新性上，主线可以更明确地写成：**counterfactual no-harm rescue gate**，即 fallback 不是由静态阈值决定，而是由在线 compressed-policy counterfactual check 验证。

### sentinel 前缀复用优化

上一版 sentinel 在 gate 阶段已经跑过 eval block 前缀，但正式 `pcic_eval` 又从 block 开头重新跑一遍，因此速度统计过于保守。新实现改为：

```text
如果 sentinel proposal 触发：
  先跑 original/proposed 两条 sentinel prefix；
  选中 winning prefix；
  正式 eval 只从 prefix 后面的 remainder token 继续；
  method_seconds = winning_prefix_seconds + rejected_probe_seconds + remainder_seconds。
```

这样质量完全不变，但避免重复计算 winning prefix。低带宽远端聚合结果：

| run | avg_delta_ppl | method_seconds | baseline_seconds | method/base | gate_seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| War b8 sentinel s4 old | -0.007080 | 19.350 | 17.002 | 1.138 | 1.659 |
| War b8 sentinel s4 reuse | -0.007080 | 18.256 | 16.766 | 1.089 | 0.826 |
| Monte b8 sentinel s4 old | -0.056621 | 18.447 | 17.185 | 1.073 | 0.568 |
| Monte b8 sentinel s4 reuse | -0.056621 | 20.536 | 19.567 | 1.050 | 0.338 |
| War eval128 sentinel s4 old | -0.079488 | 18.397 | 16.970 | 1.084 | 0.831 |
| War eval128 sentinel s4 reuse | -0.079488 | 19.460 | 18.757 | 1.037 | 0.461 |
| Monte eval128 sentinel s4 old | -0.073396 | 17.516 | 16.933 | 1.034 | 0.000 |
| Monte eval128 sentinel s4 reuse | -0.073396 | 17.617 | 17.010 | 1.036 | 0.000 |

注意：不同并行轮次的绝对秒数受服务器负载影响，所以更应该看 `method/base`。复用版在触发 sentinel 的三组中都降低了相对开销：

```text
War b8: 1.138 -> 1.089
Monte b8: 1.073 -> 1.050
War eval128: 1.084 -> 1.037
```

结论：前缀复用是正确方向，但仍不够快。下一步速度优化应集中在：

1. original/proposed sentinel probe 合批；
2. sentinel probe 只在高风险 block 触发；
3. 更低频 sentinel，例如每 2 个 block probe 一次，或只 probe top-risk proposal；
4. 最终把 landmark/sentinel 路径改成 kernel/批处理实现。

### sentinel probe 触发过滤

为进一步减少 sentinel probe，新增过滤条件：

```text
--sentinel_min_original_max_gap 0.3
--sentinel_min_original_positive_ratio 0.0
```

含义：只有原 combo 在 calibration 中的 `risk_max_loss_gap >= 0.3`，才运行 original/proposed sentinel probe；否则直接拒绝 fallback proposal，保持原 combo。

低带宽远端扫描结果：

| run | avg_delta_ppl | method/base | triggered | sentinel_accept | probe_allowed |
| --- | ---: | ---: | ---: | ---: | ---: |
| war_b8 g02 | -0.007080 | 1.092 | 4 | 4 | 6 |
| monte_b8 g02 | -0.056621 | 1.025 | 2 | 2 | 2 |
| war_eval128 g02 | -0.079488 | 1.107 | 2 | 2 | 3 |
| monte_eval128 g02 | -0.073396 | 0.994 | 0 | 0 | 0 |
| war_b8 g03 | -0.007080 | 1.109 | 4 | 4 | 5 |
| monte_b8 g03 | -0.056621 | 1.052 | 2 | 2 | 2 |
| war_eval128 g03 | -0.079488 | 1.065 | 2 | 2 | 2 |
| monte_eval128 g03 | -0.073396 | 1.032 | 0 | 0 | 0 |

其中 `g02/g03` 分别表示 `sentinel_min_original_max_gap=0.2/0.3`，positive ratio 不限制。质量都保持不变，说明 max-gap-only 过滤没有破坏 sentinel 的 no-harm 效果。

再看 gate 秒数：

| run | avg_delta_ppl | gate_seconds | probes | accepted |
| --- | ---: | ---: | ---: | ---: |
| war_b8 reuse | -0.007080 | 0.826 | - | 4 |
| war_b8 g03 | -0.007080 | 0.864 | 5 | 4 |
| monte_b8 reuse | -0.056621 | 0.338 | - | 2 |
| monte_b8 g03 | -0.056621 | 0.299 | 2 | 2 |
| war_eval128 reuse | -0.079488 | 0.461 | - | 2 |
| war_eval128 g03 | -0.079488 | 0.287 | 2 | 2 |

结论：`max_gap=0.3` 作为默认 sentinel probe 过滤是合理的。它保留了全部有效质量收益，同时减少无效 probe；绝对秒数有服务器负载噪声，但 gate 秒数和 probe 数显示方向正确。

当前默认方法更新为：

```text
Pairwise-CIC + online blockwise min-loss selection
+ fixed-risk proposal
+ sentinel_tokens=4
+ prefix reuse
+ sentinel_min_original_max_gap=0.3
+ sentinel_min_original_positive_ratio=0.0
```

### calibration-only meta fallback

进一步分析 fixed R3 proposal 的逐 block 结果后，发现一个更便宜的 gate：

```text
calib_meta_fallback

先生成 fixed-risk fallback proposal；
只在以下条件同时满足时接受 proposal：
  original_risk_max_loss_gap >= 0.5
  selected_delta_loss <= original_delta_loss + 0.1
否则拒绝 proposal，保持 original combo。
```

这个规则完全来自 calibration CSV，不需要 sentinel probe，因此没有额外 eval-prefix forward。远端 block-level 诊断中，规则在已有 proposal 上达到 `0 bad / 4 good`：

| rule | total_gain | bad | good | accepted |
| --- | ---: | ---: | ---: | ---: |
| orig_max>=0.5, orig_dl>=-0.1, sel_dl<=orig_dl+0.1 | 0.3244 | 0 | 4 | 4 |

实现参数：

```text
--rescue_strategy calib_meta_fallback
--meta_min_original_max_gap 0.5
--meta_selected_loss_slack 0.1
--meta_min_original_positive_ratio_if_increase 0.4   # refined gate
```

关键实验结果：

| run | avg_delta_ppl | method/base | triggered | meta_accept |
| --- | ---: | ---: | ---: | ---: |
| War b8 none | 1.586619 | 1.036 | 0 | 0 |
| War b8 fixed R3 | -0.100134 | 1.038 | 6 | 0 |
| War b8 sentinel | -0.007080 | 1.109 | 4 | 0 |
| War b8 calib-meta | -0.017791 | 1.010 | 2 | 2 |
| Monte b8 none | 0.108444 | 1.042 | 0 | 0 |
| Monte b8 fixed R3 | -0.056621 | 1.040 | 2 | 0 |
| Monte b8 sentinel | -0.056621 | 1.052 | 2 | 0 |
| Monte b8 calib-meta | -0.056621 | 1.032 | 2 | 2 |
| War eval128 none | -0.009529 | 1.038 | 0 | 0 |
| War eval128 fixed R3 | 0.365961 | 1.036 | 3 | 0 |
| War eval128 sentinel | -0.079488 | 1.065 | 2 | 0 |
| War eval128 calib-meta | -0.009529 | 1.011 | 0 | 0 |
| Monte eval128 none | -0.073396 | 1.039 | 0 | 0 |
| Monte eval128 calib-meta | -0.073396 | 1.037 | 0 | 0 |

held-out offset 验证（`num_blocks=4`, `eval_tokens=64`, offset 从 tokenized text 中间切片，避免只在文本开头过拟合）：

| run | offset | avg_delta_ppl | method/base | triggered | meta_accept | selected combos |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| War none | 8192 | 0.219324 | 1.024 | 0 | 0 | `0,6;0,13;0,7;0,7` |
| War calib-meta | 8192 | 0.212573 | 1.032 | 1 | 1 | `0,6;0,13;0,7;0,13` |
| Monte none | 8192 | -0.017609 | 1.057 | 0 | 0 | unchanged |
| Monte calib-meta | 8192 | -0.017609 | 1.026 | 0 | 0 | unchanged |
| War none | 16384 | 0.045617 | 1.025 | 0 | 0 | `0,7;0,7;0,7;7,6` |
| War calib-meta | 16384 | 0.003629 | 1.027 | 1 | 1 | `0,7;0,13;0,7;7,6` |
| Monte none | 16384 | 0.887313 | 1.037 | 0 | 0 | `2,0;2,0;7,13;7,13` |
| Monte calib-meta | 16384 | 0.910513 | 1.026 | 1 | 1 | `2,0;2,0;7,13;2,0` |

offset 结论：

1. `calib_meta_fallback` 在 War 两个 held-out offset 上都给出正收益，说明 gate 不只是记住文本开头的 block。
2. Monte offset 8192 没有误触发；Monte offset 16384 出现一次小幅负收益，说明当前二条件 gate 仍存在 false accept。
3. 下一步要重点诊断 Monte offset 16384 的触发 block，并检查是否需要加入 proposal 风险下降约束，例如 `selected_risk_max_gap <= original_risk_max_gap - margin` 或 `selected_risk_max_gap <= threshold`。
4. 论文主线暂时不能宣称 strict no-harm，只能表述为 low-overhead calibrated rescue gate；需要更多 held-out offset / 数据集证明误触发率可控。

Monte offset 16384 误触发诊断：

| run | block | original -> selected | eval gain ppl | original max | selected max | original pos | selected pos | selected cal worse |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| War off8192 | 3 | `0,7 -> 0,13` | +0.027005 | 0.836 | 0.522 | 0.438 | 0.438 | +0.0045 |
| War off16384 | 1 | `0,7 -> 0,13` | +0.167952 | 0.603 | 0.327 | 0.562 | 0.562 | +0.0695 |
| Monte off16384 | 3 | `7,13 -> 2,0` | -0.092803 | 0.736 | 0.120 | 0.312 | 0.375 | +0.0300 |

关键观察：误触发样本虽然降低了 `risk_max_loss_gap`，但 `risk_positive_ratio` 从 0.3125 升到 0.375；这表示 proposal 把“少数大错”换成了“更多小错”，在 eval 上不一定安全。直接要求 `selected_positive_ratio <= original_positive_ratio` 会误杀 Monte b8 的一个好样本，因此采用更温和规则：

```text
如果 selected_positive_ratio 上升，
则要求 original_positive_ratio >= 0.4。
```

这个规则的含义是：只有 original 已经表现为“高频不稳定”时，才允许 proposal 用更低 tail risk 换取略高 positive frequency；如果 original 只是孤立 spike，则拒绝把错误频率变高。

基于已有 CSV 的反事实结果：

| run | old avg_delta_ppl | refined avg_delta_ppl | old accept | refined accept | rejected accepts |
| --- | ---: | ---: | ---: | ---: | --- |
| War b8 | -0.017791 | -0.017791 | 2 | 2 | - |
| Monte b8 | -0.056621 | -0.056621 | 2 | 2 | - |
| War eval128 | -0.009529 | -0.009529 | 0 | 0 | - |
| Monte eval128 | -0.073396 | -0.073396 | 0 | 0 | - |
| War off8192 | 0.212573 | 0.212573 | 1 | 1 | - |
| Monte off8192 | -0.017609 | -0.017609 | 0 | 0 | - |
| War off16384 | 0.003629 | 0.003629 | 1 | 1 | - |
| Monte off16384 | 0.910513 | 0.887313 | 1 | 0 | `b3: 7,13 -> 2,0` |

远端实际重跑 `Monte off16384` refined gate 后，`avg_delta_ppl=0.887313`、accepted=0、combo 恢复为 `2,0;2,0;7,13;7,13`，确认代码实现与反事实一致。

### refined gate 后续 offset 验证

新增脚本：

```text
scripts/run_pcic_refined_offset_validation.sh
```

默认验证 `offset=24576,32768`，每个 offset 跑 War/Monte 的 `none` 与 refined `calib_meta_fallback`，远端只输出聚合表，不同步日志目录。

结果：

| run | avg_delta_ppl | method/base | accepted | proposal blocks | combos |
| --- | ---: | ---: | ---: | --- | --- |
| War off24576 none | 0.907596 | 1.054 | 0 | - | `0,13;7,6;0,13;0,13` |
| War off24576 refined | 0.907596 | 1.044 | 0 | `b1:7,6->7,6/a0;b3:0,13->0,13/a0` | `0,13;7,6;0,13;0,13` |
| Monte off24576 none | 1.148182 | 1.037 | 0 | - | `7,13;2,0;7,13;2,0` |
| Monte off24576 refined | 1.148182 | 1.043 | 0 | - | `7,13;2,0;7,13;2,0` |
| War off32768 none | 1.475719 | 1.032 | 0 | - | `0,6;0,13;0,6;0,7` |
| War off32768 refined | 1.475719 | 1.044 | 0 | `b1:0,13->0,13/a0;b2:0,6->0,6/a0;b3:0,7->0,7/a0` | `0,6;0,13;0,6;0,7` |
| Monte off32768 none | -0.469737 | 1.049 | 0 | - | `2,0,7,12;2,0,7,12;2,0,7,12;2,0` |
| Monte off32768 refined | -0.469737 | 1.064 | 0 | - | `2,0,7,12;2,0,7,12;2,0,7,12;2,0` |

解释：

1. refined gate 在新 offset 上没有出现新的 false accept，安全性比旧二条件 gate 更好。
2. 但 War off24576、Monte off24576、War off32768 仍有较大正 delta，说明问题不只是 rescue gate；`min_loss` 的 16-token online selector 本身会过拟合短 calibration window。
3. proposal 经常是 `original -> original`，说明现有 fixed-risk fallback 没有产生真正替代策略；需要把贡献从“rescue threshold”升级为“selector + rescue gate”的联合方法。

困难 offset 的静态 combo 诊断：

| run | avg_delta_ppl | avg_delta_loss | method/base | combos |
| --- | ---: | ---: | ---: | --- |
| War off32768 static `7,6` | 2.220705 | 0.082283 | 1.020 | all `7,6` |
| War off32768 static `0,6` | 1.373499 | 0.048672 | 1.041 | all `0,6` |
| War off32768 static `0,7` | 0.341011 | 0.022053 | 1.036 | all `0,7` |
| War off32768 static `0,13` | -0.330803 | -0.012204 | 1.042 | all `0,13` |
| Monte off24576 static `2,0` | 0.398947 | 0.013638 | 1.054 | all `2,0` |
| Monte off24576 static `2,7` | 1.931283 | 0.066567 | 1.036 | all `2,7` |
| Monte off24576 static `2,0,7,12` | 1.323170 | 0.043621 | 1.079 | all `2,0,7,12` |
| Monte off24576 static `7,13` | 1.148047 | 0.044807 | 1.011 | all `7,13` |

这个诊断非常重要：候选集合里确实存在更好的策略，尤其 War off32768 的 `0,13` 能从 `+1.475719` 变成 `-0.330803`。因此当前瓶颈不是候选集合表达能力，而是 online selector 的短窗估计不稳。

从 calibration risk 聚合看，正确静态锚点可以由 counterfactual risk 推出来：

| run | combo | avg cal delta loss | avg max gap | avg pos ratio | worst max gap |
| --- | --- | ---: | ---: | ---: | ---: |
| War off32768 | `0,13` | 0.029340 | 0.312 | 0.531 | 0.514 |
| War off32768 | `0,6` | 0.004226 | 0.560 | 0.484 | 1.073 |
| War off32768 | `0,7` | 0.029819 | 0.683 | 0.500 | 1.194 |
| War off32768 | `7,6` | 0.044759 | 1.085 | 0.516 | 1.841 |
| Monte off24576 | `2,0` | 0.000654 | 0.142 | 0.500 | 0.176 |
| Monte off24576 | `2,0,7,12` | 0.025043 | 0.562 | 0.547 | 0.748 |
| Monte off24576 | `2,7` | 0.010269 | 0.399 | 0.547 | 0.520 |
| Monte off24576 | `7,13` | 0.019336 | 0.730 | 0.500 | 1.056 |

`risk_budget` selector 在 Monte off24576 上已经验证了这个方向有用：

| run | avg_delta_ppl | method/base | accepted | combos |
| --- | ---: | ---: | ---: | --- |
| Monte off24576 none | 1.148182 | 1.037 | 0 | `7,13;2,0;7,13;2,0` |
| Monte off24576 riskbudget | 0.583588 | 1.040 | 0 | `2,0,7,12;2,0;2,0;2,0` |
| War off32768 none | 1.475719 | 1.032 | 0 | `0,6;0,13;0,6;0,7` |
| War off32768 riskbudget | 1.475719 | 1.034 | 0 | `0,6;0,13;0,6;0,7` |

一个宽松 gate 也能小幅改善 War off32768，但还远不如 static anchor：

| run | avg_delta_ppl | accepted | combos |
| --- | ---: | ---: | --- |
| War off32768 none | 1.475719 | 0 | `0,6;0,13;0,6;0,7` |
| War off32768 widegate | 1.245829 | 1 | `0,6;0,13;0,13;0,7` |
| War off32768 static `0,13` | -0.330803 | 0 | `0,13;0,13;0,13;0,13` |

### Risk Memory selector 初版

已实现 `--combo_select_policy risk_memory`：

```text
每个 block 仍先跑 Pairwise-CIC calibration；
对每个 combo 维护跨 block 的 counterfactual risk memory；
当前 block 只在 min calibration loss + risk_memory_loss_slack 内选候选；
候选排序优先最小化历史+当前的 avg max loss-gap / worst max-gap / positive ratio。
```

关键参数：

```text
--combo_select_policy risk_memory
--risk_memory_loss_slack 0.2
```

同时修正 refined rescue gate：proposal 不允许增加 max loss-gap。

```text
--meta_max_gap_increase 0.0
```

原因：risk-memory 在 War off32768 已经选到低风险锚点 `0,13` 后，旧 gate 会把 block2 反向 fallback 到更高 max-gap 的 `0,6`。这个行为和 rescue gate 的定义冲突，因此加入 max-gap 单调性约束。

验证结果：

| run | avg_delta_ppl | method/base | accepted | combos |
| --- | ---: | ---: | ---: | --- |
| War off32768 none | 1.475719 | 1.032 | 0 | `0,6;0,13;0,6;0,7` |
| War off32768 riskmemory old-gate | 1.117246 | 1.036 | 1 | `0,6;0,13;0,6;0,13` |
| War off32768 riskmemory mono-gate | 0.887356 | 1.034 | 0 | `0,6;0,13;0,13;0,13` |
| War off32768 static `0,13` | -0.330803 | 1.042 | 0 | `0,13;0,13;0,13;0,13` |
| Monte off24576 none | 1.148182 | 1.037 | 0 | `7,13;2,0;7,13;2,0` |
| Monte off24576 riskbudget | 0.583588 | 1.040 | 0 | `2,0,7,12;2,0;2,0;2,0` |
| Monte off24576 riskmemory mono-gate | 0.398947 | 1.002 | 0 | `2,0;2,0;2,0;2,0` |
| Monte off24576 static `2,0` | 0.398947 | 1.054 | 0 | `2,0;2,0;2,0;2,0` |

解释：

1. `risk_memory` 在 Monte off24576 直接达到当前候选集合的 static oracle `2,0`，把 `+1.148182` 降到 `+0.398947`。
2. `risk_memory` 在 War off32768 把后续 block 拉向低风险锚点 `0,13`，从 `+1.475719` 降到 `+0.887356`；主要剩余误差来自 block0，当前 online setting 没有历史 memory，因此仍选 `0,6`。
3. `meta_max_gap_increase=0.0` 是必要修正：rescue gate 不能把 risk-memory 的低风险选择替换成更高 max-gap 的 proposal。
4. 这说明主线应从单纯 rescue gate 升级为 selector drift control：`risk_memory` 负责减少短窗选择漂移，refined gate 负责防止 fallback 误触发。

### 完整 offset 消融：risk-memory 不是单调胜出

新增脚本：

```text
scripts/run_pcic_riskmemory_offset_ablation.sh
```

默认跑 `offset=8192/16384/24576/32768`，比较：

```text
min_loss
risk_budget
risk_memory + monotonic refined gate
```

结果：

| dataset | offset | min_loss | risk_budget | risk_memory | best |
| --- | ---: | ---: | ---: | ---: | --- |
| War | 8192 | 0.212573 | 0.200032 | 0.200032 | risk_budget/risk_memory |
| Monte | 8192 | -0.017609 | -0.195981 | -0.029401 | risk_budget |
| War | 16384 | 0.003629 | 0.006243 | 0.006243 | min_loss |
| Monte | 16384 | 0.887313 | 1.593497 | 0.218657 | risk_memory |
| War | 24576 | 0.907596 | 0.784164 | 0.617966 | risk_memory |
| Monte | 24576 | 1.148182 | 0.583588 | 0.398947 | risk_memory |
| War | 32768 | 1.475719 | 1.475719 | 0.887356 | risk_memory |
| Monte | 32768 | -0.469737 | -0.269763 | 0.159219 | min_loss |

聚合观察：

1. `risk_memory` 对困难正 delta 点有明显价值：Monte off16384、War off24576、Monte off24576、War off32768 都显著优于 `min_loss`。
2. `risk_budget` 在 Monte off8192 最好，说明“低 tail-risk”确实是有效信号，但固定排序不稳定。
3. Monte off32768 是重要反例：`min_loss=-0.469737`，而 `risk_memory=+0.159219`。这里 risk-memory 过度偏向低风险 `2,0`，错过了 calibration 明显更好的 `2,0,7,12`。
4. 因此 paper 主方法不能简单写成 “risk_memory 永远替代 min_loss”；更合理的创新点是 **memory anchor + online arbitration**。

Monte off32768 的失败机制：

| block | min-loss combo | min-loss cal delta | min-loss max-gap | risk-memory combo | 现象 |
| ---: | --- | ---: | ---: | --- | --- |
| 0 | `2,0,7,12` | -0.031827 | 0.474 | `2,0` | risk-memory 过度保守 |
| 1 | `2,0,7,12` | -0.045982 | 0.423 | `2,0` | risk-memory 过度保守 |
| 2 | `2,0,7,12` | -0.170694 | 0.338 | `2,0` | calibration 强烈支持 min-loss |
| 3 | `2,0` | 0.005863 | 0.351 | `2,0` | 两者一致 |

这个反例对论文反而有价值：它说明 selector drift 不是“越保守越好”，需要一个 arbitration gate 判断什么时候相信短窗 calibration，什么时候相信跨 block risk memory。

### Cold-start risk-memory seed

为了解决 War off32768 的 block0 没有历史 memory 的问题，实现了：

```text
--risk_memory_seed_tokens 64
```

方法：在主 eval blocks 之前，用 prefill tail 的 64 个 token 做一次 Pairwise-CIC counterfactual risk calibration，把结果作为 `risk_memory` 的虚拟历史。这个操作在线合法，因为它只使用当前 prefill 内已经可见的 token。

War off32768 结果：

| run | avg_delta_ppl | method/base | combos |
| --- | ---: | ---: | --- |
| min_loss | 1.475719 | 1.034 | `0,6;0,13;0,6;0,7` |
| risk_memory | 0.887356 | 1.034 | `0,6;0,13;0,13;0,13` |
| risk_memory seed64 | -0.330803 | 1.024 | `0,13;0,13;0,13;0,13` |
| static `0,13` oracle | -0.330803 | 1.042 | `0,13;0,13;0,13;0,13` |

这是目前最强的正结果之一：`risk_memory_seed64` 在 War off32768 直接达到 static oracle，把 `+1.475719` 拉到 `-0.330803`。

Monte seed64 检查：

| run | avg_delta_ppl | method/base | combos |
| --- | ---: | ---: | --- |
| Monte off24576 min_loss | 1.148182 | 0.978 | `7,13;2,0;7,13;2,0` |
| Monte off24576 risk_memory | 0.398947 | 1.002 | `2,0;2,0;2,0;2,0` |
| Monte off24576 risk_memory seed64 | 0.398947 | 1.039 | `2,0;2,0;2,0;2,0` |
| Monte off32768 min_loss | -0.469737 | 1.043 | `2,0,7,12;2,0,7,12;2,0,7,12;2,0` |
| Monte off32768 risk_memory | 0.159219 | 0.977 | `2,0;2,0;2,0;2,0` |
| Monte off32768 risk_memory seed64 | 0.159219 | 1.046 | `2,0;2,0;2,0;2,0` |

解释：

1. seed64 不伤 Monte off24576，但也不能修复 Monte off32768。
2. Monte off32768 需要的不是更强 memory，而是 **min-loss vs memory-anchor arbitration**。
3. 下一步最关键的方法应是：当 memory anchor 与 min-loss 不一致时，用极短 sentinel prefix 比较两者，而不是固定相信某一个。

下一步候选方法：

```text
PCIC-RM-Sentinel Arbitration

1. Pairwise-CIC 产生 current min-loss policy。
2. Risk-memory 产生 memory-anchor policy。
3. 如果两者一致，直接执行。
4. 如果两者不一致，用前 s=4/8 个 eval token 做 online counterfactual check。
5. 只复用胜者 prefix，继续跑剩余 block，控制 overhead。
```

这会把前面两个失败模式统一起来：

```text
War off32768: sentinel 应选择 memory-anchor 0,13。
Monte off32768: sentinel 应选择 current min-loss 2,0,7,12。
```

如果这个成立，paper 主线就更完整：不是简单 threshold，也不是单纯 memory，而是 **counterfactual risk memory proposes anchors, sentinel arbitration prevents both drift and over-conservatism**。

### PCIC-RM-Sentinel Arbitration 初版实现

已实现：

```text
--combo_select_policy risk_memory_sentinel
--risk_memory_seed_tokens 64
--sentinel_tokens 8
--sentinel_loss_slack 0.03
--rescue_strategy none
```

实现位置：`src/run_pcic_rescue_blockwise_local.py`。

逻辑：

```text
1. 当前 block 同时计算 min-loss policy 和 risk-memory policy。
2. 如果二者一致，直接执行。
3. 如果二者不一致，用 eval block 前 s 个 token 做 sentinel 比较。
4. 默认选择 sentinel loss 更低者；给 memory-anchor 一个小 slack。
5. 复用胜者 sentinel prefix，只继续跑 remainder，避免重复完整 block。
```

为什么要给 memory-anchor slack：

```text
memory-anchor 是跨 block 风险先验；
短 sentinel prefix 本身也会 noisy；
当 memory 只比 min-loss 差很小（例如 loss +0.02）时，继续信任 memory 可以避免短 prefix 误判。
```

复现实验脚本：

```text
scripts/run_pcic_rm_sentinel_key_experiments.sh
```

关键结果：

| run | avg_delta_ppl | method/base | sentinel decisions | memory selected | combos |
| --- | ---: | ---: | --- | ---: | --- |
| War off32768 min_loss | 1.475719 | 1.034 | - | 0 | `0,6;0,13;0,6;0,7` |
| War off32768 risk_memory seed64 | -0.330803 | 1.024 | - | 0 | `0,13;0,13;0,13;0,13` |
| War off32768 RM-sentinel s8 slack03 | -0.330803 | 1.134 | `b0:0,6|0,13->0,13; b2:0,6|0,13->0,13; b3:0,7|0,13->0,13` | 3 | `0,13;0,13;0,13;0,13` |
| Monte off32768 min_loss | -0.469737 | 1.043 | - | 0 | `2,0,7,12;2,0,7,12;2,0,7,12;2,0` |
| Monte off32768 risk_memory seed64 | 0.159219 | 1.046 | - | 0 | `2,0;2,0;2,0;2,0` |
| Monte off32768 RM-sentinel s8 slack03 | -0.757003 | 1.150 | `b0:2,0,7,12|2,0->2,0,7,12; b1:2,0,7,12|2,0->2,0; b2:2,0,7,12|2,0->2,0,7,12` | 1 | `2,0,7,12;2,0;2,0,7,12;2,0` |

结论：

1. War off32768：RM-sentinel 达到 risk-memory/static oracle，把 `+1.475719` 拉到 `-0.330803`。
2. Monte off32768：RM-sentinel 不但修复 risk-memory 过度保守，还优于纯 `min_loss`：`-0.757003` vs `-0.469737`。
3. 这是目前最像 paper 主方法的版本：memory-anchor 解决 min-loss 短窗漂移，sentinel arbitration 解决 memory-anchor 过度保守。
4. 当前代价是 method/base 约 `1.13-1.15`，但这是 Python/eager 实现；论文速度需要后续 kernel/batching 优化。

这条主线的创新性比前面版本更强：

```text
不是 SparQ/landmark 的 attention 选择变体；
不是简单的 threshold fallback；
而是在线 KV 压缩策略的 counterfactual sequential decision：

local calibration -> risk-memory anchor -> sentinel arbitration -> prefix reuse。
```

### 完整 offset 验证与 Sentinel-All

完整 offsets 上，二选一 `risk_memory_sentinel` 并非单调胜出：

| dataset | offset | min_loss | risk_memory | rm_sentinel s8/slack0.03 |
| --- | ---: | ---: | ---: | ---: |
| War | 8192 | 0.212573 | 0.200032 | -0.113199 |
| War | 16384 | 0.003629 | 0.006243 | 0.027383 |
| War | 24576 | 0.907596 | 0.617966 | 0.884401 |
| War | 32768 | 1.475719 | 0.887356 | -0.330803 |
| Monte | 8192 | -0.017609 | -0.029401 | -0.047421 |
| Monte | 16384 | 0.887313 | 0.218657 | 0.910513 |
| Monte | 24576 | 1.148182 | 0.398947 | 0.398947 |
| Monte | 32768 | -0.469737 | 0.159219 | -0.757003 |

失败模式：

1. `rm_sentinel` 能修复 War32768 / Monte32768，但会在 Monte16384 误信短 sentinel。
2. War24576 的最佳候选不是 min-loss 或 memory-anchor，而是第三个 combo；二选一仲裁无法发现。

因此实现了：

```text
--combo_select_policy risk_memory_sentinel_all
--sentinel_all_min_margin 0.1
```

逻辑：

```text
对所有候选 combo 跑 s-token sentinel；
如果 best-vs-runner-up sentinel margin >= 0.1，则接受 best；
否则回退到 risk-memory anchor。
```

完整结果：

| dataset | offset | min_loss | risk_memory | rm_sentinel | sentall_conf | sentall_conf combos | confident blocks |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| War | 8192 | 0.212573 | 0.200032 | -0.113199 | -0.119951 | `0,7;0,7;0,13;0,13` | - |
| War | 16384 | 0.003629 | 0.006243 | 0.027383 | 0.006243 | `0,13;0,13;0,13;0,13` | - |
| War | 24576 | 0.907596 | 0.617966 | 0.884401 | -1.021527 | `0,13;0,7;7,6;0,13` | `b2:7,6 m=0.420` |
| War | 32768 | 1.475719 | 0.887356 | -0.330803 | -0.330803 | `0,13;0,13;0,13;0,13` | - |
| Monte | 8192 | -0.017609 | -0.029401 | -0.047421 | -0.029401 | `2,0;2,0;2,0;2,0` | `b3:2,0 m=0.131` |
| Monte | 16384 | 0.887313 | 0.218657 | 0.910513 | 0.218657 | `2,0;2,0;2,0;2,0` | - |
| Monte | 24576 | 1.148182 | 0.398947 | 0.398947 | 0.406553 | `2,0;2,0,7,12;2,0;2,0` | `b1:2,0,7,12 m=0.113` |
| Monte | 32768 | -0.469737 | 0.159219 | -0.757003 | 0.159219 | `2,0;2,0;2,0;2,0` | - |

聚合：

| method | mean avg_delta_ppl | worst avg_delta_ppl | wins |
| --- | ---: | ---: | ---: |
| min_loss | 0.518458 | 1.475719 | 1 |
| risk_memory | 0.307377 | 0.887356 | 2 |
| rm_sentinel | 0.121602 | 0.910513 | 4 |
| sentall_conf | -0.088876 | 0.406553 | 4 |

结论：

1. `sentall_conf` 当前整体最好：平均 delta 已经转负，worst-case 从 `1.475719` 降到 `0.406553`。
2. 它修复 War24576 这种 “第三候选才是最优” 的情况，这是二选一 RM-sentinel 做不到的。
3. 它仍错过 Monte32768 的 pairwise RM-sentinel 强结果（`-0.757003`），因为 all-candidate sentinel margin 没达到 0.1 后回退到 memory anchor。
4. 下一步应做 **confidence-routed arbitration**：高置信 all-candidate sentinel 用 `sentall_conf`；否则当 pairwise memory-vs-minloss sentinel delta 绝对值足够大时，用 pairwise RM-sentinel；再否则回退 memory anchor。

这给 paper 贡献一个更清晰的层次：

```text
Counterfactual Risk Memory 负责稳定性；
Sentinel-All 负责发现短窗漏掉的第三候选；
Confidence Routing 负责避免 noisy sentinel 误判。
```

### Confidence-Routed Arbitration

已实现：

```text
--combo_select_policy risk_memory_confidence_routed
--risk_memory_seed_tokens 64
--sentinel_tokens 8
--sentinel_loss_slack 0.03
--sentinel_all_min_margin 0.1
--sentinel_pairwise_min_margin 0.05
```

复现实验脚本：

```text
scripts/run_pcic_confidence_routed_offset_validation.sh
```

路由规则：

```text
1. 先跑 all-candidate sentinel。
2. 如果 best-vs-runner-up margin >= 0.1，说明 sentinel 高置信，使用 Sentinel-All 决策。
3. 如果 all-candidate sentinel 低置信，则只看 min-loss vs memory-anchor 的 pairwise sentinel。
4. 只有 memory 比 min-loss 明显差（delta >= 0.05）时，才切到 min-loss；否则回退 memory anchor。
```

完整 offset 结果：

| dataset | offset | min_loss | risk_memory | rm_sentinel | sentall_conf | confidence_routed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| War | 8192 | 0.212573 | 0.200032 | -0.113199 | -0.119951 | -0.113199 |
| War | 16384 | 0.003629 | 0.006243 | 0.027383 | 0.006243 | -0.014605 |
| War | 24576 | 0.907596 | 0.617966 | 0.884401 | -1.021527 | -1.021527 |
| War | 32768 | 1.475719 | 0.887356 | -0.330803 | -0.330803 | -0.330803 |
| Monte | 8192 | -0.017609 | -0.029401 | -0.047421 | -0.029401 | -0.047421 |
| Monte | 16384 | 0.887313 | 0.218657 | 0.910513 | 0.218657 | 0.218657 |
| Monte | 24576 | 1.148182 | 0.398947 | 0.398947 | 0.406553 | 0.406553 |
| Monte | 32768 | -0.469737 | 0.159219 | -0.757003 | 0.159219 | -0.265406 |

聚合：

| method | mean avg_delta_ppl | worst avg_delta_ppl | wins |
| --- | ---: | ---: | ---: |
| min_loss | 0.518458 | 1.475719 | 0 |
| risk_memory | 0.307377 | 0.887356 | 2 |
| rm_sentinel | 0.121602 | 0.910513 | 4 |
| sentall_conf | -0.088876 | 0.406553 | 4 |
| confidence_routed | -0.145969 | 0.406553 | 5 |

结论：

1. `confidence_routed` 当前是最强版本：平均 delta 最低，wins 最多，worst-case 与 `sentall_conf` 持平。
2. 它保留 `sentall_conf` 对 War24576 的强收益，同时修复 War16384 / Monte8192 的低置信 noisy sentinel。
3. Monte32768 仍没达到 pairwise RM-sentinel 的 `-0.757003`，因为 pairwise min-loss fallback 只在 `delta>=0.05` 时触发；当前得到 `-0.265406`，仍显著优于 `risk_memory=0.159219`。
4. 这已经形成相对完整的 paper 方法：不是单个阈值，而是可解释的三层在线决策系统。

当前主方法建议命名：

```text
PCIC-CR: Pairwise-CIC with Counterfactual-Risk Memory and Confidence-Routed Sentinel Arbitration
```

核心贡献表述：

```text
我们把 KV cache 压缩从“每个 block 独立选最小 calibration loss”
提升为“带 counterfactual risk memory 的在线序列决策”。

Risk memory 提供稳定 anchor；
Sentinel-All 在高置信时发现第三候选；
Pairwise sentinel 在低置信时只处理 min-loss vs memory 的局部冲突；
Prefix reuse 控制仲裁开销。
```

### Key main settings 验证

新增复现实验脚本：

```text
scripts/run_pcic_confidence_routed_key_main.sh
```

验证 `b8` 与 `eval128`：

| run | avg_delta_ppl | method/base | combos |
| --- | ---: | ---: | --- |
| War b8 none | 1.586619 | 1.036 | `0,6;0,6;0,7;0,7;0,7;0,13;0,7;0,6` |
| War b8 calib-meta | -0.017791 | 1.010 | `0,6;0,6;0,7;0,13;0,7;0,13;0,7;0,13` |
| War b8 confroute | -0.134311 | 1.426 | `0,6;0,13;0,7;0,13;0,13;0,13;0,13;0,13` |
| Monte b8 none | 0.108444 | 1.042 | `2,0;2,0;2,0,7,12;2,0,7,12;7,13;2,7;2,0;2,0` |
| Monte b8 calib-meta | -0.056621 | 1.032 | `2,0;2,0;2,0,7,12;2,0;2,0;2,7;2,0;2,0` |
| Monte b8 confroute | -0.118596 | 1.435 | `2,0;2,0;2,0,7,12;2,0;2,0;2,0;2,0;2,0` |
| War eval128 none | -0.009529 | 1.038 | `0,6;0,6;0,7;0,7` |
| War eval128 calib-meta | -0.009529 | 1.011 | `0,6;0,6;0,7;0,7` |
| War eval128 confroute | -0.009529 | 1.236 | `0,6;0,6;0,7;0,7` |
| Monte eval128 none | -0.073396 | 1.039 | `2,0;2,0;2,0;2,0` |
| Monte eval128 calib-meta | -0.073396 | 1.037 | `2,0;2,0;2,0;2,0` |
| Monte eval128 confroute | -0.073396 | 1.236 | `2,0;2,0;2,0;2,0` |

结论：

1. `confidence_routed` 在 b8 上比 `calib-meta` 进一步提升：War `-0.017791 -> -0.134311`，Monte `-0.056621 -> -0.118596`。
2. 在 eval128 上不伤质量，但会带来额外开销；这说明 routing gate 需要跳过明显一致/低风险 block，或者把 sentinel probes batching/kernel 化。
3. 当前 method/base 偏高（b8 约 `1.43`，eval128 约 `1.24`），这是 Python/eager 多候选 sentinel 的实现成本，不应作为最终论文速度。
4. 下一步速度主线应是：只在 min-loss 与 memory-anchor 不一致、且 calibration risk gap 足够大时跑 sentinel；all-candidate sentinel 只在高不确定 block 启用。

速度优化方向：

```text
Fast PCIC-CR:
1. default path: risk_memory anchor, no sentinel。
2. pairwise path: only min-loss vs memory, s=4/8 prefix。
3. all-candidate path: only when pairwise is low-confidence and calibration candidates are close。
4. batch sentinel candidates in one forward/kernel。
```

当前方法应从 `calib_meta_fallback` 升级为：

```text
Pairwise-CIC + online blockwise selection
+ Counterfactual Risk Memory / Risk Anchor Selector
+ refined rescue gate
```

核心想法：

```text
短窗 calibration loss 决定局部适配；
跨 block counterfactual tail-risk memory 决定稳定锚点；
rescue gate 只在局部选择偏离稳定锚点且风险上升时介入。
```

这比单纯改阈值更像 paper contribution：它把 Pairwise-CIC 从“每个 block 独立估计”提升为“在线序列决策”，用历史 counterfactual risk 控制 selector drift。

结论：

1. refined `calib_meta_fallback` 是当前最强的低开销默认 gate：War/Monte b8 保持原有收益，eval128 不误触发，并修复了 Monte offset 16384 的一次 false accept。
2. 相比 sentinel，它牺牲了一点 War eval128 的额外收益，但大幅降低开销；更适合作为主方法。
3. sentinel 现在更适合作为 optional stronger gate / upper-bound 消融，用来证明 online counterfactual check 的上限。
4. 当前 paper 主线应更新为：

```text
Pairwise-CIC + online blockwise selection + calibration-only no-harm meta rescue gate
```

核心创新点：

```text
用 calibration loss-gap tail risk 生成 proposal，
再用 no-harm meta condition 过滤 proposal，
避免 fixed threshold rescue gate 的误触发。
```

### 关键结论

1. R3 是目前最有希望的主线：它在 War/Monte 的 2-block 和 4-block 设置下都保持负 delta，没有 R2 的三路 forward 开销。
2. 4-block 结果尤其重要：`none minloss` 在 War/Monte 都出现正 delta，而 R3 都拉回负 delta，说明 block-level risk fallback 确实在抑制在线选择漂移。
3. R3 的速度和普通 layerbudget eval 基本同量级，因为 eval 阶段不逐 token rescue，也不跑 probe stream。
4. `risk_pareto` 的负结果很有价值：它说明 tail risk 不能替代 calibration loss，只能作为 rescue/fallback 风险约束；这让论文主张更精确。
5. 8-block 结果进一步证明 R3 的价值：War 和 Monte 都存在 no-rescue 正 delta，合适的 R3 gate 能把它拉回接近或优于 baseline。
6. 128-token 结果说明固定阈值 R3 还不够稳：当 no-rescue 已经安全时，fallback 可能引入新的错误策略。
7. adaptive_block_fallback 第一版和 cal32 都是负结果，说明只靠更长 calibration 或简单 tail-risk 统计不足以构成 paper 主方法。
8. 主线应升级为 `Pairwise-CIC + risk-memory online selector + refined no-harm rescue gate`，把“短窗 selector drift control”作为核心技术目标。
9. `sentinel_block_fallback` 是当前最值得验证的 no-harm gate，因为它直接在线比较 original/proposed compressed policies，而不是继续猜阈值。
10. 当前最佳实验点是 `sentinel_tokens=4`：质量接近 s8，开销更低，适合作为下一步优化对象。
11. sentinel 前缀复用能在不改变 PPL 的情况下降低相对开销，是后续速度优化的基本实现方式。
12. max-gap-only sentinel 过滤保留质量收益，并减少无效 probe；当前默认阈值设为 `sentinel_min_original_max_gap=0.3`。
13. refined `calib_meta_fallback` 当前优先级高于 sentinel：它用 calibration-only meta rule 获得 b8 质量收益，避免 eval128 误伤，并通过 positive-ratio/max-gap monotonic gate 修复 held-out offset false accept。
14. `risk_memory` selector 是新的主线增量：它用跨 block counterfactual risk memory 抑制短窗 calibration overfitting，在 Monte off24576 达到 static oracle，并显著改善 War off32768。

### 当前 paper 主张更新

现在更合理的论文主线是：

```text
PCIC-CR: Confidence-Routed Counterfactual-Risk KV Budgeting

Pairwise-CIC estimates layer-combo quality in a short calibration window.
Counterfactual risk memory accumulates tail-risk evidence across blocks.
Confidence-routed sentinel arbitration resolves conflicts between local min-loss, memory anchors, and third-candidate policies with prefix reuse.
```

这比 “Pairwise-CIC + landmark” 或 “margin rescue” 更有创新性，因为核心贡献变成：

```text
用跨 block counterfactual loss-gap memory + confidence-routed 短前缀 sentinel arbitration，
而不是单个短窗的平均 loss / attention mass / confidence，
来控制在线 KV 压缩策略。
```

### 下一步实验

1. 以 `confidence_routed` 作为当前默认主方法，继续扩展 offset 和数据集验证。
2. 新增严格消融：`min_loss`、`risk_budget`、`risk_memory`、`risk_memory_seed64`、`risk_memory_sentinel`、`sentall_conf`、`confidence_routed`、`static oracle`。
3. 验证 routing 参数：`sentinel_all_min_margin=0.05/0.1/0.2` 与 pairwise delta 阈值 `0.03/0.05/0.1`，重点尝试进一步修复 Monte32768。
4. 在更多数据集上测试，尤其是 LongBench/RULER/Needle 类长上下文任务；服务器带宽较小时只返回 CSV 聚合小表。
5. 后续再做真正速度优化：把 Python/eager landmark fallback 和 sentinel probe 替换为 kernel/批处理实现，否则当前 seconds 仍不能代表最终论文速度。

### Fast PCIC-CR 低开销路由验证

新增实现：

```text
--combo_select_policy risk_memory_confidence_fast
--confidence_fast_all_min_delta_loss -0.05
```

复现实验脚本：

```text
scripts/run_pcic_confidence_fast_offset_validation.sh
scripts/run_pcic_confidence_fast_key_main.sh
scripts/run_pcic_confidence_fast_long_eval.sh
```

Fast PCIC-CR 的核心变化是减少 sentinel probe 频率：

```text
1. min-loss 与 memory-anchor 不一致：只跑 min-loss vs memory 的 pairwise sentinel。
2. min-loss 与 memory-anchor 一致，且 calibration delta_loss <= -0.05：才跑 all-candidate sentinel。
3. 其它 block 直接使用 memory anchor，跳过 sentinel。
```

完整 offset 验证结果：

| dataset | offset | confidence_routed | confidence_fast | fast method/base | fast combos |
| --- | ---: | ---: | ---: | ---: | --- |
| War | 8192 | -0.113199 | -0.113199 | 1.090 | `0,7;0,7;0,13;0,7` |
| War | 16384 | -0.014605 | -0.014605 | 1.171 | `0,13;0,13;0,7;0,13` |
| War | 24576 | -1.021527 | -1.021527 | 1.267 | `0,13;0,7;7,6;0,13` |
| War | 32768 | -0.330803 | -0.330803 | 1.137 | `0,13;0,13;0,13;0,13` |
| Monte | 8192 | -0.047421 | -0.047421 | 1.137 | `2,0;2,7;2,0;2,0` |
| Monte | 16384 | 0.218657 | 0.218657 | 1.100 | `2,0;2,0;2,0;2,0` |
| Monte | 24576 | 0.406553 | 0.398947 | 1.176 | `2,0;2,0;2,0;2,0` |
| Monte | 32768 | -0.265406 | -0.265406 | 1.141 | `2,0;2,0;2,0,7,12;2,0` |

聚合：

| method | mean_delta_ppl | worst_delta_ppl | mean_method/base | wins |
| --- | ---: | ---: | ---: | ---: |
| confidence_routed | -0.145969 | 0.406553 | 1.432 | 7 |
| confidence_fast | -0.146920 | 0.398947 | 1.153 | 8 |

关键 b8 / eval128 验证：

| run | avg_delta_ppl | method/base | combos |
| --- | ---: | ---: | --- |
| War b8 confroute | -0.134311 | 1.426 | `0,6;0,13;0,7;0,13;0,13;0,13;0,13;0,13` |
| War b8 conffast | -0.134311 | 1.151 | `0,6;0,13;0,7;0,13;0,13;0,13;0,13;0,13` |
| Monte b8 confroute | -0.118596 | 1.435 | `2,0;2,0;2,0,7,12;2,0;2,0;2,0;2,0;2,0` |
| Monte b8 conffast | -0.118596 | 1.106 | `2,0;2,0;2,0,7,12;2,0;2,0;2,0;2,0;2,0` |
| War eval128 confroute | -0.009529 | 1.236 | `0,6;0,6;0,7;0,7` |
| War eval128 conffast | -0.009529 | 1.055 | `0,6;0,6;0,7;0,7` |
| Monte eval128 confroute | -0.073396 | 1.236 | `2,0;2,0;2,0;2,0` |
| Monte eval128 conffast | -0.073396 | 1.039 | `2,0;2,0;2,0;2,0` |

长 eval 负结果：

| run | avg_delta_ppl | method/base | combos |
| --- | ---: | ---: | --- |
| War eval256 | 0.345124 | 1.061 | `0,6;0,13;0,13;0,13` |
| Monte eval256 | 0.078314 | 1.044 | `2,0;2,0;2,0;2,0` |
| War eval512 | 0.879153 | 1.061 | `0,6;0,13;7,6;0,7` |
| Monte eval512 | 0.054653 | 1.046 | `2,0;2,0;2,0,7,12;2,0` |

当前判断：

1. `risk_memory_confidence_fast` 是当前更合理的默认实验点：offset 平均 PPL 不劣于 `confidence_routed`，worst-case 更低，速度开销从 `1.432x` 降到 `1.153x`。
2. b8 / eval128 上 fast 完全保留 `confidence_routed` 的质量收益，同时把开销降到 `1.04-1.15x`。
3. 但它还没有达到“快于 baseline”：当前 Python/eager 实现仍有 `3.9%-15.3%` 开销。
4. eval256/eval512 不是可靠主设置：速度没有继续下降到 `<1.0x`，War PPL 变差，说明长 eval 下 combo selection 的分布漂移更明显。
5. 论文主线可以保留 `PCIC-CR / Fast PCIC-CR`，但速度 claim 不能基于当前 eager 实现；下一步必须做候选合批或真正 sparse kernel，否则只能 claim 质量/稳健性，而不能 claim wall-clock speedup。

下一步优先级：

```text
1. 继续用 eval64/eval128 做方法消融，避免被 eval256/eval512 的 drift 干扰。
2. 测 sentinel_tokens=4 的 conffast，目标是在保持质量时把 eval128 压到接近 1.0x。
3. 做 batched sentinel：一次 forward 比较多个候选，减少 Python/eager 多次调用。
4. 分离论文指标：selector 质量用 eager 验证，速度用 kernel/batched prototype 验证。

### Sentinel token 缩短实验：s4 与 s4/a8 hybrid

为了进一步降低 `Fast PCIC-CR` 的开销，测试了两种更激进的低开销版本：

```text
1. conffast_s4：所有 sentinel probe 都从 8 token 降到 4 token。
2. conffast_s4a8：pairwise sentinel 用 4 token，all-candidate sentinel 用 8 token。
```

新增脚本：

```text
scripts/run_pcic_confidence_fast_s4_validation.sh
scripts/run_pcic_confidence_fast_hybrid_s4a8_validation.sh
```

新增参数：

```text
--sentinel_all_tokens 8
```

该参数只在 all-candidate sentinel 路径覆盖 `--sentinel_tokens`；默认 `0` 时完全保持旧行为。

s4 完整 offset 结果：

| method | mean_delta_ppl | worst_delta_ppl | mean_method/base | wins_vs_s8 |
| --- | ---: | ---: | ---: | ---: |
| conffast_s8 | -0.146920 | 0.398947 | 1.153 | 6 |
| conffast_s4 | 0.142454 | 0.884401 | 1.101 | 5 |

s4 关键设置结果：

| run | s8_delta | s4_delta | s8_ratio | s4_ratio |
| --- | ---: | ---: | ---: | ---: |
| War b8 | -0.134311 | 0.195681 | 1.151 | 1.094 |
| Monte b8 | -0.118596 | -0.056621 | 1.106 | 1.073 |
| War eval128 | -0.009529 | 0.094612 | 1.055 | 1.047 |
| Monte eval128 | -0.073396 | -0.073396 | 1.039 | 1.038 |

s4/a8 hybrid 完整 offset 结果：

| method | mean_delta_ppl | worst_delta_ppl | mean_method/base | wins_vs_s8 |
| --- | ---: | ---: | ---: | ---: |
| conffast_s8 | -0.146920 | 0.398947 | 1.153 | 6 |
| conffast_hybrid | -0.095787 | 0.790202 | 1.109 | 6 |

s4/a8 hybrid 关键设置结果：

| run | s8_delta | hybrid_delta | s8_ratio | hybrid_ratio |
| --- | ---: | ---: | ---: | ---: |
| War b8 | -0.134311 | 0.195681 | 1.151 | 1.094 |
| Monte b8 | -0.118596 | -0.056621 | 1.106 | 1.071 |
| War eval128 | -0.009529 | 0.094612 | 1.055 | 1.047 |
| Monte eval128 | -0.073396 | -0.073396 | 1.039 | 1.039 |

诊断结论：

1. 单纯把 sentinel 从 s8 降到 s4 不可取：速度只从 `1.153x` 降到 `1.101x`，但 mean delta 从 `-0.146920` 退化到 `0.142454`。
2. s4/a8 hybrid 只能修复 all-candidate 路径中的 War24576，但不能修复 pairwise sentinel 的误判；War b8 和 War eval128 仍明显变差。
3. 当前质量瓶颈主要在 **pairwise sentinel 的短前缀可靠性**，不是 all-candidate sentinel。
4. 因此主方法默认仍应使用 `conffast_s8`，不能用 s4/hybrid 作为 paper 主结果。
5. 速度优化方向应从“缩短 sentinel token 数”转向“减少 probe 次数或合批 probe”，即：保持 s8 的判别可靠性，但把多个候选/多个 block 的 probe 合并执行。

对 paper 主线的影响：

```text
Pairwise-CIC + online blockwise selection + rescue gate 仍然成立；
但 rescue gate 的关键不是短到 4 token，而是要有足够稳定的 counterfactual prefix。
这反而强化了论文叙事：PCIC-CR 不是一个调阈值技巧，而是一个在线风险仲裁框架；
速度需要通过 batched sentinel / sparse kernel 实现，而不是牺牲仲裁可靠性。
```

下一步更合理的速度实验：

```text
1. 保持 sentinel_tokens=8。
2. 增加 lazy trigger：对明显会回到 memory-anchor 的 pairwise block，直接跳过 sentinel。
3. 做 batched sentinel prototype：同一 block 的候选 combo 合并到一次 forward。
4. 如果要继续测试更短前缀，应尝试 adaptive s4->s8 cascade：先用 s4，只有 margin 不足时追加到 s8，而不是直接用 s4 决策。
```

### Adaptive s4->s8 cascade sentinel

为了避免 s4 直接决策带来的误判，又尝试了更保守的 cascade：

```text
1. 先对候选 combo 跑 4-token sentinel。
2. pairwise 路径用 abs(memory_loss - min_loss_loss) 作为早停置信度。
3. all-candidate 路径用 best-vs-runner-up margin 作为早停置信度。
4. 只有早停 margin >= sentinel_cascade_accept_margin 才直接采用 s4 决策。
5. 否则复用 s4 prefix，继续补跑到 8-token sentinel，再用 s8 决策。
```

新增实现参数：

```text
--sentinel_cascade_initial_tokens 4
--sentinel_cascade_accept_margin 0.15/0.20
```

新增脚本：

```text
scripts/run_pcic_confidence_fast_cascade4to8_validation.sh
```

m0.15 完整 offset 结果：

| method | mean_delta_ppl | worst_delta_ppl | mean_method/base | wins_vs_s8 |
| --- | ---: | ---: | ---: | ---: |
| conffast_s8 | -0.146920 | 0.398947 | 1.153 | 7 |
| conffast_cascade_m0.15 | -0.146103 | 0.398947 | 1.142 | 7 |

m0.15 关键设置：

| run | s8_delta | cascade_delta | s8_ratio | cascade_ratio | early/extended/triggered |
| --- | ---: | ---: | ---: | ---: | --- |
| War b8 | -0.134311 | -0.134311 | 1.151 | 1.128 | 3/4/7 |
| Monte b8 | -0.118596 | -0.118596 | 1.106 | 1.089 | 2/2/4 |
| War eval128 | -0.009529 | -0.009529 | 1.055 | 1.056 | 0/1/1 |
| Monte eval128 | -0.073396 | -0.073396 | 1.039 | 1.038 | 0/0/0 |

m0.15 的问题：整体接近 s8，但 War off16384 从 `-0.014605` 退化到 `0.027383`，说明早停阈值仍略激进。

m0.20 完整 offset 结果：

| method | mean_delta_ppl | worst_delta_ppl | mean_method/base | wins_vs_s8 |
| --- | ---: | ---: | ---: | ---: |
| conffast_s8 | -0.146920 | 0.398947 | 1.153 | 8 |
| conffast_cascade_m0.20 | -0.146920 | 0.398947 | 1.151 | 8 |

m0.20 关键设置：

| run | s8_delta | cascade_delta | s8_ratio | cascade_ratio | early/extended/triggered |
| --- | ---: | ---: | ---: | ---: | --- |
| War b8 | -0.134311 | -0.134311 | 1.151 | 1.147 | 1/6/7 |
| Monte b8 | -0.118596 | -0.118596 | 1.106 | 1.097 | 1/3/4 |
| War eval128 | -0.009529 | -0.009529 | 1.055 | 1.056 | 0/1/1 |
| Monte eval128 | -0.073396 | -0.073396 | 1.039 | 1.039 | 0/0/0 |

结论：

1. `cascade_m0.20` 是安全版本：在完整 offset 和 key settings 上完全复现 s8 质量。
2. 但速度收益很小：完整 offset 平均只从 `1.153x` 降到 `1.151x`，b8 约降低 `0.4%-0.9%`。
3. 原因是多数关键 block 的 s4 margin 不够高，仍需要补跑到 s8；当前 Python/eager 下拆成 s4+s4 还会增加调度开销。
4. 这个实验验证了一个重要设计原则：**rescue gate 的可靠性需要 s8 级别证据，s4 只能作为 early-exit hint，不能作为主决策证据**。
5. paper 主方法仍应使用 `conffast_s8` 或 `cascade_m0.20`；如果强调最稳健结果，用 s8；如果强调机制完整性，可以把 cascade 作为 optional low-risk speed optimization。

下一步真正能带来速度收益的方向不再是 token 数，而是实现形态：

```text
1. Batched sentinel candidates：同一 block 的多个 combo 合并成一次批处理 forward。
2. Lazy pairwise skip：用更强的 calibration/memory 条件跳过明显会回到 memory-anchor 的 pairwise probe。
3. Kernel-level sparse attention：把当前 Python/eager landmark fallback 换成 fused/kernel 实现。
```

### Lazy pairwise skip gate：后验可行但泛化不足

为了继续降低 pairwise sentinel probe 频率，先做了一个非 GPU 后验分析脚本：

```text
scripts/analyze_pcic_lazy_pairwise_gate.py
```

分析对象是完整 offset 的 `conffast_s8` 与 `riskmemory_monogate` CSV。结论：

| item | value |
| --- | ---: |
| blocks | 32 |
| triggered_pairwise_blocks | 22 |
| s8_memory_selected_pairwise | 18 |
| s8_minloss_selected_pairwise | 4 |

后验网格中较好的 lazy 条件：

```text
min_loss_delta_loss >= -0.025
memory_delta_loss - min_loss_delta_loss <= 0.08
memory_delta_loss <= 0.08
```

后验估计：

| policy | mean_delta_ppl | worst_delta_ppl | saved_pairwise_probes | wrong_skips |
| --- | ---: | ---: | ---: | ---: |
| conffast_s8 | -0.146920 | 1.148294 | 0/22 | 0 |
| lazy posterior best | -0.147764 | 1.148294 | 4/22 | 1 |

因此实现了独立策略：

```text
--combo_select_policy risk_memory_confidence_lazy
--confidence_lazy_pairwise_min_delta_loss -0.025
--confidence_lazy_pairwise_max_calib_gap 0.08
--confidence_lazy_pairwise_max_memory_delta_loss 0.08
```

新增验证脚本：

```text
scripts/run_pcic_confidence_lazy_offset_validation.sh
scripts/run_pcic_confidence_lazy_key_main.sh
```

完整 offset 实测：

| method | mean_delta_ppl | worst_delta_ppl | mean_method/base | wins_vs_fast |
| --- | ---: | ---: | ---: | ---: |
| conffast_s8 | -0.146920 | 0.398947 | 1.153 | 7 |
| conflazy | -0.147764 | 0.398947 | 1.137 | 8 |

offset 结果看起来更好：平均 PPL 略优，速度从 `1.153x` 降到 `1.137x`。

但 key settings 暴露了泛化问题：

| run | conffast_delta | conflazy_delta | conffast_ratio | conflazy_ratio |
| --- | ---: | ---: | ---: | ---: |
| War b8 | -0.134311 | 0.153475 | 1.151 | 1.086 |
| Monte b8 | -0.118596 | -0.118596 | 1.106 | 1.071 |
| War eval128 | -0.009529 | 0.094612 | 1.055 | 1.037 |
| Monte eval128 | -0.073396 | -0.073396 | 1.039 | 1.039 |

结论：

1. `risk_memory_confidence_lazy` 不能作为默认主方法：它在 offset 上有效，但在 War b8 / War eval128 上误跳过关键 pairwise sentinel，导致 PPL 明显退化。
2. 这个负结果说明 calibration-only lazy skip 仍然不够可靠；pairwise sentinel 的价值正是在这些 calibration 信号不充分的 block 上纠偏。
3. 论文主线应保留 `conffast_s8`：它是当前质量最稳的版本；`conflazy` 只能作为“为什么不能只靠 calibration skip”的消融。
4. 速度主线进一步收敛：不要继续堆 calibration-only skip；应优先做 batched sentinel 或 kernel 化。

对创新性的帮助：

```text
lazy skip 的失败反而支持 paper 论点：
短窗 calibration / memory statistics 不能完全替代在线 counterfactual sentinel；
PCIC-CR 的贡献在于把 local estimate、risk memory、sentinel arbitration 组合成一个在线决策系统。
```

### Batched sentinel 可行性检查

进一步检查了当前实现中 `eval_segment` 与 attention patch 的结构：

```text
src/run_pcic_rescue_blockwise_local.py
src/evaluate_qwen3_top2_head_limit3_ppl.py
```

关键发现：

1. 当前 `attention_mode(..., layer_budget_map_path=...)` 使用的是全局 `_ACTIVE_LAYER_BUDGET_MAP_PATH`。
2. `layerbudgetattn` 在一次 forward 内只能使用一个 `layer_budget_map_path`。
3. 因此不能直接把不同 candidate combo 拼成 batch 维度一次 forward，因为 batch 内每一行需要不同 layer-budget map。
4. 如果强行 batch，不同 combo 会共享同一个 budget map，得到的 sentinel loss 不再对应真实候选，结果无效。

这意味着当前代码里“batched sentinel candidates”不是一个简单脚本级优化，而需要修改 attention patch：

```text
目标接口：
attention_mode(..., layer_budget_map_paths=[path_a, path_b, ...])

在 attention mask 生成时：
for batch_row in batch:
    使用该 row 对应的 budget map
```

或者更直接：

```text
预先把 combo -> layer mask / budget spec 编译成 tensor；
forward 时传入 batch_size 个 budget specs；
attention 内按 batch row 选择对应的 compressed layers / landmark rule。
```

工程判断：

1. 只在当前 Python 层循环里“合并候选”不可行，因为 budget map 是全局的。
2. 真正能提升速度的 batched sentinel 需要改 `evaluate_qwen3_top2_head_limit3_ppl.py` 的 attention patch，使其支持 batch-row budget。
3. 这是值得做的：PCIC-CR 的质量主线已经稳定，速度瓶颈主要来自多候选 sentinel 的多次 forward；batch-row budget 是正交工程优化，不改变方法决策。
4. 论文中可以把当前 eager 实现作为算法验证，把 batch-row budget / fused sparse attention 作为速度实现；但如果要 claim wall-clock speedup，必须完成这个 patch 或 kernel prototype。

下一步最小实现路径：

```text
1. 在 layer budget JSON loader 中增加 batch mode：多个 map path -> 多个 layer spec。
2. 在 patched attention 中保留原单 map path 行为，新增 batch row selector。
3. 写一个 smoke：同一 token、两个 combo，batch-row 输出 loss 应接近分别单跑两个 combo。
4. 通过 smoke 后，再把 sentinel candidates 改成 batch-row forward。
```

风险：

```text
Qwen3 KV cache 的 batch 维复制会增加显存。
8 卡 3090 可以承受小 batch sentinel，但需要先从 pairwise batch=2 验证。
如果 batch-row budget 的 Python mask 构造仍很重，最终还需要 fused/kernel 版本。
```

### Batched sentinel 速度上限估计

新增非 GPU 分析脚本：

```text
scripts/analyze_pcic_batched_sentinel_upper_bound.py
```

它只读取已有 `conffast_s8` CSV，估计如果 sentinel candidates 可以 batch-row 合并，当前 method/base 会降到什么程度。

估计方法：

```text
current:
    method_seconds = selected_prefix + gate_seconds + remainder
    gate_seconds = sum(non_selected_candidate_prefix_seconds)

batched upper bound:
    gate_seconds 近似替换为一次 candidate-prefix forward 的时间
```

结果：

| run | current_ratio | oracle_batch_ratio | triggered | mean_candidates |
| --- | ---: | ---: | ---: | ---: |
| War off8192 | 1.090 | 0.968 | 4 | 2.00 |
| War off16384 | 1.171 | 1.041 | 4 | 2.00 |
| War off24576 | 1.267 | 1.041 | 3 | 3.33 |
| War off32768 | 1.137 | 1.040 | 3 | 2.00 |
| Monte off8192 | 1.137 | 1.040 | 3 | 2.00 |
| Monte off16384 | 1.100 | 1.035 | 2 | 2.00 |
| Monte off24576 | 1.176 | 1.112 | 2 | 2.00 |
| Monte off32768 | 1.141 | 1.043 | 3 | 2.00 |
| War b8 | 1.151 | 1.037 | 7 | 2.00 |
| Monte b8 | 1.106 | 1.041 | 4 | 2.00 |
| War eval128 | 1.055 | 1.039 | 1 | 2.00 |
| Monte eval128 | 1.039 | 1.039 | 0 | 0.00 |

聚合：

| aggregate | mean_ppl_delta | mean_current_ratio | mean_oracle_batch_ratio |
| --- | ---: | ---: | ---: |
| all | -0.125932 | 1.131 | 1.040 |

结论：

1. Batched sentinel 值得做，但它最多把当前 eager 版本从平均 `1.13x` 降到约 `1.04x`。
2. 单靠 batched sentinel 不一定能稳定快于 baseline；它主要消除多候选 sentinel 的额外 forward 成本。
3. 剩余 `~4%` 开销来自 layerbudget attention 本身、Python mask 构造、prefix/remainder 拆分等。
4. 如果论文必须 claim wall-clock speedup，下一步必须继续做 fused/kernel sparse attention 或至少把 layer-budget mask 构造 tensor 化。
5. 如果论文主 claim 是 PPL/稳健性和在线决策创新，则当前 `conffast_s8` 已经足够支撑方法主线；速度部分应如实写成“eager prototype overhead，kernel implementation expected to close gap”。

当前推荐 paper 主线表述：

```text
PCIC-CR 是算法贡献：
Pairwise-CIC short-window estimate
+ counterfactual-risk memory
+ confidence-routed online sentinel arbitration.

速度贡献应谨慎：
当前 eager prototype 通过 Fast PCIC-CR 把开销从 1.43x 降到 1.15x；
batch-row sentinel 上限约 1.04x；
真正 speedup 需要 batch-row budget + fused sparse attention。
```

### Related Work 与创新性边界

为了判断 `Pairwise-CIC + online blockwise selection + rescue gate` 是否足够支撑 paper，需要把它和已有 KV/attention 压缩方法明确区分。

代表性已有工作：

| 方法 | 核心思想 | 与 PCIC-CR 的关系 |
| --- | --- | --- |
| SparQ Attention | 用 query 与近似 key 选择少量高相关 KV，再近似 attention 输出，目标是 bandwidth-efficient inference。 | 主要是 token/key 级 query-aware retrieval；不是跨 block 的候选策略选择，也没有 counterfactual risk memory / sentinel arbitration。 |
| H2O | 基于 heavy hitter token 做 KV eviction，保留对未来生成重要的 token。 | 是 token retention/eviction；不是 layer-combo / budget policy 的在线选择。 |
| StreamingLLM | 发现 attention sink，保留 sink + recent tokens 支持无限流式推理。 | 是固定结构的 streaming cache policy；不是数据驱动的 blockwise policy arbitration。 |
| SnapKV | 在 prompt 末端观察 attention，压缩 prompt KV cache，保留被后续 generation 需要的 cluster/token。 | 是 prompt KV token 选择；不是候选 compression policy 的在线 risk-controlled routing。 |
| PyramidKV | 根据层间信息金字塔分配不同层 KV budget。 | 更接近 layer-wise budget allocation，但主要是静态/启发式 budget 分配；PCIC-CR 的重点是每个 block 在线比较候选策略并处理选择漂移。 |
| AdaKV | 动态按 attention head 分配 KV budget，减少不重要 head 的 cache。 | 与 budget allocation 相关，但 PCIC-CR 的创新点不是单次 budget 分配，而是 risk memory + sentinel 决策框架。 |
| RazorAttention | 利用 retrieval heads 现象，把 cache 压缩集中到检索头/非检索头差异。 | 是 head/token 结构先验；PCIC-CR 是候选策略层面的在线 counterfactual selector。 |
| QUEST | query-aware token sparsity / KV page selection，用当前 query 找相关 KV。 | 与 SparQ 类似，属于 query-aware retrieval；PCIC-CR 更像在线策略选择器，可以包在不同底层 sparsity policy 外面。 |

PCIC-CR 的定位应写成：

```text
Existing methods mostly design a single KV/token/head selection rule.
PCIC-CR instead treats KV compression as an online policy-selection problem:
each block estimates multiple candidate compression policies with Pairwise-CIC,
uses counterfactual-risk memory to suppress short-window selector drift,
and invokes a short sentinel prefix only when local evidence and memory disagree.
```

也就是说，PCIC-CR 不应该被包装成“又一个 sparse attention rule”。更稳的 paper 主张是：

```text
PCIC-CR is a meta-controller for KV compression policies.
It can sit on top of layer-budget, landmark, head-budget, or query-aware retrieval methods.
The novelty is online risk-controlled arbitration, not the specific landmark cache rule.
```

创新性强点：

1. **问题定义不同**：从“如何选择 KV token/head”转为“如何在长上下文中在线选择压缩策略并避免短窗漂移”。
2. **Pairwise-CIC 作为 policy estimator**：不是直接用 attention mass 或 query-key score，而是用短 calibration window 比较候选 policy 的 counterfactual loss。
3. **Counterfactual-risk memory**：跨 block 累积候选策略的 tail-risk 证据，解决 per-block min-loss 过拟合。
4. **Confidence-routed sentinel arbitration**：只在局部 min-loss 与 memory anchor 冲突，或第三候选高置信时，用 eval prefix 做在线仲裁。
5. **负结果也支持主张**：s4、lazy skip、calibration-only gate 都出现泛化失败，说明单靠短窗统计不足，必须保留 online counterfactual sentinel。

创新性风险：

1. 如果只强调 `landmark + layer combo`，容易被认为是 PyramidKV / SnapKV / SparQ 的小变体。
2. 如果只强调 `sentinel prefix`，容易被认为是 validation-set gate / online validation 的工程技巧。
3. 如果没有更多任务和模型，ICML 级别说服力不足；当前 War/Monte PPL 只能证明方向，不足以完整支撑广泛 claim。
4. 如果速度没有 kernel/batch-row budget 支撑，不能把 wall-clock speedup 作为主贡献。

推荐 paper contribution 写法：

```text
1. We formulate long-context KV compression as online blockwise policy selection,
   where short-window calibration can drift and requires risk-aware arbitration.
2. We propose Pairwise-CIC, a counterfactual calibration estimator for comparing candidate compression policies.
3. We introduce counterfactual-risk memory to stabilize policy selection across blocks.
4. We design confidence-routed sentinel arbitration to resolve local-vs-memory conflicts and recover third-candidate wins.
5. We empirically show that calibration-only or shorter-prefix gates fail, validating the need for online counterfactual arbitration.
```

建议 title / method name：

```text
PCIC-CR: Risk-Controlled Online Policy Selection for KV Cache Compression

或：

Confidence-Routed Counterfactual Policy Selection for Long-Context KV Compression
```

下一步为了达到更强 paper 说服力，需要补的实验：

```text
1. 数据集：LongBench / RULER / Needle / Topic stress，不只 War/Monte。
2. 模型：至少 Qwen3 + Llama/Mistral 类模型中的一个。
3. 底层 policy：不只 landmark layer-budget，尝试 head-budget 或 query-aware token policy。
4. 消融：min-loss、risk-memory、sentinel-all、confidence-routed、s4、lazy、static oracle。
5. 速度：eager algorithm validation 与 batch-row/kernel speed prototype 分开报告。
```

相关工作链接：

```text
SparQ Attention: https://arxiv.org/abs/2312.04985
H2O: https://arxiv.org/abs/2306.14048
StreamingLLM: https://arxiv.org/abs/2309.17453
SnapKV: https://arxiv.org/abs/2404.14469
PyramidKV: https://arxiv.org/abs/2406.02069
AdaKV: https://arxiv.org/abs/2407.11550
RazorAttention: https://arxiv.org/abs/2407.15891
QUEST: https://arxiv.org/abs/2406.10774
```

### Hard-topic 合成文本泛化验证

为了避免只在 War/Monte 两本文学文本上过拟合，又使用服务器本地已有的 hard-topic 合成文本做轻量验证：

```text
/home/fdong/ymluo/projects/influence_bounded_synthetic_kv/data/hard_topic_eval_v2.txt
```

这个实验不下载新数据，不占服务器网络。

新增脚本：

```text
scripts/run_pcic_hard_topic_pciccr_validation.sh
scripts/run_pcic_hard_topic_pciccr_key_main.sh
scripts/run_pcic_hard_topic_none_key_main.sh
```

候选 combo 使用 War/Monte 两组候选的并集：

```text
0,6;0,7;0,13;7,6;2,0;2,7;2,0,7,12;7,13
```

初始 b4 结果：

| run | blocks | avg_delta_ppl | method/base | combos |
| --- | ---: | ---: | ---: | --- |
| hardtopic none | 4 | 0.030228 | 1.052 | `0,7;7,13;2,0,7,12;2,0,7,12` |
| hardtopic conffast | 4 | 0.003316 | 1.167 | `2,0;2,0;2,0;2,0` |

key settings 结果：

| run | blocks | none_delta | conffast_delta | none_ratio | conffast_ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| hardtopic b4 | 4 | 0.030228 | 0.003316 | 1.052 | 1.167 |
| hardtopic b8 | 8 | 0.012777 | 0.012590 | 1.049 | 1.166 |
| hardtopic eval128 | 4 | 0.009629 | 0.038679 | 1.039 | 1.101 |

观察：

1. hard-topic b4 上，PCIC-CR 把 no-rescue min-loss 的正 delta `0.030228` 压到 `0.003316`，说明 risk memory + sentinel 确实能抑制短窗 combo 漂移。
2. hard-topic b8 上，PCIC-CR 与 no-rescue 基本持平但略好：`0.012777 -> 0.012590`。
3. hard-topic eval128 上，PCIC-CR 退化：`0.009629 -> 0.038679`，说明当 eval window 变长时，固定 memory anchor `2,0` 不一定保持最优。
4. 三组 hard-topic 结果都仍是正 delta，说明当前候选 combo 不是为 hard-topic 调过的；这不是主方法失败，而是候选 policy set 需要覆盖更多任务分布。

对 paper 主线的影响：

```text
Hard-topic 验证支持“online selector 能降低短窗漂移”的方向，
但也暴露当前候选 policy set 和 memory anchor 仍偏窄。
PCIC-CR 应被写成 meta-controller；
要在更多任务上稳定领先，需要为每个任务/模型提供更丰富的 candidate policy family。
```

下一步实验建议：

```text
1. hard-topic 上加入 head-budget / query-aware token policy 作为候选，不只 layer landmark combo。
2. 对 hard-topic 单独做 static oracle，判断是否存在比 2,0 更好的稳定候选。
3. 在 LongBench/RULER/Needle 上先跑 small eval，验证 PCIC-CR 是否能在任务式长上下文中稳定降低 drift。
4. 如果 eval128 继续退化，需要把 memory anchor 从“单一累计风险最小”改成 horizon-aware memory。
```

### Hard-topic static oracle 与 seed ablation

为了判断 hard-topic 退化是“候选集合没有好策略”，还是“selector 没选到好策略”，进一步跑了 static combo oracle。实现方式：每次只传入一个 combo，因此 `min_loss` 等价于固定使用该 combo。

新增脚本：

```text
scripts/run_pcic_hard_topic_static_oracle.sh
scripts/run_pcic_hard_topic_seed_ablation.sh
```

Static oracle 结果：

| setting | oracle_combo | oracle_delta |
| --- | --- | ---: |
| b4_eval64 | `0,6` | -0.012719 |
| b4_eval128 | `0,6` | -0.020744 |

完整排序：

| setting | combo | avg_delta_ppl |
| --- | --- | ---: |
| b4_eval64 | `0,6` | -0.012719 |
| b4_eval64 | `2,0` | 0.003316 |
| b4_eval64 | `0,13` | 0.042066 |
| b4_eval64 | `2,0,7,12` | 0.051028 |
| b4_eval64 | `2,7` | 0.066519 |
| b4_eval64 | `0,7` | 0.076412 |
| b4_eval64 | `7,6` | 0.112183 |
| b4_eval64 | `7,13` | 0.134968 |
| b4_eval128 | `0,6` | -0.020744 |
| b4_eval128 | `0,7` | 0.009918 |
| b4_eval128 | `7,6` | 0.021289 |
| b4_eval128 | `2,7` | 0.038444 |
| b4_eval128 | `2,0` | 0.038679 |
| b4_eval128 | `7,13` | 0.048826 |
| b4_eval128 | `0,13` | 0.059719 |
| b4_eval128 | `2,0,7,12` | 0.072841 |

Seed ablation：

| setting | seed | avg_delta_ppl | combos |
| --- | ---: | ---: | --- |
| b4_eval64 | 0 | 0.003316 | `2,0;2,0;2,0;2,0` |
| b4_eval64 | 16 | 0.082625 | `0,13;0,7;0,7;2,0` |
| b4_eval64 | 64 | 0.003316 | `2,0;2,0;2,0;2,0` |
| b4_eval128 | 0 | 0.038679 | `2,0;2,0;2,0;2,0` |
| b4_eval128 | 16 | 0.064252 | `0,13;0,13;2,0;2,0` |
| b4_eval128 | 64 | 0.038679 | `2,0;2,0;2,0;2,0` |

结论：

1. hard-topic 候选集合里确实存在好策略：`0,6` 在 eval64/eval128 都是 static oracle，且为负 delta。
2. 当前 PCIC-CR 选不到 `0,6`，不是因为 seed64 锚错；seed0 也会稳定选到 `2,0`。
3. seed16 反而更差，说明简单改变 seed 长度不是解决方案。
4. 根因更可能是当前 `risk_memory` 的 memory_score 偏向低 tail-risk / 低 positive-ratio 的保守策略，而不是 horizon 上 PPL 最优的策略。

这给出下一版方法方向：

```text
Horizon-aware Counterfactual Risk Memory

不要只用 avg_max_gap / worst_max_gap / positive_ratio 做 memory anchor；
需要把“短 horizon sentinel loss / eval-prefix realized loss”也写入 memory，
让 memory anchor 预测未来 eval horizon 的收益，而不是只预测 calibration tail risk。
```

可实现的下一步：

```text
1. 在每个 block 结束后，把 selected combo 的 eval delta_ppl / delta_loss 也写入 memory。
2. risk_memory selector 增加 horizon_score：
   memory_score = tail_risk + lambda * historical_eval_delta_loss
3. 对没有历史 eval 的 combo，使用当前 calibration delta_loss 做 fallback。
4. 在 hard-topic 上验证是否能从 `2,0` 转向 `0,6`，同时不破坏 War/Monte。
```
```

### Hard-topic horizon probe64：长 horizon sentinel 对照

服务器网络约束：本实验只上传 `2.5KB` 小脚本，模型、文本、输出均使用服务器本地文件；运行时设置 `HF_HUB_OFFLINE=1`，没有下载模型或数据。

新增脚本：

```text
scripts/run_pcic_hard_topic_horizon_probe64.sh
scripts/run_pcic_hard_topic_horizon_probe64_forced.sh
```

实验目的：

```text
验证 hard-topic 中 `0,6` static oracle 不能被 PCIC-CR 选到，究竟是因为：
1. sentinel horizon 太短，看不到未来收益；
2. 还是当前 memory-anchor / slack gate 太保守，把可见的好候选压回了 `2,0`。
```

设置：

```text
text: /home/fdong/ymluo/projects/influence_bounded_synthetic_kv/data/hard_topic_eval_v2.txt
model: /home/fdong/hrj/prove/Qwen3-0.6B
prefill_tokens: 2048
num_blocks: 4
calibration_tokens: 16
eval_tokens_per_block: 64
candidate combos: 0,6;0,7;0,13;7,6;2,0;2,7;2,0,7,12;7,13
sentinel_tokens: 64
attn_implementation: eager
```

#### 1. 保守版：horizonprobe64 + memory slack

命令核心参数：

```text
--combo_select_policy risk_memory_sentinel_all
--risk_memory_seed_tokens 64
--sentinel_tokens 64
--sentinel_loss_slack 0.03
--sentinel_all_min_margin 0.0
```

结果：

| block | selected combo | delta_ppl | route | sentinel best | best_margin |
| ---: | --- | ---: | --- | --- | ---: |
| 0 | `2,0` | 0.031639 | `all_memory_slack` | `0,7` | 0.006 |
| 1 | `2,0` | -0.019455 | `all_memory_slack` | `0,6` | 0.013 |
| 2 | `2,0` | 0.004777 | `all_memory_slack` | `0,13` | 0.000 |
| 3 | `2,0` | -0.003697 | `all_memory_slack` | `2,0,7,12` | 0.009 |
| avg | `2,0;2,0;2,0;2,0` | 0.003316 |  |  |  |

对比：

| method | avg_delta_ppl |
| --- | ---: |
| no-rescue min-loss | 0.030228 |
| PCIC-CR conffast_s8 | 0.003316 |
| horizonprobe64 + memory slack | 0.003316 |
| static oracle `0,6` | -0.012719 |

结论：把 sentinel horizon 从 `8` 加到 `64` 后，保守版仍然选择 `2,0`，没有超过 conffast_s8。这说明当前 `sentinel_loss_slack=0.03` 的门控过保守，只要 memory anchor 的 sentinel loss 落在 slack 内，就会压制 sentinel 最优候选。

#### 2. 强制版：horizonprobe64 + forced sentinel best

为了判断长 horizon sentinel 是否真的能发现更好的候选，在远端临时生成对照脚本，只把：

```text
--sentinel_loss_slack 0.03
```

改成：

```text
--sentinel_loss_slack 0.0
```

结果：

| block | selected combo | delta_ppl | route | sentinel best | best_margin |
| ---: | --- | ---: | --- | --- | ---: |
| 0 | `0,7` | -0.024449 | `all_best_candidate` | `0,7` | 0.006 |
| 1 | `0,6` | -0.198486 | `all_best_candidate` | `0,6` | 0.013 |
| 2 | `0,13` | -0.025633 | `all_best_candidate` | `0,13` | 0.000 |
| 3 | `2,0,7,12` | -0.059192 | `all_best_candidate` | `2,0,7,12` | 0.009 |
| avg | `0,7;0,6;0,13;2,0,7,12` | -0.076940 |  |  |  |

对比：

| method | avg_delta_ppl |
| --- | ---: |
| no-rescue min-loss | 0.030228 |
| PCIC-CR conffast_s8 | 0.003316 |
| static oracle fixed `0,6` | -0.012719 |
| horizonprobe64 forced best | -0.076940 |

关键观察：

1. 长 horizon sentinel 不是无效；当不被 slack gate 压回 memory anchor 时，它能选出逐 block 更优组合，平均 `avg_delta_ppl=-0.076940`，甚至明显优于 fixed `0,6` static oracle。
2. hard-topic 的主要问题不再是“候选集合没有好策略”，而是“当前 arbitration 规则过度信任 memory anchor”。
3. 这给出比原 PCIC-CR 更强的 paper 方法方向：不是只做 counterfactual-risk memory，而是做 **horizon-aware risk arbitration**。

#### 对方法创新性的影响

这个结果把方法从“已有稀疏注意力/SparQ 类似的候选策略”进一步推到一个更有论文价值的方向：

```text
Horizon-aware Counterfactual Policy Arbitration

核心不是提出某个固定 KV 压缩规则，而是在每个 block 上用长 horizon counterfactual probe
估计候选策略对未来 token 的真实风险，然后用在线 risk controller 在 memory prior
和 local horizon evidence 之间仲裁。
```

与已有方法的差异：

1. SparQ / H2O / SnapKV / PyramidKV / AdaKV 主要回答“保留哪些 token/head/layer”。
2. PCIC-CR / Horizon-PCIC 回答“在多个压缩策略都可能失效的情况下，如何在线判断当前 block 应该信任哪个策略”。
3. horizonprobe64 的结果说明，短窗口 calibration 会误导，memory prior 也会误导；需要把“未来 horizon 上的 counterfactual loss”作为仲裁信号。
4. 论文创新点应从“某个候选压缩模式”转向“面向 KV cache 压缩的在线反事实策略选择与风险控制框架”。

下一步实验建议：

```text
1. 在 War/Monte 上跑 horizonprobe64 forced-best，确认它不会破坏原本 conffast_s8 的负 delta。
2. 加一个可发表版本的门控：不是 forced best，而是使用 margin-normalized risk gate：
   select sentinel best iff
   (memory_loss - best_loss) / uncertainty > tau
3. 把 sentinel_tokens=64 的成本降下来：
   - 先 s8/s16 粗筛 top-k；
   - 再只对 top-k 做 s64；
   - 或用 batch-row budget map 一次前向评估多个候选。
4. 在 hard-topic b8/eval128 上复测，确认 horizon-aware arbitration 是否稳定优于 fixed oracle。
```

### Horizon-PCIC：从 forced-best 到可控 risk gate

本轮把上一节的 forced-best 上界改成显式 selector：

```text
--combo_select_policy risk_memory_horizon_gate
```

核心逻辑：

```text
1. calibration 阶段仍然得到 min-loss candidate 和 risk-memory candidate。
2. 每个 block 对 candidate set 做 horizon sentinel probe。
3. 计算：
   horizon_gain = memory_loss - best_horizon_loss
   horizon_gain_ratio = horizon_gain / max(best_margin, uncertainty_floor)
4. 只有当 horizon_gain 和 horizon_gain_ratio 达到门槛时，才推翻 memory anchor。
5. 当前实验为了验证上界，使用：
   horizon_gate_min_gain=0.0
   horizon_gate_min_ratio=0.0
```

新增代码参数：

```text
--horizon_gate_min_gain
--horizon_gate_min_ratio
--horizon_gate_uncertainty_floor
```

新增脚本：

```text
scripts/run_pcic_hard_topic_horizon_gate.sh
scripts/run_pcic_warmonte_horizon_gate_s32.sh
scripts/run_pcic_warmonte_horizon_gate_s64.sh
```

#### Hard-topic 结果

`sentinel_tokens=64`，candidate union：

```text
0,6;0,7;0,13;7,6;2,0;2,7;2,0,7,12;7,13
```

| setting | selected combos | avg_delta_ppl | method/base | 备注 |
| --- | --- | ---: | ---: | --- |
| eval64 horizon gate s64 | `0,7;0,6;0,13;2,0,7,12` | -0.076940 | 1.042 | 明显优于 conffast 和 static `0,6` |
| eval128 horizon gate s64 | `0,7;2,0,7,12;0,6;0,13` | 0.004371 | 1.043 | 优于 none/conffast，但仍输 fixed `0,6` oracle |

参考：

| setting | none | conffast_s8 | static `0,6` | horizon gate |
| --- | ---: | ---: | ---: | ---: |
| hard-topic eval64 | 0.030228 | 0.003316 | -0.012719 | -0.076940 |
| hard-topic eval128 | 0.009629 | 0.038679 | -0.020744 | 0.004371 |

解释：

1. eval64 里 `sentinel_tokens=64` 等于整个 eval block，因此它是“horizon oracle 上界”，证明候选集合和 online arbitration 有足够潜力。
2. eval128 里 `sentinel_tokens=64` 只看前半段，仍然能把 conffast 的 `0.038679` 降到 `0.004371`，说明 horizon evidence 可以缓解 memory anchor 错误。
3. eval128 仍不如 fixed `0,6`，说明前半段最优不一定等于后半段最优；下一步需要做 two-horizon consistency 或历史 horizon memory。

#### War/Monte sanity check

为了验证不是只对 hard-topic 有效，额外在服务器本地 War/Monte 文本上跑了两个 sanity check。数据仍使用服务器本地文件，没有下载：

```text
data/war_and_peace_pg2600.txt
data/count_monte_cristo_pg1184.txt
```

War/Monte 原始设置：

```text
prefill_tokens=2048
num_blocks=2
calibration_tokens=16
eval_tokens_per_block=64
```

s32 结果：

| dataset | selected combos | avg_delta_ppl | method/base | 结论 |
| --- | --- | ---: | ---: | --- |
| War s32 | `0,7;0,7` | -2.135311 | 1.039 | 很强 |
| Monte s32 | `2,7;7,13` | 0.431207 | 1.039 | 失败，短 horizon 被局部前缀误导 |

s64 结果：

| dataset | selected combos | avg_delta_ppl | method/base | 结论 |
| --- | --- | ---: | ---: | --- |
| War s64 | `0,7;0,7` | -2.135311 | 1.039 | 保持强 |
| Monte s64 | `2,7;2,0` | -0.219215 | 1.037 | 从 s32 失败恢复为负 delta |

参考旧结果：

| dataset | old no-rescue blockwise avg_delta_ppl |
| --- | ---: |
| War | -0.687288 |
| Monte | -0.108427 |

结论：

```text
Horizon gate 的质量潜力是明确的：
- War：从旧 no-rescue `-0.687288` 到 s64 horizon gate `-2.135311`。
- Monte：从旧 no-rescue `-0.108427` 到 s64 horizon gate `-0.219215`。
- Hard-topic eval64：从 conffast `0.003316` 到 horizon gate `-0.076940`。

但 s32 在 Monte 上失败，说明短 horizon sentinel 不是可靠充分条件。
论文主线应该升级为：

Pairwise-CIC + online blockwise selection + horizon-aware rescue gate

其中 rescue gate 不只是“看一个短 prefix 后选最优”，而是需要带风险控制：
1. two-horizon consistency：s16/s32 初筛，s64 复核；
2. memory prior：当短 horizon 和长期 memory 冲突且证据不足时不切换；
3. uncertainty-normalized gate：用 gain/margin/历史误差控制切换；
4. batched candidate evaluation：把全候选 sentinel 的 gate cost 从串行多次 forward 降下来。
```

#### 稳健门控 ablation：阈值 vs cascade

本轮继续验证“短 horizon 会误导”的问题，并把它转成可发表的风险控制设计。

代码修复：

```text
src/run_pcic_rescue_blockwise_local.py
```

修复内容：

```text
cascade sentinel 扩展时，之前只更新 combined loss/token_count/seconds，
没有同步更新 combined ppl。

当 remainder_tokens=0 时，pcic_eval 会直接复用 selected_sentinel["ppl"]，
导致 cascade full-horizon 的 PPL 显示异常。

已修复为：
combined["ppl"] = exp(total_loss)
```

新增正式脚本：

```text
scripts/run_pcic_warmonte_horizon_gate_robust_ablation.sh
```

该脚本比较两种稳健化方式：

```text
1. s32 + min_gain=0.02
   目的：低成本过滤弱 horizon gain，避免局部 prefix 误导。

2. cascade32to64 + margin=0.02
   目的：先用 s32 粗筛；如果 early margin 不够大，就扩展到 s64 复核。
```

War/Monte 结果：

| run | avg_delta_ppl | selected_method/base | gate_s | cascade_extended | cascade_early | combos |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| War s32 min_gain=0.02 | -2.135311 | 1.031 | 6.512 | 0 | 0 | `0,7;0,7` |
| Monte s32 min_gain=0.02 | -0.009005 | 1.038 | 6.590 | 0 | 0 | `2,0;2,0` |
| War cascade32to64 m=0.02 | -2.135311 | 1.040 | 9.916 | 1 | 1 | `0,7;0,7` |
| Monte cascade32to64 m=0.02 | -0.219215 | 1.036 | 13.137 | 2 | 0 | `2,7;2,0` |

对照：

| run | avg_delta_ppl | gate_s | 结论 |
| --- | ---: | ---: | --- |
| War s32 raw | -2.135311 | 6.586 | 已足够好 |
| Monte s32 raw | 0.431207 | 6.576 | 明显失败 |
| War s64 raw | -2.135311 | 13.120 | 质量强，gate 更贵 |
| Monte s64 raw | -0.219215 | 13.098 | 质量强，修复 s32 失败 |

结论：

```text
1. min_gain=0.02 是保守 gate：
   - 能避免 Monte s32 从 `7,13` 误切换导致的崩坏；
   - 但 Monte 收益只剩 `-0.009005`，弱于 old no-rescue `-0.108427`。

2. cascade32to64 是更合理的 paper 版本：
   - War：与 s32/s64 质量相同，gate 从 s64 的 `13.120s` 降到 `9.916s`；
   - Monte：自动扩展两个 block，恢复 s64 的 `-0.219215`。

3. 这支持一个清晰创新点：
   Horizon-PCIC 不是简单“看 prefix 选最优”，而是多阶段风险仲裁：
   short horizon 负责便宜探索；
   long horizon 负责在不确定时复核；
   memory anchor 负责在证据弱时保守回退。
```

下一步需要做的实验：

```text
1. 在 hard-topic 上跑 cascade32to64，确认是否与 s64 质量一致，以及是否有 early accept。
2. 做 accept_margin sweep：0.01 / 0.02 / 0.04，找质量和 gate cost 的 Pareto 点。
3. 把 cascade gate 的决策写成论文公式：
   if margin_s <= tau_extend: extend horizon
   elif gain_s <= tau_gain: keep memory
   else: select horizon best
4. 系统层继续推进 batch-row budget map，否则全候选 gate 仍是串行成本。
```

#### Hard-topic cascade sweep

继续在 hard-topic 上验证 cascade32to64，比较 `accept_margin=0.01/0.02/0.04`。

更新脚本：

```text
scripts/run_pcic_hard_topic_horizon_gate_cascade.sh
```

实验设置：

```text
candidate combos: 0,6;0,7;0,13;7,6;2,0;2,7;2,0,7,12;7,13
sentinel_tokens: 64
sentinel_cascade_initial_tokens: 32
horizon_gate_min_gain: 0.0
horizon_gate_min_ratio: 0.0
```

结果：

| run | avg_delta_ppl | selected_method/base | gate_s | cascade_extended | cascade_early | combos |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| eval64 raw_s64 | -0.076940 | 1.042 | 62.010 | 0 | 0 | `0,7;0,6;0,13;2,0,7,12` |
| eval64 cascade_m01 | -0.076940 | 1.042 | 53.695 | 3 | 1 | `0,7;0,6;0,13;2,0,7,12` |
| eval64 cascade_m02 | -0.076940 | 1.040 | 54.436 | 3 | 1 | `0,7;0,6;0,13;2,0,7,12` |
| eval64 cascade_m04 | -0.076940 | 1.038 | 61.055 | 4 | 0 | `0,7;0,6;0,13;2,0,7,12` |
| eval128 raw_s64 | 0.004371 | 1.043 | 62.204 | 0 | 0 | `0,7;2,0,7,12;0,6;0,13` |
| eval128 cascade_m01 | 0.004371 | 1.042 | 38.477 | 1 | 3 | `0,7;2,0,7,12;0,6;0,13` |
| eval128 cascade_m02 | 0.004371 | 1.039 | 53.338 | 3 | 1 | `0,7;2,0,7,12;0,6;0,13` |
| eval128 cascade_m04 | 0.004371 | 1.042 | 62.410 | 4 | 0 | `0,7;2,0,7,12;0,6;0,13` |

关键结论：

```text
1. hard-topic 上 cascade32to64 与 raw_s64 质量完全一致：
   eval64: -0.076940
   eval128: 0.004371

2. gate cost 明显下降：
   eval64: 62.010s -> 53.695s
   eval128: 62.204s -> 38.477s

3. accept_margin=0.01 是当前 hard-topic Pareto 最好点：
   质量不变，eval128 有 3/4 blocks early accept。

4. accept_margin=0.04 太保守：
   基本退化为 raw_s64，gate cost 接近或超过 raw_s64。
```

对 paper 主线的意义：

```text
Horizon-PCIC 的 rescue gate 可以写成一个自适应计算框架：

Pairwise-CIC 提供 candidate policies；
online blockwise selection 提供 memory prior；
cascade horizon gate 在低置信时延长 horizon；
early accept 在高置信时节省 gate compute。

这比“固定 sparse attention 规则”更有创新性，因为方法核心是在线反事实风险控制，
而不是另一个 token/layer/head 保留启发式。
```

#### Top-k cascade：减少扩展候选数

为了继续压低 gate cost，本轮把 cascade 扩展从“低置信时扩展所有候选”改为可选：

```text
--sentinel_cascade_extend_topk K
```

实现位置：

```text
src/run_pcic_rescue_blockwise_local.py
```

实现逻辑：

```text
1. 先对所有候选跑 short horizon，例如 s32。
2. 如果 early margin 足够大，直接 early accept。
3. 如果需要扩展到 s64，不再扩展所有候选；
   只扩展 short-horizon loss 最低的 top-k 候选。
4. 为了保留风险控制，额外强制加入 memory anchor 和 min-loss anchor。
```

新增脚本：

```text
scripts/run_pcic_hard_topic_horizon_gate_topk_cascade.sh
```

hard-topic top-k 结果：

| run | avg_delta_ppl | selected_method/base | gate_s | cascade_extended | cascade_early | avg_extended_candidates | combos |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| eval64 raw_s64 | -0.076940 | 1.042 | 62.010 | 0 | 0 | 0.00 | `0,7;0,6;0,13;2,0,7,12` |
| eval64 cascade_all_m01 | -0.076940 | 1.042 | 53.695 | 3 | 1 | all | `0,7;0,6;0,13;2,0,7,12` |
| eval64 top2_m01 | -0.076940 | 1.043 | 25.452 | 3 | 1 | 3.67 | `0,7;0,6;0,13;2,0,7,12` |
| eval64 top3_m01 | -0.076940 | 1.042 | 31.648 | 3 | 1 | 4.67 | `0,7;0,6;0,13;2,0,7,12` |
| eval128 raw_s64 | 0.004371 | 1.043 | 62.204 | 0 | 0 | 0.00 | `0,7;2,0,7,12;0,6;0,13` |
| eval128 cascade_all_m01 | 0.004371 | 1.042 | 38.477 | 1 | 3 | all | `0,7;2,0,7,12;0,6;0,13` |
| eval128 top2_m01 | 0.004371 | 1.041 | 27.192 | 1 | 3 | 3.00 | `0,7;2,0,7,12;0,6;0,13` |
| eval128 top3_m01 | 0.004371 | 1.044 | 30.141 | 1 | 3 | 4.00 | `0,7;2,0,7,12;0,6;0,13` |

关键结论：

```text
1. top2 cascade 在 hard-topic 上不损失质量：
   eval64 仍是 -0.076940；
   eval128 仍是 0.004371。

2. gate cost 大幅下降：
   eval64 raw_s64: 62.010s
   eval64 top2:    25.452s

   eval128 raw_s64: 62.204s
   eval128 top2:    27.192s

3. 相比 cascade_all_m01，top2 也继续降低成本：
   eval64: 53.695s -> 25.452s
   eval128: 38.477s -> 27.192s

4. top2 比 top3 更优：
   质量相同，gate 更低。
```

对 paper 的意义：

```text
Horizon-PCIC 可以形成一个完整的 adaptive compute story：

all-candidate short horizon:
    保证探索覆盖；

top-k long horizon:
    只对少数高潜力候选做昂贵复核；

memory/min-loss anchor:
    保证不会因为 top-k 剪枝丢掉保守回退路径。

这已经不只是 quality selector，而是一个“反事实候选评估预算分配”方法，
创新性比简单套 SparQ/H2O/SnapKV 风格规则更强。
```

#### War/Monte top-k cascade 泛化验证

为了确认 top-k cascade 不是 hard-topic 特例，本轮继续在 War/Monte 上跑相同设计。

新增脚本：

```text
scripts/run_pcic_warmonte_horizon_gate_topk_cascade.sh
```

设置：

```text
sentinel_tokens: 64
sentinel_cascade_initial_tokens: 32
sentinel_cascade_accept_margin: 0.01
sentinel_cascade_extend_topk: 2 / 3
```

结果：

| run | avg_delta_ppl | selected_method/base | gate_s | cascade_extended | cascade_early | avg_extended_candidates | combos |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| War raw_s64 | -2.135311 | 1.039 | 13.120 | 0 | 0 | 0.00 | `0,7;0,7` |
| War cascade_all_m02 | -2.135311 | 1.040 | 9.916 | 1 | 1 | all | `0,7;0,7` |
| War top2_m01 | -2.135311 | 1.040 | 6.531 | 0 | 2 | 0.00 | `0,7;0,7` |
| War top3_m01 | -2.135311 | 1.036 | 6.465 | 0 | 2 | 0.00 | `0,7;0,7` |
| Monte raw_s64 | -0.219215 | 1.037 | 13.098 | 0 | 0 | 0.00 | `2,7;2,0` |
| Monte cascade_all_m02 | -0.219215 | 1.036 | 13.137 | 2 | 0 | all | `2,7;2,0` |
| Monte top2_m01 | -0.219215 | 1.037 | 6.526 | 2 | 0 | 2.50 | `2,7;2,0` |
| Monte top3_m01 | -0.219215 | 1.037 | 8.721 | 2 | 0 | 3.00 | `2,7;2,0` |

结论：

```text
1. Top-k cascade 在 War/Monte 上同样不损失质量。

2. War 上 top2/top3 都 early accept 两个 block：
   raw_s64 gate 13.120s -> top2 gate 6.531s。

3. Monte 上 top2 自动扩展两个 block，但只扩展少数候选：
   raw_s64 gate 13.098s -> top2 gate 6.526s。

4. top2 仍是更合理默认值：
   top3 没有质量收益，gate 更高。
```

综合 hard-topic + War/Monte：

```text
Horizon-PCIC 当前最强默认版本：

combo_select_policy = risk_memory_horizon_gate
sentinel_tokens = 64
sentinel_cascade_initial_tokens = 32
sentinel_cascade_accept_margin = 0.01
sentinel_cascade_extend_topk = 2

这条配置在三个文本分布上都保持 raw_s64 质量，并明显降低 gate cost。
```

#### 自动生成的论文结果表

为了避免手工整理表格出错，新增只读汇总脚本：

```text
scripts/summarize_horizon_pcic_results.py
scripts/run_horizon_pcic_core_suite.sh
```

它从远端已有 `outputs/*/pcic_r_blockwise_results.csv` 自动生成：

```text
docs/horizon_pcic_key_results_2026_06_29.md
docs/horizon_pcic_key_results_2026_06_29.csv
docs/horizon_pcic_paper_blueprint_2026_06_29.md
```

汇总表包含：

```text
1. Hard-topic / War / Monte 的 raw_s64 与 top2 cascade 主结果；
2. avg_delta_ppl、selected/base、serial_total/base、gate_s；
3. gate cost reduction；
4. 质量 baseline 对照；
5. 当前可写进 paper 的结论。
```

关键自动汇总结论：

```text
top2 cascade 在 Hard-topic、War、Monte 上保持 raw_s64 质量；
gate 串行成本下降约 50%–59%；
当前 paper 默认方法可写为：
Pairwise-CIC + online blockwise policy selection + horizon-aware top-k cascade rescue gate。
```

独立 paper blueprint：

```text
docs/horizon_pcic_paper_blueprint_2026_06_29.md
```

#### Batched gate proxy：系统速度下界

为了评估后续 batch-row/fused candidate probe 是否值得做，本轮给 cascade 日志增加了分阶段计时：

```text
sentinel_cascade_initial_seconds
sentinel_cascade_extension_seconds
sentinel_cascade_extended_candidates
```

然后在自动汇总脚本里增加：

```text
batched_proxy/base
```

计算方式：

```text
serial gate:
    sum(all candidate forward time)

batched proxy:
    max(short-horizon candidate time)
  + max(extended-candidate long-horizon time)
  + selected remainder time
```

这个指标不是已经实现的真实 kernel speed，而是 batch-row/fused gate 的可达下界估计。

更新后的关键表：

| run | avg_delta_ppl | serial_total/base | batched_proxy/base |
| --- | ---: | ---: | ---: |
| Hard-topic eval64 top2 | -0.076940 | 4.028 | 1.054 |
| Hard-topic eval128 top2 | 0.004371 | 2.648 | 1.039 |
| War top2 | -2.135311 | 2.596 | 1.039 |
| Monte top2 | -0.219215 | 2.596 | 1.054 |

解释：

```text
1. 串行 gate 仍然太慢：serial_total/base 大约 2.6x 到 4.0x。
2. 如果 short-horizon all-candidate 与 top-k extension 能 batch/fuse，
   端到端 proxy 接近 1.04x 到 1.05x baseline。
3. 这说明系统层最关键的工程目标不是再调 selector，
   而是实现 batch-row budget map 或 fused candidate probe。
```

对 paper 的写法：

```text
Algorithm section:
    把 Horizon-PCIC 写成 adaptive counterfactual evaluation budget allocation。

System section:
    先报告 serial implementation；
    再报告 batched proxy；
    最终需要实现 batch-row/fused gate 后报告真实 wall-clock。
```

#### Batch-row layer budget prototype

为了把 `batched_proxy/base` 变成真实 wall-clock，本轮已经完成第一步底座：attention patch 支持同一个 forward batch 内不同 batch row 使用不同 layer-budget map。

代码位置：

```text
src/evaluate_qwen3_top2_head_limit3_ppl.py
src/run_pcic_rescue_blockwise_local.py
```

新增 JSON 格式：

```json
{
  "default": {"type": "full"},
  "layers": {},
  "batch_maps": [
    {
      "default": {"type": "full"},
      "layers": {"0": {"type": "landmark", "recent": 512, "stride": 64}},
      "compressed_layers": [0, 7]
    },
    {
      "default": {"type": "full"},
      "layers": {"2": {"type": "landmark", "recent": 512, "stride": 64}},
      "compressed_layers": [2, 0]
    }
  ]
}
```

新增 helper：

```text
write_batch_layer_budget_map(...)
```

验证：

```text
BATCH_MAP_OK outputs/tmp_batch_budget_smoke.json 911
```

当前实现状态：

```text
已完成：
1. layer-budget JSON loader 支持 batch_maps；
2. _layer_budget_attention_forward 支持 batch row 分派；
3. 旧单 map 行为保持兼容；
4. batch map writer 已通过无 GPU smoke test；
5. eval_segment_batched_candidates 已实现，可以一次 forward 评估多个候选。

还没完成：
1. 主 selector 仍然默认按候选串行调用；
2. batched candidate eval 还没有接入 horizon gate 主流程；
3. 还需要在真实 top2 cascade 实验中比较 batched gate wall-clock。
```

工程下一步：

```text
实现 eval_segment_batched_candidates：

inputs:
    candidate_names
    candidate_maps -> batch_map
    shared initial_past_key_values
    shared initial_prev_logits

process:
    repeat input token batch 到 candidate_count
    repeat past_key_values batch 维度
    repeat prev_logits batch 维度
    attention_mode(layerbudgetattn, batch_map)
    每个 step 得到 per-row loss

outputs:
    dict[candidate_name] -> loss/ppl/seconds/final_past/final_logits

这个函数完成后，Horizon-PCIC 的 gate wall-clock 才能接近 batched_proxy。
```

#### Batched candidate eval smoke

新增实现：

```text
eval_segment_batched_candidates(...)
repeat_batch_value(...)
slice_batch_value(...)
```

新增验证脚本：

```text
scripts/run_pcic_batched_candidate_smoke.sh
```

smoke 设置：

```text
model: Qwen3-0.6B
text: War and Peace
prefill_tokens: 256
sentinel_tokens: 2
candidates: 0,7 和 2,0
```

结果：

| combo | serial_loss | batched_loss | abs_diff | serial_s | batched_s |
| --- | ---: | ---: | ---: | ---: | ---: |
| `0,7` | 0.00986941 | 0.00986941 | 0 | 2.0253 | 3.7101 |
| `2,0` | 0.00986941 | 0.00986941 | 0 | 1.7594 | 3.7101 |

结论：

```text
1. batched candidate eval 与 serial eval loss 完全一致，证明 batch-row budget map 的语义正确。
2. 这个 tiny smoke 下 batched wall-clock 不占优，因为 batch=2、token=2，模型加载/调度开销占主导。
3. 下一步需要把 batched eval 接入真实 top2 cascade gate，
   在 s32/s64 sentinel 长度上测试 wall-clock 是否接近 batched_proxy。
```

#### Batched gate 接入 smoke

本轮进一步把 `eval_segment_batched_candidates` 接入 horizon gate 主流程，新增开关：

```text
--sentinel_batched_candidates true
```

接入范围：

```text
1. all-candidate initial sentinel stage；
2. cascade extension stage；
3. 默认关闭，不影响旧实验结果。
```

新增脚本：

```text
scripts/run_pcic_war_batched_gate_smoke.sh
```

War top2 smoke 结果：

| run | avg_delta_ppl | selected/base | serial_total/base | gate_s | combos | batched |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| serial current | -2.135311 | 1.040 | 2.603 | 6.536 | `0,7;0,7` | 0 |
| batched gate | -2.077407 | 1.274 | 3.531 | 9.340 | `0,7;0,7` | 1 |

诊断：

```text
1. batched gate 跑通，且两个 block 选择的 combo 与 serial 相同：`0,7;0,7`。
2. 质量略有差异：`-2.135311 -> -2.077407`。
   从 sentinel loss 看，batch-row 版本与 serial 每个候选有约 1e-4 到 4e-3 的 loss 差异。
   这更像 fp16 + batch shape 改变后的数值路径差异，而不是 selector 逻辑错误。
3. 当前 wall-clock 更慢：
   因为 batch-row attention wrapper 仍在 Python 中逐 row 调用 attention，
   只是把候选组织进 batch，并没有真正 fused。
```

因此当前系统状态应表述为：

```text
已完成 functional prototype：
    batch-row budget map 可以表达 batched candidate gate；
    horizon gate 可以调用 batched candidate eval；
    选择结果在 War smoke 上与 serial 一致。

还未完成 speed prototype：
    当前 batch-row wrapper 内部仍是 Python row loop；
    需要把 landmark/recent attention 的 row-wise 分派改成 vectorized/fused 实现，
    或者写 candidate-level fused probe kernel。
```

对 paper 的影响：

```text
Algorithmic novelty 已经比较完整；
系统速度还不能声称真实 batched gate 已加速。

目前可以写：
    serial implementation + batched proxy + functional batched prototype。

不能写：
    batched gate 已经真实 wall-clock 加速。
```

#### Batch-row grouped attention smoke

上一版 batch-row attention wrapper 内部仍是逐 row loop。本轮改成按 budget 配置分组：

```text
same budget rows -> one grouped attention call
different budget rows -> separate grouped calls
```

实现位置：

```text
src/evaluate_qwen3_top2_head_limit3_ppl.py
```

War top2 smoke 结果：

| run | avg_delta_ppl | selected/base | serial_total/base | gate_s | combos | batched |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| serial_top2 | -2.135311 | 1.038 | 2.596 | 6.544 | `0,7;0,7` | 0 |
| batched_top2 grouped | -2.133787 | 1.219 | 3.309 | 8.708 | `0,7;0,7` | 1 |

对比上一版 batched gate：

```text
old batched gate:
    avg_delta_ppl = -2.077407
    gate_s = 9.340

grouped batched gate:
    avg_delta_ppl = -2.133787
    gate_s = 8.708
```

结论：

```text
1. 分组后质量更接近 serial，说明 batch-row 路径更稳定。
2. gate 从 9.340s 降到 8.708s，但仍慢于 serial top2 的 6.544s。
3. 原因是当前实现仍会在每层做 Python 分组与 index_select/index_copy，
   并且 batch 化后 KV cache batch 维度变大；还不是 fused kernel。
4. 这一步可以作为 functional grouped prototype，
   但不能作为最终 speed claim。
```

下一步系统实现方向：

```text
1. 专门为 landmark/recent 两类 budget 写 vectorized branch：
   full rows 和 landmark rows 分别一次处理，减少通用 JSON/grouping overhead。
2. 更进一步做 candidate-probe fused kernel：
   对候选维度 C 做 fused score/prob/loss，不复制完整 KV cache。
3. paper 里短期可报告：
   serial implementation + batched proxy + functional grouped prototype。
```

#### 当前不足

当前 `method/base` 只统计被选中策略的 eval 时间，不包含所有候选 sentinel gate 的串行评估时间。真实端到端速度必须解决 gate cost：

```text
Hard-topic eval64:
selected method/base = 1.042
serial gate seconds = 62.010

Hard-topic eval128:
selected method/base = 1.043
serial gate seconds = 62.204
```

因此论文实验必须分两层写：

```text
1. Algorithmic quality：证明 Pairwise-CIC + horizon-aware gate 能显著降低 PPL drift。
2. Systems speed：实现 batched row-wise budget map / fused candidate probe，避免串行候选评估。
```
