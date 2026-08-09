# Value-mediated causal closure v2：8-seed BF16 复核

## 结论

**机制证据：局部 GO；方法与完整中介链：尚未 GO。**

在 Qwen3-8B、8K 上，对同一条自定义 attention 路径先运行严格的 `epsilon=0` no-op，再只提高一个 pre-softmax attention score `0.25`。由该 score 的精确一阶答案-margin 导数给出的预测，与真实干预后的 margin 变化在 8 个独立 seeds、32 个 target 坐标上保持强相关：

| 计划 | 坐标数 | Pearson | seed-cluster 95% CI | Spearman | seed-cluster 95% CI | 符号正确率 | 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|
| Target | 32 | **0.833** | **[0.732, 0.932]** | **0.795** | **[0.667, 0.888]** | **78.1%** | **[75.0%, 84.4%]** |
| 同 layer/head/class 随机位置 | 32 | 0.416 | [-0.003, 0.654] | 0.003 | [-0.163, 0.328] | 50.0% | [34.4%, 65.6%] |

置信区间用 seed 为 cluster 进行 20,000 次 bootstrap；32 个坐标不是 32 个独立实验单位。

这支持一个严格而有限的结论：

> 在固定的 BF16 instrumented computation graph 上，小幅改变单个 attention score 后，经 Value/残差通道引起的最终答案 margin 变化，可以由同一路径的一阶局部导数预测。

它验证了跨层机制链中的局部 `score -> Value write -> output margin` 环节。它**没有**证明一种新 RoPE、一个可部署 selector，或 RoPE 对全部输出变化的完整中介比例。

## 协议与审计

- 模型：Qwen3-8B，未量化 BF16，eager attention。
- 上下文：8,192 tokens；seed 0–7。
- 每个 seed：`gold_evidence`、`conflict_evidence`、`lexical_format_distractor`、`filler` 各选择一个 target singleton。
- 每个 target 配一个相同 layer/head/class 的随机位置。
- 每次干预只把一个 pre-softmax score 增加 `0.25`。
- target 排名使用 `|positive suppression × exact d margin / d score|`；这是 answer-gradient oracle，只能用于机制诊断。
- 每个 case 先运行完全相同代码路径的 `epsilon=0` no-op；所有实际 delta 都相对 no-op 计算。
- 8/8 case 均通过：no-op 计数、单坐标回放、target/random 不同位置、prefix cache 不变及完整性审计。
- no-op 与 instrumented baseline 的答案 margin 差异在 8/8 seeds 中精确为 `0`。

native 与 instrumented 路径仍存在 BF16 实现差异，逐 seed margin 差为：`-0.0154, -0.0838, +0.0669, -0.0743, -0.0574, -0.2838, +0.0182, +0.0093`。因此这里识别的是 **instrumented graph 内部**的局部因果效应，而不是把 intervention delta 直接加回 native graph。

## 分类别结果

| 类别 | n | 平均预测 Δmargin | 平均实际 Δmargin | Pearson | Spearman | 符号正确率 |
|---|---:|---:|---:|---:|---:|---:|
| Gold evidence | 8 | +0.0774 | +0.0750 | **0.987** | **1.000** | 87.5% |
| Conflict evidence | 8 | -0.0518 | -0.0410 | **0.968** | **0.952** | 100.0% |
| Lexical/format distractor | 8 | -0.0031 | -0.0017 | 0.486 | 0.500 | 87.5% |
| Filler | 8 | +0.0001 | -0.0178 | -0.438 | 0.000 | 37.5% |

最清楚的闭环集中在有语义作用的证据坐标：增强所选 gold score **平均**提高 gold-vs-conflict margin，增强所选 conflict score **平均**降低该 margin，但两类中都有个别 seed 方向相反。真正稳定的是“具体坐标的一阶有向敏感度能预测实际变化”，而不是 token 的 gold/conflict 标签本身。对接近零敏感度的 filler，固定 `0.25` 干预已超出“信号远大于数值噪声”的局部区间，排序和符号都不可靠。

## 哪些简单量不够

在 32 个 target 坐标上：

- `suppression gap` 单独预测 actual Δmargin：Pearson `0.091`，Spearman `0.143`；
- baseline attention probability 单独预测：Pearson `0.212`，Spearman `0.078`；
- `suppression × exact sensitivity`：Pearson `0.707`，Spearman `0.781`；
- 最终完整一阶预测：Pearson `0.833`，Spearman `0.795`。

因此，“某 token 被 RoPE 压低很多”或“它原本 attention 很高”都不足以判断修复它是否改善答案；还必须知道该具体 score 通过 Value、残差流与后续层对答案 margin 的有向敏感度。

## 论文中可以与不可以声称的内容

可以声称：

1. 在严格 no-op-matched 的 BF16 局部干预中，answer-margin 一阶导数能预测选定证据 score 的真实下游效应。
2. gold 与 conflict 的分类平均作用方向分离，但存在个别反例；具体 score coordinate 的有向敏感度比类别标签更有预测力。
3. suppression 或 attention mass 本身不是可靠的正确性证书。

不可以声称：

1. 已闭合 `RoPE phase -> QK -> softmax mass -> Value -> later Query -> output` 的全部中介比例。
2. answer-gradient target ranking 是推理时可用的方法。
3. 提高所有 gold-span token 的 attention 都会有益。
4. 8K 单模型结果已泛化到 32K/64K、其他模型或自然任务。
5. 这是一个新的、可部署的位置编码方案。

## 产物

- 汇总：`merged/singleton_prediction_summary.csv`
- 逐干预：`merged/case_rows.csv`
- 逐 Value 样本：`merged/value_samples.csv`
- 每个 seed 的完整审计：`shard_gpu6/raw/` 与 `shard_gpu7/raw/`
- 可复现 seed-cluster 统计：`independent_seed_cluster_audit.json` 与 `independent_seed_cluster_audit.md`
- 运行 provenance：`corrected_provenance.json`；修复后的 shard-derived 配置为 `merged/merge_config.json`
- 旧 merge 默认参数误写版本保留为 `merged/merge_config_legacy_incorrect.json`，不得作为运行配置引用。
