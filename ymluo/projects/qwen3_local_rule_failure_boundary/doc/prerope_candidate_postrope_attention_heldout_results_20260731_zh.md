# Pre-RoPE 候选检索与 Post-RoPE 精确注意力

## 核心结论

在独立的长程两跳检索分布上，只把 **pre-RoPE QK 用作远程候选
proposal**，再对选中 token 使用原模型的 **精确 post-RoPE QK 分数**
和 V，可以稳定超过完整注意力：

| 长度 | Full PPL | 本方法 PPL | 相对 Full 质量 | Full 正确率 | 本方法正确率 |
|---:|---:|---:|---:|---:|---:|
| 8K | 3.120 | 2.541 | **122.8%** | 62.5% | 75.0% |
| 16K | 12.885 | 3.080 | **418.4%** | 29.2% | 62.5% |
| 32K | 39.079 | 10.840 | **360.5%** | 12.5% | 25.0% |
| 64K | 8.823 | 6.573 | **134.2%** | 29.2% | 29.2% |

质量比定义为：

```text
quality_ratio = exp(NLL_full - NLL_method)
```

四个长度的几何平均 PPL 从 10.851 降到 4.859，整体质量比为
**2.233x**；平均 next-token 正确率从 33.3% 提升到 47.9%。

这是 24 个冻结后的独立 seed（32–55）结果。发现阶段只使用 seed
0–7，方法选择与最终测试严格分离。

该结果已超过“Full 质量的 110%”，但范围必须准确表述：它证明的是
**长程两跳检索且含大量干扰文本时的质量提升**，不是普通文本 PPL、
LongBench 或所有生成任务上的 223.3%。

## 问题

RoPE 后的远程 QK 分数为：

```text
s_post(t, i) = q_t^T R_(i-t) k_i
```

其中相对位置旋转 `R_(i-t)` 同时编码位置与内容。当距离很长时，同一份
语义证据可能因相位旋转得到很低的分数。完整注意力仍然会把大量弱相关
token 放进 softmax 分母，形成两类问题：

1. 真正证据在候选排序中被 RoPE 相位压低；
2. 大量干扰 token 的累计 softmax 质量稀释证据。

pre-RoPE 分数为：

```text
s_pre(t, i) = q_t^T k_i
```

它丢失相对位置信息，但更接近位置无关的语义相似度。直接用它替换
模型 attention 分数会改变模型；更保守的办法是只用它找候选。

## 方法

对每层、每个 attention head：

1. 保留 16 个 sink token；
2. 保留最近 128 个 local token；
3. 在其余远程历史中按 `s_pre` 选择 top-k；
4. 总预算为当前序列长度的 2%；
5. 对全部选中 token 重新读取原始 K/V；
6. 使用原模型的 `s_post` 做精确 softmax 和 value 聚合。

因此本方法改变的是 **候选集合**，不改变候选内的原模型 QK 打分函数：

```text
C = sinks ∪ local ∪ TopK_remote(s_pre)
output = Softmax(s_post restricted to C) V_C
```

实验名为 `local_global_postscore`。

它与普通 `rope_top2` 的区别只有远程 proposal：

- `rope_top2`：用 post-RoPE 分数找候选，也用 post-RoPE 分数消费；
- 本方法：用 pre-RoPE 分数找远程候选，用 post-RoPE 分数消费。

这使实验能够单独归因“语义 proposal 是否修复 RoPE 远程漏检”。

## 独立测试

设置：

- 模型：Qwen3-8B，NF4 权重，BF16 计算；
- 长度：8K、16K、32K、64K；
- 任务：单 token 两跳规则链检索，证据嵌入长干扰文本；
- 发现集：seed 0–7；
- 独立测试：seed 32–55，共 24 个 seed；
- 每个长度与方法严格使用相同 seed 配对；
- bootstrap 单位为 seed，不把 head 或 layer 当成独立样本。

相对 Full 的配对结果：

| 长度 | 质量比 | NLL 差 95% CI | 改善 seed 比例 |
|---:|---:|---:|---:|
| 8K | **1.228x** | [-0.348, -0.060] | 75.0% |
| 16K | **4.184x** | [-1.912, -0.957] | 87.5% |
| 32K | **3.605x** | [-1.625, -0.954] | 100.0% |
| 64K | **1.342x** | [-0.615, 0.004] | 58.3% |

8K、16K、32K 显著优于 Full；64K 点估计提高 34.2%，但置信区间
刚跨过零，不能写成统计显著。

相对普通 post-RoPE top-2%：

| 长度 | 质量比 | NLL 差 95% CI |
|---:|---:|---:|
| 8K | 1.147x | [-0.355, 0.076] |
| 16K | **1.902x** | [-0.973, -0.342] |
| 32K | **1.996x** | [-1.035, -0.308] |
| 64K | 1.227x | [-0.427, 0.005] |

中长序列的收益不是简单来自稀疏化：在 16K/32K 上，它也显著超过
相同 2% 预算的 post-RoPE top-k。

## 为什么能超过 Full

Full attention 不是任务质量的理论上界。模型在超长上下文中会把非零
attention 分配给大量干扰 token。设证据集合为 `E`，干扰集合为 `D`：

```text
mass_full(E) =
sum_(i in E) exp(s_i) /
[sum_(i in E) exp(s_i) + sum_(j in D) exp(s_j)]
```

若候选 proposal 删除了大部分 `D`，同时保留足够的 `E`，则重新归一化
后证据质量可以增加。pre-RoPE proposal 还会找回被相位旋转压低、因而
不在 post-RoPE top-k 内的语义证据。两种作用共同解释了为何稀疏结果
可能优于 Full。

这不是“创造了 Full 中不存在的信息”，而是改变了有限模型对已有信息
的归一化和干扰抑制。

## 数值修正规则

另测了两种候选内分数修正：

- `pre_monotone25`：只提升 `calibrated_pre > post` 的远程分数；
- `pre_masspreserve25`：做相同提升后再平移，使远程 token 的
  `sum(exp(score))` 与原 post-RoPE 分区一致，只改变远程证据内部排序。

独立测试中它们的四长度整体质量比为 2.294x 和 2.393x，均高于纯
postscore 的 2.233x；但它们改变了原模型分数，方法更复杂，也更需要
跨任务验证。当前建议：

- **主方法**：`local_global_postscore`，归因干净、风险最低；
- **增强消融**：`pre_masspreserve25`，验证数值重排的额外上限；
- 不在缺少通用 benchmark 前把增强规则设为默认。

## 与 QKSieve 的组合

当前实验为了机制验证，显式计算了完整 pre-RoPE 和 post-RoPE QK，
所以没有系统加速，甚至会比 Full 更慢。可部署版本应当这样实现：

1. prefill 时直接在 RoPE 之前取得 K，而不是 decode 时反旋转全部 K；
2. 用 QK-balanced 变换构建 pre-RoPE Key 索引；
3. 使用当前 128-bit 整数 `[4,2]` 或后续 minifloat 索引扫描远程历史；
4. 合并 sink、local 和语义候选；
5. 只 gather 候选的原始 post-RoPE K/V；
6. 对候选执行精确 post-RoPE attention。

这个组合把两条研究线统一为：

```text
低比特 pre-RoPE 语义索引负责 recall
精确 post-RoPE K/V 负责模型兼容性
稀疏 softmax 负责抑制长上下文干扰
```

它也比同时维护 pre/post 两套完整 proxy index 更省：主候选规则只需要
一套 pre-RoPE 低比特索引，post-RoPE K/V 本来就是模型 decode 所需缓存。

## 尚未解决的问题

1. 当前只在合成两跳检索上达到超过 Full 的质量，必须补 RULER、
   LongBench 和普通主题 PPL。
2. 当前 2% 是固定比例。应测试 QKSieve 已使用的长度规则：
   `min(1280, max(256, ceil(0.06N)))`，确认收益不是特定比例造成。
3. 需要用真实 128-bit QKSieve pre-RoPE 索引替代完整 pre-QK 扫描，
   测量 proposal recall、质量与实际速度。
4. 64K 相对 Full 的置信区间仍跨零，应扩大 seed 或提升远程 proposal
   recall，而不是继续在发现集调 blend 系数。
5. 超过 Full 来源于去干扰，可能在总结、开放生成等需要广覆盖的任务
   上反向伤害；不能仅凭检索任务宣称通用提升。

## 复现入口

- 主实验：
  `src/run_local_global_rope_probe_8b.py`
- 发现集：
  `scripts/run_sage_numerical_rescue_discovery_8gpu_20260731.sh`
- 独立测试：
  `scripts/run_sage_prerope_heldout24_8gpu_20260731.sh`
- 汇总：
  `src/summarize_sage_numerical_rescue_20260731.py`
- 独立结果：
  `artifacts/20260731_sage_prerope_heldout24_8gpu/analysis/summary.json`

