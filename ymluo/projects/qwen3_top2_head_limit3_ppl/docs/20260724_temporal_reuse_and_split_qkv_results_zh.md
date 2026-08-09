# CountCap 时间复用与 Split-QKV 消费内核

更新时间：2026-07-24

## 1. 研究问题

当前 CountCap 已将每个 decode step 的候选定位压缩为 PCA48-INT4
sampled-quantile 扫描，并让约 6% 的候选直接参加精确 attention。它在长上下文中
速度较好，但 8K 附近仍慢于 Full SDPA。

本轮依次检验两个假设：

1. 相邻 decode step 复用候选集合，跳过 PCA48-INT4 全历史扫描。
2. 若扫描不是主瓶颈，则提高候选精确 QK、softmax 和 AV 阶段的 GPU 并行度。

所有实验使用 Llama-3.1-8B-Instruct、RTX 3090，且仅使用 GPU 0--3。

## 2. 时间复用的负结果

### 2.1 相邻候选并不稳定

在同一个 GovReport 样本上生成 64 token，逐层逐 head 统计相邻步候选集合：

| Prompt | 相邻候选 Jaccard | 前一步候选在当前步的召回 |
|---:|---:|---:|
| 8K | 49.08% | 65.05% |
| 16K | 48.32% | 64.29% |
| 32K | 51.14% | 66.97% |

候选集合只有约一半重合。严格 Cauchy crossing certificate 的安全率为 0。
这不是阈值偶然太紧，而是当前 sampled-quantile 选择规则使用
`score >= boundary`，至少有一个被选 token 的分数恰好等于 boundary，因此严格
boundary margin 恒为 0。只要 query 发生非零变化，要求“整个候选集合完全不变”的
证书就不可能通过。

### 2.2 周期复用没有兑现速度

实现了每 2、4、8 步刷新一次候选，其余步骤复用旧候选并立即补入所有新 token。
`reuse4` 的实测扫描跳过率为 73.33%。

| Prompt | 原 CountCap online | reuse2 | reuse4 | reuse8 |
|---:|---:|---:|---:|---:|
| 8K | 2.892 s | 2.923 s | 2.949 s | 3.061 s |
| 16K | 2.970 s | 3.013 s | 3.040 s | 3.157 s |
| 32K | 3.874 s | 3.812 s | 3.763 s | 3.774 s |

8K/16K 没有加速；32K 只有约 1%--3% 的收益。与此同时，32K 单样本分数从
0.2185 降为 reuse2 的 0.1927、reuse4 的 0.1993 和 reuse8 的 0.1661。

结论：sampled-quantile 扫描已经不是主要瓶颈。继续优化复用证书，即使成功，也不足以
解决短上下文速度问题。

## 3. Split-QKV 消费内核

### 3.1 方法

原 QKV-fused kernel 为每个 query head 启动一个 CUDA block。Llama-3.1-8B
每层只有 32 个 query heads，而 RTX 3090 有 82 个 SM，因此单层并行度不足。

新内核将同一个 head 的候选划分为 4 段：

```text
每个 split block:
  计算本段候选的精确 QK
  计算局部 softmax max 与 denominator
  计算未归一化的局部 weighted-V

归并 block:
  用各段 local max 做数值稳定的全局 softmax 归并
  合并 4 个局部 weighted-V
```

该方法不改变 PCA 索引、候选集合、候选预算或精确 K/V，只改变精确 attention 的
并行执行方式。

### 3.2 微内核

固定 6% 有效候选、12% ragged buffer capacity：

| 历史长度 | 单 block QKV | split4 | 消费内核加速 | 最大绝对误差 |
|---:|---:|---:|---:|---:|
| 8K | 0.2012 ms | 0.0375 ms | 5.37x | <= 6.1e-5 |
| 16K | 0.2599 ms | 0.0926 ms | 2.81x | <= 6.1e-5 |
| 32K | 0.4795 ms | 0.1719 ms | 2.79x | <= 6.1e-5 |

### 3.3 整模型配对

同一个 GovReport 样本、64-token decode、三次轮换执行顺序：

| Prompt | 原 QKV-fused online 中位数 | split4 online 中位数 | 相对原 CountCap |
|---:|---:|---:|---:|
| 8K | 2.955 s | 2.959 s | 0.999x |
| 16K | 2.981 s | 3.023 s | 0.986x |
| 32K | 3.882 s | 3.112 s | 1.247x |

8K/16K 的实际工作集较小，额外 block 与归并开销抵消了微内核收益。32K 候选
工作集变大后，并行度收益能够稳定兑现。

因此冻结长度感知执行规则：

```text
history < 24,576:
  使用原单-block QKV-fused kernel

history >= 24,576:
  使用 split4 QKV-fused kernel
```

这是硬件与工作集相关的解析规则，不训练 router，也不读取任务标签。

## 4. 当前最好速度结果

32K GovReport、64-token decode 的当前同运行 Full 对照：

| 方法 | Score | Online | Online speed | Total | Total speed |
|---|---:|---:|---:|---:|---:|
| Full KV | 0.21854 | 6.413 s | 1.000x | 23.657 s | 1.000x |
| 长度感知 split CountCap | 0.21854 | 3.099 s | **2.069x** | 21.167 s | **1.118x** |

Online 包含问题后缀、PCA48-INT4 检索、候选压紧、精确 QK、softmax、AV 和完整
模型 decode。Total 还包含完整 prefill 与索引构建。

## 5. 质量验证

16 个 LongBench 英文任务、每任务 2 个样本得到 32 对结果。只有 2 个样本实际超过
24K 并触发 split；这两个 31K NarrativeQA 样本的 prediction 和分数均完全一致，
配对 online 加速为 1.142x 和 1.269x。

另在 GovReport offset 115 起运行 10 对样本，其中两个样本超过 24K：

| Prompt | Prediction/score | Online speed |
|---:|---|---:|
| 25,846 | 完全一致 | 1.126x |
| 32,000 | 完全一致 | 1.260x |

因此目前真正触发 split 的真实长样本为 4/4 输出一致，中位加速约 1.20x。该结果可
证明执行内核没有明显质量回归，但样本量仍不足以替代完整 benchmark。

少数未触发 split 的短样本在两次独立稀疏运行间出现预测差异。两种 score mode 在
这些长度上实际调用同一个旧 QKV kernel，差异来自 sampled-threshold atomic
compaction 的运行间非确定顺序，不是 split 数值归并造成的。

## 6. 被否决的 Proxy-AV

还测试了直接把 PCA48-INT4 proxy score 当作 softmax logit，从而不再读取候选 K。
raw proxy 立即导致输出退化。随后实现了单 CUDA kernel：每个 head 抽取 64 个候选，
计算 exact QK，并用闭式回归拟合 `exact_score = a * proxy_score + b`。

在线仿射校准仍未恢复质量：8K GovReport 32-token 分数从 0.0769 降为 0.0297，
且 online 从 1.478 s 增至 1.695 s。说明 PCA proxy 能保持候选排序和 attention
mass，但不能安全替代最终 softmax logit。该路径不进入主方法。

## 7. 当前冻结结论

当前推荐执行结构为：

```text
Dense prompt/question encoding
-> prefill 期间增量建立 PCA48-INT4 索引
-> 256 点 sampled-quantile 选出约 6% 候选
-> 24K 以下使用单-block exact QK/softmax/AV
-> 24K 及以上使用 split4 exact QK/softmax/AV
-> 不复用过期候选
-> 不使用 proxy score 替代 exact logit
-> 不回退 Full
```

## 8. 下一步

1. 在 24K/32K/64K/128K 上联合搜索解析式 split count，而不是固定 split4。
2. 补 64K/128K attention 子系统和整模型 decode 配对，确认长文本收益是否继续扩大。
3. 对超过 24K 的真实长样本补充更大规模质量验证。
4. 8K 仍无法超过 Full SDPA。下一步若坚持纯稀疏路径，应降低候选消费的真实 K/V
   读取量；继续优化 PCA 扫描或时间复用的价值已经很低。

## 9. 结果位置

```text
results/20260724_certified_temporal_trace_8k32k_3gpu
results/20260724_periodic_reuse_ceiling_8k32k_3gpu
results/20260724_qkvsplit4_e2e_8k32k_3gpu
results/20260724_qkvsplit4_multitask_m2_4gpu
results/20260724_qkvsplit4_govreport_m10_2gpu
results/20260724_qkvsplit4_full32k_g64_v1
```
