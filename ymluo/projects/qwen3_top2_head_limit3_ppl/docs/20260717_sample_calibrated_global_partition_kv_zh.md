# 采样校准的全局 Partition KV 检索

更新时间：2026-07-17

## 1. 问题

已有实验表明，长上下文 decode 时，每个 query head 最终只使用少量历史 token，仍能接近甚至局部超过 Full Attention。真正的问题不是“能否只算 top-k”，而是：

1. 不计算完整 QK 时，如何找到重要 token；
2. 不同 layer、head 和生成位置需要的预算差异很大，如何在运行时自动选择最小预算；
3. 预算决策和 exact rerank 的额外开销，不能抵消稀疏 attention 的收益。

固定 top-2% 解决不了第二个问题。七阶段 progressive partition-UCB 能动态分配预算，但需要重复进行 exact-QK、top-k 和风险计算，32K 实测 online 时间较高。

## 2. 关键发现

### 2.1 压缩分数适合定位，但不能直接代表真实 partition

使用 PCA64 将 K 从 128 维投影到 64 维，再量化为 INT4，可以低成本扫描全部历史 token。它对高分候选有较好的定位能力，但 softmax partition 对分数误差很敏感，直接使用 proxy mass 会造成预算偏差。

### 2.2 只需极少 exact 样本，就能校准整个 partition

对每个 query head 均匀抽取 0.25% 历史 token，计算真实 QK。样本中真实指数权重与 proxy 指数权重的平均差，可以作为全体历史 token 的 control-variate 修正。

### 2.3 无需为每个预算重复校准

早期版本针对每个候选前缀重新排除落入前缀的样本，形成七次 sample-membership 与统计计算。实验发现，直接对每个 head 计算一次全局修正，再同时评估七个预算档位，质量几乎不变，但控制器可以完全向量化。

### 2.4 一次预算决策优于逐级 exact 探测

先用校准后的 proxy partition 决定预算，再只执行一次可变长度 exact rerank。真实 QKV trace 上，一次 verify-expand 只触发 0.7% 到 8.9% 的 head，质量收益很小，因此当前主方法不加入第二次扩展。

## 3. 方法

当前冻结配置：

- 全局索引：PCA64 + INT4，仅保存压缩 K；
- exact 样本：历史 token 的 0.25%；
- 最终预算档位：0.5%、1%、2%、3%、4%、6%、8%；
- 默认目标 mass：0.75；
- 置信系数：z = 0，即使用样本校准均值；
- exact 候选池：最终预算的 2 倍，最大不超过 8%；
- 每个 query head 独立决策；
- 当前 token 始终保留。

### 3.1 全局 proxy 扫描

对 query head h 和历史 token i：

```text
q_tilde_h = q_h U
k_tilde_i = INT4(k_i U)
a_h,i = q_tilde_h dot k_tilde_i
```

其中 U 是 128 x 64 的 PCA 投影矩阵，a 是近似 QK 分数。

### 3.2 样本校准

为避免指数溢出，先减去每个 head 的公共中心 m：

```text
proxy_weight_i = exp(a_i - m)
exact_weight_j = exp(s_j - m),  j 属于均匀样本 S

delta_bar = mean_j(exact_weight_j - proxy_weight_j)
```

delta_bar 是每个 query head 的单个在线校准量，不需要训练 router，也不使用任务标签。

### 3.3 一次性选择预算

按 proxy 分数排序。对预算 c，计算：

```text
selected_proxy(c) = sum(proxy top-c weights) + self_weight

tail_hat(c) = proxy_total - proxy_top_c
              + (N - c) * delta_bar

mass_hat(c) = selected_proxy(c)
              / (selected_proxy(c) + max(0, tail_hat(c)))
```

从预算梯级中选择第一个满足以下条件的 c：

```text
mass_hat(c) >= 0.75
```

若所有档位都不满足，则使用最大 8% 档位。七个档位一次向量化计算完成。

### 3.4 一次 exact rerank

若最终预算为 c，则只读取 proxy 排名前 min(2c, 8%N) 的原始 K，计算真实 QK，并从中选择真实 top-c。最后仅对这些 token 和当前 token 进行 Value 聚合。

该流程没有任务规则、训练 router、oracle 标签或逐阶段 fallback。

## 4. 当前效果

### 4.1 32K、10-case PPL

10 个 case 包括 sports/medicine 各三个窗口，以及 computer、space、politics、religion 各一个窗口。

| 方法 | 几何平均 PPL | 相对强 partition reference | 相对固定 uncertainty top-2% |
|---|---:|---:|---:|
| 固定 uncertainty top-2% | 10.6329 | +2.08% | 0% |
| 强 partition reference | 10.4159 | 0% | -2.04% |
| prefix-conditioned one-shot | 10.4408 | +0.24% | -1.81% |
| **global one-shot，target=0.75** | **10.4623** | **+0.45%** | **-1.60%** |

说明：强 partition reference 使用当前每个 case 上已完成的最强 partition 结果，其中 sports/medicine window 0 是真实 progressive-2 路径，其余八个 case 是固定最大 8% exact 候选的 partition-UCB 路径。它用于衡量质量上限，不代表统一的最快实现。

global one-shot 在固定 uncertainty top-2% 对照上赢 8/10 个 case；相对强 partition reference 的最坏单 case PPL 回退为 1.38%。

### 4.2 平均资源使用

| 指标 | global one-shot，target=0.75 |
|---|---:|
| 最终 attention links | 2.11% |
| exact-QK 原始 K 读取 | 2.90% |
| 目标位置 estimated retained mass | 82.99% |
| 32K online 时间 | 56.82 s / 512 次 token forward |

这里的 2.11% 是最终 attention 使用的历史连接；2.90% 是定位阶段 exact rerank 读取的原始 K。两者不能与 PCA64 INT4 索引存储混为一谈。

### 4.3 两主题受控速度对比

| 方法 | sports PPL | medicine PPL | 平均 online 时间 |
|---|---:|---:|---:|
| seven-stage progressive-2 | 8.9933 | 10.0865 | 124.24 s |
| prefix-conditioned one-shot，target=0.70 | 9.0536 | 10.0413 | 86.13 s |
| **global one-shot，target=0.75** | **9.0662** | **10.1205** | **54.48 s** |

global one-shot 相对 seven-stage 在该受控对比中快约 2.28 倍；两主题几何 PPL 高 0.57%。这些 online 数字是稀疏 PPL harness 内部对比，不是相对 Full Attention 的正式端到端 speedup。

### 4.4 真实 QKV trace

在 sports 和 medicine 的 5 个代表层、8 个 sample offset，共 2560 个 head-sample 上：

| 指标 | 结果 |
|---|---:|
| 真实 retained mass 均值 | 90.00% |
| 真实 retained mass 中位数 | 93.79% |
| 最终预算均值 | 2.26% |
| exact-QK 比例均值 | 2.92% |
| 稀疏输出相对 L2 误差均值 | 0.2147 |
| exact top-2% 输出相对 L2 误差 | 0.2016 |
| 真实 mass 低于 70% | 6.02% |
| 预测 mass 高估超过 5 个百分点 | 5.94% |

低 mass 主要集中在第 0 层的 diffuse attention。扩大一个预算档位只带来有限改进，而 PPL 没有显示必须对第 0 层做 full fallback，因此当前不加入层规则。

当前 global estimator 总体偏保守，但它不是严格证书：预测 mass 高于真实 mass 的比例为 22.23%。论文中应使用 sample-calibrated 或 risk-estimated，而不能使用 guaranteed、certified 等表述。

### 4.5 128K 子系统速度

| 子系统 | 原路径 | 当前路径 | 实测加速 |
|---|---:|---:|---:|
| 预算控制器 | 3.044 ms | 0.744 ms | 4.09x |
| exact-QK 候选读取 | 8.00%，0.152 ms | 2.81%，0.078 ms | 1.96x |

exact-QK 行使用 32 个 head、平均读取 2.81% 的代表性动态预算分布；它是独立 kernel 基准，不是对 128K 任务预算分布的测量。对应流量理论加速为 2.84x，实测只有 1.96x，差异来自 kernel launch、索引读取和分支等固定开销。

## 5. 索引开销

若原始 KV 使用 FP16，head dim 为 128，则每个历史 token、每个 KV head 的原始存储为：

```text
K + V = 2 * 128 * 16 bits = 4096 bits
```

PCA64 INT4 只保存压缩 K 主体：

```text
64 * 4 bits = 256 bits
256 / 4096 = 6.25%
```

加入量化 scale 和布局元数据后，当前实测索引约占原始 FP16 KV 的 6.64%。这不表示 GPU 上所有运行时内存只有 6.64%；当前 PPL harness 仍保留完整 KV，用于验证算法质量。

## 6. 创新边界

不能单独声称以下内容是创新：

- top-k sparse attention；
- 依据 attention mass 动态分配预算；
- PCA 或 INT4 压缩 K；
- exact candidate rerank。

当前更有区分度的贡献是：

1. 将压缩 QK 索引同时作为候选排序器和 softmax partition 的 control variate；
2. 使用极小的在线 exact 样本，对每个 query/head 的 partition 偏差进行无训练校准；
3. 在读取原始 K 之前一次性决定可变预算，然后只做一次 exact rerank；
4. 将预算控制器向量化，使统计自适应不再成为推理瓶颈。

它与 RAG 的边界清晰：输入仍是模型自身 KV cache，操作对象是每层、每个 head 的 attention links，不检索外部文档，也不重建 prompt。

## 7. 当前结论与限制

当前默认主线建议使用 global one-shot，target=0.75。它比旧 one-shot 快约 1.59 倍，只损失约 0.21% 的 10-case 几何 PPL；相对更昂贵的强 partition reference，质量差 0.45%。

仍未完成的关键证据：

1. LongBench、RULER 和长生成任务上的统一冻结配置；
2. 64K/128K 真实端到端 decode，而非仅子系统 replay；
3. 第二个模型与不同 GQA 结构；
4. 与 ProxyAttn、SampleAttention、Double-P、Twilight 等近邻工作的正式对齐；
5. 对样本校准误差的有限样本理论分析。

因此，这个结果已经形成了比“多级工程 router”更简单且更有论文辨识度的方法，但还不能仅凭当前实验宣称达到 ICLR 完整投稿证据。

## 8. 结果与代码位置

- 主实现：`src/run_head_top2_targeted_ppl_20260714.py`
- 增量 exact-QK CUDA：`src/qabs_cuda_kernels.py`
- 10-case global 结果：`results/20260717_partition_global_ucb_independent_32k`
- sports/medicine window 0：`results/20260717_partition_global_ucb_sweep_32k`
- 真实 QKV trace 分析：`results/20260717_one_shot_partition_trace/global_tau075_z0.json`
- 128K 控制器基准：`results/20260717_partition_controller_benchmark_128k/summary.json`
- 128K exact-QK 基准：`results/20260717_incremental_candidate_scores/summary.json`
