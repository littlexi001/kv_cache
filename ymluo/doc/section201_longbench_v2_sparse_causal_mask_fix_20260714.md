# LongBench v2 暴露的稀疏 KV 因果掩码错误与修复（2026-07-14）

## 结论先行

LongBench v2 的 503 条外部留出实验否证了原始 v466：它虽然把 KV 压到 3.11%，但准确率只有 Full KV 的 19.46%。进一步的成对实验表明，主要问题不是 block 检索错误，也不是答案格式错误，而是 HuggingFace 在非连续 KV cache 上把“逻辑 RoPE 位置”和“物理 cache 下标”混为一谈，导致较长 question/choices suffix 的 causal mask 错误。

加入显式的 physical-cache causal mask 后，M6 上 sparse KV 与 Full KV 的六条 A-D 预测完全一致；同一修复的逐 token 参考实现也得到完全相同的结果。这是当前最重要的新发现，后续版本暂称 **Logical-Physical Causal Mask（LPCM）**。但新的六领域 M60 分层验证表明：LPCM 修复了执行语义错误，不等于解决了极低预算下的检索充分性，不能根据 M6 宣称整体质量已经恢复。

## 503 条原始外部验证

设置：Llama-3.1-8B-Instruct、官方 LongBench v2 0-shot prompt、32K context cap、503 条样本。

| Method | Accuracy | Full ratio | KV ratio | Valid format | Online speed | Total speed |
|---|---:|---:|---:|---:|---:|---:|
| Full KV | 29.62% | 100.00% | 100.00% | 91.45% | 1.00x | 1.00x |
| v466 | 5.77% | 19.46% | 3.11% | 23.66% | 2.35x | 1.25x |
| v466 direct-off | 5.77% | 19.46% | 3.11% | 23.66% | 2.29x | 1.25x |

v466 与 direct-off 的准确率完全相同，说明 direct operator 不是主因。v466 只有 119 条生成了官方格式，其中 29 条正确，条件准确率为 24.37%，接近四选一随机。放宽答案正则只能把正确数从 29 恢复到 36，仍远低于 Full 的 149，因此格式错误也不是主因。

## 同 block、不同表示的因果诊断

为了区分“找错 block”和“KV 表示错误”，固定 v466 选中的完全相同 blocks，比较三种执行方式：

1. 当前非连续 sparse KV gather；
2. 把相同 blocks 按原文顺序重建为紧凑 prompt；
3. Full KV。

M6 结果：

| Representation | Score | Token/KV ratio | 说明 |
|---|---:|---:|---|
| Full KV + constrained choice | 3/6 | 100% | 对照 |
| 原始 sparse KV + constrained choice | 0/6 | 2.19% | 格式已强制正确，仍失败 |
| 相同 blocks 重建 prompt | 2/6 | 2.13% | 不改变检索结果，仅改变表示 |

相同 blocks 在 prompt rebuild 中能恢复大部分 Full 质量，证明检索器不是 M6 的主要瓶颈。问题位于非连续 KV 的位置或 mask 语义。

## 根因

设原始前缀长度为 (L\approx 32000)，稀疏选择后只保留 (m\approx 900) 个 KV。选中 token 的 K 保留原始 RoPE 坐标 (p_1,\ldots,p_m)，query suffix 的逻辑位置仍从 (L) 开始。

标准 HuggingFace cache 接口默认物理 cache 下标与逻辑位置连续一致。稀疏 gather 后：

- `past_key_values.get_seq_length()` 返回物理长度 (m)；
- `cache_position` 使用逻辑位置 (L,L+1,\ldots)；
- 自动 causal mask 却在物理长度 (m+Q) 上比较逻辑位置 (L+t)。

因为 (L\gg m+Q)，suffix 内未来 token 也可能被视为可见。问题在短 query 上不一定明显，但 LongBench v2 的 question、四个 choices 和输出说明形成较长 suffix，错误会被显著放大。

## LPCM

LPCM 将两个坐标系显式分离：

- RoPE 继续使用原始逻辑位置，保证 query 与选中 block 的相对位置关系；
- causal visibility 使用压缩后的物理 cache 拓扑。

对 suffix 中第 (t) 个 token，物理 mask 定义为：

\[
M^{\mathrm{phys}}_{t,j}=
\begin{cases}
0, & j<m,\\
0, & m\le j\le m+t,\\
-\infty, & j>m+t.
\end{cases}
\]

也就是所有已选择 prefix KV 可见，suffix 只看见自身和更早的 suffix token。实现使用一次并行 4D additive mask，形状为 `[batch, 1, query_len, physical_kv_len + query_len]`。

## 机制消融

M6 constrained-choice 结果：

| Variant | Sparse score | Full score | Sparse/Full prediction agreement | Sparse online time |
|---|---:|---:|---:|---:|
| 原始自动 mask | 0/6 | 3/6 | 明显不一致 | 约 0.075s |
| 逐 token causal replay | 3/6 | 3/6 | 6/6 | 2.2–12.2s |
| LPCM 并行 mask | 3/6 | 3/6 | 6/6 | 约 0.07s |
| compact position（不旋转 K） | 2/6 | 3/6 | 3/6 | 约 0.07s |
| RoPE rebase + compact position | 3/6 | 3/6 | 5/6 | 约 0.075s |

逐 token 与 LPCM 的预测完全一致，构成 LPCM 正确性的参考验证。compact position 和 RoPE rebase 都不如保留原始 RoPE 坐标，当前主线应使用 `original RoPE + LPCM`。

## 当前实现

- `--sparse_query_physical_mask`：启用 LPCM；
- `--sparse_query_tokenwise`：慢速正确性参考实现；
- `--sparse_position_mode original`：当前主线；
- `--constrained_choice_decode`：对 Full 和 sparse 同时启用的公平多选解码协议。

核心代码位于 `src/run_controlled_public_kv_benchmark_v1.py`。

## M60 分层验证：执行问题修复后，检索仍是瓶颈

设置：六个 LongBench v2 一级领域各 10 条，constrained-choice 对 Full 和 sparse 公平启用，sparse 使用 original RoPE + LPCM。结果如下：

| Method | Accuracy | Full ratio | KV ratio | Prediction agreement | Online speed | Total speed |
|---|---:|---:|---:|---:|---:|---:|
| Full KV | 36.67% | 100.00% | 100.00% | 100.00% | 1.00x | 1.00x |
| v466 selector + LPCM | 28.33% | 77.27% | 2.68% | 56.67% | 2.16x | 1.01x |

按领域拆分：

| Domain | Full | Ours | Ours / Full | Prediction agreement |
|---|---:|---:|---:|---:|
| Code Repository Understanding | 30% | 30% | 100% | 60% |
| Long In-context Learning | 40% | 20% | 50% | 40% |
| Long Structured Data Understanding | 40% | 20% | 50% | 60% |
| Long-dialogue History Understanding | 50% | 50% | 100% | 50% |
| Multi-Document QA | 30% | 20% | 66.7% | 80% |
| Single-Document QA | 30% | 30% | 100% | 50% |

60 条中，双方都正确 14 条、双方都错误 35 条、只有 Full 正确 8 条、只有 sparse 正确 3 条。尤其值得注意的是，静态 operator budget 本身存在明显问题：

| Routed action | Samples | Budget | Full | Ours | Ours / Full |
|---|---:|---:|---:|---:|---:|
| retrieve | 51 | 896 | 37.25% | 31.37% | 84.21% |
| code | 8 | 256 | 37.50% | 12.50% | 33.33% |
| aggregate | 1 | 128 | 0% | 0% | - |

因此，`code=256 / aggregate=128 / retrieve=896` 这种任务名驱动的静态预算不能作为最终方法。检索 score gap、entropy、query-term coverage 在成功与失败组之间也没有形成清晰分界，不能继续把这些单一特征当作可靠安全证书。

## 自由生成小样本

同一 M6 使用官方自由生成，而不是 constrained-choice：

| Method | Score | KV ratio | Valid format | Online speed | Total speed |
|---|---:|---:|---:|---:|---:|
| Full KV | 2/6 | 100% | 5/6 | 1.00x | 1.00x |
| v466 selector + LPCM | 3/6 | 2.19% | 6/6 | 5.16x | 1.44x |

这个结果说明长 decode 下的 online 加速可以兑现，但样本只有 6 条，只作为运行可行性和速度机制验证，不能作为论文主结果。

## 下一步：Counterfactual-Stability Budget Planner

同一 M60 的 uniform-2048 结果已经完成。所有非 Full action 使用相同预算，以隔离“预算充分性”和“任务路由”两个变量：

| Method | Accuracy | Full ratio | KV ratio | Prediction agreement | Online speed |
|---|---:|---:|---:|---:|---:|
| Full KV | 36.67% | 100.00% | 100.00% | 100.00% | 1.00x |
| base dynamic actions | 28.33% | 77.27% | 2.68% | 56.67% | 2.16x |
| uniform 2048 + LPCM | 38.33% | 104.55% | 6.79% | 58.33% | 1.49x |

2K 在总分上超过 Full，但领域结果并不均匀：Long In-context Learning 只有 10%，而 Full 为 40%；Code、Dialogue、Multi-Doc QA 和 Single-Doc QA 则等于或超过 Full。因此当前结果证明 2K action 有效，但还不能证明其跨领域稳定。

base 与 2K 在 42/60（70%）样本上给出相同预测，仅 18/60 会改变。这个现象把 router 目标从“预测当前答案是否正确”改写为更容易蒸馏的“增加预算是否会改变模型决策”。如果稳定样本使用 base、可能改变的样本使用 2K，oracle 平均 KV 约为 3.9%，并保持固定 2K 的全部预测。

在 design-dev 上做 5 折 OOF 的 budget-change router 诊断，选定点结果为：

| Router protocol | Score | Full ratio | 2K ratio | Mean KV | 2K action rate | Choice-decode online speed |
|---|---:|---:|---:|---:|---:|---:|
| 5-fold OOF deployable features, threshold 0.65 | 36.67% | 100.00% | 95.65% | 3.78% | 23.33% | 1.88x |

该版本已删除部署时不可获得的后验 query-coverage 特征，并通过训练/运行时特征一致性单元测试。这个 OOF 结果仍不是最终测试结果。已经冻结数据划分：design-dev 60、router-train 222、router-calibration 89、paper-test 132。下一步在 train 生成 base/2K 变化标签，只在 calibration 选择阈值，paper-test 最后只运行一次。

3072 token 预算层级已经完成：31.67% score、10.15% KV，低于 2K 的 38.33% score、6.79% KV，被 2K 严格 Pareto 支配。在已检查的 40 条中，2K block 集合有 95% 是 3K block 集合的子集，平均 Jaccard 为 0.66；因此 3K 退化主要不是选择重排，而是额外 distractor evidence 改变模型决策。正式 action space 暂时保留 base/2K，不加入 3K。

2K 官方自由生成成对测速也已完成：

| Method | Score | Full ratio | KV ratio | Valid format | Online speed | Total speed |
|---|---:|---:|---:|---:|---:|---:|
| Full KV | 28.33% | 100.00% | 100.00% | 78.33% | 1.00x | 1.00x |
| uniform 2K + LPCM | 31.67% | 111.76% | 6.79% | 83.33% | 2.94x | 1.28x |

速度并非来自 sparse 更早结束：Full 共生成 4731 tokens，ours 共生成 4831 tokens。Full decode 为 96.86 ms/token，ours 为 31.79 ms/token，token-normalized decode speed 为 3.05x。`online_seconds` 包含 KV gather、query replay 和 decode；ours 的 2.94x 已计入 1.65 秒总 gather 开销。

需要强调：这是 design-dev 结果，且 Long In-context Learning 上 ours 为 10%、Full 为 30%。总体三目标已经达到，不等于跨领域问题已解决。

完整预算实验生成了：

1. 每条样本在 base、2K、3K、Full 下的成对预测；
2. 与 Full 一致的最小预算标签和实际正确的最小预算标签；
3. `base -> 2K -> 3K` 的预测稳定性是否能作为风险信号；
4. 自适应升级策略的分数、峰值 KV 比例和包含额外探测开销的 online speed。

最终方法不再把任务名直接映射为固定预算：训练阶段用反事实预算阶梯生成“决策是否改变”标签，部署阶段由蒸馏 router 一次预测 base 或 2K，无需同时执行两个 decode。该方向同时回应了三个问题：极低平均 KV、实际可执行 router、以及可解释的风险控制。

## Choice-contrast 检索假设：小样本否证

针对多选 query 把四个候选混成同一检索词袋的问题，实现了 candidate support + top-1/top-2 contrast block scorer。在相同 896-token 预算、相同 12 条样本上：

| Selector | Score | KV ratio |
|---|---:|---:|
| uniform 896 control | 5/12 | 2.95% |
| choice-contrast 896 | 4/12 | 2.95% |

只有两条预测发生变化，其中一条把原本正确答案改错，另一条仍然错误。因此这个版本的 choice-contrast 不并入主方法。代码保留为负结果和后续更强 candidate-conditioned retriever 的起点。

## 当前结论

1. 原始 v466 的主要执行失败已被定位并修复，LPCM 有独立机制价值；
2. 固定 2K 已在 design-dev 达到 104.55% Full，但跨领域方差很大，不能直接当作最终结论；
3. budget-change router 的 OOF 结果达到 3.84% KV 和 100% Full，下一步必须通过冻结 train/calibration/test 协议确认；
4. 在预算阶梯结果和严格留出完成前，不宣称方法已经达到 ICLR 投稿实验标准。
