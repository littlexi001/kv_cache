# 跨 token 候选复用：最终实验结论

## 1. 研究问题

当前 CountCap 在每个 decode token、每一层都执行一次：

1. 低维 query 投影；
2. 256 点 sampled-quantile 阈值估计；
3. PCA48/INT4 全索引扫描；
4. 候选 token 上的 QK+V attention。

原假设是相邻 token 的 query 和候选集合足够稳定，因此可以保存上一步的候选、边界分数和 query。若能证明本步不会发生候选越界，就跳过完整索引扫描。

## 2. 严格数值证书

对相邻 query 的变化量 Δq，单个索引 key 的分数变化满足：

`|Δq · k_i| <= ||Δq||_2 ||k_i||_2`

若候选内最低分与候选外最高分的真实离散间隔，大于两侧最大分数变化，就可以证明候选集合不变。

早期实现错误地把 sampled-quantile 的连续阈值当成“候选外最高离散分数”，因此 margin 几乎总为 0。本轮已经改为显式计算完整代理分数向量，并分别取：

- 当前候选内最低代理分数；
- 当前候选外最高代理分数；
- 两者之差作为真实离散 boundary margin。

修正后，离散 margin 为正的比例接近 100%，但其数量级仍远小于 Cauchy 全局上界：

| 长度 | 候选 Jaccard | 加入新 token 后召回 | margin 中位数 | 分数变化上界中位数 | 严格证书率 |
|---:|---:|---:|---:|---:|---:|
| 8K | 49.87% | 66.30% | 0.01573 | 402.29 | 0% |
| 16K | 49.10% | 65.43% | 0.00736 | 406.22 | 0% |
| 32K | 51.96% | 68.10% | 0.00417 | 398.51 | 0% |

结论：原实现确实有边界定义问题，但修正后严格证书仍不可用。根因不是实现 bug，而是 dense top-k 边界间隔极小，而全局 key norm 上界过松。

## 3. 宽保护集合证书

为了放宽“候选集合必须完全不变”的要求，本轮又测试了上一 token 代理分数最高的 8%–25% 作为保护集合。新 token 无条件加入；若当前 6% 候选全部落在保护集合内，则本步理论上可以只扫描保护集合。

最宽的 25% 保护集合结果如下：

| 长度 | 当前候选平均召回 | 单 head 完整覆盖率 | 四-head GQA 组完整覆盖率 | 严格安全率 |
|---:|---:|---:|---:|---:|
| 8K | 95.83% | 34.86% | 12.82% | 0% |
| 16K | 95.37% | 25.11% | 7.35% | 0% |
| 32K | 96.75% | 33.63% | 13.30% | 0% |

平均召回率看起来较高，但实际内核必须按共享 KV 的四个 query head 成组执行。即使扫描上一 token 的 25% 历史，能够完整覆盖整组候选的步骤也只有 7%–13%，远低于预期的 50% 跳扫目标。

## 4. Expected-crossing 弱证书

本轮还使用 query 变化的谱加权方差估计候选越界数量，并联合 sampled attention mass 评估弱证书。

在“接受步骤中的坏输出率不超过 1%”约束下，最佳单-head 接受率为：

| 长度 | 最佳接受率 |
|---:|---:|
| 8K | 0.34% |
| 16K | 1.12% |
| 32K | 1.30% |

没有任何配置能让一整层的所有物理 GQA 组同时通过。放宽 crossing 阈值虽然可提高接受率，但接受步骤中的坏输出率会上升到约 26%–32%。

结论：当前 expected-crossing 统计量无法同时提供高跳扫率和低误差。

## 5. 当前最快路径上的强制周期复用

为避免旧执行路径低估速度上限，本轮新增了：

`qprojscan + qkvsplitauto + cacheauto + reuse{2,4,8}`

它分别约跳过 50%、75%、87.5% 的 sampled-quantile 扫描。每个长度使用同一 GovReport 样本、生成 64 token，并做两次 GPU 轮换。

| 长度 | 方法 | 单样本分数保持 | 与基线逐字一致 | Online speed vs 当前 CountCap |
|---:|---|---:|---:|---:|
| 8K | reuse2 | 105.62% | 否 | 1.008x |
| 8K | reuse4 | 115.79% | 否 | 0.984x |
| 8K | reuse8 | 100.34% | 否 | 0.947x |
| 16K | reuse2 | 100.00% | 是 | 0.992x |
| 16K | reuse4 | 105.98% | 否 | 0.994x |
| 16K | reuse8 | 99.32% | 否 | 0.955x |
| 32K | reuse2 | 88.17% | 否 | 1.001x |
| 32K | reuse4 | 91.36% | 否 | 0.976x |
| 32K | reuse8 | 82.92% | 否 | 0.949x |

单样本分数高于 100% 不表示方法更可靠，只表示生成文本改变后该样本的 ROUGE 偶然升高。32K 上 reuse2 已下降到基线的 88.17%，证明相邻候选漂移会造成实质质量风险。

速度方面，即使强制跳过一半以上扫描也没有稳定收益。当前复用实现需要为每层重新拼接、扩容和写入 ragged 候选列表，其数据搬运与 kernel launch 开销抵消了扫描节省；reuse4/8 反而更慢。

## 6. 最终判断

跨 token 候选复用不应作为当前论文和系统优化的主线，原因是：

1. 严格证书的可认证复用率为 0%；
2. expected-crossing 在低误差下的接受率不超过约 1.3%；
3. 扫描 25% 保护集合也只能让 7%–13% 的 GQA 组完整覆盖；
4. 强制周期复用会改变输出，并在 32K 明显损伤质量；
5. 当前实现跳过 50% 扫描也没有兑现稳定在线加速。

当前部署主方法保持不变：

`countcap_fullprompt_keypca_direct_qkvfused_qprojscan_qkvsplitauto_cacheauto_prefillindex`

## 7. 下一步：融合扫描与候选消费

下一阶段应直接优化单 token 内的数据流，不跨 token 猜测候选：

1. 保留 256 点 sampled-quantile 阈值估计；
2. 一个 CUDA kernel 按 tile 解码 PCA48/INT4 索引并计算代理分数；
3. token 超过阈值时，立即读取对应 FP16 K/V；
4. 立即计算精确 QK，并使用 online softmax 累积 V；
5. 不再把候选 indices、proxy scores 和 ragged counts 写回全局内存；
6. overflow 只记录计数并走现有安全重跑路径。

该设计不改变 CountCap 的数值决策，因此质量应与当前方法一致。它直接减少 sampled scan 与 candidate attention 之间的全局内存往返、临时张量分配和 kernel launch，最有希望解决 8K/16K 的交叉点问题。

第二优先级是在 dense prefill 生成 K 时增量构建 PCA/INT4 索引，或使用独立 CUDA stream 与后续 prefill chunk 重叠，以降低一次性 index-build 和 total latency。该优化主要影响首 token 前时间，不替代 decode fused kernel。

## 8. 复现文件

- 离散证书与保护集合实现：`src/run_head_top2_targeted_ppl_20260714.py`
- 证书分析：`src/analyze_discrete_certificate_trace_20260724.py`
- 保护集合实验：`scripts/launch_guard_certificate_trace_8k32k_3gpu_20260724.sh`
- 周期复用实验：`scripts/launch_qprojscan_periodic_oracle_8k32k_4gpu_20260724.sh`
- 周期复用汇总：`src/analyze_periodic_reuse_oracle_20260724.py`
- 保护集合结果：`results/20260724_guard_certificate_trace_8k32k_3gpu`
- 周期复用结果：`results/20260724_qprojscan_periodic_oracle_8k32k_4gpu`

回归测试：

`python -m pytest tests/test_countcap_fullprompt.py tests/test_head_top2_targeted_ppl.py tests/test_expected_crossing_bandef.py -q`

结果：`59 passed`。
