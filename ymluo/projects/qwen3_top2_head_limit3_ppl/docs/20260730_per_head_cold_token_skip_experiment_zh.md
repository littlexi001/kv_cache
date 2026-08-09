# Per-head 冷 token 跳过实验

## 1. 结论

这个想法可以工作，但当前证据不支持把它合入 QKSieve 主版本。

- 只保留每个 KV head 历史上经常被检索的 token 不安全。当前 Query 需要的证据，可能恰好是历史低频 token。
- 为了在跨模型 trace 上保留至少约 99.5% 的原 QKSieve attention mass，仍需扫描约 62.6% 的索引；若要求 5% 分位也不低于 98%，需扫描约 70.1%。
- 62.6% 档在 120K 上让 attention 子系统相对当前 QKSieve 再快 1.199x，但折算整模型 steady decode 预计只再快约 1.025x。32K 基本没有收益。
- 32K 六主题 PPL 没有观察到平均质量下降，但样本仅有 384 个预测 token，且跨模型 trace 仍有明显尾部风险，不能据此声称质量提高。

因此，冷 token 跳过适合作为已验证的探索结果或 120K 以上可选优化，不应替代当前“扫描全部低比特索引”的 QKSieve 主路径。

## 2. 方法

实验中的“冷”只表示某个 token 在同一 KV head 的过去校准 Query 中很少进入候选，不表示它对所有未来 Query 都不重要。

每层、每个 KV head 独立执行：

1. 使用 prompt 尾部 8 个 Query 做因果校准。
2. 每个校准 Query 用 QKSieve 近似分数选 top 2% token。
3. 统计 token 被选中的次数，并用最大近似分数打破同频次并列。
4. 将频率排名前 50% 或 60% 的 token 定义为常驻热集合。
5. 每个 decode step 实际扫描池为：

```text
候选池 =
    常驻热集合
  + 最近 256 token
  + 四个冷 block 分片中轮转的一个分片
  + 上一步已选候选
```

6. 只在候选池内扫描 QKSieve 混合位宽索引，再从原始 GPU 常驻 FP16 K/V 中取最多 1,280 个 token 做精确 QK、softmax 和 AV。

所有集合只依赖 prompt Query 和过去 decode step，没有使用当前答案、未来 Query 或 oracle 标签。热集合按 KV head 保存，GQA 中属于该 KV head 的 Query heads 共享热集合，但保留各自的上一步候选。

## 3. 跨模型 attention 质量

设置：

- 模型：Qwen3-4B、Qwen2.5-7B、Llama-3.1-8B
- 文本：sports、medicine
- 长度：32K
- 每个模型/主题抽取 5 层
- 每层前 8 步校准，后 56 步独立评估
- 每个配置共 1,680 个 layer/head/step 条件
- 质量相对量：跳过后的 attention mass / 原 QKSieve attention mass

| 配置 | 实际扫描池 | 平均 mass 保持 | 5% 分位 | 最差值 | 仅扫描量上界 |
|---|---:|---:|---:|---:|---:|
| 热 50% + recent256 + 1/4 冷分片 + carry | 62.60% | 99.534% | 97.861% | 86.504% | 1.597x |
| 热 60% + recent256 + 1/4 冷分片 + carry | 70.06% | 99.697% | 98.601% | 87.602% | 1.427x |

50% 热集合配置的分来源平均 mass 保持率：

| 来源 | 平均保持率 |
|---|---:|
| Llama-3.1-8B medicine | 99.795% |
| Llama-3.1-8B sports | 99.880% |
| Qwen2.5-7B medicine | 98.950% |
| Qwen2.5-7B sports | 99.760% |
| Qwen3-4B medicine | 99.126% |
| Qwen3-4B sports | 99.695% |

主要问题不是均值，而是尾部。50% 档在 Qwen2.5 medicine 上出现 86.50% 的单条件最差 mass 保持；把热集合增至 60% 也只能把全局最差值提高到 87.60%。这说明历史低频无法为当前 Query 提供可靠的不重要性证明。

## 4. 32K 六主题 PPL

设置：

- 模型：Qwen3-4B-Instruct
- 主题：computer、medicine、politics、religion、space、sports
- 每个主题 64 个预测 token，共 384 个
- Full、原 QKSieve full-topk、cold-skip 均使用相同窗口和冻结模板
- 两个 cold-skip 配置的 36 层审计均生效，平均扫描池分别为 62.66% 和 70.11%

| 方法 | 几何 PPL | 相对 Full 质量 | 相对原 QKSieve | Top-1 一致率 | KL(Full || 方法) |
|---|---:|---:|---:|---:|---:|
| Full Attention | 16.2909 | 100% | - | 100% | 0 |
| 原 QKSieve | 16.3036 | 99.922% | 100% | 96.094% | 0.00807 |
| cold-skip 50% | 16.2622 | 100.176% | 100.254% | 96.094% | 0.01265 |
| cold-skip 60% | 16.2886 | 100.014% | 100.092% | 96.094% | 0.01032 |

PPL 均值没有下降，但不能把小幅变好解释为方法提高了质量：

- 只有 384 个预测 token，差异小于正常数值扰动和稀疏正则化波动的量级。
- cold-skip 的 KL 反而高于原 QKSieve，说明输出分布偏移更大。
- 跨模型 attention trace 已显示 PPL 均值掩盖不了的最差条件。

## 5. 真实 CUDA attention 速度

测速包含：

- WMMA Query 投影和 INT8 Query 量化
- sampled-quantile 阈值估计
- 混合位宽索引扫描与候选写出
- 局部 token ID 到原历史位置的映射
- 原始 FP16 K/V 上的精确 QK、softmax、AV

不包含模型 MLP、投影层、LayerNorm、HF 控制流和索引首次构建。硬件为 RTX 3090，报告 36 层 attention 合计。

### 热 50%，实际扫描约 62.6%

| 历史长度 | 原 QKSieve attention | cold-skip attention | 相对原 QKSieve | 相对 Full SDPA |
|---|---:|---:|---:|---:|
| 32K | 5.788 ms | 5.745 ms | 1.007x | 3.982x |
| 64K | 6.944 ms | 6.574 ms | 1.056x | 6.776x |
| 120K | 8.929 ms | 7.445 ms | 1.199x | 10.863x |

120K 的纯检索阶段从 4.923 ms 降至 3.513 ms，为 1.401x；但 token ID 映射本身占 0.357 ms，所以 attention 总收益只有 1.199x。

### 热 60%，实际扫描约 70.1%

| 历史长度 | 相对原 QKSieve attention | 相对 Full SDPA |
|---|---:|---:|
| 32K | 0.991x | 3.917x |
| 64K | 1.031x | 6.611x |
| 120K | 1.044x | 9.456x |

更稳健的 60% 档在 120K 也只带来 4.4% attention 收益。

## 6. 整模型 decode 影响

当前 compact kernel 尚未接入 HF 全模型路径，因此下面是把独立 CUDA attention 差值代入同 runner 已测 steady decode 的估计，不是新的端到端实测。

| 长度 | 当前 QKSieve | 50% cold-skip 估计 | 相对当前 | 相对 Full 估计 |
|---|---:|---:|---:|---:|
| 32K | 46.539 ms/token | 46.497 ms/token | 1.001x | 1.899x |
| 64K | 47.330 ms/token | 46.960 ms/token | 1.008x | 3.390x |
| 120K | 61.130 ms/token | 59.646 ms/token | 1.025x | 4.684x |

120K 只多兑现约 2.5%，因为当前 QKSieve 的索引检索只占整模型 decode 的一小部分。MLP、QKV/O projection、LayerNorm、cache 管理和 kernel launch 不会因跳过冷 token 而减少。

## 7. 显存

- 当前 QKSieve 混合位宽索引约为 Full FP16 K+V 的 5.771%。
- 50% cold-skip 的目标 `uint32` 紧凑索引加 token map 约为 4.102%。
- 60% cold-skip 为 4.591%。
- 原始 FP16 K/V 仍完整常驻 GPU，因此这里减少的是辅助检索索引，不是主 KV cache。
- 当前实验 kernel 使用 `int64` token map，运行时比例为 7.526% 或 8.422%；正式实现必须改为 `uint32` 才有显存收益。

## 8. 决策

不合入当前主版本，原因如下：

1. 质量安全所需扫描比例太高，63% 到 70% 已限制速度上限。
2. 32K 无净收益，64K 整模型预计不到 1%。
3. 120K 整模型预计只有约 2.5%，不足以抵消新索引布局和维护逻辑的复杂度。
4. 历史频率不是当前 Query 相关性的安全证书，跨模型最差条件仍明显失真。

后续若继续研究，方向应改为“当前 Query 条件下的廉价排除证书”，而不是继续调热集合比例。证书必须比扫描现有混合位宽索引更便宜，同时提供可验证的 score 上界；否则只是把一次低成本全扫描换成另一种高开销筛选。

## 9. 复现入口

代码：

```text
src/analyze_qksieve_per_head_cold_skip_20260730.py
src/summarize_qksieve_per_head_cold_skip_20260730.py
src/benchmark_qksieve_per_head_cold_skip_20260730.py
src/summarize_qksieve_per_head_cold_skip_ppl_20260730.py
src/run_head_top2_targeted_ppl_20260714.py
src/run_direct_countcap_denseprompt_ppl_20260725.py
scripts/run_qksieve_per_head_cold_skip_multimodel_20260730.sh
scripts/run_qksieve_per_head_cold_skip_ppl_20260730.sh
```

远端结果：

```text
results/20260730_qksieve_per_head_coldskip_multimodel_32k
results/20260730_qksieve_per_head_coldskip_cuda_stable_gpu6.json
results/20260730_qksieve_per_head_coldskip_six_topic_ppl_32k_v2
```
