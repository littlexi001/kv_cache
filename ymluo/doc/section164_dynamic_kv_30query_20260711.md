# 30 条真实多跳问题动态 KV 检索实验

## 1. 目的

验证在相同的每层 K3 预算下，使用生成过程中不断变化的 Q 重新检索，是否优于只根据问题阶段选择一次 block 的静态检索。

数据包括：

- 2WikiMQA：10 条
- HotpotQA：10 条
- MuSiQue：10 条
- 总计：30 条真实多跳问题

## 2. 正确实验协议

第一轮实验使用 raw context + raw prompt，不能正确发挥 Qwen3 指令生成能力，结果作废。最终 v2 使用 Qwen3 chat template：

1. source context 被放入 user message 的 `Memory:` 区域。
2. chat header 和问题 token 始终可见。
3. 只有 source context 对应的远程 KV 会被 block mask。
4. source block 仍保持原始 256-token 边界。
5. prompt 要求先写至少两条短推理事实，再写 `Final answer:`。
6. 最多自由生成 128 tokens。
7. 不限制累计读取量。

比较五种方法：

| 方法 | 定义 |
|---|---|
| Question only | 不使用 source KV |
| Full source | 所有 source KV 始终可见 |
| Static K3 | 问题阶段每层选择 K3，生成阶段固定不变 |
| Dynamic C1K3 | 每生成 1 token，每层重新选择 K3 |
| Dynamic C3K3 | 每生成 3 tokens，每层重新选择 K3 |

K3 表示同一层所有 query head 共享三个 block，不同层可以选择不同 block。

## 3. 总体结果

| 方法 | Answer@128 | 结构化 Final F1 | 答案 block 曾被检索 | 累计 unique blocks | 平均耗时/条 |
|---|---:|---:|---:|---:|---:|
| Question only | 10.0% | 6.87 | 不适用 | 0 | 2.17 s |
| Full source | 40.0% | **14.17** | 全部可见 | 全部 | 5.61 s |
| Static K3 | 23.3%（7/30） | 11.47 | 90.0% | 36.2 | 15.40 s |
| Dynamic C1K3 | 26.7%（8/30） | 6.45 | 96.7% | 39.7 | 20.50 s |
| Dynamic C3K3 | **33.3%（10/30）** | 10.77 | **100%** | 38.7 | 16.97 s |

结论分成两部分：

- 按用户提出的 `128 tokens 内是否包含答案`，C3K3 比静态 K3 增加 3 条命中，且没有丢失静态方法已命中的问题。
- 按最终答案 F1，C3K3 与静态 K3 基本持平，C1K3 更差。

## 4. 配对统计

### C1K3 对 Static K3

- Answer@128：+3.33 个百分点
- bootstrap 95% CI：[-6.67, +13.33]
- 新增命中：query 0、37
- 丢失命中：query 48
- Final F1：-5.02 分
- F1 95% CI：[-11.99, +1.27]

C1 频繁刷新不稳定，增加了 33% 左右模拟耗时，并降低最终 F1，当前不推荐。

### C3K3 对 Static K3

- Answer@128：+10.0 个百分点
- bootstrap 95% CI：[0, +23.33]
- 新增命中：query 3、20、37
- 丢失命中：0
- Final F1：-0.71 分
- F1 95% CI：[-10.06, +9.44]
- F1 胜/负：6/6，其余持平

Answer@128 有一致的正方向，但样本量仍不足，置信区间下界为 0；Final F1 没有显著差异。

## 5. 分数据集结果

| 数据集 | Static Answer@128 | C3 Answer@128 | Static Final F1 | C3 Final F1 |
|---|---:|---:|---:|---:|
| 2WikiMQA | 3/10 | 4/10 | 10.00 | 5.00 |
| HotpotQA | 4/10 | 4/10 | 20.42 | 13.96 |
| MuSiQue | 0/10 | **2/10** | 4.00 | **13.33** |

C3K3 的正向收益主要出现在更困难的 MuSiQue；2WikiMQA 虽然多命中一条，但 final F1 下降；HotpotQA 命中数不变。

## 6. 代表案例

### Query 3：新增 exact answer

问题：`Who is the brother of the Melissa and Joey Theme Song singer?`

Static K3：

```text
Final answer: Joey Heatherton's brother is the person described in the memory.
```

Dynamic C3K3：

```text
1. The theme song singer is Joey Heatherton.
2. Joey Heatherton's brother is Matthew Lawrence.
Final answer: Matthew Lawrence.
```

- 第 22 个生成 token 时首次包含标准答案。
- Final F1 = 1.0。
- 但中间人物 `Joey Heatherton` 仍有事实错误，因此答案正确不代表整条推理链正确。

### Query 20：动态方法找到下一跳人物

Dynamic C3K3：

```text
1. Marty McFly's daughter is Jennifer Parker.
2. Jennifer Parker is played by Claudia Wells.
Final answer: Claudia Wells ...
```

Static K3 没有出现 `Claudia Wells`，C3K3 在第 21 个生成 token 命中。

### Query 37：生成后才取到答案 block

问题询问电影导演的出生地，答案为 `Methala`。

- 最后一个 prompt token 时，动态与静态都没有任何层选择答案 block。
- C3K3 在 generation token 6 首次重新取到答案 block。
- 随后在 generation token 51 输出 `Methala`。
- Static K3 始终只说“born into a feudal family”，没有地点。

这是当前最直接的“生成状态改变检索目标”案例。

## 7. 系统与统计限制

1. 当前仍是 exact full-dimensional QK 扫描，不是 SVD32/QAbs8 高效索引。
2. 所有 source KV 实际驻留 GPU；mask 只模拟逻辑加载，不代表已有速度收益。
3. 每层 K3 的跨层物理并集平均约 15.7 个 block，不是整个模型只有三个 block。
4. 不限制累计读取量，C3K3 平均累计访问 38.7 个 unique blocks。
5. 30 条样本下，Answer@128 的提升方向一致但未达到强统计显著性。
6. Full-source 的结构化 Final F1 只有 14.17，Qwen3-0.6B 的多跳生成能力仍是主要上限。
7. C3K3 虽然增加答案出现率，但没有提高最终 F1，说明还缺少推理状态约束和 final answer 提取机制。

## 8. 当前判断

本轮结果支持以下较窄结论：

> 动态 QK 检索能够在生成过程中找回静态问题 Q 没有持续保留的下一跳证据，并提高 128-token 轨迹中出现正确答案的概率。

本轮不能支持以下更强结论：

> 动态检索已经稳定提高完整多跳推理质量，或已经实现高效的 10M KV 在线检索。

下一步应该优先解决：

1. 为推理链增加事实一致性评分，而不只检查答案字符串。
2. 扩展到至少 100 条，确认 C3K3 的 Answer@128 提升是否稳定。
3. 比较“静态问题 Q”“最新单 token Q”“最近 3 token Q 聚合”，降低轨迹抖动。
4. 机制稳定后再将 exact QK 扫描替换为 SVD32 候选检索和全维精排。

## 9. 复现文件

- 多样本运行器：`projects/parallel_block_retrieval/src/run_dynamic_kv_multisample.py`
- 单样本动态控制器：`projects/parallel_block_retrieval/src/run_single_query_dynamic_kv_generation.py`
- 配对分析：`projects/parallel_block_retrieval/src/analyze_dynamic_kv_multisample.py`
- 多卡脚本：`projects/parallel_block_retrieval/scripts/run_dynamic_kv_multisample_server.sh`
- 正确协议结果：`projects/parallel_block_retrieval/outputs/dynamic_kv_multisample30_chat_v2/`
