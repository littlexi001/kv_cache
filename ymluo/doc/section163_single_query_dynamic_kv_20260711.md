# 单样本动态 KV 检索生成实验

## 1. 实验目的

验证以下局部假设：模型不需要在生成开始前一次性获得全部证据；随着模型生成新的推理 token，最新 Q 可以重新检索少量历史 KV block，并使后续 token 转向正确答案。

本实验只使用 `clean64` 的 query 0：

- 问题：`Where was the wife of Lou Breslow born?`
- 第一跳：`Lou Breslow's wife was Marion Byron.`
- 第二跳：`Marion Byron was born in Dayton, Ohio.`
- 标准答案：`Dayton, Ohio`
- source context：5,888 tokens，23 个 256-token block
- 答案 block：全局 block 20096，局部 block 10

## 2. 实现方式

这不是把检索文本重新拼入 prompt 的 RAG 实验，而是真实 KV 稀疏注意力模拟：

1. Qwen3-0.6B 对 source context 做一次 prefill，得到各层真实 K/V cache。
2. 问题和生成内容逐 token 前向，已生成 token 的本地 KV 始终保留。
3. 每层使用当前 token 的真实 Q 与 source K 做 QK 打分。
4. 同一层的所有 query head 共享 Top-K block；不同层允许选择不同 block。
5. 未选中的 source KV 在该层该 token 的 attention 中被屏蔽。
6. 最多自由生成 128 tokens，不限制累计读取量或累计 unique blocks。

当前 `Top-1/2/3` 是“每层共享的 1/2/3 个 block”，不是全模型物理并集只有 1/2/3 个 block。关键预测步中，Top-3 跨 28 层的物理并集为 13 个 block。该实现用于验证机制，不是加速 kernel。

## 3. 自由生成结果

`C` 表示每生成多少 token 刷新一次 block，`K` 表示每层每次保留多少 block。

| 方法 | Answer Hit@128 | 曾取到答案 block | 累计 unique blocks | 主要输出 |
|---|---:|---:|---:|---|
| Question only | 否 | 不适用 | 0 | `New York City` |
| Full source | 否 | 全部可见 | 23 | `United States` |
| C1K1 | 否 | 否 | 2 | `city of Breslow` |
| C1K2 | 否 | 是 | 20 | `Ottoman Empire` |
| C1K3 | 是，第 9 个归一化 token | 是 | 23 | `Dayton, Ohio` |
| C2K1 | 否 | 否 | 2 | `city of Breslow` |
| C2K2 | 否 | 是 | 20 | `Iznik, Turkey` |
| C2K3 | 否 | 是 | 21 | `United States` |
| C3K1 | 否 | 否 | 2 | `city of Breslow` |
| C3K2 | 否 | 是 | 20 | `Marion Byron` |
| C3K3 | 是，第 9 个归一化 token | 是 | 21 | `Dayton, Ohio` |

C1K3 和 C3K3 的输出都包含正确地点，但完整句子是 `Lou Breslow was born in Dayton, Ohio`。地点正确，主语归因错误。因此：

- 按用户提出的 `Answer Anywhere@128`，两组成功。
- 按严格语义推理，不能算完全正确。
- `K=3` 显示出必要性，但 cadence 结果非单调：C1、C3 成功，C2 失败。

在 C1K3 中，答案 block 最早于 generation token 2 被第 24 层取到；到 token 8-10，更多中高层同时取到该 block，随后输出出现 `Dayton, Ohio`。这与“生成状态变化后检索转向下一跳证据”的假设一致。

## 4. 条件预测实验

为了隔离验证“模型已经说出第一跳后，是否能够预测第二跳”，固定历史为：

```text
Lou Breslow's wife was Marion Byron.
Marion Byron was born in
```

该 prefix 不包含 `Dayton` 或 `Ohio`。比较下一个目标 token ` Dayton`：

| 上下文方式 | `Dayton` 概率 | 词表排名 | 实际续写 | Hit@128 |
|---|---:|---:|---|---:|
| Question only | 0.0026% | 1013 | `1879 in the United States` | 否 |
| Full source | 13.15% | 3 | `1911` | 否 |
| Dynamic Top-1 | 0.0015% | 2257 | `the United States` | 否 |
| Dynamic Top-2 | 7.05% | 4 | `Ohio` | 否（缺少 Dayton） |
| Dynamic Top-3 | **85.81%** | **1** | `Dayton, Ohio` | **是，第 2 个归一化 token** |

关键预测步的答案 block 覆盖：

| 方法 | 选择答案 block 的层数 | 28 层物理 block 并集 |
|---|---:|---:|
| Top-1 | 0/28 | 2 |
| Top-2 | 7/28 | 7 |
| Top-3 | 15/28 | 13 |

Top-3 将正确 token 从 full-source 的第 3 名、13.15% 提升到第 1 名、85.81%。这是本实验最强的正面结果：当生成历史已经包含中间实体 `Marion Byron` 时，动态少量 KV 检索能够在中高层集中取回答案 block，并显著增强正确下一 token。

## 5. 结论与限制

本实验支持局部机制假设，但尚未证明完整系统：

1. **支持**：更新后的生成状态能够改变 Q，并把检索焦点转向第二跳答案 block。
2. **支持**：每层 Top-3 在条件预测中足以生成 `Dayton, Ohio`，且优于 full-source 贪心结果，体现了去噪作用。
3. **未完全支持**：Qwen3-0.6B 在无 teacher prefix 的完整多跳推理中不稳定，full-source 自由生成也失败。
4. **指标限制**：自由生成成功组包含正确答案字符串，但错误地把地点归因给 Lou Breslow；Answer Hit 不能替代严格语义正确率。
5. **系统限制**：当前是 source-known、23-block 的 exact simulation，不是从全局 10M KV 索引端到端检索，也没有测速度。
6. **预算说明**：按用户要求没有限制累计读取量；Top-3 最终可访问 source 中大部分或全部 block。

因此当前最准确的结论是：

> “模型说出中间事实后，用新 Q 动态检索少量 KV，再预测下一跳答案”已经在单条真实数据上得到明确正结果；“模型无需任何辅助就能自行形成正确中间事实并完成整条推理链”尚未证明。

## 6. 复现文件

- 实现：`projects/parallel_block_retrieval/src/run_single_query_dynamic_kv_generation.py`
- 多卡脚本：`projects/parallel_block_retrieval/scripts/run_single_lou_dynamic_kv_server.sh`
- 服务器结果：`projects/parallel_block_retrieval/outputs/single_lou_dynamic_kv_v1/`
- 本地汇总：`projects/parallel_block_retrieval/outputs/single_lou_dynamic_kv_v1/summary.json`
