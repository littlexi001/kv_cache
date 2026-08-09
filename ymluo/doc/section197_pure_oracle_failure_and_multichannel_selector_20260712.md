# Section 197：Pure Oracle 失败边界与多通道 Selector

更新时间：2026-07-12

## 1. v440 True-Pure 结果

v440 在所有 action-router 完成后再次应用 runtime constraint，保证 `direct_used=0`。

| 指标 | Full（匹配 M20） | v440 | 相对结果 |
|---|---:|---:|---:|
| Score | 0.3727 | 0.2604 | 69.87% |
| KV ratio | 100% | 5.94% | 16.84x KV reduction |
| Mean online | 3.0151 s | 2.2743 s | 1.33x |
| Mean total | 4.7094 s | 3.9760 s | 1.18x |
| Direct used | 0 | 0 | 通过 Pure gate |

结论：v437 的 6.99% KV、104.06% Full 和 3.20x total speed 不能由当前纯 KV selector 复现。

## 2. v441 Pure Action Frontier

在 7 个依赖 direct operator 的任务上扫描标准 LLM decode：

| Action | Score / Full | KV | Total speed |
|---|---:|---:|---:|
| b128 p16 | 42.41% | 3.78% | 1.10x |
| b256 p16 | 48.03% | 6.92% | 1.12x |
| b256 p64 | 45.53% | 6.92% | 1.09x |
| b512 p64 | 62.55% | 13.21% | 1.11x |
| b512 p128 | 51.29% | 13.17% | 1.08x |
| b1024 p128 | 82.05% | 23.17% | 1.09x |
| b2048 p256 | 76.63% | 40.89% | 1.04x |

逐样本 oracle：

| Oracle | Score / Full | KV | Total speed |
|---|---:|---:|---:|
| Full-target 95% | 109.16% | 29.69% | 1.04x |
| Best-action 95% | 112.41% | 32.29% | 1.04x |

Oracle 虽能恢复质量，但需要约 30% KV，远高于 1%-10% 目标，且端到端速度接近 Full。因此当前 frontier 的问题不是 router，而是 selector 表达能力不足。继续训练 router 或跑 M50 没有意义，M50 gate 已停止。

## 3. 任务级失败边界

| Task | Best-action oracle KV | 观察 |
|---|---:|---|
| gov_report | 59.6% | 全局摘要不能由少量 query-local block 支撑 |
| multi_news | 59.1% | 需要跨文档全局覆盖 |
| passage_retrieval_en | 32.1% | 当前 scorer 没有稳定保留结构化标签和值 |
| samsum | 35.3% | 对话全局信息与生成格式均敏感 |
| qmsum | 19.9% | query-focused summary 需要局部证据与全局覆盖结合 |
| passage_count | 14.6% | 需要全局计数/结构通道 |
| trec | 5.5% | 低 KV 可行，主要需要示例标签覆盖 |

## 4. 新假设：通道缺失而非预算不足

当前 selector 主要优化 query-conditioned retrieval，对 QA 有效，但无法统一支持：

1. Summarization 的全局覆盖；
2. Synthetic/count 的结构与数字完整性；
3. Few-shot classification 的标签和示例覆盖。

因此 v444 测试一个不使用任务表的统一三通道 selector：

- Retrieval channel：现有 semantic/lexical/entity query scorer；
- Global coverage channel：强制在全文位置 bins 中保留分散证据；
- Structure channel：保留标签、数字、段落结构和 query-term certificate。

## 5. v444 实验

所有配置满足：

- 标准 LLM prefill/decode；
- `direct_used=0`；
- 同一个 wildcard policy 作用于全部 7 个任务；
- 不手工给不同任务分配 budget；
- budget 为 384/512，block size 为 16/64。

候选动作：

| Action | Channels | Budget | Block |
|---|---|---:|---:|
| spread_b384_p16 | retrieval + global | 384 | 16 |
| spread_b384_p64 | retrieval + global | 384 | 64 |
| structure_b384_p16 | retrieval + structure | 384 | 16 |
| tri_b384_p16 | retrieval + global + structure | 384 | 16 |
| tri_b512_p16 | retrieval + global + structure | 512 | 16 |
| tri_b512_p64 | retrieval + global + structure | 512 | 64 |

实验在 GPU 5、6、7 三路并行运行。只有当固定动作或新 oracle 在约 10% KV 下达到至少 95% Full，才进入 M50 和 risk-aware router 训练。
