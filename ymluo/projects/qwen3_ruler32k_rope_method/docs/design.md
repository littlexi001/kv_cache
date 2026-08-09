# RULER-32K 上的 RoPE 语义召回验证

## 科研问题

在模型、提示、样本和每个 head 的 2% KV 支持集预算都相同的条件下，使用 pre-RoPE 语义分数召回远程 token，是否比 exact post-RoPE Top-2% 更适合 RULER-32K？

主张被限定为“查询/解码阶段的稀疏 KV 支持集选择”。长前缀仍以原模型完整 prefill，不把本实验写成端到端稀疏 prefill 或实际加速结果。

## 可证伪假设

- H1：`local_global_postscore` 的 RULER 官方平均分高于 `rope_top2`。
- H2：提升主要出现在需要远程检索的 NIAH 和 QA，而不是所有任务都提升。
- H3：若 pre-RoPE 召回真正救回远程证据，则在可对齐任务上，答案证据 token recall 应与任务得分同向改善。
- H4（探索性）：`local_global_blend25` 在召回后补偿部分远程相位损失，可能进一步改善任务分数或正确答案首 token NLL。

任何一个方法只在少量样本上领先、置信区间跨零，均记为 pilot 信号而非稳定提升。证据 recall 下降但任务分数上升时，只能说明输出行为改善，不能声称由证据召回介导。

## 方法的精确定义

### 输入

- Qwen3-8B；同一量化和计算精度用于所有变体。
- 一条 RULER-32K prompt，最后一个 prompt token 作为第一次稀疏查询。
- 每层缓存中的原始 K/V；每层每个 attention head 独立选择支持集。

### 参数

| 参数 | 值 | 含义 | 太小/太大的影响 |
|---|---:|---|---|
| `ratio` | 0.02 | 每个 head 保留 `ceil(0.02N)` 个位置 | 太小丢证据；太大失去 2% 对照意义 |
| `local_window` | 128 | 强制保留最近 token 的上限 | 太小损害局部格式；太大挤占远程预算 |
| `sink_tokens` | 16 | 强制保留开头位置 | 太小丢系统/起始锚点；太大挤占远程预算 |
| `blend` | 0.25 | 探索性远程语义分数比例 | 0 等于 native postscore；太大改变模型原生标尺 |

### 中间变量

- `s_post(q,k_p)`：原生 post-RoPE QK 分数。
- `s_pre(q,k_p)`：通过逆 RoPE 恢复的 pre-RoPE Q/K 点积。
- `S_h`：第 h 个 head 的支持位置集合。
- `a_h`：仅在 `S_h` 上重新 softmax 的注意力权重。

### 算法

1. 对前 `N-1` 个 prompt token 做一次完整、共享的 chunked prefill。
2. 对最后一个 prompt token和每个后续生成 token，计算每层每个 head 的 pre/post-RoPE 分数。
3. `native_full` 使用所有历史位置；`rope_top2` 按 `s_post` 取每个 head 的 Top-2%。
4. `local_global_postscore` 先强制加入 sink、最近窗口和当前 token，再在剩余预算中按 `s_pre` 召回远程位置；消费阶段使用这些位置原生的 `s_post` 和原始 V 做稀疏 softmax。
5. `local_global_blend25` 使用相同支持集；远程候选消费分数为 `0.75*s_post + 0.25*calibrate(s_pre)`。
6. greedy 解码，并以 RULER 官方 string-match 规则评分。
7. 保存每个样本、任务、变体的预测、分数、候选预算审计、证据代理 recall/mass、首答案 token NLL 和耗时。

### 通过条件

- 每个稀疏 attention 观测严格保留 `ceil(0.02N)` 个互异位置。
- `full_rope_replay` 与未 patch 的 `native_full` 最大 logit 误差足够小，证明 patch 的 dense 路径等价。
- 同一样本的所有变体使用相同 prompt token IDs、相同前缀 cache 和相同 greedy 解码上限。
- 主方法相对 exact Top-2% 的 paired 官方分数差为正；若 bootstrap 95% CI 跨零，结论标记为不确定。

### 失败原因

- `protocol_mismatch`：任务、prompt、tokenizer 或评分与固定协议不一致。
- `length_mismatch`：实际 prompt 不在目标 32K 附近。
- `support_budget_violation`：支持集大小不等于 2% 预算。
- `duplicate_support`：同一个 head 的支持集中有重复位置。
- `dense_replay_mismatch`：patch 后 dense replay 不等价。
- `oom_or_runtime_failure`：显存或执行错误。
- `insufficient_evidence`：样本过少或 paired CI 跨零。

## 产物契约

- `data/*.jsonl`：Qwen tokenizer 生成的固定样本。
- `outputs/*/rows.jsonl`：逐样本、逐变体原始记录。
- `outputs/*/summary.json`：按任务和总体聚合以及 paired bootstrap。
- `outputs/*/task_scores.png`：任务级可视证据。
- `docs/visualization_results.md`：结果和失败样例解释。

