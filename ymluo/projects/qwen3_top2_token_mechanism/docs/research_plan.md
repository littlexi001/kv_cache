# 研究计划：为什么 Top-2% Attention 有效？

## 1. 核心现象

对每层、每个 attention head，在每个 query step 只允许访问 attention score 最高的约 2% 历史 token，已有曲线显示 PPL 在 2% 附近很好，且可能低于 full attention。这个现象至少有四种互相竞争的解释：

1. **质量覆盖假设**：注意力本来就高度集中，2% 已覆盖绝大多数有效 attention mass。
2. **去噪假设**：full attention 的长尾不是中性的小权重，而会累积无关 value 干扰；Top-2% 相当于推理时去噪。
3. **位置先验假设**：Top-2% 主要只是 sink + recent；QK oracle 的语义选择并非必要。
4. **稀疏远程检索假设**：sink + recent 只能解释一部分，剩余收益来自少量 query-conditioned remote token。

项目目标不是“证明 2% 永远最好”，而是区分这四个解释，并找出它们在哪些 layer/head、上下文长度和任务上成立。

## 2. 预注册假设与可证伪条件

### H1：2% 接近 attention 的有效支持集

支持证据：

- 多数 layer/head 的 Top-2% full-attention mass 很高；
- `1 / sum(p^2)` 或 `exp(entropy)` 对应的有效支持比例集中在 2% 附近；
- 2% cutoff 附近存在可辨认的 score gap。

反证：Top-2% mass 很低、有效支持比例远大于 2%，但 PPL 仍改善。此时更支持去噪或非线性层间补偿，而不是简单 mass coverage。

### H2：低于 full 的 PPL 来自去噪，而不是实现伪影

支持证据：

- `top_attention@100%` 精确复现 full；
- bottom-attention、random 和位置对照显著差于 Top-2%；
- Top-2% 的逐 token NLL 改善集中在可解释 token，而非一个错位的评分区间；
- 2% 在多个文本、长度和 chunk size 上稳定优于 full。

反证：100% 不能复现 full，或改动 chunk size 后 2% 低点消失。此时先判定为实现/评分协议问题。

### H3：Top-2% token 可被少数角色解释

统计三类互斥事件角色：

- `sink`：历史位置 `< role_sink_tokens`；
- `recent`：非 sink 且距离 query 不超过 `role_recent_tokens`；
- `remote`：其余历史位置。

再按词法类别统计：newline、whitespace、punctuation、symbol、number、word、alphanumeric、mixed/subword。类别富集度使用：

```text
selection_enrichment = selected_event_share / eligible_event_share
```

这样不会把靠前 token 因“可被选择的 query 次数更多”误判为语义富集。

### H4：等预算 sink + recent 可以替代 Top-2%

固定每个 query 的历史预算：

```text
B_t = ceil(0.02 * history_len)
```

测试 `sink_recent_sN`，其中 `N ∈ {0,1,2,4,8,16,32}`，前缀最多使用 N 个预算，其余预算给 recent。判断分三层：

1. **行为等价**：candidate - Top-2% 的配对 NLL block-bootstrap 95% CI 完全位于 `[-0.01, +0.01] nat/token`。
2. **描述性接近**：PPL 比值在 `[0.99, 1.01]`。
3. **机制接近**：同时报告 Top-2% position recall、Top-2% mass recall 和稀疏分布 cosine。

如果行为等价但 overlap 很低，结论应是“存在功能等价的局部路径”，不是“sink+recent 找回了同一批 token”。

## 3. 实验矩阵

### Phase A：协议审计与曲线复现

Selector：

- full attention；
- Top-attention ratio `{0.1,0.5,1,2,4,8,16,32,100}%`。

必须报告：PPL、逐 token NLL、100% 与 full 的误差、实际 keep ratio。

### Phase B：为什么 2% 是低点

对目标 2% 收集：

- full distribution entropy；
- entropy effective support；
- inverse-participation effective support `1/sum(p^2)`；
- Top-2% 历史 mass 和 Top-2% + self mass；
- 第 B 和 B+1 个历史 score 的 cutoff gap；
- layer/head 分布，而不只给全局均值。

对照：bottom-attention 与 deterministic random。如果它们也改善到同一水平，“高 attention score”并非关键解释。

### Phase C：这 2% 是什么 token

输出每个 token 位置的：

- selected count 与 exposure-corrected selection rate；
- full-attention mass 累积；
- sink/recent/remote 事件计数；
- tokenizer piece、decode text、词法类型；
- Top-N 累积事件占比。

后续任务数据若有证据 span，再新增：evidence overlap、answer/entity overlap、结构标记 overlap。自然语言 continuation 上不能把“高频标点”直接解释为任务证据。

### Phase D：sink + recent 等预算替代

主 controls：

- `sink_recent_s0/s1/s2/s4/s8/s16`；
- pure recent；
- pure sink；
- deterministic random；
- bottom-attention。

贡献消融（预算会降低，因此不与等预算结果混为一列）：

- `top_attention_drop_sink`；
- `top_attention_drop_recent`；
- `top_attention_drop_remote`。

这些 drop 模式回答“oracle Top-2% 中哪类 token 必要”，而等预算 sink+recent 回答“一个廉价位置规则能否替代 oracle”。

### Phase E：泛化

最小矩阵：

- 文本：至少 3 个不同领域长文本；
- context：2k、4k、8k、16k（模型允许时）；
- eval continuation：至少 512 token；
- 模型：先 Qwen3-0.6B，再加一个更大 Qwen3 checkpoint；
- 每个条件保留完全相同的 tokenization 和评分区间。

下游任务补充 accuracy/F1/ROUGE 等任务指标。PPL 等价不自动推出生成质量等价。

## 4. 主要统计与图

1. PPL vs keep ratio，x 轴 log scale，标记 full baseline。
2. 每种 selector 相对 Top-2% 的配对 delta NLL 和 95% CI。
3. layer × head 的 Top-2% mass / effective support heatmap。
4. sink allocation vs position recall、mass recall、distribution cosine、PPL。
5. sink/recent/remote 事件占比与 layer 分层。
6. 词法类别 selected share、eligible share 和 enrichment。
7. Top token 累积曲线，避免只展示几个极端样例。

## 5. 混淆因素检查表

- full prefill、sparse eval 的边界是否与原曲线一致；
- 当前 self token 是否计入 2% 预算；
- ratio 使用 `ceil` 还是 `floor`；
- attention 排序是 pre-softmax QK 还是 post-softmax（排序等价，但 mask/数值实现要一致）；
- causal future mask 是否被错误当成有限 score；
- GQA 中 attention head 与 KV head 是否混淆；
- PPL label 是否左移一位；
- 相同 prefill 是否因 cache mutation 导致 selector 间不公平；
- `drop_*` 模式实际预算降低，不能冒充等预算对照；
- 单文本的标点和换行规律不能推广成普遍 token 机制。

## 6. 决策表

| 结果 | 结论 |
| --- | --- |
| sink+recent 与 Top-2% NLL 等价，且 mass recall 高 | 2% 主要由简单位置先验解释 |
| sink+recent NLL 等价，但 overlap/mass recall 低 | 存在功能等价路径，不能说选中了同一批 token |
| sink+recent 明显更差，drop_remote 也明显更差 | query-conditioned remote token 是必要成分 |
| random/bottom 与 Top-2% 同样好 | 高 attention token 不是主要机制，检查正则化或实现效应 |
| 2% 只在一个文本或 chunk size 成立 | 局部现象，不做普遍机制结论 |
| 100% 不等于 full | 实验无效，先修协议 |

