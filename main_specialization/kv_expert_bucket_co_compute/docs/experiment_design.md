# KV Bucket and Expert Bucket Co-compute: Experiment Design

## Problem Decomposition

### Question 1: Does a bucketable QK relation exist?

已完成第一版微测试：QK 两跳 closure、query retrieval stability、weight-kernel diagnostics。

下一步需要跨文本、模型规模和 top-k 验证。

### Question 2: Is head-level necessary?

对比：

1. token-level bucket；
2. KV-head-level bucket；
3. query-head-level bucket。

固定 candidate ratio 和 active expert parameter count，比较 attention mass recall、NTP 和 specialization。

### Question 3: Is center removal necessary?

对比：

1. raw Q/K；
2. per-layer center；
3. per-layer/per-head center；
4. top-PC removal；
5. centered + normalized Q/K。

通过条件：centered 版本在相同 candidate ratio 下提高 attention mass recall 或降低 NTP loss，而不是只改变 bucket balance。

### Question 4: Should KV and expert share one bucket?

最关键的 factorial control：

| Structure | KV selector | Expert router |
|---|---|---|
| Full baseline | none | ordinary MoE |
| KV only | learned bucket A | ordinary MoE |
| Expert only | full attention | learned bucket B |
| Separate | learned bucket A | learned bucket B |
| Shared | learned bucket C | same bucket C |

共享假设只有在 `Shared` 优于 `Separate` 时才得到支持。

### Question 5: Does it work on real-data training?

先使用小模型和真实文本做短训练，随后才扩大模型。Synthetic 只用于诊断已知 feature，不作为最终有效性证据。

## Minimal Real-data Experiment

建议第一版：

1. 从相同初始化训练五个 control；
2. 固定训练 token、batch、optimizer、active expert parameters；
3. KV candidate ratio 设为约 `25%`；
4. 每层保留 local window，避免把局部语法需求混入 global bucket 失败；
5. 每 100 step 保存 stage metrics。

## Required Metrics

### Retrieval

- KV candidate ratio；
- attention mass recall；
- top-attention-token recall；
- sparse/full attention-output cosine；
- per-layer/head closure and bucket purity。

### Expert specialization

- expert load entropy；
- effective experts；
- routing margin；
- same-bucket expert overlap；
- expert ownership stability across checkpoints；
- same expert token 的 next-token-logit similarity。

### End task and system

- NTP loss / perplexity；
- downstream accuracy；
- KV bytes read；
- measured decode latency；
- bucket-index overhead。

## Pass Conditions

共享 bucket 至少满足：

1. 在相同 KV candidate ratio 下，NTP 不劣于独立 K-index；
2. 在相同 NTP 下，expert ownership 比 ordinary/separate routing 更稳定；
3. bucket 没有 collapse；
4. 实际延迟收益大于 bucket 计算开销。

## Fail Conditions

1. Shared 只达到 KV-only 的效果，没有 specialization 增益；
2. Separate 明显优于 Shared；
3. 需要接近 full KV 才能保持 NTP；
4. routing collapse 或所有 head 学到同一个 bucket；
5. 训练有效但 decode overhead 抵消访存节省。

## Insufficient Evidence

只在 synthetic、单一文本、单层或单头成立；或者只看到 attention mass 提升但没有 NTP/system 改善。
