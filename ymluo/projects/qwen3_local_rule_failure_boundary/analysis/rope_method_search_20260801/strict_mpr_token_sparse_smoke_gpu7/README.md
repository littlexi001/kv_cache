# Token-sparse Strict MPR smoke

**模型/数据：** Qwen3-8B，8K，seed 0，exact pre-RoPE Top-2% support 固定。  
**干预：** 每层每 head 最多修改 1 或 4 个 remote tokens；每个 token 最多 8 个频率对，单对相位位移不超过 0.25 rad。  
**状态：** 单 seed 因果 smoke，不是正式效果表。

## 结果

| Arm | Gold PPL | 首 token 正确 | Gold mass | 实际触发比例 | Query 时间 |
|---|---:|---:|---:|---:|---:|
| exact-pre baseline | 4.678 | 0 | 5.670% | -- | 0.42 s |
| target, top-1/head | 3.733 | 0 | 5.721% | 4.35% | 20.61 s |
| random plane, top-1/head | 4.284 | 0 | 5.666% | 4.35% | 1.04 s |
| target, top-1 + mass preserve | 4.678 | 0 | 5.670% | 4.35% | 20.44 s |
| target, top-4/head | 5.091 | 0 | 5.828% | 16.03% | 73.68 s |
| random plane, top-4/head | 4.468 | 0 | 5.740% | 16.03% | 2.64 s |
| target, top-4 + mass preserve | 5.114 | 0 | 5.736% | 16.03% | 74.65 s |
| random, top-4 + mass preserve | 4.076 | 0 | 5.675% | 16.03% | 3.39 s |

所有 arms 的 evidence recall 和两链命中完全相同，因为 support 被严格冻结。no-op、support、token cap 与随机相位范数匹配审计均通过。

## 判断

这个 smoke **不支持把 MPR 升级为新位置编码**：

1. 所有 arms 的首答案 token 都错误，PPL 改善没有恢复任务成功。
2. top-1 的定向 arm 比随机 arm 好，但二者只匹配相位位移范数；定向 arm 的实际 QK lift 约 1.082，而随机 arm 只有约 0.123，因此不是“同 score effect”的公平方法对照。
3. 对固定 Query、固定 support 和固定 V，任何相位修复都严格等价于把同样的逐 token 标量差加到 attention logits。matched additive-score control 会逐元素复现结果，所以这里没有独立的 PE 效应。
4. top-1 mass-preserve 完全回到 baseline，说明 top-1 的收益全部来自改变被选 token 相对其余 support 的 softmax 份额；保留该分区质量后，所谓 phase repair 没有剩余作用。
5. top-4 定向 arm反而恶化，random + mass-preserve 更好；效果对 cap 和控制方式不稳定。
6. 定向 solver 比 baseline 慢约 49 倍（top-1）和 174 倍（top-4），不具部署意义。

**最终决定：NO-GO as positional encoding。** 该实验只保留为“稀疏 score lift 的因果消融”；不再扩大 seeds/长度，也不再把它包装为 RoPE repair 方法。

