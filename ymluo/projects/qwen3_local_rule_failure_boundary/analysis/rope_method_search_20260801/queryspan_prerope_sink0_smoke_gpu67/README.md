# Query-span pre-RoPE retrieval: sink-free smoke

## 结论

**NO-GO。** 去掉固定 sink 后，Query-span selector 仍没有优于 final-query exact pre-RoPE Top-2%，因此第一版的失败并不只是由“sink 强制包含冲突证据”造成。

实验使用 Qwen3-8B、8K context、4 个独立 seeds、2% 总 token budget；所有 arms 均以原始位置的 post-RoPE score 和原始 V 消费候选。`sink_tokens=0`，因此 gold 与 conflict 都没有被结构性强制保留。

| 方法 | Gold PPL | 首答案 token Acc | Gold-vs-conflict margin | Gold recall | Conflict recall | Gold mass | Conflict mass | Query time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Native full attention | 1.262 | 75% | 4.344 | — | — | — | — | 0.091 s |
| Exact final-query pre-RoPE Top-2% | 1.301 | 75% | 4.281 | 39.37% | 39.48% | 22.14% | 12.17% | 0.302 s |
| Query-span token-max Top-2% | 2.805 | 75% | 3.031 | 36.14% | 40.68% | 21.93% | 11.90% | 0.172 s |
| Query-span block Top-2% | 154.317 | 50% | -1.250 | 42.29% | 60.61% | 23.70% | 13.96% | 0.653 s |

PPL 按 `exp(mean gold NLL)` 聚合；其余为 seed macro mean。

## 为什么 block 平均 recall 看似较高，PPL 却崩溃

block 版呈现明显双峰：

- seed 0：PPL 15,414，Gold recall 0.23%，Conflict recall 86.12%；
- seed 3：PPL 36,548，Gold recall 0.25%，Conflict recall 78.56%；
- seed 1/2 则几乎无损。

2% 总预算为 164 token，其中固定 local window 为 128、current token 为 1，远程只剩 35 个位置。block arm 会先选一个 64-token block，再只能从中保留 35 token；因此某个貌似同时覆盖 Query 词面的冲突 block 一旦胜出，就会排空 gold block。跨 seed 平均 recall 掩盖了这种离散灾难。

token-max 没有 block 截断问题，但仍比 exact final-query pre-RoPE 更低的 Gold recall、更高的 Conflict recall，并使 margin 降低 1.25、PPL 增加 116%。说明将 Query 拆成多个 token 后做 max 聚合，没有恢复“正确限定条件”，反而扩大了宽泛语义匹配。

还需注意以下归因边界：

- seed 同时改变冲突数字、记录顺序与 block 对齐，当前 smoke 不能把三者的影响完全拆开；
- exact-pre 使用原始 pre-RoPE 点积，而 Query-span 使用归一化 cosine，因此实验同时改变了 Query 聚合和相似度度量；若重启该方向，应加入 final-query cosine 控制；
- 只有 4 seeds、单一 8K 长度与单一模板，不能支持跨任务泛化；
- seed 3 在 native 与 exact-pre 中本就失败，存在样本难度混杂；
- 全 head 等权的 attention mass 不是 Value 因果贡献。seed 0 的 exact-pre 即使给 conflict 41.53% mass，仍输出正确答案，说明不能由平均 mass 单独推断答案。

## 主张边界

本 smoke 足以停止当前 Query-span mean/max/block 路线，但不能证明所有 multi-vector 或 set-cover 检索都无效。若未来重启该方向，必须先解决严格预算下的跨 block 分配，并用独立的 source/constraint supervision 或结构信号区分 gold 与 plausible conflict；单纯 pre-RoPE Query-token 相似度不够。

原始数据见 `merged/rows.csv` 与 `merged/summary.csv`。
