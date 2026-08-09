# Qwen3-8B 上的 RULER-32K RoPE 稀疏检索 pilot

## 一句话结果

在 13 个 RULER-32K 子任务、每任务 2 条、每 head 仅保留 2% KV 的配对实验中，pre-RoPE 远程召回、native post-RoPE 消费的 `local/global postscore` 得到 **86.15**，高于 exact post-RoPE Top-2% 的 **83.53**，也略高于 Native Full 的 **85.19**；但相对 exact 的 paired 95% CI 为 **[0.00, 6.15]**，目前只能视为值得扩样的正向 pilot 信号。

## 实验回答了什么

比较严格固定了模型、Qwen tokenizer 生成的 RULER-32K 样本、greedy 解码和每 head 2% 支持集。前缀只完整 prefill 一次；不同方法只改变最后 query 与后续解码阶段如何从历史 KV 中选择和消费 token。

| 方法 | RULER-32K 宏平均 | 相对 exact Top-2% |
|---|---:|---:|
| Native Full | 85.19 | +1.67 |
| exact post-RoPE Top-2% | 83.53 | — |
| **local/global postscore** | **86.15** | **+2.63** |
| local/global blend25 | 82.69 | -0.83 |

postscore 在 26 条样本中改善 3 条、退化 0 条、其余 23 条相同；主要恢复了 FWE 的一个关键词、CWE 的一个高频词和 multivalue 的第 4 个数字。其 NIAH 答案值 recall 从 23.43% 只升到 23.62%，说明任务分数改善并不等同于普遍、更高的答案 token recall。

## 方法判断

目前更成熟的版本是：

1. 最近 128 token 和开头 16 token 强制保留。
2. 远程预算按 pre-RoPE QK 选择。
3. 被选中的位置保留原生位置，使用原始 V 和 native post-RoPE QK 重新 softmax。

这个版本没有修改模型真正消费的分数，因此比 blend 稳定。`blend25` 虽使平均首答案 token NLL 下降，却在一条 UUID 样本中把正确前缀续写成错误后缀，最终宏平均低于 exact；它暂时应作为反例，而不是主方法。

## 结论边界

- 这是 26 条样本的 pilot，不是正式 RULER 全量结果。
- Full 与 full replay 的官方分数完全相同，但 BF16/NF4 下最大 logit 误差不为 0；Full 是行为参考，exact 与方法的严格同核比较更可靠。
- 当前实现为了可审计仍扫描完整 K，同时计算 pre/post 分数，不能据此声称加速。
- 首 token NLL 在 multiquery、UUID、CWE 等多 token 输出上不是完整质量指标，主结论只使用 RULER 官方 score。

## 建议的下一步

优先在 `cwe`、`fwe`、`niah_multivalue`、`niah_multikey_3` 各扩到 20–50 条，以最少计算确认收益和 UUID 风险。若 postscore 仍保持“改善且少退化”，再扩到完整 13 任务；blend 暂停扩样。

详细任务图、证据 recall 和样例见 [visualization_results.md](visualization_results.md)。原始结果位于 `outputs/shard0`、`outputs/shard1`，合并统计位于 `outputs/merged`。

