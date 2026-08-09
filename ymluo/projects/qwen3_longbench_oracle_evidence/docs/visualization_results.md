# LongBench Oracle evidence：18 条扩展实验

状态：**实验已闭合；结论仅适用于严格对齐的 HotpotQA 子集。**

![Qwen3-8B HotpotQA Oracle expansion](../outputs/hotpot_semantic_aligned_18_20260802/merged/oracle_expansion_summary.png)

## 结论

多测以后，Oracle 仍然明显优于 Full，而且这次配对置信区间不再跨 0：

| 条件 | QA-F1 | 严格 EM | Gold-answer PPL | 平均 context tokens |
| --- | ---: | ---: | ---: | ---: |
| Full context | 60.00 | 50.00% | 3.575 | 12,241 |
| Oracle 当前证据 span | **85.19** | **77.78%** | 3.179 | 81 |
| Oracle supporting documents | 83.70 | **77.78%** | **2.107** | 4,388 |
| BM25 documents | 60.00 | 50.00% | 4.840 | 3,361 |
| 等 token 预算随机文本（三次均值） | 14.85 | 12.96% | 332.524 | 4,388 |
| Question only | 31.85 | 27.78% | 17.700 | 0 |

Oracle document 相对 Full：

- QA-F1 提高 **23.70 points**；
- 100,000 次 paired bootstrap 95% CI 为 **[+6.67, +42.59]**；
- gold-answer mean NLL 降低 **0.529**，对应 PPL 从 3.575 降至 2.107；
- Full 答错的 9 条中救回 5 条；Full 答对的 9 条中没有严格 EM 退化。

Oracle document 相对等预算随机文本提高 **68.86 F1 points**，95% CI
为 **[+49.23, +86.82]**。因此结果不能仅由“prompt 变短”解释；短文本中
是否保留真正证据是决定性因素。BM25 与 Full 的平均 F1 完全相同，说明
一般的词法相关文档还达不到 Oracle 上界。

## 与原来 8 条结果的关系

原实验只接受旧 HotpotQA 支持句在 LongBench 中逐字出现的样本，因此
200 条中恰好只有 8 条，不是从大候选池随机抽取。新实验允许经过严格
验证的轻微版本改写，最终得到全部 **18 条**合格样本，相当于新增 10 条。

旧 8 条中 Oracle document 相对 Full 为 `+16.67` F1，CI 跨 0；扩展到
18 条后变为 `+23.70`，CI 下界为 `+6.67`。方向一致，而且不确定性明显
缩小，但 18 条仍不足以代表 LongBench 全集。

## 证据与负控审计

- 200 条输入中，18 条通过全部证据与长度门槛；178 条因至少一个支持
  事实无法在当前 LongBench passage 中严格对齐而拒绝，另有 3 条过长、
  1 条过短。
- 18 条共有 41 个支持 span：35 个 canonical exact，6 个严格 fuzzy。
  独立人工复核的 6/6 个 fuzzy span 均保持数字、实体绑定、否定、比较和
  问题所需事实。
- 来源复算确认 41/41 个 span 的字符区间与原始 LongBench passage 完全
  一致，并保存 passage 与 span SHA-256。
- 54 个随机负控与对应 Oracle document 的 token 数逐一完全相等，且均
  来自冻结样本池中的自然 Wikipedia passage。
- 其中 1 个样本的三个随机负控自然出现了答案字符串。排除该样本后，
  Oracle document 相对 Full 仍提高 **25.10 F1**，95% CI
  **[+7.45, +45.10]**；相对随机文本提高 **72.91 F1**，95% CI
  **[+53.73, +89.80]**。结论不依赖这项污染。

## 能回答与不能回答的问题

这个实验回答的是：**如果外部系统真的选出人工证据，并把它重新编码为
短 prompt，Qwen3-8B 的 LongBench HotpotQA 表现能否超过 Full？** 在这
18 条严格子集上，答案是能。

它仍是 Oracle-RAG / evidence compression，不是保持原始 position id 的
Oracle SparseKV。压缩同时缩短 RoPE 距离、缩小 softmax 竞争集合并改变
上下文组织，因此不能把增益单独归因于检索、RoPE 或 softmax 中的某一个。
严格对齐还会偏向 Wikipedia 版本稳定的样本，结果不可直接写成完整
LongBench 的总体分数。

## 可审计产物

- [图](../outputs/hotpot_semantic_aligned_18_20260802/merged/oracle_expansion_summary.png)
- [汇总 JSON](../outputs/hotpot_semantic_aligned_18_20260802/merged/summary.json)
- [条件汇总](../outputs/hotpot_semantic_aligned_18_20260802/merged/condition_summary.csv)
- [配对比较](../outputs/hotpot_semantic_aligned_18_20260802/merged/paired_summary.csv)
- [逐样本预测](../outputs/hotpot_semantic_aligned_18_20260802/merged/predictions.jsonl)
- [冻结样本](../outputs/hotpot_semantic_aligned_18_20260802/merged/sample_manifest.jsonl)
- [证据映射](../outputs/hotpot_semantic_aligned_18_20260802/merged/evidence_mapping.jsonl)
- [来源复算与敏感性分析](../outputs/hotpot_semantic_aligned_18_20260802/merged/audit/provenance_summary.json)
- [逐 span passage 哈希](../outputs/hotpot_semantic_aligned_18_20260802/merged/audit/passage_provenance.jsonl)

数据口径参考 [LongBench 官方仓库](https://github.com/THUDM/LongBench) 与
[HotpotQA 官方仓库](https://github.com/hotpotqa/hotpot)。
