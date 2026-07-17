# Qwen3-0.6B 长上下文失败机制与合成数据设计审计

日期：2026-07-17

## 先给当前确定结论

这批复核说明，旧实验中“长文本让模型找不到正确信息”混合了至少四种不同现象，不能只用一个 candidate accuracy 或一个自动抽取准确率概括：

1. **候选集删掉了真实错误类型。** 旧四条件候选评估没有把 query/start code 放进候选。Qwen3-0.6B 经常直接复制 start；删掉 start 后，短 clean 的 candidate accuracy 是 100%，放回 start 后是 0%。
2. **证据访问和状态推进不是一回事。** 64K clean 烟雾实验中，第一跳 rule cloze 仍选对 consequent；第二跳严格评估先复制输入中间状态，但排除不可能的自环答案后，正确终点排第一。模型没有完全丢失规则，更像在组合、状态更新或解码时失败。
3. **冲突会直接破坏绑定。** 8-seed 短文本校准中，clean 的第一跳/第二跳 cloze 都是 100%；加入一条共享起点的 conflict chain 后，第一跳为 0%，第二跳为 50%。这不是长度效应。
4. **提示能改变输出轨迹，但不能稳定修复冲突绑定。** 结构化 state/rule 模板有时改善回答，效果随长度和冲突条件大幅变化；thinking 提示还经常撞上 256-token 截断。

因此，更准确的研究问题应写成：

> 在固定候选、固定证据和成对样本下，长度、证据到 query 的距离、低信息 filler、相似干扰、共享起点冲突与完整竞争链，分别影响规则访问、实体/状态绑定、多跳组合和最终解码的哪一阶段？

## 数据与评估协议中发现的问题

### 1. 排除 start code 会虚高 candidate accuracy

纠正后的每个 seed 使用固定 13 个候选，始终包含：

- gold start、gold intermediate、gold final；
- conflict intermediate、conflict final；
- 4 个高相似 distractor consequent；
- 2 个 competitor final；
- 补齐的随机候选。

主指标保留 start。另报 `start_excluded_candidate_accuracy`，但只用于和历史口径对照，不能称为完整回答准确率。

短文本 8-seed 校准：

| 条件 | 完整候选准确率 | 排除 start 后 | 自由生成最后已知答案 |
|---|---:|---:|---:|
| clean | 0/8 | 8/8 | 7/8 |
| conflict1 | 0/8 | 1/8 | 0/8 |

这解释了旧实验为什么会同时出现“candidate 很好”和“自由生成很差”。

### 2. candidate mean NLL 不等于自由生成

候选分数是完整字符串的 teacher-forced 平均 NLL；自由生成逐 token 贪心选择。两者可明显分离。例如 few-shot chat 的候选排序可把 gold final 放第一，但自由生成会复制示例答案 `QC30-303`。

所有候选 token 数也不绝对相同：校准样本大多为 7 token，仍会出现 8-token code。因此本轮同时保存：

- mean-NLL 排名；
- total-NLL 排名；
- 每个候选的 token 数、总 NLL、平均 NLL 和 PPL；
- 自由生成文本及多种抽取口径。

### 3. “最后一个已知 code”依赖生成上限

模型可能先给正确终点，随后继续重复规则或提到其他状态。相同 prompt 在 32-token 截断时最后已知 code 可能正确，延长到 64 token 后反而被最后一个后来出现的 code 判错。

所以本轮并列报告：

- 严格第一行；
- 第一个已知 code；
- 显式 `answer/final answer`；
- 最后已知 code；
- 是否曾包含 gold；
- candidate 与 cloze probe。

### 4. `VERIFIED` / `DECOY` 标签并不代表真实冲突

旧协议明确告诉模型忽略 `DECOY RULE`，这测量的是“能否遵循人工类型标签”，不完全等于现实中的证据冲突。若两条冲突信息地位完全相同，则数据本身没有唯一可识别的 gold，报告准确率是不成立的。

外部有效性审计因此另外比较五种协议：

- `typed`：原 `VERIFIED` 对 `DECOY`；
- `source`：官方记录对论坛声明；
- `temporal`：新记录对旧记录；
- `scope`：目标项目对其他项目；
- `ambiguous`：双方同为无差别 `RULE`，只报告选择分布和“任意指定 gold 命中率”，不把它解释成准确率。

## 为什么旧人工审核里 filler + conflict 反而更好

旧 8B 人工审核中，短 `gold + conflict` 是 58/64 严格正确、6 个未完成；`filler + gold + conflict` 是 64/64、0 个未完成，双方都没有明确错误答案。对这 6 个配对差异做精确双侧检验为 `p = 0.03125`，所以它不只是一个完全无迹可寻的随机波动，但它测到的是**完成/输出行为**，不是更强的证据偏好：

- 短 conflict 的 candidate margin 更高、熵更低，并不比 filler 条件更不自信；
- filler 条件只是更少在 256-token 截止前保持未完成状态；
- filler 构造还改变了 gold/conflict 的相对位置、间隔和顺序，不是只增加长度；
- 新的一跳 source/scope 审计中也复现了“短 conflict 差、8K conflict 好”，并能明确归因到证据组织变化。

所以旧结果应解释为：

> filler 改变 prompt 的局部结构与生成轨迹，使模型更容易完成答案；它没有证明更长上下文本身改善了冲突推理。

## 提示结构校准

模型：Qwen3-0.6B；每格 8 个 seed。`final accuracy` 使用生成文本中的最终已知答案；`ordered path` 要求先出现 gold intermediate、后出现 gold final。

### 短文本

| prompt | clean final | conflict final | clean ordered path | conflict ordered path | 主要问题 |
|---|---:|---:|---:|---:|---|
| answer prefix | 100.0% | 12.5% | 0% | 0% | clean 有效，冲突失效 |
| two fields prefix | 87.5% | 25.0% | 0% | 0% | 字段里仍会复制 start/错误 code |
| state fields | 50.0% | 37.5% | 75.0% | 50.0% | 常找到路径但最后字段继续漂移 |
| state fields prefix | 0% | 37.5% | 0% | 37.5% | 前缀诱导不稳定 |
| rule + state fields | 12.5% | 37.5% | 12.5% | 37.5% | rule label 与 state 仍可错绑 |
| thinking, 256 tokens | 62.5% | 25.0% | 100.0% | 87.5% | 87.5%/100% 撞 token 上限 |

### 1K filler

| prompt | clean final | conflict final | clean ordered path | conflict ordered path | token-limit rate (clean/conflict) |
|---|---:|---:|---:|---:|---:|
| answer prefix | 75.0% | 0% | 50.0% | 0% | 12.5% / 0% |
| two fields prefix | 0% | 0% | 0% | 0% | 25.0% / 0% |
| state fields | 25.0% | 12.5% | 100.0% | 12.5% | 0% / 0% |
| state fields prefix | 50.0% | 25.0% | 87.5% | 25.0% | 0% / 0% |
| rule + state fields | **87.5%** | 0% | **100.0%** | 0% | 0% / 0% |
| thinking, 256 tokens | 12.5% | **50.0%** | 37.5% | 75.0% | 100% / 100% |

提示实验的结论不是“找到一个统一最佳 prompt”，而是：

- 外显状态字段可让模型更常把两步路径写出来；
- 输出字段越多，也越可能在正确路径之后继续漂移；
- 冲突条件下最优模板会改变；
- thinking 的表面改善与截断强耦合；
- prompt-only 不足以稳定修复冲突造成的绑定错误。

## 修正后的主实验设计

### 因子

- 长度：short、1K、8K、32K、64K；
- 干扰：clean、16 个低信息 note、4 个高相似规则、1 条共享起点 conflict chain、2 条完整 competitor chain、mixed；
- 主实验位置：gold chain 固定在中间；
- 后续位置扫描：prefix / middle / recent；
- 16 个成对 seed；所有条件共享同一候选池和同一 gold/conflict code。

### 三个查询

1. `full2`：正常两跳问题和自由生成；
2. `hop1 cloze`：直接补全 start 对应 VERIFIED rule 的 consequent；
3. `oracle_hop2 cloze`：把正确 intermediate 直接给 query，再补全第二条 rule。

阶段诊断采用 cloze，是为了减少“不会遵守 lookup 指令”和“没有访问到证据”之间的混淆。cloze 同时报告严格候选和排除不可能自环输入后的条件候选。

## 64K clean 烟雾结果与 RoPE 配置审计

单 seed 结果：

| 查询 | 严格候选 | 排除输入状态后 | 解释 |
|---|---:|---:|---|
| full2 | 错，复制 gold start | 错，停在 gold intermediate | 两跳组合失败 |
| hop1 cloze | 对 | 对 | 第一条规则仍可访问 |
| oracle hop2 cloze | 错，复制输入 intermediate | 对，gold final 排第一 | 第二条规则仍在，但状态/解码有复制偏置 |

正文 65,536 tokens 的 prefill 用时约 25.6 秒；512-token chunk 可在单张 24GB 3090 上完成，4K chunk 会因峰值/碎片 OOM。

随后核对发现，模型 `config.json` 的原生窗口是 **40,960**，旧 runner 却把 `original_max_position_embeddings` 写死为 32,768。这样 `65,536 + query suffix` 会被错误地从 YaRN factor 2 推到 factor 4，并正好跨过显存和 RoPE 的离散边界。正式主实验因此采用：

- `original_max_position_embeddings = 40,960`；
- 长点正文精确为 64,000 tokens；
- 64-token generation；
- prompt 后仍低于 65,536，避免无关的 RoPE 档位跃迁。

## 主实验结果（16 个配对 seed，完整校验）

完整性校验：480 个 body case、1,440 个查询结果、18,720 条候选分数，seed 0–15 无缺失。

### Clean：最终指标下降，但规则访问没有下降

| 正文 tokens | 曾生成 gold | 排除 start 后 full2 | 只在 terminal/final 候选中 | Gold final PPL | Hop1 cloze | Oracle hop2（严格） | Oracle hop2（排除输入） |
|---:|---:|---:|---:|---:|---:|---:|---:|
| short | 100.0% | 100.0% | 100.0% | 6.284 | 100.0% | 100.0% | 100.0% |
| 1,024 | 6.25% | 25.0% | 100.0% | 6.907 | 100.0% | 87.5% | 100.0% |
| 8,192 | 0% | 6.25% | 93.75% | 7.564 | 100.0% | 12.5% | 100.0% |
| 32,768 | 12.5% | 6.25% | 100.0% | 7.989 | 100.0% | 37.5% | 100.0% |
| 64,000 | 6.25% | 0% | 93.75% | 9.276 | 100.0% | 12.5% | 100.0% |

这张表否定了简单命题“序列越长，模型越找不到规则”：

- 第一条规则的直接访问在所有长度都是 16/16；
- 把正确 intermediate 直接给第二跳后，若排除“不前进、原样复制 intermediate”这个非法答案，gold final 在所有长度也都是 16/16；
- 若同时去掉 start/intermediate、只比较真正的 terminal/final 类答案，gold final 从 short 到 64K 仍是 93.75%–100%；
- 真正随长度恶化的是严格第二跳解码：模型越来越倾向复制输入 intermediate；
- full2 排名和自由生成进一步受到两跳组合、状态推进和输出漂移影响；
- clean gold-final PPL 随长度从 6.28 上升到 9.28，说明最终答案 token 的绝对概率确实变低，但不能把它等同为证据未被访问。

因此 clean 长度效应更接近：

> 规则表征仍可被局部 cloze 访问，但 query 到答案的计算路径越来越被 identity/copy prior 和中间状态吸收，最终终点的全序列优势下降。

### 不同干扰改变不同阶段

下表给出 hop cloze 的“排除输入状态”准确率；这用于测量正确 consequent 在合法转移候选中的排名。

| 条件 | 长度 | Hop1 | Oracle hop2 | 主要失败位置 |
|---|---:|---:|---:|---|
| low16 | short / 1K / 8K / 32K / 64K | 100 / 100 / 100 / 100 / 100% | 100 / 100 / 100 / 100 / 100% | 访问不坏，主要是复制/组合 |
| conflict1 | short / 1K / 8K / 32K / 64K | 0 / 0 / 12.5 / 0 / 12.5% | 50 / 100 / 100 / 62.5 / 0% | 共享起点首先破坏第一跳绑定 |
| high4 | short / 1K / 8K / 32K / 64K | 100 / 87.5 / 81.25 / 87.5 / 100% | 75 / 37.5 / 75 / 62.5 / 37.5% | 第二跳更易被相似 consequent 吸走 |
| competitor2 | short / 1K / 8K / 32K / 64K | 100 / 100 / 100 / 100 / 100% | 100 / 100 / 81.25 / 31.25 / 100% | 完整竞争链主要影响第二跳，非单调 |
| mixed | short / 1K / 8K / 32K / 64K | 31.25 / 43.75 / 75 / 100 / 100% | 93.75 / 56.25 / 81.25 / 81.25 / 31.25% | 位置、冲突和相似项交互，明显非单调 |

这里有两个重要结论：

1. **普通 filler 数量不是主因。** `low16` 下两个规则访问 probe 在所有长度均为 100%。
2. **“干扰更多所以更差”也不是充分描述。** mixed 的第一跳反而随长度提高，competitor 的第二跳非单调；改变长度也改变了 gold/conflict/competitor 的相对间隔、顺序和注意力竞争结构。

### PPL 不能单独表示正确性

64K clean 的 gold-final PPL 是 9.276，而 64K conflict 的 gold-final PPL 反而更低，为 6.207；但 conflict 第一跳只有 2/16。冲突和重复 code 让“输出某种 code”整体更可预测，却不保证绑定到正确链。因此必须同时看相对排名、probe 和错误角色。

### 自由生成自动抽取不是稳定主指标

本轮 full2 使用统一 64-token generation。short clean 的 16 个样本全部曾生成 gold，但按“最后一个已知 code”抽取却是 0/16，因为模型在正确终点之后继续输出并提到其他状态。这个结果再次证明，之前 32/64/256-token 实验之间的“答案准确率”差异有一部分来自停止位置和抽取规则。

主机制结论应以 candidate 全分数与 cloze probe 为主，自由生成用于描述输出行为，不作为唯一推理准确率。

## 数据协议外部有效性审计

两跳自然记录协议（official/date/scope）连 clean 都几乎为 0，说明表面改写本身造成地板效应，不能直接用来比较 conflict。将任务降为一跳后，clean 校准恢复：

| 长度 | 协议 | clean 严格候选 | conflict 严格候选 |
|---:|---|---:|---:|
| short | source | 75.0% | 0% |
| short | scope | 100.0% | 75.0% |
| short | temporal | 0% | 0% |
| 8K | source | 100.0% | 100.0% |
| 8K | scope | 100.0% | 100.0% |
| 8K | temporal | 100.0% | 87.5% |

8K source/scope 的 conflict 结果比短文本更好，不应解释成“长度提高推理能力”。构造 8K 时 gold 固定在中间、conflict 位于四分之一或四分之三，文本分隔和相对顺序也一起改变；模型更容易根据 authority/scope 选择 gold。这与旧人工审核里 `filler + conflict` 好于短 conflict 的现象方向一致。

真正的无标签 `ambiguous` 协议没有唯一真值：8K conflict 中任意指定的 gold 命中 6/8，只表示位置/顺序偏好，不能叫 75% 准确率。

## 位置扫描

8 个配对 seed 的 prefix / middle / recent 扫描完整通过：144 个 body、432 个查询、5,616 条候选分数。

下表仍使用排除输入状态后的 cloze accuracy：

| 长度 | 条件 | prefix Hop1 / Hop2 | middle Hop1 / Hop2 | recent Hop1 / Hop2 |
|---:|---|---:|---:|---:|
| 8K | clean | 100 / 100% | 100 / 100% | 100 / 100% |
| 8K | conflict1 | 0 / 25% | 12.5 / 100% | **100 / 100%** |
| 8K | high4 | 100 / 62.5% | 87.5 / 62.5% | **100 / 100%** |
| 64K | clean | 100 / 100% | 100 / 100% | 100 / 100% |
| 64K | conflict1 | 62.5 / 12.5% | 0 / 0% | **100 / 100%** |
| 64K | high4 | 100 / 12.5% | 100 / 37.5% | **100 / 100%** |

只比较 terminal/final 类候选时，high4 的 gold-final accuracy 更直观：

| 长度 | prefix | middle | recent |
|---:|---:|---:|---:|
| 8K | 12.5% | 12.5% | **75.0%** |
| 64K | 0% | 12.5% | **100.0%** |

位置结果说明：

1. clean 局部规则访问对位置不敏感；三种位置的两个条件 probe 都是 100%；
2. 一旦有共享起点冲突或高相似 consequent，gold 靠近 query 能把两个 probe 恢复到 100%；
3. 同样是 64K，middle conflict 两个 probe 都为 0%，recent 都为 100%。所以“总长度”不能单独解释结果，**证据到 query 的距离与冲突/相似干扰存在强交互**；
4. prefix 并不等于 recent：最前面的证据虽然同样明确，却可能在长序列后被中间或后部干扰覆盖。

这也为外部 KV retrieval 提供直接设计依据：如果检索器能把当前 head 所需的证据重新放到 query 附近，收益主要会出现在 conflict/high-sim 条件；clean 条件的访问本来就没有坏。

服务器目录：

```text
/home/fdong/ymluo/projects/qwen3_local_rule_failure_boundary/outputs/length_causal_main_20260717
/home/fdong/ymluo/projects/qwen3_local_rule_failure_boundary/outputs/prompt_structure_calibration2_20260717
/home/fdong/ymluo/projects/qwen3_local_rule_failure_boundary/outputs/data_protocol_audit_20260717
```
