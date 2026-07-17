# Qwen3-0.6B 合成证据四条件准确率与答案 PPL

日期：2026-07-17

## 实验设计

对同一批 64 个两步符号规则问题做严格的 `conflict × filler` 2×2 配对实验：

- gold chain：两条 `VERIFIED RULE`，从 start code 推导最终 code。
- conflict chain：两条相连的 `DECOY RULE`，与 gold 共用起点但导向错误终点；沿用此前合成数据协议，任务明确要求忽略 DECOY。
- filler：8192-token 中性文本，gold chain 固定埋在中间；有冲突时 conflict chain 位于前四分之一或后四分之一。
- 每个条件使用相同 gold answer、conflict answer 和 8 个候选 code；候选中包含 gold/conflict 的中间状态，用于诊断多步组合失败。

短条件的平均 prompt 长度为 131/191 token，长条件均为 8265 token。

## 指标

- `candidate accuracy`：8 个候选中，完整答案 mean NLL 最低者是否为 gold final code。这是主要的格式无关准确率。
- `generation final accuracy`：自由贪心生成后，从文本中抽取最后一个已知 code，判断是否为 gold final code。
- `gold-answer PPL`：`exp`(全部 gold final answer token 的平均 conditional NLL)。
- `best-wrong PPL`：每题最强错误候选的聚合 PPL。
- `margin = best_wrong_NLL - gold_NLL`；正值表示 gold 胜过所有错误候选。

## 结果

| 条件 | n | Candidate accuracy | 自由生成最终答案准确率 | Gold-answer PPL | Best-wrong PPL | Margin |
|---|---:|---:|---:|---:|---:|---:|
| 1. 仅正确证据链 | 64 | **100.00% (64/64)** | **98.44% (63/64)** | **6.2149** | 7.5860 | +0.1993 |
| 2. 正确链 + 冲突链 | 64 | **1.56% (1/64)** | **0.00% (0/64)** | **6.3751** | 5.4334 | -0.1598 |
| 3. Filler 中埋藏正确链 | 64 | **6.25% (4/64)** | **0.00% (0/64)** | **7.6441** | 6.5315 | -0.1573 |
| 4. Filler 中埋藏正确链 + 冲突链 | 64 | **3.12% (2/64)** | **0.00% (0/64)** | **5.2832** | 4.3194 | -0.2014 |

## 错误类型

候选预测揭示，失败主要不是随机猜测，而是停在第一步中间状态：

| 条件 | Gold final | Gold 中间状态 | Conflict final | Conflict 中间状态 |
|---|---:|---:|---:|---:|
| 仅正确链 | 64 | 0 | 0 | 0 |
| 正确链 + 冲突链 | 1 | 1 | 0 | 62 |
| Filler + 正确链 | 4 | 60 | 0 | 0 |
| Filler + 正确链 + 冲突链 | 2 | 30 | 1 | 31 |

因此：

1. 没有 filler 和冲突时，模型几乎完全掌握两步局部规则。
2. 只有冲突链时，模型大多被冲突链第一步吸引，62/64 选择 conflict intermediate。
3. 只有 filler 时，模型能找到正确链的第一步，但60/64停在 gold intermediate，说明长上下文主要破坏多步组合/状态推进，而不只是完全没检索到证据。
4. 同时有 filler 和冲突时，模型在 gold/conflict intermediate 之间近似各半，最终答案准确率仍接近零。

## 为什么第4种的 Gold-answer PPL 反而最低

第4种的 gold PPL 为 5.2832，低于第1种的 6.2149，但这不表示回答更正确。冲突链和大量相似 code 提高了“输出 code 形态”的整体可预测性，同时更强地提高了错误中间状态的概率：第4种 best-wrong PPL 为 4.3194，仍低于 gold PPL，margin 也是四种条件中最差的 -0.2014。

因此答案 PPL 必须和候选 margin/准确率一起看。只报告 gold-answer PPL 会把“所有相似答案都变得更容易预测”误判成推理能力提升。

## 配对效应

- 无 filler 时加入冲突链：candidate accuracy `-98.44` 个百分点；gold NLL `+0.0254`。
- 无冲突时加入 filler：candidate accuracy `-93.75` 个百分点；gold NLL `+0.2070`。
- 已有 filler 时再加入冲突：candidate accuracy `-3.13` 个百分点，但 margin 继续下降 `-0.0441`。

这说明冲突和长上下文单独都足以让 Qwen3-0.6B 在该两步任务上失败。出现地板效应后，准确率无法继续大幅下降，margin 比准确率更能显示第4种仍然更难。

## 限制

- 结论针对 Qwen3-0.6B、两步人造 code 规则和显式 `DECOY RULE` 标签。
- 真实矛盾文本通常没有可靠标签，需要补做“双方都声明为有效证据”的冲突协议。
- 自由生成严格格式准确率四组均为0，因为模型常输出多个 code 或解释；因此主表另外提供最终 code 抽取准确率和候选准确率。
- 下一步应扫描 chain length 1/2/4、filler 1K–32K，并把 conflict chain 数量从1扩展到多链。

## 文件

- 完整逐样本结果：`../outputs/four_condition_64seed_20260717/aggregate/results.csv`
- 条件汇总：`../outputs/four_condition_64seed_20260717/aggregate/summary.csv`
- 运行脚本：`../src/run_four_condition_answer_eval_20260717.py`
