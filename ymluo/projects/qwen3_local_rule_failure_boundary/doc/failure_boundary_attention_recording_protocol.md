# 模型失败边界：Attention 统一记录口径

## 目的

所有长上下文实验使用同一组字段，便于回答三个问题：

1. 模型在哪个长度附近从正确变为错误？
2. 翻转主要来自证据自身 QK 匹配退化，还是 Softmax 分母竞争？
3. 普通 filler、语义干扰、矛盾信息和相对位置分别改变了哪条内部通路？

## 每个长度必须保存的输出指标

- `gold_probability`、`gold_ppl`
- `answer_margin = log P(gold) - log P(strongest wrong)`
- `top1_correct`、`candidate_correct`
- `strongest_wrong_token`

`answer_margin = 0` 是答案翻转边界。只保存 PPL 不足以定位错误答案何时超过正确答案。

## 每个长度必须保存的 Attention 指标

- `evidence_mass`：当前实验定义的真实证据集合获得的总 mass
- `other_token_mass = 1 - evidence_mass`
- `outside_top20_mass`：Top-20 token 之外的总 mass
- `attention_entropy`、`effective_tokens`
- 分步骤证据 mass，例如：
  - `start_key_mass`
  - `hop1_result_mass`
  - `hop2_input_mass`
  - `hop2_result_mass`
- 冲突实验额外保存：
  - `conflict_target_mass`
  - `conflict_block_mass`
  - `conflict_label_mass`
  - `evidence_label_mass`

必须保存 `evidence_scope`。例如 `four_atomic_positions`、`hop2_result_only` 和
`verified_catalog_entry` 的分子不同，不能直接比较绝对 mass。

## Pre-softmax 与分母指标

- `evidence_qk_logit`
- `evidence_qk_cosine`
- `evidence_rank`
- `evidence_logsumexp`
- `non_evidence_logsumexp` 或 `softmax_logsumexp`
- `evidence_log_odds`

单个 head 中：

`log attention(evidence) ≈ evidence_qk_logit - softmax_logsumexp`

因此：

- 证据 logit 下降：证据自身匹配退化；
- logsumexp 上升：其他 token 的 Softmax 竞争增强；
- 二者同时发生：方向退化与分母稀释共同作用。

## 边界扫描方法

1. 用较粗步长找到 `answer_margin` 第一次变号的区间。
2. 在该区间缩小步长，直到逐 token 或达到实验允许的最小步长。
3. 同时报告首次失败与恢复；不能假设长度曲线单调。
4. 如果不同长度改变了干扰块数量，必须把 `conflict_occurrences` 单独记录，不能把结果全部归因于长度。

## 统一产物

运行 `src/build_failure_boundary_attention_registry.py` 后生成：

- `attention_failure_registry.csv`：逐实验、逐样例、逐长度的统一指标；
- `boundary_summary.csv`：每个实验组的首次失败、margin 边界和端点变化；
- `registry_manifest.json`：数据来源和口径说明。

新增实验时应扩展加载器或直接输出相同的统一字段。
