# Suppression certificate block / line aggregation

## 目的

原 safety probe 的 certificate_aurocs.csv 直接混合不同 layer、head 和 token。其 pooled AUROC 接近 0.5 时，仍可能存在两类被平均掉的结构：

1. 同一个 token 在多个 layer/head 上形成稳定信号；
2. 一整条 evidence line 的多个采样 token 共同形成信号。

本分析器只读取已有 JSONL，在 CPU 上完成聚合，不加载模型、不访问服务器，也不修改 safety runner。

## 输入

脚本：

src/analyze_suppression_certificate_block_aggregation.py

输入可以是：

- 单个 *_certificate_samples.jsonl；
- 一个 shard 目录；
- 同时给出多个 shard；
- merged 目录。若 merged 的 summary.json 保存了可访问的 source_dirs，脚本会自动跟随；若路径已失效，应显式传入 shard。

line id 不在 JSONL 中。脚本会读取同目录、同 stem 的 *_result.json，使用 case.records[].span 将 token 映射回 evidence line。找不到 result 时不会猜测精确行边界，而会写入：

line_metadata_source=class_fallback_no_result_json

已有 probe 只保存采样 token，因此这里的“line aggregation”是该 line 内**所有已采样 token**的聚合，不等价于补算整行所有未采样 token。输出中的 sampled_token_count 明确记录覆盖量。

## 运行

分析两个 shard：

    python src/analyze_suppression_certificate_block_aggregation.py \
      --inputs \
        outputs/20260801_suppression_certificate_safety_gpu67/shard_gpu6 \
        outputs/20260801_suppression_certificate_safety_gpu67/shard_gpu7 \
      --output-dir \
        outputs/20260801_suppression_certificate_block_aggregation

从 merged 入口运行：

    python src/analyze_suppression_certificate_block_aggregation.py \
      --inputs outputs/20260801_suppression_certificate_safety_gpu67/merged \
      --output-dir outputs/20260801_suppression_certificate_block_aggregation

默认分析四个指标：

- pre_score
- post_score
- pre_suppression
- grid_envelope_suppression

可用 --metrics 覆盖。positive_fraction 默认表示 score 大于 0 的 layer/head 比例，可通过 --positive-threshold 修改阈值。

## 默认输出

### token_aggregates.csv

每个 (length, seed, class, token_position) 跨 layer/head 聚合：

- mean
- max
- q90
- positive fraction

同时输出 all_sampled 和 decisive_only。

### token_aurocs.csv

对每个 metric 和 reducer 报告：

- Gold vs conflict；
- Gold vs all non-gold；
- 每个 seed 内 AUROC；
- 每个 length 跨 seeds pooled AUROC；
- 每个 length 的 seed-level macro AUROC；
- 全长度 pooled 与 macro 版本。

判断 pooled AUROC 是否掩盖结构时，应同时看：

1. evaluation_level=pooled_seeds_by_length；
2. evaluation_level=macro_mean_of_within_seed_aurocs；
3. 各个 within_seed 行。

### line_aggregates.csv

每条恢复出的 evidence line，将该行所有已采样 token × layer × head 共同聚合为 mean/max/q90/positive fraction。class fallback 会明确标记，不能与精确 record-span 结果混为一谈。

### paired_line_comparisons.csv

每个 (length, seed) 内严格配对 Gold line 与 conflict line，报告：

- 两个 line score；
- Gold-minus-conflict gap；
- strict win；
- tie；
- tie 计 0.5 的 paired win value。

### paired_line_summary.csv

按 length、scope、metric、reducer 汇总：

- paired win rate；
- strict win rate；
- mean / median / q10 / q90 score gap。

### summary.json

记录输入文件、去重数量、line metadata 来源、所有默认协议和 paired summary。

## 可选 LOSO 组合：默认关闭

默认不组合不同指标，避免在同一批 seeds 上选权重或标准化造成 leakage。

如仅用于补充诊断，可显式开启：

    python src/analyze_suppression_certificate_block_aggregation.py \
      --inputs SHARD_GPU6 SHARD_GPU7 \
      --output-dir OUTPUT_DIR \
      --enable-loso-combination

该组合没有训练分类器、没有根据标签选择权重：

- 所有 feature 固定使用相同正权重；
- held-out seed 完全不参与均值和标准差估计；
- 均值和总体标准差只使用其余 training seeds，并且不读取 gold/conflict 标签；
- 输出逐 fold 的 training_seed_ids、held_out_seed 和 uses_labels_for_weights=0。

它仍然只是 oracle 数据上的探索性组合，不应替代单指标 token/line 结果，也不应作为默认主结论。

### 8-seed LOSO 结果

在正式 8-seed safety artifact 上，固定等权、无标签标准化的 leave-one-seed-out 组合仍不稳定：

| 长度 | All sampled 配对胜率 | Decisive-only 配对胜率 |
|---:|---:|---:|
| 8K | 62.5% | 75.0% |
| 32K | 50.0% | 62.5% |
| 64K | 37.5% | 62.5% |

`all_sampled` 在 32K 等于随机、64K 低于随机；`decisive_only` 也只有 5/8 或 6/8 seeds 胜出，没有跨长度稳定性。因此，先聚合 token/head/line 仍不能把 suppression certificate 升格为 gold-vs-conflict 安全门控。后验搜索出的单长度 reducer 只可作为描述性现象，不能用作方法证据。

## 最小验证

    python -m py_compile \
      src/analyze_suppression_certificate_block_aggregation.py

    python -m pytest -q \
      tests/test_suppression_certificate_block_aggregation.py
