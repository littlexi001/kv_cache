# 2026-07-11：block size 复盘与全局 PassageRetrieval head

## 目标

继续围绕 ICLR 主线优化 RiskKV-Block，目标是：

- KV keep 在 10%~30%；
- online speed >= 2.5x；
- score >= full baseline 的 95%；
- 优先利用实验现象设计新方案，而不是盲目扫参数。

## 小 block 实验结论

基于 v200 task-gated layer33，测试了固定 token budget 下的更小 block：

| 实验 | block | 设计 | Score m20 | KV keep | Online | 结论 |
|---|---:|---|---:|---:|---:|---|
| v200 | 128 | 原 task-gated layer33 | 0.3740 | 27.70% | 1.131s | m20 很强，但 m100 不稳 |
| v201 | 64 | 小 block + scaled neighborhood | 0.3599 | 29.01% | 1.137s | 可用，但没有赢 v200 |
| v202 | 32 | 小 block + scaled neighborhood | 0.3356 | 28.88% | 1.126s | 明显掉分 |
| v203 | 16 | 小 block + scaled neighborhood | 0.3330 | 29.01% | 1.136s | 明显掉分 |
| v204 | 16 | 小 block + stronger span merge | 0.3352 | 29.34% | 1.137s | span merge 没修好 |

核心现象：

1. 小 block 本身不是 KV cache 层面的理论问题。
2. 直接用 16/32 block 做全局打分会让单块 lexical/IDF 信号变噪，QA 和多跳任务明显掉分。
3. 16 block 的 page 数约变为 8 倍，检索/scoring 开销显著增加；如果质量没有明显收益，不适合作主线。

## coarse=128 小 block 修正实验

上面 v201-v204 的一个缺陷是把 `multiscale_group_pages` 按 block size 放大后，b16 的 coarse group 变成 512 tokens，而不是 128 tokens。已启动修正版：

| 实验 | block | coarse group | multiscale weight | 状态 |
|---|---:|---:|---:|---|
| v209 | 64 | 128 tokens | 0.50 | m20 finished |
| v210 | 32 | 128 tokens | 0.50 | m20 finished |
| v211 | 16 | 128 tokens | 0.50 | running |

已完成结果：

| 实验 | Score m20 | KV keep | Online | 结论 |
|---|---:|---:|---:|---|
| v209_b64_coarse128 | 0.3504 | 28.96% | 1.216s | 不如 v200/v201 |
| v210_b32_coarse128 | 0.3426 | 28.48% | 1.208s | 不如 v200 |

当前判断：单纯换小 block 不是主突破口。更合理的长期方向是 coarse-to-fine selector：先用 128 稳定定位候选区域，再在候选区域内部做 16/32 token 级精裁，而不是所有 block 全局竞争。

## m100 主线实验

已知 m100 baseline：

- full m100 score = 0.365817；
- 95% threshold = 0.347526。

已有 m100：

| 实验 | Score | KV keep | Online | 判断 |
|---|---:|---:|---:|---|
| v194_hotpot_safe | 0.347161 | 31.41% | 1.338s | 分数只差 0.00036 过线，但 KV 超 30% |
| v200_taskgated_layer33 | 0.341057 | 26.66% | 1.184s | KV 很好，但 m100 分数不稳 |
| v193_rerun | 0.342707 | 28.66% | 1.192s | 分数不够 |

基于现象启动的新 m100：

| 实验 | 设计 | 目的 |
|---|---|---|
| v191_hotpot3072_m100 | Hotpot 3072 中间预算，m50 曾有 0.358985 / 29.95% KV | 复验最可能同时过 95% 和 KV<30 的版本 |
| v205_v194_lowrisk_layer33_m100 | v194 质量安全底座 + 仅 TREC/PassageRetrieval/SAMSum/LCC 开 layer33 | 保持 v194 质量，略降 KV |
| v206_v191_lowrisk_layer33_m100 | v191 + 低风险 layer33 | 在 v191 基础上进一步降 KV |
| v207_v191_gov_lowrisk_layer33_m100 | v191 + GovReport + 低风险 layer33 | 尝试额外降 KV，观察 GovReport 掉分 |
| v208_v191_summary_lowrisk_layer33_m100 | v191 + GovReport/QMSum + 低风险 layer33 | 更激进但可能伤摘要 |

这些不是盲扫参数，而是围绕两个观察设计：

- v194 的 Hotpot 安全策略让质量接近过线，但 KV 高；
- v200 的 task-gated layer-wise 能降 KV，但不能碰 QA/多跳核心任务。

## 新发现：PassageRetrieval 是大突破口

m100 分项中，当前 `passage_retrieval_en` 只有约 0.65，而该任务本质是结构化检索，full/AdaKV 通常接近满分。现在旧 direct head 的问题是只在已选 block 中找 `Paragraph k`，如果正确段落没有被 sparse selector 选中，就会直接错。

已实现新 head：

- 解析全上下文所有 `Paragraph k`；
- 对 abstract 和每个 paragraph 做 BM25/IDF overlap；
- 加 entity overlap、number overlap、长词 overlap；
- 直接返回最高分 `Paragraph k`；
- 不进入长 KV decode，不增加 KV cache。

本地 synthetic 测试已通过：

- 输入 3 个 paragraph；
- abstract 描述 NASA infrared telescope；
- 返回 `Paragraph 2`；
- 耗时约 0.3 ms。

等待服务器 SSH 恢复后需要立刻做：

1. 同步 `run_controlled_public_kv_benchmark_v1.py`。
2. 先跑 `passage_retrieval_en m100` smoke，验证是否从 0.65 接近 1.0。
3. 如果成立，扩展到当前最好候选 v191/v194/v205/v206 的 full m100。

预期收益：

- 如果 PassageRetrieval 从 0.65 提到 0.95，总分可提升约 `(0.30 / 16) = 0.01875`；
- 这会直接把 v191/v194 系列推过 m100 95% 阈值；
- KV 几乎不增加，因为答案由结构化 head 直接给出。

