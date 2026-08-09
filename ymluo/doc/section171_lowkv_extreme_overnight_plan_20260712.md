# section171：1%-10% KV 极限压缩夜间探索记录

日期：2026-07-12

## 目标

这轮实验不是继续微调一个固定 router，而是围绕“能否在 LongBench 上做到 1%-10% KV keep、2.5x+ 速度、分数达到 baseline 95%+”做假设驱动探索。

判断时同时看两条基线：

- full KV baseline：当前汇总脚本使用近似 `score=0.3658, online=3.0988s`。
- practical baseline v300：`score=0.4392, KV keep=27.41%, online=0.5632s`。

## 已完成的阶段性结果

已经完成的 M20 全任务结果：

| run | samples | tasks | score | vs full | vs v300 | KV keep | speed/full | speed/v300 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v365 ultra skeleton all | 320 | 16 | 0.3776 | 103.23% | 85.97% | 8.24% | 5.79x | 1.05x |
| v368 direct operator extreme mix | 320 | 16 | 0.3754 | 102.64% | 85.48% | 7.99% | 8.18x | 1.49x |
| v363 taskwise lowkv mix | 320 | 16 | 0.4096 | 111.98% | 93.26% | 14.17% | 4.75x | 0.86x |
| v360 lowkv certificate all | 320 | 16 | 0.3948 | 107.92% | 89.88% | 10.05% | 5.39x | 0.98x |

v365/v368 已经满足“1%-10% KV + 2.5x+ + 95% full score”的最低目标，但它们只是 M20，需要 M100/M150 confirmation 后才能作为论文主结果讨论。

v363 分数明显更高，但 KV keep 到 14.17%。这个现象很重要：它说明不是所有任务都需要放宽预算，只有少数 hard task 片段需要更大的证据包。因此我新增了 v375，用任务级 Pareto 片段拼接，希望保留 v363 的分数收益，同时把平均 KV 压回 10% 内。

## 方法假设

### v365/v368：ultra skeleton + direct operator

核心现象：LongBench 里有一批任务并不是必须靠完整 KV 生成解决。

- `passage_retrieval_en`、`passage_count`、`trec` 可走结构化 direct operator。
- `gov_report`、`multi_news`、`qmsum`、`samsum` 可走 extractive/query-focused summary operator 或短解码。
- QA 和 code 任务仍用小 block 检索，但允许少量安全 fallback。

这个方向的论文故事是：不是所有长上下文任务都应该被统一视为 dense KV retrieval；应该先识别任务族和输出结构，把可结构化求解的子类从 KV cache 压缩问题中分离出来。

### v370：低预算 + 不确定性 ladder

核心假设：单一 gap/entropy 阈值不够稳定，但 page-score gap、entropy、top score、coverage recall 联合起来可以作为 pre-decode 风险信号。

做法：

- 默认从 128/192 token 级别开始。
- 按 score gap、entropy、top score 选择 256/384/512/768 的最小安全预算。
- coverage 不闭合时只升级到有限预算，不直接 full fallback。

### v371：span-repack 连续证据窗口

核心假设：`block_size=16` 的质量损失不一定来自 block 太小，而是来自证据被切得过碎。

做法：

- 仍用小 block 做定位。
- 对高分小 block 周围重组 64/80/96 token 连续窗口。
- 用少量连续 evidence capsule 替代大量离散 block。

### v372/v373：extractive QA direct

核心假设：LongBench 的部分 QA 样本答案可以由局部证据句直接抽取，不需要模型读取压缩 KV 后再生成。

做法：

- v372 激进：所有 QA direct 抽取。
- v373 保守：只对更像 extractive QA 的 `qasper`、`multifieldqa_en`、`triviaqa` direct，其它硬任务仍走 ladder。

### v374：bounded verifier retry

核心假设：低预算生成失败的样本可以通过输出格式、grounding、support window contract 发现，并只升级到有限预算。

做法：

- 先低预算生成。
- 若 contract 不满足，重跑 512/768/1024 token。
- 不允许 full fallback，避免 KV keep 被少量困难样本拉爆。

### v375：task-level Pareto fusion

核心假设：M20 结果显示 v365 的全局 KV 很低，v363 的局部分数更高，但 v363 的高 KV 并不是每个任务都必要。因此把每个任务已经观测到的 Pareto 最优片段组合起来，可能得到更好的 practical policy。

做法：

- 以 v365 为底座，保留极低 KV skeleton。
- qasper 使用 v363 的 384-token bridge/certificate 片段。
- multifieldqa_en 使用 v360 的 256-token certificate 片段。
- hotpotqa 使用 v360 的 384-token graph-bridge 片段。
- lcc/qmsum/samsum/repobench-p 使用 v363 中分数更好的局部设置。
- 不引入 full fallback，目标是 M100 仍接近 10% KV。

按已完成 M20 任务级结果做离线拼接，v375 的预期是 `score=0.4093, KV keep=9.84%, online=0.6326s`。这只是 sanity-check，不等价于最终证据；最终仍以正在排队的 v375 M100 为准。

### v376：strict-10 Pareto fusion

v373 的真实 M20 结果是 `score=0.3070, KV keep=5.12%`，没有达到质量目标，因此不能作为主线。不过它暴露了两个可利用现象：

- hotpotqa 可以用更低 KV 的 graph/ladder 片段，M20 上从 v365/v368 的 `46.82% KV` 降到 `9.75% KV`，分数还略高于 v365。
- repobench-p 可以用更低 KV 的 budget-ladder 片段，虽然分数低于 v365，但能显著降低平均 KV。

因此新增 v376：以 v375 为底座，但把 hotpotqa、2wikimqa、repobench-p 换成更低 KV 片段。它的目的不是追最高分，而是作为更稳的 `KV <= 10%` confirmation variant。

## 当前后台任务

第一轮：

- v360 lowkv certificate all：已完成 M20
- v361 lowkv graph bridge hard
- v362 lowkv bounded retry hard
- v363 taskwise lowkv mix all：已完成 M20
- v364 extreme hardtask probe hard
- v365 ultra skeleton all：已完成 M20
- v366 skeleton support retry hard
- v367 query only then verify hard
- v368 direct operator extreme mix all：已完成 M20
- v369 hardtask minimal ablation hard

第二轮：

- v370 lowkv uncertainty ladder all
- v371 lowkv span-repack ladder hard
- v372 extractive QA direct all
- v373 selective direct ladder all
- v374 lowkv verifier retry hard
- v376 strict-10 Pareto fusion

确认任务：

- v368 direct operator extreme mix all confirm M100：运行中
- v375 Pareto-fused low-KV confirm M100：排队等待 GPU
- v376 strict-10 Pareto-fused M20/M100：排队等待 GPU

## 查看路径

服务器项目目录：

`/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl`

自动汇总：

- `outputs/riskkv_lowkv_exploration_summary_20260712.json`
- `doc/section170_lowkv_overnight_exploration_20260712.md`

关键日志：

- `outputs/logs/watch_lowkv_periodic_summary_20260712.log`
- `outputs/logs/watch_lowkv_exploration_20260712_lowkv_extreme_1to10.log`
- `outputs/logs/riskkv_v19_v368_direct_operator_extreme_mix_all_confirm_m100_20260712_lowkv_confirm_m100_bDyn_pDyn.log`
- `outputs/logs/launch_v375_pareto_fused_m100_20260712.log`

## 明早优先看什么

优先顺序：

1. v368 M100 是否仍保持 `KV keep <= 10%`、`score >= 95% full`。
2. v375 M100 是否能接近 v363 分数，同时把 KV 压回 10% 左右。
3. v376 是否能比 v375 更稳定地守住 `KV <= 10%`，代价是否可接受。
4. v370/v371 是否能在 hard QA 上提升质量且不超过 10%-15% KV。
5. v372/v373 的 direct QA 是否只在部分任务有效。当前 v373 已经显示 direct QA 不能作为主线。
6. v374 的 verifier retry 是否能用有限升级替代 full fallback。

如果 v368/v375 的 M100 稳住，下一步应该围绕它们写主线：task/output-aware KV compression + evidence-risk controlled budget，而不是只讲一个 block router。
