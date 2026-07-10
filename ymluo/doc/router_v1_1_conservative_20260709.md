# LongBench 动态 Router v1.1 训练记录

日期：2026-07-09

## 结论

已经训练出一个可用的保守版 LongBench 动态 router，命名为：

`longbench_dynamic_router_v1_1_conservative`

它不是固定预算方法，而是三头输出：

1. `page_router`：选择 `page_tokens / block_size`
2. `budget_router`：选择最小安全预算
3. `fallback_router`：判断是否需要 full KV fallback

## 为什么不用 v1

最初的 v1 使用逐样本 pseudo-oracle 标签：

`label = 与该样本最好分数接近的最小 budget`

这个标签在 LongBench 上噪声很大。很多样本在所有 sparse budget 下都得 0 分，于是会被标成 `B=128`，导致 router 学得过于激进。

v1 的 held-out 模拟结果：

- page router 预测策略分数：0.2602
- budget router 预测策略分数：0.2816
- 固定 B=512 分数：0.3092
- 固定 B=1024 分数：0.3309

因此 v1 不能作为主 router。

## v1.1 的标签策略

v1.1 改用任务级平均收益做校准标签：

- 对每个 LongBench task，选择平均分最高或接近最高的 block size。
- 对每个 LongBench task，选择平均分接近最高的最小安全 budget。
- 对高风险任务单独标记 fallback。

当前 fallback 任务：

- `passage_count`
- `passage_retrieval_en`

## v1.1 学到的策略

### Page / Block Size

| Task | page_tokens |
|---|---:|
| 2wikimqa | 128 |
| gov_report | 32 |
| hotpotqa | 32 |
| lcc | 256 |
| multi_news | 32 |
| multifieldqa_en | 32 |
| musique | 32 |
| narrativeqa | 64 |
| passage_count | 32 |
| passage_retrieval_en | 256 |
| qasper | 512 |
| qmsum | 32 |
| repobench-p | 64 |
| samsum | 128 |
| trec | 64 |
| triviaqa | 32 |

### Budget

| Task | budget |
|---|---:|
| 2wikimqa | 512 |
| gov_report | 1024 |
| hotpotqa | 1024 |
| lcc | 1024 |
| multi_news | 512 |
| multifieldqa_en | 1024 |
| musique | 1024 |
| narrativeqa | 1024 |
| passage_count | 1024 + fallback |
| passage_retrieval_en | 1024 + fallback |
| qasper | 1024 |
| qmsum | 512 |
| repobench-p | 1024 |
| samsum | 128 |
| trec | 1024 |
| triviaqa | 128 |

## 离线指标

训练输出：

`/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longbench_dynamic_router_v1_1_conservative_20260709`

模型文件：

`longbench_dynamic_router_v1_1_conservative.joblib`

离线指标：

- page router accuracy：0.9875
- budget router accuracy：1.0000
- fallback router accuracy：1.0000
- page policy simulated score：0.2979
- page oracle score：0.3455
- sparse budget policy simulated score：0.3499
- sparse budget oracle score：0.4088
- mean sparse budget：825.3
- fallback rate：10.67%

## 当前状态

已经把 v1.1 policy 评测排队到服务器空闲 GPU 上。

已经在跑的 v1.1 分组包括：

- `p256_b1024`: `lcc`
- `p128_b512`: `2wikimqa`
- `p32_b512`: `multi_news,qmsum`

剩余 v1.1 分组已经设置为等待 GPU 空闲后自动启动：

- `p32_b1024`: `gov_report,hotpotqa,multifieldqa_en,musique`
- `p64_b1024`: `narrativeqa,trec,repobench-p`
- `p512_b1024`: `qasper`
- `p128_b128`: `samsum`
- `p32_b128`: `triviaqa`
- full KV fallback: `passage_count`
- full KV fallback: `passage_retrieval_en`

## 解释

v1.1 不是最终形态，但它已经符合我们的方法定义：

`根据任务风险动态选择 block size、budget 和 fallback`

下一步应该在 v1.1 跑完后，把真实 LongBench 结果和固定预算 Table5 结果合并，形成：

1. fixed-budget baseline
2. task-calibrated dynamic planner
3. learned router v1.1
4. router + fallback

如果 v1.1 明显优于固定预算，后续再加入 retriever gap、top-k stability、query feature，训练样本级 router v2。
