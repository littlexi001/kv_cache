# 实验设计

## 冻结契约

- 候选预算：`min(N, 1280, max(256, ceil(0.06N)))`；
- request-local QK-balanced + query-weighted qMSE/OAS；
- 240 bit/token/KV-head Key 索引；
- sampled-quantile 配置目标为 64 个尾部锚点，实际最多扫描 512 个样本；
- 不做 exact-QK rerank；
- rank-16、block-256、INT4 ValueSketch，`alpha=0.5`；
- 无 router、任务规则、长度切换和 Full fallback。

## RULER

- 模型：Llama-3.1-8B-Instruct；
- 13 个官方任务；
- 4K/8K/16K/32K 每个 task-length 10 条；
- 64K/128K 每个 task-length 5 条；
- 共 650 个严格 Full KV--Robust pair，1,300 行；
- 数据由固定的 lm-evaluation-harness commit
  `8c05cfe04fafcdd41dd64019f2b3797ef54dcd81` 和 seed 42 生成；
- 主指标为 task-length macro score、相对 Full、最差 cell，以及 10,000 次
  task-length bootstrap 区间。

## 跨模型 LongBench

- 模型：Llama-3.1-8B、Qwen3-4B、Mistral-7B；
- 每个任务先跳过 40 条，再固定取 10 条；
- 每个模型 16 个任务、160 个严格 pair；
- 使用官方 middle truncation、7,500 token 上限和模型对应 chat template；
- 主指标为 macro score、相对 Full、分任务最差值和 task bootstrap 区间。

这批 160 样本只承担跨模型转移检查。Llama 的完整 3,750 样本 Robust
LongBench 仍需单独完成，不能由该小表替代。

## 运行入口

```bash
bash scripts/launch_qksieve_robust_ruler_20260810.sh
bash scripts/launch_qksieve_robust_multimodel_longbench_20260810.sh
```

两个入口都支持在保留有效 CSV 的情况下续跑，并在汇总前执行冻结契约审计。
