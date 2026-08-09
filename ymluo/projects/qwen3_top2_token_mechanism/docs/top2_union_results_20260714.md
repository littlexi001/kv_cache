# Top-2% 跨 Head / 跨层 Union 结果

日期：2026-07-14

## 1. 定义

对每个 eval query：

```text
S[l, h, q] = layer l, attention head h 的历史 token Top-2% 位置集合

layer_union[l, q] = union over h of S[l, h, q]
model_union[q]    = union over l,h of S[l, h, q]
```

当前 self token 不计入集合。历史 token 数约为 4096–4607；单 head 的 `ceil(2%)` 预算为 82–93，平均 87.52。

`temporal union` 另外表示跨 512 个 eval query 曾经被选中过的位置集合：

```text
layer_temporal_union[l] = union over q,h of S[l,h,q]
model_temporal_union    = union over q,l,h of S[l,h,q]
```

运行目录：

```text
remote: /home/fdong/ymluo/projects/qwen3_top2_token_mechanism/outputs/union_war4k_20260714_165630
local:  ymluo/projects/qwen3_top2_token_mechanism/outputs/union_war4k_20260714_165630
```

## 2. 核心结果

### 同一层跨 16 heads

在全部 28 层 × 512 queries 上：

```text
mean layer union tokens          = 548.44
median layer union tokens        = 557
p95 layer union tokens           = 679
mean layer union / history       = 12.60%
mean layer union / one-head 2%   = 6.26x
mean redundant head-token events = 60.84%
mean positions shared by all 16 heads = 2.11
```

若 16 个 heads 完全不重叠，理论 union 上限约为 32% history。实测为 12.6%，说明同层 heads 有明显共享；但 union 仍是单 head 预算的 6.26 倍，所以不同 heads 也保留了大量互补位置。

逐层平均 union 比例范围：

```text
lowest:  layer 17 = 9.05%, 393.5 tokens, 4.50x one-head budget
highest: layer  3 = 15.03%, 654.4 tokens, 7.47x one-head budget
```

### 整个模型跨 28×16 = 448 heads

对每个 query：

```text
mean model union tokens          = 2419.42
median model union tokens        = 2436.5
p95 model union tokens           = 2672.7
range                            = 1894 .. 2900
mean model union / history       = 55.56%
range of union fraction          = 46.23% .. 64.29%
mean model union / one-head 2%   = 27.62x
mean redundant head-token events = 93.83%
```

首个 eval query 和最后一个 eval query：

| Query | History | Single-head budget | Model union | History fraction | Budget multiple |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 4096 | 82 | 2076 | 50.68% | 25.32x |
| 4607 | 4607 | 93 | 2561 | 55.59% | 27.54x |

虽然 448 heads 产生的 head-token 选择事件有 93.8% 重复，但因为原始事件数约为 `448 × 2% = 896% history`，去重后仍覆盖平均 55.6% 的历史位置。

### 跨 512 queries 的 temporal union

```text
model temporal union = 4607 / 4607 = 100%
```

也就是说，在整段 512-token eval 期间，历史中的每一个位置都至少被某一层的某一个 head 选入过 Top-2%。因此，“每步只选 2%”不能推出“整段生成可用一个很小的静态 token 集合”。

单层 temporal union 的范围也很大：

```text
lowest:  layer 5 = 2638 / 4607 = 57.26%
highest: layer 1 = 4481 / 4607 = 97.27%
```

## 3. 完整逐层表

| Layer | Mean union | History % | x single-head budget | Redundant events | Temporal union |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 542.1 | 12.46% | 6.19x | 61.28% | 96.85% |
| 1 | 559.7 | 12.87% | 6.40x | 60.02% | 97.27% |
| 2 | 566.2 | 13.01% | 6.47x | 59.57% | 91.71% |
| 3 | 654.4 | 15.03% | 7.47x | 53.28% | 68.85% |
| 4 | 553.6 | 12.72% | 6.32x | 60.49% | 69.98% |
| 5 | 464.0 | 10.67% | 5.30x | 66.86% | 57.26% |
| 6 | 641.6 | 14.74% | 7.33x | 54.20% | 79.49% |
| 7 | 581.5 | 13.36% | 6.64x | 58.49% | 74.89% |
| 8 | 613.1 | 14.08% | 7.00x | 56.24% | 71.91% |
| 9 | 597.6 | 13.72% | 6.82x | 57.37% | 68.92% |
| 10 | 449.9 | 10.33% | 5.14x | 67.90% | 59.24% |
| 11 | 560.1 | 12.86% | 6.40x | 60.02% | 92.45% |
| 12 | 421.1 | 9.67% | 4.81x | 69.95% | 70.70% |
| 13 | 468.8 | 10.76% | 5.35x | 66.55% | 71.80% |
| 14 | 497.5 | 11.43% | 5.68x | 64.48% | 85.17% |
| 15 | 487.1 | 11.19% | 5.56x | 65.23% | 85.04% |
| 16 | 522.4 | 12.00% | 5.97x | 62.72% | 84.52% |
| 17 | 393.5 | 9.05% | 4.50x | 71.89% | 70.41% |
| 18 | 590.1 | 13.56% | 6.74x | 57.87% | 84.59% |
| 19 | 563.9 | 12.96% | 6.44x | 59.74% | 71.02% |
| 20 | 621.7 | 14.29% | 7.11x | 55.59% | 72.11% |
| 21 | 605.7 | 13.92% | 6.92x | 56.74% | 67.51% |
| 22 | 552.2 | 12.70% | 6.31x | 60.55% | 66.07% |
| 23 | 590.8 | 13.58% | 6.75x | 57.80% | 78.58% |
| 24 | 601.2 | 13.82% | 6.87x | 57.07% | 78.77% |
| 25 | 622.6 | 14.31% | 7.11x | 55.54% | 76.10% |
| 26 | 595.8 | 13.69% | 6.81x | 57.45% | 67.16% |
| 27 | 437.9 | 10.06% | 5.00x | 68.73% | 88.78% |

## 4. 解释

1. **同层 heads 既重叠又互补。** 约 60.8% head-token events 是重复的，但 union 仍从单 head 的 2% 扩张到 12.6%。
2. **少数所有-head 共享位置不能解释全部重叠。** 每层平均只有约 2.1 个位置被 16 heads 全部选中；大量重复来自“部分 head 小组共享”，不是所有 heads 都锁定同一集合。
3. **跨层互补把单步覆盖推到一半以上。** 单层平均 12.6%，整个模型为 55.6%，说明层与层之间又引入了新的位置集合。
4. **Top-2% selector 高度 query-dependent。** 整个 512-token 区间的 model temporal union 达到 100%，静态 union cache 无法直接保持 2% 物理预算。
5. **适合继续做分组共享。** 结果支持研究 head grouping：先找共享度高的 head group，再为不同 group 保留各自候选集；直接全层或全模型统一 union 会把预算扩大到约 12.6% 或 55.6%。

