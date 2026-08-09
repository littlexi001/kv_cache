# Section 143: b16 evidence block 追加实验

日期：2026-07-11

## 目的

这轮实验专门回答一个问题：`block_size=16` 是否真的因为“太碎”而不适合，还是之前只是选中的 block 数不够多。

为了避免被 direct operator、短输出 cap 等其他模块干扰，本轮只跑 6 个 evidence QA 任务：

- `narrativeqa`
- `qasper`
- `multifieldqa_en`
- `hotpotqa`
- `2wikimqa`
- `musique`

这些任务是真正依赖从长上下文中找 evidence block 的部分，因此最适合判断 b16 是否值得继续投入。

## 实验设计

| Run | 配置 | 设计意图 |
|---|---|---|
| v277 | `riskkv_task_policy_v277_b16_manyblocks_qa_20260711.json` | 直接用 16-token block，并把 QA 任务预算放大，测试“多选 b16 block”是否恢复质量。 |
| v278 | `riskkv_task_policy_v278_b16_anchor_window_qa_20260711.json` | 16-token block 只负责定位，命中后额外保留 64/96-token 局部窗口，测试 evidence 是否需要连续上下文。 |
| v279 | `riskkv_task_policy_v279_b16_ultrarecall_qa_20260711.json` | 进一步提高预算，作为 b16 多块方案的质量上界探针。 |

## 当前启动状态

三组实验已在服务器后台启动，均为 LongBench evidence QA 子集 M20：

| Run | GPU | Samples/task | Tasks |
|---|---:|---:|---|
| v277 b16 manyblocks QA | 1 | 20 | 6 个 evidence QA |
| v278 b16 anchor-window QA | 2 | 20 | 6 个 evidence QA |
| v279 b16 ultra-recall QA | 3 | 20 | 6 个 evidence QA |

输出目录：

- `outputs/riskkv_v19_v277_b16_manyblocks_qa_20260711_b16_qa_m20_m20_bDyn_pDyn`
- `outputs/riskkv_v19_v278_b16_anchor_window_qa_20260711_b16_qa_m20_m20_bDyn_pDyn`
- `outputs/riskkv_v19_v279_b16_ultrarecall_qa_20260711_b16_qa_m20_m20_bDyn_pDyn`

## 预期判据

- 如果 v277 接近或超过 v275 在这 6 个 QA 任务上的表现，说明用户的判断成立：b16 本身不是问题，关键是选足够多的细粒度 block。
- 如果 v278 明显优于 v277，说明 b16 更适合作为 locator，最终 KV 保留单元需要局部连续窗口。
- 如果 v279 仍然明显低于 v275，说明问题不是 block 太少，而是当前 scorer 对细粒度 evidence composition 的排序不够可靠。
- 如果 v279 质量很好但 KV ratio 偏高，下一步应该训练 sample-level budget router，只在高风险样本上启用 b16 ultra-recall。

## 完整结果

三组 M20 evidence QA 子集均已完成：

| Run | Score | KV keep | Online | Total | 结论 |
|---|---:|---:|---:|---:|---|
| v277 b16 many-blocks | 0.2790 | 29.74% | 0.461s | 2.256s | 只增加 b16 block 数量不能恢复质量。 |
| v278 b16 locator + anchor window | 0.3105 | 30.21% | 0.460s | 2.274s | 局部连续窗口能救回一部分 2Wiki/Musique，但总体仍低。 |
| v279 b16 ultra-recall | 0.2915 | 37.67% | 0.445s | 2.257s | 提高预算后质量仍没有上来，KV 成本更高。 |

为了避免 M20/M100 抽样差异造成误判，又用 v277 的同一批 `sample_id` 去过滤 v275 M100 `task_results.csv`，得到同样本对齐比较：

| Run | Samples | Score | KV keep | Online | Total |
|---|---:|---:|---:|---:|---:|
| v275 same IDs | 120 | 0.3833 | 42.04% | 0.376s | 2.226s |
| v277 b16 many-blocks | 120 | 0.2790 | 29.74% | 0.461s | 2.256s |
| v278 b16 anchor-window | 120 | 0.3105 | 30.21% | 0.460s | 2.274s |
| v279 b16 ultra-recall | 120 | 0.2915 | 37.67% | 0.445s | 2.257s |

同样本任务级对齐结果：

| Task | v275 score | v277 score | v278 score | v279 score | 观察 |
|---|---:|---:|---:|---:|---|
| narrativeqa | 0.2538 | 0.1154 | 0.1158 | 0.1477 | b16 对叙事类长证据明显不稳。 |
| qasper | 0.5599 | 0.4457 | 0.4424 | 0.4504 | b16 增预算后仍低于主线。 |
| multifieldqa_en | 0.5618 | 0.4129 | 0.3924 | 0.4207 | 细 block 导致答案生成更容易偏离。 |
| hotpotqa | 0.2558 | 0.2458 | 0.2602 | 0.2658 | hotpot 上差距很小，b16 可作为局部 ablation。 |
| 2wikimqa | 0.3687 | 0.2708 | 0.4187 | 0.2720 | anchor window 对 2Wiki 有帮助，但不稳定。 |
| musique | 0.3000 | 0.1833 | 0.2333 | 0.1924 | musique 仍需要更可靠的 multi-hop evidence composition。 |

## 结论

这轮实验基本否定了“只要 b16 多选一些 block 就能变好”的假设。更细的 block 带来了更低的 KV ratio，但当前 scorer 在 16-token 粒度下排序噪声更大，尤其在 NarrativeQA、Qasper、MultiFieldQA、Musique 上丢 evidence 或上下文语义断裂。

更重要的是，v279 作为高预算上界仍然没有接近 v275：它保留了 37.67% KV，但 score 只有 0.2915，低于 v275 同样本的 0.3833。这说明瓶颈不是“选得不够多”，而是 b16 粒度下的 evidence composition 和排序目标不匹配。

下一步建议：

- 主线继续使用 v275 这一类 task-conditioned memory-action router，不把 b16 多块作为主方法。
- b16 可以保留为 ablation 或辅助 locator：在 HotpotQA / 2Wiki 这类局部实体跳转任务上，它有局部正信号。
- 如果继续探索 b16，应该改 scorer，而不是继续盲目加预算：例如训练 query-evidence pair scorer、做 block-to-span repack，或者用 b16 先定位再合并成 128/256-token evidence span。
