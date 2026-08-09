# Section 150: b16 纯细粒度多块选择实验

日期：2026-07-11

## 背景

用户提出一个合理反驳：`block_size=16` 未必“太碎”，因为 token 本来就是逐个处理；如果预算按 token 控制，16-token block 会自动对应更多候选块，例如 1536 token 预算约等于 96 个小块。

之前的 b16 实验多数仍带有 `multiscale` 粗粒度平滑，`ours_multiscale_group_pages=8/4/2` 分别对应 128/64/32 token 的粗组支持。因此这轮新增一个更干净的对照：保留 b16 和高召回预算，但去掉 multiscale coarse 平滑，让排序直接发生在 16-token fine block 上。

## 新增实验

| Version | 设计 | 目的 |
|---|---|---|
| v307 | b16 pure-fine high-recall，多块选择，不启用 multiscale coarse group | 直接验证“选更多 16-token 小块”是否能恢复 QA 质量 |
| v308 | b16 pure-fine high-recall + anchor window | 验证如果 v307 失败，原因是否是缺少连续上下文，而不是细块定位本身 |

测试任务只覆盖 evidence QA 子集：

```text
narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique
```

每个任务 M100，完成后会与 `v300` 的 non-QA 任务合并，生成完整 LongBench-style 汇总。

## 输出位置

启动脚本：

```bash
scripts/launch_b16_purefine_sweep_20260711.sh
```

自动汇总脚本：

```bash
scripts/watch_combine_b16_purefine_sweep_20260711.sh
```

最终汇总表：

```text
outputs/riskkv_v19_v307_v308_b16_purefine_sweep_20260711/summary_table.csv
```

## 判据

- 如果 v307 明显优于 v301/v302/v300，说明 b16 之前主要被粗粒度平滑限制，可以把 b16 pure-fine 作为主线继续做 router。
- 如果 v308 优于 v307 但仍低于 v300，说明 b16 更适合作为 locator，最终 KV 保留应回到连续 span/window。
- 如果 v307/v308 都低于 v300，说明当前 scorer 在 16-token 粒度下证据组合不稳定；b16 应作为 ablation 或局部任务特例，而不是主方法核心。
