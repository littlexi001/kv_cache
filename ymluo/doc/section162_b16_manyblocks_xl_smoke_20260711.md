# Section 162: B16 多块高召回 smoke 实验

日期：2026-07-11

## 背景

本轮实验验证一个更直接的假设：把检索块缩小到 16 tokens 后，不一定要因为“太碎”而失败；如果 token 预算足够，可以多选很多个 16-token 微块，让定位更细，同时保持总 KV token 仍显著低于 full cache。

已有 B16 结果里，`v301/v302` 的 group 版本质量下降明显，说明“只缩小 block size、但预算不够或聚合方式不稳”不是主线；本轮改为更高召回 smoke，先在 LongBench QA 任务上用 M20 判断是否值得放大。

## 方法

新增两个 practical 配置，不使用 oracle。

### v320: B16 many-block XL

- QA 任务统一使用 `page_tokens=16`。
- 相比之前 B16 high-recall，进一步提高 token budget：
  - `narrativeqa`: 2048
  - `qasper`: 3072
  - `multifieldqa_en`: 2048
  - `hotpotqa`: 4096
  - `2wikimqa`: 3072
  - `musique`: 4096
- `qasper/musique` 保留 bridge scorer。
- 不启用 full fallback，不用 oracle。

目标：直接测试“16-token 块 + 多选块”本身是否能把 LongBench QA 拉回来。

### v321: B16 many-block + window64

在 v320 的预算上加入 `span_repack`：

- 微块仍用 `page_tokens=16` 做定位。
- 对高分微块周围回填 64-token 局部窗口。
- 使用 `span_repack_score_mode=window_sum`，用窗口内微块分数聚合排序。

目标：测试如果纯 B16 太碎，少量连续窗口回填是否能改善生成型 QA 的证据完整性。

## 运行状态

新增/同步文件：

- `configs/riskkv_task_policy_v320_b16_manyblocks_xl_smoke_20260711.json`
- `configs/riskkv_task_policy_v321_b16_manyblocks_window64_smoke_20260711.json`
- `scripts/launch_b16_manyblocks_xl_smoke_20260711.sh`
- `scripts/watch_combine_b16_manyblocks_xl_smoke_20260711.sh`

启动命令使用：

```bash
nohup env GPUS=0,2,3,6 SAMPLES=20 GPU_MAX_USED_MB=3000 GPU_MAX_UTIL=25 \
  bash scripts/launch_b16_manyblocks_xl_smoke_20260711.sh \
  > outputs/logs/launch_b16_manyblocks_xl_smoke_20260711.log 2>&1 < /dev/null &
```

watcher：

```bash
nohup bash scripts/watch_combine_b16_manyblocks_xl_smoke_20260711.sh \
  > outputs/logs/watch_combine_b16_manyblocks_xl_smoke_20260711.log 2>&1 < /dev/null &
```

汇总目录：

```text
outputs/riskkv_v19_v320_v321_b16_manyblocks_xl_smoke_20260711/
```

截至启动后检查，`v321_b16_manyblocks_window64_qasper` 已进入 Python 运行，其余任务排队等待空闲 GPU。watcher 会在 12 个任务全部完成后自动生成 `summary_table.csv` 和 `detail_table.csv`。

## 判定标准

优先看同样 M20 samples 下相对 full/v300 的分数：

- 如果 v320 明显优于之前 B16 group/purefine，说明“多选 16-token 块”有价值。
- 如果 v321 优于 v320，说明碎片化确实需要局部连续性修正。
- 如果两者仍低于 v300，B16 不应作为 LongBench QA 主线，只能作为 RULER/检索型或局部任务的加速模块。

本轮不是最终主实验，只用于判断是否把 B16 继续放大到 M100。
