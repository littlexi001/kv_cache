# Section 109: m20 Held-out 验证与 Calibrated Floor 消融（2026-07-08）

## 目标

Section 108 已经把 `floor_v2` 形式化为 calibrated risk floor。
本节启动更大的 m20 held-out 验证，目标是判断：

```text
1. calibrated floor 是否在 m20 上仍保持 m10 的稳定性；
2. free small-block router 的失败是否随样本数变大继续存在；
3. floor-only、router+floor、固定 block/topK baselines 的贡献分别是什么；
4. 是否值得直接进入 m50/full。
```

## 代码更新

正式方法名已经接入 runner：

```text
router_blocksize_calibrated
riskkv_block_calibrated
```

旧名字保留兼容：

```text
router_blocksize_floor_v2
```

新增 floor-only ablation：

```text
blocksize_calibrated_floor_only
riskkv_block_floor_only
```

对应代码：

```text
ymluo/projects/learned_hierarchical_summary_memory/src/run_qwen8b_paper_benchmarks.py
```

新增 held-out 脚本：

```text
ymluo/projects/learned_hierarchical_summary_memory/scripts/run_qwen8b_blocksize_calibrated_heldout_20260708.sh
```

脚本支持：

```text
MAX_EXAMPLES_PER_TASK
METHODS
RUN_TAG
GPU_LIST
GPU_LIST_CSV
```

## m20 主实验

启动时间：

```text
2026-07-08 03:09:26 +08:00
```

服务器 PID：

```text
742292
```

使用 GPU：

```text
0,1,2,3
```

运行设置：

```text
RUN_TAG=blocksize_calibrated_m20_20260708
MAX_EXAMPLES_PER_TASK=20
METHODS=full_raw,router_blocksize,router_blocksize_calibrated
```

输出目录：

```text
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_blocksize_calibrated_m20_20260708_longbench
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_blocksize_calibrated_m20_20260708_ruler4k
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_blocksize_calibrated_m20_20260708_ruler8k
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_blocksize_calibrated_m20_20260708_ruler16k
```

这个实验直接回答：

```text
free router vs calibrated router
```

如果 m20 上 `router_blocksize_calibrated` 仍然接近或超过 full_raw，并显著优于 `router_blocksize`，说明 calibrated floor 不是 m10 偶然结果。

## m20 消融实验

第一次启动因为 GPU_LIST 空格被 shell 拆开失败，已修复脚本支持 `GPU_LIST_CSV`。

正式启动时间：

```text
2026-07-08 03:11:50 +08:00
```

服务器 PID：

```text
745704
```

使用 GPU：

```text
4,5,6,7
```

运行设置：

```text
RUN_TAG=blocksize_calibrated_ablation_m20_20260708
MAX_EXAMPLES_PER_TASK=20
METHODS=full_raw,
        blocksize_calibrated_floor_only,
        recent_plus_b128_span_top12_b0_a0,
        recent_plus_b256_span_top3_b0_a0,
        recent_plus_b512_span_top3_b0_a0
```

输出目录：

```text
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_blocksize_calibrated_ablation_m20_20260708_longbench
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_blocksize_calibrated_ablation_m20_20260708_ruler4k
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_blocksize_calibrated_ablation_m20_20260708_ruler8k
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_blocksize_calibrated_ablation_m20_20260708_ruler16k
```

这个实验回答：

```text
1. floor-only 是否已经足够；
2. router+floor 是否比 floor-only 更省 token 或更高分；
3. 固定 b128/b256/b512 是否能解释全部收益；
4. calibrated floor 是否只是固定 topK baseline 的重命名。
```

## 结果判定标准

进入 m50/full 的 gate：

```text
1. calibrated m20 的 RULER 4k/8k/16k score >= full_raw；
2. calibrated m20 的 LongBench score 不低于 full_raw，或至少高于 free router；
3. calibrated token ratio 明显低于 full_raw，最好 <= 25%；
4. ablation 显示 calibrated policy 不是单一固定 block baseline 可以完全替代。
```

如果 m20 失败：

```text
1. 检查失败集中在哪个 task；
2. 对失败 task 单独 calibration；
3. LongBench 可能需要拆成 exact / summary 两个 group；
4. RULER 可能需要按 task type 拆 floor，而不只按 length。
```

## 对论文的意义

如果 m20 结果通过，上主文可以这样说：

```text
The calibrated floor is selected on m3 calibration sweeps and validated on a disjoint m20 held-out suite.
```

这比只报告 m10 更稳，也能抵抗 reviewer 对 sample size 的质疑。

如果 ablation 支持 router+floor 优于 fixed baselines，则创新点更清楚：

```text
RiskKV-Block is not a fixed compression ratio, not a fixed block size, and not a simple retriever.
It is a calibrated action-lattice policy over memory granularity and evidence budget.
```
