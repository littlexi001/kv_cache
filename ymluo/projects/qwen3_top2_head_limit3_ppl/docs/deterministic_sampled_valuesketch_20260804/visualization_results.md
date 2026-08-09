# 确定性采样 QKSieve：实验结果

## 1. 实验设置

- GPU：RTX 3090 24 GB。
- 长上下文模型：Llama-3.1-8B-Instruct；32K 用 1 卡，64K/128K 用 2 卡。每个长度内部的 Full 与稀疏方法使用相同设备布局。
- 长度：32,768、65,536、131,040 个历史 token。
- 质量：held-out `mixed_b` 流上的 32 个 teacher-forced token。
- 质量保持率：`exp(NLL_full - NLL_sparse)`；超过 100% 只表示这 32 个目标 token 上 NLL 略低，不代表完整任务超过 Full。
- 稳态速度：索引建成后整个模型一次 decode forward 的平均时间。一次性建索引成本单列。
- 默认方法：`qksieve_qmse_oas_requestlocal_valuesketch16_sorted_c64_k1280`，无 Full Attention 回退，无任务标签，无学习型 router。

## 2. CUDA 等价性与确定性

以下为单个 attention 层中“采样阈值、全历史代理扫描、候选压缩和 Value 尾项统计”的独立 CUDA Event 测量。`atomic/deterministic` 大于 1 表示新确定性内核更快。

| 历史长度 | c | 样本数 | Atomic ms | v40 deterministic ms | Atomic / v40 | workspace / 层 |
|---:|---:|---:|---:|---:|---:|---:|
| 32K | 64 | 1,792 | 0.1538 | 0.1599 | 0.962x | 0.406 MiB |
| 64K | 64 | 3,328 | 0.3063 | 0.2767 | 1.107x | 0.813 MiB |
| 128K | 64 | 6,656 | 0.6126 | 0.4666 | 1.313x | 1.625 MiB |

正确性结果：

- 三个长度的候选集合与原 atomic 路径完全相同，阈值最大绝对误差为 0。
- 相同输入重复 20 次，候选和尾项张量 mismatch 均为 0。
- 128K 的尾分母相对误差为 `4.12e-7`，尾系数相对误差为 `6.23e-7`；差异来自与 atomic 版不同的浮点求和顺序。
- 128K c64 的 v38 确定性内核为 0.6354 ms，v40 为 0.4666 ms，v40 快 `1.362x`，延迟下降 `26.6%`。

## 3. 真实模型质量与稳态 decode

| 历史长度 | 实际 token/head 均值 | 实际比例 | Full ms/token | QKSieve ms/token | Decode 加速 | 质量保持 | Top-1 | 固定成本 | 回本步数 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32K | 1,276.17 | 3.895% | 87.46 | 45.52 | 1.921x | 100.812% | 100% | 3.17 s | 75.7 |
| 64K | 1,276.58 | 1.948% | 151.22 | 62.77 | 2.409x | 101.358% | 100% | 3.35 s | 37.8 |
| 128K | 1,278.86 | 0.976% | 277.64 | 89.89 | 3.089x | 100.526% | 100% | 3.58 s | 19.0 |

补充统计：

- 32K/64K/128K 的 `KL(Full || sparse)` 分别为 `6.40e-4`、`2.52e-3`、`7.78e-4`。
- c64 实际候选范围分别为 `[815, 2062]`、`[817, 1962]`、`[728, 1825]`。目标均值接近 1,280，但阈值法不强制每个 head 恰好选择 1,280 个 token。
- 辅助索引约为完整 FP16 KV 的 `7.4%`。当前实现仍保留 GPU 常驻原始 FP16 K/V，因此该比例是额外索引，不是总 KV 占用。

## 4. c32 与 c64 的取舍

| 长度 | 版本 | 样本数 | 候选 min/max | Decode 加速 | 质量保持 | Top-1 |
|---:|---|---:|---:|---:|---:|---:|
| 64K | c32 | 1,792 | 689 / 2,295 | 2.426x | 100.986% | 100% |
| 64K | c64 | 3,328 | 817 / 1,962 | 2.409x | 101.358% | 100% |
| 128K | c32 | 3,328 | 654 / 2,291 | 3.129x | 100.687% | 100% |
| 128K | c64 | 6,656 | 728 / 1,825 | 3.089x | 100.526% | 100% |

c32 在 64K/128K 仅快约 `0.7%/1.3%`，但候选数量尾部明显更宽。Qwen3-4B 的 32K medicine probe 中，固定 s1024 为 `99.158%` 质量、`96.875%` top-1；c64 为 `99.694%`、`100%` top-1。两次独立 sports 进程的候选均值差从 s1024 的 `0.067 token/head` 降到 c64 的 `0.006 token/head`。因此 c64 作为稳健默认，c32 只保留为速度预设。

## 5. 分阶段 profile

profile 使用 8 个目标 token；CUDA Event 本身会增加少量开销，因此整步速度仍以第 3 节的无 profile 结果为准。

| 历史长度 | Query 准备 | 新 Key 追加 | 检索/压缩 | 稀疏精确 attention | ValueSketch 增量维护 | 已测阶段合计 | 模型其余部分 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64K | 0.204 ms | 0.270 ms | 5.152 ms | 12.479 ms | 4.053 ms | 22.159 ms | 44.041 ms |
| 128K | 0.203 ms | 0.268 ms | 7.088 ms | 20.608 ms | 1.454 ms | 29.622 ms | 62.489 ms |

这里的“模型其余部分”是 profile 整步时间减去五个显式 CUDA 阶段，包含 Q/K/V/O 投影、MLP、归一化、残差和 LM head。它不是一个独立 kernel 测量。ValueSketch 增量维护只有 6 个稳态样本，64K/128K 的差异不应解释成长度缩放规律。

## 6. 当前可以支持的结论

1. 新内核在候选集合不变的前提下消除了 atomic 压缩和尾项归约的不确定性。
2. 分位样本按 `c/p` 缩放后，128K 仍可把平均候选控制在约 1,280，并减少候选数量的长尾波动。
3. 在已缓存 KV、连续生成的场景中，稳态 decode 加速随长度从 32K 的 1.92x 增长到 128K 的 3.09x。
4. 当前证据仍是小规模 PPL probe；完整 LongBench、RULER、多模型、多 seed 结果未完成，不能把本表直接当作论文主结果。

## 7. 条件矩对照

条件矩并非完全失败。真实 Q/K/V 局部审计显示，16 维 query-crossfit 条件模型可把 held-out Value 残差预测误差平均降低 `46.27%`。但五个 32K、`k=1280` 闭环 probe 中，普通 8 维条件残差相对 ValueSketch 只带来 `+0.166` 个百分点的平均质量变化，延迟从 `124.6 ms` 增至 `463.2 ms`；query-crossfit 延迟达到 `1098.2 ms`，质量没有继续提高。因此它证明了 Key--Value 残差之间存在可利用结构，但尚未形成有效的质量--速度 Pareto。

## 8. 原始结果位置

- CUDA v40：`results/20260804_qksieve_valuesketch_deterministic_v40/`
- c64 64K/128K 无 profile：`results/20260804_qksieve_deterministic_truec64_length_v40_4gpu_v1/`
- c32 64K/128K 无 profile：`results/20260804_qksieve_deterministic_truec32_length_v40_4gpu_v1/`
- c64 32K 与 64K/128K profile：`results/20260804_qksieve_v40_c64_final_profile_5gpu_v1/`
- Qwen3-4B 重复性 probe：`results/20260804_qksieve_deterministic_c64_eval32_3gpu_v1/`
- 条件残差 fixed-budget：`results/20260804_qksieve_condres_fixedbudget_screen_8gpu_v4/`
- 条件残差 query closed-loop：`results/20260804_qksieve_condres_query_closedloop_8gpu_v3/`
- 条件残差 Wiener closed-loop：`results/20260804_qksieve_condres_wiener_closedloop_8gpu_v2/`
- 条件矩真实 Q/K/V 审计：`results/20260804_conditional_query_crossfit_generality_6gpu_v1/`
