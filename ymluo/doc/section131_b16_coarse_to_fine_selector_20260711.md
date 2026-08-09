# 2026-07-11：b16 coarse-to-fine selector 实验记录

## 动机

前一轮 b16/b32 实验显示，直接让所有小 block 全局竞争会明显掉分。但这不说明 16-token block 本身不可行，更可能的问题是定位信号太碎、噪声太高。

新的假设是：先用稳定的 128-token 粗粒度区域做召回，再在命中的粗区域内部保留 16-token 细块。这样既避免全局小块噪声，又保留更细粒度 KV 裁剪的可能收益。

## 方法

新增 `coarse_to_fine` selector：

1. fine page 仍为 16 tokens；
2. coarse group 由 8 个 fine pages 组成，即约 128 tokens；
3. 先按 coarse group 的 evidence score 选候选粗组；
4. 再只在候选粗组内部交给原有 MMR/flow/bridge selector 精细选择 16-token pages；
5. sink/recent、exact anchor、coverage certificate、graph bridge 等安全通道不受 coarse-to-fine 限制。

代码改动：

- `run_controlled_public_kv_benchmark_v1.py` 新增 `ours_coarse_to_fine_*` 配置项；
- policy 中可用 `coarse_to_fine: true` 按任务开启；
- CSV 额外记录候选粗组数、候选 fine page 数、候选 token 数。

## 已启动实验

| 实验 | 设计 | GPU | 当前状态 |
|---|---|---:|---|
| v216_b16_ctf_recall_m20 | b16；coarse group=8；candidate multiplier=3.0；neighbor groups=1 | 0 | running |
| v217_b16_ctf_aggressive_m20 | b16；coarse group=8；candidate multiplier=1.75；neighbor groups=0 | 1 | running |
| v218_b16_ctf_true128_m20 | b16；coarse group=8；multiscale smoothing 也改成 8；candidate multiplier=2.5；neighbor groups=1 | 2 | running |

对应 policy：

- `riskkv_task_policy_v216_b16_ctf_recall_v206_20260711.json`
- `riskkv_task_policy_v217_b16_ctf_aggressive_v206_20260711.json`
- `riskkv_task_policy_v218_b16_ctf_true128_v206_20260711.json`

## 判断标准

如果 b16 coarse-to-fine 能接近或超过 v200/v206 的 m20 分数，同时保持更低 token ratio 或更好 online speed，就继续扩大到 m50/m100。

如果仍明显低于 v200/v206，则结论是：当前方法更适合 128-token block；b16 可以作为 ablation，暂时不进入主方法。
