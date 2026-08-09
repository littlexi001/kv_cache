# section178：低 KV 过夜优化记录（v381-v383）

日期：2026-07-12

目标：在 LongBench question-aware 设置上，尽量达到 1%-10% KV keep、2.5x+ 速度、分数达到 full baseline 95%+。当前更现实的论文目标不是继续盲扫固定预算，而是把“低 KV 可行样本”和“必须保守处理的高风险样本”区分得更准。

## 已确认的最好实际结果

目前已完成的全任务 M20 中，最好的低 KV 实际可用方法是 v377：

| 方法 | samples | score | KV keep | speed/full | vs full | vs v300 practical |
|---|---:|---:|---:|---:|---:|---:|
| v377 global Pareto knapsack | 320 | 0.4029 | 8.48% | 5.29x | 110.15% | 91.73% |
| v376 strict10 Pareto fused | 320 | 0.3969 | 6.03% | 5.81x | 108.49% | 90.35% |

这说明 1%-10% KV + 2.5x+ + 不低于 full baseline 95% 这个目标，在 M20 上已经被实际方法达到。现在真正要做的是把这个结果在 M100/更多模型/更多长度上验证稳，并缩小与更强 practical baseline v300 的差距。

## 关键现象

1. 纯激进压缩可以非常快，但少数 QA/code 样本会被误杀，平均分会掉到 0.30 左右。
2. v377 的任务级 Pareto/knapsack 分配能在 8.48% KV 下保持 0.4029，说明“不同任务需要不同安全预算”是真现象。
3. v380 的离线 multiclass router 显示 oracle gap 很大：all 上 learned=0.4163、oracle=0.4320、KV=7.92%。但真实运行里有样本被后置 v293 规则再次升到很高 KV，说明“后置保守升级”可能在吃掉预算。
4. v382 离线重训后，以 v377 为默认基座，router all=0.4204、KV=7.29%、speed/full=6.37x；阈值 0.4 离线重放 all=0.4222、KV=7.63%，比裸阈值 0.0 更稳。

## 今晚启动的实验

| 版本 | 假设 | 配置 | 预期判断 |
|---|---|---|---|
| v381 | v380 真实运行不稳定，主要是因为 learned action 后又跑 v293 后置升级 | 复用 v380 模型，关闭 `v293_rules_after_learned` | 如果 KV 显著下降且分数不掉，说明后置升级有害 |
| v382 | 以 v377 为安全基座，再让 router 全信任地替换动作，可以缩小 oracle gap | 重新训练 base=v377 的 multiclass router，threshold=0.0 | 如果 M20 接近离线 0.420/7.3%，它是强主候选 |
| v383 | v382 可能过拟合，低置信样本应回退 v377 | 复用 v382 模型，threshold=0.4，fallback=v377 | 如果分数接近/超过 v382 且 KV 仍 <10%，它更适合作为论文方法 |
| v384 | v380 只在 2wikimqa/hotpotqa/musique 上优于 v377，全局启用会拖累 qasper/multifieldqa/code | 默认 v377，只在 2wikimqa/hotpotqa/musique 启用 v380 router | 如果 M20 超过 v377 且 KV <10%，它是最干净的 task-gated 主候选 |

## 正在运行/结果入口

主要输出目录：

- `outputs/riskkv_v19_v381_policy_multiclass_nopost_20260712_policy_multiclass_nopost_v381_m20_bDyn_pDyn`
- `outputs/riskkv_v19_v382_policy_multiclass_base_v377_20260712_policy_multiclass_base_v377_v382_m20_bDyn_pDyn`
- `outputs/riskkv_v19_v383_policy_multiclass_base_v377_conf040_20260712_policy_multiclass_base_v377_conf040_v383_m20_bDyn_pDyn`
- `outputs/riskkv_v19_v384_task_gated_v377_plus_v380_20260712_task_gated_v377_plus_v380_v384_m20_bDyn_pDyn`

对应日志：

- `outputs/logs/watch_v381_policy_multiclass_nopost_20260712.log`
- `outputs/logs/watch_v382_policy_multiclass_base_v377_20260712.log`
- `outputs/logs/watch_v383_policy_multiclass_base_v377_conf040_20260712.log`
- `outputs/logs/watch_v384_task_gated_v377_plus_v380_20260712.log`

这些 watcher 都使用 M20 gate：只有满足 `score/full >= 95%`、`1% <= KV <= 10.5%`、`speed/full >= 2.5x` 才自动进入 M100。

## 明早优先看什么

1. 如果 v383 M20 最高：把 v383 作为当前主方法，等待 M100，后续补 per-task failure analysis。
2. 如果 v382 最高但 v383 稍低：说明分类器进攻性有价值，但需要更好的置信校准，下一步做 per-task threshold 或 conformal fallback。
3. 如果 v381 明显好于 v380：论文故事里强调“先选动作，再受控 fallback”，避免无界后置升级。
4. 如果 v377 M100 稳住：v377 可以作为保底主线；v382/v383 是样本级增强模块。

## 当前方法故事雏形

可以把方法暂命名为 RiskKV-Policy：先用 query/block 统计特征和检索稳定性识别样本风险，再在一组 memory actions 中选择最小安全动作。核心创新不应说成“一个 router”，而应说成：

- block 级证据定位负责找到候选证据；
- task/global Pareto 约束负责控制总体 KV budget；
- sample-level risk-aware policy 负责在低 KV 动作和保守动作之间切换；
- fallback 不是失败补丁，而是受校准的最小安全动作选择。

这个故事能和 AdaKV/SnapKV/PyramidKV 区分开：它不是 question-agnostic 的注意力保留，也不是固定 budget 压缩，而是 question-aware、risk-aware、budget-aware 的动作规划。
