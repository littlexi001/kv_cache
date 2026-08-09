# section183：v392 自动 sample-level router 续跑

日期：2026-07-12

## 背景

当前最强已完成/运行中的短样本候选是：

| 方法 | samples | score | KV keep | speed/full | 状态 |
|---|---:|---:|---:|---:|---|
| v385 | 320 | 0.4205 | 9.80% | 5.02x | M100 运行中 |
| v389 | 320 | 0.4085 | 8.95% | 5.70x | M100 运行中 |
| v386 | 320 | 0.4008 | 9.36% | 5.66x | M100 运行中 |
| v387 | 320 | 0.3939 | 6.20% | 5.81x | M100 运行中 |

v389 是目前最强的 completed-M100 evidence task-knapsack 候选；v385 是短样本质量最高候选，但还需要 M100 验证。

## 新增 v392

新增脚本：

- `scripts/train_winner_router_v392_after_v385_v389_20260712.py`
- `scripts/watch_v392_after_v385_v389_20260712.sh`

v392 的逻辑：

1. 等待 v385 M100 和 v389 M100 都完成。
2. 把 v368/v375/v376/v377/v378/v380/v381/v382/v383/v384/v385/v389 这些 completed-M100 candidates 全部纳入候选池。
3. 以 v389 作为 base policy，训练 sample-level winner router，预测同一样本上哪个 candidate action 得分最高。
4. 自动枚举 task gate，只允许满足以下条件的组合通过：
   - all split `KV <= 10%`
   - all split gain 非负
   - calibration split gain >= -0.001
   - test split gain >= 0
5. 只有离线 gate 通过，才启动真实 m20；m20 通过后再启动 M100。

这个设计是为了利用 v385/v389 的新现象，同时避免全局 winner router 的 Hotpot/LCC 泛化风险。

## 当前状态

v392 watcher 已启动，当前处于等待状态：

```text
WAIT v385/v389 M100 results
```

它不会立刻占 GPU。只有 v385/v389 M100 完成并且离线 gate 通过，才会启动真实实验。

## 论文意义

这条线给方法故事补了一层：

- task-level knapsack 给稳定的全局预算分配；
- sample-level winner router 负责吃掉同一任务内部的 oracle gap；
- task gate / split gate 用来避免不稳定任务上的过拟合。

如果 v392 通过，这会比单纯 v389 更像一个完整可投稿方法：`budget planner + operator pool + risk-aware sample router`。
