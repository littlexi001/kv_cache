# Section 13 - Top-k Negative Update MoE Experiment

> 新增日期：2026-05-27  
> 对应项目：`ymluo/projects/qwen3_moe_topk_negative_update`

## 1. 实验目的

本实验想验证一种新的 MoE expert 更新规则：

```text
每个 token 前向选择 top-k 个 expert；
只对权重最高的前 p 个 expert 做正常梯度更新；
对剩下的 k-p 个被选中 expert 做反向梯度更新。
```

当前实验默认配置是：

```text
MOE_NUM_UNIQUE_EXPERTS=16
MOE_NUM_EXPERTS_PER_TOK=4
NEGATIVE_UPDATE_PRIMARY_SLOTS=1
NEGATIVE_UPDATE_SECONDARIES=true
NEGATIVE_UPDATE_SCALE=0.1
```

也就是：

```text
16 个 expert 中，每个 token 选择 4 个 expert；
top1 expert 正常更新；
top2/top3/top4 expert 反向更新。
```

实验核心问题：

```text
这种“一个正向 + 多个反向”的 expert 更新方式，能不能让相同 feature 的 token 更倾向于被分发到同一个 expert？
```

这里的“相同 feature”在 hierarchical synthetic data 中主要指同一个 higher-level unit。

## 2. 方法说明

### 2.1 Forward 仍然是 top-k MoE

前向计算时，router 对每个 token 输出 16 个 expert 的概率，然后选择 top-4：

```text
selected_experts = top4(router_probs)
```

前向输出仍然使用这 4 个 expert 的加权和：

```text
y = w1 * E1(x) + w2 * E2(x) + w3 * E3(x) + w4 * E4(x)
```

因此，反向更新机制不会改变 forward activation，只改变 backward gradient。

### 2.2 Top1 正常更新

对于 top1 expert：

```text
E1
```

梯度按普通训练方式更新。也就是说，如果 loss 希望当前 token 的表示往某个方向调整，top1 expert 会被训练成更好地服务这个 token。

### 2.3 其余 selected expert 反向更新

对于 top2/top3/top4：

```text
E2, E3, E4
```

forward 值保持不变，但 backward 梯度被乘以负号和 scale。

代码中的核心形式是：

```text
flipped = expert_output.detach() - scale * (expert_output - expert_output.detach())
```

当 `scale=1.0` 时，相当于对 secondary expert 做完整梯度上升。  
当前实验使用：

```text
NEGATIVE_UPDATE_SCALE=0.1
```

所以 secondary expert 的反向更新强度是正常梯度的 10%。

直观解释：

```text
top1 expert 被训练成更适合当前 token；
top2/top3/top4 expert 被轻微训练成不适合当前 token。
```

这相当于给 expert 分工施加一种竞争机制：如果某个 feature 已经由 top1 expert 负责，那么其他被 router 同时选中的 expert 会被推离这个 token 的优化方向。

## 3. 数据设置

本次结果来自垂直层级 synthetic data：

```text
SYNTHETIC_DATA_MODE=hierarchical
SEQ_LEN=128
SYNTHETIC_BLOCK_SIZE=4
SYNTHETIC_NUM_HIERARCHY_LAYERS=2
SYNTHETIC_CONTENT_TOKEN_COUNT=256
SYNTHETIC_NUM_UNITS_PER_LAYER=64
SYNTHETIC_SAMPLING_DISTRIBUTION=zipf
SYNTHETIC_ZIPF_ALPHA=1.1
```

该数据的结构是：

```text
layer0 unit = 4 个 raw token
layer1 unit = 4 个 layer0 unit
```

因此一个 higher-level unit 展开后长度是：

```text
4 * 4 = 16 tokens
```

`same_higher_same_expert` 统计同一个 higher-level unit 内的 token 是否被路由到同一个 expert。

## 4. 指标解释

### 4.1 same_higher_same_expert

该指标衡量：

```text
同 feature token pair 中，有多少 pair 被分发到同一个 expert。
```

在本实验的 hierarchical data 中，feature 是 higher-level unit id。

公式可以写成：

```text
same_higher_same_expert =
  count(feature_i == feature_j and expert_i == expert_j)
  / count(feature_i == feature_j)
```

其中：

- 排除 `i == j` 的 self pair；
- 排除 `feature_id < 0` 的无效 token；
- 当前 top-k MoE 的 metric 使用 `expert_labels[..., 0]`，也就是 top1 expert 作为该 token 的 primary expert。

因此，这个指标回答的是：

```text
同一个 higher-level unit 的 token，top1 expert 是否一致？
```

### 4.2 same_higher_occurrence_by_layer

这个指标不是看 metadata 中的真实 higher-level unit id，而是用纯位置构造 occurrence id。

在当前 hierarchical 设置下，一个 higher-level unit 长度是 16，因此：

```text
occurrence_id = position // 16
```

它用于交叉验证模型是否只是按位置 block 聚类。

### 4.3 expert_load_by_layer

该指标统计每层 top1 expert 的 token 占比。对于 16 experts，理想均衡时每个 expert 约为：

```text
1 / 16 = 0.0625
```

如果少数 expert 占比很高，说明 router/expert 分工发生塌缩。

## 5. 实验结果

训练到 step 10000：

```text
eval_loss = 0.2014
eval_acc  = 0.9422
same_higher_same_expert = 0.6038
higher_mass = 0.4794
```

训练末尾的 LM loss 和 accuracy：

| step | loss | acc |
| ---: | ---: | ---: |
| 9940 | 0.1990 | 0.9430 |
| 9950 | 0.2013 | 0.9435 |
| 9960 | 0.2007 | 0.9427 |
| 9970 | 0.2016 | 0.9417 |
| 9980 | 0.2050 | 0.9409 |
| 9990 | 0.1993 | 0.9418 |
| 10000 | 0.1975 | 0.9432 |

Per-layer selectivity：

```text
same_higher_by_layer =
  L0:0.3665
  L1:0.6222
  L2:0.8228
```

Occurrence 交叉验证：

```text
same_higher_occurrence_by_layer =
  L0:0.3844
  L1:0.6489
  L2:0.8622
```

Attention mass：

```text
higher_mass_by_layer =
  L0:0.3836
  L1:0.5617
  L2:0.4928
```

Expert load：

```text
expert_load =
[
  0.1203, 0.0194, 0.0000, 0.0024,
  0.4773, 0.0000, 0.0000, 0.0000,
  0.1089, 0.0000, 0.2556, 0.0096,
  0.0000, 0.0012, 0.0053, 0.0000
]
```

Per-layer expert load：

```text
L0:
[0.025,0.057,0.000,0.007,0.511,0.000,0.000,0.000,
 0.325,0.000,0.026,0.029,0.000,0.004,0.016,0.000]

L1:
[0.258,0.000,0.000,0.000,0.000,0.000,0.000,0.000,
 0.001,0.000,0.741,0.000,0.000,0.000,0.000,0.000]

L2:
[0.078,0.001,0.000,0.000,0.921,0.000,0.000,0.000,
 0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000]
```

## 6. 结果分析

### 6.1 LM task 学得很好

最终：

```text
eval_loss=0.2014
eval_acc=0.9422
```

这说明 top-k negative update 没有破坏模型学习 hierarchical synthetic next-token prediction。模型仍然能把主任务学到接近之前 attention-cluster 实验的水平。

### 6.2 Selectivity 明显高于随机，但没有达到极高水平

16 个 expert 随机分配时，同 expert 概率约为：

```text
1 / 16 = 0.0625
```

当前整体：

```text
same_higher_same_expert=0.6038
```

远高于随机，说明该机制确实让同 higher-level unit 的 token 倾向于共享 top1 expert。

分层看：

```text
L0:0.3665
L1:0.6222
L2:0.8228
```

越往高层，same-higher selectivity 越强。L2 达到 0.8228，说明顶层表示中出现了较强的 feature-level expert 聚合。

### 6.3 但是 expert load 严重塌缩

虽然 selectivity 上升，但 expert 使用非常不均衡。

整体上 expert 4 占：

```text
47.7%
```

expert 10 占：

```text
25.6%
```

很多 expert 几乎完全不用。

L2 更极端：

```text
expert 4 = 92.1%
```

这说明当前机制有明显副作用：它可以把相同 feature 推向相同 expert，但也容易形成 winner-take-most 的塌缩。

### 6.4 load_balance 没有生效

日志里：

```text
load_balance=0.0000
```

虽然脚本中给了：

```text
MOE_LOAD_BALANCE_LOSS_WEIGHT=0.05
```

但训练日志显示实际没有把 load-balance loss 加进去，或者模型 output 中没有返回有效的 `moe_load_balance_loss`。

这点很关键。当前结果应理解为：

```text
top-k negative update 在几乎没有 load-balance 约束时的行为
```

如果后续要研究该方法是否能同时做到 selectivity 和均衡负载，需要先确认 load-balance loss 是否真正生效。

## 7. 初步结论

本实验支持以下判断：

1. top1 正向、其余 top-k 反向的更新机制可以显著提高同 feature token 的 expert 一致性。
2. 在 hierarchical synthetic data 上，该机制不会破坏主任务收敛，最终 accuracy 仍达到约 94.2%。
3. 该机制的主要问题是 expert load 严重塌缩，尤其高层 L2 几乎集中到单个 expert。
4. 当前实验还不能证明该方法优于 attention-cluster，因为缺少有效 load-balance 以及与同配置 baseline 的直接对照。

## 8. 后续实验建议

### 8.1 加一个同配置 top-k baseline

关闭反向更新：

```bash
NEGATIVE_UPDATE_SECONDARIES=false \
RUN_NAME=topk-baseline-hierarchical \
bash ymluo/projects/qwen3_moe_topk_negative_update/scripts/run_train.sh
```

这样可以判断 selectivity 来自 top-k MoE 本身，还是来自 negative update。

### 8.2 修正或确认 load-balance loss

当前日志显示 `load_balance=0.0000`。需要检查：

```text
output_router_aux_loss=true 时是否真的返回 moe_load_balance_loss
```

以及 patched MoE forward 是否绕过了原模型中的 aux-loss 计算。

### 8.3 扫描 negative update scale

当前使用：

```text
NEGATIVE_UPDATE_SCALE=0.1
```

建议比较：

```text
0.01, 0.05, 0.1, 0.25, 1.0
```

观察 selectivity 和 expert-load collapse 的 trade-off。

### 8.4 扫描 primary slots

当前：

```text
NEGATIVE_UPDATE_PRIMARY_SLOTS=1
```

可以尝试：

```text
NEGATIVE_UPDATE_PRIMARY_SLOTS=2
```

即 top2 正常更新，top3/top4 反向更新。这样可能缓解单 expert winner-take-most。

### 8.5 在 structured-language 数据上复验

hierarchical data 比较干净，same feature 的结构非常强。下一步应该在：

```text
SYNTHETIC_DATA_MODE=structured_language
```

上复验，观察该机制在带 noise、shared entity、copy、bridge 的数据上是否仍能提高 selectivity，以及是否会进一步加重 load collapse。
