# 第三轮：Ground-truth Routing 与 Gate Selectivity

## 这一轮要回答的问题

前两轮实验说明，attention score 确实能捕捉 synthetic 数据里的 local slot / higher-level unit，但标准 MoE routing 没有自然形成同样干净的 feature bucket。因此第三轮不再直接假设“改 attention 就能解决 MoE”，而是把问题拆成三步：

1. **如果直接按照 ground-truth feature 做 routing，是否真的有好处？**
   这一步验证 feature-based expert bucket 本身是不是合理目标。

2. **当前表征里，线性 gate 是否有能力读出 ground-truth routing？**
   这一步验证失败是否来自 gate 的表达能力上限。如果表征中根本线性不可读，那么研究 inhibition / 初始化没有意义。

3. **如果 gate 已经接近 ground-truth routing，模型性能是否会接近 ground-truth dispatch？**
   这一步验证“最终 routing 准确”是否等价于“expert 已经学成 ground-truth expert”。

## 问题 1：Ground-truth Routing 是否有好处

实验直接用数据生成器里的 feature label 做 expert dispatch：

```text
local slot id / higher-level unit id -> expert id
```

映射方式测试了两种：

```text
hash:
  expert_id = feature_id % num_experts

frequency_balanced:
  先估计每个 feature 的出现频率；
  再把 feature 贪心分配给当前 token load 最低的 expert。
```

测试组合覆盖 uniform / zipf 两种数据分布，以及 local slot / higher-level unit 两种 feature 层次。

**结论：ground-truth routing 不仅可行，而且 higher-level ground-truth routing 稳定优于 learned gate。**

关键结果如下：

| 数据分布 | routing feature | mapping | loss | accuracy |
|---|---|---|---:|---:|
| uniform | learned baseline | learned gate | 0.262 | 91.5% |
| uniform | local slot | hash | 0.257 | 91.5% |
| uniform | local slot | frequency-balanced | 0.259 | 91.4% |
| uniform | higher-level unit | hash | 0.229 | 93.3% |
| uniform | higher-level unit | frequency-balanced | 0.234 | 93.0% |
| zipf | learned baseline | learned gate | 0.209 | 94.0% |
| zipf | local slot | hash | 0.207 | 94.0% |
| zipf | local slot | frequency-balanced | 0.207 | 94.0% |
| zipf | higher-level unit | hash | 0.192 | 94.7% |
| zipf | higher-level unit | frequency-balanced | 0.192 | 94.6% |

复跑 zipf + higher-level + hash / frequency-balanced 后，结果基本不变，说明这个收益不是随机种子偶然造成的。

这一点很关键：MoE 不 selective 并不是因为 feature-based routing 这个目标错了。相反，存在一个明显更好的 routing 函数，只是标准 learned gate 没有自然学到。

## 问题 2：Linear Gate 能否读出 Ground-truth Routing

为避免在线训练时 routing 反过来改变 backbone 表征，我们先做 offline linear probe：加载训练好的模型，冻结全部参数，只在固定表征上训练一个线性分类器，预测 ground-truth expert id。

测试的表征包括：

- learned baseline 的 hidden；
- attention-output router 模型的 attention output；
- ground-truth dispatch 模型的 hidden；
- ground-truth dispatch 模型的 attention output；
- supervised-gate 模型的 hidden。

**结论：普通 learned gate 的表征只能中等程度线性读出 ground-truth routing；supervised gate 能把可读性推到 97% 左右，但仍不到 100%。**

代表性结果：

| 模型 / 表征 | best probe accuracy |
|---|---:|
| learned baseline / hidden | 70.3% |
| attention-output router / attention output | 74.5% |
| ground-truth dispatch / hidden | 87.3% |
| ground-truth dispatch / attention output | 87.2% |
| supervised gate / hidden | 96.7% |

这说明两件事：

1. 标准模型的表征中确实有 hierarchy signal，但它不是非常干净的线性可分结构。
2. 通过监督信号，gate/backbone 可以把 ground-truth routing 变得接近线性可读，因此“线性 gate 完全没有能力”不是主要解释。

## 问题 3：Gate 接近 Ground-truth Routing 是否足够

我们进一步训练 supervised gate：用 ground-truth expert id 对 router 加监督 loss，但实际 dispatch 仍由 gate 决定。测试了 linear gate 和两层 MLP gate，也分别测试 hidden / attention output 作为 router input。

**结论：即使 gate 的 routing accuracy 已经达到 97% 左右，next-token 性能仍然接近 baseline，明显不如直接 ground-truth dispatch。**

代表性结果：

| 模型 | routing match | loss | accuracy |
|---|---:|---:|---:|
| learned baseline | - | 0.209 | 94.0% |
| ground-truth dispatch | 100% | 0.192 | 94.7% |
| supervised linear gate | about 97% | 0.209 | 94.0% |
| supervised MLP gate / hidden | about 97% | 0.209 | 94.0% |
| supervised MLP gate / attention output | about 97% | 0.209 | 94.0% |

更关键的是，把 supervised-gate checkpoint 在评估时强行切换为 ground-truth dispatch，loss 反而变差。这说明问题不是“评估时 gate 还差 3% 所以拖累性能”，而是训练过程中 expert 从一开始就没有按照 ground-truth feature 建立稳定 ownership。

因此，最终 gate 分类准确不等价于 expert 已经学成 ground-truth expert。ground-truth dispatch 的收益来自训练全程稳定的 feature-to-expert ownership，而不是训练结束时临时把 token 分到某个看起来正确的 expert。

## 当前结论

**结论 1：feature-based expert bucket 是合理目标。**
在 synthetic hierarchy 数据上，higher-level ground-truth routing 稳定优于 learned gate，说明我们想要的 feature-to-expert assignment 本身不是错的。

**结论 2：当前 learned gate 没学到 ground-truth routing，不只是因为表征里没有 feature。**
attention / hidden 表征里有 hierarchy signal；监督训练还能把 gate match 提到约 97%。这说明 feature signal 存在，但标准 next-token loss 没有自然把它转化为稳定、干净的 expert ownership。

**结论 3：routing accuracy 高不代表 expert ownership 正确。**
supervised gate 能接近 ground-truth label，但性能仍接近 baseline；强行 ground-truth dispatch 甚至会伤害 supervised-gate checkpoint。这说明 expert 在训练过程中学到的是与当时动态 routing 绑定的函数，而不是事后可任意替换的 ground-truth expert。

**结论 4：导师关于 inhibition / winner-take-all 的方向仍然成立，但问题应表述得更精确。**
当前需要的不是简单“让 gate 更强”，而是让 expert 在训练早期就形成稳定、互斥的 feature ownership。否则 gate 即使后来接近 ground-truth routing，也不能把已经学偏的 expert 变成 ground-truth expert。

## 下一步

1. 设计一种训练机制，让 feature-to-expert ownership 在训练早期就固定或逐渐稳定下来，而不是训练结束后再监督 gate。
2. 比较两类方法：显式 ground-truth warmup / curriculum，以及无监督 inhibition / winner-take-all loss。
3. 继续观察 expert 内部参数和表征是否随着稳定 ownership 出现更清晰的 feature specialization。
