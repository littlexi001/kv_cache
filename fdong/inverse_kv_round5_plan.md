# 第五轮：在 Synthetic 数据上系统尝试 MoE Specialization 结构

## 这一轮要回答的问题

Round 5 对应 specialization overview 中的第 3 部分：**什么结构能做到 specialization**。

前几轮实验已经说明：

1. baseline MoE 不会自然形成 ground-truth slot specialization；
2. attention 中存在可用的 hierarchy feature signal；
3. ground-truth routing 本身有收益；
4. load-balance loss 只能改变 expert usage shape，不能直接产生 feature-level specialization；
5. 为了服务 KV cache reverse indexing，最终可部署 routing signal 最好在 attention 前产生。

因此，这一轮不再只诊断现有 MoE 为什么失败，而是开始在 synthetic 数据上系统尝试不同 MoE 结构，目标是找到能稳定形成 feature-level expert bucket 的结构。

## 目标

这一轮的目标不是单纯提高 next-token prediction accuracy，而是同时验证三件事：

1. **Predictive utility：** 模型仍然能完成 synthetic next-token prediction；
2. **Feature selectivity：** expert assignment 与 ground-truth local / higher-level feature 对齐；
3. **Deployability：** routing signal 尽可能能在 attention 前产生，从而有潜力服务 KV cache reverse indexing。

## 候选结构

### 1. Ground-truth / Oracle Routing

使用 synthetic 数据生成器提供的 ground-truth feature label 直接做 expert dispatch，例如 local slot 或 higher-level slot。

这个方向不是最终部署方案，而是作为上限和 sanity check：

```text
如果 ground-truth feature routing 都没有收益，
说明 feature-based expert bucket 不是合理目标；

如果 ground-truth feature routing 有收益，
说明问题在于 learned gate 如何学到这种 routing。
```

### 2. 去掉或调整残差

测试 gate 输入中的 residual 是否让 routing 更容易读取 token identity / local shortcut，而不是读取更抽象的 feature representation。

需要比较：

1. 使用 hidden state routing；
2. 使用 attention output routing；
3. 使用去残差或弱残差后的 routing representation。

目标是判断：改变 gate 输入后，expert assignment 是否更接近 local / higher-level feature。

### 3. SD on Forward Representation

在前向表征上加入 specialization-driving signal，使相同 feature 的 representation 更接近，不同 feature 的 representation 更可分。

这个方向的核心问题是：

```text
如果 representation 本身更 feature-separable，
线性 / MLP gate 是否就能形成更稳定的 expert bucket？
```

### 4. Head-level MoE

每个 attention head 使用独立的 MoE routing。

动机是不同 attention head 可能捕捉不同 feature relation，因此 head-level routing 可能比 token-level routing 更容易形成细粒度 specialization。

需要重点观察：

1. 不同 head 是否对应不同 feature level；
2. head-level expert 是否比 token-level expert 更 selective；
3. head-level routing 是否更接近 attention retrieval bucket。

### 5. Hierarchical MoE

使用层次化 expert 结构，让不同 expert bucket 对应不同粒度的 feature，例如：

```text
token-level feature
-> local-slot-level feature
-> higher-slot-level feature
```

这个方向直接对应 synthetic 数据中的 hierarchical generation rule。它要回答的问题是：如果数据本身是 hierarchical compositional 的，MoE 结构是否也需要显式 hierarchy 才能学到干净 specialization。

### 6. Pre-attention Routing

为了服务 KV cache reverse indexing，routing input 不能依赖 attention output。候选输入包括：

1. layer input；
2. q；
3. k；
4. v；
5. attention 前可计算的其他 projection。

目标是训练一个 attention 前就能产生的 routing signal，使它既能指导 MoE dispatch，也能在 decode 时作为 KV reverse index。

## 评价指标

每个结构至少需要报告：

1. NTP loss / accuracy；
2. expert load / entropy / effective expert count；
3. local slot 与 expert assignment 的 MI / NMI；
4. higher-level slot 与 expert assignment 的 MI / NMI；
5. feature-to-expert purity；
6. same-feature same-expert rate；
7. attention retrieval bucket 与 expert bucket 的重合度；
8. 如果结构支持 pre-attention routing，还需要报告 KV 保留比例与 reverse-indexing NTP accuracy。

## 预期输出

Round 5 最终应回答：

```text
哪些 MoE 结构能在 synthetic 数据上形成更强 specialization；
这种 specialization 是 local-level、higher-level，还是 token-level shortcut；
这种结构是否比 baseline MoE 更适合作为 KV reverse index；
哪些结构值得进一步迁移到真实语料实验。
```
