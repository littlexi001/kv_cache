# 三维 Toy 与 Transformer 表征空间实验记录

## 问题与当前结论

### 问题一：提升维度让 tail 数据学好的原理是什么？

**当前结论：提升维度给 tail 提供了更多可以避开 common 主方向的表征空间。**

更准确地说，高频 common 数据仍然会优先塑造一个主方向或主子空间；维度增加以后，tail 不必全部挤在同一条反方向或同一个局部区域里，而是可以在 common 主方向之外的剩余空间中展开。这个机制解释了为什么更宽模型常常对 tail 的改善更明显：tail 原本更受表征空间竞争限制，新增维度首先缓解的是 tail 之间的竞争，而不是 head 的学习瓶颈。

但这个结论依赖训练过程能让 tail 真正使用这些新维度。**额外维度只是提供可能性，不保证 tail 一定使用这些维度。** 在简单 toy MLP 中，如果所有 tail 数据都初始化在同一维度或同一个局部方向上，最后可能仍然无法利用更多维度，而是继续在低维区域里竞争。Transformer 中情况会更复杂：token embedding 可能保留这种 packed 几何，但多层 hidden representation 又可能把 tail 重新展开。

### 问题二：tail 内部会不会继续出现 head-tail 竞争结构？

**当前结论：会。**

当 tail 内部也有频率差异时，tail 中相对更高频的部分会获得更稳定的位置、更大的 margin 和更低的 loss；更低频的 tail-tail 数据仍然更弱。也就是说，head-tail 不是只发生在 common 与 tail 之间，而是可以在 tail 内部递归出现。

这意味着长尾学习困难不是一个二分类问题，不是简单的“head 学好、tail 学不好”。更合理的理解是：频率分布会在表征空间中形成层级化的竞争结构。Common 先塑造主空间，tail 在剩余空间里竞争；tail 内部如果仍然不均匀，就会继续出现次一级的方向分配和 margin 差异。

### 问题三：真实 Transformer 是否也符合 toy MLP 中的表征竞争图景？

**当前结论：大方向符合，但 Transformer 会弱化 toy MLP 中“初始化几何决定最终几何”的强说法。**

真实 Transformer 中，token embedding / lm_head 仍然会保留初始化几何：如果一开始把 tail token packed 在一起，训练后它们在参数空间里仍然更压缩。但是经过 Transformer 层之后，hidden representation 可以重新展开；tail 的最终 hidden state 不会像简单 toy MLP 那样被初始化几何完全锁死。

所以 Transformer 上更准确的解释是：

> Common/head 会占据一个更稳定的主子空间；tail 主要进入这个主子空间之外的 residual space。更宽的 hidden dimension 可能帮助 tail，不只是因为 embedding 轴更多，而是因为多层 hidden representation 中有更多 residual 子空间可以被 tail 使用。

### 当前理论图像

现在比较稳妥的理论图像是：

1. 高频数据先决定最容易降低 loss 的主方向 / 主子空间。
2. 低频数据不是自由学习，而是在 common 已经塑形的空间里寻找可区分位置。
3. 维度越有限，tail 之间越容易被迫共享方向或压缩在低维区域。
4. 提升维度可以增加 residual 空间，让 tail 有更多可分离的位置。
5. 但是 tail 能不能真正用上这些位置，取决于初始化、频率梯度强度和模型结构。
6. Transformer 的多层结构可以把 embedding 中被压缩的 tail 表征重新展开，因此真实 Transformer 比简单线性 / MLP toy 模型更不容易被初始几何完全限制。

## 三维 Toy 实验

### 实验目的

二维实验已经说明 common 与 tail 会发生方向分离，但二维空间太小：common 占掉一个方向后，tail 实际上只能在剩余的一维或反方向附近竞争。因此我们进一步做三维 toy 实验，观察：

- common 是否仍然优先形成主方向；
- 三个 tail group 是否能在去掉 common 方向后的二维 residual plane 中展开；
- tail 内部存在 Zipf 分布时，是否在 residual plane 内继续出现次一级竞争；
- packed 初始化是否会阻止 tail 充分使用新增维度。

### 实验设置

三维实验仍然使用 toy bigram / shared embedding setting。每个 group 对应一类数据，训练后观察 embedding centroid 的三维位置、去掉 common 方向后的 residual plane 位置、奇异值和 group-wise loss / margin。

实验包括五个 run：

- `toy3d_tail3_uniform_spread`
- `toy3d_tail3_uniform_packed_common`
- `toy3d_tail3_uniform_packed_negative_common`
- `toy3d_tail3_zipf_spread`
- `toy3d_tail3_zipf_packed_common`

输出目录：

```text
fdong_embedding_dim/outputs/toy3d_controlled
```

每个 run 输出：

- `01_E_3d_trajectories.png`
- `02_E_residual_plane.png`
- `03_training_metrics.png`
- `summary.json`

### 三维实验结果

所有三维 run 最终 accuracy 都达到 `1.000`，说明任务本身都被学会了。因此这里主要分析的是学习完成后的表征空间组织，而不是 accuracy gap。

#### Uniform tail + spread 初始化

`toy3d_tail3_uniform_spread` 的最终结果：

- weighted loss：`0.000428`
- 三个奇异值：`[10.262, 8.872, 8.556]`
- tail residual singular values：`[4.179, 4.021]`
- tail pairwise cosine mean：`-0.050`
- common loss / margin：`0.000288 / 9.246`
- 三个 tail loss：`0.000743`, `0.000818`, `0.000705`
- 三个 tail margin：`8.176`, `8.311`, `8.404`

这个结果说明：当初始化允许 tail 分开时，三维空间中去掉 common 方向后，tail 确实能在剩余二维 residual plane 中展开。两个 residual singular values 都非零且接近，说明 tail 不是只挤在一条线里。

#### Uniform tail + packed 初始化

`toy3d_tail3_uniform_packed_common`：

- weighted loss：`0.002730`
- 三个奇异值：`[20.584, 18.682, 0.000]`
- tail residual singular values：`[1.880, 0.000]`
- tail pairwise cosine mean：`0.859`

`toy3d_tail3_uniform_packed_negative_common`：

- weighted loss：`0.001875`
- 三个奇异值：`[19.782, 18.566, 0.000]`
- tail residual singular values：`[1.930, 0.000]`
- tail pairwise cosine mean：`0.741`

这两个 run 的结论很关键：三维空间本身并不保证 tail 自动展开。Packed 初始化下，tail 的 residual 表征基本退化成一维，第二个 residual singular value 为 `0.000`，tail 之间 pairwise cosine 很高，说明它们仍然挤在一起。

#### Zipf tail + spread 初始化

`toy3d_tail3_zipf_spread`：

- weighted loss：`0.000393`
- 三个奇异值：`[12.025, 9.426, 9.200]`
- tail residual singular values：`[4.247, 3.801]`
- tail pairwise cosine mean：`-0.017`
- `tail1` prob `0.20`：loss `0.000411`，margin `8.662`
- `tail2` prob `0.07`：loss `0.001072`，margin `7.594`
- `tail3` prob `0.03`：loss `0.001786`，margin `7.405`

这个结果说明：当 tail 内部有 Zipf 分布时，tail 仍然可以在 residual plane 里展开，但 tail 内部会出现明显的频率排序。更高频的 tail 获得更低 loss 和更大 margin，更低频的 tail-tail 数据表现更弱。

#### Zipf tail + packed 初始化

`toy3d_tail3_zipf_packed_common`：

- weighted loss：`0.002362`
- 三个奇异值：`[26.726, 15.853, 0.000]`
- tail residual singular values：`[5.661, 0.000]`
- `tail1` prob `0.20`：loss `0.002851`，margin `6.484`
- `tail2` prob `0.07`：loss `0.006372`，margin `5.484`
- `tail3` prob `0.03`：loss `0.014238`，margin `4.808`

这个 run 把两个机制叠加在一起：packed 初始化让 tail 主要在一维 residual 方向上竞争；Zipf 频率差异又让 tail 内部出现明显强弱排序。因此最低频的 `tail3` loss 最高、margin 最低。

### 三维实验结论

三维实验支持“提升维度帮助 tail”的机制，但这个机制不是“维度一高，tail 自动变好”。更准确地说：

> 三维提供了 common 主方向之外的二维 residual plane；如果 tail 的几何和训练信号允许，它们可以在这个 residual plane 中展开，从而降低互相挤压。但如果 tail 从一开始就 packed，训练可能仍然把它们留在低维竞争结构中。

## Transformer 实验

### 实验目的

Toy MLP 的结构太简单，真实 Transformer 中有 token embedding、lm_head、多层 attention/MLP hidden representation。这个实验的目的不是直接复现 toy MLP 的几何，而是验证 toy 中的核心图像是否仍然存在：

- 高频 head/common 是否形成更稳定的主子空间；
- tail 是否主要进入 head/common 子空间之外的 residual space；
- embedding 初始化几何是否会影响最终 token embedding / lm_head；
- Transformer 层是否会把 packed embedding 重新展开。

### 实验设置

模型设置：

- hidden size：`128`
- layer 数：`2`
- attention heads：`4`
- checkpoint steps：`250`, `500`, `750`, `1000`

三种受控 embedding 初始化：

- `spread`
- `packed_common`
- `packed_negative_common`

训练 checkpoint：

```text
fdong/checkpoints/embedding-dim-transformer-spread-h128
fdong/checkpoints/embedding-dim-transformer-packed_common-h128
fdong/checkpoints/embedding-dim-transformer-packed_negative_common-h128
```

分析输出：

```text
fdong_embedding_dim/outputs/transformer_spectral_occupation.json
```

训练日志：

```text
fdong/logs/embedding-dim-transformer-spread-h128.train.log
fdong/logs/embedding-dim-transformer-packed_common-h128.train.log
fdong/logs/embedding-dim-transformer-packed_negative_common-h128.train.log
```

### Transformer 训练收敛

三组初始化最后一步 train loss 基本一样：

| 初始化 | final train loss |
| --- | ---: |
| `spread` | `0.169` |
| `packed_common` | `0.170` |
| `packed_negative_common` | `0.170` |

因此，在当前 `h128 / 2 layer / 1000 steps` 设置下，三种初始化都能学会任务，初始化几何没有造成明显的最终训练 loss 差异。

### Token embedding / lm_head 保留初始化几何

最终 checkpoint 中，tail token embedding 的几何如下：

| 初始化 | tail pairwise cosine | tail effective rank |
| --- | ---: | ---: |
| `spread` | `0.0418` | `46.07` |
| `packed_common` | `0.1083` | `39.71` |
| `packed_negative_common` | `0.1047` | `39.05` |

Packed 初始化下，tail token embedding 训练后仍然更压缩：pairwise cosine 更高，effective rank 更低。

另外，当前分析中 `lm_head_rows` 和 `embedding_rows` 的指标完全一致，因此这里不把 embedding 和 lm_head 当成两份独立证据。它们共同说明的是：参数空间中的 token-level 几何确实保留了初始化痕迹。

### Hidden representation 会重新展开 tail

最终 hidden layer 中，tail centroid 的 pairwise cosine 和 residual rank 如下：

| 初始化 | final hidden tail cosine | tail residual rank |
| --- | ---: | ---: |
| `spread` | `-0.0058` | `17.57` |
| `packed_common` | `-0.0020` | `16.91` |
| `packed_negative_common` | `-0.0044` | `16.99` |

这说明：虽然 packed 初始化让 token embedding 更挤，但经过 Transformer 层之后，tail hidden representation 基本被重新展开了。最终 hidden state 没有延续 embedding 中那种明显压缩的几何。

### Head/common 主子空间与 tail residual space

最终 hidden layer 中，head 与 tail 在 head-subspace 上的能量分布如下：

| 初始化 | head 的 head-subspace energy | tail 的 head-subspace energy | tail residual energy |
| --- | ---: | ---: | ---: |
| `spread` | `0.3907` | `0.0547` | `0.9453` |
| `packed_common` | `0.4153` | `0.0878` | `0.9122` |
| `packed_negative_common` | `0.4040` | `0.0773` | `0.9227` |

这个结果强支持我们的核心图像：head/common 形成了一个更稳定的主子空间，而 tail 的 hidden representation 主要位于这个主子空间之外的 residual space。

### Transformer 实验结论

Transformer 实验支持 toy 实验中的“common 主空间 + tail residual space”图像，但也修正了 toy 模型中过强的初始化决定论：

> Token embedding / lm_head 会保留初始化几何；但 Transformer hidden representation 可以在多层变换中重新展开 tail。真实 Transformer 中，tail 能不能学好，不只取决于 embedding 初始位置，还取决于 hidden dimension、层结构、attention/MLP 动力学以及训练过程中频率梯度的主导程度。

## 当前限制与下一步

当前结果还不能直接证明“Transformer 维度越大，tail 一定越好”，因为这次 Transformer 只做了 `h128`。如果要直接回答 width scaling，需要在 Transformer 上继续做：

- `h64 / h96 / h128 / h256` 的同一套 spectral occupation 分析；
- 每个 width 下同时报告 head/middle/tail 的 loss、margin、hidden residual rank；
- 比较 width 增大时，tail residual rank 是否上升，tail head-subspace energy 是否下降，tail loss gap 是否缩小。

在当前证据下，我们可以比较稳妥地说：

> 三维 toy 实验证明了新增 residual 方向可以缓解 tail 竞争；Transformer 实验证明了真实模型中也存在 head 主子空间与 tail residual space 的分工，并且 Transformer 层可以把被压缩的 tail embedding 重新展开。下一步需要做 width sweep，才能把这个机制和“大模型更有利于 tail 学习”直接连起来。
