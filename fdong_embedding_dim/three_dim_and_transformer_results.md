# 三维 Toy 与 Transformer 表征空间实验记录

## 问题与当前结论

### 问题一：提升维度为什么能让 tail 数据学得更好？为什么初始化不好时又不行？

**当前结论：提升维度提供的是潜在 residual 表征容量；初始化决定这些容量能否真的被 tail 使用。**

初始化观察和梯度竞争理论可以统一起来：初始化决定竞争结构是否在训练早期出现；梯度干扰 / effective rank 理论解释这种竞争为什么会影响收敛后的表征空间和预测效果。

所以，维度提升不是自动有效。只有当 tail 真正进入 common 主方向之外的 residual space，raw dimension 才会转化为 tail effective dimension，并改善 tail 学习。

### 问题二：tail 内部会不会继续出现 head-tail 竞争结构？

**当前结论：会。**

长尾学习困难不是简单的“head 学好、tail 学不好”。频率分布会在表征空间中形成层级化竞争：common 先塑造主空间，tail 在剩余空间里竞争；tail 内部如果仍然不均匀，就会继续出现次一级的方向分配和 margin 差异。

### 问题三：真实 Transformer 是否也符合这个图景？

**当前结论：大方向符合，但 Transformer 会弱化简单 toy MLP 中“初始化几何决定最终几何”的强说法。**

真实 Transformer 中，common/head 仍然会形成更稳定的主子空间，tail 主要进入这个主子空间之外的 residual space。但 Transformer 的多层结构可能把 packed embedding 中的 tail 重新展开，因此 toy 模型中的初始化结论不能被机械外推。

### 当前理论图像

现在比较稳妥的理论图像是：

1. 高频数据先决定最容易降低 loss 的主方向 / 主子空间。
2. 低频数据不是自由学习，而是在 common 已经塑形的空间里寻找可区分位置。
3. 提升维度可以增加 residual 空间，但只有当训练真的把 tail 放进这些方向时，raw dimension 才会转化为 tail effective dimension。
4. 初始化影响训练早期的竞争格局；梯度干扰 / effective rank 描述这种竞争如何影响后续收敛。
5. Tail 内部如果还有频率差异，会继续出现递归的 head-tail 结构。
6. 在真实 Transformer 中，多层结构可能把 packed embedding 中的 tail 重新展开，因此 toy 结论不能被机械外推。

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

## 梯度干扰机制实验

### 实验目的

前面的三维 toy 实验已经观察到：初始化会影响最终表征空间和预测效果。这里进一步要回答的是：

> 初始化为什么会影响后续收敛？这种影响是否可以被梯度干扰 / effective rank 理论解释？

因此，这组实验不再只是看最终 loss，而是在训练过程中记录不同 data condition 诱导出的 group-conditioned gradients。具体地说，对每个 checkpoint，分别用 `common`、`tail1`、`tail2`、`tail3` 数据计算平均梯度，然后统计：

- tail gradient effective rank；
- all gradient effective rank；
- tail representation residual effective rank；
- tail SIR；
- tail-tail gradient cosine；
- common-tail gradient cosine；
- group-wise loss / margin。

这里的 feature 不被预设为训练前固定方向。我们实际测量的是：

> 某个 data condition 在当前模型状态下诱导出的表征结构和参数更新方向。

这样可以把导师的梯度干扰理论和我们的初始化观察连接起来：初始化决定 data condition 一开始是否进入同一个低维竞争结构；梯度统计描述这个竞争结构如何在训练中表现出来。

### 实验设置

实验只比较 `dim=2` 和 `dim=3`，因为三维仍然可以直接可视化和直观理解。每个维度下比较：

- `uniform + spread`
- `uniform + packed_common`
- `zipf + spread`
- `zipf + packed_common`

输出目录：

```text
fdong_embedding_dim/outputs/toy_gradient_interference
```

总汇总表：

```text
fdong_embedding_dim/outputs/toy_gradient_interference/all_runs_summary.csv
```

### 结果一：坏初始化下，三维也不会自动变成高 effective dimension

最关键的对照是三维 `spread` 和三维 `packed_common`。

`3D uniform spread`：

- final loss：`0.000428`
- tail representation residual effective rank：`1.997`
- `tail1 / tail2 / tail3` loss：`0.000743 / 0.000818 / 0.000705`
- `tail1 / tail2 / tail3` margin：`8.176 / 8.311 / 8.404`
- final tail SIR：`1.483 / 1.390 / 1.196`

`3D uniform packed_common`：

- final loss：`0.002730`
- tail representation residual effective rank：`1.000`
- `tail1 / tail2 / tail3` loss：`0.005732 / 0.005874 / 0.005677`
- `tail1 / tail2 / tail3` margin：`5.679 / 5.767 / 5.746`
- final tail SIR：`0.990 / 0.911 / 0.991`

这说明：raw dimension 都是 `3`，但实际使用的 tail residual effective dimension 完全不同。Spread 初始化下，tail 使用了接近二维的 residual plane；packed 初始化下，tail 仍然只使用接近一维的 residual subspace。

因此，坏初始化导致长尾数据学不好，并不是因为三维空间里没有额外方向，而是因为训练过程没有把 tail 分配到这些方向上。这个结果支持导师梯度理论中的核心修正：

> 控制 tail 学习的是 effective dimension，而不是 raw dimension。

同时，它也解释了初始化为什么重要：

> 初始化会决定训练早期 tail 是否已经被放进同一个低 effective dimension 的竞争结构中。

### 结果二：好初始化下，三维提升能真正改善 tail 学习

`2D uniform spread`：

- final loss：`0.002211`
- tail representation residual effective rank：`1.000`
- tail loss 约为：`0.0045`

`3D uniform spread`：

- final loss：`0.000428`
- tail representation residual effective rank：`1.997`
- tail loss 约为：`0.0007` 到 `0.0008`

这说明：当初始化没有让 tail 和 common packed 在同一个方向附近时，维度从二维提升到三维会真实转化为更高的 tail residual effective dimension，并显著降低 tail loss、提高 tail margin。

所以“提升维度让 tail 学得更好”的具体机制是：

> 维度提升提供了更大的 residual 表征容量；spread 初始化让 tail 能使用这个容量；tail effective dimension 提升后，不同 tail data condition 在表征空间和梯度空间中的重叠降低，因此 tail loss 和 margin 改善。

### 结果三：tail 内部 Zipf 会产生递归的 head-tail 竞争

`3D zipf spread`：

- final loss：`0.000393`
- tail representation residual effective rank：`1.976`
- `tail1` prob `0.20`：loss `0.000411`，margin `8.662`，SIR `1.861`
- `tail2` prob `0.07`：loss `0.001072`，margin `7.594`，SIR `1.322`
- `tail3` prob `0.03`：loss `0.001786`，margin `7.405`，SIR `0.981`

这个结果说明，即使 tail 已经能进入二维 residual plane，tail 内部如果还有 Zipf 频率差异，也会继续出现 head-tail 排序。更高频的 tail 获得更低 loss、更高 margin 和更高 SIR；最低频的 tail 表现最弱。

`3D zipf packed_common`：

- final loss：`0.002362`
- tail representation residual effective rank：`1.000`
- `tail1` prob `0.20`：loss `0.002851`，margin `6.484`
- `tail2` prob `0.07`：loss `0.006372`，margin `5.484`
- `tail3` prob `0.03`：loss `0.014238`，margin `4.808`

这说明 packed 初始化和 Zipf 频率差异会叠加：packed 让所有 tail 都处在低 effective dimension 的竞争结构中；Zipf 让其中最低频 tail 进一步变得最差。

### 机制结论

这组实验把两种解释统一起来：

1. 我们原本观察到的是初始化对模型收敛效果的影响。
2. 梯度干扰 / effective rank 理论解释了为什么初始化会影响收敛后的表征空间和预测效果。

更准确地说：

> 初始化不是梯度竞争理论之外的另一种解释。初始化决定竞争结构是否在训练早期出现；梯度竞争理论解释这个竞争结构如何通过 low effective rank、low SIR 和 gradient overlap 影响 tail 学习。

因此，当前结论是：

> 如果 tail 初始化时避开 common 主方向，并且 tail 内部频率相对均匀，那么提升维度可以让 tail 在 residual space 中形成高 effective dimension 表示，tail 学得接近一样好。
> 如果 tail 初始化时和 common packed 在一起，即使 raw dimension 提升，tail 也可能仍然停留在低 effective dimension 子空间中，最终学得更差。
> 如果 tail 内部还有 Zipf 分布，那么 tail-high 会比 tail-low 学得更好；packed 初始化会进一步放大最低频 tail 的劣势。

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
