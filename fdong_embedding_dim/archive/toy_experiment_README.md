# Embedding-Dim Toy 实验

这个目录用于放置二维 toy bigram 实验。实验目标是研究：数据频率分布、初始化几何、有限表征方向之间如何相互作用。

## 核心假设与当前回答

我们现在想验证的核心理解是：

> 高频 common 数据会优先占据最容易降低 loss 的主方向；剩下的 tail 数据不是自由地学习，而是在 common 已经占掉 / 塑形的谱空间里，被迫寻找可区分的位置。
>
> 当 tail group 变多、维度有限、初始化位置不同的时候，模型会表现出一种“方向分配 / 方向竞争”的动力学。

围绕这个理解，当前实验实际在回答三个问题。

### 问题一：谱空间是不是像有限资源一样被分配？

**当前回答：是。** 也就是说，高频 group 拿走一个主方向之后，tail group 是否只能在剩下方向、反方向，或者被压缩的局部区域里竞争。Common 和 tail 会形成稳定分离；当 tail group 变多时，tail 之间会表现出方向竞争，尤其在 packed 初始化下更容易共享方向。

### 问题二：tail 内部是否会重复 head-tail 结构？

**当前回答：强支持。** Tail 内部如果也有 Zipf-like 频率差异，会继续产生 tail-high / tail-low 差异；但 tail-high 不一定单独占据固定坐标轴，而是更容易获得更低 loss、更高 margin、更稳定的 learned geometry 位置。

### 问题三：最终位置是数据频率决定的，还是初始化几何决定的？

**当前回答：两者都会影响。频率使得高频数据优先占据一个完整方向；当频率相同时，初始化影响非常强。** 模型不会自动选择唯一的“语义几何”；频率提供训练信号强弱，初始化决定竞争从哪里开始，二者共同决定最后的表示结构。


### 总体解释

当前 controlled sweeps 支持的主机制是：

> 频率帮助决定哪些结构获得更大的 margin 和更稳定的谱位置；有限二维几何让 tail group 竞争有限方向；初始化决定了这种竞争从哪里开始。

同时，这些实验也修正了一个过强说法：

> 高频 group 不一定占据固定坐标轴。更准确地说，高频 group 倾向于获得更好分离、更高 margin、更稳定的 learned geometry 位置。

## 实验结果正文

### 1. Toy 任务本身都充分学会了

所有 run 都收敛得很好。每个 run 中每个 group 的最终 accuracy 都达到 `1.000`。最差 group 的最终 loss 也仍然很小：

- `single_tail`：最差 group loss `0.000366`
- `tail3_uniform`：最差 group loss `0.001242`
- `tail3_zipf`：最差 group loss `0.003313`
- `tail4_uniform`：最差 group loss `0.001834`
- `tail4_zipf`：最差 group loss `0.009900`

因此，这个 toy setting 更适合被理解为：

> 当 common 和 tail 数据都已经被学会之后，表征空间会如何组织自己？

它支持的是“表征空间分配 / 方向竞争”的机制，而不是直接复现大模型实验里的 tail accuracy gap。

### 2. Common 与 tail 的方向分离

`single_tail` 实验中，无论 tail group 初始化在 `+x`、`-x`、`+y` 还是 `-y` 附近，最终 common centroid 和 tail centroid 都几乎相反：

- `single_tail_init_x_neg`：common-tail cosine `-0.999`
- `single_tail_init_x_pos`：common-tail cosine `-0.998`
- `single_tail_init_y_neg`：common-tail cosine `-0.945`
- `single_tail_init_y_pos`：common-tail cosine `-0.966`

这说明，当只有一个 tail group 时，模型会非常稳定地把 common 和 tail 拉开。正确理解不是“tail 一定去全局 `-x` 轴”，而是：

> common 方向本身也会在训练中移动；tail 倾向于移动到 common 最终方向的反方向。

当 tail group 增加到 3 个或 4 个时，tail 之间的 pairwise cosine 和 margin 差异显示出明显的方向竞争。尤其在 packed 初始化下，tail group 更容易保持相似、共享方向；在 spread 初始化下，它们更容易分开。

### 3. Tail 内部 Zipf 的递归差异

当 tail group 自己也有 Zipf-like 频率差异时，它们最终的 loss 和 margin 会沿着频率顺序排列。

`tail3_zipf_spread`：

- `tail1`，prob `0.20`：loss `0.0005`，margin `8.29`
- `tail2`，prob `0.07`：loss `0.0011`，margin `7.38`
- `tail3`，prob `0.03`：loss `0.0022`，margin `6.74`

`tail4_zipf_spread`：

- `tail1`，prob `0.25`：loss `0.0008`，margin `7.78`
- `tail2`，prob `0.09`：loss `0.0016`，margin `7.00`
- `tail3`，prob `0.04`：loss `0.0031`，margin `6.43`
- `tail4`，prob `0.02`：loss `0.0062`，margin `5.71`

所以，频率 skew 不只作用在 common-vs-tail 这一层。tail 内部如果还有 Zipf-like 结构，也会继续产生 tail-high / tail-low 差异。

### 4. 初始化对 uniform tail 几何的影响

uniform tail group 对初始化几何非常敏感。spread 初始化会让 tail centroid 明显分离；packed 初始化会让 tail centroid 保持更相似。

`tail3_uniform`：

- `spread`：tail centroid 两两 cosine 的平均值 `-0.222`
- `packed_x_pos`：tail centroid 两两 cosine 的平均值 `+0.931`

`tail4_uniform`：

- `spread`：tail centroid 两两 cosine 的平均值 `-0.203`
- `packed_x_pos`：tail centroid 两两 cosine 的平均值 `+0.832`

这说明，当 tail group 频率相同时，模型不会自动选择唯一的“语义几何”；起始几何会强烈影响最终表征。

## 实验族

一键 sweep 脚本会运行三类实验：

- `tail3`：一个 common group 加三个 tail group。
  - tail 内部分别测试 uniform 和 Zipf-like 频率分布。
  - 初始化分别测试 spread、packed +x、packed -x、packed +y。

- `tail4`：一个 common group 加四个 tail group。
  - 总共五个 group，用来提高二维表征空间的方向压力。
  - 分别测试 uniform 和 Zipf-like tail 分布。

- `single_tail`：一个 common group 加一个 tail group。
  - common group 从 +x 附近出发。
  - tail group 分别从 +x、-x、+y、-y 出发，用来测试最终 tail 方向是否存在稳定吸引子。

新增的后续实验分成两条线：

- `toy3d`：三维 toy bigram 实验。
  - 只做三维，因为三维仍然可以直接可视化。
  - 1.1：`common + 3 uniform tail`，观察 common 占据主方向后，tail 是否在剩余二维 residual plane 中展开。
  - 1.2：`common + 3 Zipf tail`，观察 residual plane 内部是否继续出现 tail-high / tail-low 竞争。
  - 每个 run 输出 3D 轨迹图、去掉 common 方向后的 residual plane 图、训练曲线和 `summary.json`。

- `transformer_embedding_init`：Transformer 上的受控 embedding 初始化与谱占据分析。
  - 2.1：构造 `spread` / `packed_common` / `packed_negative_common` 三种 token embedding 初始化。
  - 2.2：读取训练后的 checkpoint，分析 `tok embedding`、`lm_head`、每层 hidden state 的 head/middle/tail spectral occupation。
  - 2.4：比较不同 embedding 初始化是否导致不同的最终表征几何。
  - 这个实验不直接修改 `fdong` 下的训练代码，而是在 `fdong_embedding_dim` 中生成 init checkpoint，再调用现有训练入口从该 checkpoint 开始训练。

- `toy_gradient_interference`：二维 / 三维 toy 上的训练动态机制分析。
  - 目的不是再证明最终 loss 现象，而是区分“初始化几何直接预测结果”和“梯度干扰 / effective rank 动态解释”。
  - 只比较 `dim=2` 和 `dim=3`，避免进入不可直接可视化的更高维。
  - 对 `spread` / `packed_common`、tail 内部 `uniform` / `Zipf` 做 8 个 run。
  - 每个 checkpoint 分别计算 common、tail1、tail2、tail3 的 group-conditioned gradient。
  - 记录 tail gradient effective rank、all gradient effective rank、tail representation residual effective rank、tail SIR、tail-tail gradient cosine、common-tail gradient cosine。
  - 如果 packed 初始化导致 tail 学不好是因为 effective dimension 没有被真正用起来，那么应该看到 raw dimension 从 2 到 3 后，packed run 的 tail representation / gradient effective rank 仍然偏低。

## 输出文件

每个 run 会输出：

- `01_E_snapshots_real_scale.png`
- `02_E_all_trajectories_real_scale.png`
- `03_E_singular_values.png`
- `04_training_metrics.png`
- `summary.json`
- `config.json`

`summary.json` 中最有用的定量字段包括：

- 每个 group 最终的 loss / accuracy / margin
- 最终奇异值和 `sigma1 / sigma2`
- group centroid norm
- group centroid 与奇异向量的 cosine
- group centroid 两两之间的 cosine

全量实验输出默认写入 `fdong_embedding_dim/outputs/`，这个目录不进入 git。适合跨机器同步和讨论展示的轻量结果放在：

```text
fdong_embedding_dim/curated_results/
```

其中包含关键 2D / 3D 轨迹图、3D residual plane 图、梯度干扰机制图和轻量汇总表。

## 运行方式

从仓库根目录运行：

```bash
fdong_embedding_dim/scripts/run_embedding_dim_sweeps.sh
```

三维 toy 实验：

```bash
fdong_embedding_dim/scripts/run_toy_3d_experiments.sh
```

Transformer 受控 embedding 初始化实验：

```bash
fdong_embedding_dim/scripts/run_transformer_embedding_init_experiments.sh
```

二维 / 三维 toy 梯度干扰机制分析：

```bash
fdong_embedding_dim/scripts/run_toy_gradient_interference_experiments.sh
```

常用覆盖参数：

```bash
STEPS=3000 fdong_embedding_dim/scripts/run_embedding_dim_sweeps.sh
OUTDIR=fdong_embedding_dim/outputs/my_sweep fdong_embedding_dim/scripts/run_embedding_dim_sweeps.sh
RUN_FILTER=single_tail STEPS=1000 fdong_embedding_dim/scripts/run_embedding_dim_sweeps.sh
STEPS=500 fdong_embedding_dim/scripts/run_toy_3d_experiments.sh
RUN_FILTER=packed_common TOTAL_TRAINING_STEPS=500 fdong_embedding_dim/scripts/run_transformer_embedding_init_experiments.sh
STEPS=500 RECORD_EVERY=10 fdong_embedding_dim/scripts/run_toy_gradient_interference_experiments.sh
```
