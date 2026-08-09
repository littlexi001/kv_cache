# 64K 以上质量退化的机制与 Mass-Ladder QKSieve

更新日期：2026-08-03

## 1. 问题

旧 QKSieve 在 32K/64K 使用固定的 1,280 token/head 预算时质量较好，但在更长文本或困难请求上会退化。需要回答的不是“64K 之后换哪个参数”，而是：

1. 质量为什么随长度下降；
2. 哪个运行时数值能够直接反映当前 query 的难度；
3. 能否不使用任务标签、学习 router 或人工长度阈值，自然得到每层、每个 head 的预算；
4. 能否同时保持可解释性和实际速度。

## 2. 已确认的因果链

### 2.1 64K 不是突变点

固定 exact top-1,280 时，遗漏的 attention mass 随长度平滑增加：

| 历史长度 | 遗漏 attention mass |
|---:|---:|
| 32K | 6.544% |
| 64K | 8.196% |
| 96K | 10.762% |
| 128K | 12.497% |

对应的 exact-top-k KL 也从 0.00774 平滑增加到 0.01645。数据中没有 64K 突变，64K 只是旧配置开始明显暴露固定预算不足的位置。

### 2.2 QK 子空间没有在 64K 后崩溃

QK 乘积谱仍然高度集中，前 48 维能量占比只从约 97.7% 缓慢下降到 97.4%。低比特 proxy 的 RMSE 也基本稳定。因此当前退化不能主要归因于“PCA 表征在 64K 后突然失效”。

长度增加仍会让排序更困难：

- `max score error / RMSE` 从约 4.50 增加到 5.38；
- top-k 边界间隔变小；
- 边界附近的模糊 token 从约 456 增加到 560；
- proxy top-k recall 从约 93.75% 降到 91.87%。

但替换实验表明，这部分损失小于 exact top-1,280 相对 Full 的损失。主要矛盾仍是固定预算遗漏了越来越多 softmax mass，QK crossing 是次要放大项。

### 2.3 Value-tail 决定遗漏 mass 是否真正伤害输出

对一个 attention head，将选中集合记为 `S`，遗漏集合记为 `T`，Full attention 在 `S` 上的精确概率质量记为 `p`。令两部分的条件 Value 均值为 `mu_S` 和 `mu_T`：

```text
o_full   = p * mu_S + (1 - p) * mu_T
o_sparse = mu_S

o_sparse - o_full = (1 - p) * (mu_S - mu_T)
```

因此误差由两个连续量共同决定：

```text
tail_mass     = 1 - p
value_contrast = ||mu_S - mu_T||
```

这解释了两个现象：

- 相同 token 比例在不同主题、层和 head 上质量不同；
- 只增加 Key 位宽不能修复已经被候选集合遗漏的 Value 贡献。

## 3. 为什么固定 k 不可能对任意长度通用

考虑没有显著 needle 的 diffuse score bulk。若 bulk score 的指数矩存在，则总 partition 通常随 token 数近似线性增长：

```text
Z_bulk = sum_i exp(z_i) = Theta(N)
```

固定 `k` 个极值的 score 只按极值统计速度增长。以次高斯分布为例：

```text
Z_topk <= k * exp(O(sqrt(log N)))
```

所以：

```text
p_topk = Z_topk / Z_all
       <= k * exp(O(sqrt(log N))) / Theta(N)
       -> 0,  N -> infinity
```

这不是对真实 attention 分布作严格独立高斯假设，而是一个反例定理：只要上下文包含不会随长度消失的 diffuse bulk，固定 token 数就没有任意长度质量保证。因此最终方法必须让预算由当前 attention 分布决定。

## 4. 方法：Measured-Mass Ladder QKSieve

### 4.1 低比特 QK proxy

每个请求在 prefill 中根据当前 Q/K 二阶矩建立 request-local QK-balanced 坐标。当前低比特 Key 索引使用按 band 分配的整数编码；典型逻辑位宽分配为：

```text
[4, 1, 1, 1, 1, 1, 0, 0]
```

完整 Key 索引连同 scale 元数据约为 Full FP16 K/V 的 5.859%。Value-tail rank-16 INT4 sketch 约为 1.611%，总辅助索引约为 7.471%。原始 FP16 K/V 仍在 GPU 上，最终选中 token 使用 exact QK 和原始 Value。

### 4.2 几何预算梯子

对当前 query、layer 和 head 定义低比特分数 `s_tilde_i`。候选预算为：

```text
k_0 = 1280
k_j = min(N, ceil(1.5 * k_(j-1)))
```

运行时执行三步：

1. 在 1,024 个确定性均匀采样位置上计算 proxy score，只用于估计每个 `k_j` 的排序阈值；
2. 对全部低比特 Key 做一次 proxy QK 扫描，将每个 token 的 `exp(s_tilde_i)` 累积到阈值定义的质量 bin；
3. 选择第一个累计 proxy softmax mass 达到 `tau` 的 rung，当前候选 `tau=0.90`。

形式化地，选择：

```text
j* = min { j : M_tilde(S_j) >= tau }

M_tilde(S) = sum_(i in S) exp(s_tilde_i)
             / sum_i exp(s_tilde_i)
```

关键点是：采样只估计 rank threshold，softmax mass 由完整 proxy 扫描测量。即使采样漏掉一个罕见高分 token，该 token 仍会进入完整扫描的质量统计；采样误差主要改变实际候选数量和梯子粒度，而不会像“只在样本上估计 softmax”那样直接漏掉 partition 质量。

### 4.3 精确异常项加低秩 bulk

选中集合使用原始 K/V 计算 exact attention。遗漏集合不直接丢弃，而是用 rank-16 INT4 Value sketch 估计 tail numerator 和 denominator：

```text
Z_T_tilde = sum_(i in T) exp(s_tilde_i)
N_T_tilde = sum_(i in T) exp(s_tilde_i) * v_tilde_i

o_hat = (N_S_exact + N_T_tilde)
        / (Z_S_exact + Z_T_tilde)
```

它把 attention 看成“少量稀疏异常项 + 可压缩的稠密 bulk”，而不是强行假设遗漏 token 的贡献为零。

### 4.4 输出误差分解

令 tail partition 和 numerator 误差为：

```text
Delta_Z = Z_T_tilde - Z_T
Delta_N = N_T_tilde - N_T
```

则有精确关系：

```text
o_hat - o_full
= (Delta_N - o_full * Delta_Z)
  / (Z_S + Z_T + Delta_Z)
```

以及：

```text
||o_hat - o_full||
<= (||Delta_N|| + ||o_full|| * |Delta_Z|)
   / (Z_S + Z_T_tilde)
```

该分解对应三个可独立测量的误差源：

1. 预算不足导致的 tail mass；
2. QK proxy 导致的 partition 误差；
3. Value sketch 导致的 numerator 误差。

### 4.5 proxy mass 与 exact mass 的扰动界

若所有 score 误差满足：

```text
|s_i - s_tilde_i| <= epsilon
```

且 proxy 选中质量为 `m_tilde`，则 exact 遗漏质量满足：

```text
1 - m_exact
<= exp(2 * epsilon) * (1 - m_tilde)
   / (m_tilde + exp(2 * epsilon) * (1 - m_tilde))
```

这个界说明提高 proxy mass 可以连续抵消 QK 误差，但逐 token Cauchy 范数给出的 `epsilon` 在真实数据上过松。128K 实验中，实际 score 误差均值约 0.203、最大约 1.395，而 Cauchy 上界均值约 13.24、最大约 106.4，最终会选择接近 100% token。因此它只保留为理论基线，不作为部署规则。

## 5. 当前实验结果

### 5.1 real-QKV 层输出诊断

冻结 `tau=0.90、growth=1.5、floor=1280`：

| 设置 | 自动 token 比例 | exact attention mass | rank-16 INT4 tail 后层输出相对 L2 |
|---|---:|---:|---:|
| 4K religion | 32.60% | 98.41% | 0.856% |
| 32K medicine | 7.60% | 95.29% | 1.016% |
| 32K sports | 9.14% | 94.39% | 1.401% |
| 96K sports | 3.07% | 94.07% | 1.187% |
| 96K medicine | 3.34% | 95.05% | 1.159% |
| 128K computer | 5.25% | 95.21% | 1.001% |

96K 是未参与初始长度选择的中间长度，结果支持预算由当前分布连续决定，而不是依赖 64K 分支。

### 5.2 真实模型闭环 PPL

4K religion、32 个 token：

| 方法 | PPL | 相对 Full | token 比例 | steady decode |
|---|---:|---:|---:|---:|
| Full | 16.5295 | 100.000% | 100% | 1.000x |
| 固定 1,280 | 16.5334 | 99.976% | 32.27% | 0.827x |
| Mass ladder | 16.5303 | 99.995% | 32.25% | 0.788x |

32K 三个 held-out 自然主题、每个 64 token：

| 主题 | 相对 Full | token 比例 | steady decode |
|---|---:|---:|---:|
| computer | 100.068% | 5.51% | 1.455x |
| space | 100.087% | 5.41% | 1.470x |
| politics | 100.078% | 4.76% | 1.562x |

这三个主题使用相同冻结规则，没有任务标签或主题参数。

### 5.3 必须保留的反例

64K 受控两跳检索的两个新 seed 中，mass ladder 都优于固定 1,280，但尚未恢复 Full logits：

| seed | 固定 1,280 质量 | Mass ladder 质量 | Mass ladder token 比例 | gold NLL 增量 |
|---:|---:|---:|---:|---:|
| 20260881 | 79.34% | 91.09% | 3.59% | +0.187 |
| 20260882 | 96.76% | 97.35% | 3.75% | +0.054 |

Full 本身在这两个样本上也没有预测正确 gold token，因此这里主要衡量对 Full logits 的保真度，而不是任务准确率。该反例正在通过 exact-QK、QK proxy、Value-tail 和多预算对照拆解。

### 5.4 当前最大实现缺口

当前原型仍设置了 25% 的最大 rung。由于几何梯子会越过该值，实际预分配容量可达约 30%--43%。在 32K held-out 和 64K 两跳测试中，仍有约 0.4%--0.6% 的 head 未达到 90% proxy mass，最小值约 0.71--0.87。

最终方法不能把这个上限当作质量规则。需要将其改成纯系统资源约束，并使用 packed ragged candidate 或按 head 的解析成本调度处理 diffuse head。

## 6. 当前速度结论

独立 CUDA 随机压力测试中，mass-ladder 的“完整 proxy mass 扫描”已经替代全量 score materialization 加 `torch.sort`：

| 长度 | CUDA mass ladder | materialize + sort | 加速 |
|---:|---:|---:|---:|
| 32K | 0.312 ms | 0.482 ms | 1.54x |
| 64K | 0.399 ms | 0.891 ms | 2.23x |
| 128K | 0.523 ms | 1.657 ms | 3.17x |

但当前整模型实现随后又扫描一次低比特 Key，以生成候选并累计 Value-tail，因此 32K mass-ladder decode 只有约 1.46x--1.56x，低于旧固定预算的约 2.09x。质量规则已成立，不代表系统实现已最优。

下一版内核应在第一次扫描中同时完成：

1. score bin 的 partition mass；
2. 每个 bin 的 Value-sketch numerator；
3. 最大 rung 候选的 index 和 bin id。

选择 rung 后只对最大候选池做轻量过滤，不再第二次计算 proxy QK。正式测速必须分别报告：

- attention 子系统：query 投影、mass scan、候选生成、exact sparse attention、Value-tail；
- 整模型 steady decode：固定上下文、足够多生成步、排除索引构建；
- 索引构建与多轮 Agent 复用下的 break-even。

## 7. 失败条件与论文边界

最终验证必须主动覆盖：

1. attention 极度 diffuse，需要大量 token 才达到目标 mass；
2. 少量 exact 高分 token 被 QK proxy 严重低估；
3. Value 残差与 score 正相关，rank-16 sketch 漏掉关键方向；
4. 重复、近重复和计数任务形成很密的 top-k 边界；
5. 早期层和个别 head 的质量需求远高于平均值；
6. 短文本中索引与双扫描开销超过 dense attention；
7. 256K 以上 request-local 坐标漂移、RoPE 外推或数值范围变化；
8. 不同模型、GQA 比例和 head dimension。

当前可以主张：固定 1,280 的主要长度退化来自遗漏 softmax mass；measured-mass ladder 是一个无任务标签、无人工长度边界、可解释的预算规则，并已在自然文本和 real-QKV 中得到正证据。

当前不能主张：已经得到任意长度、任意请求都保持 99.5% 质量的最终方法；64K 两跳反例和 25% 最大 rung 尚未闭合，单扫描融合也尚未进入真实模型测速。
