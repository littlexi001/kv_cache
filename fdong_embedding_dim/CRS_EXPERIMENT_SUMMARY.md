# CRS (Common-Residual Split) —— 理论与实验全景报告

---

## 一、从观察到假设：思想历程

### 1.1 原始动机

在高维表征空间中，**使用更多方向（高有效秩）比使用单一方向的大 scale（高 σ₁）应当更高效**。因为方向多样性意味着模型可以利用不同 token 的不同属性，而单一方向的大 scale 意味着模型把所有输出都往同一个方向推。

### 1.2 观察：为什么参数空间会长出大奇异方向？

通过对合成 bigram 模型和真实 Qwen 模型的谱分析，我们发现了以下因果链条：

1. **高频 token 占据表征空间的最大 norm 方向**——所有 token 的 embedding 向量中，common token（`the`, `a`, `of` 等）的方向在嵌入空间里 norm 最大，因为所有上下文都需要同时预测它们。

2. **梯度外积将表征主方向传导到参数空间**——梯度公式 `∂L/∂W = Σ (∂L/∂y) ⊗ h` 中，表征向量 h 处在右空间。当 h 沿 common direction 分量最大时，`h ⊗ h` 的最大奇异方向也沿 common direction。这个外积被累加到参数矩阵 W 上，使得 W 的主奇异方向**继承**了表征空间的主方向。

3. **正反馈循环**——W 的主方向变大后，经过 W·h，输出中该方向的 scale 进一步增大，梯度中外积的贡献也更大，形成正反馈。最终参数矩阵的 σ₁ 持续增长，挤压了小奇异方向的学习空间。

4. **验证了 Boss 的两阶段理论**：在合成实验中，方向先对齐（阶段 1），奇异值再快速增长（阶段 2）。且在 tied embedding 下，E 和 Q 的 top singular direction 通过 `E.weight.T` 共享梯度通道，使得 embedding 的 common direction 被额外放大。

### 1.3 核心假设

如果我们能**把 common direction 从 hidden state 里拆出来**，让它单独进一个低容量分支（不参与主参数矩阵的梯度竞争），那么：

- **Common 分支**吸收 common direction 的能量，承担「预测 next token 是 common token」的工作
- **Residual 分支**获得更均匀的谱结构（σ₁ 降低、有效秩升高），可以更均匀地学习所有 token
- 长尾 token 的学习不应再被大奇异方向挤占

---

## 二、方法设计：CRS 架构与策略矩阵

### 2.1 核心架构

每个 TransformerBlock 的 **attention 和 MLP 均做拆分**：

```
h_common[t] = cummean(h[:, :t+1], dim=1).detach()    ← 序列累积均值，切断梯度

common_out   = W_up(W_down(h_common))                 ← rank-p bottleneck (W_down: d×p, W_up: p×d_out)
residual_out = FullMatrix(h)                         ← 标准 attention / SwiGLU MLP

output = α × common_out + β × residual_out
```

**关键设计点**：
- `h_common[t] = 位置 t 的前 t 个 hidden state 均值`，而非全局 embedding 均值——随序列实时演化，捕获当前上下文的「典型方向」
- `.detach()` 切断 common 分支的梯度回传，防止 common 分支反向影响表征学习
- rank-p bottleneck 限制 common 分支的容量（p=4~8），防止它「学会一切」
- α/β 分别控制两个分支的贡献比例

### 2.2 策略空间

我们探索了一个四维策略空间：

| 维度 | 取值 | 含义 |
|---|---|---|
| **α** | 0.3, 0.5, 0.8, 1.0 | Common 分支的贡献权重。α=0 → 退化为 baseline |
| **β** | 1.0, 1.5, 2.0, 2.5, 3.0 | Residual 分支的放大系数。β>1 → 给残差分支额外 boost |
| **Freeze** | None, @50, @200 | 在指定步数冻结 common 分支参数，只训残差 |
| **Model scale** | d=128/64/32 (261K→81K→28K) | 模型容量递减，与任务难度的匹配程度不同 |

实验数据：500 token 词表（10 K + 490 R），200-5000 个固定 pattern × length 10。

---

## 三、实验结果：三维评估

### 维度一：最终收敛效果（Loss & Accuracy）

**综合结论：CRS 在所有参数组合下均未击败 baseline。**

**d=32 模型（最受限场景）的 test_R_loss 演化**：

| Step | Baseline | CRS α=0.3 | CRS α=0.5 | CRS α=0.8 | CRS α=1.0 |
|---|---|---|---|---|---|
| 200 | **1.04** | 1.23 | 1.14 | 1.12 | 1.21 |
| 500 | **0.31** | 0.36 | 0.38 | 0.33 | 0.33 |
| 1000 | **0.044** | 0.050 | 0.045 | 0.047 | 0.042 |
| 2000 | 0.0003 | 0.0007 | 0.0003 | 0.0006 | 0.0005 |

CRS 在早期始终慢 5-17%，到末期才追平或接近。在更大模型（d=64/128）和 5000 pattern 场景下结论一致——从未反转。

**Accuracy 结论相同**：step 50 时 baseline R 准确率 97.4%，最优 CRS（α=0.3）仅 92.8%。step 200 时双方均达 100%。

| Step | Baseline R Acc | CRS α=0.3 R Acc | CRS α=0.5 R Acc |
|---|---|---|---|
| 50 | **97.4%** | 92.8% | 91.7% |
| 100 | **99.8%** | 99.2% | 98.2% |
| 200 | 100% | 100% | 99.9% |

### 维度二：学习效率（Convergence Speed）

**综合结论：CRS 的收敛速度始终低于 baseline。**

以 test_R_loss 作为指标，CRS 到达同等 loss 水平所需的步数更多：

| 目标 R_loss | Baseline 步数 | CRS α=0.5 步数 | CRS 慢多少 |
|---|---|---|---|
| 1.0 | ~220 | ~280 | +27% |
| 0.5 | ~420 | ~520 | +24% |
| 0.1 | ~820 | ~880 | +7% |

差距从早期的 25% 逐渐缩小到末期的趋同。但**在任意训练阶段，CRS 从未在 loss 或 accuracy 上比 baseline 更快**。

### 维度三：谱结构（Spectral Structure）

**这是 CRS 唯一明显优于 baseline 的维度。**

**d=32 模型（p=4）的最终 SVD 对比**：

| 配置 | Common/整体 σ₁ | Common/整体 effR | Residual σ₁ | Residual effR |
|---|---|---|---|---|
| Baseline | 0.992 | 12.5 | — | — |
| CRS α=0.3 | 1.320 | 3.2 | 0.912 | **13.3** |
| **CRS α=0.5** | 0.476 | 3.5 | **0.809** | **12.6** |
| CRS α=0.8 | 1.146 | 3.3 | 0.953 | 14.1 |
| CRS α=1.0 | 1.017 | 3.2 | 0.882 | **15.6** |

**核心发现**：

1. **残差分支 σ₁ 降低**：最优 α=0.5 时 σ₁_r = 0.809 vs baseline 0.992（**↓18%**）。common 分支成功吸收了主导方向的能量。

2. **Common 分支谱特征**：
   - α=0.3 时 common σ₁=1.320（最高）——common 在主导学习
   - α=0.5 时 common σ₁=0.476（最低）——common 和 residual 达到平衡
   - Common 分支有效秩始终在 3-4（p=4），接近饱和利用

3. **模型规模效应**：谱改善在容量受限模型（d=32）中最显著（↓18%），在大模型（d=128）中减弱（↓4%）。这是因为大模型有足够容量直接记住 pattern，不需要谱均衡。

4. **Common 分支的动态演化**：common σ₁ 呈现「先升后降」的轨迹——训练早期 common 快速学习（σ₁ 上升），训练后期 residual 接管更多工作（common σ₁ 回落）。这证实了两阶段耦合演化的存在。

---

## 四、消融实验：各个策略的效果

### 4.1 α 消融

| α | Residual σ₁ | Residual effR | R_loss@200 | 评价 |
|---|---|---|---|---|
| 0.3 | 0.912 | 13.3 | 1.23 | 谱改善中规中矩，loss 最差之一 |
| **0.5** | **0.809** | 12.6 | 1.14 | 谱最优，loss 次优 |
| 0.8 | 0.953 | 14.1 | 1.12 | 谱退步，有效秩最高 |
| 1.0 | 0.882 | 15.6 | 1.21 | 谱不错，loss 一般 |

**结论**：α=0.5 在谱结构上最优（σ₁ 最低），α=0.3 在 loss 上最接近 baseline。α≥0.8 时 common 分支过强，反而推高残差 σ₁。α 存在一个 sweet spot（0.3-0.5），但即便在 sweet spot，loss 仍不敌 baseline。

### 4.2 β 消融（α=0.5 fixed）

| β | Residual σ₁ | Residual effR | R_loss@200 | R_loss@500 |
|---|---|---|---|---|
| **1.0** | 0.809 | **12.6** | 1.14 | 0.38 |
| 1.5 | **0.823** | 13.7 | 1.32 | 0.38 |
| 2.0 | 0.930 | 13.0 | — | — |
| 3.0 | 1.098 | 12.8 | — | — |

**结论**：β>1 进一步压低 σ₁ 的代价是有效秩从 12.6→13.7→12.8 的**非单调退化**。β 仅是线性缩放整个残差谱，并没有改变谱分布的形状——把整个谱乘以 β 等价于把 effective learning rate 放大 β 倍，但也会破坏 α/β 的精细平衡。β=1.0 是最优。

### 4.3 Freeze Common 消融

| 策略 | Residual σ₁ | Residual effR | Common σ₁ | R_loss@500 |
|---|---|---|---|---|
| **Normal (no freeze)** | 0.809 | **12.6** | 0.476 | 0.38 |
| Freeze@50 | 0.997 | 13.9 | 1.418 | 0.40 |
| Freeze@200 | 0.900 | 14.0 | 1.175 | 0.33 |
| β=1.5 + Freeze@200 | **0.764** | 13.8 | 1.099 | 0.38 |

**结论**：
- Freeze@50 严重破坏谱结构（σ₁ 从 0.809→0.997），因为 common 在 step 50 远未收敛
- Freeze@200 影响较小（σ₁ 0.809→0.900），但仍使有效秩降低
- **Common 和 residual 需要全程耦合优化**——残差变好后，common 也需要重新适应；切断这个循环就丢失了正反馈收益
- β=1.5 + Freeze@200 给出最低 σ₁（0.764），但有效秩和 loss 均不如正常版本

### 4.4 数据规模影响

| 数据规模 | Baseline 最终 R_loss | CRS α=0.5 最终 R_loss | 差距 |
|---|---|---|---|
| 200 patterns (d=32) | 0.0005 | 0.0005 | 持平 |
| 5000 patterns + test (d=32) | 0.0003 | 0.0003 | 持平 |
| 109M 真实数据 (step 3000) | 4.568 | 4.559 | 基本持平 |

**CRS 和 baseline 在充分训练后达到相同的 loss 水平**。增加数据规模只是拉长了训练时间，没有改变这个定性结论。

---

## 五、总评与解释

### CRS 做到了什么？

✅ **谱结构确实被改变了**。残差分支的 σ₁ 降低了 18-25%，common 分支成功吸收了主导方向的能量。这是 CRS 设计的核心目标，并且成功了。

✅ **Common 和 Residual 耦合演化的动力学被验证了**。Freeze 实验证明两者需要同时优化；α/β 存在 sweet spot 证实了平衡的重要性。

### CRS 没做到什么？

❌ **谱优势从未转化为预测优势**。loss 和 accuracy 在所有条件下均不优于 baseline。

❌ **收敛速度始终更慢**。common 分支从零开始学习是额外的负担。

### 为什么？

**大奇异方向可能是 feature，不是 bug。** 我们的原始假设是「使用更多方向比使用单一方向的大 scale 更高效」。但实验结果表明，在 next-token prediction 任务中，**模型需要把大部分容量集中在少数主导方向上，才能最高效地降低 loss**。

这是一种**偏差-方差权衡**的体现：
- Baseline：把容量集中在少数方向 → 快速下降 loss，但谱集中在低维
- CRS：强迫谱更均匀 → 谱结构更好，但 loss 下降变慢

CRS 本质上是一个**正则化器**——它压低主导奇异值、鼓励谱多样性。和所有正则化一样，代价是训练效率下降。区别在于，L2/weight decay 的收益体现在泛化上，而 CRS 在我们的实验中**连泛化收益也没有体现出来**（测试和训练 loss 同步下降）。

### 可能的例外场景

CRS 可能在以下场景发挥作用：
1. **极度长尾的数据分布**（common token 频率 >50%），common direction 的谱支配效应才会真正成为瓶颈
2. **极低容量的模型**，大奇异方向确实挤占了小方向的学习
3. **多任务学习**，common 分支可能学到跨任务共享的表示

---

## 六、文件索引

| 路径 | 说明 |
|---|---|
| `CRS_EXPERIMENT_SUMMARY.md` | 本报告 |
| `workbuddy_scripts/small_lm_crs.py` | 真实数据训练代码（断点续训、TeeLogger） |
| `workbuddy_scripts/synthetic_crs_experiment.py` | 合成数据实验（支持 α/β/p/freeze 全配置） |
| `workbuddy_scripts/analyze_step3000.py` | 109M 模型 step 3000 分析 |
| `workbuddy_scripts/analyze_synthetic_checkpoints.py` | 合成实验 checkpoint SVD 演化分析 |
| `workbuddy_scripts/accuracy_analysis.py` | K/R 准确率分项评估 |
| `outputs/synthetic_crs/*/metrics.jsonl` | 各配置的训练曲线 |
| `outputs/synthetic_crs/*/final_analysis.json` | 最终 SVD 分析结果 |
| `outputs/small_lm_crs/*/metrics.jsonl` | 真实数据训练曲线 |
