# CRS (Common-Residual Split) 实验总结

## 问题动机

在语言模型中，高频 token（`the`, `a`, `of` 等）占据了序列中 20-30% 的位置。这些 common token 的 embedding 方向在训练过程中通过梯度外积被不断强化，形成参数矩阵的**主导奇异方向**。这些大奇异方向挤占了小奇异方向的优化空间，导致长尾 token 学习缓慢。

**CRS 假设**：如果我们将 hidden state 沿着「序列前 t 个 token 的累积均值方向」（h_common）和它的正交补（h_residual）拆开，分别用不同的参数矩阵处理，就可以让 common 分支吸收大奇异方向的能量，残差分支获得更均匀的谱结构，从而加速长尾 token 的学习。

## 模型结构

每个 TransformerBlock 的 attention 和 MLP 都做拆分：

```
h[t] = hidden state at position t
h_common[t] = cummean(h[0:t+1], dim=1).detach()

# Common branch: rank-p bottleneck (W_down: d×p, W_up: p×d_out)
branch_out = W_up(W_down(h_common))

# Residual branch: full matrix (standard attention / SwiGLU MLP)  
residual_out = FullMatrix(h)

output = α * branch_out + β * residual_out
```

关键设计：
- h_common 使用 `detach()` 切断梯度，使 common 分支只「看」而不影响 hidden state 的表示学习
- Common 分支的 rank-p 结构限制了其表达能力，防止过度学习
- α 控制 common 分支贡献强度，β 控制残差分支放大倍数

## 实验配置

### 合成数据
- 500 token 词表：10 个 K（common，每个频率 ~3%）+ 490 个 R（rare）
- 50-5000 个固定 pattern，每个长度 10（3 K + 7 R）
- 测试集使用不同随机种子生成的新 pattern，评估泛化能力

### 模型规模
| 配置 | d_model | 层数 | 参数量 |
|---|---|---|---|
| 大规模 | 128 | 1 | 261-268K |
| 中规模 | 64 | 1 | 81-85K |
| 小规模 | 32 | 1 | 28-29K |

### 消融实验维度
- **α**（common 分支权重）：0.3, 0.5, 0.8, 1.0
- **β**（残差分支放大）：1.0, 1.5, 2.0, 2.5, 3.0
- **Freeze common**：步数 50、200 时冻结 common 分支，只训残差
- **瓶颈 rank p**：4（d=32/64 时）、8（d=128 时）

## 关键发现

### 1. 谱结构：CRS 确实降低了残差分支的 σ₁

在所有参数配置下，CRS 残差分支的 top 奇异值均低于 baseline：

| 模型规模 | Baseline σ₁ | CRS 最优 σ₁_r | σ₁ 降低 |
|---|---|---|---|
| d=32, α=0.5 | 1.185 | 0.892 | -25% |
| d=64, α=0.5 | 0.905 | 0.886 | -2% |
| d=128, α=0.5 | 1.101 | 1.055 | -4% |

**模型越小（相对任务越受限），CRS 的谱分离效果越显著。**

同时，残差分支的有效秩在 d=32 时从 12.8 提升到 16.2（+27%），d=64 时从 28.4 提升到 29.6（+4%）。

### 2. α 的最优区间为 0.3-0.5

α 过低（0.1）common 分支太弱、无法吸收足够能量；α 过高（≥0.8）common 分支反而成为 rank-1 扰动叠加到残差上，使 σ₁ 回升。

### 3. β 放大无益，甚至有害

β>1 虽然进一步压低了 σ₁（β=1.5 时 σ₁=0.823，↓31%），但代价是有效秩从 16.2 降至 13.7。β 只是线性缩放整个谱，没有改善谱分布的形状。

### 4. Freeze common 破坏协同演化

在任意步数冻结 common 分支都会降低残差的有效秩（freeze@200: effR 从 16.2→14.0），且无法改善 loss。Common 和残差分支需要**全程耦合优化**。

### 5. 谱优势未转化为预测优势

无论在 200 pattern 还是 5000 pattern、无论是否测试泛化，**CRS 的 loss 和 accuracy 从未超过 baseline**。最优 CRS 配置（α=0.5, β=1.0）在早期比 baseline 慢 5-17%，差距随训练缩小但从未反转。

## 解释与讨论

**大奇异方向可能是 feature，不是 bug。** 在真实的梯度优化动态中，模型需要将大部分容量集中在少数主导方向上才能高效降低 loss。CRS 通过压制 σ₁ 换来的谱均衡，实际上砍掉了梯度最陡的优化方向——谱结构确实更好了，但学习效率反而下降了。

这个结果与我们在真实数据（109M 模型、DCLM 语料）上的观察一致：step 3000 时 CRS 的 loss 同样不优于 baseline，尽管残差分支的 σ₁ 更低。

## 可能的后续方向

1. **推迟 CRS 引入**——训练初期不做拆分，等模型学到一个「有用的大奇异方向」后再触发 CRS 分离
2. **自适应 α**——让 α 从 0 开始增长，先让大方向充分建立，再逐步拆分
3. **仅在特定层应用 CRS**——浅层可能需要大奇异方向做特征提取，深层才受益于谱均衡
4. **更大规模、更长训练**——109M 模型 5000 步可能不够，需 20K+ 步观察长期效应

## 文件清单

| 文件 | 说明 |
|---|---|
| `workbuddy_scripts/small_lm_crs.py` | 真实数据训练代码（109M 模型，断点续训，TeeLogger） |
| `workbuddy_scripts/synthetic_crs_experiment.py` | 合成数据实验（可配置 d_model/α/β/p/freeze） |
| `workbuddy_scripts/analyze_step3000.py` | 109M 模型 step 3000 对比分析 |
| `workbuddy_scripts/analyze_synthetic_checkpoints.py` | 合成实验 checkpoint SVD 分析 |
| `workbuddy_scripts/accuracy_analysis.py` | K/R 准确率分项评估 |
| `workbuddy_scripts/check_training.py` | 训练监控脚本 |
| `workbuddy_scripts/run_*.sh` | 批量运行脚本 |
| `outputs/synthetic_crs/*/metrics.jsonl` | 各配置的训练曲线 |
| `outputs/synthetic_crs/*/final_analysis.json` | 最终 SVD 分析结果 |
| `outputs/small_lm_crs/*/metrics.jsonl` | 真实数据训练曲线 |
| `outputs/small_lm_crs/step3000_analysis.json` | 109M 模型 step 3000 详细分析 |
