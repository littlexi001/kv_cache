# 低比特量化与 pre-RoPE 数值检索实验结论

## 1. 最终结论

这轮实验回答了两个问题。

第一，低比特浮点在离线 attention mass 上有约 0.5 个百分点的优势，但没有转化为真实模型 PPL 优势。128-bit 和约 80-bit 两个码率下，浮点版本都略差于对应的整数版本。因此不值得继续开发专用 minifloat LUT/CUDA kernel。

第二，把 RoPE 从“候选检索”和“最终 attention”中解耦是有效机制，但固定 32 维的 pre-RoPE 索引只在 32K 左右形成较好的轻量 Pareto 点，不能直接替代 64K 以上的 QKSieve。

当前建议冻结两种可部署配置：

| 使用目标 | 配置 | 32K PPL 质量 | 32K 稳态加速 | 索引/完整 KV |
|---|---|---:|---:|---:|
| 质量优先默认 | QKSieve INT128 `[4,2]` | 99.992% | 1.515x | 3.125% |
| 轻量索引模式 | pre-RoPE 32D INT2 | 99.824% | 1.555x | 1.5625% |

64K 下应继续使用 QKSieve INT128：

| 配置 | 64K PPL 质量 | 64K 稳态加速 | 索引/完整 KV |
|---|---:|---:|---:|
| QKSieve INT128 `[4,2]` | 100.100% | 2.705x | 3.125% |
| pre-RoPE 32D INT2 | 95.578% | 2.826x | 1.5625% |
| pre-RoPE 32D INT4 | 99.196% | 2.652x | 3.516% |

pre-RoPE INT4 在 64K 的质量接近 Full，但索引更大、速度更慢且质量低于 QKSieve，所以被 QKSieve 支配，不进入推荐配置。

## 2. 浮点和整数低比特实验

### 2.1 物理码率

QKSieve 把 128 维 Key 划分为 8 个 16 维 band。每个激活 band 除 payload 外还保存一个 FP16 scale：

```text
bits_per_token_head
  = sum_b(16 * bit_b + 16 * I[bit_b > 0])

index_ratio
  = bits_per_token_head / (2 * 128 * 16)
```

分母 4096 bit 是一个 head 上完整 FP16 K 和 V 的总成本。

### 2.2 真实 32K PPL

实验使用 Qwen3-4B、8 个独立主题、每主题 32 个预测 token，共 256 个 token。所有方法与 Full 严格配对。

| 量化格式 | PPL | 相对 Full 质量 | Top-1 一致率 | 索引比例 | 稳态加速 |
|---|---:|---:|---:|---:|---:|
| Full FP16 | 5.4560 | 100.000% | 100.00% | 0% | 1.000x |
| 整数 128-bit `[4,2]` | 5.4564 | 99.992% | 97.66% | 3.125% | 1.515x |
| minifloat 128-bit `[4,2]` | 5.4674 | 99.791% | 97.66% | 3.125% | 仅质量模拟 |
| 整数 80-bit `[4]` | 5.4806 | 99.551% | 97.27% | 1.953% | 1.522x |
| minifloat 80-bit `[4]` | 5.4988 | 99.222% | 97.27% | 1.953% | 仅质量模拟 |
| 整数 48-bit `[2]` | 5.8874 | 92.672% | 90.63% | 1.172% | 1.520x |

配对比较：

- minifloat128 / int128 的质量比为 99.799%，8 个主题中只改善 2 个；
- minifloat80 / int80 的质量比为 99.670%，8 个主题中只改善 2 个；
- int48 已跨过质量断崖；
- int80 是内存优先 Pareto 点，但质量优先仍应使用 int128。

结论是：minifloat 对代理分数重构有微小帮助，但 top-k crossing、最终 logits 和目标 token NLL 并不只由平均重构误差决定。离线 attention mass 不能替代真实 PPL。

## 3. pre-RoPE 候选、post-RoPE attention

### 3.1 设计

标准 attention 使用旋转后的 Query 和 Key：

```text
score_post(i) = <RoPE(q, t), RoPE(k_i, i)>
```

新分支把候选检索与最终 attention 分开：

1. 从缓存的 post-RoPE Key 逆旋转得到 pre-RoPE Key；
2. 只取 16 个最慢频率对，共 32 个坐标；
3. 对 Key 做 L2 归一化并存为 scale-free INT2，Query 使用 INT8；
4. 用近似 pre-RoPE 语义分数选远程候选；
5. 强制保留 16 个 sink token 和最近 128 个 token；
6. 候选集合确定后，使用原始 post-RoPE FP16 K/V 做精确 attention。

预算不依赖训练 router：

```text
B(N) = min(N, max(256, min(ceil(0.06 * N), 1280)))
```

INT2 索引每个 token/KV-head 为 64 bit：

```text
32 coordinates * 2 bit = 64 bit
64 / 4096 = 1.5625% of full FP16 K/V
```

INT4 索引包含每 token 的 FP16 scale：

```text
32 * 4 + 16 = 144 bit
144 / 4096 = 3.515625%
```

### 3.2 单索引实现

最初质量原型同时构建并扫描 QKSieve 和 pre-RoPE 两个索引，因此 32K 只有 1.28x 到 1.34x。单索引版本已经删除旧 QKSieve 的构建、Query 投影和 proxy scan，只保留：

```text
pre-RoPE index append
-> low-bit score scan
-> local/sink/remote candidate merge
-> exact post-RoPE sparse attention
```

删除重复开销后，INT2 的 32K 稳态加速从 1.342x 提升到 1.555x，质量逐 token 完全不变。

## 4. 真实 PPL 和速度

### 4.1 32K

| 方法 | 相对 Full 质量 | 主题 bootstrap 95% CI | Top-1 一致率 | 索引比例 | ms/token | 加速 |
|---|---:|---:|---:|---:|---:|---:|
| Full | 100.000% | - | 100.00% | 0% | 89.30 | 1.000x |
| QKSieve INT128 | 99.992% | [97.87%, 102.35%] | 97.66% | 3.125% | 58.94 | 1.515x |
| pre-RoPE INT2 | 99.824% | [94.90%, 105.96%] | 93.75% | 1.5625% | 57.43 | 1.555x |
| pre-RoPE INT4 | 99.873% | [96.35%, 104.90%] | 94.92% | 3.516% | 60.43 | 1.478x |

pre-RoPE INT2 相对 QKSieve INT128：

- 索引减少 50%；
- 稳态速度再提高约 2.63%；
- 平均质量比为 99.832%，置信区间跨过 100%，当前样本不能证明二者存在显著质量差。

### 4.2 64K

| 方法 | 相对 Full 质量 | 主题 bootstrap 95% CI | Top-1 一致率 | 索引比例 | ms/token | 加速 |
|---|---:|---:|---:|---:|---:|---:|
| Full | 100.000% | - | 100.00% | 0% | 161.61 | 1.000x |
| QKSieve INT128 | 100.100% | [98.80%, 101.35%] | 96.88% | 3.125% | 59.74 | 2.705x |
| pre-RoPE INT2 | 95.578% | [90.58%, 101.20%] | 89.84% | 1.5625% | 57.18 | 2.826x |
| pre-RoPE INT4 | 99.196% | [96.16%, 102.04%] | 94.92% | 3.516% | 60.94 | 2.652x |

INT2 的损失不是所有主题一致下降：

- medicine：88.06%；
- politics：84.76%；
- mixed_b：91.67%；
- space：112.05%。

这说明固定 32 维慢频子空间对不同 Query 的覆盖不同。仅根据长度切换 INT2/INT4 不足以解决跨主题风险。

## 5. RoPE 机制能否超过 Full

在独立的合成长链检索 benchmark 上，完整 pre-RoPE 候选、post-RoPE 精确 attention 的质量点估计为：

| 长度 | 相对 Full PPL 质量 | Full 正确率 | 方法正确率 |
|---:|---:|---:|---:|
| 8K | 124.8% | 62.5% | 70.8% |
| 16K | 165.8% | 58.3% | 62.5% |
| 32K | 327.6% | 25.0% | 50.0% |
| 64K | 141.1% | 41.7% | 41.7% |

这里的提升来自稀疏候选抑制随长度增长的 softmax distractor，同时 pre-RoPE 分数避免远程语义证据被相对位置相位干扰。

但必须限定结论：

- 这是合成两跳检索任务，不是普通文本 PPL 或 LongBench；
- 原型仍形成完整 pre-RoPE 分数，只用于机制验证；
- 32D INT2/INT4 的可部署近似在 64K 没有复现完整分数的提升；
- 因此可以在论文中写成机制发现和受控验证，不能声称通用任务已经达到 110% Full。

## 6. 被否定的方向

### 6.1 低比特浮点

两个真实 PPL 码率下都没有超过整数，不继续实现专用 kernel。

### 6.2 48-bit 整数

质量保持只有 92.67%，而稳态速度与 80/128-bit 几乎相同。瓶颈已转移到 top-k、最终稀疏 attention 和模型其余部分，继续压索引 bit 不会带来同比速度收益。

### 6.3 逆 RoPE 后直接做 Key-PCA

64K、8 个 discovery seed 的结果：

| 代理索引 | 物理 bit/head/token | 相对 Full 质量 | 正确率 |
|---|---:|---:|---:|
| pre-RoPE PCA32 INT4 | 144 | 101.0% | 12.5% |
| pre-RoPE PCA64 INT2 | 144 | 50.4% | 12.5% |
| pre-RoPE PCA64 INT4 | 272 | 69.5% | 12.5% |

Key-PCA 最大化 Key 方差或重构能量，不保证保留当前 Query 决定 top-k 边界的方向；256 个均匀样本也容易漏掉 64K 中的稀有证据方向。因此该方向不扩展 held-out 实验。

### 6.4 把 INT2 的慢频覆盖从 32 维扩到 64 维

该实验检验“保持 2-bit，只扩大语义坐标覆盖”能否修复 64K。8 个 discovery seed 的结果为：

| 候选代理 | 相对 Full 质量 | 正确率 |
|---|---:|---:|
| 完整 pre-RoPE | 123.6% | 62.5% |
| 32D INT2 | 75.0% | 37.5% |
| 32D INT4 | 121.9% | 62.5% |
| 64D INT2 | 50.2% | 37.5% |

64D INT2 显著变差。scale-free 2-bit 只有四个非零等级；扩大维度后，更多低幅和噪声坐标被赋予粗糙离散值，累积误差增加并制造更多 top-k crossing。该结果说明不能把总 bit 简单地平均摊到更多坐标，因此没有继续实现 64D CUDA kernel。

## 7. 当前冻结建议

1. 论文主方法继续使用 QKSieve INT128 `[4,2]`。它在 32K 和 64K 都接近或达到 Full，跨主题更稳定。
2. pre-RoPE INT2 作为轻量索引扩展，而不是替换主方法。它展示了 RoPE 解耦和 1.5625% 索引的系统价值。
3. 不开发 minifloat kernel，不继续压到 48 bit。
4. 不采用单纯 Key-PCA 的 pre-RoPE 版本。
5. 若继续优化 pre-RoPE 分支，应使用可观测的 Query 慢频能量或排名 crossing risk，而不是训练 router；在证明显著改善前不加入主方法。
6. 合成长链上的 110% 以上结果只能作为机制实验。主论文仍需在冻结配置上完成 LongBench、RULER 和多模型独立测试。

## 8. 复现入口

主要实现：

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/src/run_head_top2_targeted_ppl_20260714.py
ymluo/projects/qwen3_top2_head_limit3_ppl/src/run_direct_countcap_denseprompt_ppl_20260725.py
ymluo/projects/qwen3_top2_head_limit3_ppl/src/run_qksieve_coldskip_longcontext_quality_20260730.py
ymluo/projects/qwen3_local_rule_failure_boundary/src/run_local_global_rope_probe_8b.py
```

启动脚本：

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/scripts/launch_qksieve_ultralowbit_ppl_8gpu_20260731.sh
ymluo/projects/qwen3_top2_head_limit3_ppl/scripts/launch_qksieve_prerope32_ppl_8gpu_20260731.sh
ymluo/projects/qwen3_top2_head_limit3_ppl/scripts/launch_qksieve_prerope32_64k_ppl_8gpu_20260731.sh
```

机器可读结果：

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/results/20260731_qksieve_lowbit_final_analysis_v2/summary.json
ymluo/projects/qwen3_top2_head_limit3_ppl/results/20260731_qksieve_ultralowbit_ppl_8gpu/
ymluo/projects/qwen3_top2_head_limit3_ppl/results/20260731_qksieve_lowbit_prerope_combined_summary/summary.json
ymluo/projects/qwen3_top2_head_limit3_ppl/results/20260731_qksieve_prerope32_64k_ppl_summary/summary.json
ymluo/projects/qwen3_local_rule_failure_boundary/artifacts/20260731_prerope_lowfreq32_heldout24_summary/summary.json
ymluo/projects/qwen3_local_rule_failure_boundary/artifacts/20260731_prerope_pca64k_discovery_summary/summary.json
ymluo/projects/qwen3_local_rule_failure_boundary/artifacts/20260731_prerope_lowfreq64_64k_discovery_summary/summary.json
```
