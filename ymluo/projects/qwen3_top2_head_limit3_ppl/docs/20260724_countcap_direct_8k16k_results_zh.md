# CountCap 取消 2% 精确重排：8K/16K 结果

更新时间：2026-07-24

## 1. 实验问题

原 Dense-Suffix Key-PCA CountCap 的检索流程为：

```text
PCA48-INT4 sampled-quantile 选出约 6% 候选
-> 使用原始 K 精确打分候选
-> 候选内 top-k，压缩到 2%
-> 对 2% token 做 value attention
```

本实验测试是否可以取消候选内 top-k：

```text
PCA48-INT4 sampled-quantile 选出约 6% 候选
-> 使用原始 K 计算候选的真实 softmax logits
-> 不再排序并压回 2%
-> 直接对约 6% token 做 value attention
```

新实验方法标识：

```text
countcap_fullprompt_keypca_direct
```

对应 score mode：

```text
pca_int4_chunked_logscale16_sampleq_direct_autosplit
```

该路径始终是稀疏 attention，不包含 Full 回退。

## 2. 实验协议

| 项目 | 设置 |
|---|---|
| 模型 | Llama-3.1-8B-Instruct |
| 数据 | 同一个官方 GovReport 长样本，source row 115 |
| Prompt | 8,192 / 16,000 token |
| 生成上限 | 32 / 64 token |
| 重复 | 每个点 3 次严格配对 |
| 对照 | Full、原 2% 精排、候选直连 |
| GPU | 仅 GPU 0--3 |
| 速度 | 中位数；包含完整检索和建表开销 |

结果目录：

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/results/20260723_countcap_direct_8k16k_4gpu
```

完成状态：

```text
12 CSV
36 rows
每种方法 12 rows
ALL_COMPLETE = true
未发现 Traceback、OOM、AssertionError 或 RuntimeError
```

## 3. 结果

### 3.1 8K

| 生成上限 | 方法 | Attention 比例 | Online | Online speed | Total speed |
|---:|---|---:|---:|---:|---:|
| 32 | Full | 100% | 1.329 s | 1.000x | 1.000x |
| 32 | 原 2% 精排 | 1.99% | 2.953 s | 0.450x | 0.788x |
| 32 | 6% 候选直连 | 6.00% | 2.287 s | **0.583x** | **0.896x** |
| 64 | Full | 100% | 2.581 s | 1.000x | 1.000x |
| 64 | 原 2% 精排 | 1.99% | 4.699 s | 0.550x | 0.770x |
| 64 | 6% 候选直连 | 6.00% | 4.030 s | **0.637x** | **0.844x** |

相对于原 2% 精排，候选直连的 online speed 数值提高：

```text
8K × 32: 0.450x -> 0.583x，约提高 29%
8K × 64: 0.550x -> 0.637x，约提高 16%
```

但它仍明显慢于 Full。

### 3.2 16K

| 生成上限 | 方法 | Attention 比例 | Online | Online speed | Total speed |
|---:|---|---:|---:|---:|---:|
| 32 | Full | 100% | 2.024 s | 1.000x | 1.000x |
| 32 | 原 2% 精排 | 2.00% | 2.921 s | 0.701x | 0.948x |
| 32 | 6% 候选直连 | 6.00% | 2.309 s | **0.894x** | 1.014x |
| 64 | Full | 100% | 3.998 s | 1.000x | 1.000x |
| 64 | 原 2% 精排 | 2.00% | 4.736 s | 0.844x | 0.967x |
| 64 | 6% 候选直连 | 6.00% | 4.158 s | **0.966x** | 1.020x |

相对于原 2% 精排：

```text
16K × 32: 0.701x -> 0.894x，约提高 27%
16K × 64: 0.844x -> 0.966x，约提高 14%
```

16K × 64 的 online 延迟只比 Full 高约 4%，已经接近持平，但尚未形成可信加速。

`Total speed` 的 1.01x--1.02x 受三次独立 prefill 波动影响。当前研究重点应以 `Online speed` 为准，不能据此宣称 16K 已加速。

## 4. 质量

在该单 GovReport 样本上：

| Prompt/生成 | Full score | 原 2% 精排 | 候选直连 |
|---|---:|---:|---:|
| 8K / 32 | 0.07692 | 0.07692 | 0.07692 |
| 8K / 64 | 0.12245 | 0.12925 | 0.12925 |
| 16K / 32 | 0.07692 | 0.07692 | 0.07692 |
| 16K / 64 | 0.12925 | 0.12925 | 0.12925 |

候选直连没有产生可见质量下降，但这里只是单样本速度探针，不能替代完整 LongBench 质量实验。

## 5. 成本分解

使用 32/64-token 两点拟合：

```text
T_online(G) = T_fixed + (G - 1) × T_step
```

### 5.1 8K

| 方法 | T_fixed | T_step | 相对 Full 交叉点 |
|---|---:|---:|---:|
| Full | 0.116 s | 39.13 ms | - |
| 原 2% 精排 | 1.261 s | 54.56 ms | 不存在 |
| 6% 候选直连 | **0.598 s** | 54.49 ms | 不存在 |

取消候选内 top-k 把固定成本降低约 0.66 秒，但没有降低 steady step。8K 下稀疏 steady step 仍比 Full 慢约 39%，所以生成再长也无法回本。

### 5.2 16K

| 方法 | T_fixed | T_step | 估计交叉点 |
|---|---:|---:|---:|
| Full | 0.112 s | 61.68 ms | - |
| 原 2% 精排 | 1.164 s | 56.70 ms | 约 212 token |
| 6% 候选直连 | **0.518 s** | 57.78 ms | **约 105 token** |

候选直连把一次性固定成本降低约 0.65 秒，并把 16K 的估计 break-even 从约 212 token 提前到约 105 token。

它的 steady step 比原 2% 路径慢约 1.1 ms，是因为 value attention 从 2% 增加到约 6%；但省去候选 top-k 和重排张量后，固定成本显著下降。

## 6. 结论

本实验验证了一个明确的数值与系统现象：

> 在 8K--16K，候选内精确 top-k 的固定成本高于把 value attention 从 2% 增加到 6% 的额外成本。

因此，短序列优化不应继续执着于把最终 token 数从 6% 压到 2%。候选直连是更好的短序列执行结构。

但它没有完全解决问题：

- 8K 的 sparse steady step 本身仍慢于 Full；
- 16K × 64 只达到 0.966x online；
- 第一次 Key-PCA basis、完整历史投影和 INT4 建表仍约占 0.5 秒固定成本；
- PCA scan、候选压紧、精确 logits 和 ragged value attention 仍是多个 kernel。

## 7. 下一步

最优先方向不是继续调预算，而是：

1. 将 sampled-threshold、精确候选 logits 和 ragged value attention 进一步融合；
2. 将 Key-PCA 索引建立从第一次 decode 移出，或在 prefill 阶段隐藏其成本；
3. 单独 profile 8K steady step，确认约 15 ms/token 的差距来自 PCA scan、候选 exact QK 还是 ragged value reduction；
4. 如果 16K 优化后能够把固定成本降到约 0.2 秒，候选直连的交叉点可进一步接近 32--64 token。

当前冻结主方法仍保持 2% exact-rerank。候选直连作为实验分支，只有在多任务质量验证和进一步速度优化后才考虑替换主方法。
