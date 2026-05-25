# Section 2 — K-cache Cosine 分析与 KV Cache 压缩

> 新增日期：2026-05-14；最近同步日期：2026-05-23

## 项目：`qwen3_kcache_cosine_heatmap`

新增日期：2026-05-14

这个项目在 5000 个 DCLM tokens 上 profile Qwen3-0.6B，抽取每个 layer/head 的 K-cache 矩阵，计算 token-token pairwise cosine matrix，并输出每个 head 的热力图 PNG 和 layer/head 总览热力图。

运行单个 5k-token heatmap：

```bash
LAYERS=0 HEADS=0 \
bash ymluo/projects/qwen3_kcache_cosine_heatmap/scripts/run_analysis.sh
```

默认 `LAYERS=all HEADS=all` 会生成全部 layer 和 KV heads 的热力图。

KV-cache 压缩诊断入口：

```bash
bash ymluo/projects/qwen3_kcache_cosine_heatmap/scripts/run_compression_diagnostics.sh
```

这个扩展脚本会额外输出：

- raw K-cache cosine 和 mean-centered K-cache cosine。
- raw V-cache cosine 和 mean-centered V-cache cosine。
- K/V raw 与 centered 矩阵的奇异值数据和奇异值图。
- PCA cumulative energy，以及达到 50%、75%、90%、95%、99% 能量所需 rank。
- 采样 query 上的低秩近似验证：`|q · (k_hat - k)|`、attention KL、top-1 match、attention output vector error。

注意：这个脚本不会伪造 loss/PPL change。精确 loss/PPL 需要在模型 forward 内部把每层 attention 的 K/V 替换成压缩版本后重跑；当前脚本先给出更低成本的 attention-weighted error，用来筛选值得进一步做模型内注入实验的 layer/head/rank。

---

## K-cache Cosine 结果解读

新增日期：2026-05-14

分析文件：

```text
C:/Users/夕/Documents/summary_by_head.csv
```

这份结果覆盖：

- `224` 行，正好对应 `28 layers x 8 KV heads`。
- 每个 head 都是 `5000 x 5000` pairwise cosine matrix。
- 每个 K vector 的 `head_dim=128`。
- 计算设备是 CUDA，similarity dtype 是 `torch.float32`。

关键统计：

```text
offdiag_mean 范围：0.1267 到 0.9914
offdiag_std  范围：0.0033 到 0.2784
40 / 224 个 head 的 offdiag_mean >= 0.9
67 / 224 个 head 的 offdiag_mean >= 0.8
36 / 224 个 head 的 offdiag_mean <= 0.3
```

这里的 `offdiag_*` 都排除了对角线，也就是排除了 token 和自身的 cosine。它更能反映不同 token 之间 K 向量方向是否相似。

### 层趋势

按 layer band 聚合后的趋势很明显：

| 层范围 | 平均 `offdiag_mean` | 解释 |
| --- | ---: | --- |
| `L0-L5` | `0.883` | 早期层 K 向量高度相似，存在明显方向冗余。 |
| `L6-L17` | `0.602` | 中间层相似度下降，head 间分化开始变强。 |
| `L18-L26` | `0.403` | 后期层 K 向量更分散，压缩需要谨慎。 |
| `L27` | `0.716` | 最后一层相似度重新上升，可能有特殊输出前表示结构。 |

这个结果说明：**KV cache 压缩不应该全层统一设置压缩率。** 早期层可以更激进；`L18-L26`，尤其 `L23-L26`，应更保守。

### 极端 head

平均 cosine 最高的一些 head：

```text
L00 H2 mean=0.9914 std=0.0054
L00 H6 mean=0.9901 std=0.0059
L01 H2 mean=0.9885 std=0.0033
L00 H7 mean=0.9859 std=0.0093
L00 H5 mean=0.9856 std=0.0088
```

这些 head 的 K 向量几乎同方向，表面上看非常适合压缩。

平均 cosine 最低的一些 head：

```text
L14 H7 mean=0.1267 std=0.1765
L06 H3 mean=0.1577 std=0.2777
L06 H6 mean=0.1622 std=0.1401
L14 H3 mean=0.1703 std=0.2052
L24 H3 mean=0.1792 std=0.2361
```

这些 head 的 K 向量方向差异大，直接合并 token 或强压缩风险更高。

`offdiag_std` 最高的一些 head 也值得关注：

```text
L01 H6 mean=0.4664 std=0.2784
L06 H3 mean=0.1577 std=0.2777
L24 H7 mean=0.1926 std=0.2442
L25 H5 mean=0.1928 std=0.2383
L24 H3 mean=0.1792 std=0.2361
```

这类 head 的分布更复杂，通常表示有些 token pair 很相似，有些完全不相似。它们更适合做 cluster/block-aware 压缩，而不是统一平均。

---

## 对 KV Cache 压缩的含义

这份 cosine 结果说明 K-cache 中确实存在大量结构性冗余，但不能直接推出"cosine 高的 token 就可以合并或删除"。

最需要警惕的是：**高 raw cosine 很可能来自一个很强的公共方向，而不是 token 内容真的都一样。**

可以把某个 head 的 K 向量近似写成：

```text
k_t = μ + r_t
```

其中：

- `μ` 是该 layer/head 内共享的公共方向或公共偏置。
- `r_t` 是 token-specific residual。

如果 `μ` 很大，那么任意两个 token 的 raw cosine 都可能很高。但 attention logit 是：

```text
q · k_t = q · μ + q · r_t
```

对于同一个 query，`q · μ` 对所有 key 都是同一个常数，softmax 会把这个公共常数抵消掉。因此这个公共方向虽然会抬高 raw cosine，却不一定携带有用的 token 选择信息。

所以，raw cosine 高更像是在提醒我们：**先做公共分量 / residual 分解，再决定怎么压缩。**

---

## 下一步建议

### 1. 做 mean-centering 后重新计算 cosine

新增日期：2026-05-14；实现日期：2026-05-14

对每个 `(layer, head)` 计算：

```text
centered_k_t = k_t - mean(k)
```

然后重新画 centered cosine heatmap。如果 centered 后 cosine 大幅下降，说明之前的高相似主要来自公共方向；这时压缩重点应放在 residual 表示，而不是直接合并 token。

已实现入口：

```bash
bash ymluo/projects/qwen3_kcache_cosine_heatmap/scripts/run_compression_diagnostics.sh
```

主要输出：

```text
compression_summary_by_head.csv
plots/k_centered_cosine/
```

### 2. 做 SVD / PCA energy 分析

新增日期：2026-05-14；实现日期：2026-05-14

对每个 `(layer, head)` 的 K 矩阵做 SVD，观察多少主成分能解释主要能量：

```text
K ≈ U_r S_r V_r^T
```

如果少数主成分解释大部分能量，可以考虑：

- 低秩 residual 压缩。
- 分 layer/head 设置 rank。
- 对公共方向和 residual 分开量化。

已实现输出：

```text
singular_values.csv
svd_summary_by_head.csv
plots/svd/
```

其中 `singular_values.csv` 同时包含 K/V 的 raw 和 centered 奇异值；`svd_summary_by_head.csv` 会记录达到指定累计能量阈值所需的 rank。

### 3. 同时分析 V-cache

新增日期：2026-05-14；实现日期：2026-05-14

K 相似不代表 V 可以安全合并。token merge 或 block summary 最终改变的是 attention output：

```text
attention_output = softmax(QK^T) V
```

所以压缩策略必须同时检查：

- K 的检索误差。
- V 的值误差。
- attention output error。
- 最终 loss/PPL。

已实现内容：

```text
compression_summary_by_head.csv
singular_values.csv
svd_summary_by_head.csv
plots/v_raw_cosine/
plots/v_centered_cosine/
```

### 4. 用 attention-weighted error 验证

新增日期：2026-05-14；部分实现日期：2026-05-14

不要只看 pairwise cosine。更关键的指标是：

```text
| q · (k_hat - k) |
attention KL
output vector error
PPL / loss change
```

已实现输出：

```text
attention_validation_by_head_rank.csv
```

已实现指标：

- `q_dot_abs_error_mean`
- `attention_weighted_q_dot_abs_error_mean`
- `attention_kl_mean`
- `top1_match_fraction`
- `output_l2_error_mean`
- `output_relative_l2_error_mean`

这个指标依赖 RoPE-aligned query capture。如果当前 Hugging Face Qwen3 实现没有把 `(cos, sin)` position embeddings 暴露给 attention hook，脚本会跳过 attention validation，并在 `summary.json` 里写明原因。

尚未直接实现 `PPL / loss change`。原因是这个指标必须把压缩后的 K/V 注入模型每一层 attention forward 后重跑，不能只靠最终 cache 离线计算，否则会得到误导性结果。

如果一个 token 的 K cosine 看起来可压，但它在真实 query 上经常获得高 attention mass，那么压缩它仍然可能伤害质量。

### 5. 分层分 head 设置压缩率

根据这份结果，建议压缩强度不要统一：

- `L0-L5`：可以尝试更激进压缩，尤其是 `offdiag_mean >= 0.9` 的 heads。
- `L6-L17`：适合自适应策略，按 head 的 cosine/std/attention energy 决定。
- `L18-L26`：需要更保守，尤其是低 mean、高 std 的 heads。
- `L27`：虽然相似度回升，但最好单独验证，因为最后一层直接靠近 logits。

一个实用策略是先给每个 head 打分：

```text
compressibility_score =
  high offdiag_mean
  + low offdiag_std
  + low attention sensitivity
  + low output error after compression
```

再用这个 score 决定每个 layer/head 的压缩率。
