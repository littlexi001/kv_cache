# Dense-Suffix Key-PCA CountCap：方法与复现技术文档

更新时间：2026-07-23

本文档用于把当前方法完整交接给另一个 Codex。阅读本文档后，应能够：

1. 准确理解当前主方法解决的问题、算法流程和实现边界；
2. 区分当前方法、旧 CountCap、QK-Metric 版本和成本门控版本；
3. 在现有服务器上继续运行、恢复、验证 LongBench 实验；
4. 正确解释质量、稀疏比例、GPU KV 占用和速度指标；
5. 在不改变主方法定义的前提下继续做实验。

---

## 1. 当前冻结的主方法

当前主方法名称：

**Dense-Suffix Key-PCA CountCap**

LongBench runner 中对应的方法标识：

```text
countcap_fullprompt_keypca
```

当前方法的冻结定义：

```text
完整 prompt 使用密集 attention 编码
-> 每层、每个 query head 独立检索历史 token
-> 使用 Key-PCA 48 维 INT4 索引近似扫描全部历史
-> 保留 3%--6% 候选 token
-> 使用原始全精度 K 对候选做精确 QK 重排
-> 每个 head 最终保留 min(2% × 历史长度, 1280) 个历史 token
-> 使用原始 V 对最终 token 做稀疏 attention
-> 每生成一个新 token，增量更新 KV 和 PCA-INT4 索引
```

主方法明确**不包含**以下机制：

- 不包含 `countcap_auto`；
- 不包含成本门控；
- 不根据长度或预期生成步数回退 Full Attention；
- 不训练 router；
- 不使用任务名称、任务类别或 oracle 标签；
- 不使用字符串、BM25、RAG 或 block 级词汇匹配；
- 不在生成阶段切换到 Full Attention。

实验中保留 `full_kv` 仅用于建立同样本基线，不是主方法的一部分。

---

## 2. 问题定义

标准自回归解码在每一层、每个 attention head 上，都让当前 query 与全部历史 key 做点积，并对全部 value 加权求和。历史长度为 `N`、head 维度为 `d` 时，单个 head 每一步的主要 attention 工作量近似为：

```text
O(N × d)
```

此前的机制实验发现，在长上下文中，每个 query head 只使用真实 attention 分数最高的一小部分历史 token，通常也能保持甚至改善输出质量。于是核心问题从“是否可以只看少量 token”转化为：

> 如何在不先计算完整 QK 分数的情况下，快速找到每个 head 真正需要的少量历史 token？

直接计算完整 QK 后再取 top-k 不能带来实际加速，因为最昂贵的全历史扫描已经发生。当前方法使用一个低维、量化的 Key-PCA 索引完成廉价近似扫描，再只对较小候选集计算原始全精度 QK。

---

## 3. 必须先区分的三个比例

### 3.1 Attention link ratio

当前方法中的“2%”指：

```text
每层、每个 query head 最终真正参加 value attention 的历史 token 数
----------------------------------------------------------------
                         历史 token 总数
```

这是 attention 计算稀疏度，不是物理 KV 存储比例。

### 3.2 Candidate ratio

PCA-INT4 近似扫描后，不会直接把近似 top-2% 当作最终答案，而是先形成 3%--6% 的候选池，再使用原始 K 精确重排。候选池比最终 attention 集更大，用于降低低维投影和 INT4 量化导致的边界漏召回。

### 3.3 GPU KV storage ratio

当前 LongBench 质量与速度 harness 使用 Hugging Face `DynamicCache`，完整原始 K/V 仍留在 GPU 中，代码明确报告：

```text
gpu_kv_storage_ratio = 1.0
```

因此当前 LongBench 实验验证的是：

- 稀疏 attention 的质量；
- 检索、精确重排和稀疏 value attention 的计算速度；
- 完整生成流程的在线时间。

它**没有**证明物理 KV 显存已经压缩到 2%。如果论文要报告物理 KV 内存比例，需要另外接入 KV offload、分页缓存或只保留热 KV 的实现，并单独验证。

---

## 4. Prompt 与生成阶段如何划分

设完整 prompt 由三部分组成：

```text
[文档或上下文 prefix] + [问题/指令 suffix] + [待生成答案]
```

当前实现中，文档和问题都属于 prompt，都在答案生成前已知。

### 4.1 Dense prefix

较长的文档或上下文 prefix 使用标准 SDPA 密集 prefill。为了控制峰值显存，runner 可以按 `prefill_chunk_tokens=2048` 分块执行，但语义上仍是完整密集 causal attention。

### 4.2 Dense suffix

问题或指令 suffix 也使用标准 SDPA，一次作为完整 token segment 处理，并接在 prefix KV cache 后面。

这一步非常关键。旧 CountCap 曾把问题 suffix 拆成单 token，逐 token 走稀疏 attention：

```text
文档 dense prefill
-> 问题 token 1 sparse
-> 问题 token 2 sparse
-> ...
-> 开始生成答案
```

这种旧实现有两个问题：

1. 问题本身的隐藏状态已被稀疏近似污染，误差在生成答案前就进入模型；
2. 几十到上千个问题/指令 token 会触发大量单 token Python 和 CUDA 调度。

当前 Dense-Suffix 路径改为：

```text
[文档 + 完整问题/指令] dense prompt encoding
-> 得到第一个答案 token 的 logits
-> 从后续自回归生成步骤开始使用 sparse attention
```

因此，第一个答案 token 直接来自完整 dense prompt 的最后一个位置。生成第一个 token 本身不需要稀疏 forward；把第一个答案 token 输入模型、预测第二个答案 token 时，才发生第一次稀疏检索和索引构建。

### 4.3 为什么 dense prompt 不违背“稀疏 attention”

本文研究的目标是长上下文**解码阶段**的稀疏 attention。Prefill 只执行一次，而同一历史会被后续多个生成步重复访问。当前方法没有声称 prefill 本身是稀疏的。

---

## 5. CountCap 预算

对当前某层的历史长度 `N`，最终每个 query head 的 attention token 数为：

```text
A(N) = min(max(1, round(0.02 × N)), 1280)
```

最终 attention 比例为：

```text
r_att(N) = A(N) / N
```

候选比例为：

```text
r_cand(N) = min(0.06, max(0.03, 4 × r_att(N)))
```

常见长度对应的预算如下：

| 历史长度 N | 每 head 最终 token 数 A(N) | 最终比例 | 候选比例 |
|---:|---:|---:|---:|
| 7,500 | 150 | 2.00% | 6.00% |
| 16,000 | 320 | 2.00% | 6.00% |
| 32,000 | 640 | 2.00% | 6.00% |
| 64,000 | 1,280 | 2.00% | 6.00% |
| 128,000 | 1,280 | 1.00% | 4.00% |
| 256,000 | 1,280 | 0.50% | 3.00% |

`1280` 是 CountCap 的含义：当上下文超过 64K 后，不再让最终 attention token 数随上下文线性增加。

这个预算是解析式规则，不是训练得到的 router，也不依赖任务类型。

---

## 6. Key-PCA 索引

### 6.1 索引对象

索引按以下粒度独立维护：

```text
每一层 × 每一个 KV head
```

检索结果按每个 query head 独立产生。在 GQA 模型中，多个 query head 可以映射到同一个 KV head，但当前设置为：

```text
gqa_candidate_mode = independent
```

因此映射到同一个 KV head 的不同 query head 仍可以选择不同的历史 token。

### 6.2 PCA basis

设某层、某个 KV head 的历史 key 为：

```text
K = [k_1, k_2, ..., k_N]
```

实现每隔 32 个 token 采样一个 key：

```text
K_sample = K[..., ::32, :]
```

使用采样 key 的二阶矩阵：

```text
M = (1 / |K_sample|) × sum(k_i × k_i^T)
```

对 `M` 做特征分解，取最大 48 个特征值对应的特征向量，构成投影基：

```text
P ∈ R^(d × 48)
```

当前固定：

```text
projection_dim = 48
```

Key-PCA 使用 K 的统计量构建一次 basis，不需要等待生成 query，也不需要第二次重建全历史索引。

### 6.3 投影与量化

对历史 key：

```text
z_i = P^T k_i
```

对当前 query：

```text
u = P^T q
```

每个 `z_i` 使用分块 log-scale INT4 量化。48 个维度被划分为 3 个 16 维 chunk，INT4 code 以两个 4-bit 数打包到一个 `uint8` 中。每个 token 还保存缩放信息和压缩的 log-scale exponent。

当前 score mode：

```text
pca_int4_chunked_logscale16_sampleq_autosplit
```

近似分数可概括为：

```text
score_approx(q, k_i) = dot(P^T q, INT4_dequant(P^T k_i))
```

实际 CUDA 路径会量化投影 query，并直接在打包后的 INT4 chunk 上计算，不要求先把全部低维 key 解量化为浮点张量。

### 6.4 RoPE 口径

当前 patch 接管 Hugging Face `eager_attention_forward` 的 `query/key/value` 输入。索引直接使用该 attention 接口收到的实际 key cache，不额外对被选中的 token 重新编码 RoPE。

复现时不要另写一套“先取 raw K、再按新位置重做 RoPE”的 repack 逻辑。最可靠的复现方式是直接复用当前 attention patch。

### 6.5 增量更新

第一次稀疏 forward 会为已有完整历史建立 PCA-INT4 索引。之后每生成一个 token：

1. 原始 K/V 追加到 Hugging Face cache；
2. 只投影和量化新增 key；
3. 将新增 code 追加到已有索引；
4. PCA basis 保持不变，不重投影旧历史。

初始索引容量为：

```text
history_count + 2048
```

如果后续生成超过预留容量，索引会自动扩容，并保留已有 basis 与量化 code。扩容测试已覆盖。

### 6.6 PCA48 INT4 索引显存

以 Llama-3.1-8B 的 `head_dim=128` 为例，每个 token、每个 KV head 的主要索引数据为：

| 组成 | 每 token、每 KV head |
|---|---:|
| 48 维 INT4 code | `48 × 4 bit = 24 byte` |
| FP16 base scale | 2 byte |
| 3 个 16 维 chunk 的打包 exponent | 2 byte |
| 合计 | 约 28 byte |

原始 FP16 K 与 V 为：

```text
K + V = 128 × 2 byte × 2 = 512 byte
```

所以 PCA48 INT4 索引的主要逐 token 存储约为：

```text
28 / 512 = 5.47% of FP16 K+V
```

PCA basis 每层、每个 KV head 还需要约：

```text
128 × 48 × 2 byte = 12 KB
```

该部分会被长序列摊薄。候选索引和精确分数 buffer 是运行时临时空间。

必须再次强调：5.47% 是**附加检索索引相对原始 K+V 的大小**。当前 LongBench harness 仍保留 100% 原始 K/V，因此当前总 GPU cache 存储不是 5.47%，而是完整 K/V 再加索引及临时 buffer。

---

## 7. 两阶段检索

每个生成步、每层、每个 query head 的检索流程如下。

### 7.1 近似扫描全部历史

使用 48 维 PCA-INT4 索引计算所有历史 token 的近似 QK 分数。相对于原始 head dimension 和 FP16 K，该扫描同时减少维度与数据位宽。

### 7.2 Sampled-quantile 候选选择

这里必须区分：

```text
候选选择：近似 top 3%--6%
最终 top-k：精确 top min(2% × N, 1280)
```

256 点 sampled-quantile 只用于估计“进入候选池”的分数阈值，不直接决定最终参加 attention 的 top-k。

#### 7.2.1 采样位置

当前路径固定使用：

```text
S = sampled_quantile_sample_count = 256
```

采样不是随机采样，而是在长度为 `N` 的整个历史上均匀取 256 个区间中点。第 `j` 个采样位置为：

```text
sample_index(j)
  = floor(((2 × j + 1) × N) / (2 × S))

j = 0, 1, ..., S - 1
S = 256
```

也就是近似取：

```text
N/512, 3N/512, 5N/512, ..., 511N/512
```

这种确定性分层采样可以覆盖完整历史，避免每步生成随机数，也让同一个输入的检索结果可复现。

#### 7.2.2 计算候选阈值

设当前候选目标比例为：

```text
r = r_cand(N)
```

对 256 个采样 token 计算 PCA48-INT4 近似分数，升序排序。需要保留的采样分数数目为：

```text
m = max(1, ceil(r × 256))
```

候选阈值取第 `m` 大的采样分数：

```text
threshold = sorted_sample_scores[256 - m]
```

常用候选比例对应：

| 候选目标比例 r | 256 点中保留的 m | 阈值 |
|---:|---:|---|
| 3% | 8 | 第 8 大采样分数 |
| 4% | 11 | 第 11 大采样分数 |
| 6% | 16 | 第 16 大采样分数 |

这是一个分位数估计。它假设均匀采样分数的分布能够近似全部 `N` 个分数的分布，因此超过该阈值的完整历史 token 数应接近 `r × N`。

#### 7.2.3 全历史阈值扫描

得到 threshold 后，CUDA kernel 使用 PCA48-INT4 对全部 `N` 个历史 token 做一次近似分数扫描：

```text
for i in 0 ... N-1:
    score_i = approximate_PCA48_INT4_score(q, k_i)
    if score_i >= threshold:
        compact i into candidate_indices
```

因此，该方法省去的是“对 `N` 个分数做精确 FP16 QK 和全局排序”，并没有省去低维 INT4 的全历史线性扫描。

候选使用 warp ballot、popcount 和 atomic offset 直接压紧到连续 buffer，不需要先保存完整 FP32 score vector 后再排序。

#### 7.2.4 候选容量

采样阈值有统计误差，而且并列分数可能让候选数多于 `r × N`。当前普通 256 点路径的候选 buffer 比目标比例更大：

```text
r_capacity = min(1.0, max(2 × r, r + 0.04))
candidate_capacity = ceil(r_capacity × N)
```

同时容量不会小于最终 attention token 数 `A(N)`。

| 目标候选比例 r | 预留容量比例 |
|---:|---:|
| 3% | 7% |
| 4% | 8% |
| 6% | 12% |

内核为每个 query head 分别记录：

```text
candidate_indices
candidate_proxy_scores
candidate_count
estimated_threshold
overflow
```

#### 7.2.5 溢出处理

如果任意 head 的实际候选数超过预留容量，当前 Key-PCA 路径会：

1. 使用同一个 PCA48-INT4 索引计算完整低维近似分数；
2. 在近似分数上直接取固定数量的 candidate top-k；
3. 再进入原始 K 精确重排。

这里的 fallback 只是“sampled threshold 失败后回到完整低维近似 top-k”，不是 Full Attention，也不会让生成步骤回退到密集 FP16 attention。

#### 7.2.6 具体例子

7.5K 历史：

```text
N = 7500
A(N) = 150
r_att = 2%
r_cand = 6%
m = ceil(256 × 6%) = 16

在 256 个均匀位置计算近似分数
-> 用第 16 大分数作为阈值
-> 扫描 7500 个 PCA48-INT4 key
-> 期望得到约 450 个候选
-> buffer 最多预留约 900 个候选
-> 用原始 K 精确打分这些候选
-> 最终选出 150 个历史 token
```

128K 历史：

```text
N = 128000
A(N) = 1280
r_att = 1%
r_cand = 4%
m = ceil(256 × 4%) = 11

第 11 大采样分数作为阈值
-> 期望约 5120 个候选
-> buffer 容量约 10240
-> 精确重排后最终选 1280 个历史 token
```

256K 历史：

```text
N = 256000
A(N) = 1280
r_att = 0.5%
r_cand = 3%
m = ceil(256 × 3%) = 8

第 8 大采样分数作为阈值
-> 期望约 7680 个候选
-> buffer 容量约 17920
-> 精确重排后最终选 1280 个历史 token
```

runner 中虽然还传入：

```text
sample_fraction = 0.0025
qabs_dim_count = 8
```

但对当前 `sampleq` Key-PCA score mode，候选阈值的活动实现是固定 256 点 sampled quantile；`qabs_dim_count=8` 是兼容旧 QAbs 路径的参数，不定义当前 PCA 维度。

### 7.3 原始 K 精确重排

对候选集合 `C_h`，使用原始全精度 query 和 key 重新计算：

```text
score_exact(q_h, k_i) = dot(q_h, k_i) / sqrt(d)
```

然后在候选内部取：

```text
top A(N)
```

这一步是质量保障的关键。当前设置：

```text
skip_candidate_rerank = False
```

### 7.4 稀疏 value attention

使用最终选中的原始 V 做 softmax 和加权求和。实现会显式加入当前 token 的 self key/value，保证 causal decode 的当前位置可见。

`autosplit` 会根据候选规模和 GPU SM 数选择 value attention kernel 的分割方式，避免在所有长度上固定使用同一种 reduction 形状。

---

## 8. 单步伪代码

```text
Input:
  当前层 query q
  完整原始 KV cache: K, V
  该层各 KV head 的 PCA-INT4 index
  历史长度 N

A = min(max(1, round(0.02 * N)), 1280)
r_att = A / N
r_cand = min(0.06, max(0.03, 4 * r_att))

for each KV head:
    if index does not exist:
        sample K every 32 tokens
        build top-48 Key-PCA basis P
        project all existing K
        pack projected K as chunked logscale16 INT4
    else:
        project and append only newly added K

for each query head independently:
    project q to 48D
    scan all history with PCA-INT4 approximate scores
    estimate candidate threshold from 256 sampled scores
    collect approximately r_cand * N candidates
    recompute exact full-precision QK on candidates
    select exact top A candidates
    append current self position
    run sparse softmax and weighted sum on original V

Return:
  attention output
```

---

## 9. 与其他版本的区别

### 9.1 与旧 CountCap 的区别

旧版本：

```text
Dense document prefix
-> sparse token-by-token question suffix
-> QK-Metric warmup
-> rebuild PCA-INT4 index
-> sparse generation
```

当前版本：

```text
Dense complete prompt
-> one-shot Key-PCA index
-> sparse generation
```

主要优势：

- 问题和指令被完整密集编码，质量更稳定；
- 消除问题 suffix 的大量单 token 稀疏调用；
- 消除 QK-Metric 收集 query 后对全历史重建索引的固定成本。

### 9.2 与 QK-Metric CountCap 的区别

2026-07-22 长上下文主线使用：

```text
QK-Metric48
+ logscale16 INT4
+ sampled quantile
+ CountCap
+ exact rerank
```

QK-Metric 会先累计若干生成 query 的二阶矩，再联合 K 与 Q 的统计构建不同的 query/key 投影因子。当前实现 warmup 4 个 query 后激活，并把完整历史 K 重新投影一次。

当前 Key-PCA 版本：

- basis 只由 K 构建；
- Q 和 K 使用同一个正交 basis；
- 不等待 4 个生成 query；
- 不重建完整索引；
- 在当前 LongBench 配对实验中，质量与 QK-Metric 基本相同，但固定成本更低。

### 9.3 与 `countcap_auto` 的区别

`countcap_auto` 会根据历史长度、预期生成长度和实测成本模型在 Full 与 sparse 之间选择路径。它是一个可选工程扩展，不是当前论文主方法。

当前主方法必须直接运行：

```text
countcap_fullprompt_keypca
```

不要在主结果中用 `countcap_auto`，也不要把 Full 回退样本计入当前方法的质量或速度。

---

## 10. 当前已完成的质量结果

### 10.1 完整 16-task、16K prompt 诊断

结果目录：

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/results/20260723_countcap_adakv16k_longbench_full_8gpu
```

独立核验：

| 项目 | 数值 |
|---|---:|
| 总行数 | 7,500 |
| Full KV | 3,750 |
| Key-PCA CountCap | 3,750 |
| 严格同样本配对 | 3,750 |
| 任务数 | 16 |
| 最大 prompt | 16,000 token |
| 错误 | 0 |

宏平均：

| 方法 | Macro score | 相对 Full 质量 |
|---|---:|---:|
| Full KV | 0.471294 | 100.00% |
| Dense-Suffix Key-PCA CountCap | 0.469249 | **99.57%** |

平均统计：

| 指标 | 数值 |
|---|---:|
| 平均 prompt token | 9,053.32 |
| 平均检索历史 token | 8,835.69 |
| 每 head 平均最终 attention token | 176.70 |
| 近似 attention link ratio | 2% |
| Full 平均生成 token | 83.83 |
| CountCap 平均生成 token | 87.13 |

分任务结果：

| 任务 | 样本数 | 平均 prompt token | 首个稀疏步检索 token/head | CountCap 平均输出 token | Full | CountCap | 相对 Full |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2WikiMQA | 200 | 7,173.5 | 143.8 | 4.9 | 0.47803 | 0.48130 | 100.68% |
| GovReport | 200 | 9,373.7 | 188.1 | 440.4 | 0.21058 | 0.21102 | 100.21% |
| HotpotQA | 200 | 12,789.0 | 256.3 | 4.3 | 0.57382 | 0.57940 | 100.97% |
| LCC | 500 | 3,124.7 | 63.4 | 64.0 | 0.63196 | 0.62344 | 98.65% |
| MultiNews | 200 | 2,652.1 | 53.6 | 488.1 | 0.15973 | 0.15460 | 96.79% |
| MultiFieldQA-en | 150 | 6,948.8 | 139.2 | 18.4 | 0.56988 | 0.56581 | 99.29% |
| Musique | 200 | 15,451.0 | 309.5 | 5.9 | 0.32146 | 0.32604 | 101.43% |
| NarrativeQA | 200 | 14,480.4 | 290.0 | 7.0 | 0.26631 | 0.26506 | 99.53% |
| PassageCount | 200 | 13,518.8 | 270.4 | 4.2 | 0.08808 | 0.08375 | 95.08% |
| PassageRetrieval-en | 200 | 12,532.9 | 251.2 | 3.0 | 0.99500 | 0.99500 | 100.00% |
| Qasper | 200 | 5,047.1 | 101.1 | 19.5 | 0.46239 | 0.44930 | 97.17% |
| QMSum | 200 | 12,379.3 | 247.8 | 98.4 | 0.18488 | 0.18227 | 98.59% |
| RepoBench-P | 500 | 9,790.2 | 196.3 | 64.0 | 0.56154 | 0.55897 | 99.54% |
| SAMSum | 200 | 9,118.7 | 182.9 | 128.0 | 0.39375 | 0.39042 | 99.15% |
| TREC | 200 | 6,785.6 | 136.4 | 64.0 | 0.72500 | 0.72500 | 100.00% |
| TriviaQA | 200 | 10,948.9 | 219.4 | 32.0 | 0.91829 | 0.91662 | 99.82% |

表中的 token 口径：

- `平均 prompt token` 是文档、问题/指令和 chat wrapper 组成的完整模型输入；Full 与 CountCap 对同一样本完全相同。
- `首个稀疏步检索 token/head` 是完整 dense prompt 之后，第一次稀疏 forward 中每个 query head 选择的平均历史 token 数，不包含当前 self token。
- 生成继续进行时历史长度会增加，预算按同一比例动态增加；因此这列不是整个回答期间固定不变的 token 数。
- 不同 query head 可以选择不同 token。这一列是每个 head 的 attention link 数，不能解释为整个模型只保留了这些唯一 KV token。

重要限制：这轮 16K 实验使用了旧的自定义 Rouge 和不完整 stop-token 规则。因为 Full 与 CountCap 在完全相同协议下严格配对，**99.57% 相对质量保持率可信**；但其绝对 macro score 不能直接与 AdaKV Table 5 比较。旧 CSV 还把换行转义并把 prediction 截断到 500 字符，无法无损重算官方 Rouge。

### 10.2 7.5K、每任务 20 样本的 PCA 对比

结果目录：

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/results/20260723_countcap_fullprompt_longbench_m20_8gpu
```

| 方法 | Macro score | 相对 Full | Online speed | 全流程 speed |
|---|---:|---:|---:|---:|
| Full KV | 0.437746 | 100.00% | 1.000x | 1.000x |
| Dense-Suffix QK-Metric | 0.437770 | 100.006% | 0.448x | 0.574x |
| Dense-Suffix Key-PCA | 0.437670 | **99.983%** | **0.536x** | **0.658x** |

结论：

- Dense suffix 在 7.5K 已恢复接近无损质量；
- Key-PCA 与 QK-Metric 的质量差距可忽略；
- Key-PCA 明显降低固定成本；
- 但 7.5K 下当前稀疏 kernel 仍慢于高度优化的 Full SDPA。

### 10.3 16K、每任务 20 样本的 PCA 对比

结果目录：

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/results/20260723_countcap_adakv16k_longbench_m20_8gpu
```

| 方法 | Macro score | 相对 Full | Online speed | 全流程 speed |
|---|---:|---:|---:|---:|
| Full KV | 0.486742 | 100.00% | 1.000x | 1.000x |
| Dense-Suffix QK-Metric | 0.484498 | 99.539% | 0.522x | 0.689x |
| Dense-Suffix Key-PCA | 0.484583 | **99.556%** | **0.621x** | **0.769x** |

这组结果进一步支持：在当前固定 top-2% + exact rerank 路径中，Key-PCA 足以替代成本更高的 QK-Metric。

### 10.4 检索误差诊断

结果目录：

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/results/20260723_countcap_quality_diagnostic_m8_8gpu
```

| 方法 | Macro score | 相对 Full | 实际 link ratio |
|---|---:|---:|---:|
| Full KV | 0.406925 | 100.00% | 100% |
| 精确 QK top-2% | 0.403209 | 99.09% | 2% |
| 精确 attention-mass 自适应 | 0.405379 | 99.62% | 4.05% |
| PCA-INT4 近似 + top-2% | 0.409101 | 100.53% | 2.01% |
| PCA-INT4 近似 + 自适应 | 0.403956 | 99.27% | 4.14% |

该小样本诊断说明：

- PCA48 + INT4 的近似误差不是当前主要质量瓶颈；
- 固定 top-2% 已处于安全区域；
- 简单扩大为 4% 左右的自适应预算没有稳定改善质量。

它是机制诊断，不应替代完整 3,750 样本结果。

---

## 11. 当前已完成的速度结果

### 11.1 完整 16K LongBench 的平均速度

在第 10.1 节的完整 16K 实验中：

| 指标 | Full | CountCap | speed |
|---|---:|---:|---:|
| 平均 online 时间 | 3.532 s | 5.537 s | 0.638x |
| 平均全流程时间 | 6.922 s | 8.954 s | 0.773x |

这里的 online 时间包含：

```text
dense question/suffix
+ 第一次 PCA basis 与完整历史 INT4 索引建立
+ 每步近似检索
+ sampled-quantile
+ 原始 K 精确重排
+ 稀疏 value attention
+ generation loop
```

全流程时间还包含 dense prefix prefill。

16K 宏平均仍未加速，主要原因是很多 LongBench 样本实际短于 16K，而且短回答无法摊销一次性索引成本。

### 11.2 真长样本的二维速度网格

结果目录：

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/results/20260723_countcap_horizon_grid_long_sample_4gpu
```

使用同一个官方 GovReport 长样本，在不同 prompt 长度和最大生成长度下测量。所有稀疏开销都已计入，且没有 Full 回退。

| Prompt | 生成 8 token | 生成 32 token | 生成 64 token | 估计 break-even |
|---:|---:|---:|---:|---:|
| 8,192 | 0.247x | 0.463x | 0.553x | 当前范围内不存在 |
| 16,000 | 0.364x | 0.699x | 0.829x | 约 395 token |
| 24,576 | 0.444x | 0.904x | 1.105x | 约 44 token |
| 32,768 | 0.473x | 1.090x | **1.338x** | 约 29 token |

这张表是当前纯稀疏方法最重要的速度结论：

- 质量接近无损不等于所有长度都能加速；
- 一次性建表成本随历史长度增长；
- sparse steady step 在足够长序列上才快于 Full；
- 生成步数必须足够多，才能摊销第一次索引构建。

### 11.3 相关但不能直接归属于当前 Key-PCA 的长上下文结果

2026-07-22 的 QK-Metric CountCap 后端在 128K PPL 实验中得到：

| 指标 | 结果 |
|---|---:|
| Full PPL | 11.3884 |
| QK-Metric CountCap PPL | 11.2901 |
| 质量保持率 | 100.87% |
| 完整 online decode speed | 3.510x |
| attention 接口 speed | 13.525x |

180K--256K 聚合 online speed 为约 3.650x，质量保持率约 100.61%。

这些结果证明 CountCap + PCA-INT4 + exact rerank 后端在超长上下文有潜力，但它们使用 QK-Metric 索引，不是当前 Key-PCA 的直接实验。论文中不能把这些数字直接写成 Dense-Suffix Key-PCA 的结果；需要补做同模型、同数据、同硬件的直接 Key-PCA 128K 实验。

---

## 12. 官方 AdaKV Table-5 对齐实验

### 12.1 当前运行

正式结果目录：

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/results/20260723_countcap_adakv_table5_official75k_full_8gpu
```

启动脚本：

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/scripts/launch_countcap_adakv_table5_official75k_full_8gpu_20260723.sh
```

方法：

```text
full_kv
countcap_fullprompt_keypca
```

该实验不包含成本门控或 Full 回退。CountCap 行的每一个样本都执行固定的 Key-PCA 稀疏路径。

协议：

| 项目 | 设置 |
|---|---|
| 模型 | Llama-3.1-8B-Instruct |
| 任务 | LongBench 16 个英文任务 |
| 样本 | 3,750 |
| 方法数 | 2 |
| 预期总行数 | 7,500 |
| 最大完整 prompt | 7,500 token |
| 评分 | `rouge==1.0.1` 与官方任务 metric |
| 停止规则 | Llama 三类 stop token；SAMSum 额外首换行停止 |
| prediction | 保存完整原始文本，不截断 |
| 配对 | `(task, sample_id)` 严格 Full/CountCap 配对 |

截至 2026-07-23 18:24 的一次快照为：

```text
4020 / 7500 rows
2012 Full KV
2008 CountCap
2008 strict pairs
11 tasks observed
8 shard CSV files
ALL_COMPLETE = false
```

这只是文档编写时快照，接手者必须重新检查，不能把它当最终结果。

### 12.2 完成条件

只有同时满足以下条件才能引用正式结果：

```text
ALL_COMPLETE 存在
总行数 = 7500
full_kv = 3750
countcap_fullprompt_keypca = 3750
严格配对数 = 3750
任务数 = 16
max(prompt_tokens) <= 7500
merged/summary.json 存在且可读取
日志中无未处理异常
```

---

## 13. 代码地图

项目根目录：

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
```

本地对应目录：

```text
C:\Users\27814\Desktop\work\codex_workspace\kvcache\kv_cache-main\kv_cache-main\ymluo\projects\qwen3_top2_head_limit3_ppl
```

关键文件：

| 文件 | 作用 |
|---|---|
| `src/run_sample_calibrated_longbench_20260717.py` | LongBench 入口、CountCap 预算、dense suffix、生成循环、CSV 恢复 |
| `src/run_head_top2_targeted_ppl_20260714.py` | HF attention patch、PCA-INT4 索引、候选检索、精确重排、稀疏 attention |
| `src/qabs_cuda_kernels.py` | PCA-INT4、sampled quantile、candidate score 和 sparse value attention CUDA kernel |
| `src/run_controlled_public_kv_benchmark_v1.py` | LongBench 数据结构、prompt、官方评分辅助 |
| `src/summarize_countcap_benchmark_20260722.py` | shard 合并与 summary 生成 |
| `src/rescore_longbench_official_20260723.py` | 对保存完整 prediction 的结果重算官方分数 |
| `scripts/launch_countcap_adakv_table5_official75k_full_8gpu_20260723.sh` | 8 GPU 官方 7.5K 完整实验 |
| `tests/test_countcap_fullprompt.py` | dense suffix、stop policy 和 CountCap 路径测试 |
| `tests/test_longbench_official_scoring.py` | `rouge==1.0.1` 对齐测试 |
| `tests/test_pca_index_capacity_growth.py` | PCA 索引扩容测试 |

需要重点阅读的函数：

| 函数 | 文件 | 含义 |
|---|---|---|
| `countcap_config` | LongBench runner | 2%/1280 cap 与候选比例 |
| `resolve_method_plan` | LongBench runner | 证明只有 `countcap_auto` 会门控 |
| `generate_global_partition` | LongBench runner | dense suffix 与 sparse decode |
| `longbench_stop_token_ids` | LongBench runner | 官方停止 token |
| `_pca_int4_partial_scores` | attention patch | Key-PCA basis、索引创建、增量 append |
| `qabs_sampled_head_adaptive_attention` | attention patch | 近似候选、精确重排与 sparse value attention |
| `head_qabs_sampled_mass_mode` | attention patch | 激活当前 score mode 和运行参数 |
| `install_llama_head_top_fraction_patch` | attention patch | 安装 Llama/Qwen eager attention patch |

---

## 14. 环境与依赖

远端环境：

```text
SSH alias: df
Python: /home/fdong/miniconda3/envs/moe/bin/python
GPU: 8 × RTX 3090
Model: /home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms
LongBench data: /home/fdong/ymluo/external/KVCache-Factory/data/LongBench
```

官方 Rouge 必须可导入：

```bash
/home/fdong/miniconda3/envs/moe/bin/python -c \
  "from rouge import Rouge; print(Rouge)"
```

预期包：

```text
rouge==1.0.1
torch
transformers
numpy
pytest
```

CUDA 扩展由 `qabs_cuda_kernels.py` 按当前工程方式加载或编译。首次运行可能包含 CUDA 扩展编译时间，不应计入稳定 benchmark。

---

## 15. 复现正式 LongBench 实验

### 15.1 启动前检查

```bash
ssh df
cd /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl

nvidia-smi
pgrep -af 'run_sample_calibrated_longbench_20260717.py'

RUN_ROOT=results/20260723_countcap_adakv_table5_official75k_full_8gpu
test -e "$RUN_ROOT/ALL_COMPLETE" && echo complete || echo not_complete
```

如果已有同一个 `RUN_ROOT` 的 8 个 worker 正在运行，不要重复启动。

### 15.2 后台启动

```bash
cd /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl

RUN_ROOT=results/20260723_countcap_adakv_table5_official75k_full_8gpu
mkdir -p "$RUN_ROOT/logs"

setsid -f bash \
  scripts/launch_countcap_adakv_table5_official75k_full_8gpu_20260723.sh \
  > "$RUN_ROOT/logs/nohup.log" 2>&1
```

启动脚本会等待 GPU 空闲，然后创建 8 个 shard worker。

### 15.3 断点恢复

runner 会读取每个 shard 已有的 `sample_results.csv`，按以下键跳过已完成行：

```text
(task, sample_id, method)
```

异常后可以直接重新执行同一个 launcher。不要删除有效 CSV，不要更换输出目录后手工拼接，也不要重跑已经完成的样本。

### 15.4 监控

```bash
ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/results/20260723_countcap_adakv_table5_official75k_full_8gpu

wc -l "$ROOT"/shard[0-9]*/sample_results.csv
tail -n 30 "$ROOT"/logs/shard0.log
pgrep -af 'run_sample_calibrated_longbench_20260717.py'
nvidia-smi
```

扫描错误：

```bash
grep -RniE \
  'Traceback|CUDA out of memory|OutOfMemoryError|AssertionError|non-zero|Killed' \
  "$ROOT/logs"
```

### 15.5 独立验证

完成后执行：

```bash
cd /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl

/home/fdong/miniconda3/envs/moe/bin/python - <<'PY'
import csv
import glob
import os
from collections import Counter, defaultdict

root = "results/20260723_countcap_adakv_table5_official75k_full_8gpu"
rows = []
for path in glob.glob(root + "/shard[0-9]*/sample_results.csv"):
    with open(path, encoding="utf-8", newline="") as handle:
        rows.extend(csv.DictReader(handle))

counts = Counter(row["method"] for row in rows)
pairs = defaultdict(set)
for row in rows:
    pairs[(row["task"], row["sample_id"])].add(row["method"])

expected = {"full_kv", "countcap_fullprompt_keypca"}
assert os.path.exists(root + "/ALL_COMPLETE")
assert len(rows) == 7500
assert counts == Counter({method: 3750 for method in expected})
assert len(pairs) == 3750
assert all(methods == expected for methods in pairs.values())
assert len({row["task"] for row in rows}) == 16
assert max(int(row["prompt_tokens"]) for row in rows) <= 7500
print("validated", counts, "pairs=", len(pairs))
PY
```

最终读取：

```bash
cat results/20260723_countcap_adakv_table5_official75k_full_8gpu/merged/summary.json
```

---

## 16. 单元测试

在项目根目录执行：

```bash
/home/fdong/miniconda3/envs/moe/bin/python -m pytest \
  tests/test_countcap_fullprompt.py \
  tests/test_countcap_cost_gate.py \
  tests/test_longbench_official_scoring.py \
  tests/test_rescore_longbench_official.py \
  tests/test_pca_index_capacity_growth.py \
  -q
```

最近一次结果：

```text
24 passed
```

虽然主方法不使用成本门控，仍保留 `test_countcap_cost_gate.py` 是为了防止历史工程扩展破坏 runner；它不代表主方法包含 gate。

---

## 17. CSV 与速度统计口径

每行至少应包含：

```text
task
sample_id
method
executed_path
score
prompt_tokens
prefix_tokens
suffix_tokens
generated_tokens
prefill_seconds
query_seconds
decode_seconds
total_seconds
attention_link_ratio
candidate_fraction
prediction
```

对当前主方法，必须检查：

```text
method = countcap_fullprompt_keypca
executed_path = countcap_fullprompt_keypca
```

如果 `executed_path=full_kv`，说明运行的不是本文冻结主方法，或者误用了 `countcap_auto`。

速度定义：

```text
online_seconds = query_seconds + decode_seconds
total_seconds  = prefill_seconds + query_seconds + decode_seconds

online_speed = mean(full_online_seconds) / mean(sparse_online_seconds)
total_speed  = mean(full_total_seconds)  / mean(sparse_total_seconds)
```

解释：

- `query_seconds`：dense 问题/指令 suffix；
- `decode_seconds`：argmax、首次索引建立、所有 sparse generation forward；
- `prefill_seconds`：dense 文档 prefix；
- `online speed` 不等于单个 attention CUDA kernel speed；
- `total speed` 是当前 harness 中最接近完整请求延迟的指标。

对生成 `G` 个 token 的样本，稀疏路径的粗略模型是：

```text
T_sparse_online(N, G)
  = T_dense_suffix
  + T_index_build(N)
  + (G - 1) × T_sparse_step(N)
```

Full 路径是：

```text
T_full_online(N, G)
  = T_dense_suffix
  + (G - 1) × T_full_step(N)
```

当 `G=1` 时，不会执行下一次 model forward，因此也不会建立 PCA 索引。

---

## 18. 复现时必须保持的配置

在做主结果复现时，不要无意修改以下设置：

```text
method = countcap_fullprompt_keypca
score_mode = pca_int4_chunked_logscale16_sampleq_autosplit
projection_dim = 48
attention budget = min(round(2% × N), 1280)
candidate ratio = clamp(4 × attention ratio, 3%, 6%)
sampled quantile points = 256
skip_candidate_rerank = False
gqa_candidate_mode = independent
use_cuda_kernels = True
dense_suffix = True
attention implementation during sparse decode = eager patched path
prompt wrapper = llama3
dtype = float16
greedy decoding = argmax
```

不要把以下历史探索混入主结果：

```text
countcap_auto
QK-Metric warmup/rebuild
One-shot Risk
BandEF
adaptive mass budget
tail-value correction
temporal candidate reuse
router
task-specific policy
Full fallback
```

它们可以作为消融或后续研究，但必须使用不同 method name 和独立输出目录。

---

## 19. 已知限制

### 19.1 短序列速度

在 7.5K 和大量短回答上，Full SDPA 仍更快。原因不是 attention 稀疏度不够，而是：

- 第一次 PCA basis 与完整历史索引建立成本高；
- 低维扫描、阈值选择、精确候选重排和多个 kernel 有固定开销；
- Hugging Face Full SDPA 已高度优化；
- 短回答只有很少生成步，无法摊销建表成本。

主方法不回退 Full，因此必须如实报告这个适用区间，不能通过隐藏短序列样本制造加速。

### 19.2 物理 KV 显存尚未压缩

当前 harness 保留完整 K/V。PCA-INT4 是检索索引，不是原始 KV 的替代品。要实现真正低 GPU KV ratio，需要把未选中的原始 KV 放在 CPU、分层存储或分页系统中，同时解决每步候选 gather 的传输开销。

### 19.3 当前正式 7.5K 结果未完成

在 `ALL_COMPLETE` 出现并通过独立验证前，不得给出正式 AdaKV 横向比较结论。

### 19.4 当前直接长上下文证据仍不完整

128K 的 3.51x 来自 QK-Metric 版本。Key-PCA 在超长上下文上的直接质量与完整在线速度仍需补测。

### 19.5 模型覆盖不足

当前最完整结果来自 Llama-3.1-8B-Instruct。论文需要至少补充另一模型族，例如 Qwen3 和 Mistral，验证 Key-PCA basis、GQA head 映射和速度交叉点是否稳定。

---

## 20. 建议的下一步

不改变“无 Full 回退”的前提下，优先级如下。

### P0：完成并锁定官方 7.5K LongBench

等待当前 7,500 行正式实验完成，独立验证后记录：

- 16 任务 Full 与 CountCap 分数；
- macro quality retention；
- 平均 attention token/head；
- online 与 total speed；
- 分任务弱项；
- 与 AdaKV Table 5 的同协议横向比较。

### P1：直接补测 Key-PCA 32K/64K/128K

必须用当前冻结 method，而不是引用 QK-Metric 旧数字。至少报告：

- PPL 或 RULER/LongBench 质量；
- attention 接口速度；
- online decode 速度；
- 包含一次性索引成本的速度；
- 生成 8/32/64/128/256/1024 token 的 break-even；
- 峰值显存和 PCA-INT4 索引显存。

### P2：做物理 KV 分层

保持检索算法不变，只改变存储后端：

```text
完整 PCA-INT4 K index 常驻 GPU
+ 小型原始 KV hot cache 常驻 GPU
+ 其余原始 KV 在 CPU 或分页存储
+ 根据候选索引批量 gather
```

这样才能把“2% attention link”转化为真实显存节省。该实验必须单独统计 PCIe 传输与 gather 开销。

### P3：补模型与任务覆盖

建议至少覆盖：

- Llama-3.1-8B-Instruct；
- Qwen3-4B 或 Qwen3-8B；
- LongBench 16 个英文任务；
- RULER 4K/8K/16K/32K/64K/128K；
- 长文本 PPL；
- 至少一个长生成任务。

---

## 21. 给接手 Codex 的操作清单

1. 先阅读第 1、3、4、5、9 节，确认主方法定义。
2. 检查远端官方实验是否已有 `ALL_COMPLETE`。
3. 如果实验仍在运行，只监控，不重复启动，不删除 CSV。
4. 如果 worker 异常，保留日志与有效 CSV，修复后原目录续跑。
5. 完成后执行第 15.5 节的独立验证。
6. 读取 `merged/summary.json`，再按 task 做严格配对统计。
7. 检查所有 CountCap 行的 `executed_path`，不得出现 `full_kv`。
8. 将正式结果追加到本文档，不覆盖已完成的 16K 诊断结果。
9. 后续实验使用新的输出目录，主方法标识保持不变。
10. 任何新机制先作为独立 method/ablation，不要静默改变当前冻结配置。

最重要的研究口径：

> 当前主张是“使用低维量化 Key 索引，为每个 query head 检索少量历史 token，并在完整密集 prompt 之后执行始终稀疏的自回归 attention”。主张中没有 Full 回退，也没有训练式 router。
