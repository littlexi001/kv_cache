# Direct CountCap 冻结方法与复现说明

更新时间：2026-07-26

本文档描述当前冻结的实验方法。它取代
`20260723_dense_suffix_keypca_countcap_handoff_zh.md` 中“候选后精确重排到 2%”
的旧主方法定义。

## 1. 方法标识

LongBench runner 中的完整标识为：

```text
countcap_fullprompt_keypca_direct_qkvfused_qprojscan_qkvsplitauto_cacheauto_prefillindex
```

核心定义：

```text
完整 prompt 分块密集 prefill
-> 在首个 2048-token prefill chunk 后，每层、每个 KV head
   在线建立 sampled uncentered PCA48
-> 该基底在本次请求余下部分固定，后续 Key 只做增量投影和量化
-> 将全部历史 Key 投影并压缩为 grouped log-scale INT4 索引
-> 将当前 Query 投影并量化为 INT8
-> 用 256 个分层采样分数估计目标分位数
-> PCA48-INT4 全历史扫描，直接形成每个 query head 的候选
-> 不做候选内 exact-QK top-k 重排
-> 在候选原始 FP16 Q/K/V 上计算精确 sparse attention
-> 新 token 的 KV 与低比特索引增量追加
```

主方法不包含：

- 训练式 router；
- 任务名称或任务家族；
- oracle 标签；
- 字符串、BM25 或 RAG 检索；
- One-shot Risk；
- temporal reuse；
- 成本门控；
- Full Attention 回退；
- 前几层强制 Full；
- 候选后的全维 exact-QK top-k 重排。

实验中的 `full_kv` 只用于同样本基线。

## 2. Prompt 与 decode

文档、问题和 chat template 都属于 prompt，并用标准 dense attention 一次编码。
第一答案 token 直接来自 dense prompt 最后一个位置的 logits。从把第一答案 token
送回模型、预测第二答案 token 开始，所有层都执行 Direct CountCap。

该方法研究的是长上下文 decode，不声称加速 dense prefill。

## 3. Key-PCA 索引

对某层、某个 KV head 的 post-RoPE 历史 Key：

$$
K\in\mathbb R^{N\times d},
$$

必须注意，冻结实现使用 `prefill_chunk_tokens=2048` 和增量索引回调。第一次回调
发生在首个 dense prefill chunk 结束后，此时从

$$
K_{\mathrm{prefix}}=K[0:L],\qquad L=\min(N,2048)
$$

每隔 32 个 token 取一个样本：

$$
K_s=K_{\mathrm{prefix}}[0::32].
$$

计算未中心化二阶矩：

$$
\widehat C_K=\frac{1}{|K_s|}K_s^\mathsf TK_s.
$$

对 $\widehat C_K$ 做 `torch.linalg.eigh`，取最大 48 个特征值对应的特征向量：

$$
V_{48}\in\mathbb R^{d\times48}.
$$

因为没有减均值，这等价于对首个 prefill chunk 的采样 Key 做 truncated right
SVD。基底按 `layer × KV head` 独立建立，初次建立后在该次生成中固定。后续
prefill chunk、问题 suffix 和生成 token 的 Key 都只投影到这个固定基底并追加到
INT4 索引。

因此准确名称是 **当前请求首段条件化的 online prefix-PCA**，不是完整历史
full-SVD，也不是在整个 prompt 上每 32 个 token 重新估计一次 PCA。已有完整历史
PCA/SVD 的谱实验是理想参考；新增实验会单独报告真实首 2048-token basis 对完整
QK 分数矩阵的保真度。

历史 Key 的低维坐标为：

$$
z_i=V_{48}^\mathsf Tk_i.
$$

48 维坐标按连续 16 维一组，使用 grouped log-scale INT4 存储。当前 query 同样
投影到该子空间，并做对称 INT8 量化。INT4/INT8 只用于候选检索；候选进入
attention 后仍读取原始 FP16 K/V。

## 4. 目标预算

长度为 $N$ 时，每个 query head 的目标 attention 数量为：

$$
B(N)=
\min\left(
N,\,
1280,\,
\max\left(256,\left\lceil0.06N\right\rceil\right)
\right).
$$

| 历史长度 | 目标 token/head | 目标比例 |
|---:|---:|---:|
| 2K | 256 | 12.50% |
| 4K | 256 | 6.25% |
| 8K | 492 | 6.00% |
| 16K | 960 | 6.00% |
| 24K | 1280 | 5.33% |
| 32K | 1280 | 4.00% |
| 64K | 1280 | 2.00% |
| 128K | 1280 | 1.00% |

这是解析式、与任务无关的规则。

当 $N\le256$ 时，公式自然给出 $B(N)=N$。实现直接对全部可用历史做精确
attention；这是预算覆盖全集后的数学饱和边界，不是根据风险、任务或速度触发的
Full fallback。除此之外，冻结方法没有 Full Attention 回退。

### 4.1 目标数量不是严格 hard cap

当前实现不计算完整代理分数后做精确 top-$B$ 排序，而是用 sampled quantile
直接压紧候选。为了容纳分位数估计误差，CUDA buffer 容量为：

$$
C(N)=
\min\left(
N,\,
\max\left\{
\frac{4}{3}fN,\,
(f+0.02)N
\right\}
\right)
$$

用于高精度 1024 点采样模式；当前 256 点模式使用：

$$
\boxed{
C(N)=
\min\left(
N,\,
\max\left\{
2fN,\,
(f+0.04)N
\right\}
\right)
},
\qquad f=\frac{B(N)}{N}.
$$

实际消费数量是超过 sampled threshold 的 token 数，通常接近 $B(N)$，但最坏
可以达到 $C(N)$。因此：

- `configured_attention_tokens` 是目标值；
- buffer capacity 不是实际平均 attention 数；
- 论文必须报告实际消费均值、p95 和最大值；
- 在加入显式 proxy top-k 之前，不能称为“strict 1280 hard cap”。

### 4.2 实际消费审计

在 LongBench 16 任务上开启诊断，对 Llama-3.1-8B 和 Qwen3-4B 各取 64 条样本。
该审计记录每层、每个 query head 的真实消费，而不是用目标值代替：

| 模型 | 平均 prompt | 目标比例 | 实际比例 | 样本内 p95 比例均值 | 实际 token/head 均值 | p95 token 均值 | 单 head 最大 token | overflow head 比例 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-3.1-8B | 6476.5 | 7.116% | 7.263% | 9.442% | 420.8 | 558.6 | 1036 | 0.0218% |
| Qwen3-4B | 6512.0 | 7.063% | 7.262% | 9.546% | 427.2 | 572.4 | 1044 | 0.0316% |

平均目标比例高于 6%，是因为样本中包含短于约 4.3K 的 prompt，预算公式的
256-token floor 此时占比更高。实际均值只比目标高约 0.15--0.20 个百分点。
单样本中的最大比例可达到约 25%，同样出现在约 1.5K--1.7K 的短 prompt，
对应的绝对 token 数只有约 388--414；长 prompt 上观察到的最大绝对消费为
1036/1044，仍低于 1280。

冻结的 `qprojscan/qkvfused` 实现属于异步 sampled-quantile 路径。若某个 head
超过 $C(N)$，CUDA kernel 会在 capacity 处截断；它不会同步回到 host 执行完整
proxy top-k。本次 overflow 只占 0.0218%/0.0316% 的 head，host-side
`sampled_quantile_fallback` 为 0%。这不是 Full Attention 回退。

## 5. 256 点 sampled-quantile

对每个 query head，在历史位置上取 256 个近似均匀、带 layer-dependent offset
的样本。设目标比例为 $f$，则在样本代理分数上估计 $(1-f)$ 分位数阈值
$\widehat\tau_f$。

融合 kernel 扫描全部 PCA48-INT4 Key，并保留：

$$
\widehat s_i\ge\widehat\tau_f
$$

的 token，直到候选容量 $C(N)$。因此它的作用是避免：

```text
写出 N 个 FP32 proxy score
-> 调用全局 torch.topk
-> 再把候选交给 attention
```

采样只估计阈值，PCA48-INT4 代理分数仍会扫描全部历史 token。最终候选数量可随
当前 query 的分数分布轻微变化，但目标比例只由长度决定。

## 6. Direct sparse attention

候选集合记为 $\widehat S$。Direct 表示不再执行：

```text
候选原始 K 精确 QK
-> exact top-2%
-> value attention
```

而是直接在 $\widehat S$ 对应的原始 FP16 K/V 上计算：

$$
a_i=
\frac{
\exp(q^\mathsf Tk_i/\sqrt d)
}{
\sum_{j\in\widehat S}\exp(q^\mathsf Tk_j/\sqrt d)
},
\qquad
o=\sum_{i\in\widehat S}a_iv_i.
$$

所以 PCA/INT4 误差只影响集合 $\widehat S$，不会继续污染候选内的 QK 数值和
Value。`qprojscan` 融合 query 投影、INT8 量化、PCA48-INT4 scan 和候选压紧；
`qkvsplitauto` 根据长度和候选规模选择 QKV 消费的 split 数。

GQA 中 PCA 索引按 KV head 建立，但映射到同一 KV head 的不同 query head 使用
各自 query，独立产生候选。

## 7. Cache 的准确口径

`cacheauto` 在当前 runner 中表示：

- 短于默认 14K：Hugging Face `DynamicCache`；
- 不短于 14K：`PreallocatedDynamicCache`，避免逐 token `torch.cat`。

两者都把完整原始 FP16 K/V 留在 GPU。它不是 CPU offload，也不是 10% GPU hot
cache。PCA48-INT4 是附加检索索引。

RTX 3090 24GB 上的实测硬件要求：

- Qwen3-4B 的 64K Full/CountCap 配对可在单卡运行；
- Qwen3-4B 的 128K 单卡会在 `PreallocatedDynamicCache` 分配完整 FP16 K/V
  时 OOM，需要至少两张 24GB 卡按层均衡承载模型与 cache；
- Llama-3.1-8B 的当前 64K 配置使用两卡，128K 使用四卡。

这些卡数由完整 FP16 K/V 和模型权重决定，不能从 1%--2% 的每步 attention
消费比例推导。若未来接入 CPU/full-KV offload 或低比特物理 KV，显存结论需要
单独重测。

因此当前 final-direct LongBench 能证明：

- 稀疏 attention 质量；
- 实际 decode 路径时间；
- direct retrieval 不回退 Full。

它不能单独证明物理 GPU KV 已压缩到目标 attention 比例。此前约 10% GPU KV、
CPU pinned full KV、hot-cache 的结果属于另一套 hierarchical physical-cache
实现，不能与这里混写。

## 8. 数学解释

完整推导位于：

```text
docs/20260726_countcap_spectral_stability_mathematical_appendix_zh.md
```

最关键的四条结论是：

1. 未中心化 Key-PCA 与它实际看到的首段 sampled Key 的 truncated right SVD
   子空间等价。
2. softmax 对逐 query 常数平移不敏感，真正相关的谱矩阵是按 token 维中心化的
   $S^\circ=Q(HK)^T/\sqrt d$，不能用原始 QK 中的常数 rank-1 模态论证效果。
3. 如果候选遗漏 full-attention mass 为 $\eta$，候选内精确重算后严格有
   $\|p-\widetilde p\|_1=2\eta$，且
   $\|o-\widetilde o\|_2\le\eta\operatorname{diam}(V)$。
4. 若 full 与 sparse 的最终 logit 差为 $d$，令
   $R(d)=\max d-\min d$，则 full top-1 margin 大于 $R(d)$ 时 token 不变，
   任意目标 NLL 变化不超过 $R(d)$，并且
   $D_{\mathrm{KL}}(p_{\mathrm{full}}\|p_{\mathrm{sparse}})\le R(d)^2/8$。

不能声称 PCA 尾部“没有语义”，也不能证明任意 query 下答案不变。严格结论是
条件稳定性，跨模型实验负责验证自然输入满足这些条件的频率。

## 9. 当前已完成证据

32K 双模型中心化 QK 谱：

| 模型 | 原始 QK 有效秩 | 中心化 QK 有效秩 | 中心化最优 rank-48 | Full uncentered Key-PCA48 | 真实 first-2K basis | First-2K 中心化 cosine |
|---|---:|---:|---:|---:|---:|---:|
| Llama-3.1-8B | 1.22 | 4.72 | 99.09% | 91.10% | 62.39% | 0.7953 |
| Qwen3-4B | 2.79 | 5.43 | 98.52% | 91.45% | 59.07% | 0.7609 |

原始 QK 中被行中心化删除的 softmax 无效能量，Llama/Qwen 平均分别为
88.83%/62.70%。中心化后仍然低秩，但真实 first-2K basis 与最优 QK 子空间
之间存在明显差距；不能用原始 QK 近似 rank-1 的现象夸大生产代理精度。

32K Llama 真实 Q/K/V production-aligned 探针：

| 指标 | 结果 |
|---|---:|
| Key 有效秩均值 | 24.67 |
| PCA48 Key 谱能量 | 86.84% |
| sampled PCA48 与 full SVD48 子空间重合 | 91.50% |
| 理想 full-history sampled 代理与精确 QK Pearson | 0.9338 |
| 真实 first-2K PCA48 FP32 中心化 Pearson | 0.7771 |
| 真实 first-2K + INT4 K + INT8 Q 中心化 Pearson | 0.7712 |
| INT4 新增 score error energy / 精确 score energy | 1.01% |
| Prefix/PCA 误差与 INT4 误差 cosine | 0.00042 |
| 4% Exact-QK 候选 attention mass | 91.45% |
| 4% 理想 full-history sampled 候选 mass | 90.38% |
| 4% 生产代理候选 attention mass | 86.47% |
| 4% 生产代理 exact top-k recall | 47.34% |
| 4% 生产代理 mass-weighted recall | 92.38% |
| attention 输出界验证 | 6400/6400 通过 |

32K 体育与医学 targeted PPL：

| 方法 | PPL | 相对 Full |
|---|---:|---:|
| Full | 8.3930 | 100% |
| Direct CountCap | 8.5064 | +1.35% |

这组 PPL 使用的 target budget 为 1280，但 sampled-quantile 的执行容量可达
2560；它不是 strict-1280 质量证明。

## 10. 正在运行的正式实验

目录：

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/results/
  20260726_final_direct_multimodel_m100_ctx7500/
```

设置：

- Llama-3.1-8B-Instruct；
- Qwen3-4B-Instruct；
- LongBench 16 个英文任务；
- 每任务 100 条；
- 每模型 1600 个严格 Full/CountCap 配对；
- max context 7500；
- 无 Full 回退；
- 最终 Direct 方法标识与本文第 1 节完全一致。

完成后由下面脚本生成 16-task、RaBitQ-compatible 13-task、分任务、bootstrap CI
和同环境基线对照：

```text
src/summarize_final_direct_multimodel_comparison_20260726.py
```

随后两模型分别运行 32K、四主题、3072 个 token-logit 配对，验证 KL、top-1、
margin certificate 与 NLL。

## 11. 复现命令

在服务器项目根目录：

```bash
cd /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl

RUN_TAG=20260726_final_direct_multimodel_m100_ctx7500 \
SAMPLES=100 \
MAX_CONTEXT_TOKENS=7500 \
bash scripts/launch_countcap_final_direct_multimodel_longbench_4gpu_20260726.sh
```

本地配置与数学测试：

```powershell
$env:PYTHONPATH='ymluo/projects/qwen3_top2_head_limit3_ppl/src'
python -m pytest `
  ymluo/projects/qwen3_top2_head_limit3_ppl/tests/test_countcap_benchmark_config.py `
  ymluo/projects/qwen3_top2_head_limit3_ppl/tests/test_countcap_fullprompt.py `
  ymluo/projects/qwen3_top2_head_limit3_ppl/tests/test_direct_countcap_logit_stability.py `
  ymluo/projects/qwen3_top2_head_limit3_ppl/tests/test_final_direct_multimodel_comparison.py `
  -q
```

## 12. 最接近工作与创新边界

PCA/SVD 近似 QK 做 token selection 已被 Loki、LRQK 等工作使用；量化 Key
索引也不是首次出现。不能把“PCA48 找 top-k”单独作为创新。

当前可主张的组合是：

- 当前序列在线、per-layer/per-KV-head sampled spectral basis；
- grouped log-scale INT4 K 与 INT8 Q；
- 256 点 sampled threshold，无全局 proxy score materialization；
- 候选直接进入精确 sparse attention，无全维 exact rerank；
- 无训练、无任务特征、所有层稀疏、无 Full fallback；
- 从谱误差到 logit/NLL 的逐阶段理论和实测 error ledger；
- 长度封顶的目标预算与融合 CUDA 路径。

详细相关工作边界见：

```text
docs/20260726_countcap_closest_work_and_comparison_protocol_zh.md
```

## 13. 必须保持诚实的限制

1. 7.5K 附近 sparse decode 未必比 Full 快，一次性建索引成本尤其明显。
2. `1280` 当前是目标，不是 hard cap。
3. final-direct LongBench 仍保留完整 GPU K/V。
4. PCA48 对 tail-aligned adversarial query 没有无条件保证。
5. Loki、LRQK、RaBitQCache 是必须正面对比的近邻。
6. 旧 99.57% LongBench 是“Key-PCA + exact rerank 2%”路径，不能替代当前
   Direct 正式结果。
7. 旧 128K 3.51x 是 QK-Metric 路径，不能自动归到当前 Key-PCA Direct。
