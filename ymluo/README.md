# Qwen3 KV Cache 研究工作区

> 文档中文化与最近同步日期：2026-05-18

## 研究主题

新增日期：2026-05-14

这个工作区围绕一个核心问题展开：**能否把长上下文 KV cache 看成一个可索引、可压缩、可检索的记忆系统，而不是每次 decode 都密集扫描的扁平 token 序列。**

当前主要方向包括：

- 在精确 attention 前先做 block/chunk 级候选召回。
- 对旧 KV memory 做学习式压缩，同时保留 recent tokens 和 anchor tokens。
- 用 attention energy 估计在有限 loss 影响下可以丢弃多少上下文。
- profile K-cache 的数值、相邻 token delta、范数、pairwise cosine，理解不同 layer/head 到底存了什么。
- 用可控 synthetic retrieval 任务验证 anchor-only KV、U-Net mask schedule 与 answer-only training 是否能学到可检索记忆。
- 用 interval subsequence next-token 任务检查 U-Net stride schedule 对局部/跨步序列规律的建模能力，并导出 attention scores 做层/head 诊断。

更完整的搜索系统类比和动机见：

```text
KVCache_Indexing_Knowledge_Retrieval_2026-05-09.md
```

下面的命令默认从仓库根目录 `kv_cache` 运行。如果已经 `cd ymluo`，去掉命令里的 `ymluo/` 前缀。

## 目录结构

新增日期：2026-05-14

| 路径 | 作用 |
| --- | --- |
| `KVCache_Indexing_Knowledge_Retrieval_2026-05-09.md` | 把搜索、向量索引、层级结构、知识图谱等思想映射到 KV-cache lookup 的研究笔记。 |
| `projects/qwen3_chunk_routing` | Qwen3-0.6B chunk attention 训练框架，包含 `baseline`、`oracle`、`router` 模式。 |
| `projects/qwen3_unet_synthetic_retrieval` | 基于 `fdong` mask-based U-Net Transformer 的可控 synthetic retrieval 评估与 answer-only 训练。 |
| `projects/qwen3_interval_subseq_retrieval` | 基于 fdong-style 8-layer U-Net Qwen3 的 arithmetic subsequence next-token 训练和 attention score dump。 |
| `projects/pyramid_kv_compression` | 继续预训练实验：用学习到的 summary 替换旧的中间层 KV blocks。 |
| `projects/qwen3_kcache_avg_topk` | 推理期 sparse decode 实验：用 K-cache block 平均向量做 top-k block 选择。 |
| `projects/qwen3_kcache_norm_analysis` | Qwen3-0.6B 的 K-cache norm、attention energy、pruning loss/PPL 分析。 |
| `projects/qwen3_kcache_value_delta_analysis` | Qwen3-8B 的 K-cache 取值、范数、相邻 token delta 分布分析。 |
| `projects/qwen3_kcache_cosine_heatmap` | Qwen3-0.6B 的 K-cache token-token cosine 热力图分析，按 layer 和 KV head 输出。 |
| `logs/` | 历史日志或 workspace snapshot，普通实验入口不依赖它。 |
| `utils/` | 预留共享工具目录。 |

每个项目都有自己的 README 和 scripts；本文件只做总览和关键结论记录。

## 项目说明

新增日期：2026-05-14

### `qwen3_chunk_routing`

新增日期：2026-05-14

这个项目比较 Qwen3-0.6B 的三种 attention 模式：

- `baseline`：原始 full attention。
- `oracle`：先计算完整 attention score，再把有效历史 token 分成 20 个 chunks；保留 chunk 1、recent chunk，以及 attention mass 最高的 3 个中间 chunks。
- `router`：用轻量 learned router 从 chunk summaries 预测 top 3 中间 chunks，再做精确 attention。

运行示例：

```bash
bash ymluo/projects/qwen3_chunk_routing/scripts/run_8gpu.sh baseline
bash ymluo/projects/qwen3_chunk_routing/scripts/run_8gpu.sh oracle
bash ymluo/projects/qwen3_chunk_routing/scripts/run_8gpu.sh router
```

脚本默认使用 `torchrun --nproc_per_node=8`。它从 `MODEL_PATH` 读取 tokenizer 和 config；除非修改项目代码，否则模型权重从头初始化。

### `qwen3_unet_synthetic_retrieval`

新增日期：2026-05-14；最近同步日期：2026-05-15

这个项目把 `fdong/unet_transformer.md` 第 7 节的可控 synthetic retrieval 任务落到可运行评估和训练脚本上，用来检查 mask-based U-Net Transformer checkpoints 在只保留 anchor KV 的情况下是否还能找回答案。

当前任务直接生成 token-id 序列，避免 tokenizer segmentation 影响可控实验：

- Variant A：固定 4-token patterns。
- Variant B：随机 3-token content blocks，后接共享 anchor marker。

评估会对每个 checkpoint 和任务变体报告 answer-only loss/accuracy，并比较三条路径：

- full-sequence forward。
- teacher-forced decode with full KV cache。
- teacher-forced decode with anchor-only KV cache。

默认 checkpoint 覆盖 `baseline`、`unet-4`、`unet-4-8-4`、`unet-4-8-16-8-4`。如果 checkpoint 目录里有 `runtime_config.json`，评估脚本会直接读取；否则回退到这些 run name 的已知 stride schedule。

运行评估：

```bash
bash ymluo/projects/qwen3_unet_synthetic_retrieval/scripts/run_synthetic_eval.sh
```

快速 smoke test：

```bash
NUM_SAMPLES=8 BATCH_SIZE=1 \
bash ymluo/projects/qwen3_unet_synthetic_retrieval/scripts/run_synthetic_eval.sh
```

最近新增的训练入口支持 answer-only loss，默认训练 `unet-4` schedule 的 Variant B，并使用 `TRAIN_MODE=anchor_kv_decode`，也就是训练路径和评估里的 anchor-only KV pruning 路径一致：

```bash
bash ymluo/projects/qwen3_unet_synthetic_retrieval/scripts/run_train_synthetic.sh
```

后台训练：

```bash
bash ymluo/projects/qwen3_unet_synthetic_retrieval/scripts/nohup_train_synthetic.sh
```

这个训练只对最终 answer prediction 施加 cross-entropy，不对前 1023 个 next-token 位置施加 loss。常用覆盖参数：

```bash
MODEL_NAME=unet-4-8-4 \
VARIANT=A \
RUN_NAME=unet-4-8-4-variant-a-answer-only \
TOTAL_STEPS=20000 \
BATCH_SIZE=8 \
TRAIN_MODE=anchor_kv_decode \
bash ymluo/projects/qwen3_unet_synthetic_retrieval/scripts/run_train_synthetic.sh
```

主要输出：

```text
ymluo/projects/qwen3_unet_synthetic_retrieval/outputs/synthetic_eval/metrics.json
ymluo/projects/qwen3_unet_synthetic_retrieval/outputs/synthetic_eval/metrics.csv
ymluo/projects/qwen3_unet_synthetic_retrieval/outputs/train/<run_name>/metrics.jsonl
ymluo/projects/qwen3_unet_synthetic_retrieval/outputs/train/<run_name>/checkpoints/<step>.pth
ymluo/projects/qwen3_unet_synthetic_retrieval/outputs/train/<run_name>/checkpoints/runtime_config.json
```

### `qwen3_interval_subseq_retrieval`

新增日期：2026-05-18

这个项目训练一个小型 fdong-style `MyQwen3ForCausalLM`，用直接生成的 token-id arithmetic subsequences 做标准 causal next-token prediction。它和上面的 answer-only synthetic retrieval 不同：没有 query token、placeholder 或最终答案位置，所有 1023 个有效 next-token 位置都会贡献 loss。

默认数据配置：

```text
total_token = 10000
subseq_len = 4
seq_len = 1024
intervals = 1
```

`interval=1` 时样本来自 `[1,2,3,4]`、`[5,6,7,8]` 这类连续 4-token subsequences；`interval=2` 等更大 interval 会生成 `[2,4,6,8]`、`[10,12,14,16]` 这类跨步 pattern。每条训练样本随机抽取 256 个 subsequences 并拼成 1024-token 序列。

模型默认把 Qwen config 覆盖为 8 层，并使用 U-Net 风格 stride schedule：

```text
num_hidden_layers = 8
attention_stride_pattern = 1,1,4,4,4,4,1,1
```

当前实验结果：

- 8 层 Qwen3-0.6B 配置，stride schedule 为 `1,1,4,4,4,4,1,1`。
- 在 `interval=1`、`subseq_len=4` 的标准设置下，模型基本在 1500 step 左右就能把可学习的局部递推规则完全学会。
- 当前观测到的 next-token accuracy 是 `0.7508`，和理论上限基本一致。原因是训练序列由彼此独立的 4-token subsequences 拼接而成，例如 `[5,6,7,8,1,2,3,4]` 中，`5->6`、`6->7`、`7->8`、`1->2`、`2->3`、`3->4` 这些 subsequence 内部转移可以被完全预测；但 `8->1` 是两个独立 subsequences 的边界，数据构造时没有给 `8` 和 `1` 建立任何关系，因此模型无法从前文确定边界后的第一个 token。
- 因此每 4 个 token 里大约只有 3 个 next-token 转移是确定可学的，理论 accuracy 约为 `3/4 = 0.75`。观测值 `0.7508` 中高出 `0.75` 的约 `0.0008` 更可能来自边界位置上的随机猜测或评估噪声，而不是模型学到了跨 subsequence 边界关系。

单卡训练：

```bash
bash ymluo/projects/qwen3_interval_subseq_retrieval/scripts/run_train.sh
```

多卡 DDP 训练：

```bash
CUDA_DEVICES=0,1,2,3 \
bash ymluo/projects/qwen3_interval_subseq_retrieval/scripts/run_ddp_train.sh
```

常用覆盖参数：

```bash
INTERVALS=1,2,4 \
RUN_NAME=unet8-intervals-1-2-4 \
TOTAL_STEPS=20000 \
BATCH_SIZE=4 \
TRAIN_MODE=full_sequence_lm \
bash ymluo/projects/qwen3_interval_subseq_retrieval/scripts/nohup_train.sh
```

如果 `INTERVAL_GROUP_MODE=scaled`，最大 token id 是 `TOTAL_TOKEN * max(INTERVALS)`；超过基础 Qwen vocab 时，训练脚本默认会自动扩大 `config.vocab_size`，除非显式设置 `AUTO_RESIZE_VOCAB=false`。

保存 checkpoint 后可以导出每层/head 的 masked raw attention scores 和 softmax probabilities：

```bash
RUN_NAME=unet8-interval1-lm-ddp \
CKPT_STEP=2000 \
bash ymluo/projects/qwen3_interval_subseq_retrieval/scripts/dump_attention_scores.sh
```

主要输出：

```text
ymluo/projects/qwen3_interval_subseq_retrieval/outputs/train/<run_name>/metrics.jsonl
ymluo/projects/qwen3_interval_subseq_retrieval/outputs/train/<run_name>/checkpoints/<step>.pth
ymluo/projects/qwen3_interval_subseq_retrieval/outputs/train/<run_name>/checkpoints/runtime_config.json
ymluo/projects/qwen3_interval_subseq_retrieval/outputs/train/<run_name>/attention_scores/step_<ckpt_step>/
```

### `pyramid_kv_compression`

新增日期：2026-05-14

这个项目 patch Qwen3 attention，用 pyramid-shaped KV memory 做继续预训练：

```text
early layers:   full KV
middle layers:  compressed older KV
final layers:   full KV
```

hidden-state 序列长度保持不变。只有被选中的层会缩短 attention memory：旧的中间 blocks 会被学习到的 weighted K/V summaries 替换，anchor tokens 和 recent tokens 保持原始 KV。

推荐实验顺序：

```bash
bash ymluo/projects/pyramid_kv_compression/scripts/run_8gpu.sh sanity
bash ymluo/projects/pyramid_kv_compression/scripts/run_8gpu.sh compressor
bash ymluo/projects/pyramid_kv_compression/scripts/run_8gpu.sh attention
bash ymluo/projects/pyramid_kv_compression/scripts/run_8gpu.sh full
```

`METHOD_NOTES.md` 记录了当前风险：激进默认 schedule 已经产生过高 loss。下一轮严肃实验应从弱压缩中间层、更大的 anchor/recent window、以及 `attention` 阶段开始，再考虑 full-model training。

### `qwen3_kcache_avg_topk`

新增日期：2026-05-14

这是一个推理期 sparse decode 实验。Layer 0-2 走原始 attention；Layer 3-27 把当前 K cache 切成 blocks，对每个 block 内部的 keys 求平均，用当前 query 打分，保留 top block fraction，然后只在被选中的原始 K/V token 上做精确 attention。

生成文本：

```bash
MODEL_PATH=/mnt/workspace/lym_code/models/Qwen3-0.6B \
bash ymluo/projects/qwen3_kcache_avg_topk/scripts/run_generate.sh
```

评估 baseline vs sparse decode：

```bash
MODEL_PATH=/mnt/workspace/lym_code/models/Qwen3-0.6B \
DATA_PATH=/mnt/workspace/dclm/global-shard_01_of_10/local-shard_0_of_10 \
bash ymluo/projects/qwen3_kcache_avg_topk/scripts/run_eval.sh
```

默认 sparse 设置：

```text
BLOCK_SIZE=10
TOPK_RATIO=0.30
FIRST_SPARSE_LAYER=3
LAST_SPARSE_LAYER=27
```

### `qwen3_kcache_norm_analysis`

新增日期：2026-05-14

这个项目在 DCLM 文本前缀上运行 Qwen3-0.6B，并输出：

- token-level next-token loss 和 PPL。
- 原始 K-cache norm 的 layer/head summary。
- attention energy 的 layer/query-head summary。
- 达到不同 attention-energy threshold 所需的 top-k token 数。
- pruning 掉低 attention-energy 位置后的 loss/PPL。

运行：

```bash
bash ymluo/projects/qwen3_kcache_norm_analysis/scripts/run_analysis.sh
```

当前 3000 tokens summary 实验结论：

- 90% attention energy 已接近 full attention：loss 增加 `0.014206`，PPL 约 `1.0143x`。
- 95% attention energy 在该样本上几乎无损：loss 增加 `0.002030`，PPL 约 `1.0020x`。
- 50% 和 75% energy pruning 对质量过于激进。
- 达到同一 energy threshold 所需 token 数在 layer/head 间差异很大，因此固定 token top-k 不如 adaptive energy-based selection 合理。

详细表格和解释见：

```text
projects/qwen3_kcache_norm_analysis/attention_energy_loss_summary.md
```

### `qwen3_kcache_value_delta_analysis`

新增日期：2026-05-14

这个项目 profile 一次 Qwen3-8B forward pass，用 `past_key_values` 构建最终 K cache，并分析：

```text
k_i
delta(k_i) = k_i - k_{i-1}
```

它会输出 per-head、per-layer、global statistics，精确 histograms，timing rows，以及可选 plots。默认运行 5000 个 DCLM tokens。

运行：

```bash
bash ymluo/projects/qwen3_kcache_value_delta_analysis/scripts/run_analysis.sh
```

快速 smoke test：

```bash
MAX_TOKENS=128 CHUNK_SIZE=32 SAVE_HEAD_HISTOGRAMS=false \
bash ymluo/projects/qwen3_kcache_value_delta_analysis/scripts/run_analysis.sh
```

### `qwen3_kcache_cosine_heatmap`

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

新增日期：2026-05-14

按 layer band 聚合后的趋势很明显：

| 层范围 | 平均 `offdiag_mean` | 解释 |
| --- | ---: | --- |
| `L0-L5` | `0.883` | 早期层 K 向量高度相似，存在明显方向冗余。 |
| `L6-L17` | `0.602` | 中间层相似度下降，head 间分化开始变强。 |
| `L18-L26` | `0.403` | 后期层 K 向量更分散，压缩需要谨慎。 |
| `L27` | `0.716` | 最后一层相似度重新上升，可能有特殊输出前表示结构。 |

这个结果说明：**KV cache 压缩不应该全层统一设置压缩率。** 早期层可以更激进；`L18-L26`，尤其 `L23-L26`，应更保守。

### 极端 head

新增日期：2026-05-14

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

## 对 KV Cache 压缩的含义

新增日期：2026-05-14

这份 cosine 结果说明 K-cache 中确实存在大量结构性冗余，但不能直接推出“cosine 高的 token 就可以合并或删除”。

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

## 下一步建议

新增日期：2026-05-14

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

新增日期：2026-05-14

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

## 常用默认参数

新增日期：2026-05-14

大多数脚本都可以用环境变量配置：

```bash
MODEL_PATH=/path/to/Qwen3
DATA_PATH=/path/to/dclm
TEXT_PATH=/path/to/part-00000.txt
OUTPUT_DIR=/path/to/output
MAX_TOKENS=3000
CHUNK_SIZE=256
DEVICE=cuda
DTYPE=bfloat16
```

训练脚本默认使用 Hugging Face streaming 读取大型 DCLM 风格数据。除非确定磁盘足够并且想构建 Arrow cache，否则保持 `STREAMING=true`。

synthetic retrieval 项目的评估和训练还常用：

```bash
NUM_SAMPLES=256
BATCH_SIZE=4
VARIANTS=A,B
TRAIN_MODE=anchor_kv_decode
TOTAL_STEPS=20000
```

interval subsequence 项目的训练和 attention dump 常用：

```bash
CONFIG_DIR=/mnt/workspace/Qwen3-0.6B
TOTAL_TOKEN=10000
SUBSEQ_LEN=4
SEQ_LEN=1024
INTERVALS=1,2,4
ATTENTION_STRIDE_PATTERN=1,1,4,4,4,4,1,1
CKPT_STEP=2000
QUERY_POSITIONS=last
```

## 2026-05-19 MoE Selectivity 合成数据实验记录

指标说明：`same_higher_by_layer` 表示同一个 higher-level unit 被分发到同一个 expert 的比例；`higher_mass_by_layer` 表示 attention 落在同一 higher-level history token 上的质量；`expert_load` 表示不同 expert 的平均路由负载。

### Exp1：对其它 expert / gate 列向量施加 negative gradient

实验目的：让别的 expert / gate 的其它列向量拿到反向负梯度，观察是否能提升 expert selectivity。

| step | loss | acc | same_higher_by_layer | higher_mass_by_layer | expert_load |
| --- | ---: | ---: | --- | --- | --- |
| 100 | 1.4925 | 0.7676 | L0:0.4316 L1:0.5558 L2:0.6765 | L0:0.2997 L1:0.2578 L2:0.3173 | [0.1736, 0.2397, 0.2717, 0.3149] |
| 1100 | 0.2673 | 0.9128 | L0:0.4673 L1:0.9781 L2:1.0000 | L0:0.3636 L1:0.4257 L2:0.4200 | [0.1393, 0.3073, 0.2292, 0.3240] |
| 10000 | 0.2607 | 0.9139 | L0:0.4568 L1:0.9826 L2:1.0000 | L0:0.3808 L1:0.4341 L2:0.4273 | [0.1402, 0.3076, 0.2288, 0.3232] |

结论：为其它 expert 施加反向负梯度效果非常好，基本能够让每个 expert 专注于同一个 higher-level unit 元素。

### Exp2：小 batch / token 更新

实验目的：每次只更新一个 token，验证类似人脑 inhibition 的逐条数据学习机制是否能提升 gating selectivity。

| step | loss | acc | same_higher_by_layer | higher_mass_by_layer | expert_load |
| --- | ---: | ---: | --- | --- | --- |
| 10000 | 4.5524 | 0.0556 | L0:0.7572 L1:0.5807 L2:0.7301 | L0:0.2470 L1:0.2680 L2:0.2648 | [0.2160, 0.3237, 0.1273, 0.3328] |
| 100000 | 1.0697 | 0.7883 | L0:0.2849 L1:0.5347 L2:0.5712 | L0:0.2746 L1:0.2415 L2:0.2627 | [0.2137, 0.2238, 0.2935, 0.2687] |
| 300000 | 0.9383 | 0.8090 | L0:0.3002 L1:0.6541 L2:0.6161 | L0:0.2947 L1:0.2485 L2:0.2763 | [0.2323, 0.2549, 0.2259, 0.2866] |

结论：逐 token 学习效果不错，基本能够让同一个 higher-level unit 元素分发到同一个 expert 里。缺点是训练速度太慢，预计 100w step 才能达到之前约 1000 step 的训练 loss。

### Exp3：expert 初始化调整

实验目的：验证 expert 初始化或 gate 初始化是否显著影响最终训练出的 selectivity。

#### Exp3.1：expert 两两 Frobenius 内积正交初始化后直接训练

| step | loss | acc | same_higher_by_layer | higher_mass_by_layer | expert_load |
| --- | ---: | ---: | --- | --- | --- |
| 100 | 1.4884 | 0.7679 | L0:0.3209 L1:0.3625 L2:0.3840 | L0:0.2994 L1:0.2572 L2:0.3140 | [0.2174, 0.2864, 0.2649, 0.2311] |
| 1000 | 0.2687 | 0.9126 | L0:0.3033 L1:0.3477 L2:0.3419 | L0:0.3483 L1:0.3862 L2:0.3762 | [0.2429, 0.2413, 0.2692, 0.2463] |
| 10000 | 0.2608 | 0.9125 | L0:0.3027 L1:0.3207 L2:0.3410 | L0:0.3602 L1:0.4174 L2:0.3955 | [0.2430, 0.2486, 0.2365, 0.2717] |

结论：Frobenius 内积正交初始化没法有效让相同 higher-level unit 选择相同 expert。

#### Exp3.2：前 1000 步强制同一 higher-level unit 选择相同 expert

| step | loss | acc | same_higher_by_layer | higher_mass_by_layer | expert_load |
| --- | ---: | ---: | --- | --- | --- |
| 100 | 1.5110 | 0.7650 | L0:0.3451 L1:0.3422 L2:0.3560 | L0:0.2996 L1:0.2564 L2:0.3118 | [0.3052, 0.2031, 0.2548, 0.2367] |
| 1000 | 0.4252 | 0.8808 | L0:0.4413 L1:0.4842 L2:0.5174 | L0:0.3351 L1:0.2831 L2:0.2990 | [0.5068, 0.0635, 0.0426, 0.3869] |
| 10000 | 0.2604 | 0.9136 | L0:0.3796 L1:0.3106 L2:0.3158 | L0:0.3851 L1:0.4466 L2:0.4259 | [0.2868, 0.2260, 0.1511, 0.3360] |

结论：前置强制选择 expert 的初始化会在前 1000 步让 `same_higher_by_layer` 明显上升，但去除 force 选择后该指标明显下降，1w step 后下降到接近正常水平。

Exp3 总结：初始化参数对模型后续训练影响较小，模型会逐渐将参数向 baseline 靠近。

## 推荐阅读顺序

新增日期：2026-05-14；最近同步日期：2026-05-18

1. `KVCache_Indexing_Knowledge_Retrieval_2026-05-09.md`：理解 retrieval-system framing。
2. `projects/qwen3_kcache_cosine_heatmap/README.md`：理解 K-cache pairwise cosine 热力图生成方法。
3. `projects/qwen3_kcache_norm_analysis/attention_energy_loss_summary.md`：查看 attention pruning 的当前证据。
4. `projects/qwen3_kcache_avg_topk/README.md`：查看可运行的 block-selection sparse decode baseline。
5. `projects/qwen3_chunk_routing/README.md`：查看 oracle/router sparse chunk training。
6. `projects/qwen3_unet_synthetic_retrieval/README.md`：查看可控 synthetic retrieval 评估和 answer-only 训练入口。
7. `projects/qwen3_interval_subseq_retrieval/README.md`：查看 arithmetic subsequence next-token 训练和 attention score dump。
8. `projects/pyramid_kv_compression/METHOD_NOTES.md`：在跑 compressed KV 继续预训练前先看风险记录。
9. `projects/qwen3_kcache_value_delta_analysis/README.md`：查看 K-cache 数值和 delta 分布 profiling。

## 实践注意事项

新增日期：2026-05-14

- 精确命令参数和输出路径以每个项目自己的 README 为准。
- 分析脚本默认把输出写到各项目的 `outputs/` 目录下。
- synthetic retrieval 项目默认直接生成 token-id 序列，避免 tokenizer segmentation 对可控任务的干扰；训练默认只对最终 answer 位置施加 loss。
- interval subsequence 项目同样直接生成 token-id 序列，但目标是标准 causal LM，所有 next-token 位置都会参与 loss；不要把它和 answer-only retrieval 的指标直接混读。
- 长上下文分析会占较多显存和内存；smoke test 时先减小 `MAX_TOKENS` 和 `CHUNK_SIZE`。
- sparse decode 和 routing 实验还需要 kernel-aware 实现，理论 KV-read reduction 才能转化为真实 serving 加速。
- 对 KV cache 压缩来说，raw cosine 是很好的诊断信号，但最终判断必须回到 attention logits、attention output、loss/PPL。
