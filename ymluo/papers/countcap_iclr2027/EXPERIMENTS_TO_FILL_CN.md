# QKSieve 投稿前实验与证据清单

本文档注册一个冻结主方法和一个质量参考：

- **Reference profile**：request-local QK-balanced + qMSE + 完整 proxy
  top-k，用于隔离表示误差；
- **QKSieve-Robust（论文主方法）**：相同 request-local QK-balanced + qMSE，
  capped sampled-quantile 单遍候选写出 + exact selected attention +
  rank-16/block-256/INT4 ValueSketch，固定 `alpha=0.5`。

正文中的同路径主结论必须来自冻结 Robust。不能把 reference LongBench 质量、
旧模型级 Key-MSE 速度、host-resident KV 或不同 Prompt 协议的数字拼接成一条
结果。Fast 只用于删除 ValueSketch 的消融。

## 0. 冻结方法合同

| 项目 | 冻结值 |
|---|---|
| QK 坐标 | 每层、每 KV head 独立的 128-D QK-balanced 双正交变换 |
| Query moment | 当前 Prompt 最后 8 个位置；request-local，prefill 后在该请求内冻结 |
| Shrinkage | 固定 $\lambda=0.75$ |
| Band | 8 个连续 16-D band |
| Bit 集合 | 0/1/2/4/8 bit |
| Allocation | Query-weighted qMSE/OAS；Key-MSE 仅作历史部署消融 |
| 索引预算 | 240 bit/token/KV-head，包含每个 active band 的 FP16 scale |
| 索引大小 | 30 B/token/KV-head，等于完整 FP16 K+V 的 5.859% |
| Token 预算 | 冻结范围 4K--128K：$\min(N,1280,\max(256,\lceil0.06N\rceil))$；不声明直接外推到 256K/512K |
| Reference 候选 | 扫描完整 packed index 后直接 FP32 proxy top-k |
| Robust 候选 | $m=\min(N,512,\max(256,\lceil16/(B/N)\rceil))$ 个规则分层样本估计 quantile；扫描时直接写候选，不物化全长 score |
| 最终 attention | 候选位置原始 FP16/BF16 K/V 的精确 attention，加未选集合的低秩 Value 分子/分母估计 |
| ValueSketch | rank 16、block 256、INT4、`alpha=0.5` |
| 禁用机制 | 无 rerank、router、任务规则、recent/sink、Full fallback、64K 方法切换 |

源码级校验器：

```bash
python src/verify_qksieve_frozen_evidence_20260728.py \
  --project_root . \
  --longbench_summary RESULTS/paired_summary.json \
  --ruler_summary RESULTS/paired_summary.json \
  --samepath_summary RESULTS/samepath_summary.json \
  --multimodel_summary RESULTS/multimodel_summary.json \
  --output RESULTS/frozen_evidence_report.json
```

只有报告中 `complete=true` 时，质量和速度才可以合并成论文主结论。校验器同时检查方法合同、源码 SHA256、样本配对、任务覆盖、长度网格和峰值显存字段。

## 1. 当前已经完成的证据

| 证据 | 当前状态 | 可以支持的结论 |
|---|---|---|
| 全层/全 KV-head bit 分配 | 完成，Qwen3-4B，32K，36×8 heads，两类文本 | 自动 allocation 确实随 layer/head/context 改变 |
| QKSieve vs FIER 检索 | 完成，18,432 个 held-out query-head 条件 | 同等索引大小下 QKSieve 的 top-k recall 和 attention mass 更高 |
| 32K 同路径 PPL | 完成，FP32 proxy 主路径 | full-topk 主路径可保持 PPL，FP16 top-k 暂不采用 |
| 单层 kernel 分解 | 完成，8K–128K | top-k、scan、exact sparse attention 是主要子系统成本 |
| GQA-4 Query 融合 | 完成，FP16/BF16 × 4K/32K，并通过真实 LongBench smoke | 目标三模型可安全融合 Query projection 与 INT8；GQA-8 不推广 |
| WMMA + sampled deployment | 完成，32K/64K/120K | whole-model steady speed 为 1.90x/3.36x/4.57x；120K 相对本地 FIER 为 1.32x |
| 256K/512K attention | 完成，24% active，含检索与 exact sparse attention | 相对预展开纯 SDPA 为 0.876x/0.932x；相对真实 HF GQA+SDPA 为 2.838x/3.038x，必须同时报告两种分母 |
| 严格原生 256K 四窗口前沿 | 完成，4×(262,080 history + 64 target)，256 个配对 token | 9.98% 为 96.26%/5.35x 速度点；23.96% 为 100.25%（CI [99.40%,101.06%]）/2.99x 高保真点 |
| 256K Exact-QK oracle 预算诊断 | 完成，同一四窗口、同一 shared prefill，Exact-QK/proxy × 1,280/2,560 | Oracle 1,280/2,560 保持 100.24%/100.15%，proxy 仅 80.77%/86.13%；质量断崖来自 selector 错排，不是 1,280 的最终 attention 预算 |
| 128K--256K 同路径交叉对照 | 完成，同一冻结模板、Key-MSE 240-bit、完整 proxy top-k、四窗口和随机种子 | 128K proxy top-1,280/2,560 为 98.99%/100.01%，256K 为 80.77%/86.13%；Exact-QK 两个长度均约 100%，严格闭合长度交叉 |
| 256K selector 单因素归因 | 完成，同一四窗口、top-1,280；Exact-QK/冻结 240-bit/冻结全 INT8/request-local 240-bit | 保持率 100.24%/80.77%/100.27%/100.43%；冻结低比特模板外推是根因。Request-local 240-bit 的整模型 Steady/Online decode 为 7.77x/4.00x（不含共同 prefill） |
| Sampled-quantile 有限样本与 straggler 诊断 | 完成理论与 256K 实测 | $c=16$ 的候选数相对标准差约 25%；提高到约 64 后 6% sampled 的 KL/PPL 追平完整 proxy top-k，且候选尾部收窄使速度提高 |
| 512K 外推系统压力 | 完成，524,256 history + 32 target | 4%/6% active 的 steady 为 10.72x/7.72x；Full PPL 已为 5510.6，只能支持扩展性而非原生 512K 质量 |
| 32K Key-only 因果消融 | 完成，六主题 × 两窗口 | QK-balanced + Key-MSE 的 Top-1/KL 显著优于 Key-PCA + Key-MSE，纯 Key-PCA 不冻结 |
| LongBench m20 allocation 因果消融 | 完成，320 个严格配对 | QK-balanced + Key-MSE / qMSE 保持率 99.923%/99.772%，直接差值 CI 跨零 |
| 32K allocation 优化路径 | 完成，同一 256-token 窗口 | Key-MSE/qMSE 为 45.577/46.696 ms/token，质量持平；部署冻结 Key-MSE |
| GQA-4 子系统长度扫描 | 完成，8K–128K，包含逐 token 索引追加 | 32K 起超过 Full，64K/128K 为 2.730x/4.162x |
| 原生 MHA Fast/Robust attention A/B | 完成，RTX 3090、32Q/32KV、8K--128K、3 seeds | Fast 为 1.27--6.37x；Robust 含 ValueSketch 后为 1.08--4.12x；候选集合严格相同 |
| 原生 MHA 真实模型稳态 decode | 完成，Yarn-Llama-2-7B-128K、32/64/128K | Robust 为 1.32/2.22/3.98x；break-even 为 69/19/10 token |
| Llama official-middle LongBench | 完成，3,750/3,750 严格配对、16/16 任务、7,500 行 | Full/QKSieve macro 为 0.459398/0.458852，保持率 99.881%，bootstrap 95% CI 为 [99.424%, 100.347%] |
| RULER 4K–32K shard 5/6 | 完成，86 个严格配对 | 结果可由正式 launcher 续跑并合并，不能单独作主结果 |
| 理论证明 | 完成并写入正文/附录 | 双正交精确性、最优 score 子空间、QK-MSE、bit 分配、排序和输出误差链 |
| 正文页预算 | 完成当前版式审计 | 正文与结论在第 9 页结束，参考文献从第 10 页开始；完整命题、证明和系统图放入附录 |

Llama LongBench 已可填写论文主表；其余证据不能替代 Qwen/Mistral
LongBench、完整 RULER、多模型 PPL 和同路径整模型测速。

## 2. P0：投稿前必须完成

### 2.1 单模型完整 LongBench（Llama 已完成）

脚本：

```bash
bash scripts/launch_qksieve_fulltopk_longbench_5gpu_20260728.sh
```

协议：

- Llama-3.1-8B-Instruct；
- 16 个英文任务、3,750 个样本；
- 每个样本在同一运行中严格配对 `full_kv` 与 QKSieve；
- 官方 middle truncation，raw prompt 最大 7,500 tokens；
- 同一 stop policy、生成长度、评分代码和 GPU；
- 总计 7,500 行，不复用旧 Full CSV。

已完成并报告：

- 16 个任务的 Full、QKSieve、相对保持率；
- macro score、macro difference 和 stratified paired-bootstrap 95% CI；
- query、decode、online 时间；
- 实际 attention token 比例；
- 3,750 个严格配对与源码 SHA256。

结果：Full/QKSieve macro 为 0.459398/0.458852，保持率 99.881%；
macro difference 95% CI 为 [-0.002647, 0.001591]，retention 95% CI 为
[99.424%, 100.347%]。16 个任务均完成，最低任务保持率为 Qasper 的
97.615%，通过预注册的 macro 至少 99% 标准。

### 2.2 三模型 LongBench 泛化

脚本：

```bash
QKSIEVE_DOWNLOAD_MISSING_MODELS=1 \
bash scripts/launch_qksieve_three_model_longbench_20260728.sh
```

模型：

- Llama-3.1-8B-Instruct；
- Qwen3-4B-Instruct；
- Mistral-7B-Instruct-v0.3。

Mistral 使用 tokenizer 自带 chat template；三者保持相同 $\lambda$、Query 数、bit 预算和 token 预算，不允许按模型搜索参数。

通过标准：三个模型都完成 3,750 个严格配对，最低 macro 保持率目标为 99%。

### 2.3 RULER

脚本：

```bash
bash scripts/launch_qksieve_fulltopk_ruler_6gpu_20260728.sh
```

协议：

- 官方完整 13 个任务；
- 4K/8K/16K/32K 各任务 10 条；
- 64K/128K 各任务 5 条；
- Full 与 QKSieve 共 1,300 行、650 个严格配对；
- Llama-3.1-Instruct 使用 `llama3` chat wrapper，问题/指令后缀 dense prefill，并只取最后 8 个 Query 位置构造 QK moment。

必须按任务和长度报告 score、保持率、实际候选数和 paired bootstrap 95% CI。质量运行的 online/decode 速度只能按实际生成 token 归一化为 TPOT；固定 horizon 的正式速度结论来自 2.4 的 same-path benchmark。通过标准不是强行要求所有单元都 100%，而是不能出现随长度增长却无法由 Full 基线或任务难度解释的系统性崩溃。

### 2.4 同一路径整模型速度、请求速度与显存

Reference profile 网格脚本：

```bash
bash scripts/launch_qksieve_frozen_samepath_length_6gpu_20260728.sh
```

最终 Deployment profile 的 LongBench m20 入口：

```bash
bash scripts/launch_qksieve_global_keymse_wmma_longbench_m20_5gpu_20260730.sh
```

其注册方法为
`qksieve_global_qkbalanced_keymse_wmma_sampled`，必须核验固定模板、
`allocation_frozen`、WMMA fused、动态 `c=64` sampled quantile、无 fallback，并与
同一运行中的 Full 严格配对。

旧的 `qksieve_global_qkbalanced_qmse_wmma_sampled` 只作为 allocation
对照和已有 32K/64K/120K 系统诊断保留，不能把其速度静默改写为 Key-MSE。

网格：

- 历史长度：16K/32K/64K/128K；
- decode steps：64/256/1024；
- 每格同时运行 Full 和冻结 QKSieve；
- 64K/128K 使用两卡只是为了容纳完整 GPU-resident KV，方法本身不做 KV offload。

必须报告：

- 稳态 whole-model decode speed；
- 包含 prefill 和 index build 的请求级 speed；
- fixed overhead、break-even token 数；
- Full/QKSieve PPL；
- 每卡和总 peak allocated/reserved memory；
- 索引比例和实际 active KV。
- 单层 Query projection/INT8、packed scan、top-k、生产 fused gather+exact attention、显式 gather 诊断、历史索引构建和逐 token 索引追加。

当前单层 synthetic kernel 数字只能放在子系统表，不能替代这张同路径整模型表。

### 2.5 持久化 KV 与 Agent 复用场景

**状态：TODO，当前 request-local reference 结果不能直接填写本节。**

论文的系统主张限定为“对已经构建且可以复用的 KV cache 加速 decode”，不把
sparse prefill 作为当前贡献。但必须通过真实 cache 生命周期实验，而不能只用
steady decode 推断 Agent 场景。需要分别运行并报告：

| 协议 | 计时范围 | 主要回答的问题 |
|---|---|---|
| Cold single-use | dense prefill + 一次建索引 + decode | 普通单次请求是否加速 |
| Warm persistent | KV/index 已常驻；包含 Query setup 和完整 decode | 已缓存长前缀下的真实收益 |
| Shared-prefix reuse | 一次 prefill/index，随后 $R$ 个查询或分支 | Agent rollout 的摊销收益和 break-even |
| Append-only turn | 新 token prefill、增量 index append、随后 decode | 多轮增长时能否维持收益 |

注册网格：

- 历史长度 `16K/32K/64K/128K`；
- 每次生成 `32/64/256/1024` token；
- 复用或分支数 `R=1/2/4/8/16/32`；
- workload 至少包含 repeated QA、append-only 多轮对话和 shared-prefix branching rollout；
- 同一格同时运行优化后的 dense baseline 与 QKSieve，并核验输出质量。

必须保存：`prefill_ms`、`index_build_ms`、`cache_load_ms`、
`request_setup_ms`、`index_append_ms`、`first_token_ms`、`decode_ms`、TPOT、
throughput、峰值显存、index bytes、active KV、质量保持率，以及实测
break-even 的 $(R,G)$ 边界。Cold、warm 和摊销数字不得合并成一个 speedup。

生命周期 gate：

- 辅助索引必须与精确 KV 一起持久化，只构建一次；
- 新请求不得重新投影或量化未变化的历史 Key；
- 新 token 只能增量追加 exact KV 和对应索引；
- 每次 Query projection、风险/预算计算和候选扫描必须进入 warm-path 计时；
- 如果 cache eviction/reload，两种方法都要公平计入，或单独报告；
- 这里复用的是不可变索引，不允许复用上一步候选，不启用 learned router 或 Full fallback。

当前 QK-balanced request-local 路径如果会在新请求上重建历史索引，只能进入
cold 行。完成 persistent index 或 immutable segment 版本后，才能填写 warm、
shared-prefix 和 append-only 三行。这项工作优先级高于 sparse-prefill 优化。

Query projection + bandwise INT8 的独立融合候选已经以
`qksieve_fullprompt_auto_plain_qfused_fulltopk` 接入。GPU 验证已经完成：

- GQA-4 的 FP16/BF16、4K/32K 数值结果全部通过，Query prepare 中位加速为
  1.547x/1.755x，完整 selection 中位加速为 1.204x/1.275x；
- 真实 LongBench 8 样本 smoke 的 qfused requested/executed/frozen 标志均为
  1，和 reference 路径预测 exact-match 为 87.5%，平均分差为 -0.00213；
- GQA-8 在 32K 的完整 selection 只有约 0.883x/0.806x，因此不能把融合实现
  宣称为任意 GQA 配置上的通用优化；
- 目标 Llama/Qwen/Mistral 均为 GQA-4，可以把融合结果放入子系统表。
  Reference profile 仍是
  `qksieve_fullprompt_auto_plain_fulltopk`；Deployment profile 是上述
  global-template + WMMA + sampled 方法。在 deployment 的完整同路径任务
  实验完成前，不把两者拼成同一主结果。

正式验收产物：

```text
results/20260728_qksieve_qfused_correctness_native_g4/validation_matrix.json
results/20260728_qksieve_qfused_longbench_smoke_native_g4_retry2/
results/20260728_qksieve_qfused_breakdown_gpu5/breakdown.json
```

## 3. P0：公平基线

最直接基线是 FIER。必须在同一模型、GPU、active token 数和最终 sparse-attention kernel 下比较：

| 方法 | 必要 operating point |
|---|---|
| FIER 1-bit RTN g32 | 等索引字节、等 active KV、等质量 |
| Key-PCA uniform INT4 | 等索引字节 |
| QKSieve uniform bit | 等索引字节 |
| QKSieve automatic mixed-bit | 主方法 |

当前 FIER 检索结果是经过审计的论文规格复现，只能支持检索质量结论。真正的 packed FIER-g32 CUDA 编码与扫描实现已经完成并通过 CPU 测试、远端编译；它使用 32 B/token/KV-head 的 bit-plane 索引，并与 QKSieve 共用 top-k 和 exact sparse-attention kernel。GPU 数值验证、同路径 LongBench/RULER 和正式速度仍未运行，不能使用旧的 unpacked FP16 参考路径计时或提前填写对比表。

Quest 与两种 SparQ reference 的同 harness 质量控制已经接入：

```bash
bash scripts/launch_qksieve_public_selectors_longbench_5gpu_20260728.sh
```

该脚本在相同样本、相同长度预算和相同 exact-KV attention consumer 下比较
QKSieve、Quest P16、SparQ R32 selector-only control 和 SparQ R32
formula reference。Quest 使用 16-token page min/max bound；selector-only
路径不包含 mean-Value correction；formula reference 额外补齐论文温度、
局部窗口、selected-mass 和增量 mean-Value correction。后者仍使用 QKSieve
冻结动态预算，且没有官方优化 Key layout，因此只能标为公式完整参考，不能
标成官方 SparQ 系统。这三条公开参考路径均明确禁止用其 PyTorch latency
填写论文速度表。

Quest 优化 kernel、SparQ 官方优化系统和 RetroInfer 官方 CPU-GPU/wave-index
系统仍需分别复现。源码审计确认当前 `microsoft/RetrievalAttention` 仓库
实际提供的是 RetroInfer，而非原始 RetrievalAttention 可运行实现。因此
RetrievalAttention 只能保留 paper-reported 行，RetroInfer 单独做官方系统
复现。它们的 KV placement 和索引设定与 GPU-resident QKSieve 不同，应做
同样本质量与完整系统对比，不能虚构成 index-byte matched。详细边界见
`docs/20260728_qksieve_public_baseline_protocol_zh.md` 和
`docs/20260728_qksieve_retroinfer_official_protocol_zh.md`。

RetroInfer 固定源码审计和 LongBench 对齐外壳已完成：

```bash
bash scripts/prepare_retroinfer_official_20260728.sh
bash scripts/launch_retroinfer_aligned_longbench_5gpu_20260728.sh
```

第二个脚本严格配对官方栈内的 `Full_Flash_Attn` 与 `RetroInfer`，并分栏
保存 cache init、prefill、cache prepare、graph capture、decode、总延迟、
GPU peak 和 CPU peak RSS。当前尚未安装独立 CUDA 12.4 环境或运行 GPU，
不能提前填写结果。

TurboQuant、Q-Filters、RaBitQCache 至少完成索引质量或方法级比较，并清楚
标记是否使用官方代码。

## 4. P1：必要消融

| 消融 | 匹配条件 | 必须报告 |
|---|---|---|
| Key-PCA vs QK-balanced | 相同 240 bit、相同 top-k | score RMSE、recall、mass、PPL |
| Random rotation vs QK-balanced | 统一 1-bit、相同 256 bit | 同上 |
| Uniform vs fixed 4-4-2-1 vs auto | 相同 240 bit | layer/head 分配与下游质量 |
| Global vs per-layer/per-head | 相同总索引字节 | 质量和分配分布 |
| Without Query covariance | 相同 quantizer | 检索和任务质量 |
| $\lambda=0/.25/.5/.75/.9$ | held-out Query | drift、QK-MSE、mass |
| Active KV 0.5/1/2/4% | 同一索引 | 质量-延迟 Pareto |
| Query 样本 1/4/8/16/32 | 参数冻结 | calibration 与 held-out regret |

为拆分坐标系和 bit 分配收益，已加入三个真实 256-bit 路径：
`qksieve_fullprompt_keypca_uniform1_fulltopk` 与
`qksieve_fullprompt_qkbalanced_uniform1_fulltopk`、`qksieve_fullprompt_random_uniform1_fulltopk`。
另加入两个 240-bit Key-MSE 路径：
`qksieve_fullprompt_keypca_autokey_fulltopk` 完全不使用 Query，
`qksieve_fullprompt_qkbalanced_autokey_fulltopk` 只在 QK-balanced 坐标构造中使用 Query、但在 bit allocation 中不使用 Query。
先运行
`scripts/launch_qksieve_uniform1_ablation_longbench_m20_5gpu_20260728.sh`
做八方法受控筛选，再决定哪些方法扩展到完整 3,750 样本。

理论对应关系已经固定：

- random rotation 检验“任意正交旋转”是否足够；Haar 旋转在期望上把任意二阶矩能量按 band 维数均匀分摊；
- QK-balanced + Key-MSE 对比 QK-balanced + qMSE，只隔离 Query-weighted allocation；
- Key-PCA + Key-MSE 是完全不依赖 Query 的端点；
- Key-only allocation 的 QK-MSE 最坏近似因子由 Query moment 条件数上界；Query 越各向异性，该保证越弱。

理论对应实验必须增加：

- $\|AD^\top-I\|_{\max}$；
- QK 奇异值累计 score energy；
- 冻结候选映射上的 paired-vs-Cartesian score loss，量化 Query--Key 依赖四阶残差；
- Query anisotropy 与 QK-balanced 相对 Key-PCA 收益的关系；
- 不同 Query 样本数下的 moment operator-norm error、16-D band 边界 singular gap 和左右 subspace angle；
- cross-band 项占完整 QK-MSE 的比例；
- band 敏感度 AM/GM 比值与 uniform-to-auto QK-MSE 收益的相关性；
- held-out regret 上界中的 $\Omega$、$\Gamma$ 与实际 regret；
- 冻结坐标后，用独立 Query/Key split 枚举全部 13,817 个可行 allocation，
  报告 uniform calibration-heldout gap 和最终 selected-allocation regret；
- Key stride 8/16/32/64 消融，分别拆出 Query 样本误差、Key 样本误差，
  不能把 Cartesian 的 $m_qm_k$ 配对数当成独立样本数；
- centered score RMSE、margin certificate coverage、oracle/proxy omitted-mass ratio、attention mass；
- layer output error、最终 logit KL、top-1 margin。

## 5. P1：Query 分布漂移

必须补齐：

- 不同 Prompt template；
- 1K/2K/4K 长输出；
- 多轮对话；
- 按生成位置统计 Query covariance drift；
- 冻结 transform/allocation 后，score MSE、top-k recall、attention mass 是否退化。

增加 Query 样本只能降低有限样本误差，不能自动解决真实生成分布漂移，因此不能只做 1/4/8/16/32 的样本数消融。

当前已经完成代码层面的严格协议：

- 生产校准窗口固定为 8，额外 trace 窗口可记录 32 个 Query，两者已经解耦；
- `scripts/launch_qksieve_free_generation_drift_6gpu_20260728.sh`
  记录自然 EOS 的真实 QKSieve 生成 Query，只按实际观察到的位置报告 coverage；
- `scripts/launch_qksieve_teacher_forced_drift_6gpu_20260728.sh`
  在六个主题上保证覆盖 32K 历史后的 1K/2K/4K continuation；
- `src/analyze_qksieve_query_drift_20260728.py`
  输出 covariance drift、band-boundary singular gap、subspace angle、
  allocation regret、score RMSE、top-k recall 和 oracle/proxy omitted mass。

teacher-forced continuation 只能作为长位置机制证据，不能冒充自然生成质量。
当前这些脚本只完成 CPU 合成 trace 验证，正式 GPU 数字仍待补。

## 6. P2：系统与投稿完整性

- 在 H100 或 A100 单卡重新测 batch=1 TPOT；
- A100/H100 运行前分别设置 `TORCH_CUDA_ARCH_LIST=8.0/9.0`；所有
  20260728 launcher 只提供 3090 的 8.6 默认值，不再强制覆盖外部架构；
- 报告 Query projection/quantization、packed scan、top-k、gather、exact sparse attention、cache append、其他模型计算；
- 报告 median、p5、p95、warm-up、重复次数、CUDA/PyTorch/Transformers 版本和 GPU 时钟；
- 冻结模型、数据集 revision、随机种子、编译 flags 和匿名 artifact commit；
- 论文中统一写成“完整 FP16 K/V + 5.859% 检索索引”，不能写成只保留 5.859% KV memory；
- ICLR 2027 官方模板发布后替换临时模板；
- 删除或替换所有 `TBD`，并由证据校验器重新验证。

## 7. 当前最短执行顺序

1. 受控归因已经完成：request-local 240-bit 和冻结全 INT8 均恢复约 100%，
   根因锁定为冻结低比特模板外推。继续补 exact-set recall、attention-mass
   recall、score RMSE 和 crossing margin，并按 layer/head 定位最小刷新量。
2. 分别测试只刷新 scale、Key-MSE allocation、QK-balanced transform；随后把
   最小充分统计量在 prefill 中增量或异步构建，压缩 request-local 固定成本。
3. 在独立于 20-Newsgroups 的长文本和至少一个新模型上复核 256K Exact-QK
   top-1,280 的预算充分性，以及改进 proxy 的高保真点。
4. 完成 request-local deployment profile LongBench m20；通过后再扩到
   3,750 样本。
5. 顺序完成 Qwen3/Mistral LongBench 和 RULER 4K--128K。
6. 完成同文本 8K--128K whole-model/request/memory 网格和 packed FIER 公平
   对比。
7. 做 Query drift、跨模型 PPL 与 H100/A100 最终系统账单。
8. 运行冻结证据校验器，只有 `complete=true` 后才填写剩余主表结果。
