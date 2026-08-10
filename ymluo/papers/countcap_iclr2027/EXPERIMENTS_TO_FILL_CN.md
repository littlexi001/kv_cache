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
python src/verify_qksieve_robust_paper_evidence_20260810.py \
  --project_root . \
  --persistent_summary RESULTS/persistent/independent_summary.json \
  --longbench_summary RESULTS/longbench/paired_summary.json \
  --ruler_summary RESULTS/ruler/paired_summary.json \
  --multimodel_summary RESULTS/multimodel/multimodel_summary.json \
  --shrinkage_summary RESULTS/shrinkage/summary.json \
  --h100_summary RESULTS/h100/summary.json \
  --output RESULTS/frozen_evidence_report.json
```

只有报告中 `complete=true` 时，质量与系统结果才可以组成完整论文证据链。
校验器强制检查冻结方法合同、3,750 个同路径 LongBench 配对、650 个正式
RULER 配对、Llama/Qwen/Mistral 覆盖、persistent 生命周期和 H100 的
64K/128K 三类系统测速，以及固定 shrinkage 的严格配对敏感性；任一证据缺失
都会显式失败。

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
| 32K allocation 优化路径 | 完成，同一 256-token 窗口 | Key-MSE/qMSE 为 45.577/46.696 ms/token，质量持平；论文冻结 qMSE/OAS，Key-MSE 仅作速度消融 |
| GQA-4 子系统长度扫描 | 完成，8K–128K，包含逐 token 索引追加 | 32K 起超过 Full，64K/128K 为 2.730x/4.162x |
| 原生 MHA Fast/Robust attention A/B | 完成，RTX 3090、32Q/32KV、8K--128K、3 seeds | Fast 为 1.27--6.37x；Robust 含 ValueSketch 后为 1.08--4.12x；候选集合严格相同 |
| 原生 MHA 真实模型稳态 decode | 完成，Yarn-Llama-2-7B-128K、32/64/128K | Robust 为 1.32/2.22/3.98x；break-even 为 69/19/10 token |
| Persistent KV 生命周期 | 完成，32K/64K，cold/warm/四分支均摊/append-only | Warm 为 1.322x/2.221x；四分支均摊为 1.082x/1.785x；独立审计确认 32 层索引未重建且 replay 一致 |
| FIER 同 consumer 速度 | 完成，原生 MHA、相同 active-token schedule 和精确 sparse-attention consumer | Fast 相对 FIER 为 1.37--3.58x，Robust 在额外支付 Value-tail 后仍为 1.16--2.31x（8K--128K） |
| Llama official-middle LongBench | 完成，3,750/3,750 严格配对、16/16 任务、7,500 行 | Full/QKSieve macro 为 0.459398/0.458852，保持率 99.881%，bootstrap 95% CI 为 [99.424%, 100.347%] |
| 冻结 Robust 完整 LongBench | 完成，3,750/3,750 严格配对、16/16 任务、7,500 行、零 fallback | Full/Robust macro 为 0.459011/0.458692，保持率 99.930%，task-bootstrap 95% CI 为 [99.538%, 100.213%] |
| 冻结 Robust 正式 RULER | 修正后的审计运行进行中，13 任务、4K--128K、计划 650 个严格配对 | 缺少逐行 attention diagnostics 的旧运行已排除；完成前不写正式 RULER 主结果 |
| Llama/Qwen/Mistral 同协议 LongBench screen | 完成，16 任务、每模型 160 个独立 offset 样本、共 480 个严格配对 | 保持率分别为 98.681%/100.211%/98.487%，三个 task-bootstrap 区间均跨 100%，fallback 为 0 |
| 理论证明 | 完成并写入正文/附录 | 双正交精确性、最优 score 子空间、QK-MSE、bit 分配、排序和输出误差链 |
| 正文页预算 | 完成当前版式审计 | 正文与结论在第 9 页结束，参考文献从第 10 页开始；完整命题、证明和系统图放入附录 |

Llama reference、冻结 Robust 完整 LongBench 与三模型独立 screen 均已完成。
完整 RULER 和 H100 结果尚未完成，不能用旧实验或不同执行路径替代。

## 2. P0：投稿前必须完成

### 2.1 单模型完整 LongBench（reference 与冻结 Robust 均已完成）

脚本：

```bash
bash scripts/launch_qksieve_robust_llama_full_longbench_20260810.sh
```

协议：

- Llama-3.1-8B-Instruct；
- 16 个英文任务、3,750 个样本；
- 每个样本在同一运行中严格配对 `full_kv` 与 QKSieve；
- 官方 middle truncation，raw prompt 最大 7,500 tokens；
- 同一 stop policy、生成长度、评分代码和 GPU；
- 总计 7,500 行，不复用旧 Full CSV。

旧 reference profile 已完成并报告：

- 16 个任务的 Full、QKSieve、相对保持率；
- macro score、macro difference 和 stratified paired-bootstrap 95% CI；
- query、decode、online 时间；
- 实际 attention token 比例；
- 3,750 个严格配对与源码 SHA256。

结果：Full/QKSieve macro 为 0.459398/0.458852，保持率 99.881%；
macro difference 95% CI 为 [-0.002647, 0.001591]，retention 95% CI 为
[99.424%, 100.347%]。16 个任务均完成，最低任务保持率为 Qasper 的
97.615%，通过预注册的 macro 至少 99% 标准。

上述旧结果只证明 QK-balanced 表征和完整 proxy top-$k$ 的质量。冻结 Robust
已经在独立的相同规模协议上完成：Full/Robust macro 为
0.459011/0.458692，保持率为 99.930%，task-bootstrap 95% CI 为
[99.538%, 100.213%]。后处理器验证了 7,500 行、3,750 个严格配对、16 个
任务和零 fallback；对于两个极短 MultiNews 样本，还按每个 decode 步骤的
实际历史长度重建 sampled-quantile sample count，最大审计误差为 0。

### 2.2 三模型 LongBench 泛化

脚本：

```bash
bash scripts/launch_qksieve_robust_multimodel_longbench_20260810.sh
```

模型：

- Llama-3.1-8B-Instruct；
- Qwen3-4B-Instruct；
- Mistral-7B-Instruct-v0.3。

Mistral 使用 tokenizer 自带 chat template；三者保持相同 $\lambda$、Query 数、
bit 预算、ValueSketch 和 token 预算，不允许按模型搜索参数。当前注册的是每模型
16 任务、每任务 10 条、offset 40 的 160-pair 独立 screen，用于检验方向一致性
与最差任务；Llama 的 3,750-pair 主表承担总体质量主结论。

结果已经完成并通过冻结合同校验：

| 模型 | Full macro | Robust macro | 相对 Full | task-bootstrap 95\% CI | 平均 active attention |
|---|---:|---:|---:|---:|---:|
| Llama-3.1-8B-Instruct | 0.426435 | 0.420810 | 98.681\% | [96.393\%, 100.504\%] | 7.215\% |
| Qwen3-4B-Instruct | 0.397178 | 0.398015 | 100.211\% | [98.907\%, 101.720\%] | 7.178\% |
| Mistral-7B-Instruct-v0.3 | 0.421645 | 0.415267 | 98.487\% | [95.084\%, 100.893\%] | 6.835\% |

三个模型均包含 160 个严格配对与 16 个任务，Full fallback 为 0；有效 quantile
样本均值为 508.8/508.8/510.4。绝对下降最大的任务分别为 Llama LCC
（-0.064286）、Qwen GovReport（-0.014001）和 Mistral HotpotQA（-0.100000）。
每任务只有 10 条且三个区间均跨 100\%，因此这里只支持“没有观察到跨模型系统性
崩溃”，不宣称逐模型与 Full 等价。原始汇总 SHA256 为
`9ceadaace51808989a222df6669ca5261e233c8ebd21f67b3f2bcd9851b8bf30`。

### 2.3 RULER

脚本：

```bash
bash scripts/launch_qksieve_robust_ruler_20260810.sh
```

协议：

- 官方完整 13 个任务；
- 4K/8K/16K/32K 各任务 10 条；
- 64K/128K 各任务 5 条；
- Full 与 QKSieve 共 1,300 行、650 个严格配对；
- Llama-3.1-Instruct 使用 `llama3` chat wrapper，问题/指令后缀 dense prefill，并只取最后 8 个 Query 位置构造 QK moment。

必须按任务和长度报告 score、保持率、实际候选数和 paired bootstrap 95% CI。质量运行的 online/decode 速度只能按实际生成 token 归一化为 TPOT；固定 horizon 的正式速度结论来自 2.4 的 same-path benchmark。通过标准不是强行要求所有单元都 100%，而是不能出现随长度增长却无法由 Full 基线或任务难度解释的系统性崩溃。

审计说明：第一轮正式运行没有打开 `--collect_attention_stats`，所以虽然预测可
评分，却不能逐条证明 qMSE packed scan、有效 quantile 样本和 ValueSketch 实际
执行。该轮结果已保留但从论文证据中排除。修正后的 v2 要求每条 Robust 行同时
满足正确 `executed_path`、正的 `packed_qmse_sample_count`、
`packed_qmse_value_sketch_executed=1`、`sampled_quantile_fallback=0`，再进入
最终汇总。

### 2.4 同一路径整模型速度、请求速度与显存

RTX 3090 原生 MHA 的 attention、真实模型 decode 和 persistent-cache 结果已经
完成。投稿硬件复测入口为：

RTX 3090 表不再手工维护。`scripts/make_qksieve_rtx3090_system_rows.py` 会读取
三 seed Attention、匹配 FIER、真实 MHA Decode 和独立 persistent 生命周期
artifact，逐项检查候选一致性、MHA 形状、GPU、warmup/iteration、ValueSketch
开关和复用合同，然后生成
`data/generated/qksieve_rtx3090_system_rows.tex`。输入文件 SHA、聚合 SHA、所选
无干扰 Decode artifact 和未舍入数值记录在
`data/generated/qksieve_rtx3090_system_manifest.json`；当前聚合 SHA 为
`e30a75542a7f25f48d3218c8519fab993903d4f11d102973146a4fa41e8e9274`。
英文与中文构建脚本都会先重新生成该文件，论文表格不再复制数字。

需要区分两种构建口径：独立 Decode harness 的 Fast/Robust 构建为
`.768/1.375`、`.774/1.488`、`.743/1.839` 秒，对应 32/64/128K break-even
`24/69`、`9/19`、`4/10` token；完整 persistent 生命周期还安装并审计可复用
状态，因此 32/64K Robust 构建为 `1.759/1.998` 秒，break-even 为 `87/26`
token。两者不能混写。完整表放在系统附录，正文保留关键数值；匿名稿结论完整
结束在第 9 页，AI 声明与参考文献从第 10 页开始。

```bash
bash scripts/launch_qksieve_h100_matched_20260810.sh
```

独立的环境、命令、输出和验收说明见
`ymluo/projects/qwen3_top2_head_limit3_ppl/docs/qksieve_h100_reproduction_20260810.md`。

该脚本只运行冻结 qMSE/OAS Robust 与同张量 Full，不能替换成 Key-MSE、Fast、
旧 global template 或长度门控。每个 artifact 必须记录 GPU 型号、CUDA、PyTorch、
源码 SHA、候选数、索引字节、peak memory 和直接计时区间。
H100 汇总和总 evidence verifier 会再次检查设备名必须包含 `H100`、冻结合同与
主方法 ID 必须逐字段一致、三类结果均含 64K/128K、至少三个 seed，并拒绝缺失、
非有限或非正的延迟、加速、索引字节和显存字段；RTX 3090 结果不能误填进 H100
表。整模型进程在模型与 CUDA 扩展加载完成后、dense prefill 开始前重置 CUDA
peak statistics；原始 artifact 保存逐卡 allocated/reserved peak，汇总表报告逐卡
峰值之和。Attention 子系统在同一进程内交替测多个方法，因此不使用不公平的
进程峰值，而是报告 Full K/V、QKSieve Key index、ValueSketch 和 FIER index 的
精确 resident tensor bytes。

网格：

- H100 历史长度：64K/128K；16K/32K 保留为 RTX 3090 交叉点诊断；
- steady decode：固定生成 256 token，前 32 token 不进入稳态均值；
- persistent request：4 个 64-token 分支、一次确定性 replay 和一次 128-token
  append-only 分支；
- 每格同时运行 Full 和冻结 QKSieve；
- 64K 使用一张 80GB H100，128K 使用两张 80GB H100 只是为了容纳完整
  GPU-resident KV；方法本身不做 KV offload。

必须报告：

- 稳态 whole-model decode speed；
- 不含 dense prefill、但包含 index build 的 cold persistent speed；
- 从 dense prefill 前开始直接 wall-clock 计时、同时包含 index build 和首个完整
  decode branch 的 cold end-to-end request speed；两者不得混写；
- fixed overhead、break-even token 数；
- Full/QKSieve PPL；
- 每卡和总 peak allocated/reserved memory；
- 索引比例和实际 active KV。
- 单层 Query projection/INT8、packed scan、top-k、生产 fused gather+exact attention、显式 gather 诊断、历史索引构建和逐 token 索引追加。

当前单层 synthetic kernel 数字只能放在子系统表，不能替代这张同路径整模型表。

### 2.5 持久化 KV 与 Agent 复用场景

**状态：RTX 3090 的冻结 Robust 32K/64K 证据已完成；H100 64K/128K 复测待完成。**

论文的系统主张限定为“对已经构建且可以复用的 KV cache 加速 decode”，不把
sparse prefill 作为当前贡献。但必须通过真实 cache 生命周期实验，而不能只用
steady decode 推断 Agent 场景。需要分别运行并报告：

| 协议 | 计时范围 | 主要回答的问题 |
|---|---|---|
| Cold single-use | dense prefill + 一次建索引 + decode | 普通单次请求是否加速 |
| Warm persistent | KV/index 已常驻；包含 Query setup 和完整 decode | 已缓存长前缀下的真实收益 |
| Shared-prefix reuse | 一次 prefill/index，随后 $R$ 个查询或分支 | Agent rollout 的摊销收益和 break-even |
| Append-only turn | 新 token prefill、增量 index append、随后 decode | 多轮增长时能否维持收益 |

完整目标网格：

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

Persistent v2 已完成 32K/64K 的 cold、warm、shared-prefix 四分支均摊和
append-only 直接
计时。独立生命周期审计覆盖 32 层、6 个 snapshot 和 5 次 rewind，确认历史
Key/Value index buffer 未重建、replay token/hash 一致、生成后索引只落后精确
cache 一个注册 token。该证据允许在同一 request-local basis 下复用不变前缀、
分支和连续 append；如果新 Prompt 改变 Query 统计量，仍必须重建 basis 与索引，
不能把它宣称为任意语义新请求的零成本复用。

GQA-4 fused Query preparation 的历史数值验证保留为 kernel correctness 证据；
GQA-8 的完整 selection 在 32K 可能更慢，因此不作架构无关加速声明。冻结主
路径始终是 request-local qMSE/OAS + sampled quantile + Robust ValueSketch，
不再把旧 global-template、Key-MSE 或 full-topk 路径称为 deployment 主方法。

正式 persistent 产物：

```text
docs/qksieve_persistent_kv_20260810/raw_results/
  20260810_qksieve_persistent_kv_v2/summary.json
docs/qksieve_persistent_kv_20260810/raw_results/
  20260810_qksieve_persistent_kv_v2/independent_summary.json
```

## 3. P0：公平基线

最直接基线是 FIER。必须在同一模型、GPU、active token 数和最终 sparse-attention kernel 下比较：

| 方法 | 必要 operating point |
|---|---|
| FIER 1-bit RTN g32 | 等索引字节、等 active KV、等质量 |
| Key-PCA uniform INT4 | 等索引字节 |
| QKSieve uniform bit | 等索引字节 |
| QKSieve automatic mixed-bit | 主方法 |

FIER RTN-1 g32 的 packed CUDA 编码与扫描已经完成 GPU 数值验证和同路径速度
比较。它使用 32 B/token/KV-head 的 bit-plane 索引，并与 QKSieve 共用相同
active-token 日程、top-k 和 exact sparse-attention consumer。在 RTX 3090 原生
MHA 8K--128K 上，Fast 相对 FIER 为 1.37--3.58x，完整 Robust 在额外支付
ValueSketch 后仍为 1.16--2.31x。该结论只对应本文审计的 packed 实现，不能写成
对任意厂商优化 FIER 实现的速度结论；H100 上仍需随主系统表重新测量。

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

固定 shrinkage 的正式敏感性协议已经实现于
`scripts/launch_qksieve_shrinkage_sensitivity_20260810.sh`。该实验严格使用 prompt
最后 8 个 Query 校准，并对五个系数做跨 Llama/Qwen、体育/医学的逐条件配对；
`lambda=0.75` 仍保持冻结，实验结果只决定论文能否声明超参数稳定，不能用于重新
选择系数。最终证据门会拒绝缺轨迹、缺系数、缺稀疏率、非 prompt 校准或少于
10,000 次 bootstrap 的结果；预注册阈值是否通过则原样进入审计报告，不能隐藏
失败。当前状态为代码与协议完成、GPU 结果待补。

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
- A100/H100 运行前分别设置 `TORCH_CUDA_ARCH_LIST=8.0/9.0`；
  `launch_qksieve_h100_matched_20260810.sh` 必须记录实际架构和编译配置；
- 报告 Query projection/quantization、packed scan、top-k、gather、exact sparse attention、cache append、其他模型计算；
- 报告 median、p5、p95、warm-up、重复次数、CUDA/PyTorch/Transformers 版本和 GPU 时钟；
- 冻结模型、数据集 revision、随机种子、编译 flags 和匿名 artifact commit；
- 论文中统一写成“完整 FP16 K/V + 5.859% 检索索引”，不能写成只保留 5.859% KV memory；
- ICLR 2027 官方模板已于 2026-08-10 完成逐字节核验；最终提交前仍需重新核对
  官方规则、9 页正文、匿名性、AI 使用声明与可复现性声明；
- 删除或替换所有 `TBD`，并由证据校验器重新验证。

## 7. 当前最短执行顺序

1. 保持数值方法完全冻结，不再做参数或长度规则搜索。
2. 完成 13 任务、4K--128K、650-pair 正式 RULER，并独立核验 prompt 长度、
   零 fallback、分任务/分长度结果和 paired bootstrap 区间。
3. 已完成 Llama/Qwen/Mistral 各 160-pair 独立 LongBench screen；保留原始
   artifact、合同校验和 task-bootstrap 区间，不在 screen 上调参。
4. 完成正在运行的 Llama 冻结 Robust 3,750-pair 同路径 LongBench，将旧 reference
   结果保留为表征上界，不再作为部署主结果。
5. 在 H100 完成 64K/128K attention、steady decode、cold/warm/persistent
   request 与 peak-memory 网格，并保留 RTX 3090 结果作为可复现实验。
6. 运行总 evidence verifier；只有 `complete=true` 后才更新摘要、主表、图和
   结论，再重新编译并逐页审计英文/中文 PDF。

## 8. 最终证据收口

全部正式实验完成并同步到 `data/` 后，只运行下面这一条命令生成投稿候选版本：

```powershell
powershell -ExecutionPolicy Bypass -File .\finalize_evidence.ps1
```

脚本会强制核验冻结 SHA、persistent KV、3,750 对 LongBench、650 对 RULER、
Llama/Qwen/Mistral 三模型和 H100 64K/128K 证据；随后重新生成表格与图片，编译
匿名版、署名版和中文阅读版，并检查参考文献必须从英文 PDF 第 10 页开始、匿名版
不得泄露作者信息、三份 PDF 不得残留 `TBD`、`TODO` 或 `PLACEHOLDER`。任一条件
不满足时脚本直接失败，不能把该版本标记为最终稿。
