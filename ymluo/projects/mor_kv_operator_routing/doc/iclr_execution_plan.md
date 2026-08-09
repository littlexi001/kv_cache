# MoR-KV ICLR 2027 Evidence Plan

这份清单定义“足够投稿”的最低证据。完成一个 Qwen3-0.6B synthetic result 不等于完成论文。

## 1. 论文主张

必须同时成立：

1. attention heads 有可迁移但 query-dependent 的 retrieval operator preference；
2. routing operator identity 比只 routing budget 更有效；
3. specialist-preserving selection 修复多数共识删除少数证据的问题；
4. GQA 下能转化成真实物理 KV/cache 和 kernel 收益；
5. 风险 gate 能控制错误路由的 tail loss。

## 2. 模型矩阵

最低：

- Qwen3-8B（GQA）；
- Llama-3.1-8B-Instruct（GQA）；
- Mistral-7B-Instruct（GQA/SWA 结构差异）；
- Qwen3-0.6B 只用于机制消融和快速开发。

加分项：Qwen2.5-14B 或 Llama-3.1-70B 的少量关键点。

## 3. 数据与任务

- LongBench 全任务或官方 representative suite；
- RULER 16K/32K/64K/128K；
- InfiniteBench / LooGLE；
- NIAH、variable tracking、multi-hop tracing；
- RepoBench/long code；
- long summarization；
- reasoning decode KV：AIME/U-Math 或可公开复现替代；
- multi-query reuse scenario，回应 KVzip。

所有 router/head policy 必须有 corpus-disjoint train/dev/test；至少一个实验做 cross-dataset zero-shot transfer。

## 4. 强基线

- FullKV；
- StreamingLLM；
- H2O；
- SnapKV / PyramidKV / Ada-KV；
- Quest / RetrievalAttention 类 query-aware retrieval；
- RazorAttention；
- DuoAttention；
- HeadKV / HeadKV-R2；
- Task-KV；
- CompilerKV；
- PolyKV；
- HARD-KV；
- strongest local RiskKV / deep question-likelihood hybrid。

必须在相同有效 KV bytes、相同 context 和相同 kernel条件下比较。

## 5. 指标

质量：

- task score；
- answer NLL / PPL；
- exact attention-mass recall；
- attention output cosine/L2；
- hard-negative exposure；
- p95/p99 sample loss 和 fallback rate。

效率：

- physical KV bytes；
- metadata/index bytes；
- TTFT；
- decode tok/s；
- batch 1/4/8/16 throughput；
- router/scoring/gather/kernel 分项时间；
- GPU/CPU/PCIe traffic；
- CUDA Graph compatibility。

## 6. 核心消融

1. one operator for all heads；
2. different budget but same operator；
3. static head roles vs query-conditioned route；
4. oracle route vs learned route vs wrong route；
5. majority/RRF vs minority-max vs submodular saturation；
6. per-query-head vs GQA-group physical union；
7. no risk gate / margin gate / conformal gate；
8. no function prior / controlled-function prior / learned utility；
9. lexical, structure, QK, streaming, dense operator leave-one-out；
10. calibration sizes 16/64/256/1024；
11. rank 16/32/64；
12. block sizes 16/32/64/128/256；
13. utility lambda sweep and constraint form。

## 7. 必须关闭的工程 Gate

- 不再通过重构文本模拟 sparse KV；直接在 attention kernel/gather 上运行；
- GQA 不复制共享 K/V；
- 动态 operator 编译成少量固定 runtime templates；
- block IDs 在 gather 后恢复 causal position/RoPE 语义；
- 输出实际显存而不是理论 keep ratio；
- latency 至少 30 次重复，含 warmup 和置信区间；
- 开源可从 checkpoint + dataset 一键复现主表。

## 8. Go / No-Go

### Go

- 至少两个 8B 模型上，在同 KV bytes 下显著超过 strongest single-policy 和 PolyKV/CompilerKV；
- 至少三个自然任务族显示 operator routing 增益；
- 实际 decode speedup 不被 router/index overhead 抵消；
- wrong-route tail 由 risk gate 控制；
- submodular/minority mechanism 有独立消融贡献。

### No-Go / 改题

- 增益只存在 synthetic templates；
- 增益等价于更大 budget；
- natural tasks 上 router 无法泛化；
- GQA union 后预算膨胀消失；
- kernel latency 不优于强 single-policy；
- PolyKV layer-wise method selection复现后覆盖全部增益。

若触发 No-Go，应把工作降级为 head-function analysis 或 specialist-retrieval diagnostic，而不是继续包装成完整系统论文。

## 9. 2026-07-11 Natural Holdout Status

当前触发了“static portfolio / generic router”分支的 No-Go：

- 64 条 query-disjoint holdout 上，frozen deep-QK NLL `3.147`；
- frozen specialist `3.313`；
- frozen deep27+specialist12 `3.239`；
- 原 64 条上的小幅 static-portfolio gain 未复现；
- answer-free entropy/question-NLL gate 和 head-QK confidence gate 均无法在控制 tail regret 的同时产生有效切换。

但 operator-library oracle 仍为 `2.799`，相对 deep 有 `0.347` headroom，且 specialist 在 45.3% query 上更优。因此下一阶段不是放弃 operator identity，而是将 router teacher 改为 exact per-head attention-output distortion/omitted mass，并把目标改为 mean + CVaR constrained routing。

## 10. Causal Teacher Pilot Status

该修改已经完成首轮验证：64 natural queries、448 heads、172,032 exact labels。阈值 `relative head-output L2 <= 0.05` 下：

- fixed QK-8 violation rate 29.5%；
- static head prior：11.07 blocks，1.18% violations；
- query-disjoint head-local conformal 90%：10.95 blocks，0.86% violations；
- test oracle：8.24 blocks，0% violations。

该结果通过了“operator identity 是否比固定 operator 必要”和“query-conditioned 是否超过 static head prior”的机制 Gate。下一 Gate 是 GQA physical union、多个 query positions、answer NLL correlation、8B replication 和真实 sparse kernel。

## 11. End-to-end Risk Status

GQA physical union 与相邻 query-position Gate 已通过：5% learned route 的端到端 physical saving 为 `17.45%`；同一 query 四个相邻位置的 majority agreement 为 `78.69%`，adjacent exact agreement 为 `66.90%`。

但独立 local-risk 假设触发了新的 No-Go：

- test31 上，epsilon=1%/2%/3%/5% 的 learned physical savings 为 `4.18%/6.92%/10.68%/17.45%`；
- 对应 mean first-token delta NLL 为 `+0.0259/+0.0381/+0.0494/+0.0639`；
- 所有 paired CI 均高于 0；
- exact 1% local oracle 同样有正漂移，因此问题不只是 learned router error，而是跨 28 层的局部误差累积。

修订后的必须关闭 Gate：

1. 用 layer-group causal intervention 估计 downstream amplification；
2. 将独立 per-head threshold 改为 cross-layer propagated global risk budget；
3. 在相同 physical bytes 下证明 risk allocation 优于 uniform epsilon；
4. 用至少 4-token、最终 full-answer NLL 与 generation score确认结论；
5. 通过 GQA group gather+SDPA reference 后，再实现 fused kernel。

在这些 Gate 通过前，当前结果只能称为 strong mechanism paper prototype，不能称为 ICLR-ready system。

### 11.1 External holdout update

Cross-layer allocation 已通过第一轮外部冻结验证：第二个自然 holdout64 与旧 128 条记录零重叠。allocated epsilon=0.02 在 `6.19%` physical saving 下 mean delta NLL 为 `+0.0106`、p95 为 `0.056`；uniform epsilon=0.01 在更低的 `4.55%` saving 下反而为 `+0.0368`、p95 `0.144`。所以“amplification-aware allocation 优于 uniform local threshold”通过了 query/record-disjoint Gate。

仍未关闭：

- multi-token/full-answer 与 generation score；
- 8B transfer；
- fused kernel。将 8 次 physical-group gather+SDPA 合并为一次 padded-ragged call 后，16K 已达到 `1.17-1.18x`、32K 达到 `1.50-2.34x`，但 4K 仍只有 `0.29-0.30x`。因此长上下文 attention-subsystem Gate 有进展，4K/end-to-end speed Gate 仍是 No-Go。
