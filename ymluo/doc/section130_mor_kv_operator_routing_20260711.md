# Section 130：MoR-KV 功能专门化 Head 的检索算子路由（2026-07-11）

## 1. 项目定位

Section 129 说明 heads 具有稳定但非永久固定的 attention-pattern bias。Section 128 说明所有 heads 的候选并集能找到更多答案，但多数共识会删除只被少数专业 heads 发现的 gold blocks。

本节沿这两条证据提出：

```text
MoR-KV: Mixture-of-Retrievers for Function-Specialized KV Access
```

核心不再是“不同 head 分不同 KV budget”，而是：

> 对每个 query 动态选择 head portfolio 和 KV retrieval operator，并在 GQA 共享 K/V 的物理约束下保护 specialist minority nominations。

项目：

```text
ymluo/projects/mor_kv_operator_routing
```

## 2. 为什么普通 head-aware budget 不够新

最新工作已经覆盖多个相邻方向：

- RazorAttention / DuoAttention：retrieval heads full cache，其他 heads streaming；
- HeadKV / Task-KV：head importance 或 task-aware semantic differentiation；
- REAL：attention behavior 与 head-wise dynamic budget；
- CompilerKV：offline head reliability table + prompt risk；
- PolyKV：layer-wise compression method selection + budget allocation；
- HARD-KV：dynamic head budget 到静态 contiguous runtime layout；
- RedKnot：head-aware serving substrate。

因此，若只做 head 分类和非均匀预算，创新性不足。MoR-KV 必须保持的区别是：

1. 路由检索**算子语义**，不是只路由 budget；
2. 单位是 query-head/GQA group，不是静态 layer；
3. 每个 query 由 score signature 动态激活；
4. 用 minority/group-saturating objective 保住专业 head 的少数意见；
5. 将动作落到不复制 K/V 的 GQA physical block union。

完整 novelty audit：

```text
ymluo/projects/mor_kv_operator_routing/doc/novelty_audit.md
```

## 3. v1 方法

v1 使用两类远程 operator：

1. lexical block operator：BM25 research prototype；
2. semantic operator：真实 Qwen3-0.6B pre-RoPE Q/K、每 layer/KV-head centered SVD32。

Query router 输入不是 task label，而是正式 KV gather 前已有的跨-head score signature：

```text
top1 score
top1 - top4 margin
candidate score std
```

训练协议：

```text
train 300: head utility + router centroid
dev   100: operator/head-count/depth/quota
test  100: frozen evaluation
```

GQA 默认对每个 `(layer, kv_head)` 只保留 calibration utility 更高的 query head，避免将 16 query heads 错误解释为 16 份独立 K/V。

## 4. v1 Held-out 结果

数据为四种任务各 125 条，共 500 queries：lexical、semantic paraphrase、hard negative、multihop。每个 record 25,088 tokens，四条 record 组成约 100K-token corpus。

Router accuracy：

| split | accuracy |
| --- | ---: |
| train | 1.000 |
| dev | 0.990 |
| test | 1.000 |

这是受控数据结果，不能外推到自然任务。

Test utility 主表：

```text
utility = evidence_fraction - 0.25 * hard_negative_hit_rate
```

| Blocks | BM25 | single hybrid | wrong router | MoR-KV |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.135 | 0.135 | 0.145 | **0.208** |
| 4 | 0.530 | 0.553 | 0.413 | **0.563** |
| 8 | 0.620 | 0.610 | 0.568 | **0.645** |
| 16 | 0.640 | 0.630 | 0.588 | **0.658** |
| 39 | 0.690 | 0.680 | 0.565 | **0.698** |

在 4-block 预算：

| 方法 | evidence fraction | hard-negative hit | utility |
| --- | ---: | ---: | ---: |
| BM25 | 0.775 | 0.980 | 0.530 |
| global hybrid | **0.785** | 0.930 | 0.553 |
| wrong router | 0.605 | 0.770 | 0.413 |
| MoR-KV | 0.755 | **0.770** | **0.563** |

第一轮最重要的机制结果：错误 operator route 使 utility 从 `0.563` 降到 `0.413`，说明正确的 head/operator matching 有效；同时 MoR-KV 将 BM25 hard-negative exposure 从 `98%` 降到 `77%`。

## 5. Answer NLL

Raw utility-routed MoR 的 retrieval 指标更好，但 answer NLL 几乎没有改善：

| 4-block 方法 | Test mean answer NLL | Delta vs BM25 |
| --- | ---: | ---: |
| BM25 | 3.765 | 0.000 |
| single global hybrid | 3.706 | -0.059 |
| raw utility-routed MoR | 3.763 | -0.003 |
| wrong router | 4.942 | +1.177 |

这是否定了“优化 evidence/distractor utility 就等价于优化模型质量”的过强假设。

之后只用 dev NLL 在三个冻结 operator actions 中为每个 task family 编译策略，再在 test 上评估：

| 方法 | Test mean answer NLL |
| --- | ---: |
| BM25 | 3.765 |
| single global hybrid | 3.706 |
| raw MoR | 3.763 |
| **dev-NLL compiled MoR** | **3.520** |

相对 BM25 的 paired delta 为 `-0.246`，bootstrap 95% CI 为 `[-0.425, -0.083]`；相对 global hybrid 为 `-0.186`，CI 为 `[-0.357, 0.002]`。这支持“需要用模型 loss 编译 operator policy”，但相对 strongest hybrid 仍是边界显著，必须扩大自然 test。

## 6. 诚实判断

当前结果是 promising mechanism evidence，但远没有达到 ICLR 完成度：

1. 相对最强 global hybrid 的绝对 utility 增益只有 `+0.010`；
2. hard-negative task 自身仍有 evidence recall 损失；
3. multihop 没有改善；
4. router 的 100% 来自受控任务族；
5. v1 仍是 block nomination，不是真实 sparse-attention kernel；
6. 尚未有 8B、多模型、LongBench/RULER、实际显存和速度主表。

所以当前不能宣称“足够发 ICLR”，只能说论文方向通过了第一轮机制筛选。

### 6.1 Natural LongBench No-Go probe

在现有 64 条真实 LongBench answer NLL 上，每个 dataset 交替拆成 `33 calibration / 31 held-out`，从 BM25、risk hybrid、deep-QK 三个 39-block operator 中选择。Held-out 结果：

| 方法 | mean answer NLL |
| --- | ---: |
| BM25 | 3.829 |
| risk hybrid | 3.601 |
| **global deep-QK** | **3.304** |
| dataset-routed | 3.510 |

Dataset route 相对 global deep-QK 退化 `+0.206`，95% CI `[+0.032,+0.439]`。因此 synthetic route 尚未自然泛化；这不是可以忽略的小问题，而是当前论文主张的首要风险。

对 held-out 每条 query 事后选择三个 operators 中 NLL 最低者，oracle NLL 为 `3.247`，相对 global deep-QK `3.304` 也只有 `0.057` headroom。说明当前自然 operator library 的互补性不足；仅改进 router 不可能产生足够大的论文结果。

### 6.2 Submodular specialist objective probe

固定 operator/head policy，仅在 dev 调 submodular temperature。它在 1-block/39-block utility 上分别从 `0.208→0.218`、`0.698→0.700`，但在 4/8/16 blocks 均退化。当前 group-saturating objective 尚未通过稳定实证 Gate。

## 7. 下一步优先级

1. 在真实 LongBench 上用 corpus-disjoint split 训练 score-signature router；
2. 接入 streaming、structure 和 dense fallback operator；
3. 实现 group-saturating submodular block selection；
4. 在 Qwen3-8B 与 Llama-3.1-8B 实现真实 GQA sparse attention；
5. 与 CompilerKV、PolyKV、DuoAttention、HeadKV 和 HARD-KV 做等 KV bytes 对照；
6. 实现 risk gate，控制 wrong-route tail。

详细执行 Gate：

```text
ymluo/projects/mor_kv_operator_routing/doc/iclr_execution_plan.md
```

## 8. Natural query-disjoint holdout

新增 64 条与原 calibration `record_uid` 零重叠的自然 query；底层 10M-token blocks、records 和 block IDs 的 SHA-256 完全一致，因此复用冻结的 134GB K-SVD index，只重新计算 query profiles 和 per-head Top-16。

所有 specialist heads、action 和 `deep27 + specialist12` quota 均在 target answer NLL 前冻结。

| action | mean answer NLL |
| --- | ---: |
| **deep-QK** | **3.147** |
| BM25 | 3.239 |
| frozen specialist | 3.313 |
| frozen deep27 + specialist12 | 3.239 |

原样本上 static portfolio 的 `3.258 vs deep 3.278` 没有复现，属于探索性偶然收益。

但是 specialist 在 45.3% holdout queries 上胜过 deep，四 action oracle mean NLL 为 `2.799`，仍存在 `0.347` headroom。generic question-NLL/entropy gate、action-regret gate 和 61-feature head-QK confidence gate 均无法安全兑现这部分收益；risk optimization 会退化成不切换。

因此论文主线需要修改为：用 exact per-head omitted attention mass / attention-output distortion 生成 causal dense teacher，学习满足 mean 与 CVaR tail constraint 的 operator route，再做 GQA physical union。完整审计见 `ymluo/projects/mor_kv_operator_routing/doc/natural_holdout_report.md`。

## 9. Causal head-distortion teacher: first positive route

已在 64 natural queries、4K context、Qwen3-0.6B 全部 448 query heads 上生成 172,032 个 exact post-RoPE attention-output distortion labels。

阈值 `relative output L2 <= 0.05`：

| policy | mean logical blocks | p95 error | violation rate |
| --- | ---: | ---: | ---: |
| full | 15.29 | `2e-7` | 0.0% |
| fixed QK-8 | 8.00 | 0.1490 | 29.5% |
| static head prior | 11.07 | 0.0294 | 1.18% |
| **query-conditioned head conformal 90%** | **10.95** | **0.0275** | **0.86%** |
| test oracle | 8.24 | 0.0445 | 0.0% |

这是当前第一个 query-disjoint 正结果：router 只使用 head identity 与部署前可得的 QK/lexical score signatures，却同时优于 static head prior 的逻辑 blocks 和 tail risk。完整报告见 `ymluo/projects/mor_kv_operator_routing/doc/causal_teacher_pilot.md`。

GQA 两个 query heads 做 physical union 后，learned router 为 `12.77/15.29` blocks，即 `16.46%` 物理节省；static prior 为 `15.37%`，test oracle 为 `32.12%`。所以 GQA union 缩小但没有消除 query-head operator routing 的物理收益。

## 10. 相邻 query position 稳定性

在额外 16 条 query-disjoint 自然样本上，对每条输入最后 4 个 question-content token、全部 448 个 query heads 重新生成精确 teacher，共 172,032 条 head/action 标签。

在主阈值 `relative output L2 <= 0.05` 下：

- exact oracle mean blocks：`8.97`；
- p95 relative output error：`0.0448`；
- 同一 query 四个位置的多数动作一致率：`78.69%`；
- 相邻位置的动作完全一致率：`66.90%`。

所以 head 偏好不是永远不变：它有明显稳定先验，但约三分之一相邻 token 转移会改变“满足风险约束时最便宜的 operator”。这同时否定了永久静态 head 标签，也支持 `head prior + query-position activation` 的层级路由设计。

## 11. 端到端 causal NLL reference

新增模型内干预：在每一层仅将最终 prompt token 的 per-head attention output 替换成稀疏输出，再继续执行正常的 output projection、残差、MLP 与后续层；此前 token 完全不变。答案 continuation 严格使用既有协议：prompt 以 `Answer:` 结尾，gold answer 以前导空格开始。严格 test31、4K-context 结果：

| policy | mean blocks | head violation | mean ΔNLL vs full | paired 95% CI | p95 abs ΔNLL |
| --- | ---: | ---: | ---: | ---: | ---: |
| **learned conformal 5%** | **10.81/15.29** | **3.20%** | **+0.0642** | **[0.0394, 0.0900]** | **0.187** |
| exact 5% risk oracle | 8.05/15.29 | 0.0% | +0.0679 | [0.0358, 0.0994] | 0.210 |
| fixed QK-8 | 8.00 | 30.8% | -0.0044 | [-0.0636, 0.0582] | 0.369 |
| fixed lexical-8 | 8.00 | 47.9% | +0.338 | [-0.0069, 0.860] | 1.256 |
| fixed uniform-8 | 8.00 | 51.1% | +0.302 | [-0.0767, 0.834] | 1.565 |

所以 5% head-output 风险点不是无损：learned router 有小而显著的平均 NLL 损失。但它确实将固定 QK-8 的 p95 NLL 长尾从 `0.369` 压到 `0.187`，并把 head-error 违例从约 `31%` 降到 `3.2%`。下一道 Gate 是 1%/2%/3% 风险扫描、multi-token/full-answer NLL 和真实 sparse kernel，而不是把当前点包装成无损结果。

冻结同一个 router，只改变部署阈值后的严格 test31 Pareto：

| epsilon | logical blocks | physical GQA saving | mean ΔNLL | p95 abs ΔNLL |
| ---: | ---: | ---: | ---: | ---: |
| 0.01 | 13.67 | 4.18% | +0.0259 | 0.068 |
| 0.02 | 12.96 | 6.92% | +0.0381 | 0.110 |
| 0.03 | 12.13 | 10.68% | +0.0494 | 0.136 |
| 0.05 | 10.81 | 17.45% | +0.0639 | 0.183 |

曲线单调但没有无损的非零压缩点；甚至 exact 1% local oracle 仍有正的平均 NLL 漂移。因此更强的论文方法必须从“每个 head 独立满足阈值”升级为“跨层传播并分配一个全局 logit-risk budget”。

## 12. Cross-layer risk amplification

在 epsilon=0.02 下分别只稀疏一个七层组：

| sparse layers | physical saving | mean ΔNLL | paired 95% CI | p95 abs ΔNLL |
| --- | ---: | ---: | ---: | ---: |
| 0-6 | 2.34% | -0.0050 | [-0.0104, -0.0008] | 0.021 |
| 7-13 | 1.95% | +0.0067 | [0.0033, 0.0103] | 0.021 |
| **14-20** | **1.28%** | **+0.0319** | **[0.0215, 0.0430]** | **0.083** |
| 21-27 | 1.35% | +0.0047 | [0.0003, 0.0091] | 0.027 |

14-20 层每 1% 物理节省造成的 NLL 放大约为其它正代价层组的 7 倍。探索性地只对 `0-13 ∪ 21-27` 使用 epsilon=0.02、让 14-20 回退 full，可获得 `5.63%` 物理节省、ΔNLL `+0.0084`、95% CI `[-0.0002, 0.0157]`、p95 `0.041`；它优于 uniform epsilon=0.01 的 `4.18%` 节省、`+0.0259` ΔNLL 和 `0.068` p95。

但这一层排序看过当前 test31，因此只能算 post-hoc mechanism evidence，不能进正式主表。必须在 calibration 上冻结 amplification weights，并在新的 record_uid 零重叠 holdout 上验证。

## 13. Physical GQA execution timing

真实 reduced-compute reference 对每个物理 KV head 求 block union、gather 对应 K/V，再为共享它的两个 query heads 调一次 SDPA。包含 gather 与 8 次 group launch 的结果：

| full KV | dense | 8-call grouped | 1-call padded-ragged | batched speedup |
| ---: | ---: | ---: | ---: | ---: |
| 4K | 0.128 ms | 0.701-0.703 ms | 0.431-0.437 ms | 0.29-0.30x |
| 16K | 0.511 ms | 0.704-0.754 ms | 0.431-0.436 ms | 1.17-1.18x |
| 32K | 1.011 ms | 0.699-0.754 ms | 0.431-0.676 ms | 1.50-2.34x |

全块保留时两条 sparse path 都数值等价且不计算未选 KV。一次 padded-ragged gather+SDPA 将 crossover 从 32K 提前到 16K，但 4K 仍比 dense 慢约 3.4 倍。这是固定 512-3072 active tokens 的 attention 子系统测试，不能替代当前约 6% 物理节省点的端到端速度；4K 仍需 fused page kernel。

## 14. 第二个零重叠 holdout 的冻结验证

新建自然 holdout64_v2，同时排除 clean64 与 holdout1 的全部 128 个 `record_uid`；三者两两零重叠。新 corpus 哈希与旧版不同，因此按外部分布测试处理。router bundle 与“14-20 层保持 full”的策略在运行前冻结。

| frozen policy | physical saving | mean ΔNLL | paired 95% CI | p95 abs ΔNLL |
| --- | ---: | ---: | ---: | ---: |
| uniform epsilon=0.01 | 4.55% | +0.0368 | [0.0249, 0.0496] | 0.144 |
| uniform epsilon=0.02 | 7.56% | +0.0500 | [0.0348, 0.0668] | 0.184 |
| **allocated epsilon=0.02, sparse 0-13/21-27** | **6.19%** | **+0.0106** | **[0.0031, 0.0187]** | **0.056** |
| allocated epsilon=0.03, sparse 0-13/21-27 | 8.97% | +0.0183 | [0.0084, 0.0288] | 0.119 |
| **allocated epsilon=0.05, sparse 0-13/21-27** | **14.07%** | **+0.0363** | **[0.0198, 0.0541]** | **0.128** |

allocated policy 比 uniform-1% 节省更多物理 KV，同时平均 NLL 漂移降低约 71%、p95 降低约 61%；相对 uniform-2% 保留了大部分节省，但平均漂移降低约 79%。这是当前第一个真正 query/record-disjoint 的主张级正结果，但 residual mean shift 仍显著大于 0，因此主张是“dominating risk-quality trade-off”，不是无损压缩。

allocated epsilon=0.05 则把物理节省推到 `14.07%`：约为 uniform-1% 的三倍，但 mean drift 基本相同且 p95 更低；相对 uniform-2% 也同时有更高节省和更低误差。这是当前最强首 token capacity 点，完整答案验证正在运行。

完整答案上该 aggressive 点的物理节省为 `13.85%`，但 mean Δanswer-NLL 升到 `+0.0436`、p95 `0.269`，不再保持 quality dominance。因此它只能作为 capacity 点；默认完整答案质量点仍是 allocated epsilon=0.02 的 `6.32%/+0.0133/p95 0.089`。epsilon=0.03 的完整答案中间点正在补测。

同一外部 holdout 上评分前 4 个 gold answer tokens：allocated policy 物理节省 `6.27%`，mean ΔNLL `+0.00563`，95% CI `[-0.00113, 0.01252]`，p95 `0.0657`，均值无法与 full 区分。完整首个 reference answer 上物理节省 `6.32%`，mean ΔNLL `+0.01334`，95% CI `[0.00103, 0.02890]`，median `+0.00193`，p95 `0.0892`。完整答案损失小但可测，且由少数长尾样本驱动。

完整答案的 uniform epsilon=0.01 对照只节省 `4.97%`，mean ΔNLL `+0.02279`，median `+0.01168`，p95 `0.1120`。allocated policy 在更高压缩下将平均损失降低约 41%、median 降低约 83%、p95 降低约 20%，所以 dominance 不只存在于首 token。

## 15. Complete-answer intermediate point and risk-gate audit

After flooring ridge-plus-conformal bounds at the mathematically required zero lower limit, the corrected zero-overlap holdout64 result at allocated epsilon=0.03 saves `8.89%` physical GQA KV and changes complete-answer NLL by `+0.02598` (paired 95% CI `[0.01102, 0.04397]`), with median `+0.00812` and p95 absolute delta `0.1526`. The query-disjoint head-router summary is unchanged; only a small number of equal-budget tie breaks change. These numbers supersede the earlier `+0.02308` result.

The same run records deployable query-risk summaries from the selected per-head conformal bounds. Mean/p95/max upper bound and near-threshold fraction have absolute-answer-delta Pearson correlations of only `-0.043/-0.120/0.064/-0.132`, and calibration-derived layer-amplification weighting is still weak (`-0.048` Pearson, `0.172` Spearman). Therefore per-head conformal calibration does not automatically produce a sequence-level answer-loss certificate. A query fallback selected on these values is post-hoc and cannot be promoted to the main method without a new holdout; the next method revision needs a propagated logit/answer-risk estimator. The per-layer union across every KV head also retains essentially all `15.47` blocks, so the `8.89%` byte saving requires per-KV-head sparse page layouts rather than one shared layer-level token set.

## 16. Qwen3-8B all-layer transfer

The completed eight-query pilot covers all 36 layers, 1,152 query heads, and 55,296 exact teacher rows. Mean relative output L2 is `0.0857` for the same-budget mass oracle, `0.0931` for QK, `0.1350` for lexical, `0.1415` for uniform, and `0.2227` for streaming. At exact epsilon=0.05, the cheapest feasible action averages `5.839/7.875` logical blocks with p95 error `0.0449`; mean head-action agreement across queries is `78.98%`, and `606/1,152` heads keep one action on at least 80% of queries. Thus stable head priors and context-dependent operator activation both transfer to 8B.

After the four query heads sharing each physical KV head are unioned, epsilon=0.05 retains `7.324/7.875` blocks, only `6.99%` physical saving. Epsilon=0.02/0.10 yield `1.21%/19.11%`. This is exact-oracle capacity, not a learned router or a downstream-NLL claim. It shows that physical-GQA-aware routing must be optimized directly; logical query-head savings alone overstate the scalable system benefit. A 32-query all-layer teacher run has been launched for a valid fit/conformal/test split.

此前的单 query、layer0 smoke 给出 full 数值误差 `2e-8`，并验证两卡 `device_map=balanced` 可行；上述全层结果已经取代“仅排队”的状态。当前仍不能宣称 8B downstream-quality 结论。
