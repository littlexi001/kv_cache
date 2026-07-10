# Section 117: 低 KV Pareto 与 Coverage-Certificate 路由

日期：2026-07-09

目标：当前高质量策略的 KV ratio 约 60%，足够说明“安全动态路由”但不足以支撑“强压缩”叙事。本节记录把静态 full fallback 改成预算化安全动作，以及新增 coverage-certificate 选择器后的探索。

## 1. 已确认的质量型结果

LongBench m20，同一批样本：

| 方法 | Score | KV ratio | Online | 结论 |
|---|---:|---:|---:|---|
| v70 grounded QA verifier | 0.379631 | 62.10% | 2.811s | 比 v64/v53 有正信号，但 narrative 有过度 verifier 风险 |
| v72 2Wiki grounded-only + qasper full | 0.379897 | 61.62% | 2.810s | 当前质量/成本最好的保守点 |
| v74 mixed family certificates | 0.379897 | 62.48% | 2.822s | 与 v72 分数完全相同，但 KV 更高 |
| v75 v74 + 2Wiki retry4096 | 0.376772 | 62.69% | 2.852s | 2Wiki 变差，不采用 |

结论：

- v74 没有超过 v72，只是用更多 KV 复现同样分数。
- v75 说明简单增大 2Wiki retry budget 到 4096 不一定更安全，可能产生 grounded-but-wrong 的错误答案。
- 当前保守主线仍是 v72；如果 m50 上 v72 不稳定，则 v70/v53 是备用稳定点。

## 2. 为什么 60% KV 偏高

v72/v74 的平均 KV ratio 高，主要不是选择器本身不稀疏，而是策略层有大量静态安全动作：

- `hotpotqa`、`musique`、`qasper`、`trec`、`passage_count`、`repobench-p` 中多项直接 full fallback。
- `passage_retrieval_en` 的 output verifier 触发率接近 full，实际 KV ratio 接近 1。
- narrative / multifield / 2Wiki 上的 consistency verifier 会在部分样本触发 full fallback。

因此目前 60% KV 应该解释为“质量优先安全模式”，不能作为最终压缩主结果。

## 3. 新增低 KV Pareto 实验

### v76: no static full fallback

设计：

- 去掉原来的静态 full fallback。
- 原 full 任务改成固定预算动作。
- 目标是测试最低 KV 点，观察质量崩溃边界。

当前早期状态：

- 约 137/320 samples。
- partial Score 约 0.339。
- partial KV ratio 约 41.1%。

解释：v76 已经证明 KV 可以明显降到 40% 左右，但质量风险较大，需要完整结果和任务级分析。

### v77: mid-KV budgeted safety

设计：

- 只保留 `qasper` full。
- `hotpotqa`、`musique`、`trec`、`passage_count`、`repobench-p` 改成 2048/4096 token 的预算化安全动作。
- 目标是寻找 45%-55% KV 的中间 Pareto 点。

当前状态：

- 已启动并运行。
- 早期 partial KV ratio 约 55.6%，分数尚不能下结论。

### v78: coverage-certificate budgeted routing

新增机制：coverage certificate。

核心想法：

- 原 coverage-MMR 只是给候选 page 加分，不能保证 query 的不同实体、数字、长尾词都被覆盖。
- v78 在 MMR 前先做一个硬 set-cover 步骤：用少量预算选择能覆盖不同 query certificate terms 的 page。
- 然后再用原来的 anchor / MMR / bridge / flow 填剩余预算。

形式化地说，给定 query certificate term 集合 \(Q\)，每个 page \(p_i\) 覆盖集合 \(C_i \subseteq Q\)。在证书预算 \(B_c\) 内，先贪心选择：

\[
p^\star = \arg\max_{p_i} \left(|C_i \setminus C_{\mathrm{covered}}|, s_i, -|p_i|\right)
\]

直到证书覆盖率足够或预算耗尽。这里 \(s_i\) 是原 page evidence score。

这不是 RAG 检索，因为它不向 prompt 注入外部文本；它只决定当前输入 KV cache 中哪些 prefix block 进入下一步 decode。

当前状态：

- 已实现并启动 m20。
- early partial 约 43/320 samples，KV ratio 还偏高，因为 narrative/qasper 早期触发了安全动作。
- 需要完整结果后判断它是否比 v77 更优。

## 4. 当前论文叙事判断

目前可站住的故事：

- RiskKV-Block 不是单纯 top-k block retrieval，而是“任务风险感知的 KV memory action routing”。
- 高质量点使用最小必要安全证书：例如 v72 只在 2Wiki 上启用 grounded verifier，避免把 verifier 泛化到 narrative 造成副作用。
- 低 KV 点使用预算化安全动作和 coverage certificate，展示质量-压缩 Pareto 曲线。

如果 v76/v77/v78 中至少一个达到：

- Score 接近 v72/v70，且
- KV ratio 降到 45%-55% 或更低，

那么论文叙事会明显增强：主方法不是只有一个 60% KV 的保守点，而是一条可调 Pareto frontier。

## 5. 下一步决策

1. 等 v76/v77/v78 m20 完成。
2. 如果 v77 或 v78 在 45%-55% KV 下 score 接近 0.37-0.38，立刻启动 m50。
3. 如果 v78 比 v77 分数更好，coverage certificate 进入主方法；否则作为负结果或 ablation。
4. 如果 v76 质量崩溃但 KV 很低，将其作为 Pareto 下界，不作为主方法。
5. m50 上只扩展最有希望的 1-2 个版本，避免 GPU 被低价值组合占满。

## 6. v76-v78 中途诊断与停止

v76/v77/v78 跑到 partial 后，已经足够判断“全局低 KV”不适合作为下一步主线。停止前保存了任务级 partial 诊断：

`outputs/diagnostics_20260709/v76_v77_v78_partial_by_task_before_stop.csv`

注意：最初的任务级 partial parser 会把部分样本归到错误任务，因为日志不是每个样本都打印 header。后来已修正为按 LongBench 固定任务顺序和 `max_samples_per_task` 归因。修正后的关键观察：

| 方法 | 已解析样本 | partial Score | partial KV | 主要问题 |
|---|---:|---:|---:|---|
| v76 no static full | 178 | 0.2982 | 39.60% | narrative、musique、hotpot 明显掉分 |
| v77 mid-KV safety | 155 | 0.3183 | 46.25% | hotpot、musique 明显掉分 |
| v78 coverage certificate | 130 | 0.3254 | 46.47% | hotpot、musique、2Wiki 均不稳 |

partial 给出的正信号需要修正为：

- `qasper` 可以从 full fallback 改成 2048-token bridge 动作，partial 分数不降，KV 明显下降。
- `hotpotqa` 不能简单预算化；多个版本都显示 hotpot 从 full 改成 sparse 后掉分。

因此停止 v76/v77/v78，释放 GPU 给更有希望的 qasper-only 局部预算化版本。

## 7. v79-v81: 局部预算化消融

基于上述诊断，新启动三条 m20：

| 方法 | 相对 v72 的改动 | 目的 |
|---|---|---|
| v79 | hotpotqa 与 qasper 都预算化 | 发现 hotpot 预算化负面，已中途停止 |
| v80 | 只预算化 hotpotqa，qasper 保持 full | 隔离 hotpotqa，已中途停止 |
| v81 | 只预算化 qasper，hotpotqa 保持 full | 当前最有希望的低 KV 保守改进 |

设计原则：

- `2wikimqa` 保持 v72 的 grounded-only 策略，不再低 KV 化。
- `multifieldqa_en` 保持 v72 的 score-risk gated verifier。
- `musique` 保持 full fallback。
- 只对 partial 证明可预算化的 `hotpotqa` / `qasper` 动手。

v79/v80 已停止，原因是 hotpot 预算化在 partial 中明显不稳。v81 保留并继续运行。

## 8. v82/v83: qasper budget sweep

为了判断 qasper 预算化的最小安全 budget，新启动：

| 方法 | qasper budget | 目的 |
|---|---:|---|
| v82 | 1024 | 测试更激进的 qasper 压缩 |
| v83 | 1536 | 测试中间预算 |
| v81 | 2048 | 当前保守 qasper budgeted 版本 |
| v84 | 3072 | 测试接近 full 的安全预算，补齐 Pareto 曲线 |
| v85 | 1024 -> 2048 adaptive | 默认 1024，仅在 page-score 风险高时升到 2048 |

同时提前启动 v81 m50：

- `riskkv_v81_v72_qasper_budgeted_m50_20260709`

原因：v81 与 v72 只差 qasper 从 full fallback 改为 2048-token bridge 动作。该改动风险较小，且 m20 partial 已经显示 qasper 不降分、KV 明显下降，因此可以提前做 m50 验证。

v82/v83 第一次启动在 GPU 0/1 上遇到 OOM。原因不是配置本身，而是同一时间另有 `long_needle_age` 进程占用了 GPU 0/1 显存。随后已改用空闲 GPU 3/4 重启：

- `riskkv_v82_v72_qasper1024_m20_retry_20260709`
- `riskkv_v83_v72_qasper1536_m20_retry_20260709`

后续比较时应忽略未完成的第一次 v82/v83 输出目录，只使用带 `retry` 后缀的完整结果。

v84 已补充启动：

- `riskkv_v84_v72_qasper3072_m20_20260709`

最终 qasper sweep 将形成 `1024 / 1536 / 2048 / 3072 / full` 的预算曲线。这个曲线对论文很重要：它可以把“qasper full fallback 是经验规则”改写成“任务族风险下的最小安全预算选择”，更适合作为方法论叙事。

v84 第一次也因为 GPU 0 被 `long_needle_age` 占用显存而 OOM。已新增 watcher 等待空卡重跑：

- `scripts/watch_and_launch_v84_qasper3072_retry_20260709.sh`
- 目标输出目录：`riskkv_v84_v72_qasper3072_m20_retry_20260709`

自动选择器已改为等待 v84 的 `retry` 目录，忽略第一次失败的 v84 输出。

## 10. v85: adaptive qasper budget

固定 qasper budget 的早期结果显示：

- 1024 / 1536 可以明显降低 KV，但 qasper 分数低于 2048。
- 2048 更安全，但 KV 更高。

因此新增 v85：

- qasper 默认使用 1024-token bridge 动作。
- 如果 pre-decode page-score 风险高，则自动升到 2048。
- 风险条件沿用早期 score-risk probe：`score_risk_min_gap2=0.15` 且 `score_risk_max_entropy=0.93`。

这一步的意义是把 qasper budget 从固定超参推进到样本级自适应预算：

\[
B(x)=
\begin{cases}
2048, & r(x)=1,\\
1024, & r(x)=0.
\end{cases}
\]

其中 \(r(x)\) 是由 page-score gap/entropy 给出的 pre-decode risk certificate。若 v85 能接近 v81 的分数并接近 v82 的 KV，它会比固定 budget 更适合写进主方法。

自动选择器已更新为等待 v81-v85 全部完成后再选择 m50 候选。

## 9. 自动选择 qasper budget 并升 m50

已新增自动选择器：

- `scripts/select_qasper_budget_policy_20260709.py`
- `scripts/watch_select_qasper_budget_m50_20260709.sh`

选择规则：

1. 等 v81/v82/v83/v84 的 m20 `summary.csv` 全部出现。
2. 读取 v72 m20 作为 baseline。
3. 如果候选分数不低于 `baseline_score - 0.002`，在这些候选中选择 KV ratio 最低者。
4. 如果没有候选满足质量阈值，则选择分数最高者。
5. 自动启动所选策略的 m50。

这相当于把“预算选择”从人工调参推进到一个可复现实验协议：质量约束优先，成本最小化其次。该协议可以写进论文实验设置或附录，用于解释为什么最终选择某个 qasper budget。
