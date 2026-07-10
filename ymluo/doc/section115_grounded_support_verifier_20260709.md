# Section 115: Grounded Support Verifier 与下一轮 RiskKV-Block 迭代

日期：2026-07-09

## 当前可靠基线

截至本轮实验，最稳的实际可用方法仍是 `v64 benefit conformal counterfactual qasper full`。

| 方法 | 样本 | Score | KV ratio | Online |
|---|---:|---:|---:|---:|
| v53 consistency quality qasper full | 320 | 0.375890 | 62.89% | 2.736s |
| v64 benefit conformal counterfactual qasper full | 320 | 0.376772 | 60.89% | 2.845s |
| v65 global Coverage-MMR | 320 | 0.375736 | 60.89% | 2.792s |
| v66 task Coverage-MMR | 320 | 0.375538 | 60.89% | 2.843s |
| v67 coverage-risk gate | 320 | 0.376772 | 60.89% | 2.836s |
| v68 Coverage-MMR + coverage-risk gate | 320 | 0.375538 | 60.89% | 2.806s |
| v69 calibrated coverage-risk | 320 | 0.370553 | 59.54% | 2.834s |
| v70 grounded QA verifier | 320 | 0.379631 | 62.10% | 2.811s |
| v71 support-window QA verifier | 320 | 0.379631 | 62.10% | 2.815s |
| v72 2Wiki grounded-only verifier | 320 | 0.379897 | 61.62% | 2.810s |
| v73 v53 + 2Wiki verifier | 320 | 0.379015 | 62.77% | 2.849s |

结论：

1. v64 相比 v53 是一个干净的小提升：质量略高，KV 更低。
2. Coverage-MMR 不能作为默认排序目标；v65/v66 没有提升，且 v66 在 `multifieldqa_en` 与 `narrativeqa` 出现净损失。
3. v67 与 v64 结果一致，说明当前 coverage-risk 阈值大多没有产生有效动作变化。

## 样本级诊断

v64 相比 v53：

- matched = 320
- wins = 1
- losses = 0
- net delta = +0.282051
- 主要赢在 `multifieldqa_en` 一个样本，同时减少了部分不必要的 consistency full fallback。

v66 相比 v64：

- matched = 320
- wins = 0
- losses = 2
- net delta = -0.394643
- 损失来自 `multifieldqa_en` 与 `narrativeqa`。

v68 相比 v64：

- matched = 320
- wins = 0
- losses = 2
- net delta = -0.394643
- 损失样本与 v66 相同，coverage-risk trigger rate 约为 1.56%，不足以改变结论。

v70 相比 v64：

- matched = 320
- wins = 2
- losses = 2
- net delta = +0.915079
- 主要收益来自 `2wikimqa` 一个样本从 0 提升到 1。
- 主要代价来自 `narrativeqa` 两个小损失。
- KV ratio 从 60.89% 增加到 62.10%，增加约 1.21 个百分点。

解释：grounded QA verifier 修复了一个高价值 multi-hop QA 错误，但在 narrative QA 上会产生少量过度 fallback 或错误 retry。这个信号值得进入 m50，但还不能直接宣称主方法替换 v64。

v69 相比 v64：

- matched = 320
- wins = 0
- losses = 5
- net delta = -1.989881
- 主要损失来自 `narrativeqa`，coverage 校准把若干原本受保护的样本降成较小 KV 动作。

解释：coverage-risk 可以降低 KV，但它没有可靠地区分“可安全降预算”和“必须保守”的样本，因此目前不适合作为主线安全控制器。

v71 相比 v70：

- matched = 320
- wins = 0
- losses = 0
- net delta = 0
- support-window active rate = 18.75%
- support-window fallback rate = 0

解释：support-window verifier 在当前 m20 样本上没有改变动作，说明普通 grounding verifier 已经覆盖了这批样本中的主要可修复失败。v71 可保留为扩展消融，但当前主线应使用更简单的 v70。

## v72: 最小必要证书路由

v70 的正收益主要来自 `2wikimqa`，而 `narrativeqa` 出现两个小损失。因此下一版不再把 grounded verifier 扩到 narrative QA，而是只在 `2wikimqa` 上启用：

- output verifier
- grounding verifier
- retry 到 2048 token budget
- score-risk gated consistency verifier 保持不变

这对应一个更适合论文的故事：\textbf{不是全局加 verifier，而是给不同风险族路由最小必要安全证书}。如果 v72 保留 v70 的 `2wikimqa` 收益，同时避免 narrative 损失，那么它会比 v70 更干净，KV 也应该低于 v70。

预期：

- Score 可能略高于 v70 或接近 v70。
- KV ratio 应低于 v70，因为 narrativeqa 不再额外触发 grounded retry/full。
- 如果成立，v72 比 v70 更适合作为主方法版本。

## v73: 保守底座 + 2Wiki 证书

v64 m50 没有复现 m20 正信号：

| 方法 | 样本 | Score | KV ratio | Online |
|---|---:|---:|---:|---:|
| v53 consistency + qasper full | 800 | 0.358321 | 62.47% | 2.801s |
| v64 conformal counterfactual | 800 | 0.353371 | 59.73% | 2.813s |
| v70 grounded QA verifier | 800 | 0.358326 | 60.60% | 2.847s |

样本级对比显示，v64 相比 v53 的主要损失来自 `2wikimqa`，说明 v64 的 conformal gate 在 m50 上过度减少了 consistency full fallback。

v70 m50 基本追平 v53 分数，同时 KV ratio 从 62.47% 降到 60.60%。样本级对比显示 v70 有 7 个 wins、8 个 losses，net delta = +0.004266；主要损失来自 narrativeqa 的 grounded retry，主要收益来自 narrativeqa / `2wikimqa` 的若干修复。这进一步支持 v73 的设计：保留 v53 的 narrative 行为，只给 `2wikimqa` 增加 grounded verifier。

因此新增 v73：

- 以 v53 为底座，保留更保守的 consistency verifier。
- 只在 `2wikimqa` 上增加 output/grounding verifier 和 2048 retry。
- 目标是在不牺牲 v53 大样本稳定性的前提下，获得 v70/v72 在 `2wikimqa` 上的收益。

如果 v73 m20/m50 成立，它会比 v64/v72 更适合作为投稿主线，因为它不依赖 m20 上不稳定的降 fallback 策略。

v72 正式 m20 结果：

- Score = 0.379897
- KV ratio = 61.62%
- Online = 2.810s
- 相比 v70：wins = 2, losses = 1, net delta = +0.084921
- 相比 v64：wins = 1, losses = 0, net delta = +1.000000

结论：v72 是当前 m20 最好方法。它验证了“最小必要证书路由”的方向：只把 grounded verifier 路由到 `2wikimqa`，比把 verifier 泛化到 narrative QA 更好且更省 KV。

v73 正式 m20 结果：

- Score = 0.379015
- KV ratio = 62.77%
- Online = 2.849s
- 相比 v53：wins = 1, losses = 0, net delta = +1.000000
- 相比 v72：wins = 0, losses = 1, net delta = -0.282051

解释：v73 成功保留了 v53 的稳定性并修复一个 `2wikimqa` 样本，但缺少 v72/v64 在 `multifieldqa_en` 上的一个收益。因此 m20 主线仍是 v72；v73 的价值取决于 m50 是否更稳。

## v74/v75: 任务族混合证书

v72 和 v73 的差异说明，不同 QA 家族需要不同安全动作：

- `narrativeqa`: v53 的保守 consistency 更稳定，不应引入 grounded retry。
- `multifieldqa_en`: v64 的 score-risk gated consistency 可以避免某些 full fallback 反而变差的样本。
- `2wikimqa`: grounded verifier 能修复漏证据样本，但 2048 retry 在 m50 中仍可能给出 grounded-but-wrong 答案。

因此新增：

- v74: narrative 用 v53，multifield 用 v64，2wikimqa 用 v53 + grounded retry2048。
- v75: 同 v74，但 2wikimqa 的 retry budget 从 2048 提到 4096，测试更大 sparse retry 是否能修复 m50 中的 2Wiki 错误。

这两个版本更接近最终论文故事：\textbf{RiskKV-Block 不是单一 verifier，而是按任务族路由最小必要安全证书和 fallback 动作}。

这说明“覆盖更多 query 词”本身不等价于“保留正确证据”。Coverage 更适合做风险诊断特征，而不是直接干预排序。

## 新假设：Grounded Support Verifier

当前失败模式之一是：score-risk 很高，但两个稀疏动作的答案并不冲突，因此 consistency verifier 不会触发 full fallback；然而两个答案可能一致地错。

为此新增两个 verifier 实验：

### v70: Grounded QA Verifier

在 v64 基础上，对 `narrativeqa` 和 `2wikimqa` 开启：

- output verifier
- grounding verifier
- retry 到 2048 token budget
- 仍保留 score-risk gated consistency verifier

目标：捕捉“答案候选不在保留上下文中”的稀疏失败。

### v71: Support-Window QA Verifier

在 v70 基础上新增 support-window verifier：

给定生成的短答案候选，如果答案词虽然出现在保留上下文里，但在局部窗口内没有 query 稀有词/数字/实体支撑，则认为当前稀疏动作危险，触发 retry/full。

形式上，对答案候选 token 集合 \(A\)、query evidence token 集合 \(Q\)、保留上下文 token 序列 \(C\)，检查每个答案出现位置 \(i\) 的局部窗口：

\[
\max_{i: C_i \in A} |Q \cap C_{[i-r, i+r]}| \ge \tau.
\]

若不满足，则判定该答案缺少局部证据支撑。

这个 verifier 的定位不是 RAG 检索，而是 KV-cache memory action 的安全判别：它只决定当前 KV 子集是否足够安全，不向模型注入外部文本。

## 当前运行状态

已完成：

- v66: 完整结果已出，不优于 v64。
- v67: 完整结果已出，与 v64 基本一致。
- v68: 完整结果已出，与 v66 基本一致，作为负结果保留。
- v69: 完整结果已出，明显低于 v64，作为负结果保留。
- v70: 完整结果已出，m20 上成为当前最高分实际方法。
- v71: 完整结果已出，与 v70 完全一致，作为扩展消融保留。

正在运行或排队：

- v64 m50: 已启动，用于验证 v64 是否在更大样本上稳定优于 v53。
- v70 m50: 已启动，用于验证 grounded QA verifier 的 m20 正信号是否稳定。
- v72 m20: 已启动，测试只在 `2wikimqa` 上启用 grounded verifier 的最小必要证书路由。
- v72 m50: 已启动，提前验证 v72 若在 m20 为正时是否能稳定到更大样本。
- v73 m20: 已启动，测试 v53 保守底座 + `2wikimqa` grounded verifier。
- v73 m50: 已启动，提前验证 v73 是否能比 v53 m50 更稳。
- v74 m20/m50: 已启动，测试任务族混合证书。
- v75 m20/m50: 已启动，测试 `2wikimqa` retry4096。

## m100 进展

已有较大样本 m100 结果：

| 方法 | 样本 | Score | KV ratio | Online |
|---|---:|---:|---:|---:|
| v52 consistency quality | 1600 | 0.351705 | 58.53% | 2.919s |
| v53 consistency + qasper full | 1600 | 0.356698 | 63.10% | 2.841s |

结论：在 m100 上，v53 仍明显优于 v52，说明 qasper full / 高安全动作不是 m20 偶然现象。但 full m100 和 v37 m100 尚未完成，最终 m100 横向比较还需要等完整 baseline。

## 下一步决策规则

1. 如果 v70/v71 提升 Score 但 KV 明显增加：作为安全 verifier 消融，不作为主方法默认配置。
2. 如果 v70/v71 在不显著增加 KV 的情况下提升 Score：进入 m50 验证，并把 support-window verifier 写入主方法。
3. 如果 v64 m50 仍优于 v53 m50：v64 可作为当前主线基线。
4. 如果 v64 m50 不稳：回到 v53 作为主线，v64 只作为 m20 探索结果。
