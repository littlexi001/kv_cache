# PCIC 主线：Fixed / Online / Rescue / Oracle 证据表（2026-06-29）

目的：把 paper 主线收敛到一个可辩护的核心命题：

> Pairwise-CIC + online blockwise selection + conditional rescue gate 不是固定 sparse attention combo 的小改，而是一个在线策略选择器；rescue gate 用额外 horizon probe 修复短视选择。

本表只使用已有本地 CSV 合成，不重新跑模型，不访问网络。speed/gate 统一采用 corrected gate 口径。

补充的 blockwise policy trace 证据见：`docs/pcic_blockwise_policy_trace_2026_06_29.md`。

## 主结果

| dataset | best fixed | fixed ΔPPL | top2 ΔPPL | conditional rescue ΔPPL | oracle ΔPPL | cond-fixed | cond-oracle gap | corrected gate_s | unique combos | switches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hard-topic eval128 | `0,6` | -0.020744 | 0.004371 | -0.049633 | -0.049633 | -0.028889 | 0.000000 | 89.624 | 2 | 2 |
| War and Peace | `0,7` | -2.135311 | -2.135311 | -2.135311 | -2.135311 | 0.000000 | 0.000000 | 6.659 | 1 | 0 |
| Count of Monte Cristo | `2,0` | -0.009005 | -0.219215 | -0.219215 | -0.219215 | -0.210210 | 0.000000 | 10.011 | 2 | 1 |

## 逐项解释

### Hard-topic eval128

- 结论：conditional rescue 修复 top2 delayed-win failure，并达到 blockwise oracle。
- top2 combos：`0,7/2,0,7,12/0,6/0,13`
- conditional rescue combos：`0,6/2,0,7,12/0,6/0,6`
- oracle combos：`0,6/2,0,7,12/0,6/0,6`

### War and Peace

- 结论：easy regime；固定策略、top2 和 conditional rescue 质量相同，主要用于证明 rescue gate 不破坏质量。
- top2 combos：`0,7/0,7`
- conditional rescue combos：`0,7/0,7`
- oracle combos：`0,7/0,7`

### Count of Monte Cristo

- 结论：online blockwise selection 明显优于 best fixed，说明不是离线固定 combo 可替代。
- top2 combos：`2,7/2,0`
- conditional rescue combos：`2,7/2,0`
- oracle combos：`2,7/2,0`

## 对论文创新性的含义

- **不是 SparQ/qabs 固定算子的复述**：主贡献应写成 online policy selection，而不是某个固定 candidate。
- **不是 best fixed combo 可替代**：Monte 上 conditional/online 比 best fixed 改善 `-0.210210` ΔPPL；Hard-topic eval128 上 conditional rescue 比 best fixed 改善 `-0.028889` ΔPPL。
- **rescue gate 有必要性**：Hard-topic eval128 中 top2 与 oracle gap 为 `0.054004`，conditional rescue 将 gap 降到 `0.000000`。
- **动态性证据仍需加强**：War 是 easy regime，固定策略已经足够；后续需要更多非平稳任务证明 blockwise switch 普遍有效。
- **速度 claim 必须保守**：conditional rescue corrected gate 成本仍高，paper 现在只能声称方法学和质量证据，不能声称端到端速度已超过 baseline。

## 下一步最关键实验

1. 在标准或准标准长上下文任务上补 `best fixed / online / oracle` 三者对比。
2. 输出 blockwise trace 图，证明策略切换与文本结构、horizon risk 有对应关系。
3. 做 fused/sparse candidate probe，把 corrected gate 从方法瓶颈变成可控 overhead。

CSV：`docs/pcic_mainline_fixed_online_rescue_2026_06_29.csv`
