# Section 158: partial results and next direction

日期: 2026-07-11

## 已确认的局部结果

截至 13:20 左右, v312/v313/v315 已有部分任务完成。使用 `compare_smoke_to_baselines_20260711.py` 按 `(task, sample_id)` 对齐 full KV 与 v300 后, 结果如下:

| Method | Task | Samples | Score | vs full | vs v300 | KV | Online |
|---|---|---:|---:|---:|---:|---:|---:|
| v312 B16 window-vote quality | musique | 100 | 0.1457 | 57.6% | 57.1% | 41.28% | 0.229s |
| v313 B16 window-vote speed | multifieldqa_en | 100 | 0.4058 | 71.9% | 71.3% | 23.90% | 1.227s |
| v315 B128 BM25-bridge smoke | musique | 20 | 0.1400 | 46.7% | 46.7% | 27.63% | 0.242s |

## 解释

这些结果说明:

1. `block_size=16` 本身并没有在 LongBench QA 上自然带来更强效果。
2. 把 16-token block 扩成 window/span 后, fragmentation 问题被缓解, 但 evidence 定位仍然不准。
3. BM25 + bridge 的组合 scorer 至少没有救回 musique, 因此 LongBench QA 的难点不是单纯 exact lexical matching。
4. v313 的 KV ratio 已经落在目标区间, 但分数明显不够, 所以不能作为 practical best。

## 当前决策

B16/window-vote/BM25-bridge 作为 ablation 和负结果保留, 但不应继续成为主线。

下一步主线转向:

- v300 动态风险路由;
- 输出长度/格式控制, 先用 v316/v317 QA short-decode smoke 验证;
- 如果 short-decode 不能改善, 再训练 learned risk/evidence router, 用 full/v300/各候选方法的 same-sample oracle 标签做监督。

## 当前 practical best

在完整 M100 上仍然是 v300/v311 系列:

| Method | Score | KV | Online |
|---|---:|---:|---:|
| full KV | 0.3658 | 100.00% | 3.0988s |
| v300 main | 0.4392 | 27.41% | 0.5632s |
| v311 safe speedpatch | 0.4334 | 25.74% | 0.5069s |

v311 相对 full KV:

- score: 118.5%;
- KV: 25.74%;
- online speed: 6.11x。

它已经满足用户目标的全局形式, 但如果要写成 ICLR 主线, 还需要补足:

- 更清晰的 dynamic-action story;
- 与固定预算 KV 方法的公平对比口径;
- learned/router 消融;
- 长上下文端到端速度随长度增长的验证。
