# 研究迭代记录

## 2026-08-07：F0–F7 固定隐藏状态反事实

**猜想：** 高频相位压低长程 gold evidence，因此把 F0–F7 改成 NoPE 会提高证据读取。

**操作化：** 同一次原生前向中保持隐藏状态不变，逐 layer、逐 query head 把 F0–F7 的相对相位替换为零，重算全部历史 token 的 logits、softmax mass 和 rank。

**结果：** 5 条可定位 gold token 的 RULER-32K 样本中，深层 gold logit 平均提高 0.336，但 gold mass 平均下降 0.000401，最佳 gold rank 平均变差 82.38。A1 的 `cos≥0.9` 比例为零；F0–F7 Query 能量仅 3.61%。

**失败分解：** “gold 绝对 logit 被抑制”成立；“背景不受同样影响”不成立。取消相位也提高大量非 gold token，所以 softmax 相对竞争没有同步改善。

**猜想更新：** 不再把全局高频删除当作候选方法。保留“只在特定层/head/远程距离进行连续修正”的较弱猜想，并用端到端对照检验。

**下一不确定性：** 距离条件停止 F0–F7 相位是否能在不破坏局部位置的情况下改善 RULER score 和 gold NLL。
