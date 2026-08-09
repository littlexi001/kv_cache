# 8K value-mediated RoPE smoke

这是 Qwen3-8B、8K、seed 0 的单样本 smoke，只验证机制 probe 的基本链路；不是论文结果。

## 基线

- untouched native：Gold PPL 5.487，gold-vs-conflict margin -1.50。
- instrumented baseline：Gold PPL 6.762，margin -1.75。
- instrumented/native 有 0.25 margin 漂移，因此后续干预只和 instrumented baseline 配对比较。

## 当前 joint intervention 的局限

当前每个类别同时干预每层每个 head 的一个 token，共 36x32=1152 个 score 位置。所有类别的实际 margin 都改善了 0.125--0.500；同类别 random control 也常常不弱于 target。四类 target 的一阶预测与实际变化 Pearson 为 -0.055，平均绝对 closure error 为 0.316。

这不能否定精确局部导数

\[
\frac{\partial m}{\partial s_j}
=a_j\,g^\top(v_j-o),
\]

因为该公式只描述 baseline 附近的单个微小 score 扰动；一次联动 1152 个位置会改变后续层 Query、softmax 和 MLP，已经远离一阶局部区间。当前结果只能说明：**大规模 joint replay 不是合格的因果闭环测试，也不能证明 suppression x value sensitivity 优于随机。**

下一版将冻结 baseline 候选后，逐个做 singleton `+epsilon` 干预，并配同层、同 head、同类别的随机位置；再报告 predicted-vs-actual correlation、sign 和 closure error。

