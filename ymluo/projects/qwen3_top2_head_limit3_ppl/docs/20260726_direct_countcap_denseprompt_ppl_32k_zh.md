# Direct CountCap 32K PPL 实验

## 测试目的

测试不做精确候选重排、直接消费 PCA48 INT4 检索候选时，长度预算策略的 PPL：

`target(N) = min(N, max(256, min(ceil(0.06 * N), 1280)))`

在本实验的 32K 历史上，名义目标为每个 query head 1280 个历史 token，约占 4%。

## 实验设置

- 模型：Llama-3.1-8B-Instruct
- 数据：sports、medicine 两个 targeted 主题
- 每个主题：3 个互不重叠窗口
- 每个窗口：32000 个历史 token，评测后续 256 个 token
- 每种方法：6 个窗口、1536 个评测 token
- PCA：48 维
- 索引量化：INT4
- 分位数采样：256 个位置
- Direct 路径：`pca_int4_chunked_logscale16_sampleq_direct_qkvfused_qprojscan_qkvsplitauto`
- 不使用任务标签、router、oracle 或 Full 回退

测试采用 Dense-Prompt 协议：完整历史先做 dense prefill，第一个目标 token 由 dense 状态预测；仅后续 255 个目标位置启用对应的稀疏注意力。三种方法使用相同文本和窗口。

## PPL 结果

PPL 越低越好。表中的变化均相对 Full。

| 数据 | Full | 精确 top-2% | Direct CountCap | Direct 变化 |
|---|---:|---:|---:|---:|
| sports | 8.0669 | 8.3193 | 8.2254 | +1.96% |
| medicine | 8.7323 | 8.8576 | 8.7971 | +0.74% |
| 全部 token | 8.3930 | 8.5843 | 8.5064 | **+1.35%** |

总体平均 NLL：

| 方法 | NLL | 相对 Full 的 NLL 差 |
|---|---:|---:|
| Full | 2.127402 | 0 |
| 精确 top-2% | 2.149931 | +0.022529 |
| Direct CountCap | 2.140822 | **+0.013420** |

Direct CountCap 比精确 top-2% 更好并不矛盾：Direct 的目标比例为4%，候选buffer最多约8%，而精确方法只保留2%。该对照说明把候选直接用于attention在当前预算下具有较好的PPL，而不是说明PCA近似分数比真实QK分数更准确。

## 速度

下面统计完整模型的逐 token forward，Direct 时间包含 PCA48 INT4 查询投影、全局索引扫描、候选产生和稀疏 QK/AV。

| 方法 | decode 时间/token | 相对 Full |
|---|---:|---:|
| Full | 99.31 ms | 1.00x |
| 精确 top-2% | 93.60 ms | 1.06x |
| Direct CountCap | 46.35 ms | **2.14x** |

若把每个窗口约 17 秒的 32K dense prefill 也计入，并只生成 256 个目标 token，则协议总时间加速为 **1.47x**。生成长度增加后，固定 prefill 成本被进一步摊薄，加速会趋近 decode-only 的 2.14x。

精确 top-2% 只用于质量参照。它必须先扫描完整 QK 才知道真实 top-2%，因此不是一个高效的在线实现。

## 候选数量

采样分位数控制的是期望候选比例，不是严格 top-k。下面记录的是分位数阈值的原始命中数：

| 指标 | token/head |
|---|---:|
| 名义目标 | 1280 |
| 候选buffer容量 | 约2560 |
| 原始阈值命中均值 | 1356.87 |
| 原始阈值命中最小值 | 187 |
| 原始阈值命中最大值 | 4019 |

最终attention消费数为 `min(原始命中数, buffer容量)`，所以4019会被截到约2560。现有日志没有保存截断后每个head的平均值。

因此当前结果不能表述为“每个head严格不超过1280”，也不应把1356.87直接称为实际attention均值。若论文方法要求1280硬预算，需要在候选消费前加入无偏压缩或proxy-score top-k，并重新测量质量与速度。

## 结论

在体育和医学这两个此前较难的 PPL 主题上，Direct CountCap 将总体 PPL 增幅控制在 1.35%，完整 decode 达到 2.14x。这个结果支持取消精确 QK 重排，但当前执行预算上界约为2560而不是1280。

当前实验仍只是 2 个主题、6 个窗口的 targeted 验证，不能替代更多主题、更多长度和生成 benchmark。下一步最重要的不是继续调 PPL 参数，而是决定论文中的预算定义：保留统计意义上的 CountCap，或实现严格的 per-head hard cap 后重新验证。

## 文件

- 参考组与首次 Direct 结果：`results/20260725_direct_countcap_denseprompt_ppl_32k_4gpu`
- 带真实候选统计的 Direct 结果：`results/20260726_direct_countcap_denseprompt_ppl_32k_actual_counts`
- 测试程序：`src/run_direct_countcap_denseprompt_ppl_20260725.py`
- 四卡启动脚本：`scripts/launch_direct_countcap_denseprompt_ppl_32k_4gpu_20260725.sh`
