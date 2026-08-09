# 1B Token Context 高效搜索的本质属性探索

> **一句话结论：** 真实10M QK搜索可在head轴无损稀疏化并实测加速4.06倍；查询原型粗召回加exact精排只读25% blocks时取得46.04%最终召回，保留完整selected-head基线49.17%的93.64%，但代理评分仍线性扫blocks且BM25更强，所以性质有效，1B次线性系统尚未完成。

**日期：** 2026-07-14  
**模型：** Qwen3-0.6B真实Q/K  
**数据：** 两份独立10M-token语料，各39,062个256-token blocks  
**方法状态：** Research Exploration Agent流程版

## 阅读入口

这项研究按“当前理论、可证伪实验、结果与迭代历史”拆成三份主文档：

1. [当前研究设计](1b_context_search_research_exploration/design.md)：问题定义、可观测先验、数学模型、实现契约、算法步骤、复杂度和结论边界。
2. [可证伪实验设计](1b_context_search_research_exploration/experiment_design.md)：每个实验的输入、参数、指标、通过/失败/证据不足条件，以及下一轮冻结协议。
3. [实验结果与猜想更新](1b_context_search_research_exploration/visualization_results.md)：五张证据图、实际结果、逐阶段失败分析、profiling、证据索引和完整迭代账本。

## 当前判定

| 研究问题 | 判定 | 关键结果 |
|---|---|---|
| 单K-mean能否替代RAG | 失败 | test最佳gold evidence Recall@16仅0.4% |
| block residual K是否中低秩 | 支持 | rank90约14.0到20.7，rank16保留83.6%到93.1%能量 |
| FPS-16能否近似exact max-QK | 失败 | test Top1 agreement为14.2%/22.0% |
| 真实record是否具有K局部连续性 | 支持 | 八个LongBench任务相邻优势为0.238到0.515，打乱对照约0 |
| 两级树能否减少centroid近邻点积 | 有tradeoff | 7.79x估算点积减少对应72.78%邻居召回；16.44x只保留52.86% |
| 不同heads是否只是重复检索 | 否 | 同层非GQA-sibling Top16 Jaccard仅0.77%；全head并集从386增长到4,300 blocks |
| 专业head是否可跨问题复用 | 支持 | 480题分层切分中，train选择16 heads的test micro/macro recall为53.4%/48.0%，随机为21.2%/21.5% |
| 全head投票是否受公共吸引子干扰 | 支持 | 81个blocks被全部480题提名，只有一个在一题上是gold；解释多数投票失败 |
| Query-invariant prior能否分离 | 支持 | prior平均解释49.8%分数方差；五折z-score令universal hubs从81降到0，RRF39从22.71%升至38.13% |
| 能否无gold选择少量heads | 压力测试支持 | 整数据集留出下，train Top1-block多样性选16 heads，候选并集召回62.50%、RRF39为49.17%；仍缺完全新queries/新dataset |
| Head稀疏性是否转化为真实加速 | 支持 | 六折20-head并集单卡182.72秒，完整448-head单卡741.77秒，同召回下4.06x加速 |
| Head稀疏性是否同时压缩K存储 | 支持 | 17个实际`layer x KV-head`通道无损打包后为10.88 GB，占全profile 7.59% |
| 严格center-radius界能否剪枝 | 失败 | 16段仍平均保留99.9985% blocks，估算点积速度0.938x |
| 查询方向是否存在可学习结构 | 支持raw、z-score较弱 | 完整block轴上10.49%候选保持95.12% raw Top16；z-score在25%候选保持90.50% |
| exact prior能否修复z-score代理 | 否 | 25%候选反而从90.50%降到89.09%，说明标准化放大了方向代理残差 |
| 原型粗召回加exact精排能否保持最终召回 | 部分支持 | 25% blocks为46.04%，完整selected-head为49.17%；保留93.64%，损失3.13pp |
| 当前内部检索是否超过matched RAG | 否 | 同一480题BM25-block/record39为66.67%/81.04%，显著高于49.79% |
| 是否提高最终attention或QA | 未验证 | 下一步必须做RAG only与RAG seed + KV expansion配对实验 |
| 是否实现真实1B多卡加速 | 部分验证 | 10M selected-head真实单卡加速4.06x且packed profile已构建；尚无无争用packed多卡、100M/1B结果 |

## 当前主线

```text
第一次全局seed：BM25 / E5 / metadata
-> seed后的内部扩展：query-prototype概率路由或pre-RoPE层次索引
-> 少量候选：selected-head raw K exact max
-> 恢复真实位置：post-RoPE exact QK
-> 加载少量KV并继续生成
```

下一项决定性实验不是继续提高centroid代理指标，而是在同一批多步问题、相同RAG seed和相同读取预算下，比较：

```text
RAG only
vs.
RAG seed + residual-K hierarchy + post-RoPE exact rerank
```

必须同时报告第二证据召回、attention top-token recall、最终答案、读取tokens、分阶段wall-clock和跨卡通信。只有这些下游指标形成配对优势，当前发现的KV属性才构成完整系统贡献。

全head组合稀疏性的64题探索和480题扩展验证见 [10M Context 的 Head-Specific 稀疏分解性质](head_specific_sparse_decomposition_10m_20260714.md)。后续五折prior去偏、无标签head gate、整数据集留出与matched BM25结果见 [10M Context 中的 Query-Invariant Prior 与无标签 Head Gate](query_invariant_prior_and_unsupervised_head_gate_10m_20260714.md)。当前无gold gate已通过整数据集留出压力测试，但尚未在完全新queries/新dataset上做确认性验证。
