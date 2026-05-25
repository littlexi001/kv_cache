# Qwen3 KV Cache 研究工作区

> 最近同步日期：2026-05-23

核心问题：**能否把长上下文 KV cache 看成一个可索引、可压缩、可检索的记忆系统，而不是每次 decode 都密集扫描的扁平 token 序列。**

完整动机见 `KVCache_Indexing_Knowledge_Retrieval_2026-05-09.md`。

## 文档导航

详细内容已拆分到 `doc/` 目录，按推荐阅读顺序组织：

| Section | 文件 | 内容 |
| --- | --- | --- |
| 1 | [section01_research_overview](doc/section01_research_overview.md) | 研究总览：目录结构、推荐阅读顺序 |
| 2 | [section02_kcache_cosine_analysis](doc/section02_kcache_cosine_analysis.md) | K-cache Cosine 分析、压缩含义、下一步建议 |
| 3 | [section03_kcache_norm_analysis](doc/section03_kcache_norm_analysis.md) | K-cache Norm / Attention Energy pruning 分析 |
| 4 | [section04_sparse_decode](doc/section04_sparse_decode.md) | 推理期 Sparse Decode（block top-k 选择） |
| 5 | [section05_chunk_routing](doc/section05_chunk_routing.md) | Chunk Routing 训练（baseline / oracle / router） |
| 6 | [section06_unet_synthetic_retrieval](doc/section06_unet_synthetic_retrieval.md) | U-Net Synthetic Retrieval 评估与训练 |
| 7 | [section07_interval_subseq_retrieval](doc/section07_interval_subseq_retrieval.md) | Interval Subsequence Retrieval 训练 |
| 8 | [section08_pyramid_kv_compression](doc/section08_pyramid_kv_compression.md) | Pyramid KV Compression 继续预训练 |
| 9 | [section09_kcache_value_delta](doc/section09_kcache_value_delta.md) | K-cache Value / Delta 分析 |
| 10 | [section10_moe_selectivity](doc/section10_moe_selectivity.md) | MoE Selectivity 实验（Exp1~Exp4） |
| 11 | [section11_common_params](doc/section11_common_params.md) | 常用参数与注意事项 |

## 项目目录速查

| 路径 | 作用 |
| --- | --- |
| `projects/qwen3_kcache_cosine_heatmap` | K-cache cosine 热力图分析 |
| `projects/qwen3_kcache_norm_analysis` | K-cache norm / attention energy 分析 |
| `projects/qwen3_kcache_avg_topk` | 推理期 block-selection sparse decode |
| `projects/qwen3_chunk_routing` | Chunk attention 训练（baseline / oracle / router） |
| `projects/qwen3_unet_synthetic_retrieval` | U-Net synthetic retrieval 评估与训练 |
| `projects/qwen3_interval_subseq_retrieval` | Interval subsequence next-token 训练 |
| `projects/pyramid_kv_compression` | Pyramid KV 压缩继续预训练 |
| `projects/qwen3_kcache_value_delta_analysis` | K-cache 取值与 delta 分布分析 |
| `projects/qwen3_moe_attention_cluster` | MoE attention cluster 专家选择性实验 |
| `logs/` | 历史日志 |
| `utils/` | 共享工具（含 `moe_selectivity_experiment.py`） |

## 关键结论速览

- **K-cache 压缩**：早期层（L0-L5）cosine 高，可激进压缩；L18-L26 需保守。务必先做 mean-centering 分解公共方向再压缩。
- **Attention energy pruning**：95% energy 几乎无损（PPL +0.2%），90% 已接近 full attention。
- **MoE selectivity — attention cluster**：用 attention 权重引导 router 做 token 聚类，L2 same_higher=0.9997，expert 负载均衡。优于 negative gradient 和 forced warmup 方案。加入 pre-router 反而导致 expert 坍缩。
