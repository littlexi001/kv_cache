# QKSieve FP16 proxy top-k：32K 质量与速度结论

## 实验设置

- 模型：Qwen3-4B-Instruct。
- 历史长度：32K。
- 数据：sports、medicine，各两个独立窗口。
- 每个窗口评估 128 个 target token，共 512 token。
- 稀疏预算：每个 query head 选 1,280 个 token，约为 4%。
- 两个方法使用完全相同的 QK-balanced mixed-bit 索引、bit allocation、候选预算和精确稀疏 attention。
- 唯一区别：`torch.topk` 接收 FP32 proxy score，或先把 proxy score 转成 FP16。

## 汇总结果

| 指标 | FP32 proxy | FP16 proxy |
|---|---:|---:|
| Full PPL | 17.80095 | 17.80095 |
| Sparse PPL | 17.76585 | 17.76847 |
| 相对 Full 质量 | 100.198% | 100.183% |
| 与 Full 的 top-1 agreement | 94.531% | 94.141% |
| KL(Full $\Vert$ Sparse) | 0.009516 | 0.009506 |
| Margin certificate 覆盖率 | 40.625% | 40.625% |
| Full 稳态时间 | 88.594 ms/token | 88.903 ms/token |
| Sparse 稳态时间 | 57.003 ms/token | 57.640 ms/token |
| 稳态加速 | 1.554x | 1.542x |
| 含本窗口建索引的 online 加速 | 1.144x | 1.134x |

## 结论

FP16 proxy score 对这组 32K PPL 质量几乎没有可测损失；这一点与合成 kernel probe 中 99.94%--99.98% 的候选集合重合率一致。

但当前真实模型实现先产生 FP32 score，再单独执行 FP32 到 FP16 的转换，转换和额外 kernel launch 抵消了更快 top-k 的收益。结果是 FP16 路径反而比 FP32 慢约 1.1%。

因此：

1. 论文主方法暂时冻结为 FP32 proxy top-k。
2. FP16 可以作为数值消融，证明 top-k 排序不需要 FP32 精度。
3. 只有在 packed score scan 直接输出 FP16、并与 top-k 融合后，才应重新测试其系统收益。
4. 不能把单层合成 probe 的 8%--21% 收益写成当前整模型收益。

原始结果位于：

`results/20260728_qksieve_fp16_topk_quality_32k/`
