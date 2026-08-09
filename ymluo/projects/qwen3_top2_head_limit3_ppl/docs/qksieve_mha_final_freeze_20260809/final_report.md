# QKSieve-General：MHA 最终速度优化与冻结

## 冻结结论

从 2026-08-09 起冻结 QKSieve-General 的数值主线和原生 MHA Attention 子系统实现，不再继续修改方法结构。后续工作转为论文写作、补表和在 H100 上复测，不再根据单个速度点调整算法。

本轮唯一保留的优化是给 sampled-quantile 的统计样本数增加统一上限：

`S(N) = min(N, 512, max(256, ceil(16 / r(N))))`

其中 `r(N)=min(0.06,1280/N)`。短序列原样使用 267/410 个样本，64K 和 128K 从 800/1600 截断到 512。它不是长度路由，也不改变索引、proxy score、active-token 预算或精确 Attention 公式。运行主路径通过 `QKSIEVE_MAX_QUANTILE_SAMPLE_COUNT=512` 启用。

精确稀疏 Attention 固定为 `split=8`。实验性的 `split=16` 虽然曾把 128K 指示性速度推到 7.05x，但输出与 split=8 不等价，128K 最差余弦约 0.80，因此明确废弃。

## 冻结方法

1. 每层、每个 KV head 使用 request-local QK-balanced 坐标。
2. 用 mixed-band 低位 Key 索引近似 Query-Key 分数。
3. 通过上式的 sampled quantile 得到阈值，单遍扫描历史索引并压缩候选。
4. active-token 预算为 `min(1280,max(256,ceil(0.06N)))`。
5. 从 GPU 常驻的原始 FP16/BF16 K/V 中读取候选，执行 exact QK、softmax 和 AV。
6. 不使用任务 router、Full fallback 或 exact-QK 全历史重排。

MHA 速度表不包含 ValueSketch。原因是当前 ValueSketch 快路径为 GQA4 专用；因此本表的严格名称是“QKSieve selector + exact sparse Attention 子系统”，不能写成带 ValueSketch 的整模型 decode。

## 最终 MHA 速度

设置为 RTX 3090、`32Q/32KV/D128`、batch 1、resident FP16 K/V。Full 直接使用原生 MHA SDPA，不复制 K/V。QKSieve 时间包含 query 投影与量化、低位索引扫描、候选压缩以及 exact sparse Attention。每个点汇总 3 个 seed 和两组算法等价的同卡运行，共 6 次中位数。

| 历史长度 | Full MHA | QKSieve | 加速比 | Query prepare | Selector scan | Sparse attention | 实际 active |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8K | 0.1692 ms | 0.1386 ms | **1.22x** | 0.0336 | 0.0365 | 0.0438 | 6.16% |
| 16K | 0.3186 ms | 0.1946 ms | **1.64x** | 0.0335 | 0.0468 | 0.0910 | 5.98% |
| 32K | 0.6157 ms | 0.2312 ms | **2.66x** | 0.0299 | 0.0634 | 0.1107 | 4.06% |
| 64K | 1.2031 ms | 0.2684 ms | **4.48x** | 0.0302 | 0.0999 | 0.1186 | 1.96% |
| 128K | 2.3745 ms | 0.3810 ms | **6.23x** | 0.0303 | 0.1673 | 0.1630 | 0.98% |

辅助索引占完整 FP16 K/V 的约 5.525%。相对旧表 `1.23x/1.57x/2.49x/4.36x/5.50x`，新正式表在 128K 从 5.50x 提升到 6.23x，约提高 13%。8K 基本不变，说明统计上限没有用长文本收益交换短文本性能。

## 数值验收

- 最终 `split=8` 路径与 split=8 reference 逐点一致，记录的最大绝对误差为 0，余弦为 1。
- 512-sample 不是本轮新引入的质量假设。此前 64K 四主题自然文本验证中，完整 s512 路径的 top-1 一致率为 99.61%，候选溢出为 0。
- 本轮没有重新跑 LongBench/RULER；已有质量结果继续对应 QKSieve-General 数值主线，MHA 表只承担系统速度证据。

## 废弃项

| 候选 | 速度现象 | 数值结果 | 决策 |
|---|---|---|---|
| 固定 512 + split=16 | 128K 指示性 7.05x | 长序列输出不等价 | 废弃 |
| 固定 512 + split=10/15 | 128K 约 7.0x | 128K 仍出现明显误差 | 废弃 |
| 所有长度强制 512 | 64K/128K 更快 | 8K 无谓增加样本并变慢 | 改为统一上限 |
| `min(default,512)` + split=8 | 短文本不退化，长文本扫描更短 | 通过既有 s512 质量证据 | 冻结 |

## 论文口径

允许写：在 RTX 3090 的原生 LLaMA-2-7B 形状 MHA 上，QKSieve 完整 Attention 路径在 8K--128K 获得 1.22x--6.23x 加速，包含 selector 全部开销，索引约为完整 FP16 K/V 的 5.53%。

不允许写：6.23x 是整模型 decode 加速；该表包含 ValueSketch；该表是相对 FIER 官方 kernel 的 6.23x；或者 512-sample 在所有模型与任务上已经得到新的完整质量验证。

## 复现位置

- 冻结配置：`configs/qksieve_general_mha_frozen_20260809.json`
- MHA benchmark：`src/benchmark_qksieve_fier_mha_speed_20260808.py`
- 主运行时样本上限：`src/run_head_top2_targeted_ppl_20260714.py`
- 本地原始结果：`docs/qksieve_mha_final_freeze_20260809/raw_results`
- 远端原始结果：`/home/fdong/qksieve_iclr2027/results/20260809_mha_final_optimization`

## 论文同步

2026-08-09 已将冻结配置同步到 ICLR 2027 中英文稿：

- 主文加入原生 MHA 8K--128K 正式速度表。
- 方法公式将 sampled-quantile 样本数冻结为 `min(N,512,max(256,ceil(16/r(N))))`。
- 明确 `split=8`，并记录 `split=16` 因输出不等价而废弃。
- 将 5.859% 标为 240-bit 硬上限，将 5.525% 标为该 MHA 实验的平均实际索引率。
- 旧 GQA 与整模型数字只保留为附录诊断，不与原生 MHA 主张合并。
- 英文匿名版、英文作者版与中文阅读版均已重新编译并完成页面视觉检查。

论文 PDF 位于仓库 `output/pdf/`。本轮冻结不等于论文实验全部完成；正式 H100/A100、完整同路径跨模型质量和大规模 RULER 仍是投稿前待补项。
