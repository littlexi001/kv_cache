# QKSieve ICLR 2027 论文草稿

## 当前方法决策（2026-08-03）

- 论文主方法恢复并冻结为 **QKSieve-General**：不超过 64K 时使用 Fast
  路径，超过 64K 时使用带 rank-16 INT4 Value-tail correction 的 Robust
  路径；不使用 Full Attention 回退。
- JointKV 当前真实 32K 连续 decode 未达到质量门槛，不替换主方法；其融合
  Query/LUT/scan CUDA 实现只作为可复用的系统探索保留。
- 当前最大的部署局限是 QKSieve 仍保留完整 GPU FP16 K/V，5.859%（Fast）
  或约 7.47%（Robust）是额外辅助索引，而不是剩余 KV 占比。因此当前主张
  是稀疏 attention/decode 加速，不能主张 KV 存储压缩。
- 当前最大的结构性局限是 General 仍由 64K 人工长度边界拼接 Fast 与 Robust；
  投稿前需要用统一的数值风险量替代硬阈值，或把该边界明确限定为注册部署策略，
  不宣称它是跨长度的统一最优定律。

## 文件

- `main.tex`：匿名投稿版入口。
- `main_author.tex`：作者版入口，署名 Yiming Luo / Fudan University。
- `sections/`：正文和附录。
- `references.bib`：参考文献。
- `figures/`：论文图。
- `scripts/make_method_figure.py`：重绘方法总览图。
- `EXPERIMENTS_TO_FILL_CN.md`：投稿前需要补齐的实验清单。
- `build.ps1`：生成匿名版、作者版 PDF，并复制到仓库 `output/pdf/`。
- `sections/04_analysis.tex`：9 页正文预算内的核心理论链。
- `sections/appendix_analysis_statements.tex` 与 `sections/appendix.tex`：完整命题、边界和证明。
- `../../projects/qwen3_top2_head_limit3_ppl/docs/20260728_qksieve_theory_complete_zh.md`：与冻结实现逐项对齐的中文完整证明和适用边界。
- `../../projects/qwen3_top2_head_limit3_ppl/docs/20260728_qksieve_qfused_integration_zh.md`：Query 融合候选路径、GPU 验收阈值和升级规则。
- `../../projects/qwen3_top2_head_limit3_ppl/docs/20260728_qksieve_public_baseline_protocol_zh.md`：Quest、SparQ selector/formula reference 与 RetrievalAttention 的公平比较边界和复现协议。
- `../../projects/qwen3_top2_head_limit3_ppl/docs/20260728_qksieve_retroinfer_official_protocol_zh.md`：RetrievalAttention/RetroInfer 身份审计、固定提交与异构系统公平复现协议。
- `../../projects/qwen3_top2_head_limit3_ppl/src/run_retroinfer_aligned_longbench_20260728.py`：保持官方 RetroInfer 后端不变、只对齐 LongBench 外壳的严格配对 runner。

## 模板说明

截至 2026-07-26，ICLR 2027 官方模板尚未发布。当前
`iclr2027_conference.sty/.bst` 由官方 ICLR 2026 模板复制并只修改页眉年份，
用于提前排版，不应宣称为官方 ICLR 2027 模板。正式投稿前必须：

1. 下载 ICLR 2027 官方模板；
2. 替换本目录的 `.sty/.bst`；
3. 重新核对页数、匿名规则和 checklist；
4. 匿名投稿只编译 `main.tex`，不要提交作者版。

## 编译

在本目录运行：

```powershell
.\build.ps1
```

当前作者版共 32 页，但正文和结论在第 9 页内结束；第 9 页下半部开始参考文献，后续为参考文献与附录。提交前仍需根据正式 ICLR 2027 模板重新核验页数。

脚本使用 Codex 环境附带的 Tectonic。输出：

- `output/pdf/QKSieve_ICLR2027_draft_anonymous.pdf`
- `output/pdf/QKSieve_ICLR2027_draft_author.pdf`

## 论文口径

冻结主方法不包含：

- Full Attention 回退；
- exact-QK 候选重排；
- 训练式 router；
- 任务标签；
- sink/recent 特判；
- RAG/BM25；
- temporal reuse。

当前实现保留完整 GPU FP16 K/V。QKSieve 的 240-bit mixed-bit 索引是
额外检索索引，占完整 FP16 K+V 的 5.859%，不能写成“GPU KV 只剩
5.859%”。正式主结论还必须通过同路径证据校验器，不能拼接旧
sampled-threshold 路径与新 full-topk 路径的质量和速度。
