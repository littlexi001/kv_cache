# QKSieve ICLR 2027 论文草稿

## 当前方法决策（2026-08-10）

- 论文主方法冻结为 **QKSieve-Robust**，在注册的 4K--128K 范围内始终使用
  request-local QK-balanced + qMSE/OAS mixed-bit selector、sampled-quantile
  单遍候选写出、原始 FP16 K/V 精确稀疏 attention，以及
  rank-16/block-256/INT4 ValueSketch（`alpha=0.5`）。
- 主方法没有长度切换、学习式 router、任务规则、exact-QK 重排或 Full
  Attention 回退。Fast 仅保留为“去掉 ValueSketch”的受控消融。
- JointKV 当前真实 32K 连续 decode 未达到质量门槛，不替换主方法；其融合
  Query/LUT/scan CUDA 实现只作为可复用的系统探索保留。
- 当前最大的部署局限是 QKSieve 仍保留完整 GPU FP16 K/V，5.859%（Fast）
  或约 7.47%（Robust）是额外辅助索引，而不是剩余 KV 占比。因此当前主张
  是稀疏 attention/decode 加速，不能主张 KV 存储压缩。
- 当前注册范围是 4K--128K；256K/512K 仅作为失效机理与外推边界研究，不把
  未经正式验证的超长上下文结果写成主方法保证。

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

## 模板与投稿规则

`iclr2027_conference.sty/.bst` 已在 2026-08-10 与 ICLR 2027 官方发布包进行
逐字节核验。官方当前要求：

1. 摘要截止时间为 2026-09-18 AOE，全文截止时间为 2026-09-25 AOE；
2. 初投稿正文最多 9 页，参考文献、AI 使用声明、可复现性声明和附录不计入；
3. 投稿必须双盲，匿名版只编译 `main.tex`，不要提交作者版；
4. 所有作者须在摘要截止前确认作者集合及 OpenReview 账号；
5. 正式提交前再次以官方 Author Guidelines 为准，避免缓存页面或旧通知。

## 编译

在本目录运行：

```powershell
.\build.ps1
```

正文和结论必须在第 9 页内结束；AI 使用声明、可复现性声明、参考文献与附录
随后排列。每次补实验后都必须重新编译并逐页核验正文页数与匿名性。

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
