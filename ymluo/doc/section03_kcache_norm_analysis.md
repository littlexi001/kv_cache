# Section 3 — K-cache Norm / Attention Energy 分析

> 新增日期：2026-05-14；最近同步日期：2026-05-23

## 项目：`qwen3_kcache_norm_analysis`

新增日期：2026-05-14

这个项目在 DCLM 文本前缀上运行 Qwen3-0.6B，并输出：

- token-level next-token loss 和 PPL。
- 原始 K-cache norm 的 layer/head summary。
- attention energy 的 layer/query-head summary。
- 达到不同 attention-energy threshold 所需的 top-k token 数。
- pruning 掉低 attention-energy 位置后的 loss/PPL。

运行：

```bash
bash ymluo/projects/qwen3_kcache_norm_analysis/scripts/run_analysis.sh
```

## 当前 3000 tokens summary 实验结论

- 90% attention energy 已接近 full attention：loss 增加 `0.014206`，PPL 约 `1.0143x`。
- 95% attention energy 在该样本上几乎无损：loss 增加 `0.002030`，PPL 约 `1.0020x`。
- 50% 和 75% energy pruning 对质量过于激进。
- 达到同一 energy threshold 所需 token 数在 layer/head 间差异很大，因此固定 token top-k 不如 adaptive energy-based selection 合理。

详细表格和解释见：

```text
projects/qwen3_kcache_norm_analysis/attention_energy_loss_summary.md
```
