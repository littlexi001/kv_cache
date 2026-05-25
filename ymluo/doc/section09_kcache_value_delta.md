# Section 9 — K-cache Value / Delta 分析

> 新增日期：2026-05-14；最近同步日期：2026-05-23

## 项目：`qwen3_kcache_value_delta_analysis`

新增日期：2026-05-14

这个项目 profile 一次 Qwen3-8B forward pass，用 `past_key_values` 构建最终 K cache，并分析：

```text
k_i
delta(k_i) = k_i - k_{i-1}
```

它会输出 per-head、per-layer、global statistics，精确 histograms，timing rows，以及可选 plots。默认运行 5000 个 DCLM tokens。

运行：

```bash
bash ymluo/projects/qwen3_kcache_value_delta_analysis/scripts/run_analysis.sh
```

快速 smoke test：

```bash
MAX_TOKENS=128 CHUNK_SIZE=32 SAVE_HEAD_HISTOGRAMS=false \
bash ymluo/projects/qwen3_kcache_value_delta_analysis/scripts/run_analysis.sh
```
