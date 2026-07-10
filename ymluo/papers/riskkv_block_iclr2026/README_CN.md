# RiskKV-Block ICLR 论文草稿说明

## 文件

- `main.tex`：ICLR 2026 LaTeX 主文，已按终稿语气写好，包含方法、公式、架构图、当前实验结果、消融和待补实验占位。
- `references.bib`：引用文献。
- `main.pdf`：已编译生成的 PDF。
- `EXPERIMENTS_TO_FILL_CN.md`：正式投稿前需要补齐的实验清单。

## 编译

当前目录已经包含官方 ICLR 2026 模板文件和本地 Tectonic 编译器。Windows PowerShell 下运行：

```powershell
cd C:\Users\27814\Desktop\work\codex_workspace\kvcache\kv_cache-main\kv_cache-main\ymluo\papers\riskkv_block_iclr2026
..\..\tools\tectonic\tectonic.exe main.tex --keep-logs --keep-intermediates
```

## 投稿模式

当前版本按你的要求保留署名：

- Yiming Luo
- Fudan University
- yimingluo@fudan.edu.cn

注意：ICLR 首轮投稿通常是双盲评审。正式提交 OpenReview 前，建议把 `main.tex` 中的 `\iclrfinalcopy` 注释掉，并确认 PDF 页眉变成 under-review、作者变成 anonymous。录用后 camera-ready 再打开 `\iclrfinalcopy` 和署名。

## 当前稿件定位

论文主线是：RiskKV-Block 把 KV cache 压缩从“固定 token budget”推进到“风险约束的记忆粒度路由”。核心动作是联合选择 block size、top-k evidence budget 和 fallback，并能以 prompt-level selected spans 或 cache-native RoPE-aware KV repack 两种方式执行。

当前 PDF 已通过编译和渲染检查：没有表格越界、链接彩框或明显页面重叠。剩余的 LaTeX warning 是 underfull vbox，属于页面纵向伸缩，不影响版面。
