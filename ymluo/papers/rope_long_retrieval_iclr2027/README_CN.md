# SAGE-RoPE ICLR 2027 论文草稿

这是一套可直接编译的 ICLR 2027 官方 LaTeX 工程。当前正文严格按照：

```text
问题与价值
→ 第一层 RoPE 相位分析
→ Softmax 与跨层状态分叉
→ 由同一分析推出方法
→ held-out 实验与适用边界
```

## 本地编译

```powershell
cd ymluo\papers\rope_long_retrieval_iclr2027
powershell -ExecutionPolicy Bypass -File .\build.ps1
```

本机已经有轻量 Tectonic，不需要安装完整 TeX Live。第一次构建需要缓存
LaTeX 宏包，之后通常明显更快。输出文件为：

- `output/pdf/SAGE_RoPE_ICLR2027_draft_anonymous.pdf`
- `output/pdf/SAGE_RoPE_ICLR2027_draft_author.pdf`
- `output/pdf/SAGE_RoPE_ICLR2027_draft_zh.pdf`（中文阅读版）

如果只修改和编译中文版，可以运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\build_zh.ps1
```

中文版入口为 `main_zh.tex`，各章节位于 `sections_zh/`。它与英文稿共享
公式、图表、实验数据、引用和 TODO 边界，仅用于阅读与讨论，不作为正式
投稿文件。三张核心图也提供了中文标签。英文内容发生实质修改时，应同步
更新对应的中文章节；构建脚本会先运行 `scripts/check_bilingual_sync.py`，
检查公式、标签、引用、实验数字、环境结构和 TODO 数量是否发生漂移。

## 当前完成度

已经写入并有实验产物支持的内容：

- 第一层 64 个 RoPE 二维频率对的精确推导；
- 固定 pre-RoPE Q/K、只改变距离的数值重构；
- 从 attention/MLP/residual 到最终 margin 的逐层有限差分；
- activation patching 因果案例；
- 24 个 held-out seeds、8K/16K/32K/64K 的 SAGE-Post 主结果；
- 对 Full Attention 和 exact post-RoPE Top-2% 的配对 bootstrap 比较。

目前用红色 `TODO` 明确标出的内容尚不能写成已经完成：真实长上下文
benchmark、跨模型验证、本地语序/短文本 PPL、实际近似索引和端到端效率、
以及更大样本的 64K 统计验证。完整优先级见 `TODO_EXPERIMENTS.md`。

## 协作原则

1. 不要直接修改 `data/*.csv` 中的数字；应从上游实验重新生成并记录来源。
2. 主文中的强结论必须能映射到具体公式、表格或 artifact。
3. 不把 143K 的位置扩展诊断写成模型原生窗口结果。
4. 不写“距离越远 QK 单调下降”；相位模型和实验都是振荡/非单调的。
5. 不在完整 pre-RoPE 扫描仍存在时宣称系统加速。
