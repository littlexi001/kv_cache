# Qwen3-0.6B 全 Head 功能图谱

本项目把已有的 head 级实验与新增因果干预合并成一张可审计的 448-head 图谱。

主要输出：

- `outputs/head_function_atlas.csv`：全部 28×16 个 query heads 的完整字段。
- `docs/all_448_head_cards_20260715.md`：按层列出的逐 head 功能目录。
- `docs/qwen3_0p6b_head_function_atlas_20260715.md`：研究结论、方法、限制与后续实验。
- `docs/head_function_results_summary_20260716_zh.md`：九类功能、冲突证据、Oracle PPL 与外部检索结果的精简总结。
- `outputs/plots/`：功能、置信度和证据关注热图。

图谱区分“保守标签”和“强制主签名”。保守标签只有在跨 head 显著性规则通过时才给出单一功能；强制主签名为每个 head 取九类受控探针中相对最强的一类，用于完整枚举，但不能被解释为该 head 唯一、永久的内在功能。

重新生成：

```bash
python src/build_head_function_atlas.py
```
