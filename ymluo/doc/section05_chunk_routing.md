# Section 5 — Chunk Routing 训练

> 新增日期：2026-05-14；最近同步日期：2026-05-23

## 项目：`qwen3_chunk_routing`

新增日期：2026-05-14

这个项目比较 Qwen3-0.6B 的三种 attention 模式：

- `baseline`：原始 full attention。
- `oracle`：先计算完整 attention score，再把有效历史 token 分成 20 个 chunks；保留 chunk 1、recent chunk，以及 attention mass 最高的 3 个中间 chunks。
- `router`：用轻量 learned router 从 chunk summaries 预测 top 3 中间 chunks，再做精确 attention。

运行示例：

```bash
bash ymluo/projects/qwen3_chunk_routing/scripts/run_8gpu.sh baseline
bash ymluo/projects/qwen3_chunk_routing/scripts/run_8gpu.sh oracle
bash ymluo/projects/qwen3_chunk_routing/scripts/run_8gpu.sh router
```

脚本默认使用 `torchrun --nproc_per_node=8`。它从 `MODEL_PATH` 读取 tokenizer 和 config；除非修改项目代码，否则模型权重从头初始化。
