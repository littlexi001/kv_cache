# Section 4 — 推理期 Sparse Decode

> 新增日期：2026-05-14；最近同步日期：2026-05-23

## 项目：`qwen3_kcache_avg_topk`

新增日期：2026-05-14

这是一个推理期 sparse decode 实验。Layer 0-2 走原始 attention；Layer 3-27 把当前 K cache 切成 blocks，对每个 block 内部的 keys 求平均，用当前 query 打分，保留 top block fraction，然后只在被选中的原始 K/V token 上做精确 attention。

生成文本：

```bash
MODEL_PATH=/mnt/workspace/lym_code/models/Qwen3-0.6B \
bash ymluo/projects/qwen3_kcache_avg_topk/scripts/run_generate.sh
```

评估 baseline vs sparse decode：

```bash
MODEL_PATH=/mnt/workspace/lym_code/models/Qwen3-0.6B \
DATA_PATH=/mnt/workspace/dclm/global-shard_01_of_10/local-shard_0_of_10 \
bash ymluo/projects/qwen3_kcache_avg_topk/scripts/run_eval.sh
```

默认 sparse 设置：

```text
BLOCK_SIZE=10
TOPK_RATIO=0.30
FIRST_SPARSE_LAYER=3
LAST_SPARSE_LAYER=27
```
