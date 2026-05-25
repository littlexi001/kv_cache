# Section 6 — U-Net Synthetic Retrieval

> 新增日期：2026-05-14；最近同步日期：2026-05-23

## 项目：`qwen3_unet_synthetic_retrieval`

新增日期：2026-05-14；最近同步日期：2026-05-15

这个项目把 `fdong/unet_transformer.md` 第 7 节的可控 synthetic retrieval 任务落到可运行评估和训练脚本上，用来检查 mask-based U-Net Transformer checkpoints 在只保留 anchor KV 的情况下是否还能找回答案。

当前任务直接生成 token-id 序列，避免 tokenizer segmentation 影响可控实验：

- Variant A：固定 4-token patterns。
- Variant B：随机 3-token content blocks，后接共享 anchor marker。

评估会对每个 checkpoint 和任务变体报告 answer-only loss/accuracy，并比较三条路径：

- full-sequence forward。
- teacher-forced decode with full KV cache。
- teacher-forced decode with anchor-only KV cache。

默认 checkpoint 覆盖 `baseline`、`unet-4`、`unet-4-8-4`、`unet-4-8-16-8-4`。如果 checkpoint 目录里有 `runtime_config.json`，评估脚本会直接读取；否则回退到这些 run name 的已知 stride schedule。

运行评估：

```bash
bash ymluo/projects/qwen3_unet_synthetic_retrieval/scripts/run_synthetic_eval.sh
```

快速 smoke test：

```bash
NUM_SAMPLES=8 BATCH_SIZE=1 \
bash ymluo/projects/qwen3_unet_synthetic_retrieval/scripts/run_synthetic_eval.sh
```

最近新增的训练入口支持 answer-only loss，默认训练 `unet-4` schedule 的 Variant B，并使用 `TRAIN_MODE=anchor_kv_decode`，也就是训练路径和评估里的 anchor-only KV pruning 路径一致：

```bash
bash ymluo/projects/qwen3_unet_synthetic_retrieval/scripts/run_train_synthetic.sh
```

后台训练：

```bash
bash ymluo/projects/qwen3_unet_synthetic_retrieval/scripts/nohup_train_synthetic.sh
```

这个训练只对最终 answer prediction 施加 cross-entropy，不对前 1023 个 next-token 位置施加 loss。常用覆盖参数：

```bash
MODEL_NAME=unet-4-8-4 \
VARIANT=A \
RUN_NAME=unet-4-8-4-variant-a-answer-only \
TOTAL_STEPS=20000 \
BATCH_SIZE=8 \
TRAIN_MODE=anchor_kv_decode \
bash ymluo/projects/qwen3_unet_synthetic_retrieval/scripts/run_train_synthetic.sh
```

主要输出：

```text
ymluo/projects/qwen3_unet_synthetic_retrieval/outputs/synthetic_eval/metrics.json
ymluo/projects/qwen3_unet_synthetic_retrieval/outputs/synthetic_eval/metrics.csv
ymluo/projects/qwen3_unet_synthetic_retrieval/outputs/train/<run_name>/metrics.jsonl
ymluo/projects/qwen3_unet_synthetic_retrieval/outputs/train/<run_name>/checkpoints/<step>.pth
ymluo/projects/qwen3_unet_synthetic_retrieval/outputs/train/<run_name>/checkpoints/runtime_config.json
```
