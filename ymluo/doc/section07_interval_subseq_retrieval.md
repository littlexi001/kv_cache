# Section 7 — Interval Subsequence Retrieval

> 新增日期：2026-05-18；最近同步日期：2026-05-23

## 项目：`qwen3_interval_subseq_retrieval`

新增日期：2026-05-18

这个项目训练一个小型 fdong-style `MyQwen3ForCausalLM`，用直接生成的 token-id arithmetic subsequences 做标准 causal next-token prediction。它和上面的 answer-only synthetic retrieval 不同：没有 query token、placeholder 或最终答案位置，所有 1023 个有效 next-token 位置都会贡献 loss。

默认数据配置：

```text
total_token = 10000
subseq_len = 4
seq_len = 1024
intervals = 1
```

`interval=1` 时样本来自 `[1,2,3,4]`、`[5,6,7,8]` 这类连续 4-token subsequences；`interval=2` 等更大 interval 会生成 `[2,4,6,8]`、`[10,12,14,16]` 这类跨步 pattern。每条训练样本随机抽取 256 个 subsequences 并拼成 1024-token 序列。

模型默认把 Qwen config 覆盖为 8 层，并使用 U-Net 风格 stride schedule：

```text
num_hidden_layers = 8
attention_stride_pattern = 1,1,4,4,4,4,1,1
```

当前实验结果：

- 8 层 Qwen3-0.6B 配置，stride schedule 为 `1,1,4,4,4,4,1,1`。
- 在 `interval=1`、`subseq_len=4` 的标准设置下，模型基本在 1500 step 左右就能把可学习的局部递推规则完全学会。
- 当前观测到的 next-token accuracy 是 `0.7508`，和理论上限基本一致。原因是训练序列由彼此独立的 4-token subsequences 拼接而成，例如 `[5,6,7,8,1,2,3,4]` 中，`5→6`、`6→7`、`7→8`、`1→2`、`2→3`、`3→4` 这些 subsequence 内部转移可以被完全预测；但 `8→1` 是两个独立 subsequences 的边界，数据构造时没有给 `8` 和 `1` 建立任何关系，因此模型无法从前文确定边界后的第一个 token。
- 因此每 4 个 token 里大约只有 3 个 next-token 转移是确定可学的，理论 accuracy 约为 `3/4 = 0.75`。观测值 `0.7508` 中高出 `0.75` 的约 `0.0008` 更可能来自边界位置上的随机猜测或评估噪声，而不是模型学到了跨 subsequence 边界关系。

单卡训练：

```bash
bash ymluo/projects/qwen3_interval_subseq_retrieval/scripts/run_train.sh
```

多卡 DDP 训练：

```bash
CUDA_DEVICES=0,1,2,3 \
bash ymluo/projects/qwen3_interval_subseq_retrieval/scripts/run_ddp_train.sh
```

常用覆盖参数：

```bash
INTERVALS=1,2,4 \
RUN_NAME=unet8-intervals-1-2-4 \
TOTAL_STEPS=20000 \
BATCH_SIZE=4 \
TRAIN_MODE=full_sequence_lm \
bash ymluo/projects/qwen3_interval_subseq_retrieval/scripts/nohup_train.sh
```

如果 `INTERVAL_GROUP_MODE=scaled`，最大 token id 是 `TOTAL_TOKEN * max(INTERVALS)`；超过基础 Qwen vocab 时，训练脚本默认会自动扩大 `config.vocab_size`，除非显式设置 `AUTO_RESIZE_VOCAB=false`。

保存 checkpoint 后可以导出每层/head 的 masked raw attention scores 和 softmax probabilities：

```bash
RUN_NAME=unet8-interval1-lm-ddp \
CKPT_STEP=2000 \
bash ymluo/projects/qwen3_interval_subseq_retrieval/scripts/dump_attention_scores.sh
```

主要输出：

```text
ymluo/projects/qwen3_interval_subseq_retrieval/outputs/train/<run_name>/metrics.jsonl
ymluo/projects/qwen3_interval_subseq_retrieval/outputs/train/<run_name>/checkpoints/<step>.pth
ymluo/projects/qwen3_interval_subseq_retrieval/outputs/train/<run_name>/checkpoints/runtime_config.json
ymluo/projects/qwen3_interval_subseq_retrieval/outputs/train/<run_name>/attention_scores/step_<ckpt_step>/
```
