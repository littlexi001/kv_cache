# Section 11 — 常用参数与注意事项

> 新增日期：2026-05-14；最近同步日期：2026-05-23

## 常用默认参数

大多数脚本都可以用环境变量配置：

```bash
MODEL_PATH=/path/to/Qwen3
DATA_PATH=/path/to/dclm
TEXT_PATH=/path/to/part-00000.txt
OUTPUT_DIR=/path/to/output
MAX_TOKENS=3000
CHUNK_SIZE=256
DEVICE=cuda
DTYPE=bfloat16
```

训练脚本默认使用 Hugging Face streaming 读取大型 DCLM 风格数据。除非确定磁盘足够并且想构建 Arrow cache，否则保持 `STREAMING=true`。

synthetic retrieval 项目的评估和训练还常用：

```bash
NUM_SAMPLES=256
BATCH_SIZE=4
VARIANTS=A,B
TRAIN_MODE=anchor_kv_decode
TOTAL_STEPS=20000
```

interval subsequence 项目的训练和 attention dump 常用：

```bash
CONFIG_DIR=/mnt/workspace/Qwen3-0.6B
TOTAL_TOKEN=10000
SUBSEQ_LEN=4
SEQ_LEN=1024
INTERVALS=1,2,4
ATTENTION_STRIDE_PATTERN=1,1,4,4,4,4,1,1
CKPT_STEP=2000
QUERY_POSITIONS=last
```

MoE attention cluster 项目常用：

```bash
DEBUG_NUM_HIDDEN_LAYERS=3
DEBUG_HIDDEN_SIZE=128
MOE_NUM_UNIQUE_EXPERTS=4
MOE_HEAD_LEVEL=false
ATTENTION_CLUSTER_WEIGHT=0.05
ATTENTION_CLUSTER_TOPK=4
SYNTHETIC_SAMPLING_DISTRIBUTION=uniform
```

## 实践注意事项

- 精确命令参数和输出路径以每个项目自己的 README 为准。
- 分析脚本默认把输出写到各项目的 `outputs/` 目录下。
- synthetic retrieval 项目默认直接生成 token-id 序列，避免 tokenizer segmentation 对可控任务的干扰；训练默认只对最终 answer 位置施加 loss。
- interval subsequence 项目同样直接生成 token-id 序列，但目标是标准 causal LM，所有 next-token 位置都会参与 loss；不要把它和 answer-only retrieval 的指标直接混读。
- 长上下文分析会占较多显存和内存；smoke test 时先减小 `MAX_TOKENS` 和 `CHUNK_SIZE`。
- sparse decode 和 routing 实验还需要 kernel-aware 实现，理论 KV-read reduction 才能转化为真实 serving 加速。
- 对 KV cache 压缩来说，raw cosine 是很好的诊断信号，但最终判断必须回到 attention logits、attention output、loss/PPL。
- MoE selectivity 实验中，`same_higher_by_layer` 高不代表一定有 selectivity，需要结合 `expert_load` 分布判断是否坍缩。
