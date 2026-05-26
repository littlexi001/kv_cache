# Qwen3 MoE Structured Language

This project studies why attention-neighborhood router clustering works on
clean hierarchical synthetic data but can fail on real text.

The default dataset is `structured_language`, a synthetic token stream with:

- topic spans sampled from a Zipf distribution;
- private topic entities plus ambiguous shared entities;
- shared function and verb tokens across topics;
- filler/noise tokens that do not always belong to a clear topic;
- copy templates with delayed repetition;
- bridge templates that intentionally mix two topics in one local context.

The goal is to separate two ideas that were coupled in the original hierarchy:

```text
high attention pair != always same semantic/router group
same topic/entity/function can overlap in the same sequence
```

Metadata is still tensor-shaped and compatible with the existing metrics:

```text
metadata[:, :, 0] = syntax role id
metadata[:, :, 1] = topic id
metadata[:, :, 2] = entity id
metadata[:, :, 3] = span id
metadata[:, :, 4] = template/relation id
```

So the existing `same_higher_by_layer` metric means same-topic same-expert in
this project.

Run:

```bash
bash ymluo/projects/qwen3_moe_structured_language/scripts/run_train.sh
```

Four paired experiment runners:

```bash
# Small-data MoE baseline: no attention-cluster auxiliary loss.
bash ymluo/projects/qwen3_moe_structured_language/scripts/run_train_baseline.sh
bash ymluo/projects/qwen3_moe_structured_language/scripts/run_eval_baseline.sh

# Small-data test: attention-cluster enabled.
bash ymluo/projects/qwen3_moe_structured_language/scripts/run_train_test.sh
bash ymluo/projects/qwen3_moe_structured_language/scripts/run_eval_test.sh

# Larger/harder data baseline.
bash ymluo/projects/qwen3_moe_structured_language/scripts/run_train_big_baseline.sh
bash ymluo/projects/qwen3_moe_structured_language/scripts/run_eval_big_baseline.sh

# Larger/harder data test: attention-cluster enabled.
bash ymluo/projects/qwen3_moe_structured_language/scripts/run_train_big_test.sh
bash ymluo/projects/qwen3_moe_structured_language/scripts/run_eval_big_test.sh
```

Useful sweeps:

```bash
STRUCTURED_NOISE_RATE=0.5 \
STRUCTURED_AMBIGUITY_RATE=0.6 \
RUN_NAME=structured-noise50-amb60 \
bash ymluo/projects/qwen3_moe_structured_language/scripts/run_train.sh
```

```bash
ATTENTION_CLUSTER_WEIGHT=0.001 \
ATTENTION_CLUSTER_NEGATIVE_WEIGHT=0.01 \
RUN_NAME=structured-low-attn-neg \
bash ymluo/projects/qwen3_moe_structured_language/scripts/run_train.sh
```

Important defaults:

```text
SEQ_LEN=256
SYNTHETIC_DATA_MODE=structured_language
STRUCTURED_TOPIC_COUNT=8
STRUCTURED_ENTITIES_PER_TOPIC=8
STRUCTURED_SHARED_ENTITY_COUNT=16
STRUCTURED_NOISE_RATE=0.25
STRUCTURED_AMBIGUITY_RATE=0.35
STRUCTURED_COPY_RATE=0.25
STRUCTURED_BRIDGE_RATE=0.25
MOE_HEAD_LEVEL=false
USE_PRE_ROUTER=true
PRE_ROUTER_INPUT=q
ATTENTION_CLUSTER_WEIGHT=0.01
MOE_LOAD_BALANCE_LOSS_WEIGHT=0.01
```

Eval a checkpoint:

```bash
CKPT_FILE=ymluo/projects/qwen3_moe_structured_language/outputs/train/structured-language-attn-cluster/checkpoints/10000.pth \
bash ymluo/projects/qwen3_moe_structured_language/scripts/run_eval.sh
```
