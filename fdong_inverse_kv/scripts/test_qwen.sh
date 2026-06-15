export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

LOCAL_BATCH_SIZE=64
TEST_BATCH_SIZE=2048
SEQ_LEN=1024
DATA_SHUFFLE=true  # 注意：bash 中 true/false 是字符串
USE_BF16=true

USE_MOE=false
MOE_INTERMEDIATE_SIZE=1536
EXPERT_PER_TOKEN=2
NUM_EXPERTS=16
GATING_REFERENCE="switch"

NUM_WORKERS=4
CONFIG_DIR="../../Qwen3-0.6B"
DATA_DIR="../../dclm/global-shard_01_of_10"
RUN_NAME="3B-test"
# ========== 使用 RUN_NAME 构建路径和文件名 ==========
CKPT_DIR="../checkpoints/${RUN_NAME}"

# ========== 构建 Python 命令 ==========
ARGS=""
ARGS+=" --local_batch_size $LOCAL_BATCH_SIZE"
ARGS+=" --test_batch_size $TEST_BATCH_SIZE"
ARGS+=" --seq_len $SEQ_LEN"
ARGS+=" --num_workers $NUM_WORKERS"
ARGS+=" --config_dir $CONFIG_DIR"
ARGS+=" --data_dir $DATA_DIR"
ARGS+=" --ckpt_dir $CKPT_DIR"  # ← 关键：传入构建好的路径

# 处理布尔参数
[ "$DATA_SHUFFLE" = "true" ] && ARGS+=" --data_shuffle"
[ "$USE_BF16" = "true" ] && ARGS+=" --use_bf16"
[ "$USE_MOE" = "true" ] && ARGS+=" --use_moe"

# MOE 和 Active Learning 的子参数（即使主开关关了也可以传，但通常只在开启时传）
ARGS+=" --moe_intermediate_size $MOE_INTERMEDIATE_SIZE"
ARGS+=" --expert_per_token $EXPERT_PER_TOKEN"
ARGS+=" --num_experts $NUM_EXPERTS"
ARGS+=" --gating_reference $GATING_REFERENCE"

python test_qwen.py ${ARGS}
    # >>../logs/alnew-${ACTIVE_LEARNING_K}-${ACTIVE_LEARNING_PERCENT}-${ACTIVE_LEARNING_FROM}.log 2>&1 &

