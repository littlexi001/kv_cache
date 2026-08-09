# 四个 P0 方法的运行说明

## 四台机器分配

| 任务 ID | 方法 | 用途 |
|---:|---|---|
| 0 | `optimized_phase_complementary` | 无近程约束的相位互补主方法 |
| 1 | `optimized_phase_complementary_local` | 加入近程相位保持的完整主方法 |
| 2 | `native_rope` | 同训练预算的原生 RoPE 基线 |
| 3 | `rnope_every4` | 每第 4 层使用 NoPE 的外部基线 |

每个任务独占一台 8 卡机器。四台机器必须使用同一份项目代码、模型目录、
DCLM 数据和 `.env` 参数。

## 首次安装

在每台机器执行：

```bash
cd /mnt/workspace/qwen3_06b_aliyun_pe_pretraining_iclr27
bash scripts/install.sh
cp -n configs/experiment.env.example .env
```

检查 `.env` 至少包含：

```bash
MODEL_ROOT=/mnt/workspace/Qwen3-0.6B
DCLM_ROOT=/mnt/workspace/dclm
RUN_ROOT=/mnt/workspace/pe_pretrain_100b_iclr27
GPU_LIST=0,1,2,3,4,5,6,7
INITIALIZATION=from_scratch
SEQ_LEN=8192
MICRO_BATCH=1
GRAD_ACCUM=32
GLOBAL_BATCH_SIZE=256
TARGET_TOKENS=100000000000
LEARNING_RATE=1e-4
```

这里的 `from_scratch` 表示只读取 Qwen3-0.6B 的结构和 tokenizer，不读取
`model.safetensors`。四个条件都从同一个随机种子初始化，才能比较 PE
架构本身。

## 先生成两个新增策略

任务 0 或任务 1 首次启动时会自动生成；也可以在任意机器手动检查：

```bash
bash scripts/prepare_p0_strategies.sh
python src/validate_contract.py \
  --model-root /mnt/workspace/Qwen3-0.6B \
  --dclm-root /mnt/workspace/dclm \
  --strategy configs/strategies/optimized_phase_complementary.json \
  --sequence-length 8192
python src/validate_contract.py \
  --model-root /mnt/workspace/Qwen3-0.6B \
  --dclm-root /mnt/workspace/dclm \
  --strategy configs/strategies/optimized_phase_complementary_local.json \
  --sequence-length 8192
```

必须保留两个 `.optimization.json` 文件，它们记录了离线目标是否真的
改善、最终倍率矩阵和优化轨迹。

## 10M smoke test

Smoke test 使用单独输出目录，不能把它的 checkpoint 直接续成正式 100B
任务，因为短跑和正式任务的 cosine 学习率总步数不同。

四台机器分别执行，其中 `N` 为 0、1、2、3：

```bash
cd /mnt/workspace/qwen3_06b_aliyun_pe_pretraining_iclr27
RUN_ROOT=/mnt/workspace/pe_smoke_10m_iclr27 \
TARGET_TOKENS=10000000 \
MILESTONE_TOKENS=10000000 \
WARMUP_STEPS=1 \
ALLOW_NON_100B=1 \
GPU_LIST=0,1,2,3,4,5,6,7 \
bash scripts/run_p0_worker.sh N
```

通过条件：进程完成、loss/gradient 有限、无 OOM、四个任务实际 global
batch 都是 256，并且策略 profile 与预期一致。10M 只验证代码和数值，不能
判断方法有效。

## 正式 100B 运行

Smoke test 通过后，在四台机器分别执行：

```bash
cd /mnt/workspace/qwen3_06b_aliyun_pe_pretraining_iclr27
GPU_LIST=0,1,2,3,4,5,6,7 bash scripts/run_p0_worker.sh N
```

脚本在后台启动，SSH 或网页断开不会终止训练。它会在约 0.1B、1B、10B、
25B、50B、75B 和 100B token 保存 checkpoint、执行评测并继续训练。

## 查看状态与 TensorBoard

```bash
bash scripts/status.sh
tail -f /mnt/workspace/pe_pretrain_100b_iclr27/<strategy>/nohup.log
```

TensorBoard：

```text
http://<该机器 IP>:6006
```

如果不开放端口，可在本地执行：

```bash
ssh -L 6006:127.0.0.1:6006 <user>@<server-ip>
```

## 停止单个任务

```bash
bash scripts/stop_strategy.sh <strategy>
```

例如：

```bash
bash scripts/stop_strategy.sh optimized_phase_complementary
```

## 10B 筛选规则

先比较同一实际 token 数下的四个 checkpoint。新增方法只有同时满足以下
条件才继续消耗到 100B：

1. 至少一个长程指标或 gold-answer NLL 优于 `native_rope`；
2. held-out DCLM PPL 不比 `native_rope` 差 2% 以上；
3. manifest SHA256、实际 token 数和完成样本数一致；
4. 提升不只出现在一个手写 synthetic probe 上。

一个 seed 的结果只能筛选候选。最终论文结果仍需对原生 RoPE、RNoPE 和
最多两个候选方法运行至少三个种子，并补充完整 RULER、LongBench、短上下文
PPL 和基础能力评测。
