# Head-level KV/Expert Shared Buckets

本目录管理 head-level KV bucket 与 expert bucket 共享结构的设计、实现、实验和结果。

## 当前核心问题

已有 seq-compress / sparse-inference 相关实验显示：在每层每个 head 只保留 top-2% attention token 时，推理质量基本不下降，某些情况下甚至优于 full attention。现在的核心问题不是再次证明 top-2% 有效，而是解释：

> top-2% token 为什么有效？使它们在某个 head 上得到高 QK score 的共同 feature 到底来自哪里？

需要定位的 feature 可能来自三类空间：

1. **hidden-native feature**：输入 hidden state 本身已经聚类，Q/K 只是读出已有相似性；
2. **Q/K-extracted feature**：hidden 中相似性不明显，经过该 head 的 `W_Q/W_K` 后才变明显；
3. **bilinear/SVD feature**：高 score 主要来自完整 $W_Q/W_K$ SVD 后按 head 切输出空间得到的少数 query/key feature pairs。

如果能解释 top-2% token 的共同 feature 在哪里，就能决定 inverse KV 的 bucket 和 expert specialization 应该定义在哪个空间里。

当前研究文件：

1. [设计文档](./docs/design.md)
2. [实验设计](./docs/experiment_design.md)
3. [实验结果](./docs/visualization_results.md)

目录约定：

```text
fdong_inverse_kv/
├── docs/       # 当前理论、实验设计和结果解释
├── src/        # 模型与分析代码
├── scripts/    # 可复现的训练和评估入口
└── outputs/    # 本地生成结果，默认不提交 Git
```

## 当前实现

当前代码实现训练阶段的第一版结构：

1. Qwen3 每层每个 attention query head 独立 routing；
2. `layer_input/q/k/v` 可作为 router input；
3. 可选 detached exclusive causal mean centering；
4. 同 bucket token 才进入该 head 的 attention，并可保留 local/sink fallback；
5. 同一 bucket id 选择该 head 的同编号 expert；
6. 只使用 next-token prediction loss，router 通过 expert 路径获得梯度。

当前尚未实现 bucketed KV-cache decode。训练原型会拒绝 `use_cache=True`，避免静默使用与训练结构不一致的普通 KV cache。

## 远端训练

在远端仓库的 `fdong_inverse_kv/scripts` 下执行：

```bash
bash pretrain_qwen.sh
```

脚本默认保留现有服务器路径：

```text
CONFIG_DIR=../../Qwen3-0.6B
DATA_DIR=../../dclm/global-shard_01_of_10
```

常用覆盖方式：

```bash
RUN_NAME=k-centered-e4 \
ROUTER_INPUT=k \
CENTER_ROUTER_INPUT=true \
NUM_EXPERTS=4 \
EXPERT_INTERMEDIATE_SIZE=3072 \
bash pretrain_qwen.sh
```

`EXPERT_INTERMEDIATE_SIZE=3072` 是 ordinary MoE 的基准宽度。Shared head-level 模型按等参数公式自动推导实际宽度：

$$
\mathrm{head\_width}=\frac{3\times3072}{16+2}=512
$$

因此 ordinary expert 是 `1024 -> 3072 -> 1024`，shared head expert 是 `64 -> 512 -> 1024`。16 个 head 输出求和并除以 `sqrt(16)`。4 experts 时两者总参数量分别约为 `1.388888B` 和 `1.389003B`。

`runs/` 中当前已同步的四组阶段性结果来自旧版 `64 -> 3072 -> 64` head expert。它们保留为历史结果，但旧 checkpoint 不能加载到新版结构中；新版实验必须使用新的 `RUN_NAME`。

## 四组首轮实验

四个实验使用相同 Qwen3 配置、DCLM 数据、batch size、优化器、expert 数量和 expert intermediate size。建议一次只启动一个实验；每个命令会自行进入后台。

### 1. Ordinary MoE baseline

标准 full attention，每层 4 个完整 `1024 -> 3072 -> 1024` top-1 experts：

```bash
RUN_NAME=ordinary-moe-e4 \
ARCHITECTURE=ordinary_moe \
NUM_EXPERTS=4 \
EXPERT_INTERMEDIATE_SIZE=3072 \
bash pretrain_qwen.sh
```

### 2. K routing + causal centering

```bash
RUN_NAME=shared-fullout-k-centered-e4 \
ARCHITECTURE=shared_bucket \
ROUTER_INPUT=k \
CENTER_ROUTER_INPUT=true \
NUM_EXPERTS=4 \
EXPERT_INTERMEDIATE_SIZE=3072 \
bash pretrain_qwen.sh
```

### 3. Raw K routing

```bash
RUN_NAME=shared-fullout-k-raw-e4 \
ARCHITECTURE=shared_bucket \
ROUTER_INPUT=k \
CENTER_ROUTER_INPUT=false \
NUM_EXPERTS=4 \
EXPERT_INTERMEDIATE_SIZE=3072 \
bash pretrain_qwen.sh
```

### 4. Layer-input routing

第一轮默认也测试去中心化的 layer input，使它与实验 2 只相差 router representation：

```bash
RUN_NAME=shared-fullout-layer-input-centered-e4 \
ARCHITECTURE=shared_bucket \
ROUTER_INPUT=layer_input \
CENTER_ROUTER_INPUT=true \
NUM_EXPERTS=4 \
EXPERT_INTERMEDIATE_SIZE=3072 \
bash pretrain_qwen.sh
```

对应输出目录分别为：

```text
fdong_inverse_kv/runs/ordinary-moe-e4/
fdong_inverse_kv/runs/shared-fullout-k-centered-e4/
fdong_inverse_kv/runs/shared-fullout-k-raw-e4/
fdong_inverse_kv/runs/shared-fullout-layer-input-centered-e4/
```

每个目录包含：

```text
runtime_config.json       # 完整实验配置，可提交 Git
train_metrics.jsonl       # 每步 loss/routing/KV candidate 指标，可提交 Git
training_complete.json    # 完成状态，可提交 Git
checkpoint-*.pt           # 模型与优化器状态，被 .gitignore 忽略
```

因此过程指标可以通过 Git 同步；checkpoint 需要手动下载。控制台 `logs/*.log` 也被忽略，因为关键信息已经结构化写入 `train_metrics.jsonl`。

查看进度，例如：

```bash
tail -f ../logs/shared-fullout-k-centered-e4.log
```

训练动态同时写入：

```text
../runs/shared-fullout-k-centered-e4/train_metrics.jsonl
```

训练完成后生成紧凑汇总：

```bash
python3 extract_train_metrics.py ../runs/shared-fullout-k-centered-e4
```

## 本地验证

本地不读取 DCLM，也不需要模型权重：

```bash
cd fdong_inverse_kv/scripts
python3 smoke_test.py
bash single_thread_debug_qwen.sh
```

本地配置副本位于 `configs/qwen3_0.6b/config.json`，只包含 Qwen3-0.6B 架构配置。

`scripts` 中原先复制进来的其他分析和测试文件保留为远端目录/接口参考。当前正式训练链路只依赖 `pretrain_qwen.py`、`single_thread_debug_qwen.py`、`train_common.py`、`models/myqwen.py` 和 `utils/data_utils.py`。
