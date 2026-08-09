# 固定前缀、精确后缀的 CUDA Graph Decode

## 问题

QKSieve 的低比特索引和稀疏 attention 已经减少了计算量，但普通 HuggingFace
逐 token 执行包含大量 Python 调度、kernel launch 和动态 cache 操作。这些固定
开销会掩盖稀疏 attention 的收益。

本子问题检验一个可证伪命题：如果固定长上下文前缀的内存地址，只让生成后缀
增长，并把整个 decode step 捕获为一张 CUDA Graph，那么整模型速度应随上下文
长度增加而明显超过最优 Full-attention Graph，同时保持与同一计算公式的普通执行
一致。

## 先验与数学模型

设不可变前缀长度为 `P`，已经生成的后缀长度为 `t`。第 `t` 步的有效 KV 长度为：

`N_t = P + t + 1`

其中最后一个位置是当前 token。QKSieve 只对前缀建立一次低比特索引，并从前缀
检索候选集合 `R_h(q)`。所有先前生成的后缀 token 精确保留：

`C_h(q,t) = R_h(q) union {P, ..., P+t-1}`

当前 token `P+t` 由 attention kernel 作为 self token 单独加入。未入选前缀 token
仍由 frozen ValueSketch 近似，不对动态后缀做近似。

因此前缀扫描成本不随生成步数变化，后缀精确计算只随 `t` 增长。对短生成任务，
其主要成本近似为：

`T_qksieve(N_t) = T_model_base + T_prefix_scan(P) + T_exact(K+t) + T_tail`

而 Full attention 的主要长度相关成本为 `T_full(N_t) = T_model_base + c*N_t`。

## 实现契约

输入：

- 已完成 prefill 的固定容量 K/V cache。
- 固定前缀长度 `P`。
- GPU 上的一元素 `int32 active_key_count`。
- 每层 frozen QKSieve 索引、ValueSketch 和候选 workspace。

每步过程：

1. 使用设备端 `cache_position` 将新 K/V 原地写入预分配 cache。
2. 将 `active_key_count` 更新为 `cache_position + 1`。
3. 只扫描固定前缀的低比特索引，生成每个 query head 的候选。
4. 将 `[P, active_key_count-1)` 中除当前 self 外的后缀 token 追加为精确候选。
5. 精确计算候选 QK、softmax 和 AV，并合并 frozen ValueSketch 前缀尾部补偿。
6. 在 Graph 内执行 logits、greedy argmax、下一 token 写入和位置加一。

固定地址包括模型参数、K/V cache、query 编码、候选、ValueSketch workspace、输入
token 和位置张量。一张 Graph 可覆盖多个后缀长度，不需要重新捕获。

## 代码

- `src/preallocated_dynamic_cache_20260724.py`：设备端位置原地写 K/V。
- `src/qksieve_valuesketch_cuda_20260801.py`：动态有效长度 attention 和精确后缀追加。
- `src/run_head_top2_targeted_ppl_20260714.py`：固定前缀 fast path。
- `src/run_direct_countcap_denseprompt_ppl_20260725.py`：整模型增长 Graph benchmark。
- `src/benchmark_qksieve_dynamic_suffix_graph_20260807.py`：attention 层严格等价测试。

## 当前边界

该实现适合 prefix cache 复用、多轮问答和 agent 场景。当前只实现 greedy 单 batch
decode；采样、连续 batching、请求退出和多 batch page 管理尚未接入。Full 的动态
非全零 mask 在当前 PyTorch SDPA 上触发慢 kernel，因此不能把该慢路径作为论文
主 baseline。
