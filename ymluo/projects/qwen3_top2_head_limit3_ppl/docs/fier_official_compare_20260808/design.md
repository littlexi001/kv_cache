# FIER 官方实现公平测速：设计

## 可证伪问题

在同一张 RTX 3090、同一个 LongChat-7B MHA 模型、相同历史长度和相同生成长度下，FIER 官方发布代码能否复现论文报告的 1.2--1.5x decode 加速，并且能否与 QKSieve 做同口径比较？

## 先验与数学口径

一次 decode 的时间写成：

`T_decode = T_non_attention + T_select + T_sparse_attention`。

Full 对照为：

`T_full = T_non_attention + T_full_attention`。

公平比较要求 Full 与稀疏方法共享模型权重、KV layout、prefill 历史、GPU、生成步数和同步边界。加速比定义为 `T_full / T_method`，不能用 Attention 子系统速度除以整模型速度。

## 实现契约

- FIER 固定为官方提交 `e0b34153591dd7a55171f09f30abee35b0f08356`。
- 模型使用 `longchat-7b-v1.5-32k`，MHA 为 32 Q heads / 32 KV heads。
- 首轮审计复现官方随机 hidden-state 协议，但在完整 decode 循环前后执行 CUDA 同步。
- Full 使用同一个 FIER paged-KV backend，并把预算设为全部历史；这样 Full 会真正读取 prefill KV。
- FIER 使用与 QKSieve 相同的 active-token 预算，正式目标为每 head 1,280 token。
- 输出必须记录请求预算、控制器 page size、实际 page budget 和有效 token 上界。

## 已发现的官方代码边界

1. 发布仓库缺少 CMake 引用的 `kernels/3rdparty`，需要从 Quest 官方提交补齐头文件。
2. 官方 speed script 在每一步 `time.perf_counter` 前后没有 CUDA 同步。
3. 官方 speed script 的普通 Full 路径没有传入 past KV，因此没有读取 prefill 历史。
4. 当前 release decode 路径调用 `bit=2`，且 `quantize_and_pack` 返回随机 code、scale、zero，而不是从真实历史 K 构建索引。
5. `fier_init` 的公开参数默认 page size 为 1，但 controller 内部硬编码 page size 为 8；默认 token budget 因此不是实际 active-token 数。

这些问题意味着：原样 release 可以用于程序路径审计，不能直接作为“质量与速度同时成立”的论文对照。
