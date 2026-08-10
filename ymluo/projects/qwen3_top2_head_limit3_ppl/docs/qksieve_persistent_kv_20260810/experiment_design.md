# Persistent-KV 实验设计

## 研究问题

冻结版 QKSieve-Robust 能否在同一 request-local QK 坐标系内的共享前缀分支与
连续追加生成中复用一次构建的辅助索引，并把索引成本均摊，而不是在每个分支中
隐式重建？

## 实验对象

- 模型：`NousResearch/Yarn-Llama-2-7b-128k`，32 Query heads、32 KV heads、head dimension 128，即原生 MHA。
- 硬件：RTX 3090；后续 H100 使用同一脚本复测。
- 上下文：由固定文本重复编码得到 32K 与 64K 前缀。
- 方法：原生 Full FP16 SDPA；冻结 `QKSieve-Robust`，240-bit Key index、rank-16 block-256 INT4 ValueSketch、`alpha=0.5`、最多每 head 1,280 个精确 token。
- 分支：从同一前缀 logits 的前四个 token 分别开始，每个分支生成 32 token；最后重复第一个分支。
- 连续追加：回退到前缀后连续生成 128 token。
- 进程重复标签：`20260810/20260811/20260812`；所有方法共享模型、固定前缀、
  分支与生成长度。输入和 greedy decode 不随标签变化，因此区间只描述进程重复。
- 执行方式：32K 和 64K 顺序运行，避免 CUDA JIT、CPU 特征分解和显存带宽互相污染。
- 运行时：CUDA 扩展在请求计时前显式加载；QK factor、完整 Key index 与 ValueSketch
  的构建全部计入 cold 请求。

## 指标

1. `cold_persistent_request_ms_per_token`：一次建索引加第一次分支生成的总时间除以生成 token 数，不含 prefill。
2. `cold_end_to_end_request_ms_per_token`：dense prefill、一次建索引和第一次分支的完整请求时间除以生成 token 数。
3. `shared_prefix_warm_mean_ms_per_token`：索引已存在时，各热分支的平均整模型延迟。
4. `shared_prefix_amortized_ms_per_token`：一次建索引与四个分支总生成时间之和，除以四个分支的总 token 数。
5. `append_only_ms_per_token`：不回退、连续追加 128 token 的平均整模型延迟。
6. `reuse_tokens_equal`：第一次与最后一次相同分支的 token 序列是否逐项相等。
7. `index_buffers_reused_without_rebuild`：所有层的 Key/Value 索引地址及两类重建计数是否在分支间保持不变。
8. `rewind_value_layers_correct`：每次回退是否覆盖模型的全部 ValueSketch 层。
9. `persistent_contract_passed`：完整 Key 预建、Value 预建/安装、长度、指针和回退检查是否同时通过。

长度检查区分两个时刻：预建结束时 Key/Value index 必须与 cache 等长；Decode
结束时二者必须严格落后 cache 一个 token，因为最后输入 token 会在下一步才转为
可检索历史。lag 为 0 或大于 1 都视为失败。

## 通过、失败与证据不足

通过：

- 四项生命周期检查都为 `true`；
- 32K/64K 热分支不包含建索引成本；
- 64K 热分支相对 Full 有明确加速；
- 原始 JSON、日志和硬件信息完整保存。

失败：

- 相同分支 token/hash 不一致；
- 任何索引指针或重建计数变化；
- kernel 读取旧分支后缀，表现为输出不一致或越界；
- 热分支仍重新运行 QK factor、Key 编码或 ValueSketch 构建。

证据不足：

- 只有微基准而没有真实模型；
- Full 与 QKSieve 使用不同模型布局或不同 GPU 数；
- 并发任务污染延迟；
- 只报告稳态速度，不报告首次建索引和均摊结果。

## 路径

- 核心接口：`src/run_head_top2_targeted_ppl_20260714.py`
- 基准：`src/benchmark_qksieve_persistent_kv_20260810.py`
- 启动：`scripts/run_qksieve_persistent_kv_case_20260810.sh`
- 输出：`results/20260810_qksieve_persistent_kv_v3_multiseed/`

## v1 审计结论

v1 不能进入论文表格。其第一次稀疏 Decode 与并发 CUDA 扩展加载发生重叠，32K/64K
首步分别出现约 215 秒/109 秒异常延迟；同时审计器只识别 Python 快路径的
ValueSketch runtime，把普通执行路径的真实 Value 索引错误记录为 0 层。v2 保留 v1
原始文件用于失败分析，但不复用其 cold、均摊或“全部正确”结论。

v3 在匹配软件栈与冻结源码上对每个长度执行三个独立进程，并同时记录不含 prefill
的 cold-index 和包含 prefill 的 cold-E2E；论文系统表只读取 v3 独立汇总。

## 已知限制

当前分支差异由首 token 构造，主要验证缓存生命周期；它不等价于完整多轮对话
质量测试，也不覆盖新问题使 Query 统计和 request-local 坐标改变的情况。质量
结论仍由独立 LongBench、RULER 与 PPL 实验给出。
