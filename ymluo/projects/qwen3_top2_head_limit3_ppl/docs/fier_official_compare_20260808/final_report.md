# FIER 官方代码测速与 QKSieve 对照报告

## 结论

FIER 官方 CUDA backend 已在 RTX 3090 上编译并运行。修正 CUDA 异步计时、Full KV 复用和预算换算后，FIER 在 8K 历史下为 `0.993x`，在 16K 历史下为 `1.177x`。这说明官方系统路径确实在长度增加后开始获益，但短文本没有速度优势。

当前不能把这些数值写成 FIER 的完整质量--速度复现。发布代码的 speed path 使用随机 packed selector 输入，而且实际调用 2-bit，而不是论文定义的真实 1-bit Key 索引。可发表的表述应是“官方 release 后端审计”；FIER 论文中的 1.2--1.5x 可作为论文报告值引用，不能与本次随机 selector 结果混为一谈。

## 1. 可证伪问题

在相同 LongChat-7B 模型、相同 RTX 3090、相同历史长度、相同 256-token decode 和相同 active-token 上限下，FIER 官方 release 是否比共享同一 paged-KV backend 的 Full Attention 更快？

通过条件：同步计时后的 `Full latency / FIER latency > 1`。失败条件：该比值小于或等于 1。证据不足条件：Full 不读取完整历史、实际预算不等于请求预算，或者 selector 没有由真实历史 K 构造。

## 2. 计算模型

一次 decode 的时间写成：

`T_full = T_common + T_full_attention`

`T_fier = T_common + T_selector + T_sparse_attention`

当历史较短时，`T_selector` 可能大于省下的 Full Attention 时间；当历史增长而 active-token 预算固定时，`T_full_attention` 近似随历史长度增长，`T_sparse_attention` 主要随预算增长，因此存在长度交叉点。

实现契约如下：

- Full 和 FIER 使用同一模型权重与官方 paged-KV backend。
- 两者都先写入历史 KV，再进行 256 次单-token forward。
- FIER 的有效上限为 160 个 page，每 page 8 token，即 1,280 token/head。
- 完整 decode 循环前后均执行 CUDA 同步。
- 只改变 selector 是否启用；模型、dtype、GPU、历史长度和生成长度保持不变。

## 3. 实验设置

| 项目 | 设置 |
|---|---|
| FIER 提交 | `e0b34153591dd7a55171f09f30abee35b0f08356` |
| Quest 提交 | `01c1623bf9395009520874e989e29f683203b357` |
| 模型 | LongChat-7B-v1.5-32K，FP16，32 Q heads / 32 KV heads |
| GPU | NVIDIA RTX 3090 24GB |
| 软件 | PyTorch 2.5.0+cu121，CUDA compiler 12.2 |
| 历史长度 | 8,192 和 16,384 token |
| 生成长度 | 256 token |
| 重复 | 每个条件 3 次 |
| 汇总 | 三次完整 decode 的 ms/token 中位数 |
| FIER 预算 | 1,280 active tokens/head |
| 输入 | 官方 speed protocol 风格的随机 hidden states |

16K 无法按官方脚本一次性 prefill：脚本会建立未使用的 `16K x 16K` 稠密 mask，并保留整段 MLP 激活。测量脚本改为 2K 分块追加 KV，并跳过未被自定义 Attention 消费的 mask。这不改变 decode 算法，但意味着 16K 结果是“官方 decode path + 显存修复”，不是官方脚本逐行原样运行。

## 4. 结果

| 历史长度 | Full | FIER release | Full / FIER |
|---:|---:|---:|---:|
| 8K | 23.849 ms/token | 24.010 ms/token | 0.993x |
| 16K | 28.735 ms/token | 24.413 ms/token | 1.177x |

单张 RTX 3090 无法同时容纳 LongChat-7B FP16 权重和 32K 完整 MHA KV。为得到一个不跨 GPU、也不把 KV 卸载到 CPU 的 32K 容量诊断，Full 与 FIER 共同改用 NF4 double-quant 模型权重；K/V、paged-KV backend、1,280 token/head 预算和 256-token 同步 decode 均保持不变。该组结果必须与上面的 FP16 表分开：

| 权重 | 历史长度 | Full | FIER release | Full / FIER |
|---|---:|---:|---:|---:|
| NF4 | 8K | 48.541 ms/token | 58.142 ms/token | 0.835x |
| NF4 | 16K | 48.381 ms/token | 58.585 ms/token | 0.826x |
| NF4 | 32K | 49.548 ms/token | 58.878 ms/token | 0.842x |

NF4 下 FIER 在三个长度都额外增加约 9--10 ms/token，并未出现随长度增长而跨过 Full 的现象。原因是 bitsandbytes NF4 权重解量化已经成为整模型 decode 的主要公共开销，Full 在 8K--32K 只增长约 1 ms/token，FIER selector 的固定成本无法被节省的 Attention 时间抵消。因此，`0.842x` 是 3090 容量受限条件下的诊断结果，不是论文 FP16/RTX 4090 配置的复现值。

8K 结果否定了“FIER 在所有长度都加速”的假设。16K 结果支持“固定稀疏预算存在长度交叉点”：历史翻倍后，Full 增加 4.887 ms/token，而 FIER 只增加 0.403 ms/token。

作为独立的 Attention 子系统证据，项目内相同 MHA 形状测试中，QKSieve 在 8K/16K 分别为 `1.227x/1.566x`，FIER-style 路径为 `0.953x/1.098x`。该结果说明 QKSieve selector 的扫描成本更低，但不能直接证明整模型 decode 按相同比例领先。

标准 Hugging Face LongChat 路径上的配对结果如下。稳态排除前 16 个生成 token；online 包含索引构建、首步初始化和 256-token decode，不包含 prefill。

| 历史长度 | Full 稳态 | QKSieve 稳态 | 稳态速度比 | 256-token online 速度比 |
|---:|---:|---:|---:|---:|
| 8K | 37.070 ms/token | 39.371 ms/token | 0.942x | 0.444x |
| 16K | 52.316 ms/token | 35.824 ms/token | 1.461x | 0.752x |

QKSieve 在 16K 的持续生成阶段已经比 Full 快 46.1%，但固定索引构建为 7.302 秒，首 token 为 1.360 秒；按当前实现约需生成 522 token 才能抵消固定成本。8K 的稳态仍慢于 Full，不存在有限的摊平点。这个结果把后续系统目标收窄为两项：降低 8K selector 固定开销，以及让索引在 prefill/缓存创建时增量生成，而不是在首个 decode 请求前集中构建。

## 5. 失败分析

### 5.1 官方计时脚本

官方 `bench_textgen_fier.py` 在每步调用 `time.perf_counter()`，但没有在计时边界同步 CUDA。GPU kernel 是异步提交的，因此原脚本的单步 Python 时间不能直接作为 GPU decode latency。

### 5.2 预算换算

`fier_init` 用公开参数 `token_budget // page_size` 得到 page budget，但控制器内部 page size 固定为 8。若照 README 默认传入 `page_size=1`，请求 1,280 token 会变成最多约 10,240 个 active token。当前实验传入 8，使有效上限回到 1,280。

### 5.3 selector 数据流

release decode path 创建 `torch.empty` 的历史 Key，调用 `bit=2` 的打包函数；该打包函数返回随机整数 code 与随机 scale/zero。因而系统运行成功只证明 kernel/backend 能执行，不能证明真实 1-bit FIER 的排序质量。

### 5.4 32K

官方控制器为完整 MHA K/V 预分配显存。LongChat-7B FP16 权重加 32K MHA KV 超过单张 24GB 3090 容量：NF4 模型加载后分配 3.866 GB，32K cache 初始化后总分配已达 23.344 GB。因此当前可报告两条相互独立的证据：论文原结果使用 RTX 4090，并报告 32K 超过 1.5x；本机相同 NF4 权重配对下，FIER 为 58.878 ms/token、Full 为 49.548 ms/token，即 0.842x。前者是外部论文结果，后者是容量 workaround，不应混为同一复现口径。

## 6. QKSieve 对照边界

目前可安全写入论文的比较有两类：

1. 引用 FIER 论文公开结果：单张 RTX 4090、LLaMA-2-7B、最长 32K、decode 256 token，报告 1.2--1.5x。
2. 报告本项目同硬件、同形状的 Attention 路径：QKSieve 在 8K--128K 为 1.227x--5.502x，且 selector 完整包含 query preparation、索引扫描、top-k 和稀疏 Attention。

不能把第二类 Attention 数值直接写成相对 FIER 整模型 decode 的领先倍数。最终论文需要在统一 serving backend 中实现真实 FIER 1-bit 索引和 QKSieve 索引，再用相同请求做端到端配对。

## 7. 下一项不确定性

最关键的未完成项不是再调 FIER 的预算，而是建立同一 backend 的真实 selector 对照：在 prefill 阶段由真实 K 构建 FIER 1-bit g32 索引，decode 阶段读取该索引，保存 top-k，并同时测召回率、LongBench 质量和同步 decode latency。只有这一项完成后，才允许声称 QKSieve 在相同质量下比官方 FIER 更快。

## 复现位置

- 本地测量脚本：`ymluo/projects/qwen3_top2_head_limit3_ppl/src/benchmark_fier_official_audit_20260808.py`
- 远端 FIER：`/home/fdong/qksieve_iclr2027/external/FIER_official`
- 远端结果：`/home/fdong/qksieve_iclr2027/results/20260809_fier_official_audit`
- 本地原始 JSON：`ymluo/projects/qwen3_top2_head_limit3_ppl/docs/fier_official_compare_20260808/raw_results`
