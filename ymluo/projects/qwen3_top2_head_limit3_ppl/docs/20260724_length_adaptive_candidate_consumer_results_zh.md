# 长度自适应候选 Attention Consumer

## 1. 问题

当前 CountCap 的在线路径为：

```text
原始 query
-> Key-PCA48 + INT4 全局索引
-> 256 点 sampled-quantile
-> 约 6% 候选
-> 候选上的精确 QK、softmax 和 AV
```

8K/16K Nsight profile 显示，PCA 阈值估计与全局扫描只占在线 CUDA
kernel 时间约 5.0%/4.7%，而候选上的精确 QK、softmax 和 AV
分别占 12.7%/21.7%。因此，继续减少 PCA 算术量不是当前最重要的优化。

## 2. 被否定的原位 KV Cache

首先实现了 `PreallocatedDynamicCache`：

- 保持 Hugging Face `DynamicCache` 的 causal-mask 语义；
- 为每层预分配 K/V buffer；
- decode 时原位追加，不再执行整段 `torch.cat`；
- 稀疏 attention kernel 支持带 stride 的有效前缀视图，不隐式复制整段 K/V。

正确性验证：

- 随机小型 Llama 分段 prefill 的全部 logits 与标准 `DynamicCache`
  逐元素相同；
- 非连续 KV 上的 stride-aware CUDA attention 与 contiguous 输入
  逐元素相同；
- 8K/16K LongBench 配对生成和分数全部相同。

速度结果：

| 长度 | Base online | 原位 cache online | 加速 |
|---:|---:|---:|---:|
| 8K | 2.5885 s | 2.5893 s | 0.9997x |
| 16K | 2.6468 s | 2.6120 s | 1.0133x |

原位 cache 保留为系统实现选项，但其 steady decode 收益不足，不能解决
8K/16K 的主要瓶颈。

## 3. QProjScan 与 Split Consumer 组合

旧 split4 实验使用的是较慢的检索前端，无法单独判断候选 consumer 的收益。
本轮首次组合：

```text
当前最快 qprojscan 检索前端
-> 相同候选集合
-> split2 或 split4 精确候选 attention
-> 数值稳定的分段 softmax 归并
```

每个 split block 计算本段候选的：

- 局部最大 logit；
- 局部 softmax denominator；
- 未归一化 weighted-V。

归并 kernel 使用各段最大值恢复全局 softmax。该方法不改变 PCA 索引、
候选预算、候选 token 或精确 K/V，只改变候选 attention 的执行结构。

## 4. 严格配对结果

协议：Llama-3.1-8B-Instruct、RTX 3090、GovReport、64-token decode。
所有表中方法的预测与任务分数完全一致。

### 4.1 8K 和 16K

三样本均值：

| 长度 | Base | split2 | split4 |
|---:|---:|---:|---:|
| 8K | 2.6209 s | 2.6365 s, 0.994x | 2.6345 s, 0.995x |
| 16K 混合样本 | 2.6411 s | 2.6123 s, 1.011x | 2.6165 s, 1.009x |

16K 组中只有一个样本真正达到 16K，其余为 13.6K 和 7.4K。对同一个
16K 样本做三种方法顺序轮换后：

| 方法 | Online 中位数 | 相对 Base |
|---|---:|---:|
| Base qprojscan | 2.7235 s | 1.000x |
| qprojscan + split2 | 2.6500 s | **1.028x** |
| qprojscan + split4 | 2.6947 s | 1.011x |

### 4.2 24K 和 32K

| 长度 | Base | split2 | split4 |
|---:|---:|---:|---:|
| 24K | 3.3484 s | 2.9072 s, 1.152x | **2.7081 s, 1.236x** |
| 32K，中位数 | 3.8971 s | 3.2864 s, 1.186x | **3.0495 s, 1.278x** |

32K 使用三种执行顺序轮换。三个 split4 online 时间为
3.0593/3.0365/3.0495 s，收益稳定，不是冷启动顺序造成的。

## 5. 当前冻结的自动规则

设本步实际候选数为 `C`。当前 RTX 3090 执行规则为：

```text
C < 900        : single block
900 <= C < 1280: split2
C >= 1280      : split4
```

该规则只读取本步已经计算出的候选数，不使用任务名、训练 router、oracle
或质量回退。对应方法名：

```text
countcap_fullprompt_keypca_direct_qkvfused_qprojscan_qkvsplitauto_prefillindex
```

自动版本单样本复核：

| 长度 | 自动动作 | Base online | Auto online | 加速 |
|---:|---|---:|---:|---:|
| 8K | single | 2.5410 s | 2.5876 s | 0.982x，计时噪声 |
| 16K | split2 | 2.7127 s | 2.6434 s | **1.026x** |
| 24K | split4 | 3.3419 s | 2.7264 s | **1.226x** |
| 32K | split4 | 3.8960 s | 3.0711 s | **1.269x** |

8K 自动分支实际调用与 Base 相同的 single-block kernel，没有额外候选计算；
表中差异来自单次运行噪声。四个长度的预测和分数全部严格一致。

## 6. 预分配 cache 与 stride-aware consumer

第一版预分配 cache 只消除了 `DynamicCache` 的 `torch.cat`，但 split attention
的 C++ 包装层仍执行：

```text
key.contiguous()
value.contiguous()
```

因此，32 个生成 token、31 个稀疏层、K/V 两份恰好产生
`32 × 31 × 2 = 1984` 次整段 KV 物化。v102 删除这两个物化点，
并让 split kernel 直接读取 key/value 的 batch、head、token 和 dim stride。

32K、32-token decode 的 Nsight GPU kernel 汇总如下：

| 路径 | GPU kernel 总时间 | 整段 KV copy | split attention |
|---|---:|---:|---:|
| auto split + DynamicCache | 1485.7 ms | 384.5 ms | 187.7 ms |
| 预分配 cache，包装层仍 contiguous | 1416.7 ms | 377.0 ms | 191.6 ms |
| **v102 stride-aware + 预分配 cache** | **1087.9 ms** | **50.7 ms** | 189.9 ms |

v102 相对原 auto 路径把 GPU kernel 总时间降低 **26.78%**。剩余
50.7 ms 主要是单 token 原位写入和小张量 copy，不再是整段 KV 物化。

## 7. 相对 Full KV 的长度交叉点

协议：Llama-3.1-8B-Instruct、RTX 3090、同一 GovReport 样本、64-token decode。
每个长度使用四种方法顺序轮换，表内为四次中位数。

| 长度 | Full KV | 旧 qprojscan | auto split | v102 split+inplace | v102 / Full |
|---:|---:|---:|---:|---:|---:|
| 8K | 2.5506 s | 2.6106 s | **2.5886 s** | 2.6500 s | 0.962x |
| 16K | 3.9394 s | 2.7374 s | 2.6926 s | **2.6583 s** | **1.482x** |
| 24K | 5.1689 s | 3.3643 s | 2.7300 s | **2.7107 s** | **1.907x** |
| 32K | 6.3985 s | 3.9057 s | 3.0666 s | **2.6999 s** | **2.370x** |

三条稀疏路径在每个长度下的预测和分数均严格一致。该表只用于系统测速；
它是一个截断 GovReport 样本，不能代替完整 LongBench 质量表。

预分配 cache 在 8K 有固定开销，但从 16K 开始获益。因此部署入口增加
纯成本门控：

```text
prompt_tokens < 14K : DynamicCache
prompt_tokens >= 14K: preallocated cache
```

阈值来自 8K/16K 实测成本交叉，允许通过
`COUNTCAP_INPLACE_CACHE_MIN_TOKENS` 按硬件覆盖。它只切换 cache 执行后端，
不改变候选、预算、attention token 或输出，不是 Full attention 回退。

统一部署方法名：

```text
countcap_fullprompt_keypca_direct_qkvfused_qprojscan_qkvsplitauto_cacheauto_prefillindex
```

独立进程复核中，8K 的原 auto 与 cache-auto 分别为 2.5895 s 和 2.5948 s，
预测及分数严格一致。

## 8. 当前结论

1. qprojscan、候选数自适应 consumer 和 stride-aware 预分配 cache 可以组合，
   且不改变当前主方法的检索与质量。
2. 实际 online 速度交叉点已从接近 32K 下移到 8K–16K 之间；
   16K/24K/32K 相对 Full KV 分别达到 1.48x/1.91x/2.37x。
3. 8K 仍比 Full KV 慢约 1.5%（使用不启用预分配的 auto 路径），是下一步
   唯一需要单独突破的短上下文点。
4. 下一步应融合 sampled-quantile threshold scan 与候选 attention 消费，
   减少 8K 下的 kernel launch、候选索引落地和读回；不再调预算或训练 router。
5. 主方法仍是纯稀疏 attention，不加入 Full attention 回退。

## 9. 复现位置

核心代码：

```text
src/run_head_top2_targeted_ppl_20260714.py
src/run_sample_calibrated_longbench_20260717.py
src/qabs_cuda_kernels.py
src/preallocated_dynamic_cache_20260724.py
```

实验脚本：

```text
scripts/launch_inplacecache_pair_8k16k_2gpu_20260724.sh
scripts/launch_qprojscan_split_grid_8k16k_2gpu_20260724.sh
scripts/launch_qprojscan_split_order_rotation_16k_3gpu_20260724.sh
scripts/launch_qprojscan_split_24k32k_4gpu_20260724.sh
scripts/launch_qprojscan_splitauto_validation_4gpu_20260724.sh
scripts/launch_qprojscan_splitauto_nsys_32k_1gpu_20260724.sh
scripts/launch_splitauto_inplace_32k_rotation_3gpu_20260724.sh
scripts/launch_splitauto_inplace_nsys_32k_1gpu_20260724.sh
scripts/launch_full_crossover_splitauto_inplace_8k32k_4gpu_20260724.sh
scripts/launch_cacheauto_validation_8k16k_2gpu_20260724.sh
scripts/launch_cacheauto_isolated_8k_2gpu_20260724.sh
```

结果目录与脚本同名，位于 `results/`。
