# QKSieve 256K/512K cold-skip 结果

## 1. 实验边界

模型为 Qwen3-4B-Instruct：

```text
max_position_embeddings = 262144
rope_theta = 5000000
rope_scaling = null
```

因此：

- 256K attention 算子位于模型原生上限；质量实验必须满足
  `history + eval <= 262144` 才能作为原生范围证据。
- 512K 超出原生上限，速度测试仍有效，但模型质量只能作为外推压力测试，不能作为论文主结果。

速度使用 RTX 3090 上的独立物理 CUDA attention 路径。质量使用 GPU 0--6，
单次 Full prefill 后顺序回滚同一份 KV cache，对比 Full、QKSieve 和 cold-skip，
避免重复进行长提示 prefill。

## 2. Attention 子系统速度

所有 QKSieve 路径均包含：

- WMMA Query 投影与量化；
- sampled-quantile 阈值估计；
- 混合位宽索引扫描和候选写出；
- 最多 1,280 个原始 FP16 K/V token 的精确 QK、softmax、AV。

cold-skip 额外包含局部位置到原始 token ID 的映射。索引首次构建和模型非
attention 部分不计入本表。

### 256K

| 方法 | 检索时间 | 完整 attention | 相对原 QKSieve | 相对 Full SDPA |
|---|---:|---:|---:|---:|
| Full SDPA | - | 175.008 ms | - | 1.00x |
| 原 QKSieve | 7.703 ms | 12.650 ms | 1.00x | 13.834x |
| cold-skip 50%，扫描 62.6% | 6.142 ms | 10.878 ms | **1.163x** | **16.088x** |
| cold-skip 60%，扫描 70.1% | 6.649 ms | 11.419 ms | 1.108x | 15.326x |

### 512K

| 方法 | 检索时间 | 完整 attention | 相对原 QKSieve | 相对 Full SDPA |
|---|---:|---:|---:|---:|
| Full SDPA | - | 380.282 ms | - | 1.00x |
| 原 QKSieve | 11.647 ms | 17.513 ms | 1.00x | 21.714x |
| cold-skip 50%，扫描 62.6% | 9.091 ms | 14.485 ms | **1.209x** | **26.254x** |
| cold-skip 60%，扫描 70.1% | 9.736 ms | 14.986 ms | 1.169x | 25.376x |

50% 档的 token ID 映射在 256K/512K 分别占 0.431/0.732 ms。扫描信息减少
37.4%，但检索只获得 1.254x/1.281x，原因仍是映射、Query 准备、候选写出和
kernel 固定成本没有随扫描范围同比下降。

## 3. 旧 256K 边界外推质量诊断

设置：

- 文本：未参与冻结模板构建的 `mixed_b`
- history：262,144 token
- 预测：16 token
- attention budget：每个 Query head 最多 1,280 token
- Full、QKSieve、cold-skip 共享完全相同的 prefill KV
- shared Full prefill：856.52 秒

该旧协议的历史本身已经占满 262,144，因此 16 个预测位置实际超过模型声明
上限 16 token。下表只能作为严格配对的边界压力诊断，不能再称为原生 256K
质量。正式补测使用 `history=262080, eval=64`，使总长度恰好为 262,144。

| 方法 | PPL | 相对 Full PPL 质量 | 相对 QKSieve | Top-1 一致率 | KL(Full || 方法) |
|---|---:|---:|---:|---:|---:|
| Full Attention | 5.7518 | 100% | - | 100% | 0 |
| 原 QKSieve | **5.1052** | 112.67% | 100% | 100% | 0.07031 |
| cold-skip 50% | 5.5731 | 103.21% | **91.60%** | 100% | 0.08696 |
| cold-skip 60% | 5.5182 | 104.23% | **92.52%** | 100% | 0.08775 |

这 16 个 token 上，三个稀疏方法的 argmax 都与 Full 相同，不能把 QKSieve
更低的 PPL 解释为稳定质量提升。但在完全配对的条件下，cold-skip 相对原
QKSieve 的 NLL 分别增加 0.0877 和 0.0778，对应 PPL 质量只保留 91.60% 和
92.52%。同时 KL 也从 0.0703 上升到约 0.087。

该结果与 32K trace 的尾部风险一致：随着历史变长，当前 Query 需要的证据更
容易落入历史低频集合。扩大热集合可以缓解，但不能消除这个问题。

质量样本只有一个窗口、16 个轻微外推 token，因此它是方向性诊断，不是可投稿
的 256K 质量表。要形成正式结论，还需要严格原生范围的多个独立窗口和原生
支持相应长度的任务基准。

## 4. 256K 整模型速度解释

质量 runner 中原 QKSieve full-topk 的实测 steady decode 为 76.881 ms/token，
Full 为 602.776 ms/token。cold-skip 的 PPL 路径为了验证质量仍物化完整 proxy
score 后再 mask，不能使用其 wall-clock 时间。

将独立 CUDA attention 差值代入同一 QKSieve runner：

```text
50% cold-skip:
76.881 - (12.650 - 10.878) = 75.109 ms/token
estimated speedup = 1.024x

60% cold-skip:
76.881 - (12.650 - 11.419) = 75.650 ms/token
estimated speedup = 1.016x
```

因此即使到 256K，attention 子系统的 1.163x 也只预计兑现为约 2.4% 的整模型
收益。瓶颈仍主要位于模型 dense backbone、跨 GPU 层传输和最终 1,280-token
精确 attention，而不是低比特索引扫描。

## 5. 512K 质量尝试

512K 外推压力测试使用：

```text
7 x RTX 3090
prefill chunk = 1024
history = 524288
eval tokens = 8
```

运行约 64 分钟后，Full prefill 在 GPU 1 申请额外 480 MiB 时 OOM：

```text
PyTorch allocated = 22.05 GiB
GPU free = 453.38 MiB
requested = 480.00 MiB
```

没有生成可用的 512K PPL，不能填入质量数值。继续通过更小 chunk、8 卡或
80 GB H100 强行完成在工程上可行，但该模型没有 512K RoPE scaling，得到的
绝对质量也不能作为正式 benchmark。当前保留 512K 物理速度结果，不把缺失的
PPL 用估计值替代。

## 6. 决策

256K 进一步加强了“不合入 cold-skip 主路径”的判断：

1. 50% 档 attention 子系统在 256K/512K 可再快 1.163x/1.209x。
2. 整模型预计只再快约 2%--4%。
3. 256K 配对 PPL 相对原 QKSieve 已出现约 7.5%--8.4% 的损失。
4. 质量风险明显大于整模型速度收益。

更值得优化的是最终精确 K/V 消费、Q/K projection epilogue 融合和静态 decode
执行，而不是继续根据历史检索频率删除索引 token。

## 7. 复现入口

```text
src/benchmark_qksieve_per_head_cold_skip_20260730.py
src/run_qksieve_coldskip_longcontext_quality_20260730.py
src/run_direct_countcap_denseprompt_ppl_20260725.py

results/20260730_qksieve_per_head_coldskip_cuda_256k_512k_gpu6.json
results/20260730_qksieve_coldskip_quality_256k_sharedprefill
results/20260730_qksieve_coldskip_quality_512k_extrap_sharedprefill/run.log
```
