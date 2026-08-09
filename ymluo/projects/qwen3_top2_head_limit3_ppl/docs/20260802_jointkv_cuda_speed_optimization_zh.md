# JointKV-Sieve CUDA 速度优化结果（2026-08-02）

## 1. 结论

本轮优化将 128K、Llama 类 GQA 配置下的完整 Attention 路径从约
`4.24x` 提高到同卡配对中位数 **`8.65x`**。完整路径包含 query
投影、逐步 LUT 构建、基础和残差检索、compact top-k、精确稀疏
Attention、tail mass 扣除及输出融合；不包含一次性索引构建，也不是
整模型 Decode 速度。

长度曲线如下。`Full ms` 和 `Ours ms` 均来自同卡交替测量，速度取五次
或七次配对的中位数。

| 历史长度 | Full ms | Ours ms | 完整 Attention 加速 | 新路径与旧全局选择重合率 |
|---:|---:|---:|---:|---:|
| 8K | 0.1697 | 0.2810 | 0.60x | 95.78% |
| 16K | 0.3219 | 0.2855 | 1.13x | 93.25% |
| 32K | 0.6322 | 0.2891 | 2.18x | 94.96% |
| 64K | 1.2321 | 0.2847 | 4.34x | 97.03% |
| 128K | 2.4378 | 0.2818 | **8.65x** | 97.74% |

相较优化前，本轮将有效交叉点从约 32K 降到了 16K 以下。8K 仍然慢于
Full，不能宣称全长度无条件加速。

## 2. 最终实现

### 2.1 融合基础扫描与局部候选生成

每个 warp 扫描 32 个历史 token，同时完成：

1. 64-bit principal Key code 打分；
2. joint K/V ID correction；
3. risk LUT 修正；
4. 每个 GQA query 的 tail cluster mass 累计；
5. warp 内保留 8 个基础候选。

因此不再物化 `[KV head, N]` 的完整 priority 数组，也不再对它执行一次
大规模全局 top-k。128K 下，每个 KV head 产生 32,768 个基础候选。

### 2.2 融合残差读取、打分与 shortlist

残差内核直接使用基础候选索引读取原 residual code，在寄存器中完成
48-bit residual 打分并形成 warp-local shortlist。删除了 residual code、
joint ID 和 risk code 的三次独立 `gather`，全局 top-k 只作用于紧凑
shortlist。

shortlist 宽度使用统一公式，而不是按长度手写规则。设最终精确预算为
`K`，基础候选包含 `W` 个 candidate warp，则：

`r = next_power_of_two(max(4, ceil(2K / W)))`

每个 candidate warp 保留 `r` 个 token。8K、16K、32K、64K、128K 对应
的 `r` 分别为 16、16、16、8、4。128K shortlist 为 4,096 个 token，
最终从中选 1,280 个 token。

### 2.3 每步 query byte-LUT

原始 CUDA 扫描对每个 token 执行 64 或 48 次逐 bit 符号加法。现在把
每 8 bit 的 256 种符号组合预计算成 query LUT：

- 64-bit 基础编码需要 8 次查表；
- 48-bit 残差编码需要 6 次查表；
- LUT 使用 FP32，单步大小约 448 KiB；
- LUT 每个 Decode step 重建，建表时间已经计入完整路径。

在 128K 上，基础扫描由约 `0.274 ms` 降至约 `0.092 ms`，LUT 建表约
`0.014 ms`。

### 2.4 精确稀疏 Attention 与 tail 修正

最终仍读取最多 1,280 个 GPU-resident FP16/BF16 K/V token，执行精确
稀疏 Attention。默认使用 split=8。所有 token 的 joint-ID tail mass 在
基础扫描中累计，再扣除已选 token 的 mass，最后将 64 个 Value centroid
与稀疏输出融合。

## 3. 128K 配对测速

测试配置：RTX 3090、BF16、32 query heads、8 KV heads、head dimension
128、1,280 token/head。七轮同卡交替测量，每轮各执行 50 次 Full 与
JointKV-Sieve：

- Full：约 2.434--2.440 ms；
- JointKV-Sieve：六轮约 0.281--0.282 ms，一轮 0.288 ms；
- 加速范围：8.45--8.68x；
- 中位数：**8.65x**；
- 均值：**8.63x**。

## 4. 小规模质量检查

为了验证 warp-local 候选不是纯随机张量上的速度技巧，在 Qwen3-0.6B、
8K、相同最终 KV 预算下，对比了旧全局 20% refine 和新局部选择：

| 文本 | 全局 refine 质量保持 | warp-local 质量保持 | top-1 agreement |
|---|---:|---:|---:|
| 医学 | 94.85% | **96.22%** | 100% |
| 编译器 | 100.0007% | **100.0010%** | 100% |

这只能说明两个 targeted probe 中没有观察到退化，不能替代 128K PPL、
LongBench/RULER 和第二模型验证。成熟的 adaptive JointKV 主方法质量也
不能直接由这两条结果推出。

## 5. 失败或未采用的优化

- **FP16 query LUT**：最终选择与 FP32 LUT 重合 99.94%，但完整速度
  只有约 8.22x，没有超过 FP32。LUT 已主要命中缓存，half-to-float
  转换抵消了带宽收益。
- **LUT selected-mass subtraction**：约 0.0253 ms，与逐 bit 版本约
  0.0247--0.0257 ms 持平，未采用。
- **split=16**：精确稀疏 Attention 子内核由约 0.0916 ms 降至
  0.0814 ms，但完整路径收益不稳定并出现一次长尾，默认保留 split=8。
- **每个 residual warp 只保留 2 个 token**：128K 重合率降至 85.18%，
  且完整速度反而低于保留 4 个的版本。

## 6. 当前边界与下一步

1. 当前 `8.65x` 是完整 Attention 子系统，不是整模型 Decode。
2. 新 CUDA 路径尚未接入真实 HF 模型逐层 Decode，不能从 Attention
   microbenchmark 直接宣称端到端加速。
3. 128K 的真实质量尚未验证；在完成质量验证前，warp-local 路径应称为
   fast candidate，而不是冻结的论文主方法。
4. 8K 仍为 0.60x。若不允许成本门控回到 Full，下一步需要融合 compact
   top-k、精确稀疏 Attention、selected-mass subtraction 和 tail blend，
   单纯减少历史扫描已无法解决短序列固定开销。

## 7. 复现入口

- CUDA 实现：`src/jointkv_sieve_cuda_20260802.py`
- 分项及配对测速：`src/benchmark_jointkv_sieve_direct_stages_20260802.py`
- PPL runner：`src/run_jointkv_residual_ppl_20260802.py`
- 质量探针脚本：`scripts/run_jointkv_warp_local_quality_probe_20260802.sh`
- 本地原始结果：`results/20260802_jointkv_cuda_optimized/`
