# 确定性采样 QKSieve：方法设计

## 1. 问题定义

目标是在自回归 decode 的每一层、每个 Query head 上，从长度为 `N` 的历史 KV 中找出少量候选，只对候选执行精确 FP16 QK 和 Value 聚合，并用低秩统计近似未选中的 Value 尾项。

本阶段不使用 Full Attention 回退。需要同时满足：

- 质量：32K、64K、128K 的 held-out PPL 质量尽量保持 Full Attention 的 99% 以上。
- 速度：稳态 decode 在长上下文上显著快于 Full Attention。
- 稳定性：相同输入重复执行时，候选顺序和尾项归约必须完全一致。
- 通用性：预算与阈值采样数由序列长度决定，不使用任务标签或学习型 router。

## 2. 可证伪猜想

**猜想：** 请求内估计的 QK 低比特代理足以定位高注意力 token；未选中 token 的 Value 贡献可由 rank-16 INT4 低秩统计近似；若分位点样本数随目标比例缩放，并使用固定顺序归约，则 32K--128K 下可以同时获得稳定候选、接近 Full 的质量和随长度增长的 decode 加速。

该猜想在下列任一条件出现时被否定：

1. 代理候选与原实现的候选集合不一致，且不是阈值并列导致。
2. 相同张量重复运行时，候选顺序或 Value 尾项发生变化。
3. 32K--128K 任一长度的质量保持率显著低于 99%。
4. 64K 或 128K 的稳态 decode 加速不能覆盖新增索引扫描和尾项计算。

## 3. 数学模型

### 3.1 长度预算

每个 Query head 的目标候选数为：

$$
k(N)=\min\left(N,\max\left(256,\min\left(\lceil0.06N\rceil,1280\right)\right)\right).
$$

因此 2K、4K 使用 256 个 token，8K--16K 使用约 6%，24K 以后逐渐封顶为 1280。本文当前重点长度 32K、64K、128K 均使用至多 1280 个历史 token/head。

### 3.2 QK 代理分数

Prefill 后，从当前请求的 Query/Key 二阶矩估计 OAS 收缩协方差，并构造 QK-balanced 表征。每个 head 的 128 维 Key 被分为 8 个、每个 16 维的 band。分配器在 8 个 band 上使用总成本 15 的约束；每个 band 可取 0/1/2/4/8 bit，启用一个 band 还计入 1 个 scale 元数据成本。分配目标是最小化该请求 Query 分布下的预期 QK 分数均方误差。

记投影和量化后的 Query、Key 为 `q_hat` 和 `k_hat_i`，代理分数为：

$$
\hat s_i = \langle \hat q,\hat k_i\rangle.
$$

该代理只负责候选定位。最终选中 token 仍从 GPU 常驻的原始 FP16 K/V 计算精确 attention。

### 3.3 按尾部有效样本数缩放的分位点

目标保留比例为 `p=k(N)/N`。设希望在目标上尾中观察到的样本数为 `c`，采样数不是固定 256 或 1024，而是：

$$
m(N,c)=\operatorname{clip}_{[256,8192]}
\left(256\left\lceil\frac{c/p}{256}\right\rceil\right).
$$

含义是：若采样分布能代表全历史，则样本中预计约有 `c` 个分数落入目标上尾。其分位统计相对波动约随 `1/sqrt(c)` 缩小；固定样本数在 `p` 随长度变小时没有这一性质。当前比较 `c=32` 与 `c=64`：c32 优先速度，c64 优先候选数量稳定性。

采样位置使用分段中心点，并加入由 Query-head 行号确定的相位：

$$
i_j=\left(\left\lfloor\frac{(2j+1)N}{2m}\right\rfloor+
((131h+17)\bmod\max(1,\lfloor N/m\rfloor))\right)\bmod N.
$$

对这 `m` 个代理分数排序，取目标上尾的边界作为阈值 `tau_h`。全历史只执行一次线性代理扫描，保留满足 `s_hat_i >= tau_h` 的 token。候选数量可以围绕 1280 小幅变化，但不进行完整 top-k 排序。

### 3.4 Value 尾项

对每个 KV head 的 Value 建立 rank-16 基：

$$
v_i \approx \mu_v + U z_i,
$$

其中 `z_i` 在 256-token block 内逐维 INT4 量化。选中集合 `S` 使用原始 FP16 V；未选中集合的 softmax 分母与低秩系数在扫描代理分数时累计：

$$
D_T=\sum_{i\notin S} e^{(\hat s_i-\tau)/\sqrt d},
\qquad
c_T=\sum_{i\notin S} e^{(\hat s_i-\tau)/\sqrt d}\,\hat z_i.
$$

尾项 Value 分子近似为：

$$
N_T \approx D_T\mu_v+Uc_T.
$$

最终输出由候选的精确 FP16 attention 分子/分母与尾项近似在同一 softmax 标尺下合并。

## 4. 确定性 CUDA 算法

### 输入

- 每层当前 Query 的低比特码和 scale。
- 当前请求的 packed Key 索引及 band 位宽元数据。
- rank-16 INT4 Value code、block minimum 和 scale。
- 原始 GPU 常驻 FP16 K/V。
- `N`、`k(N)`、`m(N)`。

### 步骤

1. **采样阈值：** 对每个 Query head 计算 `m(N)` 个分段样本分数并排序，得到 `tau_h`。
2. **一次历史扫描：** 每个 256-token CUDA block 同时计算一个 KV head 对应的 4 个 Query head 代理分数。
3. **bitmask：** 每个 warp 用 32-bit ballot 写出选中 mask，不对全局候选计数器执行 atomicAdd。
4. **block 局部尾项：** 每个 block 写出 1 个选中分母、1 个尾分母和 16 个尾系数的局部和。
5. **前缀压缩：** 每个 Query head 用固定整数前缀扫描按 token 编号递增写出候选下标。
6. **固定树归约：** 对每个 Query head 的 18 个尾项分量执行固定顺序树归约。
7. **精确稀疏 attention：** 对候选原始 FP16 K/V 计算精确 attention，并与 rank-16 尾项合并。

### 输出

- 按 token 编号递增的候选下标和每个 head 的候选数。
- 阈值、overflow 标志。
- 选中分母、尾分母、16 维尾系数。
- 每层 attention 输出。

### 通过条件

- 与原子扫描版阈值完全相同，候选集合完全相同。
- 相同输入重复 50 次，候选下标和三个尾项张量逐 bit 相同。
- overflow 为 0。
- 新 workspace 占用相对完整 KV 很小。

### 已知失败原因

- `m` 太小：阈值方差增大，候选数和质量对微小 Query 扰动敏感。
- Value rank 太小：已有 rank-8/rank-12 probe 出现 top-1 翻转。
- 全局原子压缩：候选集合相同，但候选顺序随 block 调度变化。
- 全局原子浮点归约：尾项约有 `1e-4` 到 `1e-3` 绝对抖动，可在多步 decode 中放大。
- 短上下文：索引和阈值计算的固定开销可能大于稀疏 attention 节省。

## 5. 参数表

| 参数 | 当前值 | 含义 | 太小时 | 太大时 |
|---|---:|---|---|---|
| `direct_fraction` | 6% | 中等长度目标比例 | 漏掉高分 token | attention 成本升高 |
| `min_tokens` | 256 | 短上下文下限 | 2K/4K 质量下降 | 短文本更慢 |
| `max_tokens` | 1280 | 长上下文上限 | 128K 尾部风险增大 | 长文本 attention 变慢 |
| `tail anchors c` | 默认 64，fast 版 32 | 样本上尾有效点数 | 分位点不稳定 | 采样排序更慢 |
| `Value rank` | 16 | 尾项低秩维数 | 已观察到 top-1 翻转 | 索引和扫描成本增加 |
| `Value bits` | 4 | 低秩系数量化位宽 | 尾项误差增大 | 辅助显存和带宽增大 |
| `Value block` | 256 | INT4 scale/min 共享块 | 元数据增多 | 局部量化误差增大 |

## 6. 代码契约

- 主状态配置：`src/run_head_top2_targeted_ppl_20260714.py`
- CUDA 扫描与压缩：`src/mixedblock_spectral_cuda_20260729.py`
- Value 尾项 attention：`src/qksieve_valuesketch_cuda_20260801.py`
- 长上下文 probe：`src/run_qksieve_coldskip_longcontext_quality_20260730.py`
- 论文默认 variant：`qksieve_qmse_oas_requestlocal_valuesketch16_sorted_c64_k1280`
- 速度优先 variant：`qksieve_qmse_oas_requestlocal_valuesketch16_sorted_c32_k1280`

## 7. 声明边界

- 当前结果测量的是已有 KV 可复用场景下的稳态 decode；prefill 单独报告，不混入 decode 加速。
- 原始 FP16 K/V 当前仍常驻 GPU。`7.4%` 左右指辅助索引相对完整 KV 的额外占用，不代表只保存了 7.4% 的总 KV。
- 当前 probe 证明数值和系统可行性，不等价于完整 LongBench/RULER 论文表格。
- 没有 Full Attention 回退，也没有任务标签或学习型 router。
