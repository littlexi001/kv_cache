# FIER 官方实现公平测速：结果

## 实验设置

本目录比较 FIER 官方发布代码与 QKSieve。所有表必须先说明模型、GPU、上下文长度、生成长度、active-token 预算、是否包含索引构建以及 CUDA 同步边界。

## 实际运行状态

- FIER 官方提交：`e0b34153591dd7a55171f09f30abee35b0f08356`。
- 模型：LongChat-7B-v1.5-32K，FP16，MHA 结构。
- GPU：单张 NVIDIA RTX 3090 24GB。
- 生成：固定 256 步；每个条件独立运行 3 次，报告每次完整 decode 循环的中位数。
- active-token 预算：每个 head 最多 1,280 token。
- 计时：在完整 decode 循环前后执行 `torch.cuda.synchronize()`；不包含模型加载和 prefill。
- Full 与 FIER 共用官方 paged-KV backend，均复用已写入的历史 KV。
- 16K 采用 2K 分块 prefill，并跳过官方模型中未被 `FierAttention` 使用的稠密 causal mask；两项改动只降低 prefill 峰值显存，不改变被测 decode 路径。

## Decode 结果

速度比定义为 `Full ms/token / FIER ms/token`，大于 1 表示 FIER 更快。

| 历史长度 | Full | FIER release | FIER 速度比 | 实际 token 上限 |
|---:|---:|---:|---:|---:|
| 8K | 23.849 ms/token | 24.010 ms/token | 0.993x | 1,280/head |
| 16K | 28.735 ms/token | 24.413 ms/token | 1.177x | 1,280/head |

8K 下 selector 与稀疏 Attention 的总开销尚未低于 Full Attention，因而没有加速。16K 下 Full Attention 随历史长度增长，而 FIER 的稀疏部分接近固定预算，开始得到 17.7% 的 decode 加速。这个趋势与论文报告的“上下文越长，加速越明显”一致，但本机 16K 结果仍略低于论文的 1.2--1.5x 范围。

32K 的完整 FP16 MHA KV 与模型权重无法同时装入 24GB 3090，因此另做了 Full/FIER 共同使用 NF4 double-quant 权重的容量实验：

| 历史长度 | NF4 Full | NF4 FIER release | FIER 速度比 |
|---:|---:|---:|---:|
| 8K | 48.541 ms/token | 58.142 ms/token | 0.835x |
| 16K | 48.381 ms/token | 58.585 ms/token | 0.826x |
| 32K | 49.548 ms/token | 58.878 ms/token | 0.842x |

这一组只能说明 bitsandbytes NF4 权重路径下，FIER selector 的约 9--10 ms/token 额外成本无法被 Attention 节省抵消；它不是论文 FP16/RTX 4090 的 32K 复现。

原始 JSON 位于远端：

- `/home/fdong/qksieve_iclr2027/results/20260809_fier_official_audit/full_8k_d256_r3.json`
- `/home/fdong/qksieve_iclr2027/results/20260809_fier_official_audit/fier_8k_b1280_d256_r3.json`
- `/home/fdong/qksieve_iclr2027/results/20260809_fier_official_audit/full_16k_d256_r3_chunk2k_nomask.json`
- `/home/fdong/qksieve_iclr2027/results/20260809_fier_official_audit/fier_16k_b1280_d256_r3_chunk2k_nomask.json`

## 与 QKSieve 的当前可比证据

同一张 RTX 3090 上的 MHA-shaped Attention 路径测试显示：

| 历史长度 | QKSieve Attention 速度比 | 项目内 FIER-style Attention 速度比 |
|---:|---:|---:|
| 8K | 1.227x | 0.953x |
| 16K | 1.566x | 1.098x |

该表只测一层完整 Attention 路径，不能与上表的整模型 decode 速度直接相除。它支持的窄结论是：在相同 MHA 形状和近似索引存储比例下，QKSieve 的 selector 扫描更短，因此 Attention 子系统的交叉点更早。

另用标准 Hugging Face LongChat 路径完成了 256-token QKSieve/Full 配对。稳态排除前 16 个生成 token；online 包含索引构建、首步初始化和全部 256 个生成 token，但不包含 prefill。

| 历史长度 | Full 稳态 | QKSieve 稳态 | 稳态速度比 | QKSieve 固定构建 | 256-token online 速度比 |
|---:|---:|---:|---:|---:|---:|
| 8K | 37.070 | 39.371 ms/token | 0.942x | 10.134 s | 0.444x |
| 16K | 52.316 | 35.824 ms/token | 1.461x | 7.302 s | 0.752x |

16K 的 QKSieve 稳态路径明显快于 Full，但固定成本仍未被 256 个输出 token 摊平。用固定成本加首步额外开销除以每个稳态 token 节省的时间，估计 break-even 约为 522 个生成 token。8K 下稳态本身就慢于 Full，因此不存在生成得更久即可摊平的 break-even。

这张表与 FIER 官方 backend 表使用不同模型实现和不同 PyTorch 版本，不能用原始毫秒直接判断谁更快。可以比较各自相对其 Full 的趋势：FIER release 在 16K 为 1.177x；QKSieve 在 16K 的稳态为 1.461x，但 256-token online 只有 0.752x。QKSieve 当前优势在稳态 selector/Attention，短板是一次性索引构建和首步初始化。

## 当前允许的结论

FIER 官方 CUDA backend 可以运行，修正同步与 Full 对照后，16K 已复现随长度增加而出现的 decode 加速趋势。但当前 release 的速度路径不能用于证明论文中 1-bit selector 的质量：decode 中先创建未初始化的 `past_key`，再调用 `bit=2`，而 `quantize_and_pack` 返回随机 packed code、scale 和 zero。这次运行因此是后端速度审计，不是“真实 FIER 质量与速度同时成立”的复现。

## 当前不允许的结论

- 不能把项目内 RTN-1 复现写成“FIER 官方实现”。
- 不能把 QKSieve 的 Attention 子系统 5.50x 与 FIER 的整模型 1.2--1.5x 直接相除。
- 不能用官方 release 的随机 packed selector 路径证明 FIER 的真实质量与速度。
- 不能把 FP16 8K/16K 结果外推成 FP16 32K 数值，也不能把 NF4 32K 的 0.842x 写成论文 FP16/RTX 4090 配置的复现。
