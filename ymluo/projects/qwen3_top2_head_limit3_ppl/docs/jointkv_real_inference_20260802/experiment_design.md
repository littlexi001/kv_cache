# JointKV 真实推理实验设计

## 实验设置

| 项目 | 设置 |
|---|---|
| 模型 | Qwen3-0.6B，28 层，16 Query heads，8 KV heads，head dim 128 |
| 硬件 | NVIDIA RTX 3090，单卡运行每个条件 |
| 数值类型 | FP16 模型、FP16 exact K/V、126-bit 逻辑辅助索引 |
| 位置编码 | 模型原生 post-RoPE Q/K，`rope_theta=1e6` |
| prefill | Full dense SDPA，各条件一致 |
| decode | teacher-forced 自回归，每一步更新真实 HF DynamicCache |
| fallback | 无 |
| tail correction | 无 |
| 阈值样本 | 512 个均匀位置，按 Query head 独立 |
| exact suffix | prefill 后的既有 decode token 全部保留，当前 token 作为 self |

测试文本为未参与本次运行调参的长程事实文本：生物医学和编译器优化。8K 使用连续
128 个 teacher-forced token；32K 使用生物医学文本连续 64 个 token。当前数据仍是
机制 probe，不等同于 LongBench/RULER 或第二模型验证。

## 条件

`overfetch` 控制样本阈值对应的候选比例：

- `1x`：目标约 `min(6% * N, 1280)`。
- `2x`：阈值目标扩大 2 倍，用于高质量点。
- 32K 额外测 `3x/4x`，检验增加候选是否能消除多步质量下降。

workspace 至少为目标候选的 3-8 倍。正式结果要求所有层、head 的 overflow 总数为 0。

## 指标

质量指标：

- `Full PPL` 与 `Sparse PPL`：连续 teacher-forced token 的几何困惑度。
- `质量保持率 = exp(Full NLL - Sparse NLL)`；100% 表示两者 PPL 相同。
- `Top-1 一致率`：Sparse 与 Full 下一 token argmax 相同的比例。
- `Full-to-Sparse KL`：Full 概率分布到 Sparse 分布的平均 KL。

速度指标均由同步 CUDA event 直接测量：

- `Full/Sparse decode ms/token`：完整模型 forward，不做阶段延迟相加。
- `稳态 decode 加速 = Full ms/token / Sparse ms/token`。
- `一次性在线时间 = dense prefill + 索引构建 + 所有 decode 步`。
- 单层阶段测试分别直接测 query/LUT、selector、exact sparse attention 和完整路径。

## 判定标准

理想通过条件：连续多步质量保持率不低于 99.5%，Top-1 不低于 99%，且稳态 decode
加速大于 1。任何 workspace overflow、Full/Sparse 非配对、autograd 开启或随机索引
都使结果无效。

当前实验的作用是验证真实接入并找出失败阶段，不因为某个速度点通过就把整个方法
判定为完成。

## 结果位置

主要 JSON 位于：

`results/20260802_jointkv_real_inference/`

正式长步结果文件名包含：

- `hf_biomed_8k_e128_s512_over1.json`
- `hf_biomed_8k_e128_s512_over2.json`
- `hf_compiler_8k_e128_s512_over1.json`
- `hf_compiler_8k_e128_s512_over2.json`
- `hf_biomed_32k_e64_s512_over1.json` 至 `over4.json`
- `layer0_biomed_32k_s512_over3_cap6_profiled.json`
