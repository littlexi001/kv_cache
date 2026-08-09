# 实验方案

## 条件

- GPU：NVIDIA RTX 3090 24GB。
- PyTorch：2.7.1+cu118。
- 层级布局：batch=1，32 Q heads，32 KV heads，head_dim=128。
- 长度：8K、16K、32K、64K、128K。
- active tokens/head：8K 和 16K 取 6%；32K 及以上最多 1280。
- QKSieve 索引：mixed-block、PCA/QK-balanced 坐标、低位编码；本实验只测在线读取与计算，不计离线生成随机速度张量的时间。
- FIER 对照：项目内 RTN-1 group-32 packed selector；它不是官方仓库端到端复现，因此仅作为同环境工程 A/B。

## 指标

1. Full MHA SDPA 的每层毫秒数。
2. query prepare、selector scan、精确稀疏 attention 的独立 CUDA 时间。
3. 完整注意力路径加速：`Full SDPA / sparse complete`。
4. LongChat-7B 整模型稳态 decode 的 ms/token。
5. QKSieve 一次性 QK 因子预计算时间和包含预计算的短生成延迟。

## 通过条件

- 层级：QKSieve 在 8K 快于 Full，且 32K 以上超过 2x。
- 整模型：必须在同一模型和同长度上快于 Full，才能宣称 decode 加速。
- 任何使用不同 decode 步数、不同 split 或不同进程的比值只能标为指示性结果。

## 失败条件

- Full 路径复制 GQA K/V。
- 只报告 selector，不包含精确稀疏 attention。
- 把索引预计算隐藏到稳态数字中，或反过来把一次性成本均摊到未实际生成的 token 上。
- CUDA profiler 自身导致运行挂起时，不使用该轮总延迟。

