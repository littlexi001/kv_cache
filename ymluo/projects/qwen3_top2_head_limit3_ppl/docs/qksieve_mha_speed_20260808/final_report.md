# QKSieve 在原生 MHA 上的速度结论

> 本文档已由 `docs/qksieve_mha_final_freeze_20260809/final_report.md` 取代。冻结后的正式结果为 8K--128K `1.22x/1.64x/2.66x/4.48x/6.23x`；以下内容保留为优化前记录。

## 结论

在 LLaMA-2 7B 同形状的原生 MHA attention 上，QKSieve 已经显示出清楚的长度扩展优势：8K 为 1.23x，32K 为 2.49x，64K 为 4.36x，128K 为 5.50x。这个结果不含 GQA KV 复制，也包含 query 投影、低位索引扫描和精确稀疏 attention。

在真实 LongChat-7B 的 8K decode 上，优化 split 后目前是 35.90 ms/token，Full smoke 为 36.65 ms/token，约 1.02x。由于两者 decode 步数不同，这只能作为指示性结果，论文正式表必须补同轮重复实验。

## 为什么 MHA 比 GQA 更适合当前方法

Full MHA 每层读取 32 个 KV heads；Qwen3 GQA 的 Full 路径只读取 8 个 KV heads。QKSieve 的 selector 也随 KV head 数增长，但压缩索引每 token/head 约为完整 K/V 的 5.53%，因此 MHA 中 Full 基线增加的显存流量远大于 QKSieve 的索引流量。长度增长后，固定 query 投影成本不变，QKSieve 的优势继续扩大。

## 工程发现

1. MHA 支持不是简单改 `num_key_value_groups=1`。旧的 ValueSketch 快路径硬编码 GQA4，必须关闭 ValueSketch并走通用 selector。
2. 32 个独立 KV heads 更容易出现病态协方差。新增 FP64 与尺度相关对角正则后，真实模型可稳定建索引。
3. 自动 sparse-attention split 规则不适用于 MHA。8K 约 492 candidates 时 single split 占 23.66 ms/token；split=8 才恢复盈利。
4. 6 秒的 QK 因子预计算不能在短回答中均摊。论文应把“可复用索引的多轮 Agent 场景”和“一次性请求”分开报告。

## 声明边界

- 已证明的是 RTX 3090 上原生 MHA 形状的 attention 子系统速度，以及 LongChat-7B 8K 的可运行性。
- 尚未证明 LongChat-7B 在 16K/32K 的整模型加速，也尚未完成 MHA 质量评测。
- FIER 数字来自项目内 packed RTN-1 实现，不应表述为官方 FIER 复现结果。
- 当前最有价值的下一项实验是同卡、同进程设置、三次重复的 8K/16K Full 与 QKSieve split sweep；不是继续增加随机长度点。

当前源码已把候选容量不超过 4096 时的默认路径改为 split=8，并通过 91 个单元测试；远端性能复测尚未完成。

## 复现入口

- 层级脚本：`src/benchmark_qksieve_fier_mha_speed_20260808.py`
- 整模型脚本：`src/run_qksieve_fier_autoregressive_speed_20260808.py`
- 远端层级结果：`/home/fdong/qksieve_iclr2027/results/20260808_mha_speed/mha_layer_full.json`
- 远端真实结果目录：`/home/fdong/qksieve_iclr2027/results/20260808_mha_speed`
