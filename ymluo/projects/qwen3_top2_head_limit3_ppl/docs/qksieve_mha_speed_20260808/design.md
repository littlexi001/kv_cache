# MHA 速度实验设计

## 问题

此前 QKSieve 主要在 Qwen3 的 GQA 上测速。GQA 的 Full Attention 只读取 8 个 KV heads，而 LLaMA-2 7B 的 MHA 需要读取 32 个 KV heads。若仍用 GQA 结果判断检索方法的系统价值，会低估减少 KV 读取量带来的收益。

可证伪假设：在相同的 32Q/32KV/128 维布局、相同 FP16 K/V 和相同 active-token 预算下，QKSieve 从 8K 起应快于 Full MHA，且加速比随历史长度增长。

## 物理先验与数学模型

单 token decode 的 Full MHA 读取量近似为：

$$B_{full}=2H_{kv}Nd\,s,$$

其中 $H_{kv}=32$、历史长度为 $N$、head 维度 $d=128$、FP16 元素大小 $s=2$ Byte，系数 2 表示 K 和 V。

QKSieve 的在线注意力成本写为：

$$T_{sparse}=T_{qproj}+T_{scan}(N)+T_{attn}(k),$$

其中 $k=\min(0.06N,1280)$。辅助索引约占完整 K/V 的 5.53%，但精确注意力仍从 GPU 常驻 FP16 K/V 中读取选中的 token。

## 实现约束

- MHA 层级基准必须直接使用 `[1,32,N,128]` 的 K/V，禁止 `repeat_interleave`。
- Full、QKSieve 和 FIER 共用相同的 FP16 K/V。
- QKSieve 包含 query 投影、量化索引扫描、候选压缩和精确稀疏 attention。
- 层级计时使用 CUDA Events，并在计时前后同步。
- 整模型使用 LongChat-7B-32K，其配置明确为 32 query heads、32 KV heads。
- 索引预计算与稳态 decode 分开报告。

## 数值鲁棒性

真实 MHA 首次运行发现一个 head 的协方差矩阵使 FP32 `eigh` 不收敛。通用求解器现采用：显式对称化、CPU FP64 重试、按矩阵尺度逐级加入 $10^{-10}$ 到 $10^{-4}$ 的对角正则。正常矩阵仍走原 FP32 路径，NaN/Inf 直接报错。

