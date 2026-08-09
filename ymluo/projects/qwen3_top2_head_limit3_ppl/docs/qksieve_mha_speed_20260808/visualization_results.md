# MHA 速度结果

## 如何读表

`Full` 和 `QKSieve` 是单层单 token attention 的 CUDA 时间。加速比大于 1 表示 QKSieve 更快。该表已经包含 query 投影、索引扫描和精确稀疏 attention，但不包含 MLP 等非 attention 模块。

| 历史长度 | Full MHA | Query prepare | Selector scan | Sparse attention | QKSieve complete | QKSieve 加速 | FIER complete | FIER 加速 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8K | 0.1693 ms | 0.0338 | 0.0365 | 0.0423 | 0.1380 | **1.23x** | 0.1776 | 0.95x |
| 16K | 0.3186 ms | 0.0339 | 0.0470 | 0.0994 | 0.2034 | **1.57x** | 0.2902 | 1.10x |
| 32K | 0.6155 ms | 0.0339 | 0.0668 | 0.1244 | 0.2471 | **2.49x** | 0.4474 | 1.38x |
| 64K | 1.2033 ms | 0.0326 | 0.1167 | 0.1064 | 0.2762 | **4.36x** | 0.7636 | 1.58x |
| 128K | 2.3750 ms | 0.0316 | 0.2225 | 0.1560 | 0.4316 | **5.50x** | 1.4263 | 1.67x |

QKSieve 索引占完整 FP16 K/V 的 5.525%；FIER 对照索引为 6.25%。

## 真实 LongChat-7B，8K

| 路径 | 稳态 decode | 相对 Full | 备注 |
|---|---:|---:|---|
| Full MHA | 36.65 ms/token | 1.000x | 8-token smoke |
| FIER top-1280 | 36.94 ms/token | 0.992x | 8-token smoke |
| QKSieve 自动 split | 39.35 ms/token | 0.931x | 自动选到低效的 single-split kernel |
| QKSieve split=8 | 35.90 ms/token | 约 1.02x | 64-token run；尚缺同轮 64-token Full 配对 |

QKSieve 的一次性 QK 因子预计算为 6.07 秒。因此 64-token 在线延迟若包含预计算仍为 159.84 ms/token；该设置适合索引可复用的多轮会话，不适合只生成几十个 token 的一次性请求。

## 失败分析

默认 8K 路径的 CUDA stage profile 为：query prepare 3.15、retrieval 2.31、sparse attention 23.66、key append 0.39 ms/token。异常大头是 single-split 稀疏 attention，不是索引扫描。强制 split=8 后稳态从 39.35 降到 35.90 ms/token，但 profiler 复测期间服务器 SSH 失去响应，尚无有效的 split=8 stage breakdown。

当前源码已经把默认规则改为：`candidate_capacity <= 4096` 时使用 split=8；大缓冲再依据共享内存上限选择 4/8/16。该修改通过 91 个本地单元测试，远端默认路径复测仍待 SSH 恢复。

## 当前结论

原生 MHA 明显放大了 QKSieve 的 attention 子系统优势，并把层级交叉点提前到 8K。整模型 8K 只得到约 1.02x 指示性收益，因为非 attention 底座和一次性索引成本仍占主导。16K 及以上的真实模型交叉点尚未完成，不能用层级结果代替整模型数字。
