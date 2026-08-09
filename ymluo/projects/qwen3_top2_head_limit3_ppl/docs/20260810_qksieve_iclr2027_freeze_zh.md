# QKSieve ICLR 2027 方法冻结与复现合同

## 1. 冻结决定

从 2026-08-10 起，停止修改论文主方法的数值定义。后续工作只允许：

1. 修复不改变输出定义的实现错误；
2. 优化与现有输出逐元素等价的 kernel；
3. 补齐 benchmark、模型、硬件、统计检验和论文图表。

冻结实现提交：

```text
328e01718deebfdfc80dbd8e588a1a95a1832b59
```

机器可读配置：

```text
configs/qksieve_robust_iclr2027_frozen_20260810.json
```

## 2. 冻结主方法

论文主方法为 **QKSieve-Robust**。QKSieve-Fast 只作为去除 ValueSketch 的速度/质量消融，不作为通用高保真结论。

### 2.1 Key 检索

1. 每层、每个 KV head 根据当前请求构造 QK-balanced 双正交坐标。
2. 将 128 维 Key 分成 8 个连续 16 维 band。
3. 在 240 bit/token/KV-head 的硬预算下，用 Query-weighted MSE 分配 `0/1/2/4/8` bit。
4. 每个 decode Query 在同一低比特坐标中生成 proxy score。
5. 最多抽取 512 个规则分层位置估计目标分位数，再单遍扫描 packed index 并直接写候选。

每个 Query head 的候选预算固定为：

```text
B(N) = min(N, 1280, max(256, ceil(0.06 N)))
```

候选选出后，不做 exact-QK rerank，直接读取候选位置的原始 FP16/BF16 K/V，计算精确 sparse QK-softmax-AV。

### 2.2 遗漏 Value 补偿

只在候选上做 attention 会丢失大量单个权重很小、但聚合后方向一致的 Value。Robust 在同一次 Key-index 扫描中额外估计未选集合的 softmax 分母和低秩 Value 分子：

```text
Z_tail = sum_{i not in S} exp(score_proxy_i - threshold)
c_tail = sum_{i not in S} exp(score_proxy_i - threshold) * c_i
```

其中 Value 使用 request-local rank-16 PCA，系数按 256-token block 做 affine INT4。最终输出为：

```text
o = [u_selected + alpha * exp(threshold - m) *
     (Z_tail * value_mean + value_basis * c_tail)]
    /
    [Z_selected + alpha * exp(threshold - m) * Z_tail]

alpha = 0.5
```

`alpha=0.5` 来自预注册弱例，随后在六个未见 seed 上冻结验证。它不是 router，也不根据任务或长度调整。

### 2.3 明确禁用

- 无学习式 router；
- 无任务规则；
- 无 Full Attention fallback；
- 无 sink/recent token 特判；
- 无 exact-QK rerank；
- 无 64K 方法切换；
- 不把 oracle 或逐样本最优动作作为可部署结果。

## 3. 存储

| 配置 | 辅助索引 / 完整 FP16 K+V | 用途 |
|---|---:|---|
| QKSieve-Fast | 约 5.86% | 去除 ValueSketch 的消融 |
| QKSieve-Robust | 约 7.47% | 论文主方法 |

上述比例只统计逐 token 辅助索引。论文的 GPU-resident 速度路径仍保存完整精确 K/V；QKSieve 减少的是每步 attention 读取和计算，不宣称在该路径删除完整 KV。

## 4. 已有质量证据

| 测试 | Fast | Robust | 说明 |
|---|---:|---:|---|
| 32K 三条连续文本 pooled PPL 保持率 | 98.578% | 100.517% | 同 selector、预算和文本 |
| 96K 三条连续文本 pooled PPL 保持率 | 95.099% | 99.768% | Robust 修复最差文本 86.892% 到 100.296% |
| 原生 128K 十二个 topic/seed case | 96.319%（六个同 seed） | 99.805% | Robust 含六个 held-out seed；最差 96.980% |

完整 3,750 样本 LongBench 的 99.881% 来自冻结前的 reference selector 路径，不能与 Robust MHA 速度拼成同路径主结论。投稿前必须补做冻结 Robust 的完整或严格预注册同路径质量表。

## 5. 已有 MHA 速度证据

RTX 3090，MHA 32Q/32KV，head dimension 128，原始 K/V 常驻 GPU。所有数值均包含 Query 准备、低比特扫描、候选压缩和 sparse attention；Robust 还包含 Value-tail 扫描与 consumer。

| 历史长度 | Fast attention | Robust attention |
|---:|---:|---:|
| 8K | 1.27x | 1.08x |
| 16K | 1.67x | 1.41x |
| 32K | 2.65x | 2.09x |
| 64K | 4.55x | 3.36x |
| 128K | 6.37x | 4.12x |

真实 `Yarn-Llama-2-7B-128K` 稳态 greedy decode：

| 历史长度 | Full | Fast | Robust | Fast 加速 | Robust 加速 |
|---:|---:|---:|---:|---:|---:|
| 32K | 84.18 ms | 51.56 ms | 63.97 ms | 1.63x | 1.32x |
| 64K | 144.75 ms | 52.97 ms | 65.25 ms | 2.73x | 2.22x |
| 128K | 268.17 ms | 55.41 ms | 67.38 ms | 4.84x | 3.98x |

Robust 的一次性索引构建为 1.375/1.488/1.839 秒，32/64/128K 的 break-even 为 69/19/10 个生成 token。64-token 冷请求计入构建后为 0.81/1.31/2.15x。多轮问答、agent 或共享前缀复用只支付一次构建成本，随后使用稳态速度。

## 6. 复现入口

MHA attention A/B：

```bash
bash scripts/launch_mha_valuesketch_attention_ab_20260809.sh
```

真实 MHA decode 严格 A/B：

```bash
bash scripts/launch_qksieve_mha_valuesketch_decode_ab_strict_20260809.sh
```

冻结 `alpha=0.5` 的封口验证：

```bash
bash scripts/launch_qksieve_robust_freeze_validation_20260810.sh
```

相关单元测试：

```bash
python -m pytest -q \
  tests/test_head_top2_targeted_ppl.py \
  tests/test_qksieve_tail_resolution_sample_count_20260730.py \
  tests/test_qksieve_valuesketch_workspace_20260804.py
```

当前结果：112 passed。

## 7. 投稿前只补证据，不再改方法

按优先级执行：

1. 冻结 Robust 的同路径 LongBench 与 RULER；
2. Llama/Qwen/Mistral 跨模型质量；
3. persistent KV 的 cold、warm、shared-prefix、append-only 四种协议；
4. 同 kernel 入口的 FIER 公平速度；
5. H100 64K/128K attention、decode、请求级速度；
6. allocation、sample count、Value rank/bit/block/alpha 消融；
7. bootstrap CI、最差 case、显存和失败边界；
8. 最终论文表格、图、页数和 claim audit。

若某项实验失败，应如实报告或降低 claim，不能重新搜索主方法参数后把测试集结果当作冻结方法。
