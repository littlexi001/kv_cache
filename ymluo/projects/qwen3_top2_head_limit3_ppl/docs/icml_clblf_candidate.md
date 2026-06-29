# ICML 候选方法：CLB-LF

## 结论先行

当前最值得继续投入的方向不是 `qabs8cand3attn`。后者本质上已经接近 SparQ/Quest 一类 query-aware retrieval。更有投稿潜力的方向是：

```text
CLB-LF: Calibrated Layer Budgeting with Landmark Fallback
```

一句话定义：先用少量校准 token 估计每层对远程上下文的依赖，再给不同层分配不同 attention budget；高远程依赖层保留 full attention，低敏感层使用 `recent window + landmark stride`，避免 recent-only 对 PPL 的破坏。

## 与已有工作的边界

必须避开的“低创新”包装：

- 不能把 `qabs8cand3attn` 包成新方法；它和 SparQ Attention 的 query-aware selective KV fetching 太接近。
- 不能只做 head-level retrieval/streaming；这和 RazorAttention、DuoAttention 的主线太接近。
- 不能只说 fixed local window；这类方法在长上下文 PPL 上很容易崩。

当前 CLB-LF 的可辩护创新点：

1. **层级预算而非 token/head 检索**：核心决策单位是 layer budget，不是在每个 query 上动态 top-k token。
2. **校准式 remote-mass 排序**：用短校准段得到每层远程注意力依赖强度，避免人工指定 bottom/top layers。
3. **landmark fallback 保留全局低频信息**：非 full 层不是 recent-only，而是 recent 加低频全局 landmark。
4. **可扩展到 layer-wise mixed budgets**：同一模型中不同层可使用 full、recent、landmark-r/s 等不同预算。
5. **可叠加置信度/位置感知 controller**：后续可以在 token level 调节压缩强度，但不把 novelty 建在 SparQ 式 query top-k 上。

## 关键实验事实

### War and Peace 80k / eval200

Baseline：`30.0984s / PPL 49.2224`

| 模式 | Decode 时间 | PPL | 相对速度 | PPL 差值 |
| --- | ---: | ---: | ---: | ---: |
| `fulll25landmarkr1024s64attn` | 29.5950s | 49.6204 | +1.67% | +0.3981 |
| `fulll25landmarkr2048s64attn` | 29.7186s | 49.2745 | +1.26% | +0.0522 |
| `fulll25landmarkr4096s64attn` | 29.8449s | 48.9016 | +0.84% | -0.3207 |
| `fulll25landmarkr2048s128attn` | 29.6978s | 49.0622 | +1.33% | -0.1601 |

### War and Peace 80k / eval1000

| 模式 | Decode 时间 | PPL | 相对速度 | PPL 差值 |
| --- | ---: | ---: | ---: | ---: |
| `baseline` | 151.1200s | 28.8883 | - | - |
| `fulll25landmarkr4096s64attn` | 150.3802s | 28.7829 | +0.49% | -0.1054 |

### Monte Cristo 80k / eval200

继续使用 War 20k 校准出的 layer map。

| 模式 | Decode 时间 | PPL | 相对速度 | PPL 差值 |
| --- | ---: | ---: | ---: | ---: |
| `baseline` | 30.0134s | 36.1664 | - | - |
| `fulll25landmarkr4096s64attn` | 29.9301s | 34.7897 | +0.28% | -1.3766 |

解释：PPL 改善不一定说明压缩“增强模型”，更可能是测试片段波动、attention 近似带来的正则化效应、或 baseline eager attention 数值路径差异。因此论文里不能只报一次 PPL 下降，必须做多数据、多 seed/区间验证。

## 当前不足

按 ICML 标准，目前证据还不够：

- decode 加速只有 `0.3%--1.3%`，工程收益偏弱；
- 只在 Qwen3-0.6B 上验证，attention 不是唯一瓶颈，层压缩收益容易被 MLP 和 Python overhead 吃掉；
- 目前安全压缩层大约只有 3 层，压到 4 层时 War 80k PPL 明显变差；
- 还没有证明 layer map 可以跨模型、跨长度、跨数据稳定迁移。

## 下一步实验门槛

要把 CLB-LF 提升到可投稿强度，至少需要满足下面任一组合：

1. **更大速度收益**：在 80k decode 上稳定超过 baseline `5%`，PPL 差不超过 `0.2`。
2. **更强质量收益**：在多个文本上 PPL 稳定不差于 baseline，同时速度稳定正收益。
3. **更大模型收益**：在 1.7B/4B/8B 上证明 attention 占比更高时收益显著放大。
4. **更低 overhead 实现**：把 layer fallback 路径做成 fused/compiled kernel，减少 Python 和 mask 构造开销。

## 服务器恢复后的优先实验

### 1. 收单层消融

```bash
cd /home/u21307130306/kvcache/kv_cache/ymluo/projects/qwen3_top2_head_limit3_ppl
find outputs/icml_war80_eval200_single_layer_lm4096 -name ppl_by_mode.csv | wc -l
```

如果数量达到 28，执行：

```bash
bash scripts/run_clblf_safe_layer_search_server.sh
```

该脚本会自动生成 `safe_top{k}_layers_last.json` 并测试 `fulll{28-k}landmarkr4096s64attn`。

### 2. 选择候选标准

优先保留满足以下条件的候选：

```text
War 80k/eval200: speedup > 2%, delta_ppl <= 0.2
War 80k/eval1000: speedup > 1%, delta_ppl <= 0.2
Monte 80k/eval200: speedup > 1%, delta_ppl <= 0.2
```

如果没有候选满足，需要转向更大模型或 kernel 优化；继续在 Qwen3-0.6B 上调参收益有限。

## 相关工作链接

- SparQ Attention: https://arxiv.org/abs/2312.04985
- Quest: https://arxiv.org/abs/2406.10774
- MInference: https://arxiv.org/abs/2407.02490
- RazorAttention: https://arxiv.org/abs/2407.15891
- DuoAttention: https://arxiv.org/abs/2410.10819
- TriAttention: https://arxiv.org/abs/2604.04921
