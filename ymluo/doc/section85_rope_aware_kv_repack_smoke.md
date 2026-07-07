# Section 85: RoPE-aware KV repack smoke

日期：2026-07-06

## 目的

验证当前方法是否可以从 prompt-rebuild 版本推进到真正 cache-native 版本，避免和 RAG 边界不清晰。

新增脚本：

```bash
ymluo/projects/learned_hierarchical_summary_memory/src/run_rope_aware_kv_repack_smoke.py
```

比较的方法：

```text
full_kv_cache
naive_kv_gather_absolute_query_pos
naive_kv_gather_compact_query_pos
rope_delta_repack_compact_query_pos
rope_delta_repack_shifted_query_pos
prompt_rebuild_selected_pages
```

其中 `rope_delta_repack_*` 是真正 cache-native 的关键对照：

```text
先从 full prefill cache 中 gather 选中 KV page；
然后把 cached key 从原始 RoPE position 旋转到新的 planned position；
value 不改；
query 使用对应的新 position 继续 decode。
```

## 实验输出

Qwen3-0.6B quick：

```text
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/rope_repack_smoke_qwen06b_quick_shift_20260706
```

Qwen3-8B，约 3.5k context，page=256：

```text
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/rope_repack_smoke_qwen8b_3k5_p256_shift_20260706
```

Qwen3-8B，4k context，page=512：

```text
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/rope_repack_smoke_qwen8b_4k_p512_shift_20260706
```

说明：8B 6k/page1024 单卡 3090 OOM，所以先用 3.5k/4k smoke 验证机制。

## 关键结果

### Qwen3-0.6B

single_exact：

| method | NLL | exact |
|---|---:|---:|
| full | 0.9847 | true |
| naive absolute | 3.7396 | false |
| naive compact | 3.5763 | false |
| RoPE compact | 0.8700 | true |
| RoPE shifted | 3.7396 | false |
| prompt rebuild | 0.8664 | true |

two_hop：

| method | NLL | exact |
|---|---:|---:|
| full | 1.3749 | true |
| naive absolute | 4.0525 | false |
| naive compact | 4.1047 | false |
| RoPE compact | 1.4131 | true |
| RoPE shifted | 3.7931 | false |
| prompt rebuild | 1.6638 | true |

结论：小模型上，RoPE compact repack 明显把 naive sparse KV gather 修回来了，接近甚至略优于 prompt rebuild。

### Qwen3-8B 3.5k/page256

two_hop：

| method | NLL | exact |
|---|---:|---:|
| full | 1.4359 | true |
| naive absolute | 1.2769 | false |
| naive compact | 1.4157 | false |
| RoPE compact | 1.3776 | true |
| RoPE shifted | 0.9317 | true |
| prompt rebuild | 1.6018 | true |

decoy_exact：

| method | NLL | exact |
|---|---:|---:|
| full | 0.5852 | true |
| naive absolute | 0.2815 | true |
| naive compact | 0.2786 | true |
| RoPE compact | 0.9974 | false |
| RoPE shifted | 0.2801 | true |
| prompt rebuild | 0.4233 | true |

结论：8B 上不是“compact 一招通吃”。two-hop 需要 RoPE repack 才把 exact 修回来；decoy case 里 shifted/absolute 更稳，compact 会破坏当前记录判断。

### Qwen3-8B 4k/page512

single_exact：

| method | NLL | exact |
|---|---:|---:|
| full | 0.8047 | true |
| naive absolute | 3.0175 | false |
| naive compact | 0.9405 | false |
| RoPE compact | 2.2629 | true |
| RoPE shifted | 3.0175 | false |
| prompt rebuild | 0.8713 | true |

decoy_exact：

| method | NLL | exact |
|---|---:|---:|
| full | 0.5841 | true |
| naive absolute | 1.1466 | false |
| naive compact | 0.8402 | false |
| RoPE compact | 1.0447 | false |
| RoPE shifted | 0.1878 | true |
| prompt rebuild | 0.3431 | true |

结论：page 变大后，single_exact 更偏向 compact repack，decoy_exact 更偏向 shifted repack。这说明位置规划本身应成为方法的一部分。

## 对 ICLR 方法的含义

这个 smoke 给出了一个比 prompt/RAG 更清楚的主线：

```text
不是把 chunk 重新拼回 prompt；
而是在 full prefill 后，对 selected KV pages 做 position-aware cache repack。
```

关键发现：

```text
1. naive sparse KV gather 确实不够，很多 case 会失败；
2. RoPE delta repack 能在 single/two-hop 上修复 naive gather；
3. compact repack 和 shifted repack 各有适用场景；
4. 因此最终方法不应该只做 top-k page selection，还要做 position-mode planning。
```

建议把论文方法升级成：

```text
Risk-aware typed KV cache planner
= page/action planner + position-mode planner + safety ladder
```

position-mode planner 至少包含：

```text
absolute / shifted / compact
```

更准确地说：

```text
single lookup / bridge reasoning: 倾向 compact 或 shifted；
decoy / current-vs-obsolete / order-sensitive evidence: 倾向 shifted 或 absolute；
不确定时升 k 或 full fallback。
```

这样和 RAG 的边界就清晰了：

```text
RAG: retrieve text chunks -> re-prefill selected text prompt
ours: retrieve/plan KV pages -> RoPE-aware KV repack -> continue decoding without re-prefilling raw text
```

## 下一步

下一步应实现一个 `position-mode planner`：

```text
输入：router features + retriever stability + task family + selected page layout
输出：compact / shifted / absolute / full fallback
```

训练标签可以直接来自这次 smoke 的扩展版本：

```text
对每个 case 同时跑 compact、shifted、absolute；
把 exact 成功且 NLL 最低的 mode 标成 oracle；
如果所有 sparse mode 都失败，则 full fallback。
```

目标不是证明一个固定 repack 规则万能，而是证明：

```text
learned position-aware KV planning 可以把 prompt-rebuild 的质量迁移到 cache-native 系统。
```
