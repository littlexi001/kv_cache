# 第 45 节：分层 KV 摘要 PPL 小规模验证

日期：2026-07-03

## 0. 实验目标

本节直接验证用户提出的分层 KV cache 想法：

```text
10k-token block 摘要 -> 1k-token 摘要 -> 100-token 原始叶子块 -> 原始 KV attention
```

实验对象是 Qwen3-0.6B 在 War and Peace 上的普通续写 PPL。这里没有把文本摘要重新拼回 prompt，而是在 KV 层做分层路由：

```text
每个摘要是固定 token range 上的 mean K-cache vector。
decode 时用当前 query vector 做路由：
q -> top 10k block -> top 1k mid ranges -> top 100-token raw leaf ranges
```

最终 attention 实际看到的是：

```text
sink64 + recent512 + selected raw 100-token leaf KV + self
```

也就是说，摘要只用于路由；真正参与最终 attention 的仍然是被召回的原始 K/V token。

## 1. 服务器运行

服务器：

```text
fdong@10.176.37.31
```

脚本：

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/src/run_hierarchical_kv_summary_ppl.py
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/scripts/run_hierarchical_kv_summary_ppl_server.sh
```

服务器输出目录：

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/hierarchical_kv_summary_ppl_20k512_20260703
```

本地同步目录：

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/hierarchical_kv_summary_ppl_20k512_20260703
```

主要配置：

```text
model = /home/fdong/hrj/prove/Qwen3-0.6B
text = data/war_and_peace_pg2600.txt
prefill_tokens = 20000
eval_tokens = 512
block_tokens = 10000
mid_tokens = 1000
leaf_tokens = 100
sink_tokens = 64
recent_tokens = 512
route_refresh_tokens = 16
dtype = float16
attention = eager
```

## 2. 实验结果

| 方法 | PPL | Loss | 平均 attention token 数 | 相对 full token 比例 |
| --- | ---: | ---: | ---: | ---: |
| full_baseline | 16.505 | 2.804 | 20000 | 100.00% |
| recent512 | 19.990 | 2.995 | 577 | 2.89% |
| recent1024 | 18.946 | 2.942 | 1089 | 5.45% |
| hierkv_1x1 | 19.147 | 2.952 | 665 | 3.33% |
| hierkv_2x2 | 18.812 | 2.934 | 955 | 4.78% |
| hierkv_2x4 | 18.013 | 2.891 | 1345 | 6.73% |

其中 `hierkv_2x2` 用的 attention token 比 `recent1024` 更少，但 PPL 更低：

```text
recent1024:
  PPL = 18.946
  avg attention tokens = 1089

hierkv_2x2:
  PPL = 18.812
  avg attention tokens = 955
```

增加被召回的 raw leaf 数量后，质量继续提升：

```text
hierkv_1x1:
  PPL = 19.147
  avg attention tokens = 665

hierkv_2x2:
  PPL = 18.812
  avg attention tokens = 955

hierkv_2x4:
  PPL = 18.013
  avg attention tokens = 1345
```

这个趋势支持核心方向：

```text
query-dependent hierarchical KV summary
比单纯扩大 recent window 更能以相近 token budget 找到有用远程上下文。
```

## 3. 重要限制

当前端到端耗时不能作为最终速度结论。这个实现只是 Python attention patch：

```text
full_baseline seconds = 17.17
recent512 seconds = 28.14
hierkv_2x2 seconds = 83.71
```

`hierkv` 当前更慢，原因是：

```text
1. Python 里逐层、逐 head 做路由。
2. 每个 decode token 都要做 summary score 和 gather。
3. 没有 fused range/block-sparse kernel。
4. 当前测试关注质量/token budget，不关注最终 latency。
```

因此，本节的有效信号是：

```text
在相似 attention-token budget 下，分层 KV 路由是否比 fixed recent window 更好。
```

而不是：

```text
当前 Python 原型是否已经更快。
```

## 4. 当前解释

正向证据：

```text
1. 固定 recent window 不是 Pareto-optimal。
2. 10k/1k/100 的 K-cache mean 摘要能召回有用远程 raw KV。
3. 在这个 smoke 中，召回更多 raw leaf 后 PPL 单调下降。
4. hierkv_2x2 用更少 token 击败 recent1024。
```

仍未证明的部分：

```text
1. 学出来的摘要或生成式摘要是否会优于 mean-K 摘要。
2. 这个趋势是否能扩展到 QA、RULER、LongBench，而不只是普通 LM continuation。
3. 接入 fused range/block-sparse kernel 后是否有真实 latency speedup。
4. 路由粒度应该是 per-head、per-KV-head、layer group，还是 task-level planner。
```

## 5. 下一步

建议下一步优先做三件事：

```text
1. 把这个 fixed hierarchy 跑到更长上下文，例如 39k prefill。
2. 在 RULER / synthetic QA 上测 answer accuracy，而不只测 PPL。
3. 把 selected leaf ranges 接到已有 range_sdpa / KV gather 路径，减少 Python loop。
```
