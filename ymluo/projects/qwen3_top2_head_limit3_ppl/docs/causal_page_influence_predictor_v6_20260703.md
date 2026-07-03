# Causal Page Influence Predictor V6（2026-07-03）

## 目标

这次实验处理两个问题：

1. 让 3090 环境下之前不能公平跑的官方 KV 方法至少进入可运行 smoke：
   - H2O：原实现 prefix 阶段构造完整 attention matrix，长上下文 OOM。
   - AdaKV：环境没有 `flash_attn`，并且还缺 `nvtx` / `tiny_api_cuda`。
   - Quest：仓库里只有工具函数，没有接入 runtime monkeypatch。
2. 把我们的 typed/page scorer 从 lexical/entity heuristic 升级成 teacher-distilled 的 causal page influence predictor。

核心变化是：不再只问“query 和 page 字面/embedding 像不像”，而是用 full-context teacher 的目标 token 做标签，估计“加入这个 page 后，teacher target 的 NLL 降低了多少”。

## KVCache-Factory runtime patch

远端位置：

```text
/home/fdong/ymluo/external/KVCache-Factory
```

本地保存的可复现 patch：

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/patches/kvcache_factory_runtime_patch_v6_20260703.patch
```

已处理：

| 方法 | 原问题 | 本次处理 |
| --- | --- | --- |
| H2O | 36k narrativeqa 在 3090 上 prefix score OOM | 改为 query-chunk 累加 H2O attention mass，不一次性创建 `[heads, q_len, k_len]` |
| Quest | 有 page min/max scorer，但没有 generation runtime | 新增 `--method Quest`，接入 Llama SDPA monkeypatch，支持 `kv_head` |
| AdaKV | 无 `flash_attn`，且 flat-cache update 依赖 `nvtx/tiny_api_cuda` | 新增 AdaKV SDPA fallback；flat per-head cache update 增加纯 PyTorch fallback |

8B smoke 输出：

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/kvcache_factory_h2o_chunk_smoke_20260703
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/kvcache_factory_quest_smoke_20260703
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/kvcache_factory_adakv_sdpa_smoke_20260703
```

注意：AdaKV fallback 能跑，但没有 flash-attn 时速度不代表官方优化实现上限；H2O chunked score 是等价避免 OOM，但会更慢。

## Causal page influence predictor

新增代码：

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/src/run_causal_page_influence_predictor_v6.py
ymluo/projects/qwen3_top2_head_limit3_ppl/scripts/run_causal_page_influence_predictor_v6_server.sh
```

方法流程：

1. 用 full KV 生成 teacher answer。
2. 取 teacher 生成的前 `target_tokens` 作为蒸馏目标。
3. 先只保留 sink + recent，计算目标 token 平均 NLL，得到 `base_nll`。
4. 对候选 page 逐个加入 sink + recent，重新计算目标 token NLL。
5. 标签定义为：

```text
label_delta_nll = base_nll - page_nll
```

值越大，说明这个 page 对恢复 full-context teacher logits 越有因果贡献。

6. 用 page 特征训练 ridge predictor：

```text
lexical / entity / structural / coverage
semantic mean embedding cosine
late-interaction MaxSim
page position / length / sink / recent
old heuristic score
```

7. 推理时用 predictor 给所有 page 排序，按预算 gather KV page，然后走同一套 sparse KV generate。

## 8B 结果

模型：

```text
/home/fdong/qwen/LlaMa-3.1-8B
```

配置：

```text
tasks = qasper, hotpotqa, passage_retrieval_en
budget = 512 context tokens
sink = 64
recent = 256
page = 256
max_context = 8192
max_new = 32
prompt_wrapper = llama3
attn = sdpa
```

### 1-shot smoke

输出：

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/causal_page_influence_predictor_v6_1shot_b512_20260703_smoke8b
```

| method | score | online sec | kept prefix | keep frac |
| --- | ---: | ---: | ---: | ---: |
| full_kv | 0.1629 | 1.2573 | 6786.7 | 1.000 |
| heuristic_page_gather | 0.0833 | 0.9958 | 560.7 | 0.0948 |
| causal_ridge_page_gather | 0.1913 | 0.9980 | 560.7 | 0.0948 |
| causal_label_oracle | 0.1629 | 0.9965 | 560.7 | 0.0948 |

### 2-shot smoke

输出：

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/causal_page_influence_predictor_v6_2shot_b512_20260703_2shot8b
```

| method | score | online sec | kept prefix | keep frac |
| --- | ---: | ---: | ---: | ---: |
| full_kv | 0.0815 | 1.2627 | 6675.8 | 1.000 |
| heuristic_page_gather | 0.0536 | 0.9914 | 560.7 | 0.100 |
| causal_ridge_page_gather | 0.1075 | 0.9923 | 560.7 | 0.100 |
| causal_label_oracle | 0.0430 | 0.9922 | 560.7 | 0.100 |

按任务看，causal ridge 的主要收益来自 HotpotQA：heuristic 在一个样本里选到高 lexical/late-interaction 但无因果帮助的页，causal predictor 选到了包含 `Miller v. California` 证据的页，F1 从 0 提升到 0.316，并超过 full KV 该样本的 0.231。

## 解释

这个结果说明两件事：

1. heuristic scorer 容易被“看起来相关”的 page 误导。
   例如 HotpotQA 里，lexical 和 late-interaction 高的页不一定会降低 teacher target NLL；teacher delta-NLL 能识别“真的让答案 logits 变好”的 page。

2. causal predictor 有发展价值。
   在 2-shot smoke 里，causal ridge 平均分数高于 full KV 和 heuristic，同时只保留约 10% prefix token，online decode-side 时间约低 21%。但这不是端到端快于 full，因为当前仍然先做 full prefix prefill，再 gather KV。

## 限制

这些数字不能作为最终论文结论：

- 当前 ridge 是同一批 sampled pages 上训练和评测，属于 in-sample smoke。
- 每个样本只测 8-10 个 page label，不是全 page oracle。
- `causal_label_oracle` 只在被测 page 里做 upper bound，所以它可能低于 ridge；它不是全页 oracle。
- passage retrieval 里 full KV 本身也失败，所以该任务当前不能说明压缩方法优劣。
- 速度口径是 online decode-side，不是 end-to-end；端到端要等 prefill-time compression 或 fused sparse prefill。

## 下一步

1. 做 held-out split：用 train sampled IDs 标 causal labels，eval IDs 只用 predictor。
2. 扩大标签覆盖：每样本测更多 page，或者用分层采样覆盖高 heuristic、低 heuristic、均匀位置。
3. 把 label 从 teacher generated token 扩展到 answer/reference token，避免 teacher 本身答错时蒸馏错误模式。
4. 接 KVCache-Factory 官方 sweep：FullKV / StreamingLLM / SnapKV / PyramidKV / H2O / AdaKV / Quest / ours causal page predictor，同 sampled IDs、同预算。
5. 做真正速度闭环：保留 online latency，同时新增 end-to-end latency；我们的最终目标还是 prefill 阶段也跳过无关 page。
