# V8: Hierarchical Memory Update + End-to-End Timing

日期：2026-07-03

## 目标

这轮把原来只统计 `eval_seconds` 的 range-SDPA 实验，改成分阶段计时：

```text
ingest/page/index -> prefill -> calibration -> route -> range build -> typed record -> eval
```

这样可以同时报告三种速度口径：

- `eval_seconds`：原来的模型 forward 口径，只统计已有 KV 后的 typed record/query/answer scoring。
- `query_pipeline_seconds`：每次 query 的真实在线路径，包含 routing、token range 构建、typed record 构建/tokenize、range-SDPA eval。
- `end_to_end_seconds`：更保守口径，额外包含 page/index ingest、full context prefill、calibration。

## Memory 更新频率

本轮先按下面的策略落地到实验脚本：

| 层级 | 更新频率 | 实验字段 |
|---|---|---|
| sink | 固定保留 | `sink_tokens` |
| recent window | 每次 query 自然携带 | `recent_tokens` |
| L0 current page | 新 token / 新 turn 追加 | `l0_current_tokens` |
| L1 page | 自然段落边界，256-1024 tokens 封页 | `paragraph_min_tokens=256`, `paragraph_max_tokens=1024` |
| L2 section | 每 8 个 L1 page 合并 | `section_max_paragraphs=8` |
| L3 chapter/document | 每 64 个 L1 page 合并 | `chapter_max_pages=64` |
| global memory index | 每 128 个 L1 page 批量更新 | `global_index_update_pages=128` |

## 代码

修改：

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/src/run_longrange_book_index_sparse_eval.py
```

新增脚本：

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/scripts/run_range_sdpa_e2e_timing_v8_server.sh
```

新增输出列包括：

```text
l1_page_build_seconds
l1_page_index_seconds
l2_section_build_seconds
l2_section_index_seconds
l3_chapter_build_seconds
l3_chapter_index_seconds
global_index_seconds
ingest_index_seconds
prefill_seconds
calibration_seconds
route_seconds
range_build_seconds
text_verifier_seconds
typed_route_seconds
typed_record_build_seconds
typed_record_tokenize_seconds
query_pipeline_seconds
end_to_end_seconds
```

## 运行

服务器命令：

```bash
cd /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
STAMP=20260703_e2e_timing_v8_10k20k39k_x2 \
CONTEXT_TOKENS=10000,20000,39000 \
TASKS_PER_LENGTH=2 \
LAYOUTS=e05_d90,e20_d80,e35_d70 \
bash scripts/run_range_sdpa_e2e_timing_v8_server.sh
```

输出目录：

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/range_sdpa_e2e_timing_v8_20260703_e2e_timing_v8_10k20k39k_x2
```

配置：

```text
task_variant = chain_story_conflict
contexts = 10k, 20k, 39k
layouts = e05_d90, e20_d80, e35_d70
tasks = 6 per context per mode
modes = full, chain_typedhier_role_auto_p1
sparse_attention_impl = range_sdpa
typed_record_format = answerline_summary
typed_record_answer_override = true
skip_lm_answer_when_override = true
```

注意：这里的 full 和 typed 都走了 typed sidecar/answerline 配置，所以这个实验主要比较“同一 typed system path 下 full attention vs typed sparse range-SDPA”的速度，不是纯 direct-LM full baseline。

## 结果

| Context | Mode | Acc | Query PPL | Eval sec | Query pipeline sec | End-to-end sec | KV kept | Decoy hit |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 10k | full | 83.3% | 17.15 | 4.57 | 4.58 | 8.16 | 100.0% | 100% |
| 10k | typed range-SDPA | 83.3% | 21.15 | 3.70 | 3.71 | 7.29 | 10.33% | 0% |
| 20k | full | 100.0% | 15.71 | 6.63 | 6.64 | 13.05 | 100.0% | 100% |
| 20k | typed range-SDPA | 100.0% | 16.96 | 4.38 | 4.40 | 10.80 | 5.73% | 0% |
| 39k | full | 83.3% | 14.95 | 8.27 | 8.28 | 26.82 | 100.0% | 100% |
| 39k | typed range-SDPA | 83.3% | 16.98 | 3.98 | 4.00 | 22.54 | 2.81% | 0% |

速度：

| Context | Eval speedup | Query pipeline speedup | End-to-end speedup |
|---:|---:|---:|---:|
| 10k | 1.24x | 1.23x | 1.12x |
| 20k | 1.51x | 1.51x | 1.21x |
| 39k | 2.08x | 2.07x | 1.19x |

## 分阶段耗时

| Context | Mode | Ingest index | Prefill | Calibration | Route | Range build | Typed route | Typed record build | Eval |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 10k | typed | 0.018 | 1.931 | 1.628 | 0.0041 | 0.00002 | 0.0035 | 0.0005 | 3.699 |
| 10k | full | 0.018 | 1.931 | 1.628 | 0.0000 | 0.00001 | 0.0050 | 0.0005 | 4.572 |
| 20k | typed | 0.036 | 4.746 | 1.621 | 0.0068 | 0.00003 | 0.0059 | 0.0006 | 4.382 |
| 20k | full | 0.036 | 4.746 | 1.621 | 0.0000 | 0.00000 | 0.0075 | 0.0006 | 6.634 |
| 39k | typed | 0.069 | 16.845 | 1.621 | 0.0144 | 0.00002 | 0.0126 | 0.0005 | 3.976 |
| 39k | full | 0.069 | 16.845 | 1.621 | 0.0000 | 0.00000 | 0.0145 | 0.0005 | 8.269 |

## 分析

1. 分页和索引本身很便宜。
   10k/20k/39k 的 ingest index 分别约 `0.018s / 0.036s / 0.069s`。这是 CPU 侧 TF-IDF/page/section/chapter/global index 构建成本，远小于 prefill 和 eval。

2. routing 和 range build 几乎不是瓶颈。
   typed route 在 39k 也只有约 `0.014s`，range build 基本是微秒级。真正耗时仍然在模型 forward。

3. prefill 会显著压低端到端加速。
   39k 的 prefill 是 `16.845s`，typed 和 full 都要付同样成本，所以 query-side 可以有 `2.07x`，但 end-to-end 只有 `1.19x`。

4. 如果一个长文档/长期记忆会服务多次 query，ingest 和 prefill 应该摊销。
   这时更应该看 `query_pipeline_seconds`。如果每次都是一次性新文档新 query，则应该看 `end_to_end_seconds`。

5. 当前 typed path 的优势主要来自减少 eval attention 成本。
   KV kept 从 10k 的 `10.33%` 降到 39k 的 `2.81%`，所以 context 越长，eval/query-side speedup 越明显。

## 结论

这轮补齐了更公平的时间口径：

```text
原来的 2.1x 是 query/eval-side speedup，不是完整 end-to-end speedup。
把分页、路由、range build、typed record 都算进去后，query pipeline 仍然约 2.07x。
把 prefill 也算进去后，39k end-to-end speedup 约 1.19x。
```

下一步如果要让 end-to-end 也明显超过 full，需要继续做：

```text
prefill-time sparse path
KV page cache 复用
多 query amortized memory index
fused sparse prefill / range-SDPA prefill
```
