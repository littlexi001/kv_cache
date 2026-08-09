# Section 125: m50 激进压缩与输出契约 Router 记录（2026-07-10）

## 目标

继续逼近实际可用目标：

- KV keep: 10%-30%
- online speedup: 2.5x 左右
- LongBench m50 score: 达到 full KV baseline 的 95%+

m50 full baseline：

| 方法 | Score | KV keep | Online | 95% 阈值 |
|---|---:|---:|---:|---:|
| full_raw m50 | 0.371970 | 100.00% | 3.283s | 0.353371 |

## 已完成 m50 结果

| 方法 | Score | KV keep | Online | 判断 |
|---|---:|---:|---:|---|
| v162 tree-router mid recovery | 0.355825 | 41.40% | 1.853s | 过 95%，但 KV 仍高 |
| v163 conditional rescue | 0.344165 | 35.72% | 1.410s | 未过 95% |
| v164 aggressive conditional | 0.315095 | 22.91% | 1.296s | 质量崩 |
| v165 budget ladder aggressive | 0.322332 | 26.67% | 1.330s | ladder 阈值不够稳 |
| v166 v163 passage trim | 0.316156 | 31.41% | 1.273s | PassageRetrieval 被砍坏 |

## 关键结论

1. 当前唯一过 m50 95% 的是 v162，但它依赖 RepoBench-P 和 TREC 全任务 full fallback，因此 KV keep 达到 41.40%，不满足 10%-30% 的目标。
2. v163 说明把 RepoBench/TREC 从 full fallback 释放出来可以把 KV 降到 35.72%，但 score 掉到 0.344165，主要损失来自 GovReport、RepoBench-P、TREC 和 MuSiQue。
3. v166 关闭 PassageRetrieval output fallback 后整体 m50 掉到 0.316156，说明 PassageRetrieval 不能直接释放 verifier。
4. PassageRetrieval retry budget probe 显示 3072/4096/6144 都不划算：质量约 0.627-0.700，但 KV keep 已经 81%-93%，接近 full。

## 新增方法点：Output-Contract Router

LongBench 的主要瓶颈不只是“找不到 block”，还有压缩后输出格式失稳：

- TREC 会泄漏 few-shot 示例格式，如继续输出 `Question:` / `Type:`。
- RepoBench-P 会输出解释、markdown fence 或 “I'll help...” 这类自然语言模板。

因此新增一个 narrow output-contract router：

- 只检测明显的 prompt leakage / natural-language code failure。
- 不判断语义是否正确，不使用 oracle label。
- 触发后先中等预算 retry；retry 仍不满足格式时才 full fallback。

这条线比整任务 full fallback 更符合方法故事：不是 RAG 检索，而是 KV 压缩后的 action safety controller。

## 正在跑的实验

| 实验 | 目的 |
|---|---|
| v170 | v163 + GovReport 不截断 + TREC selective full |
| v171 | v170 + RepoBench gap3<=0.0479 selective full |
| v172 | v171 但关闭 PassageRetrieval output full fallback，测低 KV 下界 |
| v173 | v172 + RepoBench gap3<=0.1454，更强 RepoBench 恢复 |
| v174 | v171 + output-contract verifier for TREC/RepoBench |
| v175 | output-contract release gates：减少 Repo/TREC 预解码 full gate，主要靠输出契约触发 |

同时启动了 task-only probe：

| Probe | Tasks |
|---|---|
| v174_contract_probe | TREC + RepoBench-P |
| v175_contract_probe | TREC + RepoBench-P |

## 下一步

1. 等 v170-v175 m50 完整结果；如果 v174/v175 的 Repo/TREC probe 有效，把 output-contract router 作为主线。
2. 如果 v174/v175 仍不能过 0.353371，下一步训练样本级 router，而不是继续 task-level policy。
3. 目标仍是 m50/m100 上达到 score >= 95% full，同时 KV keep <= 30%；当前证据显示还未完成。

## 2026-07-10 追加：更激进压缩的新方向

当前 RULER 表现已经足够稳定，LongBench 的问题不是统一预算不够低，而是少数任务需要不同动作：

- TREC：原来 v162 依赖 full KV 才有 0.70 分数，KV keep 为 100%。新加入 short-prompt LLM classification head，只给模型类别集合、当前问题和少量近邻 few-shot 示例，不再保留长上下文 KV。
- PassageRetrieval：新加入 direct structured head，从已选 block 中直接输出 `Paragraph k`，避免 output verifier 把 KV 拉到 80% 以上。
- HotpotQA/MuSiQue：历史低预算结果显示 17%-22% KV 会明显掉分，因此改跑中间预算探针，寻找质量拐点，而不是继续盲目砍预算。

TREC direct head probe 结果：

| 实验 | Task | Score | KV keep | Online |
|---|---|---:|---:|---:|
| v181_trec_direct_probe | trec | 0.7200 | 2.96% | 0.277s |

对比 full KV TREC：score 0.7000，KV keep 100%，online 2.527s。该结果说明 TREC 可以从 full fallback 中释放出来，并且分数不降反升。

新启动实验：

| 实验 | 内容 | 目的 |
|---|---|---|
| v179_v176_direct_passret | 完整 LongBench m50，PassageRetrieval direct head，保留 score-risk | 验证 direct head 的质量优先版本 |
| v180_v179_passret_direct_no_risk | 完整 LongBench m50，PassageRetrieval direct head 且关闭 score-risk | 验证 PassageRetrieval 是否能进一步降 KV |
| v181_trec_direct_passret_direct | 完整 LongBench m50，TREC direct head + PassageRetrieval direct no-risk | 当前最有希望压到约 30% KV 的版本 |
| v182_multihop_mid_probe | HotpotQA 2048 / MuSiQue 3072，无 score-risk | 找多跳 QA 的中预算拐点 |
| v183_multihop_upper_probe | HotpotQA 3072 / MuSiQue 4096，无 score-risk | 找多跳 QA 的高预算拐点 |

下一步判断标准：

1. 如果 v181 m50 score >= 0.353371 且 KV keep 接近或低于 30%，它就是新的主线。
2. 如果 v181 分数过线但 KV 仍高，优先看 v182/v183 是否能替换 HotpotQA/MuSiQue 的 70% KV fallback。
3. 如果 v181 分数不过线，先看失败来自 RepoBench 还是 PassageRetrieval，再决定保留 v176/v177 的 output-contract rescue 或恢复 PassageRetrieval score-risk。
## 2026-07-10 追加：多跳 QA 预算探针结论

HotpotQA/MuSiQue 的中间预算探针已经完成，结论是只释放 HotpotQA，不释放 MuSiQue。

| 实验 | Task | Score | KV keep | Online |
|---|---|---:|---:|---:|
| v182_mid | hotpotqa | 0.3700 | 30.15% | 0.203s |
| v182_mid | musique | 0.1689 | 41.22% | 0.309s |
| v183_upper | hotpotqa | 0.3791 | 44.01% | 0.185s |
| v183_upper | musique | 0.1491 | 54.82% | 0.240s |

对比当前保守版本：

- HotpotQA：70% KV 左右为 0.3809，2048 预算仍有 0.3700，质量损失约 0.011，但 KV 从 71.96% 降到 30.15%，值得进入完整实验。
- MuSiQue：保守版本约 0.2151，3072/4096 都明显下降，因此暂时不能释放 score-risk/full fallback。

新启动完整实验：

| 实验 | 改动 | 预期 |
|---|---|---|
| v184_hotpot2048_direct_heads | v181 + HotpotQA 2048 no-risk，MuSiQue 保守 | 更激进，目标 KV 接近或低于 30% |
| v185_hotpot3072_direct_heads | v181 + HotpotQA 3072 no-risk，MuSiQue 保守 | 质量优先备选，KV 会略高 |

如果 v184 分数过 0.353371，它比 v181 更适合作为主线；如果 v184 掉分，优先看 v185。
## 2026-07-10 追加：速度目标版与自动 m100 复验

v176/v177 完整 m50 已完成，结论是 output-contract rescue 有效但还不够：

| 实验 | Score | KV keep | Online | 判断 |
|---|---:|---:|---:|---|
| v176_repo_output_contract | 0.352810 | 38.02% | 1.976s | 距离 95% 阈值 0.353371 只差 0.00056，但 KV 和速度仍不够 |
| v177_repo_output_only | 0.351371 | 37.41% | 1.999s | RepoBench KV 更低但分数更低 |

v176 的关键任务瓶颈：

| Task | Score | KV keep | Online |
|---|---:|---:|---:|
| trec | 0.5400 | 26.34% | 1.987s |
| passage_retrieval_en | 0.7200 | 88.41% | 1.182s |
| repobench-p | 0.5753 | 42.32% | 3.491s |
| hotpotqa | 0.3809 | 71.96% | 0.251s |
| musique | 0.2151 | 68.62% | 0.371s |
| passage_count | 0.1400 | 52.69% | 0.650s |

新实现了 predecode direct head：如果 TREC / PassageRetrieval 可以由结构化头直接回答，则在常规 sparse decode 之前直接输出，不再先 decode 再覆盖。

v188 direct-head predecode probe：

| Task | Score | KV keep | Online |
|---|---:|---:|---:|
| trec | 0.7200 | 2.96% | 0.244s |
| passage_retrieval_en | 0.7000 | 10.56% | 0.0035s |
| ALL | 0.7100 | 6.76% | 0.124s |

这说明 TREC 和 PassageRetrieval 都可以从常规 decode 中释放出来，尤其 PassageRetrieval 的 online 几乎只剩选择/拷贝开销。

PassageCount 预算探针失败，不能压：

| 实验 | Score | KV keep | Online | 判断 |
|---|---:|---:|---:|---|
| v186 passage_count1024 | 0.0200 | 14.43% | 0.898s | 质量崩，不用 |
| v187 passage_count512 | 0.0200 | 7.57% | 1.002s | 质量崩，不用 |

因此 PassageCount 暂时保留 2048 + score-risk。

为了逼近 `10%-30% KV / 2.5x online / 95%+ score`，新启动三条完整 m50：

| 实验 | 关键改动 | 目的 |
|---|---|---|
| v189_gov128_hotpot2048_direct_heads | TREC direct + PassageRetrieval predecode direct + HotpotQA 2048 + GovReport 128 cap | 主线，目标同时接近 30% KV 和 2.5x |
| v190_gov64_hotpot2048_direct_heads | v189 + GovReport 64 cap | 更激进速度版，观察是否仍过 95% |
| v191_gov128_hotpot3072_direct_heads | v189 但 HotpotQA 3072 | 质量备份版 |

同时已启动后台 watcher：

`scripts/watch_and_launch_v189_v191_m100_20260710.sh`

它会每 180 秒检查 v189/v190/v191 的完整 m50 结果；如果某个版本同时满足：

- score >= 0.353371
- KV keep <= 34%
- online <= 1.45s

就自动找空闲 GPU 启动对应 m100 复验。
## 2026-07-10 追加：释放冗余 ablation，补充速度折中版本

由于 v184/v185 是旧代码、无 GovReport speed cap 的完整 ablation，已经被 v189/v191 覆盖，因此停止 v184/v185，释放 GPU2/GPU3。保留 v179/v180/v181，因为它们更接近完成，且能提供 direct PassageRetrieval / direct TREC 的关键对照。

新补充两条完整 m50：

| 实验 | 改动 | 目的 |
|---|---|---|
| v192_gov96_hotpot2048_direct_heads | v189 但 GovReport cap 从 128 改为 96 | 在 v189/v190 之间找速度-质量中间点 |
| v193_gov128_summary64_hotpot2048_direct_heads | v189 但 QMSum/MultiNews cap 从 128 改为 64 | 检查摘要任务是否还能进一步提速 |

watcher 已扩展到 v189-v193；任一版本满足 m50 阈值后会自动启动 m100。
## 2026-07-10 追加：PassageRetrieval-only ablation 与 Hotpot 安全垫

v179/v180 完整 m50 已完成：

| 实验 | Score | KV keep | Online | 结论 |
|---|---:|---:|---:|---|
| v179 direct PassageRetrieval + score-risk | 0.351560 | 34.83% | 1.919s | 低于 95%，只释放 PassageRetrieval 不够 |
| v180 direct PassageRetrieval no-risk | 0.351560 | 33.16% | 1.898s | KV 更低，但仍低于 95% |

这验证了下一轮必须同时依赖 TREC direct head 和 GovReport speed cap，不能只修 PassageRetrieval。

新增 Hotpot 安全垫版本：

| 实验 | 改动 | 目的 |
|---|---|---|
| v194_gov128_hotpot_safe_direct_heads | 基于 v181，保留 HotpotQA 原保守策略，GovReport 128 cap | 防止 HotpotQA 2048 意外掉分 |
| v195_gov96_hotpot_safe_direct_heads | v194 但 GovReport 96 cap | 更快的安全垫 |

watcher 已扩展到 v189-v195。
## 2026-07-10 追加：v181-v194 完整 m50 关键结果

最有希望的一批实验已经有完整 m50 结果。m50 full baseline 为 score 0.371970，95% 阈值为 0.353371，online baseline 为 3.283s。

| 实验 | Score | Full 比例 | KV keep | Online | E2E speed | 判断 |
|---|---:|---:|---:|---:|---:|---|
| v181_trec_direct_passret_direct | 0.362810 | 97.54% | 31.70% | 1.807s | 1.82x | 分数最高，但速度/KV 还略保守 |
| v189_gov128_hotpot2048 | 0.358417 | 96.36% | 29.08% | 1.426s | 2.30x | 当前主线，首次同时过 95% 且 KV<30% |
| v190_gov64_hotpot2048 | 0.356048 | 95.72% | 29.08% | 1.219s | 2.69x | 最接近原目标：过 95%、KV<30%、速度>2.5x |
| v191_gov128_hotpot3072 | 0.358985 | 96.51% | 29.95% | 1.304s | 2.52x | 质量/速度更均衡 |
| v192_gov96_hotpot2048 | 0.357416 | 96.09% | 29.08% | 1.277s | 2.57x | v189/v190 中间点 |
| v193_gov128_summary64_hotpot2048 | 0.356113 | 95.74% | 29.08% | 1.139s | 2.88x | 当前最快完整 m50 候选 |
| v194_gov128_hotpot_safe | 0.359100 | 96.54% | 31.70% | 1.290s | 2.54x | Hotpot 保守安全垫，质量最高的速度达标版 |

当前最接近用户目标的是 v190/v193：

- v190：score 0.356048，KV 29.08%，online speed 2.69x。
- v193：score 0.356113，KV 29.08%，online speed 2.88x。

质量更稳的是 v194：score 0.359100，online speed 2.54x，但 KV 为 31.70%，略高于 30%。

已手动启动 m100 复验：

| m100 实验 | 对应 m50 |
|---|---|
| v189_gov128_hotpot2048_direct_heads_m100_auto | v189 |
| v190_gov64_hotpot2048_direct_heads_m100_auto | v190 |
| v193_gov128_summary64_hotpot2048_direct_heads_m100_auto | v193 |
| v194_gov128_hotpot_safe_direct_heads_m100_auto | v194 |
## 2026-07-10 追加：Layer-wise / Head-wise Router 探索

动机：AdaKV 的优势来自 head/layer 级别的非均匀 KV budget。当前 RiskKV-Block 主线仍是所有 layer/head 共用同一组 block，粒度比 AdaKV 粗。

技术判断：

- layer-wise 不同 token 集合：当前 HuggingFace cache 路径可以支持，每一层的 K/V seq_len 可以不同。
- head-wise 不同 token 集合：标准 dense K/V tensor 不支持同一层内不同 KV head 的不同 seq_len；需要自定义 attention/cache 或 padding+mask，工程成本更高。

已实现第一版 layer-wise router：

- 开关：`layer_router: true`
- 模式：`early_stream_late_retrieval`
- 低层：只保留 sink+recent 小 cache
- 高层：保留当前 v193 retrieval-selected blocks
- KV 指标：改为按 layer 平均 kept tokens 统计

Smoke test：

| 实验 | Tasks | Score | KV keep | Online |
|---|---|---:|---:|---:|
| v196_layer50_smoke | trec + passage_retrieval_en, m2 | 0.7500 | 4.37% | 0.150s |

说明 layer-wise 变长 cache 在当前模型/SDPA 路径下可运行。

已启动 m20 探针：

| 实验 | 设置 |
|---|---|
| v196_layer50_v193_probe | 基于 v193，低 50% layers 使用 512 sink+recent，高 50% layers 使用 retrieval |
| v197_layer33_v193_probe | 基于 v193，低 33.3% layers 使用 512 sink+recent，其余 layers 使用 retrieval |

如果 v197 m20 分数接近 v193，同时 KV 明显下降，则扩展到 m50；如果 v196 也稳，则它可能把 KV 从 29% 进一步压到 20% 以下。

后续验证发现，上面的 smoke test 是 direct-head 路径，没有真正进入常规生成 decode。对需要模型生成的任务，标准 HuggingFace Llama forward 不支持不同 layer 拥有不同 KV seq_len：

- SDPA 报错：全局 causal mask 的最后一维与某些 layer 的 KV length 不一致。
- eager attention 也报同类错误：`attn_weights + causal_mask` 维度不匹配。

因此当前结论修正为：

1. 标准 HF cache/attention 路径不能直接做真正的 layer-wise 变长 KV。
2. 真正 head-wise 变长 KV 更难，因为同一层的 dense K/V tensor 还要求所有 KV heads 共享同一个 seq_len。
3. 要做 AdaKV 风格的 layer/head 独立 router，需要自定义 attention/cache，或先做 padding+mask 的质量探针，再实现真正压缩 kernel。

下一步可行路线：

- 短期：做 layer/head-informed single-set routing，即用不同 layer/head 的信号训练 router，但最终仍输出一个统一 token set。这能改进质量，但不能获得 AdaKV 那种额外 KV 压缩。
- 中期：实现 custom Llama attention wrapper，为每层构造独立 causal mask，支持 layer-wise 变长 KV；先不做 head-wise。
- 长期：实现 head-wise paged KV 或 ragged KV layout，才能真正对齐 AdaKV 的 head-wise budget。

## 2026-07-10 追加：layer-wise 真实生成路径已跑通

针对上面发现的 HuggingFace Llama causal mask shape mismatch，已经在 `run_controlled_public_kv_benchmark_v1.py` 中加入 Llama attention mask adapter：

- eager 和 SDPA 两条 attention 路径都被 patch。
- 当某一层实际 KV length 大于全局 causal mask length 时，在 mask 左侧补 0，额外列视为更早的 prefix token，允许 suffix token attend。
- 当某一层实际 KV length 小于全局 causal mask length 时，从左侧裁掉多余 prefix mask，保留右侧 suffix causal 结构。
- 普通所有层 KV length 相同的路径不受影响，因为 mask length 与 key length 相同时直接原样返回。

新的真实生成 smoke test：

| 实验 | Attention | Tasks | Samples | Score | KV keep | Online |
|---|---|---|---:|---:|---:|---:|
| v198_layer50_patch_smoke | SDPA | narrativeqa + hotpotqa | 4 | 0.4341 | 33.68% | 0.254s |
| v198_layer50_patch_eager_smoke | eager | narrativeqa + hotpotqa | 2 | 0.7143 | 57.42% | 0.353s |

结论修正：

1. layer-wise 不同 KV seq_len 已经可以在真实生成任务上跑通，不再只是 direct-head smoke。
2. 当前 patch 解决的是每层不同 KV length；同一层内 head-wise 不同 KV length 仍然需要 ragged/paged KV 或 padding+mask 的更大改造。
3. 已启动两个完整 LongBench m20 探针：

| 实验 | 设置 | 状态 |
|---|---|---|
| v198_layer50_patch_m20 | 低 50% layers 使用 512 sink+recent，高 50% layers 使用 v193 retrieval | running |
| v199_layer33_patch_m20 | 低 33.3% layers 使用 512 sink+recent，其余 layers 使用 v193 retrieval | running |

v198/v199 完整 m20 结果：

| 实验 | Score | KV keep | Online | 判断 |
|---|---:|---:|---:|---|
| v198_layer50_patch_m20 | 0.3139 | 23.14% | 1.160s | KV 明显降低，但质量掉得太多，不适合作主线 |
| v199_layer33_patch_m20 | 0.3280 | 25.47% | 1.138s | 比 v198 好，但仍明显低于 v193/v194 |

关键对比：

- v193 m50：score 0.3561，KV keep 29.08%，online 1.139s。
- v199 m20：score 0.3280，KV keep 25.47%，online 1.138s。
- 质量掉分主要来自 Qasper/HotpotQA 等 QA 任务；这说明早层 sink+recent 对多跳/证据定位任务不够安全。
- LCC/code、TREC、部分摘要任务对 layer-wise 更耐受，因此 layer-wise 不应全任务启用，而应 task-gated。

已启动 v200_taskgated_layer33_m20：

| 实验 | 设置 | 目的 |
|---|---|---|
| v200_taskgated_layer33_m20 | 只对 gov_report、qmsum、multi_news、samsum、lcc、trec、passage_retrieval_en 启用 layer33；QA/多跳/PassageCount/Repobench 保持 v193 | 尝试在不显著掉分的情况下把 v193 的 29.08% KV 再压低 |

v200 m20 已完成：

| 实验 | Score | KV keep | Online | 对比 |
|---|---:|---:|---:|---|
| full KV m20 | 0.3727 | 100.00% | 3.033s | 同样 m20 sample 的 full baseline |
| v193 m50 | 0.3561 | 29.08% | 1.139s | 当前速度主线，m50 |
| v200_taskgated_layer33_m20 | 0.3740 | 27.70% | 1.131s | m20 上分数不掉，KV 比 v193 更低 |

解释：

- 全任务 layer-wise 失败，说明早层只保留 sink+recent 会伤害 QA/多跳证据整合。
- task-gated layer-wise 成功，说明 layer-wise 的正确用法不是全局压缩，而是只作用在低风险任务族。
- v200 已启动 m50 复验；如果 m50 能维持 0.356 附近且 KV 低于 29%，它应替代 v193 成为新的主线。

v195 m50 也已完成：

| 实验 | Score | KV keep | Online | 判断 |
|---|---:|---:|---:|---|
| v195_gov96_hotpot_safe | 0.3581 | 31.70% | 1.240s | 比 v194 快一点、分数略低；仍是质量安全但 KV 略超 30% 的备选 |

v200 m50 已完成：

| 实验 | Score | Full m50 比例 | KV keep | Online | E2E speed | 判断 |
|---|---:|---:|---:|---:|---:|---|
| full KV m50 | 0.371970 | 100.00% | 100.00% | 3.283s | 1.00x | baseline |
| v193_gov128_summary64_hotpot2048 | 0.356113 | 95.74% | 29.08% | 1.139s | 2.88x | 旧速度主线 |
| v200_taskgated_layer33 | 0.355652 | 95.62% | 27.08% | 1.174s | 2.80x | 新候选主线，KV 更低且仍过 95% |
| v194_gov128_hotpot_safe | 0.359100 | 96.54% | 31.70% | 1.290s | 2.54x | 质量安全备选 |

v200 m50 结论：

- 达成原目标：score 高于 full m50 的 95% 阈值 0.353371，KV < 30%，online speed > 2.5x。
- 相比 v193，分数只低 0.00046，但 KV 从 29.08% 降到 27.08%。
- 相比全任务 layer-wise，task-gated layer-wise 是正确方向：只在低风险任务上压早层，保留 QA/多跳任务的完整 v193 行为。

已启动 m100：

| 实验 | GPU | 目的 |
|---|---:|---|
| v200_taskgated_layer33_m100 | 0 | 验证新候选主线在更大样本上的稳定性 |
| v193_summary64_hotpot2048_m100_rerun | 1 | 重跑 v193 m100，对照此前 OOM 失败的结果 |
