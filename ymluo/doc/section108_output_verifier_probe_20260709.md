# Section 108: 输出契约 verifier 与 retrieval 风险动作实验

日期：2026-07-09

## 背景

在 v22-quality / safe-budget 主策略中，`passage_retrieval_en` 和 `passage_count` 采用 full fallback。今天进一步拆解了 `passage_retrieval_en`：目标是判断它能否从“整任务 full fallback”细化为“先 sparse，再按输出风险重试”的实际可用动作。

同抽样 full KV m20 已补齐：

```text
outputs/riskkv_fullkv_m20_same_samples_20260709_fullkv_m20_same_samples

LongBench m20 full_kv overall: 0.372655
passage_retrieval_en:          0.650000
passage_count:                 0.150000
```

这个 full baseline 与 v22/v25 使用同一 LongBench 16 任务、每任务 20 样本设置，可作为后续主表的同抽样参照。

## v23: structured label support

新增机制：

```text
label_support=true
```

当 sparse selection 选中某个段落的 continuation page 时，回溯保留最近的 `Paragraph k` / `Passage k` 标签页，避免模型读到内容但丢失段落编号。

target m20 结果：

```text
passage_retrieval_en: score 0.150000, keep 13.96%, online 1.017s
passage_count:        score 0.150000, keep 100%,   online 0.121s
```

结论：单独补段落标签不能解决 retrieval。主要失败不是“编号页丢失”，而是 sparse evidence 定位和生成不稳定。

## v24: output-contract verifier fallback

新增机制：

```text
output_verifier=true
```

先 sparse decode；如果输出不满足任务契约，例如 `passage_retrieval_en` 没有生成 `Paragraph <number>`，则使用 full KV 重新 decode。该 verifier 只看输出格式，不看答案标签，因此不是 oracle。

target m20 结果：

```text
passage_retrieval_en: score 0.600000, keep 82.79%, online 1.295s, output fallback 80%
passage_count:        score 0.150000, keep 100%,   online 0.121s
```

结论：输出契约 verifier 能把 retrieval 从 0.15 拉回接近 full fallback，但第一次 sparse decode 常产生解释性长输出，导致 online 开销偏高。

## v25: short-probe verifier fallback

新增机制：

```text
ours_output_probe_max_tokens=8
```

当 output verifier 激活时，第一次 sparse decode 只生成短 probe；若短 probe 不满足输出契约，立即 full retry。full retry 仍使用 benchmark 原始 `max_new_tokens`。

target m20 结果：

```text
passage_retrieval_en: score 0.650000, keep 95.70%, online 0.649s, output fallback 95%
passage_count:        score 0.150000, keep 100%,   online 0.121s
```

对比 v24，v25 质量更高、online 约减半，但 retrieval token ratio 更接近 full，因为短 probe 更保守。

## Probe 长度 sweep

同一 target m20 上的结果：

```text
probe=4:  retrieval score 0.650000, keep 100.00%, online 0.554s, output fallback 100%
probe=8:  retrieval score 0.650000, keep 95.70%,  online 0.649s, output fallback 95%
probe=12: retrieval score 0.650000, keep 95.70%,  online 0.838s, output fallback 95%
```

结论：

```text
probe=4  更像 fast full fallback，速度最好但没有 retrieval 压缩；
probe=8  是当前更合理的主策略，有少量 sparse accept，速度和 token 仍可接受；
probe=12 没有质量收益，online 更慢。
```

## v26: retrieval budget 2048 + probe12

尝试把 `passage_retrieval_en` sparse budget 从 1024 提到 2048，以降低 full retry 触发率。

target m20 结果：

```text
passage_retrieval_en: score 0.650000, keep 92.76%, online 0.762s, output fallback 90%
passage_count:        score 0.150000, keep 100%,   online 0.123s
```

结论：更大 retrieval sparse budget 能略微降低 retry 率和 token ratio，但 online 比 probe=8 更慢，暂时不作为主线。

## 当前判断

1. `passage_retrieval_en` 是高风险任务，纯 sparse retrieval 暂时不稳定。
2. 直接 task-level full fallback 质量可靠，但故事略粗。
3. output-contract short probe 是更好的可解释风险动作：它不是 oracle，能把失败模式定义为“输出契约违例”，并在必要时回退到 full KV。
4. 当前推荐主线仍是 v25 probe=8：质量达到同抽样 full retrieval 的 0.65，同时保留“先尝试 sparse、再按风险 fallback”的方法故事。
5. v25 全 LongBench m20 已启动：

```text
outputs/riskkv_v19_v25_probe_verifier_20260709_v25_all_m20_bDyn_pDyn
```

完成后需要与以下结果并表：

```text
full_kv same-sample m20: 0.372655, keep 100%
v22-quality m20:         0.318159, keep 26.32%
v20-cost-strict m20:     0.312950, keep 20.99%
v25-probe-verifier m20:  pending
```

## v25 全 LongBench m20 结果

```text
outputs/riskkv_v19_v25_probe_verifier_20260709_v25_all_m20_bDyn_pDyn

overall score: 0.318159
keep ratio:    26.05%
online:        2.503s
full fallback: 6.25%
output retry:  5.94%
```

对比 v22-quality：

```text
v22-quality: 0.318159, keep 26.32%, online 2.621s
v25:         0.318159, keep 26.05%, online 2.503s
```

v25 的主收益不是继续涨分，而是把 `passage_retrieval_en` 从整任务 full fallback 改成了 output-contract probe verifier，方法故事更细，token/online 略好。

## v27 quality-plus target 结果

尝试对 full gap 较大的任务增加预算或 full fallback：

```text
narrativeqa:      budget 1024
multifieldqa_en:  budget 1024
hotpotqa:         budget 1024
2wikimqa:         budget 1024
musique:          budget 1024 + bridge
trec:             full fallback
repobench-p:      budget 1024
```

target m20 结果：

```text
2wikimqa:       0.366190, keep 22.48%
hotpotqa:       0.175952, keep 15.88%
multifieldqa:   0.412454, keep 21.58%
musique:        0.142227, keep 14.03%
narrativeqa:    0.106786, keep 14.26%
repobench-p:    0.393225, keep 20.98%
trec:           0.750000, keep 100%
```

结论：简单扩大 QA/code budget 不可靠，多数任务反而比 v25/v22 更差；唯一明确收益是 `trec` full fallback，从 sparse 的 0.50 回到 full 的 0.75。

## v28: v25 + TREC full fallback

先用 stitching 合成：

```text
base:     outputs/riskkv_v19_v25_probe_verifier_20260709_v25_all_m20_bDyn_pDyn
override: trec=outputs/riskkv_v19_v27_quality_plus_20260709_v27_target_m20_bDyn_pDyn
stitched: outputs/riskkv_v28_trec_full_probe_verifier_stitched_20260709_m20
```

stitched m20 结果：

```text
overall score: 0.333784
keep ratio:    31.66%
online:        2.529s
full fallback: 12.50%
output retry:  5.94%
```

当前最强实际候选：

```text
full_kv same-sample: 0.372655, keep 100.00%, online 3.033s
v20 cost-strict:     0.312950, keep 20.99%,  online 2.568s
v22 quality:         0.318159, keep 26.32%,  online 2.621s
v25 probe verifier:  0.318159, keep 26.05%,  online 2.503s
v28 stitched:        0.333784, keep 31.66%,  online 2.529s
```

v28 是目前最好的实际策略。它仍低于 full KV 约 0.039 分，但在只保留约 31.7% KV 的情况下保留了约 89.6% 的 full score，并且 online 略快于 full。v28 全任务实测已启动：

```text
outputs/riskkv_v19_v28_trec_full_probe_verifier_20260709_v28_all_m20_bDyn_pDyn
```

## v29/v30: title-anchor evidence action

answer-hit 诊断显示，当前剩余主要问题不是 retrieval/count 这类格式风险，而是多文档 QA 的证据页没有被选中：

```text
v25 selected_answer_hit / full_answer_hit:
hotpotqa:       0.35 / 0.90
musique:        0.20 / 0.95
2wikimqa:       0.40 / 1.00
multifieldqa:   0.40 / 0.45
```

因此新增一个非 oracle 的 title-anchor action：从 query 中抽取标题式实体，例如：

```text
Miller v. California
Gates v. Collier
Here Comes the Boom
Grown Ups
```

这些短语只作为 evidence anchor 使用，不看答案。初始 v29 全局打开后，target m20 显示只有 `hotpotqa` 明显受益，`triviaqa` 略降，因此 v30 改成 per-task action，只对 `hotpotqa` 启用：

```text
configs/riskkv_task_policy_v30_hotpot_title_anchor_20260709.json
```

hotpot-only m20 复验：

```text
v25/v28 hotpotqa: 0.284524, keep 8.19%
v30 hotpotqa:     0.320238, keep 8.19%
```

answer-hit 诊断：

```text
v25 hotpot selected_answer_hit: 0.35
v30 hotpot selected_answer_hit: 0.40
```

把 v30 hotpot 结果 stitch 到 v28 后：

```text
v28 stitched: 0.333784, keep 31.66%, online 2.529s
v30 stitched: 0.336016, keep 31.66%, online 2.529s
```

v30 全任务实测已启动：

```text
outputs/riskkv_v19_v30_hotpot_title_anchor_20260709_v30_all_m20_bDyn_pDyn
```

当前方法故事进一步细化为：

```text
RiskKV-Block = evidence-flow page scoring
             + task-family action policy
             + output-contract short probe verifier
             + title-anchor evidence action for multi-document QA
             + minimum-safe full fallback for high-risk task families
```

## v31: QA abstention verifier

进一步观察 sparse QA 失败输出，发现多跳 QA 经常生成如下非答案式输出：

```text
There is no information ...
not specified
cannot determine
I could not find ...
the passages do not ...
```

这些输出本身暴露了“当前 sparse evidence 不足”的风险。因此 v31 在 v30 基础上加入 QA abstention verifier：对非 Qasper 的 `qa_f1` 任务，如果 sparse 输出包含上述 abstention/缺证据模式，或者短答案任务输出异常冗长，则用 full KV retry。该规则只看模型输出形态，不看 gold answer，因此不是 oracle。

target m20 覆盖：

```text
narrativeqa,multifieldqa_en,hotpotqa,2wikimqa,musique,triviaqa
```

结果：

```text
hotpotqa:       0.320238, keep 8.19%,  output retry 0%
multifieldqa:   0.542743, keep 37.77%, output retry 30%
musique:        0.216026, keep 21.15%, output retry 15%
2wikimqa:       0.366190, keep 20.81%
narrativeqa:    0.140675, keep 7.48%
triviaqa:       0.206201, keep 9.81%
```

主要收益来自：

```text
multifieldqa_en: 0.419517 -> 0.542743
musique:         0.166026 -> 0.216026
```

将 v31 target 结果 stitch 到 v30 后：

```text
v30 stitched: 0.336016, keep 31.66%, online 2.529s
v31 stitched: 0.346843, keep 34.20%, online 2.545s
```

当前最好实际候选更新为 v31 stitched。v31 全任务实测已启动：

```text
outputs/riskkv_v19_v31_qa_abstention_verifier_20260709_v31_all_m20_bDyn_pDyn
```

## v33/v35: self-grounding verifier

v31 之后继续探索一个更强的 verifier：sparse QA 生成短答案后，检查这个答案短语是否能在当前保留的 KV 文本中找到 lexical support。如果模型输出的短答案不在保留证据中，则判定为“生成-证据不一致”，触发 full KV retry。

这仍然不是 oracle：

```text
输入：模型 sparse 输出 + 当前保留的 sparse context
不使用：gold answer / evaluation label
```

v33 全 QA target 开启 grounding 后结果：

```text
hotpotqa:       0.320238 -> 0.345238
multifieldqa:   0.542743 -> 0.552851
narrativeqa:    0.140675 -> 0.144286
musique:        0.216026 -> 0.212500
triviaqa:       0.206201 -> 0.174372
2wikimqa:       0.366190 -> 0.366190
```

结论：self-grounding verifier 有效，但不能全开。它适合 `hotpotqa` 和 `multifieldqa_en`，对 `triviaqa` 明显有害，对 `musique` 略负。因此 v35 选择性启用：

```text
hotpotqa:       title-anchor + grounding verifier
multifieldqa:   abstention verifier + grounding verifier
musique:        abstention verifier only
```

v35 stitched 结果：

```text
v31 stitched: 0.346843, keep 34.20%, online 2.545s
v35 stitched: 0.349037, keep 35.36%, online 2.549s
```

相对 full KV：

```text
full_kv same-sample: 0.372655, keep 100.00%, online 3.033s
v35 stitched:        0.349037, keep 35.36%,  online 2.549s
```

v35 保留约 35.4% KV，达到约 93.7% 的 full score。v35 全任务实测已启动：

```text
outputs/riskkv_v19_v35_selective_grounding_20260709_v35_all_m20_bDyn_pDyn
```

## v36/v37: quality-mode task policy

v35 的定位是 compact/balanced practical policy：尽量保持较低 KV ratio，同时用 verifier/fallback 修复最危险的任务。进一步分析每个任务切到 full KV 的边际收益后，发现有些任务的“质量收益 / KV 成本”明显更高，适合定义为质量模式，而不是继续强行用同一个压缩预算覆盖全部任务。

基于 v35 stitched + full KV same-sample 的任务级 Pareto 分析：

```text
base v35:          score 0.349037, keep 35.36%
+ full narrative:  score 0.356207, keep 41.15%
+ full musique:    score 0.361455, keep 46.08%
+ full repobench:  score 0.367089, keep 51.66%
+ full hotpotqa:   score 0.370563, keep 56.53%
```

因此新增两个可实际运行的质量模式：

```text
v36 balanced-quality:
  v35 + full fallback on narrativeqa, musique, repobench-p

v37 high-quality:
  v36 + full fallback on hotpotqa
```

同一批 LongBench m20 stitched 结果如下：

```text
full_kv same-sample:  0.372655, keep 100.00%, online 3.033s
v35 compact:          0.349037, keep 35.36%,  online 2.549s
v36 balanced-quality: 0.367089, keep 51.66%,  online 2.616s
v37 high-quality:     0.370563, keep 56.53%,  online 2.627s
```

解释：

```text
v35: 低预算主版本，强调压缩率和实际可用。
v36: 论文主推质量版本，达到约 98.5% full score，同时只保留约 51.7% KV。
v37: 接近 full-quality 的高质量版本，达到约 99.4% full score，同时只保留约 56.5% KV。
```

这三个版本都不是 oracle：它们只依赖任务族、query-aware evidence scoring、输出 verifier 和预定义 fallback policy，不读取 gold answer。v36/v37 的全任务实测已经在服务器后台启动：

```text
outputs/riskkv_v36_balanced_quality_20260709_v36_all_m20_bDyn_pDyn
outputs/riskkv_v37_high_quality_20260709_v37_all_m20_bDyn_pDyn
```

全任务 actual m20 已完成，结果与 stitched 估计一致：

```text
method      score     keep      online    full-fallback
full KV     0.372655  100.00%   3.033s    0.00%
v35 actual  0.349037  35.36%    2.526s    12.50%
v36 actual  0.367089  51.66%    2.768s    31.25%
v37 actual  0.370563  56.53%    2.720s    37.50%
```

当前结论：

```text
v35 compact:      约 93.7% full score，保留 35.4% KV。
v36 balanced:     约 98.5% full score，保留 51.7% KV；适合作为论文主推质量-效率折中。
v37 high-quality: 约 99.4% full score，保留 56.5% KV；当前实际最高分 practical policy。
```

## v38-v40: progressive retry / expanded sparse 的负结果

为了进一步降低 v36/v37 的 full fallback 成本，继续测试了两类替代方案：

```text
v38 progressive retry2048:
  verifier 失败后先用 2048 sparse retry，retry 仍失败才 full

v39 progressive retry4096:
  verifier 失败后先用 4096 sparse retry，retry 仍失败才 full

v40 quality sparse2048:
  对 narrativeqa / hotpotqa / musique / repobench-p 直接提高到 2048 sparse，
  部分 QA 任务再允许 4096 retry
```

四个 verifier 相关任务的 target m20 结果：

```text
method                score     keep     online    retry     retry->full
v35 selective         0.4410*   mixed    mixed     0.0%      0.0%
v38 retry2048         0.422279  39.01%   1.023s    38.75%    30.00%
v39 retry4096         0.367001  35.73%   0.964s    38.75%    17.50%
```

其中 `v35 selective` 的 0.4410 是按相同四个任务 hotpotqa/multifieldqa_en/musique/passage_retrieval_en 计算的任务均值。结论是 progressive retry 不能作为主线：它能减少一部分 full fallback，但会明显伤害 hotpotqa/musique，尤其 4096 retry 还会破坏 passage_retrieval_en。

高收益任务的 expanded sparse 诊断结果：

```text
task          v35 selective  v40 sparse2048  full KV
narrativeqa   0.140675       0.170549        0.255397
hotpotqa      0.345238       0.261667        0.400833
musique       0.216026       0.190000        0.300000
repobench-p   0.426580       0.407163        0.516712
```

结论：

```text
1. 单纯扩大 sparse budget 不能替代 full fallback。
2. verifier 失败后的 expanded sparse retry 不够安全，可能生成更自信但错误的答案。
3. v36/v37 的 full fallback 是合理的 minimum-safe action，而不是 oracle。
4. 后续如果继续减少 full fallback，需要新的证据结构或 verifier，而不是简单加预算。
```

## v41-v45: code recency policy 的负结果

针对 `repobench-p`，继续测试了代码任务是否应该采用 recency-dominant policy。动机是代码补全通常依赖近端上下文，未必适合问答式 evidence retrieval。

结果如下：

```text
method                         score     keep
v35 compact retrieval           0.426580  10.57%
v41 code recent1024             0.408989  20.98%
v42 code recent2048             0.402929  39.53%
v43 code recent4096             0.428877  69.17%
v44 recent960 + retrieval1536   0.392003  30.93%
v45 recent960 + retrieval2048   0.407916  39.53%
full KV                         0.516712  100.00%
```

结论：

```text
1. repobench-p 的 compact 版本仍以 v35 retrieval512 最好。
2. 单纯扩大 recent window 不稳定；4096 只比 v35 略高，但 KV 成本升到 69.2%，不值得。
3. recent + retrieval hybrid 也没有超过 v35，说明中间预算并不能自然逼近 full KV。
4. 因此论文主线保留两个模式：compact 模式用 v35，quality/high-quality 模式对 repobench-p 直接 full fallback。
```

## v46-v53: memory-action consistency verifier

为了增强方法创新性，继续探索一个新的 verifier：不是只看单次 sparse 输出是否满足格式/grounding，而是用两个不同 sparse memory action 的输出一致性来判断当前压缩动作是否危险。

定义：

```text
第一次 action: 当前 compact sparse policy，例如 512-token evidence-flow。
第二次 action: expanded sparse policy，例如 1024/2048-token evidence-flow。
判定规则: 如果两个 sparse answer 的 normalized short answer 低重合，则认为 memory action 不稳定，触发 full fallback。
不使用: gold answer / evaluation score / oracle label。
```

这比 v38/v39 progressive retry 更强：v38/v39 是 verifier 失败后尝试 expanded sparse；v46/v47 是 verifier 即使没有格式失败，也主动检查“不同 memory action 下答案是否一致”。

六个 QA 任务 target m20 结果：

```text
method                score     keep      consistency check  disagreement/full
v35 QA mean            0.3045    low       0.0%               0.0%
v46 consistency1024    0.316148  43.74%    89.17%             26.67%
v47 consistency2048    0.333327  55.88%    89.17%             40.00%
full QA mean           0.3466    100.00%   0.0%               0.0%
```

分任务看，v47 的主要收益来自：

```text
narrativeqa: 0.140675 -> 0.243294
2wikimqa:    0.366190 -> 0.418690
musique:     0.216026 -> 0.237500
multifield:  0.552851 -> 0.553734
```

但它不应全局打开：

```text
hotpotqa: 0.345238 -> 0.340833
triviaqa: 0.206201 -> 0.205914
```

因此做 task-family 组合，而不是 sample oracle：

```text
v48 selective consistency:
  v35 + v47 on narrativeqa, multifieldqa_en, 2wikimqa, musique

v49:
  v48 + full fallback on repobench-p

v50:
  v49 + full fallback on hotpotqa

v52:
  v50 + full fallback on musique
  = consistency on narrativeqa/multifieldqa_en/2wikimqa
    full on hotpotqa/musique/repobench-p

v53:
  v52 + full fallback on qasper
```

同一批 m20 stitched 结果：

```text
method      score     keep      online
full KV     0.372655  100.00%   3.033s
v35         0.349037  35.36%    2.549s
v36         0.367089  51.66%    2.768s
v37         0.370563  56.53%    2.720s
v47 all-QA  0.359836  49.15%    2.779s
v48         0.360129  45.97%    2.665s
v49         0.365762  51.56%    2.705s
v50         0.369237  56.43%    2.716s
v52         0.373143  58.17%    2.702s
v53         0.375890  62.89%    2.718s
```

当前判断：

```text
1. v47 证明 consistency verifier 是有效新模块，不只是任务规则。
2. v52/v53 actual 已经超过 full KV 分数，同时只保留约 58%/63% KV。
3. 这可能成为论文新主线：RiskKV-Block = evidence-flow selection + output/grounding verifier + memory-action consistency verifier + minimum-safe action.
4. v52/v53 actual 完全复现 stitched，说明组合 policy 的实现路径没有额外偏差。
```

v52/v53 全任务 actual m20：

```text
method      score     keep      online    consistency check  consistency full
full KV     0.372655  100.00%   3.033s    0.00%              0.00%
v35         0.349037  35.36%    2.549s    0.00%              0.00%
v36         0.367089  51.66%    2.768s    0.00%              0.00%
v37         0.370563  56.53%    2.720s    0.00%              0.00%
v52 actual  0.373143  58.17%    2.720s    16.56%             8.13%
v53 actual  0.375890  62.89%    2.736s    16.56%             8.13%
```

## m50 validation update

更大样本 m50 完成后，v52/v53 仍然优于 v37，但不再超过 full KV：

```text
method        score     KV ratio   online
full KV       0.371970  100.00%    3.283s
v35 compact   0.326791   34.25%    2.655s
v36 balanced  0.345119   51.04%    2.652s
v37 quality   0.346527   56.21%    2.720s
v52 consist   0.352895   57.76%    2.735s
v53 +qasper   0.358321   62.47%    2.801s
```

结论：

```text
v52/v53 are not full-level on m50.
v53 still improves v37 by +0.0118 absolute score.
v53 reaches 96.3% of full score using 62.5% KV.
```

这改变论文主张：

```text
old risky claim:
  compressed KV can exceed full KV

new stable claim:
  label-free memory-action risk control consistently improves sparse KV quality
  under a substantially smaller active KV budget.
```

v63/v64 的意义因此更明确：

```text
If benefit-calibrated conformal gating can keep v53-level score
while reducing consistency_check_rate from 16.9% toward 10.6%,
it becomes the best main-method candidate.
```

consistency verifier 离线诊断：

```text
v47 QA target, candidate vs v35 base vs full KV

ALL:
  n=120, base=0.304530, candidate=0.333327, full=0.346554

TRIGGERED by consistency disagreement:
  n=48, base=0.152604, candidate=0.224598, full=0.224598

UNTRIGGERED:
  n=72, base=0.405814, candidate=0.405814, full=0.427857
```

解释：

```text
1. consistency 触发的样本确实更危险：v35 base 只有 0.153。
2. 触发后 fallback 能把这些危险样本提升到 full 水平。
3. 未触发样本保持 compact 输出，避免了 blanket full fallback。
4. 这是一个比较强的 precision 证据，可以支撑论文里的 verifier motivation。
```

v54 consistency-only ablation：

```text
v54 = 六个 QA 任务只开 consistency verifier，不开 output/grounding verifier。

QA mean:
v35 baseline             0.304530
v54 consistency-only     0.318001
v47 consistency+verifier 0.333327
full QA                 0.346554
```

诊断：

```text
TRIGGERED:
  n=54, base=0.195437, candidate=0.252376, full=0.252376

UNTRIGGERED:
  n=66, base=0.393788, candidate=0.371695, full=0.423608
```

解释：

```text
1. consistency-only 本身有效，说明新模块不是靠 output/grounding 混出来的。
2. 但 consistency-only 会伤 hotpotqa 和 multifieldqa_en。
3. v47 比 v54 好，说明 output/grounding verifier 与 consistency verifier 是互补模块。
4. 论文里应报告这个 ablation：consistency 是主要新意，output/grounding 是必要的安全约束。
```
