# Experiments

`main/experiments/` 是执行与证据层。这里不存放所有历史实验，只保留和当前正在验证的命题最相关的 active experiment line。

当前 active line：

```text
slot_context_dominance_router_specialization/
```

它服务的当前命题是：普通 top-1 MoE router / expert bucket 是否能形成稳定、可解释、具有 causal utility 的 feature-level expert specialization。

## Active Experiments

| Experiment | Hypothesis | Status | Decision |
|---|---|---|---|
| `slot_context_dominance_router_specialization/execution_plan.md` | stronger slot context plus slot-centroid router initialization 是否足以产生 slot-aligned routing 和 slot-specific expert utility | pre-run | 用 assignment-utility agreement 判断 weak context signal 解释是否成立 |
| `H0529a_zipfian_frequency_shortcut/summary.md` | token-Zipfian frequency 是否放大 flat router tail specialization failure | completed | Zipfian 改变 load structure，但没有削弱 tail $S_{\mathrm{sense}}$；frequency alone 不是充分解释 |
| `H0530a_hierarchical_common_sense_moe/summary.md` | common/sense hierarchy 是否改善 Zipfian tail functional specialization | completed | hierarchy 提高 token/family routing association，但 tail functional specialization 和 route-function alignment 变差 |
| `H0521_router_feature_specialization/H0526_top1_sparse_router_failure_mode/summary.md` | trainable top-1 sparse router 是否形成 functional expert specialization | completed | sparse audit 通过、任务学会，但 AB-specific specialization 不稳定；下一步只加 load-balance |
| `H0521_router_feature_specialization/H0525a_trainable_router_a_reproduction/summary.md` | H0524a pattern 是否在 trainable-router corrected A 中复现 | completed | router gradient/delta 通过；split 不复现；不进入 B trajectory |
| `H0521_router_feature_specialization/H0521a_seq2_ab_feature_specialization/summary.md` | 最小 `seq_len=2` A/B/AB setup 是否形成干净 expert bucket | completed | routing assignment 混合；functional mixing 尚未证明 |
| `H0521_router_feature_specialization/H0521b_seq32_contiguous_ab/summary.md` | `seq_len=32` source-token routing 是否形成干净 A/B/background bucket | completed | source-token bucket 不干净；下一步做 source-token rule audit |

## Archive

旧实验不删除，但降级到：

```text
archive/
```

Archive 的作用是保留历史证据、旧计划和可追溯材料；它不是当前命题判断的第一阅读入口。

| Archived folder | Why archived |
|---|---|
| `archive/E0_E3_synthetic_data_understanding/` | early synthetic data understanding line; now only background support |
| `archive/E1_E2_E15_E25_synthetic_router_retrieval/` | old router retrieval planning line |
| `archive/D1_D2_dclm_transfer_router_retrieval/` | DCLM transfer plan; not current active proposition |
| `archive/R1_synthetic_router_retrieval_0512/` | prior summarized evidence; use only when tracing old retrieval claims |
| `archive/R3_perfect_router_retrieval/` | prior perfect-router condition; supporting material |
| `archive/R4_attention_output_router_retrieval/` | prior attention-output router retrieval line; supporting material |

KV retrieval 相关实验暂时只作为 downstream-effect archive。当前不要从
retrieval 指标反推 specialization 成立；先看 route assignment 和 expert
causal utility 是否对齐。

## Experiment Folder Contract

每个 active experiment folder 应包含：

```text
README.md
execution_plan.md
summary.md
detailed.md
figures/
```

`summary.md` 承接旧 closure 的作用，是实验完成后的第一阅读入口。它必须 self-contained：对应 anchor、最重要指标、直接结果、解释、限制、claim update、关键图片和下一步。普通实验结果不再另写 hypothesis closure。

`detailed.md` 放最完整的实验记录：数据构造、模型与训练设置、条件矩阵、seed、命令记录、job id、完整表格、关键图、失败或部分运行、artifact map、失败分析和补充诊断。它必须先放一个快速 recap，再解释最重要指标为什么决定结论，最后列补充指标。

`detailed.md` 必须以这个部分开头：

```markdown
## 0. Quick Recap

目的：

假设：

实验思路：

结论：

证据：
```

这个 recap 是给快速恢复上下文用的，不替代后面的证据分析。

`detailed.md` 后文必须扩展同一条研究故事，并明确链接：

- anchor；
- `summary.md`；
- sync report, if any；
- code workspace；
- runner / launch script；
- config；
- key code files；
- data or manifest；
- result dir；
- figure dir；
- key tables；
- logs / checkpoints；
- repro command。

写完 `detailed.md` 后，active anchor 的 evidence 区必须反向链接到这个
`detailed.md`。

如果有关键图片，必须复制到报告所在文件夹的 `figures/`，并在 `summary.md` 和 `detailed.md` 中用相对路径嵌入，同时写出图片支持的判断。这是强制规则；有图但不嵌入的 summary / detailed 视为不完整。

可运行脚本、配置、原始数据、模型权重、完整输出和 runtime figures 默认放在：

```text
XingyuD/Attention_Search_Experiments/
```

`main/experiments/` 只保留能帮助理解 hypothesis judgment 的计划、摘要、关键表格、关键图和可复盘结论。
