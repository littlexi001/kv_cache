# PCIC Minimal Component Ablation Runbook（2026-06-29）

目的：用最小服务器实验补齐 paper 主线最容易被审稿人追问的三个消融：

```text
no-rescue / no-memory / no-pairwise
```

该 runbook 明确只同步少量脚本/代码文件，不下载外部数据，不占用服务器网络带宽。

## 1. 需要同步的小文件

从本地同步到服务器项目目录：

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
```

文件：

```text
src/run_pcic_rescue_blockwise_local.py
scripts/run_pcic_minimal_component_ablation_suite.sh
scripts/summarize_pcic_minimal_component_ablation.py
docs/pcic_minimal_component_ablation_2026_06_29.md
docs/pcic_component_evidence_matrix_2026_06_29.md
docs/horizon_pcic_method_spec_2026_06_29.md
```

这些都是小文本文件；不需要同步模型、数据集或大输出。

## 2. 服务器环境约束

进入服务器后：

```bash
cd /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE=1
```

默认模型路径：

```bash
MODEL=/home/fdong/hrj/prove/Qwen3-0.6B
PY=/home/fdong/miniconda3/envs/moe/bin/python
```

Hard-topic 文本：

```bash
HARD=/home/fdong/ymluo/projects/influence_bounded_synthetic_kv/data/hard_topic_eval_v2.txt
```

RULER-style variable 文本若存在则自动运行：

```bash
data/pcic_ruler_style_variable_eval_2026_06_29.txt
```

若不存在，suite 会跳过 RULER variable，不会联网下载。

## 3. 先做 dry-run / summary-only

不跑模型，只生成当前表：

```bash
bash scripts/run_pcic_minimal_component_ablation_suite.sh
```

预期：

- 已有 corrected core 结果会以 `historical` 状态填充；
- 严格消融还会显示 `missing`；
- 输出文档：

```text
docs/pcic_minimal_component_ablation_2026_06_29.md
docs/pcic_minimal_component_ablation_2026_06_29.csv
```

## 4. 正式运行严格消融

只在服务器空闲时运行：

```bash
RUN_EXPERIMENTS=1 bash scripts/run_pcic_minimal_component_ablation_suite.sh
```

该 suite 会跳过已有输出，因此可断点续跑。

如果想先只跑较少任务，不需要改脚本，使用 `ONLY_CASES` 过滤。`ONLY_CASES` 是空格分隔的 case-name 子串。

P0 最小运行：

```bash
ONLY_CASES="hard_memoryonly hard_nohistory hard_nopairwise" \
RUN_EXPERIMENTS=1 \
bash scripts/run_pcic_minimal_component_ablation_suite.sh
```

P1 扩展到 RULER-style variable：

```bash
ONLY_CASES="rulervar_memoryonly rulervar_nopairwise" \
RUN_EXPERIMENTS=1 \
bash scripts/run_pcic_minimal_component_ablation_suite.sh
```

P2 扩展到 Monte：

```bash
ONLY_CASES="monte_memoryonly monte_nohistory monte_nopairwise" \
RUN_EXPERIMENTS=1 \
bash scripts/run_pcic_minimal_component_ablation_suite.sh
```

优先级建议：

```text
P0:
  hard memory_only_no_rescue
  hard no_history_memory
  hard no_pairwise_probe

P1:
  ruler_variable memory_only_no_rescue
  ruler_variable no_pairwise_probe

P2:
  monte memory_only_no_rescue
  monte no_history_memory
  monte no_pairwise_probe
```

原因：

- Hard-topic eval128 已有明确 delayed-win failure，是证明 rescue gate 的最好 case；
- RULER-style variable 已有 top2 failure，是证明不只在 hard-topic 上有效的最好 smoke；
- Monte 主要证明 online selection / fixed policy 差异。

## 5. 消融定义

| ablation | 实现 | 解释 |
| --- | --- | --- |
| `no_validation_anchor_top2` | historical fallback / no anchor top2 | 保留 pairwise/horizon top-k rescue，但不加 validation prior anchor |
| `memory_only_no_rescue` | `--combo_select_policy risk_memory` | 严格无 sentinel/horizon candidate arbitration |
| `main_cond_rescue` | historical fallback / cond anchor | 当前主方法 |
| `no_history_memory` | `--risk_memory_use_history false` | 不 seed、不更新跨 block risk memory |
| `no_pairwise_probe` | `--pairwise_candidate_probe false` | 关闭候选间 sentinel/horizon 对比，只保留 memory anchor |
| `min_loss_no_pairwise_proxy` | `--combo_select_policy min_loss` | 额外负对照，不是严格 no-pairwise |

## 6. 成功判据

### Rescue gate 必要性

期待：

```text
main_cond_rescue 优于 memory_only_no_rescue
main_cond_rescue 优于 no_validation_anchor_top2
```

特别看：

```text
hard / ruler_variable 的 avg_delta_ppl
```

如果 `memory_only_no_rescue` 接近主方法，说明 rescue gate 的必要性证据不足，需要找更非平稳的任务或更长 block。

### Risk memory 必要性

期待：

```text
no_history_memory 的 block trace 更不稳
或 avg_delta_ppl 劣于 main_cond_rescue
```

如果差异不明显，论文中不能强写 memory 是核心贡献，只能写成稳定化组件。

### Pairwise / horizon candidate probing 必要性

期待：

```text
no_pairwise_probe 劣于 main_cond_rescue
```

如果差异不明显，则说明当前候选集或任务不够暴露 pairwise arbitration 的价值；需要在 RULER variable / hard-topic b8 上继续找 case。

## 7. 结果写法边界

可以写：

```text
Horizon-PCIC 的核心贡献是 online counterfactual policy selection；
conditional rescue gate 在 hard-topic / RULER-style variable 上修复 short-horizon failure；
fixed policy 与 memory-only selector 不足以覆盖所有非平稳 block。
```

不能写，除非后续实验补齐：

```text
端到端速度已经超过 baseline；
Pairwise-CIC / risk memory 在所有任务上都不可或缺；
已在正式 LongBench/RULER 上充分验证。
```

## 8. 运行后需要检查

```bash
cat docs/pcic_minimal_component_ablation_2026_06_29.md
```

重点看：

```text
missing strict cases: 0
hard memory_only_no_rescue vs main_cond_rescue
hard no_pairwise_probe vs main_cond_rescue
ruler_variable no_pairwise_probe vs main_cond_rescue
```

若有失败日志：

```bash
ls outputs/logs/*ablate*.log
tail -n 80 outputs/logs/<failed>.log
```

## 9. 后续接入

跑完后需要更新：

```text
docs/pcic_component_evidence_matrix_2026_06_29.md
docs/horizon_pcic_method_spec_2026_06_29.md
docs/horizon_pcic_icml_readiness_2026_06_29.md
```

如果严格消融支持预期结论，则 paper 主线的创新性证据会更接近可投稿标准；如果不支持，则需要收缩 claim，把 Pairwise-CIC / memory 写成辅助机制而不是核心不可替代贡献。
