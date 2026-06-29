# PCIC Minimal Component Ablation（2026-06-29）

该表由 `scripts/run_pcic_minimal_component_ablation_suite.sh` / `scripts/summarize_pcic_minimal_component_ablation.py` 生成。
默认 suite 不跑模型；设置 `RUN_EXPERIMENTS=1` 才会启动实验。
可用 `ONLY_CASES="hard_memoryonly hard_nohistory hard_nopairwise"` 先跑 P0 小集合。
`historical` 表示该行复用了已有 corrected core CSV；`missing` 表示严格消融尚未实际运行。

## Results

| task | ablation | status | avg ΔPPL | method/base | gate_s | extended | early | combos |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| hard | `no_validation_anchor_top2` | `historical` | 0.004371 | 2.972 | 32.530 | 1 | 3 | `0,7/2,0,7,12/0,6/0,13` |
| hard | `memory_only_no_rescue` | `missing` | - | - | - | - | - | `outputs/server_pcic_ablate_hard_memoryonly_eval128_seed64_eager/pcic_r_blockwise_results.csv` |
| hard | `main_cond_rescue` | `historical` | -0.049633 | 6.227 | 89.624 | 4 | 0 | `0,6/2,0,7,12/0,6/0,6` |
| hard | `no_history_memory` | `missing` | - | - | - | - | - | `outputs/server_pcic_ablate_hard_nohistory_eval128_seed64_eager/pcic_r_blockwise_results.csv` |
| hard | `no_pairwise_probe` | `missing` | - | - | - | - | - | `outputs/server_pcic_ablate_hard_nopairwise_eval128_seed64_eager/pcic_r_blockwise_results.csv` |
| hard | `min_loss_no_pairwise_proxy` | `missing` | - | - | - | - | - | `outputs/server_pcic_ablate_hard_minloss_eval128_seed64_eager/pcic_r_blockwise_results.csv` |
| monte | `no_validation_anchor_top2` | `historical` | -0.219215 | 3.388 | 9.913 | 2 | 0 | `2,7/2,0` |
| monte | `main_cond_rescue` | `historical` | -0.219215 | 3.392 | 10.011 | 2 | 0 | `2,7/2,0` |
| monte | `memory_only_no_rescue` | `missing` | - | - | - | - | - | `outputs/server_pcic_ablate_monte_memoryonly_seed64_eager/pcic_r_blockwise_results.csv` |
| monte | `no_history_memory` | `missing` | - | - | - | - | - | `outputs/server_pcic_ablate_monte_nohistory_seed64_eager/pcic_r_blockwise_results.csv` |
| monte | `no_pairwise_probe` | `missing` | - | - | - | - | - | `outputs/server_pcic_ablate_monte_nopairwise_seed64_eager/pcic_r_blockwise_results.csv` |
| monte | `min_loss_no_pairwise_proxy` | `missing` | - | - | - | - | - | `outputs/server_pcic_ablate_monte_minloss_seed64_eager/pcic_r_blockwise_results.csv` |
| ruler_variable | `no_validation_anchor_top2` | `historical` | 0.017397 | 3.699 | 24.768 | 2 | 1 | `2,0/2,0,7,12/2,0` |
| ruler_variable | `memory_only_no_rescue` | `missing` | - | - | - | - | - | `outputs/server_pcic_ablate_rulervar_memoryonly_seed64_eager/pcic_r_blockwise_results.csv` |
| ruler_variable | `main_cond_rescue` | `historical` | -0.000564 | 5.388 | 41.103 | 3 | 0 | `0,13/2,0/2,0` |
| ruler_variable | `no_pairwise_probe` | `missing` | - | - | - | - | - | `outputs/server_pcic_ablate_rulervar_nopairwise_seed64_eager/pcic_r_blockwise_results.csv` |

## Interpretation

- `no_validation_anchor_top2`：保留 pairwise/horizon top-k rescue，但去掉 validation-prior anchor。
- `memory_only_no_rescue`：使用 `--combo_select_policy risk_memory`，不运行 sentinel/horizon candidate arbitration，作为严格 no-rescue 对照。
- `main_cond_rescue`：当前主方法，对照 rescue gate 是否修复 failure。
- `no_history_memory`：设置 `--risk_memory_use_history false`，不 seed、不更新跨 block 历史，测试 historical prior 的贡献。
- `no_pairwise_probe`：设置 `--pairwise_candidate_probe false`，关闭候选间 sentinel/horizon 对比，只保留 memory anchor。
- `min_loss_no_pairwise_proxy`：用 calibration min-loss 作为额外负对照；它不是严格 no-pairwise，只用于辅助解释。

missing strict cases: 10
