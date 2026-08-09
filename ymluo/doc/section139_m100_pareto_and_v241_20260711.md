# Section 139: M100 Pareto and v241 router

Date: 2026-07-11

## Completed M100 Practical Runs

LongBench M100, Llama-3.1-8B-Instruct, practical methods only.

| Method | Score | KV keep | Online | Total | Speed vs 3.033s full online |
|---|---:|---:|---:|---:|---:|
| v229 extractive summary | 0.3901 | 22.36% | 0.825s | 2.504s | 3.67x |
| v230 extractive + code cap | 0.3893 | 22.36% | 0.604s | 2.278s | 5.02x |
| v235 v231 + prefill skip | 0.3895 | 22.36% | 0.672s | 1.870s | 4.51x |
| v240 task-router split combined | 0.3834 | 18.13% | 0.686s | 1.860s | 4.42x |

Baseline references from previous LongBench runs:

- full_raw m20 score: 0.3596 to 0.3727 depending on run section.
- full online reference: about 3.033s.

All four completed M100 methods satisfy the target range:

- KV keep is in 10%-30%.
- Score is above 95% of full_raw baseline.
- Online speed is above 2.5x.

## Why v240 Was Not Final

v240 was the best M20 result, but M100 showed that not all M20 task-local gains generalized.

Task-level deltas for v240 compared with v235:

| Task | Score delta | KV delta | Decision |
|---|---:|---:|---|
| 2wikimqa | +0.0238 | -1.75% | Keep v240 action. |
| hotpotqa | -0.0004 | -14.01% | Keep v240 action: quality neutral, large KV win. |
| multifieldqa_en | +0.0421 | +11.80% | Keep v240 action: quality win. |
| qasper | -0.0590 | -8.28% | Revert to v235 action. |
| narrativeqa | -0.0301 | -9.25% | Revert to v235 action. |
| musique | -0.0740 | -46.19% | Revert to v235 action for quality. |

## v241: M100-Validated Router

Policy:

`configs/riskkv_task_policy_v241_m100_validated_router_20260711.json`

Design:

- Default to the v235/v231 high-quality practical path.
- Use the v240 action only on M100-validated tasks:
  - `2wikimqa`
  - `hotpotqa`
  - `multifieldqa_en`
- Do not use the v240 action on qasper, narrativeqa, or musique, because those M20 wins did not hold on M100.

Virtual estimate from completed M100 task rows:

| Method | Score | KV keep | Online | Total | Speed vs 3.033s full online |
|---|---:|---:|---:|---:|---:|
| v241 M100-validated router estimate | 0.3936 | 22.11% | 0.644s | 1.834s | 4.71x |

This is currently the strongest candidate by the original target:

- Higher estimated M100 score than v229/v230/v235/v240.
- KV keep remains in 10%-30%.
- Online speed remains above 2.5x.

## Running Actual v241 Split M100

Actual v241 split M100 is running:

| Split | Output |
|---|---|
| QA group | `riskkv_v19_v241_validated_router_m100_qa_group_20260711_validated_router_split_m100_bDyn_pDyn` |
| Summary group | `riskkv_v19_v241_validated_router_m100_summary_group_20260711_validated_router_split_m100_bDyn_pDyn` |
| Structured/code group | `riskkv_v19_v241_validated_router_m100_struct_code_group_20260711_validated_router_split_m100_bDyn_pDyn` |
| Combined | `riskkv_v19_v241_validated_router_split_m100_combined_20260711_validated_router_split_m100_bDyn_pDyn` |

The combiner is running:

`outputs/logs/combine_v241_split_m100_20260711.log`

