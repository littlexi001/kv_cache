# Section 98: 输入侧预算规划器与当前 ICML 缺口（2026-07-07）

## 当前判断

目前主线方法应写成：

**Risk-constrained KV budget planning with RoPE-aware cache repacking and output-level safety verification**

它和 RAG / prompt compression 的边界已经比较清楚：

- 不重新构造 prompt 作为主路径。
- 先做 full-context prefill，保留原模型 KV 表示。
- 在 KV cache 内选择和重排页面，使用 RoPE-aware repack 保持 compact query position。
- 用 risk planner / verifier 决定最小安全 KV budget。

但现在还不能把 output-level verifier prefix 作为最终唯一形态，因为它在 LongBench 4k 上为了验证多个 candidate 会产生额外 decode 开销。

## 已验证的关键结果

### 13-task mixed, 4k, m=1

`output_level_risk_kv_planner`：

- Score: 69.23%，match full。
- KV: 14.05%。
- Online speed: 0.925x（sum-based aggregation）。

加 `floor2` 后：

- Score: 69.23%，仍 match full。
- KV: 26.08%。
- Online speed: 1.019x。
- E2E speed: 1.013x。

结论：短上下文里，跳过过小且不稳定的 k1 能把速度拉回到略高于 full；这是一个有用的 runtime policy 证据。

### LongBench, 4k, m=4

原始 output verifier prefix：

- Score: 20.00%，match full。
- KV: 18.75%。
- Online speed: 0.711x。

`floor2`：

- Score: 20.00%，match full。
- KV: 29.38%。
- Online speed: 0.801x。

`floor3`：

- Score: 20.00%，match full。
- KV: 43.75%。
- Online speed: 0.768x。

结论：LongBench 的主要问题不是 KV 选太少，而是 output verifier 在运行时反复 decode candidate。继续加 floor 只能牺牲 KV，不足以解决速度。

### RULER scaling

RULER 8k + floor2：

- Score: 100%，match full。
- KV: 13.14%。
- Online speed: 1.080x。
- E2E speed: 1.037x。

RULER 16k, m=1, 8 tasks sharded：

- Score: 100%，match full。
- KV: 6.40%。
- Online speed: 1.702x。
- E2E speed: 1.190x。

结论：长度上来后，KV 压缩开始兑现成真实 online speedup。这个 scaling 是目前最强的论文证据之一。

## 当前缺口

1. LongBench 4k 仍然亏速，说明 output-level verifier 不适合作为唯一 runtime policy。
2. RULER 16k m=2 的整包运行会在 24GB GPU 上 OOM，需要 case-level sharding 才能完成。
3. 现有样本量还偏小，m1/m4 结果不能直接支撑 ICML 主实验。
4. 需要把方法从“输出后验证”推进到“输入侧风险预算规划 + 可选输出验证”的组合结构。

## 下一步方法形态

建议把最终方法写成两层：

1. **Input-side risk-constrained budget planner**
   - 输入：retriever gap、top-k stability、task family、context length、layout features。
   - 输出：最小安全 KV budget，例如 k2/k3/k4/k6/full。
   - 优点：运行时只 decode 一次，能解决 LongBench 的 verifier 开销。

2. **Output-level verifier / fallback**
   - 用于高风险或分布外样本。
   - 在长上下文或风险高时做 safety check。
   - 不作为所有样本的默认路径。

这样创新点会比单纯 prompt/router 更强：它是一个 cache-native、risk-calibrated、runtime-aware 的 KV memory controller。

## 已完成的新代码准备

本地已加入：

- `--case_start`
- `--case_limit`
- `--runtime_methods`

用于把长上下文 benchmark 按 case 分片运行，避免 16k 多样本单进程 OOM。
`--runtime_methods` 用于显式只跑目标方法，例如 `full_kv_cache,variable_budget_kv_planner` 或 `full_kv_cache,output_level_risk_kv_planner`。这可以跳过 prompt rebuild、naive gather、shifted RoPE 等辅助 baseline，降低 16k 显存压力，也让 input-side planner 的 runtime 测量更干净。

本地还修复了：

- `RuntimeVariableBudgetPlanner` 现在会尊重 checkpoint 里的 `use_text_features`。
- 如果训练时不用 text features，runtime 会填零，避免训练/运行分布不一致。

已准备但尚未因 SSH 不通而同步的脚本：

- `scripts/run_variable_budget_runtime_sweep_20260707.sh`
- `scripts/summarize_runtime_scaling_20260707.py` 已扩展支持 `variable_budget_kv_planner`。
- `scripts/sync_and_launch_variable_budget_runtime_20260707.ps1` 可在 Windows 本地一键检测 SSH、同步补丁、远端语法检查并后台启动 variable-budget sweep。
- `scripts/collect_icml_runtime_status_20260707.ps1` 可在 Windows 本地一键检查远端进程/GPU/summary 状态，运行汇总脚本，把 `runtime_scaling_summary_20260707` 拉回本地，并自动生成 ICML 主表、SVG 图和 readiness report。
- `scripts/icml_runtime_manifest_20260707.json` 记录 8 个主实验、3 个 scaling 补充实验、同步文件列表和验收标准。
- `scripts/audit_icml_runtime_manifest_20260707.py` 会检查 manifest、汇总脚本、方法名和关键输出目录是否一致。
- `scripts/test_runtime_controls_20260707.py` 会加载临时 planner checkpoint，测试 `--runtime_methods`、`selected_tau` 和 `use_text_features` 的实际行为。
- `scripts/make_icml_readiness_report_20260707.py` 会把 LongBench、Mixed13、RULER 8k 和 RULER 扩样结果按门槛自动判定为 `PASS`、`BORDERLINE`、`FAIL` 或 `PENDING`。

补充：runtime 现在支持 `--variable_budget_tail_threshold -1`，表示直接读取 conformal checkpoint 中的 `selected_tau`。因此后续不仅会跑 fixed `tail_threshold=0.35`，也会跑 `conformal_auto`。这个版本更适合作为论文主方法，因为阈值来自校准集，而不是手工挑选。

## 服务器恢复后要做

1. 同步以下文件到服务器：
   - `src/run_rope_aware_kv_repack_benchmark.py`
   - `scripts/summarize_runtime_scaling_20260707.py`
   - `scripts/run_variable_budget_runtime_sweep_20260707.sh`
   - `scripts/run_output_verifier_floor_sweep_20260707.sh`
   - `scripts/run_ruler_scaling_expansion_20260707.sh`
   - `scripts/run_ruler16k_case1_shards_20260707.sh`
   - `scripts/run_ruler16k_floor2_recovery_20260707.sh`

   或者直接在本地运行：

   ```powershell
   powershell -ExecutionPolicy Bypass -File ymluo/projects/learned_hierarchical_summary_memory/scripts/sync_and_launch_variable_budget_runtime_20260707.ps1
   ```

   这个脚本会先检查 `fdong@10.176.37.31` 是否可 SSH，通了才会同步和启动。

   启动后可以运行：

   ```powershell
   powershell -ExecutionPolicy Bypass -File ymluo/projects/learned_hierarchical_summary_memory/scripts/collect_icml_runtime_status_20260707.ps1
   ```

   它会检查 GPU、后台进程、关键 summary 文件，并拉取最新 aggregate 表；同时生成 `icml_tables`、`icml_figures` 和 `icml_readiness/icml_readiness_report.md`。

2. 启动 variable-budget runtime sweep：
   - LongBench m4 best-calibrated planner。
   - LongBench m4 min-safe planner。
   - LongBench m4 conformal-auto planner。
   - mixed13 best-calibrated planner。
   - mixed13 min-safe planner。
   - mixed13 conformal-auto planner。
   - RULER8k best-calibrated planner。
   - RULER8k conformal-auto planner。

3. 继续等待 / 收尾：
   - RULER 4k m3 floor2。
   - RULER 8k m3 floor2。
   - RULER 16k case1 shards。

4. 汇总后看两个硬指标：
   - LongBench 是否从 0.80x 左右提升到接近或超过 1.0x online。
   - RULER m3/m2 是否继续保持 full-level score 和随长度增长的 speedup。

5. 对照 `scripts/icml_runtime_manifest_20260707.json` 和 `icml_readiness/icml_readiness_report.md` 做验收：
   - LongBench m4 必须 match full score。
   - 至少一个 input-side planner 要明显优于 output verifier floor2/floor3 的 online speed。
   - mixed13 要保持 full-level score 且 online speed 接近或超过 1.0x。
   - RULER 8k 要保持 full-level score 且 online speed > 1.0x。
   - RULER m3/m2 扩样结果要支撑 scaling claim。

   本地可先运行：

   ```powershell
   python ymluo/projects/learned_hierarchical_summary_memory/scripts/audit_icml_runtime_manifest_20260707.py
   python ymluo/projects/learned_hierarchical_summary_memory/scripts/test_runtime_controls_20260707.py
   ```

   当前审计通过：8 个主实验、3 个 scaling 补充实验都已被汇总脚本覆盖；runtime control 测试也通过。

## 对 ICML 的现实判断

当前结果已经有方法雏形和一个强 scaling 现象，但还不够稳妥：

- 如果 input-side planner 能在 LongBench/mixed 上 match full 且 online speed 接近或超过 1x，同时 RULER 8k/16k 扩展样本保持 100% 或接近 full，那么方法主线可以支撑一篇强 workshop / 有希望冲主会的论文。
- 如果 LongBench 仍然亏速，只能把主贡献收窄为 long-context KV runtime scaling，ICML 主会风险会比较高。

因此，下一步最关键的不是再做一个更复杂的 verifier，而是证明 input-side planner 能把安全性和速度同时拿回来。
