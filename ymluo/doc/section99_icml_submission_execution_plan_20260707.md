# Section 99: ICML 投稿执行计划与论文主线（2026-07-07）

## 一句话主线

我们要投的不是 “prompt 压缩” 或普通 RAG，而是：

**面向长上下文推理的风险约束 KV cache 预算控制器。**

核心路径是：

1. 对完整上下文做一次 full prefill。
2. 在 KV cache 空间中选择少量页面。
3. 用 RoPE-aware repack 保持 compact query position。
4. 用输入侧 risk planner 预测最小安全 KV budget。
5. 对高风险样本可选使用 output-level verifier fallback。

论文应该强调 cache-native memory controller，而不是 text reconstruction。

## 建议方法名

短名：

**RiskKV**

完整名：

**Risk-Constrained KV Budget Planning with RoPE-Aware Cache Repacking**

如果 conformal planner 的 runtime 结果足够好，可以写成：

**Conformal RiskKV**

这样比 “two-stage calibrated” 或 “router” 更像论文方法，也更容易和 RAG/prompt compression 区分开。

## 目前已有证据

### 已经比较强的点

1. Cache-native 路径已经跑通：full prefill + KV gather/repack + compact decode。
2. RoPE-aware compact query position 明显优于 naive/shifted 版本。
3. Output-level verifier 在 replay 和 runtime 都能 match full，并显著降低 KV。
4. Input-side RiskKV 已经在 RULER 8k/16k 出现真实 online speedup：
   - 8k m5 conformal floor2: 100.00% score, 18.25% KV, 1.075x online。
   - 16k m3 conformal: 100.00% score, 8.44% KV, 1.669x online。
5. 16k OOM 问题已经通过 case sharding 和 `--runtime_methods` 找到工程解法。
6. 最新 readiness report 是 `ICML_CANDIDATE`，主线可以按 RiskKV input planner 推进。

### 仍不足的点

1. LongBench/Mixed13 速度仍只是接近持平，不应声称短上下文显著加速。
2. 4k m5 需要 k2 safety floor 才能 match full；这要写成明确的 lower-bound budget 机制。
3. 需要补 overhead 消融，解释 planner/repack 固定开销为什么在短上下文抵消速度收益。
4. 需要把论文主张收窄为 cache-native long-context serving scaling。

2026-07-07 晚上更新：variable-budget runtime、LongBench m8、Mixed13 m2、RULER m5/m3 和 conformal floor2 recovery 已跑完，readiness report 当前为 `ICML_CANDIDATE`。详细结果见 `section101_riskkv_runtime_results_20260707.md`。

## 最关键的下一批实验

### 主实验 A：Input-side planner runtime

脚本：

`scripts/run_variable_budget_runtime_sweep_20260707.sh`

目标：

验证输入侧 planner 是否能解决 LongBench 上 output verifier 的多候选 decode 开销。

已经得到的关键结果：

| Setting | Method | Score | KV | Online speed | E2E speed |
|---|---|---:|---:|---:|---:|
| LongBench m8 | full | 25.00% | 100% | 1.000x | 1.000x |
| LongBench m8 | conformal-auto | 32.50% | 26.24% | 0.988x | 0.992x |
| LongBench m4 | conformal-auto | 25.00% | 26.25% | 0.996x | 0.997x |
| LongBench m4 | output verifier floor2 | 20.00% | 29.38% | 0.801x | 0.864x |

过线条件：

- LongBench score 必须 match 或超过 full。
- 至少一个 input-side planner online speed 明显高于 output verifier floor2/floor3。
- 如果能接近或超过 1.0x，主方法叙事成立；当前是质量/KV 很强、速度接近持平。

### 主实验 B：Mixed13 runtime

目的：

证明方法不是只在 RULER synthetic tasks 有效。

过线条件：

- Score 维持 69.23% full-level。
- Online speed 接近或超过 1.0x。
- KV 明显低于 full，最好低于 30%。

当前 m2 结果：

- min-safe: 69.23% score / 23.04% KV / 0.993x online。
- conformal: 65.38% score / 15.53% KV / 0.991x online。

### 主实验 C：RULER scaling

当前最强 scaling 证据来自 RULER：

| Length | Samples | Score | KV | Online speed |
|---:|---:|---:|---:|---:|
| 4k | 40 | 100.00% | 26.30% | 0.991x |
| 8k | 40 | 100.00% | 18.25% | 1.075x |
| 16k | 24 | 100.00% | 8.44% | 1.669x |

已经补完：

- RULER 4k/8k m5 input planner。
- RULER 16k m3 case2 shards。
- RULER 4k/8k m5 conformal floor2 recovery。

过线条件：

- 扩样后仍接近 full-level score。
- 8k/16k speedup 趋势保持。

## 必做消融

### 消融 1：KV-native vs prompt rebuild

要证明不是 prompt compression：

| Method | Uses KV cache | Rebuilds prompt | Score | KV/token ratio | Online speed |
|---|---:|---:|---:|---:|---:|
| Prompt rebuild | No | Yes | | | |
| RoPE compact k2 | Yes | No | | | |
| RiskKV | Yes | No | | | |

重点写法：

prompt rebuild 可能 E2E 看起来快，因为它不做 full prefill；但它不是服务端多轮 decode 的同一计算模型。我们的在线指标更贴近已-prefill 长上下文服务场景。

### 消融 2：RoPE-aware repack

比较：

- naive absolute query position。
- compact query position without RoPE correction。
- shifted query position。
- RoPE-aware compact query position。

目的：

证明不是简单裁 KV，而是需要位置一致性修正。

### 消融 3：Planner 形态

比较：

- fixed k2。
- output verifier prefix。
- input-side variable planner。
- conformal-auto planner。

目的：

证明最终方法兼顾安全和速度：output verifier 安全但慢，input-side planner 快，conformal-auto 提供可校准风险。

### 消融 4：Risk threshold / safety floor

比较：

- no floor。
- floor2。
- learned/conformal threshold。

目的：

解释 RULER 8k 中 k1 失败，证明 long-context lower bound 是必要机制，不是事后补丁。

## 论文图表清单

### Figure 1: System diagram

流程：

Full context prefill -> page scoring -> budget planner -> RoPE-aware KV repack -> compact decode -> optional verifier/fallback。

重点：

画 KV cache，不画文档检索框，避免看起来像 RAG。

### Figure 2: Speed scaling

x 轴：context length。

y 轴：online speedup。

曲线：

- full baseline: 1.0。
- fixed k2 RoPE compact。
- RiskKV / output verifier floor2。
- prompt rebuild 可作为灰色参考。

### Figure 3: Accuracy-KV Pareto

x 轴：KV ratio。

y 轴：score 或 exact match。

展示 full、fixed k、prompt rebuild、RiskKV。

### Table 1: Main runtime results

覆盖：

- LongBench m4。
- Mixed13。
- RULER 4k/8k/16k。

### Table 2: Ablations

覆盖：

- RoPE mode。
- planner type。
- floor/conformal risk。

### Table 3: Risk calibration

如果 conformal-auto 成功，需要报告：

- selected tau。
- calibration risk bound。
- test failure rate。
- KV ratio。

## 写作风险与应对

### 风险 1：LongBench full score 太低

应对：

不要把论文主张写成“提高 QA 准确率”。主张应是：

在不牺牲 full-context model output 的前提下，降低 active KV 并提升长上下文在线 decoding 效率。

### 风险 2：短上下文速度不明显

应对：

明确系统目标是 long-context serving。4k 是 sanity check，8k/16k 是核心场景。

### 风险 3：方法看起来像简单 router

应对：

强调三个非平凡部分：

1. RoPE-aware cache repacking。
2. Risk-constrained variable budget selection。
3. Output-level verifier / conformal fallback 的安全闭环。

### 风险 4：RULER synthetic 不够

应对：

LongBench/mixed13 必须作为 main table 的一部分；RULER 主要支撑 length scaling。

## 投稿判断门槛

### 可以冲 ICML 主会的条件

1. input-side planner 在 LongBench m4 match full，且 online speed 明显优于 output verifier floor2/floor3。
2. mixed13 match full，online speed 接近或超过 1.0x。
3. RULER 8k/16k 扩样后保持 full-level score 和 speedup。
4. conformal-auto 至少不明显差于 bestcal，并能提供风险校准叙事。
5. 消融证明 RoPE-aware repack 和 risk planning 都必要。

### 更适合 workshop / CCFB 的条件

1. RULER scaling 很强，但 LongBench 仍明显亏速。
2. input-side planner 不能稳定 match full。
3. 结果主要依赖手工 floor，而 conformal/learned risk 不稳。

## 当前下一步

服务器 `10.176.37.31` 已恢复，m5/m3 和 floor2 recovery 已跑完。下一步：

1. 运行收集脚本：

   ```powershell
   powershell -ExecutionPolicy Bypass -File ymluo/projects/learned_hierarchical_summary_memory/scripts/collect_icml_runtime_status_20260707.ps1
   ```

2. 运行 manifest 审计：

   ```powershell
   python ymluo/projects/learned_hierarchical_summary_memory/scripts/audit_icml_runtime_manifest_20260707.py
   ```

3. 查看 readiness report：

   ```powershell
   Get-Content ymluo/projects/learned_hierarchical_summary_memory/outputs/remote_runtime_status_20260707/runtime_scaling_summary_20260707/icml_readiness/icml_readiness_report.md
   ```

4. 根据 RULER m5/m3 扩样结果更新主表和论文 claim：
   - 若 m5/m3 仍保持 8k/16k speedup，主线维持 RiskKV input planner。
   - 若 4k 仍接近持平，把 4k 明确写成 sanity check，不作为加速 claim。
   - 若 conformal 在 Mixed13 上略掉点，主表报告 min-safe，conformal 放 risk/KV trade-off。
