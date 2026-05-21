# Inverse KV Experiment Index

这个文件用于记录 `checkpoints/`、`logs/` 和 `experiments/` 中已经跑过的主要实验。表里的 accuracy 优先来自 `experiments/*.json` 的评估结果；没有 JSON 评估时，记录训练 log 中能看到的 final token loss。

注意：部分早期 checkpoint 目录仍使用 `oracle-*` 命名。这里统一解释为 **ground-truth dispatch**，避免和实验室语境里的 oracle 混淆。

## 主要结论速查

1. **Uniform baseline**：标准 learned MoE 大约 `91.5%` NTP accuracy。
2. **Zipf baseline**：标准 learned MoE 在 zipf 数据上大约 `94.4%` NTP accuracy。
3. **Ground-truth higher-level dispatch 最强**：zipf + higher-level + hash / frequency-balanced 大约 `94.6%~94.7%`。
4. **Supervised gate 没带来 NTP 提升**：即使监督 gate，NTP 仍大约 `93.95%`，接近 baseline。
5. **Naive inhibition 没证明 specialization**：NTP 接近 baseline，但 routing 主要 collapse 到单 expert，不能视为有效 slot specialization。
6. **Load-balance sweep 不改善 NTP**：zipf + attention-output router 加 LB 后 NTP 基本不变，但会破坏 hot expert locality。

## Experiment Table

| Family | Checkpoint / run | Log | Data | Main variables | Step | NTP loss | NTP acc | Result source | Notes |
|---|---|---|---|---|---:|---:|---:|---|---|
| Learned baseline | `inverse-kv-local-h128-l3-top1` | `inverse-kv-local-h128-l3-top1.log` | uniform | token-level MoE, router input hidden, top1 | 5000 | 0.2623 | 91.50% | `learned_baseline_uniform_step5000.json` / `moe_variant_selectivity_step5000_include_self.json` | Uniform learned baseline. |
| Learned baseline | `inverse-kv-zipf-baseline` | `inverse-kv-zipf-baseline.train.log` | zipf | token-level MoE, router input hidden, top1 | 5000 | 0.1978 | 94.43% | `moe_variant_selectivity_zipf_step5000_include_self.json` | Zipf learned baseline used in 2x4 variant comparison. |
| Router input variant | `inverse-kv-attn-output-router` | `inverse-kv-attn-output-router.log` / `.nohup.log` | uniform | token-level MoE, router input attention_output | 5000 | 0.2641 | 91.38% | `moe_variant_selectivity_step5000_include_self.json` | Tests whether attention output improves routing selectivity. |
| Router input variant | `inverse-kv-zipf-attn-output-router` | `inverse-kv-zipf-attn-output-router.train.log` | zipf | token-level MoE, router input attention_output | 5000 | 0.1979 | 94.43% | `moe_variant_selectivity_zipf_step5000_include_self.json` | Zipf attention-output router baseline. |
| Head-level MoE variant | `inverse-kv-head-moe-hidden-router` | `inverse-kv-head-moe-hidden-router.nohup.log` | uniform | head-level MoE, router input hidden | 5000 | 0.2633 | 91.41% | `moe_variant_selectivity_step5000_include_self.json` | Per-head expert routing. |
| Head-level MoE variant | `inverse-kv-zipf-head-moe-hidden-router` | `inverse-kv-zipf-head-moe-hidden-router.train.log` | zipf | head-level MoE, router input hidden | 5000 | 0.1979 | 94.44% | `moe_variant_selectivity_zipf_step5000_include_self.json` | Zipf head-level hidden router. |
| Combined MoE variant | `inverse-kv-attn-output-head-moe` | `inverse-kv-attn-output-head-moe.nohup.log` | uniform | head-level MoE + attention_output router | 5000 | 0.2617 | 91.29% | `moe_variant_selectivity_step5000_include_self.json` | Combines attention-output routing and head-level experts. |
| Combined MoE variant | `inverse-kv-zipf-attn-output-head-moe` | `inverse-kv-zipf-attn-output-head-moe.train.log` | zipf | head-level MoE + attention_output router | 5000 | 0.1981 | 94.43% | `moe_variant_selectivity_zipf_step5000_include_self.json` | Zipf combined variant. |
| Ground-truth dispatch | `oracle-uniform-local-hash` | `oracle-uniform-local-hash.log` | uniform | ground-truth local slot -> expert, hash mapping | 5000 | 0.2570 | 91.52% | `ground_truth_selectivity_uniform_step5000.json` | Local slot dispatch is close to baseline. |
| Ground-truth dispatch | `oracle-uniform-local-frequency_balanced` | `oracle-uniform-local-frequency_balanced.log` | uniform | ground-truth local slot -> expert, frequency-balanced mapping | 5000 | 0.2590 | 91.42% | `ground_truth_selectivity_uniform_step5000.json` | Local frequency-balanced dispatch. |
| Ground-truth dispatch | `oracle-uniform-higher-hash` | `oracle-uniform-higher-hash.log` | uniform | ground-truth higher-level slot -> expert, hash mapping | 5000 | 0.2293 | 93.30% | `ground_truth_selectivity_uniform_step5000.json` | Strong uniform result; higher-level bucket is useful. |
| Ground-truth dispatch | `oracle-uniform-higher-frequency_balanced` | `oracle-uniform-higher-frequency_balanced.log` | uniform | ground-truth higher-level slot -> expert, frequency-balanced mapping | 5000 | 0.2335 | 92.98% | `ground_truth_selectivity_uniform_step5000.json` | Higher-level frequency-balanced dispatch. |
| Ground-truth dispatch | `oracle-zipf-local-hash` | `oracle-zipf-local-hash.log` | zipf | ground-truth local slot -> expert, hash mapping | 5000 | 0.2067 | 94.03% | `ground_truth_selectivity_zipf_step5000.json` | Local slot dispatch is close to baseline. |
| Ground-truth dispatch | `oracle-zipf-local-frequency_balanced` | `oracle-zipf-local-frequency_balanced.log` | zipf | ground-truth local slot -> expert, frequency-balanced mapping | 5000 | 0.2067 | 94.04% | `ground_truth_selectivity_zipf_step5000.json` | Local frequency-balanced dispatch. |
| Ground-truth dispatch | `oracle-zipf-higher-hash` | `oracle-zipf-higher-hash.log` | zipf | ground-truth higher-level slot -> expert, hash mapping | 5000 | 0.1922 | 94.73% | `ground_truth_selectivity_zipf_step5000.json` | Best main zipf result. |
| Ground-truth dispatch | `oracle-zipf-higher-frequency_balanced` | `oracle-zipf-higher-frequency_balanced.log` | zipf | ground-truth higher-level slot -> expert, frequency-balanced mapping | 5000 | 0.1922 | 94.63% | `ground_truth_selectivity_zipf_step5000.json` | Best main zipf result, frequency-balanced. |
| Ground-truth rerun | `oracle-rerun-zipf-higher-hash-seed20260519` | `oracle-rerun-zipf-higher-hash-seed20260519.log` | zipf | higher-level ground-truth dispatch, hash, rerun seed 20260519 | 5000 | 0.1922 | 94.73% | `ground_truth_top2_rerun_seed20260519_selectivity.json` | Confirms zipf higher-level hash is stable. |
| Ground-truth rerun | `oracle-rerun-zipf-higher-frequency_balanced-seed20260519` | `oracle-rerun-zipf-higher-frequency_balanced-seed20260519.log` | zipf | higher-level ground-truth dispatch, frequency-balanced, rerun seed 20260519 | 5000 | 0.1922 | 94.64% | `ground_truth_top2_rerun_seed20260519_selectivity.json` | Confirms zipf higher-level frequency-balanced is stable. |
| Supervised gate | `inverse-kv-supervised-gate-zipf-high-hash` | `inverse-kv-supervised-gate-zipf-high-hash.nohup.log` | zipf | linear gate, hidden router input, supervised by high-slot hash label, dispatch by learned gate | 5000 | 0.2090 | 93.95% | `supervised_gate_zipf_high_hash_step5000.json` | Gate supervision does not recover ground-truth dispatch NTP gain. |
| Supervised gate detached | `inverse-kv-supervised-gate-detach-zipf-high-hash` | `inverse-kv-supervised-gate-detach-zipf-high-hash.nohup.log` | zipf | linear gate, hidden router input, router CE detach input | 5000 | - | - | training log only | Final train token_loss about 0.207; no selectivity JSON found. |
| MLP supervised gate | `inverse-kv-mlp-gate-hidden-supervised-zipf-high-hash` | `inverse-kv-mlp-gate-hidden-supervised-zipf-high-hash.log` | zipf | 2-layer MLP gate, hidden router input, supervised high-slot hash label | 5000 | 0.2090 | 93.96% | `mlp_supervised_gate_zipf_high_hash_selectivity_step5000.json` | Nonlinear gate still does not recover ground-truth dispatch NTP gain. |
| MLP supervised gate | `inverse-kv-mlp-gate-attention_output-supervised-zipf-high-hash` | `inverse-kv-mlp-gate-attention_output-supervised-zipf-high-hash.log` | zipf | 2-layer MLP gate, attention_output router input, supervised high-slot hash label | 5000 | 0.2090 | 93.95% | `mlp_supervised_gate_zipf_high_hash_selectivity_step5000.json` | Attention-output supervised MLP gate. |
| Inhibition | `inhibition-uniform-hidden` | `inhibition-uniform-hidden.log` | uniform | learned gate, hidden router input, inhibition weight 0.05, no ground-truth label | 5000 | 0.2617 | 91.33% | `inhibition_selectivity_uniform_step5000.json` | Same-slot rate high, but expert load nearly collapses. |
| Inhibition | `inhibition-uniform-attention_output` | `inhibition-uniform-attention_output.log` | uniform | learned gate, attention_output router input, inhibition weight 0.05, no ground-truth label | 5000 | 0.2617 | 91.28% | `inhibition_selectivity_uniform_step5000.json` | Collapses to single expert in all layers. |
| Inhibition | `inhibition-zipf-hidden` | `inhibition-zipf-hidden.log` | zipf | learned gate, hidden router input, inhibition weight 0.05, no ground-truth label | 5000 | 0.2089 | 93.96% | `inhibition_selectivity_zipf_step5000.json` | Mostly collapses to one expert. |
| Inhibition | `inhibition-zipf-attention_output` | `inhibition-zipf-attention_output.log` | zipf | learned gate, attention_output router input, inhibition weight 0.05, no ground-truth label | 5000 | 0.2089 | 93.96% | `inhibition_selectivity_zipf_step5000.json` | Collapses to single expert in all layers. |
| Load-balance sweep | `inverse-kv-zipf-attn-output-router-lb-0p001` | `inverse-kv-zipf-attn-output-router-lb-0p001.train.log` | zipf | attention_output router, load-balance weight 0.001 | 5000 | 0.1979 | 94.45% | `moe_variant_selectivity_zipf_attn_output_lb_sweep_step5000.json` | LB flattens expert load without clear NTP gain. |
| Load-balance sweep | `inverse-kv-zipf-attn-output-router-lb-0p01` | `inverse-kv-zipf-attn-output-router-lb-0p01.train.log` | zipf | attention_output router, load-balance weight 0.01 | 5000 | 0.1980 | 94.44% | `moe_variant_selectivity_zipf_attn_output_lb_sweep_step5000.json` | LB sweep. |
| Load-balance sweep | `inverse-kv-zipf-attn-output-router-lb-0p1` | `inverse-kv-zipf-attn-output-router-lb-0p1.train.log` | zipf | attention_output router, load-balance weight 0.1 | 5000 | 0.1979 | 94.43% | `moe_variant_selectivity_zipf_attn_output_lb_sweep_step5000.json` | LB sweep. |

## Other Checkpoints / Logs

These are present but should not be treated as primary finished baselines:

| Checkpoint / artifact | Status |
|---|---|
| `oracle-rerun-zipf-higher-hash-seed999999` | Only `1.pth`; smoke / aborted run. |
| `oracle-rerun-zipf-higher-frequency_balanced-seed999999` | Only `1.pth`; smoke / aborted run. |
| `smoke-load-balance` | One-step smoke test. |
| `inverse-kv-local-moe-top1` | Directory exists but no `.pth` / runtime config in the current scan. |
| `experiments/baseline`, `experiments/unet-*` | Earlier sequence-compression / Unet-style experiments, not part of the current inverse-KV MoE selectivity table. |

## Reading Notes

- For MoE selectivity, always check expert load entropy or MI/NMI together with same-slot same-expert rate. Same-slot rate alone is misleading under expert collapse.
- For comparing model quality, use NTP accuracy from the same evaluation JSON when possible. Some older JSON files evaluate the same checkpoint under slightly different scripts, so source files are listed explicitly.
- For new experiments, keep checkpoint directory name, log name, and JSON run name identical when possible; it makes this table much easier to maintain.
