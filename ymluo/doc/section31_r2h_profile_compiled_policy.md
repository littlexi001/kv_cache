# Section 31: R2H-KV Profile-Compiled Head Policy

Date: 2026-06-29

## 0. Goal

This section records the first implementation of:

```text
profile_by_layer_head.csv -> recommended_hybrid_policy.json
```

The goal is to move from manually selected layer lists to a measured profile and a compiler that recommends GPU-friendly head-group templates.

## 1. New Script

Added:

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/src/compile_r2h_hybrid_policy.py
```

Inputs:

```text
remote_mass.json
qabs_overlap/overlap_by_layer_head.csv
adjacent/top2_token_adjacent_by_layer_head.csv
```

Outputs:

```text
profile_by_layer_head.csv
recommended_hybrid_policy.json
recommended_layerbudget_map.json
summary.json
```

Local copies:

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/r2h_profile_codex_0629_local/profile_by_layer_head.csv
ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/r2h_profile_codex_0629_local/recommended_hybrid_policy.json
```

Server output:

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/r2h_profile_codex_0629/compiled_policy_conservative
```

## 2. Profile Generation

Server:

```text
host: df
project: /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
model: /home/fdong/hrj/prove/Qwen3-0.6B
text: /home/fdong/ymluo/projects/influence_bounded_synthetic_kv/data/hard_topic_eval_v2.txt
```

Common profile config:

```text
prefill_tokens = 2048
eval_tokens = 64
recent_tokens = 512
top_fraction = 0.03
qabs mode = qabs8cand3rerank
dtype = bfloat16
attention = eager
```

Profile commands executed:

```text
src/analyze_full_head_remote_mass.py
src/analyze_qabs_top2_overlap.py
src/analyze_adjacent_stability.py
src/compile_r2h_hybrid_policy.py
```

## 3. Head-Level Features

Each `(layer, head)` row records:

```text
remote_mass_mean
recent_mass_mean
qabs_candidate_energy
qabs_selected_energy
qabs_true_top_overlap
reuse_previous_energy
reuse_intersection_energy
reuse_jaccard
recommended_template
recommendation_reason
confidence
```

The compiler currently uses simple rules:

```text
remote_mass low:
  LOCAL_R512

qabs energy high and reuse stable:
  QSKETCH_PAGE_B2

qabs energy ok:
  QSKETCH_PAGE_B4

remote mass high and qabs energy low:
  FULL

otherwise:
  HYBRID_SAFE_B4
```

This is intentionally GPU-friendly:

```text
head-level decisions are compiled into head groups,
but runtime templates are a small finite set.
```

## 4. Conservative Compile Result

The first less conservative compile was too aggressive when folded back into the current layer-level runtime.

The conservative compile used:

```text
local_remote_mass = 0.03
qabs_energy_good = 0.97
qabs_energy_ok = 0.93
qabs_overlap_good = 0.35
reuse_energy_good = 0.85
reuse_jaccard_good = 0.35
fragile_remote_mass = 0.12
fragile_qabs_energy = 0.88
```

Template counts over all 448 heads:

| Template | Count |
| --- | ---: |
| FULL | 314 |
| HYBRID_SAFE_B4 | 72 |
| LOCAL_R512 | 8 |
| QSKETCH_PAGE_B4 | 50 |
| QSKETCH_PAGE_B2 | 4 |

Interpretation:

```text
This profile is conservative.
Only 54/448 heads are recommended for QSKETCH_PAGE.
Most heads remain FULL until better backend evidence or risk calibration is available.
```

This is acceptable for the first compiler because the hard-topic prompt is risk-sensitive and qabs8cand3 has modest overall coverage:

```text
overall qabs8cand3rerank candidate_attention_mass_mean ≈ 0.705
overall true_top_overlap ≈ 0.343
```

## 5. GPU-Friendly Policy Format

`recommended_hybrid_policy.json` contains:

```json
{
  "method": "R2H-KV",
  "gpu_friendly": {
    "head_group_size": 4,
    "templates": {
      "FULL": {"backend": "full"},
      "LOCAL_R512": {"backend": "recent", "recent": 512},
      "QSKETCH_PAGE_B2": {
        "backend": "qsketch_page",
        "runtime_fallback": "qabs8cand3reuse",
        "selected_pages": 2,
        "page_size": 64
      },
      "QSKETCH_PAGE_B4": {
        "backend": "qsketch_page",
        "runtime_fallback": "qabs8cand3reuse",
        "selected_pages": 4,
        "page_size": 64
      },
      "HYBRID_SAFE_B4": {
        "backend": "hybrid_safe",
        "runtime_fallback": "headmix_qabs_reuse",
        "selected_pages": 4,
        "page_size": 64
      }
    }
  },
  "layers": {
    "...": {
      "head_groups": [
        {"heads": [0, 1, 2, 3], "template": "..."}
      ]
    }
  }
}
```

Important:

```text
This is the intended fine-grained head-group policy.
Current layerbudgetattn cannot execute this exact head-group policy yet.
```

So the compiler also emits:

```text
recommended_layerbudget_map.json
```

as a coarse fallback for current runtime.

## 6. Fallback Runtime Evaluation

The conservative `recommended_layerbudget_map.json` was evaluated with current `layerbudgetattn`.

Server output:

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/r2h_profile_policy_conservative_eval_codex_0629
```

Result:

| Mode | Loss | PPL | PPL Ratio | Seconds | Time Ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 1.65158 | 5.21522 | 1.0000 | 2.1226 | 1.000 |
| profile-compiled fallback | 1.66098 | 5.26449 | 1.0094 | 3.3502 | 1.578 |

Interpretation:

```text
The profile-compiled fallback is quality-safe under a 1% PPL-ratio target,
but it is slower in the current eager/dynamic backend.
```

This is expected:

```text
recommended_hybrid_policy.json is head-group/page-template oriented,
but current runtime can only approximate it with coarse layer-level budgets.
```

## 7. Comparison to Manual Refined Hybrid

Previous manual smoke:

```text
r2h_refined_hybrid:
  layers 0,4,5,7-14
  qabs + headmix
  PPL ratio = 0.9991
```

Profile-compiled conservative fallback:

```text
PPL ratio = 1.0094
```

The manual policy is better, but the automatic compiler is doing a harder task:

```text
it derives decisions from profile metrics,
and emits a future head-group policy plus a current-runtime fallback.
```

The gap suggests the next compiler should include counterfactual delta-loss calibration, not only proxy metrics like qabs energy and remote mass.

## 8. Current Takeaways

1. The profile pipeline works end-to-end.
2. `profile_by_layer_head.csv` is generated for all 448 heads.
3. `recommended_hybrid_policy.json` is head-group and GPU-template oriented.
4. Current runtime fallback reaches PPL ratio `1.0094`, which is usable as a conservative first policy.
5. The fallback is slower because current backends are not the final GPU-friendly q-sketch/page kernels.
6. To beat the manual refined policy, the compiler needs direct counterfactual risk features.

## 9. Next Step

Add a fourth profile input:

```text
counterfactual_delta_loss_by_layer_head.csv
```

or at least:

```text
counterfactual_delta_loss_by_layer.csv
```

Then update the compiler:

```text
if proxy says compressible but delta_loss says risky:
  FULL or HYBRID_SAFE

if proxy is moderate but delta_loss is safe:
  allow QSKETCH/HYBRID
```

This should close the gap between:

```text
manual refined hybrid: PPL ratio 0.9991
profile-compiled fallback: PPL ratio 1.0094
```

