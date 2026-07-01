# Section 37: K-SVD Sample Stability

Date: 2026-06-30

## 0. Goal

This experiment tests whether the low-rank K-cache right-singular subspace is stable when estimated from different numbers of samples:

```text
V_r estimated from first x K tokens  ≈  V_r estimated from first y K tokens
```

This matters because the previous low-rank classifier/probe uses an SVD basis. If the basis changes drastically with sample count, then the low-rank explanation would be fragile.

New files:

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/src/analyze_k_svd_sample_stability.py
ymluo/projects/qwen3_top2_head_limit3_ppl/scripts/run_k_svd_sample_stability_server.sh
```

## 1. Important Detail: KV Heads

Qwen3-0.6B config:

```text
hidden_size = 1024
num_attention_heads = 16
num_key_value_heads = 8
head_dim = 128
```

This experiment analyzes the stored K cache, so it uses KV heads, not attention heads.

Selected KV heads:

```text
0,2,4,6
```

## 2. Server Run

Server output:

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/k_svd_sample_stability_0630_v1
```

Local copy:

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/k_svd_sample_stability_0630_v1
```

Run command:

```bash
OUT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/k_svd_sample_stability_0630_v1 \
bash scripts/run_k_svd_sample_stability_server.sh
```

Scale:

```text
variants = compact_kv,json_kv,needle_sentence,topic_table
tasks_per_variant = 4
tasks used = 16
layers = 0,4,8,13,20,27
kv_heads = 0,2,4,6
sample sizes = 64,128,256,512,768
ranks = 4,8,16,32,64,128
pair rows = 12456
runtime = 25.7s
```

Note:

```text
Not every task has 768 context tokens.
So pairs involving 768 have fewer cases and mainly come from longer variants such as needle_sentence.
```

## 3. Metrics

For each layer/KV-head/task:

1. Take first `x` K tokens and estimate `V_r(x)`.
2. Take first `y` K tokens and estimate `V_r(y)`.
3. Compare them with:

```text
diag_abs_cos_mean:
  Mean abs cosine of same-index singular vectors.
  This is sign-invariant but rotation-sensitive.

principal_cos_mean:
  Mean principal-angle cosine between the two r-dimensional subspaces.

subspace_overlap:
  mean squared principal cosine = trace(Px Py) / r.
  1 means identical subspaces.

energy_capture_ratio:
  Energy of first-y K captured by V_r(x),
  divided by energy captured by V_r(y).
```

The most reliable metrics are:

```text
subspace_overlap
energy_capture_ratio
```

because individual singular vectors can rotate inside a near-degenerate subspace.

## 4. Main Results

### Adjacent sample-size pairs

| x -> y | Rank | Subspace overlap | Principal cos mean | Diag abs cos mean | Energy capture ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| 64 -> 128 | 4 | 0.728 | 0.816 | 0.616 | 0.876 |
| 64 -> 128 | 8 | 0.737 | 0.818 | 0.469 | 0.870 |
| 64 -> 128 | 16 | 0.743 | 0.818 | 0.336 | 0.871 |
| 64 -> 128 | 32 | 0.761 | 0.832 | 0.237 | 0.887 |
| 64 -> 128 | 64 | 0.791 | 0.852 | 0.165 | 0.924 |
| 128 -> 256 | 4 | 0.746 | 0.821 | 0.637 | 0.884 |
| 128 -> 256 | 8 | 0.769 | 0.841 | 0.491 | 0.879 |
| 128 -> 256 | 16 | 0.779 | 0.847 | 0.374 | 0.887 |
| 128 -> 256 | 32 | 0.797 | 0.859 | 0.268 | 0.903 |
| 128 -> 256 | 64 | 0.839 | 0.890 | 0.193 | 0.940 |
| 256 -> 512 | 4 | 0.764 | 0.838 | 0.673 | 0.901 |
| 256 -> 512 | 8 | 0.787 | 0.852 | 0.532 | 0.898 |
| 256 -> 512 | 16 | 0.813 | 0.870 | 0.401 | 0.902 |
| 256 -> 512 | 32 | 0.825 | 0.878 | 0.297 | 0.914 |
| 256 -> 512 | 64 | 0.866 | 0.908 | 0.215 | 0.948 |
| 512 -> 768 | 4 | 0.831 | 0.880 | 0.755 | 0.945 |
| 512 -> 768 | 8 | 0.866 | 0.910 | 0.632 | 0.944 |
| 512 -> 768 | 16 | 0.888 | 0.921 | 0.522 | 0.947 |
| 512 -> 768 | 32 | 0.899 | 0.931 | 0.409 | 0.954 |
| 512 -> 768 | 64 | 0.921 | 0.947 | 0.302 | 0.973 |

Interpretation:

```text
The subspace is already moderately stable at 64-128 samples.
It becomes clearly stable by 256-512 samples.
At 512-768 samples, rank32/rank64 are very stable.
```

## 5. Non-Adjacent Pairs

| x -> y | Rank | Subspace overlap | Principal cos mean | Energy capture ratio |
| --- | ---: | ---: | ---: | ---: |
| 128 -> 512 | 16 | 0.674 | 0.768 | 0.808 |
| 128 -> 512 | 32 | 0.699 | 0.786 | 0.834 |
| 128 -> 512 | 64 | 0.771 | 0.839 | 0.892 |
| 256 -> 768 | 16 | 0.742 | 0.813 | 0.853 |
| 256 -> 768 | 32 | 0.773 | 0.841 | 0.872 |
| 256 -> 768 | 64 | 0.823 | 0.877 | 0.918 |

Interpretation:

```text
When y is much larger than x, stability drops, especially for 128 -> 512.
But even then, rank32 captures about 83% of the rank32 energy that V_r(y) captures,
and rank64 captures about 89%.
```

## 6. Layer-Level Stability

For the important `256 -> 512` pair:

| Layer | Rank | Subspace overlap | Principal cos mean | Energy capture ratio |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 32 | 0.841 | 0.892 | 0.995 |
| 4 | 32 | 0.838 | 0.889 | 0.920 |
| 8 | 32 | 0.833 | 0.883 | 0.901 |
| 13 | 32 | 0.801 | 0.859 | 0.869 |
| 20 | 32 | 0.826 | 0.880 | 0.918 |
| 27 | 32 | 0.810 | 0.866 | 0.883 |
| 0 | 64 | 0.864 | 0.908 | 0.996 |
| 4 | 64 | 0.887 | 0.923 | 0.955 |
| 8 | 64 | 0.873 | 0.914 | 0.939 |
| 13 | 64 | 0.848 | 0.894 | 0.917 |
| 20 | 64 | 0.859 | 0.903 | 0.946 |
| 27 | 64 | 0.866 | 0.909 | 0.934 |

Layer 13 is the least stable among the selected layers, but it is still usable at rank32/rank64.

For `512 -> 768`, all selected layers become more stable:

```text
rank32 overlap: about 0.878-0.920
rank64 overlap: about 0.909-0.938
```

## 7. Key Observation

Single-vector alignment is not a reliable stability signal:

```text
diag_abs_cos_mean often decreases as rank grows.
```

For example, in `256 -> 512`:

```text
rank4  diag_abs_cos_mean = 0.673
rank32 diag_abs_cos_mean = 0.297
rank64 diag_abs_cos_mean = 0.215
```

But the subspace is stable:

```text
rank32 subspace_overlap = 0.825
rank64 subspace_overlap = 0.866
```

This means the leading singular vectors can rotate within the same low-rank subspace. Therefore the correct object to compare is the span of `V_r`, not individual `V_i` vectors.

## 8. Conclusion

The sampling stability hypothesis is mostly supported:

```text
V_r estimated from the first x samples is close to V_r estimated from the first y samples
when x is at least about 256 and y is not dramatically larger.
```

Practical recommendation:

```text
Use at least 256 K tokens to estimate V_r for quick probes.
Use 512+ tokens when rank32/rank64 stability matters.
Compare subspaces with principal angles or energy capture, not same-index vector cosine.
```

For the previous low-rank classifier direction:

```text
rank32/rank64 bases are stable enough for offline analysis and likely stable enough
for calibration-based candidate selection, as long as the calibration window is not too tiny.
```
