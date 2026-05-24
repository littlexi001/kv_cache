# Round5 MoE Structure Analysis

数据设定：`b4-u512-vocab512-zipf1.0`；模型设定：`2-layer h64`；checkpoint step: `1000`。

## Summary Table

| Run | Router | Train loss | Eval loss | NTP acc | Inside-local | Local-boundary-not-high | High-boundary | Local same-expert | High same-expert | Eff. experts | Attn-local mass | Attn-high mass | Attn-expert mass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `round5-full-attn-output-exp-resid` | `attention_output/full` | 0.7370 | 0.6893 | 87.25% | 95.19% | 80.15% | 10.35% | 36.67% | 28.56% | 3.9336 | 41.47% | 66.36% | 46.93% |
| `round5-full-q-exp-resid` | `q/full` | 0.6030 | 0.5940 | 89.02% | 96.53% | 83.96% | 10.91% | 38.71% | 29.95% | 3.9442 | 41.45% | 68.26% | 48.15% |
| `round5-full-k-exp-resid` | `k/full` | 0.5990 | 0.5933 | 89.09% | 96.48% | 84.49% | 11.06% | 43.03% | 33.35% | 3.7355 | 41.40% | 68.23% | 49.48% |
| `round5-full-v-exp-resid` | `v/full` | 0.6070 | 0.5819 | 89.27% | 96.72% | 84.66% | 10.65% | 36.41% | 26.81% | 3.9566 | 40.58% | 68.00% | 41.73% |
| `round5-full-layer-input-exp-resid` | `layer_input/full` | 0.5660 | 0.5539 | 89.82% | 97.11% | 85.87% | 10.93% | 36.29% | 27.84% | 3.9764 | 40.07% | 66.52% | 42.26% |
| `round5-full-hidden-exp-resid` | `hidden/full` | 0.6400 | 0.6058 | 88.63% | 96.16% | 83.61% | 10.27% | 36.52% | 28.79% | 3.9809 | 39.55% | 65.08% | 44.00% |
| `round5-head-attn-output-exp-resid` | `attention_output/head` | 0.7340 | 0.7026 | 86.77% | 94.58% | 80.05% | 10.17% | 35.63% | 29.70% | 3.9751 | 43.00% | 66.35% | 48.84% |
| `round5-head-q-exp-resid` | `q/head` | 0.6200 | 0.6066 | 88.68% | 96.43% | 82.80% | 10.35% | 39.32% | 31.53% | 3.9611 | 41.43% | 68.33% | 51.42% |
| `round5-head-k-exp-resid` | `k/head` | 0.6160 | 0.6063 | 88.90% | 96.62% | 82.99% | 10.96% | 46.23% | 37.01% | 3.8145 | 43.39% | 71.19% | 54.24% |
| `round5-head-v-exp-resid` | `v/head` | 0.5950 | 0.5638 | 89.80% | 97.18% | 85.56% | 10.81% | 36.55% | 29.53% | 3.9498 | 43.55% | 73.94% | 44.70% |
| `round5-head-layer-input-exp-resid` | `layer_input/head` | 0.5410 | 0.5300 | 90.28% | 97.53% | 86.54% | 11.39% | 36.98% | 29.35% | 3.9639 | 42.08% | 72.81% | 44.45% |
| `round5-head-hidden-exp-resid` | `hidden/head` | 0.5940 | 0.5751 | 89.25% | 96.72% | 84.60% | 10.48% | 35.10% | 27.93% | 3.9947 | 41.68% | 69.90% | 43.66% |
| `round5-spectral-attn-output-exp-resid` | `attention_output/spectral` | 0.8770 | 0.8634 | 84.45% | 92.85% | 74.82% | 9.61% | 35.20% | 29.01% | 13.5828 | 44.14% | 67.57% | 34.78% |
| `round5-spectral-q-exp-resid` | `q/spectral` | 0.9220 | 0.8860 | 83.95% | 92.43% | 73.83% | 9.74% | 42.58% | 32.33% | 12.5392 | 42.85% | 68.06% | 36.66% |
| `round5-spectral-k-exp-resid` | `k/spectral` | 0.8950 | 0.8783 | 84.15% | 92.47% | 74.71% | 9.72% | 42.60% | 32.94% | 11.7595 | 42.01% | 67.33% | 34.79% |
| `round5-spectral-v-exp-resid` | `v/spectral` | 0.9590 | 0.9352 | 82.65% | 91.08% | 72.55% | 9.01% | 33.69% | 27.44% | 13.0376 | 42.73% | 67.40% | 31.89% |
| `round5-spectral-layer-input-exp-resid` | `layer_input/spectral` | 1.0030 | 0.9482 | 82.51% | 90.94% | 72.39% | 8.90% | 37.47% | 29.96% | 14.7634 | 40.37% | 65.76% | 29.34% |
| `round5-spectral-hidden-exp-resid` | `hidden/spectral` | 0.9170 | 0.9066 | 83.15% | 91.72% | 72.75% | 8.80% | 36.68% | 28.57% | 15.0147 | 41.51% | 65.46% | 30.62% |

## Notes

- `Local same-expert` / `High same-expert` 是同一 feature 内 token 被分到同一 expert bucket 的平均比例；head-router 按 head 平均，spectral 按 routed band 平均。
- `Attn-local/high mass` 是 attention score 落在同 local/high slot 的比例，包含 self attention。
- `Attn-expert mass` 是 attention score 落在同 expert bucket token 上的比例，表示 routing bucket 与 attention retrieval bucket 的重合程度。
