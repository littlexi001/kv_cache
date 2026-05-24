# Round5 MoE Structure Analysis

数据设定：`b4-u512-vocab512-zipf1.0`；模型设定：`2-layer h64`；checkpoint step: `2000`。

## Summary Table

| Run | Router | Train loss | Eval loss | NTP acc | Inside-local | Local-boundary-not-high | High-boundary | Local same-expert | High same-expert | Eff. experts | Attn-local mass | Attn-high mass | Attn-expert mass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `round5-full-attn-output-exp-resid` | `attention_output/full` | 0.4250 | 0.4139 | 91.80% | 98.27% | 91.35% | 12.05% | 34.39% | 28.08% | 3.9462 | 50.73% | 76.34% | 51.33% |
| `round5-full-q-exp-resid` | `q/full` | 0.3680 | 0.3637 | 92.72% | 98.89% | 93.68% | 12.33% | 38.54% | 30.98% | 3.9298 | 50.08% | 78.16% | 51.61% |
| `round5-full-k-exp-resid` | `k/full` | 0.3620 | 0.3648 | 92.72% | 98.85% | 93.83% | 12.43% | 43.50% | 35.42% | 3.6992 | 48.48% | 77.34% | 52.87% |
| `round5-full-v-exp-resid` | `v/full` | 0.3690 | 0.3675 | 92.63% | 98.78% | 93.60% | 12.48% | 33.74% | 25.94% | 3.9678 | 46.23% | 75.47% | 44.12% |
| `round5-full-layer-input-exp-resid` | `layer_input/full` | 0.3530 | 0.3565 | 92.77% | 98.88% | 94.01% | 12.40% | 34.86% | 28.35% | 3.9706 | 47.39% | 75.18% | 45.76% |
| `round5-full-hidden-exp-resid` | `hidden/full` | 0.3730 | 0.3765 | 92.43% | 98.65% | 93.12% | 12.38% | 36.87% | 28.40% | 3.9806 | 48.02% | 74.70% | 48.00% |
| `round5-head-attn-output-exp-resid` | `attention_output/head` | 0.4200 | 0.4099 | 91.79% | 98.21% | 91.56% | 12.10% | 37.03% | 30.73% | 3.9581 | 53.37% | 78.01% | 55.21% |
| `round5-head-q-exp-resid` | `q/head` | 0.3740 | 0.3747 | 92.54% | 98.79% | 93.21% | 12.13% | 41.27% | 33.19% | 3.9840 | 49.58% | 77.97% | 56.82% |
| `round5-head-k-exp-resid` | `k/head` | 0.3550 | 0.3639 | 92.75% | 98.93% | 93.72% | 12.30% | 50.34% | 43.16% | 3.5054 | 53.34% | 83.52% | 63.21% |
| `round5-head-v-exp-resid` | `v/head` | 0.3450 | 0.3528 | 92.95% | 99.01% | 94.50% | 12.18% | 37.28% | 29.27% | 3.9612 | 52.32% | 83.41% | 47.14% |
| `round5-head-layer-input-exp-resid` | `layer_input/head` | 0.3370 | 0.3462 | 92.97% | 98.98% | 94.63% | 12.53% | 36.04% | 29.38% | 3.9822 | 48.31% | 80.64% | 46.75% |
| `round5-head-hidden-exp-resid` | `hidden/head` | 0.3570 | 0.3599 | 92.81% | 98.90% | 94.11% | 12.43% | 34.38% | 27.99% | 3.9939 | 50.53% | 81.15% | 47.11% |
| `round5-spectral-attn-output-exp-resid` | `attention_output/spectral` | 0.4820 | 0.4874 | 90.55% | 97.72% | 87.08% | 11.62% | 35.19% | 29.35% | 13.7743 | 55.08% | 79.59% | 41.97% |
| `round5-spectral-q-exp-resid` | `q/spectral` | 0.5050 | 0.5113 | 90.20% | 97.46% | 86.24% | 11.72% | 41.68% | 31.96% | 12.7323 | 53.34% | 79.78% | 41.51% |
| `round5-spectral-k-exp-resid` | `k/spectral` | 0.4690 | 0.4743 | 90.93% | 98.05% | 87.83% | 11.64% | 38.51% | 30.15% | 12.1473 | 56.91% | 85.26% | 42.13% |
| `round5-spectral-v-exp-resid` | `v/spectral` | 0.4990 | 0.5023 | 90.24% | 97.47% | 86.51% | 11.49% | 36.52% | 27.47% | 12.8272 | 54.78% | 82.98% | 37.32% |
| `round5-spectral-layer-input-exp-resid` | `layer_input/spectral` | 0.5250 | 0.5221 | 90.07% | 97.48% | 85.60% | 11.31% | 35.96% | 27.56% | 15.0174 | 55.98% | 84.19% | 36.92% |
| `round5-spectral-hidden-exp-resid` | `hidden/spectral` | 0.5030 | 0.5192 | 90.07% | 97.44% | 85.85% | 11.09% | 37.82% | 28.09% | 14.9591 | 52.48% | 77.50% | 37.46% |

## Notes

- `Local same-expert` / `High same-expert` 是同一 feature 内 token 被分到同一 expert bucket 的平均比例；head-router 按 head 平均，spectral 按 routed band 平均。
- `Attn-local/high mass` 是 attention score 落在同 local/high slot 的比例，包含 self attention。
- `Attn-expert mass` 是 attention score 落在同 expert bucket token 上的比例，表示 routing bucket 与 attention retrieval bucket 的重合程度。
