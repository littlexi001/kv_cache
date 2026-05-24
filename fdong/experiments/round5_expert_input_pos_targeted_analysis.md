# Round5 MoE Structure Analysis

数据设定：`b4-u512-vocab512-zipf1.0`；模型设定：`2-layer h64`；checkpoint step: `2000`。

## Summary Table

| Run | Router | Train loss | Eval loss | NTP acc | Inside-local | Local-boundary-not-high | High-boundary | Local same-expert | High same-expert | Eff. experts | Attn-local mass | Attn-high mass | Attn-expert mass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `round5-expertpos-targeted-rhead-k-eresid` | `k/head` | 0.3500 | 0.3630 | 92.80% | 98.97% | 93.77% | 12.46% | 51.98% | 44.26% | 3.5483 | 52.89% | 83.16% | 63.53% |
| `round5-expertpos-targeted-rhead-k-eattn` | `k/head` | 0.3620 | 0.3678 | 92.81% | 98.98% | 93.87% | 12.18% | 48.37% | 39.68% | 3.6857 | 63.40% | 87.21% | 68.19% |
| `round5-expertpos-targeted-rhead-k-elayerin` | `k/head` | 0.3940 | 0.3957 | 92.26% | 98.64% | 92.63% | 11.26% | 42.04% | 35.26% | 3.9258 | 47.63% | 75.56% | 55.55% |
| `round5-expertpos-targeted-rhead-k-eq` | `k/head` | 0.4390 | 0.4424 | 91.37% | 98.15% | 89.96% | 10.83% | 41.86% | 34.39% | 3.8701 | 36.53% | 64.39% | 48.12% |
| `round5-expertpos-targeted-rhead-k-ek` | `k/head` | 0.4570 | 0.4594 | 90.89% | 97.72% | 89.26% | 10.58% | 37.92% | 30.67% | 3.9346 | 40.23% | 67.37% | 47.47% |
| `round5-expertpos-targeted-rhead-k-ev` | `k/head` | 0.4470 | 0.4418 | 91.57% | 98.34% | 90.34% | 10.76% | 46.43% | 39.12% | 3.8655 | 57.22% | 84.56% | 62.75% |
| `round5-expertpos-targeted-rhead-k-ehidden` | `k/head` | 0.3600 | 0.3633 | 92.81% | 98.91% | 94.03% | 12.46% | 52.88% | 46.03% | 3.4853 | 52.96% | 83.92% | 64.02% |
| `round5-expertpos-targeted-rhead-layerin-eresid` | `layer_input/head` | 0.3390 | 0.3459 | 93.01% | 99.01% | 94.66% | 12.58% | 35.97% | 29.25% | 3.9744 | 48.10% | 80.53% | 46.22% |
| `round5-expertpos-targeted-rhead-layerin-eattn` | `layer_input/head` | 0.3550 | 0.3556 | 92.99% | 99.07% | 94.43% | 12.23% | 39.74% | 31.67% | 3.9551 | 62.22% | 86.78% | 59.93% |
| `round5-expertpos-targeted-rhead-layerin-elayerin` | `layer_input/head` | 0.3870 | 0.3938 | 92.19% | 98.56% | 92.54% | 11.21% | 38.27% | 29.27% | 3.9863 | 46.67% | 74.35% | 46.81% |
| `round5-expertpos-targeted-rhead-layerin-eq` | `layer_input/head` | 0.4080 | 0.4106 | 92.02% | 98.54% | 91.71% | 11.24% | 37.11% | 29.76% | 3.9389 | 40.15% | 71.75% | 41.96% |
| `round5-expertpos-targeted-rhead-layerin-ek` | `layer_input/head` | 0.4290 | 0.4418 | 91.36% | 98.12% | 90.11% | 10.73% | 36.22% | 29.46% | 3.9461 | 39.37% | 69.91% | 42.08% |
| `round5-expertpos-targeted-rhead-layerin-ev` | `layer_input/head` | 0.4190 | 0.4202 | 91.95% | 98.57% | 91.36% | 10.88% | 35.84% | 28.76% | 3.9276 | 56.70% | 84.92% | 51.51% |
| `round5-expertpos-targeted-rhead-layerin-ehidden` | `layer_input/head` | 0.3400 | 0.3458 | 93.00% | 99.05% | 94.50% | 12.46% | 36.05% | 30.06% | 3.9740 | 48.15% | 80.49% | 46.49% |
| `round5-expertpos-targeted-rhead-q-eattn` | `q/head` | 0.3660 | 0.3724 | 92.82% | 99.05% | 93.74% | 11.90% | 50.62% | 42.28% | 3.7744 | 65.82% | 88.50% | 72.58% |
| `round5-expertpos-targeted-rhead-q-ek` | `q/head` | 0.4640 | 0.4784 | 90.57% | 97.54% | 88.21% | 10.65% | 40.02% | 29.94% | 3.9608 | 37.58% | 62.74% | 48.59% |
| `round5-expertpos-targeted-rfull-k-eattn` | `k/full` | 0.3740 | 0.3749 | 92.70% | 98.94% | 93.42% | 12.15% | 50.08% | 44.32% | 2.9656 | 65.55% | 88.55% | 70.99% |
| `round5-expertpos-targeted-rfull-layerin-eattn` | `layer_input/full` | 0.3520 | 0.3603 | 92.83% | 98.97% | 94.09% | 12.05% | 37.70% | 30.26% | 3.8144 | 62.31% | 86.65% | 58.02% |

## Notes

- `Local same-expert` / `High same-expert` 是同一 feature 内 token 被分到同一 expert bucket 的平均比例；head-router 按 head 平均，spectral 按 routed band 平均。
- `Attn-local/high mass` 是 attention score 落在同 local/high slot 的比例，包含 self attention。
- `Attn-expert mass` 是 attention score 落在同 expert bucket token 上的比例，表示 routing bucket 与 attention retrieval bucket 的重合程度。
