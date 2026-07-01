# Section 40: calibrated V_r and candidate recall

## 1. 实验目的

这组实验验证两个问题：

1. `calibrated V_r` 是否可用：能不能只用前面一小段历史 token 估计每层每 head 的低秩右奇异子空间 `V_r`，后续 query 固定使用这个 `V_r` 做投影，而不是每个 query 重新 SVD。
2. `candidate recall` 是否足够高：用低秩投影分数先选一个候选集合，例如 5% 或 8% 历史 token，能否覆盖 full-QK attention 真正的 top 2% token，特别是能否覆盖主要 attention mass。

如果成立，下一步可以做两阶段选择：

1. calibrated `V_r` 做低成本 candidate proposal；
2. 只在 candidate 内做 full-QK rerank 或 attention，从而逼近原始 top2 行为。

## 2. 实验设置

代码：

- `ymluo/projects/qwen3_top2_head_limit3_ppl/src/analyze_calibrated_lrsvd_candidate_recall.py`
- `ymluo/projects/qwen3_top2_head_limit3_ppl/scripts/run_calibrated_lrsvd_candidate_recall_server.sh`

服务器输出：

- selected-head 版本：`/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/calibrated_lrsvd_candidate_recall_0630_v1`
- all-head 版本：`/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/calibrated_lrsvd_candidate_recall_allheads_0630_v1`

本地同步结果：

- `ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/calibrated_lrsvd_candidate_recall_allheads_0630_v1/candidate_recall_summary.csv`
- `ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/calibrated_lrsvd_candidate_recall_allheads_0630_v1/summary.json`
- `ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/calibrated_lrsvd_candidate_recall_allheads_0630_v1/tasks.csv`

模型和采样：

- Model: Qwen3-0.6B
- `head_dim=128`, `num_attention_heads=16`, `num_key_value_heads=8`
- layers: `0, 4, 8, 13, 20, 27`
- all-head run: heads `0..15`
- variants: `compact_kv`, `json_kv`, `needle_sentence`, `topic_table`
- tasks: 4 per variant, 16 tasks total
- observed layer-head-query rows: 3072
- ranks: `16, 32, 64`
- calibration samples: `128, 256, 512`
- candidate fractions: `2%, 5%, 8%`
- target top fraction: full-QK top `2%`

注意：`calib=512` 时 `compact_kv` 的上下文太短，因此 overall 的 rows 是 2304，而不是 3072。

## 3. 方法

对每个 task、layer、head：

1. 取历史 K 的前 `calib_samples` 个 token，计算 SVD，得到固定 `V_r`。
2. 对后续 sampled query row：
   - full-QK 分数定义真 top2 token；
   - lowrank score 使用投影后的 q/k dot：

```text
score_lr(q, k) = (q V_r) dot (k V_r)
```

3. 按 lowrank score 选 top candidate fraction。
4. 统计 candidate 是否召回 full-QK top2：
   - `top2_recall`: token-level recall；
   - `top2_attention_mass_recall`: 真 top2 attention mass 中被 candidate 覆盖的比例；
   - `candidate_attention_mass_mean`: candidate 集合本身覆盖的总 attention mass。

这里的关键指标优先看 `top2_attention_mass_recall`。因为 token recall 低不一定代表行为差，如果漏掉的是 attention mass 很小的 top2 边缘 token，对 PPL 影响可能有限。

## 4. All-head overall 结果

| calib | rank | candidate | top2 token recall | top2 mass recall | candidate mass |
|---:|---:|---:|---:|---:|---:|
| 128 | 16 | 2% | 39.01% | 77.34% | 63.40% |
| 128 | 16 | 5% | 58.93% | 86.56% | 73.13% |
| 128 | 16 | 8% | 68.80% | 90.79% | 78.27% |
| 128 | 32 | 2% | 49.18% | 86.36% | 70.62% |
| 128 | 32 | 5% | 69.17% | 93.26% | 78.91% |
| 128 | 32 | 8% | 77.42% | 95.81% | 82.88% |
| 128 | 64 | 2% | 62.87% | 94.10% | 76.60% |
| 128 | 64 | 5% | 81.75% | 97.99% | 83.08% |
| 128 | 64 | 8% | 88.29% | 98.93% | 85.98% |
| 256 | 16 | 2% | 40.03% | 76.56% | 62.85% |
| 256 | 16 | 5% | 60.63% | 87.50% | 74.13% |
| 256 | 16 | 8% | 70.89% | 92.15% | 79.67% |
| 256 | 32 | 2% | 51.47% | 87.53% | 71.53% |
| 256 | 32 | 5% | 71.57% | 94.73% | 80.36% |
| 256 | 32 | 8% | 80.01% | 97.05% | 84.29% |
| 256 | 64 | 2% | 65.27% | 95.01% | 77.24% |
| 256 | 64 | 5% | 83.84% | 98.53% | 83.73% |
| 256 | 64 | 8% | 89.81% | 99.28% | 86.60% |
| 512 | 16 | 2% | 39.02% | 74.15% | 61.35% |
| 512 | 16 | 5% | 59.77% | 85.26% | 72.60% |
| 512 | 16 | 8% | 69.81% | 89.87% | 78.08% |
| 512 | 32 | 2% | 51.29% | 87.30% | 71.97% |
| 512 | 32 | 5% | 71.81% | 94.53% | 80.61% |
| 512 | 32 | 8% | 80.55% | 96.89% | 84.45% |
| 512 | 64 | 2% | 66.80% | 95.41% | 78.23% |
| 512 | 64 | 5% | 85.83% | 98.67% | 84.36% |
| 512 | 64 | 8% | 91.50% | 99.36% | 87.07% |

## 5. 主要结论

### 5.1 calibrated V_r 是可用的

固定用前 `128/256/512` 个历史 token 估计 `V_r`，后续仍然能高 recall 地覆盖 full-QK top2 的主要 attention mass。尤其是：

- `calib=128, rank=64, candidate=5%`: top2 mass recall `97.99%`
- `calib=256, rank=64, candidate=5%`: top2 mass recall `98.53%`
- `calib=512, rank=64, candidate=5%`: top2 mass recall `98.67%`

这说明 `V_r` 不需要每个 query 在线重新估计。用 prefix calibration 得到的低秩子空间已经足够接近后续 query 需要的可分性方向。

### 5.2 rank64 是最稳的配置，rank32 是可压缩配置

`rank64` 的表现非常稳定：

- `candidate=2%` 时已经有 `95%` 左右的 mass recall；
- `candidate=5%` 时达到 `98%+`；
- `candidate=8%` 时接近 `99%+`。

`rank32` 也有实用价值：

- `calib=256, rank=32, candidate=5%`: mass recall `94.73%`
- `calib=256, rank=32, candidate=8%`: mass recall `97.05%`

如果下一步目标是尽量接近 full top2，优先测 `rank64 + candidate5%/8%`。如果目标是更激进降成本，可以测 `rank32 + candidate8%`。

### 5.3 candidate token recall 不等于 mass recall

例如：

- `calib=512, rank=64, candidate=5%`: token recall `85.83%`，但 mass recall `98.67%`
- `calib=512, rank=64, candidate=2%`: token recall `66.80%`，但 mass recall `95.41%`

这说明低秩 candidate 主要抓住了 attention mass 最大的 top2 token；被漏掉的 token 很多可能是 top2 里的边缘 token。这对 PPL 可能比 token recall 更重要。

## 6. 分主题结果

`rank=64, calib=512`：

| variant | candidate | top2 token recall | top2 mass recall |
|---|---:|---:|---:|
| json_kv | 2% | 71.48% | 95.97% |
| json_kv | 5% | 90.30% | 99.07% |
| json_kv | 8% | 94.88% | 99.65% |
| needle_sentence | 2% | 61.28% | 93.73% |
| needle_sentence | 5% | 80.49% | 97.77% |
| needle_sentence | 8% | 87.80% | 98.83% |
| topic_table | 2% | 69.15% | 96.45% |
| topic_table | 5% | 88.15% | 99.13% |
| topic_table | 8% | 92.84% | 99.58% |

`needle_sentence` 相对更难，token recall 最低，但 5% candidate 仍然能覆盖 `97.77%` 的 top2 mass。

## 7. 分层结果

`rank=64, calib=512, candidate=5%`：

| layer | top2 token recall | top2 mass recall |
|---:|---:|---:|
| 0 | 87.22% | 95.98% |
| 4 | 78.73% | 97.82% |
| 8 | 93.71% | 99.44% |
| 13 | 95.67% | 98.91% |
| 20 | 95.47% | 99.75% |
| 27 | 64.17% | 99.44% |

第 27 层的 token recall 明显低，但 mass recall 仍然高。这说明末层 top2 token 集合可能更分散，但主要质量仍集中在低秩候选能抓到的 token 上。

## 8. 推荐下一步实验

建议直接进入 PPL/下游验证，不再只停留在 recall：

1. `calib256-rank64-candidate5% + candidate 内 full-QK rerank/top2`
2. `calib256-rank64-candidate8% + candidate 内 full-QK rerank/top2`
3. 对照压缩配置：`calib256-rank32-candidate8% + rerank/top2`

优先使用 `calib=256`，因为：

- 和 `calib=512` 的效果非常接近；
- 对较短上下文更友好；
- calibration 成本更低。

实现上应该缓存 `K V_r`，否则如果每次重新投影历史 K，速度实验会低估方法潜力。

