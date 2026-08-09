# 4K 谱实验设置

## 设置

- 模型：Llama-3.1-8B-Instruct，FP16。
- 文本：sports 主题自然文本，seed 20260803。
- 历史长度：4,096 token。
- 层与 head：全部 32 层，每层 8 个 KV head，共 256 个矩阵。
- Key 二阶矩：每 32 个历史 token 采一个 Key，共约 128 个样本/head。
- Query 二阶矩：8 个连续 query step，每个 KV head 对应 4 个 GQA query head，共 32 个样本/head。
- 对照：`lambda=0` 与当前实际使用的 `lambda=0.75`。

## 指标

- `top-r energy = sum(i<=r) sigma_i^2 / sum_i sigma_i^2`。
- `rank95/rank99`：累计平方奇异值能量首次达到 95%/99% 的最小秩。
- `stable rank = sum_i sigma_i^2 / sigma_1^2`。
- `entropy effective rank = exp(-sum_i p_i log p_i)`，其中 `p_i` 是归一化平方奇异值。
- `sigma_r/sigma_1`：第 r 个奇异值相对最大奇异值的大小。

## 实现与产物

- trace：服务器 `results/20260803_qk_product_spectrum_llama31_8b_4k/traces/llama31_8b_sports4k.pt`。
- 分析代码：`src/analyze_qk_product_spectrum_20260803.py`。
- 启动脚本：`scripts/run_qk_product_spectrum_4k_20260803.sh`。
- 结果：`results/20260803_qk_product_spectrum_llama31_8b_4k/analysis/`。

## 通过、失败与证据不足

- 通过“谱集中”：当前 `lambda=0.75` 的中位 rank95 明显小于 128，且 top48 能量接近 99%。
- 失败“统一低秩”：任一批 layer-head 的 rank95 明显高于统一候选秩。
- 证据不足“全局通用”：一个 4K sports 请求不能代表其他主题、长度和模型。
