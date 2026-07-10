# 正式投稿前实验补齐清单

## 必补主表

1. 多模型主结果表
   - 模型：Qwen3-8B、Llama-3.1-8B、Mistral-7B，条件允许再加 Qwen2.5-14B 或 Llama-3.1-70B。
   - 数据：LongBench、RULER full、RULER 8k/16k/32k。
   - 指标：score、active token/KV ratio、online speed、E2E speed、peak memory。

2. Baseline 对比表
   - Full KV/raw。
   - Sliding window 或 StreamingLLM。
   - H2O。
   - SnapKV。
   - PyramidKV 或 DynamicKV。
   - Fixed block top-k。
   - RiskKV-Block。

3. Router v2 表
   - 输入特征：当前 router features、retriever gap、top-k stability、task family、block size candidate。
   - 标签：oracle 或 worst-case label，判断当前动作是否危险，以及最小安全动作。
   - 报告：router label accuracy、danger recall、score、token ratio、E2E speed。

## 必补图

1. Speed vs context length
   - X 轴：4k、8k、16k、32k，条件允许加 64k。
   - Y 轴：online speed 和 E2E speed。
   - 曲线：Full、fixed top-k、block-size router、cache-native RiskKV。

2. Accuracy-memory Pareto
   - X 轴：active token/KV ratio。
   - Y 轴：task score。
   - 点：b32、b64、b128、b256、b512、fixed top-k、router selection。

## 必补消融

1. 去掉 block-size routing，只保留固定 block。
2. 去掉 fallback。
3. 去掉 identifier overlap。
4. 去掉 retriever gap/top-k stability features。
5. prompt-level selected spans vs cache-native KV repack。
6. RoPE-aware repack vs naive gather。
7. fallback safety floor：k1/k2/k3、summary fallback、full fallback。

## 建议优先级

第一优先级是把当前 96-example mixed suite 扩到正式规模，并训练 router v2。第二优先级是补齐 LongBench/RULER 上和已有 KV cache baseline 的横向对比。第三优先级是多模型复现实验。只要主表、多模型、baseline 和两张核心图补齐，这篇稿子的完整度就接近正式投稿形态。
