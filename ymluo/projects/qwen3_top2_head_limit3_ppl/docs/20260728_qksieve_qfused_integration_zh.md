# QKSieve Query 融合算子：接入状态与验收协议

## 当前结论

已把“Query 128x128 投影 + 8 个 16D band 的 INT8 量化”接入为独立实验路径：

```text
qksieve_fullprompt_auto_plain_qfused_fulltopk
```

对应 score mode：

```text
pca_hierarchical_autoqmsetotal15z_qkmetric_qfused_packed_fulltopk
```

当前冻结主方法仍是：

```text
qksieve_fullprompt_auto_plain_fulltopk
```

融合路径尚未执行 GPU correctness 和 latency smoke，因此不能用于替换现有质量或速度结果。

## Coalesced GQA-reuse v2

静态审计发现 v1 存在两项系统问题：

1. 一个 warp 为单个输出维做 reduction，basis 读取沿输入维跨行，不能形成连续读取；
2. 同一 KV head 对应的多个 GQA Query group 分别启动 block，重复读取相同的
   128x128 basis。

v2 改为：

- 每个 batch/KV-head 只启动一个 128-thread block；
- thread 对应连续 output dimension；
- 对固定 input dimension，warp 连续读取 basis 最后一维；
- 一个 block 同时累计 1–16 个 GQA Query group，共享每次 basis 读取；
- 投影后仍逐 Query、逐 16D band 计算 INT8 scale/code。
- FP32 累加结果先舍入为模型的 FP16/BF16，再执行 INT8 量化，匹配冻结
  `einsum -> model dtype tensor -> INT8` 路径的数值边界。

这只改变算子布局，不改变数学运算、packed index 或候选预算。
CUDA extension 名称升级为 `qksieve_query_project_ext_v2`，避免复用旧二进制缓存。

## 执行流程

1. Prompt 尾部 Query 仍走原始浮点投影，用于构造 QK-balanced 坐标并完成 per-layer/per-KV-head mixed-bit allocation。
2. allocation 冻结前禁止启用融合 kernel。
3. allocation 冻结后，融合 kernel 一次完成当前 Query 的 128x128 投影和 bandwise INT8 量化。
4. 后续 packed index scan、完整 proxy top-k、原始 FP16 K/V 上的 exact sparse attention 与冻结路径相同。
5. 运行时 diagnostics 记录实际使用的是 `fused_projection_int8`、`unfused_calibration` 还是 `unfused_projection_then_int8`。

这样可以保证融合只优化 decode 热路径，不改变 Query 校准样本、QK-balanced 变换、bit allocation、packed Key index、active-token budget 或最终 exact-KV attention。

## GPU 验收

恢复 GPU 后运行：

```bash
QKSIEVE_GPU=0 \
bash scripts/run_qksieve_qfused_correctness_20260728.sh
```

3090 默认编译 `sm_86`。换到 A100/H100 时必须显式设置：

```bash
# A100
TORCH_CUDA_ARCH_LIST=8.0 QKSIEVE_GPU=0 \
bash scripts/run_qksieve_qfused_correctness_20260728.sh

# H100
TORCH_CUDA_ARCH_LIST=9.0 QKSIEVE_GPU=0 \
bash scripts/run_qksieve_qfused_correctness_20260728.sh
```

输出：

```text
results/20260728_qksieve_qfused_correctness/
  validation_matrix.json
  float16_g4/correctness_and_latency.json
  float16_g8/correctness_and_latency.json
  bfloat16_g4/correctness_and_latency.json
  bfloat16_g8/correctness_and_latency.json
  */run.log
```

脚本在 4K 和 32K、FP16/BF16、4/8 个 GQA group 上逐 trial 比较。
correctness 指标使用所有 Query head 中的最坏值，不能用全局平均掩盖单个
head 的错误：

| 指标 | 默认门槛 |
|---|---:|
| INT8 code exact match | >= 90% |
| Query scale relative p99 | <= 1% |
| proxy score normalized RMSE | <= 1% |
| per-head top-k recall | >= 99.5% |
| exact sparse-attention output cosine | >= 0.999 |
| exact sparse-attention output RMSE | <= 0.01 |
| Query prepare speedup | >= 1.05x |
| 完整 selection speedup | >= 1.00x |

同时报告：

- unfused/fused Query prepare latency；
- Query prepare speedup；
- unfused/fused 完整 selection latency；
- 包含 packed scan 和 top-k 后的 selection speedup。

四种 dtype/GQA 配置的 correctness 和速度门槛都进入
`validation_matrix.json::all_passed`。只看 Query prepare 微内核更快不构成
升级依据；完整 selection latency 必须有稳定收益，否则融合对整模型没有意义。

## 理论验收条件

设两条路径解量化后的 Query 分别为 $\widehat z$ 和 $\widetilde z$，固定代理 Key 为 $\widehat k_i$，则

$$
|\widetilde s_i-\widehat s_i|
\le
\|\widetilde z-\widehat z\|_2\|\widehat k_i\|_2.
$$

令右侧对所有 token 的最大值为 $\epsilon_{\mathrm{fuse}}$。如果参考 top-k 边界满足

$$
\widehat s_{(B)}-\widehat s_{(B+1)}
>
2\epsilon_{\mathrm{fuse}},
$$

则两条路径选择相同的 top-$B$ 集合。相同候选集合和原始 FP16 K/V 在实数运算下给出相同的 exact sparse-attention 输出；GPU reduction 的有限精度差异由输出 cosine/RMSE 继续约束。

## 升级规则

只有以下条件同时满足，才允许把融合路径升级为冻结主方法：

1. correctness smoke 所有长度和 trial 全部通过；
2. 完整 selection latency 稳定优于 unfused 路径，而非仅微内核更快；
3. 小规模真实模型 paired quality smoke 无不可解释退化；
4. diagnostics 确认所有 decode 层实际执行 fused path；
5. 没有 Full fallback、exact rerank、recent/sink 或任务规则。

升级后必须重新运行：

- 完整 LongBench；
- 13-task RULER；
- 三模型统一超参数实验；
- packed FIER 公平对比；
- 16K-128K same-path 速度、显存和 breakdown；
- 最终冻结证据校验器。

旧 unfused 质量与新 fused 速度不能拼接成同一论文结果。

## 真实模型执行证据

微基准通过后，还不能说明 HuggingFace LongBench 路径确实调用了融合
kernel。当前实现为每个稀疏 attention 记录三个数值标志：

- `packed_qmse_fused_query_prepare_requested`：score mode 是否请求融合；
- `packed_qmse_fused_query_prepare_executed`：当前层是否实际执行融合；
- `packed_qmse_allocation_frozen_before_query`：调用前 mixed-bit allocation
  是否已经冻结。

这些标志按层、按生成步采集，再写入每个样本的 CSV。恢复 GPU 后按顺序执行：

```bash
QKSIEVE_GPU=0 \
bash scripts/run_qksieve_qfused_correctness_20260728.sh

QKSIEVE_GPU=0 \
bash scripts/run_qksieve_qfused_longbench_smoke_20260728.sh
```

第二个脚本首先要求
`validation_matrix.json::all_passed=true`，然后在完全相同的样本和预算上配对
运行 Full、冻结 unfused QKSieve 和实验 qfused QKSieve。验收器要求：

1. 每个样本形成严格的三方法配对；
2. 两条 QKSieve 路径除 score mode 外的预算、投影维度和索引 bit 数一致；
3. unfused 的 requested/executed 均为 0；
4. qfused 的 requested/executed 和 allocation-frozen 均为 1；
5. qfused 与 frozen 的 prediction exact-match rate 不低于 87.5%，平均
   LongBench score 绝对差不超过 0.01。

smoke 开启了逐层 diagnostics，因此其中的计时只能用于排查执行路径，不能作为
论文速度结果。若 smoke 通过，仍需关闭 diagnostics 后在冻结 same-path
benchmark 中测量端到端速度。
