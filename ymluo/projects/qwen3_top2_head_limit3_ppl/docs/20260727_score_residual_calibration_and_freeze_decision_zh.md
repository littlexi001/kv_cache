# 分数残差校正实验与方法冻结决策

## 1. 研究问题

当前主方法用 QK-balanced 坐标、分层可变比特量化和 QK-metric scale
估计每个 query-head 对历史 token 的分数，再直接消费稀疏候选。代理分数仍有一个
稳定误差：

```text
exact_i(q) = q^T k_i
proxy_i(q) = q8^T khat_i
residual_i(q) = exact_i(q) - proxy_i(q)
```

本轮验证的问题是：能否只用 prefill query 的数值统计，为每个历史 token 保存极少量
校正元数据，进一步提高检索质量，并保持真实速度。

## 2. 数值方法

### 2.1 均值残差校正

对 token `i`，用 prefill query 校准残差均值：

```text
b_i = 0.5 * mean_q residual_i(q)
corrected_i(q) = proxy_i(q) + b_i
```

`0.5` 对应闭式 ridge 收缩，不训练网络、不使用任务标签，也没有 router、fallback
或精确重排。

若使用真实校准均值而不收缩，校准集上有：

```text
mean_q[exact_i(q) - corrected_i(q)] = 0
MSE_corrected = Var_q[residual_i(q)] <= E_q[residual_i(q)^2]
```

因此它能严格消除校准残差的均值，但该结论只保证分数 MSE，不保证语言模型 NLL。

### 2.2 仿射校正代理

同时还测试了每 token 的闭式 ridge 仿射校正：

```text
corrected_i(q) = a_i * proxy_i(q) + b_i
```

`a_i,b_i` 均由 prefill query 的最小二乘充分统计量得到。INT8 元数据几乎不损失
FP16 仿射校正的代理收益，但本轮没有把它加入冻结方法，原因见第 5 节。

## 3. 代理检索实验

协议：

- 3 个模型：Qwen3-4B、Qwen2.5-7B、Llama-3.1-8B。
- 体育、医学两类文本。
- 32K 和 96K 长度。
- 8 条 trace、59,200 个严格配对 head-query 样本。
- 校准 query 与评测 query 分离。
- 评测 top-1% token 检索。

| 方法 | top-1% recall | 选中 attention mass | attention mass recall | Pearson | RMSE |
|---|---:|---:|---:|---:|---:|
| QK-metric scale | 72.7991% | 76.8849% | 96.9044% | 0.970079 | 0.756410 |
| + bias FP16 | 73.0258% | 76.9160% | 96.9631% | 0.970336 | 0.754066 |
| + bias INT4 | 73.0213% | 76.9156% | 96.9624% | 0.970327 | 0.754157 |
| + affine INT8 | 73.2282% | 76.9441% | 97.0146% | 0.970547 | 0.752199 |
| + EB bias INT4 | 72.8924% | 76.9011% | 96.9407% | 0.970259 | 0.754845 |

主要观察：

- 固定 bias FP16 的 recall 提升 `+0.2267` 个百分点。
- bias INT4 保留了约 `98%` 的 recall 增益，说明偏置本身可低比特存储。
- affine INT8 提升最大，为 `+0.4291` 个百分点。
- 固定收缩优于经验贝叶斯收缩；后者不能作为替代。
- 固定 bias 在 1,920 个 layer-step block 中有 `82.9%` 的 block 改善 recall；
  affine INT8 为 `85.2%`。

这些结果证明校正现象跨模型、跨主题存在，但仍只是检索代理证据。

## 4. 真实 CUDA 实现

已把 bias FP16 接入 packed CUDA 路径：

1. prefill 时计算 exact query mean 和 INT8 proxy query mean。
2. 编码每个 K token 时，在同一 CUDA kernel 内计算 residual mean。
3. 每个 token/head 额外存一个 FP16 bias。
4. sampled-quantile 扫描时多一次 FP16 读取和加法。
5. 最终仍对候选执行原始 FP16 K/V 稀疏 attention。

正确性检查：

- 开启校正前后 packed codes 完全相同。
- key scales 最大误差为 `0`。
- GPU 分数增量与存储 bias 的最大误差小于 `1e-6`。
- 18 个数值单元测试全部通过。

索引开销由约 `5.86%` Full FP16 KV 增至约 `6.25%`，增加量就是
`16 / 4096 = 0.390625%` Full KV。

## 5. 真实 PPL 与速度结论

### 5.1 128K 独立四窗口

协议与旧 qscale 主结果完全对齐：4 个独立窗口、每窗口 256 个目标 token，
共 1,024 个严格配对 token。

| 方法 | PPL | 相对 Full 质量 | 95% CI | 稳态整模型速度 | 256 token 含建索引 |
|---|---:|---:|---:|---:|---:|
| Full KV | 12.88480 | 100% | - | 1.000x | 1.000x |
| 旧 qscale 主方法 | 12.89227 | 99.9421% | [98.8572%, 101.0359%] | 4.3129x | 3.6017x |
| + mean bias FP16 | 12.89868 | 99.8924% | [98.8230%, 100.9605%] | 4.3123x | 3.5605x |

mean bias 相对旧 qscale：

- 质量保持率为 `99.9503%`，即自身约退化 `0.0497%`。
- bootstrap 95% 区间为 `[99.8873%, 100.0097%]`。
- 只有 `5.4%` 的 bootstrap 样本认为 mean bias 更好。
- 稳态速度无可测差异；固定建索引时间从约 `3.48s` 增到 `3.72s`。

### 5.2 决策

**不把 mean bias 或 affine calibration 加入冻结主方法。**

原因不是速度，而是：

1. top-k recall 改善没有转化成 NLL 改善。
2. 真实端点略差于更简单的旧 qscale。
3. 为挽救它继续调 shrinkage 会把工作变成验证窗口调参。
4. affine 虽然代理收益更高，但同属同一代理目标；在没有下游证据前，不值得增加
   CUDA 元数据和论文复杂度。

本轮最重要的科研结论是：

> 分数 MSE、top-k recall 和 attention mass recall 都不是最终 PPL 的充分替代指标。
> 后续新数值模块必须直接面向 attention output 或 token NLL 的稳定性，而不能只追求
> 更像 Full Attention 的候选集合。

## 6. 当前冻结候选

截至本轮，仍应保留：

`pca_hierarchical_autoqmsetotal15z_qkmetric_qscale_packed_direct`

它包含：

- QK-balanced 双正交坐标；
- 每 head 的物理分层比特分配；
- `{0,1,2,4,8}` bit 量化；
- QK-metric optimal scale；
- INT8 query；
- sampled-quantile 候选定位；
- 长度自适应候选预算，上限 1,280 token/head；
- 无 router、无 task label、无 fallback、无精确重排；
- 直接稀疏 K/V attention。

128K 已确认结果：

- 相对 Full PPL 质量 `99.9421%`；
- 每 head 实际 attention token 约 `1.05%`；
- 索引约 `5.86%` Full FP16 KV；
- 整模型稳态速度 `4.3129x`；
- 256-token 生成含建索引 `3.6017x`；
- break-even 约 `15.2` 个生成 token。

## 7. 下一步

高优先级：

1. 完成官方 RaBitQCache 同环境对照，而不是继续使用公式模拟。
2. 在冻结主方法上补 Llama-3.1-8B、LongBench、RULER 和真实生成速度。
3. 设计直接约束 attention output 误差的数值目标，例如 value-sensitive covariance
   或 softmax Jacobian 二次型，并先用独立 trace 验证它与 NLL 的相关性。

低优先级：

- bias INT4 CUDA 打包；
- 仿射元数据 CUDA；
- 调整 bias shrinkage；
- 继续追求极小的 top-k recall 增益。

这些低优先级工作在当前证据下不会提高投稿质量。
