# QKSieve 公共稀疏检索基线协议

## 目的

公共基线必须拆成两类，不能把不同问题的数字混在一张“公平速度表”中：

1. **selector quality control**：相同样本、相同 active-token schedule、相同
   原始 FP16 K/V、相同 exact sparse-attention consumer，只改变候选选择器；
2. **official system comparison**：使用作者官方实现，允许其使用不同 KV
   布局、CPU offload、ANN index 或 Value 修正，但必须完整报告硬件、显存、
   索引、质量和端到端延迟。

当前代码完成第一类中的 Quest selector、SparQ selector-only control，以及
一个公式完整的 SparQ matched-budget reference。它们都尚未构成官方优化系统
的速度复现。

## Quest P16 selector

方法名：

```text
quest_p16_fullprompt_matchedbudget
```

score mode：

```text
quest_p16_fulltopk
```

实现与 Quest 算法一致的 page criticality：

1. 每个 KV head 按 16 token 分页；
2. 每页、每个 Key 维度维护最小值和最大值；
3. 给定 Query 后计算

   $$
   u_p(q)=\sum_j\max\{q_jk^{\min}_{p,j},q_jk^{\max}_{p,j}\};
   $$

4. 选择 page score 最大的页面；
5. 页面内 token 全部交给和 QKSieve 相同的 exact-KV attention kernel。

page metadata 增量更新，不会在每个 decode step 重扫完整 Key。page size 为
16 时，FP16 min/max metadata 是 32 B/token/KV-head，即 256 bit。目标
token 数使用 QKSieve 的冻结长度预算，但 page 粒度会使实际加载量最多上浮
15 个 token，结果中单独报告该上浮。

## SparQ R32 selector control

方法名：

```text
sparq_r32_selector_fullprompt_matchedbudget
```

score mode：

```text
sparq_r32_selector_fulltopk
```

实现步骤：

1. 每个 Query head 选择绝对值最大的 32 个 Query 维度；
2. 只在这些维度上计算近似 QK score；
3. 直接选取冻结 active-token budget 对应的 top-k；
4. 使用与 QKSieve 相同的原始 FP16 K/V 和 exact sparse-attention kernel。

该路径有意不使用 SparQ 的 selected-mass 估计和 mean-Value correction，因此
论文中必须称为 **SparQ selector control**，不能称为完整官方 SparQ。
它隔离的问题是“大绝对值 Query 维度能否比 QKSieve mixed-bit index 更好地
找到相同数量的 token”。

SparQ selector 没有额外低比特索引，但高效实现需要 dimension-addressable
Key layout。当前 HuggingFace reference path 不能用于速度结论。

## SparQ R32 formula reference

方法名：

```text
sparq_r32_formula_fullprompt_matchedbudget
```

score mode：

```text
sparq_r32_meanvalue_fulltopk
```

该路径在 selector control 之外补齐 SparQ 论文中的四个关键步骤：

1. 每个 Query head 仍取绝对值最大的 \(r=32\) 个维度 \(I\)；
2. 使用论文温度

   $$
   \tau=\sqrt{d\,\frac{\lVert q_I\rVert_1}{\lVert q\rVert_1}}
   $$

   对全历史近似 score 做 softmax；
3. 在 top-\(k\) 选择时强制包含最近
   \(\lfloor k/4\rfloor\) 个位置，并以近似概率和
   \(\alpha=\sum_{i\in S}\widetilde p_i\) 估计 selected mass；
4. 在原始 FP16 K/V 上计算精确稀疏输出 \(y_S\)，再使用增量维护的
   Value 均值做

   $$
   y=\alpha y_S+(1-\alpha)\overline v.
   $$

为保持 active-token 公平性，实现令 \(k=B(N)+1\)，其中额外一个位置是当前
token；传给公共 exact-KV consumer 的历史候选仍严格为 \(B(N)\) 个。因此，
该路径是“公式完整、同预算”的质量参考，但仍不是官方系统复现：它使用
QKSieve 的动态预算而非 SparQ 论文固定的 \(k=128\)，也没有官方优化
dimension-addressable layout。它的 PyTorch latency 不可用于速度结论。

## LongBench 配对协议

恢复 GPU 后运行：

```bash
bash scripts/launch_qksieve_public_selectors_longbench_5gpu_20260728.sh
```

正式配置包含 16 个英文 LongBench 任务、3,750 个严格五方法配对：

- Full KV；
- 冻结 QKSieve；
- Quest P16 selector；
- SparQ R32 selector control；
- SparQ R32 formula reference。

五个方法使用相同 prompt、stop policy 和生成长度。四个稀疏方法使用相同

$$
B(N)=\min\{N,1280,\max(256,\lceil0.06N\rceil)\}
$$

目标预算；Quest 单独报告 page rounding 后的 loaded-token ratio。

输出：

```text
results/20260728_qksieve_public_selectors_longbench_official_middle_5gpu/
  shard0..4/sample_results.csv
  public_selector_summary.json
  logs/
  ALL_COMPLETE
```

`public_selector_summary.json` 会拒绝样本不配对、预算不一致、score mode
不一致或索引 rate 不一致的结果。报告中的 latency 明确标为无效，不能填写
论文系统速度表。

## RetrievalAttention 与 RetroInfer

RetrievalAttention 把 KV cache 视为向量数据库，并依赖近似近邻检索与
CPU-GPU 协同。它与 QKSieve 的 GPU-resident full-KV 设定不同，不能伪装成
“同一个 PyTorch selector”放进上述五方法 harness。

进一步审计发现，当前 `microsoft/RetrievalAttention` 仓库的可运行代码是
后续方法 RetroInfer，而不是原始 RetrievalAttention 实现。因此正确做法是：

1. RetrievalAttention 只保留 paper-reported 行，不写成已复现；
2. 固定当前官方仓库提交，单独复现 RetroInfer；
3. 锁定与 QKSieve 相同的 LongBench/RULER 样本、prompt 和生成设置；
4. 分别报告 GPU/CPU KV placement、ANN/wave index memory、构建时间、TPOT、
   prefill、传输和峰值显存；
5. 质量按严格相同样本配对；
6. 系统速度表按完整系统报告，不宣称 index-byte matched。

完整身份审计和复现边界见
`docs/20260728_qksieve_retroinfer_official_protocol_zh.md`。

## 尚未完成

- Quest page-score 与 gather 的优化 CUDA kernel；
- SparQ 官方优化 Key layout、kernel 和原生系统复现；
- RetroInfer 官方仓库复现；
- 三者在相同 GPU 上的正式端到端速度。

因此当前代码只能关闭“公共 selector 质量比较缺失”这一部分，不能关闭
“公共系统速度基线缺失”。
