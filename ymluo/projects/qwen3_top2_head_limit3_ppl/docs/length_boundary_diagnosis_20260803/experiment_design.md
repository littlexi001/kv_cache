# 实验设计

## 研究问题

1. 32K--128K 的退化最先来自固定预算、Key 排序、阈值近似还是 Value 尾部？
2. 低维 Key 坐标能否解释 Value sketch 残差？
3. 解析风险值能否在未知主题、长度和模型上识别需要更高 rank 的 head？
4. 通过质量门槛后，融合实现能否提高 attention 与稳态 decode 速度？

## 阶段 A：固定目标因果诊断

- 模型：Llama-3.1-8B-Instruct，FP16。
- 长度：32K、64K、96K、128K。
- 目标：每个主题固定同一 128K 流末尾的 8 个 token。
- 主题：computer、mixed_b、politics、medicine、religion、sports。
- 预算：每个 KV head 固定 1,280 token。
- 对照：Full、exact-QK top-k、proxy full-top-k、sampled threshold、rank-16 Value tail。
- 保持不变：目标 token、seed、模型、原始 K/V、top-k 数量。

判定：若 exact top-k 与 Full 的差明显大于 proxy 与 exact 的差，则主因是固定预算和 tail，而不是 Key 量化。

## 阶段 B：真实 Q/K/V 机制实验

- Llama 32K：sports、medicine，层 0/8/16/24/31。
- Llama 128K：computer，层 0/8/16/24/31。
- Qwen3-4B 32K：至少 8 个 decode step，层 0/8/17/26/35。
- 每个 KV head 和其 GQA query head 独立统计。

比较：

1. selected only。
2. rank-16/32/64/96/128 INT4 Value sketch。
3. 块残差均值。
4. Key 条件残差，`d=8/16/32`。
5. 条件残差加解析风险升级。

指标：局部 attention 输出相对 L2、绝对 L2、cosine、p90/p99/max、selected mass、风险与真实误差的 Pearson/Spearman 相关、实际平均 rank、索引比例。

### 阶段 B2：prefill 到 decode 的时间迁移

- 模型：Qwen3-4B-Instruct。
- 请求：sports、medicine，各 32K 历史。
- 层：0/8/17/26/35；每层记录 prefill 最后 8 个 query 和后续 32 个 decode query。
- 候选集合：exact top-1,280 与请求局部 QK proxy top-1,280。
- Value：普通 rank-16 INT4 PCA 与 W_o-metric rank-16 INT4 PCA。
- 校准：使用最后 1/2/4/8 个 prefill query；比较每层单系数与每 KV head 系数。
- 对照：`alpha=0/0.5/1`，以及只用于诊断的 decode oracle。

实现算法：先固定 QK 和 Value basis；对 prefill query 计算 `s_t`、`d_t`、`o_t`，按设计文档中的闭式最小二乘解出 `alpha`；之后不再更新，在 32 个 decode query 上测经 `W_o` 投影后的相对 L2。

通过条件：两个主题上，prefill 校准的 mean 与 p90 均优于固定 `alpha=1`，且增加 calibration query 后结果不发生大幅反转。失败条件：只在训练用 prefill query 上变好、decode p90 变差，或每 KV head 解明显过拟合。

脚本与产物：

- 捕获：`src/collect_real_qk_trace_20260715.py --prefill_query_tail_tokens 8`
- 分析：`src/analyze_qksieve_prefill_tail_calibration_20260803.py`
- 结果：`results/20260803_prefill_tail_calibration_qwen32k_2topic_v1/`

### 阶段 B3：输出误差感知选择

保持 QK basis、Value rank、INT4、top-k 数量和所有输入完全相同，只替换 top-k 排序键：

1. `proxy_score`：当前方法。
2. `proxy_score + log(rho_fp)`：FP32 风险，判断数学准则本身。
3. `proxy_score + log(rho_int8)`：部署候选。
4. `proxy_score + log(rho_int4)`：更低存储候选。
5. `exact_score + log(rho_fp)`：诊断 QK proxy 对新准则的剩余损失，不是部署方法。

首先复用 Qwen3-4B 的 sports/medicine 32K 两请求，每个请求 32 个 decode query、5 层；随后只在通过时进入 Llama 128K 三个困难主题。主指标是经 `W_o` 后的 mean/p90/max 相对 L2；通过条件是 INT8 在两个主题都稳定优于普通 proxy top-k，且接近 FP32 风险版本。若只在一个主题改善，或 exact-score 风险版本也无改善，则该误差上界虽正确但对实际选择无用，应停止。

脚本与产物：

- `src/analyze_qksieve_prefill_tail_calibration_20260803.py`
- `results/20260803_residual_priority_qwen32k_2topic_v1/`

### 阶段 B4：无长度阈值的风险覆盖预算

先在同一批 Qwen3-4B sports/medicine 32K real-QKV trace 上比较：

1. 固定 `k=1280`、按 proxy QK 排序。
2. 固定 `k=1280`、按 `proxy_score + log(rho_int4)` 排序。
3. 选择覆盖 90%/95%/97.5%/99% proxy attention mass 的最小集合。
4. 选择覆盖 90%/95%/97.5%/99% proxy output-risk mass 的最小集合。

每个条件报告经 `W_o` 后的 mean/p90/p99/max 相对 L2、实际每 head token 数分布、真实 attention mass 和真实 residual-risk mass。主要比较必须在相近平均 token 数下进行，不能仅用更大预算换质量。

通过条件：至少一个风险覆盖工作点在两个主题上都优于相近预算的 attention-mass 规则，且 p90 不反转。失败条件：预算接近 Full、只改善训练主题、均值改善但 p90/max 显著恶化，或风险码量化破坏排序。

通过后再加入块级 QK 误差上界，测量证书 `B_V` 与真实 Value 输出误差的比值；证书不得低于真实误差，且中位松弛倍数应足够小，才能用于在线预算扩展。

脚本与产物：

- `src/analyze_qksieve_output_risk_budget_20260803.py`
- `results/20260803_output_risk_budget_qwen32k_2topic_v1/`

### 阶段 B5：误差分解与全局 head-token 分配

先用固定 `k in {640,1280,2560,4096}` 分解三条路径：

1. `exact_weight_sketch`：所有 token 使用 exact QK 权重，只近似未选 Value，测纯 Value 项。
2. `hybrid_full_value`：所有 token 使用原始 Value，但 selected 用 exact 权重、tail 用 proxy 权重，测混合权重项。
3. `hybrid_sketch`：当前完整近似。
4. `coherent_proxy_full_value`：所有 token 使用 proxy 权重和原始 Value，测一致 proxy 的纯 score 项。
5. `coherent_proxy_sketch`：所有 token 使用 proxy 权重，selected 用原始 Value、tail 用 Value sketch。

若固定 `k` 的三项误差都随 `k` 单调下降，而每 head 独立 coverage 仍失败，则问题归因于跨 head 预算归一化，不归因于增加 token 本身。随后在每层、每 decode step 固定总槽位 `B = num_query_heads * k`，比较：

- 每 head 固定 `k`；
- 全局 top-`B` 的 `ptilde_h,i * rho_group,i`；
- 全局 top-`B` 的 `ptilde_h,i * rho_head,i`；
- 只按全局 proxy attention mass 分配。

通过条件：在相同总 token-head 槽位下，head-specific 全局风险分配在两个主题的 mean 与 p90 都优于固定每 head `k`，且没有 head 因分配为零产生极端误差。若仅靠增加总预算改善，或单个 head 获得几乎全部槽位，则该分配规则失败。

实测结论：该条件只在 teacher-forced 单层输出上通过，整模型闭环失败。
因此无约束全局 top-B 已停止，不进入正式方法。

### 阶段 B6：逐 head 输出误差证书

输入：同一批 4K Llama 全 32 层 trace、Qwen 32K sports/medicine trace，
以及 Llama 128K computer trace。QK 索引、Value rank-16 INT4 sketch、
`W_o` 和输入 query 均保持不变。

算法：按设计文档中的
`c_h,i = ptilde_h,i [rho_h,i + delta_h,i d_h,i]` 排序；每个 head 独立
选择最小前缀，使归一化尾部风险不超过 `tau`。先测试 exact score-error
尺度，再测试 256 个精确探针估计和块级量化上界。exact 版本只判断数学
目标是否可行，不作为部署结果。

扫描：`tau in {0.005, 0.01, 0.02, 0.04}`，并保留固定每 head
`k in {640,1280,2560}` 对照。所有结果报告实际 token 数的
mean/p10/p50/p90/max，不只报告平均值。

通过条件：在 4K、32K、128K 和两个模型上，部署版在不高于固定
`k=1280` 的平均槽位下，层输出相对 L2 的 mean、p99 和 max 均不变差；
接入模型后 hidden drift 在任何层不出现超过前一层 4 倍的突增，PPL
质量保持不低于 99.5%。失败条件：任一 head 被分到少于其证书要求的
预算、均值改善但 p99/max 反转、探针估计系统性低估 exact 风险，或预算
在相邻 query 间大幅跳变。

调试产物：逐层 hidden drift、每 head 实际预算、证书值与真实输出误差、
探针低估率，以及相邻 decode step 的候选 Jaccard 和预算变化。

## 阶段 C：整模型质量

先做 3 个困难主题、每主题固定 8 token 的 128K PPL 对照；通过后才扩展到六主题与未知 seed。Full、固定 rank-16、候选新方法必须共享输入和目标 token。

门槛：

- 三主题几何 PPL 质量保持率不低于 99.5%。
- 每个主题不得低于 99.0%。
- 新方法必须优于固定 rank-16，且风险升级不能使用主题或长度标签。
- 只在上述门槛通过后做 LongBench/RULER 大实验。

## 阶段 D：系统实现与计时

先独立实现和测量，不用延迟相减估算：

1. 低比特 QK 扫描与 top-k。
2. tail partition、平方权重、`d=8` 条件矩的一次融合扫描。
3. top-k exact QK/V。
4. Value tail 合并。
5. 完整 attention 子系统。
6. 64K/128K 稳态 decode，至少 256 个生成 step，报告中位数和 p90。

速度比较使用相同模型、dtype、batch、历史长度、生成步数、CUDA graph 设置和 GPU 常驻 K/V。索引构建单独报告；Agent 多轮复用场景另算 break-even。

## 当前运行与产物

- 固定目标：`results/20260803_qksieve_fixed_target_hard3_alpha05_6gpu/`
- 真实尾部分析：`results/20260803_cvtail_trace128k_v2/`
- 条件残差：`results/20260803_conditional_residual_trace128k_r16_v1/`
- rank 风险扫描：`results/20260803_riskrank{16,32,64,96,128}_trace128k_v5/`
- 128K PPL 因果对照：`results/20260803_qksieve_layer0r128_hard3_128k_6gpu_v2/`

已知限制：当前 128K real-QKV 只有一个主题和一个 decode 位置；在跨主题、跨 step、第二模型验证前，条件残差只能称为候选机制。

## 阶段 E：双证书 rate 与罕见事件压力测试

### 实验对象

- Discovery：Qasper 3K、religion 4K、sports/medicine 32K 与 96K。
- Held-out：重新捕获的 computer 96K，seed `20260843`，Qwen3-4B，层 `0/7/14/21/28/35`。
- 合成反例：均匀分散误差、近边界平台、块相关误差、单个隐藏 needle、query 方向漂移。

### 固定比较

1. fixed rate-15；
2. fixed rate-23；
3. 256 探针、同样本拟合与评估的 KL；
4. 256 探针 cross-fit KL；
5. cross-fit KL 加逐 token 尾部质量上界。

所有条件使用同一 QK-balanced basis、同一 Value sketch、同一 RSS token 规则和相同输入。不得用主题或长度选择 rate。

### 指标和判定

- rate 决策：平均 rate、各 rate 占比、误升级率、漏升级率。
- 质量：局部输出相对 L2 的 mean/p90/p99/max，闭环 PPL 质量保持率。
- 预算：Key 索引比例、Value 辅助索引比例、实际精确 K/V token 比例和三者总流量代理。
- 证书：`D_r` 与真实 full-history KL 的相关性；`mass_tail_upper` 覆盖真实遗漏质量的比例及松弛倍数。

通过条件：held-out 中不存在“规则选择低 rate、但局部输出误差超过 fixed rate-23 两倍”的层；隐藏 needle 的漏检率为零；正常分布下额外 token 不超过 fixed rate-23 的 10%。若上界虽然覆盖但中位松弛倍数超过 20，判为不能部署，只保留理论诊断。
# 概率越界救援压力测试

## 变量与固定项

- 固定模型、trace、query、原始 K/V、QK-balanced basis、rate-15 编码和最终 top-k 数量。
- 只改变越界概率估计：无救援、高斯 RMS、经验 add-one survival、rate-23 对照。
- 长度：4K、32K、96K；主题至少包含 religion、medicine、sports、computer。
- 真实 trace 指标：真实 top-k 召回、attention mass、attention 输出相对 L2、额外 exact-K 数、预测越界数与真实漏选数。
- 分布诊断：标准化误差的方差、偏度、峰度、p95/p99，以及不同层和 head 的越界事件相关性。

## 反例

1. 分散高斯误差：高斯方法应校准良好。
2. Student-t 重尾误差：经验方法应比高斯方法少漏检。
3. 256-token 块相关误差：检查独立性假设是否导致 Bernstein 救援数偏小。
4. 高残差隐藏 needle：残差范数应使其进入救援集合。
5. 普通残差范数但定向对齐的隐藏 needle：两种方法均可能失败，用于标明不可辨识边界。
6. 请求外 basis 漂移：应由请求局部 basis 更新解决，而不是靠扩大救援集合掩盖。

## 判定

- 通过：所有真实 trace 上，救援后的局部输出误差不差于 rate-23，平均总读取字节更低；合成高残差 needle 零漏检；重尾/块相关反例的漏检率有明确上界或触发更保守的经验估计。
- 证据不足：只有单一主题或单一层通过，或者 PPL 样本少于 8 token。
- 失败：任何真实条件的最坏输出误差超过 rate-23 两倍；普通条件需要接近 Full 的救援数；或真实 decode 的新增阶段时间大于节省的 attention 时间。

## 候选保留最小实验

### 固定项

- rate-15 请求局部 QK-balanced 索引、256 个 exact-QK 探针、经验 add-one 越界概率和 `delta=0.01` 保持不变。
- 同一 trace、同一 query、同一原始 K/V 和同一基础 `k` 下成对比较。
- 先只做 real-QKV 小实验：Llama religion 4K、Qwen medicine/sports 32K、Qwen medicine/sports 96K；每个条件覆盖 5 层和全部 KV/GQA head。

### 对照

1. `base-k`：rate-15 代理直接 top-k。
2. `rerank-k`：风险候选与 base 合并，exact 重排后压回 k。
3. `keep-union`：保留去重后的全部 base 与风险候选。
4. `rate-23-k`：高码率索引的固定 k 对照。
5. `exact-k`：只用于诊断固定预算上限，不作为可部署方法。

### 指标与决策

- 质量：真实 top-k 召回、真实 attention mass、selected-only 输出相对 L2；随后加入 rank-16 INT4 Value tail 后报告局部输出相对 L2。
- 流量：基础 k、救援候选数、去重后实际 token 数、exact K 读取数和 Value 读取数。
- 鲁棒性：报告每层/head/step 的 mean、p90、p99 和 maximum，不能只报宏平均。
- 通过：`keep-union` 在五个条件均不差于 `rerank-k`，至少四个条件优于 rate-23-k 的平均输出误差，且实际 token 数明显低于各条件 Full。
- 失败：任何主题的 p99 反转超过 20%，或者 96K 并集平均超过历史的 5%。失败时不进入闭环 PPL，只记录“越界候选适合排序修复，不适合预算扩展”。
