# Value-mediated RoPE suppression probe

## 1. 研究问题

已有实验已经否定了“只要提高远程证据的 QK、recall 或 attention mass，答案就会改善”这一充分性假设：block transport 和 Native Phase Envelope 都能提高部分中间指标，却使 Gold PPL 变差。

本实验检验更精确的机制：

> RoPE 抑制是否只有在被抑制 token 的 Value 写入方向能够提高正确答案 margin 时，才会成为真正的输出损失？

它是一个 **oracle mechanism audit**，不是可部署检索器。正确答案 `9` 和冲突答案 digit 用于定义反向传播的审计 margin；默认 singleton 排名也使用由该 margin 得到的 baseline gradient。因此答案标签虽然不出现在 prompt 中，但参与 oracle 候选排序，不能作为论文方法的在线信号。

## 1.1 Novelty 边界

本实验**不声称首次提出 value-aware scoring、gradient attribution 或 output-aware KV selection**。特别是：

- LOCOS（arXiv:2607.01002）已经使用逐位置的 `attention × direct OV-to-logit` 分数、给出 gradient interpretation，并在 Qwen3-8B 上进行 held-out 因果消融；
- CriticalKV、LaProx 和 VATP 等工作已经覆盖 generic output/value-aware KV 评分。

因此这里唯一待检验的窄问题是：

> 在 RoPE-specific pre/post suppression event 上，包含完整后续层 Jacobian 的 exact downstream sensitivity，能否比 QK gap、attention mass 和 LOCOS-style direct-OV proxy 更准确地预测 matched score intervention 的符号与幅度？

若不能显著超过这些近邻 proxy，本方向不构成新的机制贡献。

## 2. 数据与目标

数据直接复用 `run_suppression_certificate_safety_probe_8b.py` 的英文构造：

1. Gold evidence：`The school register lists Xiaoming's age as nine years.`
2. Plausible conflict：`A family note lists Xiaoming's age as four/six/eight/two years.`
3. Lexical/format distractors：相同 school-register 格式、不同人名和年龄；
4. Filler：单 token 句号。

问题要求只输出一个 digit。Gold target 是无前导空格的单 token `9`；冲突 target 是与该 seed 对应的 digit。输入中没有 `VERIFIED`、`UNVERIFIED` 或其他显式真假标签。

默认长度为 8K 和 32K，8 个 seeds。每类采样 8 个 token，并强制保留 decisive age token。

## 3. 不可变 prefix 与 autograd final query

前缀仅执行一次标准 Qwen3-8B eager-attention prefill，并在 `torch.no_grad()` 下生成 KV cache。runner 保存 legacy prefix 的 storage identity、shape 和 tensor version，并在所有 replay 后验证它没有发生原地修改。

native 对照保留一次标准 final-query forward。所有 instrumented baseline 和因果 replay 则直接复用 safety probe 的只读 helper：每一层只在局部执行 `cat(prefix KV, current KV)`，不调用 `DynamicCache.update`，也不让某一层拼接后的长 cache 存活到下一层。这既保持 final-query attention 语义不变，也避免每个 singleton pass 复制一份完整 prefix。

只有最后一个 Query token 打开 autograd。模型参数全部冻结，final-token embedding 是唯一显式 gradient leaf。这样既保留完整的层间 Query/Value/MLP 传播，又不会为 8B 权重分配参数梯度。

runner 先执行一次完全未 instrument 的 native final query，用于测量 explicit-QK 测量路径的 instrumentation drift。机制结论以 instrumented native-score baseline 为共同干预基线。

## 4. Value-mediated 导数

某层某 head 的原生 attention 为：

$$
a_j=\frac{\exp(s_j)}{\sum_k\exp(s_k)},
\qquad
o=\sum_j a_jv_j.
$$

定义 Gold 与 conflict digit 的输出 margin：

$$
m=z_{\text{gold}}-z_{\text{conflict}}.
$$

令：

$$
g=\frac{\partial m}{\partial o}.
$$

softmax Jacobian 给出：

$$
\frac{\partial o}{\partial s_j}
=a_j(v_j-o).
$$

因此：

$$
\boxed{
\frac{\partial m}{\partial s_j}
=a_jg^\top(v_j-o)
}
$$

这一区分三种情况：

- 大于零：提高该 token 的 attention logit，局部上提高 Gold-vs-conflict margin；
- 小于零：提高它反而帮助冲突答案；
- 接近零：即使 attention mass 增加，对该输出决策也基本无效。

runner 对每个 `layer × head × sampled token` 保存：

- post-RoPE score；
- local-grid envelope score；
- suppression gap；
- 原生 attention probability；
- `dm_dscore = a*g^T(v-o)`；
- `suppression_gap × dm_dscore`；
- LOCOS-style `attention × direct OV-to-gold-logit`；
- LOCOS-style `attention × direct OV-to-gold-vs-conflict-margin`；
- 使用 `V-o` 的 centered direct-OV margin derivative；
- token class、位置、距离与 decisive-token 标记。

其中 LOCOS-style 对照为：

$$
L_j^{\mathrm{gold}}
=a_j u_{\mathrm{gold}}^\top W_O^{(h)}v_j,
$$

$$
L_j^{\mathrm{margin}}
=a_j (u_{\mathrm{gold}}-u_{\mathrm{conflict}})^\top
W_O^{(h)}v_j.
$$

更贴近 score derivative、但仍不含后续层 Jacobian 的 direct-OV 对照为：

$$
D_j^{\mathrm{OV}}
=a_j (u_{\mathrm{gold}}-u_{\mathrm{conflict}})^\top
W_O^{(h)}(v_j-o_h).
$$

它们与 exact downstream derivative 的关键差异是：LOCOS/direct-OV 只看当前层直接写向 unembedding 的路径；exact derivative 的

$$
g=\frac{\partial m}{\partial o_h}
$$

包含剩余 attention、residual、MLP、normalization 和输出层的完整 Jacobian。

这里的 suppression gap 为：

$$
C_j=s_j^{\mathrm{grid}}-s_j^{\mathrm{post}},
$$

其中 grid 使用可实现的相对距离集合
`{1,2,4,8,16,32,64,128}`。独立频率上界不作为选择器。

## 5. Singleton score-lift 因果验证

### 5.1 为什么不再默认联合干预

已完成的 8K seed-0 smoke 同时修改 `36 × 32 = 1,152` 个 score。target/all 的一阶预测均值为 `-0.00325`，实际 margin 变化却为 `+0.3125`；Spearman 为 `-0.258`，sign accuracy 只有 `50%`。实际 margin 还只出现 `0.125/0.375/0.5` 等粗粒度值。这不能否定单点导数公式，只说明把 1,152 个局部导数直接相加后，非线性交互与 BF16 输出量化使 closure 失真。

因此，旧 joint arm 仅保留为 `--run-joint-interventions` 显式开启的非线性压力测试，默认关闭；它不再承担公式验证。

### 5.2 冻结 top-N 候选

baseline 完成一次反向传播后，按每个 class 全局排序，默认冻结 top-16：

$$
R_j=\left|\max(C_j,0)\frac{\partial m}{\partial s_j}\right|.
$$

排序只读取 frozen baseline metric；一旦候选确定，后续任何干预结果都不能改变候选或顺序。每个 target 配一个同 `layer/head/class`、但位置不同的确定性 random control。random 选择只依赖 seed 与候选坐标，不消耗全局 RNG，也不读取 intervention outcome。

这个默认排名使用答案定义的 gradient，明确属于 oracle 诊断。可选排名还包括 `|C_j dm/ds|`、`|dm/ds|` 和仅使用正 suppression gap 的对照。

### 5.3 每次只修改一个 score

每次 replay 只修改一个冻结的 `(layer, head, token)`；干预不搬动位置、不修改 Q/K/V、不改变 support：

$$
s_j' = s_j + \epsilon,
\qquad
\epsilon=0.25.
$$

`0.25` 是统一 cap，也是正式 launcher 的固定 lift；不存在按 class、长度或干预结果调整的幅度。每个 class 分别执行 top-16 target 和 16 个 matched-random replay。

由 baseline 导数得到的一阶预测为：

$$
\widehat{\Delta m}
=
\epsilon
\left.
\frac{\partial m}{\partial s_{l,h,j}}
\right|_{\text{baseline}}.
$$

实际变化为：

$$
\Delta m
=m_{\text{intervention}}-m_{\text{baseline}}.
$$

### 5.4 FP32 双 token margin

PPL、Gold probability、next-token accuracy 和 full-vocabulary margin 仍来自模型原生 logits。因果 closure 使用最终 hidden state 与 Gold/conflict 两个 unembedding 行重新做 FP32 点积：

$$
m_{32}
=
h_L^{\top}
\left(u_{\mathrm{gold}}-u_{\mathrm{conflict}}\right).
$$

这消除了 BF16 LM-head dot product 的粗粒度，但 final hidden state 与模型权重来源仍可能是 BF16，所以不能恢复 transformer 内部已经舍入掉的信息。输出同时保留原生 `gold_conflict_margin_model_logits` 与主因果指标 `gold_conflict_margin_fp32_pair`。

### 5.5 报告指标

singleton 与 joint 必须按 `intervention_scope` 分开聚合，绝不混合。报告：

- predicted 与 actual margin change；
- sign accuracy；
- Pearson / Spearman correlation；
- absolute closure error；
- symmetric closure error：

$$
E_{\mathrm{sym}}
=
\frac{|\widehat{\Delta m}-\Delta m|}
{|\widehat{\Delta m}|+|\Delta m|+10^{-8}};
$$

- Gold PPL、Gold probability、next-token accuracy；
- Gold-vs-full-vocabulary 和 Gold-vs-conflict margin；
- native 与 instrumented baseline drift。

聚合表还直接比较下列量与 actual margin change 的 Pearson/Spearman：

- selected suppression-gap sum；
- selected attention-mass sum；
- LOCOS direct-OV gold/margin sum；
- centered direct-OV 一阶预测；
- suppression × direct-OV；
- suppression × exact downstream sensitivity；
- cap × exact downstream sensitivity。

主表是 `singleton_prediction_summary.csv`；`first_order_prediction_summary.csv` 保留所有 scope，并明确带 `intervention_scope` 列。

## 6. 解释边界

以下结果可以支持 value-mediated 机制：

- `dm/ds` 或 `suppression × dm/ds` 比 suppression gap、attention mass、LOCOS/direct-OV controls 更能预测实际 margin effect；
- target intervention 的 predicted/actual 符号高度一致；
- Gold evidence 的正向效应明显高于 conflict/filler；
- matched-random 无法复现 target 的效应；
- instrumentation drift 很小，prefix immutable 检查通过。

以下结果不能被包装成可部署方法：

- 直接使用 Gold/conflict margin 的 gradient 选择 token（本实验的默认 top-N 正是这种 oracle 选择）；
- 使用答案标签调 score lift 或 intervention 数量；
- 从该 oracle 导数直接声称推理时可以识别正确事实。

如果 Gold 和 plausible conflict 的 suppression 都很强，而它们的 Value-mediated 导数方向相反，这支持后续研究无标签 causal-utility proxy；它本身不等于已经得到该 proxy。

## 7. Go / No-Go 门槛

机制方向升级为 GO 至少需要：

- informative sign accuracy 不低于 80%；
- predicted 与 actual 的 Spearman correlation 不低于 0.7；
- target 明显优于 matched-random；
- Value 项相较 suppression-only 显著提高 exact intervention effect 的解释能力；
- exact downstream sensitivity 显著优于 LOCOS/direct-OV proxy，而不是复现已有 generic attribution；
- 在第二模型、两跳证据和自然长上下文数据上复现。

任一以下情况意味着 NO-GO 或需要改为非线性路径积分：

- 一阶导数与实际效应符号接近随机；
- closure error 随层数快速失控；
- Gold、conflict 和 filler 的 value-mediated 分布无法区分；
- 结论只存在于一个合成模板或单个模型。

## 8. 文件与正式 launcher

- Runner：`src/run_value_mediated_rope_probe_8b.py`
- Tests：`tests/test_value_mediated_rope_probe.py`
- Launcher：`scripts/run_value_mediated_probe_gpu67_20260801.sh`
- Output：`outputs/20260801_value_mediated_rope_probe_singleton_gpu67/`

launcher 严格限制为：

- physical GPU 6：seeds 0–3；
- physical GPU 7：seeds 4–7；
- lengths：8,192 和 32,768；
- Qwen3-8B NF4、BF16 compute、eager attention；
- score lift：0.25；
- 每 class 冻结 top-16 singleton 候选；
- 默认排名：`abs_positive_suppression_x_dm_dscore`；
- joint arm 默认关闭。

创建和测试这些文件不会启动服务器实验。
