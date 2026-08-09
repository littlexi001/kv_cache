# Qwen3-8B 英文单-token 128K 长度退化机制研究

## 研究问题

固定一条 clean 两跳规则链，在只增加无关 filler 的情况下，回答正确 token 的
PPL 为什么变坏？退化来自以下哪一部分：

1. 历史 token 增多造成 softmax 分母稀释；
2. 无关 token 的极值竞争增强，目标证据排名下降；
3. query 与目标 key 的方向匹配随距离或位置外推变坏；
4. 证据虽然被检索到，但 value 读取、跨层传递或两跳组合失败；
5. 为延长上下文采用的 RoPE/YaRN 配置本身改变了短程行为。

## 固定数据

- 模型：Qwen3-8B，FP16。
- seed：0；条件：clean；答案严格为一个 token。
- 链：`river → window → basket`。
- 规则：

  ```text
  VERIFIED RULE: IF river IS ACTIVE THEN window BECOMES ACTIVE.
  VERIFIED RULE: IF window IS ACTIVE THEN basket BECOMES ACTIVE.
  ```

- 三个英文词在裸文本、规则和查询上下文中都严格对应一个稳定 tokenizer token；
  filler 词表中排除这三个词。
- 主扫描：证据放在 filler 中部，长度 0–128K，每 500 token 一个点，共 257 点。
- 每个长度使用同一条确定性 filler 前缀；因此相邻长度主要改变新增尾部、证据绝对位置
  和 query-evidence 距离，而不是重新随机采样整段文本。

## 记录的内部量

对 36 层 × 32 个 query head，在预测答案前的最后一个 query 位置记录：

- 三个单-token 证据位置的真实 post-softmax attention mass；
- pre-softmax QK logit、Q/K cosine、目标 key norm 和 query norm；
- 目标 key 在全部历史 token 中的 rank；
- 每个 head 的最大竞争 logit 与 `logsumexp`；
- 目标是否进入固定 Top-100 和动态 Top-2%；
- 正确答案概率、PPL、词表 Top-1 与 Top-5。

对单个 head，目标证据 `e` 的 full-attention 权重为：

\[
a_e = \frac{\exp(s_e)}{\sum_{j=1}^{N}\exp(s_j)}.
\]

因此即使目标 logit `s_e` 不变，历史 token 数 `N` 增加也会扩大分母；同时无关
logit 的最大值会随候选数增加而上升。两者分别由 `logsumexp` 和
`max_logit - target_logit` 测量。

若目标仍位于每个 head 的 Top-2%，截断并重归一化后的后验权重为：

\[
a'_e = \frac{a_e}{M_{2\%}},\qquad
M_{2\%}=\sum_{j\in\mathrm{Top2\%}} a_j.
\]

因此 Top-2% 能否帮助模型取决于两个条件：目标证据仍在候选集内，以及被保留的总
softmax mass `M_2%` 足够小。前者测 recall，后者决定重归一化放大倍数。

## 因果对照

主扫描完成后，在 8K、32K、64K、96K、128K 运行以下干预：

| 对照 | 改变 | 判别目的 |
|---|---|---|
| prefix full2 | 证据靠近开头 | 最长 query-evidence 距离 |
| recent full2 | 证据距 query 约数百 token | 固定总长度，消除远距离检索 |
| middle hop1 | 只问 `river → window` | 第一跳独立检索能力 |
| middle oracle-hop2 | 直接给 `window`，只问 `window → basket` | 绕过中间状态生成与绑定 |
| native factor-1 | 8K/32K 不做上下文外推 | 检查 factor-4 是否污染短长度 |
| YaRN factor-2 | 32K/64K | 区分外推强度与 token 数效应 |

解释原则：

- target logit/cosine 稳定、`logsumexp` 上升：以分母稀释为主；
- target logit/cosine 下降且 rank 变差：Q/K 检索方向也退化；
- recent 明显恢复：距离/位置是主要因素；
- 两个单跳好、full2 差：跨跳状态绑定或组合失败；
- 检索指标稳定但 PPL 仍坏：瓶颈位于 value 汇聚或更后的 residual/logit 读出；
- native/factor-2 明显优于 factor-4：一部分退化来自位置编码扩展配置。

## 主扫描结果

### 答案置信度随长度显著下降，但单点强烈振荡

固定链在 0 token 时 Gold PPL 为 1.3086；8K、32K、64K、96K、128K 的
PPL 分别为 138.5377、8.0243、342.2755、1131.3318、19347.1110。32K
单点的短暂恢复说明不能用少数锚点拟合单调曲线。对每 500-token 密集点分桶后，
趋势更稳定：

| filler 区间 | 点数 | Top-1 正确 | Top-5 召回 | median PPL | mean NLL |
|---|---:|---:|---:|---:|---:|
| 0–7.5K | 16 | 6.2% | 81.2% | 12.8783 | 2.6364 |
| 8–31.5K | 48 | 2.1% | 29.2% | 55.3200 | 3.9623 |
| 32–40.5K | 18 | 5.6% | 27.8% | 63.0393 | 4.2513 |
| 41–63.5K | 46 | 0% | 2.2% | 158.5687 | 5.0353 |
| 64–95.5K | 64 | 0% | 0% | 2358.3813 | 7.6623 |
| 96–128K | 65 | 0% | 1.5% | 4129.5548 | 8.0977 |

NLL 与长度 Pearson 相关系数为 0.7786、Spearman 为 0.7880；两条结果 token
的总 attention mass 与 NLL 的 Pearson 为 -0.8376、Spearman 为 -0.8947。

### 不是单一的 softmax 分母稀释

从 0 到 128K，最终结果 token 的全 layer-head 平均诊断变化为：

| 指标 | 0 | 128K | 变化 |
|---|---:|---:|---:|
| target QK logit | 4.8191 | -0.0772 | -4.8963 |
| Q/K cosine | 0.1553 | 0.0541 | -0.1012 |
| head logsumexp | 13.3591 | 17.4082 | +4.0491 |
| target rank | 52.9 | 27674.8 | +27622.0 |
| max competitor gap | 7.7921 | 15.9169 | +8.1247 |
| target attention mass | 0.006386 | 0.000663 | 约 9.6× 下降 |

因此观察到三件事同时发生：目标 Q/K 方向匹配变弱、softmax 分母增大、无关
token 的极值竞争变强。query norm 从 16.4959 增到 18.3902，并未衰减；主要变化
是方向而不是 query 向量长度。

### 第二跳桥接位置最脆弱

128K 时三个关键单-token 位置为：

| 位置 | mass | mean logit | mean rank | Top-2% heads | cosine |
|---|---:|---:|---:|---:|---:|
| 第一跳结果 `window` | 0.000602 | 0.5013 | 25811.4 | 41.5% | 0.0699 |
| 第二跳规则输入 `window` | 0.000092 | -1.3079 | 33107.8 | 24.7% | 0.0221 |
| 最终结果 `basket` | 0.000663 | -0.0772 | 27674.8 | 35.9% | 0.0541 |

同一个 `window` 在第一条规则 consequent 和第二条规则 antecedent 是两个不同
位置；后者的所有检索指标最差。这表明两跳任务还存在中间状态与第二条规则绑定
失败，不能只追踪最终 `basket`。

### 为什么动态 Top-2% 仍然有机会工作

把 8K 时最终证据 rank≤100 的 281 个 layer-head 固定为该样本的“检索 heads”。
追踪同一批 heads 得到：

| 长度 | Top-2% budget | 证据仍在 Top-2% | 仍在 Top-100 | target mass | target logit | target rank |
|---:|---:|---:|---:|---:|---:|---:|
| 8K | 162 | 100.0% | 100.0% | 0.023220 | 9.8719 | 35.4 |
| 32K | 642 | 97.9% | 78.6% | 0.015825 | 10.2985 | 202.5 |
| 64K | 1282 | 95.4% | 60.1% | 0.004563 | 10.0031 | 330.9 |
| 96K | 1922 | 93.6% | 54.4% | 0.005693 | 9.6995 | 572.4 |
| 128K | 2562 | 86.5% | 29.2% | 0.002684 | 8.6457 | 1850.1 |

绝对 rank 明显变差，但动态 2% 的预算也随长度增长，所以 128K 仍覆盖 86.5%
的原检索 heads；与此同时 full attention 下的目标 mass 已下降约 8.7 倍。这个组合
正好满足 Top-2% 截断可能有效的条件：大部分证据没有被剪掉，删除长尾后重新归一化
可以放大证据权重。后续干预另外计算实际完整分布上的 Top-2% 保留质量与放大倍数。

### 退化发生在哪些层

把每 6 层合为一段，并比较 0 与 128K 的最终结果位置，得到：

| layer 段 | attention mass 长/短比 | target logit 变化 | cosine 变化 | logsumexp 变化 | rank 变差 |
|---|---:|---:|---:|---:|---:|
| 0–5 | 0.0024× | -12.6935 | -0.1604 | +3.7316 | +61,202 |
| 6–11 | 0.0378× | -7.1460 | -0.1434 | +4.5236 | +38,356 |
| 12–17 | 0.1781× | -1.0125 | -0.0322 | +4.7928 | +16,391 |
| 18–23 | 0.1037× | -0.2587 | -0.0345 | +4.5521 | +5,944 |
| 24–29 | 0.3184× | -2.9950 | -0.1013 | +3.9414 | +17,861 |
| 30–35 | 0.0460× | -5.2721 | -0.1353 | +2.7528 | +25,977 |

因此不是所有层以同一种方式失败。最前 12 层和最后 6 层的 Q/K 方向退化最强；
12–23 层的 target logit 相对稳定，主要损失来自 `logsumexp` 增长，即有越来越多的
竞争 token 分享 softmax。中间层一度保住局部关联，并不足以保证最终 residual stream
仍能稳定读出 `basket`。

## 位置、位置编码与 Top-2% 因果对照

### 证据位置不是简单的“越近越好”

| filler 长度 | prefix full2 PPL | middle full2 PPL | recent full2 PPL |
|---:|---:|---:|---:|
| 8K | 31.8477 | 138.5377 | 5.5940 |
| 32K | 89.5515 | 8.0243 | 6.9250 |
| 64K | 15.6354 | 342.2755 | 7.8015 |
| 96K | 110.6330 | 1131.3318 | 47.7948 |
| 128K | 179.3916 | 19347.1110 | 904.6862 |

8K–96K 时 recent 最好，说明缩短 query-evidence 距离通常有效；但 128K 时 prefix
反而优于 recent，而 middle 最差两个数量级。这排除了“只有距离”这一单因解释：
模型同时存在 lost-in-the-middle、绝对位置外推敏感性，以及 filler 内容截断点造成的
非单调振荡。

### RoPE 外推配置会改变曲线，但不是唯一原因

| 长度 | native factor-1 | YaRN factor-2 | YaRN factor-4 |
|---:|---:|---:|---:|
| 8K | 13.9675 | — | 138.5377 |
| 32K | 24.4046 | 125.5507 | 8.0243 |
| 64K | — | 654.5193 | 342.2755 |

factor-1 在 8K 更好，却在 32K 不如 factor-4；factor-2 在 32K/64K 都不如
factor-4。外推配置会显著改变单点结果，但不存在一个 factor 在所有长度都占优，
因此不能把长程退化全部归咎于某个固定 RoPE factor。

### 单次最终-query重归一化不足以解释稀疏模型收益

从完整 attention 分布做后验 Top-2% 截断，平均仍保留约 92%–97% 的原 softmax
质量，所以目标证据的直接重归一化放大通常只有约 1.03–1.08 倍。这与此前真实稀疏
运行中观察到的巨大 PPL 改善不在同一量级。更可能的解释是：逐层剪掉低质量 value
会改变后续 residual 和 query 的形成轨迹，收益经过多层累积；它不是只在最后一个
query 上把现有 attention mass 除以一个常数。这个假设需要用真实逐层 Top-2% 重跑、
并逐层交换 full/sparse residual 才能正式验证。

## 单跳完形对照：关联还在，普通问答没有稳定调用它

普通单跳提示仍包含任务说明和答案格式，可能把“检索关联”和“理解/执行指令”混在
一起。为此新增更直接的完形提示：给出起点，并要求补全精确匹配的 VERIFIED RULE
后件。正文、证据位置和 filler 完全不变。

| 长度 | full2 普通问答 | hop1 普通问答 | oracle-hop2 普通问答 | hop1 精确完形 | oracle-hop2 精确完形 |
|---:|---:|---:|---:|---:|---:|
| 8K | 138.5377 | 1183.9522 | 116.3908 | 1.0142 | 1.0239 |
| 32K | 8.0243 | 46.4750 | 55.1029 | 1.0357 | 1.0330 |
| 64K | 342.2755 | 219.5791 | 82.7661 | 1.0595 | 1.1804 |
| 96K | 1131.3318 | 390.6416 | 187.2600 | 1.1179 | 1.8835 |
| 128K | 19347.1110 | 1693.9822 | 9655.1729 | 1.2175 | 1.2019 |

128K 时两个精确完形的正确 token 概率约为 82%–83%，而 full2 普通问答仅约
0.0052%。所以不能说长文本已经抹掉 `river → window` 或 `window → basket` 的局部
关联。更准确的结论是：模型仍有一条能读出关联的内部路径，但普通长上下文问答没有
稳定形成和调用正确的 query；两跳时还要把第一跳状态绑定到第二条规则，再把结果传到
答案槽，误差会继续累积。

这个对照也说明，Gold PPL 不是“纯检索能力”指标，它同时测 prompt 路由、答案槽
预测和跨层读出。完形提示重复了规则形式并给定 antecedent，任务明显更容易，因此它
是局部关联是否仍可访问的上限测试，不是对自然问答准确率的替代。

## 机制结论

本样本的长度退化不是单因果，而是以下过程叠加：

1. **分母与极值竞争增长。** token 变多使 `logsumexp` 上升，无关 token 的最大
   logit 也变强；即使证据 logit 不变，softmax mass 仍会下降。
2. **Q/K 方向本身漂移。** 128K 时证据 cosine、target logit 和 rank 同时恶化，
   尤其发生在前 12 层和最后 6 层，因此不只是机械稀释。
3. **位置与外推配置敏感。** recent 通常恢复，但 128K 是 prefix 最好、middle 最差；
   这符合 lost-in-the-middle 与绝对位置外推共同作用，而非单一相对距离规律。
4. **query 路由和两跳状态绑定失败。** 精确单跳完形在 128K 仍近乎满置信度，说明
   局部关联可访问；普通问题和完整两跳的巨大差距主要在如何构造 query、传递中间状态
   和读出答案。
5. **桥接位置最脆弱。** 第二条规则里的 `window` 比第一条规则结果位置和最终
   `basket` 都更差，是两跳组合最自然的断点。

因此，后续优化不应只追求最后一层提高某个证据 token 的 attention。更有针对性的
方向是：在每层/每 head 保护可用证据候选，显式维持中间状态到第二跳 antecedent 的
绑定，并验证稀疏 attention 是否通过过滤 value 噪声改善后续 query/residual 轨迹。

## 下一轮最有判别力的实验

- 在 8–16 条不同英文单-token 链、多个 filler seed 上复现位置与完形结论，给出均值和
  bootstrap 区间，避免把 seed 0 的非单调峰谷误当总体规律。
- 做 layer-swap：前 `L` 层使用真实 Top-2% sparse、后面恢复 full（以及反向），定位
  稀疏收益从哪一层开始进入 residual stream。
- 对 full/sparse 同时保存每层 query、attention output 和 residual；比较 sparse 是否
  先改善桥接 `window` 的可线性解码性，再改善 `basket` 输出概率。
- 固定总长度和绝对位置，只改变无关 filler 的词汇重复度、主题相似度与最大竞争 logit，
  分开测“候选数量”与“高相似竞争者”效应。
- 把精确完形逐步改回自然问题（去掉 VERIFIED、去掉规则模板、要求两跳），做提示难度
  阶梯，找出 PPL 首次崩坏发生在检索、组合还是答案格式阶段。

## 结果文件

远程原始目录：

```text
/home/fdong/ymluo/projects/qwen3_local_rule_failure_boundary/outputs/
  attention_confidence_qwen3_8b_english_single_token_128k_20260718/
  attention_confidence_qwen3_8b_english_mechanism_probes_20260718/
```

主输出包括 `analysis_summary.csv`、`length_bin_summary.csv`、
`role_mechanism_summary.csv`、`retrieval_head_retention.csv`、
`head_trends.csv` 和 `analysis_report.md`；干预输出汇总为
`combined_analysis/mechanism_comparison.csv` 与 `mechanism_report.md`。
