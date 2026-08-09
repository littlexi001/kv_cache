# Section 129：Attention Head 功能画像与跨输入稳定性（2026-07-11）

## 1. 研究问题

本项目直接研究：

1. 不同 attention head 可以按哪些关注模式分类？
2. 某个 head 若偏好 sink、recent、句法/结构、标点或语义证据，这种偏好在不同输入上是否一直保持？

需要先限定术语：本文的“功能”是从 attention weight 得到的**可观测关注模式**，不是未经干预验证的因果功能。实验允许一个 head 同时具有多个标签，并把 causal ablation 留作第二阶段。

项目路径：

```text
ymluo/projects/qwen3_head_function_stability
```

## 2. 为什么不能只看一条输入或强制单标签

单条文本上的 attention 热力图会把内容、位置和 token 数量混在一起。比如 recent 区域有 16 个候选 token，而一个远端 evidence 可能只有 1–3 个 token；直接比较 attention mass 会天然偏向前者。

本实验做三项修正：

1. 使用 32 条受控输入，覆盖语义 QA、英中句法、JSON/代码/XML/Markdown 结构和自然文本；
2. 用 `attention_mass / causal_key_availability` 做机会数校正；
3. 输出九维多标签画像，并分别测配对改写稳定性和跨域稳定性。

## 3. 功能类别

| 类别 | 定义 |
| --- | --- |
| `self` | 当前 token |
| `previous_token` | 前一个 token |
| `local_recent` | 距离 2–16 的局部历史 |
| `sink` | 序列最前 4 个 token |
| `punctuation` | 排除 sink/recent 后的远端标点或分隔符 |
| `lexical_copy` | 排除 sink/recent 后的远端同词 token |
| `syntactic_dependency` | 手工标注的英中长距离主谓依存 |
| `structural_anchor` | JSON/括号/XML/fence 的匹配开端 |
| `semantic_evidence` | 与末尾问题对应的答案值 token/span |

手工类别只取标注 query span 的最后一个 token。语义任务中该位置是 `Answer:` 之后用于预测首个答案 token 的 causal-LM 状态；若把问题尾部 8 个 token 一起平均，会显著稀释语义检索信号。

逐样本分数：

```text
score(h, c, x)
  = mean over applicable queries q
      clip(log2(attention_mass(h,c,x,q) / key_availability(c,x,q)), -8, 8)
```

每类再跨所有 heads 计算 robust z-score。主标签必须同时满足 `z >= 1.0` 和绝对 `log2_enrichment > 0`；这避免把“比其他 head 少忽略一些目标 token”误称为专门化。未超过阈值的 head 标为 `mixed_or_common`。

## 4. “是否一直保持一致”的判据

实验不使用一个含混的稳定性数字，而是同时报告：

| 指标 | 回答的问题 |
| --- | --- |
| head-rank Spearman | 换输入后，仍是同一批 heads 在该类别上排名靠前吗？ |
| paired-paraphrase Spearman | 同一事实或依存关系仅改写措辞后，排序保持吗？ |
| profile cosine | 单个 head 的完整多类画像是否相似？ |
| primary-label consistency | 单个 head 的逐输入主模式是否与总体标签一致？ |
| domain agreement | 分别在 QA、句法、结构、自然文本中聚合后，主模式是否一致？ |

这允许识别三种现象：

1. `stable_bias`：有稳定偏好，但具体 key 仍由输入决定；
2. `intermediate`：同任务/改写较稳定，跨任务会改变；
3. `context_sensitive`：画像或主标签随输入明显切换。

## 5. 服务器与模型

已按现有 `ymluo/doc` 记录核验：

```text
server: fdong@10.176.37.31
project: /home/fdong/ymluo/projects/qwen3_head_function_stability
python: /home/fdong/miniconda3/envs/moe/bin/python
model: /home/fdong/hrj/prove/Qwen3-0.6B
GPU: RTX 3090
```

复现命令：

```bash
ssh fdong@10.176.37.31
cd /home/fdong/ymluo/projects/qwen3_head_function_stability
CUDA_VISIBLE_DEVICES=4 \
OUT_DIR=/home/fdong/ymluo/projects/qwen3_head_function_stability/outputs/full_v3 \
bash scripts/run_server.sh
```

## 6. 正式运行

正式结果目录：

```text
server: /home/fdong/ymluo/projects/qwen3_head_function_stability/outputs/full_v3
local:  ymluo/projects/qwen3_head_function_stability/outputs/full_v3
```

实验规模：

| 项目 | 数值 |
| --- | ---: |
| model | Qwen3-0.6B |
| layers | 28 |
| attention heads / layer | 16 |
| total heads | 448 |
| controlled inputs | 32 |
| semantic QA | 12 |
| English/Chinese syntax | 8 |
| JSON/code/XML/Markdown structure | 8 |
| natural prose | 4 |
| full runtime | 22.0 s |

正式运行前先通过了 2 samples × 2 layers × 4 heads 的真实模型 smoke test，以及 5 个不加载模型的单元测试。

## 7. 不同 Head 可以分成什么功能

分类采用保守条件：主类别必须同时满足 `robust z >= 1.0` 和绝对 `log2_enrichment > 0`。448 个 heads 中 172 个得到主标签，276 个为 `mixed_or_common`：

| 主模式 | Head 数 | Top heads |
| --- | ---: | --- |
| self | 48 | L0/H3, L11/H9, L6/H7, L9/H7, L0/H7 |
| previous token | 38 | L15/H3, L2/H12, L0/H2, L1/H3, L20/H0 |
| punctuation | 19 | L2/H10, L2/H3, L1/H15, L1/H12, L10/H6 |
| structural anchor | 17 | L3/H3, L16/H2, L11/H14, L8/H9, L6/H5 |
| syntactic dependency | 14 | L6/H1, L11/H1, L5/H11, L16/H1, L13/H3 |
| lexical copy | 12 | L2/H3, L1/H8, L1/H15, L2/H11, L2/H10 |
| semantic evidence | 9 | L21/H13, L22/H7, L21/H11, L24/H11, L19/H13 |
| sink | 9 | L19/H4, L27/H8, L27/H3, L5/H0, L27/H0 |
| local recent | 6 | L1/H4, L1/H2, L0/H10, L4/H1, L2/H1 |
| mixed or common | 276 | 没有类别同时超过相对和绝对阈值 |

若允许多标签，共有 66 个 heads 同时满足两个或更多功能标签。因此实验不支持“每个 head 只有一种功能”的硬划分。

层次上有明显但不绝对的趋势：

1. 标点、lexical copy 和 local recent 的 strongest heads 较多出现在早层；
2. 句法和 structural anchor 的 strongest heads 主要位于中层；
3. semantic evidence 的 9 个主标签 heads 全部集中在 L19–L24；
4. sink heads 分布更散，但 L27 有多个突出 heads。

语义项最强的是 `L21/H13`，平均 `log2_enrichment=2.23`，即在用于预测首个答案 token 的位置，对正确答案值的 attention mass 约为机会数基线的 `2^2.23 ≈ 4.7` 倍。

主标签图：

![Primary head function map](../projects/qwen3_head_function_stability/outputs/full_v3/plots/primary_function_map.png)

## 8. Head 的偏好在不同输入上是否永远一致

结论是：**存在很强的稳定偏置，但不是永远固定；稳定程度与功能类型有关。**

| 模式 | 跨全部输入 head-rank Spearman | 同一关系改写 Spearman |
| --- | ---: | ---: |
| self | 0.899 | 0.980 |
| previous token | 0.890 | 0.978 |
| local recent | 0.895 | 0.983 |
| sink | 0.877 | 0.983 |
| punctuation | 0.816 | 0.960 |
| lexical copy | 0.666 | 0.964 |
| syntactic dependency | 0.778 | 0.936 |
| structural anchor | 0.609 | 0.840 |
| semantic evidence | 0.765 | 0.954 |

可以得到三个层次的判断：

1. **位置型 heads 最稳定。**self/previous/recent/sink 的跨输入相关性均为 `0.877–0.899`。
2. **同一关系只改写措辞时，大多数专门化仍然保持。**九类的 paired Spearman 为 `0.840–0.983`。
3. **换内容或任务域后会重排。**structural anchor、lexical copy、semantic evidence 的全输入相关性明显低于 paired 改写相关性，说明它们有可复用偏置，但是否激活以及关注哪个 token 强烈依赖当前输入。

按预注册启发式阈值分类：

| 稳定性类型 | Head 数 | 占比 |
| --- | ---: | ---: |
| stable bias | 273 | 60.9% |
| intermediate | 112 | 25.0% |
| context sensitive | 63 | 14.1% |

所有 heads 的完整画像跨输入 cosine 平均为 `0.726`、中位数为 `0.794`；同一关系改写的平均 cosine 为 `0.952`。在 172 个有主标签的 heads 中，逐输入主标签一致率平均为 `0.857`，跨域主标签一致率平均为 `0.714`。这再次说明“偏置稳定”与“每个输入都做完全相同的事”不是同一个命题。

跨输入画像图：

![Cross-input head stability](../projects/qwen3_head_function_stability/outputs/full_v3/plots/cross_input_stability_map.png)

## 9. 对三个研究问题的直接回答

1. **不同 head 能否分类？**可以按可观测 attention pattern 分成 self、previous、recent、sink、punctuation、lexical copy、syntax、structure 和 semantic evidence；但应保留多标签与未分类组。
2. **分类是否永久不变？**不是。模型中广泛存在稳定偏置，尤其是位置型偏置；但语义、复制和结构 heads 会随内容与任务明显调节。最准确的表述是“head 有稳定 prior，实际 attention target 是 query/context dependent”。
3. **能否据此直接删 head 或压缩 KV？**还不能。attention pattern 是观察证据，必须进一步用定向 link/head ablation 验证损失、答案准确率和结构/句法目标 token loss。

## 10. 结果文件

```text
report.md
summary.json
head_profiles.csv
category_head_rankings.csv
category_stability.csv
per_sample_head_features.csv
paired_input_stability.csv
controlled_samples.csv
plots/primary_function_map.png
plots/cross_input_stability_map.png
plots/category_specialization_maps.png
```

## 11. 解释边界与第二阶段

Attention pattern 只能说明某个 head 把权重放在哪里，不能单独证明这些 link 对最终预测必要。若第一阶段找到清晰的专门化 heads，第二阶段应：

1. 只屏蔽该 head 指向目标类别的 links，并重新归一化；
2. 同时做整 head ablation，区分“目标 link 重要”与“head 整体重要”；
3. 对语义任务测 answer NLL/accuracy；
4. 对句法与结构最小对测目标 token loss；
5. 使用随机同层 head、等量随机 links 和 attention-mass-matched links 作为对照。
