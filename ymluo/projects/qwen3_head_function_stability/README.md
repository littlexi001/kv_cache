# Qwen3 Head Function and Cross-input Stability

这个项目回答两个问题：

1. Qwen3-0.6B 的不同 attention head 呈现哪些可重复的关注模式？
2. 同一个 head 的模式在改写、任务和输入域变化后是否保持一致？

这里把“功能”严格定义为**可观测 attention pattern**，不是直接定义成因果功能。每个 head 可以有多个标签；若要证明某个 head 对任务必不可少，还需做定向 head/link ablation。

## 功能画像

受控输入共 32 条，覆盖语义检索、英文/中文长距离句法、JSON/代码/XML/Markdown 结构和自然文本。脚本测量九类模式：

| 类别 | 测量对象 |
| --- | --- |
| `self` | 当前 token |
| `previous_token` | 前一个 token |
| `local_recent` | 距离 2–16 的局部历史 |
| `sink` | 最前 4 个 token |
| `punctuation` | 远端标点/分隔符 token |
| `lexical_copy` | 远端同词 token |
| `syntactic_dependency` | 手工标注的英中长距离主谓 anchor |
| `structural_anchor` | 括号、JSON、XML、代码围栏的匹配开端 |
| `semantic_evidence` | 与末尾问题对应的答案值 token/span |

三类手工 anchor 默认只使用标注 query span 的最后一个 token。对语义 QA 来说，这是 `Answer:` 后用于预测首个答案 token 的 causal-LM 位置；对句法与结构样例则是目标词或闭合符的最后一个 subtoken。这样不会把整段问题中尚未形成完整查询的早期位置平均进去。

每个 `(sample, layer, head, category)` 的分数为：

```text
log2_enrichment = mean_q clip(log2(attention_mass / causal_key_availability), -8, 8)
```

因此分数 `1` 表示该 head 给这一类 key 的质量约为均匀注意力基线的 2 倍。最终标签要求该类别的绝对 enrichment 大于 0，同时在所有 heads 之间的 robust z-score 不低于 1；这避免把“比其他 head 少忽略一些”误称为专门化，也避免仅因某类 token 数量多就把所有 heads 都标成同一类。

## 稳定性指标

- `head_rank_spearman_mean`：换输入后，全体 heads 对同一功能的排序是否保持。
- `paired_paraphrase_spearman_mean`：只比较同一事实/依存关系的受控改写。
- `profile_cosine_mean`：单个 head 的九维画像跨输入平均余弦。
- `primary_label_consistency`：逐样本主标签与全局主标签的一致率。
- `domain_agreement`：分域主标签与全局主标签的一致率。

预期解释不是“永远一致/永远不一致”的二元结论，而是区分：稳定偏置、同任务稳定但跨域切换、以及明显上下文敏感。

## 服务器运行

项目沿用 `ymluo/doc` 现有服务器约定：

```bash
ssh fdong@10.176.37.31
cd /home/fdong/ymluo/projects/qwen3_head_function_stability
CUDA_VISIBLE_DEVICES=4 bash scripts/run_server.sh
```

默认环境与模型：

```text
Python: /home/fdong/miniconda3/envs/moe/bin/python
Model:  /home/fdong/hrj/prove/Qwen3-0.6B
dtype:  float16
attention backend: eager
```

快速 smoke test：

```bash
SAMPLE_LIMIT=2 LAYERS=0-1 HEADS=0-3 MAKE_PLOTS=false \
OUT_DIR=/tmp/qwen3_head_function_smoke \
bash scripts/run_server.sh
```

正式运行：

```bash
OUT_DIR=/home/fdong/ymluo/projects/qwen3_head_function_stability/outputs/full_v1 \
CUDA_VISIBLE_DEVICES=4 \
bash scripts/run_server.sh
```

## 输出

| 文件 | 内容 |
| --- | --- |
| `report.md` | 自动生成的主要结果表 |
| `head_profiles.csv` | 每个 layer/head 的多标签画像与稳定性 |
| `category_head_rankings.csv` | 每类功能的 head 排名 |
| `category_stability.csv` | 跨输入和配对改写的 head-rank 稳定性 |
| `per_sample_head_features.csv` | 可审计的逐样本原始特征 |
| `paired_input_stability.csv` | 每个 head 在配对输入上的画像相似度 |
| `controlled_samples.csv` | 实际输入、域和标注类别 |
| `plots/primary_function_map.png` | layer × head 主模式图 |
| `plots/cross_input_stability_map.png` | layer × head 稳定性图 |
| `summary.json` | 配置、总体计数和运行时间 |

## 本地纯逻辑测试

测试不加载模型：

```bash
python -m unittest discover \
  -s ymluo/projects/qwen3_head_function_stability/tests \
  -p 'test_*.py'
```

## 结论边界

- Attention weight 不是 explanation；本项目首先回答“关注模式是否专门化和稳定”。
- `primary_function` 只是相对最突出类别，不表示该 head 只有一个功能。
- `mixed_or_common` 表示没有类别超过当前 robust-z 阈值，不表示 head 无用。
- 受控数据适合归因，但不能替代 LongBench、代码库、对话等自然分布复验。
- 下一阶段应对每类 top heads 做定向 link/head ablation，并测 answer NLL、句法最小对准确率和结构闭合 token loss。
