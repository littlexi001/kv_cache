# Qwen3-0.6B 长上下文大海捞针年龄事实实验设计

日期：2026-07-09

## 目标

本项目测试 Qwen3-0.6B 在 8k 到 256k 长上下文中，是否还能从背景文本中检索并使用一条短事实：

```text
问题：小明今年是几岁？
埋伏事实：小明今年是九岁。
标准答案：九岁
```

主指标：

1. 最后答案的 conditional PPL。
2. 生成答案时对证据 span 的 attention mass。
3. 模型最终回答是否正确。

本实验先只测 full-context baseline，不引入 KV 压缩或 router。后续 KV memory 方法应以这条长度退化曲线作为参照。

## 实验矩阵

长度：

```text
8k, 16k, 32k, 64k, 128k, 256k
```

插入深度：

```text
10%, 50%, 90%
```

随机种子：

```text
0, 1, 2, 3, 4
```

每个 case 只埋一条事实，不是在同一个上下文里同时埋 10%、50%、90% 三个位置。

完整规模：

```text
6 lengths * 3 depths * 5 seeds = 90 cases
```

最小 smoke：

```text
6 lengths * depth=50% * seed=0 = 6 cases
```

## 位置延申配置

模型：

```text
Qwen/Qwen3-0.6B
```

Qwen3 原生上下文按 32k 级别处理；64k 以上需要显式 RoPE scaling。不能只改 `max_position_embeddings`。

| 目标长度 | RoPE 配置 | 实验定位 |
|---:|---|---|
| 8k | native | 原生范围 |
| 16k | native | 原生范围 |
| 32k | native | 接近原生上界 |
| 64k | YaRN factor 2.0 | 可靠长上下文区间 |
| 128k | YaRN factor 4.0 | 官方长上下文验证区间 |
| 256k | YaRN factor 8.0 | 外推 stress，不作为官方能力结论 |

示例配置语义：

```json
{
  "max_position_embeddings": 65536,
  "rope_scaling": {
    "rope_type": "yarn",
    "factor": 2.0,
    "original_max_position_embeddings": 32768
  }
}
```

运行时必须保存最终生效配置：

```text
model.config.max_position_embeddings
model.config.rope_scaling
model.config.rope_theta
transformers version
```

## 数据构造

每条样本结构：

```text
[无关 filler tokens]
[小明今年是九岁。]
[无关 filler tokens]

请只根据上文回答。如果上文没有相关信息，回答“无法确定”。
问题：小明今年是几岁？
答案：
```

目标长度只统计 haystack 主体 token 数，不包含最后问题和答案前缀。

Filler 要求：

1. 不出现“小明”“九岁”“年龄”等干扰词。
2. 用 tokenizer 精确控制长度。
3. 使用多段自然中文模板轮换，不使用全随机汉字或单句无限重复。

## PPL 指标

PPL 只计算答案条件概率，不计算整篇 haystack：

```text
NLL("九岁" | haystack + question + "答案：")
PPL = exp(NLL / answer_token_count)
```

输出字段：

| 字段 | 含义 |
|---|---|
| `target_length` | 8192...262144 |
| `actual_prompt_tokens` | 实际 prompt token 数 |
| `depth_percent` | 10/50/90 |
| `seed` | 随机种子 |
| `answer_nll` | 平均 token NLL |
| `answer_ppl` | `exp(answer_nll)` |

## 回答正确率

生成设置：

```text
do_sample = false
temperature = 0
max_new_tokens = 16
use_cache = true
```

归一化规则：

```text
九岁、9岁、九 岁 -> correct
无法确定、不知道、文中没有提到、不能确定 -> miss
其他 -> wrong
```

报告：

```text
accuracy
miss_rate
wrong_rate
generated_text
```

## Evidence Attention Mass

不要在 128k/256k 上保存完整 attention matrix。证据 attention mass 用流式方式计算：

1. 正常 prefill 得到 KV cache。
2. 对答案生成关键位置取 query 向量。
3. 对所有 key 分块计算 `q · k / sqrt(d)`。
4. 用稳定 softmax 累加证据 span 概率质量。
5. 只保存聚合结果，不保存完整 `[query_len, key_len]` attention。

证据 span：

```text
E = tokens("小明今年是九岁。")
```

定义：

\[
m_{l,h,t} = \sum_{j \in E} \mathrm{Attn}_{l,h,t,j}.
\]

报告聚合：

```text
mass_mean_all_layers_heads
mass_last_layer_mean_heads
mass_top_head
mass_top5_heads_mean
evidence_rank_by_page_mass
normalized_mass = mass / (|E| / context_length)
```

重点看两个 query position：

1. 生成第一个答案 token 前。
2. 生成完整答案“九岁”中第二个 token 前。

## 分阶段运行

### Phase 0: smoke

```text
lengths = 8k, 16k
depth = 50%
seed = 0
```

验收：

1. 8k/16k 能稳定回答“九岁”。
2. PPL 输出正常。
3. attention mass 输出非空，且没有物化完整 attention matrix。

### Phase 1: native window

```text
lengths = 8k, 16k, 32k
depths = 10%, 50%, 90%
seeds = 0..4
rope = native
```

规模：

```text
3 lengths * 3 depths * 5 seeds = 45 cases
```

### Phase 2: validated long context

```text
lengths = 64k, 128k
depths = 10%, 50%, 90%
seeds = 0..4
rope = YaRN factor 2.0 / 4.0
```

6 个 length-depth bucket：

```text
64k@10%, 64k@50%, 64k@90%   -> YaRN factor 2.0
128k@10%, 128k@50%, 128k@90% -> YaRN factor 4.0
```

规模：

```text
2 lengths * 3 depths * 5 seeds = 30 cases
```

### Phase 3: 256k stress

```text
length = 256k
depths = 10%, 50%, 90%
seeds = 0..2 first, then 0..4 if stable
rope = YaRN factor 8.0
```

256k 只作为外推压力测试。正结果不能写成官方支持 256k，负结果也不否定 128k 内能力。

## 服务器运行与输出

推荐按长度从短到长排队：

```text
8k -> 16k -> 32k -> 64k -> 128k -> 256k
```

输出目录：

```text
ymluo/projects/qwen3_long_needle_age/outputs/long_needle_age_20260709/
```

每次 run 保存：

```text
env.json
config_effective.json
cases.jsonl
generation_results.csv
answer_ppl.csv
evidence_attention_mass.csv
summary_by_length.csv
summary_by_length.md
run.log
```

`env.json` 至少包含：

```text
git commit
hostname
CUDA_VISIBLE_DEVICES
GPU name and memory
torch version
transformers version
flash-attn / xformers / triton version if used
model path
```

显存注意：

1. Qwen3-0.6B 参数小，但 256k full KV cache 仍可能达到数十 GB。
2. attention mass 必须 chunked。
3. 256k 优先使用 80GB GPU；如果只有 24GB/48GB，先确认 128k。

## 结果图

主图：

1. length vs accuracy，按 depth 分线。
2. length vs answer PPL。
3. length vs evidence attention mass / normalized mass。
4. length-depth heatmap，value 为 accuracy 或 evidence mass。

判断逻辑：

1. PPL 上升、attention mass 下降、准确率下降同步出现：主要是证据检索/定位退化。
2. attention mass 高但答案错：模型看到了证据，但绑定或输出失败。
3. PPL 低但生成错：decode、thinking 或输出格式有额外问题。

## 后续代码拆分

建议后续实现四个脚本：

```text
build_long_needle_age_cases.py
run_long_needle_generation.py
run_long_needle_answer_ppl.py
run_long_needle_attention_mass.py
```

先实现 Phase 0 smoke，再加服务器 launcher：

```text
run_long_needle_age_phase1_phase2_20260709.sh
run_long_needle_age_256k_stress_20260709.sh
summarize_long_needle_age_20260709.py
```

当前阶段只确定实验设计，不写代码。

## 参考

- Qwen3 Transformers long-context 文档：`https://github.com/QwenLM/Qwen3/blob/main/docs/source/inference/transformers.md#enabling-long-context`
- Qwen3-0.6B HF config：`https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json`
- vLLM context extension 示例：`https://docs.vllm.ai/en/latest/features/context_extension/`
