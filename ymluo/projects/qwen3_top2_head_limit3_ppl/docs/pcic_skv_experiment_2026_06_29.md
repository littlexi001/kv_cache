# PCIC-SKV 远端实验记录（2026-06-29）

## 目标

本轮继续推进 `PCIC-SKV`：

```text
Pairwise Counterfactual Influence Cache with Synthetic KV Compensation
```

目标不是只用固定 `landmark` fallback，而是验证 synthetic KV compensation 是否能补偿被压缩层丢失的远程上下文。

## 实现内容

### 1. 扩展 `layerbudgetattn`

修改文件：

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/src/evaluate_qwen3_top2_head_limit3_ppl.py
```

新增 layer budget 类型：

- `synthetic` / `synthkv`：压缩层用 synthetic KV 替换远程上下文。
- `landmark_synthetic` / `hybrid_skv` / `pcic_skv`：压缩层保留 landmark，同时额外加入 synthetic KV 补偿。

当前支持两种 synthetic 方法：

- `mean`：每个远程分段构造均值 K/V prototype。
- `mass`：每个远程分段用 query-dependent softmax 聚合 V，并用 logsumexp 作为该 prototype 的 score。

### 2. 新增 PCIC-SKV 实验脚本

新增文件：

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/src/run_pcic_skv_local.py
ymluo/projects/qwen3_top2_head_limit3_ppl/scripts/run_pcic_skv_local.ps1
```

脚本功能：

- 加载模型一次；
- 共享 prefill cache；
- 对指定 Pairwise-CIC 组合测试多种 fallback；
- 输出 `pcic_skv_results.csv` 和 `summary.md`；
- 支持 `landmark`、`synthetic`、`hybrid` 三类 fallback。

## 远端环境

服务器：

```text
ssh fdong@10.176.37.31
```

项目目录：

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
```

环境：

```text
conda env: moe
python: /home/fdong/miniconda3/envs/moe/bin/python
torch: 2.4.0
transformers: 4.53.0
GPU: 8 x RTX 3090
```

注意：远端 Hugging Face Xet 下载会断连，因此运行时加：

```text
HF_HUB_DISABLE_XET=1
```

## 实验配置

共同配置：

```text
model: Qwen/Qwen3-0.6B
prefill_tokens: 4096
eval_tokens: 128
chunk_size: 16
recent_tokens: 512
landmark_stride: 64
synthetic_prototypes: 16
dtype: bfloat16
attn_implementation: eager
```

测试文本：

```text
data/war_and_peace_pg2600.txt
data/count_monte_cristo_pg1184.txt
```

测试组合来自之前 Pairwise-CIC 的强组合。

## 实验一：替换式 SKV

替换式 SKV 指压缩层不再用 landmark，而是直接用 synthetic prototype 代表远程上下文。

输出目录：

```text
outputs/server_pcic_skv_war4096_key
outputs/server_pcic_skv_monte4096_key
```

### War and Peace

Baseline：

```text
4.3510s / PPL 32.233544
```

| combo | fallback | method | delta_ppl | seconds | speedup |
| --- | --- | --- | ---: | ---: | ---: |
| `7,6` | landmark | - | -0.8608 | 4.4454 | -2.12% |
| `0,6` | landmark | - | -0.8322 | 4.5100 | -3.53% |
| `0,7` | landmark | - | -0.8213 | 4.3958 | -1.02% |
| `7,6` | synthetic | mean | -0.7064 | 4.6516 | -6.46% |
| `0,13` | synthetic | mean | -0.6215 | 4.6670 | -6.77% |
| `0,7` | synthetic | mean | -0.5603 | 4.6251 | -5.93% |
| `7,6` | synthetic | mass | +0.0222 | 5.6228 | -22.62% |

结论：

- `synthetic mean` 有补偿效果，但不如真实 landmark。
- `synthetic mass` 在当前实现下质量和速度都差，不适合作为下一阶段主线。

### Monte Cristo

Baseline：

```text
4.2974s / PPL 17.561929
```

| combo | fallback | method | delta_ppl | seconds | speedup |
| --- | --- | --- | ---: | ---: | ---: |
| `2,0,7,12` | landmark | - | -0.6951 | 4.5152 | -4.83% |
| `7,13` | landmark | - | -0.5803 | 4.3845 | -1.99% |
| `2,7` | landmark | - | -0.3422 | 4.4899 | -4.29% |
| `7,13` | synthetic | mean | -0.2172 | 4.6129 | -6.84% |
| `2,0,7,12` | synthetic | mean | -0.1347 | 4.9501 | -13.19% |
| `2,0` | synthetic | mass | -0.0995 | 5.5216 | -22.17% |

结论：

- Monte 上替换式 SKV 明显弱于 landmark。
- 单纯用 synthetic prototype 替换远程 token 会损失太多细粒度信息。

## 实验二：Hybrid-SKV

Hybrid-SKV 指：

```text
recent + landmark + synthetic prototypes
```

也就是不删除 landmark，而是在 landmark 基础上额外加入 synthetic KV 补偿。

输出目录：

```text
outputs/server_pcic_hybrid_skv_war4096_key
outputs/server_pcic_hybrid_skv_monte4096_key
```

### War and Peace

Baseline：

```text
4.3527s / PPL 32.233544
```

| combo | fallback | method | delta_ppl | seconds | speedup |
| --- | --- | --- | ---: | ---: | ---: |
| `7,6` | hybrid | mean | -0.8523 | 4.7747 | -8.84% |
| `0,7` | hybrid | mean | -0.6579 | 4.7788 | -8.92% |
| `0,13` | hybrid | mean | -0.5050 | 4.7127 | -7.64% |
| `0,7` | hybrid | mass | -0.2249 | 5.7266 | -23.99% |
| `0,13` | hybrid | mass | +0.0092 | 5.7505 | -24.31% |
| `7,6` | hybrid | mass | +0.1299 | 5.9671 | -27.06% |

结论：

- `hybrid mean` 接近 landmark，但没有超过 landmark。
- `hybrid mass` 仍然不好，且很慢。

### Monte Cristo

Baseline：

```text
4.3189s / PPL 17.561929
```

| combo | fallback | method | delta_ppl | seconds | speedup |
| --- | --- | --- | ---: | ---: | ---: |
| `2,0,7,12` | hybrid | mean | -0.6785 | 5.2482 | -17.71% |
| `7,13` | hybrid | mean | -0.5140 | 4.6894 | -7.90% |
| `2,7` | hybrid | mean | -0.3923 | 4.7230 | -8.56% |
| `2,0,7,12` | hybrid | mass | -0.1638 | 7.2452 | -40.39% |
| `2,7` | hybrid | mass | -0.1632 | 5.7517 | -24.91% |
| `7,13` | hybrid | mass | -0.0161 | 5.7306 | -24.63% |

结论：

- `hybrid mean` 在 Monte 上接近 landmark，但速度代价明显。
- `hybrid mean` 有一定补偿能力，但当前不是速度友好的实现。

## 当前判断

### 已验证成立

1. Pairwise-CIC 仍是有效预算来源：强组合在两本文本上都能降低 PPL。
2. Synthetic KV 不是完全无效：`mean` prototype 能提供部分补偿。
3. Hybrid-SKV 比替换式 SKV 更合理：保留 landmark 后质量明显更接近 landmark。

### 当前失败点

1. 替换式 SKV 不如 landmark，说明纯 prototype 太粗。
2. Hybrid-SKV 没有超过 landmark，说明简单均值 prototype 的信息增益不足。
3. `mass` 方法当前又慢又不稳定，不建议继续作为主路线。
4. 所有 Python/eager 实现都没有速度优势；后续速度必须靠更少 token 或 kernel 优化。

## 下一步方法建议

当前最合理的方向不是继续堆 synthetic prototype，而是改成：

```text
PCIC-R: Pairwise-CIC with Rescue
```

或者：

```text
PCIC-Selective-SKV
```

具体做法：

1. 默认用 Pairwise-CIC 找到安全压缩层。
2. 压缩层 fallback 先用目前最稳的 `landmark`，不要强行替换为 synthetic。
3. 加 token-level rescue gate：当预测 margin 低、entropy 高、或当前 token 风险高时，该 token 的压缩层回退 full attention。
4. Synthetic KV 只在“远程上下文非常长、landmark 数量受限”的情况下作为附加补偿，而不是替代 landmark。
5. 后续可以让 synthetic prototype 不是简单均值，而是按 query cluster / value residual / landmark residual 构造。

因此，投稿主线应从：

```text
PCIC-SKV = Pairwise budget + synthetic compensation
```

调整为：

```text
PCIC-R/SKV = Pairwise budget + robust rescue + selective synthetic compensation
```

现阶段最强、最稳的实验 baseline 仍是：

- War：`7,6` + landmark，`delta_ppl=-0.8608`
- Monte：`2,0,7,12` + landmark，`delta_ppl=-0.6951`

下一步优先实现 rescue gate，而不是继续优化 `mass` synthetic。
