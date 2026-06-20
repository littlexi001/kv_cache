# Section 23: Qwen3-0.6B Routed Top4 Head 预训练实验总结

## 1. 实验动机

前面的实验主要是在已经训练好的 Qwen3-0.6B 上做 post-hoc 的 attention head-token 裁剪。例如每个 head 只保留 top 2% 历史 token，或者进一步限制一个 token 最多进入 3 个 head。结果显示：直接在训练好的 dense attention 模型上删 head-token 连接，很容易导致 PPL 崩掉。

这个现象有一个合理解释：

```text
原模型从训练开始就是 dense attention。
它可能已经学会让某些 token 同时服务很多 head，特别是 sink token、最近 token 和高频结构 token。
后处理阶段强行删除这些连接，会破坏已经形成的计算图。
```

所以本实验换了一个问题：

```text
如果模型从随机初始化开始，就被要求每个 token 只能写入 4 个 attention head，
模型能不能自己学出适应该约束的表示和路由方式？
```

这个实验的目标不是马上证明可以省 KV cache，而是先验证：

```text
routed top4 head 约束是否可以从训练初期稳定学习，而不是像 post-hoc pruning 一样直接崩掉。
```

## 2. 实验假设

可证伪假设：

```text
Qwen3-style 模型如果从随机初始化开始训练 hard top4 head routing，
那么它可以在不发生 NaN、不发生明显 router collapse 的情况下，让 CE loss 明显下降。
```

如果训练 CE 接近随机初始化水平，或者 router 很快只使用极少数 head，或者训练频繁 NaN，则该实现失败。

如果训练 CE 从 `log(vocab_size)` 附近明显下降，并且 router 仍然有多 head 使用，则说明 routed 训练范式至少是可训练的。

## 3. 模型结构

项目位置：

```text
ymluo/projects/qwen3_routed_top4_mha_pretrain
```

基础配置来自官方模型目录：

```text
/mnt/workspace/Qwen3-0.6B/config.json
```

使用 Qwen3-style decoder-only causal LM。主要参数沿用 Qwen3-0.6B：

```text
hidden_size = 1024
num_hidden_layers = 28
num_attention_heads = 16
head_dim = 128
vocab_size = 151936
```

注意：这个实验把 attention 实现成 Qwen3-style MHA：

```text
num_query_heads = 16
num_key_value_heads = 16
```

每一层都有一个独立 gate：

```text
gate_logits[l, t] = W_gate[l] x[l, t]
selected_heads[l, t] = top4(gate_logits[l, t])
```

其中：

```text
l: layer index
t: token position
x[l, t]: 该层 attention 输入前的 token hidden state
W_gate[l]: 第 l 层独立 gate 矩阵
```

得到 hard routing mask：

```text
M[l, t, h] = 1, if head h is in top4(gate_logits[l, t])
M[l, t, h] = 0, otherwise
```

## 4. Attention 计算规则

对每个 token，每层只选择 4 个 head。

实现里的规则是：

```text
1. 先照常计算 16 个 head 的 Q/K/V。
2. 对未选中的 token-head 槽位，将 Q/K/V 乘 0。
3. attention 时，一个 key token j 对 head h 可见，当且仅当：
   j <= t 且 M[l, j, h] = 1
4. 一个 query token t 在未选中的 head h 上，输出被置 0。
5. 16 个 head 槽位仍然保留，最后经过 o_proj 回到 hidden_size。
```

也就是说，当前实现是：

```text
dense tensor + mask
```

不是：

```text
真正 ragged KV cache
```

因此当前实验不省计算，也不真实省显存。它测试的是训练范式：

```text
模型能不能在每个 token 只写入 4 个 head 的约束下学起来。
```

真正 KV cache 节省要在后续把 dense tensor 改成 ragged/head-local KV storage 才能验证。

## 5. Gate 的训练方式

前向传播使用 hard top4：

```text
hard = one_hot(top4(gate_logits))
```

反向传播使用 straight-through estimator：

```text
probs = softmax(gate_logits / temperature)
route = hard + probs - stopgrad(probs)
```

这样前向行为是离散 top4，梯度近似从 softmax 概率传回 gate。

训练中还加了两个辅助 loss：

```text
router_load_loss:
  鼓励平均 gate probability 在 16 个 head 上更均衡。

router_z_loss:
  惩罚 gate logits 的 logsumexp 过大，避免 router logit scale 失控。
```

总 loss：

```text
total_loss = CE
           + 0.01 * router_load_loss
           + 0.001 * router_z_loss
```

其中 CE 是 next-token cross entropy。

## 6. 训练数据

训练数据根目录：

```text
/mnt/workspace/dclm
```

最初用户给过一个文件：

```text
/mnt/workspace/dclm/global-shard_01_of_10/local-shard_0_of_10/part-00000.txt
```

这个文件现在只作为路径示例，不再作为唯一训练文件。

当前默认数据采样规则：

```text
1. 递归扫描 /mnt/workspace/dclm/**/*.txt。
2. 使用 dataset_sample_seed = 1234 打乱文件列表。
3. 默认采样 dataset_sample_files = 1024 个 txt 文件。
4. 每个文件最多读取 tokenize_max_chars_per_file = 250000 字符。
5. 全局最多读取 tokenize_max_chars = 200000000 字符。
6. 用 /mnt/workspace/Qwen3-0.6B tokenizer 进行 tokenization。
7. 把 token 写入本次 run 的 token cache。
```

token cache 默认位置：

```text
<output_dir>/token_cache/train_tokens.uint32.bin
```

metadata 位置：

```text
<output_dir>/token_cache/train_tokens_meta.json
```

metadata 记录：

```text
all_file_count
sampled_file_count
dataset_sample_seed
max_chars
max_chars_per_file
total_chars
total_tokens
sampled_files
```

这个设计的原因是 DCLM 文件太多、总内容太大，不能每次把全部数据 tokenization。每次 run 采样一部分文件，可以让训练覆盖整个数据树，同时控制 token cache 构建时间。

## 7. 训练设置

服务器 GPU：

```text
8 x 80GB GPU
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

默认训练命令：

```bash
cd ymluo/projects/qwen3_routed_top4_mha_pretrain
bash scripts/nohup_train_8x80g.sh
```

默认输出目录：

```text
/mnt/workspace/lym_code/scripts/kv_cache/kv_cache/ymluo/projects/qwen3_routed_top4_mha_pretrain/output/routed_top4_qwen3_0p6b_runs
```

关键超参数：

```text
seq_len = 2048
per_device_batch_size = 1
gradient_accumulation_steps = 8
world_size = 8
max_train_seconds = 72000
learning_rate = 3e-4
warmup_steps = 1000
weight_decay = 0.1
max_grad_norm = 1.0
router_top_k = 4
router_aux_loss_coef = 0.01
router_z_loss_coef = 0.001
router_temperature = 1.0
router_noise_std = 0.1
save_steps = 500
```

每个 optimizer step 的 token 数：

```text
tokens_per_step
= seq_len * per_device_batch_size * gradient_accumulation_steps * world_size
= 2048 * 1 * 8 * 8
= 131072 tokens
```

因此如果日志显示：

```text
step = 8700
```

则训练 token 数约为：

```text
8700 * 131072 = 1140326400 tokens
```

即约：

```text
11.40 亿 tokens
```

## 8. 工程问题和修复

### 8.1 NCCL timeout

训练最初遇到过类似报错：

```text
WorkNCCL ... OpType=ALLREDUCE ... Timeout(ms)=600000
```

原因不是模型训练 step 内部出错，而是：

```text
rank0 正在构建 token cache；
其他 rank 过早进入 NCCL barrier；
等待超过 600 秒后 NCCL watchdog 超时。
```

修复方式：

```text
1. rank0 负责构建 train_tokens.uint32.bin 和 train_tokens_meta.json。
2. 非 rank0 不立即进入 NCCL barrier。
3. 非 rank0 先通过 filesystem polling 等待 token cache 文件出现。
4. cache 完成后，所有 rank 再进入短 barrier。
```

默认等待上限：

```text
CACHE_WAIT_TIMEOUT_SECONDS = 86400
```

即 24 小时。

### 8.2 训练速度

当前日志中典型吞吐：

```text
tok/s ≈ 27000 到 28000，全局 8 卡
```

每卡约：

```text
3400 tokens/s
```

这对 8 张 80GB 卡来说偏慢，但符合当前原型实现的预期，因为当前实现并没有利用 top4 routing 省计算。

主要原因：

```text
1. 仍然计算完整 16 个 head 的 Q/K/V。
2. 仍然构造 dense attention score，形状约为 [batch, 16, seq, seq]。
3. top4 只是通过 mask 把未选 token-head 槽位置零。
4. 使用 gradient checkpointing，节省显存但增加重算。
5. 没有使用 FlashAttention、Megatron、DeepSpeed 或 ragged KV kernel。
```

所以当前 top4 routing 是结构训练实验，不是加速实验。

## 9. 当前训练结果

用户提供的训练日志片段：

```text
step=8480 loss=2.4444 ce=2.4223 load=0.5922 z=16.1810 entropy=1.3850 tok/s=27813.9 elapsed_h=11.17
step=8500 loss=2.5631 ce=2.5404 load=0.6209 z=16.5061 entropy=1.3827 tok/s=27535.5 elapsed_h=11.20
step=8700 loss=2.6043 ce=2.5815 load=0.6092 z=16.7265 entropy=1.3790 tok/s=27320.3 elapsed_h=11.46
```

训练 token 数估计：

```text
step 8500:
8500 * 131072 = 1114112000 tokens
约 11.14 亿 tokens

step 8700:
8700 * 131072 = 1140326400 tokens
约 11.40 亿 tokens
```

随机初始化时 CE 期望接近：

```text
log(vocab_size) = log(151936) ≈ 11.93
```

当前 train CE 已降到约：

```text
2.4 到 2.6
```

对应 train PPL：

```text
PPL = exp(CE)
exp(2.5) ≈ 12.2
```

这个结果说明：

```text
1. routed top4 模型可以从随机初始化稳定训练。
2. CE loss 有大幅下降，不是只跑通 forward/backward。
3. post-hoc pruning 崩掉并不意味着 routed-head 约束本身无法训练。
```

但是还不能说明：

```text
1. 下游任务能力已经接近官方 Qwen3-0.6B。
2. routed 架构优于 dense baseline。
3. 真实 ragged KV cache 节省后的性能和质量已经成立。
```

原因是当前看到的是 train loss，不是 held-out validation loss，也不是下游任务 accuracy。

## 10. 下游评测方案

已经补充了 checkpoint-vs-baseline 评测脚本。

准备评测数据：

```bash
cd ymluo/projects/qwen3_routed_top4_mha_pretrain
pip install datasets transformers
bash scripts/prepare_downstream_eval_data.sh
```

默认任务：

```text
piqa
hellaswag
winogrande
arc_easy
arc_challenge
boolq
```

默认每个任务采样：

```text
500 validation examples
```

比较最新 routed checkpoint 和官方 Qwen3-0.6B：

```bash
bash scripts/eval_checkpoint_vs_baseline.sh
```

指定 checkpoint：

```bash
CHECKPOINT_DIR=/path/to/checkpoint-0008500 bash scripts/eval_checkpoint_vs_baseline.sh
```

评测输出：

```text
output/downstream_eval_results/<run_name>_<timestamp>/summary.json
output/downstream_eval_results/<run_name>_<timestamp>/multiple_choice_details.jsonl
output/downstream_eval_results/<run_name>_<timestamp>/eval_args.json
```

生成简表：

```bash
python eval/summarize_eval_results.py \
  output/downstream_eval_results/<run_name>_<timestamp>/summary.json
```

多选题 scoring 规则：

```text
score(choice) = mean log p(choice_token | prompt, earlier_choice_tokens)
prediction = argmax_choice score(choice)
accuracy = correct / total
```

这个规则对 routed checkpoint 和官方 baseline 完全一致。

## 11. Held-out text PPL 实验

为了确认 train CE 下降不是因为 DCLM token cache 重复或过于容易，需要增加 held-out text PPL 实验。

这个实验应该优先使用非 DCLM 文本。当前新增的默认方案是：

```text
held-out dataset = WikiText-103 validation
source = HuggingFace datasets: wikitext / wikitext-103-raw-v1 / validation
默认最多读取 5000000 字符
```

准备 held-out 文本：

```bash
cd ymluo/projects/qwen3_routed_top4_mha_pretrain
bash scripts/prepare_heldout_ppl_text.sh
```

比较 routed checkpoint 和官方 Qwen3-0.6B：

```bash
bash scripts/eval_heldout_ppl_vs_baseline.sh
```

指定 checkpoint：

```bash
CHECKPOINT_DIR=/path/to/checkpoint-0008500 bash scripts/eval_heldout_ppl_vs_baseline.sh
```

指定自己的非 DCLM 文本：

```bash
HELDOUT_TEXT_PATH=/path/to/non_dclm_validation.txt bash scripts/eval_heldout_ppl_vs_baseline.sh
```

评测指标：

```text
CE = held-out next-token cross entropy
PPL = exp(CE)
```

解释方式：

```text
如果 train CE 很低，但 held-out CE/PPL 很高：
  说明 train loss 可能主要来自重复数据、容易预测数据，或者泛化还不足。

如果 held-out CE/PPL 也随 checkpoint step 持续下降：
  说明 routed top4 模型确实在学习更通用的语言建模能力。

如果 routed checkpoint 明显差于官方 Qwen3-0.6B：
  这不等于 routed 架构失败，因为官方模型是完整预训练模型，而当前 checkpoint 只训练了约十几亿 token。
```

因此，这个实验的核心比较不是“当前 routed 模型是否已经超过官方模型”，而是：

```text
1. routed checkpoint 的 held-out PPL 是否显著低于随机初始化；
2. held-out PPL 是否随训练 step 下降；
3. routed checkpoint 和官方 Qwen3-0.6B 的差距有多大。
```

## 12. 如何解释与官方 Qwen3-0.6B 的比较

官方 Qwen3-0.6B 是完整预训练模型。当前 routed checkpoint 是从随机初始化开始，训练约十几亿 token 的中间 checkpoint。

因此下游比较不是公平训练预算比较。

合理解释方式：

```text
如果 routed checkpoint 远低于官方 Qwen3：
  这不说明 routed 架构失败，因为训练 token 和训练工程都远少于官方模型。

如果 routed checkpoint 明显高于随机水平：
  说明 routed top4 训练已经学到可迁移语言能力。

如果 routed checkpoint 随训练 step 持续提升：
  说明该训练范式值得继续投入。

如果 train CE 下降但下游 accuracy 很差：
  需要检查数据重复、validation loss、prompt 格式和训练泛化。
```

## 13. 当前结论

当前最稳妥的结论：

```text
Qwen3-style routed top4 MHA 模型可以从随机初始化开始训练。
在约 11.4 亿 token 后，train CE 已从随机水平约 11.93 降到约 2.5。
这说明 routed-head 约束本身不是不可训练的。
```

当前不能下的结论：

```text
1. 不能说它已经达到官方 Qwen3-0.6B 的能力。
2. 不能说它已经实现 KV cache 真实节省。
3. 不能说 top4 routing 一定优于 dense attention。
4. 不能仅凭 train loss 判断泛化质量。
```

## 14. 下一步建议

优先级最高：

```text
1. 跑 held-out text PPL，确认 train CE 下降不是数据重复导致。
   默认脚本已经新增：prepare_heldout_ppl_text.sh 和 eval_heldout_ppl_vs_baseline.sh。
2. 跑下游多选任务，与官方 Qwen3-0.6B 和随机 baseline 比较。
3. 保存不同 step 的 checkpoint，画 step vs validation CE / downstream accuracy。
```

其次：

```text
1. 做 dense MHA 同等训练 token baseline。
2. 用 PyTorch scaled_dot_product_attention 或 FlashAttention 替换手写 attention。
3. 研究 router entropy、hard_load_min、hard_load_max 随 step 的变化。
4. 实现真正 ragged KV cache，测试实际 cache 占用和推理速度。
```

如果下游表现较差，应先检查：

```text
1. validation CE 是否也在下降；
2. DCLM 采样是否过于重复；
3. tokenizer 和 checkpoint 加载是否正确；
4. 多选题 prompt 格式是否适合当前未 instruction-tuned 的 base 模型；
5. routed checkpoint 是否还需要更多 token 才能出现下游能力。
```
