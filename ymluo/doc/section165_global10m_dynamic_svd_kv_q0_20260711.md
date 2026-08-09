# 10M 全局动态 SVD32-KV 检索：Lou Breslow 两跳出生地实验

## 1. 实验问题

问题是：`Where was the wife of Lou Breslow born?`

证据链为：

1. `Lou Breslow's wife was Marion Byron.`
2. `Marion Byron was born in Dayton, Ohio.`
3. 最终答案：`Dayton, Ohio`

目标不是在已知 source 内扫描，而是在不知道 source 的情况下，从接近 10M tokens 的全局语料中检索 block，并在生成过程中每 3 个 token 刷新一次查询，检查第二跳证据能否被后续生成状态激活。

## 2. 数据和索引

- 真实语料 token 数：9,999,872
- block 数：39,062
- 每个 block：256 tokens
- 合成向量：无
- 模型：Qwen3-0.6B
- 检索索引：Qwen 真实前向得到的 pre-RoPE K
- 原始 K 维度：128
- 粗检索维度：SVD32
- 粗检索候选：全局 Top-512
- full128 精排后工作集：Top-3 blocks，即 768 个检索 token

当前索引使用 4 个 Q/K 通道：

| Layer | Query head | KV head |
|---:|---:|---:|
| 3 | 10 | 5 |
| 21 | 8 | 4 |
| 6 | 7 | 3 |
| 16 | 14 | 7 |

## 3. 在线检索流程

1. 初始查询只使用问题文本产生的真实 pre-RoPE Q，不拼接该问题的完整 source。
2. 每张 GPU 扫描自己负责的索引 shard，在 SVD32 空间计算 late-interaction 分数。
3. 多卡汇总得到全局 Top-512 候选。
4. 对候选 block 使用原始 128 维 K 重新打分，选出 Top-3。
5. 根据 block ID 读取原始 token，将 3 个 block 重新送入 Qwen 全模型 prefill，计算所有层和头的完整 K/V cache。
6. 模型 greedy 生成 3 个 token，捕获这 3 个 token 在真实上下文状态下的 pre-RoPE Q。
7. 用新的 Q 重复全局 SVD32 检索、full128 精排和 Top-3 KV 重建，直到新生成 128 tokens 或遇到 EOS。

这里的“动态查询”不是把生成文本送入外部 embedding 模型，而是直接使用 Qwen 生成 token 的内部 Q 向量。

## 4. 三个对照

| 模式 | 初始信息 | 刷新方式 | 用途 |
|---|---|---|---|
| Static K3 | 只有问题 | 只检索一次 | 检查一次静态工作集是否足够 |
| Dynamic C3K3 | 只有问题 | 每 3 个新 token 全局检索 | 无泄漏端到端主实验 |
| Dynamic C3K3 + hop1 seed | 问题加正确第一跳句子 | 每 3 个新 token 全局检索 | 只诊断第二跳机制，不计作端到端成绩 |

第三组显式提供 `Lou Breslow's wife was Marion Byron.`，因此如果第三组成功而第二组失败，说明瓶颈主要是第一跳发现或生成；如果第三组也失败，则说明从生成态 Q 激活第二跳 block 的机制本身还不可靠。

## 5. 防止信息泄漏

旧的离线 profile 查询在问题 Q 前拼接过该问题完整 source，因此旧 `query_results.csv` 中第一跳 block 的排名不能当作无泄漏结果。本实验重新在线计算问题 Q，问题初始状态看不到 source、gold block ID 或答案文本。

gold block ID 只用于运行后评估排名轨迹，不参与候选生成和排序。

## 6. KV 实现边界

10M 持久索引当前保存了用于检索的 SVD K 和 raw K，但没有保存 V。因此本实验命中 block 后通过原 token 重新前向，得到全模型 K/V。这验证的是“10M 全局 K 检索 + 小工作集完整 KV 重建 + 动态生成”闭环。

它还不是“从 10M 持久 KV 数据库直接加载完整 K/V”。后者需要额外 profile 并存储 V，或实现与位置变换兼容的 KV 重定位。

## 7. 结果

### 7.1 主实验：纯 full128 精排 K3

主实验使用 3 张空闲 GPU，C3 表示每生成 3 tokens 刷新一次。

| 模式 | Answer@128 | 第一跳进入 K3 | 答案进入 Top-512 | 答案进入 K3 | 生成 tokens | 平均检索耗时 |
|---|---:|---:|---:|---:|---:|---:|
| Static K3 | 否 | 是 | 否 | 否 | 48 | 首次冷启动 580.1 ms |
| Dynamic C3K3 | 否 | 是 | 是，最好粗排 384 | 否，最好精排 276 | 55 | 58.6 ms |
| C3K3 + hop1 seed | 否 | 是 | 是，最好粗排 2 | 否，最好精排 24 | 6 | 58.6 ms |

输出分别是：

- Static：`Final answer: Lou Breslow was born in the United States.`
- Dynamic C3K3：`Final answer: The wife of Lou Breslow was born in the United States.`
- hop1 seed：`Final answer: Marion Byron.`

无泄漏问题 Q 的初始 SVD32 粗排把第一跳 block 20088 排在 132，但 raw128 精排将其提升到第 1，因此第一跳被放入 K3。这证明当前全局系统不是依赖 source 内扫描，并且能从 39,062 blocks 中找到第一跳。

第二跳没有通过纯 full128 K3。尤其在 hop1 seed 组中，答案 block 20096 已经达到 SVD32 粗排第 2，却被 raw128 精排降到第 24。这里出现了明确的粗排与精排目标不一致。

### 7.2 C1 不能单独解决

将刷新间隔从 3 tokens 改成 1 token，使用 4 张 GPU，其余保持纯 full128 K3：

| 模式 | Answer@128 | 答案最好粗排 | 答案最好精排 | 答案进入 K3 |
|---|---:|---:|---:|---:|
| Dynamic C1K3 | 否 | 未进入候选 | 未进入候选 | 否 |
| C1K3 + hop1 seed | 否 | 1 | 9 | 否 |

C1 无泄漏输出错误的 `New York`。更频繁刷新改变了生成和检索轨迹，但没有提高可靠性，反而累计选中过 160 个不同 block，说明只使用最新局部 Q 容易发生检索漂移。

### 7.3 K3 风险保留策略

为避免 raw128 精排完全删除粗空间强信号，增加一个仍然只有 3 blocks 的诊断策略：保留 SVD32 粗排前 2 个 block，最后 1 个位置使用 raw128 精排。

| 模式 | Answer@128 | 答案进入 K3 | 首次进入时机 | 最终输出 |
|---|---:|---:|---:|---|
| C1K3，2 coarse + 1 exact | 否 | 是 | 第 29 个新 token | `Marion Byron was born in New York City.` |
| C1K3 + hop1 seed，2 coarse + 1 exact | 否 | 是 | 第 5 个新 token | `Final answer: Marion Byron.` |

这组结果非常重要：10M 检索器确实能够将答案 block 20096 放入当前 3-block 工作集，但加载发生在模型已经开始或完成错误答案之后。重建 KV cache 不会自动让自回归模型撤销已经生成的错误 token。

因此，当前失败不能再简单表述为“全局检索找不到答案”；更准确的结论是“第二跳信号不稳定，而且检索成功时通常晚于答案承诺”。

### 7.4 Anchor 和 hop2 probe 诊断

将初始问题 Q 永久拼到动态 Q 后，问题 Q 在当前 max 聚合中长期压过生成态实体信号，答案 block 反而没有进入候选。该 anchor 设计失败。

又测试了人工第二跳状态 `Marion Byron was born in`，分别保留 3 个和 8 个尾部 Q。两者都没有从全局索引召回答案，模型继续生成 `New York City`。这说明当前 4 个通道的 max late-interaction 并不能稳定表示“实体 + 关系”的组合；扩大窗口本身不能解决组合语义被常见人物/出生文本干扰的问题。

### 7.5 多卡检索耗时

不同诊断运行恰好使用了当时空闲的不同 GPU 数。以下为单查询动态检索的近似稳态均值，包含全局 SVD32 扫描、Top-512 汇总和 raw128 候选精排，不包含首次索引冷启动：

| GPU 数 | 近似稳态耗时/次 | 相对 3 卡 |
|---:|---:|---:|
| 3 | 58.6 ms | 1.00x |
| 4 | 52.9 ms | 1.11x |
| 6 | 39–40 ms | 约 1.48x |
| 7 | 39.3 ms | 约 1.49x |

从 6 卡到 7 卡已经基本平台化，单查询下 collective、候选精排和 Python 调度开始主导。该数字来自不同策略运行，不是严格的同配置 scaling benchmark，因此只作为在线延迟量级，不能替代独立多卡加速实验。

## 8. 结论

这次已经完成了真正的 10M 全局闭环，而不是 source 内精确扫描：

1. 初始问题 Q 不知道 source，仍能从 39,062 blocks 找到第一跳 block。
2. 每次动态刷新都扫描完整 10M SVD32 分布式索引，再用 raw128 K 精排候选。
3. 选中 block 的原 token 被重新 prefill，生成使用全模型完整 K/V。
4. 答案 block 在风险保留版中确实进入过 K3，但最终答案仍错误。

所以目前不能宣称该方法已经解决两跳出生地点问题。它证明了“10M 全局检索 + 动态小工作集”工程闭环可运行，也定位出三个算法瓶颈：

1. raw128 最大点积精排会删除 SVD 空间中的强第二跳候选。
2. 仅用最近 1–3 个生成 Q 容易漂移，简单拼接固定问题 Q 又会压制动态信号。
3. 自回归模型在证据到达前已经承诺错误地点，后到 KV 无法修改历史 token。

下一版不应继续只调 K 或窗口，而应实现“检索后再承诺”的控制器：先生成第一跳实体到草稿状态，在输出地点前构造独立的实体-关系检索 query；对粗排和精排做可学习融合或 rank fusion；确认第二跳证据进入工作集后，再恢复答案 token 的生成。

## 9. 产物

- 主实现：`src/run_global_dynamic_svd_kv_single.py`
- 服务器启动器：`scripts/run_global_dynamic_svd_kv_single_server.sh`
- 纯 full128 C3：`outputs/global_dynamic_svd_kv_q0_v1/result.json`
- 纯 full128 C1：`outputs/global_dynamic_svd_kv_q0_c1_v1/result.json`
- C1 风险保留：`outputs/global_dynamic_svd_kv_q0_reserve2_c1_v1/result.json`
- Anchor 诊断：`outputs/global_dynamic_svd_kv_q0_anchor_reserve2_c3_v1/result.json`
- hop2 probe：`outputs/global_dynamic_svd_kv_q0_hop2probe_c3_v1/result.json`
- hop2 probe Q8：`outputs/global_dynamic_svd_kv_q0_hop2probe_q8_c3_v1/result.json`
