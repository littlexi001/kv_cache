# RiskKV-Block 通用续写控制器 v2：PG19 与因果复述检索

日期：2026-07-14

## 1. 本轮要解决的问题

原来的 RiskKV-Block 主要面向有显式问题的 QA：用问题定位相关 block，再保留对应 KV。普通自由续写没有显式问题，把最后 256 个 token 直接当作 query 会产生两个问题：

1. 语义相关不等于下一段真正需要的局部连续状态；
2. query 之后才出现的历史复述无法被一次性静态检索提前知道。

因此，本轮没有继续微调原 block scorer，而是把通用方案拆成两个前端、一个共享 KV 后端：

- query-aware 前端：继续使用语义 block router，服务 QA、证据定位和结构化检索；
- queryless 前端：使用 local continuity、semantic prior 和在线 recurrence detector；
- 共享后端：均从同一上下文的预计算 KV page 中 gather，不访问外部文档，不属于 RAG。

## 2. 冻结方法

### 2.1 初始 2K 动作

对 32K 历史，remote KV 预算为 2,048 token，另回放最后 256 个可见 token，总 KV 为：

\[
B_{total}=B_{remote}+B_{query}=2048+256=2304,
\]

即原始 32K 历史的 7.20%。remote 预算分配为：

\[
B_{remote}=B_{sink}+B_{local}+B_{semantic}=32+1536+480.
\]

- `sink=32`：保留最前部稳定 token；
- `local=1536`：连续保留最近上下文，保护语法、文体和短程状态；
- `semantic=480`：使用现有 block scorer 保留少量远程主题信息。

### 2.2 在线 recurrence detector

对 remote history 建立 8-token rolling hash 索引。生成第 \(t\) 个 token 前，只使用已经可见的 query 和生成前缀：

\[
s_t=(x_{t-7},\ldots,x_t).
\]

若 \(s_t\) 在历史中命中位置 \(p_t\)，还不能立即重建，因为普通短语可能偶然相同。v2 要求连续三个解码步满足：

\[
p_t=p_{t-1}+1,\qquad p_{t-1}=p_{t-2}+1.
\]

只有历史源位置连续前进，才把它视为真实复述。该门控会带来 2-token 检测延迟，但能过滤源位置跳变的短公共片段。

### 2.3 recurrence 动作

若确认的匹配已经在 local/semantic KV 中，则不重建。若匹配位于远程历史，则把 480-token semantic 槽替换成从匹配源开始的连续 echo KV：

\[
B_{remote}=32\;\text{sink}+1536\;\text{local}+480\;\text{echo}=2048.
\]

总预算不增加。一个 echo episode 只在首次确认时重建一次，后续连续匹配复用同一批 KV page。

### 2.4 与 recent fallback 的关系

早期版本把“距离较近的复述”路由到 2.8K pure recent。实测表明该规则错误：医学窗口 0 中，2.8K recent PPL 为 15.310，而 2K echo 为 14.828。原因是 echo 精确保留源片段及其 continuation，而 pure recent 会同时引入大量无关间隔。

最终 v2 将两个信号正交化：

- 已确认复述：始终使用 2K recurrence echo；
- 独立 continuity-risk 信号：才允许升级到 2.8K/4K recent；
- 当前冻结版本尚未启用未经独立验证的 continuity-risk 升级。

## 3. 严格因果协议

- 模型：Llama-3.1-8B-Instruct；
- 历史长度：32,000 token；
- 每个窗口预测 256 个 target token；
- block size：16；
- 稀疏 KV 使用原始逻辑位置、物理 causal mask 和 LPCM；
- detector 在预测当前 token 前只能读取已经观察到的前缀；
- full、静态和 controller 均保留逐 token 公平时延路径；
- paper-test 未被读取、未参与阈值选择。

## 4. Targeted 体育与医学结果

共 6 个窗口，专门覆盖之前最差的主题。PPL 越低越好。

| 方法 | KV ratio | PPL | PPL / Full | cache rebuild |
|---|---:|---:|---:|---:|
| Full KV | 100.00% | 8.3941 | 1.0000 | 0 |
| 2K tokenwise static | 7.20% | 11.2667 | 1.3422 | 0 |
| 2K causal echo | 7.20% | 9.8496 | 1.1734 | 2 |
| 2K universal controller v2 | 7.20% | **9.8496** | **1.1734** | **2** |

分主题结果：

| 主题 | Full PPL | Static PPL / Full | v2 PPL / Full |
|---|---:|---:|---:|
| 体育 | 8.0679 | 10.9062 / 1.3518 | **9.3530 / 1.1593** |
| 医学 | 8.7336 | 11.6392 / 1.3327 | **10.3726 / 1.1877** |

最差窗口变化：

- 体育窗口 0：18.168 -> 11.459，Full 为 8.790；
- 医学窗口 0：20.951 -> 14.828，Full 为 10.224；
- 其他四个窗口没有远程稳定复述，controller 与静态路径一致。

## 5. PG19 单文档独立验证

20 Newsgroups 是多帖子拼接流，可能放大签名和模板复述。为排除这种数据伪影，新增 PG19 test Parquet 入口；每个 32K 历史和 256-token target 均来自同一本完整书。

6 本有效书、1,536 个 target token 的聚合结果：

| 方法 | KV ratio | PPL | PPL / Full | cache rebuild |
|---|---:|---:|---:|---:|
| Full KV | 100.00% | 13.8518 | 1.0000 | 0 |
| Sink + recent 2K | 7.20% | 14.7345 | 1.0637 | 0 |
| 2K tokenwise static | 7.20% | 14.8767 | 1.0740 | 0 |
| 2K causal echo | 7.20% | 14.8767 | 1.0740 | 0 |
| 2K universal controller v2 | 7.20% | **14.8767** | **1.0740** | **0** |

结论：在普通连续长篇文本中，v2 没有误触发，也不会凭空改变静态结果。7.2% KV 下相对 Full 的 PPL 增幅为 7.4%，说明通用自由续写已经明显好于最初六主题实验，但尚未稳定达到预设的 `PPL/Full <= 1.05`。

## 6. 门控消融

| 门控 | Targeted PPL / Full | PG19 book 0 | 结论 |
|---|---:|---:|---|
| 8-token + 连续 3 步 | **1.1734** | 无重建，PPL 8.734 | 冻结版本 |
| 10-token + 连续 1 步 | 1.1726 | 2 次误重建，PPL 8.751 | 质量略高但不安全 |
| 8-token + 16-token 后向确认 | 1.2135 | 无重建，PPL 8.734 | 安全但触发过晚 |

PG19 假匹配的 source trajectory 在多个位置间跳变，最长只有 2 个连续步；体育真实复述的历史位置持续按 `+1` 前进。该轨迹差异解释了为什么连续 3 步比简单增加固定 match length 更合适。

## 7. 在线开销

在此前 18-window 统一逐 token harness 中：

- tokenwise static：平均 7.7625 秒；
- causal echo：平均 7.8025 秒；
- 增量开销：0.515%；
- 18 个窗口仅 4 次远程重建；
- 平均 hash lookup 0.000776 秒、KV gather 0.008468 秒、suffix replay 0.167131 秒、decode 7.626101 秒。

本轮 targeted v2 中，controller 平均 7.855 秒，tokenwise static 平均 7.977 秒；差异落在运行噪声内，没有观察到可分辨的额外在线开销。

注意：表中的 full PPL 路径使用 chunked teacher forcing，不能与逐 token controller 的秒数直接计算生成加速。attention 理论上界仍约为 `1 / 0.072 = 13.89x`；真正端到端速度需要使用统一自回归 decode harness。

## 8. 当前结论与下一步

本轮得到的可部署结论不是“一个 semantic router 适合所有任务”，而是：

1. query-aware 和 queryless 必须采用不同前端信号；
2. local、semantic、recurrence 是三种功能不同的 KV memory；
3. recurrence 应由生成时可见的稳定轨迹触发，不能由静态 query 猜测；
4. recurrence 与预算升级必须解耦；
5. 当前 2K v2 已解决大部分重复型灾难窗口，但极端通用 PPL 仍需要安全预算或更强连续状态压缩。

安全预算实验已经完成：

| Remote 预算 | 32K 总 KV ratio | 体育 PPL/Full | 医学 PPL/Full | 聚合 PPL/Full |
|---:|---:|---:|---:|---:|
| 2K | 7.20% | 1.1593 | 1.1877 | 1.1734 |
| 4K | 13.60% | 1.1412 | 1.0602 | 1.0999 |
| 8K | 26.40% | 1.1186 | **1.0342** | 1.0756 |
| 12K | 39.20% | 1.1148 | **1.0333** | 1.0733 |
| 16K | 52.00% | **1.0292** | **1.0239** | **1.0265** |

4K/8K 明显改善医学连续性，但体育 gap 下降很慢。12K 仍与 8K 基本相同，16K 才出现明显相变并达到 `PPL/Full <= 1.05`。体育真实复述源距 query 约 15,334 token：12K local 看不到它，16K local 恰好覆盖。复述在 target 117 才首次显现，任何严格因果的在线 detector 都不可能在前 117 个 token 预知它。

因此，在没有可跨主题泛化的未来源预测器时，要保证该最坏样本的 95% PPL，连续 local floor 至少约为 15.3K。16K 在 32K 上占 52%，attention token-ratio 上界仅约 1.92x，不能满足低 KV 主目标；它应被视为 queryless 最坏情况 fallback 或方法边界，而不是主动作。

8K 在 32K 上比例较高，但固定绝对预算在更长上下文中会下降：计入 256-token query 后，在 64K/128K 历史上分别约为 13.2%/6.6%，对应 attention token-ratio 上界约 7.6x/15.2x。该换算只是理论上界，不代表当前 kernel 的端到端实测速度。

为处理“复述开始之前不可见”的剩余误差，本轮继续验证了 history-side proactive motif memory：尝试在生成前从历史中主动选取高复现模板、实体签名和它们的 continuation，作为 queryless prior。

### 8.1 Proactive motif memory 可行性诊断

先对两个真实复述源做了只使用历史的 retrospective ranking，共 1,955 个 480-token 候选 span：

| 真实源 | rarity rank | history n-gram recurrence rank | template rank | 综合 motif rank |
|---|---:|---:|---:|---:|
| 体育窗口 0 | 1,538 | 771 | 481 | 1,197 |
| 医学窗口 0 | 1,390 | 146 | 182 | 397 |

真实源并不是简单的“历史中最稀有、重复最多或模板标记最多”的 span。480-token 预算只允许选择约 1 个连续候选，因此该启发式召回接近随机，不能直接变成主方法。

进一步使用 20 Newsgroups 全类别自动挖掘未来 16-token 复述标签，训练 history-only GBDT prior。训练时完整排除 `rec.sport.baseball`、`sci.med` 和 `comp.graphics` 三个测试类别：

| Split | Positive windows | Hit@1 | Hit@4 | Hit@8 | MRR |
|---|---:|---:|---:|---:|---:|
| Train | 15 | 93.3% | 100.0% | 100.0% | 0.967 |
| Dev | 3 | 0.0% | 33.3% | 33.3% | 0.175 |
| Held-out topics | 7 | **0.0%** | 14.3% | 28.6% | 0.081 |

该 prior 明显过拟合主题，说明仅依赖位置、词频、稀有度、局部熵和历史 n-gram 次数的小 router 不具备跨主题预测未来复述源的能力。下一版 proactive memory 必须使用模型状态或 attention-derived block representation，并在更大自监督语料上训练；继续组合手工词频特征没有足够证据。

## 9. 实现与结果文件

- 多主题/控制器入口：`ymluo/projects/qwen3_top2_head_limit3_ppl/src/run_causal_echo_ppl_20260714.py`
- PG19 入口：`ymluo/projects/qwen3_top2_head_limit3_ppl/src/run_pg19_causal_echo_ppl_20260714.py`
- 冻结控制器：`ymluo/projects/qwen3_top2_head_limit3_ppl/src/riskkv_universal_controller.py`
- PG19 并行脚本：`ymluo/projects/qwen3_top2_head_limit3_ppl/scripts/launch_pg19_causal_echo_20260714.sh`
- PG19 汇总脚本：`ymluo/projects/qwen3_top2_head_limit3_ppl/scripts/summarize_pg19_causal_echo_20260714.py`
- Motif 可行性诊断：`ymluo/projects/qwen3_top2_head_limit3_ppl/scripts/diagnose_proactive_motif_memory_20260714.py`
- 自监督 history prior：`ymluo/projects/qwen3_top2_head_limit3_ppl/scripts/train_history_recurrence_prior_20260714.py`
- 单元测试：`ymluo/projects/qwen3_top2_head_limit3_ppl/tests/test_multitopic_lpcm_ppl.py`、`test_pg19_causal_echo_ppl.py`、`test_riskkv_universal_controller.py`、`test_proactive_motif_memory.py`、`test_history_recurrence_prior.py`
- 本地原始结果：`ymluo/projects/qwen3_top2_head_limit3_ppl/results/20260714_pg19_echo/`
