# RoPE 长程检索方法搜索报告

**模型：** Qwen3-8B  
**状态日期：** 2026-08-01  
**结论口径：** 只使用已落盘、可复核的实验结果；正在运行或仅完成实现的实验均标为 TODO。

> **最终审计更新：** 本文件早期章节保留了方法搜索过程；其中“strict sparse phase rescue conditional-GO”等中间判断已被后续 formal safety、token-sparse MPR、sink-free Query-span、singleton replay 与 KVQ-R 实验否定。当前最终结论以 [`final_method_verdict_20260801.md`](final_method_verdict_20260801.md) 为准：新 PE / frozen-inference phase repair 为 **NO-GO**，论文建议转为跨层因果机制研究。

## 1. Executive conclusion

目前最稳定的有效结果仍然是：**用 pre-RoPE 内容分数召回远程候选，再用原生 post-RoPE 分数消费候选**。在 16K、32K 和 64K 的受控两跳检索中，它明显优于 full attention 和 exact post-RoPE Top-2%，说明 RoPE 确实会把一部分语义相关的远程证据压出原生 Top-2%。

但是，现有实验**尚未证明直接修改远程 RoPE 相位可以稳定提升生成质量**：

- 静态远程 NoPE、距离封顶、距离压缩和频率渐退缺少跨长度稳定性；
- 未校准 phase kernel 会产生灾难性退化；
- blockwise phase transport 和 Native Phase Envelope（NPE）虽然能提高部分 QK 分数、evidence recall 或 attention mass，却使 PPL 和准确率变差；
- support-matched Minimal Phase Rescue（MPR）只有很小、长度间不稳定的收益，尚不足以成为论文方法。

因此，当前结论不是“新的 RoPE repair 已成立”，而是：

> **pre-RoPE 语义召回有效；宽泛的远程相位修改已被实验否定。唯一仍值得验证的方向，是冻结模型上由真实 suppression event 触发、严格稀疏、严格 no-op、使用冻结干预计划的最小相位修复。**

## 2. 创新边界

文献审计表明，下列宽泛主张不应作为 ICLR 2027 的核心创新：

1. 远程 token 不使用或削弱 RoPE；
2. 近程使用 RoPE、远程使用 NoPE；
3. 距离封顶、分桶、压缩或 chunk-wise position remapping；
4. pre-RoPE 与 post-RoPE 分数的静态或距离依赖融合；
5. 训练或校准一个 content/head/frequency-dependent phase gate；
6. proposal 与 native-RoPE consumption 分离；
7. 对同一 KV 使用多个 RoPE phase/anchor 后进行 LME 或概率混合。
8. 仅把 attention 与 Value norm、projected Value、输出扰动或答案 unembedding 方向结合。

其中，多相位边缘化与 NeurIPS 2024 的 MoICE 高度重合。把 mixture-of-softmax 改成 softmax-of-LME 不足以形成独立方法贡献，因此裸 multi-phase marginalization 为 **NO-GO**。

同样，generic value/output-aware 叙事也已被 VATP、CriticalKV、LaProx 和 LOCOS 覆盖；attention gradient attribution 本身也不是新工具。若后续 value-mediated probe 成立，可守贡献必须限定为 **RoPE 频率相位抑制经过 softmax 与 Value 写入影响最终答案 margin 的逐项因果闭环**，而不是“首次发现 attention mass 不等于 token 重要性”。

尚可能形成可守创新的窄交集是：

- 模型权重完全冻结，无训练、无校准语料；
- 触发条件来自同一 Query--Key 的 pre/post-RoPE 反事实抑制，而非单纯距离；
- 只改极少数频率对、head 或证据 token，并显式限制相位位移；
- 未触发时与原模型逐元素一致；
- 干预位置和相位计划冻结，避免后续 Query 漂移造成自适应重选混淆；
- 建立 QK、attention mass、residual 写入、答案 logit/PPL 的因果闭环；
- 在冲突证据和局部顺序任务上通过安全性验证。

## 3. 实验统一口径

- **模型：** Qwen3-8B，NF4 权重，BF16 计算；已完成主实验使用服务器 GPU 6--7。
- **数据：** 受控英文两跳证据检索，远程证据位于前缀，Query 位于末尾。
- **干预范围：** 冻结模型和 prefix KV，只修改最终 Query 的 attention；这不是训练后完整位置编码实验。
- **稀疏预算：** 约 2%。
- **PPL：** 在匹配 seeds 上先平均 gold NLL，再计算 `exp(mean NLL)`。
- **Acc：** 首答案 token 正确率，不是多 token 自由生成准确率。
- **Recall：** 被保留的 gold evidence token 比例。
- **Both：** 两条 gold evidence line 均有 token 命中的平均比例。
- **Mass：** gold evidence 获得的平均 attention mass。
- `full_rope` 的 Recall/Both 恒为 100%，只表示所有 token 都被保留，不代表模型成功使用了证据。
- 除 blockwise transport 为 4 seeds 外，主要已完成表格均为 8 seeds。

## 4. 结果矩阵

### 4.1 共同基线

| 长度 | 方法 | PPL | Acc | Recall | Both | Mass |
|---:|---|---:|---:|---:|---:|---:|
| 8K | Full attention | 2.600 | 62.5% | 100.0% | 100.0% | 5.265% |
| 8K | Exact post-RoPE Top-2% | 2.377 | 62.5% | 32.2% | 64.1% | 5.193% |
| 8K | Exact pre-RoPE Top-2% + native consumer | 2.334 | 62.5% | 14.2% | 55.8% | 5.391% |
| 8K | Local + pre-RoPE retrieval | 2.339 | 50.0% | 14.2% | 55.9% | 5.433% |
| 16K | Full attention | 7.509 | 37.5% | 100.0% | 100.0% | 5.228% |
| 16K | Exact post-RoPE Top-2% | 3.139 | 62.5% | 35.7% | 59.9% | 5.058% |
| 16K | Exact pre-RoPE Top-2% + native consumer | 2.566 | 75.0% | 41.0% | 78.6% | 5.256% |
| 16K | Local + pre-RoPE retrieval | 2.601 | 62.5% | 41.1% | 78.3% | 5.265% |
| 32K | Full attention | 14.607 | 12.5% | 100.0% | 100.0% | 4.992% |
| 32K | Exact post-RoPE Top-2% | 10.472 | 37.5% | 39.3% | 60.5% | 4.788% |
| 32K | Exact pre-RoPE Top-2% + native consumer | 3.958 | 37.5% | 47.2% | 80.4% | 5.097% |
| 32K | Local + pre-RoPE retrieval | 3.847 | 62.5% | 47.2% | 80.1% | 5.092% |
| 64K | Full attention | 8.362 | 25.0% | 100.0% | 100.0% | 4.618% |
| 64K | Exact post-RoPE Top-2% | 6.494 | 50.0% | 35.4% | 54.8% | 3.897% |
| 64K | Exact pre-RoPE Top-2% + native consumer | 4.449 | 50.0% | 44.8% | 78.0% | 4.012% |

pre-RoPE retrieval 是目前最稳定的有效组件，但 proposal--consumption separation 已有正式先例；它适合作为强 baseline 和系统组件，不能单独支撑“新 RoPE”主张。

### 4.2 静态远程 RoPE 修改

下表单元格依次为 `PPL / Acc / Recall / Mass`；baseline 均为同长度 `full_rope`。

| 变体 | 8K | 32K | 判断 |
|---|---|---|---|
| `remote_nope_cal_full` | 5.068 / 37.5 / 100 / 5.342 | 11.057 / 37.5 / 100 / 3.320 | 不稳定；NO-GO |
| `distance_fade_4k_full` | 2.913 / 62.5 / 100 / 5.215 | 6.182 / 50.0 / 100 / 3.666 | 32K 的 95% CI 跨零，且 8K 退化 |
| `distance_fade_8k_full` | 2.435 / 75.0 / 100 / 5.572 | 8.229 / 50.0 / 100 / 4.106 | 长度间不稳定 |
| `distance_fade_16k_full` | 2.440 / 62.5 / 100 / 5.341 | 7.387 / 37.5 / 100 / 4.850 | 长度间不稳定 |
| `phase_coherent_w4k_c4_cal_full` | 2.799 / 75.0 / 100 / 5.395 | 18.417 / 12.5 / 100 / 5.227 | 32K 退化 |
| `phase_coherent_w8k_c4_cal_full` | 3.018 / 50.0 / 100 / 5.304 | 7.831 / 25.0 / 100 / 4.758 | 8K 显著退化 |
| `phase_coherent_norm_w4k_c4_cal_full` | 2.547 / 75.0 / 100 / 5.348 | 10.567 / 12.5 / 100 / 4.818 | 无稳定优势 |
| 未校准 phase-coherent 系列 | PPL 35--1376；Acc 0--12.5% | PPL 353--1063；Acc 0% | 灾难性尺度失配；NO-GO |
| `distance_saturate_w4k_t4k_full` | 4.742 / 37.5 / 100 / 5.244 | 19.611 / 12.5 / 100 / 4.663 | NO-GO |
| `distance_saturate_w4k_t16k_full` | 2.730 / 62.5 / 100 / 5.050 | 16.759 / 25.0 / 100 / 4.481 | NO-GO |
| `distance_log_w4k_t4k_full` | 5.037 / 37.5 / 100 / 5.084 | 60.441 / 12.5 / 100 / 4.807 | NO-GO |

### 4.3 Counterfactual selection

| 长度 | 方法 | PPL | Acc | Recall | Both | Mass |
|---:|---|---:|---:|---:|---:|---:|
| 8K | `cfs_w4k_lift50_postscore` | 2.661 | 62.5% | 12.2% | 46.4% | 5.385% |
| 32K | `cfs_w128_lift100_postscore` | 6.599 | 37.5% | 46.8% | 81.3% | 4.772% |

32K 时 CFS 描述性优于 post-RoPE Top-2% 的 PPL 10.472，但仍明显弱于更简单的 exact pre-RoPE selector（PPL 3.958）；8K 也无优势。当前版本 **NO-GO**。

### 4.4 Support-matched Minimal Phase Rescue

这一实验固定 exact pre-RoPE Top-2% 的候选 support，只修改候选消费分数，是当前最接近 phase repair 因果消融的结果。

| 长度 | 变体 | PPL | Acc | Recall | Mass | $\Delta$NLL vs exact-pre [95% CI] |
|---:|---|---:|---:|---:|---:|---:|
| 8K | exact-pre baseline | 2.334 | 62.5% | 14.2% | 5.391% | 0 |
| 8K | `mpr_pre_w4k_lift25` | 2.191 | 62.5% | 14.2% | 5.492% | -0.064 [-0.138, +0.014] |
| 8K | `mpr_pre_w4k_lift25_masspreserve` | 2.183 | 50.0% | 14.1% | 5.414% | -0.067 [-0.131, -0.001] |
| 16K | exact-pre baseline | 2.566 | 75.0% | 41.0% | 5.256% | 0 |
| 16K | `mpr_pre_w4k_lift25` | 2.508 | 75.0% | 41.1% | 5.334% | -0.023 [-0.113, +0.061] |
| 32K | exact-pre baseline | 3.958 | 37.5% | 47.2% | 5.097% | 0 |
| 32K | `mpr_pre_w4k_lift25` | 3.806 | 50.0% | 47.2% | 5.224% | -0.039 [-0.224, +0.103] |
| 32K | `mpr_pre_w4k_lift25_gap1` | 3.671 | 50.0% | 47.3% | 5.221% | -0.075 [-0.186, +0.029] |

只有 8K mass-preserving 变体的 CI 极窄地低于零，但 Acc 从 62.5% 降至 50%；16K 和 32K 均跨零。普通 MPR 还会在约 52%--66% 的远程候选上触发并修改全部 64 个频率对，不符合“稀少事件、最小干预”。

结论：当前 MPR 尚不成立；严格稀疏 trust-region 版本仍是 **conditional-GO/TODO**。

严格 frozen-reference / 8-frequency-pair 版本现已完成一个 8K、seed 0 smoke：exact pre-RoPE baseline PPL 4.678，定向 Strict MPR 为 3.422，匹配 plane count 与 shift-$L_2$ 的随机频率对照为 5.002。support mismatch、非触发 no-op 和 random matching error 均为 0 或数值零。然而所有 arm 的首 token 都错误；原始 trigger 比例仍为 51.64%，共调用 solver 12,493 次，两个定向 arm 各耗时约 238 秒；random + partition-preserve 也得到 3.642 PPL。该结果只说明“值得继续做更严格消融”，不能证明方法成立。正式扩大前先改为每层每 head 最多 top-1 / top-4 token 的双重稀疏版本。

### 4.5 Blockwise coherent phase transport

32K、4 seeds，baseline 为同 seeds 的 `rope_top2`。

| 方法 | PPL | Acc | Recall | Both | Mass |
|---|---:|---:|---:|---:|---:|
| `rope_top2` | 7.129 | 50.0% | 39.1% | 60.3% | 4.632% |
| `block16_selector_only` | 8.102 | 25.0% | 66.6% | 71.5% | 5.790% |
| `block16_transport` | 19.427 | 25.0% | 67.7% | 72.6% | 6.490% |
| `block16_transport_masspreserve` | 16.072 | 25.0% | 66.8% | 71.2% | 5.955% |
| `block16_random_matched` | 24.034 | 25.0% | 67.9% | 72.7% | 6.343% |
| `block32_selector_only` | 14.326 | 25.0% | 67.6% | 70.9% | 6.103% |
| `block32_transport` | 28.667 | 25.0% | 69.0% | 72.6% | 6.938% |
| `block32_transport_masspreserve` | 46.982 | 25.0% | 67.7% | 70.9% | 6.245% |
| `block32_random_matched` | 51.268 | 25.0% | 68.9% | 72.5% | 6.607% |

块内相对距离误差严格为 0，但 transport 仍恶化 PPL。block16 中，Mass 从 5.790% 增至 6.490%，PPL 却从 8.102 恶化到 19.427，直接说明“保持块内顺序并增加 evidence mass”仍不足以恢复答案。

8K 的 block32 在扣除 local window、sink 和当前 token 后放不下一个完整远程块，Recall=0、PPL 约 590K；这是预算退化，不作为方法效果使用。整体 **NO-GO**。

### 4.6 Native Phase Envelope rollback

baseline 为 exact pre-RoPE Top-2% + native post-RoPE consumer；`npe_native_pre_top2` 的 no-op 最大误差为 0。

| 长度 | 方法 | PPL | Acc | Recall | Mass | $\Delta$NLL [95% CI] |
|---:|---|---:|---:|---:|---:|---:|
| 8K | native | 2.334 | 62.5% | 14.2% | 5.391% | 0 |
| 8K | rollback | 3.044 | 62.5% | 13.8% | 5.949% | +0.265 [+0.034, +0.595] |
| 8K | rollback + mass preserve | 2.496 | 50.0% | 14.1% | 5.470% | +0.067 [-0.060, +0.195] |
| 8K | random matched | 2.950 | 50.0% | 13.9% | 5.233% | +0.234 [+0.058, +0.423] |
| 32K | native | 3.958 | 37.5% | 47.2% | 5.097% | 0 |
| 32K | rollback | 4.949 | 12.5% | 46.6% | 4.972% | +0.223 [-0.497, +0.765] |
| 32K | rollback + mass preserve | 4.445 | 25.0% | 47.2% | 5.129% | +0.116 [+0.015, +0.219] |
| 32K | random matched | 4.905 | 37.5% | 47.2% | 4.509% | +0.215 [-0.289, +0.769] |
| 64K | native | 4.449 | 50.0% | 44.8% | 4.012% | 0 |
| 64K | rollback | 21.069 | 0.0% | 44.9% | 3.503% | +1.555 [+0.781, +2.238] |
| 64K | rollback + mass preserve | 5.848 | 25.0% | 44.9% | 3.973% | +0.274 [-0.022, +0.632] |
| 64K | random matched | 11.105 | 12.5% | 45.3% | 3.539% | +0.915 [+0.627, +1.198] |

certificate 在 native baseline 中已触发 55.1%、61.4%、69.1%；rollback 平均移动约 3.4K、10.0K、18.9K token，不属于最小修复。提高 phase score、Recall 或 Mass 均未转化成稳定的答案收益，mass preservation 也不能消除退化。当前 adaptive NPE **明确 NO-GO**。

## 5. 已证伪方向

| 命题 | 实验证据 | 结论 |
|---|---|---|
| 远程统一关闭或削弱 RoPE 即可改善检索 | Remote NoPE、distance saturation/log 和静态 phase kernel 跨长度不稳定或退化 | 已证伪 |
| 提高 evidence recall 就足够 | Block transport 将 Recall 提高到约 67%--69%，PPL 反而恶化 | 已证伪 |
| 提高 evidence attention mass 就足够 | 8K NPE rollback：Mass 5.391%→5.949%，PPL 2.334→3.044 | 已证伪 |
| 保持远程 softmax 分区总质量就足够 | MPR/NPE mass-preserve 没有稳定收益 | 已证伪 |
| 保持块内局部顺序后可以安全搬运远程块 | 块内距离误差为 0，transport 仍显著退化 | 已证伪 |
| 当前 suppression gap 是稀少、可靠的事件证书 | NPE 对 55%--69% 候选触发 | 已证伪 |
| 多 phase/LME 本身具有足够方法新颖性 | MoICE 已覆盖同 KV 多 RoPE angle 聚合的核心结构 | 创新性 NO-GO |

这些结果同时说明：QK 分数、Recall 和 attention mass 都只是中间变量。修复必须进一步证明被增强的 Value 写入方向对答案 logit 有正贡献，而且不会同时增强冲突证据或无关竞争 token。

## 6. 当前唯一 conditional-GO 路线

下一步只保留 **strict sparse counterfactual phase rescue**：

1. 使用 exact pre-RoPE Top-2% 固定候选 support；
2. 在未修改 baseline trajectory 上冻结每层、每 head 的触发位置和干预计划；
3. 只有当 suppression gap 超过 BF16 重构误差保护阈值时触发；
4. 最多修改 8 个频率对，相位绝对变化不超过 0.25 rad；
5. 显式最小化相位位移，并与 random-support、mass-preserve 控制严格匹配；
6. 非触发位置逐元素 no-op；
7. 同时报告 QK lift、evidence mass、residual/logit 变化、Gold PPL、Acc；
8. 必须通过 gold/conflict safety 和 local-order counterfactual control。

只有满足以下门槛才升级为 GO：

- 在独立 seeds 和至少 16K/32K/64K 三个长度上稳定优于 exact pre-RoPE selector；
- paired NLL 置信区间稳定低于 0，且 Acc 不下降；
- targeted intervention 明显优于等预算 random control；
- gold repair 改善 gold-vs-conflict margin，而 conflict repair 不会同等获益；
- 局部顺序、否定、最近指代等任务无可测退化；
- 至少在第二个模型和自然长上下文 benchmark 上复现。

若这些条件不能满足，应停止把 phase repair 作为论文主方法，转为“RoPE 长程抑制机制诊断 + pre-RoPE retrieval repair”的论文路线。

## 7. 未完成项

| 项目 | 当前状态 | 报告规则 |
|---|---|---|
| Strict sparse MPR：8 个频率对、phase cap 0.25、frozen support、random/mass controls | 8K seed-0 smoke 已完成；PPL 有正信号但 Acc 未恢复，且 51.64% remote token 触发 | 只作为 provisional；正在实现每 head top-1/top-4 token cap 后再扩大 |
| Frozen-reference NPE rollback/random controls | 已实现并通过单元测试；尚无正式 artifact | TODO |
| Suppression-certificate safety：8 seeds × 8K/32K/64K | **正式实验正在重跑** | 等重跑完成后再报告 |
| Value-mediated RoPE probe：$\partial m/\partial s_j$、phase suppression × downstream sensitivity、matched intervention | runner/protocol 正在实现；尚无 artifact | 仅作为 RoPE-specific 因果诊断；不得声称通用 value-aware attribution 新颖 |
| 旧 safety smoke | 仅 8K seed 0，且 prompt 的答案 token 边界/目标格式错误 | **无效，不得引用其 PPL、Acc 或 AUROC 作为论文结果** |
| Local-order counterfactual control | protocol、runner 和 launcher 已完成；尚无实验 artifact | TODO |
| MPR 64K | 未运行 | TODO |
| CFS、block transport 64K | 未运行 | TODO |
| 多模型、自然长上下文、不同 needle/conflict 类型 | 未运行 | TODO |
| 完整多 token decoding 与训练时位置编码验证 | 未运行 | TODO |

在上述 TODO 完成前，论文中最多可以表述为“我们发现了一个可测量的 RoPE 长程抑制机制，并验证 pre-RoPE retrieval 能恢复部分被压低的证据”，不能表述为“我们已经提出并验证了一个成熟的新 RoPE”。
