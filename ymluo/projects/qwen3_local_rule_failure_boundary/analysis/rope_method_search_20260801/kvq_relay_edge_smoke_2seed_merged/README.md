# KVQ-R relay-edge smoke

## 结论

**NO-GO；不扩跑。** Qwen3-8B、8K、mixed 两跳数据的两个独立 seeds 均通过 prefix-KV 完整 SHA-256 不变性检查，且 label-free Top-64 同时召回两条 gold evidence；但两个 case 都未通过有限差分审计，并且即使把无效 case 的分数仅作描述性检查，KVQ-R 也没有优于更简单的无 Value control。

| Seed | Gold 两跳候选覆盖 | FD pass | KVQ-R AUROC | Shuffled-V | Random-V | Reverse edge | K-K | Pre-score pair |
|---:|:---:|:---:|---:|---:|---:|---:|---:|---:|
| 0 | ✓ | ✗ | 1.000 | 0.469 | 0.625 | 0.969 | 0.219 | **1.000** |
| 1 | ✓ | ✗ | 0.750 | 0.250 | 0.125 | **0.813** | 0.313 | **1.000** |

由于 `finite_difference_audit_pass=0`，正式汇总正确地将两例都排除，valid-case AUROC 为 `null`；上表 AUROC 只用于排错，不能作为方法效果。

## 有限差分为何失败

baseline Query 重构最大误差为 0，prefix KV 也完全不变。失败集中在第 27 层的少量 source blocks：

| Seed | Layer | 通过 block 数 | Relative error 最大值 | Cosine 最小值 |
|---:|---:|---:|---:|---:|
| 0 | 9 / 18 / 27 / 34 | 64 / 64 / **59** / 61 | 0.252 / 0.330 / **0.892** / 0.370 | 0.968 / 0.944 / **0.542** / 0.929 |
| 1 | 9 / 18 / 27 / 34 | 64 / 63 / **58** / 64 | 0.265 / 0.381 / **0.830** / 0.336 | 0.965 / 0.927 / **0.605** / 0.942 |

多数 block 的 epsilon-halving 方向一致，但少量深层 block 对 BF16/NF4 有明显非线性或数值不稳定。预注册规则要求整 case 的全部候选通过，因此不能后验删除这些 block 再宣称成功。

## 更直接的淘汰理由

即使未来用 FP32 或更小 epsilon 修复 FD 稳定性，当前 score 仍未证明 Value-mediated directed relay：

- seed 0 的简单 `pre_score_pair` 与 KVQ-R 同为 1.000；
- seed 1 的 `pre_score_pair=1.000`，高于 `KVQ-R=0.750`；
- reverse edge 在 seed 0/1 分别为 0.969/0.813，与或高于 KVQ-R，缺少可靠方向性；
- 因而高 AUROC 可由“两个端点各自都与最终 Query 相关”解释，不需要 $K\rightarrow V\rightarrow Q\rightarrow K$ 接力。

这已经触发协议中的 control stop rule，所以没有必要仅为获得有效 AUROC 而放宽 FD 阈值或继续跑更多 seeds/长度。

## 可保留的发现

Top-64 pre-RoPE block proposal 在两例中均覆盖两条 gold 证据，说明候选生成仍有价值；但当前 relay edge 不增加可辨识信息。若论文继续走方法路线，应把精力放在“为什么某个候选对答案有因果效用”，而不是把两个单点 relevance 分数重新组合成路径。

原始合并结果见 `case_rows.csv`、`case_summary.csv` 与 `summary.json`；每个 seed 的 `raw.pt` 和完整审计保存在相邻的 `kvq_relay_edge_smoke_gpu6_seed0/`、`kvq_relay_edge_smoke_gpu7_seed1/`。
