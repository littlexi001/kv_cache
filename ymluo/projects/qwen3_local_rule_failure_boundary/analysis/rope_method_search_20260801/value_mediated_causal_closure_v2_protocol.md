# Value-mediated causal closure v2（最小 BF16 验证）

## 为什么需要 v2

旧 runner 使用带 autograd 的 instrumented baseline 作为 singleton replay 的
比较基线。实际 intervention 则使用 `input_ids + inference_mode`。在 BF16 下，
两条执行路径之间的公共数值偏移可能被错误记成单 token 的因果效应。

v2 在每个 case 中新增一次 `epsilon=0` replay：

- 与 target/random intervention 使用同一个 patched custom-attention forward；
- 使用相同的 `input_ids + inference_mode + read-only prefix cache`；
- 不改变任何 attention score；
- 所有实际 `delta_gold_nll`、`delta_gold_ppl` 和
  `delta_gold_conflict_margin` 都只相对该 no-op 计算。

原生 baseline 和 autograd baseline 仍保留，但分别只用于原生一致性检查和
候选/一阶导数冻结，不再充当 causal delta 的参考点。

## 每个 case 的强制审计

`case_replay_audit` 会 fail closed 地检查：

1. 恰好存在一条 custom-attention `epsilon=0` no-op；
2. no-op 的 score 改动数为 0；
3. 每条 singleton/joint intervention 的 delta 均由 no-op 重算；
4. 每个 singleton candidate 恰好有 target 和 matched-random 两条 replay；
5. target/random 的 layer、head、class 相同，但 token 位置不同；
6. 每条 singleton replay 恰好只改一个 score；
7. 所有 replay 后 prefix KV cache 的 identity/version 不变。

## 最小 8K BF16 smoke

脚本：

`scripts/run_value_mediated_probe_v2_bf16_smoke_gpu67_20260801.sh`

配置：

- Qwen3-8B，unquantized BF16（刻意不启用 NF4）；
- 8K；GPU 6/7 各一个 seed；
- 每类只采样 2 个 token；
- 每类只冻结 top-1 candidate；
- 每个 candidate 运行 target/random 配对控制；
- `epsilon=0.25`（与旧 smoke 可直接配对），另有严格 `epsilon=0` no-op；
- 不运行 joint intervention。

远程启动命令（只允许 GPU 6/7）：

```bash
bash /home/fdong/ymluo/projects/qwen3_local_rule_failure_boundary/scripts/run_value_mediated_probe_v2_bf16_smoke_gpu67_20260801.sh
```

预计每张 24GB GPU 的峰值显存为约 **20--23 GiB**：BF16 权重约
15--16 GiB，8K KV cache 约 1.1 GiB，其余为 eager attention、单 query
autograd 图、临时 cache wrapper 和 CUDA workspace。该估计接近 24GB 上限；
若实测 OOM，应先把 `--prefill-chunk-size 64` 降为 32，而不是切换到本地 GPU
或占用 GPU 0--5。若仍 OOM，再记录 BF16 24GB 不可行，不应悄悄改用 NF4。

## 判读

首先查看：

- `custom_noop_delta_from_instrumented`：旧实验公共偏移的大小；
- `case_replay_audit.passed`：必须为 `true`；
- target/random 相对 no-op 的实际 margin delta；
- 一阶预测与实际 delta 的 Pearson、Spearman、sign accuracy、closure error。

只有在去掉 no-op 公共偏移后，target 仍显著优于 matched random，且一阶预测
在独立 seed 上保持方向与排序一致，才能把 value-mediated quantity 当作有效的
局部因果解释。否则应判为 NO-GO，而不是继续扩大样本掩盖执行路径偏差。
