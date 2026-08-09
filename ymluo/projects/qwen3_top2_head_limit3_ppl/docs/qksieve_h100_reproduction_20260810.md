# QKSieve-Robust H100 复现实验

## 目的

在同一组 H100 张量和同一冻结方法上，独立测量：

1. 64K/128K 原生 MHA attention；
2. 64K/128K 真实模型稳态 decode；
3. cold、cold end-to-end、warm、shared-prefix 和 append-only 请求；
4. Full、QKSieve-Fast、QKSieve-Robust 与审计版 FIER 的 resident bytes、延迟和速度比。

当前仓库没有 H100 实测结果。本文件只定义复现与验收合同，不能作为结果证据。

## 冻结合同

- 主方法：`qksieve_qmse_oas_requestlocal_valuesketch16_sorted_c64`；
- 数值冻结提交：`328e01718deebfdfc80dbd8e588a1a95a1832b59`；
- 审计实现提交：`f300fb280a597ceb124d454cdfc9a0a1665d6a04`；
- token 预算：`min(N,1280,max(256,ceil(0.06N)))`；
- quantile 有效样本上限：512；
- ValueSketch：rank 16、block 256、INT4、`alpha=0.5`；
- 无 router、长度切换、rerank 或 Full fallback；
- 原始 FP16 K/V 始终常驻 GPU，辅助索引不能被写成压缩后总 KV 占用。

机器可读合同位于：

```text
configs/qksieve_robust_iclr2027_frozen_20260810.json
```

## 硬件与软件

- 至少两张 80GB H100；
- 64K 使用一张卡，128K 使用两张卡；
- CUDA 与 PyTorch 必须彼此兼容；
- CUDA 扩展按 `TORCH_CUDA_ARCH_LIST=9.0` 编译；
- 默认模型为 MHA 的 `Yarn-Llama-2-7b-128k`，不能替换成 GQA 模型后仍称为同一速度口径。

启动器会逐卡检查设备名称包含 `H100` 且显存不少于 75,000 MiB。RTX 3090、
A100 或较小显存设备会直接生成 `FAILED`，不能进入 H100 汇总。

## 启动命令

```bash
export ROOT=/path/to/qksieve_iclr2027
export PYTHON=/path/to/venv/bin/python
export MODEL=${ROOT}/models/Yarn-Llama-2-7b-128k
export RUN_ROOT=${ROOT}/results/20260810_qksieve_h100_matched_v1
export GPU_64K=0
export GPU_128K=0,1
export TORCH_CUDA_ARCH_LIST=9.0

nohup bash ${ROOT}/scripts/launch_qksieve_h100_matched_20260810.sh \
  >${RUN_ROOT}.launcher.log 2>&1 < /dev/null &
```

启动器固定运行三个 seed：`20260810,20260811,20260812`。Attention 使用 20 次
warmup 和 80 次计时；decode 生成 256 token，并排除前 32 token 后报告稳态均值；
persistent 实验使用四个 64-token 分支和一个 128-token append-only 分支。

## 直接测量口径

| 输出 | 计时范围 |
|---|---|
| Attention | selector、候选写出和 exact sparse attention；不含索引构建 |
| Steady decode | 完整模型逐 token forward；不含共同 dense prefill |
| Cold persistent | 一次索引构建、Query setup 和完整 decode 分支；不含 dense prefill |
| Cold end-to-end | 从 dense prefill 前开始，到首个完整 decode 分支结束 |
| Warm | K/V 与索引常驻后的 Query setup 和 decode |
| Shared-prefix amortized | 一次 prefill/index 在四个分支间均摊 |
| Append-only | 新 token prefill、增量索引 append 和后续 decode |

完整路径延迟必须直接计时，不能由独立 kernel 时间相加。单层 kernel 只用于解释
瓶颈，不能替代真实 decode 或请求级结果。

## 结果文件

```text
${RUN_ROOT}/manifest.txt
${RUN_ROOT}/attention/seed*.json
${RUN_ROOT}/decode/n*/seed*/*.json
${RUN_ROOT}/persistent/n*/seed*/*.json
${RUN_ROOT}/summary.json
${RUN_ROOT}/ALL_COMPLETE
```

`manifest.txt` 必须保存模型、冻结合同、源码 SHA256、Python、PyTorch、
Transformers、CUDA、cuDNN、驱动和 GPU UUID。`summary.json` 必须同时包含：

- 64K 与 128K；
- 三个 seed；
- attention 的 Full/Fast/Robust/FIER 延迟与 resident bytes；
- Robust 稳态 decode；
- 五种 persistent 请求速度；
- 每卡峰值 allocated/reserved memory 之和；
- 明确的 claim boundary。

## 验收

先生成论文表：

```bash
python ymluo/papers/countcap_iclr2027/scripts/make_qksieve_h100_tables.py \
  --summary ${RUN_ROOT}/summary.json
```

再运行总证据校验器：

```bash
python ymluo/projects/qwen3_top2_head_limit3_ppl/src/verify_qksieve_robust_paper_evidence_20260810.py \
  --project_root ymluo/projects/qwen3_top2_head_limit3_ppl \
  --persistent_summary RESULTS/persistent/independent_summary.json \
  --longbench_summary RESULTS/longbench/paired_summary.json \
  --ruler_summary RESULTS/ruler/paired_summary.json \
  --multimodel_summary RESULTS/multimodel/multimodel_summary.json \
  --h100_summary ${RUN_ROOT}/summary.json \
  --output RESULTS/frozen_evidence_report.json
```

只有设备、软件、长度、seed、冻结合同、正延迟、显存和索引字段全部通过，H100
结果才能进入正文。任何缺失字段、非 H100 设备或方法漂移都按失败处理。
