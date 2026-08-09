# QKSieve 与 RetrievalAttention / RetroInfer 的官方基线边界

## 1. 先纠正系统身份

`microsoft/RetrievalAttention` 仓库目前公开的可运行实现是 **RetroInfer**，
不是 2024 年论文中的原始 RetrievalAttention 实现。

审计依据：

- 2024-09-26 到 2025-05-15 以前的提交只有 README、LICENSE 等仓库元数据；
- 第一个包含 `attn_hub/`、`cache_hub/`、CUDA kernel、LongBench 和 RULER
  代码的提交 `4fc50f6`，README 标题和方法均为 RetroInfer；
- 当前固定提交为
  `6b1228c346836769da0ed525dadf05bb7010e96b`，README 同样明确写
  RetroInfer，并引用 arXiv:2505.02922；
- 仓库保留 RetrievalAttention 论文引用，是方法演进关系，不代表当前代码
  是原始 RetrievalAttention 的官方实现。

因此论文必须拆成两行：

1. **RetrievalAttention (paper-reported)**：只能引用论文数字和方法描述，
   不写成“我们复现”；
2. **RetroInfer (official system)**：使用当前公开代码做正式复现。

把 RetroInfer 结果标成 RetrievalAttention official reproduction 会造成基线
身份错误，审稿时属于严重可复现性问题。

## 2. 固定版本

| 组件 | 固定值 |
|---|---|
| 官方仓库 | `https://github.com/microsoft/RetrievalAttention.git` |
| RetroInfer commit | `6b1228c346836769da0ed525dadf05bb7010e96b` |
| Starmys weighted FlashAttention | `56d96228ada74d6df806b0083bf018d0d57f57e9` |
| CUTLASS 候选 pin | `e64a9136dd929639e5f7c969fe5af3bf7415cd4f` |
| 官方环境 | Python 3.10.16、CUDA 12.4、PyTorch 2.5 路径 |

CUTLASS 和 weighted FlashAttention 的固定提交目前只用于消除“安装时拉取
浮动 HEAD”的不确定性，必须在目标 GPU 上编译通过后才能标记为最终依赖。

只做源码准备和审计：

```bash
bash scripts/prepare_retroinfer_official_20260728.sh
```

该脚本不会安装包、编译 kernel 或占用 GPU。审计结果写入：

```text
results/20260728_retroinfer_official_checkout_audit.json
```

## 3. 官方原生协议

当前仓库的原生配置为：

- retrieval budget ratio：1.8%；
- attention-estimation cluster ratio：23.2%；
- CPU-GPU 模式的 cache ratio：5%；
- LongBench 直接从 `THUDM/LongBench` 加载；
- 原生脚本固定 `ignore_eos=True`；
- 原生 throughput 数字来自 80GB A100 和大容量 NUMA CPU 环境。

第一阶段应不修改官方源码，分别运行 `Full_Flash_Attn` 与 `RetroInfer`，
确认能够复现仓库口径。该阶段回答“官方系统是否能复现”，不能直接回答
“在 QKSieve 协议下谁更好”。

## 4. 公平对齐协议

第二阶段保持 RetroInfer 的 cache、wave index、attention kernel 和
CPU-GPU 执行不变，只对齐评测外壳：

1. 使用与 QKSieve 完全相同的 16 个英文 LongBench 样本和 `sample_id`；
2. 使用相同的 middle truncation、7,500-token 上限和 Llama-3 chat wrapper；
3. 使用相同的任务生成上限；
4. 使用 Llama 的 EOS、`<|end_of_text|>`、`<|eom_id|>`、`<|eot_id|>`，
   SAMSum 额外按首个换行 token 停止；
5. 使用相同的 LongBench metric 实现和逐样本严格配对；
6. 同时运行官方 Full-Flash 后端，防止把模型实现差异归因于稀疏方法。

对齐层只能改变输入、停止和评分协议，不能修改 RetroInfer 的检索预算、
聚类、缓存放置或 kernel。结果必须标为
**RetroInfer official backend under aligned evaluation protocol**，不能冒充
官方仓库原生数字。

对齐评测入口已经准备：

```bash
bash scripts/launch_retroinfer_aligned_longbench_5gpu_20260728.sh
```

它在每个样本内严格配对：

```text
retroinfer_stack_full_flash
retroinfer_official_aligned
```

每条结果记录 prompt hash、stop token、官方 commit、retrieval/estimation/cache
比例，以及 cache init、prefill、cache prepare、CUDA graph capture、decode、
总请求时间、GPU peak 和 CPU peak RSS。分析器只接受 3,750 个严格配对和
16 个任务：

```text
src/analyze_retroinfer_aligned_longbench_20260728.py
```

## 5. 速度与显存必须报告

RetroInfer 是 CPU-GPU 异构系统，QKSieve 是 GPU-resident exact-KV 系统。
二者不能只比较 attention token ratio。正式表至少需要：

- 相同 GPU、CPU、NUMA 绑定和 batch=1；
- prefill、index build、cache prepare、首 token、稳态 TPOT 和请求总延迟；
- CPU 常驻内存、page-locked memory、GPU allocated/reserved peak；
- CPU-GPU 传输字节和传输时间；
- retrieval、estimation、exact attention 的实际 token/cluster 比例；
- 16K/32K/64K/128K 与 64/256/1,024 输出长度；
- Full-Flash、RetroInfer、QKSieve 三者的完整系统速度。

官方 A100 数字与本地 RTX 3090 数字只能分别报告，不能直接计算 speedup。

## 6. 当前完成与未完成

已完成：

- 官方仓库身份和历史审计；
- 当前官方提交固定；
- 原生配置、LongBench 停止策略和依赖浮动点审计；
- 无 GPU 的 checkout 校验脚本；
- 对齐 LongBench runner、五卡 launcher、严格配对分析器和 CPU 测试。

未完成：

- 目标服务器的独立 Python 3.10/CUDA 12.4 环境；
- CUTLASS 与 weighted FlashAttention 固定提交编译验证；
- 官方原生 LongBench/RULER 复现；
- 公平对齐 LongBench 的真实 GPU 数字；
- 与冻结 QKSieve 的同硬件系统对比。

在上述项目完成前，论文中只能把 RetroInfer 放在待补系统基线中；原始
RetrievalAttention 只能作为 paper-reported baseline。
