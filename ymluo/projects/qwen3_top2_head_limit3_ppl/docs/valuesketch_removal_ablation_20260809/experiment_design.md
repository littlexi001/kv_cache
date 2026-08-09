# ValueSketch 去留消融：实验设计

## 设置

- 模型：Llama-3.1-8B-Instruct，FP16，标准 GQA4。
- 硬件：三张 RTX 3090，每条文本流独占一张卡。
- 文本：War and Peace、The Count of Monte Cristo、QKSieve 大型实现源码。
- 每条文本测试两个长度：32,768 历史 token 后评估 64 token；98,304 历史 token 后评估 32 token。文本不足时确定性循环。
- seed：20260809。
- 条件：Full Attention、QKSieve 无补偿、QKSieve rank-16 INT4 ValueSketch 补偿。
- 候选：每 head 最多 1,280 token，即 32K 下 3.90625%、96K 下 1.30208%。
- 质量：`quality_retention = exp(NLL_full - NLL_method)`；大于 100% 只表示本批目标 token 的 NLL 更低。
- 速度：runner 中完整模型逐 token forward 的平均延迟；固定准备成本单列。

## 路径

- launcher：`scripts/launch_qksieve_valuesketch_removal_ablation_20260809.sh`
- runner：`src/run_qksieve_coldskip_longcontext_quality_20260730.py`
- 远端结果：`/home/fdong/qksieve_iclr2027/results/20260809_valuesketch_removal_ablation_32k_v1`

## 限制

这是快速决策实验，不是论文最终质量表。它没有覆盖问答任务、体育/医学弱项、多个 seed 或原生 MHA ValueSketch kernel；结论必须限定为是否值得继续把补偿作为默认主线。
