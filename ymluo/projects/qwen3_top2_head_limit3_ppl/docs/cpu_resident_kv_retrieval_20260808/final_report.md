# CPU 常驻完整 KV 的 QKSieve 精确取回实验

## 结论

这条路线值得继续。QKSieve 在“完整 KV 无法放入单卡显存”的场景中，比在 KV
全驻 GPU 场景更有明确优势：GPU只扫描占完整KV 5.859%的低比特Key索引，CPU
保留完整FP16 K/V，每层只搬运真实候选需要的精确向量。

在128K真实QKSieve候选上，实测 attention/offload 子系统加速为 **20.70x**；
把host路径加到保守resident Decode底座后，估算为 **9.68x**。该结果通过了
预设的1.5x门槛，但完整autoregressive CPU-cache集成仍未完成，因此论文目前
只能把20.70x作为子系统实测，把9.68x标为估算。

## 1. 问题

Qwen3-4B的完整FP16 KV为144 KiB/token。64K、128K和256K分别需要9、18和
36 GiB。Full-offload虽然解决容量问题，却需要每生成一个token再次跨PCIe搬运
全部KV。

QKSieve不删除CPU中的完整状态。它在GPU中保存240-bit/token/KV-head的
QK-balanced索引，以近似分数定位候选，再读取候选的原始FP16 K/V。因此低比特
误差只影响地址选择，不会进一步量化最终参加Attention的K/V。

## 2. 方法

每层执行以下步骤：

1. GPU用QKSieve低比特索引产生per-query-head候选ID。
2. 同一KV head对应的4个query head在GPU求候选并集。
3. fetch ID写入pinned CPU buffer。
4. CPU把每个候选token的128维K和V作为连续行，用`index_select`压紧到pinned
   staging buffer。
5. staging buffer连续H2D。
6. GPU用映射表恢复各query head自己的候选，执行精确QK、softmax和Value聚合。

不采用大page。128K真实候选下，page=2和page=16分别令传输增加1.58倍和
5.69倍，均降低最终速度。

## 3. 真实候选性质

64K/128K真实追踪表明，同一GQA组的候选并集约为单head预算的2.27--2.34倍，
而非最理想的1倍或最坏的4倍。相邻生成token候选Jaccard为0.36--0.42，说明
后续可以加入小型精确KV热缓存，但当前结果没有依赖该优化。

## 4. 效果

128K下，Full每层从CPU搬运512 MiB；QKSieve真实候选平均只搬13.82 MiB，流量
降低37.03倍。检索路径每层p50为：ID回传0.037 ms、CPU gather 0.629 ms、
H2D 1.179 ms、GPU remap与稀疏Attention 0.261 ms，总计2.136 ms。Full的
H2D与native-GQA Attention为44.212 ms，因此子系统加速20.70倍。

采用更保守的128K resident底座后：

```text
Full-offload      约 1626.23 ms/token
QKSieve-Host      约  168.07 ms/token
保守估算加速            9.68x
```

64K对应保守估算为7.05x。即使人工设置4个query head候选完全不重合，128K的
优化后保守估算仍为8.09x。

## 5. 失败结果与解释

1. 逐元素`torch.gather`很慢。128K真实候选为4.01 ms/layer；整行
   `index_select`仅0.629 ms/layer。旧实现慢在计算接口没有利用128维连续布局，
   不是CPU取回这一物理先验本身失败。
2. page=2只有7.38x保守加速，page=16只有2.53x。真实候选的空间聚集不足以抵消
   多搬token的成本，因此当前不能采用常规PagedAttention式粗页取回。

## 6. 论文边界与下一步

本实验已经证明CPU数据路径有足够大的速度空间，但还没有完成论文级系统闭环。
下一步应实现真正的host-resident cache：prefill后把完整KV放入CPU，decode每层
直接消费QKSieve候选，并在同一autoregressive循环中测TPOT和峰值显存。

随后必须在相同GPU显存、CPU线程数和PCIe条件下比较Full-offload及至少三个强
基线，例如RetrievalAttention、PQCache、InfiniGen或ShadowKV。论文主表必须同时
报告质量、TPOT、CPU到GPU字节数、持久GPU索引、staging峰值和索引构建成本。

当前允许的主张是“QKSieve显著减少CPU-resident KV的精确取回成本”。在完整系统
与强基线完成前，不能主张已经取得9.68x端到端autoregressive实测加速。
