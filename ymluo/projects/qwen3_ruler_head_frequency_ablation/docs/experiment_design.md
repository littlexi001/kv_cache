# 实验设计

## 研究问题

冻结 Qwen3-8B 权重，只删除深层特定 head-group 的特定 RoPE 频率旋转，能否超过原生 RoPE 的 RULER-32K 分数？

## 固定条件

- 模型：Qwen3-8B 原始 checkpoint，不加载续训 adapter。
- 数据：现有 13 类 RULER-32K，每类 2 条，共 26 条。
- 上下文：约 32K token。
- 推理：BF16 计算、4-bit 权重、SDPA、greedy decoding。
- 原生基线：同一 checkpoint、同一 prompt、全部频率保留标准 RoPE；已测官方均分为 85.19%。
- 所有权重、prompt、答案、解码参数与随机性保持不变。

## 分层地毯搜索

直接原子穷举需要 `18 deep layers × 8 head-groups × 64 frequencies = 9216` 个配置。实验采用以下固定搜索流程。

### 阶段 A：粗搜索

- 深层范围：layer 18–35。
- layer blocks：18–23、24–29、30–35。
- head-groups：0–7。
- frequency bands：0–7、8–15、…、56–63，共 8 段。
- 配置数：`3 × 8 × 8 = 192`，另加原生 RoPE。
- 筛选集：6 条固定 RULER-32K，包含原生成功、部分成功和失败样本。

一个粗配置会在一个 6-layer block、一个 head-group 和一个 8-frequency band 上使用单位旋转。

### 阶段 B：细化

对阶段 A 排名前 8 且具有正向信号的 `(layer block, head-group, frequency band)` 区域分别测试：

1. block 内每个单层 + 整个 8-frequency band，共最多 48 个配置；
2. 整个 block + band 内每个单频率，共最多 64 个配置；
3. 最优单层与最优单频率的交叉组合，共最多 8 个配置。

### 阶段 C：完整验证

- 在完整 26 条 RULER-32K 上测试最优 8 个单项配置。
- 按筛选得分进行逐项累加，测试最多 8 个组合配置。
- 所有最终配置都与本次运行重新计算的原生 RoPE 配对比较。

## 筛选指标

每条样本记录：

- 官方 RULER score；
- Gold answer mean NLL；
- Gold answer PPL；
- 首答案 token 准确率；
- 预测文本与运行时间。

对配置 `v` 和样本 `x` 定义

$$
\Delta_{\mathrm{score}}(v,x)=\mathrm{Score}(v,x)-\mathrm{Score}(\mathrm{native},x),
$$

$$
\Delta_{\mathrm{NLL}}(v,x)=\mathrm{NLL}(\mathrm{native},x)-\mathrm{NLL}(v,x).
$$

正的 NLL 差表示正确答案概率提高。为防止单个极端样本控制排名，使用

$$
U(v)=\operatorname{mean}_x\Delta_{\mathrm{score}}(v,x)
+0.05\operatorname{mean}_x\operatorname{clip}(\Delta_{\mathrm{NLL}}(v,x),-2,2).
$$

阶段 A 的排名只用于生成候选，不能作为最终结论。

## 通过、失败和证据不足

- 通过当前 26 条 probe：最终配置的官方均分严格高于同次原生 RoPE，且 Gold NLL 不系统性变差。
- 失败：所有完整验证配置均不高于原生 RoPE。
- 证据不足：只在 6 条筛选集提高，或提高来自单个样本而其他样本普遍退化。
- 若在当前 probe 通过，论文级结论仍需更多 seed、其他长度及 LongBench 验证。

## 调试产物

每个配置保存精确的 layer、head-group、frequency pairs；每条样本保存配对指标；汇总文件保存改进、退化、持平样本数。原生 replay 的 logits 最大误差必须小于 `1e-4`，否则停止搜索。

## 独立稳定性协议

地毯搜索使用的 seed 42 只用于提出候选，不再用于选择最终方法。

- validation：RULER seeds 43、44，每个 seed 的 13 类任务各 1 条，共 26 条独立样本；用于选择 alpha 和候选结构。
- test：RULER seeds 45、46、47，每个 seed 的 13 类任务各 2 条，共 78 条独立样本；方法固定后只评测一次。
- 外部验证：LongBench 与 PG19 PPL 不参与 RULER 参数选择。

validation 比较以下候选：

1. `L25/G3/F46` 的 alpha `0, 0.125, 0.25, 0.5, 0.75`；
2. `L18–23/G4/F47` 的 alpha `0, 0.25, 0.5`；
3. `L25/G3/F40–47` 的 alpha `0, 0.25`；
4. F46 与 F47 两处同时使用 alpha `0` 或 `0.25`；
5. 原生 RoPE。

候选进入 test 必须同时满足：

- 两个 validation seeds 的官方分数均不低于各自原生 RoPE；
- 合并 Gold NLL 改善为正；
- 在满足前两项的配置中按 Gold NLL 改善排序，最多保留 3 个；
- 原始 `L25/G3/F46, alpha=0` 无论排序如何都保留为发现阶段对照。

最终方法需要同时满足：

- 三个 test seeds 合并后的官方 RULER score 不低于原生 RoPE；
- test Gold NLL 的 paired bootstrap 95% CI 全部大于 0；
- LongBench QA/EM 不系统性退化；
- PG19 4K PPL 相对变化不超过 1%，32K PPL 不退化并最好改善。
