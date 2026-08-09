# 面向 ICLR 2027 的 RoPE 长程方法文献与新颖性审计

**检索截止：2026-08-01（Asia/Shanghai）**

**审计对象**

- **方向 A：远程 RoPE 忽略或削弱。** 包括远距离 NoPE、RNoPE/HoPE、维度截断、相位饱和、距离封顶、相对位置分桶与分组。
- **方向 B：新的 local + global 位置编码或双通道几何。** 包括同层或跨层的局部精细几何与全局内容/粗粒度几何、双分数通道，以及 phase repair。
- 当前 SAGE / dual-max / pre-RoPE proposal + post-RoPE consumption 仅作为内部参照，不算上述两个拟议方向。

**证据规范**

- 只把原论文、官方 proceedings、ACL Anthology、OpenReview/ICLR 官方页面和 arXiv 原稿作为事实来源。
- “正式”表示已进入正式会议或期刊；“预印本”表示截至检索日只有 arXiv 或仍在评审。
- 重点覆盖 2024--2026；NoPE 的 2023 奠基论文仅用于追溯先例。
- 本报告是截至检索日的 novelty audit，不可能排除尚未公开或 ICLR 2027 同期投稿。提交前应再做一次标题、摘要和公式级增量检索。

## 1. 执行结论

### 1.1 一句话判断

| 方向 | 当前新颖性判断 | 原因 |
|---|---|---|
| A：远程 RoPE 忽略/削弱 | **红灯：不能以一般方法贡献投稿** | “删掉远程位置”“远程 NoPE”“距离封顶/分组/分桶”“只保留局部精细位置”分别已被 p-RoPE、HoPE、RNoPE-SWA、DroPE、SelfExtend、LM-Infinite、InfLLM、AdaGroPE、STRING、LaMPE、CALIOPE 等覆盖。 |
| B：local + global 双通道几何 | **黄红灯：一般双通道设计已高度拥挤** | 层级双通道已有 RNoPE-SWA、SWAN-GPT、P-RoPE；子空间双通道已有 DeepSeek MLA、p-RoPE、HoPE；分段/距离双尺度已有 BiPE、SelfExtend、AdaGroPE、LaMPE；冻结模型的 local+sinks+remote memory/fixed-remote-geometry 已有 InfLLM；双复数分量已有 RoPE++。 |
| B 中的 generic phase repair | **红灯：名称与核心操作均已有直接先例** | PSC 已直接使用 “Phase Shift Calibration”；TAPA、CARoPE、Selective RoPE 已做 token/content-dependent phase；Resonance RoPE、MrRoPE、LPES 等已做频率/相位修正；RoPE++ 已增加互补相位分量。 |
| generic value-aware / output-aware token scoring | **红灯：不能作为本项目的新方法贡献** | VATP 已把 attention 与 Value norm 相乘；CriticalKV 已纳入 projected Value 和输出扰动；LaProx 已建模 attention、投影 Value、跨 head/layer 输出；LOCOS 已将逐位置 OV 写入投影到答案 unembedding。 |
| 冻结模型上的因果、稀疏、反事实 phase repair | **暂定黄灯：仍可能形成可守的窄缺口** | 检索中尚未发现同时满足“完全冻结、无需校准语料、由当前 query--evidence 的 pre/post-RoPE 抑制事件触发、只修最小频率平面/候选块、无事件时严格退化为原模型、并给出因果归因”的工作。这个判断是组合式缺口，不是对任一单组件的新颖性判断。 |

### 1.2 最危险的投稿叙事

以下叙事极可能被审稿人直接判为增量：

1. “距离超过 \(W\) 后关闭 RoPE。”
2. “局部窗口用 RoPE，远程区域用 NoPE。”
3. “把 RoPE 与 pre-RoPE/NoPE 分数做静态或距离依赖的加权和。”
4. “只旋转一部分维度，剩余维度保留内容通道。”
5. “按层交替 local-RoPE 与 global-NoPE。”
6. “学习一个内容相关的 phase/frequency gate。”
7. “校准/修复 RoPE phase 以改善长程检索。”
8. 若同时声称稀疏加速：“先用廉价频率/低维分数选候选，再在候选上做原始或高保真 RoPE 注意力。”FASA 已在 ICLR 2026 给出 frequency-chunk proposal--exact consumption；SALS 已在 NeurIPS 2025 给出 pre-RoPE/RoPE-free 低秩 proposal--RoPE consumption。两阶段分离与 pre-RoPE selector 都不能单独作为结构创新。
9. “attention mass 不够，所以乘 Value norm、投影 Value 或答案方向再选 token/head。”VATP、CriticalKV、LaProx 与 LOCOS 已分别覆盖这些宽泛版本；即便使用 exact gradient，也必须把贡献限定为 **RoPE phase suppression 的因果分解与验证**，不能声称首个 value/output-aware importance。

### 1.3 建议的研究定位

如果继续方向 B，主张应从“新位置编码”收缩为：

> **对冻结 RoPE LLM 中已被实测为相位抑制的远程证据，进行事件触发、最小化、稀疏且可回退的反事实相位恢复。**

这要求方法同时满足：

- **冻结性：** 不改模型权重，不训练新 PE、不做短校准或轻量 continual pretraining；
- **事件触发：** 不是只由距离决定，而是由同一 query--key/block 的 content evidence 强而 post-RoPE evidence 弱这一可观测冲突决定；
- **最小干预：** 只改造成抑制的频率平面、head 或证据块，并显式最小化相位位移；
- **局部不变：** 局部窗口仍使用原始 RoPE；不触发时与原模型逐元素一致；
- **因果闭环：** 修复使证据 rank/mass、下游 hidden state 和答案 logit 按预测方向恢复，而不仅是 benchmark 平均分上升；
- **次序安全：** 在需要顺序、否定、最近指代和代码位置的任务上不退化；
- **数值排除：** 排除 BF16 位置误差、缓存/重算差异和 softmax denominator 变化等替代解释。

单独满足其中任何一项都不够；可能的新颖性来自这个严格的交集。

## 2. 方法分类表

### 2.1 NoPE、远程消融与 local/global 混合

| 方法 | 年份/状态 | 核心机制与干预粒度 | 是否适用于既有冻结模型 | 与 A/B 的关系 |
|---|---|---|---|---|
| [NoPE：The Impact of Positional Encoding on Length Generalization](https://proceedings.neurips.cc/paper_files/paper/2023/hash/4e85362c02172c0c6567ce593122d31c-Abstract-Conference.html) | NeurIPS 2023，正式 | 不显式加入 PE；因果 mask 可诱导隐式位置。 | 需从头训练 | A 的最早概念先例。 |
| [Length Generalization of Causal Transformers without Position Encoding](https://aclanthology.org/2024.findings-acl.834/) | Findings ACL 2024，正式 | 分析 NoPE 的 attention distraction，并用 head temperature tuning 扩展有效长度。 | 需训练/调温 | 说明 NoPE 不是无代价全局检索器；温度必须作为控制。 |
| [p-RoPE / Round and Round We Go](https://proceedings.iclr.cc/paper_files/paper/2025/hash/e6d58fc68c0f3c36ae6e0e64478a69c0-Abstract-Conference.html) | ICLR 2025，正式 | 截去部分低频 rotary 分量，保留高频位置分量；论文把高频与位置、低频与语义联系起来。 | 主要为训练时架构 | A 的“部分忽略 RoPE”和 B 的内容/位置子空间直接先例。 |
| [HoPE](https://aclanthology.org/2025.acl-long.1123/) | ACL 2025，正式 | 将造成长期衰减/全局 U 形的某些 RoPE 分量替换为 position-independent 分量，同时保留局部位置分量。 | 主要从头训练，论文规模至 3B | 对 A 和 B 都是最强近邻之一。 |
| [RNoPE-SWA](https://proceedings.neurips.cc/paper_files/paper/2025/hash/5c9ab393551b7a39b4c02d88fe5e7e69-Abstract-Conference.html) | NeurIPS 2025，正式；[arXiv](https://arxiv.org/abs/2501.18795) | 约 1:3 交替 global full-attention NoPE 层与 local SWA-RoPE 层，形成检索/局部处理分工。 | 需预训练/继续训练新架构 | B 的层级 local/global 最直接先例；也覆盖 A 的远程 NoPE。 |
| [SWAN-GPT](https://aclanthology.org/2025.emnlp-main.123/) | EMNLP 2025，正式；[arXiv](https://arxiv.org/abs/2504.08719) | 同样交替 NoPE 全局层与 SWA-RoPE 局部层。 | 需训练 | 使 “RNoPE-SWA 只是单篇偶然设计” 这一辩护失效。 |
| [DroPE](https://iclr.cc/virtual/2026/poster/10009468) | ICLR 2026，正式；[arXiv](https://arxiv.org/abs/2512.12167) | 预训练后删除所有位置编码，再在原训练长度做短 recalibration；无需长上下文微调。 | **不是完全冻结**，需要短校准 | A 的“删 RoPE”强先例；任何全局/远程删除方案必须与其区分。 |
| [Selective RoPE](https://iclr.cc/virtual/2026/poster/10011040) | ICLR 2026，正式；[arXiv](https://arxiv.org/abs/2511.17388) | 由输入投影得到累积旋转角，并带可学习 phase gate；可选择是否旋转。 | 需训练 | 直接覆盖“内容依赖地关闭/改变相位”。 |
| [Partial RoPE systematic study](https://arxiv.org/abs/2603.11611) | 2026，预印本 | 系统改变 rotary 维度比例；报告少量维度也可接近 full RoPE，并讨论 NoPE 训练不稳。 | 从头训练研究 | 进一步压缩“少旋转一些维度”的新颖空间。 |
| [Periodic RoPE, P-RoPE](https://arxiv.org/abs/2605.27980) | 2026，预印本 | SWA 局部 RoPE 层与全局 NoPE 层堆叠，目标是避免 position exhaustion。 | 需训练 MiniWin | 名称与 p-RoPE 不同；机制与 RNoPE/SWAN 高度同源。 |
| [NAPE in Long-Context Generalization with Sparse Attention](https://iclr.cc/virtual/2026/poster/10009641) | ICLR 2026，正式 | 一半 heads 用 NoPE 获取内容驱动的远程检索，另一半用 ALiBi 保持局部性；作者明确称其为 practical default 而非贡献。 | 从头训练 | 说明“按 head 分 local/global PE”本身甚至已被视为常规配置。 |
| [SmallThinker](https://arxiv.org/abs/2507.20984) | 2025，预印本 | 面向本地部署的从头训练模型，采用 NoPE--RoPE hybrid sparse attention。 | 从头训练 | 进一步说明混合 NoPE/RoPE 已进入系统架构实践，但不是冻结模型 repair。 |

### 2.2 距离重映射、封顶、分组与多粒度位置

| 方法 | 年份/状态 | 核心机制与干预粒度 | 训练要求 | 与 A/B 的关系 |
|---|---|---|---|---|
| [SelfExtend](https://proceedings.mlr.press/v235/jin24b.html) | ICML 2024，正式 | local neighbor attention 保留精细位置；远程用 grouped attention 压缩/复用相对位置。 | 训练免费 | A 的远程分组和 B 的 bi-level local/global 直接先例。 |
| [DCA](https://openreview.net/forum?id=If4xW9vF7U) | ICML 2024，正式；[arXiv](https://arxiv.org/abs/2402.17463) | chunk decomposition；分别进行 intra-chunk、inter-chunk 与 successive-chunk 位置重映射。 | 训练免费 | 覆盖 chunk/local/global 多几何。 |
| [LM-Infinite](https://aclanthology.org/2024.naacl-long.222/) | NAACL 2024，正式 | 距离限制与 \(\Lambda\)-形 attention mask，使超长输入保持训练内相对距离模式。 | 训练免费 | A 的 distance cap/远程饱和重要先例。 |
| [InfLLM](https://proceedings.neurips.cc/paper_files/paper/2024/hash/d842425e4bf79ba039352da0f658a906-Abstract-Conference.html) | NeurIPS 2024，正式；[arXiv](https://arxiv.org/abs/2402.04617) | 冻结模型中组合 local window、initial tokens/sinks 与按当前 query 检索的远程 memory blocks；被取回的远程 token 在最终 attention 中统一使用局部窗口边界的固定相对距离 \(l_L\)。 | 训练免费 | **若 A/B 同时包含远程候选检索和固定/封顶远程几何，这是最强整体近邻之一。** 它已覆盖“内容选远程、位置消费时统一压回局部边界”。 |
| [FIRE](https://proceedings.iclr.cc/paper_files/paper/2024/hash/2f55a8b7b1c2c6312eb86557bb9a2bd5-Abstract-Conference.html) | ICLR 2024，正式 | 学习相对位置的连续函数并 progressive interpolation；统一/覆盖 T5 RPE、ALiBi、Kerple 风格。 | 需训练 | “学习分桶/距离函数”并非新方向。 |
| [BiPE](https://proceedings.mlr.press/v235/he24c.html) | ICML 2024，正式 | token 内的 intra-segment absolute PE 与 segment 间 relative PE 结合。 | 需从头训练 | B 的分段双尺度几何直接先例。 |
| [STRING](https://proceedings.iclr.cc/paper_files/paper/2025/hash/884baf65392170763b27c914087bde01-Abstract-Conference.html) | ICLR 2025，正式 | Shifted RoPE 将训练良好的相对位置区间平移到长序列位置，覆盖失效位置。 | 训练免费 | A 的远程位置替换/重映射先例。 |
| [AdaGroPE](https://aclanthology.org/2025.acl-long.28/) | ACL 2025，正式 | 局部窗口保留精细相对位置；距离越远，位置复用/分组次数渐增，并随输入长度自适应。 | 训练免费 | **A 的截断/饱和/分桶最强近邻；也是 B 的局部精细+全局粗粒度最强近邻。** |
| [Set Encoding](https://aclanthology.org/2025.acl-long.197/) | ACL 2025，正式 | 对集合元素复用相同 position IDs，并用 mask 阻断跨元素干扰。 | 训练免费 | 展示极端位置碰撞/复用可行，但场景限于集合结构。 |
| [LaMPE](https://aclanthology.org/2026.findings-acl.1608/) | Findings ACL 2026，正式 | 输入长度感知的 sigmoid 位置映射，加 multi-grained attention 分配局部与远程位置分辨率。 | 训练免费 | 对自适应分桶和 B 的多粒度 local/global 都是非常近的先例。 |
| [CALIOPE](https://aclanthology.org/2026.findings-eacl.120/) | Findings EACL 2026，正式 | 在冻结 RoPE LLM 推理时对 RoPE 输入位置做确定性、严格单调的 chunk-aware remapping；Moses/Hourglass/Decay calibrators 改变区间间距。 | 训练免费 | 与“冻结模型 + 推理时位置修复”直接竞争；静态 position repair 必须比较。 |

### 2.3 频率缩放、频带选择与相位校准

| 方法 | 年份/状态 | 核心机制 | 训练要求 | novelty 含义 |
|---|---|---|---|---|
| [YaRN](https://proceedings.iclr.cc/paper_files/paper/2024/hash/874a4d89f2d04b4bcf9a2c19545cf040-Abstract-Conference.html) | ICLR 2024，正式 | 不同频带使用不同插值/外推策略，并调整 attention temperature。 | 少量微调或推理缩放配置 | 所有频带改动的标准 baseline。 |
| [CLEX](https://proceedings.iclr.cc/paper_files/paper/2024/hash/3df38ca67befaed9c03b95ffee07d9f8-Abstract-Conference.html) | ICLR 2024，正式 | 用连续长度外推函数学习位置缩放动力学。 | 需训练 | 学习连续频率/长度变换的先例。 |
| [LongRoPE](https://proceedings.mlr.press/v235/ding24i.html) | ICML 2024，正式 | 搜索非均匀、逐维的插值因子并渐进延长，兼顾短上下文恢复。 | 搜索 + 少量长上下文微调 | 逐维频率修正基线。 |
| [Resonance RoPE](https://aclanthology.org/2024.findings-acl.32/) | Findings ACL 2024，正式 | 将 RoPE 波长/特征调整到更有利于训练短测长的共振形式，可与 YaRN 组合。 | 训练/缩放配方 | “修复频率特征间隙/相位周期”先例。 |
| [PSC](https://aclanthology.org/2024.emnlp-main.341/) | EMNLP 2024，正式 | **Phase Shift Calibration**：以小型 head-wise block-diagonal MLP 在 RoPE 前/后校准既有缩放方法的相位偏移，参数少于 1%，与 LoRA 微调联合。 | 需要学习校准模块 | “phase repair/calibration”字面与操作上的最直接先例。 |
| [Ms-PoE / Found in the Middle](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6ffdbbe354893979367f93e2121e37dd-Abstract-Conference.html) | NeurIPS 2024，正式 | head-specific 多尺度位置索引缩放，融合不同 receptive fields。 | 训练免费 | head-wise local/global scale 融合先例。 |
| [LongRoPE2](https://proceedings.mlr.press/v267/shang25a.html) | ICML 2025，正式 | needle-driven perplexity 引导逐维缩放搜索，配合 mixed-context training。 | 搜索 + 约 10B token 微调 | 强缩放基线；不能只比 PI/YaRN。 |
| [FoPE](https://proceedings.mlr.press/v267/hua25a.html) | ICML 2025，正式 | 将每个位置维度建模为 Fourier-series frequency mixture，处理破坏性或训练不足的频率。 | 需预训练/适配 | “用更丰富频谱修复 RoPE”已有直接先例。 |
| [PEPE](https://aclanthology.org/2025.findings-emnlp.1149/) | Findings EMNLP 2025，正式 | 周期外推位置编码，控制未见位置分布。 | 需要适配 | 周期 phase/extrapolation 先例。 |
| [MrRoPE](https://iclr.cc/virtual/2026/poster/10011844) | ICLR 2026，正式；[OpenReview](https://openreview.net/forum?id=1J63FJYJKg) | 用 mixed-radix 统一 RoPE 扩展，并提出 training-free 的 uniform/progressive radix conversion。 | 训练免费 | 2026 年强测试时频率变换 baseline；普通 progressive frequency mapping 难称新颖。 |
| [LPES](https://aclanthology.org/2026.findings-acl.1059/) | Findings ACL 2026，正式 | 为每层分配不同 RoPE scaling factor，用 Bézier 参数化和遗传算法搜索，训练免费且无推理额外延迟。 | 离线搜索，无微调 | layer-specific frequency repair 先例。 |
| [Frequency Bands / FMRoPE](https://openreview.net/pdf/4dd46cea98fadb375d28fcf897debdf638db365b.pdf) | ICLR 2026，正式 | 分析训练上下文与 RoPE base 如何共同决定被模型实际使用的频带，并以 FMRoPE 在推理时修改 base。 | 训练免费干预 | 固定或静态 frequency-band remapping 的 2026 近邻。 |
| [How Data Shapes RoPE Frequency Usage](https://arxiv.org/abs/2607.07678) | 2026，预印本 | 提出训练数据的依赖距离尺度决定已使用频率；说明 field--resolution trade-off 与测试时缩频成功/失败条件。 | 分析性工作 | 任何频率选择论证都应解释为何不是已有 scale-matching 结论。 |

### 2.4 内容依赖、可学习相位与更一般几何

| 方法 | 年份/状态 | 核心机制 | 训练要求 | 与 B/phase repair 的关系 |
|---|---|---|---|---|
| [DAPE](https://proceedings.neurips.cc/paper_files/paper/2024/hash/2f050fa9f0d898e3f265d515f50ae8f9-Abstract-Conference.html) | NeurIPS 2024，正式 | 根据输入与固定先验动态调整 data-adaptive PE，兼顾 local 与 anti-local patterns。 | 需训练 | 内容自适应 PE 先例。 |
| [CoPE](https://openreview.net/forum?id=sIGWTd1DcW) | ICLR 2025，正式 | context-conditioned positions；模型只对语义上选中的 token 增加位置计数。 | 需训练 | token-dependent position 的直接先例。 |
| [DeepSeek-V2 MLA](https://arxiv.org/abs/2405.04434) | 2024，官方技术报告 | decoupled RoPE：内容 NoPE 子通道与位置敏感 RoPE 子通道分别产生内积并相加。 | 预训练架构 | **同一 head/score 内双通道几何的强工程先例。** |
| [CARoPE](https://arxiv.org/abs/2507.23083) | 2025，预印本 | token embedding 条件化的 head-specific 动态频率/相位。 | 需从头训练 | generic “content-dependent RoPE phase” 已被覆盖。 |
| [TAPA](https://arxiv.org/abs/2509.12635) | 2025--2026，预印本 | Token-Aware Phase Attention 引入可学习 phase function，抑制 token-agnostic distance bias；用轻量 continual pretraining 扩到长上下文。 | 需预训练/继续训练 | generic token-aware phase 与 B 的直接近邻。 |
| [PaTH](https://papers.neurips.cc/paper_files/paper/2025/hash/59c27bf8d56d3d50c7aeaf7535dee975-Abstract-Conference.html) | NeurIPS 2025，正式 | 以 token-conditioned Householder 变换的连乘定义相对位置几何，可从头训练或继续训练转换 RoPE 模型。 | 需训练/继续训练 | “内容决定逐 token 相对变换路径”已有正式先例；宽泛 learned geometry claim 不可守。 |
| [Selective RoPE](https://iclr.cc/virtual/2026/poster/10011040) | ICLR 2026，正式 | 输入依赖的任意角旋转，带 phase gate；可作用于 softmax/linear transformer。 | 需训练 | 对“学习何时/如何旋转”形成最强优先权。 |
| [GRAPE](https://iclr.cc/virtual/2026/poster/10007924) | ICLR 2026，正式；[arXiv](https://arxiv.org/abs/2512.07805) | 以 group actions 统一 multiplicative rotations 与 additive biases；包含 learned commuting subspaces、non-commuting mixtures 及 content-adaptive path forms。 | 需训练 | “提出更一般双通道/群几何位置编码”的宽泛主张已非常危险。 |
| [RoPE++ / Beyond Real](https://iclr.cc/virtual/2026/poster/10010807) | ICLR 2026，正式；[arXiv](https://arxiv.org/abs/2512.07525) | 恢复通常被舍弃的复数内积虚部，形成 real + imaginary 双分量 attention；虚部分量更偏长程。 | 新架构训练 | **若 B 是两个互补相位/几何分数通道，这是最直接近邻之一。** |
| [LeRoPE](https://arxiv.org/abs/2607.10134) | 2026，预印本 | 每个 RoPE frequency 配置可学习标量。 | 需训练 | “让模型学习每频率权重”不新。 |

### 2.5 机制诊断与稀疏注意力的相关边界

| 工作 | 年份/状态 | 关键结论 | 对本项目的约束 |
|---|---|---|---|
| [Understanding RoPE Extensions from an Attention Perspective](https://aclanthology.org/2025.coling-main.600/) | COLING 2025，正式 | 保持训练长度内 attention pattern、降低位置不确定性对长程检索重要。 | 支持“先诊断 attention pattern”，但不是本项目独有解释。 |
| [Exploring Context Window of Large Language Models via Decomposed Positional Vectors](https://proceedings.neurips.cc/paper_files/paper/2024/hash/1403ab1a427050538ec59c7f570aec8b-Abstract-Conference.html) | NeurIPS 2024，正式 | 从 hidden states 中分解隐式 positional vectors，并提出 training-free positional-vector replacement 与 attention-window extension。 | 主要针对隐式/NoPE 位置而非 RoPE，但使“首个冻结模型位置修复”的宽泛 claim 也不可用。 |
| [The Rotary Position Embedding May Cause Dimension Inefficiency in Attention Heads for Long-Distance Retrieval](https://aclanthology.org/2025.findings-acl.697/) | Findings ACL 2025，正式 | 控制实验显示大角度旋转范围会使部分 attention 维度低效；三种 LLM 上这些维度也无助于长上下文问答。 | “RoPE 使远程检索维度失效”已有直接实证先例；本项目只能贡献更细的 matched intervention、跨层因果链与输出后果。 |
| [Decoupling Positional and Symbolic Attention](https://iclr.cc/virtual/2026/poster/10009178) | ICLR 2026，正式；[arXiv](https://arxiv.org/abs/2511.11579) | 给出 positional/symbolic head 的定义与互斥性，并因果控制 head 可访问频率。 | 频率/head 功能归因必须与其指标和干预对照。 |
| [Probing RoPE through Frequency Entropy](https://openreview.net/forum?id=1JZuEDq62N) | ICLR 2026，正式 | 在 rotation-pair 粒度衡量 RoPE 周期性和频带使用。 | 需要作为 frequency attribution 诊断基线。 |
| [RoPE Distinguishes Neither Positions Nor Tokens in Long Contexts, Provably](https://arxiv.org/abs/2605.15514) | 2026，预印本 | 长度增加时 locality 与 relevance consistency 失效概率趋近随机；调 base 存在位置/内容区分折衷。 | 支持问题重要性，但也要求新方法同时报告位置与 token 区分。 |
| [RoPE as Phase Modulation](https://openreview.net/forum?id=zxyMneble7) | 2026，TMLR 在审 | 从 aliasing、DC drift、深度累积与精度给出 RoPE base 上下界。 | phase 解释必须控制 aliasing 与数值精度。 |
| [When Precision Meets Position](https://openreview.net/forum?id=gwXfZ3xkUq) | TMLR 2025，正式 | BF16 可造成 RoPE 位置偏差，并提出 AnchorAttention。 | 所有 phase suppression/repair 实验都必须有 FP32 RoPE 重算控制。 |
| [FASA](https://openreview.net/forum?id=FnSgecCEwg) | ICLR 2026，正式；[arXiv](https://arxiv.org/abs/2602.03152) | 离线校准 head 的主导 RoPE frequency chunks；用低成本频块提出 token，再收集原始 K/V、保留原位置并做候选内全维精确 attention。 | proposal--consumption 分离与 frequency-aware selection 已被覆盖；SAGE 只能从选择信号与因果诊断上区分。 |
| [SALS](https://papers.neurips.cc/paper_files/paper/2025/hash/00a0ebcad584c59dbc439c2af8793638-Abstract-Conference.html) | NeurIPS 2025，正式；[arXiv](https://arxiv.org/abs/2510.24273) | 经无梯度 post-training calibration，从 pre-RoPE keys 的协方差做低秩分解；在线以 RoPE-free 低维 query/key 选 Top-K，再近似恢复 K、施加标准 RoPE 并在候选内 attention。 | **pre-RoPE/RoPE-free proposal → RoPE consumption 已被直接覆盖。** SAGE 若用全维原 K/V，只能主张 selected support 上的 exactness、不同 selector 与因果诊断。 |
| [TriAttention](https://arxiv.org/abs/2604.04921) | 2026，预印本 | 用 pre-RoPE Q/K concentration centers、三角距离偏好和范数评分 KV 重要性。 | “pre-RoPE 信息 + RoPE 三角结构选候选”也已有近邻。 |
| [Attention Score is not All You Need / VATP](https://aclanthology.org/2024.emnlp-main.1178/) | EMNLP 2024，正式 | 以 attention score 与 Value 向量的 $\ell_1$ norm 联合衡量 token 重要性。 | “attention 不是全部、Value 也重要”及最简单的 attention$\times$Value 方法已被直接覆盖。 |
| [CriticalKV](https://arxiv.org/abs/2502.03805) | ICML 2026；arXiv v2 | 从 attention 输出扰动出发，将 attention、$VW_O$ 与预训练参数纳入最坏情形扰动上界，并据此选择 KV。 | generic output-perturbation / projected-Value selection 已被覆盖；只比较 attention mass 与输出变化不新。 |
| [LaProx](https://arxiv.org/abs/2605.07234) | 2026，预印本 | 把 cache eviction 写成 layer-wise matrix approximation，显式建模 attention 与 projected Value 的乘积及跨 head 依赖，并在 Qwen3-8B 等模型上验证。 | generic attention$\times$projected-Value、跨 head/layer output-aware selection 也已拥挤。 |
| [LOCOS](https://arxiv.org/abs/2607.01002) | 2026，预印本 | 对每个位置的 $\alpha W_OV$ 投影到答案 token 的 unembedding 方向，以 needle/off-needle spatial contrast 识别非字面检索 head；在 Qwen3-8B 等模型上做 held-out 因果消融。 | 与“被注意的位置是否真的向正确答案写入”高度重合；答案方向投影、OV write-aware head detection 和 gradient interpretation 不能作为本项目独立 novelty。 |
| [Self-Attention Attribution](https://arxiv.org/abs/2004.11207) | AAAI 2021，正式 | 对 attention matrix 使用 integrated gradients 做 interaction attribution，并以 attribution 做 attention pruning。 | 对 attention score/weight 求目标梯度属于成熟归因工具；本项目若使用 $\partial m/\partial s$，新意只能来自 RoPE-specific phase mediation、matched intervention 和长程失效闭环。 |

## 3. 两个方向分别最接近的工作

### 3.1 方向 A：远程 RoPE 忽略/削弱

方向 A 不是由一篇工作覆盖，而是被四种等价分解方式包围：

| 拟议操作 | 最接近先例 | 公式/概念上的碰撞 |
|---|---|---|
| 超过距离阈值后 \(R_{\Delta}\rightarrow I\) | HoPE、p-RoPE、DroPE | HoPE/p-RoPE 在子空间上令部分旋转成为 identity；DroPE 在所有层删除 PE。 |
| local 用 RoPE，remote 用 NoPE | RNoPE-SWA、SWAN-GPT、P-RoPE | 它们在层/attention span 上明确实现 local RoPE + global NoPE。 |
| 远程距离封顶 \(g(\Delta)=\min(\Delta,W)\) | LM-Infinite、STRING、InfLLM | 前两者把 OOD 或失效距离映回有效/受限区间；InfLLM 将所有已检索远程 token 的消费距离固定为局部窗口边界 \(l_L\)。 |
| 远程分桶或渐进位置复用 | SelfExtend、AdaGroPE、LaMPE | 都保留局部高分辨率，远程使用 grouped/multi-grained/dynamic remapping。 |
| local+sinks+内容检索 remote，再以受限几何消费 | InfLLM | 已在冻结模型上把 query-conditioned 远程块检索与固定远程相对位置结合；若拟议 A 含 remote memory/候选选择，仅改变检索打分或固定距离常数会非常增量。 |
| 冻结模型推理时静态位置修复 | AdaGroPE、STRING、LaMPE、CALIOPE、MrRoPE、LPES | 均无需更新模型权重；“training-free”本身不能构成差异。 |

**方向 A 的最近单篇工作取决于是否带远程选择。** 对“局部保真、远程逐渐忽略精确距离/分桶”的纯位置方法，最近的是 **AdaGroPE**；若同时含 local+sinks、query-conditioned 远程块选择和冻结模型推理，整体最近的是 **InfLLM**。若 A 是 hard cutoff，则 LM-Infinite、HoPE 与 RNoPE-SWA 更直接；若 A 是全删或大面积 NoPE，则 DroPE 更直接。

**结论：** A 只能作为 B 的消融或失败边界，不宜单独作为 ICLR 2027 主方法。若坚持 A，唯一可能的区别不是映射函数本身，而是“远程位置抑制是否由当前内容冲突触发，并对未触发 pair 保持 exact identity”。

### 3.2 方向 B：local + global 双通道几何

先按“通道放在哪个轴上”对齐：

| 双通道轴 | 已有工作 | 已覆盖的设计 |
|---|---|---|
| **层/架构** | RNoPE-SWA、SWAN-GPT、P-RoPE、SmallThinker | local SWA-RoPE 与 global/full NoPE 或 hybrid sparse channels。 |
| **head** | Ms-PoE、NAPE | 不同 head 使用不同位置尺度，或 NoPE 与局部偏置。 |
| **维度/子空间** | DeepSeek MLA、p-RoPE、HoPE、Partial RoPE | content/NoPE 子空间与 positional/RoPE 子空间共存。 |
| **segment/chunk/距离** | BiPE、SelfExtend、DCA、AdaGroPE、LaMPE、InfLLM | intra-segment/local 精细几何与 inter-segment/global 粗几何；InfLLM 还在冻结模型中把取回的远程 token 统一映到固定局部边界距离。 |
| **复数分量/score** | RoPE++ | real 与 imaginary 两个互补 attention 分量。 |
| **token/query 条件** | DAPE、CoPE、CARoPE、TAPA、Selective RoPE、GRAPE | 内容决定位置计数、频率、phase 或 group action。 |
| **层/频率校准** | PSC、Resonance RoPE、MrRoPE、LPES | 对 phase/frequency 的逐 head、逐层、逐频带修正。 |

因此以下三个自然公式均缺乏独立新颖性：

1. \(z_{ij}=z^{content}_{ij}+z^{position}_{ij}\)：与 DeepSeek MLA 的 decoupled RoPE 及 RoPE++ 的双分量 score 同类。
2. \(z_{ij}=\lambda(\Delta)z^{RoPE}_{ij}+[1-\lambda(\Delta)]z^{NoPE}_{ij}\)：与 partial/HoPE 的子空间混合、RNoPE 的层混合和 AdaGroPE 的距离混合概念等价；若 \(\lambda\) 静态只由距离决定，风险最高。
3. \(\phi'_{ij}=\phi^{RoPE}_{ij}+\Delta\phi_{ij}\)：与 PSC 的相位校准、TAPA/CARoPE/Selective RoPE 的内容条件 phase 直接相邻。

**方向 B 的最近工作取决于最终实现：**

- 若是“local RoPE + global NoPE”：**RNoPE-SWA / SWAN-GPT**；
- 若是“同一 attention 内 content + position 两个 score”：**DeepSeek MLA**；
- 若是“局部精细 + 远程粗粒度位置映射”：**AdaGroPE / LaMPE / SelfExtend**；
- 若是“冻结模型的 local+sinks + query-conditioned remote memory + 受限远程几何”：**InfLLM**；
- 若是“两个相位几何通道”：**RoPE++**；
- 若是“内容条件化 phase”：**Selective RoPE / TAPA / CARoPE**；
- 若是“修正既有 RoPE phase”：**PSC**。

没有一个“B 的普通版本”仍处在空白区。特别地，InfLLM 使“冻结模型、局部窗口不变、内容选择远程证据、再用受限位置几何消费”也不能作为宽泛 novelty；可守差异必须落在**由 pre/post-RoPE 反事实抑制触发的最小相位干预**，而不是 local/remote 系统分工本身。

### 3.3 InfLLM、SAGE 与拟议 phase repair 的公式边界

令因果距离 \(\Delta=i-j>0\)，原始 RoPE logit 写为

\[
\ell(\Delta)=q_i^\top R_{-\Delta}k_j.
\]

- **InfLLM：** 对已检索的 remote memory token 使用固定映射 \(g(\Delta)=l_L\)，因此最终消费分数为 \(\ell(l_L)\)；它不会在远程候选上恢复原始 \(\Delta\)。所以“把所有 remote token 压到窗口边界相位”已被直接覆盖。
- **内部 SAGE：** proposal 用 full-dimensional pre-RoPE 内容分数 \(\ell(0)=q_i^\top k_j\) 选候选；随后丢弃 proposal 分数，在选中 support 上恢复原位置、以 \(\ell(\Delta)\) 消费。这里的 “exact” 只能写成 **exact on the admitted support / unmodified original-position scoring on the selected support**；它不等于 dense attention 输出，因为 softmax support 已改变。与 InfLLM 的关键区别是 **proposal 不改最终几何**。
- **FASA：** 用离线选出的少数 post-RoPE frequency chunks 提议候选，再收集原始 K/V、以原位置全维 attention 消费。它不改 phase，却已预占 query-aware frequency proposal、proposal--consumption 分离和 selected-support full-fidelity attention。
- **SALS：** 从 pre-RoPE keys 的低秩子空间计算 RoPE-free Top-K，再近似恢复 K 并施加标准 RoPE。因此“pre-RoPE proposal 后回到 RoPE”也不是 SAGE 的独占结构；SAGE 只能从全维 selector、原始 K/V、selected-support exactness、local/sink 约束和因果动机上区分。
- **拟议 phase repair：** 若最终分数是 \(\ell(\Delta+\delta_{ij})\)，真正可能的新颖部分只能是 \(\delta_{ij}\) 由当下 pre/post 抑制证据触发、在 pair/head/frequency-plane 上稀疏且最小，并对未触发交互严格为零；若 \(\delta_{ij}=l_L-\Delta\) 或只由距离决定，就退化为 InfLLM/LM-Infinite/AdaGroPE 类静态重映射。

## 4. 尚未被覆盖的明确 novelty gap

### 4.1 可守的组合式缺口

截至 2026-08-01，检索未发现如下完整组合：

1. **现成、冻结、已经用 RoPE 预训练的大模型；**
2. 在其**原生窗口内**也观测到远程 evidence 的 content score 高、post-RoPE score/rank/mass 显著下降，而非只处理超出预训练长度的 OOD；
3. 由这个**逐 query--head--candidate/block 的反事实抑制事件**触发修复，而不是根据绝对距离、输入长度或离线 head 统计静态触发；
4. 求解**最小相位干预**，只调整造成证据抑制的少数 rotation planes/head/block；
5. local pairs、未触发 remote pairs、K/V 内容和原始位置都不变；无触发时计算严格退化为原模型；
6. 不学习 router、PE、frequency 或 adapter，不需要 calibration set；
7. 给出从 attention rank/mass 到 hidden state 再到 answer logit 的**因果恢复链**；
8. 在检索任务之外，通过 order-sensitive、local syntax、code、否定与近指任务验证不会把位置敏感关系“内容化”。

可将其形式化为受约束的最小修复，而不是新的通用 PE：

\[
\min_{\delta\phi_{ij,\mathcal F}}\|\delta\phi_{ij,\mathcal F}\|_2
\quad
\text{s.t.}\quad
z^{repair}_{ij}\ge z^{target}_{ij},\;
\delta\phi=0\ \text{for local/non-triggered pairs}.
\]

其中 \(\mathcal F\) 是由实测 counterfactual suppression 定位出的少数频率平面，而不是预先固定的低频或高频集合。可行性与目标分数必须在**不使用答案/证据标签的推理时信号**下定义，否则会退化为 oracle。

还必须显式区分三层已有先例：

1. **Value-aware importance 已有。** attention$\times\|V\|$、attention$\times\|VW_O\|$ 和 output perturbation 分别已有 VATP、CriticalKV、LaProx。
2. **答案方向的 OV 写入已有。** LOCOS 已计算 $\alpha\,u_y^\top W_OV$ 并给出 gradient interpretation。
3. **目标对 attention 的梯度已有。** gradient / integrated-gradient attribution 不是新工具。

因此可守问题只能更窄：对同一个远程证据，精确分解“哪一个 RoPE 频率对的相位变化，经 softmax 和 Value 写入后，实际推动或压低了最终答案 margin”，再用冻结 support、随机频率与 matched-$L_2$ 干预做因果闭环。即使这个诊断成立，也仍需要一个**不使用 gold answer 的可部署 gate**，否则只能作为机制论文的 oracle 分析，而不能包装成推理方法。

### 4.2 必须避免的过宽 claim

不要声称：

- 首个 local/global positional encoding；
- 首个 RoPE + NoPE hybrid；
- 首个 partial/dimension-wise RoPE；
- 首个 content-dependent phase；
- 首个 phase calibration/repair；
- 首个 frequency-aware attention；
- 首个在冻结模型上 training-free remapping；
- 首个 cheap proposal + exact full-dimensional consumption；
- 首个证明 RoPE 破坏远程检索。

可以尝试但必须用限定词声称：

- 首个针对**已实测 counterfactual phase suppression event**而非距离 OOD 的冻结模型干预；
- 首个在**pair/block 级**求最小 phase correction 且对所有未触发交互给出 exact no-op 保证；
- 首个把 phase-plane 干预与证据 rank/mass、下游 answer logit 的因果恢复同时闭环；
- 首个在原生窗口与外推窗口、检索与顺序敏感任务上共同验证上述机制。

### 4.3 新颖性成立所需的四个“反例测试”

若下列任一测试成立，方法会被降级为已有方法变体：

1. **只用距离就能复现 gate：** 等价于 AdaGroPE/LaMPE/CALIOPE/LM-Infinite 类映射。
2. **固定频带消融就能复现收益：** 等价于 p-RoPE/HoPE/MrRoPE 类频带方案。
3. **学习一个小 phase MLP 就能复现：** 落入 PSC/TAPA/Selective RoPE 类。
4. **只做 content proposal 而不真正修改位置几何就能复现：** 贡献回到 SAGE/FASA/SALS/RetrievalAttention 类稀疏检索，而非新 PE。

## 5. 哪种新方法最可能被认为非增量

按风险从高到低排序：

| 风险 | 方法草案 | 审稿人最可能指出的先例 |
|---|---|---|
| **极高** | 远程 hard NoPE；超过 \(W\) 直接不旋转 | HoPE、p-RoPE、RNoPE-SWA、DroPE、LM-Infinite |
| **极高** | local RoPE + global NoPE 两路或交替层 | RNoPE-SWA、SWAN-GPT、P-RoPE、NAPE |
| **极高** | static distance-dependent RoPE/NoPE 加权和 | AdaGroPE、SelfExtend、LaMPE、HoPE/partial RoPE |
| **极高** | generic phase repair/calibration | PSC、Resonance RoPE、MrRoPE、LPES |
| **高** | learned content-dependent phase/frequency gate | TAPA、Selective RoPE、CARoPE、DAPE、CoPE、GRAPE |
| **高** | real/imaginary 或两个相位 score 通道 | RoPE++ |
| **高** | 只按维度保留一部分 RoPE | p-RoPE、HoPE、Partial RoPE、DeepSeek MLA |
| **极高** | pre-RoPE/RoPE-free proposal 后恢复 RoPE attention | SALS；若是全维而非低秩，则仍只是 selector 与 exactness 条件不同 |
| **高** | 频率感知 top-k proposal 后 exact attention | FASA；另有 TriAttention |
| **中高** | 训练免费、冻结模型的静态位置 remapping | SelfExtend、DCA、STRING、AdaGroPE、LaMPE、CALIOPE、MrRoPE、LPES |
| **中** | 无学习的 query-specific、suppression-triggered 最小相位修复 | 尚无完全相同工作，但与 PSC/TAPA/Selective/FASA 的组件距离很近，必须靠严格组合与因果证据守住。 |

**最不建议投入主线实验的方案：** “超过局部窗口后用 NoPE，并平滑混合两种分数。”它同时撞上层级、维度、距离和分数四条已有路线，最多适合作为一个强消融。

## 6. 必须比较的 baseline

### 6.1 最小可接受主表

若论文主推方向 B 的冻结 phase repair，主表至少应包含：

1. 原始 full RoPE dense attention；
2. full NoPE；
3. local unchanged + remote NoPE hard cutoff；
4. local unchanged + remote distance cap；
5. local unchanged + remote progressive grouping/bucketing；
6. p-RoPE 或 HoPE 风格的固定频带/partial-RoPE；
7. SelfExtend；
8. AdaGroPE；
9. CALIOPE 或 LaMPE（最好两者都有）；
10. InfLLM matched-budget（对齐 sinks、local window、remote token/block budget；若块化混杂过强，再加 token-level \(g(\Delta)=\min(\Delta,l_L)\) faithful proxy）；
11. MrRoPE-Pro；
12. PSC 风格的小 phase calibrator（作为“允许训练”的上界/近邻）；
13. Selective RoPE 或 TAPA 风格 content-phase（允许训练的概念近邻）；
14. RoPE++ 或 MLA-like dual-score（若宣称双通道几何）；
15. 拟议的 counterfactual sparse repair。

其中 1--11 是**冻结/测试时路线的公平对手**，12--14 是**概念最近但训练条件不同的上界或架构对手**。不能因为后者需要训练就完全不讨论。

### 6.2 若声称 local/global 架构贡献

还必须比较：

- RNoPE-SWA；
- SWAN-GPT；
- P-RoPE；
- BiPE；
- DeepSeek MLA-like decoupled RoPE；
- NAPE/head-wise NoPE+local bias；
- RoPE++。

如果无法在相同规模从头训练全部模型，至少应：

- 实现关键结构的受控小模型比较；
- 在大模型上做等价的 inference-time proxy；
- 明确把架构先例列入 related work，不能用“无法复现”绕开 novelty 对齐。

### 6.3 若同时声称稀疏或系统效率

还必须加入：

- [FASA](https://openreview.net/forum?id=FnSgecCEwg)；
- [SALS](https://papers.neurips.cc/paper_files/paper/2025/hash/00a0ebcad584c59dbc439c2af8793638-Abstract-Conference.html)；
- [QUEST](https://proceedings.mlr.press/v235/tang24l.html)；
- [SparQ Attention](https://proceedings.mlr.press/v235/ribar24a.html)；
- [Loki](https://papers.neurips.cc/paper_files/paper/2024/hash/1e027da6bec9ceb2ec37951ceeccae93-Abstract-Conference.html)；
- [RetrievalAttention](https://proceedings.neurips.cc/paper_files/paper/2025/hash/4e36d4049fb0fea195a8267c8dcd0824-Abstract-Conference.html)；
- [InfLLM](https://proceedings.neurips.cc/paper_files/paper/2024/hash/d842425e4bf79ba039352da0f658a906-Abstract-Conference.html)；
- [TriAttention](https://arxiv.org/abs/2604.04921)；
- 内部 SAGE-Post/pre-RoPE proposal + original-position post-RoPE consumption（仅在 admitted support 上 exact）；
- post-RoPE exact top-k、pre-RoPE exact top-k、union/dual-max、随机 remote + local+sinks。

若方法使用 local+sinks+remote support，至少再做同预算的 proposal/consumer 四格：

| Proposal | Consumer | 对应含义 |
|---|---|---|
| pre-RoPE \(\ell(0)\) | original-position \(\ell(\Delta)\) | SAGE-style；只改 admission |
| clipped \(\ell(l_L)\) | clipped \(\ell(l_L)\) | InfLLM-like；选择与消费均用固定远程 phase |
| clipped \(\ell(l_L)\) | original-position \(\ell(\Delta)\) | 分离“clipped selector”与“恢复原几何”的作用 |
| original post-RoPE \(\ell(\Delta)\) | original-position \(\ell(\Delta)\) | 标准 post-RoPE sparse top-k control |

另需加入 **remote-NoPE consumption**（remote 最终 logit 直接用 \(\ell(0)\)），用来判断收益来自更好的候选准入，还是仅仅来自删除远程 RoPE。

所有稀疏 baseline 必须匹配：

- 实际选中 token 数；
- local 与 sink 配额；
- prefill/decode 阶段；
- QK 扫描 FLOPs、gather bytes、KV cache bytes；
- exact softmax denominator 的候选集合；
- 端到端 latency，而非只报告选择器 kernel。

### 6.4 关键消融

| 维度 | 必做消融 |
|---|---|
| 触发信号 | 距离-only；input-length-only；pre/post score gap；rank gap；attention mass gap；随机同率 gate |
| 干预粒度 | layer；head；frequency plane；token pair；block |
| 远程几何 | NoPE；distance cap；log bucket；progressive grouping；最小 phase correction |
| 频带选择 | 固定低频；固定高频；Frequency Entropy；离线 head calibration；在线 counterfactual selection |
| 修复对象 | 只改 phase；只改 temperature/amplitude；只改 denominator；只改 candidate set；phase + candidate；SALS-style 低秩 pre-RoPE selector |
| 回退性质 | 无触发 exact no-op；soft blend；hard gate；强制 local exact RoPE |
| 数值 | BF16 cached RoPE；FP32 重算 RoPE；不同 sin/cos cache；原始 vs 重新索引位置 |
| 训练条件 | 完全冻结；短校准；LoRA/PSC；轻量 continual pretraining |

## 7. 评价协议：怎样证明不是“又一个 NIAH trick”

### 7.1 分开报告四类场景

1. **原生窗口内远程检索：** 证明问题不是单纯 RoPE OOD/extrapolation。
2. **超出预训练窗口的外推：** 与缩放、remapping、NoPE 系列比较。
3. **位置不敏感检索：** NIAH、multi-key/value、RULER retrieval。
4. **位置敏感任务：** copy/reverse、最近实体、顺序/否定、代码依赖、时序问答。

### 7.2 推荐公开 benchmark

- RULER；
- LongBench / LongBench-v2；
- InfiniteBench；
- NoLiMa 或等价的低词面重合检索；
- BabiLong/cross-chunk reasoning；
- PG19 或长文 perplexity；
- 多跳证据、代码、摘要任务；
- 自建 controlled phase-suppression benchmark，但必须和公开 benchmark 同时报告。

### 7.3 机制指标

不能只报 exact match。至少报告：

- evidence token/block 的 post-RoPE rank；
- pre-RoPE 到 post-RoPE 的 rank drop 与 score/mass drop；
- 修复前后证据 attention mass；
- top distractor margin；
- 被修复频率平面的数量和相位位移范数；
- 触发率、误触发率、local pair 改动率；
- hidden-state patching/activation replacement 对 answer logit 的恢复量；
- answer logit 与 evidence mass 的 paired effect；
- 位置敏感反例的 harm rate；
- 各 head/layer 的 paired bootstrap CI，而不是只看 pooled mean。

### 7.4 模型覆盖

最低建议：

- 两个模型家族，例如 Qwen 与 Llama/Mistral；
- 至少两个规模；
- 一个原生长上下文模型和一个通过 scaling 扩展的模型；
- 多 seed/多 query 模板；
- BF16 与 FP32 RoPE control。

只在单个 Qwen checkpoint、单种 needle 模板上成功，不足以支持 ICLR 级“新位置几何”结论。

## 8. 对论文路线的具体建议

### 8.1 方向 A 的处理

- 降为**诊断性 baseline 与失败边界**；
- 实现 hard NoPE、distance cap、AdaGroPE-like progressive bucket 三个代表点；
- 用它们回答“静态远程去位置化是否足够”；
- 预期结论应是：静态 A 能救部分检索，但会伤害顺序敏感关系，且无法精确定位实际被 RoPE 抑制的 pair。

### 8.2 方向 B 的方法收缩

优先探索“counterfactual phase rescue”，而不是“dual-channel PE”：

1. 原始 local RoPE 路径始终保留；
2. 从 pre/post-RoPE gap 生成无需训练的 conflict certificate；
3. 只在 certificate 通过的 remote candidate/block 上求解析或小步闭式 phase correction；
4. correction 受最小范数、频带稀疏与 no-op 约束；
5. 最终 attention 仍消费原始 V，并明确是改 score geometry 还是只改 candidate set；
6. 把 SAGE/FASA/SALS/InfLLM 作为 selection/remote-consumption baseline，把 PSC/Selective RoPE 作为 phase baseline。

### 8.3 go / no-go 门槛

在扩大 benchmark 前，先要求以下小规模闭环全部成立：

- 相对 full RoPE，修复显著提高已知 suppression cases 的 evidence rank/mass；
- 相对 remote NoPE、cap、AdaGroPE-like bucket，收益不是简单距离映射可解释；
- 随机 matched-rate gate 无同等收益；
- FP32 重算排除 BF16 伪影；
- local/order-sensitive harm 接近 0；
- 无触发样本逐元素等于原模型；
- phase 干预优于只扩 candidate 或只改 temperature；
- 至少两个模型家族复现。

若不能同时满足，应停止把它包装为新 PE，转回“RoPE failure diagnosis + sparse retrieval repair”的更窄论文。

## 9. 最终审计判断

1. **方向 A 单独投稿：不建议。** 其所有自然实例都已有强先例，尤其 InfLLM、AdaGroPE、RNoPE-SWA、HoPE、DroPE、LaMPE 与 CALIOPE。
2. **普通方向 B：不建议。** local/global、content/position、real/imaginary、layer/head/dimension/segment 等通道分解均已有正式工作；冻结推理场景还有 InfLLM 这一直接整体近邻。
3. **generic phase repair：不建议。** PSC 已占据名称和基本操作；Selective RoPE/TAPA/CARoPE 已占据内容条件相位；RoPE++ 已占据双相位分量。
4. **最可能形成 ICLR 2027 新颖性的路线：** 冻结模型中、由实测反事实抑制触发、对少数 pair/block/frequency planes 做最小且可回退的因果 phase rescue；主贡献必须是事件定义、最小修复算法、no-op/局部不变性质和因果闭环，而不是“又一个 PE 映射函数”。
5. **最大外部风险：** 2024--2026 已有 InfLLM、SALS，以及 ICLR 2026 同期的 FASA、Selective RoPE、RoPE++、MrRoPE、GRAPE、DroPE、Frequency Entropy 等；2026 下半年到 ICLR 2027 截稿前仍可能出现更接近的工作。公式定型后必须对最终公式再做一次逐项检索。

---

### 名称注意

- **p-RoPE**：ICLR 2025《Round and Round We Go》中的 partial RoPE/频带截断思想。
- **P-RoPE**：2026 预印本《Periodic RoPE for Infinite Context LLMs》，是 local RoPE + global NoPE 层堆叠。
- 不应在论文中混用两者缩写。
