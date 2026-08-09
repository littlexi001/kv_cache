# 面向超大外部记忆的生成状态检索：10M实测与1B外推

> **一句话结论：真实10M实验支持“层级缩域 + 地址成熟 + 稀疏页面复用 + 工作集条件效用 + scope内因果拓扑”；LongMemEval相同12页按旧到新并显示日期使NLL从4.383降到4.131，共享前缀KV分叉又把有效完整性probe从0.588s降到0.434s（1.35x），但直接KV页系统和通用检索器仍未完成；本文不做1B实测。**

## 1. 研究问题

目标不是让模型把 10M-1B tokens 一次性放进工作上下文，而是：

> 随着生成状态变化，持续从外部记忆中找到当前一步需要的少量信息，并把模型实际读取的上下文稳定控制在几百到几千 tokens。

设外部记忆含 `N` 个 tokens，切成 `B=N/L` 个 blocks。若 `L=256`：

- 10M tokens 对应约 39,062 blocks；
- 1B tokens 对应约 3.9M blocks。

因此，最终方案必须同时满足：

1. **质量：** 当前一步的必要证据进入几十个候选 blocks；
2. **动态性：** 推理状态变化后能够切换证据；
3. **复杂度：** 不能每生成一个 token 都线性扫描 3.9M blocks；
4. **工作集：** 只加载几百到几千 tokens 的 KV 或原文；
5. **通用性：** 性质要在自然 QA、新闻、代码、书籍等语料成立，而非只在合成向量上成立。

### 1.1 当前得到支持的性质总表

| 可利用性质 | 当前证据强度 | 对算法的直接含义 |
|---|---|---|
| 访问模式稀疏 | QA、XSum、PG19 支持 | 先选 `LOCAL / RETRIEVE / SCAN`，不是每步都远程检索 |
| 检索事件在时间上稀疏 | 2Wiki、MuSiQue 轨迹显著高于置换 null | 状态变化或不确定性触发检索，命中后短时复用 |
| 片段的生成效用具有短时持续性 | PG19/代码repo相邻64-token段的候选效用Spearman为0.347/0.325；前段最优窗口73.3%/76.7%在后段仍有正效用 | 命中后先KEEP并复用，效用下降或状态突变时再刷新 |
| 地址随生成逐步成熟 | Hybrid Top8 any-hit随前缀增长在XSum为26%→72%、旧PG19为16.7%→50.0%、代码为46.7%→73.3%；77本严格9.9M PG19中`book8 -> segment32`同书命中64→512为81.8%→97.4%，12胜0负，p=4.88e-4 | 早期保留多分支，地址成熟后收缩；不能假设初始query已充分表达需求 |
| 地址准确率与检索边际价值存在中间窗口 | past-only PG19中Hybrid同书命中持续升高，但512-token读取的NLL增益在状态64/128/256/512时为+0.0586/+0.0667/+0.0262/+0.0085；128显著高于512 | 检索应在“地址已可表达、recent尚未覆盖需求”时触发，而非等地址最稳定或按固定周期读取 |
| 外部记忆具有层级scope | PG19/代码同scope高效用事件率47.3%/46.6%，异scope为10.6%/11.8%；Top3 scope召回93.3%/96.7% | 先路由文档/会话/repo/文件/时间段，再做block级精排；大scope需继续分层 |
| 证据基数相对记忆规模高度稀疏 | LongMemEval共享10M中，每题平均只有2.02个证据session、6.25个精确64-token blocks，而全库有156,250 blocks | 系统应优化少量证据集合的coverage与完整性，不应把工作集大小随总记忆线性增长 |
| 层级缩域有效，但上层表示必须保真 | LongMemEval中已知owner后，BM25 session3把候选缩到平均131 blocks并达到83.3% Top8精确命中；全局E5 session8聚合反而比全局block E5低20.8个百分点全部证据召回 | 保留`owner/project -> session -> turn/block`结构和多key；不能把大scope任意均值成单向量 |
| 时间局部性是实体/关系槽位条件的，不是全局recent | 严格自然日口径下，LongMemEval knowledge-update证据的中位owner内新近排名为23.5，仅25%在latest8；multi-session为24和12.5% | 先检索`owner/entity/relation`相关写入，再在该槽位内部应用last-write/range规则；为全局recent保留独立小配额即可 |
| 拒答需要完整性与反事实检查，而不只是相关性检索 | 16条abstention中，BM25 owner-session3仍有81.2%命中官方hard-negative session，E5 owner-session8为100% | “找到了相关内容”不能推出可回答；reader前必须检查问题所需槽位是否齐全、是否只有近邻反例或旧值 |
| 层级容器允许次线性在线访问 | 100M真实PG19中Top3/Top8 book scopes只扫描0.544%/1.68% blocks，Top8同书命中保留全局BM25的92%/96% | 把全局极值竞争改写为`scope postings -> scope内block postings`，并让Top-D随router置信度变化 |
| 上层router也受1B极值竞争，深度应随地址成熟收缩 | 真实嵌套PG19上50M→100M的Top8预测72.2%、实际72.5%；后验外推1B时Top8/Top128 book recall仅55.8%/76.5%。达到80%时prefix64/128/256/512分别需Top1024/512/128/64 scopes | 一层book路由不足以支撑1B；早期保留宽scope分支，并增加`book -> chapter/segment -> block`，地址成熟后快速缩窄 |
| 第三级segment可显著缩小索引访问并改善reader | 77本严格9.9M PG19中，`book8 -> segment8`把block精排域从154,688降到约499，最终只读512 tokens；跨状态相对query-only/global BM25/E5的配对ΔNLL为-0.0557/-0.0304/-0.0277，CI均不跨0 | 层级的首要作用是结构保真、减少访问与抑制干扰；深度必须用独立query校准，30条探索中的segment32最优未在77条复现 |
| 工作集纯度可比粗召回率更重要 | 100M层级Top3虽比全局BM25少6.6个百分点Top8同书命中，但状态128/512的NLL均显著更优 | 控制器不能只优化any-hit；应联合优化coverage、候选纯度、reader效用与干扰代价 |
| scope路由错误可由score geometry预测 | 按query分组5折外推，100M中仅用规模/状态的宏AUC为0.564，加入margin、熵、score mass后为0.768 | margin/熵可控制是否保留更多scope分支，但必须用train-only校准，不能读取oracle scope |
| 地址成熟不等于score分布变尖，路由置信度也不等于读取效用 | 77本中真实scope平均排名64→512由3.52降到1.45，但归一化entropy由0.959升到0.976、Top8 mass由0.146降到0.118；router/frontier轨迹模型反而显著弱于静态策略 | 不应把margin、entropy或集合churn当作通用STOP/utility信号；coverage gate与reader utility gate必须分开 |
| 已观察token的回放loss包含弱而稳定的候选效用信号，但极值选择会过拟合 | 77本、231个状态事件中，64-token回放loss与未来action效用的事件内Spearman为0.169；直接七选一弱于状态先验，先验收缩仅改善-0.00093且CI跨0 | 回放适合做小幅校正或两动作确认，不适合无正则地扫描许多候选；只有增益超过额外forward成本时才触发 |
| 有益scope扩展稀疏且部分可预测 | 100M中只有22.8%的D扩展改善NLL；geometry+Top8 churn的分组外推AUC为0.760，oracle STOP显著胜固定D3/D8 | 默认动作应是STOP；只在预期边际效用为正时扩展，并利用连续生成中的历史反馈更新 |
| utility符号比幅度更容易预测 | 当前连续Delta NLL回归Spearman为-0.029/-0.091；最佳学习STOP相对固定D仍不显著 | 需要surprisal、hidden/Value响应或短counterfactual probe估计收益强度，不能只校准二元概率 |
| 已观察token的反事实loss能预测未来边际效用 | 严格“先检索、后观察64 tokens、再预测未来128 tokens”中，180个扩展事件的Spearman为0.271（p=2.36e-4）；状态512的符号AUC为0.740 | 把连续生成本身当成无标签反馈：少量候选并行重放最近窗口，决定EXPAND/STOP，而不是预先知道gold或未来答案 |
| 未见候选时无法辨识候选效用 | 严格PG19中，当前工作集自身的attention/uncertainty/hidden信号AUC仅0.595/0.464/0.419，连续utility相关接近0；它们没有读取新增候选 | 当前状态只能判断“是否可能需要外部信息”，不能判断“哪个未见候选有用”；至少要计算候选条件的廉价交互统计 |
| 候选条件QK能提供动态地址，但不天然优于RAG | PG19中geometry AUC 0.737；加入同状态E5候选特征为0.813，模型原生QK为0.827，QK-E5差值CI跨0 | QK适合作为模型原生候选信号或RAG补充通道，不能仅凭attention相似度替代文本检索 |
| 可寻址性与生成效用必须分开 | 真实10M代码中67.1%候选对未来64 tokens无正效用；按query聚类后无单个QK/Value特征通过FDR，模型特征选块也未超过“检索排名+repo scope” | 目标函数必须显式减去冗余与干扰；高QK只表示模型会读取候选，不保证读取会改善下一步生成 |
| 检索轨迹具有“慢scope、快block”双时间尺度 | 真实10M代码RRF的相邻Top3 scope Jaccard为0.623-0.753，逐query比Top8 block高0.309-0.361且CI全大于0；上一轮Top3 scope覆盖下一轮Top8 blocks的67.5%-91.7% | scope router可低频更新并缓存scope pages；block工作集仍需在scope内高频重排，不能把旧Top8直接冻结 |
| 粗frontier可复用但不能完全代替刷新 | XSum/PG19/代码中，上一轮RRF Top512覆盖下一轮Top8的62.5%-85.4%，历史并集提高到73.3%-89.2%；但只重排旧Top512会在部分轨迹损失4-13.3个百分点source any@8 | 维护约0.5K-1.1K候选frontier可减少全局访问；仍需由状态事件或保底周期注入新候选 |
| 生成状态会产生稀疏的跨scope地址创新，而非单调改善每次Top-K | LongMemEval全500题、8个独立10M shards中，追加64个无答案计划tokens会使平铺BM25和未知owner路由变差；但层级Top8每次只新增约1-3个64-token页面，保留历史页后可显著提高全部证据session覆盖 | 新状态结果不能直接替换旧Top-K；应以`KEEP + sparse APPEND + delayed EVICT`维护持久frontier |
| 时间复用可把多次检索限制在约1K-token工作集，但优势只在层级路由成立 | 已知owner的五状态frontier平均只含约12个唯一页面、复用约70%；Hybrid固定12页时相对静态同预算把任一证据/全部证据覆盖提高2.8/3.4个百分点；平铺全局BM25同预算反而更差 | 价值来自状态改变上层session/scope路径，而不是普通query expansion；KV页缓存可避免每步重复prefill，平铺检索继续使用静态Top-K |
| 证据条件状态能产生有用的自然地址创新，但页面效用高度不对称 | Qwen3-8B只读初始512 tokens后生成32-token事实/缺口状态，动态工作集平均673 tokens；相对静态768 tokens，全部证据覆盖提高2.55个百分点（p=.0357），但总体答案NLL与F1差异均不显著。20个rescue样本NLL改善1.044，8个loss样本恶化1.899 | 控制器应估计新页面的完整性增益和被挤出页面的机会成本；检索coverage gate不能替代reader utility gate |
| 页面变化元数据和可分离相关性都不足以预测替换效用 | 500题按8个独立10M shards留一外推；轨迹/状态特征预测动态收益AUC仅0.517-0.527，加入新旧页面词法与E5交互后仍仅0.512-0.514；最佳E5门NLL相对全静态仅-0.0015，CI跨0 | churn、跨scope和bi-encoder相似度只能提出候选，不能批准EVICT；需要显式slot-page支持/冲突、条件集合增益、cross-encoder/QK或廉价反事实probe |
| 页面效用是相对当前工作集的非可分离集合函数 | Qwen3-8B同时读取固定8页与两组候选，独立判断各工作集完整性；动态收益AUC由E5的0.514升到0.587。零阈值门以平均730 tokens把NLL从4.383降到4.262，CI `[-0.236,-0.018]`，rescue/loss选择动态70%/12.5% | 粗RAG只负责proposal；APPEND/EVICT必须比较`complete(W+new)`与`complete(W+old)`，不能给每页一个上下文无关分数 |
| 时间更新记忆具有有向版本链，正确呈现顺序是旧到新 | 72个knowledge-update中相同静态12页按旧到新排列使NLL下降0.475、F1提高10.0pp；加日期后F1总增16.8pp。旧到新相对新到旧的NLL优势0.712，CI不跨0；相同顺序消融在自然XSum Top8新闻片段上仅0.0032 NLL且CI跨0 | 对`current/latest/update`应检索`entity-relation`的多版本写入，保留时间/来源并按因果拓扑读入；普通无版本页面不强制排序 |
| 相关度负责选页，但有向scope应在reader前恢复因果拓扑 | LongMemEval全500题固定同一静态Top12：日期-only NLL仅-0.0276且CI跨0，顺序-only为-0.1641、日期+旧到新为-0.2517；固定日期时旧到新相对新到旧为-0.1748，CI不跨0。代码10M旧到新相对反向为-0.0169且CI不跨0，PG19/XSum则不显著 | 检索分数不能兼任reader排列；对会话、事件、版本和连续代码scope按内部拓扑组织，对独立文章和稀疏非链式片段保持原策略 |
| 候选工作集共享长前缀，可用KV分叉降低条件效用验证成本 | 两组完整性prompt平均逻辑长度1937 tokens，其中固定8页、问题和状态高度共享；公共前缀只算一次后执行token减少32.1%。全500延迟0.580s降到0.463s；同进程62题为0.588s降到0.434s、61/62更快，NLL门控仍显著改善 | 候选动作不必各自重复prefill整个`W`；缓存`KV(question,state,W_fixed)`，只为少量APPEND/EVICT后缀分叉，成本从`m(P+S)`降为`P+mS` |
| frontier失效可预测，但精确证据刷新更稀疏 | 三域480次状态转移中，Top8超过25%离开旧Top512的事件占36.7%，target-free特征按query外推AUC 0.834、跨域AUC 0.797；真正只靠全局刷新才能救回source的事件仅7.9%，跨域门控AUC最高0.615 | 常态重排旧frontier；用无训练query-state漂移或候选agreement触发全局刷新，并保留低频保底。不能把集合漂移门控直接当作证据utility门控 |
| 稳定hub主要是QK表示属性而非文本语料属性 | XSum两折cross-fit中，另一半query学到的Top1% hub只吸收文本RRF Top64的0.28%/0.018%提名，却吸收QK head-RRF的14.9%-25.5%/3.9%-9.1%（状态8/64） | KV索引应按layer/head/profile估计并减去hub prior；BM25/E5不应套用同样的全局hub惩罚 |
| scope效用是局部核与长程身份的混合 | 代码same-repo later候选>16K后macro平均效用约0；PG19 bidirectional同书>16K仍有用，但past-only中效用随0-4K、4-16K、16-64K、>64K逐级衰减 | recent、scope内长程和关系边三路保留独立配额；因果与infill记忆必须分开建模 |
| 当前状态是少量带角色的多向量 | MuSiQue state pointer 用更少 Q 保持召回 | 保留实体、变量、子目标等 pointer，不做全句单向量平均 |
| Query 地址具有中低维原型结构 | train/dev 原型对 test Q 的 cosine≥0.8 覆盖 98.6% | 可以建立 `head x role x prototype` 倒排，而非全库 QK 扫描 |
| Head 是异质专家，部分共识高度富集 | 跨 head 桶重叠低，4-head gold 富集 28x/77x | 用软共识提升精度，同时保留单 head rescue 分支 |
| 词法地址与模型内部地址互补具有任务条件性 | MuSiQue 上有增益，10M XSum 上直接融合反而略降 PPL | BM25/embedding 做主种子；KV 仅在跨视角一致或独立验证通过时启用 |
| 极值尾部只允许粗候选尺度稳定 | 1B 乐观外推 Top391 仍为 61.5/45.0%；近10M XSum/PG19 的 Top512 any-hit 为97%-98%/96.7%，而 Top8 明显更低 | 分层缩域到几百候选，再精排到 1-4 blocks |
| K低秩是存储性质，不等于语义可检索 | 10M新闻SVD32保留95%-99%能量；past-only PG19的56个profile平均保留87.0%，但两域的全局QK Top8都弱 | 低秩用于压缩和候选内精排，不能单凭能量保留宣称全局检索有效 |
| block内K的高同向性主要来自profile公共分量 | PG19八个真实profile中raw方向一致性0.903，减去全局均值后仅0.309 | 建索引前必须去除layer/head公共方向与hub；raw coherence不能当作block语义一致性的证据 |
| K-mean可强压缩token轴，但不是充分地址 | 64-token K-mean相对token索引压缩64倍；Top8同scope命中至多10%，token-max也未挽救；两者读取均恶化PPL | 仅在block residual小且排序margin足够时使用mean bound；自然全局检索应以文本/scope为主，QK用于缩域后验证或特殊pointer事件 |
| 有用证据具有局部片段结构 | 两个连续4-block窗口在XSum/PG19/代码上分别得到探索性最佳PPL 23.14/38.81/6.77，但配对CI均跨0 | 粗检索找锚点，工作集加载少数连续片段，而非彼此孤立的Top-K blocks |
| 主题相关排序不等于生成效用排序 | PG19中BM25/E5/Hybrid rank与未来Delta NLL的Spearman仅0.064/0.059/0.145；代码为0.147/0.087/0.147 | 粗检索保召回后，需要状态条件的效用估计器或reader/verifier精排 |
| 未来效用可由scope与短时试读联合预测，但强度依赖任务 | bidirectional PG19/代码跨域模型AUC达0.712-0.738；past-only PG19的A/B效用Spearman仍为0.243，但2-4 probe尚不稳定 | 先用结构先验富集，再自适应扩大probe预算；不要把固定2-4或静态相似度写成普适结论 |
| 探测预算兼具计算约束与统计正则化 | bidirectional PG19在4后继续搜索会过拟合；past-only则要约32个现有候选才显著胜静态Top1；3090上4-probe为45.7ms | 预算必须由候选效用密度、A/B持续性和多尺度覆盖共同决定，不能固定为越小或越大越好 |

这些性质不是都已达到论文级普适性：QA、新闻、书籍和代码repo已跨域支持地址成熟与小工作集，LongMemEval全500题已在8个独立10M shards上确认对话层级、时间frontier和8B reader闭环；但该对话语料包含模拟memory sessions，答案指标尚未使用官方LLM judge，自然日志/邮件仍未验证，真实1B不在当前实测范围。

## 2. 当前实验口径

**范围声明：本文当前主张建立在真实约10M-token实验上。不会运行或声称运行真实1B-token索引；文中1B数字只用于说明尺度边界，均明确标注为由较小真实尺度得到的统计或复杂度外推。后续新增实验也固定在10M tokens。**

### 2.1 真实 10M 外部记忆

- 9,999,872 个真实 tokens；
- 39,062 个 `256-token` blocks；
- Qwen3-0.6B 的真实 pre-RoPE Q/K；
- K 投影到冻结的 SVD32 空间；
- 28 层、16 query heads、8 KV heads；
- 不使用合成高斯向量。

### 2.2 动态自然任务

机制诊断使用 31 条自然 2WikiMQA 两跳问题。生成实验给模型第一跳 oracle evidence，再逐 token 生成 bridge，记录每个生成状态的全层全头 Q，并在完整 39,062-block 轴上精确扫描。

这个集合适合做机制诊断，但规模很小，而且 0.6B 的 bridge 生成正确率只有 `3/31`。因此：

- “生成轨迹”结果用于判断信号是否存在及如何变化；
- “oracle 状态”结果用于剥离生成错误；
- 尚不能把 31 条结果当成通用质量结论。

外部复验另使用官方 MuSiQue：

- 2,000 个自然两跳问题，train/dev/test 为 1,000/500/500；
- 10M tokens 由真实 MuSiQue 段落组成，不含高斯向量和合成文本；
- test 500 共 1,000 个显式推理步骤、15,995 个 token 级 Q 状态；
- 查询不包含 source context，第二步只写入 oracle bridge，用于单独测量“正确状态能否寻址”；
- 16 个 heads 在实验前由排除全部 LongBench MuSiQue 的 LODO fold 冻结。

MuSiQue K 索引由 Qwen3-0.6B 真实前向产生，但采用 `block_local_fallback`：每个 256-token block 独立编码，再保存 pre-RoPE K，而不是对一个单一 10M causal 序列做 full prefill。SVD32 平均保留能量为 86.4%。因此本实验严格验证的是“外部 block/KV memory bank 的寻址”，尚未验证跨 block 隐状态依赖、RoPE 重定位和把命中 KV 直接注入生成上下文后的等价性。

## 3. 新发现一：证据方向会短暂出现，而且不是随机命中

原生逐 token Q 的第二跳 RRF39：

| 指标 | 结果 |
|---|---:|
| 初始状态命中 | 0/31 |
| 任一生成状态曾命中 | 9/31，29.03% |
| 最终状态命中 | 2/31，6.45% |
| 连续命中至少 2 个状态 | 6/31，19.35% |
| 连续命中至少 3 个状态 | 2/31，6.45% |

为了排除“24 次尝试总会随机撞中”的解释，把完整第二跳 gold block 集合在 31 条轨迹间随机打乱 20,000 次：

| 指标 | 真实对齐 | 打乱均值 | lift | 经验 p |
|---|---:|---:|---:|---:|
| 任一状态曾命中 | 29.03% | 4.17% | 6.96x | <5e-5 |
| 连续至少 2 状态 | 19.35% | 1.16% | 16.63x | <5e-5 |
| 8-token 时间窗最终命中 | 19.35% | 1.05% | 18.45x | <5e-5 |

**解释：** 模型生成时确实会短暂暴露与下一步证据对齐的内部方向。它不是稳定地址，更像稀疏的“检索事件”。

但 raw ranking 的变化本身不是可靠事件信号：bridge 字符串进度发生变化的 17 个 transition，RRF39 Jaccard 为 `0.047`；其余 706 个 transition 为 `0.055`，几乎没有差别。也就是说，token-Q 每步都在变，真正的语义状态变化只是其中很小一部分。

**设计含义：** 不应只在固定间隔查询，也不应只使用最后一个 token。应研究事件检测、短时缓存和证据命中的持续性置信度。

500 条外部 MuSiQue 的结果更明显：

| 真实 10M 全扫描指标 | 结果 |
|---|---:|
| 单个初始 token 的第二跳 RRF39 | 2.0% |
| 32 个两步 token 状态中，第二跳曾进入 RRF39 | **71.2%** |
| 32 个状态中，任一选定 head 的 Top16 曾命中 | **93.2%** |
| 全扫描时间，6×3090、15,995 states、16 heads | 122.8 s |

这里的 93.2% 是“大候选层曾命中”，不是最终只读 39 blocks 的召回；71.2% 也使用了整条 oracle 轨迹的最佳时刻。它们证明信号存在，但不等于在线系统已经知道何时读取。

证据：[trajectory_temporal_properties_2wiki_10m_20260715.json](1b_context_search_research_exploration/evidence/trajectory_temporal_properties_2wiki_10m_20260715.json)

外部证据：[musique_token_trajectory_external_10m_20260715.json](1b_context_search_research_exploration/evidence/musique_token_trajectory_external_10m_20260715.json)

## 4. 新发现二：稳定性与选择性存在明确冲突

相邻原生 Q 并不稳定：

- selected-head Q cosine：`0.737`；
- 每 head Top16 Jaccard：`0.044`；
- 最终 RRF39 Jaccard：`0.055`；
- 99.45% 的相邻状态 RRF39 Jaccard 不超过 0.25。

固定检索 suffix 可以让 Q cosine 达到约 `0.995`、RRF39 Jaccard 达到 `0.56-0.60`，但第二跳召回变成 `0%`。简单 EMA 也形成单调 Pareto 曲线：

| Q 表示 | 相邻 Q cosine | RRF39 Jaccard | 第二跳曾命中 |
|---|---:|---:|---:|
| 原生 token-Q | 0.737 | 0.055 | 29.03% |
| 2-token EMA | 0.894 | 0.114 | 19.35% |
| 4-token EMA | 0.963 | 0.238 | 6.45% |
| 8-token EMA | 0.986 | 0.386 | 3.23% |

**解释：** 证据方向是高频、短暂的极值信号。平滑虽然降低抖动，也会把真正有用的尖峰一起抹掉。

**设计含义：** 需要“保留事件的稳健聚合”，而不是向量平均。例如对候选 block 做短窗 recurrence、峰值、分位数或序贯检验。

证据：[q_stability_selectivity_pareto_2wiki_10m_20260715.json](1b_context_search_research_exploration/evidence/q_stability_selectivity_pareto_2wiki_10m_20260715.json)

## 5. 新发现三：推理状态是稀疏多向量地址，不是单向量地址

为了剥离 0.6B bridge 生成错误，构造两帧 oracle 状态：

1. 原始问题状态；
2. 写入正确 bridge 后的第二步状态。

两种表示结果差异很大：

| 状态表示 | 第二跳候选层召回 | 压到 RRF39 后 |
|---|---:|---:|
| 16 个 state-token Q 直接平均 | 0% | 0% |
| 保留 16 个 token x 16 heads | 70.97% | 12.90% |

这里的 `70.97%` 是“至少一个 token-head 的 Top16 命中”，候选并集最多可达 4,096 blocks，不是最终工作集召回。它说明：

1. 正确状态里通常存在证据方向；
2. 方向只存在于少数 token-head 通道；
3. 全 token 平均会抵消稀疏方向；
4. 全通道 RRF 又会让大量无关方向淹没证据。

这与 block K-mean 失败具有相同数学原因：我们要找的是支持函数极值，不能用均值代替。

证据：[oracle_multivector_raw_2wiki_10m_20260715.json](1b_context_search_research_exploration/evidence/oracle_multivector_raw_2wiki_10m_20260715.json)

## 6. 新发现四：当前状态指针可以无标签地路由 Q token

对第一步只使用当前 lookup entity 对应的 Q；对第二步只使用已经写入 compact state 的 bridge entity 对应 Q。该规则不读取 gold block 或最终答案。

| token 路由 | 平均第二步 Q 数 | 第一跳 RRF39 | 第二跳 RRF39 |
|---|---:|---:|---:|
| 全部 token | 16.00 | 16.13% | 12.90% |
| 最后 4 tokens | 4.00 | 6.45% | 25.81% |
| 最后 8 tokens | 8.00 | 19.35% | 25.81% |
| state pointer | 7.45 | **22.58%** | **29.03%** |

state-pointer 第二跳相对打乱证据的 null mean 为 `1.02%`，lift `28.46x`，`p<5e-5`。

在外部 MuSiQue test 500 上，按“当前步骤”聚合而不是挑选整条轨迹的最佳时刻：

| token 路由 | 第一步平均 Q 数 | 第二步平均 Q 数 | 第一跳 RRF39 | 第二跳 RRF39 | 两步都命中 |
|---|---:|---:|---:|---:|---:|
| 全部 state tokens | 15.99 | 16.00 | 66.8% | 46.4% | 33.2% |
| state pointer | **8.14** | **5.79** | **68.6%** | **46.8%** | **35.0%** |

因此，指针路由分别减少约 49% 和 64% 的 Q 方向，同时没有牺牲召回。随机打乱 gold block 后，第一、二跳均值只有 `0.261%` 和 `0.247%`；真实 lift 为 `262.8x` 和 `189.1x`，均 `p<5e-5`。

**解释：** 生成状态中的“当前实体/变量/约束”是可利用的稀疏寻址坐标。自然语言状态不应整体压成一个 embedding，而应保留一组具有角色的指针。

**通用化：** 指针不必只指实体。在代码中可以是变量、函数、错误位置；在新闻中可以是事件、时间、人物；在数学中可以是当前子目标、符号和约束。

证据：[state_pointer_routing_raw_2wiki_10m_20260715.json](1b_context_search_research_exploration/evidence/state_pointer_routing_raw_2wiki_10m_20260715.json)

外部证据：[musique_state_pointer_external_10m_20260715.json](1b_context_search_research_exploration/evidence/musique_state_pointer_external_10m_20260715.json)

### 6.1 新发现五：状态指针 Q 本身具有可学习的低维原型结构

只使用 MuSiQue train+dev 的状态指针 Q 学习每个 head 的 128 个球面 k-means 原型，再在完全隔离的 test 500 上测最近原型：

| 指标（16 heads 宏平均） | 结果 |
|---|---:|
| train/dev pointer vectors / head | 19,767 |
| test pointer vectors / head | 6,963 |
| SVD32 中 rank90 | 19.75 |
| 有效秩 | 20.04 |
| test 最近原型 cosine 均值 | **0.895** |
| test 最近原型 cosine 的 p05 均值 | 0.787 |
| test 覆盖率，cosine≥0.8 | **90.7%** |
| test 覆盖率，cosine≥0.9 | 47.9% |

**解释：** 生成状态虽然组合很多，但模型用于外部寻址的方向并没有充满 32 维球面，而是集中在可由有限原型覆盖的中低维区域。这为 `head x state-role x prototype` 倒排表提供了依据：在线先把少数 pointer Q 路由到最近原型，再访问这些原型的 block postings，而不是扫描全部 3.9M blocks。

**限制：** 原型覆盖只证明 query 可路由，不保证对应 postings 一定包含正确 block。128 原型的第一版实验印证了这一点：最近一个原型的 postings 并集平均覆盖 58.5% blocks，第一/二跳 gold 仍在并集中的概率为 80.2%/68.0%，但压到 1% blocks 后只有 6.6%/4.0%。原型 cosine 高，不代表 block score margin 足以容忍量化误差。

证据：[musique_pointer_query_manifold_20260715.json](1b_context_search_research_exploration/evidence/musique_pointer_query_manifold_20260715.json)

### 6.2 属性转化实验：细原型、稀疏 head 和 hub-IDF 可以形成有效倒排

离线对每个 `head x prototype` 计算 `g_b(p)=max_i <p,k_i>`，只保存 Top512 block postings；在线把每个 state pointer Q 路由到最近原型，用 posting-rank RRF 和跨原型 document frequency 的 IDF 去掉 universal hubs。所有原型只由 MuSiQue train+dev 学习，test gold 不参与选择。

固定最多访问 391 blocks，即完整 10M block 轴的 1.00%：

| query 原型数 / heads | 第一跳候选 recall | 第二跳候选 recall | 结论 |
|---|---:|---:|---|
| 128 / 16 | 6.6% | 4.0% | 原型过粗，失败 |
| 512 / 16 | 18.0% | 13.2% | 增加分辨率有效 |
| 512 / 8 | 19.6% | 18.0% | 少数强 heads 优于全 16 heads |
| 2048 / 8 | **35.6%** | **41.0%** | 形成可用的粗候选层 |

2048 原型将 test 最近原型 cosine 提高到 `0.926`，cosine≥0.8 覆盖率为 `98.6%`。8-head routed postings 的原始并集仍有 42.2% blocks，但含 gold 的概率为第一跳 87.2%、第二跳 88.2%；hub-IDF 把它压到 1% 后得到上表结果。

再对 391 个候选执行真实 SVD32 QK 精排：

| 最终加载 blocks | tokens | 第一跳 recall | 第二跳 recall | 两步都命中 |
|---:|---:|---:|---:|---:|
| 1 | 256 | 28.8% | 18.0% | 6.0% |
| 3 | 768 | 32.0% | 29.2% | 9.2% |
| 4 | 1,024 | **32.0%** | **31.2%** | **9.8%** |
| 8 | 2,048 | 33.4% | 33.0% | 10.6% |
| 16 | 4,096 | 34.2% | 35.2% | 11.6% |
| 39 | 9,984 | 34.8% | 36.8% | 12.4% |

同样 8 heads 的完整 10M 全扫描 Top4 为第一跳 40.8%、第二跳 27.6%、两步同时 13.0%；Top39 为 62.4%/47.0%/31.2%。因此：

1. 1% candidate route 在第一跳仍有明显损失；
2. 第二跳 Top4 反而高于 full-scan Top4，说明粗筛有时能去掉 QK/RRF hub 干扰；
3. 从 Top4 增加到 Top39 收益很小，大部分可恢复证据已集中在前 1K tokens；
4. 主要剩余瓶颈已从“精排”前移到“候选生成”和“两步复合成功率”。

本次 2048-prototype/8-head postings 为 49MB，prototype 文件为 2.1MB。6×3090 对 1000 个步骤精排 391→39 blocks 的 wall time 为 14.1s；实验版 CPU routing 同时计算五种聚合器耗时 6.8s。两者都不包含 Q profile 生成，也不是冷盘端到端系统时间。

### 6.3 与 BM25 的配对互补性

同一 10M corpus、同一 MuSiQue test 500、同一显式 step state 上：

| 方法 | 最多加载 tokens | 第一跳 recall | 第二跳 recall | 两步都命中 |
|---|---:|---:|---:|---:|
| BM25 Top3 | 768 | **71.6%** | 60.2% | 42.8% |
| 动态 KV Top4 | 1,024 | 32.0% | 31.2% | 9.8% |
| BM25 Top3 ∪ KV Top4 | 1,792 | **75.4%** | **68.8%** | **52.0%** |
| BM25 Top16 | 4,096 | 86.4% | 89.8% | 77.6% |
| BM25 Top16 ∪ KV Top4 | 5,120 | 87.2% | 90.4% | 79.0% |

KV Top4 单独很弱，但能额外救回 3.8% 的第一跳和 8.6% 的第二跳 BM25-Top3 失败样本。由此得到的正确工程判断是：

- 文本可寻址场景优先走 BM25/embedding；
- 动态 KV 通道作为 model-state-conditioned residual channel；
- 两通道一致时高置信加载，不一致时才扩展、验证或回退；
- 论文贡献应定位在动态内部状态、KV 复用和互补失败集，而不是宣称检索质量胜过 RAG。

10M warm BM25 postings 查询约为 0.69ms/step，当前实验版 KV routing+rerank 约为 20.9ms/step，而且还未计 Q 生成；10M 上 BM25 在质量和速度上都更强。KV 路线能否在 100M-1B、频繁迭代状态和免重复 prefill 场景取得系统优势，仍需端到端证明。

证据：[musique_pointer_prototype_postings_p2048_h8_10m_20260715.json](1b_context_search_research_exploration/evidence/musique_pointer_prototype_postings_p2048_h8_10m_20260715.json)

精排证据：[musique_pointer_prototype_p2048_h8_exact_rerank_10m_20260715.json](1b_context_search_research_exploration/evidence/musique_pointer_prototype_p2048_h8_exact_rerank_10m_20260715.json)

公平全扫描证据：[musique_state_pointer_external_h8_10m_20260715.json](1b_context_search_research_exploration/evidence/musique_state_pointer_external_h8_10m_20260715.json)

互补性证据：[musique_bm25_kv_complementarity_10m_20260715.json](1b_context_search_research_exploration/evidence/musique_bm25_kv_complementarity_10m_20260715.json)

BM25 证据：[musique_official_bm25_top3_test500_v3.json](1b_context_search_research_exploration/evidence/musique_official_bm25_top3_test500_v3.json)、[musique_official_bm25_online_profile_v2.json](1b_context_search_research_exploration/evidence/musique_official_bm25_online_profile_v2.json)

### 6.4 规模外推：全局固定 Top-K 不是可扩展属性

为了回答“10M 上能找到，增加 100 倍真实干扰后是否还能找到”，先做一个有意偏乐观的反证诊断：对每个步骤允许用 gold 挑选 8 个最佳 token-head 方向，再根据真实 10M 中每个方向超过 gold 的 block 比率，用 Jeffreys Beta 后验和 beta-binomial posterior predictive 增加同分布干扰 blocks。

10M 中，最佳方向的 gold rank 已很强：第一跳中位数为 3，第二跳为 6；但固定小 Top-K 仍会被 1B 的极值尾部吞没：

| 全局候选预算 | 10M 第一/二跳 | 100M 预测 | 1B 预测 |
|---:|---:|---:|---:|
| Top4 | 63.4% / 45.8% | 24.9% / 14.3% | **8.1% / 4.5%** |
| Top16 | 81.2% / 70.8% | 45.6% / 29.7% | 16.4% / 9.2% |
| Top39 | 88.4% / 81.8% | 61.5% / 45.0% | 25.0% / 14.5% |
| Top391 | 97.6% / 98.4% | 88.6% / 82.3% | **61.5% / 45.0%** |
| Top512 | 98.2% / 98.8% | 90.5% / 86.1% | 65.9% / 49.9% |

这不是实际 1B 运行结果，而是基于真实 10M distractor tail 的后验预测；它还使用 gold 选方向并部分忽略跨 head 相关性，因此偏乐观。即便如此，全局 Top4 仍明显崩溃，足以否定“相关 block 与无关 block 存在固定绝对 margin，所以固定 Top-K 可直接扩展”的假设。

真正得到支持的是更弱但有用的性质：**某些 token-head 方向能把 gold 稳定推入几百个粗候选，但不能直接推入最终几个 blocks。** 因而 1B 检索必须先通过文档、时间、符号、词法或 prototype postings 把比较域缩小，再执行 QK 精排；不能在 3.9M blocks 上直接取 Top4。

证据：[musique_gold_rank_scale_law_10m_to_1b_20260715.json](1b_context_search_research_exploration/evidence/musique_gold_rank_scale_law_10m_to_1b_20260715.json)

### 6.5 跨 head 共识：高富集、低召回，适合作为软证据

2048-prototype/8-head postings 并未被极少数 universal hubs 完全垄断：全局 block posting frequency 的 Gini 为 `0.298`，最频繁 1% blocks 只占 4.46% posting entries。不过单 head 内 Gini 为 `0.531`，说明每个 head 仍有自己的高频块。

随机两个 prototype 桶的 Jaccard：同 head 为 2.68%，跨 head 仅 0.71%。因此不同 head 的偶然交集很小，而 gold block 往往被多个 head 同时支持：第一跳支持 head 数中位数为 3，第二跳为 2。

| 最少支持 heads | 第一跳候选占比 / gold recall / 富集 | 第二跳候选占比 / gold recall / 富集 |
|---:|---:|---:|
| 1 | 43.10% / 87.2% / 2.0x | 41.27% / 88.2% / 2.1x |
| 2 | 15.21% / 73.2% / 4.8x | 10.79% / 70.2% / 6.5x |
| 3 | 4.56% / 52.0% / 11.4x | 2.15% / 46.8% / 21.7x |
| 4 | **1.11% / 31.2% / 28.0x** | **0.38% / 29.0% / 77.0x** |
| 5 | 0.21% / 16.2% / 76.4x | 0.06% / 10.4% / 178.8x |

这里的“富集”是 gold recall 除以同等比例随机 block 的命中概率。它说明跨 head concurrence 是很强的置信特征；但若硬性要求 4 个 heads，一半以上 gold 会被删除，所以它不能作为唯一候选门槛。更合理的聚合是：

```text
candidate_score = soft_head_concurrence + posting_rank + lexical_or_symbolic_agreement - hub_penalty
```

此外候选规模存在长尾：第一跳 4-head 共识的中位数只有 92 blocks，但均值 435、p95 为 1,436。系统必须设置绝对 candidate cap，并把“桶异常膨胀”视为低置信度事件，而不能只按固定 head 阈值读取。

证据：[musique_prototype_posting_structure_p2048_h8_10m_20260715.json](1b_context_search_research_exploration/evidence/musique_prototype_posting_structure_p2048_h8_10m_20260715.json)

## 7. 新发现六：静态 block prior 可分离，但会发生查询分布漂移

此前 480 个真实 LongBench queries 上：

- query-independent `head x block` prior 平均解释约 49.8% QK 分数方差；
- cross-fitted z-score 能去掉 universal hubs，并提高同分布 RRF39；
- 冻结 16 heads 后可获得 4.06x 扫描加速，K 存储降至 7.59%。

但在新的 oracle step-state Q 上，使用完全排除 2WikiMQA 的 train-only prior：

- state-pointer 第二跳 RRF39 从 `29.03%` 降到 `6.45%`；
- 所有 token 的候选层召回也从 `70.97%` 降到 `61.29%`。

**解释：** `mu(head, block)` 并非与 query 分布完全独立。普通 QA query、生成 token-Q、显式 step-state Q 的分布不同，固定均值/方差可能过度校正。

**设计含义：** prior 要么使用分布鲁棒的解析变换，要么在线缓慢更新，并按状态类型或 query family 条件化。不能把同一数据分布上的 z-score 收益直接外推到长程生成。

这一现象与 ParisKV 强调的 decoding drift 属于同一类风险，但这里的漂移发生在 query-state 分布而不只是新生成 K 的聚类中心。

证据：[state_pointer_routing_zscore_2wiki_10m_20260715.json](1b_context_search_research_exploration/evidence/state_pointer_routing_zscore_2wiki_10m_20260715.json)

## 8. 与此前静态属性实验合并

### 8.1 得到支持的静态性质

1. **Head 功能稀疏且异质。** 不同 heads 的 Top-K 重叠很低，少数 retrieval heads 可跨问题复用。
2. **K residual 中低秩。** 真实 block residual K 的 rank90 约为 14-21，但统一 rank16 并非无损。
3. **位置局部连续性。** 去除公共方向后，同一真实文档的相邻 block centroid 更相似，可建立分层局部索引。
4. **查询方向具有中等低维流形。** SVD32 查询方向有效秩约 11.6、rank90 约 18.1，但原型覆盖不足以给出有用的最坏情况界。
5. **小工作集可能优于 full context。** 真实 XSum 新闻中，512-token `Recent+Hybrid` 的 PPL 为 30.21，优于 Full-40K 的 31.63；延迟新闻恢复中 E5-512 为 21.33，优于 Full-40K 的 26.92。外部信息的边际效用高度集中，移除无关上下文还能减轻 attention dilution。
6. **多视角共识具有乘法富集。** 4-head concurrence 在只保留 1.11%/0.38% blocks 时，对第一/二跳 gold 分别产生 28.0x/77.0x 富集；但不同 heads 仍以互补为主，必须软聚合。
7. **极值尾部决定尺度，而不是平均相似度。** 1B 外推中固定 Top4 即使在 oracle-best 方向下也只剩 8.1%/4.5%；几百个粗候选比最终几个候选更具有尺度稳定性。

新闻证据：[xsum_news_40k_working_set_20260713.json](1b_context_search_research_exploration/evidence/xsum_news_40k_working_set_20260713.json)

### 8.2 已被否定或显著削弱的假设

1. **一个 K-mean 就是 block 语义地址：否。** K concentration 约 0.52，但不同 blocks 的 K-mean cosine 仍高达约 0.84-0.93，稀疏实体方向被公共方向淹没。
2. **四段均值足够：否。** Recall@16 仍不可用。
3. **中心-半径安全剪枝：工程上无效。** 为保证零漏召回平均保留 99.9985% blocks。
4. **固定检索 probe 能兼得稳定和召回：否。** 排名稳定但召回为 0。
5. **简单时间平均可以降噪：否。** 稳定性提高时召回单调下降。
6. **全 head/全 token 多数投票可靠：否。** 公共 hub 与句法方向淹没少数语义方向。
7. **跨 head 硬交集可以同时保持高召回和高压缩：否。** 4-head 共识富集很高，但 gold recall 只有 31.2%/29.0%。
8. **10M 的全局固定 Top-K 可以等比例扩展到 1B：否。** 干扰项增加会提高极值上界，固定小 K 的生存率快速下降。
9. **SVD32 能量保留高就意味着 QK 能做全局语义搜索：否。** 10M新闻K保留95%-99%能量，但raw-QK Top8 any-hit只有7%，PPL不优于Query-only。
10. **QA中的词法/KV互补可以直接泛化到自然生成：否。** XSum上始终加入QK使PPL从Hybrid的23.67变为23.80；只有极少数跨视角一致事件显示高效用。

### 8.3 通用任务不是一种检索模式

现有真实新闻、书籍和 QA 结果共同表明，不能把所有长上下文统一成“远程 Top-K 检索”：

| 访问模式 | 真实例子 | 当前证据 | 合理动作 |
|---|---|---|---|
| 局部连续读取 | 连续新闻、小说续写、代码局部编辑 | XSum 中 Recent512 优于 Full40K；但 6 本 PG19 上 2,048-token recent PPL 14.73，仍差于 Full32K 的 13.85 | 保留 recent/local，不频繁远程检索 |
| 稀疏远程读取 | 多跳 QA、延迟新闻恢复、跨文件符号引用 | MuSiQue 的必要事实集中在少数 blocks；XSum delayed 的 E5-512 PPL 21.33，优于 Full40K 的 26.92 | 先粗召回，再逐步加载 1-3 blocks |
| 全局聚合读取 | 全书摘要、计数、全库一致性、跨项目重构 | 当前尚无充分实验；理论上单个 Top-K 无法覆盖全局统计 | 分层摘要、流式 scan 或 map-reduce，而非强行稀疏点查 |

PG19 实验只覆盖 6 本书、每本 256 个评测 tokens，因此它是反例诊断，不是稳定 benchmark 结论。不过它已经说明“几千 tokens 永远足够”并不成立；动态 echo 在这组局部续写上与静态选择 PPL 相同，却把在线时间从约 0.26s 增到 7.9s，也说明检索事件必须有触发条件。

由此得到一个更广泛的系统属性：**每一步需要的信息通常稀疏，但稀疏的类型不同。** 通用控制器首先应在 `LOCAL / RETRIEVE / SCAN` 三种模式间路由，然后才决定用什么地址和预算。

书籍证据：[pg19_32k_working_set_20260714.json](1b_context_search_research_exploration/evidence/pg19_32k_working_set_20260714.json)

### 8.4 无问题模板的真实 10M 新闻：地址成熟与粗候选生存

为了避免结论只来自两跳 QA，新实验使用自然新闻 continuation，而不是问题/答案模板：

- 9,999,488 个 distractor tokens 来自 XSum train 的 20,731 篇真实 BBC 新闻；
- 100 条 source/query/target 来自完全不重叠的 XSum test；
- 每条 test 新闻的前 512 tokens 是远程 source，随后 64 tokens 是已经生成的自然前缀，再预测后续 128 tokens；
- 40K、1M、10M 使用同一批 queries 和嵌套 distractor 前缀；每个虚拟记忆额外包含当前 query 的 8 个 source blocks；
- 检索只能看已出现的 prefix，不能看 target；block 为 64 tokens，最终读取预算固定为 8 blocks/512 tokens。

该协议没有合成文本，但为了获得可测 gold，确实控制了 source 与真实干扰新闻的排列；它是自然文本上的受控远程恢复，不等于完全原生新闻流。

#### 地址成熟：更多已生成内容显著提高寻址能力

在同一个 10M 记忆中：

| prefix tokens | BM25 Top8 any-hit | E5 Top8 any-hit | BM25+E5 Top8 any-hit | Hybrid source-block recall |
|---:|---:|---:|---:|---:|
| 8 | 22% | 25% | 26% | 7.38% |
| 16 | 25% | 29% | 33% | 9.50% |
| 32 | 34% | 42% | 49% | 15.63% |
| 64 | 53% | 65% | **72%** | **23.50%** |

Hybrid 从 8 到 64 tokens 有 48 条新命中、2 条丢失，Top8 any-hit 增加 46 个百分点，配对精确 `p=2.27e-12`；source-block recall 增加 16.13 个百分点，bootstrap 95% CI 为 `[12.13, 20.38]`。prefix 与 E5 gold 最佳 rank 的 Spearman 为 `-0.453`，`p=1.12e-21`。

这证明状态轨迹不只是噪声来源：**已生成内容会逐步暴露实体、事件和主题，使外部地址变得更可分。** 但候选仍持续变化：10M Hybrid 相邻 prefix 的 Top8 Jaccard 只有 `0.222/0.245/0.307`。早期 8→16 的命中保持率为 76.9%，到 32→64 才提高到 93.9%。因此应在早期保留多分支，地址成熟后再收缩工作集。

#### 尺度：最终 Top8 不稳定，Top512 粗候选仍高召回

使用完整 64-token prefix：

| 记忆规模 | 方法 | Top8 any-hit | Top8 source recall | Top512 any-hit | Top512 source recall |
|---:|---|---:|---:|---:|---:|
| 40K | E5 | 99% | 68.13% | 100% | 100% |
| 1M | E5 | 87% | 37.38% | - | 86.38% |
| 10M | E5 | 65% | 21.00% | **98%** | **63.25%** |
| 10M | BM25+E5 | **72%** | **23.50%** | 97% | 61.75% |

从 40K 增到 10M 时，E5 Top8 命中从 99% 降到 65%，34 条配对样本丢失、0 条新增，`p=1.16e-10`。这是真实新闻上的实际尺度实验，不是后验外推，并与 MuSiQue 极值模型给出相同结论：**几百个粗候选能保持较高覆盖，最终几个候选不能直接跨尺度保持。**

#### 下游效用：512 tokens 有用，但错误读取会伤害模型

10M 中固定读取 512 tokens 后预测 128-token continuation：

| 方法 | PPL | 相对 Query-only 平均 Delta NLL | source any-hit |
|---|---:|---:|---:|
| Oracle source512 | **20.39** | -0.2746 | 100% |
| E5 | **23.61** | -0.1280 | 65% |
| BM25+E5 | 23.67 | -0.1255 | 72% |
| BM25 | 24.57 | -0.0881 | 53% |
| Query only | 26.83 | 0 | 0% |
| raw SVD32 QK | 26.95 | +0.0043 | 7% |
| Random512 | 28.87 | +0.0731 | 0% |

E5 恢复了 oracle NLL 改善量的约 46.6%。E5 命中任一 source 时平均 `Delta NLL=-0.198`，未命中时约为 `+0.001`；随机读取则显著伤害模型。这支持“外部信息效用稀疏且具有符号”的性质：读对少量证据有益，读入无关信息并非中性，因此工作集控制器需要 `READ/KEEP/EVICT/STOP`，而不是固定填满预算。

#### 局部片段结构：锚点应该扩展成少数连续窗口

source 的8个blocks本来连续，但普通Top8把它们当成8个独立点。新的无训练局部聚合器不改变BM25/E5分数，而是在Top512内给连续窗口累计RRF质量，再按同一个512-token预算选择一个8-block窗口、两个4-block窗口或四个2-block窗口：

| Hybrid工作集形状 | PPL | source-block recall | any-hit |
|---|---:|---:|---:|
| 8个独立Top blocks | 23.666 | 23.50% | **72%** |
| 1×8-block连续窗口 | 23.458 | **42.63%** | 43% |
| **2×4-block连续窗口** | **23.139** | 35.75% | 57% |
| 4×2-block连续窗口 | 23.702 | 26.50% | 62% |

两个4-block窗口相对独立Top8把source recall提高12.25个百分点，bootstrap 95% CI为`[7.88,16.88]`，但any-hit下降15点，CI为`[-23,-8]`。PPL的`Delta NLL=-0.0225`，95% CI `[-0.0517,0.0057]`仍跨0，因此23.139是探索性最优，不是统计显著胜出。

这个Pareto揭示了一个重要结构：**证据效用更接近少数完整局部片段，而不是最大化“碰到任意一个相关block”的问题覆盖率。** 1×8窗口过于集中，4×2又过于碎片，2×4在片段完整性和多分支覆盖之间取得当前最佳平衡。对KV memory而言，这意味着索引可以只找锚点，然后按原位置批量加载相邻KV；但该结果受控source天然连续，仍需在原生PG19叙事复现和代码定义/调用链上验证。

局部工作集结果：[xsum_10m_retrieval_ppl_locality_20260715.json](1b_context_search_research_exploration/evidence/xsum_10m_retrieval_ppl_locality_20260715.json)

局部性配对分析：[xsum_10m_locality_working_set_analysis_20260715.json](1b_context_search_research_exploration/evidence/xsum_10m_locality_working_set_analysis_20260715.json)

#### QK 反例与仍可利用的稀疏事件

四通道真实 K 的 SVD32 保留能量为 `94.96%-99.44%`，10M 索引为 2.56GB；8 卡正式 profiling 约 47s。但 raw max-QK 在 10M/64-token prefix 上 Top8 any-hit 只有 7%，PPL 26.95，K 去公共均值后的 cosine 也没有改善。由此应明确：

> K 能低秩压缩，不代表当前 Q 能把语义相关 block 从海量自然文本中排到前面。

不过 QK 命中的 7 条平均 `Delta NLL=-0.407`，未命中的 93 条为 `+0.035`，说明它是稀疏高效用事件而非均匀弱信号。一个无 gold 的校准现象是：当 QK Top8 与 E5 Top8 至少重合 2 个 blocks 时，仅触发 6 条，其中 5 条命中 source，Fisher `p=1.64e-6`。但据此在 QK 与 Hybrid 间门控，PPL 只从 23.666 到 23.566，配对 `Delta NLL=-0.0042`、95% CI `[-0.0212, 0.0069]`，尚无显著收益。

因此跨视角 agreement 和跨时间 recurrence 可以用于识别 rare QK events，但目前只是**可校准性证据**，不是“KV 已超过 RAG”的结果。

文本检索证据：[xsum_10m_dynamic_text_retrieval_20260715.json](1b_context_search_research_exploration/evidence/xsum_10m_dynamic_text_retrieval_20260715.json)

QK索引证据：[xsum_10m_svd32_k_profile_20260715.json](1b_context_search_research_exploration/evidence/xsum_10m_svd32_k_profile_20260715.json)

QK检索证据：[xsum_10m_dynamic_qk_retrieval_20260715.json](1b_context_search_research_exploration/evidence/xsum_10m_dynamic_qk_retrieval_20260715.json)

配对性质证据：[xsum_10m_dynamic_properties_20260715.json](1b_context_search_research_exploration/evidence/xsum_10m_dynamic_properties_20260715.json)

PPL证据：[xsum_10m_retrieval_ppl_20260715.json](1b_context_search_research_exploration/evidence/xsum_10m_retrieval_ppl_20260715.json)

### 8.5 跨域到真实长篇书籍：地址成熟复现，片段读取方向复现

为检查上述性质是否只是短新闻的产物，新增PG19长篇书籍continuation实验：

- 30条query分别来自30本真实书籍，每条为连续的`512-token source + 64-token query + 128-token target`；
- 外部记忆由PG19 test的100本书、128个不重复真实文本片段组成；受测source/query/target区间从base中严格删除，再按query单独放回source；
- 同一本书的其他远端部分允许存在，用于保留小说的层级叙事结构，但不存在受测target泄漏；
- PG19 test去泄漏后的唯一文本不足严格10M，因此最大规模为9,900,032 tokens，而不是复制文本凑数；
- 与XSum相同，按64-token blocks检索，最终reader只读取512 tokens，且检索不使用target。

#### 地址成熟与候选规模

在9.9M外部记忆上：

| 64-token prefix方法 | Top8 source recall | Top8 any-hit | Top512 source recall | Top512 any-hit |
|---|---:|---:|---:|---:|
| BM25 | 6.67% | 30.0% | 34.58% | 80.0% |
| E5 | 8.33% | 40.0% | 43.33% | 96.7% |
| BM25+E5 RRF | **11.25%** | **50.0%** | **48.33%** | **96.7%** |

对Hybrid而言，prefix从8增至64 tokens时Top8 any-hit从16.7%增至50.0%，10胜0负，exact `p=0.00195`；source recall提高7.92个百分点，bootstrap 95% CI为`[3.33,12.92]`。E5也从6.7%增至40.0%，10胜0负。该现象与XSum一致：**当前生成状态不是静态query的附属文本，而是在逐步形成更可区分的地址。**

但尺度问题同样复现。Hybrid/64-token prefix从40K扩到9.9M时，Top8 any-hit由86.7%降至50.0%，11负0胜，`p=0.00098`；候选身份的Top8 Jaccard只有0.080。与此同时Top512仍保留96.7% any-hit。这再次支持“两级压缩”：先把15.5万个blocks缩到约512个高召回候选，再用更贵的状态条件信号压到数个局部片段。

#### 512-token工作集的生成效用

使用Qwen3-0.6B读取选出的512 tokens，并预测随后128 tokens：

| 工作集 | PPL | source-block recall | any-hit |
|---|---:|---:|---:|
| Query only | 44.20 | 0 | 0 |
| Random512 | 47.20 | 0 | 0 |
| BM25 Top8 | 41.31 | 6.67% | 30.0% |
| E5 Top8 | 39.59 | 8.33% | 40.0% |
| Hybrid Top8 | 39.17 | 11.25% | **50.0%** |
| Hybrid两个连续4-block片段 | **38.81** | **18.33%** | 36.7% |
| Oracle source512 | 36.50 | 100% | 100% |

Hybrid相对Query-only的平均`Delta NLL=-0.1208`，bootstrap 95% CI为`[-0.2354,-0.0384]`，说明近10M书籍中512-token检索工作集确实改善自然续写，而随机读取会恶化PPL。两个4-block片段相对离散Top8进一步得到`Delta NLL=-0.0090`、20胜10负，但95% CI为`[-0.0379,0.0219]`，尚不显著；它把source完整度提高7.08个百分点，CI为`[1.25,13.33]`，同时牺牲13.33个百分点any-hit。

因此跨域后可以保留但应降级表述为：**“少数连续片段”是比“孤立Top-K blocks”更合适的工作集归纳偏置，目前在新闻和书籍上均有方向性质量收益；最佳窗口数和长度仍需由状态与置信度自适应决定。**

数据协议：[pg19_9p9m_continuation_protocol_20260715.json](1b_context_search_research_exploration/evidence/pg19_9p9m_continuation_protocol_20260715.json)

文本检索：[pg19_9p9m_dynamic_text_retrieval_20260715.json](1b_context_search_research_exploration/evidence/pg19_9p9m_dynamic_text_retrieval_20260715.json)

动态配对分析：[pg19_9p9m_dynamic_properties_20260715.json](1b_context_search_research_exploration/evidence/pg19_9p9m_dynamic_properties_20260715.json)

PPL与局部片段：[pg19_9p9m_retrieval_ppl_locality_20260715.json](1b_context_search_research_exploration/evidence/pg19_9p9m_retrieval_ppl_locality_20260715.json)

局部性配对分析：[pg19_9p9m_locality_working_set_analysis_20260715.json](1b_context_search_research_exploration/evidence/pg19_9p9m_locality_working_set_analysis_20260715.json)

### 8.6 不依赖预定义source的效用地形：短时持续、可替代片段与层级scope

source recall只能说明“找回了人为指定的片段”，不能证明片段真的改善下一步生成。为此增加target-NLL效用探针：

1. 每条PG19 query先由BM25、E5和Hybrid各取Top64，再加入64个随机窗口；去重后平均121.5个候选；
2. 每个候选扩展成4个连续blocks，即256 tokens；候选生成不使用target；
3. 将真实128-token continuation拆成已观察段A和未来段B，各64 tokens；
4. 用窗口对A的`Delta NLL`作为已经发生后的效用观测，只在完全未参与选择的B上评估是否持续。

这是**属性探针而非可部署算法**：离线遍历全部候选来测A的效用仍然太贵；它回答的是“过去的真实生成效用能否预测未来效用”，从而判断KEEP/EVICT控制是否有可学习信号。

#### 性质一：效用存在显著但非永久的短时持续

- 同一窗口在A与B上的效用，逐query平均Spearman为`0.347`，跨query bootstrap 95% CI为`[0.273,0.424]`；
- A上最优的检索窗口有`73.3%`在B上仍为正效用；
- 它有`43.3%`进入B效用Top 10%，相对10%随机基准的binomial `p=2.30e-6`；
- 仍有26.7%的A最优窗口在B上转为负效用，说明“命中后永久保留”同样错误。

| 用于未来B的256-token窗口 | B段PPL | 相对Query-only的平均NLL改善 |
|---|---:|---:|
| Query only | 44.16 | 0 |
| 静态E5 Top1 | 42.52 | +0.0378 |
| 用A选择BM25/E5/Hybrid候选 | **40.90** | **+0.0767**，CI `[0.0369,0.1197]` |
| 用A选择、但排除指定source | 42.31 | +0.0428，CI `[0.0078,0.0815]` |
| 用A从随机候选选择 | 44.38 | -0.0048，CI跨0 |
| 用未来B直接选候选 | 32.37 | target-leaking诊断上界，不可部署 |

A条件选择相对静态E5 Top1进一步改善NLL `0.0389`，95% CI `[0.0095,0.0711]`，20胜8负；相对静态Hybrid Top1改善`0.0705`，CI `[0.0245,0.1240]`。随机候选即使用A选择也不能改善B，说明持续性不是“从很多随机窗口里取最大值”的普遍过拟合。

#### 性质二：自然效用不是唯一gold，而是可替代片段族

平均121.5个检索窗口中：

- B上正效用窗口占38.5%；
- 超过该query随机窗口95分位的占19.6%，平均23.4个；
- 正效用质量的participation-ratio有效支持数为29.25，即候选的24.0%；
- 其中平均22.4个高效用窗口不重叠预定义source。

因此“每一步全库中客观上只有1-3个有用blocks”不是通用事实。更准确的结构是：**外部记忆中可能存在几十个功能上可替代的局部片段，但当前工作集只需选择其中少数；难点是效用排序，不是把全部有用片段读入。** 这也解释了为何source recall和PPL并不严格单调对应。

BM25、E5、Hybrid的候选rank与未来`Delta NLL`的平均Spearman分别只有`0.064/0.059/0.145`。而future-B oracle在同一候选池可达PPL 32.37，说明主要损失已从“候选池没有信息”转移到“相关性分数不能识别真正的生成效用”。

#### 性质三：scope层级是强先验，但必须校正文档长度

重建PG19 memory的完整来源并逐token核对9,899,520个base tokens后，只有0.078%的blocks跨书边界。得到：

- 检索候选中同书窗口占24.6%，随机窗口只有0.10%；
- “B效用超过随机95分位”的事件率，同书为47.3%，异书为10.6%；odds ratio `7.54`，Fisher `p=1.05e-109`；
- 同一query内，同书窗口相对异书窗口平均多改善`0.0902 NLL`，CI `[0.0593,0.1216]`；
- 用A在同书候选中选窗，B段PPL 40.53；只在异书中选为45.79，配对NLL优势`0.1221`，CI `[0.0749,0.1726]`，22胜7负。

把Top512 block质量按书聚合时，原始求和会偏向长书；按文档block数平方根归一化后，Hybrid的目标书Top1/Top3召回为`76.7%/93.3%`。Top3文档平均含5,120 blocks，把154,680-block全库缩小`30.2x`。但在保存的Top512排名中直接过滤后再取Top8，Hybrid any-hit仅从50.0%增至53.3%，E5从40.0%增至46.7%。所以层级scope解决的是**比较域规模**，不会自动解决scope内部的效用精排。

这个性质可自然推广：书籍对应文档，代码对应repository/file/module，Agent记忆对应session/episode/tool，日志对应service/time range。真正通用的索引应先估计当前状态所属的少数scope，再在scope内部使用词法、语义、QK和效用模型联合精排。

效用地形：[pg19_9p9m_candidate_utility_landscape_20260715.json](1b_context_search_research_exploration/evidence/pg19_9p9m_candidate_utility_landscape_20260715.json)

定性成功/失败案例：[pg19_9p9m_candidate_utility_examples_20260715.json](1b_context_search_research_exploration/evidence/pg19_9p9m_candidate_utility_examples_20260715.json)

逐token来源校验：[pg19_9p9m_block_provenance_20260715.json](1b_context_search_research_exploration/evidence/pg19_9p9m_block_provenance_20260715.json)

层级与效用分析：[pg19_9p9m_candidate_provenance_analysis_20260715.json](1b_context_search_research_exploration/evidence/pg19_9p9m_candidate_provenance_analysis_20260715.json)

### 8.7 第三自然域：10M真实代码仓库continuation

为避免性质只在自然语言中成立，使用LongBench-v2的50个`Code Repository Understanding`真实repository上下文构造代码域记忆：

- 30个query repo从合格repo中按固定seed随机抽取，不再选择最短scope；
- 每个repo固定从第4096 token开始取`512 source + 64 query + 128 target`，位置和检索均不使用target；
- 受测区间从base删除，再按query单独放回source；
- 10,000,000 tokens对应156,250个64-token blocks，只有0.010%的base blocks跨repo边界；
- context包含源码、README和仓库文档，因此代表真实mixed repository memory，不等同于纯AST benchmark。

#### 地址成熟、尺度与512-token工作集

10M/64-token prefix结果：

| 方法 | Top8 source recall | Top8 any-hit | Top512 source recall | Top512 any-hit |
|---|---:|---:|---:|---:|
| BM25 | 25.00% | 66.7% | 46.67% | 93.3% |
| E5 | 24.58% | 63.3% | 49.17% | 86.7% |
| BM25+E5 RRF | **25.83%** | **73.3%** | **55.00%** | **93.3%** |

Hybrid的prefix 8→64使Top8 any-hit从46.7%增至73.3%，11胜3负，exact `p=0.057`；source recall提高13.33个百分点，bootstrap 95% CI `[5.82,21.67]`。BM25和E5的any-hit分别从36.7%→66.7%和36.7%→63.3%，`p=0.022/0.021`。因此“生成状态逐步形成地址”已在新闻、书籍、代码repo三种无问题模板continuation上方向一致。

使用Qwen3-0.6B预测随后128 tokens：

| 512-token工作集 | PPL | source recall | any-hit |
|---|---:|---:|---:|
| Query only | 8.56 | 0 | 0 |
| Random512 | 9.65 | 0 | 0 |
| BM25 Top8 | 7.33 | 25.00% | 66.7% |
| E5 Top8 | 7.55 | 24.58% | 63.3% |
| Hybrid Top8 | 7.07 | **25.83%** | **73.3%** |
| Hybrid两个连续4-block片段 | **6.77** | 23.33% | 43.3% |
| Oracle source512 | 6.51 | 100% | 100% |

Hybrid相对Query-only的`Delta NLL` CI为`[-0.297,-0.098]`，随机512则显著恶化。两个4-block片段相对离散Hybrid的`Delta NLL=-0.0433`，但CI `[-0.178,0.069]`仍跨0；它在source recall更低时得到更好PPL，再次证明预定义source并不覆盖全部自然效用。

#### 代码效用持续与repository scope

沿用A/B各64 tokens的held-out效用协议：

- 候选A/B效用Spearman为`0.325`，CI `[0.257,0.392]`；
- A最优窗口76.7%在B仍为正效用，70.0%进入B效用Top10%，binomial `p=5.80e-15`；
- A条件选择把B段PPL从8.70降至7.05，NLL改善`0.2104`，CI `[0.1031,0.3448]`；
- 随机候选经A选择仍为PPL 8.86；
- 正效用participation-ratio支持数18.84，占候选16.3%，仍是“可替代片段族”而非唯一gold；
- BM25/E5/Hybrid rank与B效用Spearman只有`0.147/0.087/0.147`。

repository来源进一步得到：

- 检索候选中同repo占16.8%，随机占2.66%；
- 高效用事件率同repo为46.6%，异repo为11.8%，odds ratio `6.54`，`p=6.47e-75`；
- 同repo候选相对异repo平均多改善`0.1755 NLL`，CI `[0.1112,0.2569]`；
- A条件同repo选择的B段PPL为6.52，异repo为8.42，配对优势`0.2551 NLL`，CI `[0.1365,0.3960]`，23胜6负；
- 长度归一化Hybrid repo路由Top1/Top3为86.7%/96.7%。

但Top3 repo平均包含39,851 blocks，只缩域3.9倍；代码repo远大于PG19单书，说明代码必须使用`repo -> file/module -> symbol/definition -> local span`多级scope。直接过滤旧Top512再取Top8也几乎不增加any-hit，仍需在缩域后重新计算symbol/QK/utility精排。

数据协议：[longbench_code_10m_continuation_protocol_20260715.json](1b_context_search_research_exploration/evidence/longbench_code_10m_continuation_protocol_20260715.json)

文本检索：[longbench_code_10m_dynamic_text_retrieval_20260715.json](1b_context_search_research_exploration/evidence/longbench_code_10m_dynamic_text_retrieval_20260715.json)

动态配对分析：[longbench_code_10m_dynamic_properties_20260715.json](1b_context_search_research_exploration/evidence/longbench_code_10m_dynamic_properties_20260715.json)

PPL与局部片段：[longbench_code_10m_retrieval_ppl_locality_20260715.json](1b_context_search_research_exploration/evidence/longbench_code_10m_retrieval_ppl_locality_20260715.json)

局部性配对分析：[longbench_code_10m_locality_analysis_20260715.json](1b_context_search_research_exploration/evidence/longbench_code_10m_locality_analysis_20260715.json)

效用地形：[longbench_code_10m_candidate_utility_landscape_20260715.json](1b_context_search_research_exploration/evidence/longbench_code_10m_candidate_utility_landscape_20260715.json)

repository scope分析：[longbench_code_10m_scope_analysis_20260715.json](1b_context_search_research_exploration/evidence/longbench_code_10m_scope_analysis_20260715.json)

### 8.8 跨域在线效用门控：先缩域，再有限试读

前面的效用地形证明了信号存在，但每题离线遍历约120个候选并不构成可用系统。这里进一步问：**不用未来B段，能否只看静态rank、当前scope，以及候选对已经观察到的A段的影响，预测它对紧接着B段的效用？**

协议如下：

- PG19与代码各30条query；每个候选是256-token连续窗口；
- A为当前已经观察到的64 tokens，B为随后的64 tokens；
- 标签是候选B段效用是否超过该query随机窗口的95分位数；B从不进入特征或选择规则；
- `rank/scope`不需要候选前向；`observed utility`需要把候选短暂接入A段计算一次counterfactual Delta NLL；
- 在PG19训练逻辑回归后直接测试代码，反向再从代码训练、测试PG19，不做目标域微调。

跨域候选事件预测结果：

| 特征 | PG19 -> 代码 AUC / AP | 代码 -> PG19 AUC / AP | 额外候选试读 |
|---|---:|---:|---:|
| 静态BM25/E5/Hybrid rank与agreement | 0.586 / 0.277 | 0.553 / 0.259 | 0 |
| 当前scope是否一致 | 0.666 / 0.303 | 0.712 / 0.354 | 0 |
| A段实测效用 | 0.663 / 0.366 | 0.716 / 0.388 | 每个候选1次 |
| rank + scope + A段效用 | **0.712 / 0.420** | **0.738 / 0.434** | 每个候选1次 |

这说明静态相似度rank几乎不能独立回答“这个block现在读了是否有用”；可跨域的两个主信号反而是：**当前状态属于哪个scope**，以及**这个block刚才是否实际降低了模型loss**。不过，对所有候选试读仍然太贵，因此又测试了有界策略：先按`same scope -> Hybrid rank`排序，只试读前2/4个候选，然后选A段效用最高者。

| 策略 | PG19未来B段PPL | 代码未来B段PPL | 相对原Hybrid Top1的胜/负/平 |
|---|---:|---:|---:|
| 原Hybrid Top1，无试读 | 43.89 | 7.94 | - |
| scope优先Top1，无试读 | 42.49 | 7.53 | PG19 7/4/19；代码 3/2/25 |
| scope优先，试读2个后选优 | 41.76 | 7.26 | PG19 15/6/9；代码 10/4/16 |
| scope优先，试读4个后选优 | **40.64** | 7.24 | PG19 17/7/6；代码 12/8/10 |
| scope优先，试读8个后选优 | 41.30 | 7.33 | PG19 19/8/3；代码 13/10/7 |

PG19的4-probe相对原Hybrid Top1平均改善`0.0769 NLL`，95% bootstrap CI `[0.0326,0.1287]`；代码方向一致，但30条下CI跨0。使用“首次`Delta NLL_A >= 0.05`就停止”的同一规则，PG19平均只试读2.17个候选、PPL 41.88；代码平均3.00个、PPL 7.49。阈值0.1在代码更好，说明绝对loss阈值仍有域漂移，简单除以query-only NLL也没有完全消除。

更重要的是，PG19从4个继续搜索到8/16个，PPL反而回升到41.30/41.38。原因不是更多信息必然有害，而是从越来越多候选中最大化短A段效用，会更容易选中**只对A偶然有利、对B不持续的极值**，即winner's curse。有限预算因此不只是省计算，也是在正则化瞬时效用估计。

当前可支持的在线机制是：

```text
scope posterior缩域
  -> 静态多通道粗排
  -> 并行试读2-4个候选的短A段
  -> 首个达到KEEP阈值者进入工作集
  -> 后续token上继续监测效用，下降时EVICT/REFRESH
```

在单张RTX 3090上用真实代码候选做微基准：每候选输入为256-token窗口 + 64-token状态 + 64-token已观察A段，Qwen3-0.6B的1/2/4/8/16候选批量试读中位延迟分别为`32.96/34.46/45.73/91.00/171.86 ms`。因此四候选并行试读只比单候选增加约12.8ms，吞吐从30.3升至87.5 candidates/s，峰值显存1.41GiB。这个数字只含reader forward与LM head，不含10M粗检索、KV/text I/O和后续生成；它证明小预算探测在0.6B上可承受，不能直接外推到8B或端到端系统。

这还不是零成本方案：每次试读仍需短reader forward；而且这里的same-scope是受控continuation中的已知元数据，真实系统需要从生成状态预测scope。已确认的属性是“效用可以被结构先验富集，并在很短时间内持续”，尚未确认的是最便宜的无标签代理信号。

跨域效用与有界探测证据：[cross_domain_utility_gates_20260715.json](1b_context_search_research_exploration/evidence/cross_domain_utility_gates_20260715.json)

候选试读微基准：[candidate_utility_probe_benchmark_20260715.json](1b_context_search_research_exploration/evidence/candidate_utility_probe_benchmark_20260715.json)

### 8.9 scope不是平坦标签：局部邻域与长程身份是两条不同路径

同book/repo效用富集仍可能有混淆：它究竟来自全局scope身份，还是仅因为检索到了被挖去片段附近的文本？利用构造时保存的原始token位置，把所有检索候选拆成`heldout source / same-scope before / same-scope after / other-scope`，并进一步按距挖空边界的token距离分桶。事件仍定义为B段效用超过本query随机候选95分位数。

| 域与区域 | 候选数 / 覆盖query | 效用事件率（micro / query-macro） | 平均Delta NLL_B（micro / query-macro） |
|---|---:|---:|---:|
| PG19 other book | 2785 / 30 | 10.6% / 10.1% | -0.0367 / -0.0434 |
| PG19 same book, before | 77 / 21 | 53.2% / 56.6% | +0.0581 / +0.0607 |
| PG19 same book, after | 690 / 27 | 45.8% / 47.9% | +0.0308 / +0.0443 |
| Code other repo | 2951 / 30 | 11.8% / 11.5% | -0.0701 / -0.0634 |
| Code same repo, before | 110 / 6 | 54.5% / 59.0% | +0.0723 / +0.0816 |
| Code same repo, after | 378 / 7 | 38.4% / 42.3% | +0.0263 / +0.0343 |

两域都确认same-scope不是简单由heldout source候选造成：排除source后，same-scope仍显著高于other-scope。但距离分解显示两种不同结构：

- **代码更像局部依赖核。** same-repo later context在0-256 tokens的query-macro事件率87.5%、平均`Delta NLL_B=+0.3383`；超过16K后降至28.3%，平均效用为`-0.0074`。repository身份仍提高命中概率，但真正强效用主要集中在近邻定义、调用和局部文档。
- **书籍同时存在局部核和长程身份。** same-book later context在0-256 tokens的macro事件率75.0%、平均效用`+0.1249`；超过16K仍有48.4%事件率和`+0.0641`平均效用，远高于other-book。这可能来自人物、叙事主题、文体和重复情节，而不是单纯token邻近。

因此更通用的先验不是单一`same_scope` bit，而是多尺度混合：

```text
p_useful(b | z_t)
  = local_kernel(relative_position, adjacency, time)
  + scope_identity(document, repo, session, entity)
  + relational_route(symbol, citation, call edge, event link)
```

算法上应保留独立候选配额：一条取recent/adjacent局部片段，一条在当前scope内做长程检索，一条沿显式关系边跳转；最后再由在线效用门控合并。若只用距离，会漏掉PG19的长程同书效用；若只用repo/book标签，又会在代码中保留大量超过16K且平均无益的块。

这个分析也暴露了当前benchmark边界：候选来自BM25/E5/Hybrid而非全库均匀抽样，代码same-repo距离桶仅覆盖6-7条query；受控数据还包含被挖空目标之后的later context。对静态代码库，later只是拼接顺序，不等于真实未来；对严格因果书籍生成，它不能作为可访问历史。因此下一版自然实验必须构造**past-only memory**，并把later-context结果单独报告，避免把infill能力误写成因果长期记忆能力。

PG19距离证据：[pg19_10m_distance_scope_20260715.json](1b_context_search_research_exploration/evidence/pg19_10m_distance_scope_20260715.json)

代码距离证据：[longbench_code_10m_distance_scope_20260715.json](1b_context_search_research_exploration/evidence/longbench_code_10m_distance_scope_20260715.json)

### 8.10 严格past-only自然长篇：因果地址成熟成立，但2-4 probe并不普适

为消除前述later-context偏差，新增严格因果PG19实验：

- 30本test书籍作为query book，50本disjoint validation书籍只作真实干扰；
- 每题保留最近512 tokens为本地状态，预测随后128 tokens；
- 外部记忆只含query book更早的32K-131K历史和其他书籍，不含本地状态、target或该书任何未来文本；
- 记忆精确为9,900,032真实tokens，即154,688个64-token blocks；无重复填充、无合成文本、无预定义source；
- 所有30本query history完整写入，程序验证future block违规数为0。

#### 因果地址随状态成熟

| 最近状态长度 | BM25同书Top8 / 占比 | E5同书Top8 / 占比 | Hybrid同书Top8 / 占比 |
|---:|---:|---:|---:|
| 64 | 66.7% / 30.8% | 76.7% / 38.3% | 73.3% / 49.2% |
| 128 | 86.7% / 46.7% | 73.3% / 45.0% | 90.0% / 61.7% |
| 256 | 90.0% / 63.3% | 83.3% / 58.8% | 93.3% / 69.2% |
| 512 | 96.7% / 69.2% | 90.0% / 62.1% | **100% / 77.5%** |

Hybrid从64到512 tokens有8条新增同书Top8命中、0条丢失，exact `p=0.0078`；同书占比平均增加28.3个百分点，bootstrap CI `[17.5,40.0]`。BM25为9胜0负，`p=0.0039`。同时Hybrid相邻状态Top8 Jaccard只有`0.25-0.29`：状态增长不是简单累积旧结果，而是在不断替换具体地址，但替换后的候选越来越集中于正确scope。

#### 512-token远程读取质量

使用512-token本地状态，再从9.9M外部记忆加载512 tokens：

| 方法 | 预测随后128-token PPL | 相对Query-only Delta NLL 95% CI |
|---|---:|---:|
| Query only | 22.38 | - |
| Random512 | 22.88 | `[+0.0093,+0.0347]`，显著恶化 |
| BM25 Top8 | 22.25 | `[-0.0256,+0.0149]` |
| E5 Top8 | 22.08 | `[-0.0334,+0.0074]` |
| Hybrid Top8 | 21.96 | `[-0.0454,+0.0075]` |
| E5两个连续4-block片段 | **21.86** | `[-0.0533,+0.0041]` |

方向与旧PG19一致，但30题下检索方法CI仍跨0。随机读取显著有害，再次证明工作集不应被固定预算强行填满；同书Top8达到100%却仍只有有限PPL改善，也证明**scope正确不等于块级效用正确**。

#### 地址成熟与读取价值存在“中间窗口”

固定使用Hybrid两个连续4-block片段，对不同长度的最近状态分别计算Query-only与retrieval增益：

| 最近状态 | Hybrid同书Top8 | Query-only PPL | 读取512 tokens后PPL | NLL改善 / 95% CI |
|---:|---:|---:|---:|---:|
| 64 | 73.3% | 26.40 | 24.89 | +0.0586 / `[0.0070,0.1136]` |
| 128 | 90.0% | 24.98 | **23.37** | **+0.0667** / `[0.0247,0.1125]` |
| 256 | 93.3% | 23.33 | 22.72 | +0.0262 / `[-0.0036,0.0578]` |
| 512 | 100% | 22.38 | 22.19 | +0.0085 / `[-0.0115,0.0295]` |

地址准确率随状态单调提高，但边际读取价值在128 tokens达到峰值，之后下降。128相对512的检索增益差为`+0.0582 NLL`，CI `[0.0183,0.1009]`，21条更偏向128、9条更偏向512。这不是“检索越晚越好”：早期64 tokens地址尚不充分；中期128 tokens已经能指向正确历史，同时recent自身还不够；到512 tokens时本地状态更强，远程块的新增信息被部分吸收。

因此检索触发应同时估计两个量：`address_information(t)`是否足够，以及`residual_need(t | recent)`是否仍然较高。二者乘积而非任一单调指标决定读取价值：

```text
expected_read_value(t) ~= address_confidence(t) * residual_information_need(t)
```

这个“中间窗口”直接支持事件触发检索，也解释了为什么等模型生成很久再检索，虽然地址更准，却可能已经错过最有价值的读取时机。

#### 无gold因果效用地形

每题扫描BM25/E5/Hybrid Top64并集约120个候选窗口，target按A/B各64 tokens拆分：

- A/B候选效用Spearman为`0.243`，CI `[0.172,0.307]`；
- A最优候选70.0%在B仍为正效用，36.7%进入B效用Top10%，`p=8.91e-5`；
- BM25/E5/Hybrid rank与未来B效用Spearman仅`0.062/0.023/0.047`；
- 用A选择全并集候选，把B段PPL从21.23降至20.33，NLL改善CI `[0.0207,0.0689]`；
- 对随机候选做同样A选择仍为PPL 21.26，说明提升不是“尝试够多总会挑中”；
- 诊断性oracle-B为18.57，表明当前候选中仍存在大量尚未被在线信号识别的可用空间。

排除任何source标签后，同书候选的未来效用事件率为27.3%，异书仅6.2%，odds ratio `5.64`，`p=3.39e-35`。但效用随过去距离衰减：

| 同书距离 | 事件率 | 平均Delta NLL_B |
|---|---:|---:|
| 0-4K | 37.8% | +0.0287 |
| 4K-16K | 35.4% | +0.0203 |
| 16K-64K | 20.9% | +0.0075 |
| >64K | 8.4% | -0.0013 |

这修正了bidirectional PG19中“>16K同书效用仍强”的结论：严格因果历史里确实有长程同书增益，但它呈明显衰减，而不是平坦的book prior。

#### 小预算反例与多尺度覆盖

静态Hybrid Top1的B段PPL为20.90。按原rank试读2/4个后为20.67/20.78，CI均跨0；约32个才显著降到20.32。仅按距离取最近同书块更糟：无试读Top1为21.18，因为最近512 tokens已在本地状态中，紧邻历史可能冗余。按`same scope -> distance -> rank`排序后，试读8/16个可降至20.39/20.19，改善CI分别约`[0.00003,0.0509]`和`[0.0067,0.0612]`。

因此，“2-4 probe足够”只在先前bidirectional任务成立，不是通用属性。past-only自然生成更接近：**recent形成冗余区，真正互补的历史分布在数个距离尺度；在当前弱效用估计下，需要覆盖8-16个结构化候选，或学习更强的零/低成本utility prior。** 这是一条比固定K更有价值的系统规律。

数据协议：[pg19_past_only_10m_protocol_20260715.json](1b_context_search_research_exploration/evidence/pg19_past_only_10m_protocol_20260715.json)

文本检索：[pg19_past_only_10m_text_retrieval_20260715.json](1b_context_search_research_exploration/evidence/pg19_past_only_10m_text_retrieval_20260715.json)

动态配对分析：[pg19_past_only_10m_dynamic_properties_20260715.json](1b_context_search_research_exploration/evidence/pg19_past_only_10m_dynamic_properties_20260715.json)

PPL：[pg19_past_only_10m_ppl_20260715.json](1b_context_search_research_exploration/evidence/pg19_past_only_10m_ppl_20260715.json)

状态长度与检索价值：[pg19_past_only_10m_state_value_20260715.json](1b_context_search_research_exploration/evidence/pg19_past_only_10m_state_value_20260715.json)

因果效用地形：[pg19_past_only_10m_utility_20260715.json](1b_context_search_research_exploration/evidence/pg19_past_only_10m_utility_20260715.json)

scope、距离与有界探测：[pg19_past_only_10m_utility_structure_20260715.json](1b_context_search_research_exploration/evidence/pg19_past_only_10m_utility_structure_20260715.json)

### 8.11 真实K-mean假设检验：高同向性不等于可寻址性

这一节直接检验“同一block的K大致同向，因此可用当前Q与`k_mean`做矩阵检索”的假设。实验仍使用8.10节的严格past-only PG19：9,900,032个真实tokens、154,688个64-token blocks、30个因果query，不含target、未来文本、合成向量或预定义source。为隔离RoPE位置搬移问题，统一使用pre-RoPE Q/K。

- 对28层、每层2个代表性KV组，共56个`layer x KV-group` profile建立SVD32 K-mean索引；
- 每个profile的SVD由1024个历史blocks拟合，平均能量保留87.03%，范围74.50%-98.81%；
- 56-profile K-mean索引为554.4MB；8个代表profile的token-level索引为5.07GB，而相同8个profile的K-mean估算仅79.2MB，即token轴精确压缩64倍；
- query只使用当前状态最后64/128/256/512 tokens产生的Q，不读取target；最终读取预算固定为Top8，即512 tokens。

#### K为何看起来高度同向

对8个token-level profile，先计算每个block内token K的方向一致性：

| K口径 | mean | median | p10-p90 |
|---|---:|---:|---:|
| raw projected K | **0.903** | 0.908 | 0.839-0.991 |
| 减去profile全局均值 | **0.309** | 0.305 | 0.180-0.441 |

去均值后平均下降0.594。这说明raw K确实“看起来同向”，但主要原因是同一层/head共享一个很强的query-independent公共方向，而不是每个block形成了清晰、彼此可区分的内容方向。这个公共方向适合被压缩或显式剔除，却不能承担block地址。

#### K-mean与token-max都没有形成可靠全局地址

以“Top8中是否出现当前书籍历史block”为无需gold source的scope指标；随机Top8的期望any-hit为4.46%、平均同书占比为0.57%。状态512时：

| 方法 | Top8同书any-hit | Top8同书占比 |
|---|---:|---:|
| K-mean 56-profile max | 3.3% | 0.4% |
| K-mean 56-profile RRF | **10.0%** | 1.3% |
| K-mean 8-profile max / RRF | 3.3% / 6.7% | 0.4% / 0.8% |
| token-max 8-profile max / RRF | 0% / 3.3% | 0% / 0.4% |

最好的单一K-mean profile在全部30题上可达16.7% any-hit，但train前15题与test后15题的profile质量Spearman只有0.104；按train选出的最优profile在test为0%，Top4 profile在test平均5%。这不是一个可稳定路由的“语义head”。更重要的是，K-mean与token-max的Top8 Jaccard接近0；均值不保留token极值，但恢复token极值同样没有得到可靠地址，所以主要瓶颈不只是mean压缩损失，而是自然续写状态Q与全局历史K之间缺少稳定的内容寻址对齐。

#### 最终判据是生成效用，而非几何自洽

固定读取512 tokens并预测随后128 tokens：

| 状态 | Query-only | Random | BM25+E5 | K-mean56 max | K-mean56 RRF | token-max8 max |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 24.98 | 25.95 | **23.74** | 26.16 | 26.48 | 25.73 |
| 512 | 22.38 | 22.88 | **21.96** | 22.90 | 23.31 | 23.04 |

状态128时Hybrid相对Query-only的NLL变化为`-0.0508`，95% CI `[-0.1011,-0.0058]`，显著有益；K-mean56 max为`+0.0462`，CI `[+0.0133,+0.0797]`，显著有害。状态512时K-mean和token-max也全部显著恶化。因而当前自然数据支持的结论是：

1. **K低秩、raw同向和可压缩均成立，但三者都不是可检索性的充分条件；**
2. **K-mean压缩会丢失token极值，但token-max失败说明这不是首要矛盾；**
3. **文本/scope索引仍应承担自然内容的主召回，QK应退到候选内验证、直接KV加载前的page bound，或实体/变量pointer等经过门控的稀疏事件；**
4. **该反例不否定所有KV索引：目前只覆盖pre-RoPE、2个代表KV组/层和8个token-level profile，尚未检验全部query heads、post-RoPE相对位置校准与专门训练的retrieval heads。**

K profile与存储：[pg19_past_only_kmean_profile_20260715.json](1b_context_search_research_exploration/evidence/pg19_past_only_kmean_profile_20260715.json)

QK检索：[pg19_past_only_kmean_qk_retrieval_20260715.json](1b_context_search_research_exploration/evidence/pg19_past_only_kmean_qk_retrieval_20260715.json)

几何、稳定性与PPL汇总：[pg19_past_only_kmean_geometry_20260715.json](1b_context_search_research_exploration/evidence/pg19_past_only_kmean_geometry_20260715.json)

### 8.12 真实100M压力测试：层级缩域保留大部分召回并改善生成质量

为实际检验“scope能否阻止固定Top-K随总库增长而崩坏”，把8.10节9.9M严格因果PG19记忆扩展到100,000,000真实tokens：

- 原9.9M中的30个query histories、local state和target完全不变；
- 追加90.1M来自PG19 train的真实文本，共983本不重复书籍；query来自test split，新增干扰来自train split；
- 总计1,562,500个64-token blocks、1,118个book scopes；不含合成文本、重复填充、target、query book未来文本或预定义source；
- 用嵌套的9.9M/20M/50M/100M前缀做同query配对比较，统一使用100M词表与IDF，避免各尺度重新拟合造成口径漂移；
- 在线router只读取当前状态文本，不读取真实book id；`oracle scope`只作诊断上限；最终工作集始终是Top8，即512 tokens。

独立审计确认：9.9M前缀的token、scope和原位置逐块完全一致，queries/targets/metadata的SHA256完全一致，30个query scopes中future/local block违规为0；全部因果与去重复检查通过。

比较全局unigram BM25与两级`book-scope BM25 -> scope内block BM25`。离线解码、block索引和scope索引分别耗时54.9s、40.5s、34.6s；block/scope索引为461.8/103.7MiB。以下在线时间均为CPU实测，不包含reader forward。

#### 十倍干扰下的召回、纯度和在线比较域

状态512时的嵌套尺度曲线：

| memory | 方法 | scope router | Top8同书any-hit | Top8同书占比 | 平均候选blocks |
|---:|---|---:|---:|---:|---:|
| 9.9M | 全局BM25 | - | 96.7% | 71.7% | 154,688 |
| 9.9M | Top3 scopes | 90.0% | 90.0% | 80.4% | 5,502 |
| 9.9M | Top8 scopes | 100% | 100% | 81.7% | 16,332 |
| 50M | 全局BM25 | - | 86.7% | 52.5% | 781,250 |
| 50M | Top3 scopes | 80.0% | 80.0% | 73.3% | 8,072 |
| 50M | Top8 scopes | 86.7% | 83.3% | 69.6% | 24,305 |
| 100M | 全局BM25 | - | **83.3%** | 42.5% | 1,562,500 |
| 100M | Top3 scopes | 76.7% | 76.7% | **65.0%** | 8,502 |
| 100M | Top8 scopes | **83.3%** | 80.0% | 62.5% | 26,196 |

100M上Top3 scopes只扫描0.544% blocks，即183.8倍候选压缩；在线4.63ms，相对全局31.19ms加速6.74倍，同时保留92%的Top8 any-hit。Top8 scopes扫描1.68%，压缩59.6倍，9.60ms、加速3.25倍，保留96%的any-hit。状态128也复现：Top3/Top8分别压缩190.4/72.5倍、加速5.18/2.57倍，保留全局any-hit的90%/95%。

这不是完全尺度不变：全局BM25在状态512从96.7%降到83.3%，Top8 scopes从100%降到80%；book router本身也会受到新增书籍极值干扰。得到支持的是**层级路由把退化限制在一个小得多的比较域，并允许用Top-D连续交换计算与召回**，而不是“一层book路由已经解决1B”。

#### 层级方法召回略低，reader质量反而更高

将100M检索出的Top8原文块交给同一个Qwen3-0.6B预测随后128 tokens：

| 当前状态 | 方法 | Top8同书any-hit / 占比 | PPL | 相对Query-only Delta NLL 95% CI |
|---:|---|---:|---:|---:|
| 128 | Query-only | 0 / 0 | 24.98 | - |
| 128 | Random512 | 0 / 0 | 25.74 | `[+0.0031,+0.0596]` |
| 128 | 全局BM25 | 66.7% / 28.3% | 24.25 | `[-0.1010,+0.0203]` |
| 128 | Top3 scopes | 60.0% / **45.8%** | **23.40** | `[-0.1386,-0.0107]` |
| 128 | Top8 scopes | 63.3% / 42.9% | 23.67 | `[-0.1227,-0.0059]` |
| 128 | Oracle scope | 100% / 100% | 23.47 | `[-0.1131,-0.0190]` |
| 512 | Query-only | 0 / 0 | 22.38 | - |
| 512 | Random512 | 0 / 0 | 22.63 | `[-0.0028,+0.0242]` |
| 512 | 全局BM25 | 83.3% / 42.5% | 22.57 | `[-0.0055,+0.0231]` |
| 512 | Top3 scopes | 76.7% / **65.0%** | 21.99 | `[-0.0380,+0.0013]` |
| 512 | Top8 scopes | 80.0% / 62.5% | **21.97** | `[-0.0390,+0.0002]` |
| 512 | Oracle scope | 100% / 100% | 21.38 | `[-0.1047,-0.0070]` |

更严格的直接配对比较中，Top3相对全局BM25的NLL差在状态128/512为`-0.0358/-0.0263`，95% CI分别`[-0.0613,-0.0133]`和`[-0.0471,-0.0088]`；Top8也分别为`-0.0243/-0.0272`，CI均严格低于0。也就是说，层级方法不是只“更快但略损质量”，而是在当前实验中**更快且reader质量更好**。

原因是any-hit只问“8块里是否至少有一个同书块”，没有惩罚另外7块的干扰。层级路由把状态128/512的同书占比从28.3%/42.5%提高到约43%-46%/62%-65%，工作集纯度的收益超过了少量coverage损失。该结论同时修正评价指标：最终控制目标应是`coverage x conditional utility - interference cost`，而不是最大化召回率后固定填满预算。

逐题诊断进一步显示，这个纯度信号也受生成时刻控制。状态128时，Top3相对全局BM25的“同书占比增益”与“reader NLL增益”的Spearman为`0.458`，`p=0.0109`；状态512时只有`0.152`，`p=0.423`。在Top3方法中，状态128命中/未命中同书scope的平均reader效用为`+0.110/-0.002 NLL`，到状态512则收敛为`+0.019/+0.013`。因此scope纯度是**中期地址已成熟、recent仍不充分时**的有效代理，不是所有生成时刻都成立的静态utility score；这与8.10节的中间价值窗口相互印证。

100M数据协议：[pg19_past_only_100m_protocol_20260715.json](1b_context_search_research_exploration/evidence/pg19_past_only_100m_protocol_20260715.json)

因果与前缀审计：[pg19_past_only_100m_audit_20260715.json](1b_context_search_research_exploration/evidence/pg19_past_only_100m_audit_20260715.json)

嵌套尺度检索：[pg19_past_only_100m_hier_bm25_20260715.json](1b_context_search_research_exploration/evidence/pg19_past_only_100m_hier_bm25_20260715.json)

状态128 PPL：[pg19_past_only_100m_hier_ppl_s128_20260715.json](1b_context_search_research_exploration/evidence/pg19_past_only_100m_hier_ppl_s128_20260715.json)

状态512 PPL：[pg19_past_only_100m_hier_ppl_s512_20260715.json](1b_context_search_research_exploration/evidence/pg19_past_only_100m_hier_ppl_s512_20260715.json)

配对汇总：[pg19_past_only_100m_hier_analysis_20260715.json](1b_context_search_research_exploration/evidence/pg19_past_only_100m_hier_analysis_20260715.json)

### 8.13 scope置信度性质：路由错误可预测，但recall校准不是utility控制

8.12节使用固定Top3/Top8 scopes。本节检验一个更强的性质：**能否只根据当前query产生的scope分数形状，判断应该保留1、3、8、16还是32个scope？**

在相同100M索引上新增Top1/3/8/16/32检索，并记录不使用target的在线特征：Top1-2、Top3-4等score margin，Top1/3/8/16/32正分质量占比，归一化熵、HHI、Top1 z-score、query词项数和有效scope数。实验共`30 queries x 4 memory scales x 4 state lengths = 480`个状态；按query id做5折GroupKFold，同一query的所有尺度和状态不会跨越train/test。标签只在训练折使用“真实book是否进入Top-D”，测试选择不读取book id或target。

#### score geometry确实包含query级路由置信度

| 特征 | 全尺度宏AUC | 100M宏AUC | 100M D1 / D3 / D8 / D16 / D32 AUC |
|---|---:|---:|---:|
| 仅memory规模、state长度、query词项数 | 0.632 | 0.564 | 0.563 / 0.548 / 0.546 / 0.561 / 0.603 |
| 加入margin、熵与score mass | **0.796** | **0.768** | **0.840 / 0.745 / 0.740 / 0.740 / 0.774** |

因此，某条query的scope router是否可靠，并不只由“100M很难”或“状态有多长”决定；分数几何包含显著的query-specific不确定性。这支持用无target confidence决定是否保留多scope分支，也说明固定Top-D不是信息论上的必然选择。

#### 但扩大scope会出现明确的干扰拐点

100M固定深度结果如下；候选数是第二级实际比较的blocks，reader仍只读取最终Top8共512 tokens：

| 状态 | scope深度 | scope覆盖 | Top8同书命中 / 占比 | 候选blocks | PPL |
|---:|---:|---:|---:|---:|---:|
| 128 | 3 | 60.0% | 60.0% / **45.8%** | 8,205 | **23.40** |
| 128 | 8 | 66.7% | 63.3% / 42.9% | 21,562 | 23.67 |
| 128 | 16 | 73.3% | 66.7% / 37.9% | 43,984 | 23.95 |
| 128 | 32 | 83.3% | 73.3% / 37.5% | 85,481 | 23.87 |
| 512 | 3 | 76.7% | 76.7% / **65.0%** | 8,502 | 21.99 |
| 512 | 8 | 83.3% | 80.0% / 62.5% | 26,196 | **21.97** |
| 512 | 16 | 86.7% | 83.3% / 62.1% | 52,599 | 22.01 |
| 512 | 32 | 90.0% | 83.3% / 57.1% | 99,205 | 22.34 |

D16相对D8在状态128的NLL差为`+0.0117`，95% CI `[+0.0035,+0.0209]`；D32相对D8在状态512为`+0.0169`，CI `[+0.0032,+0.0333]`，均显著恶化。路由层覆盖一直增加，但进入最终Top8的同书占比下降，而且D32在状态512不再增加Top8 any-hit。新增scope让更多无关blocks参与第二级极值竞争，出现了与效用probe相同的winner's curse和干扰成本。

#### recall校准的adaptive D尚未形成Pareto优势

用严格out-of-fold预测概率选择最小满足70%/80%/90% scope-hit置信度的D，低置信时回退到D32。代表性结果：

| 状态 | 方法 | 平均D | Top8命中 / 占比 | 候选blocks | PPL |
|---:|---|---:|---:|---:|---:|
| 128 | Adaptive C70 | 13.67 | 63.3% / 45.4% | 36,417 | 23.42 |
| 128 | 固定D3 | 3 | 60.0% / 45.8% | **8,205** | **23.40** |
| 512 | Adaptive C70 | 5.07 | 76.7% / **67.5%** | 15,787 | 21.97 |
| 512 | 固定D3 | 3 | 76.7% / 65.0% | **8,502** | 21.99 |
| 512 | Adaptive C80 | 8.97 | 80.0% / 66.7% | 30,012 | **21.96** |
| 512 | 固定D8 | 8 | 80.0% / 62.5% | **26,196** | 21.97 |

Adaptive C70与固定D3的配对NLL差在状态128/512为`+0.0009/-0.0008`，CI均跨0；C80与固定D8在状态512为`-0.0005`，CI也跨0。也就是说，它们都显著优于全局BM25，却没有严格支配简单的固定D3/D8。

这条结果非常关键：

1. **score geometry是可利用的地址不确定性属性；**
2. **“真实scope是否在候选内”不是最终控制目标；**
3. **当D增加时，scope coverage、最终block纯度和reader utility会分叉；**
4. **下一版停止器应预测`marginal generation utility - interference - search cost`，而不是把scope-hit概率校准到固定阈值。**

这也构成与普通Top-K RAG更实质的边界：系统不仅要给文档排序，还要在连续生成中判断“新增一层候选是否仍有边际因果价值”，并在没有价值时执行STOP。

带score geometry的100M检索：[pg19_past_only_100m_hier_bm25_v2_20260715.json](1b_context_search_research_exploration/evidence/pg19_past_only_100m_hier_bm25_v2_20260715.json)

分组交叉验证与reader配对汇总：[pg19_past_only_scope_adaptive_20260715.json](1b_context_search_research_exploration/evidence/pg19_past_only_scope_adaptive_20260715.json)

状态128 adaptive PPL：[pg19_past_only_scope_adaptive_ppl_s128_20260715.json](1b_context_search_research_exploration/evidence/pg19_past_only_scope_adaptive_ppl_s128_20260715.json)

状态512 adaptive PPL：[pg19_past_only_scope_adaptive_ppl_s512_20260715.json](1b_context_search_research_exploration/evidence/pg19_past_only_scope_adaptive_ppl_s512_20260715.json)

### 8.14 边际utility性质：大多数scope扩展无益，廉价信号能预测符号但不能预测幅度

8.13节说明scope-hit置信度不足以决定STOP。本节把监督目标直接改为真实reader反事实：对100M、状态128/512的每条query，比较`D3->D8`、`D8->D16`和`D16->D32`前后的未来128-token NLL，共180个严格配对扩展事件。训练折可以观察未来NLL标签，测试选择只使用：

- 当前状态长度、起止D和候选增长量；
- scope margin、熵、score mass与HHI；
- 扩展前后最终Top8的Jaccard/churn；
- 不使用真实scope、target token或测试折NLL。

仍按query id做5折GroupKFold，防止同一书籍状态泄漏。

#### 有益扩展是少数事件，而非默认情况

180次扩展中只有`22.8%`降低未来NLL，平均扩展反而恶化`0.0060 NLL`：

| 状态与扩展 | 有益比例 | 平均NLL改善 |
|---|---:|---:|
| 128，D3->D8 | 26.7% | -0.0115 |
| 128，D8->D16 | 23.3% | -0.0117 |
| 128，D16->D32 | 36.7% | +0.0034 |
| 512，D3->D8 | 26.7% | +0.0009 |
| 512，D8->D16 | 13.3% | -0.0020 |
| 512，D16->D32 | **10.0%** | **-0.0149** |

负值表示扩展后更差。特别是状态512的后两次扩展，绝大多数只是把已经充分的工作集重新暴露给更多极值干扰。因而更合理的先验是`STOP`，而不是“低置信就继续扩大”。

#### 无target信号可以预测utility符号，但预测幅度仍失败

| 预测器 | 分组外推AUC | 状态128 / 512 AUC | 预测概率与真实Delta NLL Spearman |
|---|---:|---:|---:|
| 仅状态、D与候选规模 | 0.538 | 0.386 / 0.599 | -0.118 |
| 加scope geometry | 0.601 | 0.504 / 0.679 | -0.102 |
| geometry + Top8 churn，逻辑回归 | 0.705 | 0.610 / 0.787 | -0.193 |
| geometry + Top8 churn，受限深度森林 | **0.760** | **0.699 / 0.820** | -0.173 |

加入最终排名变化后，模型明显能区分“扩展更可能有益还是有害”，尤其在状态512。但连续收益回归失败：Ridge/受限深度森林的Delta NLL外推Spearman仅`-0.029/-0.091`，sign AUC也只有`0.317/0.389`。二元分类能识别一部分事件类型，却没有学到收益强度；微小正收益和大收益仍被当作同一标签。

#### 学习STOP尚未显著胜固定D，但oracle显示有真实空间

逻辑回归在阈值0.30时，状态512平均选择D5.6、17,007个二级候选，PPL为`21.928`；固定D3为8,502候选、PPL `21.988`。配对NLL差`-0.0027`，CI `[-0.0083,+0.0013]`，方向有利但尚不显著。状态128的学习策略最优行为基本是全部停在D3，无法超过PPL `23.401`。

诊断性oracle只在当前扩展真实改善NLL时继续：

| 状态 | 平均D | 候选blocks | Oracle STOP PPL | 固定D3 PPL | NLL差 / 95% CI |
|---:|---:|---:|---:|---:|---:|
| 128 | 5.4 | 14,325 | **23.178** | 23.401 | `-0.0095 / [-0.0171,-0.0033]` |
| 512 | 4.6 | 15,810 | **21.834** | 21.988 | `-0.0070 / [-0.0131,-0.0023]` |

oracle并不是可部署结果，但它证明“少量、按需扩scope”确实比任何固定D更好；当前缺失的是可泛化的收益幅度代理，而不是候选中没有可用信息。

由此得到三条更广泛的性质：

1. **scope扩展效用是稀疏事件，默认STOP具有统计依据；**
2. **状态越成熟，继续扩展的正事件越少，utility prior必须依赖生成时刻；**
3. **score geometry与ranking churn只够做风险筛查，真正的幅度估计需要模型响应信号。**

这使下一步实验目标非常明确：比较无需额外reader的`surprisal change / hidden-state novelty / Value alignment`与1-4个批量短probe，测它们对连续Delta NLL幅度的外推，而不是继续增加静态相似度特征。

边际utility、分组外推与oracle上限：[pg19_past_only_scope_utility_stop_20260715.json](1b_context_search_research_exploration/evidence/pg19_past_only_scope_utility_stop_20260715.json)

### 8.15 生成轨迹是在线监督信号：先检索、后观察仍能预测未来utility

8.14节缺少的是模型响应信号。本节检验一个不需要gold、也不读取未来target的代理：**已经真实生成出来的token，在不同候选工作集下的反事实loss，能否预测这些候选对下一段生成的价值。**

为排除“检索和评分使用同一段文本”的循环相关，采用更严格的时间切分：

- 状态128：仅用最早64 tokens检索，随后64 tokens作为已观察probe，再预测未来128 tokens；
- 状态512：仅用最早448 tokens检索，随后64 tokens作为已观察probe，再预测未来128 tokens；
- 检索深度为D3、D8、D16、D32；每个深度最终都只加载Top8个64-token blocks；
- probe与未来target都不参与候选选择，记忆严格past-only，无预定义source。

对每个`D -> next D`，定义已观察边际收益和未来边际收益：

```text
observed_gain = NLL(probe | W_D) - NLL(probe | W_nextD)
future_gain   = NLL(future | W_D) - NLL(future | W_nextD)
```

共30条query、2种状态、3次扩展，得到180个严格配对事件：

| Probe窗口 | 总体Spearman / p | 总体符号AUC | 状态128 Spearman / AUC | 状态512 Spearman / AUC |
|---:|---:|---:|---:|---:|
| 8 | 0.189 / 0.0110 | 0.625 | 0.279 / 0.688 | 0.050 / 0.505 |
| 16 | 0.270 / 2.46e-4 | 0.639 | **0.370 / 0.652** | 0.101 / 0.633 |
| 32 | 0.221 / 0.00287 | 0.613 | 0.198 / 0.588 | **0.289 / 0.696** |
| 64 | **0.271 / 2.36e-4** | **0.649** | 0.295 / 0.620 | 0.247 / **0.740** |

因此，候选对刚发生轨迹的影响确实具有跨越下一个128-token窗口的时间持续性；这不是由未来答案、真实book id或同一probe文本参与检索造成的。它给出了一个比静态相似度更接近最终目标的在线信号：模型已经生成的内容本身可以充当弱监督。

将probe用于序贯`EXPAND/STOP`后：

| 状态 | 最佳规则 | 平均D / 二级候选 | 自适应PPL | 固定D3 PPL | 配对NLL差 / 95% CI |
|---:|---|---:|---:|---:|---:|
| 128 | probe16，gain>0 | 4.1 / 10,903 | **24.677** | 24.846 | `-0.00681 / [-0.01598,-0.00004]` |
| 512 | probe32，gain>0.0025 | 4.87 / 15,280 | 21.760 | **21.752** | `+0.00040 / [-0.00594,+0.00885]` |

状态128的自适应策略显著胜D3，且均值优于所有固定D，但相对D8/D16/D32的CI仍跨0；状态512没有超过已经很强的固定D3，不过显著避免了D16/D32的过度扩展。其含义不是“probe总能提高质量”，而是：**较早、地址仍在成熟的状态有更多按需扩展空间；成熟状态的正确默认动作往往就是STOP。**

代价也必须计入：当前未融合实现平均执行约2.2-2.33次probe forward，状态128/512分别增加167.6/83.0ms。已有小batch实验表明候选可并行，但部署前仍需把多个D的probe合并成一次批处理，并研究更便宜的hidden/Value代理。

此前“检索query包含最近probe tokens”的在线版本，在状态512上probe16与未来收益Spearman为0.326，并能显著胜固定D3；严格时间切分后这一策略优势消失，但相关信号仍显著。故保守结论是：**真实的时间持续性成立，在线同窗反馈还叠加了query更新带来的额外优势；两者不能混称为同一个因果效应。**

严格100M预检索：[pg19_past_only_100m_hier_bm25_preprobe_20260715.json](1b_context_search_research_exploration/evidence/pg19_past_only_100m_hier_bm25_preprobe_20260715.json)

严格probe与未来效用分析：[pg19_past_only_scope_retrospective_preprobe_analysis_20260715.json](1b_context_search_research_exploration/evidence/pg19_past_only_scope_retrospective_preprobe_analysis_20260715.json)

状态128/512 reader结果：[pg19_past_only_preprobe_ppl_s128_20260715.json](1b_context_search_research_exploration/evidence/pg19_past_only_preprobe_ppl_s128_20260715.json)、[pg19_past_only_preprobe_ppl_s512_20260715.json](1b_context_search_research_exploration/evidence/pg19_past_only_preprobe_ppl_s512_20260715.json)

### 8.16 真实对话记忆：证据基数、层级粒度、更新槽位与RAG对照

为避免结论只来自合成两跳问题或连续书籍，本节加入LongMemEval-S cleaned。它是带时间戳的半合成长程对话基准：目标会话由受控任务生成，干扰会话包含ShareGPT、UltraChat与模拟交互，因此比人工高斯向量真实，但不能称为完全自然日志。

我们把64条分层抽样问题的完整历史与29个完整干扰owner拼成一个**恰好10,000,000 tokens**的共享记忆：

- 156,250个64-token blocks、4,525个sessions；
- 64条问题含48条可回答与16条abstention；
- 选择问题时不使用答案，证据由官方`answer_session_ids`和`has_answer`标注定位；
- 每条可回答问题平均2.02个证据sessions、6.25个精确blocks，中位数分别为2和4；
- 最终统一只读取Top8 blocks，即512 tokens。

先比较平坦检索和两级检索。表中`全部证据`要求所有官方证据session都出现在最终Top8中；E5在线时间包含单条query embedding和GPU矩阵检索，BM25是CPU稀疏实现，二者时间只表示本原型的实际开销，不代表优化后极限。

| 路径 | 平均二级候选blocks | 精确证据any@8 | 最新精确证据@8 | 全部证据sessions@8 | 在线时间/题 |
|---|---:|---:|---:|---:|---:|
| 全局block BM25 | 156,250 | 54.2% | 33.3% | 41.7% | 1.04 ms |
| 全局block E5-base-v2 | 156,250 | **58.3%** | **43.8%** | **45.8%** | 8.04 ms |
| 全局BM25 session8 -> block | 354 | **52.1%** | **37.5%** | **43.8%** | 0.572 ms |
| 全局E5 session8 -> block | 332 | 37.5% | 25.0% | 22.9% | 7.55 ms |
| 已知owner，BM25 session3 -> block | 131 | **83.3%** | 58.3% | **70.8%** | 0.963 ms |
| 已知owner，E5 session3 -> block | 125 | 79.2% | **62.5%** | 64.6% | 7.89 ms |
| 已知owner，BM25 session8 -> block | 337 | 81.2% | 62.5% | 77.1% | 1.01 ms |
| 已知owner，E5 session8 -> block | 315 | 81.2% | **70.8%** | **81.2%** | 7.89 ms |

E5全局block相对BM25的精确命中净增4.2个百分点，但配对只有8胜6负，`p=0.791`，不能宣称显著。更关键的反例是：全局E5 session8把长session截断并压成一个向量后，全部证据召回比BM25 session8低20.8个百分点，配对2胜12负，`p=0.0129`。**语义模型更强不等于粗粒度摘要更保真；层级索引的上层key必须保留细事实，不能把整个容器任意平均。**

已知owner不是答案泄漏，而对应真实系统通常已知的当前用户、项目、租户或仓库namespace。它把全局156,250 blocks先缩到平均1,665 blocks；再取BM25 Top3 sessions只比较131 blocks，累计缩域约1,192倍，同时Top8精确命中由全局54.2%提高到83.3%。但如果必须从混合全库自动识别owner，当前粗文本router的Top1/Top3/Top8 owner recall只有29.7%/45.3%/62.5%。所以环境元数据应作为硬scope，未知scope则保留多个分支，不能假设单个自动owner路由已经解决。

层级深度也有明确拐点：BM25 owner-session从D1扩到D3时，`all evidence sessions@8`有20胜0负，`p=1.91e-6`；D3到D8只有4胜1负，同时精确block出现0胜1负；D8以后最终Top8不再变化。其可利用属性是：**复合问题需要少量并行session覆盖，但候选继续增加会进入干扰区，D不是越大越好。**

时间属性则不是“越新越好”。在排除分钟级时间异常后的严格因果子集中：

- knowledge-update证据中位年龄23.8天，中位owner内新近排名23.5，只有25%落在latest8；
- multi-session证据中位新近排名24，只有12.5%落在latest8；
- 直接取owner全局latest3/latest8的精确命中只有10.4%/25.0%；
- 先取语义Top16再选其中latest3仍只有41.7%，低于纯BM25 session3的83.3%。

这说明知识更新应建模成**槽位条件时间链**：先找`owner + entity + relation`的所有相关写入，再在该小集合中解析最新值、先后关系或时间范围。对整个owner历史套global recency会错过大量远期但当前仍有效的事实。

数据时间轴也经过单独审计。官方定义要求问题发生在全部会话之后；cleaned数据的470条可回答问题中，41条有证据时间在分钟级晚于`question_date`，共70个sessions，平均领先6.74小时，但**自然日级未来证据为0**。因此官方all-history评测仍按全部会话执行；严格在线因果结论只使用证据不晚于提问时刻的40/48条本地样本，不能把同日时刻噪声隐藏掉。

最后，16条abstention暴露出另一种结构：BM25全局Top8有56.2%命中官方hard-negative session，已知owner的BM25 session3为81.2%，E5 owner-session8为100%。这些近邻内容与问题高度相关，却有意缺少一个必要事实。系统需要的不是“降低相关检索”，而是检索后判断所需槽位是否完整，并区分当前值、旧值、近邻实体与不存在的信息。

E5离线索引使用8张3090，对全部156,250 blocks编码87.25秒、4,525 sessions编码40.52秒；这是标准轻量稠密RAG对照，不是最强RAG。LongMemEval论文已经证明更强稠密retriever、round粒度、事实key扩展、时间query扩展和extract-before-read有效。因此本节支持的是通用记忆属性与失败边界，不把“层级RAG”本身当作新贡献。

共享10M数据与BM25结果：[longmemeval_shared_10m_data_20260715.json](1b_context_search_research_exploration/evidence/longmemeval_shared_10m_data_20260715.json)、[longmemeval_10m_hier_bm25_20260715.json](1b_context_search_research_exploration/evidence/longmemeval_10m_hier_bm25_20260715.json)

结构、因果与深度分析：[longmemeval_10m_properties_20260715.json](1b_context_search_research_exploration/evidence/longmemeval_10m_properties_20260715.json)、[longmemeval_full_temporal_audit_20260715.json](1b_context_search_research_exploration/evidence/longmemeval_full_temporal_audit_20260715.json)

E5与配对RAG对照：[longmemeval_10m_e5_rag_20260715.json](1b_context_search_research_exploration/evidence/longmemeval_10m_e5_rag_20260715.json)、[longmemeval_10m_bm25_e5_comparison_20260715.json](1b_context_search_research_exploration/evidence/longmemeval_10m_bm25_e5_comparison_20260715.json)

### 8.17 零额外前向的辨识边界：只看当前工作集，不足以判断未见候选是否有用

8.15节的counterfactual probe有效，但需要对扩展工作集做额外reader forward。本节先问更严格的问题：能否只利用模型正常生成当前工作集`W_D`时已经产生的信号，预测尚未读取的`W_nextD - W_D`是否会改善未来生成？

沿用严格PG19 100M pre-probe协议，共30条query、状态128/512、三种`D -> next D`扩展，得到180个事件。候选在观察64-token probe之前固定；当前正常forward可以看到`W_D + 已观察64 tokens`，但不对新增scope做forward，未来128 tokens只用于事后定义Delta NLL。

| 零额外reader特征 | 未来utility符号AUC | 结论 |
|---|---:|---|
| scope geometry + ranking churn | **0.737** | 能识别一部分扩展风险，但主要描述候选竞争结构 |
| 当前attention响应 | 0.595 | 当前工作集如何被注意，不能说明未见scope会带来什么 |
| 当前不确定性 | 0.464 | “模型现在不确定”不等于“这个新增候选能解决不确定性” |
| 当前hidden-state摘要 | 0.419 | 不含候选身份，无法形成候选级排序 |
| 状态与预算元数据 | 0.396 | 只学到弱prior |

所有零额外信号对连续Delta NLL幅度的外推都失败，最佳相关仅约0.041，单特征也没有通过FDR。这个负结果给出一个一般性的**可辨识性边界**：只给定同一个`(z_t, W_t)`，两个未见候选`C1/C2`可能分别有正、负效用；任何不依赖候选内容的函数都只能给二者相同预测。因此当前状态适合决定`LOCAL / RETRIEVE / SCAN`，却不能单独完成候选精排。

证据：[pg19_past_only_zero_extra_signals_20260715.json](1b_context_search_research_exploration/evidence/pg19_past_only_zero_extra_signals_20260715.json)、[pg19_past_only_zero_extra_utility_analysis_20260715.json](1b_context_search_research_exploration/evidence/pg19_past_only_zero_extra_utility_analysis_20260715.json)

### 8.18 候选条件信号：QK提供模型原生地址，但与同状态E5相当而非更强

接着保持候选集合、状态和未来标签完全不变，只增加“不运行扩展reader”的候选侧统计。对180次扩展涉及的724个固定候选blocks，比较两类信号：

1. **文本候选互补性：** 用完整已观察状态的E5向量与候选文本计算query affinity、相对当前工作集的新颖性和覆盖；
2. **模型原生候选响应：** 离线保存候选block在7层、全部KV heads上的block-local pre-RoPE K/V sidecar；在线用当前正常forward的Q计算RoPE校准QK、head选择性和Value响应，不运行`W_nextD` reader forward。

| 预测器 | utility符号AUC | 相对geometry的query-cluster bootstrap差值 |
|---|---:|---:|
| geometry | 0.737 | - |
| geometry + 同状态E5候选特征 | 0.813 | `[+0.026,+0.138]` |
| geometry + 模型原生QK | **0.827** | `[+0.016,+0.179]` |
| QK 相对 E5 | +0.013 AUC | `[-0.047,+0.073]` |

这说明**候选条件交互是必要信息**，而QK确实可以作为随生成状态变化的模型原生地址。但QK没有显著胜同状态E5，QK+E5/Value也没有稳定提高符号AUC。连续幅度上，紧凑`E5+QK/Value`的Spearman为0.160、cluster CI `[0.009,0.316]`，显著好于geometry；相对E5单独的增量CI仍跨0。

工程成本也不是零：候选sidecar为1.328GB；本原型当前状态forward平均34.9ms，候选响应特征平均26.8ms。它比完整扩展reader便宜，并能只对粗候选计算，但尚不能直接扩到1B全库。正确用法是`scope/text postings -> 几百候选 -> QK/Value统计`，不是全局QK扫描。

证据：[pg19_past_only_candidate_complementarity_observed64_features_20260715.json](1b_context_search_research_exploration/evidence/pg19_past_only_candidate_complementarity_observed64_features_20260715.json)、[pg19_past_only_model_native_response_20260715.json](1b_context_search_research_exploration/evidence/pg19_past_only_model_native_response_20260715.json)、[pg19_past_only_model_native_fair_utility_analysis_20260715.json](1b_context_search_research_exploration/evidence/pg19_past_only_model_native_fair_utility_analysis_20260715.json)

### 8.19 跨域到真实10M代码：高响应不等于有用，效用模型必须显式惩罚干扰

为检验8.18是否只是PG19书籍域现象，在LongBench-v2真实代码continuation上构造第二个严格协议：

- 10M mixed-repo代码记忆，30条query；
- 当前状态为64-token query + 已观察目标A段64 tokens；
- 每个候选窗口256 tokens，目标是候选对未来B段64 tokens的Delta NLL；
- 候选仅取BM25、E5、RRF各Top32的并集，共1,760个窗口、5,539个唯一blocks；不使用random或oracle-only候选；
- 模型原生sidecar只取第7/15/27层、全部KV heads，共4.356GB；当前状态forward缓存后平均40.0ms，候选响应特征平均14.4ms/窗口。

候选效用比书籍域更尖锐：只有32.9%的候选改善未来B段，平均Delta NLL为`-0.0453`，即多数主题相关候选实际有害。按query分组的5折外推与逐query选择结果如下：

| 选择策略 | 每题所选窗口平均未来Delta NLL | 相对`rank+repo scope`的配对差值95% CI |
|---|---:|---:|
| 最佳静态retriever rank | +0.0461 | `[-0.199,+0.027]` |
| OOF rank + repo scope | +0.1216 | baseline |
| OOF rank + scope + 文本特征 | +0.1100 | `[-0.063,+0.036]` |
| OOF rank + scope + QK/Value | +0.1160 | `[-0.022,+0.010]` |
| OOF rank + scope + 文本 + QK/Value | +0.1247 | `[-0.043,+0.045]` |
| 已观察A段counterfactual probe | **+0.1812** | `[-0.010,+0.136]` |
| 未来B段oracle | +0.3104 | `[+0.118,+0.268]` |

如果错误地把1,760个窗口当独立样本，会看到大量极小p值；但窗口嵌套在同一query内。修正为“每个query内先算Spearman，再对30个query做符号置换、bootstrap和FDR”后，没有任何单个QK/Value特征通过FDR。最强的Value alignment信号平均query内Spearman为0.142，CI `[0.037,0.251]`，但校正后`q=0.391`；late-layer token-max QK反而与效用负相关`-0.091`，同样`q=0.391`。

因此PG19上的QK增益不能直接外推到代码。更稳健的结论是：

1. **addressability：** QK、BM25、E5说明候选是否能被当前状态“叫到”；
2. **incremental information：** 候选是否补充当前recent/workset尚未包含的信息；
3. **interference：** 候选是否引入高吸引力但错误、冗余或作用域冲突的模式；
4. **utility：** 前三者与读取成本共同决定未来Delta NLL。

跨域通用系统不能把第1项当第4项。当前最可靠的低成本主干仍是结构scope和强文本召回；QK是候选条件补充信号，只有经过跨域校准或短probe验证后才应影响KEEP/EVICT。

证据：[longbench_code_model_native_utility_analysis_20260715.json](1b_context_search_research_exploration/evidence/longbench_code_model_native_utility_analysis_20260715.json)、[longbench_code_query_cluster_candidate_signals_20260715.json](1b_context_search_research_exploration/evidence/longbench_code_query_cluster_candidate_signals_20260715.json)

### 8.20 检索轨迹的双时间尺度：scope慢变，block快变，粗frontier只能部分复用

如果每生成少量tokens都重新搜索1B全库，即使单次索引是次线性的，累计成本仍然很高。可扩展系统还需要**时间局部性**：上一状态的scope、候选frontier和已加载pages能否服务下一状态？为此重新分析三类真实约10M轨迹：XSum新闻100条、PG19书籍30条、LongBench-v2代码30条；每条都记录prefix 8/16/32/64下BM25、E5和RRF的Top512，检索不使用target。

先看RRF的block级复用。`previous Top512 coverage`表示下一轮完整10M检索Top8中，有多少已经存在于上一轮Top512；`history`把此前所有Top512做有界并集。

| 真实10M域 | 相邻Top8 Jaccard范围 | 上一轮Top512覆盖下一轮Top8 | 历史frontier覆盖下一轮Top8 | 最大平均frontier |
|---|---:|---:|---:|---:|
| XSum新闻 | 0.222-0.307 | 64.8%-78.5% | 64.8%-79.3% | 1,091 blocks |
| PG19书籍 | 0.192-0.244 | 62.5%-75.4% | 62.5%-76.3% | 1,124 blocks |
| 代码repo | 0.263-0.439 | **72.1%-85.4%** | **72.1%-89.2%** | 1,006 blocks |

因此旧Top8本身不稳定，不能直接KEEP；但0.5K-1.1K的粗frontier确实保留了大多数下一轮强候选。下一轮Top8首次命中source、且上一轮Top512完全没有任何source block的比例只有：XSum 3%-6%、PG19 3.3%、代码0%-6.7%。这支持低成本candidate rerank与page cache，但不是100%证明。

为了避免把“候选覆盖”误写成“可直接替代全库搜索”，进一步按新query的完整排名只重排旧Top512。该操作在RRF轨迹中都可由保存的Top512交集精确审计：

| 域 | 只重排上一Top512相对完整搜索的source any@8差值 |
|---|---:|
| XSum | -4、-6、-7个百分点；后两段CI不含0 |
| PG19 | -3.3、0、-13.3个百分点；32->64显著下降 |
| 代码 | +3.3、-10、+3.3个百分点；CI均含0 |

历史frontier能减少部分损失，但仍不能稳定等价于完整刷新。其原因不是旧frontier完全没有证据，而是新状态会改变候选内部的竞争结构，还会引入少量此前完全未进入frontier的新证据。正确设计应是：**常态只重排旧frontier，检测到scope/实体/关系事件时注入新postings，并保留低频全局刷新；不能永久关闭全局入口。**

代码数据有真实repository scope，可以直接比较层级稳定性：

| prefix变化 | Top8 block Jaccard | Top3 scope Jaccard | scope-block配对差 / 95% CI | 旧Top3 scope覆盖新Top8 blocks |
|---|---:|---:|---:|---:|
| 8->16 | 0.263 | **0.623** | `+0.361 / [+0.227,+0.490]` | 67.5% |
| 16->32 | 0.361 | **0.670** | `+0.309 / [+0.187,+0.430]` | 77.9% |
| 32->64 | 0.439 | **0.753** | `+0.315 / [+0.193,+0.435]` | 91.7% |

旧Top8 scopes对新Top8 blocks的覆盖进一步达到94.2%/93.3%/97.5%。这给出了比“检索结果会持续”更精确的性质：**上层语义/环境scope慢变，细粒度证据block快变。** 系统应为二者使用不同更新时钟：scope route低频刷新，scope内block和active working set高频刷新。

证据与逐query审计：[cross_domain_10m_incremental_frontier_hubness_20260715.json](1b_context_search_research_exploration/evidence/cross_domain_10m_incremental_frontier_hubness_20260715.json)、[cross_domain_10m_incremental_frontier_hubness_rows_20260715.jsonl](1b_context_search_research_exploration/evidence/cross_domain_10m_incremental_frontier_hubness_rows_20260715.jsonl)

### 8.21 稳定hub是模型表示属性：文本候选近似分散，QK极值存在profile条件公共吸引力

8.11节发现K含有强layer/head公共方向。本节从检索结果侧问：这些公共分量是否产生跨query反复占据Top-K的稳定hub blocks？先看文本RRF的Top64：

- XSum的6,068次base nominations分散到5,801个blocks，Simpson有效blocks为5,545；最频繁block只出现在4%的queries；
- PG19为1,862次/1,848个唯一blocks，最频繁block覆盖6.7%；
- 代码为1,812次/1,766个唯一blocks，最频繁block覆盖10%；
- leave-one-query-out hub惩罚在三域RRF上都没有改变prefix64的source any@8。

所以普通BM25/E5候选没有表现出足以解释主要错误的稳定全局hub，不能把KV去偏方法机械套到RAG通道。

同一批XSum 100条query的真实QK结果完全不同。采用两折cross-fit：在一半query上找出现频率最高的1%候选blocks，在从未参与hub定义的另一半query上计算这些blocks占Top64的比例，然后交换两半。表中CI以query为独立单位配对bootstrap：

| Top64方法 | 状态8 hub mass | 状态64 hub mass |
|---|---:|---:|
| 文本BM25+E5 RRF | **0.284%** `[0.157%,0.439%]` | **0.018%** `[0%,0.053%]` |
| QK raw head-RRF | 14.90% `[12.94%,16.77%]` | 9.05% `[7.64%,10.52%]` |
| QK K-centered head-RRF | **25.46%** `[24.56%,26.30%]` | 3.93% `[3.18%,4.72%]` |
| QK raw token-max | 15.62% `[12.95%,18.45%]` | 0.031% `[0%,0.078%]` |
| QK K-centered token-max | 1.98% `[1.58%,2.42%]` | 3.29% `[2.43%,4.22%]` |

除状态64的raw token-max外，QK稳定hub mass均显著高于文本RRF。去K均值能把早期token-max hub从15.6%降到2.0%，却使head-RRF达到25.5%；这说明hub不只是单一全局均值，还取决于layer/head与跨head聚合规则。简单全局centering不是充分修复。

QK轨迹也比文本更不稳定：prefix16之后，多数QK方法的相邻Top8 Jaccard只有0.001-0.037，旧Top512覆盖新Top8通常只有2.6%-18.3%；相同XSum文本RRF分别为0.245-0.307和73.6%-78.5%。这解释了为什么QK适合**当前状态下的候选响应/精排**，却不适合无门控地承担全局、跨步稳定主召回。

但是hub去除不是QK召回不足的全部答案：最强频率惩罚只让XSum raw head-RRF的Top8 source any从7%变为9%，CI仍含0；QK Top512 source any也只有21%-28%，远低于文本通道此前的96%-98%。因此应把QK score写成profile条件分解：

```text
score_p(z_t, b) = hub_prior_p(b) + state_delta_p(z_t, b) + noise_p(z_t, b)
```

其中`p=(layer, head, aggregation, state regime)`。离线可估计`hub_prior_p`并做whitening/IDF式惩罚，但最终是否读取仍需scope/text候选与utility验证，不能把去hub本身当成完整检索器。

QK与文本原始轨迹：[xsum_10m_dynamic_qk_retrieval_rows_20260715.jsonl](1b_context_search_research_exploration/evidence/xsum_10m_dynamic_qk_retrieval_rows_20260715.jsonl)、[xsum_10m_dynamic_text_retrieval_rows_20260715.jsonl](1b_context_search_research_exploration/evidence/xsum_10m_dynamic_text_retrieval_rows_20260715.jsonl)

### 8.22 无target的增量刷新门：集合失效可预测，证据事件更稀疏且更难跨域

8.20节只证明了旧frontier大多可复用，并没有回答部署时最关键的问题：**当前不能查看完整全库排序和gold source，如何知道这一步必须刷新？** 为此在同一批真实约10M轨迹上构造严格在线事件：

1. 上一状态的RRF Top512作为旧frontier；
2. 当前状态只在旧512内计算BM25、E5和RRF，不访问当前全库排名；
3. 特征仅包括query token/E5漂移、BM25-E5 agreement、候选内分数margin/熵/正分数比例、前后候选分数Spearman和Top1变化；
4. 当前完整10M检索只用于离线标注“旧frontier失效”和“刷新是否救回source”；
5. 同一query的三个状态转移始终在同一折，并额外做leave-one-domain-out，禁止相邻状态和域信息泄漏。

三域共160条query、480次状态转移。先区分两个不同目标：

| 域 | Top8超过25%离开旧Top512 | 真正因不刷新而漏掉source | 平均Top8 miss fraction |
|---|---:|---:|---:|
| XSum新闻 | 37.0% | 8.33% | 27.7% |
| PG19书籍 | 44.4% | 6.67% | 31.1% |
| 代码repo | 27.8% | 7.78% | 20.8% |
| 合计 | **36.7%** | **7.92%** | **27.1%** |

这说明候选集合经常漂移，但真正影响当前source coverage的新证据事件很稀疏。集合变化不是读取价值，二者不能共用一个标签。

候选失效预测是稳定的。只用当前旧frontier可计算的44个target-free特征，按query分组5折的logistic AUC为0.834，query-cluster bootstrap 95% CI `[0.793,0.867]`；leave-one-domain-out AUC仍为0.797，CI `[0.754,0.835]`。其中只用状态漂移的AUC为0.713，加入BM25/E5候选内agreement后明显提高。连续miss fraction的跨域random-forest Spearman为0.615，说明当前候选竞争几何确实携带“旧frontier正在失效”的信息。

但严格source刷新事件只有38个正例，跨域泛化明显更弱：

| source刷新门 | query分组AUC | leave-one-domain-out AUC | 结论 |
|---|---:|---:|---|
| 状态漂移logistic | 0.658 | 0.412 | 域内相关，跨域反转 |
| agreement forest | 0.671 | 0.552 | 略高于随机，仍弱 |
| compact forest | 0.703 | **0.615** | 当前最好，但CI `[0.533,0.698]` |
| 全特征logistic | **0.725** | 0.613 | 域内最好，跨域不稳定 |

因此当前不能声称已经学会“何时必需刷新证据”。更复杂的监督门控没有稳定超过简单query embedding漂移；主要原因是source事件稀少，而且`source any@8`本身只表示人工source覆盖，不等于生成效用。

在统一刷新预算下比较策略。`source any@8`按每个当前状态计算，成本把旧512候选重排和完整10M索引访问都计入；速度倍数是索引block访问量相对每步全局搜索的缩减，不包含reader forward：

| 策略 | 刷新率 | source any@8 | 相对从不刷新 | strict事件召回 | 索引访问缩减 |
|---|---:|---:|---:|---:|---:|
| 从不全局刷新 | 0% | 46.04% | - | 0% | 304.6x |
| 固定在prefix32刷新 | 33.3% | 48.12% | +2.08点 | 34.2% | 2.97x |
| 固定在prefix64刷新 | 33.3% | 48.54% | +2.50点 | 39.5% | 2.97x |
| query E5漂移Top20% | 20% | 48.33% | **+2.29点**，CI `[+0.63,+3.96]` | 39.5% | **4.92x** |
| query E5漂移Top33% | 33.3% | **49.58%** | **+3.54点**，CI `[+1.46,+5.63]` | **57.9%** | 2.97x |
| 每步全局刷新 | 100% | 51.25% | +5.21点 | 100% | 1.00x |
| 只刷新正收益事件oracle | 7.92% | 53.96% | +7.92点 | 100% | 12.13x |

oracle高于“每步刷新”不是矛盾：完整新排序偶尔会挤掉旧frontier里已有的source；oracle只执行能带来正增益的刷新，跳过有害刷新。它只用于量化上限，部署时不可使用。

当前最稳健的工程结论不是训练复杂gate，而是一个保守的三级控制：

```text
FAST:  每步只在旧512 frontier内重排
EVENT: query-state漂移或BM25/E5 agreement恶化时注入新postings
SAFE:  低频周期性全局刷新，防止长期累积漏召回
```

本实验仍只测retrieval coverage，没有把刷新策略接入未来NLL或答案正确率；10M上的4.92x是索引访问缩减，不是端到端生成加速。下一步必须用真实连续生成utility作为事件标签，并在1B CPU/SSD层级索引上报告page I/O、cache命中和整体吞吐。

证据与逐事件预测：[cross_domain_10m_incremental_refresh_gate_20260715.json](1b_context_search_research_exploration/evidence/cross_domain_10m_incremental_refresh_gate_20260715.json)、[cross_domain_10m_incremental_refresh_gate_predictions_20260715.jsonl](1b_context_search_research_exploration/evidence/cross_domain_10m_incremental_refresh_gate_predictions_20260715.jsonl)、[xsum_10m_incremental_refresh_features_20260715.json](1b_context_search_research_exploration/evidence/xsum_10m_incremental_refresh_features_20260715.json)、[pg19_10m_incremental_refresh_features_20260715.json](1b_context_search_research_exploration/evidence/pg19_10m_incremental_refresh_features_20260715.json)、[longbench_code_10m_incremental_refresh_features_20260715.json](1b_context_search_research_exploration/evidence/longbench_code_10m_incremental_refresh_features_20260715.json)

### 8.23 1B的主要瓶颈是上层极值竞争：事件稀疏必须与多级缩域相乘

真实1B索引尚未运行，但现有严格past-only PG19包含嵌套的9.9M、20M、50M和100M真实文本，可以先检验一个可校准的外推模型。对每个query-state，记`k_N`为N-token记忆中BM25分数超过正确book的错误scopes数，`n_N`为全部错误scopes数。100M使用1,118本真实书、1,562,500个64-token blocks，四个尺度共120条query-state轨迹；固定100M IDF后，`k_N`随嵌套尺度0次违反单调性。

先用当前尺度的`k_N/n_N`拟合Beta-Binomial后验，预测下一真实尺度，而不是直接跳到1B：

| 真实外推 | 错误scopes预测/实际 | count Spearman | Top3预测/实际 | Top8预测/实际 |
|---|---:|---:|---:|---:|
| 9.9M -> 20M | 8.38 / 8.64 | 0.959 | 76.6% / 75.0% | 85.6% / 85.0% |
| 20M -> 50M | 20.23 / 20.59 | 0.923 | 70.7% / 69.2% | 77.5% / 78.3% |
| 50M -> 100M | 39.32 / 38.96 | **0.966** | 64.3% / 65.8% | **72.2% / 72.5%** |

相邻真实尺度校准良好后，再假设新增PG19书籍与现有distractor scopes可交换、book大小分布有界，把1,118个scopes扩到约11,180个。结果是统计外推，不是真实1B运行：

| 1B book router深度 | 预测正确book recall | query-cluster 95% CI | prefix512 recall |
|---|---:|---:|---:|
| Top3 | 50.1% | `[36.9%,62.9%]` | 62.2% |
| Top8 | 55.8% | `[41.8%,69.2%]` | 69.4% |
| Top32 | 66.4% | `[51.8%,79.5%]` | 77.0% |
| Top128 | 76.5% | `[63.9%,88.1%]` | 86.4% |
| Top256 | 82.3% | `[71.9%,91.6%]` | 88.4% |
| Top512 | 87.8% | `[78.8%,95.4%]` | 99.7% |
| Top1024 | 92.5% | `[85.2%,98.3%]` | 100% |

平均达到80%/90%/95%预计需要Top256/1024/2048 books。更重要的是地址成熟会改变所需宽度：达到80%时，prefix64/128/256/512分别需要Top1024/512/128/64；达到90%时prefix64在当前Top2048范围内仍不足，prefix128/256/512分别需要Top1024/256/512。这不是严格单调的单query策略，而是宏平均后验，足以说明**固定Top-D不是1B稳定属性，D必须依赖状态成熟度和router不确定性。**

仅扩大book Top-D也不可行。按100M实际被选books的大小，1B时D8/D32分别需要在22,770/89,970个block索引项内精排；若20%的生成事件刷新，连同常驻512 frontier，平均每步访问约5,066/18,506个block索引项，相对每步扫描15,625,000 blocks是3,084x/844x缩减。但D1024若仍把book内blocks平铺，估计要触碰291万个候选blocks；即使只在20%事件刷新，平均也约58万项，只有26.8x缩减。

这给出了两个不能互相替代的乘法项：

```text
1B search reduction
  = hierarchical domain reduction
  x sparse refresh-event reduction
  x bounded final working-set selection
```

单独20%刷新最多约5x；单独一层book路由又会为高recall保留数百至上千books。合理结构必须是：

```text
book/owner/repo coarse router
  -> chapter/segment/session/file router
  -> bounded 0.5K frontier
  -> utility-selected 1-3 blocks / 0.5K-4K reader tokens
```

线性存储外推中，当前文本block/scope BM25 sidecar约为4.84GB/1.09GB；这不包含原文、KV、SSD布局和动态更新，也不代表实测RAM驻留或吞吐。该实验只支持“多级层次和状态自适应宽度是1B必要条件”，不能替代真实1B CPU/SSD端到端实验。

证据：[pg19_past_only_1b_router_extrapolation_20260715.json](1b_context_search_research_exploration/evidence/pg19_past_only_1b_router_extrapolation_20260715.json)、[pg19_past_only_100m_hier_bm25_v2_rows_20260715.jsonl](1b_context_search_research_exploration/evidence/pg19_past_only_100m_hier_bm25_v2_rows_20260715.jsonl)

### 8.24 当前10M主实验：book -> segment -> block在固定512-token工作集下改善reader

按当前研究范围，不需要实际运行1B。为了验证8.23节导出的第三级结构是否在10M真实数据上成立，在9.9M严格past-only PG19上增加固定连续segment层：每个segment最多64个blocks，即4,096 tokens；query只使用已观察的512-token状态，不使用future target或预定义source。30条query的最终reader候选始终为Top8 blocks，即512 tokens。

| 方法 | book候选 | segment候选 | block精排域 | 查询时间 | Top8同书 | Top8局部4K | 同书纯度 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 全局BM25 | 135 | 2,477 | 154,688 | **5.09ms** | 96.7% | 80.0% | 70.0% |
| book8内平铺 | 8 | 253 | 15,994 | 7.54ms | **100%** | **90.0%** | 80.4% |
| book8 -> segment128 | 8 | 253 | 8,135 | 9.63ms | **100%** | **90.0%** | 80.4% |
| book8 -> segment32 | 8 | 253 | 2,018 | 7.73ms | **100%** | **90.0%** | 81.2% |
| book8 -> segment8 | 8 | 253 | **493** | 7.31ms | **100%** | 86.7% | **87.5%** |
| book32 -> segment8 | 32 | 916 | 493 | 17.54ms | 96.7% | 83.3% | 84.6% |

`book8 -> segment8`相对全局把block精排域缩小约314倍，并提高候选纯度；局部4K命中比book8平铺低3.3个百分点，但仍比全局高6.7个百分点。它说明固定连续segment能够作为真实的中间路由单元，而不是只在公式里存在。

速度结果同样重要：10M的全局稀疏BM25 postings已经很快，三级路由增加了book和segment两次固定开销，因此7.31ms没有超过5.09ms。不能根据候选数量直接宣称端到端更快；层级索引的价值首先是结构保真、I/O上界和更高纯度，速度交叉点需要在同一硬件上由更大索引或SSD分层实测。

随后在同一批30条query上运行Qwen3-0.6B reader：状态长度512 tokens、未来target 128 tokens；除query-only外，每种方法都只加载Top8个64-token blocks，即固定512-token工作集。以下PPL差异因此来自“选了什么”，不是“给了模型更多tokens”。

| 方法 | block精排域 | reader tokens | PPL | ΔNLL vs query-only | 95% CI |
|---|---:|---:|---:|---:|---:|
| Query only | 0 | 0 | 22.377 | 0 | - |
| Random512 | 全库随机 | 512 | 22.883 | +0.0223 | `[+0.0097,+0.0348]` |
| 全局BM25 | 154,688 | 512 | 22.255 | -0.0055 | `[-0.0252,+0.0133]` |
| book8内平铺 | 15,994 | 512 | 22.040 | -0.0152 | `[-0.0338,+0.0028]` |
| book8 -> segment128 | 8,135 | 512 | 22.047 | -0.0149 | `[-0.0332,+0.0032]` |
| **book8 -> segment32** | **2,018** | **512** | **21.955** | **-0.0190** | **`[-0.0373,-0.0008]`** |
| book8 -> segment8 | 493 | 512 | 22.091 | -0.0129 | `[-0.0309,+0.0057]` |

`book8 -> segment32`相对全局BM25的逐query ΔNLL为-0.0136，95% CI `[-0.0232,-0.0054]`；相对过宽的segment128为-0.0042，CI `[-0.0090,-0.0002]`；相对更窄的segment8为-0.0062，但CI `[-0.0165,+0.0035]`跨0。这里不能宣称32是普适最优值，只能说当前数据出现了**中间预算甜点**：过宽会引入干扰，过窄会丢掉局部证据，而最终reader token预算保持不变。

时间必须分开解释。CPU检索原型中，全局BM25为5.09ms，`book8 -> segment32`为7.73ms；reader forward约37.9ms。把两者机械相加只能得到约45.6ms的顺序原型估计，不能当成经过流水线和I/O优化后的端到端吞吐。当前10M结论是**质量更好、访问域小76.6倍，但尚未比优化过的全局稀疏BM25更快**。

证据：[三级检索汇总](1b_context_search_research_exploration/evidence/pg19_past_only_multilevel_10m_tuning_20260715.json)、[reader PPL](1b_context_search_research_exploration/evidence/pg19_past_only_multilevel_10m_ppl_s512_20260715.json)、[逐query配对分析](1b_context_search_research_exploration/evidence/pg19_past_only_multilevel_10m_analysis_20260715.json)

### 8.25 地址成熟的真正形式：排名改善，但score不一定变尖

为了直接测试“随着生成状态变化持续检索”，在同一真实9.9M-token PG19记忆上固定`book8 -> segment32 -> Top8 blocks`，只改变模型已观察的状态长度。候选精排域始终约2,020个blocks，最终工作集始终512 tokens。

| 已观察状态 | 真实book平均排名 | book Top8命中 | 最终Top8同书命中 | 局部4K命中 | 同书纯度 | block精排域 |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 9.80 | 83.3% | 73.3% | 56.7% | 48.8% | 2,021 |
| 128 | 5.50 | 90.0% | 86.7% | 73.3% | 60.8% | 2,019 |
| 256 | 4.13 | 96.7% | 96.7% | 83.3% | 78.3% | 2,017 |
| 512 | **1.40** | **100%** | **100%** | **90.0%** | **81.2%** | 2,018 |

64到512状态的逐query变化为：真实book排名-8.4，95% CI `[-16.87,-1.87]`；Top8同书命中+26.7个百分点，8胜0负，p=0.0078；局部4K命中+33.3个百分点，11胜1负，p=0.0063。30条轨迹中93.3%的真实book排名单调不升，Top8命中100%单调不降。**状态增加改善了地址，但没有要求更大的候选域或工作集。**

然而可在线观察的分数几何给出反直觉结果：

| 在线量 | 状态64 | 状态512 | 配对变化95% CI |
|---|---:|---:|---:|
| query非零词项数 | 19.7 | 129.1 | `[+103.7,+114.7]` |
| Top8正分mass | 0.151 | 0.134 | `[-0.0272,-0.0072]` |
| 归一化score entropy | 0.963 | 0.973 | `[+0.0048,+0.0165]` |
| Top1 z-score | 4.85 | 5.23 | `[-0.13,+0.92]` |

也就是说，地址成熟表现为**正确scope的相对排名改善**，并不表现为全局分布更尖。每个状态转移中，entropy、Top8 mass和Top1 z的变化与真实排名改善的Spearman都接近0。静态margin/entropy gate因此不是可靠的通用刷新或STOP条件。

最后用未来128-token ΔNLL做离线标签，并严格按query做5折OOF控制器比较：

| 策略 | 在线使用target | 平均ΔNLL vs query-only | 说明 |
|---|---:|---:|---|
| 训练折选择的固定方法 | 否 | -0.0197 | 每个fold只从训练query选一个固定方法 |
| 只按状态长度选择方法 | 否 | **-0.0227**，CI `[-0.0448,-0.0024]` | 比固定方法多改善-0.0030，但差值CI跨0 |
| 静态geometry随机森林 | 否 | -0.0095 | 使用margin/entropy/mass等，明显退化 |
| 每事件oracle | 是，不可部署 | -0.0559 | 仅表示action选择仍有较大上限 |

保守结论是：**状态长度本身已经是有用的预算先验，但现有静态置信特征还不能可靠决定reader action。** 一个合理系统应让早期状态保留更宽的scope分支，地址成熟后缩窄；是否扩segment或刷新frontier，则使用相邻状态的query漂移、候选集合变化、跨通道agreement和少量候选试读，而不是只看当前分数是否尖锐。

证据：[四状态三级检索](1b_context_search_research_exploration/evidence/pg19_past_only_multilevel_10m_all_states_20260715.json)、[地址成熟配对分析](1b_context_search_research_exploration/evidence/pg19_past_only_10m_address_maturation_20260715.json)、[动态控制器](1b_context_search_research_exploration/evidence/pg19_past_only_multilevel_10m_dynamic_controller_20260715.json)、[状态64 reader](1b_context_search_research_exploration/evidence/pg19_past_only_multilevel_10m_ppl_s64_20260715.json)、[状态128 reader](1b_context_search_research_exploration/evidence/pg19_past_only_multilevel_10m_ppl_s128_20260715.json)、[状态256 reader](1b_context_search_research_exploration/evidence/pg19_past_only_multilevel_10m_ppl_s256_20260715.json)

### 8.26 77本独立书籍确认实验：强缩域成立，静态/轨迹gate失败，回放loss只有弱增量

8.24-8.25节只有30个query，足以发现性质但不足以选择层级深度。本节使用PG19 test中全部77本满足严格长度条件的书，每本只产生一个query；另有70本真实distractor书。记忆仍为9,900,032 tokens、154,688个64-token blocks，不含合成文本。为了填满10M而不复制数据，把每本允许的真实past history上限由131K提高到256K；77本query history全部完整写入，query-book future block违规为0。

#### 地址成熟在更多独立书籍上复现

固定`book8 -> segment32`作为诊断路径，候选精排域始终约2,020 blocks：

| 状态tokens | 正确book平均排名 | Top8同书命中 | 局部4K命中 | 同书纯度 | block精排域 |
|---:|---:|---:|---:|---:|---:|
| 64 | 3.52 | 81.8% | 48.1% | 58.3% | 2,017 |
| 128 | 2.27 | 94.8% | 63.6% | 72.9% | 2,024 |
| 256 | 1.66 | 97.4% | 68.8% | 82.6% | 2,020 |
| 512 | **1.45** | **97.4%** | **79.2%** | **85.4%** | 2,019 |

64→512的真实book平均排名变化为-2.06，95% CI `[-3.39,-0.92]`；Top8同书命中增加15.6个百分点，12胜0负，`p=4.88e-4`；局部4K命中增加31.2个百分点，26胜2负，`p=3.03e-6`。与此同时，score entropy由0.959升至0.976，Top8 score mass由0.146降至0.118，再次确认**地址成熟是正确scope的相对排序性质，不是分布变尖性质**。

#### 固定512-token reader下，segment8优于平铺BM25和E5，尚未显著胜Hybrid

所有方法都只给Qwen3-0.6B读取8个64-token blocks，并预测随后128 tokens：

| 状态 | Query only | Random512 | 三级`book8-seg8` | 全局BM25 RAG | E5 RAG | BM25+E5 Hybrid |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 32.335 | 33.380 | **30.069** | 31.803 | 30.979 | 30.330 |
| 256 | 29.998 | 30.857 | **28.377** | 29.042 | 28.986 | 28.520 |
| 512 | 28.067 | 28.690 | **26.993** | 27.323 | 27.873 | 27.327 |

以书籍query为bootstrap单位，并先在每本书内平均三个状态：

| 配对基线 | 三级方法ΔNLL | query-cluster 95% CI | 胜/负 | 结论 |
|---|---:|---:|---:|---|
| Query only | **-0.0557** | `[-0.0780,-0.0363]` | 59/18 | 显著改善 |
| 全局BM25 | **-0.0304** | `[-0.0415,-0.0199]` | 56/21 | 显著改善 |
| E5 | **-0.0277** | `[-0.0454,-0.0118]` | 53/24 | 显著改善 |
| BM25+E5 Hybrid | -0.00865 | `[-0.0188,+0.0014]` | 43/34 | 方向更好，尚不显著 |

这也修正了30-query探索中的深度结论：当时`segment32`最好；扩大到77本后，`segment8`在128/256/512三个状态都最好或近似最好。不能把32写成模型固有常数。更稳定的性质是：**先用自然scope和连续segment把竞争域缩到约500，再从中选最终8 blocks；具体中间深度需要独立query校准。**

#### 在线成本边界

| 状态 | 三级499候选 | 全局BM25 | E5 exact | Hybrid exact |
|---:|---:|---:|---:|---:|
| 128 | 3.88ms | **2.63ms** | 6.99ms | 10.52ms |
| 256 | 4.19ms | **3.65ms** | 6.98ms | 11.47ms |
| 512 | **4.93ms** | 5.10ms | 6.94ms | 12.88ms |

三级方法的block精排域约499，相对154,688缩小约310倍。层级BM25的block+segment+book索引构建为11.54s、约92.4MB；E5 passage编码为64.58s，154,688×768 float32 embedding约475.2MB。当前E5是exact matrix retrieval，生产ANN可降低在线计算，因此这些数字只能证明本原型的结构访问和存储优势，不能代表所有dense RAG实现。

#### 为什么轨迹变化仍不能直接控制reader

在77本、231个状态事件上，静态router策略、router trajectory和frontier trajectory相对query-only的ΔNLL分别为-0.0540、-0.0477和-0.0447。router trajectory相对静态反而恶化+0.00624，CI `[+0.00203,+0.01085]`；frontier trajectory恶化+0.00929，CI `[+0.00445,+0.01453]`。因此scope/block集合churn虽然能预测“候选是否变化”，不能推出“变化后的候选能改善模型”。

进一步对每个当前候选重放最近已观察的64 tokens，不读取未来128-token target。回放ΔNLL与未来action ΔNLL的事件内去中心化Spearman为0.169，说明存在弱而稳定的自监督信号；但：

| 控制策略 | 平均未来ΔNLL vs query-only | 相对状态策略 |
|---|---:|---:|
| 状态先验 | -0.05361 | - |
| 先验 + 回放loss收缩 | **-0.05454** | -0.00093，CI `[-0.00301,+0.00088]` |
| 直接取回放loss最小action | -0.04473 | 明显更差 |
| future oracle | -0.07849 | 不可部署上限 |

七个action顺序回放平均需要296ms，每个候选forward约35.9ms；即使批处理后，当前不显著的0.00093 NLL增益也不足以支付该成本。回放loss目前应定位为**少量边界action的确认信号**，而不是遍历候选的主检索器。

#### 与RAG的严格边界

当前最强三级方法仍使用BM25和真实book/segment容器，算法上属于结构化层级RAG；“层级检索”本身不能作为与RAG不同的贡献。当前已经得到支持、但尚未形成端到端优势的非RAG部分是：

1. 检索query随生成状态持续变化，而不是一次性问题embedding；
2. 已加载KV/frontier跨token复用，只在事件发生时刷新；
3. 用模型自身已观察token的causal loss估计候选utility；
4. 目标是`Address × InformationGain - Interference - ReadCost`，而不只是文本相关度。

要真正超过RAG，下一阶段必须在相同层级BM25/E5候选上证明：量化KV page bound或直接KV复用能以更低prefill/I/O成本取得额外utility，并且无future target的gate能够稳定决定KEEP/REFRESH/EVICT。当前结果尚未证明这一点。

证据：[77-query数据审计](1b_context_search_research_exploration/evidence/pg19_past_only_10m_q77_data_20260715.json)、[四状态层级检索](1b_context_search_research_exploration/evidence/pg19_past_only_multilevel_10m_q77_all_states_20260715.json)、[地址成熟](1b_context_search_research_exploration/evidence/pg19_past_only_10m_q77_address_maturation_20260715.json)、[层级与RAG配对](1b_context_search_research_exploration/evidence/pg19_past_only_10m_q77_hierarchy_vs_rag_20260715.json)、[轨迹控制器](1b_context_search_research_exploration/evidence/pg19_past_only_10m_q77_trajectory_controller_20260715.json)、[64-token因果回放](1b_context_search_research_exploration/evidence/pg19_past_only_10m_q77_retrospective_action_controller_20260715.json)

### 8.27 LongMemEval全500题：状态创新应追加到层级frontier，而不是替换旧Top-K

#### 协议与因果口径

完整500题的历史无法同时装入一个10M记忆，因此按问题类型和历史长度平衡划分为8个互不重叠的shards；每个shard都独立包含严格10,000,000 tokens和156,250个64-token blocks，不能把它们写成一次80M检索。500题中470题可回答、30题为abstention。Qwen3-8B只看到问题，不读取memory或答案，生成64-token实体/关系/时间检索计划；参考答案只做事后字符串泄漏审计，排除问题中原有选项后新增答案重合为0。

每题有八个状态：问题文本的25%/50%/75%/100%，以及完整问题后8/16/32/64个生成计划tokens。每个状态都从自己的10M shard检索，最终Top8仍只对应512 tokens。owner metadata表示实际系统可用的tenant/user namespace，不是证据oracle。

#### 问题到达时地址成熟，但生成计划不是单调增益

| 方法 | 25%问题任一证据 / 全部session | 100%问题任一证据 / 全部session | 加64计划tokens后 |
|---|---:|---:|---:|
| 全局block BM25 | 5.3% / 4.9% | **55.3% / 43.8%** | 51.1% / 38.9% |
| 全局session3→block | 4.0% / 4.5% | **48.7% / 30.6%** | 41.9% / 23.2% |
| 已知owner session3→block | 20.0% / 21.1% | **85.3% / 73.8%** | 84.7% / 73.4% |
| 未知owner8→session3→block | 4.3% / 4.5% | **47.0% / 30.2%** | 39.1% / 24.7% |

问题由25%增长到100%时地址显著成熟；但完整问题之后继续追加计划文本通常只是在重述、扩写和稀释词法约束。未知owner Top8路由命中由完整问题的66.6%降到64-token计划后的54.7%。因此**生成状态增加不等于当前Top-K单调变好**，盲目用新检索替换旧结果会丢证据。

#### 持久frontier把有害平均变化转化为稀疏救回

从完整问题Top8开始，依次保留plan8/16/32/64的Top8并去重：

| 路由 | 五状态唯一工作集 | 页面复用 | 静态Top8→动态并集任一证据 | 全部证据session | 动态并集相对同数量静态结果 |
|---|---:|---:|---:|---:|---|
| 全局block | 952 tokens | 62.8% | 55.3%→60.2% | 43.8%→49.6% | 任一证据-2.3pp；全部-0.4pp，均不优 |
| 全局session3→block | 1,016 tokens | 60.3% | 48.7%→55.3% | 30.6%→35.1% | +3.0pp，p=.020；+4.3pp，p=1.91e-6 |
| owner Hybrid3→block | 769 tokens | **70.0%** | 84.0%→90.2% | 68.5%→73.8% | +3.6pp，p=9.11e-4；+4.9pp，p=2.38e-7 |
| owner session3→block | **763 tokens** | **70.2%** | 85.3%→90.2% | 73.8%→78.1% | +1.3pp，p=.263；+2.8pp，p=.00235 |
| owner-router8→session3 | 1,151 tokens | 55.0% | 47.0%→54.7% | 30.2%→37.0% | +2.6pp，p=.073；+5.5pp，p=2.16e-7 |

平铺全局BM25失败是关键负对照：给完整问题一次性读取相同数量的静态Top-K即可追平或超过动态并集。层级方法不同，因为增加block深度仍被限制在静态选择的sessions中；生成状态虽然平均更差，却偶尔把另一个必要session路由进来。保留旧页避免loss，追加少数新页捕获rescue，所以收益主要出现在`all_evidence_sessions`而非单个相关块。

任务拆分支持这一解释。owner Hybrid同预算动态并集相对静态结果，在multi-session上的全部证据覆盖为38.0%→48.8%，temporal-reasoning为57.5%→63.0%；single-session-assistant已经为100%→100%。动态机制不是所有问题都需要，应由关系槽位完整性触发。

#### 固定768-token预算与最强RAG边界

使用最简单的在线策略：完整问题先保留Top8，此后按首次出现顺序追加新页，到12页即停止。owner Hybrid在相同12-page/768-token预算下：

- 任一证据覆盖：静态86.4%→动态89.1%，20胜7负，净增2.8个百分点，`p=.0192`；
- 全部证据session：静态70.0%→动态73.4%，`p=.00154`；
- 全局session3的全部证据覆盖也由31.1%→33.6%，`p=.00183`；
- 平铺全局block在16页时反而少3.4个百分点任一证据、少3.8个百分点全部证据。

这里需要避免过度结论。最强静态`owner session3`在12页时为88.5%任一证据、76.4%全部证据；其动态12页为89.4%/77.9%，配对差均不显著。因此当前实验支持**层级route innovation和持久frontier属性**，没有证明最终系统已经超过最强静态RAG。

#### 对直接KV页复用的含义

已知owner的四次计划刷新平均每次只新增0.92-1.35个64-token pages，即59-86 tokens；47%-67%的刷新Top8完全不变。若每一步都把8个文本块重新prefill，五状态累计为2,560 tokens；按页面缓存只处理唯一块，平均为763-769 tokens，理论上减少约70%的重复prefill。P95工作集为owner session3的1,280 tokens、Hybrid的1,344 tokens，仍在“几百到几千tokens”目标内。

但当前8B检索计划本身平均需要2.69s/题，它只是构造无答案状态轨迹的属性探针，不能计作可部署加速。真正系统必须复用模型本来就会生成的状态，并验证pre-RoPE/position-remapped KV pages能直接加载、无需重复文本prefill。

证据：[全500题、8个独立10M shards配对汇总](1b_context_search_research_exploration/evidence/longmemeval_all500_independent_10m_state_frontier_20260715.json)

### 8.28 证据条件自然状态闭环：检索完整性提高，但reader收益被页面替换损失抵消

8.27节的64-token search plan只读取问题，主要用于分离“问题状态增长”和“持久frontier”两个性质。本节构造更接近真实生成的闭环，但仍不读取答案：

1. 用完整问题在已知owner的BM25 `session3 -> block`索引中取初始Top8，共512 tokens；
2. Qwen3-8B只读取问题和这8页，生成最多32 tokens的“已支持事实 + 尚缺关系槽位”，提示词禁止猜测和输出最终答案；
3. 在状态prefix 8/16/32时刷新同一层级索引；保留初始页面，只按首次出现顺序追加新页，到12页停止；
4. 参考答案和gold session不参与生成、检索或页面选择，只用于事后评估。

因此这不是把答案改写成query，也不是oracle bridge。状态只能复制已经读到的事实，或者指出仍然缺少的关系。

#### 检索结果

以下只统计470个可回答问题，每题仍在自己的独立真实10M-token shard中检索：

| 工作集 | 平均读取tokens | 任一精确证据block | 全部证据sessions | 平均session recall |
|---|---:|---:|---:|---:|
| 静态Top8 | 512 | 85.32% | 73.83% | 85.12% |
| 静态Top12 | 768 | 88.51% | 76.60% | 86.61% |
| 证据状态动态Top12 | **672.5** | **89.57%** | **79.15%** | **88.67%** |

动态方法相对静态Top12的任一精确block为`+1.06pp`，13胜8负，`p=.383`；全部证据session为`+2.55pp`，20胜8负，`p=.0357`；连续session recall为`+2.06pp`，bootstrap 95% CI `[+0.76,+3.44]`。因此证据条件状态确实能以更少平均读取量发现部分静态深挖遗漏的session，但主要收益是**复合证据完整性**，不是单块召回。

三次刷新平均每次只新增0.845页，中位数0页，P95为4页；总计平均新增2.54页，中位数3页，最多4页。状态生成平均1.52s，三次BM25层级查询合计约3.11ms。当前慢点是额外8B状态生成，不是10M索引查询；部署时只能复用模型自然生成状态或用更轻量slot state，不能把这1.52s写成加速。

事后泄漏审计也揭示了状态来源：初始精确证据未命中的69题中，只有4.35%的状态提及参考答案，只有2.90%加入问题中原本没有的参考答案；初始命中的401题则分别为23.69%和20.45%。这支持状态主要是在**压缩已读证据和显式化缺口**，不是靠参数记忆稳定猜出答案。

#### Qwen3-8B reader结果

为检验coverage是否真的转化为生成质量，对静态Top12和动态Top12运行同一个Qwen3-8B reader。主指标是teacher-forced参考答案NLL；token F1、EM、答案包含率只是确定性短答案生成的探索指标，不等同于LongMemEval官方LLM judge。

| 指标 | 静态Top12 | 证据状态动态Top12 | 配对结论 |
|---|---:|---:|---|
| 全500题参考答案NLL | **4.3831** | 4.4034 | `Delta=+0.0203`，95% CI `[-0.129,+0.166]` |
| 470题答案NLL | **4.3253** | 4.3430 | `Delta=+0.0177`，CI跨0 |
| token F1 | 37.39% | **39.11%** | `+1.72pp`，95% CI `[-0.78,+4.25]` |
| exact match | 24.26% | **25.11%** | `+0.85pp`，24胜20负，`p=.652` |
| 答案包含率 | 26.38% | **28.51%** | `+2.13pp`，24胜14负，`p=.143` |
| 30题拒答正确率 | 70.00% | **73.33%** | 2胜1负，样本不足 |

总体上，动态方法平均少读约94 tokens，F1、EM和答案包含率方向更好，但所有生成质量差异均不显著；参考答案NLL甚至方向略差。**不能据此宣称已经超过静态RAG。**

最重要的结果来自按检索完整性变化拆分：

| 动态相对静态的证据变化 | 题数 | reader `Delta NLL` | token F1变化 |
|---|---:|---:|---:|
| rescue：静态不完整、动态完整 | 20 | **-1.044** | **+17.7pp** |
| loss：静态完整、动态不完整 | 8 | +1.899 | -18.1pp |
| 两者都完整 | 352 | +0.064 | +1.35pp |
| 两者都不完整 | 90 | -0.093 | +1.39pp |

这说明检索变化与reader变化在机制上严格同向：救回缺失session时收益很大，挤掉静态第9-12页中的必要证据时损失更大。20次rescue的总收益被8次loss和大量中性题抵消。当前最关键的问题不再是“状态能否找到新页面”，而是：

```text
APPEND new_page
iff expected_completeness_gain(new_page | required_slots, W_t)
   > displacement_cost(evicted_page | required_slots, W_t)
```

也就是说，`KEEP + APPEND`仍不够；固定12页时必须有**slot completeness + displacement utility**门。

#### 测试时无答案的跨10M-shard门控

为检查上述门是否能由廉价轨迹元数据实现，进一步以8个独立10M shards轮流作为完全未见测试集。训练折可用reader NLL学习“动态减静态”的期望收益，但测试折严格禁止参考答案、gold evidence、召回标签、reader NLL、生成答案和事后答案重合标记。三组预先定义的特征依次为：

- `trajectory_only`：新增页数、工作集重合、session route变化、连续性和页面簇形状；
- `structured`：再加问题类型、状态长度和显式`unknown/missing/need`等缺口标记；
- `structured + state text`：再加训练折拟合的状态文本TF-IDF。

结果没有学到可部署门控：

| 测试时可见特征 | 动态优于静态的AUC | 选择动态比例 | 选择后NLL | 相对全静态NLL |
|---|---:|---:|---:|---:|
| trajectory only | 0.517 | 40.0% | 4.4174 | +0.0343，CI跨0 |
| structured | 0.527 | 39.4% | 4.4066 | +0.0235，CI跨0 |
| structured + state text | 0.527 | 39.8% | 4.4090 | +0.0259，CI跨0 |
| 始终静态 | - | 0% | **4.3831** | - |
| 始终动态 | - | 100% | 4.4034 | +0.0203 |

三种ridge模型在所有外层折都选择最强正则`alpha=1000`，预测值与真实`Delta NLL`的Spearman只有0.007-0.025。最佳structured门对20个rescue选择动态60%，对8个loss也选择动态50%，几乎不能区分两者。作为不可部署诊断上限，逐题oracle二选一可把NLL降到3.9824，说明选择空间有较大潜力，但轨迹元数据没有暴露这项选择所需的信息。

#### 看候选内容是否足够

随后解码每题静态/动态候选页，但不重新扫描或编码10M全库，加入两类测试时可见交互：问题/状态/状态新增实体与新页、被挤出页的词法覆盖与冗余，以及E5-base-v2的question-page、state-page和page-current-workset相似度。外层验证仍严格按10M shard留一：

| 候选条件特征 | 动态收益AUC | 选择后NLL | 相对全静态NLL |
|---|---:|---:|---:|
| 词法支持/冗余 | 0.512 | 4.3942 | +0.0111，CI跨0 |
| 词法 + 状态文本 | 0.513 | 4.3929 | +0.0098，CI跨0 |
| 词法 + E5候选交互 | **0.514** | **4.3816** | -0.0015，95% CI `[-0.108,+0.105]` |
| 词法 + E5 + 状态文本 | 0.514 | 4.3887 | +0.0056，CI跨0 |

E5版本数值上偶然追平静态NLL，但AUC、Spearman和配对CI都不支持真实门控能力；它对rescue/loss选择动态的比例仍为55.0%/37.5%，样本极少且没有形成reader收益。这个负结果很关键：**可分离的query-page相似度仍是相关性，不是`page | current workset, required slots`的条件集合效用。**

这与8.17节的“未见候选不可辨识”边界一致。状态变化量只能告诉系统可能需要刷新，BM25/E5只能告诉系统候选与query相似；两者都不能判断新页是否补足关系槽位、旧页是否仍有生成效用。下一步必须加入非可分离交互，例如slot-page支持/冲突、候选相对现有workset的信息增益、cross-encoder、量化QK page bound，或只对边界页面运行短counterfactual probe；不能继续堆churn、margin、词袋和bi-encoder分数。

证据：[证据条件状态检索汇总](1b_context_search_research_exploration/evidence/longmemeval_all500_evidence_conditioned_state_20260715.json)、[Qwen3-8B reader配对汇总](1b_context_search_research_exploration/evidence/longmemeval_all500_evidence_conditioned_reader_20260715.json)、[跨10M-shard无答案门控](1b_context_search_research_exploration/evidence/longmemeval_all500_answer_free_utility_gate_20260715.json)、[候选词法/E5条件门](1b_context_search_research_exploration/evidence/longmemeval_all500_candidate_content_utility_gate_20260715.json)

### 8.29 非可分离集合效用：读取当前工作集后，才能决定APPEND/EVICT

8.28节说明轨迹标量、词法和E5都不能区分动态页面的rescue与loss。本节不再给每页独立打相关分，而是直接比较两个**条件集合**：

```text
fixed = 初始8页
static action  = fixed + 静态BM25第9-12页
dynamic action = fixed + 状态刷新提出的新页
```

Qwen3-8B同时看到问题、32-token检索状态、fixed页面和两组候选，不读取参考答案、不生成答案，只判断哪组更能补足全部实体关系、日期、更新和中间事实。直接要求输出A/B出现严重位置偏差：交换候选顺序后只有0.4%的题保持符号一致，因此该分数被保留为失败对照。有效协议改为两个同构forward：分别判断`complete(static action)`与`complete(dynamic action)`的`YES-NO` log-odds，再取差值。

#### 条件分数是否真的识别reader效用

| 测试时可见信号 | 动态优于静态AUC | 与`-Delta NLL` Spearman | rescue平均分 | loss平均分 |
|---|---:|---:|---:|---:|
| 轨迹/状态标量 | 0.517-0.527 | 约0.01-0.03 | - | - |
| 词法/E5候选交互 | 0.512-0.514 | 约0.02 | - | - |
| A/B集合直接选择 | 0.568 | 0.130 | +1.11 | -2.22 |
| **独立完整性差分** | **0.587** | **0.166** | **+2.04** | **-3.91** |

绝对AUC仍不高，但它首次在未见10M shard上形成可用的效用排序。最简单的零阈值不需要训练：动态完整性高于静态时用动态页，否则保留静态页。

| 策略 | 选择动态 | 平均工作集 | 答案NLL | 相对全静态 | rescue/loss选择动态 |
|---|---:|---:|---:|---:|---:|
| 全静态Top12 | 0% | 768 tokens | 4.3831 | - | 0% / 0% |
| 全动态 | 100% | 674 tokens | 4.4034 | +0.0203 | 100% / 100% |
| E5候选门 | 46% | 743 tokens | 4.3816 | -0.0015，CI跨0 | 55.0% / 37.5% |
| **完整性零阈值门** | 47% | **730 tokens** | **4.2620** | **-0.1211，95% CI `[-0.2357,-0.0184]`** | **70.0% / 12.5%** |
| 8-shard训练阈值 | 49% | 729 tokens | 4.2687 | -0.1144，CI不跨0 | 70.0% / 12.5% |
| probe组合ridge | 38.6% | 744 tokens | 4.2729 | -0.1102，CI不跨0 | 75.0% / **0%** |

零阈值门相对全动态也改善`-0.1415 NLL`，CI `[-0.2466,-0.0438]`。答案token F1由静态37.39%提高到39.61%，EM由24.26%提高到25.74%，但这些短答案生成差异尚未单独达到显著。平均probe成本为1.316s/题，约四次1.0K-token级8B forward；它是属性证明，不是可部署速度结果。

#### 收益边界

该平均NLL收益主要来自避免或救回少数灾难性动作，不是每题都小幅改善：234个非tie中126胜108负，符号检验`p=.266`、Wilcoxon `p=.185`；去掉两端1%后均值从`-0.1211`缩到`-0.0797`，5%截尾后为`-0.0284`。零阈值在8个shards中6个改善；组合ridge为7个改善。因而当前可写成：**条件完整性probe能识别高价值尾部事件并显著改善标准平均NLL，但稳定的逐题效用校准尚未完成。**

与RAG的边界也更清楚。BM25/E5继续负责从10M提出十几页；新增能力不是另一种embedding，而是计算：

```text
U(new, old | z, W, required_slots)
  = Complete(W + new) - Complete(W + old) - Interference - ReadCost
```

它显式条件于已经读入的`W`和被替换的`old`，因此不是普通`score(query,page)`可以表达的可分离排序。下一步工程目标是把四次8B probe蒸馏成slot-page cross-encoder、量化QK page response或只对不确定边界触发的一次短probe。

证据：[全500题条件集合probe与8-shard门控](1b_context_search_research_exploration/evidence/longmemeval_all500_pairwise_set_utility_probe_20260715.json)

### 8.30 时间不是全局recent：版本链需要多值、来源和旧到新因果顺序

完整性probe在multi-session上收益最大，但在knowledge-update上反而使NLL增加0.162。事后机制审计发现：原32-token状态没有包含正确最新值的54题中，完整性门平均增加`+0.256 NLL`；包含正确值的24题则为`-0.049`。典型状态直接把旧值写成事实，例如“最近家庭旅行是Hawaii”或“每两周见一次治疗师”，随后检索和verifier继续围绕旧值形成正反馈。

#### 只把状态写得更长、更谨慎并不能解决

对78个knowledge-update问题重新生成64-token状态，要求列出所有候选值、日期、来源并在不能确定时标记`LATEST UNRESOLVED`。72题可回答；每题仍只在自己的10M shard检索，选择不使用答案：

| 指标 | 32-token compact state | 64-token temporal multivalue |
|---|---:|---:|
| 状态事后包含正确最新值 | 33.33% | 37.50%，8胜5负，`p=.581` |
| 动态Top12任一精确证据 | 97.22% | 97.22% |
| 动态Top12全部证据session | 97.22% | 97.22% |
| 动态reader NLL | 3.7354 | 3.7315，`Delta=-0.0039`，CI跨0 |
| 动态reader F1 | 48.95% | 47.64% |

状态文本改变了，但gold coverage完全不变，reader也没有改善。这里的瓶颈已经不是召回。页面检查显示，静态Top12同时含旧Hawaii记录和明确写着“Family Trip to Paris”的新记录；动态frontier虽然仍命中官方gold block/session，却为了新页面挤掉了更容易读取的Paris邻接页。粗粒度`any gold block/session`无法度量**原子答案是否可见、版本是否齐全、更新关系是否可执行**。

#### 相同页面，只改变时间组织

随后固定每题页面ID和token预算，只改变Qwen3-8B reader看到的页头和顺序。`old-to-new`按session时间升序排列，同一session内保持block顺序：

| 页面组织 | 静态Top12 NLL / F1 / EM | 动态工作集 NLL / F1 / EM |
|---|---:|---:|
| 原BM25相关度顺序，无日期 | 3.413 / 46.16% / 34.72% | 3.731 / 47.64% / 34.72% |
| 只显示日期，不重排 | 3.443 / 49.97% / 38.89% | 3.738 / 52.55% / 41.67% |
| 只按旧到新重排，不显示日期 | **2.939** / 56.18% / 44.44% | 3.535 / 51.01% / 41.67% |
| **显示日期 + 旧到新** | **2.923 / 62.99% / 50.00%** | **3.480 / 59.60% / 45.83%** |
| 显示日期 + 新到旧 | 3.635 / 51.48% / 41.67% | 4.291 / 48.68% / 36.11% |

相对原顺序，静态旧到新无日期已经显著改善`Delta NLL=-0.4746`，CI `[-0.9436,-0.0355]`，F1 `+10.03pp`；加日期后的总改善为`Delta NLL=-0.4900`、F1 `+16.83pp`、EM `+15.28pp`，三者CI或配对检验均支持。固定日期后，旧到新相对原相关度顺序仍改善`-0.5198 NLL`；旧到新相对新到旧改善`-0.7119 NLL`，CI `[-1.4147,-0.1147]`。动态较小工作集也得到F1 `+11.96pp`，但NLL CI跨0，且仍弱于静态12页，说明它丢失的版本/邻接页尚未被控制器补回。

#### 不是所有页面都应按时间强排

为防止把该性质过度泛化，在完全自然的10M XSum新闻记忆上固定已有BM25+E5 Top8和512-token预算，只比较原始memory位置、反向位置和检索分数顺序。100题PPL分别为23.686/23.763/23.698；反向相对原始位置`Delta NLL=+0.0032`，95% CI `[-0.0054,+0.0116]`，检索分数顺序为`+0.0005`，CI也跨0。因此顺序收益不是Transformer对所有升序block ID的偏好，而是**同一实体/关系存在互相覆盖的版本写入时的有向结构**。

更准确的时间属性是：

```text
slot = (owner, entity, relation)
version_chain(slot) = [(value_i, time_i, source_i, confidence_i)]
READ = retrieve relevant writes -> preserve competing values -> order old-to-new
       -> apply latest/before/during/aggregate operator
```

普通RAG常按相关度独立排序页面；这里真正需要的是把版本写入恢复成一个有向小图或事件链，再把几十页以内的局部工作集按因果拓扑交给reader。这既能解释“总记忆10M但当前只需少数写入”，也为直接KV页加载提供稳定顺序和provenance。

证据：[多值状态负结果](1b_context_search_research_exploration/evidence/longmemeval_knowledge_update_temporal_multivalue_state_20260715.json)、[时间reader完整消融](1b_context_search_research_exploration/evidence/longmemeval_knowledge_update_temporal_reader_ablation_20260715.json)、[自然XSum固定页面顺序反例](1b_context_search_research_exploration/evidence/xsum_10m_selected_page_order_ablation_20260715.json)

### 8.31 条件集合效用可以利用公共工作集做KV分叉，但不能压成一句直接比较

8.29节的有效完整性分数需要分别运行`Complete(W+static)`和`Complete(W+dynamic)`，再相减；原实现还保留了两个失败的A/B顺序对照，共约四次8B forward。首先尝试把动作压成一次直接判断：固定语义标签`OLD/NEW`，询问“用NEW替换OLD是否提高完整性”，输出`YES/NO`。同时做隐藏固定8页的消融：

| 一次forward协议 | 平均prompt | 动态收益AUC | 零阈值相对静态NLL | 平均耗时 |
|---|---:|---:|---:|---:|
| 看完整固定工作集，OLD在前 | 1218 tokens | 0.560 | +0.0046，CI跨0 | 0.392s |
| 隐藏固定工作集，只看OLD/NEW | 656 tokens | 0.571 | -0.0146，CI跨0 | **0.215s** |
| 两次独立完整性差分 | 约2×1K tokens | **0.587** | **-0.1211，CI不跨0** | 0.580s |

直接动作prompt对所有候选形成强NO先验：rescue/loss的平均log-odds虽然方向可分，但绝对值都为负；跨shard训练阈值也没有救回NLL。隐藏固定工作集没有显著更差，不是“固定页不重要”，而是这个直接比较协议根本没有稳定执行集合差分。该负结果说明不能仅凭自然语言要求模型“考虑替换损失”，必须保留两个可校准的绝对完整性判断或训练专门的差分模型。

#### 简单batch不等于加速

把两个独立prompt左填充后放进一次batch，质量仍成立：门控NLL为4.2624，相对静态`-0.1207`，CI不跨0；但平均耗时为0.5949s，原顺序两次forward为0.5796s，只有`0.97x`。batch扩大后，3090在约1K-token的8B prefill上没有闲置算力可利用；“模型调用次数从2到1”不能替代真实墙钟测量。

#### 公共前缀KV只算一次

两个完整性prompt不是任意两条独立序列，它们共享问题、检索状态和固定8页，只在边界候选页处分叉。实现时先对两个完整token序列寻找最长公共token前缀，运行一次prefix forward并保存KV；随后复制cache，把静态/动态后缀放入一个branch batch：

```text
P = tokens(question, state, fixed workset)
S_j = tokens(candidate action j, decision suffix)

naive(m actions)  = m * (P + S)
prefix branch     = P + m * S
saving            = (m - 1) * P
```

全500题结果：

| 指标 | 两次完整forward | 公共前缀 + 双分支 |
|---|---:|---:|
| 逻辑prompt tokens | 1937 | 1937 |
| 实际执行tokens | 1937 | **1314，-32.1%** |
| 平均墙钟 | 0.580s | **0.463s，1.25x** |
| 动态收益AUC | 0.587 | **0.593** |
| 门控答案NLL | 4.2620 | **4.2403** |
| 相对静态NLL | -0.1211 | **-0.1428，CI `[-0.2604,-0.0362]`** |

另在partition0的62题中把两种路径放在同一进程和模型内，并逐题交替执行顺序。顺序基线0.5876s，公共前缀0.4342s，墙钟加速`1.35x`；61/62题更快，配对节省时间的95% CI为`[-0.1689,-0.1290]`秒，“共享先跑”和“顺序先跑”分别为1.32x/1.39x。BF16下cache分叉与完整prefill不是bit-exact，utility符号一致率为85.5%；因此不能用原分数直接替换而不复验。全500 reader门控已经重新闭环，证明当前数值扰动没有破坏质量，甚至略改善排序。

这给出一个区别于普通RAG相似度加速的系统属性：**连续推理中的候选动作通常只改变当前工作集的一小部分，反事实工作集之间具有高KV前缀共享率。** 检索器提出少数边界动作后，可以缓存当前`W`的验证KV并做树形分叉，不需要为每个候选重复prefill问题、状态和已固定证据。

证据：[单动作一forward负结果](1b_context_search_research_exploration/evidence/longmemeval_all500_single_action_utility_probe_20260715.json)、[简单batch结果](1b_context_search_research_exploration/evidence/longmemeval_all500_batched_completeness_probe_20260715.json)、[全500公共前缀结果](1b_context_search_research_exploration/evidence/longmemeval_all500_shared_prefix_completeness_probe_20260715.json)、[同进程配对速度](1b_context_search_research_exploration/evidence/longmemeval_partition0_shared_prefix_paired_latency_20260715.json)

### 8.32 从版本链推广到scope内因果拓扑：相关度选页，拓扑决定读取顺序

8.30节只在72个knowledge-update问题上验证旧到新。为判断它是否只是题型特例，在全部500题、相同页面ID和相同512/768-token页面预算上完成完整因子消融：

| 静态Top12页面组织 | NLL | token F1 | EM |
|---|---:|---:|---:|
| BM25相关度顺序，无日期 | 4.3831 | 36.84% | 22.80% |
| 相关度顺序，只显示日期 | 4.3555 | 38.58% | 24.00% |
| **旧到新，不显示日期** | **4.2190** | **40.62%** | **26.20%** |
| **旧到新 + 日期** | **4.1314** | **41.31%** | **26.80%** |
| 新到旧 + 日期 | 4.3062 | 38.49% | 24.40% |

相对原相关度顺序：日期-only的`Delta NLL=-0.0276`且CI跨0；顺序-only为`-0.1641`，CI `[-0.2919,-0.0394]`，F1 `+3.78pp`；日期+旧到新为`-0.2517`，CI `[-0.3948,-0.1111]`，F1 `+4.47pp`。固定日期后，旧到新相对相关度顺序仍为`-0.2241 NLL`；旧到新相对新到旧为`-0.1748`，CI `[-0.3243,-0.0303]`。因此主效应来自有向顺序，日期元数据在正确顺序上再提供`-0.0876 NLL`。

收益也不只存在于knowledge-update。顺序-only在静态页面上对knowledge-update/multi-session分别改善`-0.4286/-0.2725 NLL`，普通assistant/user单会话也为负；temporal-reasoning为`+0.0337`且F1下降，说明“问题名字含temporal”不等于页面形成更新链。用`question + state + page dates`直接学习顺序收益的8-shard外推也失败：静态/动态benefit-sign AUC仅0.429/0.499，ridge连续相关接近0；默认全排序反而强于逐题门控。这里更稳定的决策单元是**memory scope的生成机制**，不是问题措辞。

一个只识别knowledge-update的answer-free router本身可以学到：问题文本AUC 0.859，加入静态页日期分布为0.940，召回82.1%；但它只覆盖版本更新，错过multi-session等一般因果历史，最终收益明显小于对整个对话history scope恢复顺序。因而该router适合选择`latest/before/range`时间算子，不适合决定是否保留对话因果拓扑。

#### 跨域边界

为避免把LongMemEval的对话格式偏好误写成通用规律，固定各域BM25+E5页面并只改变scope内顺序：

| 真实约10M域 | 页面关系 | 旧到新相对检索顺序 | 旧到新相对新到旧 | 结论 |
|---|---|---:|---:|---|
| LongMemEval 500 | 多session事件、更新、偏好 | **-0.1641 NLL** | **-0.1748**，CI不跨0 | 强因果拓扑 |
| 代码30 | 同repo/连续源码页 | -0.0078，CI跨0 | **-0.0169**，CI不跨0 | 反向会破坏局部代码顺序 |
| PG19 past-only 77 | 同书但检索页较稀疏 | -0.0027，CI跨0 | -0.0048，CI跨0 | 未形成稳定连续链 |
| XSum 100 | 独立新闻片段 | 约0 | 反向差+0.0032，CI跨0 | 无需强排 |

所以应把外部记忆表示为带类型的scope：

```text
retrieve pages by relevance
  -> group by scope / relation component
  -> if scope has directed topology:
       topological_order(pages)   # session time, version edge, code/file dependency
     else:
       keep calibrated retrieval order
  -> reader / direct KV load
```

该属性与“global recent”完全不同：检索仍可找到很旧的相关写入，只在已选中的小集合内恢复原有因果偏序。它也解释了为什么总记忆可以很大而工作集很小：模型不需要重读完整历史，只需当前关系分量上的少数节点及其有向边。

证据：[全500时间因子消融](1b_context_search_research_exploration/evidence/longmemeval_all500_temporal_factor_ablation_20260715.json)、[无答案版本router](1b_context_search_research_exploration/evidence/longmemeval_all500_answer_free_version_router_20260715.json)、[无答案顺序utility门](1b_context_search_research_exploration/evidence/longmemeval_all500_answer_free_order_utility_gate_20260715.json)、[PG19顺序反例](1b_context_search_research_exploration/evidence/pg19_past_only_10m_scope_order_ablation_20260715.json)、[代码顺序对照](1b_context_search_research_exploration/evidence/code_10m_scope_order_ablation_20260715.json)、[XSum顺序反例](1b_context_search_research_exploration/evidence/xsum_10m_selected_page_order_ablation_20260715.json)

## 9. 理论模型

对生成时刻 `t`、层 `l`、head `h`、状态 token `u` 与 memory block `b`，定义：

```text
s(t,l,h,u,b) = max_i <q(t,l,h,u), k(l,h,b,i)>
```

其中 `i` 是 block 内 token。进一步分解：

```text
k(l,h,b,i) = m(l,h) + c(l,h,b) + r(l,h,b,i)
```

- `m`是layer/head共享的全局公共方向；
- `c`是block级内容分量；
- `r`是token级残差、句法和局部极值。

于是精确block分数为：

```text
max_i <q,k_i> = <q,m> + <q,c_b> + max_i <q,r_bi>
```

而K-mean近似只保留前两项和平均残差。它要保持Top-K排序，至少需要`<q,c_b>`的block间margin大于不同block的`max_i<q,r_bi>`变化。PG19中raw coherence为0.903但去`m`后仅0.309，且K-mean与token-max Top8几乎不重合，说明raw同向性主要测到了`m`，没有证明上述margin条件。**因此“block内K同向”只能推出可压缩，不能直接推出可寻址。**

进一步分解检索分数：

```text
s = mu(D,l,h,b) + delta(z_t,l,h,u,b) + epsilon
```

- `D`：当前 query-state 分布；
- `mu`：分布条件下的 block/head 公共吸引力；
- `z_t`：当前推理状态；
- `delta`：真正随状态变化的证据信号；
- `epsilon`：token identity、句法、数值误差与索引近似噪声。

8.21节说明`mu`不能只按整个数据分布写成常数；对QK索引，它包含可跨query复现、且强烈依赖profile的block hub prior：

```text
s_p(z_t,b) = h_p(b) + delta_p(z_t,b) + epsilon_p(z_t,b)
p = (layer, head, aggregation, state regime)
```

`h_p(b)`可由与当前query分离的历史query提名频率、分位数或均值估计。文本RRF的cross-fit hub mass接近0，QK head-RRF可达9%-25%，所以去偏应是KV profile条件操作，而不是对所有检索器统一减同一个block popularity。K-centering只处理一部分公共方向，不能替代聚合后hub校准。

对单位归一化的 K，定义 block 支持函数 `g_b(q)=max_i <q,k_i>`。对任意 query 原型 `p`：

```text
|g_b(q) - g_b(p)| <= ||q - p||_2 = sqrt(2 - 2 cos(q,p))
```

因此，若状态指针 Q 能被少量 train-only 原型覆盖，原型附近的 block 支持分数不会任意变化；可以离线为原型建立候选表，在线只对候选做精确 QK。这个界本身较松，最终是否可剪枝还取决于 block 间的 score margin，所以必须用 candidate-recall 曲线验证，不能仅凭 cosine 宣称次线性检索成立。

设无关 block 分数分布为 `F_0`，gold 分数为 `s*`，则它在 `B` 个 blocks 中的期望 rank 近似为：

```text
E[rank(s*)] = 1 + (B - 1) * (1 - F_0(s*))
```

当 `B` 从 39K 增到 3.9M 时，若 tail probability 不随分区或多视角联合约束下降 100 倍，固定 Top-K 必然恶化。因此可扩展系统必须利用至少一种结构降低有效比较域：层级分区、时间/文档局部性、符号/词法路由、prototype postings，或多个近似独立地址的联合证据。

自然生成还需要引入地址成熟度。令 `z_t` 为已生成到时刻 `t` 的状态，目标 block 为 `b*`，则我们关心的不只是瞬时分数，而是：

```text
address_information(t) = I(z_t; b*)
confidence_t(b) = prior(b) + sum_{j <= t, event_j} w_j * evidence_j(b)
```

XSum 中 prefix 增长显著改善 gold rank，说明 `address_information(t)` 通常随相关实体和事件逐步出现而增加；但候选 Jaccard 很低，说明不能简单平均全部历史 Q。更合理的是只累计通过跨时间 recurrence、跨索引 agreement 或 verifier 的事件证据，并在置信度未成熟时保留多个候选分支。

跨域frontier实验把时间局部性分成两个尺度。令`S_t`为Top-D scopes，`F_t`为约512-1,100个粗候选，`W_t`为最终1-3个片段：

```text
S_t = UPDATE_SCOPE(S_(t-1))        only on slow scope/entity events
F_t = RERANK(F_(t-1), z_t)
      union NEW_POSTINGS(S_t, z_t) on fast address events
W_t = UTILITY_SELECT(F_t, z_t, W_(t-1))
```

代码中`S_t`的Jaccard显著高于Top8 blocks，说明上层route可以低频更新；但旧Top512重排仍会损失部分source recall，说明`NEW_POSTINGS`不能删除。若全局/上层刷新概率为`rho_t`，单步期望搜索成本可写成：

```text
E[C_t] = C_rerank(|F_(t-1)|)
         + rho_t * C_new_postings(S_t)
         + gamma_t * C_global_refresh(B)
```

其中`rho_t`由实体、关系、scope和检索分数事件触发，`gamma_t`是更低频保底刷新。现有10M结果证明第一项可覆盖大多数下一轮强候选，但没有证明`gamma_t=0`；永久关闭全局入口会产生4-13个百分点的source损失。

LongMemEval全500题进一步说明，生成状态下的地址过程不应假设为单调rank提升。令每个时刻最终候选为`A_t=TopK(z_t)`，持久工作集为`W_t`，新增页面数为：

```text
I_t = |A_t - W_(t-1)|
W_t = BOUNDED_KEEP(W_(t-1) union A_t)
|W_T| <= K + sum_(t=1..T) I_t
```

若相邻地址高度相关，则`I_t << K`，多次检索不要求工作集线性增长；若少数`I_t`来自不同scope，它们又可能提高复合证据覆盖：

```text
coverage(W_T) = 1[required_scopes subset scopes(W_T)]
```

实测已知owner层级路由每次平均只新增0.92-1.35个页面，五次Top8的唯一工作集约12页，而不是40页。平铺BM25的同预算负对照表明，`union A_t`本身并不创造价值；只有当状态变化改变了上层session/scope路径时，动态frontier才优于在同一路径内继续加深静态Top-K。因此更准确的属性是：**地址创新率低，但跨scope创新的边际完整性价值高；KEEP的作用是把不对称的偶发rescue保留下来，同时避免后续状态的replacement loss。**

8.28节的reader闭环又说明，`BOUNDED_KEEP`不能只按首次出现截断。设当前问题要求的关系槽位为`R_t`，页面`p`覆盖的候选槽位为`r(p)`，在固定容量下追加`p_new`必须挤出`p_old`，则门控目标至少应包含：

```text
gain(p_new, p_old | W_t, R_t)
  = completeness(R_t, W_t + p_new - p_old)
  - completeness(R_t, W_t)
  - interference(p_new | z_t, W_t)
  - displacement_utility(p_old | z_t, W_t)
```

20个完整性rescue对应平均`Delta NLL=-1.044`，8个完整性loss对应`+1.899`，说明正负效用明显不对称。即使新页面来自新session，也不能自动替换静态深排页面；`APPEND/EVICT`应是一对联合动作，而不是先追加、超预算后按年龄机械淘汰。

8.29节把该目标进一步收紧为**非可分离集合效用**。对候选动作集合`A`，定义：

```text
U(A | z_t, W_t, R_t)
  = NLL(y | z_t, W_t) - NLL(y | z_t, W_t union A)
    - lambda_read * tokens(A)

DeltaU(new, old | z_t, W_t, R_t)
  = Complete(W_t + new; R_t) - Complete(W_t + old; R_t)
    - Interference(new, W_t) - Displacement(old, W_t)
```

一般情况下，`U({p1,p2}) != U({p1}) + U({p2})`：两个页面可能分别支持同一槽位而冗余，也可能分别补足两跳链条而互补。因而普通RAG的`score(query,page)`只能做proposal，不能独立决定`APPEND/EVICT`。全500题实验中，可分离E5交互AUC仅0.514，而读取当前工作集后比较两组完整性的AUC为0.587，并首次显著降低平均答案NLL；这给出了“集合条件性”不仅是建模偏好、而是可测属性的证据。

集合条件性并不必然要求把整个工作集重复读取`m`次。令`P_t`为问题、状态和当前固定工作集的公共prefix，`S_j`为候选动作`j`的边界后缀，则反事实验证复杂度可写成：

```text
C_naive(m)  = m * C_prefill(P_t + S_j)
C_branch(m) = C_prefill(P_t) + sum_j C_suffix(S_j | KV(P_t))
```

当生成状态每次只替换1-4页时，`|P_t| >> |S_j|`，公共KV分叉会随候选数放大收益。实测双分支执行tokens减少32.1%、同进程墙钟加速1.35x；这使“非可分离”不再等价于“每个候选都完整重跑”。数值路径在BF16下并非bit-exact，因此分叉verifier仍需按自身分数重新校准，不能直接继承完整prefill阈值。

8.22节进一步把刷新控制拆成两个随机事件：

```text
D_t = 1[full Top8有超过25%不在旧frontier]     # candidate drift
G_t = 1[global refresh对目标utility产生正增益] # useful refresh
```

三域中`P(D_t)=36.7%`，而source覆盖代理下`P(G_t)=7.9%`。target-free候选特征可以跨域预测`D_t`，却只能较弱预测`G_t`。因此刷新决策不应写成`refresh = drift`，而应写成成本敏感控制：

```text
refresh_t = 1[
    P(G_t=1 | state drift, frontier geometry, history) * E[gain_t]
    > C_global - C_frontier
]
```

当前query embedding漂移只是`P(G_t)`的无训练代理；周期保底对应非零`gamma_t`。真正目标必须把`G_t`从source覆盖替换为未来NLL、答案子目标完成度或可执行反馈，并显式允许“刷新会挤入干扰项”的负收益。

past-only结果补充了第二个随时间变化的量：recent变长会降低外部记忆的剩余信息需求。定义：

```text
residual_need(t) = I(y_future; M_external | z_t, W_recent)
expected_read_value(t) ~= g(address_information(t)) * residual_need(t) - read_cost
```

`address_information(t)`通常上升，而`residual_need(t)`通常下降，所以读取价值可以在中间时刻达到峰值，而不是单调增加。事件控制器应在地址置信度刚越过可用阈值且剩余需求仍高时触发；这比“每N tokens检索”或“等地址最稳定再检索”更符合实测。

对生成效用本身，必须显式条件于当前已加载工作集`W_t`，定义未来长度`Delta`上的边际效用：

```text
u_t(b; Delta, W_t) = NLL(y[t:t+Delta] | z_t, W_t)
                   - NLL(y[t:t+Delta] | z_t, W_t union {b})
```

`u_t>0`表示在已有recent和证据之上，加载片段仍有额外改善。past-only结果说明位置最近不保证边际效用最高：若信息已被512-token recent吸收，邻接片段可能冗余；因此效用不是block固有分数，而是对`W_t`的条件集合函数。PG19显示`u_t`与`u_(t+64)`存在中等正相关而非恒定不变，因此工作集控制应利用短期滞后：

```text
KEEP(b) if posterior(u_(t+1)(b) > 0 | observed u_t, state_change) is high
EVICT/REFRESH otherwise
```

8.17节给出一个直接的候选可辨识性约束。若效用估计器只看`f(z_t, W_t)`而不看候选`C`，那么对同一状态下的任意`C1/C2`都有相同预测；但真实情况允许`u_t(C1)>0`且`u_t(C2)<0`。所以候选级utility至少需要一个交互项：

```text
u_hat_t(C | W_t) = f(z_t, W_t, phi(z_t, C), psi(C, W_t))
```

其中`phi`可以是BM25、dense embedding、QK page bound、实体/符号边等**状态-候选可寻址性**，`psi`描述候选相对当前工作集的新颖性、冲突和作用域关系。当前状态自身的不确定性只能作为是否启动检索的prior，不能替代这两个候选条件项。

结合PG19和代码结果，更合理的近似分解是：

```text
E[u_t(C | W_t)] ~= Address(z_t, C)
                   * InformationGain(z_t, C | W_t)
                   - Interference(z_t, C, W_t)
                   - ReadCost(C)
```

`Address`高只说明候选会吸引模型；当候选重复recent、来自错误scope、包含相似符号或产生错误Value方向时，`InformationGain`可接近0而`Interference`很大。代码域中多数候选Delta NLL为负、late-layer token-max QK与utility弱负相关，正是“强地址、负效用”的实例。因此QK适合缩域后的候选响应特征，不是独立的utility verifier。

严格pre-probe实验把这个滞后变成了在线可观测量。设候选在时刻`t0`已经选出，`o[t0:t]`是随后真实发生、现在已经可见的轨迹，定义：

```text
observed_u_past(W -> W') = NLL(o[t0:t] | W) - NLL(o[t0:t] | W')
future_u(W -> W')       = NLL(y[t:t+Delta] | W) - NLL(y[t:t+Delta] | W')
```

若`corr(observed_u_past, future_u)>0`，控制器就能在不读取未来target的情况下，用已经发生的生成计算EXPAND/STOP。实测总体相关0.271支持这一条件，但不同状态和probe长度差异明显，所以它是短时、状态条件的反馈，不是永久block权重。

跨域结果进一步把这个posterior具体化为：

```text
p_keep(b,t) = P(u_(t+Delta)(b) > tau_random |
                scope(b) ~ scope(z_t), retrieval_ranks,
                observed_u_t(b), state_change)
```

其中scope与rank负责无额外前向的先验，`observed_u_t`是只对少量候选执行的短时似然更新。若探测候选数为`m`，最大化带噪声的`observed_u_t`会随`m`增加而放大极值选择偏差；但past-only实验又表明`m`过小可能无法覆盖互补时间尺度。因此`m`应由效用持续性、候选多样性与估计噪声共同决定，而非固定越小或越大越好。

可以把probe预算写成带覆盖与极值惩罚的序贯目标：

```text
m_t* = argmin_m  E[NLL_future(selected_m)]
                 + lambda_compute * m
                 + lambda_extreme * sigma_A * sqrt(2 log m)
                 + lambda_miss * P(useful stratum not covered | m)
```

第三项抑制在大量候选中追逐短A段噪声极值，第四项防止预算太小而没有覆盖local、mid-range、long-range和relation等候选层。bidirectional PG19主要受第三项影响，past-only PG19主要暴露第四项；二者共同解释了为何不存在跨任务固定最优K。

进一步令`scope(b)`表示文档、会话、项目、文件或时间段。PG19中`P(u>tau | same_scope)`显著大于`P(u>tau | other_scope)`，所以有效比较域不是固定全库，而是状态条件的层级域：

```text
B_eff(t) = union of TopD scopes under z_t
retrieve blocks only inside B_eff(t)
```

这给出次线性访问的第二条来源：不仅压缩向量维度，还通过scope posterior减少参与极值竞争的blocks数。scope聚合必须校正文档大小，否则长文档会仅因包含更多blocks而获得更高累计分数。

LongMemEval进一步说明，scope不是一个平坦标签，而是带环境元数据的有向层级：

```text
tenant / owner / project
  -> document / session / repository
    -> turn / file / symbol / event
      -> block / token / KV page
```

设`a_s`是scope `s`的上层地址表示。层级路由有用不仅要求同scope富集，还要求`a_s`对下层证据近似充分：

```text
I(b*; a_s | z_t) sufficiently high
and
candidate_cost(children(s)) << candidate_cost(all blocks)
```

把长session或整个owner均值成一个向量会降低第一项，导致粗路由虽便宜却提前丢失事实。更合理的上层地址是多key集合、稀疏posting、事实槽位摘要或子节点score的稳健聚合，而不是容器全文的单均值。

对于动态事实，令槽位`r=(scope, entity, relation)`，该槽位存在按时间排序的写入链：

```text
E_r = {(t_i, value_i, provenance_i)}
value(r, t) = latest valid write in E_r before t
```

查询应先按`z_t`检索相关槽位或写入，再在槽位内执行`latest / before / during / aggregate`算子。对需要更新解析的槽位，reader输入不是相关度无序集合，而是保留`value/time/provenance`并按旧到新拓扑排列：

```text
READ(r) = TOPOLOGICAL_OLD_TO_NEW(retrieve_writes(r))
ANSWER  = TEMPORAL_OPERATOR(READ(r), query_time)
```

旧到新使模型先建立旧状态、再观察覆盖它的新写入；新到旧则更容易在后文重新激活旧值。LongMemEval固定相同页面的消融支持这一点，而自然XSum固定页面顺序的反例没有显著差异，说明这是**版本链条件属性**，不是全局位置或block ID偏好。全局recent相当于先按时间剪掉语义域，只有当目标槽位本身最近时才有效；这解释了knowledge-update证据中位新近排名23.5、global latest8召回很低的现象。

全500与跨域结果把版本链推广成scope条件的有向图。令`G_s=(V_s,E_s,type_s)`表示会话、事件流、文件、调用链或普通文章scope，则最终读取顺序应为：

```text
pages_s = relevance_retrieve(z_t, V_s)
ordered_s = topological_sort(pages_s, E_s)  if type_s is directed
            calibrated_rank(pages_s)         otherwise
```

LongMemEval对话history的`E_s`由session时间和版本覆盖边给出；连续代码页至少具有文件位置边；XSum独立新闻片段没有共同有向分量，PG19稀疏同书页也未形成稳定链。因而排序策略应由scope结构决定，而不是看到时间词就逐题切换。它将“相关性”与“可执行上下文组织”分开：前者决定节点集合，后者恢复节点间关系。

拒答还要求把目标从“至少命中一个相关block”改写为需求覆盖。令`R(q,z_t)`为当前子问题所需槽位集合，则：

```text
answerable(q, W_t) = all required slots in R(q,z_t) are supported by W_t
                     and no unresolved conflict remains
```

hard negative可以与问题和某个槽位高度相关，却缺少另一个必要槽位。因此高相似度、甚至命中近邻会话，都不能替代coverage、provenance和冲突检查。

100M结果进一步把在线复杂度具体化。若scope倒排只访问与query匹配的scope postings，再在Top-D scopes内访问block postings，则：

```text
online_cost(t) ~= postings_scope(z_t)
                 + sum_{s in TopD scopes} postings_block(z_t, s)
                 + reader_cost(W_t)
```

它不再与全库block数线性等价；实测Top3 scopes只让0.544%的100M blocks进入第二级，获得6.74倍CPU查询加速。但质量不能只写成scope coverage。更合适的读取目标是：

```text
expected_value(W_t) ~= P(useful scope retained)
                      * E[useful block utility | routed scopes]
                      - interference_cost(W_t)
```

层级Top3的any-hit略低但NLL显著优于全局BM25，直接证明第二项和干扰代价不能被召回率替代。Top-D因此既是计算预算，也是工作集纯度控制变量。

8.13节说明即使`P(useful scope retained | score geometry)`可校准，也不能直接把它当作D的停止规则。更完整的scope深度目标应为：

```text
D_t* = argmax_D  E[utility_t(TopK blocks from TopD scopes) | z_t, W_t]
                  - lambda_search * candidate_blocks(D)
                  - lambda_interference * competing_mass(D)
```

其中`competing_mass(D)`衡量新增scope在最终Top-K极值竞争中引入的无关分数质量。实测D16/D32提高scope coverage却显著恶化NLL，证明它不能被常数项忽略。score margin和熵适合构造`P(scope retained)`，但STOP还必须使用候选纯度、跨视角一致性、短reader probe或历史效用持续来估计后两项。

边际utility实验进一步说明二元概率仍不充分。令`Delta u_D`为从D扩到下一层的reader收益，则：

```text
E[Delta u_D | x] = P(Delta u_D > 0 | x) * E[Delta u_D | Delta u_D > 0, x]
                    - P(Delta u_D <= 0 | x) * E[-Delta u_D | Delta u_D <= 0, x]

EXPAND iff E[Delta u_D | x] > lambda_search * Delta C_D
                                + lambda_interference * Delta M_D
```

当前特征能把第一项的二元AUC做到0.760，却不能预测条件幅度，因此固定概率阈值没有形成Pareto优势。后续模型必须使用cost-sensitive或连续utility目标，并通过短期已观测surprisal、hidden/Value响应或小批量counterfactual probe补足幅度信息。

距离分解说明`B_eff`还不能只是一组scope。更合适的是保留不同结构来源的分层并集：

```text
C_t = C_local(t) union C_scope(t) union C_relation(t) union C_content(t)
```

`C_local`覆盖位置/时间邻域，`C_scope`覆盖同文档、同项目或同session的远程片段，`C_relation`沿实体、引用、调用图和事件边跳转，`C_content`才是BM25/E5/QK等内容检索。各分支应有独立最小配额，再通过效用门控竞争最终工作集；否则强局部先验会淹没真正的长程依赖，或大scope会淹没局部精确证据。

理想检索集合不是对全部方向求平均，而是：

```text
R_t = TopK_b Aggregate_{(u,h) in Router(z_t)} s(t,l,h,u,b)
```

关键研究问题变成：

1. 如何无标签地得到小的 `Router(z_t)`；
2. 如何聚合少数支持方向而不被大量无关方向稀释；
3. 如何把 `TopK_b` 从线性扫描变成次线性索引；
4. 如何判断应该保留、替换还是扩展当前工作集。
5. 如何让无关 block 的联合通过率随视角数近似乘法下降，同时保留只被少数专家 head 发现的证据。
6. 如何在线估计地址成熟度，在“过早收缩”和“重复检索浪费”之间选择触发时机。
7. 如何用生成中已经观测到的loss、置信度和状态变化估计未来`u_t`，而不离线遍历全部候选。
8. 如何学习长度无偏、允许多scope分支的层级router，并在scope切换时快速失效旧工作集。
9. 如何校准跨域KEEP阈值，并控制候选试读数，避免瞬时效用极值造成winner's curse。

## 10. 属性导出的系统草图

```text
生成状态 z_t
  -> 访问模式路由：LOCAL / RETRIEVE / SCAN
  -> scope路由：document / session / repo / file / time range；margin/熵估计地址不确定性
  -> 环境硬scope：tenant / owner / active project；未知时保留多个软scope分支
  -> 记录级路由：session / turn / event / fact-slot多粒度key，禁止大容器单均值替代全部子地址
  -> RETRIEVE 时提取指针/角色 token（4-8 个 Q 方向）
  -> 少数 retrieval heads（而非全部 heads）
  -> 词法/符号/时间/文档/prototype 多级倒排
  -> 候选合并 + 跨视角软共识 + 分布鲁棒去 hub
  -> 按layer/head/aggregation减QK hub prior；文本通道不套用未经验证的同一惩罚
  -> 文本/scope主召回；仅对几百个候选计算同状态E5、QK page bound、Value响应与workset新颖性
  -> 估计 Address × InformationGain - Interference；高attention但低增益候选不进入工作集
  -> 版本链组织：保留entity/relation写入的value/time/provenance，按旧到新排列
  -> scope拓扑恢复：对话/事件/连续代码按有向边组织；独立文章不强制位置排序
  -> 时间算子：在版本链内执行latest / before / range / aggregate，而非全局recent
  -> 条件集合比较：Complete(W+new) 对 Complete(W+old)，联合批准APPEND/EVICT
  -> 缓存问题、状态和固定工作集KV；只对新旧边界页做counterfactual分叉
  -> 只对低置信边界候选做分批短试读或回溯重放；用已经发生的轨迹估计边际效用，收益不足则STOP
  -> 完整性检查：所需槽位是否齐全、是否只有旧值/近邻实体/hard negative；不足则继续检索或拒答
  -> 从高置信锚点扩展并加载1-3个局部片段
  -> active working set 保持 0.5K-4K tokens
  -> 双时钟更新：scope低频刷新，block/frontier按状态事件重排；保留低频全局refresh
```

### 10.1 不能缺少的组件

- **Recent/local 区：** 最近生成与刚读取证据保持 dense attention；
- **访问模式控制器：** 局部续写不应承担远程索引开销，全局聚合也不能伪装成 Top-K 点查；
- **External retrieval 区：** 完整历史保存在 CPU/SSD，不永久丢弃；
- **多索引：** 至少按 layer/head 与状态角色拆分；
- **尺度隔离：** 先将 3.9M blocks 路由到几百至几千个 coarse candidates，再从中选最终 blocks；
- **层级scope路由：** 先按文档/会话/项目/文件/时间段缩域，并显式消除scope长度偏置；
- **环境namespace与多粒度key：** 已知tenant/owner/project作为硬过滤，未知scope才做软路由；session、turn、event和fact key同时保留，避免大容器聚合丢失细事实；
- **槽位条件版本链：** 时间规则只在已检索到的`scope/entity/relation`写入链内应用；保留多值、时间和来源，按旧到新交给reader，global recent仅作为独立小配额；
- **Scope拓扑层：** relevance负责选页，session时间、版本覆盖、文件位置和依赖边负责reader顺序；普通无边scope保持校准后的检索顺序；
- **事件控制器：** 联合估计地址置信度与recent之后的剩余信息需求，在二者乘积较高时读取；状态指针或效用变化时刷新；
- **两级时间缓存：** 缓存慢变Top-D scopes及其pages，同时维护0.5K-1.1K粗frontier；常态只重排frontier，实体/关系/scope事件注入新postings，保底周期允许全局刷新；
- **刷新控制器：** 分开估计candidate drift与useful refresh；当前可部署基线使用query-state漂移触发加低频周期保底，不能用候选Jaccard或source标签在线作弊；
- **Profile条件hub校准：** QK按layer/head/聚合方式维护离线hub prior或分位数；公共吸引力从动态state delta中分离，文本BM25/E5仅在独立证据支持时去偏；
- **候选条件是硬约束：** 当前hidden/uncertainty只控制是否检索；候选排序必须至少读取文本摘要、scope关系、QK bound或其他候选sidecar，不能从当前状态凭空预测未见候选效用；
- **效用反馈：** 用已发生token的surprisal与状态变化决定KEEP/EVICT，而不是每一步重做全量检索；
- **自适应候选试读：** 对scope富集后的候选分批计算短counterfactual loss；高置信早停，低持续性或低多样性时扩展到8/16/32；
- **地址与效用分离：** BM25/E5/QK负责可寻址性，novelty/conflict/Value与短reader估计信息增益和干扰；QK高分本身不能充当验证；
- **完整性与拒答门：** 验证当前工作集是否覆盖全部必要关系、时间点和并行条件，不能把相关hard negative当成可回答证据；
- **条件集合选择器：** 比较`Complete(W+new)`与`Complete(W+old)`，显式计入被挤出页效用、干扰和读取成本；不允许用独立页面相关分直接批准淘汰；
- **反事实KV分叉：** 固定工作集prefix只prefill一次，多个APPEND/EVICT动作共享其KV，仅计算差异后缀；batch大小和数值阈值需在目标GPU上重新校准；
- **工作集策略：** READ、KEEP、EVICT、STOP 是独立动作。
- **片段化加载：** 检索排名决定锚点，原始位置邻域决定一次I/O和KV加载的连续范围。

### 10.2 复杂度目标

ParisKV、Quest 等方法证明 compact metadata 线性扫可以支持百万 tokens，但对 1B 仍然不够。我们的目标应是：

```text
每次事件刷新成本 = O(route_1 + D_1 * route_2 + C)
普通生成步成本   = O(U * H * |F_t|) + reader(W_t)
期望索引成本     = C_frontier + rho_t * C_hier_refresh + gamma_t * C_global
```

- `U`：4-8 个状态指针方向；
- `H`：少数 retrieval heads；
- `B`：约 3.9M blocks；
- `C`：几百到几千个粗候选，而不是全部 blocks。
- `F_t`：跨步缓存的0.5K-1.1K候选frontier；只有事件或保底refresh才访问全局/上层postings。
- `D_1`：第一层scope分支数，必须随地址成熟度和score geometry变化；1B后验表明固定book Top8不够稳定。
- `C_hier_refresh`：多级scope缩域后的新postings访问，不能退化为把Top-D books内所有blocks平铺扫描。

100M CPU原型已实际达到候选比较域缩小59.6-190.4倍和2.57-6.74倍在线加速；差距来自scope postings、候选稀疏矩阵运算和Python开销尚未完全随候选数同比缩放。1B系统仍需把两级索引放到CPU/SSD分层，并避免每次构造全长score数组。

## 11. 与普通 RAG 的边界

| 维度 | 普通 RAG | 本项目目标 |
|---|---|---|
| query | 一段文本的全局 embedding | 当前生成状态的 token x head 多向量 |
| 更新频率 | 通常问题级或少数轮 | 随推理状态和事件动态更新 |
| memory | 原文 chunks | 原文 + 模型原生 K/V + 压缩索引 |
| 排名 | 单一语义相似度 | head-specific QK、角色指针、Value/验证 |
| reader组织 | 常直接按retrieval score拼接 | 先选节点，再按scope版本/因果/依赖边恢复拓扑 |
| 候选验证 | 每组context通常重新prefill | 固定工作集KV共享，候选动作只分叉差异后缀 |
| 读取代价 | 检索后重新 prefill 文本 | 可直接加载已计算 KV，避免重复 prefill |
| 目标 | 找主题相关文档 | 找当前一步对生成有因果作用的少量状态 |

边界并不是“完全不用 BM25/embedding”。字符串与 RAG 可以作为低成本 seed 或 fallback；创新点必须来自模型原生、轨迹条件、多向量、KV 复用与次线性访问。

全500顺序消融还说明，普通RAG即使召回同一批正确页面，也可能因按score排列而破坏history内部的因果边。这里新增的不是另一种相似度，而是`retrieve nodes -> restore edges -> read`；公共前缀实验又证明条件集合验证可以直接复用当前工作集KV。两者都位于retrieval proposal之后，是当前与“换embedding、调Top-K”最清楚的系统边界。

LongMemEval共享10M的标准E5对照使这个边界更具体：全局block E5的Top8精确命中58.3%，略高于BM25的54.2%，但配对不显著；已知owner后的E5/BM25 session8精确命中同为81.2%。这说明普通RAG已经是很强的主召回器，本项目不能靠替换embedding宣称贡献。另一方面，把session压成单E5向量后全局层级召回显著下降，说明可研究的问题是**如何保留层级、槽位、生成状态和KV utility，而不是是否使用RAG**。

LongMemEval论文也已系统验证round级value、事实key扩展、时间query扩展与extract-before-read；这些都应作为必须比较的强RAG baseline。当前E5-base-v2只是一项轻量对照，尚未覆盖Stella/GTE、turn/round索引、fact expansion和端到端reader，不能据此声称超过最强RAG。

候选条件实验给出更精确的边界。在严格PG19上，模型原生QK的utility符号AUC为0.827，同状态E5为0.813，二者差值CI跨0；在真实代码上，QK/Value加入`retriever rank + repo scope`后，逐query所选候选的平均未来收益反而从0.1216变为0.1160，配对CI跨0。**因此当前证据支持“模型原生地址可以补充RAG”，不支持“QK检索普遍优于RAG”。** 真正可能形成区别的是：RAG缩域后不重新prefill全部文本，而是用当前token/head状态查询预计算KV sidecar，并在连续生成中用真实utility反馈维护工作集。

轨迹与hub实验又增加了一个系统边界：文本RRF的候选在跨query上近似分散，而QK具有强profile条件hub且跨步Top-K漂移更快。普通RAG可以低频重算query embedding并复用文档索引；KV通道必须额外处理head/profile去偏、状态事件和高频候选响应。它的潜在收益不是“更稳定”，而是当文本query尚未显式表达当前内部变量时，能够提供模型状态原生地址，并直接复用已计算KV。若这些条件增益和prefill节省不能覆盖更高的在线更新成本，就不应启用KV通道。

还必须避免把现代RAG已有能力误写成新贡献：FLARE已经根据即将生成的句子持续检索，Self-RAG已经按需检索并自我反思，HORMA已经组织并导航层级Agent记忆，R3AG已经把retrieval relevance与generation utility分开学习。因此，**“动态检索”“层级路由”“生成效用精排”单独都不是本项目的新颖性。** 可成立的边界应是它们在1B模型原生外部KV上的联合：token/head级状态地址、已计算KV直接加载、次线性scope/posting访问，以及连续生成中的KEEP/EVICT而非反复文本prefill。

当前配对结果进一步表明，BM25/E5不应只作为“失败时兜底”，而应作为主检索通道；KV通道的近期目标是补充它们的失败集合，并降低反复文本prefill成本。只有在更大规模或高频动态生成中证明“直接KV复用 + 次线性状态检索 + utility门控”具有端到端优势后，二者的系统边界才真正成立。

最新集合probe把系统边界从“换一种retriever”推进到“proposal之后如何维护有限工作集”。BM25/E5从10M提出候选，条件完整性模型读取`question + current workset + candidate action`后再批准保留或替换；把两个候选的公共工作集前缀只prefill一次后，完整性判断从同进程顺序执行的0.588s降到0.434s（1.35x），全500题门控NLL为4.2403，对应静态RAG的4.3831。它仍是额外Qwen3-8B文本probe，不是已完成的直接KV页系统。因而当前研究贡献候选是**工作集条件效用、公共KV分叉与scope内因果拓扑**，不是声称检索质量已经全面超过RAG。

## 12. 当前判断

### 已验证

- 动态 Q 中存在显著高于随机的下一步证据事件；
- 证据地址集中在少数 token-head 方向；
- 当前实体/变量式 state pointer 在外部 MuSiQue 上能用约一半至三分之一 Q 保持有限工作集召回；
- train/dev 学到的 2048 个原型可覆盖 98.6% 的 test pointer Q（cosine≥0.8）；
- head 轴、K residual 与真实文档局部结构具有可压缩性。
- 4-head posting 共识可将 gold 富集 28.0x/77.0x，证明多视角联合有强选择性，但必须软聚合。
- 真实新闻 PPL 证明几百 token 工作集可能优于 full context，而不只是更快。
- 真实 prototype postings 已将 10M 访问压到 1%，并在 1024-token 工作集上得到 32.0%/31.2% 的单步证据召回。
- BM25 Top3 与 KV Top4 的互补组合在 1792 tokens 内达到 75.4%/68.8%，两步同时命中 52.0%。
- 无问题模板的 10M XSum 证明地址随 prefix 成熟，Hybrid Top8 any-hit 从 26% 增至 72%。
- 真实新闻实际扩展证明 Top512 仍有 97%-98% any-hit，而 Top8 只有 65%-72%，支持分层粗到细。
- 10M XSum 的 E5-512 PPL 23.61 显著优于 Query-only 26.83，Oracle512 为 20.39；随机读取恶化到 28.87。
- 10M XSum 的两个4-block连续窗口在同预算下把Hybrid PPL探索性降到23.14，并显著提高source完整召回，但PPL配对CI仍跨0。
- 9.9M PG19复现地址成熟：Hybrid Top8 any-hit随prefix 8→64从16.7%增至50.0%，10胜0负。
- 9.9M PG19的Hybrid-512将PPL从Query-only 44.20降至39.17；两个连续片段探索性降至38.81，方向复现但相对Hybrid的配对CI仍跨0。
- 9.9M PG19中Top512 any-hit为96.7%，Top8只有50.0%，再次支持“粗候选保活、局部片段精排”。
- PG19相邻两个64-token生成段的候选效用Spearman为0.347；前段最优窗口73.3%在后段仍为正效用，支持命中后的短时复用。
- 前段效用选择把未来段PPL从44.16降至40.90，并显著优于静态E5 Top1的42.52；排除预定义source仍为42.31。
- PG19同书高效用事件率47.3%，异书10.6%；长度归一化Hybrid文档路由Top3召回93.3%，平均缩域30.2倍。
- 10M代码repo复现地址成熟：Hybrid Top8 any-hit为46.7%→73.3%；Hybrid-512将PPL从8.56降至7.07，随机读取恶化到9.65。
- 代码repo复现效用持续：A/B效用Spearman 0.325，A最优窗口76.7%在B仍有正效用；同repo/异repo高效用事件率为46.6%/11.8%。
- 代码repo Top3 scope召回96.7%，但只缩域3.9倍，证明scope层级有效且粒度必须继续细化到file/module/symbol。
- PG19与代码的跨域效用事件预测中，静态rank-only AUC仅0.553-0.586，scope-only为0.666-0.712，scope加短A段实测效用达到0.712-0.738。
- 在bidirectional PG19/代码中，same-scope优先后只试读2-4个候选可把未来PPL从43.89/7.94降至41.76或40.64/7.26或7.24，证明部分任务可收缩为小预算在线序贯选择。
- PG19试读候选从4增至8/16时未来PPL反而回升，证明瞬时效用存在极值选择偏差，候选预算兼具计算限制和统计正则化作用。
- 排除heldout source后same-scope效用仍跨域成立；代码效用随距离明显衰减，而PG19相隔>16K的same-book候选仍有正平均效用，支持“局部核 + 长程身份”双路径。
- Qwen3-0.6B在单张3090上并行试读4个384-token候选序列中位耗时45.7ms，仅比单候选33.0ms多12.8ms；小候选效用探测可批处理。
- 严格past-only PG19 9.9M已通过0 future-block违规检查；无预定义source时，Hybrid同书Top8随状态64→512从73.3%升至100%，8胜0负，`p=0.0078`。
- past-only中随机512显著恶化PPL至22.88，Hybrid/E5局部片段探索性改善至21.96/21.86但CI跨0；scope命中不是充分效用条件。
- past-only无gold效用A/B Spearman为0.243，A选择可把B段PPL从21.23显著降至20.33；同书/异书效用事件率27.3%/6.2%，odds ratio 5.64。
- past-only效用随历史距离衰减；最近块因与recent冗余并不最优，结构化8-16 probe才稳定胜静态Top1，否定了固定2-4 probe的普适性。
- past-only中地址准确率随状态64→512持续提高，但读取边际NLL增益在128 tokens达到0.0667、到512降为0.0085；128显著高于512，证明存在事件触发的中间价值窗口。
- past-only真实K中raw block方向一致性为0.903，但去除layer/head全局公共方向后仅0.309；高coherence主要是公共偏置，不是block语义地址。
- SVD32在56个PG19 K profile中平均保留87.0%能量，64-token K-mean可把token轴压缩64倍；但聚合Top8同书any-hit至多10%，token-max也未挽救，说明低秩和K-mean是存储性质而非充分检索性质。
- 在状态128/512上，文本Hybrid PPL为23.74/21.96，而K-mean56 max为26.16/22.90、token-max8为25.73/23.04；自然全局QK读取会显著恶化生成质量。
- 真实100M PG19由原9.9M因果记忆加983本disjoint train书籍构成，无合成和重复文本；Top3/Top8 scope只扫描0.544%/1.68% blocks，Top8 any-hit保留全局BM25的92%/96%，CPU查询加速6.74/3.25倍。
- 100M层级Top3/Top8虽比全局BM25少量降低any-hit，却在状态128和512都显著改善配对NLL；候选纯度和干扰成本比二元召回指标更接近真实生成目标。
- 480个状态的query-grouped 5折外推中，scope score margin/熵/mass把100M路由命中宏AUC从结构基线0.564提高到0.768，证明地址不确定性可由无target信号预测。
- 扩scope存在显著干扰拐点：D16在状态128、D32在状态512均显著差于D8；scope coverage增加不保证最终Top8纯度或reader效用增加。
- 100M的180次scope扩展中仅22.8%改善未来NLL；geometry+Top8 churn对扩展符号的query-grouped AUC为0.760，状态512达0.820，证明边际utility事件具有部分可预测结构。
- Oracle utility STOP以平均D5.4/4.6显著优于固定D3/D8，证明自适应小工作集存在真实上限；当前瓶颈是代理信号而非候选不可用。
- 严格pre-probe协议中，检索只使用probe之前的64/448 tokens；随后已观察64-token反事实收益与未来128-token收益Spearman为0.271（p=2.36e-4），状态512的符号AUC达0.740，证明生成轨迹可提供无gold在线utility反馈。
- 状态128的probe16 STOP平均选择D4.1、PPL 24.677，显著优于固定D3的24.846；状态512未超过最强固定D3，但显著避免D16/D32过扩展，说明成熟状态应更偏向STOP。
- LongMemEval全500题已划分为8个互不重叠的独立10M shards；470个可回答问题平均只有1.89个证据sessions、9.17个精确64-token blocks，而每个shard有156,250 blocks，支持“总记忆巨大但单步证据基数稀疏”。
- 首个64题shard中，已知owner后BM25 session3平均只比较131个blocks、最终读取512 tokens，Top8精确命中83.3%；相对156,250-block全局比较域缩小约1,192倍。
- 首个64题shard的层级深度存在干扰拐点：owner-session D1→D3的全部证据覆盖20胜0负（`p=1.91e-6`），D3→D8仅4胜1负且出现精确block损失，D8以后最终Top8不再变化。
- 首个64题shard中，标准E5全局block Top8精确命中58.3%，BM25为54.2%，但配对8胜6负（`p=0.791`）；已知owner的session8二者均为81.2%，说明retriever家族不是当前决定性变量。
- 首个64题shard中，全局E5 session8聚合的全部证据覆盖比BM25低20.8个百分点，配对2胜12负（`p=0.0129`），证明层级上层表示必须保留细粒度事实，不能任意单向量平均。
- LongMemEval自然日级未来会话/证据违规均为0；分钟级同日噪声影响41/470条可回答问题，严格在线因果分析已单独过滤，官方all-history口径与因果口径已分开。
- knowledge-update与multi-session证据的中位owner内新近排名为23.5/24，latest8覆盖仅25.0%/12.5%，支持“时间局部性必须在实体/关系槽位内使用”。
- abstention hard negative在已知owner的BM25 session3/E5 session8中命中81.2%/100%，证明拒答瓶颈是证据完整性与冲突判断，不是简单降低相关召回。
- LongMemEval全500题中，问题从25%增长到100%时，全局BM25任一证据命中由5.3%升至55.3%，已知owner session3由20.0%升至85.3%；地址成熟跨对话域成立。
- 完整问题后继续追加生成计划会使单状态检索平均变差，但已知owner层级Top8每次只新增0.92-1.35个64-token页面；五状态唯一工作集约763-769 tokens、页面复用约70%，支持`KEEP + sparse APPEND`而非Top-K替换。
- 在同数量页面下，owner Hybrid动态并集相对静态结果把任一证据/全部证据覆盖提高3.6/4.9个百分点；固定12页时仍提高2.8/3.4个百分点。平铺BM25同预算无增益，证明价值来自跨session路由创新而非普通query expansion。
- 但最强静态owner-session3的12页绝对分数仍为88.5%/76.4%，其动态12页89.4%/77.9%的差异不显著；当前支持状态化层级frontier属性，不支持“已超过最强RAG”。
- 证据条件闭环中，8B只读初始Top8后生成32-token事实/缺口状态；动态工作集平均672.5 tokens，相对静态768 tokens把全部证据session覆盖由76.60%提高到79.15%，`p=.0357`，session recall提高2.06个百分点且CI不跨0。
- 同一个Qwen3-8B reader上，动态相对静态的参考答案NLL为`+0.0203`，95% CI `[-0.129,+0.166]`；token F1方向提高1.72个百分点但CI跨0，说明检索完整性改善尚未转化为总体显著生成收益。
- 20个动态完整性rescue样本平均`Delta NLL=-1.044`、F1提高17.7个百分点；8个loss样本为`+1.899`、F1降低18.1个百分点。动态页面本身有价值，但固定容量下的displacement loss更大，完整性与淘汰门是当前直接瓶颈。
- 8-shard留一外推的无答案门控中，轨迹、结构和状态文本三类特征的动态收益AUC仅0.517-0.527，选择后NLL均不优于全静态；页面变化和缺口措辞不足以判断候选内容效用。
- 加入新页/被挤出页的词法支持、冗余和E5候选交互后，动态收益AUC仍仅0.512-0.514；最佳E5门相对全静态`Delta NLL=-0.0015`且CI跨0，证明普通bi-encoder相关性仍不能估计条件集合效用。
- Qwen3-8B独立判断`Complete(W+static)`与`Complete(W+dynamic)`后，动态收益AUC提高到0.587；零阈值门平均读取730 tokens，把答案NLL由4.3831降至4.2620，95% CI `[-0.2357,-0.0184]`，对rescue/loss选择动态70.0%/12.5%。
- 条件完整性门的收益集中在少量大救回：234个非tie中126胜108负，符号检验和Wilcoxon均不显著，5%截尾均值只剩`-0.0284 NLL`；它证明可学习属性存在，但不是稳定逐题最优控制器。
- knowledge-update的64-token多值状态没有改变97.22%的检索覆盖，也没有改善reader，说明状态改写不是主瓶颈；固定相同Top12页面按旧到新排列使F1由46.16%升至56.18%，加日期后升至62.99%、NLL降0.4900。
- 固定日期后，旧到新相对新到旧改善0.7119 NLL且CI不跨0；同一顺序消融在自然10M XSum固定Top8上只有0.0032 NLL且CI跨0，确认该规律属于实体/关系版本链而非全局顺序偏好。
- LongMemEval全500题固定相同静态Top12后，日期-only相对baseline的NLL仅改善0.0276且CI跨0，顺序-only旧到新改善0.1641并且CI不跨0，日期+旧到新改善0.2517；固定日期时旧到新相对新到旧改善0.1748，证明主要增益来自恢复因果历史，日期是辅助地址。
- 代码10M固定候选页的旧到新相对反向读取改善0.0169 NLL且CI不跨0；PG19稀疏页面和XSum独立新闻片段均不显著，支持“只在有向scope内恢复拓扑”，而不是对所有检索结果做全局位置排序。
- 把条件集合效用压成一句`OLD vs NEW`单forward比较只有AUC 0.571，不能改善NLL；把两次完整性prompt简单padding成batch也只有0.974x。性能来自共享公共前缀KV，而不是只减少API调用次数。
- 公共前缀KV分叉把逻辑执行tokens减少32.1%；全500题旧计时0.580s降到0.463s，同进程62题由0.588s降到0.434s（1.35x，61/62题更快），并保持有效门控：NLL 4.3831降到4.2403、F1提高2.42个百分点。
- 只用问题与页面日期可以无答案识别版本查询（AUC 0.940），但问题/状态/日期对“本题是否受益于旧到新排序”的外推失败；正确路由单位更像`history/code-chain/independent-corpus`这类scope类型，而不是问题词面。
- 严格PG19零额外reader实验表明，当前attention/uncertainty/hidden无法可靠预测未见scope的边际utility；候选内容缺失是可辨识性限制，不是继续堆状态特征就能解决。
- 在同一PG19候选上，候选条件E5和模型原生QK把utility符号AUC从geometry的0.737提高到0.813/0.827；QK与E5差值CI跨0，说明动态QK地址成立但尚未优于RAG。
- 真实10M代码的1,760个候选中只有32.9%对未来64 tokens有正效用；按query聚类和FDR后无单个QK/Value特征显著，证明addressability与utility必须分开。
- 代码域OOF模型原生选块平均未来Delta NLL为0.1160，相对`rank+repo scope`的0.1216无增益；已观察A段probe达到0.1812但相对baseline的配对CI仍跨0，跨域utility门控尚未解决。
- 三个真实约10M域的RRF轨迹中，旧Top512覆盖新Top8的62.5%-85.4%，有界历史frontier提高到73.3%-89.2%；粗候选具有部分时间局部性，可以跨步缓存。
- 直接只重排旧Top512仍在XSum/PG19部分轨迹显著损失4-13.3个百分点source any@8，证明增量frontier必须允许事件注入和低频全局刷新。
- 代码Top3 scope的相邻Jaccard为0.623-0.753，逐query显著高于Top8 block的0.263-0.439；旧Top8 scopes覆盖新Top8 blocks达93.3%-97.5%，支持“慢scope、快block”双时钟。
- XSum两折cross-fit中，文本RRF的Top1%稳定hub mass仅0.284%/0.018%，QK head-RRF为14.9%-25.5%/3.9%-9.1%；稳定hub是KV profile属性而非通用语料属性。
- 三个真实约10M域、480次状态转移中，target-free frontier特征可按query预测Top8候选失效（AUC 0.834），并跨域保持AUC 0.797；说明增量刷新存在可观测的在线状态信号。
- 真正只靠全局刷新才能救回source的事件仅7.9%；无训练query E5漂移以20%刷新率达到48.33% source any@8，相对从不刷新提高2.29个百分点并把索引访问缩减4.92倍。该倍数不含reader，不能写成端到端加速。
- PG19嵌套尺度后验在50M→100M上把Top8 scope recall预测为72.2%、实际72.5%；外推1B后Top8/Top128预计仅55.8%/76.5%，说明上层router本身也受极值竞争，必须继续分层。
- 1B后验中达到80% recall的book深度随prefix64/128/256/512从Top1024/512/128/64收缩；地址成熟不仅提高召回，也能直接降低路由宽度。
- 77本不同真实PG19书籍组成的严格9.9M主实验包含154,688个64-token blocks；每本只产生一个query，另有70本真实干扰书，future-block违规为0，不含合成文本或复制填充。
- 在同一10M三级索引上，状态64→512使真实book平均排名由3.52改善到1.45，Top8同书命中由81.8%升到97.4%，局部4K命中由48.1%升到79.2%；候选域保持约2,020、最终读取保持512 tokens，复现“地址成熟不要求工作集增长”。
- 77本确认实验中，`book8 -> segment8`把block精排域从154,688缩到约499，再加载Top8共512 tokens；跨128/256/512状态相对query-only、全局BM25和E5的配对ΔNLL为-0.0557/-0.0304/-0.0277，CI均不跨0。
- `book8 -> segment8`相对BM25+E5 Hybrid的配对ΔNLL为-0.00865，95% CI `[-0.0188,+0.0014]`；当前只能说方向更好，不能宣称已经超过强RAG。30-query中`segment32`最优也未复现，层级深度必须用独立query校准。
- 77本中地址成熟时score entropy仍由0.959升到0.976、Top8 mass由0.146降到0.118；router/frontier trajectory策略相对静态策略分别恶化+0.00624/+0.00929 NLL，证明地址正确性、score集中度、候选变化和reader utility是四个不同对象。
- 只重放已经观察过的64 tokens时，回放效用与未来效用的事件内Spearman为0.169；先验加回放相对状态先验仅改善-0.00093 NLL且CI跨0，而七action顺序回放需296ms。该信号只能用于少量边界动作确认，不能作为全候选扫描器。

### 尚未解决

- 1024-token 工作集的两步同时命中只有 9.8%，复合成功率仍不足；
- 2048 原型的 1% candidate route 仍只覆盖第一跳 35.6%、第二跳 41.0%；
- 10M 上固定 posting 深度有效，不代表增加 100 倍干扰项后仍有效；
- 乐观后验预测中，1B 全局 Top4 只剩 8.1%/4.5%，分层索引尚未真正通过 1B 实测；
- 当前只证明近似倒排访问，不是带动态更新、CPU/SSD 分层和连续生成的完整 1B 系统；
- KV 通道在 10M 上的质量和速度均未超过 BM25；
- 在自然新闻上，raw SVD32 QK 的 PPL 26.95，直接与文本通道融合也没有净增益，QA 中的 KV 互补性尚不能泛化；
- past-only PG19再次否定未门控的全局QK：K-mean与token-max均接近随机且读取有害；尚未完成全部query heads、post-RoPE位置校准、候选内QK bound和专门retrieval-head训练。
- past-only PG19已取消预定义source并扩大到77本独立书籍，但仍是单一书籍域；尚未在新闻流、对话、日志和代码因果历史中复现同一层级与效用控制协议；
- 有界效用探针在bidirectional任务平均2-5个有效；77本past-only的64-token回放只有弱预测信号，七action顺序回放成本过高且没有显著超过状态先验，仍缺少零额外前向、跨域稳定的在线效用代理；
- BM25/E5/Hybrid rank与未来生成效用的相关性仅0.06-0.15，scope内部精排仍是主要质量瓶颈；
- 书籍域已在10M做到book/segment/block三级，对话session层级已在500题验证；代码file/symbol、对话turn/fact和日志service/time层级仍未验证；
- LongBench代码context混合源码、README和文档，尚未在纯代码definition/call-chain与可执行pass-rate上验证；
- 0.6B 生成器本身会造成严重状态错误；
- prior 与 query-state 分布漂移尚无鲁棒方案。
- 当前100M实验使用真实book边界，router已由状态文本在线预测Top-D books而不读取query book id；但对话、日志与混合代码中的scope边界本身仍需自动发现。
- bidirectional与past-only PG19的长程距离曲线不同，说明infill外部知识库和严格因果历史不能共享未经校准的scope/distance prior。
- 100M book router仍随尺度退化：状态512的Top8 scope recall从10M的100%降到100M的83.3%；自然容器能缩域，但一层book路由尚未达到1B稳定性。
- 当前1B数字是由真实10M-100M嵌套PG19校准的Beta-Binomial外推，假设新增books可交换且scope大小有界；它不是实际1B索引、CPU/SSD I/O或端到端吞吐结果。
- 10M三级检索已在77本独立PG19书籍、三个reader状态和标准BM25/E5/Hybrid RAG下完成配对确认；样本量仍不足以训练复杂控制器，中间segment预算和状态动作仍需跨新闻、代码、对话复现。
- 77本query-grouped外推中，router/frontier trajectory均显著弱于静态策略；当前缺少能从无target轨迹变化稳定预测reader utility的控制器。
- recall校准的adaptive C70/C80尚未在候选成本与PPL上严格支配固定D3/D8；score geometry能预测路由错误，但仍缺少无target的边际生成效用和干扰预测器。
- 当前静态廉价特征对Delta NLL连续幅度的分组外推失败；77本回放probe把事件内Spearman提高到0.169，但收缩策略相对状态先验的改善CI跨0，且尚未跨域复现。
- PG19上的候选条件QK符号AUC没有跨域复现在真实代码选块收益上；目前没有训练后可跨书籍/代码直接共用的零额外forward utility模型。
- 模型原生candidate sidecar仍较大：PG19七层724 blocks为1.328GB，代码三层5,539 blocks为4.356GB；1B需要低秩/量化page summary、分层按需加载和严格端到端I/O基准。
- 已有不读取target和当前全库排序的target-free refresh gate，但精确source事件跨域AUC最高仅0.615，且学习门没有稳定超过简单query漂移；尚未用未来NLL/答案效用训练，也未接入连续生成系统。
- QK hub prior只在XSum上做了cross-fit，且简单频率惩罚最多带来2个百分点、CI含0；需要按head/profile做train-only whitening并在新闻、书籍、代码和QA外推。
- LongMemEval全500题多shard复验已完成，但语料包含模拟memory sessions，生成状态是额外8B search plan而非自然回答轨迹；仍需真实用户日志、邮件/工单或开源对话历史。
- 当前已知owner实验对应真实系统可用的tenant/user/project namespace；全500题中自动owner Top8在完整问题上只有66.6%，追加长计划后降至54.7%，未知scope仍是层级系统的主要漏召回点。
- 当前E5-base-v2不是最强RAG，尚未比较Stella/GTE、turn/round粒度、fact/keyphrase expansion和官方end-to-end reader；不能宣称当前方法已经超过RAG。
- LongMemEval已完成10M检索后的Qwen3-8B reader NLL、短答案F1/EM和拒答评估，但尚未运行官方LLM judge，也没有把命中页面作为已计算KV直接注入；当前只能称为文本页面reader闭环。
- LongMemEval条件完整性probe已经通过公共前缀KV分叉把同进程延迟从0.588s降到0.434s，但它仍是Qwen3-8B文本verifier，不是直接加载预计算页面KV；只验证了双分支，BF16效用符号一致率约85%-91%，多候选树、跨模型校准和更小probe仍未完成。
- 全500题已经证明对话因果顺序的平均价值，并在连续代码上得到较弱复现；但当前仍依赖数据已有session日期、源位置或局部代码位置，没有自动发现`entity/relation/version/dependency`边，也没有完成scope类型路由与通用拓扑排序。

因此，当前 idea 中的**小工作集、地址随生成成熟、层级缩域、稀疏跨scope创新、条件集合效用、公共KV分叉和scope内因果拓扑已得到真实10M支持**。最重要的新结论不是“动态query总能提高召回”，而是：粗RAG负责提出少量候选；控制器必须相对当前`W`比较新旧页面集合；会话、版本与代码链还必须在reader前恢复局部因果拓扑。共享KV证明非可分离集合效用不必为每个候选重跑全部前缀，但当前仍使用额外8B文本probe，收益也偏重尾；直接KV页加载、廉价控制器、自动scope图和跨域闭环尚未完成，所以不能宣称已有端到端系统超过最强RAG。研究范围只新增真实10M实验，不运行真实1B；1B仅用于解释为什么固定全局Top-K会因极值竞争失效，以及系统为何必须层级化和次线性化。

## 13. 通用化验证与下一步

官方 MuSiQue test 500 的外部复验已经完成，状态指针、多向量瞬时证据和 query 原型覆盖均成立。XSum与PG19跨域确认地址成熟、粗候选生存和512-token工作集效用；LongMemEval全500题确认状态化session frontier、工作集条件效用和对话因果顺序，并完成共享前缀8B reader闭环；连续代码给出较弱但同方向的拓扑证据。原生QK单独检索自然文本仍弱于E5/Hybrid，动态reader也未总体胜过最强静态RAG，不能把两跳QA寻址器或query expansion直接当成通用方案。

下一阶段还需加入：

1. **当前真实规模主实验：** 按当前研究范围只新增真实10M实验，不运行真实1B；真实10M书籍三级检索、77本reader与RAG对照，以及对话全500题的8个独立10M shards和reader闭环均已完成。下一步在同一10M上加入直接KV页加载与连续生成控制；既有100M结果和1B router外推只用于解释尺度边界；
2. **互补主实验：** 在 BM25/E5 失败集合上测 KV 的条件增益，并训练无 gold 的 agreement/confidence gate；
3. **新闻/书籍：** 10M XSum、9.9M PG19受控continuation和past-only无source书籍实验已完成；下一步扩大past-only query数，并在原生新闻流中复现因果scope/距离/效用曲线；
4. **代码仓库：** 10M mixed-repo continuation已完成；下一步解析file/module/symbol层级，把指针改为定义、调用点、类型和错误位置，测definition/call-chain recall与生成pass rate；
5. **全局任务反例：** 摘要、统计、全库一致性检查可能需要覆盖式读取，应测试 READ/SCAN/STOP 控制器而非假设每步总是 1-3 blocks；
6. **规模与系统：** 当前只在10M做真实系统实验；在同一10M索引上加入CPU内存预算、SSD page模拟、异步预取和连续生成流水线，报告索引更新时间、page访问、缓存命中、P50/P95与吞吐，不用构造真实1B；
7. **模型尺度：** 比较 0.6B 与 8B 的同状态 Q 流形、指针路由和 verifier，区分寻址能力与 reader 能力。
8. **自适应候选预算：** score geometry预测scope错误已通过分组外推，但recall阈值策略未胜固定D；下一步把纯度下降、跨视角agreement和候选竞争质量加入STOP模型，不再默认低置信就扩到D32。
9. **在线效用估计：** 当前状态自身的零额外信号已证实不可辨识未见候选；下一步固定粗候选，比较同状态E5、量化QK page bound、候选相对workset novelty/conflict与8/16/32-token probe，训练`Address × Gain - Interference`而非raw similarity。
10. **层级scope：** book/repo和对话session级富集已复现；下一步在代码file/symbol、对话turn/fact和日志service/time range上验证自动scope posterior，并报告Top-D召回、长度偏置和实际缩域倍数。
11. **KV边界实验：** 不再继续无门控全局K-mean；按layer/head/aggregation从train queries估计hub prior，比较whitening、频率IDF、候选内QK page bound、post-RoPE相对位置校准和pointer-event gate，报告相对纯文本候选的条件增益。
12. **对话多粒度主实验：** LongMemEval 500条的session层级BM25状态frontier和Qwen3-8B reader已完成；下一步增加round/turn/fact多key、Stella/GTE、官方fact expansion与LLM judge，并按相同512/768/1024-token预算报告检索、直接KV复用和最终答案质量。
13. **类型化scope因果图：** 相同页面的日期/顺序全500消融和代码方向复现已经完成；下一步自动抽取`owner/entity/relation/version/dependency/time operator`，把会话、版本、事件、代码调用链表示成有向局部图，相关度负责选节点、拓扑负责排列，并对独立文章显式关闭该算子。
14. **分叉完整性与淘汰控制器：** 双分支公共前缀KV已经取得1.35x同进程加速；下一步测`m=2/4/8`候选树、直接预计算页面KV注入和更短verifier，联合估计slot gain、displacement、interference与read cost，输出`APPEND / KEEP / EVICT / ANSWER / ABSTAIN`，同时报告BF16校准、截尾NLL、逐题胜率、hard-negative误答率和P50/P95延迟。
15. **跨域校准门：** 以query为独立单位，在书籍、代码、新闻、对话分别训练/外推utility gate；必须报告跨域leave-one-domain-out、模型尺度迁移、paired Delta NLL和选择后工作集PPL，禁止用候选行数伪造样本量。
16. **增量refresh gate：** 三域target-free gate与固定周期比较已完成；下一步把标签从source coverage改为未来NLL/答案子目标增益，扩大稀有正事件，并在10M CPU/SSD模拟层级索引中比较query漂移、agreement gate、固定周期和每步全局搜的page I/O、cache命中与端到端吞吐。

## 14. 相关工作定位

- [Quest](https://arxiv.org/abs/2406.10774)：query-aware page bound，证明 token 重要性依赖当前 Q。
- [RetrievalAttention](https://arxiv.org/abs/2409.10516)：对 KV 建 ANNS，并指出 Q/K 分布不匹配使普通 ANN 失效。
- [MagicPIG](https://arxiv.org/abs/2410.16179)：指出 attention 不总是 Top-K 稀疏，使用 LSH sampling 估计 attention。
- [SnapKV](https://arxiv.org/abs/2404.14469)：用prompt末端observation window预测生成期head级关注位置，是“短观察窗口具有时间持续性”的直接先例。
- [H2O](https://arxiv.org/abs/2306.14048)：把KV保留写成动态次模优化，并结合recent与历史heavy hitters，支持边际效用与预算共同建模。
- [DuoAttention](https://arxiv.org/abs/2410.10819)：区分 retrieval heads 与 streaming heads，支持 head 功能异质性。
- [ParisKV](https://arxiv.org/abs/2602.07721)：百万 token 的 GPU-native 粗到细 KV retrieval，并显式处理 decoding drift。
- [Self-Indexing KVCache](https://arxiv.org/abs/2603.14224)：把1-bit压缩Key同时作为稀疏attention索引，证明压缩sidecar与寻址结构可以统一；它仍以attention近似为目标，不直接解决候选生成效用和跨步KEEP/EVICT。
- [FIER](https://arxiv.org/abs/2508.08256)：用1-bit keys做细粒度token级KV检索，说明page局部性并不总能覆盖稀疏关键token。
- [Recursive Language Models](https://arxiv.org/abs/2512.24601)：把长 prompt 当成外部环境，通过程序化分解与递归调用读取局部片段。
- [FLARE](https://arxiv.org/abs/2305.06983)：根据下一句预测和低置信token触发迭代检索，证明“随生成更新query”已是RAG已有方向。
- [Self-RAG](https://arxiv.org/abs/2310.11511)：通过reflection tokens学习按需检索、生成与批判，覆盖访问模式控制的一部分。
- [HORMA](https://arxiv.org/abs/2606.11680)：把Agent经验组织成文件系统式层级并训练导航器选择最小充分上下文，直接覆盖层级memory routing。
- [R3AG](https://arxiv.org/abs/2604.22849)：显式区分retrieval quality与generation utility，说明“相关性不等于生成效用”也已有RAG研究。
- [BRIGHT](https://arxiv.org/abs/2407.12883)：显示推理密集检索中现有retriever表现很弱，支持“主题相似不足以解决状态条件证据检索”的问题设定。
- [LongMemEval](https://arxiv.org/abs/2410.10813)：系统比较session/round value、事实key扩展、时间query扩展与extract-before-read；为对话长期记忆的RAG强基线和粒度/时间/完整性属性提供直接依据。
- [RICO](https://arxiv.org/abs/2506.12149)：用模型梯度和无监督question perplexity优化候选混合，直接支持“检索目标应接近模型效用而非外部相似度”，但计算路径与本文的在线KV sidecar不同。
- [SALS](https://arxiv.org/abs/2510.24273)：在RoPE-free低秩空间做候选选择、只重建少量KV，支持低秩适合作为候选内压缩与筛选，而非全局语义充分性的定位。
- [Every Token Counts / HSA-UltraLong](https://arxiv.org/abs/2511.23319)：把超长上下文需要的核心性质概括为稀疏、随机访问和长度外推，并报告16M训练模型；它支持问题设定，但仍需真实非合成效用和1B系统验证。
- [Random-Access Reading](https://arxiv.org/abs/2405.13216)：训练Transformer跳读长文档，说明目标导向任务不必顺序消费全部tokens；本项目进一步关心生成状态变化后如何刷新地址和控制已加载工作集。
- [SPIN](https://arxiv.org/abs/2604.26837)：把动态稀疏attention与CPU/GPU分层KV、page抽象和局部缓存协同设计，说明算法稀疏只有与I/O和工作集管理结合才会形成端到端收益。
- [H2MT](https://arxiv.org/abs/2605.24930)：离线构造语义层级并在线粗到细路由，说明层级缩域已有强先例；本项目不能只以“层级检索”作为新颖性。

这些工作已证明“百万token稀疏KV访问”和“动态层级RAG”都可以工程化。稀疏、随机访问和长度泛化也已有明确表述。本项目若要形成新的研究贡献，需要进一步解决：**1B block级次线性KV索引、推理轨迹条件的token/head状态寻址、已计算KV的直接复用、极小active working set，以及基于工作集条件真实未来效用的在线REFRESH/KEEP/EVICT。**
