# LongBench v2 外部留出验证（2026-07-14）

## 目标

在不使用 LongBench v2 标签训练、调参或修改路由规则的前提下，直接验证 v466 operator-contract 方法的外部分布泛化能力。该实验不是继续在 LongBench v1 上挑选参数，而是回答三个更关键的问题：

1. v466 在全新任务分布上能否保持 Full KV 的准确率；
2. 极低 KV 比例和端到端加速是否仍然成立；
3. v466 的 direct operator 是否会破坏 LongBench v2 的四选一输出协议。

## 数据与官方协议

- 数据：`zai-org/LongBench-v2/data.json`，共 503 条样本；
- 领域：Single-Document QA、Multi-Document QA、Long In-context Learning、Code Repository Understanding、Long-dialogue History Understanding、Long Structured Data Understanding；
- 难度：192 条 easy，311 条 hard；
- 长度：180 条 short，215 条 medium，108 条 long；
- Prompt：使用官方 0-shot prompt；
- 生成长度：`max_new_tokens=128`；
- 评分：仅接受官方格式 `The correct answer is (A-D)` 或不带括号的对应格式；
- 汇报：Overall、Easy、Hard、Short、Medium、Long，并补充领域分组、KV ratio、online speed 和 total speed。

当前服务器是 24GB RTX 3090。正式实验首先测试 32K 上下文截断；如果在无并发占用时仍然 OOM，再统一降到 24K，并在论文中明确标记为 `LongBench v2 (24K cap)`，不能称为完整 128K 官方设置。

## 三组对照

| 组别 | 方法 | 目的 |
|---|---|---|
| Full | `full_kv` | 同模型、同 prompt、同样本和同截断长度的质量与速度基线 |
| v466 | `riskkv_operator_contract_v466_retrieve896_code256_20260713.json` | 原样迁移当前最佳 practical 方法，不针对 v2 调参 |
| v466 direct-off | `riskkv_operator_contract_v466_direct_off_20260714.json` | 保留相同路由与稀疏 KV 动作，只关闭 aggregate/structured direct answer |

direct-off 不是新方法，而是机制消融。如果 v466 明显差于 direct-off，说明当前 direct operator 与四选一协议不兼容；如果两者接近，则主要差距来自路由或 KV 选择本身。

## 实现

- Runner：`src/run_controlled_public_kv_benchmark_v1.py`
- 启动：`scripts/run_longbench_v2_operator_eval_20260714.sh`
- 汇总：`scripts/summarize_longbench_v2_operator_eval_20260714.py`
- 自动等待与汇总：`scripts/watch_longbench_v2_operator_eval_20260714.sh`

v2 loader、prompt 和 scorer 是独立分支，不改变已有 LongBench v1 与 RULER 数据路径。小样本 smoke 按领域轮询抽样，避免前三条恰好都来自同一领域。

## 当前状态

- [x] 官方 503 条数据下载和字段校验；
- [x] 官方 prompt、答案抽取与分组字段接入；
- [x] v466 direct-off 消融配置；
- [x] 本地语法、配置继承和 scorer 单元检查；
- [x] 无并发条件下的 32K 三方 smoke；
- [ ] 503 条 Full、v466、direct-off 正式运行（已在 GPU 5/6/7 后台启动）；
- [x] 自动汇总 watcher 已启动，等待三组 `task_results.csv` 后自动执行。

第一次 32K smoke 启动后，另一条 7 卡任务同时在每张候选卡占用约 656MB，三条分支均在 SDPA 预填充末段 OOM。该结果不能判断 32K 本身不可运行。重试已启用 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，并要求启动前显存占用低于 200MB。

无并发重试成功，32K 上下文可以在单张 24GB RTX 3090 上运行。3 条 smoke 仅用于执行正确性检查，不用于选方法：

| Method | Score | KV ratio | Valid format | Online speed | Total speed | Route error |
|---|---:|---:|---:|---:|---:|---:|
| Full | 33.33% | 100.00% | 100.00% | 1.00x | 1.00x | 0 |
| v466 | 0.00% | 2.19% | 0.00% | 3.22x | 1.32x | 0 |
| v466 direct-off | 0.00% | 2.19% | 0.00% | 3.18x | 1.34x | 0 |

这 3 条样本分别走 1 条 code 和 2 条 retrieve，因此 v466 与 direct-off 行为一致。Full 也只答对 1 条，不能从该 smoke 估计准确率；但 v466 的官方答案格式率为 0 是正式实验需要重点检查的失败信号。

在不运行模型的全量路由预检查中，503 条的动作分布为：retrieve 450、code 44、aggregate 8、structured 1。direct operator 最多影响 9 条，正式结果主要反映 sparse retrieval 和 code action 的外部分布泛化。

正式任务于 2026-07-14 02:42（Asia/Shanghai）启动：

- Full：`outputs/20260714_longbench_v2_full_m503_c32000`
- v466：`outputs/20260714_longbench_v2_v466_m503_c32000`
- v466 direct-off：`outputs/20260714_longbench_v2_v466_direct_off_m503_c32000`
- 自动汇总：`outputs/20260714_longbench_v2_comparison_m503_c32000`
- 日志：`outputs/logs/20260714_longbench_v2_*.log`

## 结果表（待自动填充）

| Method | Overall | Easy | Hard | Short | Medium | Long | KV ratio | Online speed | Total speed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 100% | 1.00x | 1.00x |
| v466 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 |
| v466 direct-off | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 |

在正式结果完成前，不使用该实验宣称 v466 已经泛化到 LongBench v2，也不根据 smoke 的 3 条样本修改路由。
