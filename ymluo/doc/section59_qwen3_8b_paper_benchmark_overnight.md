# Section 59: Qwen3-8B 论文 benchmark 长跑实验

## 目标

本节启动 Qwen3-8B 上的长跑 benchmark，用经典长上下文论文常用的 benchmark 数据检查 summary memory / raw retrieval / router policy 的效果。

重点不再是 toy smoke，而是使用本地已有的正式 benchmark 数据：

- LongBench
- RULER

## 新增脚本

主 runner：

```bash
ymluo/projects/learned_hierarchical_summary_memory/src/run_qwen8b_paper_benchmarks.py
```

overnight 启动脚本：

```bash
ymluo/projects/learned_hierarchical_summary_memory/src/run_qwen8b_paper_benchmarks_overnight.sh
```

## 模型

使用：

```bash
Qwen/Qwen3-8B
```

服务器上原本没有本地 Qwen3-8B 目录，因此脚本会通过 Hugging Face 自动下载到 cache。实际日志显示已经成功 fetch 5 个文件并加载 5 个 checkpoint shards。

## Benchmark 数据

LongBench 数据来自：

```bash
/home/fdong/ymluo/external/KVCache-Factory/data/LongBench
```

使用子任务：

```text
hotpotqa
2wikimqa
musique
passage_retrieval_en
passage_count
qasper
gov_report
multi_news
```

RULER 数据来自：

```bash
/home/fdong/ymluo/external/KVCache-Factory/data/RULER
```

使用子任务：

```text
niah_single_1
niah_single_2
niah_multikey_1
niah_multiquery
niah_multivalue
vt
cwe
fwe
```

RULER context length：

```text
4096
8192
16384
```

每个任务取前 5 个样例。总计约 160 个 case，每个 case 跑 9 种 memory 方法。

## 对比方法

```text
full_raw
summary10
summary100
summary1000
static_hier
retrieval_raw_k1
retrieval_raw_k2
router
router_conservative
```

其中：

- `summary10/100/1000`：对上下文按 block 做 extractive summary。
- `static_hier`：远处 block 用短 summary，近处 block 用长 summary。
- `retrieval_raw_k1/k2`：summary memory + query-aware raw block retrieval。
- `router`：加载前面蒸馏的小 router。
- `router_conservative`：router 输出后加保守升级规则，降低 summary10 过度压缩风险。

## 输出目录

主输出：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen3_8b_paper_benchmarks_overnight_20260704
```

日志：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/logs/qwen3_8b_paper_benchmarks_overnight_20260704.log
```

运行中会持续写：

```bash
trials.partial.csv
```

完成后会写：

```bash
trials.csv
summary.csv
summary.json
```

## 当前启动状态

进程：

```bash
PID=676417
```

启动后已经确认：

- Qwen3-8B 成功下载/加载；
- GPU0 显存占用约 19.5GB；
- 已开始 generation；
- `trials.partial.csv` 已经开始写入。

查看进度：

```bash
ps -p 676417 -o pid,etime,cmd
tail -80 /home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/logs/qwen3_8b_paper_benchmarks_overnight_20260704.log
wc -l /home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen3_8b_paper_benchmarks_overnight_20260704/trials.partial.csv
nvidia-smi
```

## 注意

当前 runner 是 prompt-level compression benchmark，不是 CUDA kernel 级 KV cache patch。它可以评估：

- task accuracy / ROUGE；
- prompt token ratio；
- generation wall time；
- router 是否能按任务切换策略。

但它还不能代表最终 CUDA kernel 级实现的真实性能上限。这个实验主要用于 paper benchmark 质量趋势和策略选择验证。
