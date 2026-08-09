# Qwen3 Top-2% Attention Token Mechanism

这个项目研究三个问题：

1. 为什么每层、每个 attention head 只保留历史 token 中 attention score 最高的 2%，PPL 会达到低点，甚至优于 full attention？
2. 这 2% 到底是哪些 token，它们更像 sink、recent、标点/换行、语义实体，还是远程证据？
3. 在严格相同的 2% 历史-token 预算下，只保留 `sink + recent` 能否达到与 oracle Top-2% 相同的效果？

用户原问题中的 `sin` 在本项目中按常见的 `sink`（attention sink）解释。

## 最重要的实验口径

- Top-2% 的预算只计算历史 token：`B_t = ceil(0.02 * history_len)`。
- 当前 query token 默认额外保留，不占历史-token 预算。
- `sink_recent_sN` 在同一个 `B_t` 内保留前 `min(N, B_t)` 个 sink token，其余预算全部给最近历史 token；它不是额外增加 sink 预算。
- `top_attention` 是使用完整 QK score 得到的 oracle selector，用于解释机制，不是可部署的 KV-cache 方法。
- “效果一样”不靠肉眼看两条 PPL，而按逐 token 配对 NLL 的 block-bootstrap 95% CI 判断。默认等价界限为 `±0.01 nat/token`；相对 PPL 相差不超过 1% 作为描述性辅助条件。

## 已有证据与新项目边界

仓库前置实验已经观察到：

- ratio 曲线覆盖约 `0.1%, 0.5%, 1%, 2%, 4%, 8%, 16%, 32%, 100%`；
- 在一组 4k 文本诊断中，Top-2% 选择事件约 56.7% 来自 recent 区域，但约 41.4% 来自远程中部；
- token 0 是强 attention sink，但 sink bucket 的总事件质量并不占主导；
- 不同 layer/head 的位置偏好差异明显。

因此，本项目不重复堆积相同图，而新增四类证据：

- attention 集中度、有效支持集大小和 2% cutoff gap；
- Top-2% token 的位置角色、词法类型、选择富集度和 attention mass；
- 等预算 sink/recent 分配 sweep 对 oracle Top-2% 的位置召回、mass 召回和分布 cosine；
- full、Top-ratio 曲线和所有 control 的逐 token 配对 NLL。

详细方案见 [docs/research_plan.md](docs/research_plan.md)，预注册参数见 [configs/main_experiment.json](configs/main_experiment.json)。

首轮 War and Peace 4k→512 远程结果见 [docs/war4k_results_20260714.md](docs/war4k_results_20260714.md)。该条件下 4% PPL 最低，最佳等预算 sink+recent 仍显著差于 oracle Top-2%。

Top-2% 跨 head、跨层和跨 query 的 union 结果见 [docs/top2_union_results_20260714.md](docs/top2_union_results_20260714.md)。同层 16 heads 的单步 union 平均为历史的 12.60%，整个模型单步 union 为 55.56%，整个 512-query 区间的 model temporal union 为 100%。

## 运行

Linux/GPU 服务器完整实验：

```bash
bash ymluo/projects/qwen3_top2_token_mechanism/scripts/run_server.sh
```

常用覆盖参数：

```bash
MODEL=/home/fdong/hrj/prove/Qwen3-0.6B \
TEXT=/path/to/eval.txt \
PREFILL_TOKENS=4096 \
EVAL_TOKENS=512 \
CUDA_VISIBLE_DEVICES=0 \
bash ymluo/projects/qwen3_top2_token_mechanism/scripts/run_server.sh
```

Windows 本地 smoke：

```powershell
powershell -ExecutionPolicy Bypass -File `
  ymluo\projects\qwen3_top2_token_mechanism\scripts\run_smoke.ps1
```

只运行核心 evaluator：

```powershell
python ymluo\projects\qwen3_top2_token_mechanism\src\run_selector_ppl.py `
  --model_name_or_path ymluo\models\Qwen3-0.6B `
  --text_path external\needle-in-a-haystack\needlehaystack\PaulGrahamEssays\worked.txt `
  --output_dir ymluo\projects\qwen3_top2_token_mechanism\outputs\manual_run `
  --prefill_tokens 4096 `
  --eval_tokens 512 `
  --ratio_grid 0.001,0.005,0.01,0.02,0.04,0.08,0.16,0.32,1.0 `
  --target_ratio 0.02
```

对已有 run 重新汇总，无需再次加载模型：

```powershell
python ymluo\projects\qwen3_top2_token_mechanism\src\analyze_diagnostics.py `
  --run_dir ymluo\projects\qwen3_top2_token_mechanism\outputs\RUN_NAME

python ymluo\projects\qwen3_top2_token_mechanism\src\compare_selectors.py `
  --run_dir ymluo\projects\qwen3_top2_token_mechanism\outputs\RUN_NAME `
  --make_plot
```

## 核心输出

- `ppl_by_selector.csv`：full、Top-ratio 曲线和等预算 controls 的 PPL。
- `token_nll_by_selector.csv`：逐 token、逐 selector NLL，用于配对统计。
- `top2_token_events.csv`：每个 token 位置的 Top-2% 选择次数、mass 和 sink/recent/remote 事件数。
- `top2_concentration_by_layer_head.csv`：entropy、effective support、Top-2% mass、cutoff gap。
- `sink_recent_overlap_by_layer_head.csv`：每个 layer/head 和 sink allocation 的位置召回、mass 召回、分布 cosine。
- `top2_union_by_layer_query.csv`：每个 query、每层跨 heads 的 Top-2% 位置 union。
- `top2_union_by_model_query.csv`：每个 query 跨整个模型全部 attention heads 的位置 union。
- `top2_union_summary.csv`：layer/model union 的均值、分位数、预算倍数和重复率。
- `top2_temporal_union.csv`：跨整个 eval 区间曾被选中过的历史位置 union。
- `analysis/paired_selector_comparison.csv`：对 full 和 Top-2% 的配对 NLL、95% CI 与等价判断。
- `analysis/token_category_summary.csv`：按 eligible exposure 校正后的词法类别富集度。
- `analysis/position_role_summary.csv`：Top-2% 选择事件在 sink/recent/remote 三类中的占比。

## 必须先通过的 sanity checks

1. `top_attention@100%` 与 full attention 的 `delta_loss` 应接近 0。
2. `sink_recent_s0` 与 `recent` 应逐 token 相同。
3. 非 drop-ablation selector 的实际历史-token keep ratio 应与名义 ratio 一致。
4. 所有 selector 必须评分完全相同的 token index。
5. 若 2% 低点只在单一文本、单一 chunk size 或单次运行出现，不解释为普遍机制。

## 测试

```powershell
python -m pytest ymluo\projects\qwen3_top2_token_mechanism\tests -q
```
