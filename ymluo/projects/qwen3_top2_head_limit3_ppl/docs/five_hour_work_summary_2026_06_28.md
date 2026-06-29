# 2026-06-28 五小时工作总结

## 目标

本轮工作的目标是：围绕长上下文 KV cache / attention decode 优化，寻找一个有机会投稿 ICML 的方法方向。要求不是单纯调参，而是同时满足：

- 有足够创新性，不能只是 SparQ、Quest、DuoAttention、RazorAttention 的重复实现；
- decode 速度要比 full-attention baseline 快；
- PPL 不能明显变差，最好能和 baseline 持平或更好；
- 文档和结论使用中文记录，便于后续继续推进。

## 相关工作边界判断

首先重新审视了已有方法，明确了哪些方向不适合作为主创新点：

- `qabs8cand3attn` 与 SparQ Attention 的 query-aware selective KV fetching 太接近，不能作为 ICML 主创新；
- head-level retrieval / streaming 与 RazorAttention、DuoAttention 太接近，也不适合作为唯一主线；
- recent-only 或固定窗口方法速度可能快，但 80k 长上下文下 PPL 很容易崩；
- 仅做 post-RoPE attention warmup 排序不够稳定，20k 和 80k 的 layer remote-mass 排名会迁移。

因此，本轮把主线从 `qabs8cand3attn` 转向：

```text
CLB-LF: Calibrated Layer Budgeting with Landmark Fallback
```

核心思想是：用少量校准 token 判断每层远程上下文依赖强度；高依赖层保留 full attention；低敏感层不直接 recent-only，而是使用 `recent window + landmark stride` 保留低频全局信息。

## 代码实现

主要修改文件：

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/src/evaluate_qwen3_top2_head_limit3_ppl.py
```

新增/完善的能力包括：

- 新增 `--full_head_map_path`、`--full_layer_map_path`、`--layer_budget_map_path`；
- 新增 full-layer budget 相关模式：
  - `fulll{n}recent{w}attn`
  - `fulll{n}landmarkr{recent}s{stride}attn`
  - `fullladapt{low}to{high}m{margin}recent{window}attn`
  - `layerbudgetattn`
- 新增 landmark fallback 路径，让非 full 层保留：
  - recent window；
  - stride landmark tokens；
  - self token；
- 新增通用 JSON layer budget map，支持每层指定：
  - `full`
  - `recent`
  - `landmark`
- 为 CSV 输出补充：
  - `full_head_map_path`
  - `full_layer_map_path`
  - `layer_budget_map_path`
  - adaptive budget 统计字段；
- 给 `full_layer_indices()` 增加 `(map_path, requested)` 缓存，减少 decode 时重复构造 set 的 Python 开销。

新增工具：

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/src/select_safe_layer_budget.py
```

用途：读取 28 个单层压缩消融结果，自动筛选安全压缩层，并生成专用 `full_layer_map_path`，用于继续测试 `fulll{28-k}landmarkr4096s64attn`。

新增远端续跑脚本：

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/scripts/run_clblf_safe_layer_search_server.sh
```

用途：服务器恢复后自动汇总单层消融、生成候选 map、并行验证更强 CLB-LF 组合。

新增同步脚本：

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/scripts/sync_clblf_to_server.ps1
```

用途：服务器恢复后，一键同步本地新增代码、文档和日志到服务器。

新增投稿候选文档：

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/docs/icml_clblf_candidate.md
```

用途：单独记录 CLB-LF 的创新边界、实验事实、当前不足和下一步投稿门槛。

## 关键实验结果

### 1. 20k / 60k 早期信号

早期实验显示 full-layer gating 比 qabs 系列更有希望：

- War 20k / eval100：
  - `fulll26recent512attn`：`3.4778s / PPL 15.4297`
  - baseline：`3.5327s / PPL 15.6031`
  - 速度和 PPL 都优于 baseline；
- Monte 20k / eval100：
  - `fulll26recent512attn`：`3.4565s / PPL 29.0047`
  - baseline：`3.5160s / PPL 29.4340`
  - 同样优于 baseline；
- War 60k / eval200：
  - baseline：`23.2466s / PPL 29.2983`
  - `fulll27recent512attn`：`23.0057s / PPL 29.5082`
  - 速度略快，但 PPL 有小幅损失。

结论：full-layer budget 有信号，但 recent-only 在更长上下文下质量不够稳定。

### 2. 80k landmark fallback 关键突破

War and Peace 80k / eval200 baseline：

```text
baseline: 30.0984s / PPL 49.2224
```

关键结果：

| 模式 | Decode 时间 | PPL | 相对速度 | PPL 差值 | 判断 |
| --- | ---: | ---: | ---: | ---: | --- |
| `fulll25landmarkr1024s64attn` | 29.5950s | 49.6204 | +1.67% | +0.3981 | 速度好但 PPL 偏差仍大 |
| `fulll25landmarkr2048s64attn` | 29.7186s | 49.2745 | +1.26% | +0.0522 | 质量基本贴近 baseline |
| `fulll25landmarkr4096s64attn` | 29.8449s | 48.9016 | +0.84% | -0.3207 | 首个速度和 PPL 同时优于 baseline 的点 |
| `fulll25landmarkr2048s128attn` | 29.6978s | 49.0622 | +1.33% | -0.1601 | 也同时优于 baseline |

这说明 landmark fallback 明显优于 recent-only fallback，尤其是 `recent=4096, stride=64` 时质量信号最好。

### 3. 更长 eval 验证

War and Peace 80k / eval1000：

| 模式 | Decode 时间 | PPL | 相对速度 | PPL 差值 |
| --- | ---: | ---: | ---: | ---: |
| `baseline` | 151.1200s | 28.8883 | - | - |
| `fulll25landmarkr4096s64attn` | 150.3802s | 28.7829 | +0.49% | -0.1054 |

说明 `fulll25landmarkr4096s64attn` 在更长 eval 上仍保持速度正收益和 PPL 不差于 baseline。

### 4. 跨文本验证

Monte Cristo 80k / eval200，继续使用 War 20k 校准出的 layer map：

| 模式 | Decode 时间 | PPL | 相对速度 | PPL 差值 |
| --- | ---: | ---: | ---: | ---: |
| `baseline` | 30.0134s | 36.1664 | - | - |
| `fulll25landmarkr4096s64attn` | 29.9301s | 34.7897 | +0.28% | -1.3766 |

说明该 layer map 至少在另一个长文本上没有明显失效。但 PPL 下降可能包含文本片段波动或数值路径差异，后续不能只用单点结果支撑论文结论。

### 5. 进一步网格搜索

War and Peace 80k / eval200：

| 模式 | Decode 时间 | PPL | 判断 |
| --- | ---: | ---: | --- |
| `fulll25landmarkr4096s128attn` | 31.3075s | 49.1491 | PPL 好，但该轮并行计时偏慢 |
| `fulll25landmarkr4096s256attn` | 31.2699s | 49.0803 | PPL 好，但该轮并行计时偏慢 |
| `fulll24landmarkr4096s64attn` | 31.1007s | 51.2558 | 压 4 层后质量明显变差 |
| `fulll24landmarkr4096s128attn` | 31.2093s | 50.9776 | 压 4 层仍不稳 |
| `fulll25recent4096attn` | 30.5962s | 49.2266 | recent-only 质量接近，但速度没有优势 |
| `fulll24recent4096attn` | 30.5439s | 51.5232 | 压 4 层质量崩 |

结论：当前安全压缩层大约是 3 层；直接压到 4 层会破坏 PPL。下一步必须通过单层消融找出真正安全的压缩层，而不是只按 remote-mass 排序压缩。

## 当前最强候选

当前最强候选是：

```text
fulll25landmarkr4096s64attn
```

方法解释：

- 28 层模型中保留 25 层 full attention；
- 3 个低敏感层使用 `recent=4096 + stride=64 landmark`；
- full layers 由 remote-mass calibration map 决定；
- 相比 recent-only，landmark fallback 保留远程低频信息；
- 相比 SparQ/Quest，不做每 token query-aware top-k retrieval；
- 相比 DuoAttention/RazorAttention，不把核心创新放在 head retrieval/streaming 分类。

当前证据：

- War 80k/eval200：速度 `+0.84%`，PPL `-0.3207`；
- War 80k/eval1000：速度 `+0.49%`，PPL `-0.1054`；
- Monte 80k/eval200：速度 `+0.28%`，PPL `-1.3766`。

## 为什么还不能说已经达到 ICML 标准

当前方向有希望，但还不够强：

- 速度收益仍只有 `0.3%--1.3%`，论文说服力不足；
- 只在 Qwen3-0.6B 上验证，attention 不是唯一瓶颈，收益被 MLP 和 Python overhead 吃掉；
- 还没有在更大模型上验证；
- 还没有证明 layer budget 可以跨模型、跨长度稳定迁移；
- 还没有完成 28 层单层压缩消融，无法确定是否能安全压缩更多层；
- 通用 `layerbudgetattn` 质量好但速度慢，说明需要专用快速路径或 fused kernel。

因此，目前结论应表述为：

```text
CLB-LF 是当前最有潜力继续推进的投稿方向，但还没有达到 ICML 级别的完整证据。
```

## 服务器状态与阻塞

后半段尝试继续跑 28 层单层 `landmark4096s64` 消融，但服务器断连：

- `Test-Connection 10.176.34.117` 返回 `False`；
- `ssh -o ConnectTimeout=10 u21307130306@10.176.34.117` 多次超时；
- 本地 ROCm 环境可识别 7900XTX，但本地 `fdong/Qwen3-0.6B` 只有 tokenizer/config，没有模型权重，无法本地跑 PPL。

因此，最后将目标状态标记为 blocked，并把恢复脚本准备好。

## 服务器恢复后的操作

### 1. 同步本地修改

在本地执行：

```powershell
powershell -ExecutionPolicy Bypass -File ymluo\projects\qwen3_top2_head_limit3_ppl\scripts\sync_clblf_to_server.ps1
```

### 2. 检查昨晚单层消融是否完成

在服务器执行：

```bash
cd /home/u21307130306/kvcache/kv_cache/ymluo/projects/qwen3_top2_head_limit3_ppl
find outputs/icml_war80_eval200_single_layer_lm4096 -name ppl_by_mode.csv | wc -l
ps -u u21307130306 -o pid,etime,cmd | grep evaluate_qwen3_top2_head_limit3_ppl | grep -v grep
```

### 3. 继续安全层搜索

如果单层消融已完成或大部分完成，执行：

```bash
bash scripts/run_clblf_safe_layer_search_server.sh
```

该脚本会：

1. 汇总单层消融；
2. 生成 `safe_top{k}_layers_last.json`；
3. 并行测试 `fulll{28-k}landmarkr4096s64attn`；
4. 输出候选 PPL 和速度。

### 4. 下一步成功标准

若要继续向 ICML 方向推进，下一阶段至少要达到：

```text
War 80k/eval200: speedup > 2%, delta_ppl <= 0.2
War 80k/eval1000: speedup > 1%, delta_ppl <= 0.2
Monte 80k/eval200: speedup > 1%, delta_ppl <= 0.2
```

如果达不到，应转向：

- 更大模型验证；
- fused/compiled attention fallback kernel；
- 更细粒度 layer-wise mixed budget，而不是继续手工调 `fulll25`。

## 文件清单

本轮新增/修改的关键文件：

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/src/evaluate_qwen3_top2_head_limit3_ppl.py
ymluo/projects/qwen3_top2_head_limit3_ppl/src/select_safe_layer_budget.py
ymluo/projects/qwen3_top2_head_limit3_ppl/scripts/run_clblf_safe_layer_search_server.sh
ymluo/projects/qwen3_top2_head_limit3_ppl/scripts/sync_clblf_to_server.ps1
ymluo/projects/qwen3_top2_head_limit3_ppl/docs/icml_clblf_candidate.md
ymluo/projects/qwen3_top2_head_limit3_ppl/docs/five_hour_work_summary_2026_06_28.md
remote-change-log.md
```

## 总结

这 5 小时完成的核心工作不是“找到最终可投稿方法”，而是把方向从低创新的 qabs/SparQ-like 路线，推进到了一个更有论文叙事空间的 CLB-LF 路线，并拿到了第一个 80k 长上下文下速度和 PPL 同时不差于 baseline 的候选点。

下一步真正决定这个方向能不能冲 ICML 的关键，是服务器恢复后完成单层消融和安全层组合搜索。如果能把速度收益从当前不足 1% 提升到 2%--5%，同时保持 PPL 差值小于 0.2，这条线才值得继续写成论文主方法。
