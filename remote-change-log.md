# 远端变更记录

## 2026-06-28 qwen3_top2_head_limit3_ppl 的 SparQ/QABS 速度优化

目标项目：

- 本地工作区：`C:\Users\27814\Desktop\work\codex_workspace\kvcache\kv_cache-main\kv_cache-main\ymluo\projects\qwen3_top2_head_limit3_ppl`
- 服务器 Codex 副本：`/home/u21307130306/kvcache_codex/kv_cache/ymluo/projects/qwen3_top2_head_limit3_ppl`
- 用户指定目标路径：`/home/u21307130306/kvcache/kv_cache/ymluo/projects/qwen3_top2_head_limit3_ppl`

修改文件：

- `src/evaluate_qwen3_top2_head_limit3_ppl.py`
- `src/qabs_cuda_kernels.py`
- `scripts/run_qabs_fast_speed_server.sh`

实现内容：

- 新增 SparQ 风格的 `qabs8cand3attn` 模式：用 query 绝对值最大的 8 个通道做部分 QK 打分，选出 3% candidate token，然后只在 candidate/protected token 集合上做精确 attention。
- 新增 CUDA partial-score kernel，包括 dim-major K-cache 路径，让历史 token 维度读取更连续。
- 用已选 candidate 直接构造最终 token index，避免 decode 路径上 dense keep-mask 到 index 的转换开销。
- 单模式测速时复用 prefill KV cache，避免不必要的 KV cache clone 开销。
- 调整测速脚本默认配置为 `qabs8cand3attn,baseline`，开启 CUDA candidate scoring，关闭自定义 CUDA final attention；当前 candidate 规模下 PyTorch gather + matmul 更快。

服务器环境记录：

- Python：`/home/u21307130306/miniconda3/envs/cudatest/bin/python`
- 模型权重：`/home/u21307130306/kvcache_codex/kv_cache/fdong/Qwen3-0.6B`
- 评测文本：`/home/u21307130306/kvcache_codex/kv_cache/ymluo/projects/qwen3_kcache_l2_neighbor_analysis/data/needle_in_haystack/needle_in_haystack.jsonl`
- 在 `cudatest` 环境中补装了缺失的 `accelerate`。
- 在导入 `transformers` 前增加了本地 guard，绕过当前环境中 `torchvision::nms` fake registration 缺失导致的导入失败。

实测结果：

| 实验 | Prefill | Decode tokens | 模式 | Decode 时间 | PPL |
| --- | ---: | ---: | --- | ---: | ---: |
| Baseline | 60,000 | 100 | `baseline` | 8.8460s | 1.0012 |
| 优化后 | 60,000 | 100 | `qabs8cand3attn` | 8.3625s | 1.0193 |
| 复用重排参考 | 60,000 | 100 | `qabs8cand3reuse` | 15.197s | 1.0131 |
| Baseline | 20,000 | 100 | `baseline` | 3.746s | 1.0462 |
| 优化后 | 20,000 | 100 | `qabs8cand3attn` | 4.587s | 1.0756 |

结论：

- 在 60k/100 长上下文 decode 测速中，`qabs8cand3attn` 达到速度目标，比 baseline 快约 5.5%。
- 原来的精确 rerank/reuse 路径 `qabs8cand3reuse` 仍然慢于 baseline。
- 当前最快路径有质量代价：PPL 比 baseline 更差，需要后续在速度和质量之间继续调参。
- 最后一轮同步到 `/home/u21307130306/kvcache/kv_cache` 时，服务器 SSH 端口 22 超时；本地文件已经更新，服务器恢复连接后需要重新同步。

## 2026-06-28 War and Peace 独立长文本复测

复测原因：

- 之前 needle-in-haystack 文本上的 PPL 约为 1，数值偏低，不适合作为唯一质量判断。
- 重新使用一份独立自然长文本做验证，检查 PPL 是否正常，以及速度优势是否稳定。

复测文本：

- 来源：Project Gutenberg `War and Peace`，下载地址 `https://www.gutenberg.org/files/2600/2600-0.txt`
- 服务器路径：`/home/u21307130306/kvcache/kv_cache/ymluo/projects/qwen3_top2_head_limit3_ppl/data/war_and_peace_pg2600.txt`
- 文本字符数：约 327 万字符
- Qwen3 tokenizer 统计：前约 320 万字符约 77.3 万 tokens，足够覆盖 60k prefill + decode。

复测命令配置：

- 模型：`/home/u21307130306/kvcache_codex/kv_cache/fdong/Qwen3-0.6B`
- 模式：`qabs8cand3attn,baseline`
- `QABS_CUDA_FINAL_KERNEL=false`
- `QABS_CUDA_CANDIDATE_KERNEL=true`
- `REUSE_PREFILL_CACHE=true`

复测结果：

| 文本 | Prefill | Decode tokens | 模式 | Decode 时间 | PPL |
| --- | ---: | ---: | --- | ---: | ---: |
| War and Peace | 60,000 | 100 | `qabs8cand3attn` | 10.1858s | 37.4879 |
| War and Peace | 60,000 | 100 | `baseline` | 8.9927s | 34.8498 |
| War and Peace | 20,000 | 100 | `qabs8cand3attn` | 4.4705s | 15.5547 |
| War and Peace | 20,000 | 100 | `baseline` | 3.5570s | 15.5953 |

复测结论：

- War and Peace 上的 PPL 明显比 needle-in-haystack 文本正常，说明之前 PPL 偏低主要是数据文本特性导致。
- 在 War and Peace 60k 和 20k 上，`qabs8cand3attn` 没有比 baseline 更快；之前的 60k 速度优势不稳定，不能作为最终结论。
- 质量上，20k 位置 `qabs8cand3attn` 与 baseline 基本接近；60k 位置 `qabs8cand3attn` 比 baseline 更差。
- 后续如果目标仍是“在自然长文本上速度稳定超过 baseline”，需要继续优化 decode 路径，而不是只保留当前 `qabs8cand3attn`。

## 2026-06-28 ICML 方向方法搜索与实验筛选

目标：

- 找一个不是简单复现 SparQ/Quest 的长上下文 decode attention 方法。
- 约束是速度要能超过 baseline，PPL 要接近或优于 baseline，并且要有足够论文创新性。
- 当前阶段先做快速原型筛选，不把 PyTorch 稀疏原型的速度当作最终系统上限，但会用它判断质量信号和工程瓶颈。

已对照的相关工作：

- SparQ Attention：用 query 最大通道近似打分，选择少量 KV 再做 attention；当前 `qabs8cand3attn` 与它高度相似，单独作为论文方法创新性不足。
- Quest：按 KV cache page 的 key min/max 上界做 query-aware page 选择，重点是减少 KV 读取。
- MInference：主要面向 long-context prefill 的动态稀疏 attention pattern 和 kernel。
- RetrievalAttention/RetroInfer：把 KV cache 视作检索库，通过 attention-aware vector retrieval 降低访问量。

本轮新增/筛选过的方法：

- `qabs8cand3top2attn`：三集合版本，先取 3% candidate，再精确 QK rerank 到 top2%，最后合并 protected sink/recent/self。
- `qabs8cand3top2hlim3attn`：在 top2% 后增加每个 token 最多被 3 个 head 选中的约束。
- `qabs8cand3top2globalattn` / `qabs16cand3top2globalattn`：把每个 head 固定 top2% 改成跨 head 的全局 token-head budget，允许重要 head/token 拿更多预算。
- `qabs8/16cand3sharedattn`：跨 head 共享 token 集，减少重复选 token。
- `qabs8/16cand3sharedr2/r4/r8attn`：共享 token 集的时间复用版本，每 N 个 decode token 刷新一次。
- `knormtop2`、`kdimq8cand3`、`ksign8cand3`、`kdomq8k8cand3`、`blockroute64`、`qbb64q8s3cand3`：用于排除较直接的 key-norm、posting/index、block route、block bound 类已有路线。

关键实验结果：

| 数据 | Prefill | Decode | 模式 | Decode 时间 | PPL | 判断 |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| War and Peace | 20,000 | 100 | `baseline` | 3.5248s | 15.5953 | 基线 |
| War and Peace | 20,000 | 100 | `qabs8cand3attn` | 4.4665s | 15.5547 | 质量接近，但慢，且像 SparQ |
| War and Peace | 20,000 | 100 | `qabs8cand3top2attn` | 4.4119s | 15.7763 | 三集合成立，但没有速度/质量优势 |
| War and Peace | 20,000 | 100 | `qabs8cand3top2globalattn` | 6.5761s | 15.2474 | PPL 优于 baseline，但慢 |
| War and Peace | 20,000 | 100 | `qabs16cand3top2globalattn` | 6.1850s | 13.8687 | 最强质量信号，但慢 |
| War and Peace | 20,000 | 100 | `qabs16cand3sharedattn` | 4.2373s | 15.6021 | 共享集合质量接近，但仍慢 |
| War and Peace | 60,000 | 100 | `baseline` | 9.0030s | 34.3854 | 基线 |
| War and Peace | 60,000 | 100 | `qabs16cand3sharedattn` | 9.6272s | 44.7404 | 长上下文质量崩得较明显 |
| War and Peace | 60,000 | 100 | `qabs16cand3top2globalattn` | 12.1514s | 35.1186 | 质量接近 baseline，但仍慢 |
| War and Peace | 10,000 | 100 | `baseline` | 2.4568s | 35.8789 | 已有替代路线筛选基线 |
| War and Peace | 10,000 | 100 | `knormtop2` | 4.1678s | 79.5067 | 质量差且慢 |
| War and Peace | 10,000 | 100 | `kdimq8cand3` | 10.9612s | 40.9685 | 慢且质量差 |
| War and Peace | 10,000 | 100 | `ksign8cand3` | 5.6887s | 44.4108 | 慢且质量差 |
| War and Peace | 10,000 | 100 | `kdomq8k8cand3` | 5.6683s | 64.3064 | 慢且质量差 |
| War and Peace | 10,000 | 100 | `blockroute64` | 8.0054s | 39.8630 | 慢且质量差 |
| War and Peace | 10,000 | 100 | `qbb64q8s3cand3` | 9.7716s | 53.8809 | 慢且质量差 |

阶段性判断：

- 目前还没有找到一个“现有原型直接可投 ICML”的方法；不能把当前结果包装成已经成功。
- `qabs8cand3attn` 基本是 SparQ 风格，速度也没有在自然长文本上稳定超过 baseline，不适合作为主创新。
- shared token-set 和 temporal reuse 没有通过筛选：短上下文也没赢速度，60k 质量下降明显。
- 最值得继续投入的是 `qabs16cand3top2globalattn` 背后的思想：跨 head 的全局 token-head budget。它在 20k 上 PPL 明显优于 baseline，60k 上质量接近 baseline，但当前 PyTorch 原型慢。
- 论文级创新不能只讲“top-k 稀疏 attention”，应讲成“query-aware candidate recall + exact rerank + 全局 token-head 预算分配 + fused/page-level 实现”，核心是把每个 head 固定预算改成全局受约束的 token-head 选择问题。

下一步建议：

- 方法名暂定为 `Budgeted TriSet Attention` 或 `Global-Budget Token-Head Attention`。
- 实现上不要继续堆 Python/PyTorch gather/topk；需要把 candidate selection、exact rerank、global budget、protected token 合并成 fused CUDA 或 page/block-level bound kernel。
- 先加诊断指标：exact full-attention top token recall、attention mass recall、每 token 被多少 head 共享、global budget 的 head/token 分布。
- 若诊断证明 global budget 用更少 token-head pair 保留更多 attention mass，再做 fused kernel；否则该方向要及时止损。

## 2026-06-28 追加诊断：global-budget 质量机制与 qabs16 预算扫描

新增诊断脚本修改：

- 文件：`src/analyze_qabs_top2_overlap.py`
- 增加 `torchvision::nms` fake-registration guard，避免服务器 `transformers` 导入失败。
- 原脚本只支持 `qabsXXcandYYrerank`，现在支持：
  - `qabsXXcandYYtop2attn`
  - `qabsXXcandYYtop2globalattn`
  - `qabsXXcandYYsharedattn`
  - `qabsXXcandYYsharedtop2attn`
- 新增输出字段：
  - `selection`
  - `selected_tokens`
  - `selected_token_fraction`
  - per-query 级别的 final selected token 数量，用于观察 global-budget 是否在跨 head 重分配。

4k/64 attention-mass 诊断：

- 输出目录：`outputs/icml_global_budget_diag_4k`
- 数据：War and Peace
- 配置：prefill 4096，eval 64，sampled query 16，top_fraction 2%

| 模式 | true top2 overlap | selected attention mass | candidate attention mass | 结论 |
| --- | ---: | ---: | ---: | --- |
| `qabs8cand3top2attn` | 0.4163 | 0.7149 | 0.7173 | qabs8 召回不足 |
| `qabs16cand3top2attn` | 0.5750 | 0.8179 | 0.8224 | qabs16 明显更稳 |
| `qabs16cand3top2globalattn` | 0.5176 | 0.8144 | 0.8224 | global 没有提升 mass recall |
| `qabs16cand3sharedattn` | 0.2809 | 0.6758 | 0.6758 | shared token-set 质量信号弱 |
| `qabs16cand3sharedtop2attn` | 0.2809 | 0.6746 | 0.6758 | shared rerank 也不理想 |

global-budget 分配诊断：

- `qabs16cand3top2globalattn` 的平均 selected tokens/head 仍为 84，但标准差约 37.9。
- 部分 layer/head 平均只拿 1 到 8 个 token，部分 layer/head 接近拿满 125 个候选 token。
- 说明 global-budget 确实在做强 head 间重分配，但 4k attention-mass 证据不支持“它比 per-head top2 更保留 attention mass”。
- 当前 global-budget 的 PPL 改善可能来自某些 token/head 对 logits 的非线性影响，不应只用 attention-mass 作为唯一解释。

20k qabs16 candidate/global 预算扫描：

- 输出目录：`outputs/icml_qabs16_budget_sweep_20k`
- 数据：War and Peace
- 配置：prefill 20k，eval 100，CUDA candidate kernel 开，final CUDA kernel 关

| 模式 | Decode 时间 | PPL | 结论 |
| --- | ---: | ---: | --- |
| `baseline` | 3.5969s | 15.5953 | 基线 |
| `qabs16cand1top2globalattn` | 6.7070s | 14.9334 | PPL 好但太慢 |
| `qabs16cand1p5top2globalattn` | 6.3826s | 14.3225 | PPL 好但太慢 |
| `qabs16cand2top2globalattn` | 6.4075s | 14.0837 | PPL 好但太慢 |
| `qabs16cand3top2globalattn` | 6.5011s | 13.8687 | 质量最好但慢 |
| `qabs16cand2top2attn` | 4.4240s | 14.0919 | 比 global 快很多，仍慢于 baseline |
| `qabs16cand3top2attn` | 4.4961s | 13.9876 | 质量强，速度未过关 |

20k qabs16 direct-attention 预算扫描：

- 输出目录：`outputs/icml_qabs16_direct_sweep_20k`

| 模式 | Decode 时间 | PPL | 结论 |
| --- | ---: | ---: | --- |
| `baseline` | 3.6003s | 15.5953 | 基线 |
| `qabs16cand1attn` | 4.4836s | 14.9880 | PPL 好，慢 |
| `qabs16cand1p5attn` | 4.0797s | 14.3080 | PPL 好，慢约 13% |
| `qabs16cand2attn` | 4.1322s | 14.1716 | PPL 好，慢 |
| `qabs16cand2p5attn` | 4.0760s | 14.1216 | PPL 好，慢约 13% |
| `qabs16cand3attn` | 4.1495s | 14.2012 | PPL 好，慢 |
| `qabs16cand4attn` | 3.8566s | 14.7040 | 当前最接近速度目标，但仍慢约 7% |

60k 复验：

- 输出目录：`outputs/icml_qabs16_selected_60k`

| 模式 | Decode 时间 | PPL | 结论 |
| --- | ---: | ---: | --- |
| `baseline` | 8.9914s | 34.3854 | 基线 |
| `qabs16cand2p5attn` | 10.1919s | 36.3833 | 慢且质量差 |
| `qabs16cand4attn` | 9.6747s | 35.5834 | 速度接近但仍慢，PPL 差约 1.20 |
| `qabs16cand3top2attn` | 9.6829s | 35.7167 | 速度接近但仍慢，PPL 差约 1.33 |
| `qabs16cand3top2globalattn` | 12.1721s | 35.1186 | 质量最接近但速度差 |

final CUDA kernel 复测：

- 输出目录：`outputs/icml_direct_finalcuda_20k`
- `qabs16cand4attn` 开 final CUDA kernel 后从约 3.86s 变为 7.46s。
- `qabs16cand2p5attn` 开 final CUDA kernel 后从约 4.08s 变为 5.71s。
- 结论：现有 final CUDA kernel 对 direct qabs 路径不是优化，后续不要继续沿这个 kernel 微调。

新的阶段性判断：

- `qabs16` 比 `qabs8` 有明显更强质量信号；这是目前最可靠的经验规律。
- `qabs16cand4attn` 是最接近“速度可过 baseline”的原型点，但 60k 下仍慢约 7.6%，PPL 也差约 1.20，不能作为已完成结果。
- `qabs16cand3top2attn` 在 20k 上质量很好，但 60k 不够稳。
- `qabs16cand3top2globalattn` 的论文创新性最好，但当前速度最差；如果继续它，必须把故事转成“受约束 token-head 预算 + fused/page-level 实现”，不能靠当前 PyTorch 原型。
- 下一步工程优先级应从“方法枚举”转到“减少 qabs16 direct/top2 的调度和 gather 开销”：候选 topk、KV gather、softmax/V 聚合需要合并；否则很难超过高度优化的 baseline attention。

## 2026-06-28 深夜追加：周期远程检索融合、低维 sweep 与 reuse 排除

本轮目标：继续寻找一个足够接近 ICML 投稿要求的方法，即不仅有创新叙事，还必须在 War and Peace 自然长文本上满足“decode 速度快于 baseline，PPL 接近或优于 baseline”。

环境修正：
- 服务器 `cudatest` 环境缺少 `ninja`，导致 `qabs_cuda_candidate_kernel` 实际没有加载。
- 已执行：`/home/u21307130306/miniconda3/envs/cudatest/bin/python -m pip install ninja`。
- 后续远端命令需要显式设置：`export PATH=/home/u21307130306/miniconda3/envs/cudatest/bin:$PATH`，否则非交互 SSH shell 仍找不到 `ninja`。

新增实现：
- 文件：`src/evaluate_qwen3_top2_head_limit3_ppl.py`
- 新增 `_qabs_remote_recent_attention_forward(...)`：周期检索模式下把远程候选 token 用索引 gather，把 protected recent/self 保持为连续 slice，尝试减少“最近窗口也被 index gather”的开销。
- 该路径没有作为默认路径保留；当前默认仍使用原 `_qabs_final_indices_from_selected(...) + _qabs_attention_from_final_indices(...)`。
- 新融合路径仅在环境变量 `QABS_PERIODIC_FUSED_REMOTE_RECENT=true` 时启用，原因是短测没有带来稳定收益，并且 `recent512` 当前复测出现异常 PPL，需要避免默认回归。

20k/100 融合周期检索复测：
- 输出目录：`outputs/icml_periodic_fused_recent512_20k`
- 配置：`prefill_tokens=20000`，`eval_tokens=100`，`protect_recent_tokens=512`，`qabs_cuda_candidate_kernel=true`

| 模式 | Decode 时间 | PPL | 判断 |
| --- | ---: | ---: | --- |
| `baseline` | 3.4784s | 15.6031 | 基线 |
| `qabs16cand1p5r8attn` + fused | 4.4509s | 15.6237 | 慢，质量接近但未优于 |
| `qabs16cand2r8attn` + fused | 4.0979s | 15.6106 | 慢，质量接近 |
| `qabs16cand2r16attn` + fused | 3.7262s | 16.0237 | 速度接近但仍慢，PPL 变差 |
| `recent512` | 2.6848s | 1622.4916 | 当前复测异常，不作为可信质量结论 |

低维周期检索 sweep：
- 输出目录：`outputs/icml_periodic_fused_dim_sweep_20k`
- 配置同上，融合路径开启。

| 模式 | Decode 时间 | PPL | 判断 |
| --- | ---: | ---: | --- |
| `baseline` | 3.4570s | 15.6031 | 基线 |
| `qabs8cand2r8attn` | 4.4444s | 19.1466 | 慢且质量差 |
| `qabs8cand3r8attn` | 4.1117s | 18.9889 | 慢且质量差 |
| `qabs12cand2r8attn` | 3.8533s | 16.7631 | 慢且质量差 |
| `qabs12cand2r16attn` | 3.8981s | 17.0005 | 慢且质量差 |

结论：降低 qabs 维度不能解决问题；`qabs16` 的质量信号仍然明显强于 `qabs8/qabs12`。

Direct qabs16 cand sweep 复测：
- 输出目录：`outputs/icml_direct_recheck_cuda_20k`
- 配置：`protect_recent_tokens=0`，`qabs_cuda_candidate_kernel=true`。

| 模式 | Decode 时间 | PPL | 判断 |
| --- | ---: | ---: | --- |
| `baseline` | 3.3771s | 15.6031 | 基线 |
| `qabs16cand2p5attn` | 4.1639s | 17.5655 | 慢且质量差 |
| `qabs16cand4attn` | 3.8228s | 17.8921 | 慢且质量差 |
| `qabs16cand6attn` | 3.5997s | 15.5532 | PPL 略优，但仍慢约 6.6% |

结论：`qabs16cand6attn` 是 direct 路线当前最接近的点，但还不能满足“速度快于 baseline”。

Reuse/rerank 路线复测：
- 输出目录：`outputs/icml_reuse_cand6_20k` 与 `outputs/icml_reuse_select_cuda_20k`

| 模式 | Decode 时间 | PPL | 判断 |
| --- | ---: | ---: | --- |
| `baseline` | 3.4701s | 15.6031 | 基线 |
| `qabs16cand6reuse` | 6.4920s | 14.5400 | 质量好但太慢 |
| `qabs16cand6reusefinal` | 6.1593s | 14.1331 | 质量最好但太慢 |
| `lagrefresh8qabs16cand6` | 5.8928s | 179.5069 | 质量崩 |
| `lagrefresh16qabs16cand6` | 5.9182s | 256.8494 | 质量崩 |
| `qabs16cand6reusefinal` + reuse CUDA selector | 6.5281s | 14.1331 | fused selector 更慢 |

结论：reuse/rerank 给出了强质量信号，但当前实现和选择策略离速度目标太远；lag refresh 直接复用旧集合会显著破坏质量。

阶段性判断：
- 还没有找到一个可以诚实声称“已达到 ICML 级结果”的方法。
- 当前最强可讲的研究方向不是 `qabs8cand3attn`，因为它高度类似 SparQ，且速度/PPL 均不稳定。
- 最有潜力的方向仍是“query-aware remote memory + sliding recent core + 周期/事件触发刷新”的结构，但必须做真正 fused kernel 或 page/block-level remote retrieval；仅用 PyTorch gather/topk 不可能赢过 baseline。
- 如果继续做，下一步应从算法原型转向系统实现：把 remote gather、recent contiguous attention、softmax、V reduce 融合到一个专用 kernel，并把 remote 候选缓存成固定容量结构，避免每步重建索引和拼接张量。
- 同时需要修复/复核 `recent512` 当前异常 PPL；它理论上应是速度下界，但当前复测质量异常，不能作为论文证据。

## 2026-06-28 继续推进：Remote-Mass Layer Gating 初步过线结果

本轮目标：在已有 qabs/SparQ-like 路线未能稳定超过 baseline 后，寻找一个更低调度开销、同时有论文故事的长上下文 decode attention 方法。

先排除/修正的点：
- `recent512` 的异常 PPL 不是 fast path bug。强制走 dense-mask 路径后，20k/20 token 上 `recent512` 仍然 PPL 很高，说明 War and Peace 在该位置需要远程上下文，单纯 sliding window 不可用。
- 新增 `landmarkr{recent}s{stride}attn`，即 head-phased static landmarks + recent。速度可以略快于 baseline，但 PPL 崩溃：
  - `landmarkr1024s64attn`：3.2345s / PPL 265.9194
  - `landmarkr512s32attn`：3.2555s / PPL 756.8337
  - baseline：3.4790s / PPL 15.6031
  - 结论：静态锚点不够，不能作为主线。
- `qabs16cand6attn` 开 final CUDA 后更慢：
  - `qabs16cand6attn` + final CUDA：5.1925s / PPL 16.4563
  - `qabs16cand8attn` + final CUDA：5.4118s / PPL 15.3518
  - baseline：3.4713s / PPL 15.6031
  - 结论：当前 final CUDA kernel 仍不是可用优化点。

新增代码：
- 文件：`src/evaluate_qwen3_top2_head_limit3_ppl.py`
- 新增模式：`landmarkr{recent}s{stride}attn`
- 新增模式：`fullh{n}recent{w}attn`
  - 前 N 个 head full，其余 recent；后来支持 `FULL_HEAD_MAP_PATH` 按层指定 head 排序。
- 新增模式：`fulll{n}recent{w}attn`
  - 选 N 个 layer 保留 full attention，其余 layer 使用 recent window。
  - 支持 `FULL_LAYER_MAP_PATH`，若 JSON 含 `mean_remote_mass`，则按每层远程 attention mass 总和降序选 full layer。
- 新增脚本：`src/analyze_full_head_remote_mass.py`
  - 用 baseline attention 在少量 decode token 上统计每层/每头 recent window 外 attention mass。
  - 输出 JSON：`outputs/icml_fullh_remote_mass_20k/head_map_recent512.json`

方法暂命名：`Remote-Mass Layer Gating`（RMLG）
- Calibration 阶段：用少量 token 的真实 attention mass，估计哪些 layer 最依赖远程上下文。
- Decode 阶段：高 remote-mass layer 保持 full attention；低 remote-mass layer 只保留 recent window。
- 优点：不做 per-token topk、不做 qabs partial score、不做 gather-heavy candidate rerank，调度开销远低于 SparQ/qabs 路线。
- 论文风险：layer/head 级 dense/sparse 混合已有相关工作，不能只讲“部分层 full”；创新点必须落在 remote-mass 校准、在线/低成本 layer gating、跨长度迁移和系统实现上。

20k/100 结果：
- Calibration：20k 位置，recent window=512，eval 32 token，输出 `head_map_recent512.json`。
- 使用同一个 map 做 layer 选择。

| 模式 | Decode 时间 | PPL | 判断 |
| --- | ---: | ---: | --- |
| `baseline` | 3.5069s | 15.6031 | 基线 |
| `fulll24recent1024attn` | 3.4167s | 16.0241 | 快约 2.6%，PPL 差 0.42 |
| `fulll25recent512attn` | 3.4670s | 15.6925 | 快约 1.1%，PPL 差 0.09 |
| `fulll26recent512attn` | 3.4679s | 15.4297 | 快约 1.1%，PPL 反而更好 |
| `fulll27recent512attn` | 3.4918s | 15.5636 | 快约 0.4%，PPL 更好 |
| `fulll26recent1024attn` | 3.4646s | 15.5943 | 快约 1.2%，PPL 基本持平 |

20k 结论：
- `fulll26recent512attn` 和 `fulll26recent1024attn` 是目前第一个同时满足“速度略快于 baseline、PPL 不差”的结果。
- 这比 qabs 路线更有工程可行性，因为它避免了每步候选选择和大规模 gather。

60k/100 单模式验证：
- 多模式 60k 共享 prefill cache 在 clone 阶段 OOM；因此改为单模式运行，每次仍使用 shared prefill，但不 clone，decode 时间仍不包含 prefill。
- 使用 20k calibration 生成的同一个 layer map，直接迁移到 60k。

| 模式 | Decode 时间 | PPL | 判断 |
| --- | ---: | ---: | --- |
| `baseline` | 11.4278s | 35.1346 | 单模式 60k 基线 |
| `fulll26recent512attn` | 11.2763s | 35.4248 | 快约 1.3%，PPL 差 0.29 |
| `fulll26recent1024attn` | 11.3051s | 35.6747 | 快约 1.1%，PPL 差 0.54 |

60k 结论：
- `fulll26recent512attn` 在 60k 仍然保持速度优势，PPL 小幅变差但没有崩。
- 这是目前最接近“可继续打磨成论文方法”的结果。
- 但速度收益还很小，只有约 1%–1.3%；需要更强实现或更细粒度 gating 才够有说服力。

下一步建议：
1. 做重复运行，确认 `fulll26recent512attn` 的 1% 速度优势不是噪声。
2. 在不同文本/位置验证：20k、40k、60k、80k；至少再用一份非 War and Peace 文本。
3. 把 layer map 从“离线 JSON + 环境变量”改成正式参数或在线 warmup calibration。
4. 做 ablation：随机选 26 层、前 26 层、后 26 层、remote-mass 选 26 层，证明 remote-mass gating 有必要。
5. 如果要冲 ICML，方法故事应是：`remote-mass calibrated layer gating + sliding local fallback + zero top-k/gather decode path`，不是 SparQ/qabs 复刻。

## 2026-06-28 继续验证：RMLG 重复运行、消融与跨文本结果

本轮目标：验证 `Remote-Mass Layer Gating`（RMLG）的速度收益是否是噪声，并证明 remote-mass 选层不是任意选层都能达到。

### 1. Calibration map 与 layer 排序

War and Peace 20k / recent512 / 32 token calibration 得到的 remote-mass layer 排序：

```text
[25, 24, 27, 26, 3, 21, 23, 22, 20, 5, 8, 18, 19, 4, 9, 6, 7, 16, 10, 15, 11, 12, 14, 13, 17, 2, 1, 0]
```

remote mass 最高的层主要集中在后层，同时 layer 3 也很重要；最低的是 layer 0/1/2。这说明“哪些 layer 需要 full attention”不是简单的前层/后层规则。

### 2. `fulll24recent512attn` 选层消融

配置：War and Peace，prefill 20k，eval 100，单模式运行，recent=512，保留 24/28 层 full，其余 4 层 recent。

| 选层策略 | Decode 时间 | PPL | 判断 |
| --- | ---: | ---: | --- |
| remote-mass top24 | 3.2452s | 16.0650 | 最好 |
| front24 | 3.2595s | 32.3419 | PPL 崩，说明前 24 层不是好选择 |
| back24 | 3.2761s | 17.0033 | 好于 front/random，但差于 remote-mass |
| random24(seed=1234) | 3.2857s | 19.4331 | 明显差 |

结论：remote-mass calibration 有实质作用；不是任意保留 24 层 full 都行。

### 3. `fulll26recent512attn` 20k 重复运行

配置：War and Peace，prefill 20k，eval 100，remote-mass map，`fulll26recent512attn,baseline` 同进程共享 prefill cache。

| Repeat | RMLG 时间 | RMLG PPL | Baseline 时间 | Baseline PPL | 速度收益 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.4599s | 15.4297 | 3.5235s | 15.6031 | 1.81% |
| 2 | 3.4857s | 15.4297 | 3.5394s | 15.6031 | 1.52% |
| 3 | 3.4877s | 15.4297 | 3.5351s | 15.6031 | 1.34% |
| mean | 3.4778s | 15.4297 | 3.5327s | 15.6031 | 1.55% |

结论：20k 上速度优势不是单次噪声；三次都快，PPL 稳定优于 baseline。

### 4. 跨文本验证：The Count of Monte Cristo

数据：Project Gutenberg `The Count of Monte Cristo`，下载到：
`data/count_monte_cristo_pg1184.txt`

流程：
- 对 Monte Cristo 自身做 20k / recent512 / 32 token remote-mass calibration。
- 再跑 `fulll26recent512attn,baseline`。

| 文本 | 模式 | Decode 时间 | PPL | 判断 |
| --- | --- | ---: | ---: | --- |
| Monte Cristo | `fulll26recent512attn` | 3.4565s | 29.0047 | 快且 PPL 更好 |
| Monte Cristo | `baseline` | 3.5160s | 29.4340 | 基线 |

结论：RMLG 不是 War and Peace 单文本特例；在第二本文学长文本上也同时快于 baseline 且 PPL 更好。

### 5. 目前最强结论

- 当前最有希望的方法是 `Remote-Mass Layer Gating (RMLG)`。
- 它相对 qabs/SparQ-like 路线的优势：
  - 不做 per-token topk。
  - 不做 qabs partial-score 扫描。
  - 不做 sparse gather-heavy rerank。
  - 只在 layer 级切换 full/recent，工程路径更简单。
- 已有证据：
  - War and Peace 20k 三次重复：稳定快约 1.3%–1.8%，PPL 更好。
  - War and Peace 60k 单模式：快约 1.3%，PPL 小幅变差但不崩。
  - Monte Cristo 20k：快约 1.7%，PPL 更好。
  - 选层消融证明 remote-mass top layers 明显优于 front/back/random。

### 6. 仍不足之处

- 速度收益还小，约 1%–2%；如果要 ICML，需要进一步扩大收益。
- 目前 calibration 仍依赖真实 attention 输出；论文方法需要改成低成本 warmup 或可预测 proxy。
- 需要更多模型/数据/长度：至少 40k、80k、另一个模型规模。
- 需要查清相关工作边界。已知 SparQ/Quest/MInference/RetrievalAttention 等都做长上下文稀疏/检索 attention；RMLG 的论文贡献不能只是“部分层 sparse”，必须强调 remote-mass calibrated layer gating、跨长度迁移、以及 zero top-k/gather decode path。

下一步建议：
1. 用 60k 再跑一次 `fulll26recent512attn` 和 baseline，确认 60k 速度差不是噪声。
2. 做 `fulll24/25/26/27` 在 Monte Cristo 上的 sweep，确认最优保留层数是否稳定。
3. 把 `FULL_LAYER_MAP_PATH` 环境变量改成正式 CLI 参数，输出 map 元数据到 summary，方便复现实验。
4. 设计低成本 calibration：只采样少数层/少数 head 或用 hidden-state/query norm proxy 预测 remote-mass layer。

### 7. 相关工作边界（快速核对）

- SparQ Attention: https://arxiv.org/abs/2312.04985
  - query-aware selective KV fetching，和早期 `qabs8cand3attn` 高度相似；RMLG 不走 per-token top-k KV 选择。
- Quest: https://arxiv.org/abs/2406.10774
  - query-aware KV cache page selection，用 key min/max page bound 估计重要 page；RMLG 不做 page-level query scoring，而是 layer-level full/recent gating。
- MInference: https://arxiv.org/abs/2407.02490
  - 主要面向 long-context prefill 的动态稀疏 pattern/kernel；RMLG 当前目标是 decode attention。

论文定位建议：RMLG 不能表述成“又一个 sparse attention”；应表述成 decode 阶段的 `calibrated layer-wise remote dependency gating`，核心卖点是用极低调度开销绕开 qabs/SparQ 的 per-token selection/gather 瓶颈。

## 2026-06-28 深夜继续验证：RMLG CLI 固化与动态层预算

本轮目标：把 RMLG 从临时环境变量实验固化成可复现实验入口，并继续尝试能否扩大“快于 baseline 且 PPL 不差”的空间。

### 1. 代码改动

文件：`ymluo/projects/qwen3_top2_head_limit3_ppl/src/evaluate_qwen3_top2_head_limit3_ppl.py`

- 新增正式参数：`--full_head_map_path`、`--full_layer_map_path`，替代临时 `FULL_HEAD_MAP_PATH` / `FULL_LAYER_MAP_PATH` 环境变量；环境变量仍作为 fallback。
- `ppl_by_mode.csv` 新增 `full_head_map_path`、`full_layer_map_path` 字段，便于复现实验。
- 新增动态层预算模式：`fullladapt{low}to{high}m{margin}recent{window}attn`。
  - 例：`fullladapt24to27m3p0recent512attn`。
  - 每个 decode token 先看上一 token logits 的 top1-top2 margin。
  - 若 margin >= threshold，使用低预算 `fulll{low}recent{window}attn`；否则使用高预算 `fulll{high}recent{window}attn`。
  - 这个设计比 per-token KV top-k 更便宜：只做一次 logits margin 判断，然后仍走 layer-level full/recent 切换。

验证：本地 `py_compile` 通过；脚本已同步到服务器。

### 2. CLI 参数 smoke

输出：`outputs/icml_cli_map_smoke/ppl_by_mode.csv`

配置：War and Peace，prefill 1024，eval 2，`fulll26recent512attn,baseline`，通过 `--full_layer_map_path outputs/icml_layer_ablation_maps/remote.json` 指定 map。

结果：CSV 正确写入 `full_layer_map_path`，说明正式 CLI 链路可用。

### 3. Monte Cristo 20k / eval 200：静态 N sweep

输出：`outputs/icml_fulll_remote_map_monte_sweep_20k/ppl_by_mode.csv`

| 模式 | Decode 时间 | PPL | 判断 |
| --- | ---: | ---: | --- |
| `fulll24recent512attn` | 6.8064s | 37.3783 | 最好；快且 PPL 优于 baseline |
| `fulll25recent512attn` | 6.8081s | 38.2071 | 快且 PPL 优于 baseline |
| `fulll26recent512attn` | 6.8652s | 38.5280 | 快且 PPL 优于 baseline |
| `fulll27recent512attn` | 6.9344s | 39.0267 | 快但 PPL 略差 |
| `baseline` | 6.9920s | 38.8316 | 基线 |

结论：在 Monte Cristo 的 200-token 评估里，`N=24` 比 `N=26` 更好，说明最优 full-layer 预算与文本/位置有关；这支持“自适应预算”方向。

### 4. War and Peace 20k / eval 200：静态 N sweep

输出：`outputs/icml_fulll_remote_map_war_sweep_20k_eval200/ppl_by_mode.csv`

| 模式 | Decode 时间 | PPL | 判断 |
| --- | ---: | ---: | --- |
| `fulll24recent512attn` | 6.8946s | 21.2101 | 更快但 PPL 变差 |
| `fulll25recent512attn` | 6.9040s | 20.8658 | 更快但 PPL 小幅变差 |
| `fulll26recent512attn` | 6.9443s | 20.7700 | 更快且略优于 baseline |
| `fulll27recent512attn` | 6.9936s | 20.4540 | PPL 最好，速度略快 |
| `baseline` | 7.0285s | 20.7808 | 基线 |

结论：War 20k / 200-token 下，质量最稳的是 `N=27`，速度收益较小；`N=26` 是更均衡点。

### 5. 跨文本 map 迁移

输出：
- `outputs/icml_crossmap_war_eval_monte_map_20k/ppl_by_mode.csv`
- `outputs/icml_crossmap_monte_eval_war_map_20k/ppl_by_mode.csv`

War eval + Monte map：

| 模式 | Decode 时间 | PPL | 判断 |
| --- | ---: | ---: | --- |
| `fulll24recent512attn` | 6.8417s | 20.8027 | 快，PPL 接近 baseline |
| `fulll26recent512attn` | 6.9052s | 20.7700 | 快，略优于 baseline |
| `baseline` | 7.0003s | 20.7808 | 基线 |

Monte eval + War map：

| 模式 | Decode 时间 | PPL | 判断 |
| --- | ---: | ---: | --- |
| `fulll24recent512attn` | 6.8956s | 40.2529 | 快但 PPL 明显差 |
| `fulll26recent512attn` | 6.9446s | 38.5280 | 快且优于 baseline |
| `baseline` | 7.0158s | 38.8316 | 基线 |

结论：`N=26` 的 layer map 有一定跨文本迁移性；`N=24` 更激进，对 map/文本更敏感。

### 6. War and Peace 60k / eval 200：静态与 sink

输出：
- `outputs/icml_war60_eval200_baseline/ppl_by_mode.csv`
- `outputs/icml_war60_eval200_fulll24/ppl_by_mode.csv`
- `outputs/icml_war60_eval200_fulll26/ppl_by_mode.csv`
- `outputs/icml_war60_eval200_fulll27/ppl_by_mode.csv`
- `outputs/icml_war60_eval200_fulll*_sink*/ppl_by_mode.csv`

核心结果：

| 模式 | Sink | Decode 时间 | PPL | 相对 baseline |
| --- | ---: | ---: | ---: | --- |
| `baseline` | 0 | 23.2466s | 29.2983 | 基线 |
| `fulll24recent512attn` | 0 | 22.6570s | 31.9165 | 快 2.5%，PPL 差太多 |
| `fulll26recent512attn` | 0 | 22.8856s | 30.4311 | 快 1.6%，PPL +1.13 |
| `fulll27recent512attn` | 0 | 23.0057s | 29.5082 | 快 1.0%，PPL +0.21 |
| `fulll26recent512attn` | 256 | 22.8414s | 29.8365 | 快 1.7%，PPL +0.54 |
| `fulll27recent512attn` | 512 | 23.0997s | 29.4033 | 快 0.6%，PPL +0.10 |

结论：长上下文 60k 下，`N=27` 的质量最好但速度收益小；sink 能改善质量，但不能同时明显扩大速度收益。`N=26 + sink256` 是更偏速度的折中点。

### 7. 动态层预算：20k 与 60k

方法：`fullladapt24to27m{threshold}recent512attn`。低置信 token 用 `N=27`，高置信 token 用 `N=24`。

War 20k / eval 200，输出：`outputs/icml_adapt_layer_budget_war20_eval200/ppl_by_mode.csv`

| 模式 | Decode 时间 | PPL | 判断 |
| --- | ---: | ---: | --- |
| `fullladapt24to27m1p0recent512attn` | 6.9242s | 20.8415 | 快但 PPL 略差 |
| `fullladapt24to27m2p0recent512attn` | 6.8743s | 20.7179 | 快且略优于 baseline |
| `fullladapt24to27m3p0recent512attn` | 6.9239s | 20.4356 | 快且明显优于 baseline |
| `fullladapt24to27m5p0recent512attn` | 6.9721s | 20.5791 | 快且优于 baseline |
| `baseline` | 7.0188s | 20.7808 | 基线 |

Monte 20k / eval 200，输出：`outputs/icml_adapt_layer_budget_monte20_eval200/ppl_by_mode.csv`

| 模式 | Decode 时间 | PPL | 判断 |
| --- | ---: | ---: | --- |
| `fullladapt24to27m2p0recent512attn` | 7.0430s | 38.5988 | PPL 优于 baseline，但时间略慢 |
| `fullladapt24to27m3p0recent512attn` | 6.9817s | 38.8073 | 快且基本持平 baseline |
| `fullladapt24to27m5p0recent512attn` | 7.0015s | 38.8997 | 快但 PPL 略差 |
| `baseline` | 7.0241s | 38.8316 | 基线 |

War 60k / eval 200，输出：
- `outputs/icml_war60_eval200_adapt24to27_m2/ppl_by_mode.csv`
- `outputs/icml_war60_eval200_adapt24to27_m3/ppl_by_mode.csv`
- `outputs/icml_war60_eval200_adapt24to27_m3_sink256/ppl_by_mode.csv`

| 模式 | Sink | Decode 时间 | PPL | 判断 |
| --- | ---: | ---: | ---: | --- |
| `baseline` | 0 | 23.2466s | 29.2983 | 基线 |
| `fullladapt24to27m2p0recent512attn` | 0 | 22.9691s | 29.8011 | 快 1.2%，PPL +0.50 |
| `fullladapt24to27m3p0recent512attn` | 0 | 23.1376s | 29.7787 | 快 0.5%，PPL +0.48 |
| `fullladapt24to27m3p0recent512attn` | 256 | 22.9919s | 29.6187 | 快 1.1%，PPL +0.32 |

结论：动态层预算比纯静态 `N=24` 稳得多，且在 War 20k 出现“快且 PPL 明显更好”的结果；但在 60k 上收益仍偏小。它比原始 RMLG 更像论文方法，因为它把“remote-mass layer ranking”和“token confidence budget controller”结合起来，而不是固定阈值剪层。

### 8. 当前最适合继续打磨的论文方向

暂定方法名：`Confidence-Guided Remote-Mass Layer Gating`（C-RMLG）。

核心故事：
1. 用少量 warmup token 估计每层 remote attention dependency，得到 layer-level remote-mass ranking。
2. Decode 时只在高 remote-mass 层保留 full attention，其他层退化为 recent/sink attention，避免 SparQ/qabs 的 per-token KV top-k 和 gather-heavy path。
3. 用上一 token logits margin 做低成本置信度控制：高置信 token 使用更低 full-layer budget，低置信 token 自动提高 full-layer budget。
4. 这形成一种 coarse-grained、硬件友好的动态稀疏：选择发生在 layer budget 级别，而不是 token/key 级别。

当前证据：
- 静态 RMLG：War 20k、Monte 20k、War 60k 都能跑出快于 baseline 的点。
- 消融：remote-mass top layers 明显优于 front/back/random。
- 跨文本：`N=26` 对 War/Monte map 互换仍基本可用。
- 动态 C-RMLG：War 20k 上 `m3` 同时比 baseline 快且 PPL 更好；Monte 20k 上接近持平；War 60k 上仍快但 PPL 小幅变差。

风险：
- 速度收益目前还是小，尤其 60k 只有约 0.5%--1.7%。如果要冲 ICML，需要继续扩大收益，或者换更大模型/更长上下文证明 attention bottleneck 更明显时收益放大。
- 目前 calibration 仍用真实 attention，需要设计低成本 warmup proxy；否则论文贡献会被认为是工程启发式。
- 动态预算还需要记录低/高预算触发比例、margin 分布、不同 threshold 的稳定性。

下一步优先级：
1. 给 adaptive 模式记录低/高预算触发次数和平均 margin，写入 CSV/summary。
2. 在 80k 或 100k 上测 `baseline`、`fulll27`、`fulll26+sink256`、`fullladapt24to27m3+sink256`。
3. 找一个更大模型或至少 Qwen3 更大参数版本；0.6B 的 attention 占比太低，速度收益可能被 MLP/调度开销淹没。
4. 把 calibration 从真实 attention 改成低成本 proxy：例如每层 query/key norm、RoPE distance sensitivity、或者少量层采样后插值。

### 9. 补充：adaptive 统计字段

又补了一次小改动：`ppl_by_mode.csv` 现在对 `fullladapt...` 模式记录：

- `adaptive_low_tokens`
- `adaptive_high_tokens`
- `adaptive_low_fraction`
- `adaptive_mean_margin`
- `adaptive_margin_threshold`

Smoke 输出：`outputs/icml_adapt_stats_smoke/ppl_by_mode.csv`。在 4-token smoke 中，`fullladapt24to27m3p0recent512attn` 触发低预算 3 次、高预算 1 次，`adaptive_low_fraction=0.75`，字段链路确认可用。

## 2026-06-28 继续推进：80k 暴露的问题与 landmark fallback

本轮目标：验证 C-RMLG 是否能在更长上下文上保持“快且 PPL 好”，并尝试修复 80k 质量问题。

### 1. 带统计 adaptive sweep：War/Monte 20k eval200

重新跑带 `adaptive_low_tokens`/`adaptive_high_tokens` 的 20k sweep。

输出：
- `outputs/icml_adapt_stats_war20_eval200_v2/ppl_by_mode.csv`
- `outputs/icml_adapt_stats_monte20_eval200_v2/ppl_by_mode.csv`

War and Peace 20k：

| 模式 | 时间 | PPL | low/high | low fraction | 判断 |
| --- | ---: | ---: | ---: | ---: | --- |
| `fullladapt20to27m2p0recent512attn` | 6.8892s | 21.1138 | 64/136 | 0.320 | 过激，PPL 差 |
| `fullladapt20to27m3p0recent512attn` | 6.8535s | 20.8202 | 42/158 | 0.210 | 快但 PPL 略差 |
| `fullladapt22to27m2p0recent512attn` | 6.8757s | 20.9625 | 66/134 | 0.330 | PPL 差 |
| `fullladapt22to27m3p0recent512attn` | 6.9297s | 20.5812 | 40/160 | 0.200 | 快且优于 baseline |
| `fullladapt24to27m2p0recent512attn` | 6.9554s | 20.7179 | 67/133 | 0.335 | 快且略优 |
| `fullladapt24to27m3p0recent512attn` | 6.9796s | 20.4356 | 39/161 | 0.195 | 快且最好 |
| `baseline` | 7.0387s | 20.7808 | - | - | 基线 |

Monte Cristo 20k：

| 模式 | 时间 | PPL | low/high | low fraction | 判断 |
| --- | ---: | ---: | ---: | ---: | --- |
| `fullladapt20to27m2p0recent512attn` | 6.9128s | 39.5604 | 63/137 | 0.315 | PPL 差 |
| `fullladapt20to27m3p0recent512attn` | 6.8763s | 38.5934 | 43/157 | 0.215 | 快且优于 baseline |
| `fullladapt22to27m2p0recent512attn` | 6.8887s | 40.7229 | 68/132 | 0.340 | 明显差 |
| `fullladapt22to27m3p0recent512attn` | 6.9357s | 39.7626 | 42/158 | 0.210 | 差 |
| `fullladapt24to27m2p0recent512attn` | 6.9564s | 38.5988 | 66/134 | 0.330 | 快且优于 baseline |
| `fullladapt24to27m3p0recent512attn` | 6.9952s | 38.8073 | 41/159 | 0.205 | 快且接近 baseline |
| `baseline` | 7.0546s | 38.8316 | - | - | 基线 |

结论：低预算触发比例大约 20%--34%。`20to27/22to27` 更激进，但质量不稳定；`24to27` 更稳。置信度 controller 有用，但不是免费午餐。

### 2. War 80k eval200：20k map 迁移失败

输出：
- `outputs/icml_war80_eval200_baseline/ppl_by_mode.csv`
- `outputs/icml_war80_eval200_fulll27/ppl_by_mode.csv`
- `outputs/icml_war80_eval200_fulll26_sink256/ppl_by_mode.csv`
- `outputs/icml_war80_eval200_adapt24to27_m3_sink256/ppl_by_mode.csv`

| 模式 | 时间 | PPL | 判断 |
| --- | ---: | ---: | --- |
| `baseline` | 30.0984s | 49.2224 | 基线 |
| `fulll27recent512attn` | 30.0254s | 50.5624 | 只快 0.2%，PPL 差 1.34 |
| `fulll26recent512attn + sink256` | 29.6932s | 49.9447 | 快 1.3%，PPL 差 0.72 |
| `fullladapt24to27m3p0recent512attn + sink256` | 29.9184s | 50.5412 | PPL 差 |

结论：20k calibration map 迁移到 80k 时质量明显不稳。即使只把 1 层从 full 改成 recent，也会显著伤 PPL。这说明 `recent-only fallback` 过硬，且 remote-mass layer 排序随位置变化。

### 3. 80k remote-mass calibration

输出：`outputs/icml_fullh_remote_mass_80k/head_map_recent512.json`

20k layer 排序：
```text
[25, 24, 27, 26, 3, 21, 23, 22, 20, 5, 8, 18, 19, 4, 9, 6, 7, 16, 10, 15, 11, 12, 14, 13, 17, 2, 1, 0]
```

80k layer 排序：
```text
[4, 25, 27, 23, 24, 22, 26, 3, 21, 7, 11, 20, 9, 5, 8, 16, 18, 19, 14, 13, 6, 12, 15, 17, 10, 1, 2, 0]
```

差异：layer 4 在 80k 升到第 1，说明 remote dependency 不是完全固定的 layer 属性；需要位置自适应或稳定 proxy。

但用 80k map 复测没有改善：

| 模式 | 时间 | PPL | 判断 |
| --- | ---: | ---: | --- |
| `fulll26recent512attn + sink256` + 80k map | 29.5594s | 50.5965 | 更快但 PPL 更差 |
| `fulll27recent512attn` + 80k map | 29.8207s | 50.5624 | 与 20k map 基本一样 |
| `fullladapt24to27m3p0recent512attn + sink256` + 80k map | 29.6493s | 50.7576 | 更差 |

判断：单纯用更近位置重新校准 layer 排序不能解决 80k PPL 问题；非 full 层完全丢远程信息是核心瓶颈。

### 4. 新增 full-layer + landmark fallback 模式

代码：`src/evaluate_qwen3_top2_head_limit3_ppl.py`

新增模式：`fulll{N}landmarkr{recent}s{stride}attn`

含义：
- remote-mass top-N 层保留 full attention；
- 其他层不再只看 recent，而是看 `sink + landmark stride + recent + self`；
- 目标是在低 remote 层保留少量全局位置信息，缓解 80k recent-only 质量崩坏。

Smoke：`outputs/icml_fulll_landmark_smoke/ppl_by_mode.csv` 通过。

War 80k eval200 结果：

| 模式 | 时间 | PPL | 判断 |
| --- | ---: | ---: | --- |
| `baseline` | 30.0984s | 49.2224 | 基线 |
| `fulll27recent512attn` | 30.0254s | 50.5624 | PPL 差 1.34 |
| `fulll27landmarkr512s64attn` | 29.6822s | 50.3109 | landmark 有改善，但仍差 1.09 |
| `fulll26recent512attn + sink256` | 29.6932s | 49.9447 | PPL 差 0.72 |
| `fulll26landmarkr512s64attn` | 29.6592s | 49.6542 | 更好，快 1.46%，PPL 差 0.43 |
| `fulll26landmarkr512s128attn` | 29.6431s | 50.4594 | stride 太稀，质量差 |

结论：landmark fallback 是正确方向，确实比 recent-only 恢复质量；但 80k 上仍未达到“PPL 保持良好”的强标准。

### 5. 相关工作边界更新

快速核对到的关键相关工作：

- SparQ Attention: https://arxiv.org/abs/2312.04985
  - query-aware selective KV fetching；我们早期 `qabs8cand3attn` 与其相近，因此不能作为创新主线。
- Quest: https://arxiv.org/abs/2406.10774
  - query-aware page selection，ICML 2024；说明 page/token 级 query-aware 稀疏已经成熟。
- MInference: https://arxiv.org/abs/2407.02490
  - 主要面向 prefill 的 dynamic sparse attention/kernel。
- RazorAttention: https://arxiv.org/abs/2407.15891
  - retrieval heads + streaming heads + compensation token，属于 head-level KV compression。
- DuoAttention: https://arxiv.org/abs/2410.10819
  - retrieval heads full KV、streaming heads constant KV；decode speedup 很强。它和我们的 head-level `fullh` 方向非常接近，因此我们不能只做 head retrieval/streaming。
- TriAttention: https://arxiv.org/abs/2604.04921
  - 2026 新工作，指出 post-RoPE recent-query importance 不稳定，转向 pre-RoPE Q/K concentration 和 trigonometric distance preference。这对我们很关键：80k 实验也显示用少量 post-RoPE attention calibration 的迁移性有限。

### 6. 阶段性判断

当前最像 ICML 方向的故事需要从 `RMLG` 升级为：

`Position-aware Confidence-Guided Remote-Mass Layer Gating with Landmark Fallback`

但当前实验还不够强：
- 20k：可以做到快且 PPL 好；
- 60k：能做到快且 PPL 小幅变差；
- 80k：仍不能满足 PPL 良好，说明方法还需要改进。

下一步更有希望的方向：
1. 引入 pre-RoPE 或位置距离 proxy，避免只靠 post-RoPE warmup attention 做 layer ranking。
2. 对非 full 层不要只用 recent；应使用“少量全局 landmark/compensation token + recent”，并学习/估计每层 stride。
3. 需要把 full/recent/landmark 的层预算从固定 N 改成 layer-wise budget：每层可以是 full、stride64、stride128、recent-only，而不是二分类。
4. 如果只在 Qwen3-0.6B 上做，速度收益会被 MLP 和 Python 调度淹没；要冲论文，最好拿到 1.7B/4B/8B 模型或实现 fused kernel。

## 2026-06-28 继续推进：80k landmark budget sweep

本轮目标：针对 80k 上 `recent-only fallback` 伤 PPL 的问题，系统扫 `full-layer + landmark fallback` 的 N、stride、recent window。

### 1. 新增/复用模式

模式：`fulll{N}landmarkr{recent}s{stride}attn`

含义：
- remote-mass top-N layers：full attention；
- 其余 layers：`sink + landmark stride + recent + self`；
- 与 `fulll{N}recent{R}attn` 相比，非 full 层不再完全丢远程信息。

代码位置：`src/evaluate_qwen3_top2_head_limit3_ppl.py`

### 2. War 80k / eval200 sweep 汇总

Baseline：`30.0984s / PPL 49.2224`

| 模式 | Decode 时间 | PPL | 速度收益 | PPL 差值 | 判断 |
| --- | ---: | ---: | ---: | ---: | --- |
| `fulll25landmarkr1024s64attn` | 29.5950s | 49.6204 | +1.67% | +0.3981 | 当前最佳折中 |
| `fulll26landmarkr2048s64attn` | 29.8867s | 49.6376 | +0.70% | +0.4152 | 质量接近但速度收益小 |
| `fulll26landmarkr512s64attn` | 29.6592s | 49.6542 | +1.46% | +0.4319 | 次优折中 |
| `fulll26landmarkr1024s64attn` | 29.7268s | 49.8188 | +1.23% | +0.5964 | 不如 512/2048 |
| `fulll25landmarkr512s32attn` | 30.1165s | 49.8933 | -0.06% | +0.6710 | 质量尚可但不快 |
| `fulll26recent512attn + sink256` | 29.6932s | 49.9447 | +1.35% | +0.7224 | landmark fallback 明显更好 |
| `fulll26landmarkr512s16attn` | 30.6354s | 50.0780 | -1.78% | +0.8556 | stride 太密，慢且无收益 |
| `fulll26landmarkr512s64attn + sink256` | 30.1943s | 50.1132 | -0.32% | +0.8908 | sink 反而伤害 |
| `fulll25landmarkr512s64attn` | 30.2376s | 50.1444 | -0.46% | +0.9220 | 不如 recent1024 |
| `fulll26landmarkr512s32attn` | 30.3150s | 50.1501 | -0.72% | +0.9278 | stride32 不划算 |
| `fulll27landmarkr512s64attn` | 29.6822s | 50.3109 | +1.38% | +1.0885 | 只改 layer0 不稳定 |
| `fulll26landmarkr512s128attn` | 29.6431s | 50.4594 | +1.51% | +1.2370 | stride 太稀 |
| `fulll24landmarkr512s64attn` | 29.7524s | 53.3588 | +1.15% | +4.1364 | 剪太多层，质量崩 |

输出目录包括：
- `outputs/icml_war80_eval200_fulll25_landmark1024s64`
- `outputs/icml_war80_eval200_fulll26_landmark2048s64`
- `outputs/icml_war80_eval200_fulll26_landmark512s64`
- `outputs/icml_war80_eval200_fulll26_landmark1024s64`
- `outputs/icml_war80_eval200_fulll25_landmark512s32`
- `outputs/icml_war80_eval200_fulll26_landmark512s16`
- `outputs/icml_war80_eval200_fulll26_landmark512s64_sink256`
- `outputs/icml_war80_eval200_fulll25_landmark512s64`
- `outputs/icml_war80_eval200_fulll26_landmark512s32`
- `outputs/icml_war80_eval200_fulll24_landmark512s64`

### 3. 关键发现

1. landmark fallback 比 recent-only fallback 明显更稳：
   - `fulll26recent512attn + sink256`: PPL +0.7224
   - `fulll25landmarkr1024s64attn`: PPL +0.3981，速度还更快。
2. 不是 landmark 越密越好：stride16/32 都更慢且质量不更好；stride64 反而最稳。
3. 不是 N 越大越好：`fulll27...` 只让 layer0 fallback，PPL 反而比 `fulll26...` 差；这说明 bottom layers 的相互作用不是简单单调关系。
4. `N=24` 质量崩，说明可压缩层数上限大概在 2--3 层，至少在 Qwen3-0.6B / 80k 这个设置下如此。

### 4. 当前论文方法判断

现在更合理的方法雏形是：

`Calibrated Layer Budgeting with Landmark Fallback (CLB-LF)`

比之前的 RMLG 更准确：
- 不只是 full/recent 二分类；
- 每层可以分配 full、landmark+recent、recent-only 等不同预算；
- landmark fallback 是保持长上下文 PPL 的必要组件；
- 置信度 controller 可作为 token-level budget modulation，但 80k 上还不够稳定。

当前最强证据：
- 20k：adaptive/full-layer gating 可做到快且 PPL 更好；
- 60k：静态/动态 gating 可做到快且 PPL 小幅变差；
- 80k：`fulll25landmarkr1024s64attn` 做到 +1.67% speed，PPL +0.398；这是目前 80k 最接近“速度快、PPL 保持良好”的点。

仍不足：
- 80k PPL 还没有完全持平 baseline；
- 速度收益仍小于 2%；
- 目前只在 Qwen3-0.6B 上验证，attention 不是绝对瓶颈。

下一步：
1. 实现 layer-wise budget map，而不是只靠 top-N：例如 JSON 指定每层 `full / landmark64 / landmark128 / recent`。
2. 用 20k/80k remote mass 和 bottom-layer sweep 自动搜索每层预算，目标是把 80k PPL 差压到 <0.2，同时速度保持 >1%。
3. 如果服务器只能用 Qwen3-0.6B，应做 fused/less-Python 的专门实现；否则速度收益会被调度噪声吞掉。

## 2026-06-28 80k 关键突破：CLB-LF 的 4096 landmark 版本

### 1. 方法更新

当前最有希望的投稿雏形命名为：`CLB-LF`（Calibrated Layer Budgeting with Landmark Fallback）。核心不是 SparQ 式 query-aware token top-k，也不是 DuoAttention/RazorAttention 式 head-level retrieval/streaming，而是：

- 先用短校准段估计各层远程依赖强度，得到 layer budget；
- 高远程依赖层保留 full attention；
- 少数低敏感层使用 `recent window + landmark stride`，避免 recent-only 丢失远程信息；
- 关键实现是专用 `fulll{N}landmarkr{recent}s{stride}attn`，避免通用 JSON layerbudget 的 Python 分支开销。

### 2. War and Peace 80k / eval200 关键结果

Baseline：`30.0984s / PPL 49.2224`

| 模式 | Decode 时间 | PPL | 相对速度 | PPL 差值 | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| `fulll25landmarkr1024s64attn` | 29.5950s | 49.6204 | +1.67% | +0.3981 | 早期速度最好但 PPL 仍偏高 |
| `fulll25landmarkr2048s64attn` | 29.7186s | 49.2745 | +1.26% | +0.0522 | 质量基本贴近 baseline |
| `fulll25landmarkr4096s64attn` | 29.8449s | 48.9016 | +0.84% | -0.3207 | 首个速度和 PPL 同时优于 baseline 的点 |
| `fulll25landmarkr2048s128attn` | 29.6978s | 49.0622 | +1.33% | -0.1601 | 速度略好，PPL 也优于 baseline |

对应输出：
- `outputs/icml_war80_eval200_fulll25_landmark2048s64/ppl_by_mode.csv`
- `outputs/icml_war80_eval200_fulll25_landmark4096s64/ppl_by_mode.csv`
- `outputs/icml_war80_eval200_fulll25_landmark2048s128/ppl_by_mode.csv`

### 3. 更长 eval 与跨文本验证

War and Peace 80k / eval1000：

| 模式 | Decode 时间 | PPL | 相对速度 | PPL 差值 |
| --- | ---: | ---: | ---: | ---: |
| `baseline` | 151.1200s | 28.8883 | - | - |
| `fulll25landmarkr4096s64attn` | 150.3802s | 28.7829 | +0.49% | -0.1054 |

Monte Cristo 80k / eval200，继续使用 War 20k 校准出的 layer map：

| 模式 | Decode 时间 | PPL | 相对速度 | PPL 差值 |
| --- | ---: | ---: | ---: | ---: |
| `baseline` | 30.0134s | 36.1664 | - | - |
| `fulll25landmarkr4096s64attn` | 29.9301s | 34.7897 | +0.28% | -1.3766 |

对应输出：
- `outputs/icml_war80_eval1000_baseline/ppl_by_mode.csv`
- `outputs/icml_war80_eval1000_fulll25_landmark4096s64/ppl_by_mode.csv`
- `outputs/icml_monte80_eval200_baseline/ppl_by_mode.csv`
- `outputs/icml_monte80_eval200_fulll25_landmark4096s64/ppl_by_mode.csv`

### 4. 进一步网格搜索结果

War and Peace 80k / eval200：

| 模式 | Decode 时间 | PPL | 判断 |
| --- | ---: | ---: | --- |
| `fulll25landmarkr4096s128attn` | 31.3075s | 49.1491 | PPL 好，但该轮并行计时偏慢 |
| `fulll25landmarkr4096s256attn` | 31.2699s | 49.0803 | PPL 好，但该轮并行计时偏慢 |
| `fulll24landmarkr4096s64attn` | 31.1007s | 51.2558 | 压 4 层后质量崩，20k map 中 layer17 不安全 |
| `fulll24landmarkr4096s128attn` | 31.2093s | 50.9776 | 同上，不建议 |
| `fulll25recent4096attn` | 30.5962s | 49.2266 | recent-only 质量可贴近，但速度没有优势 |
| `fulll24recent4096attn` | 30.5439s | 51.5232 | 压 4 层质量崩 |

### 5. 当前判断

`fulll25landmarkr4096s64attn` 是目前最强候选：

- 在 80k/200 上：`+0.84%` decode speed，PPL 比 baseline 低 `0.3207`；
- 在 80k/1000 上：`+0.49%` decode speed，PPL 比 baseline 低 `0.1054`；
- 在 Monte Cristo 80k/200 上：`+0.28%` decode speed，PPL 比 baseline 低 `1.3766`；
- 质量信号明显强于之前的 `qabs8cand3attn`、recent-only、full-head sparse、reuse/rerank；
- 创新叙事应放在“校准式层预算 + landmark fallback + 位置/置信度扩展”，不能包装成 SparQ attention。

不足：

- decode speed 仍只有 `0.3%--0.8%`，距离 ICML 级别的工程说服力不够；
- Qwen3-0.6B 上 attention 不是唯一瓶颈，压 3/28 层的收益会被 MLP 和 Python 调度吞掉；
- 如果要冲投稿，需要扩大到更大模型或实现更低开销 kernel，或者找到能安全压缩 5--8 层的 layer budget。

### 6. 当前阻塞/下一步

已启动 28 层单层 `landmark4096s64` 消融，目标是找出除 layer0/1/2 以外还能安全压缩的层，再构造自定义 full-layer map：把安全压缩层放到排序末尾，用 `fulll{28-k}landmarkr4096s64attn` 走专用快速路径。

该消融运行期间本地 SSH 到服务器 `10.176.34.117` 连续超时，`Test-Connection` 也失败，说明服务器网络暂时不可达。待服务器恢复后需要继续检查：

```bash
cd /home/u21307130306/kvcache/kv_cache/ymluo/projects/qwen3_top2_head_limit3_ppl
find outputs/icml_war80_eval200_single_layer_lm4096 -name ppl_by_mode.csv | wc -l
ps -u u21307130306 -o pid,etime,cmd | grep evaluate_qwen3_top2_head_limit3_ppl | grep -v grep
```

如果单层消融完成，按 PPL 从低到高选择安全层集合，生成自定义 `full_layer_map_path`，再跑：

```bash
CUDA_VISIBLE_DEVICES=0 /home/u21307130306/miniconda3/envs/cudatest/bin/python src/evaluate_qwen3_top2_head_limit3_ppl.py \
  --model_name_or_path /home/u21307130306/kvcache_codex/kv_cache/fdong/Qwen3-0.6B \
  --text_path data/war_and_peace_pg2600.txt \
  --output_dir outputs/icml_war80_eval200_custom_safe_layers_lm4096s64 \
  --prefill_tokens 80000 --eval_tokens 200 \
  --chunk_size 512 --eval_chunk_size 1 \
  --modes fulll{28-k}landmarkr4096s64attn \
  --full_layer_map_path outputs/icml_custom_maps/safe_layers_last.json \
  --reuse_prefill_cache true \
  --protect_sink_tokens 0 \
  --disable_sparse_stats true \
  --log_every 1000 \
  --make_plots false
```

## 2026-06-28 续：服务器断连期间的本地准备

### 1. 新增安全层选择工具

新增脚本：`src/select_safe_layer_budget.py`

用途：读取 28 个单层 fallback 消融目录：

```text
outputs/icml_war80_eval200_single_layer_lm4096/layerXX/ppl_by_mode.csv
```

并和 baseline CSV 比较，自动输出：

- `single_layer_summary.csv`：每层单独压缩后的 PPL/速度 delta；
- `safe_top{k}_layers_last.json`：把最安全的 k 个压缩层放到 `top_layers` 末尾；
- 对应的专用模式建议：`fulll{28-k}landmarkr4096s64attn`。

关键点：`full_layer_indices()` 使用 `top_layers[:N]` 作为 full layers，因此要让某些层走 fallback，必须把这些层排到 `top_layers` 最后，而不是写在最前面。

### 2. 新增远端续跑脚本

新增脚本：`scripts/run_clblf_safe_layer_search_server.sh`

服务器恢复后执行：

```bash
cd /home/u21307130306/kvcache/kv_cache/ymluo/projects/qwen3_top2_head_limit3_ppl
bash scripts/run_clblf_safe_layer_search_server.sh
```

该脚本会：

1. 用 `src/select_safe_layer_budget.py` 汇总单层消融；
2. 生成 `outputs/icml_safe_layer_maps_lm4096s64/safe_top{k}_layers_last.json`；
3. 并行验证 `safe_top{k}` 候选：`fulll{28-k}landmarkr4096s64attn`；
4. 打印每个候选的 `ppl_by_mode.csv`。

### 3. 小优化

`src/evaluate_qwen3_top2_head_limit3_ppl.py` 中给 `full_layer_indices()` 增加了 `(map_path, requested)` 级缓存，避免 decode 每层重复构造同一个 full-layer set。该优化不改变算法语义，只减少 Python 调度开销；本地已通过：

```bash
python -m py_compile src/evaluate_qwen3_top2_head_limit3_ppl.py src/select_safe_layer_budget.py
```

### 4. 当前状态

服务器 `10.176.34.117` 仍不可达：

- `Test-Connection 10.176.34.117` 返回 `False`；
- `ssh -o ConnectTimeout=10 u21307130306@10.176.34.117` 超时。

本地 ROCm venv 可识别 7900XTX：`torch 2.9.1+rocm7.2.1 / cuda True / device_count 1`，但本地 `fdong/Qwen3-0.6B` 只有 tokenizer/config，没有 safetensors 权重，因此无法本地跑 PPL。

下一步仍是等服务器恢复后同步这三个文件并继续：

- `src/evaluate_qwen3_top2_head_limit3_ppl.py`
- `src/select_safe_layer_budget.py`
- `scripts/run_clblf_safe_layer_search_server.sh`

## 2026-06-28 续：投稿候选文档与同步脚本

### 1. 新增投稿候选文档

新增：`docs/icml_clblf_candidate.md`

该文档把当前路线明确为 `CLB-LF: Calibrated Layer Budgeting with Landmark Fallback`，并写清楚：

- 为什么不能继续把 `qabs8cand3attn` 当主创新点；
- CLB-LF 与 SparQ/Quest、RazorAttention/DuoAttention 的边界；
- 当前 War/Monte 80k 实验结果；
- 目前 ICML 证据不足的位置；
- 服务器恢复后的最低验证门槛。

### 2. 新增 Windows 同步脚本

新增：`scripts/sync_clblf_to_server.ps1`

服务器恢复后在本地执行：

```powershell
powershell -ExecutionPolicy Bypass -File ymluo\projects\qwen3_top2_head_limit3_ppl\scripts\sync_clblf_to_server.ps1
```

它会同步：

- `src/evaluate_qwen3_top2_head_limit3_ppl.py`
- `src/select_safe_layer_budget.py`
- `scripts/run_clblf_safe_layer_search_server.sh`
- `docs/icml_clblf_candidate.md`
- `remote-change-log.md`

并在服务器上运行 py_compile 与 chmod。

### 3. 本地校验

已通过：

```powershell
python -m py_compile ymluo\projects\qwen3_top2_head_limit3_ppl\src\evaluate_qwen3_top2_head_limit3_ppl.py ymluo\projects\qwen3_top2_head_limit3_ppl\src\select_safe_layer_budget.py
```

PowerShell 脚本也已用 `[scriptblock]::Create(...)` 做语法检查。

### 4. 服务器状态

再次测试服务器：

- `Test-Connection -ComputerName 10.176.34.117 -Count 2 -Quiet` 为 `False`；
- `ssh -o BatchMode=yes -o ConnectTimeout=10 u21307130306@10.176.34.117 'echo ok'` 仍超时。

这已经是连续多轮同一网络阻塞。当前还能继续推进的本地工作已经基本完成；真正的下一步必须依赖服务器恢复，或者本地补齐 Qwen3-0.6B 权重后用 7900XTX 跑小规模验证。
