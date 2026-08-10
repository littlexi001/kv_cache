# QKSieve 持久化 KV 生命周期实验

## 结论

冻结的 QKSieve-Robust 索引可以与精确 KV 一起常驻，并跨共享前缀分支和
append-only decode 复用。固定确定性工作负载上三次独立进程实测中，warm 整模型
decode 在 32K/64K 的中位数分别为 1.474x/2.503x；四个 32-token 分支均摊一次
索引构建后为 1.220x/2.040x。包含 dense prefill 的单次 32-token cold-E2E 在
32K 为 0.963x，在 64K 为 1.013x；因此 32K 的短输出尚不能抵消构建成本，64K
单次请求大致打平，复用前缀后收益才明显。

## 可证伪命题

对于已经物化并会复用的长上下文 KV，QKSieve 的历史 Key 索引和 ValueSketch
只构建一次；后续分支、回滚和追加生成不重新编码未变化的历史。若这个命题成立，
warm 与 append-only 延迟应接近，索引指针和重建计数应在所有分支中保持不变，
并且复用次数增加后一次性构建成本应被 sparse decode 节省抵消。

## 先验与数学计账

1. 历史 KV 在分支请求间不变，因此其辅助索引也是可缓存状态。
2. 当前 token 的 Query 仍需每步准备，不能排除在 sparse decode 计时之外。
3. 一次性构建不能混入 warm 数字，也不能从分阶段微基准拼接出整模型延迟。

对复用次数 $R$、每个分支生成长度 $G$，计账为：

$$T_{full}=R(H_f+GD_f),$$

$$T_{sparse}=I+R(H_s+GD_s)+A,$$

其中 $I$ 是 QK 坐标、完整 Key index 与 ValueSketch 的一次性构建，$A$ 是新增
token 的增量维护。只有直接计时满足 $T_{full}>T_{sparse}$ 时，才报告请求级加速。

## 实现契约

- 模型：Yarn-Llama-2-7B-128K，原生 32 Query/32 KV head MHA，FP16。
- 方法：冻结的 always-on QKSieve-Robust；无 Full fallback、router、任务规则、
  精确重排或候选复用。
- 索引：request-local QK-balanced 坐标、240-bit mixed-bit Key index、rank-16
  block-256 INT4 ValueSketch，候选预算为冻结的 $B(N)$。
- CUDA 扩展预加载在请求计时外；QK 因子、全历史 Key 编码和 ValueSketch 构建
  全部计入 cold。
- 每个长度在同一 request-local QK 坐标系下执行四个 32-token 分支，回滚后
  重放第一个分支；append-only 在同一请求内连续生成 128 token。
- 实现允许索引严格落后 KV 一个 token：生成 token $t$ 时不需要 token $t$ 自己
  的 Key/Value；它在下一步消费前增量写入。

## 实验设置

32K 使用两张 RTX 3090，64K 使用三张；Full 和 QKSieve 使用匹配的卡数与模型
分片。每个长度分别运行 `20260810/20260811/20260812` 三个独立进程；输入、分支
和 greedy decode 都是固定确定性的，因此这些标签只区分进程重复，不代表三个
独立工作负载。Full 与 QKSieve 的 cold、warm、四分支均摊和 append-only 都是
完整模型的直接墙钟计时；分阶段构建时间只用于诊断。

原始结果位于 `raw_results/20260810_qksieve_persistent_kv_v3_multiseed/`。汇总程序重新读取
每层状态并独立检查生命周期，不信任运行程序写出的总布尔值。

## 结果

| 历史长度 | Cold-index | Cold-E2E | Warm | 四分支均摊 | Append-only | 一次构建 |
|---:|---:|---:|---:|---:|---:|---:|
| 32K | 0.806x [0.693, 0.809] | 0.963x [0.948, 0.971] | 1.474x [1.177, 1.487] | 1.220x [1.001, 1.229] | 1.472x [1.174, 1.484] | 1.486 s [1.469, 1.566] |
| 64K | 1.312x [1.295, 1.319] | 1.013x [1.011, 1.013] | 2.503x [2.488, 2.522] | 2.040x [2.022, 2.053] | 2.493x [2.479, 2.511] | 1.650 s [1.647, 1.681] |

方括号为三次进程重复的 bootstrap 中位数 95% 区间，不表示工作负载分布或跨硬件
方差。Warm 的 Full/QKSieve 延迟中位数在 32K 为 84.002/57.032 ms/token，在
64K 为 145.081/57.973 ms/token。Cold-index 不含 dense prefill；Cold-E2E 包含
prefill，两种口径不能混写。按中位数计算的 break-even 为 56/19 个生成 token。

阶段级直接计时如下：

| 历史长度 | QK 因子 | Key 编码 | ValueSketch | 合计 |
|---:|---:|---:|---:|---:|
| 32K | 0.511 s | 0.517 s | 0.471 s | 1.486 s |
| 64K | 0.495 s | 0.556 s | 0.605 s | 1.650 s |

QK 因子主要处理每层的小矩阵，因此随历史长度变化很小；Key 与 Value 阶段扫描
历史，随长度增长。前三项是各自的进程中位数，总构建是整段墙钟中位数，不能按列
严格相加。三项同量级，单独优化 mixed-bit scan 不能消除 cold 瓶颈。

## 生命周期审计

- 32 层的 Key 与 Value 都完成预建。
- 每个长度解析 6 个状态快照和 5 次回滚。
- 预建后索引长度等于 KV；decode 后每层都严格落后一个 token。
- 所有快照中的 Key/Value buffer 指针和重建计数完全不变。
- 回滚后重放的 token 序列和 SHA256 与第一次分支完全一致。
- 独立汇总输出 `all_correct=true`。

## 失败解释与命题更新

32K cold-E2E 的失败没有否定 sparse attention 的稳态优势。它否定的是“32-token
单次请求足以摊薄完整索引构建和 prefill”的假设。64K 中 Full decode 每 token
更慢，而 QKSieve warm 延迟基本不随长度增长，所以同样 32-token 输出大致达到
break-even。方法主张因此收窄为：持久化或共享前缀是 32K 的必要使用条件；64K
单次短请求仅能打平，不能据此宣称明显请求级加速。

## 结论边界

本实验只支持 RTX 3090、该 MHA 模型和 32K/64K 的系统生命周期结论。重放一致
只证明同一方法的确定性，不证明与 Full 的任务质量相同。它也不证明在新用户问题
改变 Query 统计后还能复用旧的 request-local 索引；冻结契约下这种跨请求变化
需要重新构建。质量由独立 LongBench、RULER 和 PPL 实验承担。H100 三次重复、
128K、更多输出长度和多模型结果仍待补齐。
