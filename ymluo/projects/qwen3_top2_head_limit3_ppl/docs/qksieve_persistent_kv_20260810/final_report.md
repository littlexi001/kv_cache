# QKSieve 持久化 KV 生命周期实验

## 结论

冻结的 QKSieve-Robust 索引可以与精确 KV 一起常驻，并跨共享前缀分支和
append-only decode 复用。单 seed RTX 3090 实测中，warm 整模型 decode 在
32K/64K 分别为 1.322x/2.221x；四个 32-token 分支均摊一次索引构建后为
1.082x/1.785x。32K 单次 32-token cold 请求为 0.702x，说明短输出仍无法摊薄
1.759 s 的构建成本；64K 同口径为 1.125x。

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
分片。随机种子为 `20260810`。Full 与 QKSieve 的 cold、warm、四分支均摊和
append-only 都是完整模型的直接墙钟计时；分阶段构建时间只用于诊断。

原始结果位于 `raw_results/20260810_qksieve_persistent_kv_v2/`。汇总程序重新读取
每层状态并独立检查生命周期，不信任运行程序写出的总布尔值。

## 结果

| 历史长度 | Cold | Warm | 四分支均摊 | Append-only | 一次构建 |
|---:|---:|---:|---:|---:|---:|
| 32K | 0.702x | 1.322x | 1.082x | 1.319x | 1.759 s |
| 64K | 1.125x | 2.221x | 1.785x | 2.211x | 1.998 s |

Warm 的 Full/QKSieve 延迟在 32K 为 83.714/63.347 ms/token，在 64K 为
144.910/65.256 ms/token。QKSieve 的 cold 延迟为 119.551/129.108 ms/token。

阶段级直接计时如下：

| 历史长度 | QK 因子 | Key 编码 | ValueSketch | 合计 |
|---:|---:|---:|---:|---:|
| 32K | 0.724 s | 0.518 s | 0.517 s | 1.759 s |
| 64K | 0.735 s | 0.603 s | 0.658 s | 1.997 s |

QK 因子主要处理每层的小矩阵，因此随历史长度变化很小；Key 与 Value 阶段扫描
历史，随长度增长。三项同量级，单独优化 mixed-bit scan 不能消除 cold 瓶颈。

## 生命周期审计

- 32 层的 Key 与 Value 都完成预建。
- 每个长度解析 6 个状态快照和 5 次回滚。
- 预建后索引长度等于 KV；decode 后每层都严格落后一个 token。
- 所有快照中的 Key/Value buffer 指针和重建计数完全不变。
- 回滚后重放的 token 序列和 SHA256 与第一次分支完全一致。
- 独立汇总输出 `all_correct=true`。

## 失败解释与命题更新

32K cold 的失败没有否定 sparse attention 的稳态优势。它否定的是“32-token
单次请求足以摊薄完整索引构建”的假设。64K 中 Full decode 每 token 更慢，而
QKSieve warm 延迟只小幅增长，所以同样 32-token 输出已经越过 break-even。
方法主张因此收窄为：持久化或共享前缀是 32K 的必要使用条件；64K 在本协议下
连单次 32-token cold 请求也已加速。

## 结论边界

本实验只支持 RTX 3090、该 MHA 模型和 32K/64K 的系统生命周期结论。重放一致
只证明同一方法的确定性，不证明与 Full 的任务质量相同。它也不证明在新用户问题
改变 Query 统计后还能复用旧的 request-local 索引；冻结契约下这种跨请求变化
需要重新构建。质量由独立 LongBench、RULER 和 PPL 实验承担。H100 三 seed、
128K、更多输出长度和多模型结果仍待补齐。
