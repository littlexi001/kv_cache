# Persistent-KV 复用设计

## 可证伪命题

在多轮问答和 Agent 共享长前缀的场景中，QKSieve 的 Key 索引与 ValueSketch 可以随精确 KV 一起常驻。分支切换只回退有效长度，不重建张量；随后追加的新 token 原位覆盖旧后缀。因此，热请求应保持与第一次相同分支完全一致的生成结果，并获得冻结 Robust 路径的稳态速度。

若分支回退后 token 发生变化、任何层的索引指针改变、索引重建计数增加，或热请求仍承担完整索引构建成本，则该命题失败。

## 先验条件

1. 精确 K/V、packed Key index 和 ValueSketch 都使用固定容量张量。
2. attention 与扫描 kernel 都显式接收当前有效长度，只读取 `[0,n)`。
3. Decode 每次最多追加一个 token，Key 与 Value 索引均已有原位 append kernel。
4. 同一共享前缀下，request-local QK 坐标系和位宽分配保持冻结。

## 数学模型

设共享前缀长度为 `P`，预分配容量为 `C`，第 `b` 个分支后缀为 `S_b`。精确缓存和辅助索引均写入固定数组：

```text
K[0:C), V[0:C), I_K[0:C), I_V[0:C).
```

当前有效长度为 `n`。回退算子只执行：

```text
R_P: n <- P, indexed_K <- P, indexed_V <- P.
```

它不修改坐标基、位宽、量化 scale 或任何张量地址。新分支第 `t` 个 token 写入位置 `P+t`。由于 kernel 只读取 `[0,n)`，旧分支在 `[P,C)` 中未覆盖的内容不可见。

## 实现契约

入口：`rewind_active_qksieve_cache(cache, P)`。

步骤：

1. 在修改状态前验证 KV cache、每层 Key index 和 Value index 均至少覆盖 `P`。
2. 调用预分配 KV cache 的 `crop(P)`。
3. 将每层 `packed_qmse_indexed_count` 和 packed index 的有效长度设为 `P`。
4. 将 ValueSketch runtime 及对应 state 的有效长度设为 `P`。
5. 保留所有底层张量、basis、allocation、metadata 与 CUDA workspace。
6. 后续 Decode 通过已有 append kernel 覆盖旧后缀。

审计接口 `active_qksieve_persistent_state_signature()` 输出每层重建计数及 Key/Value 张量地址。任何分支后发生变化都视为失败。

## 输出

基准输出必须分别报告：

- 已有 KV、首次建索引再生成的冷请求延迟；
- 建好索引后的共享前缀热分支延迟；
- 多个分支均摊一次建索引后的延迟；
- 不回退、连续追加的 ms/token；
- 重复同一分支的 token/hash 是否完全一致；
- 所有层是否复用原索引且无重建。

## 声明边界

该设计证明的是缓存生命周期和实际速度，不单独证明 LongBench/RULER 质量。跨独立请求、不同前缀或坐标统计明显变化时仍需新建 request-local 索引。
