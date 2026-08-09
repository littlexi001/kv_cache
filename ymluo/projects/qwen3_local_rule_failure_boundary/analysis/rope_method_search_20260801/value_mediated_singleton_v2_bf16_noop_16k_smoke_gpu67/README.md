# Value-mediated causal closure v2：16K 外推 smoke

> **状态：已被 16K、8-seed 复核取代。** 正式统计见 `../value_mediated_singleton_v2_bf16_noop_16k_8seed_gpu67/README.md`；本文件仅保留最初 2-seed smoke。

## 结论

**8K 的证据坐标局部闭环在 16K 两个新 case 上方向一致，但样本仅 2 seeds，只能视为外推 smoke。**

| 子集 | n | Pearson(predicted, actual) | Spearman | 符号正确率 |
|---|---:|---:|---:|---:|
| 全部 target（4 类 × 2 seeds） | 8 | 0.874 | 0.810 | 62.5% |
| Gold + conflict evidence | 4 | **0.988** | **1.000** | **100%** |
| Matched random（全部类别） | 8 | 0.665 | 0.024 | 50.0% |

random 的 Pearson 被少数大效应点拉高，但秩相关几乎为零；证据 target 的一阶排序与真实干预排序完全一致。该结果与 8K 的主结论一致：闭环最稳定地出现在 gold/conflict 证据坐标，对 filler 的近零局部信号并不稳定。

## 逐点结果

| Seed | 类别 | 预测 Δmargin | 实际 Δmargin |
|---:|---|---:|---:|
| 0 | Gold | -0.1972 | -0.1969 |
| 0 | Conflict | -0.1766 | -0.1916 |
| 0 | Lexical | -0.0372 | -0.0226 |
| 0 | Filler | -0.0016 | +0.1786 |
| 1 | Gold | +0.1055 | +0.1454 |
| 1 | Conflict | -0.1982 | -0.2653 |
| 1 | Lexical | -0.0102 | +0.0920 |
| 1 | Filler | +0.0004 | -0.0995 |

这里再次表明：`gold` 标签并不保证“提高 attention 一定有益”。Seed 0 的 gold 坐标对答案 margin 的局部作用就是负的，而且公式正确预测了该负方向。论文中应强调 **coordinate-specific causal utility**，而不是按 token 标签统一增强。

## 审计与限制

- Qwen3-8B，未量化 BF16，16,384 tokens，seed 0/1，物理 GPU 6/7。
- 每次只提高一个 score `0.25`，所有 delta 相对同一路径 `epsilon=0` no-op。
- 两个 seeds 的 no-op 相对 instrumented margin 差均严格为 `0`。
- no-op 相对 native margin 差为 `+0.0325` 与 `+0.0804`；两者均未改变首 token 决策。
- 只有两个 seeds，不能提供可靠置信区间，也不能单独用于论文主表。
- target ranking 使用正确答案 margin gradient，不是可部署方法。

## 32K 状态

同协议 32K BF16 在两张 24GB RTX 3090 上均于 native eager final-query 的 GQA `repeat_kv` 分配处 OOM（还需 256 MiB）。没有改用 GPU 0–5、没有切换本机 GPU，也没有用量化结果冒充 BF16 外推；详细记录见相邻目录 `value_mediated_singleton_v2_bf16_noop_32k_smoke_gpu67/README.md`。
