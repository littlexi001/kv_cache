# Qwen3 KV Cache 随 Prefix 增长的几何性质结论

Date: 2026-05-28

## 0. 文档定位

这个文档用于记录 `fdong_seq_compress` 方向下关于 KV cache 几何性质的阶段性结论。

当前问题不是“马上设计一个 indexer”，而是先回答更前置的问题：

> 对同一条长文本，随着 prefix 逐渐变长，LLM 每层每头的 K/V cache 作为高维点云，其数学结构如何变化？

只有先理解这些结构，后面才知道 KV cache indexing / compression 应该利用什么性质，避免直接做结果导向的工程尝试。

本轮实验对应输出目录：

```text
fdong_seq_compress/outputs/qwen3_geometry_mps_long_20260528_142454
```

实验配置：

```text
model: fdong/Qwen3-0.6B
device: mps
dtype: float16
text: fdong_seq_compress/data/synthetic_texts/long_english_12000_words.txt
prefix lengths: 512, 1024, 2048, 4096, 8192, 12000
layers: 0-27
KV heads: 0-7
kinds: K, V
head_dim: 128
```

## 1. 本轮最重要结论

这次实验最清楚的结论是：

> K-cache 更像可索引的 address space；V-cache 更像承载内容的 content space。二者的几何性质差异很大，不应该用同一种方式压缩。

具体对比如下。这里先只写定性结论，具体数值放在后续各节。

| 性质 | K-cache | V-cache |
| --- | --- | --- |
| 有效维度 | 随 sequence length sublinear 增长，远慢于 token 数增长 | 也随 sequence length sublinear 增长，但有效维度整体比 K 更高 |
| 各向异性 | 强，存在明显 common direction / cone effect | 相比 K 弱得多，整体更分散 |
| 去均值后的相似性 | centering 后 token-token 平均相似性显著下降，说明 raw similarity 受公共方向影响很大 | centering 后平均相似性也接近 0，但 raw similarity 本来就不高 |
| 局部平滑性 | 强，相邻 token 的 K 向量高度相似，更像一条平滑高维轨迹 | 弱，相邻 token 的 V 向量差异明显更大 |
| 小 block 结构 | 支持小尺度连续 block，尤其是 4/8/16 token block | 不支持简单连续 block average，即使很小 block 内部也较分散 |
| 主子空间稳定性 | 主子空间随 prefix 增长逐渐稳定，但弱方向仍会旋转 | 主子空间更稳定，长 prefix 下 dominant subspace 几乎不再明显变化 |
| 新 token novelty | 新 token 对已有 top subspace 仍有明显 residual，不能被很小固定 basis 完全解释 | 也有明显 residual，且内容侧 residual 不应被忽略 |
| 层间差异 | 浅层 K 尤其各向异性强，后层 K 的公共方向相对减弱 | 后层 V 的有效维度更高，内容性更强 |
| 适合承担的角色 | 更像 address / index / routing space | 更像 content / evidence / information payload |
| 对压缩的直接启发 | 适合研究去 common direction 后的小 block index、delta、change point、routing summary | 更适合保留 exact residual、multi-slot content，或在 K index 命中后按需 gather，不适合直接平均压缩 |

这与“attention 可以拆成 index layer + content layer”的高层想法是对齐的：

```text
K / index side:   先找哪些历史位置或 block 值得看
V / content side: 再读取更精确的内容表示
```

## 2. 有效维度：KV 不是随 token 数线性扩维

从 prefix `512 -> 12000`，sequence 长度增长约 23 倍，但 K/V 的 effective rank 只小幅增长。

中位数统计如下：

```text
K effective_rank: 28.8 -> 37.3
K rank95:         56   -> 68
K stable_rank:    5.4  -> 6.7

V effective_rank: 60.3 -> 62.4
V rank95:         82   -> 90
V stable_rank:    10.9 -> 11.4
```

这里的 `rank95` 表示 centered SVD 下达到 95% energy 需要的 rank。

这个结果支持：

> 长 context 的 KV cache 点云并不是随着 token 数线性开辟新维度，而是在一个远小于 token 数的有效几何空间中逐渐填充。

但它不支持一个过强结论：

> 不能说 KV cache 可以被极低 rank 无损表示。

尤其是 V-cache，`head_dim=128` 时，`rank95` 仍然接近 90。这说明 V 的内容空间仍然相当丰富，不能只靠很小 rank 的压缩码替代。

## 3. K/V 几何差异：K 更像地址，V 更像内容

K-cache 的中位数表现：

```text
K raw offdiag cosine: ~0.66 at 512, ~0.63 at 12000
K adjacent cosine:    ~0.88 at 512, ~0.91 at 12000
```

V-cache 的中位数表现：

```text
V raw offdiag cosine: ~0.13 at 512, ~0.16 at 12000
V adjacent cosine:    ~0.41 at 512, ~0.45 at 12000
```

这说明：

- K 向量之间整体更相似；
- K 沿 token 序列变化更平滑；
- V 向量之间差异更大；
- V 沿 token 序列变化不如 K 平滑。

因此，K/V 的角色可以被更明确地区分：

```text
K: address / index / query matching
V: content / evidence / information payload
```

后续如果设计 KV compression，不应默认：

```text
compressed K = block average K
compressed V = block average V
```

更合理的思路是：

```text
用 K 的几何结构做索引；
对 V 保留更精细的读取路径，或设计更谨慎的 content compression。
```

## 4. K 的各向异性：raw 相似度被 common direction 污染

K 的 raw cosine 很高，但 centered cosine 接近 0：

```text
K raw offdiag cosine:      ~0.63
K centered offdiag cosine: ~0
```

这说明 raw K 空间里存在很强的 common direction / cone effect。

这对后续 index 设计非常关键：

> 如果直接用 raw K 的 dot product 或 cosine 做 nearest neighbor / block score，很可能会被 common component 主导，而不是捕捉真正有区分度的语义或检索结构。

后续应当系统测试：

```text
centered K
whitened K
remove top principal components
normalized residual K
layer/head-specific centering
```

也就是说，K-cache indexing 的第一步可能不是发明复杂 indexer，而是先把 K 空间中无区分度的公共方向去掉。

## 5. 主子空间稳定，但新 token 仍有明显 residual

相邻 prefix 的 top-16 subspace overlap 随长度增长而变高：

```text
K subspace overlap: 0.87 at 1024 -> 0.94 at 12000
V subspace overlap: 0.91 at 1024 -> 0.995 at 12000
```

这说明 dominant subspace 会随长文本增长逐渐稳定，尤其是 V-cache。

但新 token 相对前一 prefix top-16 basis 的 novelty residual 仍然明显：

```text
K novelty residual ratio: ~0.63 at 12000
V novelty residual ratio: ~0.68 at 12000
```

所以这里的正确结论是：

> 主能量结构稳定，但 top-16 这样的很小 basis 不能解释所有新 token 信息。

这支持一种中间立场：

- 可以维护长期稳定的 low-dimensional summary / basis；
- 但不能指望一个很小的固定 basis 捕捉全部 KV；
- compression 应该允许 residual、fallback、或者多尺度补充信息。

## 6. 局部平滑性：K 明显平滑，V 不明显

K 的 adjacent cosine 很高：

```text
K adjacent cosine median: ~0.88 -> ~0.91
```

V 的 adjacent cosine 明显低：

```text
V adjacent cosine median: ~0.41 -> ~0.45
```

这说明 K-cache 沿文本位置更像一条平滑高维轨迹，而 V-cache 更像每个 token 携带相对独立的内容。

这支持：

- K 可以考虑 delta compression；
- K 可以考虑 segment boundary / change point；
- K 可以考虑小窗口 summary；
- V 不应简单依赖相邻 token 平滑性。

## 7. Block 结构：K 支持小 block，V 不支持简单 block average

在 prefix `12000` 时，K 的 block within/between ratio：

```text
block 4:   0.36
block 8:   0.56
block 16:  0.82
block 32:  1.10
block 64:  1.56
block 128: 2.13
```

ratio 小于 1 表示 block 内部方差小于 block centroid 之间的方差，也就是这个 block 尺度有一定几何意义。

因此，K-cache 支持：

```text
4 / 8 / 16 token 小 block
```

但不太支持很大的固定连续块。

V 的 block within/between ratio：

```text
block 4:   1.43
block 8:   2.64
block 16:  4.49
block 32:  7.50
block 64:  13.32
block 128: 23.39
```

这说明 V 即便在很小 block 内也不够紧，大 block 更明显失效。

因此，本轮结果反驳了一个简单方案：

> 不应直接把连续 block 内的 V 做 average，当作 compressed content。

更可能成立的是：

```text
K block summary 用于 index；
V 保留原始 token、multi-slot 表示、或按需 gather。
```

## 8. Layer 差异：压缩策略必须 layer-aware

不同层的几何性质差异很大。

浅层 K 极度各向异性：

```text
layer 0 K raw cosine: ~0.98
layers 0-6 K raw cosine median: ~0.92
```

后层 V 的 effective rank 更高：

```text
V effective rank layers 0-6:   ~63
V effective rank layers 21-27: ~80
```

这说明：

> KV compression 不应是一套所有层共享的规则。

可能的方向是：

- 浅层 K：重点处理 common direction、局部平滑和小 block index；
- 中层 K/V：观察是否有更强语义 block 或 entity-level structure；
- 深层 V：更加谨慎，避免压掉任务相关内容；
- 不同 layer/head 可能需要不同 block size、rank、residual budget。

## 9. 本轮异常：prefix 12000 时 layer 21 K 全零

本轮发现一个异常：

```text
prefix=12000
kind=K
layer=21
all 8 KV heads mean_row_norm=0
```

但同一层在 `512/1024/2048/4096/8192` prefix 下正常，V 也正常。

因此暂时不要把这个现象解释为模型真实结构。它更可能来自：

- MPS 长上下文 kernel / cache 行为；
- float16 数值或设备行为；
- Hugging Face cache extraction edge case；
- 当前脚本在极长 prefix 下的某个边界问题。

后续需要单独复核：

```text
只跑 layer 21 / prefix 12000
用 DTYPE=float32 对比
用 CPU 小范围对比
检查 past_key_values 对应 layer 的原始 shape 和 norm
```

本轮全局中位数分析已排除这个异常行，因此总体结论不依赖它。

## 10. 当前支持与反驳的假设

### 支持的假设

1. KV cache 的有效维度远小于 sequence length，且随 prefix 增长很慢。
2. K-cache 比 V-cache 更适合作为 index / address space。
3. K-cache 存在强 common direction，raw K 相似度需要先做去公共方向处理。
4. K-cache 沿 sequence 更平滑，适合研究 delta、change point、小 block summary。
5. 小尺度 K block 有几何意义，尤其是 4/8/16 token block。
6. 主子空间会随 prefix 增长逐渐稳定，尤其是 V-cache。
7. 压缩策略应当 layer-aware、head-aware，而不是统一规则。

### 反驳或削弱的假设

1. “一个很小 rank 的 basis 可以解释全部 KV”不成立。
2. “K 和 V 可以用同一种压缩方式处理”不成立。
3. “连续 block averaging 可以同时压缩 K 和 V”不成立，尤其对 V 不成立。
4. “越大 block 越适合压缩”不成立；K 的有效 block 尺度更偏小。
5. “raw cosine / raw dot product 就能直接反映 K 的有用相似度”不成立，因为 common direction 很强。

## 11. 对下一步的启发

下一步不应直接做完整 CSA-style trainable compressor，而应继续做几何诊断和小型验证。

建议优先级：

### 11.1 复核异常

先确认 `prefix=12000, layer=21, K=0` 是否是 MPS 或脚本问题。

### 11.2 做 centered / whitened K 的几何分析

当前 raw K 明显被 common direction 主导。下一轮应比较：

```text
raw K
centered K
remove top-1 PC K
remove top-k PC K
whitened K
```

看这些变换后：

- effective rank 如何变化；
- block within/between ratio 是否更清楚；
- adjacent smoothness 是否保留；
- layer/head 差异是否更可解释。

### 11.3 研究 K-index + V-residual 的结构

本轮最自然的结构启发是：

```text
K side:
  small-block summary / centered K / low-rank or centroid index

V side:
  preserve exact token V
  or use multi-slot residual content
  or only在被 K index 命中后 gather 原始 V
```

也就是说，压缩不应该理解为：

```text
同时压 K 和 V
```

而更像：

```text
压 index，不轻易压 content。
```

### 11.4 引入真实文本对照

当前长文本是 synthetic coherent English report。它适合观察长程重复实体、主题迁移、跨段回指，但仍然是人造文本。

后续应加入：

- 真实新闻长文；
- 教科书章节；
- 代码文件；
- 多文档拼接；
- 用户对话 + 工具返回内容。

看几何结论是否稳定。

## 12. 当前一句话结论

本轮 Qwen3-0.6B 的长文本 KV 几何实验表明：

> 随着 prefix 变长，K/V cache 都表现出远低于 token 数的有效维度；但 K 和 V 的几何角色明显不同。K 更像平滑、各向异性、可小块索引的地址空间；V 更像高信息量、弱局部压缩性的内容空间。因此，后续 KV cache compression 的合理方向不是简单平均 K/V，而是先在去 common direction 的 K 空间里建立小尺度 index，再用更谨慎的方式保留或读取 V 内容。
