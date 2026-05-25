# Section 8 — Pyramid KV Compression

> 新增日期：2026-05-14；最近同步日期：2026-05-23

## 项目：`pyramid_kv_compression`

新增日期：2026-05-14

这个项目 patch Qwen3 attention，用 pyramid-shaped KV memory 做继续预训练：

```text
early layers:   full KV
middle layers:  compressed older KV
final layers:   full KV
```

hidden-state 序列长度保持不变。只有被选中的层会缩短 attention memory：旧的中间 blocks 会被学习到的 weighted K/V summaries 替换，anchor tokens 和 recent tokens 保持原始 KV。

推荐实验顺序：

```bash
bash ymluo/projects/pyramid_kv_compression/scripts/run_8gpu.sh sanity
bash ymluo/projects/pyramid_kv_compression/scripts/run_8gpu.sh compressor
bash ymluo/projects/pyramid_kv_compression/scripts/run_8gpu.sh attention
bash ymluo/projects/pyramid_kv_compression/scripts/run_8gpu.sh full
```

`METHOD_NOTES.md` 记录了当前风险：激进默认 schedule 已经产生过高 loss。下一轮严肃实验应从弱压缩中间层、更大的 anchor/recent window、以及 `attention` 阶段开始，再考虑 full-model training。
