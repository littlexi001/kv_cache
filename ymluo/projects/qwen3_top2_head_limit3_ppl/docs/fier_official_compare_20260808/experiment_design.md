# FIER 官方实现公平测速：实验设计

## 第一阶段：最小可运行性测试

- GPU：RTX 3090 单卡。
- 模型：LongChat-7B-v1.5-32K，FP16。
- 长度：5K；生成 16 token；1 次重复。
- 条件：官方 release FIER、同 backend Full。
- 通过：模型成功 prefill 和 decode，控制器实际预算被记录，输出 JSON 完整。
- 失败：编译失败、OOM、控制器预算与请求预算不符、CUDA kernel 报错。

## 第二阶段：同步 256-token 测速

- 长度：8K、16K、32K。
- 预算：每 head 1,280 token；另测 FIER 论文常用预算。
- 每个条件重复 3 次，报告 median、p10、p90。
- Full 与 FIER 在同一物理 GPU 上轮换运行。
- 计时不包含模型加载，分别报告 prefill 和 decode；decode 包含每步选择、top-k、稀疏 Attention 和模型其余层。

## 第三阶段：与 QKSieve 对齐

- QKSieve 使用同一模型、历史和 256 个生成步骤。
- 分别报告：完整 Attention 路径、稳态 decode、含索引构建的 online decode。
- 只有在 active-token 预算和质量都对齐时，才能宣称谁更快。

## 判定

- `通过`：官方 FIER 和 QKSieve 均为真实索引、真实 KV、同步计时，并有匹配质量结果。
- `仅系统证据`：运行了官方 kernel/backend，但 selector 输入是随机索引。
- `不足`：Full 没读取历史 KV、计时未同步或两者预算不同。
