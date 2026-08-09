# Value-mediated causal closure v2：32K BF16 尝试记录

## 状态

**未产生实验结果：两张指定 GPU 均因显存不足而 fail-closed。**

- 模型：Qwen3-8B，未量化 BF16。
- 长度：32,768 tokens。
- 设备：远程物理 GPU 6/7，各为 RTX 3090 24GB。
- 两个 shard 在相同位置失败：native eager final-query 中，Qwen3 GQA 的 `repeat_kv` 试图再分配 256 MiB。
- 失败时单卡约使用 23.37 GiB，其中 PyTorch 已分配约 22.51 GiB。
- 两张卡均未生成 `raw/*_result.json`，因此不存在可分析或可引用的 32K 数值。

没有使用 GPU 0–5、本机 GPU、量化模型或改变协议来掩盖该失败。后续若要完成 32K BF16，应先把 final-query native control 改成严格等价、不会物化重复 KV heads 的 grouped-GQA kernel，并重新验证其与当前 native kernel 的数值等价性；不能简单跳过 native control 后把结果当成同一协议。

远程原始目录：

`/home/fdong/ymluo/projects/qwen3_local_rule_failure_boundary/outputs/20260801_value_mediated_singleton_v2_bf16_noop_32k_smoke_gpu67`

