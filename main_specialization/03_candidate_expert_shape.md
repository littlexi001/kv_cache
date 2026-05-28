# Candidate Design: Expert Input Shape

Expert input shape 决定 expert 的 computation ownership 是作用在完整 token state 上，还是只作用在某个子空间上。

1. **Full expert input / output：** expert 输入和输出都是完整 hidden size。当前最推荐的默认方案是 `router=head`，但 `expert=full`：每个 head 可以独立决定 routing，但被选中的 expert 仍处理完整 token state。
2. **Head expert input / output：** 每个 head 的 expert 只处理该 head 子空间。它会带来更强的 specialization inductive bias，但已有结果显示 NTP 明显下降，因此只适合作为 ablation 或机制分析。
3. **Spectral-band expert input / output：** expert 只处理某个 spectral band，并写回对应子空间。这是更强的 feature-subspace ownership，但当前尚未证明可稳定训练。
4. **Hybrid design：** router 使用子空间，expert 使用 full token vector。这是目前最有价值的中间方案：既让 gate 看到更干净的 feature signal，又不牺牲 expert 的完整表达能力。
