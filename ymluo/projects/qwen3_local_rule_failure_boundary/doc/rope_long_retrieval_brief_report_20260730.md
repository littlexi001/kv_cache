# RoPE 长程检索简要汇报

长上下文会改变证据与 Query 的 RoPE 相位，使远程证据 QK 分数发生振荡并可能被压低，随后经 attention、Value 和残差流累计为答案 PPL 上升。为此，我们设计了 SAGE-RoPE：近程保留 RoPE，远程用校准后的 pre/post 双路分数召回，以 75% post-RoPE、25% pre-RoPE 融合，并只在约 2% 候选上计算 softmax。在 Qwen3-8B 两跳任务的 24 个全新 seeds 上，32K 的 Gold PPL 相比 exact post-RoPE Top-2% 从 8.056 降至 5.605，首 token 准确率从 20.8% 升至 37.5%，两链均命中率从 60.9% 升至 77.4%；64K 的 PPL 从 4.412 降至 3.914，但准确率未提高且置信区间略跨 0。SAGE 因而适合作为有门控的远程补偿，而不应在 8K–16K 无条件启用。
