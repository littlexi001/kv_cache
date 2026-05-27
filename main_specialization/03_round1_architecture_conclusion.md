# Round1 Architecture Conclusion

Round1 在 clean synthetic data 上得到的主结论与 Round2 基本一致：

1. query/key vector 比 ordinary hidden router 更容易形成 slot-level specialization；
2. expert 输入使用 full token vector / `attention_output + residual` 比 pure attention output 更稳；
3. 自然训练不足以得到足够硬的 specialization，需要额外 routing objective。

主要差异是：Round1 中 `k/head` 在简单 slot feature 上表现更像明确最优。
