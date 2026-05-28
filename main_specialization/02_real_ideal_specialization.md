# Real-Corpus Ideal Specialization

理想 specialization 定义：分到同一 expert 的 token，其 next-token logits 分布应当相似。

当前已有观察：

1. 同一个 expert 中的 token / context，其表征 cosine similarity 高，约为 0.97；
2. 不同 expert 中的 token / context，表征 cosine similarity 低，约为 0.20；
3. 也有反例：表征 cosine similarity 高，但一起学的效果很差，反例比例约为 10%；
4. 线性分发无法支持 ground truth feature 分发。
