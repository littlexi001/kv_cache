# Evaluation Metrics

1. **NTP Acc：** NTP accuracy / loss 不显著变差，最好在困难样本或长程依赖样本上有收益；
2. **Feature selectivity：** expert assignment 与 synthetic ground truth 或真实语料 proxy feature 显著对齐；
3. **Deployability：** routing signal 能够在需要的位置提前产生，并能服务 KV cache reverse indexing 或其他下游系统目标。
