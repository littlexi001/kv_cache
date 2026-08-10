# Query 二阶矩收缩系数敏感性：实验设计

## 设置

- 模型：`Meta-Llama-3.1-8B-Instruct`、`Qwen3-4B-Instruct-2507`。
- 文本：20 Newsgroups 中相互独立的体育与医学流，每条历史 32,000 token。
- 层：每个模型 5 个从早到晚均匀分布的层；全部 KV head 与映射后的 Query head。
- Query：采集 64 个 decode 位置；prompt 最后 8 个 Query 只用于校准，不从前 8 个
  decode 位置偷取校准信息。
- 固定量：模型、token 流、随机种子、Key 样本、位宽预算、量化、候选比例及 exact
  attention consumer。
- 唯一变量：`lambda in {0,0.25,0.5,0.75,0.9}`。

## 配对与统计

严格配对键为“模型/主题、layer、held-out step、KV head、Query head、候选比例”。
任意系数缺少一个键都拒绝汇总。均值之外，先在每个“模型/主题/layer”内计算
相对 `lambda=0.75` 的差，再对这些 cluster 做 10,000 次 bootstrap。

预注册的稳定性阈值是：每个候选比例下，`0.75` 相对五点最优值的 attention mass
损失不超过 1 个百分点、top-2% recall 损失不超过 2 个百分点、score RMSE 不超过
最优值的 1.10 倍。这些阈值只判断“是否稳定”，不允许据此改主方法。

## 运行与产物

```bash
ROOT=/home/fdong/qksieve_iclr2027 \
GPUS=0,1,2,3 \
bash scripts/launch_qksieve_shrinkage_sensitivity_20260810.sh
```

- 原始 trace：`RUN_ROOT/traces/*.pt`
- 每个系数逐条件结果：`RUN_ROOT/analysis/*/lambda_*/per_head.csv`
- 严格汇总：`RUN_ROOT/summary.json`
- 日志与环境清单：`RUN_ROOT/logs/`、`RUN_ROOT/manifest.txt`

已知局限：它直接检验 selector 数值机制，不替代 LongBench、RULER 或 PPL；当前
网格覆盖 32K，若 64K/128K 正式质量出现异常，需要增加长度对照但不能修改冻结值。
