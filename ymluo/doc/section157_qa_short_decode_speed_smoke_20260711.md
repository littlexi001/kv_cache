# Section 157: QA short-decode speed smoke

日期: 2026-07-11

## 背景

当前很多 sparse KV 配置的 attention/KV 理论压缩已经足够高, 但端到端 online speed 不总是兑现。一个直接原因是 LongBench QA 的生成输出可能过长: 即使 KV gather/attention 更快, decode token 数一长, 端到端时间仍然被生成阶段吃掉。

从日志看, 一些错误输出本身也是解释性长句或“not enough information”式长回答。对于 `qa_f1` 任务, 标准答案通常是短实体、数字、短短语; 因此短解码可能同时改善速度和格式, 但也可能截断需要解释的答案。

## 实验设计

本轮不改变 v300 的检索/风险路由主线, 只加入 QA 短解码:

| Version | 设计 | 目的 |
|---|---|---|
| v316 | balanced short decode | 温和限制输出长度, 看是否无损提速 |
| v317 | aggressive short decode | 更激进限制输出长度, 看速度上界和分数损失 |

任务:

```text
narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique
```

样本数:

```text
M20 per task
```

## 关键设置

v316:

- narrativeqa: 32 tokens
- qasper: 64 tokens
- multifieldqa_en: 48 tokens
- hotpotqa / 2wikimqa / musique: 32 tokens

v317:

- narrativeqa: 24 tokens
- qasper: 48 tokens
- multifieldqa_en: 32 tokens
- hotpotqa / 2wikimqa / musique: 24 tokens

## 判据

如果 v316 相对 v300 的 same-sample score 基本不降, 但 online speed 更高, 可以纳入 practical best, 因为这是独立于检索质量的端到端收益。

如果 v317 明显降分, 说明 aggressive cap 只能作为任务特定分支, 不能作为默认策略。

如果两者都无法提速, 说明当前端到端瓶颈不在 QA 输出长度, 下一步应回到 KV gather/attention kernel 或风险路由减少 full fallback。
