# Qwen3-0.6B：16 路 PE 预训练实验包

当前默认方式是真正的从头预训练：程序读取
`/mnt/workspace/Qwen3-0.6B` 中的模型结构和 tokenizer，但不读取已有的
`model.safetensors` 权重。16 个条件使用同一个随机种子初始化，每个条件
独占一台 8 卡阿里云交互式实例。

## 固定训练协议

- 每个任务训练 100B token；16 个任务总计最多处理 1.6T token；
- global batch size：256 条序列；
- 序列长度：8,192 token；
- 每卡 micro-batch 1，8 卡，梯度累积 32；
- 学习率峰值 `1e-4`，warmup 500 步，之后 cosine decay；
- 在约 0.1B、1B、10B、25B、50B、75B、100B token 自动保存并评测；
- Trainer 实时写入 TensorBoard，默认端口 6006。

每一步处理：

```text
256 × 8192 = 2,097,152 token
```

100B 对应 47,684 步，实际最终处理 100,000,595,968 token，多出的比例约
0.0006%。

## 每台机器安装

```bash
unzip qwen3_06b_aliyun_pe_pretraining_iclr27.zip
cd qwen3_06b_aliyun_pe_pretraining_iclr27
bash scripts/install.sh
cp configs/experiment.env.example .env
```

## 启动 16 个任务

在编号为 `N` 的机器运行：

```bash
bash scripts/run_pretrain_worker.sh N  # N 为 0 到 15
```

脚本默认使用 `0,1,2,3,4,5,6,7` 八张卡，训练和 TensorBoard 都在后台
运行，网页或 SSH 断开不会停止。

正式启动前可以只检查分配和训练合同，不创建进程：

```bash
DRY_RUN=1 bash scripts/run_pretrain_worker.sh N
```

如果控制机能够免密 SSH 到全部实例，可以一次性派发：

```bash
cp configs/sixteen_hosts.example configs/sixteen_hosts.conf
# 填写 16 台机器
bash scripts/launch_sixteen_hosts.sh configs/sixteen_hosts.conf
```

查看状态和统一停止：

```bash
bash scripts/status_sixteen_hosts.sh configs/sixteen_hosts.conf
bash scripts/stop_sixteen_hosts.sh configs/sixteen_hosts.conf
```

TensorBoard 地址为 `http://<机器地址>:6006`。如果不希望公开端口，请使用
安全组限制或 SSH 端口转发。

新实验统一写入 `/mnt/workspace/pe_pretrain_100b_iclr27`，不会恢复以前
`pe_runs_iclr27` 中的续训练 checkpoint。

## 方法与文档

任务 0 是原生 RoPE 基线；其余条件包括全 NoPE、周期 NoPE、深层 NoPE、
高/中/低频消融、全局减速、平滑层×频率、远程位置压缩，以及新的
period-aware 和深层 phase-diverse 函数。任务映射位于
`configs/sixteen_machine_plan.json`，每个方法都有独立文档：
`docs/methods/00_*.md` 到 `15_*.md`。

完整说明：

- `docs/pretraining_100b_protocol.md`：100B 训练与一键运行；
- `docs/design.md`：研究假设和数学形式；
- `docs/experiment_design.md`：评价指标与通过/失败条件；
- `docs/visualization_results.md`：TensorBoard 和结果作图规范。

每个 checkpoint 会自动评测 DCLM PPL、受控 RULER 风格任务和 LongBench
子集，再继续训练。合并结果时会核对实际 token 数和 DCLM manifest 哈希。

如果想恢复以前的 checkpoint 续训练模式，可设置
`INITIALIZATION=checkpoint`；不要把它和从头预训练结果混在一起比较。
