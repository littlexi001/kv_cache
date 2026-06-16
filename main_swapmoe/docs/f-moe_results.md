# 华为指标

### 模型架构

Qwen 0\.6B改的MOE

1. `old_baseline`：
普通 shared MoE。每层有一个 always\-on shared expert，size `1536`；另外有 4 个 routed expert，每个 size `1536`，每个 token 选 1 个。router 用普通 `mlp_input / x_moe`，不做 SVD 分段。

2. `old_svroute_fullinput`：
老 subspace MoE。一个 common expert size `1536`；然后把 `o_proj` 的 SVD 频段分成 3 组\(0\-2% common 部分部分发,2\-20%,20\-50%，rest\)，每组有 4 个 expert，每个 expert size `512`，每组 top1。router 用旧版 hidden 表征，也就是 `mlp_input / x_moe` 侧；expert 输入是 full hidden，不是 component。

3. `attnpreo_svroute_fullinput`：
修正后的 subspace MoE。：common `1536`，三组 routed\(0\-2% common 部分部分发,2\-20%,20\-50%，rest\)，每组 4 expert，每个 `512`。route 用 `attn_pre_o @ V_band`，并乘 singular value。

4. `fmoe_v1`：
common 频段是 top `0%-1%`，直接进 common expert，size `1536`；`1%-10%` 是 middle group，4 个 expert，每个 `768`，top1；`10%-100%` 是 tail group，16 个 expert，每个 `768`，top1。route 用 `attn_pre_o @ V_band`，并乘 singular value。

5. `fmoe_v2`：
和 v1 一样，只是 common 扩大到 `0%-2%`，middle 是 `2%-10%`，tail 是 `10%-100%`。expert 数和 size 不变。

6. `fmoe_v3`：【middle 就是第一组 expert 】
频段和 v1 一样，`0%-1% / 1%-10% / 10%-100%`，但三个 active 部分的 size 全改成 `1024`。所以 common/middle/tail 都是 `1024`，active 仍然是 `3072`。

7. `fmoe_v4`：
频段和 v1 一样，size 也和 v1 一样，但是 tail expert 数从 16 增加到 32。也就是为了测试“长尾更稀疏，给更多小 expert”这件事。

8. `fmoe_v5`：
common 扩大到 `0%-5%`，middle 是 `5%-20%`，tail 是 `20%-100%`。expert 数和 size 跟 v1 一样。这个是测试“分段位置不同”。

**频段维度大概是多少**

这里百分比是按 `o_proj` 的奇异方向排序来的。虽然 `attn_pre_o` 是 2048 维，但 `o_proj` 只有 1024 个有效 singular directions，所以分段 rank 大概按 1024 算：



```Plain Text
fmoe_v1_common1_mid1_10_tail16
common: 0-1%, middle: 1-10%, tail: 10-100%
size: common 1536, middle 4x768, tail 16x768

fmoe_v2_common2_mid2_10_tail16
common: 0-2%, middle: 2-10%, tail: 10-100%
size: common 1536, middle 4x768, tail 16x768

fmoe_v3_common1_mid1_10_tail16_c1024
common: 0-1%, middle: 1-10%, tail: 10-100%
size: common 1024, middle 4x1024, tail 16x1024

fmoe_v4_common1_mid1_10_tail32
common: 0-1%, middle: 1-10%, tail: 10-100%
size: common 1536, middle 4x768, tail 32x768

fmoe_v5_common5_mid5_20_tail16
common: 0-5%, middle: 5-20%, tail: 20-100%
size: common 1536, middle 4x768, tail 16x768
```





### 下游任务性能

具体的位置在/mnt/workspace/Hrj/current\_downstream\_eval\_results\_harness\_oldstyle/run\_all\.nohup\.log

old\_baseline                 avg 0\.3803, PPL 158\.88

attnpreo\_svroute\_fullinput   avg 0\.3766, PPL 203\.78

fmoe\_v1                      avg 0\.3763, PPL 211\.35

fmoe\_v2                      avg 0\.3736, PPL 223\.78

fmoe\_v4                      avg 0\.3709, PPL 242\.41

fmoe\_v3                      avg 0\.3708, PPL 248\.95

old\_svroute\_fullinput        avg 0\.3705, PPL 194\.68

fmoe\_v5                      avg 0\.3704, PPL 229\.95

### 分发稳定性

1. 纯计算expert swap 计算一个序列中总共换了多少次expert

具体的位置在这个文件夹里面，但是东西有点太多了：/mnt/workspace/Hrj/current\_analysis\_outputs/continuity
对应的代码是：/mnt/workspace/Hrj/scripts\_hyper/hrj\_continuity\_bands\_current\.py

然后这的测试方式是给定32768 tokens个token来看

attnpreo low\_mid   0\.330

fmoe\_v2 middle     0\.334

fmoe\_v1 middle     0\.337

fmoe\_v4 middle     0\.341

fmoe\_v3 middle     0\.342

fmoe\_v5 middle     0\.349

attnpreo mid       0\.351

attnpreo tail      0\.358

old\_baseline       0\.472

fmoe\_v2 tail       0\.516

fmoe\_v3 tail       0\.520

fmoe\_v1 tail       0\.521

fmoe\_v5 tail       0\.525

fmoe\_v4 tail       0\.567

2. 类似Oracle MOE 设置一个具体的MB 看要swap 进出多少次

代码/mnt/workspace/Hrj/scripts\_hyper/run\_expert\_swap\_cost\_current\.sh

文件：/mnt/workspace/Hrj/current\_analysis\_outputs/expert\_swap\_cost

对于不同的group 怎么塞，方式是

```Plain Text
baseline routed expert size = 1536
memory budget = 可以塞 2 个 baseline routed expert
              = 2 * 1536 = 3072 units
```

我们按每个 group 的 expert size 来决定每组能缓存几个 expert。

规则：

1. 每个 routed group 至少尽量分到 1 个 cache slot。

2. 如果还有 budget，希望每组能到 2 个 slot。

3. 如果 budget 不够所有 group 都 2 个：

    - 先保证大 expert 的 group；

    - 如果大 expert 只能塞 1 个，剩余全给小 expert group。

4. 最后得到每个模型的 per\-group cache slots。

5. 之后每个 group 独立做 LRU cache。



![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=ZWQ3YTIwMjBkMzI4NjQ0MGM1ZDA2OThmZDU5ZjllZTJfYjQ4Yzg5M2ZkMGE4MjU2ZTZmMzVkYzFiMWFlYzgwZGZfSUQ6NzY1MTgwNDkyNDA3NDAwMzQxMF8xNzgxNTk4MDMxOjE3ODE2ODQ0MzFfVjM)

### Loss

代码：/mnt/workspace/Hrj/scripts\_hyper/hrj\_loss\_curves\_current\.py
图的文件夹：/mnt/workspace/Hrj/current\_analysis\_outputs/loss


![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=NzZjMTNkMjhjNWQyNWNhOTMzYjQ3OWFkZTk0YmEyOWZfYzEzMjE3MDYxNDc3ZWMxNDY0YjAwNTliYjFmMWI5ZDZfSUQ6NzY1MTY4NjEyNDk1MTI0Mzk3Ml8xNzgxNTk4MDMxOjE3ODE2ODQ0MzFfVjM)

### 预测性

代码：/mnt/workspace/Hrj/scripts\_hyper/run\_predictability\_current\_oldstyle\_full\.sh/
/mnt/workspace/Hrj/scripts\_hyper/hrj\_predictability\_oneclick\_current\_oldstyle\.py

对应的结果：/mnt/workspace/Hrj/predictability\_outputs\_current\_oldstyle\_full

然后总结的结果在这里：/mnt/workspace/Hrj/predictability\_outputs\_current\_oldstyle\_top2\_full/\_summary/predictability\_summary\_all\.csv

【然后一些变式的值低是因为他的expert数量太多了导致的捏】按 `sametoken upper` 从高到低排，核心表是：

最短结论：`old/full_mlp_input` 最高；FMoE 的 `middle` 紧随其后；FMoE 的 `tail` 绝对值低，但因为随机基线低很多，所以仍然显著可预测。


Rank	Model	Feature	Group	E	Random	Same-token Upper	Cross-token Diag
1	old_svroute_fullinput	full_mlp_input	2	4	0.25	0.6838	0.5205
2	old_svroute_fullinput	full_mlp_input	1	4	0.25	0.6766	0.5424
3	old_baseline	full_mlp_input	0	4	0.25	0.6744	0.5437
4	old_baseline	router	0	4	0.25	0.6706	0.5275
5	old_svroute_fullinput	full_mlp_input	0	4	0.25	0.6533	0.507
6	old_svroute_fullinput	router	2	4	0.25	0.6091	0.5328
7	fmoe_v3	full_mlp_input	middle	4	0.25	0.5859	0.5953
8	fmoe_v4	full_mlp_input	middle	4	0.25	0.5823	0.6129
9	fmoe_v1	full_mlp_input	middle	4	0.25	0.5795	0.6155
10	old_svroute_fullinput	router	1	4	0.25	0.5733	0.533
11	fmoe_v2	full_mlp_input	middle	4	0.25	0.5697	0.5984
12	fmoe_v5	full_mlp_input	middle	4	0.25	0.5647	0.5117
13	old_svroute_fullinput	router	0	4	0.25	0.5482	0.541
14	attnpreo_svroute_fullinput	full_mlp_input	low_mid	4	0.25	0.5437	0.566
15	attnpreo_svroute_fullinput	full_mlp_input	mid	4	0.25	0.5206	0.3141
16	attnpreo_svroute_fullinput	full_mlp_input	tail	4	0.25	0.511	0.5502
17	fmoe_v4	router	middle	4	0.25	0.5096	0.5939
18	fmoe_v3	router	middle	4	0.25	0.5086	0.5896
19	fmoe_v1	router	middle	4	0.25	0.5065	0.5941
20	fmoe_v5	router	middle	4	0.25	0.5064	0.5908
21	attnpreo_svroute_fullinput	router	low_mid	4	0.25	0.5031	0.5989
22	attnpreo_svroute_fullinput	router	tail	4	0.25	0.4919	0.5553
23	attnpreo_svroute_fullinput	router	mid	4	0.25	0.4879	0.5732
24	fmoe_v2	router	middle	4	0.25	0.4862	0.5673
25	fmoe_v2	full_mlp_input	tail	16	0.0625	0.3617	0.3944
26	fmoe_v3	full_mlp_input	tail	16	0.0625	0.357	0.2815
27	fmoe_v1	full_mlp_input	tail	16	0.0625	0.3536	0.3833
28	fmoe_v2	router	tail	16	0.0625	0.3514	0.352
29	fmoe_v3	router	tail	16	0.0625	0.3476	0.3929
30	fmoe_v5	full_mlp_input	tail	16	0.0625	0.3472	0.3728
31	fmoe_v1	router	tail	16	0.0625	0.3417	0.3777
32	fmoe_v5	router	tail	16	0.0625	0.3286	0.3368
33	fmoe_v4	full_mlp_input	tail	32	0.0313	0.2925	0.3227
34	fmoe_v4	router	tail	32	0.0313	0.2736	0.3293
