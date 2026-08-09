# 完整层块 × KV Head × RoPE 频带表：可视化审计

## Plot contract

- 标题：完整层块 × KV Head × RoPE 频带扫描。
- 问题：同一 Head–频带干预在一个六层区域内是否具有平均正向或负向趋势？
- 数据：Qwen3-8B，6 条固定 RULER-32K 发现样本；L18–L35 的 1,152 个单层干预配置。
- 干预：在一个层、一个 KV Head 组、一个连续 8 维频带内，将 RoPE 旋转设为单位旋转。
- 指标：`100 × mean_l(score_intervention(l) - score_native)`，其中 `l` 遍历层块内六层。
- 单位：百分点（pp）；越大越好。
- 横轴：KV Head 组 G0–G7；Gg 对应 Query Heads Q(4g)–Q(4g+3)。
- 纵轴：F0–F7 到 F56–F63 的八个连续 RoPE 频带。
- 分面：L18–L23、L24–L29、L30–L35。
- 颜色：蓝色为平均提高，橙色为平均下降，灰色为接近 0；数值文本是主要编码。
- 允许结论：可以定位层区域、Head 组和频带的平均发现集趋势。
- 限制：不能代替独立 seeds；六层均值可能隐藏单层极值。

## Audit result

| 检查项 | 状态 | 证据或修复 |
|---|---|---|
| 完整性 | pass | 1,152 个单层配置均存在；192 个聚合单元，每格恰好 6 层。 |
| 指标定义与单位 | pass | 页面首屏在表格前定义均值公式含义与 pp。 |
| 轴与图例 | pass | 三个层块、八个频带、八个 Head 组和颜色方向均可见。 |
| 可见数字 | pass | 所有 192 格直接显示一位小数；点选后显示两位小数、最佳/最差层和平均 Gold NLL 改善。 |
| 单位兼容 | pass | 表格只编码官方分数变化（pp）；Gold NLL 仅出现在点选详情中，不与 pp 共用颜色或轴。 |
| 交互 | pass | 点选 L24–L29/G3/F40–F47 后显示 `+2.50 pp`、最佳 L25 `+10.83 pp`、最差 L24 `0.00 pp`。 |
| 页面脚本 | pass | 192 个按钮、3 个分面；浏览器控制台无 error/warn。 |
| 响应式布局 | pass | 1024 和 736 px 无横向溢出；360 px 时仅每张表内部横向滚动，页面本身不溢出；文字为 14 px。 |
| HTTP | pass | `http://127.0.0.1:4196/rope-head-frequency-complete-preview.html` 返回 200。 |
| 独立审阅 | 未使用 | 当前任务的工具策略不允许启动子代理；由主代理按同一清单完成审计。 |

## Artifacts

- 聚合数据：`outputs/dense_layer_band_sweep_20260806/complete_layer_block_heatmap.json`
- 完整页面截图：`outputs/dense_layer_band_sweep_20260806/complete_heatmap_full.png`
- 单元格选择截图：`outputs/dense_layer_band_sweep_20260806/complete_heatmap_selected_cell.png`
- 内联页面：`C:/Users/27814/.codex/visualizations/2026/07/18/019f7582-4120-7bc0-8df4-ec00ebf1ab44/rope-head-frequency-complete.html`
- 本地检查服务器：`scripts/serve_complete_heatmap.ps1`

## 非聚合逐层表

- 标题：逐层 RoPE Head × Frequency 完整扫描。
- 状态：pass。
- 指标：`100 × (单层干预官方分 - 原生 RoPE 官方分)`。
- 单位：百分点（pp），越大越好。
- 数据：L18–L35、8 个 KV Head 组、8 个频带，共 1,152 个单层配置；每个配置使用 6 条发现样本。
- 聚合：无；每张表严格对应一个层。
- 轴：横向 G0–G7，纵向 F0–F7 至 F56–F63；18 张表分别对应 L18–L35。
- 图例：蓝色提高、橙色下降、灰色官方分数不变；格内直接显示 pp。
- 可见数字：1,152 个格均显示一位小数；点击后显示两位小数、干预后官方分、Gold NLL 改善和样本提高/下降数。
- 完整性检查：18 个 grid、1,152 个按钮，无缺失格。
- 脚本检查：浏览器无 error/warn；L33/G3/F40–F47 点选结果为 `-5.00 pp`、Gold NLL 改善 `+0.0485`。
- 响应式检查：1024/736 px 页面无横向溢出；360 px 时仅表格内部横向滚动；文字 14 px。
- HTTP：`http://127.0.0.1:4196/rope-head-frequency-single-layer-preview.html` 返回 200。
- 截图：`outputs/dense_layer_band_sweep_20260806/single_layer_heatmap_header.png` 与 `single_layer_heatmap_selected_cell.png`。
- 允许结论：可以读取每个具体层、Head 组、频带的发现集单层效应。
- 剩余不确定性：6 样本官方分离散，0.0 不代表 Gold NLL 或输出概率完全不变。
- 独立审阅：当前工具策略不允许启动子代理，由主代理完成同一审计清单。
