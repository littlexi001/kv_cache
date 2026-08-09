# Qwen3-0.6B 全部 448 个 Query Head 功能目录

每个 head 都给出一个主签名，但请优先看“保守标签”和“置信度”。“弱→某类”表示该 head 没通过单功能专门化阈值，只是在九类探针中该类相对最强。局部因果一致指删除同类 attention links 后，最大输出变化类别是否与主签名一致。

## Layer 0

| Head | 保守标签 | 完整主签名 | 多标签 | 稳定性 | 局部因果 | 外部检索（实测） | 自然任务安全算子 | 置信度 |
|---|---|---|---|---|---|---|---|---|
| L00H00 | 标点与边界 | 标点与边界 | 标点与边界 | stable_bias | ≠标点与边界 | 位置 | QK top-block (0.86) | 高 |
| L00H01 | 当前 token/self | 当前 token/self | 当前 token/self | stable_bias | ≠当前 token/self | 位置 | QK top-block (0.98) | 高 |
| L00H02 | 前一 token | 前一 token | 前一 token;当前 token/self | stable_bias | ≠前一 token | 位置 | QK top-block (0.58) | 高 |
| L00H03 | 当前 token/self | 当前 token/self | 当前 token/self | stable_bias | ≠当前 token/self | 位置 | streaming (1.00) | 高 |
| L00H04 | 标点与边界 | 标点与边界 | 标点与边界 | stable_bias | ≠前一 token | 位置 | streaming (0.94) | 高 |
| L00H05 | 当前 token/self | 当前 token/self | 当前 token/self | stable_bias | ≠当前 token/self | 位置 | streaming (0.81) | 高 |
| L00H06 | 标点与边界 | 标点与边界 | 标点与边界;局部近期上下文;前一 token | stable_bias | ≠语义证据 | 位置 | 全量 (0.67) | 高 |
| L00H07 | 当前 token/self | 当前 token/self | 当前 token/self | stable_bias | ≠当前 token/self | 重复片段 | QK top-block (0.91) | 高 |
| L00H08 | 混合/通用 | 弱→语义证据 | 局部近期上下文 | stable_bias | ≠语义证据 | 位置 | QK top-block (0.86) | 中 |
| L00H09 | 当前 token/self | 当前 token/self | 当前 token/self | stable_bias | ≠当前 token/self | 重复片段 | streaming (0.98) | 高 |
| L00H10 | 局部近期上下文 | 局部近期上下文 | 局部近期上下文;前一 token | context_sensitive | ≠前一 token | 位置 | streaming (0.97) | 中 |
| L00H11 | 混合/通用 | 弱→语义证据 | 句法依赖 | stable_bias | ≠语义证据 | 位置 | 全量 (0.95) | 中 |
| L00H12 | 当前 token/self | 当前 token/self | 当前 token/self | stable_bias | ≠当前 token/self | 位置 | streaming (1.00) | 高 |
| L00H13 | 标点与边界 | 标点与边界 | 标点与边界;局部近期上下文 | stable_bias | ≠语义证据 | 位置 | QK top-block (0.53) | 高 |
| L00H14 | 当前 token/self | 当前 token/self | 当前 token/self | stable_bias | ≠当前 token/self | 位置 | QK top-block (0.94) | 高 |
| L00H15 | 当前 token/self | 当前 token/self | 当前 token/self | stable_bias | ≠当前 token/self | 重复片段 | QK top-block (0.98) | 高 |

## Layer 1

| Head | 保守标签 | 完整主签名 | 多标签 | 稳定性 | 局部因果 | 外部检索（实测） | 自然任务安全算子 | 置信度 |
|---|---|---|---|---|---|---|---|---|
| L01H00 | 混合/通用 | 弱→标点与边界 | 前一 token | stable_bias | ≠前一 token | 位置 | QK top-block (0.62) | 低 |
| L01H01 | 前一 token | 前一 token | 前一 token;句法依赖;当前 token/self | intermediate | ≠前一 token | 位置 | QK top-block (0.97) | 中 |
| L01H02 | 前一 token | 前一 token | 前一 token;局部近期上下文 | stable_bias | ≠前一 token | 位置 | streaming (1.00) | 高 |
| L01H03 | 前一 token | 前一 token | 前一 token;当前 token/self | stable_bias | ≠前一 token | 位置 | streaming (0.53) | 高 |
| L01H04 | 局部近期上下文 | 局部近期上下文 | 局部近期上下文 | stable_bias | ≠局部近期上下文 | 位置 | streaming (1.00) | 高 |
| L01H05 | 混合/通用 | 弱→标点与边界 | 局部近期上下文;前一 token;当前 token/self | stable_bias | ≠语义证据 | 位置 | 全量 (0.98) | 低 |
| L01H06 | 混合/通用 | 弱→标点与边界 | 当前 token/self;结构锚点 | stable_bias | ≠结构锚点 | 重复片段 | 全量 (0.89) | 低 |
| L01H07 | 当前 token/self | 当前 token/self | 当前 token/self | stable_bias | ≠当前 token/self | 位置 | QK top-block (0.80) | 高 |
| L01H08 | 同词回指/复制 | 同词回指/复制 | 同词回指/复制;标点与边界;结构锚点 | stable_bias | ≠同词回指/复制 | 位置 | QK top-block (0.89) | 高 |
| L01H09 | 标点与边界 | 标点与边界 | 标点与边界;前一 token;当前 token/self | stable_bias | ≠标点与边界 | 位置 | streaming (0.56) | 高 |
| L01H10 | 混合/通用 | 弱→标点与边界 | 同词回指/复制;局部近期上下文 | stable_bias | ≠同词回指/复制 | 位置 | 全量 (0.50) | 中 |
| L01H11 | 标点与边界 | 标点与边界 | 标点与边界;局部近期上下文;结构锚点 | stable_bias | ≠标点与边界 | 位置 | streaming (0.91) | 高 |
| L01H12 | 标点与边界 | 标点与边界 | 标点与边界 | stable_bias | ≠标点与边界 | 位置 | 全量 (0.97) | 高 |
| L01H13 | 混合/通用 | 弱→标点与边界 | — | stable_bias | ≠标点与边界 | 位置 | 全量 (0.97) | 中 |
| L01H14 | 前一 token | 前一 token | 前一 token;当前 token/self | stable_bias | ≠前一 token | 位置 | streaming (0.78) | 高 |
| L01H15 | 标点与边界 | 标点与边界 | 标点与边界;同词回指/复制;当前 token/self | intermediate | ≠同词回指/复制 | 重复片段 | QK top-block (0.56) | 中 |

## Layer 2

| Head | 保守标签 | 完整主签名 | 多标签 | 稳定性 | 局部因果 | 外部检索（实测） | 自然任务安全算子 | 置信度 |
|---|---|---|---|---|---|---|---|---|
| L02H00 | 局部近期上下文 | 局部近期上下文 | 局部近期上下文;前一 token;结构锚点 | intermediate | ≠前一 token | 位置 | streaming (0.98) | 中 |
| L02H01 | 局部近期上下文 | 局部近期上下文 | 局部近期上下文;前一 token | context_sensitive | ≠前一 token | 位置 | QK top-block (0.66) | 中 |
| L02H02 | 标点与边界 | 标点与边界 | 标点与边界;同词回指/复制;当前 token/self | stable_bias | ≠同词回指/复制 | 位置+重复片段 | 全量 (0.89) | 高 |
| L02H03 | 同词回指/复制 | 同词回指/复制 | 同词回指/复制;标点与边界;结构锚点 | stable_bias | ≠同词回指/复制 | 重复片段 | 全量 (0.89) | 高 |
| L02H04 | 当前 token/self | 当前 token/self | 当前 token/self | stable_bias | ≠当前 token/self | 重复片段 | 全量 (0.52) | 高 |
| L02H05 | 标点与边界 | 标点与边界 | 标点与边界;当前 token/self | stable_bias | ≠标点与边界 | 重复片段 | 全量 (0.88) | 高 |
| L02H06 | 局部近期上下文 | 局部近期上下文 | 局部近期上下文;结构锚点;句法依赖 | stable_bias | ≠句法依赖 | 位置 | streaming (0.98) | 中 |
| L02H07 | 标点与边界 | 标点与边界 | 标点与边界;句法依赖 | stable_bias | ≠标点与边界 | 位置 | QK top-block (0.56) | 高 |
| L02H08 | 混合/通用 | 弱→标点与边界 | 同词回指/复制;当前 token/self;结构锚点 | stable_bias | ≠当前 token/self | 重复片段 | 全量 (0.97) | 中 |
| L02H09 | 当前 token/self | 当前 token/self | 当前 token/self | stable_bias | ≠当前 token/self | 重复片段 | 全量 (0.81) | 高 |
| L02H10 | 标点与边界 | 标点与边界 | 标点与边界;同词回指/复制 | stable_bias | ≠同词回指/复制 | 重复片段 | 全量 (0.89) | 中 |
| L02H11 | 同词回指/复制 | 同词回指/复制 | 同词回指/复制;标点与边界;当前 token/self | stable_bias | ≠同词回指/复制 | 重复片段 | 全量 (0.62) | 高 |
| L02H12 | 前一 token | 前一 token | 前一 token;当前 token/self | stable_bias | ≠前一 token | 位置 | 全量 (0.48) | 高 |
| L02H13 | 前一 token | 前一 token | 前一 token;当前 token/self | stable_bias | ≠前一 token | 位置 | 全量 (0.77) | 高 |
| L02H14 | 标点与边界 | 标点与边界 | 标点与边界;同词回指/复制;句法依赖;结构锚点 | stable_bias | ≠同词回指/复制 | 位置 | 全量 (0.94) | 高 |
| L02H15 | 标点与边界 | 标点与边界 | 标点与边界;局部近期上下文;句法依赖 | stable_bias | ≠标点与边界 | 位置 | 全量 (0.98) | 高 |

## Layer 3

| Head | 保守标签 | 完整主签名 | 多标签 | 稳定性 | 局部因果 | 外部检索（实测） | 自然任务安全算子 | 置信度 |
|---|---|---|---|---|---|---|---|---|
| L03H00 | 混合/通用 | 弱→语义证据 | 结构锚点 | stable_bias | ≠语义证据 | 位置+重复片段 | 全量 (0.97) | 中 |
| L03H01 | 混合/通用 | 弱→结构锚点 | — | stable_bias | ≠标点与边界 | 位置 | 全量 (0.94) | 低 |
| L03H02 | 结构锚点 | 结构锚点 | 结构锚点 | intermediate | ≠局部近期上下文 | 位置 | QK top-block (0.53) | 中 |
| L03H03 | 结构锚点 | 结构锚点 | 结构锚点 | intermediate | ≠结构锚点 | 位置 | streaming (0.89) | 高 |
| L03H04 | 混合/通用 | 弱→序列起点/sink | — | stable_bias | ≠序列起点/sink | 位置 | QK top-block (0.89) | 中 |
| L03H05 | 混合/通用 | 弱→前一 token | — | stable_bias | ≠前一 token | 位置 | streaming (0.88) | 中 |
| L03H06 | 混合/通用 | 弱→序列起点/sink | — | stable_bias | ≠前一 token | 位置 | QK top-block (0.97) | 低 |
| L03H07 | 前一 token | 前一 token | 前一 token | stable_bias | ≠前一 token | 位置 | streaming (0.80) | 高 |
| L03H08 | 混合/通用 | 弱→序列起点/sink | — | stable_bias | ≠序列起点/sink | 位置 | QK top-block (0.66) | 中 |
| L03H09 | 混合/通用 | 弱→序列起点/sink | — | stable_bias | ≠序列起点/sink | 位置+重复片段 | streaming (0.75) | 中 |
| L03H10 | 混合/通用 | 弱→语义证据 | — | stable_bias | ≠结构锚点 | 位置 | 全量 (0.98) | 低 |
| L03H11 | 混合/通用 | 弱→同词回指/复制 | — | stable_bias | ≠同词回指/复制 | 位置+重复片段 | 全量 (0.95) | 中 |
| L03H12 | 混合/通用 | 弱→语义证据 | — | intermediate | ≠语义证据 | 位置 | 全量 (0.56) | 低 |
| L03H13 | 混合/通用 | 弱→同词回指/复制 | — | intermediate | ≠同词回指/复制 | 位置 | 全量 (0.81) | 低 |
| L03H14 | 混合/通用 | 弱→语义证据 | — | intermediate | ≠语义证据 | 位置 | QK top-block (0.50) | 低 |
| L03H15 | 混合/通用 | 弱→标点与边界 | — | stable_bias | ≠标点与边界 | 位置 | streaming (0.64) | 中 |

## Layer 4

| Head | 保守标签 | 完整主签名 | 多标签 | 稳定性 | 局部因果 | 外部检索（实测） | 自然任务安全算子 | 置信度 |
|---|---|---|---|---|---|---|---|---|
| L04H00 | 混合/通用 | 弱→标点与边界 | 局部近期上下文 | stable_bias | ≠语义证据 | 位置 | streaming (0.92) | 中 |
| L04H01 | 前一 token | 前一 token | 前一 token;局部近期上下文;当前 token/self | stable_bias | ≠前一 token | 位置 | streaming (1.00) | 高 |
| L04H02 | 混合/通用 | 弱→序列起点/sink | — | stable_bias | ≠序列起点/sink | 位置 | QK top-block (0.78) | 中 |
| L04H03 | 混合/通用 | 弱→序列起点/sink | — | stable_bias | ≠序列起点/sink | 位置 | QK top-block (0.55) | 中 |
| L04H04 | 当前 token/self | 当前 token/self | 当前 token/self | stable_bias | ≠当前 token/self | 位置 | streaming (1.00) | 高 |
| L04H05 | 前一 token | 前一 token | 前一 token | intermediate | ≠前一 token | 位置 | QK top-block (0.53) | 高 |
| L04H06 | 混合/通用 | 弱→前一 token | — | intermediate | ≠前一 token | 位置 | streaming (1.00) | 低 |
| L04H07 | 混合/通用 | 弱→序列起点/sink | — | intermediate | ≠前一 token | 位置 | streaming (1.00) | 低 |
| L04H08 | 混合/通用 | 弱→序列起点/sink | — | stable_bias | ≠当前 token/self | 位置 | streaming (0.56) | 低 |
| L04H09 | 混合/通用 | 弱→序列起点/sink | — | stable_bias | ≠序列起点/sink | 位置 | 全量 (0.95) | 中 |
| L04H10 | 当前 token/self | 当前 token/self | 当前 token/self | stable_bias | ≠当前 token/self | 重复片段 | 全量 (0.81) | 高 |
| L04H11 | 混合/通用 | 弱→语义证据 | — | intermediate | ≠语义证据 | 位置 | 全量 (0.73) | 低 |
| L04H12 | 混合/通用 | 弱→同词回指/复制 | — | stable_bias | ≠同词回指/复制 | 位置+重复片段 | 全量 (0.97) | 中 |
| L04H13 | 同词回指/复制 | 同词回指/复制 | 同词回指/复制 | stable_bias | ≠同词回指/复制 | 重复片段 | 全量 (0.64) | 高 |
| L04H14 | 前一 token | 前一 token | 前一 token | stable_bias | ≠前一 token | 位置 | streaming (0.98) | 高 |
| L04H15 | 当前 token/self | 当前 token/self | 当前 token/self | stable_bias | ≠当前 token/self | 位置 | streaming (0.81) | 高 |

## Layer 5

| Head | 保守标签 | 完整主签名 | 多标签 | 稳定性 | 局部因果 | 外部检索（实测） | 自然任务安全算子 | 置信度 |
|---|---|---|---|---|---|---|---|---|
| L05H00 | 序列起点/sink | 序列起点/sink | 序列起点/sink | stable_bias | ≠序列起点/sink | 位置 | streaming (1.00) | 高 |
| L05H01 | 当前 token/self | 当前 token/self | 当前 token/self | stable_bias | ≠当前 token/self | 位置 | streaming (0.98) | 高 |
| L05H02 | 混合/通用 | 弱→序列起点/sink | — | stable_bias | ≠序列起点/sink | 位置 | QK top-block (0.95) | 中 |
| L05H03 | 混合/通用 | 弱→序列起点/sink | — | stable_bias | ≠序列起点/sink | 位置 | QK top-block (0.95) | 中 |
| L05H04 | 混合/通用 | 弱→标点与边界 | 句法依赖 | stable_bias | ≠语义证据 | 位置 | streaming (0.84) | 低 |
| L05H05 | 混合/通用 | 弱→标点与边界 | — | stable_bias | ≠语义证据 | 重复片段 | 全量 (0.97) | 低 |
| L05H06 | 混合/通用 | 弱→序列起点/sink | — | stable_bias | ≠序列起点/sink | 位置 | streaming (0.91) | 中 |
| L05H07 | 混合/通用 | 弱→序列起点/sink | — | context_sensitive | ≠前一 token | 位置 | streaming (1.00) | 低 |
| L05H08 | 混合/通用 | 弱→前一 token | — | intermediate | ≠前一 token | 位置 | streaming (0.98) | 低 |
| L05H09 | 前一 token | 前一 token | 前一 token | stable_bias | ≠前一 token | 位置 | streaming (0.95) | 高 |
| L05H10 | 混合/通用 | 弱→同词回指/复制 | — | intermediate | ≠同词回指/复制 | 位置 | streaming (0.70) | 低 |
| L05H11 | 句法依赖 | 句法依赖 | 句法依赖 | context_sensitive | ≠句法依赖 | 位置 | streaming (0.98) | 中 |
| L05H12 | 混合/通用 | 弱→序列起点/sink | — | context_sensitive | ≠局部近期上下文 | 位置 | streaming (1.00) | 低 |
| L05H13 | 混合/通用 | 弱→序列起点/sink | — | context_sensitive | ≠局部近期上下文 | 位置 | streaming (1.00) | 低 |
| L05H14 | 混合/通用 | 弱→当前 token/self | — | stable_bias | ≠前一 token | 位置 | streaming (0.98) | 低 |
| L05H15 | 当前 token/self | 当前 token/self | 当前 token/self | stable_bias | ≠当前 token/self | 位置 | streaming (1.00) | 高 |

## Layer 6

| Head | 保守标签 | 完整主签名 | 多标签 | 稳定性 | 局部因果 | 外部检索（实测） | 自然任务安全算子 | 置信度 |
|---|---|---|---|---|---|---|---|---|
| L06H00 | 混合/通用 | 弱→句法依赖 | — | intermediate | ≠句法依赖 | 位置 | streaming (0.97) | 低 |
| L06H01 | 句法依赖 | 句法依赖 | 句法依赖 | context_sensitive | ≠句法依赖 | 位置 | streaming (0.77) | 中 |
| L06H02 | 混合/通用 | 弱→前一 token | — | intermediate | ≠前一 token | 位置 | streaming (0.66) | 低 |
| L06H03 | 混合/通用 | 弱→前一 token | — | intermediate | ≠前一 token | 位置 | QK top-block (0.86) | 低 |
| L06H04 | 结构锚点 | 结构锚点 | 结构锚点 | intermediate | ≠结构锚点 | 位置 | streaming (0.62) | 中 |
| L06H05 | 结构锚点 | 结构锚点 | 结构锚点 | context_sensitive | ≠结构锚点 | 位置 | streaming (0.95) | 中 |
| L06H06 | 同词回指/复制 | 同词回指/复制 | 同词回指/复制 | stable_bias | ≠同词回指/复制 | 重复片段 | 全量 (0.69) | 高 |
| L06H07 | 当前 token/self | 当前 token/self | 当前 token/self | stable_bias | ≠当前 token/self | 重复片段 | streaming (0.95) | 高 |
| L06H08 | 标点与边界 | 标点与边界 | 标点与边界 | stable_bias | ≠标点与边界 | 位置 | QK top-block (0.89) | 高 |
| L06H09 | 混合/通用 | 弱→句法依赖 | — | context_sensitive | ≠句法依赖 | 位置 | QK top-block (0.53) | 低 |
| L06H10 | 同词回指/复制 | 同词回指/复制 | 同词回指/复制 | intermediate | ≠同词回指/复制 | 重复片段 | 全量 (0.91) | 中 |
| L06H11 | 混合/通用 | 弱→同词回指/复制 | — | context_sensitive | ≠结构锚点 | 位置+格式 | 全量 (0.84) | 低 |
| L06H12 | 同词回指/复制 | 同词回指/复制 | 同词回指/复制 | stable_bias | ≠同词回指/复制 | 位置 | streaming (0.70) | 高 |
| L06H13 | 同词回指/复制 | 同词回指/复制 | 同词回指/复制 | stable_bias | ≠同词回指/复制 | 重复片段 | 全量 (0.72) | 高 |
| L06H14 | 混合/通用 | 弱→标点与边界 | — | stable_bias | ≠同词回指/复制 | 位置+重复片段 | 全量 (0.86) | 低 |
| L06H15 | 句法依赖 | 句法依赖 | 句法依赖 | intermediate | ≠语义证据 | 位置 | QK top-block (0.67) | 中 |

## Layer 7

| Head | 保守标签 | 完整主签名 | 多标签 | 稳定性 | 局部因果 | 外部检索（实测） | 自然任务安全算子 | 置信度 |
|---|---|---|---|---|---|---|---|---|
| L07H00 | 当前 token/self | 当前 token/self | 当前 token/self | stable_bias | ≠当前 token/self | 位置+重复片段 | 全量 (0.89) | 高 |
| L07H01 | 混合/通用 | 弱→局部近期上下文 | — | intermediate | ≠局部近期上下文 | 位置 | streaming (0.97) | 低 |
| L07H02 | 前一 token | 前一 token | 前一 token | stable_bias | ≠前一 token | 位置 | QK top-block (0.50) | 高 |
| L07H03 | 前一 token | 前一 token | 前一 token | stable_bias | ≠前一 token | 位置 | streaming (0.66) | 高 |
| L07H04 | 混合/通用 | 弱→同词回指/复制 | 当前 token/self | stable_bias | ≠当前 token/self | 位置+重复片段 | 全量 (0.88) | 低 |
| L07H05 | 混合/通用 | 弱→语义证据 | — | stable_bias | ≠语义证据 | 位置 | QK top-block (0.50) | 中 |
| L07H06 | 混合/通用 | 弱→句法依赖 | — | context_sensitive | ≠同词回指/复制 | 位置 | 全量 (0.80) | 低 |
| L07H07 | 混合/通用 | 弱→标点与边界 | 句法依赖 | stable_bias | ≠标点与边界 | 位置 | QK top-block (0.62) | 中 |
| L07H08 | 混合/通用 | 弱→前一 token | — | intermediate | ≠前一 token | 位置 | streaming (0.98) | 低 |
| L07H09 | 混合/通用 | 弱→前一 token | — | intermediate | ≠前一 token | 位置 | streaming (0.95) | 低 |
| L07H10 | 前一 token | 前一 token | 前一 token;句法依赖 | stable_bias | ≠句法依赖 | 位置 | streaming (0.95) | 高 |
| L07H11 | 前一 token | 前一 token | 前一 token | intermediate | ≠前一 token | 位置 | streaming (0.80) | 中 |
| L07H12 | 混合/通用 | 弱→前一 token | — | intermediate | ≠前一 token | 位置 | streaming (0.66) | 低 |
| L07H13 | 当前 token/self | 当前 token/self | 当前 token/self | stable_bias | ≠当前 token/self | 位置 | QK top-block (0.98) | 高 |
| L07H14 | 混合/通用 | 弱→语义证据 | — | intermediate | ≠语义证据 | 位置 | streaming (0.88) | 低 |
| L07H15 | 混合/通用 | 弱→标点与边界 | 当前 token/self | stable_bias | ≠标点与边界 | 重复片段 | 全量 (0.89) | 中 |

## Layer 8

| Head | 保守标签 | 完整主签名 | 多标签 | 稳定性 | 局部因果 | 外部检索（实测） | 自然任务安全算子 | 置信度 |
|---|---|---|---|---|---|---|---|---|
| L08H00 | 结构锚点 | 结构锚点 | 结构锚点 | intermediate | ≠结构锚点 | 位置 | 全量 (0.45) | 中 |
| L08H01 | 混合/通用 | 弱→同词回指/复制 | — | intermediate | ≠同词回指/复制 | 重复片段 | 全量 (0.80) | 低 |
| L08H02 | 混合/通用 | 弱→标点与边界 | 结构锚点;句法依赖 | stable_bias | ≠标点与边界 | 位置 | 全量 (0.92) | 中 |
| L08H03 | 混合/通用 | 弱→标点与边界 | — | stable_bias | ≠标点与边界 | 位置 | 全量 (0.97) | 中 |
| L08H04 | 混合/通用 | 弱→序列起点/sink | — | stable_bias | ≠序列起点/sink | 位置 | 全量 (0.88) | 中 |
| L08H05 | 混合/通用 | 弱→序列起点/sink | — | stable_bias | ≠序列起点/sink | 位置 | 全量 (0.73) | 中 |
| L08H06 | 混合/通用 | 弱→同词回指/复制 | — | intermediate | ≠序列起点/sink | 位置+词法 | 全量 (0.50) | 低 |
| L08H07 | 混合/通用 | 弱→结构锚点 | — | stable_bias | ≠序列起点/sink | 位置 | 全量 (0.97) | 低 |
| L08H08 | 混合/通用 | 弱→序列起点/sink | — | stable_bias | ≠序列起点/sink | 位置 | 全量 (0.84) | 中 |
| L08H09 | 结构锚点 | 结构锚点 | 结构锚点 | context_sensitive | ≠结构锚点 | 位置 | streaming (0.61) | 中 |
| L08H10 | 混合/通用 | 弱→局部近期上下文 | — | intermediate | ≠局部近期上下文 | 位置 | 全量 (0.66) | 低 |
| L08H11 | 混合/通用 | 弱→标点与边界 | — | intermediate | ≠标点与边界 | 位置 | 全量 (0.75) | 低 |
| L08H12 | 混合/通用 | 弱→标点与边界 | — | intermediate | ≠局部近期上下文 | 位置 | 全量 (0.95) | 低 |
| L08H13 | 混合/通用 | 弱→序列起点/sink | — | stable_bias | ≠序列起点/sink | 位置 | 全量 (0.67) | 中 |
| L08H14 | 当前 token/self | 当前 token/self | 当前 token/self | stable_bias | ≠当前 token/self | 位置 | streaming (0.73) | 高 |
| L08H15 | 混合/通用 | 弱→标点与边界 | 结构锚点;句法依赖 | stable_bias | ≠标点与边界 | 位置 | QK top-block (0.59) | 中 |

## Layer 9

| Head | 保守标签 | 完整主签名 | 多标签 | 稳定性 | 局部因果 | 外部检索（实测） | 自然任务安全算子 | 置信度 |
|---|---|---|---|---|---|---|---|---|
| L09H00 | 结构锚点 | 结构锚点 | 结构锚点 | context_sensitive | ≠同词回指/复制 | 位置 | QK top-block (0.58) | 中 |
| L09H01 | 混合/通用 | 弱→标点与边界 | — | stable_bias | ≠局部近期上下文 | 位置 | streaming (0.98) | 低 |
| L09H02 | 混合/通用 | 弱→语义证据 | — | context_sensitive | ≠语义证据 | 位置 | 全量 (0.94) | 低 |
| L09H03 | 混合/通用 | 弱→标点与边界 | — | intermediate | ≠标点与边界 | 位置 | 全量 (0.75) | 低 |
| L09H04 | 混合/通用 | 弱→序列起点/sink | — | context_sensitive | ≠序列起点/sink | 位置 | QK top-block (0.50) | 低 |
| L09H05 | 混合/通用 | 弱→序列起点/sink | — | context_sensitive | ≠同词回指/复制 | 位置 | streaming (0.75) | 低 |
| L09H06 | 混合/通用 | 弱→标点与边界 | — | stable_bias | ≠标点与边界 | 位置 | 全量 (0.86) | 中 |
| L09H07 | 当前 token/self | 当前 token/self | 当前 token/self | stable_bias | ≠当前 token/self | 位置 | streaming (0.97) | 高 |
| L09H08 | 句法依赖 | 句法依赖 | 句法依赖;结构锚点 | context_sensitive | ≠句法依赖 | 位置 | QK top-block (0.64) | 中 |
| L09H09 | 当前 token/self | 当前 token/self | 当前 token/self | stable_bias | ≠当前 token/self | 位置 | QK top-block (0.78) | 高 |
| L09H10 | 前一 token | 前一 token | 前一 token | stable_bias | ≠前一 token | 位置 | streaming (0.86) | 高 |
| L09H11 | 前一 token | 前一 token | 前一 token | stable_bias | ≠前一 token | 位置 | QK top-block (0.73) | 高 |
| L09H12 | 同词回指/复制 | 同词回指/复制 | 同词回指/复制;结构锚点;当前 token/self | stable_bias | ≠同词回指/复制 | 位置+重复片段 | QK top-block (0.50) | 高 |
| L09H13 | 混合/通用 | 弱→同词回指/复制 | 当前 token/self | stable_bias | ≠同词回指/复制 | 位置+重复片段 | 全量 (0.81) | 中 |
| L09H14 | 混合/通用 | 弱→序列起点/sink | — | stable_bias | ≠语义证据 | 位置 | 全量 (0.69) | 低 |
| L09H15 | 混合/通用 | 弱→序列起点/sink | — | stable_bias | ≠标点与边界 | 位置+重复片段 | 全量 (0.89) | 低 |

## Layer 10

| Head | 保守标签 | 完整主签名 | 多标签 | 稳定性 | 局部因果 | 外部检索（实测） | 自然任务安全算子 | 置信度 |
|---|---|---|---|---|---|---|---|---|
| L10H00 | 混合/通用 | 弱→标点与边界 | 句法依赖 | stable_bias | ≠标点与边界 | 位置 | streaming (0.97) | 中 |
| L10H01 | 标点与边界 | 标点与边界 | 标点与边界;句法依赖 | stable_bias | ≠标点与边界 | 位置 | QK top-block (0.89) | 高 |
| L10H02 | 当前 token/self | 当前 token/self | 当前 token/self | stable_bias | ≠当前 token/self | 位置 | streaming (0.83) | 高 |
| L10H03 | 句法依赖 | 句法依赖 | 句法依赖 | stable_bias | ≠句法依赖 | 位置 | streaming (1.00) | 高 |
| L10H04 | 混合/通用 | 弱→前一 token | — | intermediate | ≠前一 token | 位置 | streaming (0.97) | 低 |
| L10H05 | 当前 token/self | 当前 token/self | 当前 token/self | stable_bias | ≠当前 token/self | 位置 | QK top-block (0.78) | 高 |
| L10H06 | 标点与边界 | 标点与边界 | 标点与边界;句法依赖 | stable_bias | ≠标点与边界 | 位置 | QK top-block (0.56) | 高 |
| L10H07 | 混合/通用 | 弱→标点与边界 | — | stable_bias | ≠标点与边界 | 位置 | 全量 (0.92) | 中 |
| L10H08 | 前一 token | 前一 token | 前一 token | stable_bias | ≠前一 token | 位置 | streaming (0.72) | 高 |
| L10H09 | 当前 token/self | 当前 token/self | 当前 token/self;前一 token | intermediate | ≠当前 token/self | 位置 | streaming (1.00) | 中 |
| L10H10 | 句法依赖 | 句法依赖 | 句法依赖 | stable_bias | ≠句法依赖 | 位置 | QK top-block (0.73) | 高 |
| L10H11 | 混合/通用 | 弱→标点与边界 | — | stable_bias | ≠标点与边界 | 位置 | 全量 (0.72) | 中 |
| L10H12 | 混合/通用 | 弱→前一 token | — | context_sensitive | ≠前一 token | 位置 | streaming (0.69) | 低 |
| L10H13 | 混合/通用 | 弱→前一 token | — | intermediate | ≠前一 token | 位置 | streaming (0.97) | 低 |
| L10H14 | 混合/通用 | 弱→局部近期上下文 | 前一 token | stable_bias | ≠语义证据 | 位置 | streaming (0.77) | 低 |
| L10H15 | 前一 token | 前一 token | 前一 token | intermediate | ≠前一 token | 位置 | streaming (0.97) | 中 |

## Layer 11

| Head | 保守标签 | 完整主签名 | 多标签 | 稳定性 | 局部因果 | 外部检索（实测） | 自然任务安全算子 | 置信度 |
|---|---|---|---|---|---|---|---|---|
| L11H00 | 混合/通用 | 弱→句法依赖 | — | context_sensitive | ≠同词回指/复制 | 位置 | QK top-block (0.56) | 低 |
| L11H01 | 句法依赖 | 句法依赖 | 句法依赖 | context_sensitive | ≠句法依赖 | 位置 | QK top-block (0.52) | 中 |
| L11H02 | 当前 token/self | 当前 token/self | 当前 token/self | stable_bias | ≠同词回指/复制 | 重复片段 | streaming (0.61) | 高 |
| L11H03 | 混合/通用 | 弱→同词回指/复制 | — | stable_bias | ≠同词回指/复制 | 位置+重复片段 | 全量 (0.92) | 中 |
| L11H04 | 混合/通用 | 弱→标点与边界 | 句法依赖 | stable_bias | ≠标点与边界 | 位置 | QK top-block (0.59) | 中 |
| L11H05 | 句法依赖 | 句法依赖 | 句法依赖 | intermediate | ≠句法依赖 | 位置 | streaming (0.48) | 中 |
| L11H06 | 混合/通用 | 弱→序列起点/sink | — | context_sensitive | ≠前一 token | 位置 | streaming (0.55) | 低 |
| L11H07 | 混合/通用 | 弱→序列起点/sink | — | context_sensitive | ≠标点与边界 | 位置 | QK top-block (0.47) | 低 |
| L11H08 | 同词回指/复制 | 同词回指/复制 | 同词回指/复制;当前 token/self;前一 token | intermediate | ≠同词回指/复制 | 位置 | 全量 (0.86) | 中 |
| L11H09 | 当前 token/self | 当前 token/self | 当前 token/self | stable_bias | ≠当前 token/self | 位置 | streaming (1.00) | 高 |
| L11H10 | 结构锚点 | 结构锚点 | 结构锚点 | intermediate | ≠同词回指/复制 | 位置 | 全量 (0.77) | 中 |
| L11H11 | 混合/通用 | 弱→同词回指/复制 | — | stable_bias | ≠同词回指/复制 | 位置+词法 | 全量 (0.78) | 中 |
| L11H12 | 同词回指/复制 | 同词回指/复制 | 同词回指/复制 | intermediate | ≠同词回指/复制 | 位置+重复片段 | 全量 (0.94) | 中 |
| L11H13 | 同词回指/复制 | 同词回指/复制 | 同词回指/复制 | stable_bias | ≠同词回指/复制 | 位置+重复片段 | 全量 (0.83) | 高 |
| L11H14 | 结构锚点 | 结构锚点 | 结构锚点;句法依赖;前一 token | stable_bias | ≠结构锚点 | 位置 | streaming (1.00) | 高 |
| L11H15 | 结构锚点 | 结构锚点 | 结构锚点 | intermediate | ≠前一 token | 位置 | streaming (1.00) | 中 |

## Layer 12

| Head | 保守标签 | 完整主签名 | 多标签 | 稳定性 | 局部因果 | 外部检索（实测） | 自然任务安全算子 | 置信度 |
|---|---|---|---|---|---|---|---|---|
| L12H00 | 混合/通用 | 弱→结构锚点 | — | stable_bias | ≠语义证据 | 位置 | QK top-block (0.50) | 低 |
| L12H01 | 混合/通用 | 弱→同词回指/复制 | — | stable_bias | ≠同词回指/复制 | 位置 | 全量 (0.50) | 中 |
| L12H02 | 混合/通用 | 弱→局部近期上下文 | — | context_sensitive | ≠标点与边界 | 位置 | 全量 (0.56) | 低 |
| L12H03 | 混合/通用 | 弱→同词回指/复制 | — | context_sensitive | ≠同词回指/复制 | 位置 | 全量 (0.59) | 低 |
| L12H04 | 句法依赖 | 句法依赖 | 句法依赖 | stable_bias | ≠句法依赖 | 位置 | streaming (0.83) | 高 |
| L12H05 | 前一 token | 前一 token | 前一 token;句法依赖 | stable_bias | ≠前一 token | 位置 | streaming (1.00) | 高 |
| L12H06 | 当前 token/self | 当前 token/self | 当前 token/self | stable_bias | ≠当前 token/self | 位置 | QK top-block (0.69) | 高 |
| L12H07 | 混合/通用 | 弱→语义证据 | — | stable_bias | ≠语义证据 | 位置 | 全量 (0.97) | 中 |
| L12H08 | 句法依赖 | 句法依赖 | 句法依赖 | intermediate | ≠同词回指/复制 | 位置 | 全量 (0.97) | 中 |
| L12H09 | 混合/通用 | 弱→标点与边界 | — | stable_bias | ≠标点与边界 | 位置 | 全量 (0.98) | 中 |
| L12H10 | 混合/通用 | 弱→结构锚点 | — | intermediate | ≠标点与边界 | 位置 | 全量 (0.80) | 低 |
| L12H11 | 混合/通用 | 弱→标点与边界 | 句法依赖 | stable_bias | ≠标点与边界 | 位置 | streaming (0.48) | 中 |
| L12H12 | 混合/通用 | 弱→句法依赖 | — | context_sensitive | ≠标点与边界 | 位置 | 全量 (0.45) | 低 |
| L12H13 | 混合/通用 | 弱→语义证据 | — | intermediate | ≠标点与边界 | 位置 | 全量 (0.69) | 低 |
| L12H14 | 混合/通用 | 弱→局部近期上下文 | — | context_sensitive | ≠标点与边界 | 位置 | 全量 (0.77) | 低 |
| L12H15 | 混合/通用 | 弱→句法依赖 | — | context_sensitive | ≠标点与边界 | 位置 | 全量 (0.95) | 低 |

## Layer 13

| Head | 保守标签 | 完整主签名 | 多标签 | 稳定性 | 局部因果 | 外部检索（实测） | 自然任务安全算子 | 置信度 |
|---|---|---|---|---|---|---|---|---|
| L13H00 | 混合/通用 | 弱→同词回指/复制 | — | intermediate | ≠同词回指/复制 | 位置+重复片段 | 全量 (0.81) | 低 |
| L13H01 | 混合/通用 | 弱→标点与边界 | — | intermediate | ≠标点与边界 | 位置+重复片段 | 全量 (0.97) | 低 |
| L13H02 | 结构锚点 | 结构锚点 | 结构锚点 | context_sensitive | ≠标点与边界 | 位置 | 全量 (0.72) | 低 |
| L13H03 | 句法依赖 | 句法依赖 | 句法依赖;结构锚点 | intermediate | ≠标点与边界 | 位置 | 全量 (0.62) | 中 |
| L13H04 | 结构锚点 | 结构锚点 | 结构锚点 | context_sensitive | ≠语义证据 | 位置+重复片段 | 全量 (0.75) | 低 |
| L13H05 | 局部近期上下文 | 局部近期上下文 | 局部近期上下文 | stable_bias | ≠同词回指/复制 | 位置 | 全量 (0.53) | 中 |
| L13H06 | 当前 token/self | 当前 token/self | 当前 token/self | stable_bias | ≠当前 token/self | 位置 | QK top-block (0.64) | 高 |
| L13H07 | 当前 token/self | 当前 token/self | 当前 token/self;前一 token | stable_bias | ≠当前 token/self | 位置 | 全量 (0.70) | 高 |
| L13H08 | 混合/通用 | 弱→局部近期上下文 | — | stable_bias | ≠局部近期上下文 | 位置 | QK top-block (0.73) | 中 |
| L13H09 | 混合/通用 | 弱→标点与边界 | — | intermediate | ≠标点与边界 | 位置 | QK top-block (0.55) | 低 |
| L13H10 | 混合/通用 | 弱→标点与边界 | 句法依赖 | stable_bias | ≠标点与边界 | 位置 | streaming (0.77) | 中 |
| L13H11 | 当前 token/self | 当前 token/self | 当前 token/self | stable_bias | ≠同词回指/复制 | 位置 | QK top-block (0.69) | 高 |
| L13H12 | 混合/通用 | 弱→局部近期上下文 | — | stable_bias | ≠前一 token | 位置 | 全量 (0.83) | 低 |
| L13H13 | 前一 token | 前一 token | 前一 token | stable_bias | ≠前一 token | 位置 | QK top-block (0.69) | 高 |
| L13H14 | 混合/通用 | 弱→当前 token/self | — | intermediate | ≠当前 token/self | 位置 | 全量 (0.59) | 低 |
| L13H15 | 混合/通用 | 弱→当前 token/self | — | intermediate | ≠标点与边界 | 位置 | 全量 (0.62) | 低 |

## Layer 14

| Head | 保守标签 | 完整主签名 | 多标签 | 稳定性 | 局部因果 | 外部检索（实测） | 自然任务安全算子 | 置信度 |
|---|---|---|---|---|---|---|---|---|
| L14H00 | 混合/通用 | 弱→标点与边界 | 句法依赖 | stable_bias | ≠标点与边界 | 位置 | 全量 (0.92) | 中 |
| L14H01 | 混合/通用 | 弱→标点与边界 | — | stable_bias | ≠标点与边界 | 位置 | 全量 (0.98) | 中 |
| L14H02 | 句法依赖 | 句法依赖 | 句法依赖 | intermediate | ≠句法依赖 | 位置 | QK top-block (0.52) | 中 |
| L14H03 | 混合/通用 | 弱→标点与边界 | — | stable_bias | ≠标点与边界 | 位置 | 全量 (0.73) | 中 |
| L14H04 | 标点与边界 | 标点与边界 | 标点与边界;句法依赖 | stable_bias | ≠标点与边界 | 位置 | 全量 (0.42) | 高 |
| L14H05 | 混合/通用 | 弱→序列起点/sink | — | context_sensitive | ≠前一 token | 位置 | streaming (0.55) | 低 |
| L14H06 | 混合/通用 | 弱→序列起点/sink | — | stable_bias | ≠语义证据 | 位置 | 全量 (0.52) | 低 |
| L14H07 | 混合/通用 | 弱→序列起点/sink | — | stable_bias | ≠序列起点/sink | 位置 | 全量 (0.67) | 中 |
| L14H08 | 前一 token | 前一 token | 前一 token;当前 token/self | stable_bias | ≠前一 token | 位置 | streaming (1.00) | 高 |
| L14H09 | 前一 token | 前一 token | 前一 token | stable_bias | ≠前一 token | 位置 | streaming (0.94) | 高 |
| L14H10 | 混合/通用 | 弱→序列起点/sink | — | stable_bias | ≠同词回指/复制 | 位置 | 全量 (0.91) | 低 |
| L14H11 | 混合/通用 | 弱→语义证据 | — | context_sensitive | ≠句法依赖 | 位置 | 全量 (0.91) | 低 |
| L14H12 | 当前 token/self | 当前 token/self | 当前 token/self | stable_bias | ≠当前 token/self | 位置 | streaming (0.97) | 高 |
| L14H13 | 当前 token/self | 当前 token/self | 当前 token/self;结构锚点 | stable_bias | ≠当前 token/self | 位置 | QK top-block (0.62) | 高 |
| L14H14 | 混合/通用 | 弱→当前 token/self | — | stable_bias | ≠同词回指/复制 | 位置+词法 | 全量 (0.80) | 低 |
| L14H15 | 混合/通用 | 弱→同词回指/复制 | 当前 token/self | intermediate | ≠同词回指/复制 | 位置 | 全量 (0.64) | 低 |

## Layer 15

| Head | 保守标签 | 完整主签名 | 多标签 | 稳定性 | 局部因果 | 外部检索（实测） | 自然任务安全算子 | 置信度 |
|---|---|---|---|---|---|---|---|---|
| L15H00 | 混合/通用 | 弱→结构锚点 | — | context_sensitive | ≠句法依赖 | 位置 | QK top-block (0.69) | 低 |
| L15H01 | 混合/通用 | 弱→标点与边界 | — | context_sensitive | ≠句法依赖 | 位置 | QK top-block (0.59) | 低 |
| L15H02 | 混合/通用 | 弱→语义证据 | — | context_sensitive | ≠语义证据 | 位置 | 全量 (0.55) | 低 |
| L15H03 | 前一 token | 前一 token | 前一 token | stable_bias | ≠前一 token | 位置 | streaming (1.00) | 高 |
| L15H04 | 混合/通用 | 弱→前一 token | — | intermediate | ≠句法依赖 | 位置 | streaming (0.75) | 低 |
| L15H05 | 混合/通用 | 弱→句法依赖 | — | intermediate | ≠句法依赖 | 位置 | QK top-block (0.69) | 低 |
| L15H06 | 混合/通用 | 弱→序列起点/sink | — | context_sensitive | ≠局部近期上下文 | 位置+词法 | 全量 (0.61) | 低 |
| L15H07 | 混合/通用 | 弱→前一 token | — | intermediate | ≠局部近期上下文 | 位置 | QK top-block (0.61) | 低 |
| L15H08 | 混合/通用 | 弱→标点与边界 | 句法依赖 | stable_bias | ≠标点与边界 | 位置 | QK top-block (0.81) | 中 |
| L15H09 | 混合/通用 | 弱→语义证据 | — | intermediate | ≠语义证据 | 位置 | streaming (0.70) | 低 |
| L15H10 | 混合/通用 | 弱→序列起点/sink | — | intermediate | ≠同词回指/复制 | 重复片段 | 全量 (0.77) | 低 |
| L15H11 | 混合/通用 | 弱→结构锚点 | — | stable_bias | ≠句法依赖 | 位置 | streaming (0.59) | 低 |
| L15H12 | 前一 token | 前一 token | 前一 token | stable_bias | ≠前一 token | 位置 | streaming (0.83) | 高 |
| L15H13 | 混合/通用 | 弱→前一 token | — | intermediate | ≠前一 token | 位置 | streaming (0.72) | 低 |
| L15H14 | 混合/通用 | 弱→前一 token | — | intermediate | ≠句法依赖 | 位置 | streaming (0.81) | 低 |
| L15H15 | 混合/通用 | 弱→局部近期上下文 | — | intermediate | ≠局部近期上下文 | 位置 | 全量 (0.53) | 低 |

## Layer 16

| Head | 保守标签 | 完整主签名 | 多标签 | 稳定性 | 局部因果 | 外部检索（实测） | 自然任务安全算子 | 置信度 |
|---|---|---|---|---|---|---|---|---|
| L16H00 | 前一 token | 前一 token | 前一 token | stable_bias | ≠前一 token | 位置 | streaming (1.00) | 高 |
| L16H01 | 标点与边界 | 标点与边界 | 标点与边界;句法依赖;结构锚点 | stable_bias | ≠标点与边界 | 位置 | streaming (0.81) | 高 |
| L16H02 | 结构锚点 | 结构锚点 | 结构锚点 | context_sensitive | ≠结构锚点 | 位置 | streaming (0.55) | 中 |
| L16H03 | 混合/通用 | 弱→结构锚点 | — | context_sensitive | ≠结构锚点 | 位置 | streaming (0.64) | 低 |
| L16H04 | 混合/通用 | 弱→语义证据 | 前一 token | stable_bias | ≠语义证据 | 位置 | streaming (0.92) | 中 |
| L16H05 | 混合/通用 | 弱→语义证据 | 句法依赖 | stable_bias | ≠语义证据 | 位置 | 全量 (0.70) | 中 |
| L16H06 | 混合/通用 | 弱→语义证据 | — | context_sensitive | ≠语义证据 | 位置 | QK top-block (0.64) | 低 |
| L16H07 | 混合/通用 | 弱→语义证据 | — | context_sensitive | ≠语义证据 | 位置 | QK top-block (0.78) | 低 |
| L16H08 | 当前 token/self | 当前 token/self | 当前 token/self | stable_bias | ≠语义证据 | 位置 | QK top-block (0.52) | 高 |
| L16H09 | 混合/通用 | 弱→语义证据 | 当前 token/self | stable_bias | ≠语义证据 | 位置 | QK top-block (0.67) | 中 |
| L16H10 | 混合/通用 | 弱→局部近期上下文 | — | stable_bias | ≠语义证据 | 位置 | QK top-block (0.64) | 低 |
| L16H11 | 混合/通用 | 弱→序列起点/sink | — | stable_bias | ≠序列起点/sink | 位置+词法 | 全量 (0.70) | 中 |
| L16H12 | 当前 token/self | 当前 token/self | 当前 token/self | stable_bias | ≠同词回指/复制 | 位置 | 全量 (0.75) | 中 |
| L16H13 | 当前 token/self | 当前 token/self | 当前 token/self;结构锚点;句法依赖 | stable_bias | ≠同词回指/复制 | 位置 | QK top-block (0.61) | 高 |
| L16H14 | 混合/通用 | 弱→序列起点/sink | — | stable_bias | ≠序列起点/sink | 位置+词法 | QK top-block (0.47) | 中 |
| L16H15 | 序列起点/sink | 序列起点/sink | 序列起点/sink | stable_bias | ≠序列起点/sink | 位置 | streaming (0.98) | 高 |

## Layer 17

| Head | 保守标签 | 完整主签名 | 多标签 | 稳定性 | 局部因果 | 外部检索（实测） | 自然任务安全算子 | 置信度 |
|---|---|---|---|---|---|---|---|---|
| L17H00 | 混合/通用 | 弱→语义证据 | — | context_sensitive | ≠语义证据 | 位置 | QK top-block (0.80) | 低 |
| L17H01 | 混合/通用 | 弱→语义证据 | — | context_sensitive | ≠语义证据 | 位置 | QK top-block (0.81) | 低 |
| L17H02 | 混合/通用 | 弱→语义证据 | 结构锚点 | intermediate | ≠语义证据 | 位置 | QK top-block (0.58) | 低 |
| L17H03 | 混合/通用 | 弱→语义证据 | — | intermediate | ≠语义证据 | 位置 | 全量 (0.61) | 低 |
| L17H04 | 结构锚点 | 结构锚点 | 结构锚点 | intermediate | ≠结构锚点 | 位置 | streaming (1.00) | 中 |
| L17H05 | 混合/通用 | 弱→结构锚点 | — | context_sensitive | ≠句法依赖 | 位置 | streaming (0.66) | 低 |
| L17H06 | 前一 token | 前一 token | 前一 token;当前 token/self | stable_bias | ≠前一 token | 位置 | streaming (0.98) | 高 |
| L17H07 | 混合/通用 | 弱→同词回指/复制 | — | stable_bias | ≠当前 token/self | 位置 | 全量 (0.92) | 低 |
| L17H08 | 混合/通用 | 弱→语义证据 | — | context_sensitive | ≠语义证据 | 位置 | 全量 (0.69) | 低 |
| L17H09 | 混合/通用 | 弱→语义证据 | — | context_sensitive | ≠语义证据 | 位置 | QK top-block (0.69) | 低 |
| L17H10 | 混合/通用 | 弱→语义证据 | — | context_sensitive | ≠语义证据 | 位置 | QK top-block (0.53) | 低 |
| L17H11 | 混合/通用 | 弱→语义证据 | — | context_sensitive | ≠语义证据 | 位置 | QK top-block (0.53) | 低 |
| L17H12 | 混合/通用 | 弱→语义证据 | 当前 token/self;结构锚点;句法依赖;前一 token | stable_bias | ≠语义证据 | 位置 | QK top-block (0.66) | 中 |
| L17H13 | 混合/通用 | 弱→语义证据 | 当前 token/self;前一 token | stable_bias | ≠语义证据 | 位置 | 全量 (0.59) | 中 |
| L17H14 | 前一 token | 前一 token | 前一 token | stable_bias | ≠前一 token | 位置 | streaming (0.94) | 高 |
| L17H15 | 当前 token/self | 当前 token/self | 当前 token/self;前一 token | intermediate | ≠当前 token/self | 位置 | streaming (0.58) | 中 |

## Layer 18

| Head | 保守标签 | 完整主签名 | 多标签 | 稳定性 | 局部因果 | 外部检索（实测） | 自然任务安全算子 | 置信度 |
|---|---|---|---|---|---|---|---|---|
| L18H00 | 混合/通用 | 弱→标点与边界 | — | stable_bias | ≠标点与边界 | 位置+词法 | QK top-block (0.53) | 中 |
| L18H01 | 结构锚点 | 结构锚点 | 结构锚点 | context_sensitive | ≠结构锚点 | 位置+词法 | QK top-block (0.53) | 中 |
| L18H02 | 混合/通用 | 弱→语义证据 | — | context_sensitive | ≠语义证据 | 位置 | 全量 (0.80) | 低 |
| L18H03 | 当前 token/self | 当前 token/self | 当前 token/self | intermediate | ≠当前 token/self | 位置 | QK top-block (0.75) | 中 |
| L18H04 | 混合/通用 | 弱→序列起点/sink | — | intermediate | ≠局部近期上下文 | 位置+词法 | QK top-block (0.95) | 低 |
| L18H05 | 混合/通用 | 弱→当前 token/self | — | context_sensitive | ≠当前 token/self | 位置+词法 | QK top-block (0.78) | 低 |
| L18H06 | 混合/通用 | 弱→前一 token | — | stable_bias | ≠语义证据 | 位置 | QK top-block (0.62) | 低 |
| L18H07 | 前一 token | 前一 token | 前一 token;当前 token/self | stable_bias | ≠前一 token | 位置 | QK top-block (0.84) | 高 |
| L18H08 | 前一 token | 前一 token | 前一 token | intermediate | ≠前一 token | 位置 | QK top-block (0.61) | 高 |
| L18H09 | 前一 token | 前一 token | 前一 token | stable_bias | ≠前一 token | 位置 | streaming (0.95) | 高 |
| L18H10 | 混合/通用 | 弱→结构锚点 | — | context_sensitive | ≠句法依赖 | 位置 | QK top-block (0.77) | 低 |
| L18H11 | 结构锚点 | 结构锚点 | 结构锚点 | intermediate | ≠语义证据 | 位置+词法 | QK top-block (0.77) | 中 |
| L18H12 | 混合/通用 | 弱→标点与边界 | 句法依赖 | stable_bias | ≠语义证据 | 位置 | QK top-block (0.84) | 中 |
| L18H13 | 混合/通用 | 弱→语义证据 | 句法依赖 | stable_bias | ≠语义证据 | 位置 | QK top-block (0.59) | 中 |
| L18H14 | 前一 token | 前一 token | 前一 token | stable_bias | ≠前一 token | 位置 | streaming (0.97) | 高 |
| L18H15 | 前一 token | 前一 token | 前一 token;当前 token/self | stable_bias | ≠前一 token | 位置 | streaming (0.73) | 高 |

## Layer 19

| Head | 保守标签 | 完整主签名 | 多标签 | 稳定性 | 局部因果 | 外部检索（实测） | 自然任务安全算子 | 置信度 |
|---|---|---|---|---|---|---|---|---|
| L19H00 | 混合/通用 | 弱→语义证据 | 句法依赖 | intermediate | ≠语义证据 | 位置 | QK top-block (0.86) | 中 |
| L19H01 | 混合/通用 | 弱→句法依赖 | — | context_sensitive | ≠序列起点/sink | 位置 | QK top-block (0.73) | 低 |
| L19H02 | 混合/通用 | 弱→语义证据 | 结构锚点;前一 token | stable_bias | ≠前一 token | 位置 | streaming (0.81) | 低 |
| L19H03 | 混合/通用 | 弱→语义证据 | 结构锚点 | context_sensitive | ≠语义证据 | 位置 | QK top-block (0.59) | 低 |
| L19H04 | 序列起点/sink | 序列起点/sink | 序列起点/sink | stable_bias | ≠序列起点/sink | 位置+词法 | streaming (0.80) | 高 |
| L19H05 | 混合/通用 | 弱→语义证据 | — | intermediate | ≠语义证据 | 位置 | QK top-block (0.52) | 中 |
| L19H06 | 前一 token | 前一 token | 前一 token;当前 token/self | stable_bias | ≠前一 token | 位置 | streaming (0.72) | 高 |
| L19H07 | 当前 token/self | 当前 token/self | 当前 token/self;前一 token | stable_bias | ≠当前 token/self | 位置+重复片段 | 全量 (0.55) | 高 |
| L19H08 | 混合/通用 | 弱→语义证据 | — | stable_bias | ≠语义证据 | 位置 | QK top-block (0.83) | 中 |
| L19H09 | 混合/通用 | 弱→语义证据 | — | intermediate | ≠语义证据 | 位置 | 全量 (0.62) | 低 |
| L19H10 | 混合/通用 | 弱→序列起点/sink | — | intermediate | ≠前一 token | 位置 | streaming (0.91) | 低 |
| L19H11 | 混合/通用 | 弱→前一 token | — | stable_bias | ≠前一 token | 位置 | streaming (1.00) | 中 |
| L19H12 | 混合/通用 | 弱→语义证据 | — | intermediate | ≠语义证据 | 位置 | streaming (0.58) | 中 |
| L19H13 | 语义证据 | 语义证据 | 语义证据 | context_sensitive | ≠语义证据 | 位置 | QK top-block (0.62) | 中 |
| L19H14 | 结构锚点 | 结构锚点 | 结构锚点 | context_sensitive | ≠结构锚点 | 位置 | 全量 (0.53) | 中 |
| L19H15 | 混合/通用 | 弱→序列起点/sink | — | stable_bias | ≠语义证据 | 位置 | QK top-block (0.78) | 低 |

## Layer 20

| Head | 保守标签 | 完整主签名 | 多标签 | 稳定性 | 局部因果 | 外部检索（实测） | 自然任务安全算子 | 置信度 |
|---|---|---|---|---|---|---|---|---|
| L20H00 | 前一 token | 前一 token | 前一 token | stable_bias | ≠前一 token | 位置 | streaming (0.73) | 高 |
| L20H01 | 前一 token | 前一 token | 前一 token | stable_bias | ≠前一 token | 位置 | streaming (0.88) | 高 |
| L20H02 | 混合/通用 | 弱→序列起点/sink | — | stable_bias | ≠结构锚点 | 位置 | streaming (0.84) | 低 |
| L20H03 | 序列起点/sink | 序列起点/sink | 序列起点/sink | stable_bias | ≠序列起点/sink | 位置 | streaming (1.00) | 高 |
| L20H04 | 混合/通用 | 弱→语义证据 | — | intermediate | ≠语义证据 | 位置 | QK top-block (0.75) | 中 |
| L20H05 | 句法依赖 | 句法依赖 | 句法依赖;结构锚点 | intermediate | ≠句法依赖 | 位置 | streaming (0.50) | 中 |
| L20H06 | 混合/通用 | 弱→语义证据 | — | intermediate | ≠语义证据 | 位置 | QK top-block (0.72) | 低 |
| L20H07 | 混合/通用 | 弱→语义证据 | — | intermediate | ≠语义证据 | 位置 | QK top-block (0.66) | 低 |
| L20H08 | 混合/通用 | 弱→语义证据 | — | stable_bias | ≠语义证据 | 位置 | QK top-block (0.78) | 中 |
| L20H09 | 混合/通用 | 弱→标点与边界 | — | context_sensitive | ≠句法依赖 | 位置 | streaming (0.97) | 低 |
| L20H10 | 混合/通用 | 弱→序列起点/sink | — | stable_bias | ≠局部近期上下文 | 位置+词法 | QK top-block (0.52) | 低 |
| L20H11 | 混合/通用 | 弱→序列起点/sink | — | stable_bias | ≠语义证据 | 位置+词法 | QK top-block (0.48) | 低 |
| L20H12 | 混合/通用 | 弱→语义证据 | — | stable_bias | ≠语义证据 | 位置 | streaming (0.89) | 中 |
| L20H13 | 混合/通用 | 弱→语义证据 | — | intermediate | ≠语义证据 | 位置 | QK top-block (0.66) | 中 |
| L20H14 | 混合/通用 | 弱→语义证据 | — | context_sensitive | ≠语义证据 | 位置+词法 | QK top-block (0.67) | 低 |
| L20H15 | 混合/通用 | 弱→语义证据 | — | intermediate | ≠语义证据 | 位置 | 全量 (0.62) | 中 |

## Layer 21

| Head | 保守标签 | 完整主签名 | 多标签 | 稳定性 | 局部因果 | 外部检索（实测） | 自然任务安全算子 | 置信度 |
|---|---|---|---|---|---|---|---|---|
| L21H00 | 混合/通用 | 弱→语义证据 | — | intermediate | ≠语义证据 | 位置 | 全量 (0.86) | 中 |
| L21H01 | 语义证据 | 语义证据 | 语义证据;结构锚点 | intermediate | ≠语义证据 | 位置 | 全量 (0.59) | 高 |
| L21H02 | 混合/通用 | 弱→语义证据 | — | intermediate | ≠语义证据 | 位置 | streaming (0.94) | 低 |
| L21H03 | 混合/通用 | 弱→标点与边界 | 结构锚点 | stable_bias | ≠标点与边界 | 位置 | streaming (0.91) | 中 |
| L21H04 | 混合/通用 | 弱→语义证据 | 句法依赖 | context_sensitive | ≠语义证据 | 位置 | QK top-block (0.78) | 低 |
| L21H05 | 混合/通用 | 弱→标点与边界 | — | intermediate | ≠标点与边界 | 位置 | QK top-block (0.81) | 低 |
| L21H06 | 混合/通用 | 弱→序列起点/sink | — | stable_bias | ≠序列起点/sink | 位置 | QK top-block (0.52) | 中 |
| L21H07 | 混合/通用 | 弱→序列起点/sink | — | stable_bias | ≠序列起点/sink | 位置 | streaming (0.62) | 中 |
| L21H08 | 混合/通用 | 弱→序列起点/sink | — | stable_bias | ≠语义证据 | 位置+词法 | 全量 (0.92) | 低 |
| L21H09 | 混合/通用 | 弱→序列起点/sink | — | stable_bias | ≠语义证据 | 位置+词法 | 全量 (0.73) | 低 |
| L21H10 | 混合/通用 | 弱→序列起点/sink | — | stable_bias | ≠序列起点/sink | 位置 | streaming (0.59) | 中 |
| L21H11 | 语义证据 | 语义证据 | 语义证据 | stable_bias | ≠语义证据 | 位置 | QK top-block (0.84) | 高 |
| L21H12 | 混合/通用 | 弱→序列起点/sink | — | intermediate | ≠语义证据 | 位置 | QK top-block (0.78) | 低 |
| L21H13 | 语义证据 | 语义证据 | 语义证据 | intermediate | ≠语义证据 | 位置 | 全量 (0.73) | 高 |
| L21H14 | 混合/通用 | 弱→语义证据 | — | intermediate | ≠语义证据 | 位置 | streaming (0.59) | 低 |
| L21H15 | 混合/通用 | 弱→语义证据 | — | context_sensitive | ≠语义证据 | 位置 | QK top-block (0.73) | 低 |

## Layer 22

| Head | 保守标签 | 完整主签名 | 多标签 | 稳定性 | 局部因果 | 外部检索（实测） | 自然任务安全算子 | 置信度 |
|---|---|---|---|---|---|---|---|---|
| L22H00 | 混合/通用 | 弱→序列起点/sink | — | stable_bias | ≠标点与边界 | 位置 | 全量 (0.89) | 低 |
| L22H01 | 混合/通用 | 弱→序列起点/sink | — | stable_bias | ≠语义证据 | 位置 | 全量 (0.64) | 低 |
| L22H02 | 混合/通用 | 弱→标点与边界 | — | intermediate | ≠标点与边界 | 位置 | QK top-block (0.52) | 低 |
| L22H03 | 混合/通用 | 弱→标点与边界 | — | stable_bias | ≠标点与边界 | 位置 | 全量 (0.52) | 中 |
| L22H04 | 混合/通用 | 弱→语义证据 | — | stable_bias | ≠语义证据 | 位置 | QK top-block (0.62) | 中 |
| L22H05 | 混合/通用 | 弱→序列起点/sink | — | stable_bias | ≠语义证据 | 位置 | streaming (0.81) | 低 |
| L22H06 | 混合/通用 | 弱→语义证据 | 前一 token;当前 token/self | stable_bias | ≠前一 token | 位置 | QK top-block (0.88) | 低 |
| L22H07 | 语义证据 | 语义证据 | 语义证据;结构锚点 | stable_bias | ≠语义证据 | 位置 | 全量 (0.56) | 高 |
| L22H08 | 混合/通用 | 弱→语义证据 | 结构锚点 | context_sensitive | ≠语义证据 | 位置 | QK top-block (0.77) | 低 |
| L22H09 | 混合/通用 | 弱→语义证据 | — | context_sensitive | ≠语义证据 | 位置 | QK top-block (0.70) | 低 |
| L22H10 | 混合/通用 | 弱→语义证据 | — | intermediate | ≠语义证据 | 位置 | QK top-block (0.81) | 低 |
| L22H11 | 混合/通用 | 弱→序列起点/sink | — | stable_bias | ≠序列起点/sink | 位置 | streaming (0.91) | 中 |
| L22H12 | 混合/通用 | 弱→语义证据 | — | stable_bias | ≠语义证据 | 位置 | QK top-block (0.53) | 中 |
| L22H13 | 混合/通用 | 弱→序列起点/sink | — | stable_bias | ≠语义证据 | 位置 | streaming (0.55) | 低 |
| L22H14 | 混合/通用 | 弱→语义证据 | 当前 token/self | stable_bias | ≠语义证据 | 位置 | QK top-block (0.64) | 中 |
| L22H15 | 混合/通用 | 弱→语义证据 | 当前 token/self;前一 token | stable_bias | ≠语义证据 | 位置 | streaming (0.61) | 中 |

## Layer 23

| Head | 保守标签 | 完整主签名 | 多标签 | 稳定性 | 局部因果 | 外部检索（实测） | 自然任务安全算子 | 置信度 |
|---|---|---|---|---|---|---|---|---|
| L23H00 | 混合/通用 | 弱→语义证据 | — | intermediate | ≠语义证据 | 位置 | QK top-block (0.58) | 中 |
| L23H01 | 混合/通用 | 弱→语义证据 | — | intermediate | ≠语义证据 | 位置 | QK top-block (0.77) | 中 |
| L23H02 | 前一 token | 前一 token | 前一 token | stable_bias | ≠前一 token | 位置 | streaming (1.00) | 高 |
| L23H03 | 当前 token/self | 当前 token/self | 当前 token/self | intermediate | ≠当前 token/self | 位置 | streaming (0.58) | 高 |
| L23H04 | 混合/通用 | 弱→语义证据 | — | stable_bias | ≠语义证据 | 位置 | 全量 (0.64) | 中 |
| L23H05 | 混合/通用 | 弱→语义证据 | — | stable_bias | ≠语义证据 | 位置 | 全量 (0.50) | 中 |
| L23H06 | 当前 token/self | 当前 token/self | 当前 token/self | stable_bias | ≠当前 token/self | 位置+重复片段 | streaming (0.88) | 高 |
| L23H07 | 当前 token/self | 当前 token/self | 当前 token/self | stable_bias | ≠当前 token/self | 位置+重复片段 | QK top-block (0.80) | 高 |
| L23H08 | 混合/通用 | 弱→序列起点/sink | — | intermediate | ≠序列起点/sink | 位置 | streaming (0.56) | 低 |
| L23H09 | 混合/通用 | 弱→序列起点/sink | — | intermediate | ≠序列起点/sink | 位置 | streaming (0.80) | 低 |
| L23H10 | 混合/通用 | 弱→序列起点/sink | — | stable_bias | ≠序列起点/sink | 位置 | streaming (0.69) | 中 |
| L23H11 | 混合/通用 | 弱→序列起点/sink | — | stable_bias | ≠序列起点/sink | 位置 | streaming (0.89) | 中 |
| L23H12 | 混合/通用 | 弱→序列起点/sink | — | stable_bias | ≠句法依赖 | 位置 | streaming (0.92) | 低 |
| L23H13 | 句法依赖 | 句法依赖 | 句法依赖 | intermediate | ≠句法依赖 | 位置 | streaming (0.67) | 中 |
| L23H14 | 语义证据 | 语义证据 | 语义证据 | intermediate | ≠语义证据 | 位置 | 全量 (0.53) | 高 |
| L23H15 | 混合/通用 | 弱→语义证据 | — | intermediate | ≠语义证据 | 位置 | QK top-block (0.55) | 中 |

## Layer 24

| Head | 保守标签 | 完整主签名 | 多标签 | 稳定性 | 局部因果 | 外部检索（实测） | 自然任务安全算子 | 置信度 |
|---|---|---|---|---|---|---|---|---|
| L24H00 | 混合/通用 | 弱→结构锚点 | — | context_sensitive | ≠当前 token/self | 位置 | streaming (1.00) | 低 |
| L24H01 | 混合/通用 | 弱→序列起点/sink | — | stable_bias | ≠序列起点/sink | 位置 | streaming (1.00) | 中 |
| L24H02 | 当前 token/self | 当前 token/self | 当前 token/self;句法依赖 | intermediate | ≠当前 token/self | 位置 | streaming (0.83) | 中 |
| L24H03 | 混合/通用 | 弱→序列起点/sink | — | stable_bias | ≠序列起点/sink | 位置 | QK top-block (0.89) | 中 |
| L24H04 | 混合/通用 | 弱→序列起点/sink | — | stable_bias | ≠语义证据 | 位置 | QK top-block (0.47) | 低 |
| L24H05 | 混合/通用 | 弱→语义证据 | — | stable_bias | ≠语义证据 | 位置 | QK top-block (0.53) | 中 |
| L24H06 | 混合/通用 | 弱→语义证据 | — | stable_bias | ≠语义证据 | 位置 | QK top-block (0.55) | 中 |
| L24H07 | 语义证据 | 语义证据 | 语义证据 | stable_bias | ≠语义证据 | 位置 | QK top-block (0.50) | 高 |
| L24H08 | 混合/通用 | 弱→序列起点/sink | — | stable_bias | ≠序列起点/sink | 位置 | QK top-block (0.58) | 中 |
| L24H09 | 混合/通用 | 弱→前一 token | — | intermediate | ≠前一 token | 位置 | streaming (1.00) | 低 |
| L24H10 | 混合/通用 | 弱→语义证据 | — | stable_bias | ≠语义证据 | 位置 | 全量 (0.73) | 中 |
| L24H11 | 语义证据 | 语义证据 | 语义证据 | stable_bias | ≠语义证据 | 位置 | 全量 (0.58) | 高 |
| L24H12 | 语义证据 | 语义证据 | 语义证据 | stable_bias | ≠语义证据 | 位置 | 全量 (0.53) | 高 |
| L24H13 | 混合/通用 | 弱→语义证据 | — | stable_bias | ≠语义证据 | 位置 | QK top-block (0.50) | 中 |
| L24H14 | 混合/通用 | 弱→语义证据 | — | stable_bias | ≠语义证据 | 位置 | QK top-block (0.64) | 中 |
| L24H15 | 混合/通用 | 弱→语义证据 | — | stable_bias | ≠语义证据 | 位置 | 全量 (0.66) | 中 |

## Layer 25

| Head | 保守标签 | 完整主签名 | 多标签 | 稳定性 | 局部因果 | 外部检索（实测） | 自然任务安全算子 | 置信度 |
|---|---|---|---|---|---|---|---|---|
| L25H00 | 当前 token/self | 当前 token/self | 当前 token/self | stable_bias | ≠当前 token/self | 位置 | streaming (0.98) | 高 |
| L25H01 | 混合/通用 | 弱→当前 token/self | — | stable_bias | ≠当前 token/self | 位置 | streaming (0.80) | 中 |
| L25H02 | 混合/通用 | 弱→标点与边界 | — | stable_bias | ≠标点与边界 | 位置 | QK top-block (0.72) | 中 |
| L25H03 | 混合/通用 | 弱→序列起点/sink | — | intermediate | ≠序列起点/sink | 位置 | streaming (1.00) | 低 |
| L25H04 | 混合/通用 | 弱→语义证据 | — | intermediate | ≠语义证据 | 位置 | QK top-block (0.50) | 低 |
| L25H05 | 混合/通用 | 弱→语义证据 | — | stable_bias | ≠语义证据 | 位置 | 全量 (0.62) | 中 |
| L25H06 | 混合/通用 | 弱→语义证据 | — | stable_bias | ≠语义证据 | 位置 | QK top-block (0.62) | 中 |
| L25H07 | 混合/通用 | 弱→语义证据 | — | stable_bias | ≠语义证据 | 位置 | QK top-block (0.52) | 中 |
| L25H08 | 混合/通用 | 弱→语义证据 | 当前 token/self | intermediate | ≠语义证据 | 位置 | QK top-block (0.62) | 低 |
| L25H09 | 混合/通用 | 弱→语义证据 | 当前 token/self | intermediate | ≠同词回指/复制 | 位置+重复片段 | QK top-block (0.70) | 低 |
| L25H10 | 混合/通用 | 弱→语义证据 | — | intermediate | ≠语义证据 | 位置 | QK top-block (0.52) | 中 |
| L25H11 | 混合/通用 | 弱→语义证据 | — | stable_bias | ≠语义证据 | 位置 | QK top-block (0.55) | 中 |
| L25H12 | 混合/通用 | 弱→语义证据 | — | stable_bias | ≠语义证据 | 位置 | 全量 (0.70) | 中 |
| L25H13 | 混合/通用 | 弱→语义证据 | — | stable_bias | ≠语义证据 | 位置 | 全量 (0.67) | 中 |
| L25H14 | 混合/通用 | 弱→语义证据 | — | stable_bias | ≠语义证据 | 位置+词法 | 全量 (0.55) | 中 |
| L25H15 | 混合/通用 | 弱→序列起点/sink | — | stable_bias | ≠序列起点/sink | 位置 | streaming (0.77) | 中 |

## Layer 26

| Head | 保守标签 | 完整主签名 | 多标签 | 稳定性 | 局部因果 | 外部检索（实测） | 自然任务安全算子 | 置信度 |
|---|---|---|---|---|---|---|---|---|
| L26H00 | 混合/通用 | 弱→语义证据 | — | intermediate | ≠语义证据 | 位置 | QK top-block (0.64) | 中 |
| L26H01 | 混合/通用 | 弱→语义证据 | — | intermediate | ≠语义证据 | 位置 | QK top-block (0.72) | 低 |
| L26H02 | 混合/通用 | 弱→语义证据 | — | intermediate | ≠语义证据 | 位置 | 全量 (0.78) | 中 |
| L26H03 | 混合/通用 | 弱→语义证据 | — | intermediate | ≠语义证据 | 位置 | QK top-block (0.48) | 中 |
| L26H04 | 混合/通用 | 弱→语义证据 | — | intermediate | ≠语义证据 | 位置 | 全量 (0.70) | 中 |
| L26H05 | 混合/通用 | 弱→语义证据 | — | intermediate | ≠语义证据 | 位置 | 全量 (0.67) | 中 |
| L26H06 | 混合/通用 | 弱→序列起点/sink | — | stable_bias | ≠序列起点/sink | 位置 | QK top-block (0.56) | 中 |
| L26H07 | 混合/通用 | 弱→语义证据 | — | intermediate | ≠语义证据 | 位置 | 全量 (0.88) | 中 |
| L26H08 | 混合/通用 | 弱→语义证据 | — | stable_bias | ≠语义证据 | 位置 | 全量 (0.95) | 中 |
| L26H09 | 当前 token/self | 当前 token/self | 当前 token/self | stable_bias | ≠当前 token/self | 位置 | streaming (0.91) | 高 |
| L26H10 | 混合/通用 | 弱→标点与边界 | — | stable_bias | ≠标点与边界 | 位置 | QK top-block (0.55) | 中 |
| L26H11 | 混合/通用 | 弱→序列起点/sink | — | stable_bias | ≠序列起点/sink | 位置 | streaming (1.00) | 中 |
| L26H12 | 序列起点/sink | 序列起点/sink | 序列起点/sink | stable_bias | ≠语义证据 | 位置 | 全量 (0.83) | 高 |
| L26H13 | 混合/通用 | 弱→序列起点/sink | — | stable_bias | ≠序列起点/sink | 位置 | QK top-block (0.52) | 中 |
| L26H14 | 混合/通用 | 弱→语义证据 | — | context_sensitive | ≠语义证据 | 位置 | 全量 (0.73) | 低 |
| L26H15 | 混合/通用 | 弱→语义证据 | — | context_sensitive | ≠语义证据 | 位置 | QK top-block (0.53) | 低 |

## Layer 27

| Head | 保守标签 | 完整主签名 | 多标签 | 稳定性 | 局部因果 | 外部检索（实测） | 自然任务安全算子 | 置信度 |
|---|---|---|---|---|---|---|---|---|
| L27H00 | 序列起点/sink | 序列起点/sink | 序列起点/sink | stable_bias | ≠序列起点/sink | 位置 | streaming (0.61) | 高 |
| L27H01 | 混合/通用 | 弱→序列起点/sink | — | stable_bias | ≠序列起点/sink | 位置 | streaming (0.73) | 中 |
| L27H02 | 混合/通用 | 弱→序列起点/sink | — | stable_bias | ≠序列起点/sink | 位置 | QK top-block (0.56) | 中 |
| L27H03 | 序列起点/sink | 序列起点/sink | 序列起点/sink | stable_bias | ≠序列起点/sink | 位置 | QK top-block (0.81) | 高 |
| L27H04 | 混合/通用 | 弱→序列起点/sink | — | stable_bias | ≠当前 token/self | 位置 | streaming (0.77) | 低 |
| L27H05 | 混合/通用 | 弱→当前 token/self | — | stable_bias | ≠当前 token/self | 位置 | streaming (1.00) | 中 |
| L27H06 | 混合/通用 | 弱→序列起点/sink | — | stable_bias | ≠序列起点/sink | 位置 | streaming (0.97) | 中 |
| L27H07 | 混合/通用 | 弱→语义证据 | 当前 token/self;前一 token | stable_bias | ≠语义证据 | 位置 | streaming (0.89) | 中 |
| L27H08 | 序列起点/sink | 序列起点/sink | 序列起点/sink | stable_bias | ≠序列起点/sink | 位置 | streaming (1.00) | 高 |
| L27H09 | 混合/通用 | 弱→序列起点/sink | — | intermediate | ≠语义证据 | 位置 | QK top-block (0.89) | 低 |
| L27H10 | 序列起点/sink | 序列起点/sink | 序列起点/sink | stable_bias | ≠序列起点/sink | 位置 | streaming (0.80) | 高 |
| L27H11 | 混合/通用 | 弱→序列起点/sink | — | stable_bias | ≠序列起点/sink | 位置 | streaming (0.97) | 中 |
| L27H12 | 混合/通用 | 弱→序列起点/sink | — | stable_bias | ≠序列起点/sink | 位置 | streaming (1.00) | 中 |
| L27H13 | 当前 token/self | 当前 token/self | 当前 token/self;前一 token | stable_bias | ≠前一 token | 位置 | streaming (0.77) | 高 |
| L27H14 | 当前 token/self | 当前 token/self | 当前 token/self | stable_bias | ≠当前 token/self | 位置 | streaming (0.83) | 高 |
| L27H15 | 当前 token/self | 当前 token/self | 当前 token/self;前一 token | stable_bias | ≠当前 token/self | 位置 | streaming (0.94) | 高 |

## 字段边界

- 功能标签描述的是本实验集合中的可观测 attention/输出签名，不是不可变的神经元语义。
- 外部检索（实测）来自 War and Peace 4K 的 2% oracle-mask imitation；它衡量位置召回，不等于生成 PPL 已被验证。
- 自然任务安全算子来自 64 个查询、相对 head-output L2≤0.05 的 teacher；agreement 越低，越应按 query 动态路由。
- 逐 head 数值、冲突敏感性、NLL 干预和推荐方法请查 `../outputs/head_function_atlas.csv`。
