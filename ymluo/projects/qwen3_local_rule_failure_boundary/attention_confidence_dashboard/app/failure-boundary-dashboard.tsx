"use client";

import { useMemo, useState } from "react";

export type FailureScope = "overall" | "layer" | "head";
type FailureMetric =
  | "total"
  | "start_key"
  | "hop1_result"
  | "hop2_input"
  | "hop2_result"
  | "hop2_logit"
  | "logsumexp";

type FailurePoint = {
  length: number;
  promptTokens: number;
  goldPpl: number;
  goldProbability: number;
  fullVocabMargin: number;
  fullVocabCorrect: boolean;
  candidateMargin: number;
  candidateCorrect: boolean;
  candidatePrediction: string;
  topToken: string;
  topProbability: number;
  letProbability: number;
  roleMass: number[][][];
  roleLogit: number[][][];
  headLogsumexp: number[][];
  headEntropy: number[][];
};

export type FailureBoundaryPayload = {
  schemaVersion: number;
  experiment: string;
  model: string;
  condition: string;
  chain: string[];
  placement: string;
  lengths: number[];
  numLayers: number;
  numHeads: number;
  roleOrder: string[];
  roleLabels: Record<string, string>;
  summary: {
    candidate_correct: number;
    full_vocab_correct: number;
    global_mass_corr_full_vocab_margin: number;
    late_layers: [number, number];
    late_mass_corr_full_vocab_margin: number;
    late_mass_corr_candidate_margin: number;
    global_correct_mean_mass: number;
    global_wrong_mean_mass: number;
    late_correct_mean_mass: number;
    late_wrong_mean_mass: number;
    failure_transitions: number;
    failure_with_late_mass_decrease: number;
    recovery_transitions: number;
    recovery_with_late_mass_increase: number;
  };
  points: FailurePoint[];
};

const ROLE_COLORS = ["#e6a95d", "#55c2a5", "#7ab8ff", "#f0746e"];
const METRIC_OPTIONS: Array<{ key: FailureMetric; label: string; unit: string }> = [
  { key: "total", label: "全部证据 mass", unit: "%" },
  { key: "start_key", label: "起始键", unit: "%" },
  { key: "hop1_result", label: "第一跳结果", unit: "%" },
  { key: "hop2_input", label: "第二跳输入", unit: "%" },
  { key: "hop2_result", label: "最终结果", unit: "%" },
  { key: "hop2_logit", label: "最终证据 QK logit", unit: "logit" },
  { key: "logsumexp", label: "Softmax logsumexp", unit: "LSE" },
];

function formatPercent(value: number) {
  if (value >= 0.01) return `${(value * 100).toFixed(2)}%`;
  if (value >= 0.0001) return `${(value * 100).toFixed(3)}%`;
  return value.toExponential(2);
}

function average(values: number[]) {
  return values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length);
}

function roleIndex(payload: FailureBoundaryPayload, metric: FailureMetric) {
  return payload.roleOrder.indexOf(metric);
}

function scopeHeadValues(
  point: FailurePoint,
  payload: FailureBoundaryPayload,
  metric: FailureMetric,
  layer: number,
) {
  if (metric === "logsumexp") return point.headLogsumexp[layer];
  if (metric === "hop2_logit") {
    const index = payload.roleOrder.indexOf("hop2_result");
    return point.roleLogit[layer].map((head) => head[index]);
  }
  const index = roleIndex(payload, metric);
  return point.roleMass[layer].map((head) =>
    metric === "total"
      ? head.slice(0, 4).reduce((sum, value) => sum + value, 0)
      : head[index],
  );
}

function scopedMetric(
  point: FailurePoint,
  payload: FailureBoundaryPayload,
  metric: FailureMetric,
  scope: FailureScope,
  layer: number,
  head: number,
) {
  if (scope === "head") return scopeHeadValues(point, payload, metric, layer)[head];
  if (scope === "layer") return average(scopeHeadValues(point, payload, metric, layer));
  return average(
    Array.from({ length: payload.numLayers }, (_, layerIndex) =>
      scopeHeadValues(point, payload, metric, layerIndex),
    ).flat(),
  );
}

function roleMassForScope(
  point: FailurePoint,
  payload: FailureBoundaryPayload,
  scope: FailureScope,
  layer: number,
  head: number,
) {
  return payload.roleOrder.map((_, role) => {
    if (scope === "head") return point.roleMass[layer][head][role];
    if (scope === "layer") {
      return average(point.roleMass[layer].map((values) => values[role]));
    }
    return average(
      point.roleMass.flatMap((heads) => heads.map((values) => values[role])),
    );
  });
}

function pathFor(
  values: number[],
  width: number,
  height: number,
  minValue: number,
  maxValue: number,
) {
  const span = Math.max(1e-12, maxValue - minValue);
  return values
    .map((value, index) => {
      const x = 42 + (index / Math.max(1, values.length - 1)) * (width - 64);
      const y = 16 + (1 - (value - minValue) / span) * (height - 42);
      return `${index === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");
}

function BoundaryCurve({
  payload,
  values,
  selectedIndex,
  label,
  percent,
}: {
  payload: FailureBoundaryPayload;
  values: number[];
  selectedIndex: number;
  label: string;
  percent: boolean;
}) {
  const width = 920;
  const height = 220;
  const minValue = Math.min(...values);
  const maxValue = Math.max(...values);
  const selectedX =
    42 + (selectedIndex / Math.max(1, values.length - 1)) * (width - 64);
  const selectedValue = values[selectedIndex];
  const selectedY =
    16 +
    (1 - (selectedValue - minValue) / Math.max(1e-12, maxValue - minValue)) *
      (height - 42);
  const ticks = [34, 48, 60, 80, 100];
  return (
    <div className="failure-curve">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${label} 随长度变化`}>
        <title>{label} 随 filler 长度变化</title>
        {[0, 0.5, 1].map((ratio) => {
          const y = 16 + ratio * (height - 42);
          return <line key={ratio} className="failure-grid-line" x1="42" x2={width - 22} y1={y} y2={y} />;
        })}
        {payload.points.map((point, index) => {
          const x = 42 + (index / Math.max(1, values.length - 1)) * (width - 64);
          return (
            <rect
              key={point.length}
              x={x - 2.2}
              y={height - 17}
              width="4.4"
              height="5"
              className={point.fullVocabCorrect ? "failure-state-correct" : "failure-state-wrong"}
            />
          );
        })}
        <path className="failure-primary-line" d={pathFor(values, width, height, minValue, maxValue)} />
        <line className="failure-selected-line" x1={selectedX} x2={selectedX} y1="12" y2={height - 12} />
        <circle className="failure-selected-dot" cx={selectedX} cy={selectedY} r="5" />
        <text x="4" y="22">{percent ? formatPercent(maxValue) : maxValue.toFixed(3)}</text>
        <text x="4" y={height - 28}>{percent ? formatPercent(minValue) : minValue.toFixed(3)}</text>
        {ticks.map((tick) => {
          const index = payload.lengths.indexOf(tick);
          const x = 42 + (index / Math.max(1, values.length - 1)) * (width - 64);
          return <text key={tick} x={x} y={height - 1} textAnchor="middle">{tick}</text>;
        })}
      </svg>
      <div className="failure-curve-caption">
        <span><i className="correct" />首 token 正确</span>
        <span><i className="wrong" />首 token 为解释前缀</span>
        <b>{label}：{percent ? formatPercent(selectedValue) : selectedValue.toFixed(4)}</b>
      </div>
    </div>
  );
}

function MarginCurve({
  payload,
  selectedIndex,
}: {
  payload: FailureBoundaryPayload;
  selectedIndex: number;
}) {
  const width = 920;
  const height = 180;
  const values = payload.points.map((point) => point.fullVocabMargin);
  const minValue = Math.min(...values, 0);
  const maxValue = Math.max(...values, 0);
  const span = Math.max(1e-12, maxValue - minValue);
  const zeroY = 16 + (1 - (0 - minValue) / span) * (height - 42);
  const selectedX =
    42 + (selectedIndex / Math.max(1, values.length - 1)) * (width - 64);
  const selectedY =
    16 + (1 - (values[selectedIndex] - minValue) / span) * (height - 42);
  return (
    <div className="failure-curve margin">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="完整词表答案 margin 随长度变化">
        <title>basket 相对最强非答案 token 的 log probability margin</title>
        <line className="failure-zero-line" x1="42" x2={width - 22} y1={zeroY} y2={zeroY} />
        <path className="failure-margin-line" d={pathFor(values, width, height, minValue, maxValue)} />
        <line className="failure-selected-line" x1={selectedX} x2={selectedX} y1="12" y2={height - 12} />
        <circle
          className={values[selectedIndex] >= 0 ? "failure-margin-dot correct" : "failure-margin-dot wrong"}
          cx={selectedX}
          cy={selectedY}
          r="5"
        />
        <text x="4" y="22">{maxValue.toFixed(2)}</text>
        <text x="4" y={height - 28}>{minValue.toFixed(2)}</text>
        <text x={width - 26} y={zeroY - 5} textAnchor="end">margin = 0</text>
      </svg>
    </div>
  );
}

export function FailureBoundaryDashboard({
  payload,
  selectedIndex,
  scope,
  layer,
  head,
  onScopeChange,
  onLayerChange,
  onHeadChange,
}: {
  payload: FailureBoundaryPayload;
  selectedIndex: number;
  scope: FailureScope;
  layer: number;
  head: number;
  onScopeChange: (scope: FailureScope) => void;
  onLayerChange: (layer: number) => void;
  onHeadChange: (head: number) => void;
}) {
  const [metric, setMetric] = useState<FailureMetric>("total");
  const point = payload.points[selectedIndex];
  const metricOption = METRIC_OPTIONS.find((option) => option.key === metric)!;
  const values = useMemo(
    () =>
      payload.points.map((row) =>
        scopedMetric(row, payload, metric, scope, layer, head),
      ),
    [head, layer, metric, payload, scope],
  );
  const roleMass = roleMassForScope(point, payload, scope, layer, head);
  const totalRoleMass = roleMass.reduce((sum, value) => sum + value, 0);
  const heatValues = point.roleMass.flatMap((heads, layerIndex) =>
    heads.map((roles, headIndex) => {
      if (metric === "hop2_logit") {
        return point.roleLogit[layerIndex][headIndex][3];
      }
      if (metric === "logsumexp") return 0;
      const index = roleIndex(payload, metric);
      return metric === "total"
        ? roles.slice(0, 4).reduce((sum, value) => sum + value, 0)
        : roles[index];
    }),
  );
  const heatMax = Math.max(...heatValues.map((value) => Math.abs(value)), 1e-9);
  const state = point.fullVocabCorrect
    ? {
        title: "答案 token 仍占优",
        detail: `basket 的完整词表 margin 为 ${point.fullVocabMargin.toFixed(3)}。`,
        tone: "correct",
      }
    : point.candidateCorrect
      ? {
          title: "输出格式翻转，不是证据答案选错",
          detail: `合法候选仍选择 ${point.candidatePrediction}，但首 token 变为 ${point.topToken}。`,
          tone: "warning",
        }
      : {
          title: "合法候选也已选错",
          detail: `candidate margin 为 ${point.candidateMargin.toFixed(3)}。`,
          tone: "wrong",
        };

  return (
    <section className="failure-dashboard" data-testid="failure-boundary-dashboard">
      <div className="failure-workspace">
        <aside className="failure-controls">
          <section>
            <p className="section-kicker">VIEW SCOPE</p>
            <div className="segmented">
              {(["overall", "layer", "head"] as FailureScope[]).map((value) => (
                <button
                  key={value}
                  className={scope === value ? "active" : ""}
                  onClick={() => onScopeChange(value)}
                >
                  {value === "overall" ? "全模型" : value === "layer" ? "单层" : "单 Head"}
                </button>
              ))}
            </div>
          </section>
          <section className={scope === "overall" ? "disabled-control" : ""}>
            <label htmlFor="failure-layer"><span>Layer</span><b>{layer}</b></label>
            <input
              id="failure-layer"
              type="range"
              min="0"
              max={payload.numLayers - 1}
              value={layer}
              disabled={scope === "overall"}
              onChange={(event) => onLayerChange(Number(event.target.value))}
            />
            <div className="range-labels"><span>0</span><span>{payload.numLayers - 1}</span></div>
          </section>
          <section className={scope !== "head" ? "disabled-control" : ""}>
            <label htmlFor="failure-head"><span>Head</span><b>{head}</b></label>
            <input
              id="failure-head"
              type="range"
              min="0"
              max={payload.numHeads - 1}
              value={head}
              disabled={scope !== "head"}
              onChange={(event) => onHeadChange(Number(event.target.value))}
            />
            <div className="range-labels"><span>0</span><span>{payload.numHeads - 1}</span></div>
          </section>
          <section>
            <p className="section-kicker">ATTENTION METRIC</p>
            <div className="failure-metric-list">
              {METRIC_OPTIONS.map((option) => (
                <button
                  key={option.key}
                  className={metric === option.key ? "active" : ""}
                  onClick={() => setMetric(option.key)}
                >
                  <span>{option.label}</span><b>{option.unit}</b>
                </button>
              ))}
            </div>
          </section>
        </aside>

        <div className="failure-analysis">
          <section className={`failure-verdict ${state.tone}`}>
            <div>
              <p className="section-kicker">SELECTED LENGTH · {point.length} TOKENS</p>
              <h2>{state.title}</h2>
              <p>{state.detail}</p>
            </div>
            <div className="failure-probability-compare">
              <article>
                <span>basket</span>
                <strong>{formatPercent(point.goldProbability)}</strong>
                <i style={{ width: `${Math.min(100, point.goldProbability * 100)}%` }} />
              </article>
              <article>
                <span>Let</span>
                <strong>{formatPercent(point.letProbability)}</strong>
                <i style={{ width: `${Math.min(100, point.letProbability * 100)}%` }} />
              </article>
            </div>
          </section>

          <section className="failure-chart-panel">
            <div className="panel-heading compact">
              <div>
                <p className="section-kicker">ATTENTION TRACE · EVERY VALID LENGTH</p>
                <h3>{metricOption.label} · {scope === "overall" ? "全模型" : scope === "layer" ? `Layer ${layer}` : `Layer ${layer} / Head ${head}`}</h3>
              </div>
              <strong>{metricOption.unit === "%" ? formatPercent(values[selectedIndex]) : values[selectedIndex].toFixed(4)}</strong>
            </div>
            <BoundaryCurve
              payload={payload}
              values={values}
              selectedIndex={selectedIndex}
              label={metricOption.label}
              percent={metricOption.unit === "%"}
            />
          </section>

          <section className="failure-chart-panel">
            <div className="panel-heading compact">
              <div>
                <p className="section-kicker">OUTPUT DECISION BOUNDARY</p>
                <h3>basket 对最强任意 token 的 margin</h3>
              </div>
              <strong className={point.fullVocabMargin >= 0 ? "positive" : "negative"}>
                {point.fullVocabMargin >= 0 ? "+" : ""}{point.fullVocabMargin.toFixed(3)}
              </strong>
            </div>
            <MarginCurve payload={payload} selectedIndex={selectedIndex} />
          </section>

          <section className="failure-role-panel">
            <div className="panel-heading compact">
              <div><p className="section-kicker">EVIDENCE MASS BREAKDOWN</p><h3>当前 scope 的四个证据位置</h3></div>
              <strong>{formatPercent(totalRoleMass)}</strong>
            </div>
            <div className="failure-role-bars">
              {payload.roleOrder.map((role, index) => (
                <div key={role}>
                  <span>{payload.roleLabels[role]}</span>
                  <div><i style={{ width: `${Math.max(0.5, roleMass[index] / Math.max(...roleMass, 1e-12) * 100)}%`, background: ROLE_COLORS[index] }} /></div>
                  <b>{formatPercent(roleMass[index])}</b>
                </div>
              ))}
            </div>
          </section>

          <section className="failure-heatmap-panel">
            <div className="panel-heading compact">
              <div><p className="section-kicker">LAYER × HEAD · SELECTED LENGTH</p><h3>{point.length} tokens · {metricOption.label}</h3></div>
              <span>点击格子切换到对应 Head</span>
            </div>
            {metric === "logsumexp" ? (
              <div className="failure-empty">切换到证据 mass 或 QK logit 后显示逐 Head 热力图。</div>
            ) : (
              <div className="failure-head-map" style={{ gridTemplateColumns: `44px repeat(${payload.numHeads}, minmax(8px, 1fr))` }}>
                <div />
                {Array.from({ length: payload.numHeads }, (_, headIndex) => (
                  <span key={headIndex}>{headIndex % 4 === 0 ? headIndex : ""}</span>
                ))}
                {Array.from({ length: payload.numLayers }, (_, layerIndex) => (
                  <div key={layerIndex} style={{ display: "contents" }}>
                    <span>L{layerIndex}</span>
                    {Array.from({ length: payload.numHeads }, (_, headIndex) => {
                      const value = metric === "hop2_logit"
                        ? point.roleLogit[layerIndex][headIndex][3]
                        : metric === "total"
                          ? point.roleMass[layerIndex][headIndex].slice(0, 4).reduce((sum, item) => sum + item, 0)
                          : point.roleMass[layerIndex][headIndex][roleIndex(payload, metric)];
                      const intensity = Math.min(1, Math.abs(value) / heatMax);
                      const color = metric === "hop2_logit" && value < 0 ? "240,116,110" : "85,194,165";
                      return (
                        <button
                          key={`${layerIndex}-${headIndex}`}
                          className={scope === "head" && layer === layerIndex && head === headIndex ? "selected" : ""}
                          style={{ backgroundColor: `rgba(${color},${(0.06 + intensity * 0.94).toFixed(3)})` }}
                          aria-label={`Layer ${layerIndex} Head ${headIndex}, ${metricOption.label} ${value}`}
                          title={`L${layerIndex} H${headIndex} · ${metricOption.unit === "%" ? formatPercent(value) : value.toFixed(4)}`}
                          onClick={() => {
                            onLayerChange(layerIndex);
                            onHeadChange(headIndex);
                            onScopeChange("head");
                          }}
                        />
                      );
                    })}
                  </div>
                ))}
              </div>
            )}
          </section>

          <section className="failure-findings">
            <div>
              <b>全模型证据 mass</b>
              <strong>r = {payload.summary.global_mass_corr_full_vocab_margin.toFixed(3)}</strong>
              <span>错误点比正确点低 {Math.abs((payload.summary.global_wrong_mean_mass / payload.summary.global_correct_mean_mass - 1) * 100).toFixed(1)}%</span>
            </div>
            <div className="emphasis">
              <b>关键晚层 30–33</b>
              <strong>r = {payload.summary.late_mass_corr_full_vocab_margin.toFixed(3)}</strong>
              <span>{payload.summary.failure_with_late_mass_decrease}/{payload.summary.failure_transitions} 次失败伴随其下降</span>
            </div>
            <div>
              <b>合法候选检索</b>
              <strong>{payload.summary.candidate_correct}/{payload.points.length}</strong>
              <span>本样例没有在 river/window/basket 中选错</span>
            </div>
          </section>
        </div>
      </div>
    </section>
  );
}
