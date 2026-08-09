"use client";

import { useEffect, useMemo, useState } from "react";

type Scope = "overall" | "layer" | "head";
type Metric =
  | "mass"
  | "meanAttention"
  | "enrichment"
  | "meanLogit"
  | "maxLogit"
  | "logsumexp"
  | "bestRank";

type CandidateScore = {
  word: string;
  token_id: number;
  log_probability: number;
  probability: number;
};

type Distractor = {
  name: string;
  age: string;
  text: string;
  token_count: number;
  span: [number, number];
  age_span: [number, number];
};

type Point = {
  distractorCount: number;
  totalTokens: number;
  fillerCount: number;
  fillerGapCounts: number[];
  categoryCounts: Record<string, number>;
  categorySpans: Record<string, Array<[number, number]>>;
  goldSpan: [number, number];
  goldAgeSpan: [number, number];
  querySpan: [number, number];
  goldText: string;
  queryText: string;
  distractors: Distractor[];
  promptText: string;
  decodedTokens: string[];
  tokenCategories: string[];
  goldProbability: number;
  goldPpl: number;
  fullVocabMargin: number;
  fullVocabCorrect: boolean;
  candidateMargin: number;
  candidateCorrect: boolean;
  candidatePrediction: string;
  topToken: string;
  topProbability: number;
  strongestNonGold: { token: string; probability: number };
  strongestWrongCandidate: CandidateScore;
  nextTokenTop10: Array<{ token: string; probability: number }>;
  candidateScores: CandidateScore[];
  headCategoryMass: number[][][];
  headCategoryMeanAttention: number[][][];
  headCategoryEnrichment: number[][][];
  headCategoryMeanLogit: number[][][];
  headCategoryMaxLogit: number[][][];
  headCategoryLogsumexp: number[][][];
  headCategoryBestRank: number[][][];
  headEntropy: number[][];
  headEffectiveTokens: number[][];
  headLogsumexp: number[][];
  headMaxLogit: number[][];
  timing: Record<string, number>;
};

type Payload = {
  schemaVersion: number;
  experiment: string;
  model: string;
  totalTokens: number;
  goldEvidence: string;
  query: string;
  goldAnswer: string;
  numberWords: string[];
  answerTokenIds: Record<string, number>;
  categoryOrder: string[];
  categoryLabels: Record<string, string>;
  numLayers: number;
  numHeads: number;
  summary: {
    fullVocabCorrectCount: number;
    candidateCorrectCount: number;
    goldAgeMassVsMarginPearson: number;
    goldLineMassVsMarginPearson: number;
    distractorAgeMassVsMarginPearson: number;
    distractorLineMassVsMarginPearson: number;
    irrelevantMassVsMarginPearson: number;
    goldAgeMassVsPplPearson: number;
    firstFullVocabFailure: number | null;
    firstCandidateFailure: number | null;
  };
  points: Point[];
};

const DATA_PATH = "/data/age_distractor_fixed300/payload.json.gz";
const CATEGORY_COLORS: Record<string, string> = {
  gold_line: "#55c2a5",
  gold_age: "#e6a95d",
  distractor_lines: "#f0746e",
  distractor_ages: "#c4a7e7",
  irrelevant_periods: "#738092",
};
const CATEGORY_SHORT: Record<string, string> = {
  gold_line: "正确证据句",
  gold_age: "答案词 nine",
  distractor_lines: "干扰句",
  distractor_ages: "干扰年龄词",
  irrelevant_periods: "无关句号",
};
const METRICS: Array<{ key: Metric; label: string; note: string }> = [
  { key: "mass", label: "Attention mass", note: "该类别获得的 softmax 概率总和" },
  { key: "meanAttention", label: "单 token attention", note: "mass ÷ 类别 token 数" },
  { key: "enrichment", label: "均匀基线倍数", note: "单 token attention 相对 1/300" },
  { key: "meanLogit", label: "平均 QK", note: "softmax 之前的平均 QK/√d" },
  { key: "maxLogit", label: "最大 QK", note: "类别中最强 token 的 QK/√d" },
  { key: "logsumexp", label: "类别 LogSumExp", note: "类别内全部 QK 的总竞争强度" },
  { key: "bestRank", label: "最佳排名", note: "该类别最高 QK 在 300 个 token 中的排名" },
];

async function readPayload(response: Response): Promise<Payload> {
  const fallback = response.clone();
  try {
    return await response.json() as Payload;
  } catch {
    // Static hosts may return raw gzip instead of setting Content-Encoding.
  }
  if (!fallback.body || typeof DecompressionStream === "undefined") {
    throw new Error("浏览器不支持 gzip 解压。");
  }
  const stream = fallback.body.pipeThrough(new DecompressionStream("gzip"));
  return JSON.parse(await new Response(stream).text()) as Payload;
}

function average(values: number[]) {
  return values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length);
}

function formatPercent(value: number) {
  if (value >= 0.01) return `${(value * 100).toFixed(2)}%`;
  if (value >= 0.0001) return `${(value * 100).toFixed(3)}%`;
  return `${(value * 100).toExponential(2)}%`;
}

function signed(value: number, digits = 3) {
  return `${value >= 0 ? "+" : ""}${value.toFixed(digits)}`;
}

function tensor(point: Point, metric: Metric) {
  if (metric === "mass") return point.headCategoryMass;
  if (metric === "meanAttention") return point.headCategoryMeanAttention;
  if (metric === "enrichment") return point.headCategoryEnrichment;
  if (metric === "meanLogit") return point.headCategoryMeanLogit;
  if (metric === "maxLogit") return point.headCategoryMaxLogit;
  if (metric === "logsumexp") return point.headCategoryLogsumexp;
  return point.headCategoryBestRank;
}

function scopedValue(
  point: Point,
  metric: Metric,
  categoryIndex: number,
  scope: Scope,
  layer: number,
  head: number,
) {
  const values = tensor(point, metric);
  if (scope === "head") return values[layer][head][categoryIndex];
  if (scope === "layer") return average(values[layer].map((item) => item[categoryIndex]));
  return average(values.flatMap((heads) => heads.map((item) => item[categoryIndex])));
}

function displayMetric(value: number, metric: Metric) {
  if (metric === "mass" || metric === "meanAttention") return formatPercent(value);
  if (metric === "enrichment") return `${value.toFixed(2)}×`;
  if (metric === "bestRank") return `#${Math.round(value)}`;
  return value.toFixed(3);
}

function MetricCard({
  label,
  value,
  detail,
  color,
}: {
  label: string;
  value: string;
  detail: string;
  color: string;
}) {
  return (
    <article className="age-metric" style={{ "--age-accent": color } as React.CSSProperties}>
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{detail}</small>
    </article>
  );
}

function Curve({
  payload,
  selectedIndex,
  categoryIndex,
  metric,
  scope,
  layer,
  head,
  onSelect,
}: {
  payload: Payload;
  selectedIndex: number;
  categoryIndex: number;
  metric: Metric;
  scope: Scope;
  layer: number;
  head: number;
  onSelect: (index: number) => void;
}) {
  const evidence = payload.points.map((point) =>
    scopedValue(point, metric, categoryIndex, scope, layer, head),
  );
  const margin = payload.points.map((point) => point.fullVocabMargin);
  const width = 760;
  const height = 250;
  const pad = { left: 48, right: 38, top: 22, bottom: 34 };
  const minY = Math.min(...evidence);
  const maxY = Math.max(...evidence);
  const minMargin = Math.min(0, ...margin);
  const maxMargin = Math.max(0, ...margin);
  const x = (index: number) =>
    pad.left + index * ((width - pad.left - pad.right) / Math.max(1, payload.points.length - 1));
  const y = (value: number) =>
    pad.top + (maxY - value) * ((height - pad.top - pad.bottom) / Math.max(1e-12, maxY - minY));
  const marginY = (value: number) =>
    pad.top + (maxMargin - value) *
      ((height - pad.top - pad.bottom) / Math.max(1e-12, maxMargin - minMargin));
  const evidencePath = evidence.map((value, index) => `${index ? "L" : "M"}${x(index)},${y(value)}`).join(" ");
  const marginPath = margin.map((value, index) => `${index ? "L" : "M"}${x(index)},${marginY(value)}`).join(" ");
  const color = CATEGORY_COLORS[payload.categoryOrder[categoryIndex]];

  return (
    <div className="age-curve">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="干扰数量扫描曲线">
        {[0, 0.5, 1].map((fraction) => {
          const gy = pad.top + fraction * (height - pad.top - pad.bottom);
          return <line key={fraction} x1={pad.left} x2={width - pad.right} y1={gy} y2={gy} className="age-grid" />;
        })}
        {minMargin <= 0 && maxMargin >= 0 ? (
          <line x1={pad.left} x2={width - pad.right} y1={marginY(0)} y2={marginY(0)} className="age-zero" />
        ) : null}
        <path d={evidencePath} fill="none" stroke={color} strokeWidth="2.4" vectorEffect="non-scaling-stroke" />
        <path d={marginPath} fill="none" className="age-margin-line" />
        <line x1={x(selectedIndex)} x2={x(selectedIndex)} y1={pad.top} y2={height - pad.bottom} className="age-selected-line" />
        {payload.points.map((point, index) => (
          <g key={point.distractorCount} onClick={() => onSelect(index)} className="age-curve-hit">
            <circle cx={x(index)} cy={y(evidence[index])} r={index === selectedIndex ? 5 : 3} fill={color} />
            <circle cx={x(index)} cy={marginY(margin[index])} r="3" fill={point.fullVocabCorrect ? "#55c2a5" : "#f0746e"} />
            <rect x={x(index) - 14} y={pad.top} width="28" height={height - pad.top - pad.bottom} fill="transparent" />
            <text x={x(index)} y={height - 12} textAnchor="middle">{point.distractorCount}</text>
          </g>
        ))}
        <text x="4" y={pad.top + 4}>{displayMetric(maxY, metric)}</text>
        <text x="4" y={height - pad.bottom}>{displayMetric(minY, metric)}</text>
        <text x={width - 4} y={pad.top + 4} textAnchor="end">margin {signed(maxMargin, 2)}</text>
        <text x={width - 4} y={height - pad.bottom} textAnchor="end">margin {signed(minMargin, 2)}</text>
      </svg>
      <div className="age-curve-legend">
        <span><i style={{ background: color }} />{CATEGORY_SHORT[payload.categoryOrder[categoryIndex]]} · {METRICS.find((item) => item.key === metric)?.label}</span>
        <span><i className="margin" />完整词表 margin</span>
      </div>
    </div>
  );
}

function Heatmap({
  payload,
  point,
  metric,
  categoryIndex,
  layer,
  head,
  onSelect,
}: {
  payload: Payload;
  point: Point;
  metric: Metric;
  categoryIndex: number;
  layer: number;
  head: number;
  onSelect: (layer: number, head: number) => void;
}) {
  const values = tensor(point, metric).flatMap((heads) => heads.map((item) => item[categoryIndex]));
  const min = Math.min(...values);
  const max = Math.max(...values);
  const scale = (value: number) => {
    let normalized = (value - min) / Math.max(1e-12, max - min);
    if (metric === "bestRank") normalized = 1 - normalized;
    const alpha = 0.09 + normalized * 0.91;
    return `color-mix(in srgb, ${CATEGORY_COLORS[payload.categoryOrder[categoryIndex]]} ${Math.round(alpha * 100)}%, #111923)`;
  };
  return (
    <div className="age-heatmap" style={{ gridTemplateColumns: `30px repeat(${payload.numHeads}, minmax(8px, 1fr))` }}>
      <span />
      {Array.from({ length: payload.numHeads }, (_, headIndex) => <span key={headIndex}>{headIndex % 4 === 0 ? headIndex : ""}</span>)}
      {Array.from({ length: payload.numLayers }, (_, layerIndex) => (
        <div key={layerIndex} style={{ display: "contents" }}>
          <span>{layerIndex}</span>
          {Array.from({ length: payload.numHeads }, (_, headIndex) => {
            const value = tensor(point, metric)[layerIndex][headIndex][categoryIndex];
            return (
              <button
                key={headIndex}
                title={`L${layerIndex} H${headIndex} · ${displayMetric(value, metric)}`}
                aria-label={`Layer ${layerIndex}, Head ${headIndex}`}
                className={layer === layerIndex && head === headIndex ? "selected" : ""}
                style={{ background: scale(value) }}
                onClick={() => onSelect(layerIndex, headIndex)}
              />
            );
          })}
        </div>
      ))}
    </div>
  );
}

function TokenLayout({ payload, point }: { payload: Payload; point: Point }) {
  const segments = [
    { label: "证据", count: point.categoryCounts.gold_line, color: CATEGORY_COLORS.gold_line },
    { label: "无关句号", count: point.categoryCounts.irrelevant_periods, color: CATEGORY_COLORS.irrelevant_periods },
    { label: "干扰句", count: point.categoryCounts.distractor_lines, color: CATEGORY_COLORS.distractor_lines },
    { label: "问题", count: point.querySpan[1] - point.querySpan[0], color: "#7ab8ff" },
  ];
  return (
    <div className="age-layout">
      <div className="age-layout-bar">
        {point.tokenCategories.map((category, index) => (
          <i
            key={index}
            style={{
              background:
                category === "query"
                  ? "#7ab8ff"
                  : CATEGORY_COLORS[category] ?? CATEGORY_COLORS.distractor_lines,
            }}
            title={`token ${index}: ${point.decodedTokens[index]}`}
          />
        ))}
      </div>
      <div className="age-layout-legend">
        {segments.map((segment) => (
          <span key={segment.label}><i style={{ background: segment.color }} />{segment.label} {segment.count}</span>
        ))}
        <b>总长 {payload.totalTokens} tokens</b>
      </div>
    </div>
  );
}

export default function AgeDistractorPage() {
  const [payload, setPayload] = useState<Payload | null>(null);
  const [error, setError] = useState("");
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [scope, setScope] = useState<Scope>("overall");
  const [layer, setLayer] = useState(30);
  const [head, setHead] = useState(0);
  const [metric, setMetric] = useState<Metric>("mass");
  const [category, setCategory] = useState("gold_age");

  useEffect(() => {
    const controller = new AbortController();
    fetch(DATA_PATH, { signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error(`数据请求失败：HTTP ${response.status}`);
        return readPayload(response);
      })
      .then(setPayload)
      .catch((reason) => {
        if (reason?.name !== "AbortError") setError(String(reason));
      });
    return () => controller.abort();
  }, []);

  const point = payload?.points[selectedIndex] ?? null;
  const categoryIndex = payload ? payload.categoryOrder.indexOf(category) : -1;
  const currentValues = useMemo(() => {
    if (!payload || !point) return [];
    return payload.categoryOrder.map((_, index) =>
      scopedValue(point, metric, index, scope, layer, head),
    );
  }, [payload, point, metric, scope, layer, head]);

  if (!payload || !point || categoryIndex < 0) {
    return (
      <main className="age-page">
        <header className="age-topbar">
          <div className="brand-mark">A<span>9</span></div>
          <div><p className="eyebrow">QWEN3-8B CONTROLLED RETRIEVAL LAB</p><h1>固定 300-token 年龄干扰实验</h1></div>
        </header>
        <div className="age-loading">{error || "正在读取 10 条逐层逐 Head 实验数据…"}</div>
      </main>
    );
  }

  const selectedMetric = METRICS.find((item) => item.key === metric)!;
  const goldAgeMass = scopedValue(
    point,
    "mass",
    payload.categoryOrder.indexOf("gold_age"),
    scope,
    layer,
    head,
  );
  const distractorAgeMass = scopedValue(
    point,
    "mass",
    payload.categoryOrder.indexOf("distractor_ages"),
    scope,
    layer,
    head,
  );
  const categoryMassValues = payload.categoryOrder.map((_, index) =>
    scopedValue(point, "mass", index, scope, layer, head),
  );
  const categoryMeanAttentionValues = payload.categoryOrder.map((_, index) =>
    scopedValue(point, "meanAttention", index, scope, layer, head),
  );
  const categoryEnrichmentValues = payload.categoryOrder.map((_, index) =>
    scopedValue(point, "enrichment", index, scope, layer, head),
  );
  const maxCategoryMass = Math.max(...categoryMassValues, 1e-12);
  const maxCategoryMeanAttention = Math.max(...categoryMeanAttentionValues, 1e-12);

  return (
    <main className="age-page">
      <header className="age-topbar">
        <div className="brand-mark">A<span>9</span></div>
        <div>
          <p className="eyebrow">QWEN3-8B · FIXED LENGTH CAUSAL PROBE</p>
          <h1>固定 300-token 年龄干扰实验</h1>
        </div>
        <a href="/">返回长上下文 Attention Lab</a>
        <div className="status-cluster">
          <span className="status-dot live" />
          <div><b>真实 8B 数据已接入</b><small>10 条样例 · 36 层 × 32 Heads</small></div>
        </div>
      </header>

      <section className="age-control-deck">
        <div className="age-slider-control">
          <div><span>干扰信息数量</span><strong>{point.distractorCount} <small>条</small></strong></div>
          <input
            data-testid="age-distractor-slider"
            type="range"
            min="0"
            max={payload.points.length - 1}
            value={selectedIndex}
            onInput={(event) => setSelectedIndex(Number(event.currentTarget.value))}
            onChange={(event) => setSelectedIndex(Number(event.target.value))}
          />
          <div className="range-labels"><span>0 条</span><span>总长始终 300 token</span><span>9 条</span></div>
        </div>
        <div className="age-fact"><span>主证据</span><code>Xiaoming&apos;s age is <b>nine</b> years.</code></div>
        <div className="age-fact"><span>严格答案</span><code>Answer: <b>nine</b></code></div>
      </section>

      <section className="age-metric-strip">
        <MetricCard label="GOLD ANSWER PPL" value={point.goldPpl.toFixed(3)} detail={`p(nine) = ${formatPercent(point.goldProbability)}`} color="#e6a95d" />
        <MetricCard label="完整词表 MARGIN" value={signed(point.fullVocabMargin)} detail={`top-1: ${point.topToken || "∅"} · ${point.fullVocabCorrect ? "正确" : "错误"}`} color={point.fullVocabCorrect ? "#55c2a5" : "#f0746e"} />
        <MetricCard label="年龄候选 MARGIN" value={signed(point.candidateMargin)} detail={`预测 ${point.candidatePrediction} · 对手 ${point.strongestWrongCandidate.word}`} color={point.candidateCorrect ? "#55c2a5" : "#f0746e"} />
        <MetricCard label="NINE ATTENTION MASS" value={formatPercent(goldAgeMass)} detail={`${scope === "overall" ? "全模型均值" : scope === "layer" ? `Layer ${layer}` : `L${layer} H${head}`}`} color="#e6a95d" />
        <MetricCard label="干扰年龄 MASS" value={formatPercent(distractorAgeMass)} detail={`${point.distractorCount} 个年龄 token 合计`} color="#c4a7e7" />
      </section>

      <TokenLayout payload={payload} point={point} />

      <div className="age-workspace">
        <aside className="age-controls">
          <section>
            <p className="section-kicker">聚合范围</p>
            <div className="segmented">
              {(["overall", "layer", "head"] as Scope[]).map((value) => (
                <button key={value} className={scope === value ? "active" : ""} onClick={() => setScope(value)}>
                  {value === "overall" ? "全模型" : value === "layer" ? "单层" : "单 Head"}
                </button>
              ))}
            </div>
          </section>
          <section className={scope === "overall" ? "disabled-control" : ""}>
            <label>Layer <b>{layer}</b></label>
            <input type="range" min="0" max={payload.numLayers - 1} value={layer} disabled={scope === "overall"} onChange={(event) => setLayer(Number(event.target.value))} />
          </section>
          <section className={scope !== "head" ? "disabled-control" : ""}>
            <label>Head <b>{head}</b></label>
            <input type="range" min="0" max={payload.numHeads - 1} value={head} disabled={scope !== "head"} onChange={(event) => setHead(Number(event.target.value))} />
          </section>
          <section>
            <p className="section-kicker">观察指标</p>
            <div className="age-metric-list">
              {METRICS.map((item) => (
                <button key={item.key} className={metric === item.key ? "active" : ""} onClick={() => setMetric(item.key)}>
                  <b>{item.label}</b><small>{item.note}</small>
                </button>
              ))}
            </div>
          </section>
          <section>
            <p className="section-kicker">热力图类别</p>
            <div className="age-category-list">
              {payload.categoryOrder.map((item) => (
                <button key={item} className={category === item ? "active" : ""} onClick={() => setCategory(item)}>
                  <i style={{ background: CATEGORY_COLORS[item] }} />
                  <span>{CATEGORY_SHORT[item]}</span>
                  <b>{point.categoryCounts[item]}</b>
                </button>
              ))}
            </div>
          </section>
        </aside>

        <div className="age-analysis">
          <section className="age-panel age-verdict">
            <div>
              <p className="section-kicker">CURRENT SAMPLE · {point.distractorCount} DISTRACTORS</p>
              <h2>{point.fullVocabCorrect ? "模型仍把 nine 排在完整词表第一位" : `模型已经改答 ${point.topToken || "其他 token"}`}</h2>
              <p>这里所有样例总长、主证据位置和问题位置都不变；唯一被系统改变的是干扰年龄句数量，因此可以直接观察语义竞争如何改写 QK、softmax mass 和输出 margin。</p>
            </div>
            <div className="age-answer-race">
              {point.candidateScores.slice(0, 5).map((candidate) => (
                <article key={candidate.word}>
                  <span>{candidate.word}</span>
                  <div><i style={{ width: `${Math.max(1, candidate.probability / Math.max(...point.candidateScores.map((row) => row.probability)) * 100)}%` }} /></div>
                  <b>{formatPercent(candidate.probability)}</b>
                </article>
              ))}
            </div>
          </section>

          <section className="age-panel">
            <div className="panel-heading">
              <div><p className="section-kicker">CATEGORY COMPARISON</p><h2>{selectedMetric.label} · 五类 token</h2></div>
              <div className="heading-meta"><span>{selectedMetric.note}</span><b>{scope === "overall" ? "36×32 均值" : scope === "layer" ? `Layer ${layer}` : `Layer ${layer} · Head ${head}`}</b></div>
            </div>
            <div className="age-category-bars">
              {payload.categoryOrder.map((item, index) => {
                const finiteValues = currentValues.filter(Number.isFinite);
                const max = Math.max(...finiteValues, 1e-12);
                const width = metric === "bestRank"
                  ? (1 - (currentValues[index] - Math.min(...finiteValues)) / Math.max(1e-12, max - Math.min(...finiteValues))) * 100
                  : Math.abs(currentValues[index]) / Math.max(...currentValues.map((value) => Math.abs(value)), 1e-12) * 100;
                return (
                  <div key={item}>
                    <span><i style={{ background: CATEGORY_COLORS[item] }} />{CATEGORY_SHORT[item]} <small>{point.categoryCounts[item]} tokens</small></span>
                    <div><i style={{ width: `${Math.max(1, width)}%`, background: CATEGORY_COLORS[item] }} /></div>
                    <b>{displayMetric(currentValues[index], metric)}</b>
                  </div>
                );
              })}
            </div>
          </section>

          <section className="age-panel">
            <div className="panel-heading">
              <div><p className="section-kicker">MASS VS. PER-TOKEN MEAN</p><h2>Attention 总量与每个 token 的平均 Attention</h2></div>
              <div className="heading-meta"><span>右侧消除各类别 token 数量差异</span><b>平均 Attention = Mass ÷ token 数</b></div>
            </div>
            <div className="age-mass-average-grid">
              <article>
                <header><span>类别总 Attention mass</span><b>SUM</b></header>
                <div className="age-dual-bars">
                  {payload.categoryOrder.map((item, index) => (
                    <div key={item}>
                      <span><i style={{ background: CATEGORY_COLORS[item] }} />{CATEGORY_SHORT[item]}</span>
                      <div><i style={{ width: `${Math.max(point.categoryCounts[item] ? 1 : 0, categoryMassValues[index] / maxCategoryMass * 100)}%`, background: CATEGORY_COLORS[item] }} /></div>
                      <b>{point.categoryCounts[item] ? formatPercent(categoryMassValues[index]) : "—"}</b>
                    </div>
                  ))}
                </div>
              </article>
              <article>
                <header><span>类别内每个 token 的平均 Attention</span><b>MEAN</b></header>
                <div className="age-dual-bars">
                  {payload.categoryOrder.map((item, index) => (
                    <div key={item}>
                      <span><i style={{ background: CATEGORY_COLORS[item] }} />{CATEGORY_SHORT[item]}</span>
                      <div><i style={{ width: `${Math.max(point.categoryCounts[item] ? 1 : 0, categoryMeanAttentionValues[index] / maxCategoryMeanAttention * 100)}%`, background: CATEGORY_COLORS[item] }} /></div>
                      <b>
                        {point.categoryCounts[item]
                          ? <>{formatPercent(categoryMeanAttentionValues[index])}<small>{categoryEnrichmentValues[index].toFixed(2)}× uniform</small></>
                          : "—"}
                      </b>
                    </div>
                  ))}
                </div>
              </article>
            </div>
          </section>

          <section className="age-panel">
            <div className="panel-heading">
              <div><p className="section-kicker">DISTRACTOR SWEEP</p><h2>从 0 到 9 条干扰：内部信号与最终 margin</h2></div>
              <div className="heading-meta"><span>点击曲线节点切换样例</span><b>当前位置 {point.distractorCount} 条干扰</b></div>
            </div>
            <Curve payload={payload} selectedIndex={selectedIndex} categoryIndex={categoryIndex} metric={metric} scope={scope} layer={layer} head={head} onSelect={setSelectedIndex} />
          </section>

          <section className="age-panel">
            <div className="panel-heading">
              <div><p className="section-kicker">LAYER × HEAD MAP</p><h2>{CATEGORY_SHORT[category]} · {selectedMetric.label}</h2></div>
              <div className="heading-meta"><span>颜色按当前样例的 1,152 个 head 归一化</span><b>点击格子进入单 Head</b></div>
            </div>
            <Heatmap
              payload={payload}
              point={point}
              metric={metric}
              categoryIndex={categoryIndex}
              layer={layer}
              head={head}
              onSelect={(nextLayer, nextHead) => {
                setLayer(nextLayer);
                setHead(nextHead);
                setScope("head");
              }}
            />
          </section>

          <section className="age-panel age-prompt">
            <div className="panel-heading">
              <div><p className="section-kicker">PROMPT AUDIT</p><h2>第 {point.distractorCount} 条样例的实际内容</h2></div>
              <div className="heading-meta"><span>句号直接按单 token ID 重复插入</span><b>{point.fillerCount} 个无关句号</b></div>
            </div>
            <div className="age-prompt-grid">
              <article><span>主证据 · token 0 开始</span><code>{point.goldText.trim()}</code></article>
              <article><span>干扰信息 · 均匀插入</span><code>{point.distractors.length ? point.distractors.map((item) => item.text.trim()).join("\n") : "（无干扰信息）"}</code></article>
              <article><span>结尾问题 · token {point.querySpan[0]}–{point.querySpan[1] - 1}</span><code>{point.queryText.trim()}</code></article>
            </div>
          </section>

          <section className="age-findings">
            <article><span>完整词表正确</span><strong>{payload.summary.fullVocabCorrectCount}/10</strong><small>首次失败：{payload.summary.firstFullVocabFailure === null ? "未出现" : `${payload.summary.firstFullVocabFailure} 条干扰`}</small></article>
            <article><span>年龄候选正确</span><strong>{payload.summary.candidateCorrectCount}/10</strong><small>只在 one–ten 中比较</small></article>
            <article><span>nine mass ↔ margin</span><strong>{payload.summary.goldAgeMassVsMarginPearson.toFixed(3)}</strong><small>Pearson r；正值表示证据 mass 越高，输出 margin 越高</small></article>
            <article><span>干扰年龄 mass ↔ margin</span><strong>{payload.summary.distractorAgeMassVsMarginPearson.toFixed(3)}</strong><small>负值表示干扰吸走 attention 时，更容易压低正确答案</small></article>
          </section>
        </div>
      </div>

      <footer className="age-footer">
        <span>Qwen3-8B · fixed 300 tokens · gold at start · query at end · age words are single tokens</span>
        <span>QK = post-RoPE Q·K/√d · attention = softmax(QK)</span>
      </footer>
    </main>
  );
}
