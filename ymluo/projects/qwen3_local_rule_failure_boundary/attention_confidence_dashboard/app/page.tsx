"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  FailureBoundaryDashboard,
  type FailureBoundaryPayload,
} from "./failure-boundary-dashboard";

type Scope = "overall" | "layer" | "head";
type ExperimentMode = "legacy" | "single_token" | "english_single_token";
type TraceKind = "evidence" | "other" | "top100";
type ScoreSpace = "post_softmax" | "pre_softmax" | "relative_position" | "rope_pairs" | "failure_boundary";
type PreMetric = "logit" | "share";
type RelativeMetric = "ppl" | "logit" | "cosine" | "mass";
type RopePairMetric = "post" | "pre" | "delta" | "kernel";

type RopePairManifest = {
  schema_version: number;
  model: string;
  target_context_tokens: number;
  prompt_tokens: number;
  query_position: number;
  key_length: number;
  placement: string;
  role_positions: Record<string, number>;
  num_layers: number;
  num_attention_heads: number;
  num_key_value_heads: number;
  head_dim: number;
  pair_count: number;
  pair_layout: string;
  bin_size: number;
  bin_count: number;
  bin_aggregation: string;
  post_definition: string;
  pre_definition: string;
  delta_definition: string;
  rope_theta: number;
  rope_factor_label: number;
  attention_scaling: number;
  inv_freq: number[];
  files: { head_pattern: string };
  timing?: Record<string, number>;
};

type RopePairHeadPayload = {
  schema_version: number;
  layer: number;
  head: number;
  kv_head: number;
  shape: [number, number];
  post_f16_b64: string;
  pre_f16_b64: string;
  post_min: number;
  post_max: number;
  pre_min: number;
  pre_max: number;
};

type RelativePointMetrics = {
  ppl: number;
  evidenceMass: number;
  evidenceLogit: number;
  evidenceCosine: number;
  goldProbability?: number;
  evidenceKeyNorm?: number;
  queryNorm?: number;
  evidenceRank?: number;
  logsumexp?: number;
  maxLogit?: number;
  top2HeadFraction?: number;
};

type RelativePositionRow = {
  fillerTokens: number;
  promptTokens: number;
  keyLength: number;
  evidencePosition: number;
  queryPosition: number;
  fixedDistance: number;
  middleDistance: number;
  fixed: RelativePointMetrics;
  middle: RelativePointMetrics;
};

type RelativeAggregate = {
  short: Record<string, number>;
  long: Record<string, number>;
  median_ppl_factor_long_over_short: number;
  correlations_with_log_ppl: Record<string, {
    raw_spearman: number;
    length_residual_spearman: number;
    adjacent_delta_spearman: number;
  }>;
  attention_log_mass_decomposition: {
    delta_evidence_logit: number;
    delta_logsumexp: number;
    delta_geometric_log_mass: number;
    numerator_factor: number;
    competition_factor: number;
    combined_factor: number;
  };
  logsumexp_vs_log_key_length_ge_8k: { slope: number; r_squared: number; p_value: number };
  ppl_bins: Array<{ label: string; count: number; gold_ppl_median: number; gold_ppl_mean: number }>;
};

type FixedRelativePayload = {
  schemaVersion: number;
  experiment: string;
  model: string;
  condition: string;
  chain: string[];
  queryMode: string;
  seed: number;
  fillerStep: number;
  fixedBodyOverhead: number;
  rowCount: number;
  fixed: RelativeAggregate;
  middle: RelativeAggregate;
  comparison: {
    aligned_count: number;
    long_fixed_over_middle_ppl_median: number;
    long_logit_fixed_minus_middle_mean: number;
    long_cosine_fixed_minus_middle_mean: number;
    long_mass_fixed_over_middle_median: number;
  };
  rows: RelativePositionRow[];
};

type PreSoftmaxRole = {
  mean_logit: number;
  mean_cosine: number;
  mean_rank: number;
  mean_rank_percentile: number;
  top2pct_head_fraction: number;
  top100_head_fraction: number;
  mean_max_logit_gap: number;
};

type PreSoftmaxRow = {
  length: number;
  prompt_tokens: number;
  key_length: number;
  top2pct_budget: number;
  gold_ppl: number;
  mean_head_logsumexp: number;
  mean_head_max_logit: number;
  mean_query_norm: number;
  roles: Record<"hop1_result" | "hop2_input" | "hop2_result", PreSoftmaxRole>;
};

type PreSoftmaxPayload = {
  schema_version: number;
  model: string;
  aggregation: string;
  role_order: Array<"hop1_result" | "hop2_input" | "hop2_result">;
  rows: PreSoftmaxRow[];
  limitations: { full_token_logits_saved: boolean; description: string };
};

type FullPreSoftmaxManifest = {
  schema_version: number;
  model: string;
  code_mode: "english_single_token";
  placement: string;
  query_mode: string;
  target_context_tokens: number;
  prompt_tokens: number;
  key_length: number;
  num_layers: number;
  num_attention_heads: number;
  gold_codes: string[];
  role_order: string[];
  storage_dtype: string;
  probability_definition: string;
  files: {
    tokens: string;
    overall: string;
    layer_pattern: string;
    head_pattern: string;
    token_type_heatmap?: string;
  };
};

type FullPreSoftmaxTokens = {
  schema_version: number;
  key_length: number;
  storage_dtype: string;
  token_ids_u32_b64: string;
  token_text: Record<string, string>;
  body_tokens: number;
  spans: Record<string, Array<[number, number]>>;
};

type FullPreSoftmaxScope = {
  schema_version: number;
  scope: Scope;
  layer: number | null;
  head: number | null;
  key_length: number;
  storage_dtype: string;
  logits_f16_b64: string;
  probabilities_f16_b64?: string;
  logsumexp?: number;
  top_logit_positions: number[];
  min_logit: number;
  max_logit: number;
};

type FullPreDecoded = {
  logits: Float32Array;
  probabilities: Float32Array;
  tokenIds: Uint32Array;
};

type TokenTypeHeatmapPayload = {
  schema_version: number;
  model: string;
  target_context_tokens: number;
  key_length: number;
  num_layers: number;
  num_attention_heads: number;
  shape: [number, number, number];
  aggregation: { raw_logit: string; share: string };
  storage_dtype: string;
  token_ids_u32_b64: string;
  token_counts_u32_b64: string;
  token_text: Record<string, string>;
  mean_logits_f16_b64: string;
  probability_mass_f16_b64: string;
};

type TokenTypeHeatmapDecoded = {
  tokenIds: Uint32Array;
  tokenCounts: Uint32Array;
  meanLogits: Float32Array;
  probabilityMass: Float32Array;
};

type PreHeadLengthPayload = {
  schema_version: number;
  model: string;
  code_mode: string;
  gold_codes: string[];
  lengths: number[];
  key_lengths: number[];
  role_order: string[];
  num_layers: number;
  num_attention_heads: number;
  shape: [number, number, number, number];
  storage_dtype: string;
  aggregation: { role_logit: string; role_mass: string };
  role_logits_f16_b64: string;
  role_mass_f32_b64: string;
  role_best_rank_u32_b64: string;
  head_logsumexp_f16_b64: string;
  head_max_logit_f16_b64: string;
};

type PreHeadLengthDecoded = {
  roleLogits: Float32Array;
  roleMass: Float32Array;
  roleBestRank: Uint32Array;
  headLogsumexp: Float32Array;
  headMaxLogit: Float32Array;
};

type TokenTypeLengthManifest = {
  schema_version: number;
  model: string;
  experiment: string;
  normalization: string;
  raw_logit_definition: string;
  share_definition: string;
  lengths: number[];
  key_lengths: number[];
  num_layers: number;
  num_attention_heads: number;
  shape_per_token: [number, number, number];
  token_count: number;
  tokens: Record<string, {
    display: string;
    file: string;
    token_ids: number[];
    present_length_count: number;
    first_length: number | null;
    last_length: number | null;
  }>;
};

type TokenTypeLengthPayload = {
  schema_version: number;
  token: string;
  display: string;
  token_ids: number[];
  lengths: number[];
  shape: [number, number, number];
  occurrence_counts_u32_b64: string;
  mean_logits_f16_b64: string;
  probability_mass_f32_b64: string;
};

type TokenTypeLengthDecoded = {
  occurrenceCounts: Uint32Array;
  meanLogits: Float32Array;
  probabilityMass: Float32Array;
};

type PreMetricSeries = {
  key: string;
  label: string;
  color: string;
  points: Array<{ length: number; logit: number; share: number }>;
};

type FullPreBar = {
  position: number;
  text: string;
  role: string;
  logit: number | null;
  share: number;
};

const FULL_PRE_ROOT = "/data/english_single_token/full_pre_softmax_128k";
const TOKEN_TYPE_LENGTH_ROOT = "/data/english_single_token/token_type_all_lengths";
const ROPE_PAIR_ROOT = "/data/english_single_token/rope_pair_64k";
const FAILURE_BOUNDARY_FILE = "/data/failure_boundary_dense/payload.json.gz";

type TracePoint = {
  length: number;
  value: number;
  matchedPositions: number;
};

const TRACE_CACHE = new Map<string, TracePoint[]>();

const EXPERIMENT_MODES: Record<ExperimentMode, { label: string; detail: string; manifest: string; dataRoot: string }> = {
  legacy: {
    label: "多 token 编码",
    detail: "GA89-987 风格",
    manifest: "/data/manifest.json",
    dataRoot: "/data",
  },
  single_token: {
    label: "中文单 token",
    detail: "每个证据编号 = 1 token",
    manifest: "/data/single_token/manifest.json",
    dataRoot: "/data/single_token",
  },
  english_single_token: {
    label: "英文单 token · 128K",
    detail: "river → window → basket",
    manifest: "/data/english_single_token/manifest.json",
    dataRoot: "/data/english_single_token",
  },
};

type Summary = {
  length: number;
  body_tokens: number;
  prompt_tokens: number;
  gold_ppl: number;
  gold_mean_nll: number;
  overall_entropy: number;
  overall_effective_tokens: number;
  overall_role_mass: number[];
  prefill_seconds: number;
  file: string;
};

type Manifest = {
  title: string;
  model: string;
  model_config: {
    num_layers: number;
    num_attention_heads: number;
    num_key_value_heads: number;
    head_dim: number;
    rope_factor: number;
  };
  condition: string;
  code_mode?: ExperimentMode;
  placement: string;
  gold_codes: string[];
  role_order: string[];
  max_top: number;
  length_step: number;
  completed_lengths: number[];
  summaries: Summary[];
};

type LengthData = {
  model: string;
  target_context_tokens: number;
  body_tokens: number;
  prompt_tokens: number;
  gold_codes: string[];
  spans: Record<string, Array<[number, number]>>;
  attention: {
    max_top: number;
    key_length: number;
    role_order: string[];
    head_positions: number[][][];
    head_scores: number[][][];
    head_entropy: number[][];
    head_effective_tokens: number[][];
    head_role_mass: number[][][];
    head_recent512_mass: number[][];
    head_sink16_mass: number[][];
    layer_positions: number[][];
    layer_scores: number[][];
    layer_entropy: number[];
    layer_role_mass: number[][];
    overall_positions: number[];
    overall_scores: number[];
    overall_entropy: number;
    overall_effective_tokens: number;
    overall_role_mass: number[];
    overall_recent512_mass: number;
    overall_sink16_mass: number;
  };
  answer: {
    gold_answer: string;
    gold_token_count: number;
    gold_mean_nll: number;
    gold_ppl: number;
    gold_token_scores: Array<{
      index: number;
      token_id: number;
      token: string;
      probability: number;
      nll: number;
    }>;
    next_token_top5: Array<{ token_id: number; token: string; probability: number }>;
  };
  token_table: Array<[number, number, string, string]>;
  timing: { prefill_seconds: number; query_seconds: number; total_seconds: number };
};

const ROLE_LABELS: Record<string, string> = {
  start_key: "起始编号",
  hop1_result: "第一跳结果",
  hop2_input: "第二跳输入",
  hop2_result: "第二跳结果",
  rule1_line: "第一条规则",
  rule2_line: "第二条规则",
  filler: "Filler",
  query: "查询",
  other: "其它 token",
};

const ROLE_COLORS: Record<string, string> = {
  start_key: "#e6a95d",
  hop1_result: "#55c2a5",
  hop2_input: "#7ab8ff",
  hop2_result: "#f0746e",
  rule1_line: "#8bcbb8",
  rule2_line: "#a6bfe0",
  filler: "#738092",
  query: "#c4a7e7",
  other: "#424b59",
};

function demoPpl(length: number) {
  return 6.28 + 0.58 * Math.log1p(length / 9000) + 1.45 * Math.pow(length / 64000, 1.35);
}

function makeDemoManifest(): Manifest {
  const lengths = Array.from({ length: 257 }, (_, index) => index * 500);
  const roleOrder = ["start_key", "hop1_result", "hop2_input", "hop2_result", "rule1_line", "rule2_line"];
  return {
    title: "Qwen3-8B · Clean two-hop attention confidence sweep",
    model: "Qwen3-8B",
    model_config: {
      num_layers: 36,
      num_attention_heads: 32,
      num_key_value_heads: 8,
      head_dim: 128,
      rope_factor: 2,
    },
    condition: "clean",
    placement: "middle",
    gold_codes: ["GA42-318", "GB67-541", "GC81-762"],
    role_order: roleOrder,
    max_top: 100,
    length_step: 500,
    completed_lengths: lengths,
    summaries: lengths.map((length) => {
      const decay = 1 / (1 + length / 18000);
      return {
        length,
        body_tokens: Math.max(54, length),
        prompt_tokens: Math.max(54, length) + 83,
        gold_ppl: demoPpl(length),
        gold_mean_nll: Math.log(demoPpl(length)),
        overall_entropy: 4.4 + 1.7 * Math.log1p(length / 8000),
        overall_effective_tokens: 82 + length / 42,
        overall_role_mass: [0.016 * decay, 0.052 * decay, 0.044 * decay, 0.061 * decay, 0.11 * decay, 0.12 * decay],
        prefill_seconds: 0.5 + length / 6200,
        file: `data/length_${length}.json`,
      };
    }),
  };
}

function demoDistribution(
  length: number,
  layer: number,
  head: number,
  count: number,
  tokenRows: Map<number, [number, number, string, string]>,
) {
  const body = Math.max(54, length);
  const prompt = body + 83;
  const ruleStart = length === 0 ? 0 : Math.max(0, Math.floor(body / 2) - 27);
  const special: Array<[number, string, string]> = [
    [ruleStart + 8, "GA42", "start_key"],
    [ruleStart + 22, "GB67", "hop1_result"],
    [ruleStart + 35, "GB67", "hop2_input"],
    [ruleStart + 49, "GC81", "hop2_result"],
    [prompt - 1, " ", "query"],
    [0, "The", "filler"],
  ];
  const positions: number[] = [];
  const scores: number[] = [];
  const phase = (layer * 17 + head * 23 + length / 500) % 97;
  const decay = 1 / (1 + length / 18000);
  special.forEach(([position, text, role], index) => {
    const safePosition = Math.max(0, Math.min(prompt - 1, position));
    positions.push(safePosition);
    const roleBoost = role === "hop2_result" ? 0.055 : role === "hop1_result" ? 0.043 : 0.025;
    scores.push(roleBoost * decay * (0.72 + ((layer + head + index) % 7) / 20));
    tokenRows.set(safePosition, [safePosition, 1000 + safePosition, text, role]);
  });
  for (let rank = positions.length; rank < count; rank += 1) {
    const position = Math.max(0, Math.min(prompt - 1, Math.floor(((rank * 997 + phase * 131) % prompt))));
    positions.push(position);
    scores.push(0.018 * Math.exp(-rank / 19) * (0.82 + ((rank + phase) % 5) / 18));
    if (!tokenRows.has(position)) {
      const fillerText = ["the", "·", "context", " ", "record", "↵"][position % 6];
      tokenRows.set(position, [position, 2000 + position, fillerText, position >= body ? "query" : "filler"]);
    }
  }
  const order = positions.map((_, index) => index).sort((a, b) => scores[b] - scores[a]);
  return {
    positions: order.map((index) => positions[index]),
    scores: order.map((index) => scores[index]),
  };
}

function makeDemoLengthData(length: number, manifest: Manifest): LengthData {
  const { num_layers: layers, num_attention_heads: heads } = manifest.model_config;
  const tokenRows = new Map<number, [number, number, string, string]>();
  const headPositions: number[][][] = [];
  const headScores: number[][][] = [];
  const headEntropy: number[][] = [];
  const headEffective: number[][] = [];
  const headRoles: number[][][] = [];
  const headRecent: number[][] = [];
  const headSink: number[][] = [];
  const layerPositions: number[][] = [];
  const layerScores: number[][] = [];
  const layerEntropy: number[] = [];
  const layerRoles: number[][] = [];
  const decay = 1 / (1 + length / 18000);

  for (let layer = 0; layer < layers; layer += 1) {
    const positionsForLayer: number[][] = [];
    const scoresForLayer: number[][] = [];
    const entropyForLayer: number[] = [];
    const effectiveForLayer: number[] = [];
    const rolesForLayer: number[][] = [];
    const recentForLayer: number[] = [];
    const sinkForLayer: number[] = [];
    for (let head = 0; head < heads; head += 1) {
      const distribution = demoDistribution(length, layer, head, 100, tokenRows);
      positionsForLayer.push(distribution.positions);
      scoresForLayer.push(distribution.scores);
      const entropy = 4.1 + Math.log1p(length / 9000) + ((layer + head) % 9) / 18;
      entropyForLayer.push(entropy);
      effectiveForLayer.push(Math.exp(entropy));
      const specialization = 0.65 + ((layer * 3 + head * 5) % 13) / 15;
      rolesForLayer.push([
        0.012 * decay * specialization,
        0.047 * decay * specialization,
        0.039 * decay * specialization,
        0.056 * decay * specialization,
        0.1 * decay * specialization,
        0.11 * decay * specialization,
      ]);
      recentForLayer.push(0.17 + length / 640000 + (head % 5) / 100);
      sinkForLayer.push(0.04 + (layer % 6) / 100);
    }
    const aggregate = demoDistribution(length, layer, -1, 100, tokenRows);
    headPositions.push(positionsForLayer);
    headScores.push(scoresForLayer);
    headEntropy.push(entropyForLayer);
    headEffective.push(effectiveForLayer);
    headRoles.push(rolesForLayer);
    headRecent.push(recentForLayer);
    headSink.push(sinkForLayer);
    layerPositions.push(aggregate.positions);
    layerScores.push(aggregate.scores);
    layerEntropy.push(entropyForLayer.reduce((sum, value) => sum + value, 0) / heads);
    layerRoles.push(rolesForLayer[0].map((_, role) => rolesForLayer.reduce((sum, values) => sum + values[role], 0) / heads));
  }
  const overall = demoDistribution(length, -1, -1, 100, tokenRows);
  const summary = manifest.summaries.find((row) => row.length === length) ?? manifest.summaries[0];
  const bodyTokens = Math.max(54, length);
  const ruleStart = length === 0 ? 0 : Math.max(0, Math.floor(bodyTokens / 2) - 27);
  return {
    model: manifest.model,
    target_context_tokens: length,
    body_tokens: bodyTokens,
    prompt_tokens: bodyTokens + 83,
    gold_codes: manifest.gold_codes,
    spans: {
      start_key: [[ruleStart + 8, ruleStart + 9]],
      hop1_result: [[ruleStart + 22, ruleStart + 23]],
      hop2_input: [[ruleStart + 35, ruleStart + 36]],
      hop2_result: [[ruleStart + 49, ruleStart + 50]],
      rule1_line: [[ruleStart, ruleStart + 28]],
      rule2_line: [[ruleStart + 28, ruleStart + 54]],
    },
    attention: {
      max_top: 100,
      key_length: Math.max(54, length) + 83,
      role_order: manifest.role_order,
      head_positions: headPositions,
      head_scores: headScores,
      head_entropy: headEntropy,
      head_effective_tokens: headEffective,
      head_role_mass: headRoles,
      head_recent512_mass: headRecent,
      head_sink16_mass: headSink,
      layer_positions: layerPositions,
      layer_scores: layerScores,
      layer_entropy: layerEntropy,
      layer_role_mass: layerRoles,
      overall_positions: overall.positions,
      overall_scores: overall.scores,
      overall_entropy: summary.overall_entropy,
      overall_effective_tokens: summary.overall_effective_tokens,
      overall_role_mass: summary.overall_role_mass,
      overall_recent512_mass: 0.18 + length / 640000,
      overall_sink16_mass: 0.052,
    },
    answer: {
      gold_answer: manifest.gold_codes[2],
      gold_token_count: 4,
      gold_mean_nll: Math.log(summary.gold_ppl),
      gold_ppl: summary.gold_ppl,
      gold_token_scores: [0, 1, 2, 3].map((index) => ({
        index,
        token_id: 8400 + index,
        token: ["GC", "81", "-", "762"][index],
        probability: Math.exp(-Math.log(summary.gold_ppl) * (0.72 + index / 5)),
        nll: Math.log(summary.gold_ppl) * (0.72 + index / 5),
      })),
      next_token_top5: [
        { token_id: 8400, token: "GC", probability: 0.31 / Math.sqrt(summary.gold_ppl / 6.28) },
        { token_id: 8401, token: "GB", probability: 0.17 },
        { token_id: 8402, token: "GA", probability: 0.11 },
        { token_id: 8403, token: "The", probability: 0.06 },
        { token_id: 8404, token: "\n", probability: 0.04 },
      ],
    },
    token_table: Array.from(tokenRows.values()),
    timing: { prefill_seconds: summary.prefill_seconds, query_seconds: 0.12, total_seconds: summary.prefill_seconds + 0.36 },
  };
}

function compactToken(text: string) {
  if (text === " ") return "␠";
  if (text === "\n") return "↵";
  return text.replaceAll("\n", "↵").replaceAll("\t", "⇥") || "∅";
}

function formatPercent(value: number) {
  if (value >= 0.01) return `${(value * 100).toFixed(2)}%`;
  if (value >= 0.0001) return `${(value * 100).toFixed(3)}%`;
  return value.toExponential(2);
}

async function readGzipJson<T>(response: Response, file: string): Promise<T> {
  if (!file.endsWith(".gz")) return response.json() as Promise<T>;
  // The local Vite server advertises `.json.gz` with Content-Encoding: gzip,
  // so Fetch has already decompressed the body. Static hosts may instead send
  // the same file as raw gzip. Try decoded JSON first, then handle raw gzip.
  const fallback = response.clone();
  try {
    return await response.json() as T;
  } catch {
    // Continue with the untouched clone when the host returned raw gzip bytes.
  }
  if (!fallback.body || typeof DecompressionStream === "undefined") {
    throw new Error("This browser does not support gzip stream decompression.");
  }
  const stream = fallback.body.pipeThrough(new DecompressionStream("gzip"));
  return JSON.parse(await new Response(stream).text()) as T;
}

async function readLengthPayload(response: Response, file: string): Promise<LengthData> {
  return readGzipJson<LengthData>(response, file);
}

function base64Bytes(payload: string) {
  const binary = atob(payload);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
  return bytes;
}

function halfToFloat(value: number) {
  const sign = value & 0x8000 ? -1 : 1;
  const exponent = (value >>> 10) & 0x1f;
  const fraction = value & 0x03ff;
  if (exponent === 0) return sign * Math.pow(2, -14) * (fraction / 1024);
  if (exponent === 0x1f) return fraction ? Number.NaN : sign * Number.POSITIVE_INFINITY;
  return sign * Math.pow(2, exponent - 15) * (1 + fraction / 1024);
}

function decodeF16Base64(payload: string) {
  const bytes = base64Bytes(payload);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const output = new Float32Array(bytes.byteLength / 2);
  for (let index = 0; index < output.length; index += 1) {
    output[index] = halfToFloat(view.getUint16(index * 2, true));
  }
  return output;
}

function decodeF32Base64(payload: string) {
  const bytes = base64Bytes(payload);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const output = new Float32Array(bytes.byteLength / 4);
  for (let index = 0; index < output.length; index += 1) {
    output[index] = view.getFloat32(index * 4, true);
  }
  return output;
}

function decodeU32Base64(payload: string) {
  const bytes = base64Bytes(payload);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const output = new Uint32Array(bytes.byteLength / 4);
  for (let index = 0; index < output.length; index += 1) {
    output[index] = view.getUint32(index * 4, true);
  }
  return output;
}

function applyScopePattern(pattern: string, layer: number, head: number) {
  return pattern
    .replace("{layer:02d}", String(layer).padStart(2, "0"))
    .replace("{head:02d}", String(head).padStart(2, "0"));
}

function fullPreScopeFile(manifest: FullPreSoftmaxManifest, scope: Scope, layer: number, head: number) {
  if (scope === "overall") return manifest.files.overall;
  if (scope === "layer") return applyScopePattern(manifest.files.layer_pattern, layer, head);
  return applyScopePattern(manifest.files.head_pattern, layer, head);
}

function roleAtPosition(position: number, tokens: FullPreSoftmaxTokens) {
  const priority = ["hop2_result", "hop1_result", "hop2_input", "start_key", "rule2_line", "rule1_line"];
  for (const role of priority) {
    if ((tokens.spans[role] ?? []).some(([start, end]) => position >= start && position < end)) return role;
  }
  return position >= tokens.body_tokens ? "query" : "filler";
}

function lengthDataUrl(mode: ExperimentMode, summary: Summary) {
  const fileName = summary.file.split("/").pop();
  return `${EXPERIMENT_MODES[mode].dataRoot}/${fileName}`;
}

function traceDistribution(data: LengthData, scope: Scope, layer: number, head: number) {
  if (scope === "head") {
    return {
      positions: data.attention.head_positions[layer]?.[head] ?? [],
      scores: data.attention.head_scores[layer]?.[head] ?? [],
      roles: data.attention.head_role_mass[layer]?.[head] ?? [],
    };
  }
  if (scope === "layer") {
    return {
      positions: data.attention.layer_positions[layer] ?? [],
      scores: data.attention.layer_scores[layer] ?? [],
      roles: data.attention.layer_role_mass[layer] ?? [],
    };
  }
  return {
    positions: data.attention.overall_positions,
    scores: data.attention.overall_scores,
    roles: data.attention.overall_role_mass,
  };
}

function discreteMeanCos(omega: number, lowDistance: number, highDistance: number) {
  const low = Math.ceil(Math.min(lowDistance, highDistance));
  const high = Math.floor(Math.max(lowDistance, highDistance));
  const count = Math.max(1, high - low + 1);
  const denominator = Math.sin(omega / 2);
  if (Math.abs(denominator) < 1e-10) return Math.cos(omega * (low + high) / 2);
  return Math.cos(omega * (low + high) / 2) * Math.sin(count * omega / 2) / (count * denominator);
}

function robustExtent(values: Float32Array) {
  if (!values.length) return 1;
  const absolute = Array.from(values, (value) => Math.abs(value)).sort((a, b) => a - b);
  return Math.max(1e-6, absolute[Math.min(absolute.length - 1, Math.floor(absolute.length * 0.995))]);
}

function RopePairDashboard({
  manifest,
  payload,
  loading,
  error,
  layer,
  head,
  onLayerChange,
  onHeadChange,
}: {
  manifest: RopePairManifest;
  payload: RopePairHeadPayload | null;
  loading: boolean;
  error: string;
  layer: number;
  head: number;
  onLayerChange: (value: number) => void;
  onHeadChange: (value: number) => void;
}) {
  const [metric, setMetric] = useState<RopePairMetric>("post");
  const [selectedPair, setSelectedPair] = useState(40);
  const [hovered, setHovered] = useState({ bin: 0, pair: 40 });
  const heatRef = useRef<HTMLCanvasElement>(null);
  const lineRef = useRef<HTMLCanvasElement>(null);
  const decoded = useMemo(() => payload ? {
    post: decodeF16Base64(payload.post_f16_b64),
    pre: decodeF16Base64(payload.pre_f16_b64),
  } : null, [payload]);

  const values = useMemo(() => {
    const totalCells = manifest.bin_count * manifest.pair_count;
    if (metric === "kernel") {
      const output = new Float32Array(totalCells);
      for (let bin = 0; bin < manifest.bin_count; bin += 1) {
        const keyStart = bin * manifest.bin_size;
        const keyEnd = Math.min(manifest.key_length, keyStart + manifest.bin_size) - 1;
        const nearDistance = manifest.query_position - keyEnd;
        const farDistance = manifest.query_position - keyStart;
        for (let pair = 0; pair < manifest.pair_count; pair += 1) {
          output[bin * manifest.pair_count + pair] = discreteMeanCos(
            manifest.inv_freq[pair], nearDistance, farDistance,
          );
        }
      }
      return output;
    }
    if (!decoded) return null;
    if (metric === "post") return decoded.post;
    if (metric === "pre") return decoded.pre;
    const output = new Float32Array(totalCells);
    for (let index = 0; index < totalCells; index += 1) output[index] = decoded.post[index] - decoded.pre[index];
    return output;
  }, [decoded, manifest, metric]);

  const series = useMemo(() => {
    if (!values) return null;
    const selected = new Float32Array(manifest.bin_count);
    const overall = new Float32Array(manifest.bin_count);
    for (let bin = 0; bin < manifest.bin_count; bin += 1) {
      let sum = 0;
      for (let pair = 0; pair < manifest.pair_count; pair += 1) {
        const value = values[bin * manifest.pair_count + pair];
        sum += value;
        if (pair === selectedPair) selected[bin] = value;
      }
      overall[bin] = metric === "kernel" ? sum / manifest.pair_count : sum;
    }
    return { selected, overall };
  }, [manifest, metric, selectedPair, values]);

  useEffect(() => {
    const canvas = heatRef.current;
    if (!canvas || !values) return;
    const draw = () => {
      const rect = canvas.getBoundingClientRect();
      const width = Math.max(480, Math.round(rect.width));
      const height = Math.max(300, Math.round(rect.height));
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
      const context = canvas.getContext("2d");
      if (!context) return;
      context.scale(dpr, dpr);
      const style = getComputedStyle(canvas);
      const panel = style.getPropertyValue("--panel").trim();
      const positive = style.getPropertyValue("--gold").trim();
      const negative = style.getPropertyValue("--red").trim();
      const ink = style.getPropertyValue("--ink").trim();
      const faint = style.getPropertyValue("--faint").trim();
      context.fillStyle = panel;
      context.globalAlpha = 1;
      context.fillRect(0, 0, width, height);
      const margin = { left: 42, right: 8, top: 6, bottom: 24 };
      const plotWidth = width - margin.left - margin.right;
      const plotHeight = height - margin.top - margin.bottom;
      const cellWidth = plotWidth / manifest.bin_count;
      const cellHeight = plotHeight / manifest.pair_count;
      const extent = metric === "kernel" ? 1 : robustExtent(values);
      for (let bin = 0; bin < manifest.bin_count; bin += 1) {
        for (let pair = 0; pair < manifest.pair_count; pair += 1) {
          const value = values[bin * manifest.pair_count + pair];
          context.globalAlpha = 0.06 + 0.94 * Math.min(1, Math.abs(value) / extent);
          context.fillStyle = value >= 0 ? positive : negative;
          context.fillRect(
            margin.left + bin * cellWidth,
            margin.top + pair * cellHeight,
            Math.max(1, cellWidth + 0.35),
            Math.max(1, cellHeight + 0.35),
          );
        }
      }
      context.globalAlpha = 1;
      context.strokeStyle = ink;
      context.lineWidth = 1;
      context.strokeRect(margin.left, margin.top + selectedPair * cellHeight, plotWidth, Math.max(1, cellHeight));
      context.fillStyle = faint;
      context.font = "9px ui-monospace, monospace";
      context.textAlign = "right";
      [0, 8, 16, 24, 32, 40, 48, 56, 63].forEach(pair => {
        context.fillText(String(pair), margin.left - 6, margin.top + (pair + 0.7) * cellHeight);
      });
      context.textAlign = "left";
      context.fillText("Key 0", margin.left, height - 7);
      context.textAlign = "center";
      context.fillText(`Key ${Math.round(manifest.key_length / 2).toLocaleString()}`, margin.left + plotWidth / 2, height - 7);
      context.textAlign = "right";
      context.fillText(`Key ${(manifest.key_length - 1).toLocaleString()}`, width - margin.right, height - 7);
    };
    draw();
    const observer = new ResizeObserver(draw);
    observer.observe(canvas);
    return () => observer.disconnect();
  }, [manifest, metric, selectedPair, values]);

  useEffect(() => {
    const canvas = lineRef.current;
    if (!canvas || !series) return;
    const draw = () => {
      const rect = canvas.getBoundingClientRect();
      const width = Math.max(480, Math.round(rect.width));
      const height = Math.max(240, Math.round(rect.height));
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
      const context = canvas.getContext("2d");
      if (!context) return;
      context.scale(dpr, dpr);
      const style = getComputedStyle(canvas);
      const panel = style.getPropertyValue("--panel").trim();
      const line = style.getPropertyValue("--line").trim();
      const faint = style.getPropertyValue("--faint").trim();
      const overallColor = style.getPropertyValue("--green").trim();
      const pairColor = style.getPropertyValue("--red").trim();
      context.fillStyle = panel;
      context.fillRect(0, 0, width, height);
      const margin = { left: 54, right: 10, top: 12, bottom: 26 };
      const plotWidth = width - margin.left - margin.right;
      const plotHeight = height - margin.top - margin.bottom;
      const combined = new Float32Array(series.selected.length + series.overall.length);
      combined.set(series.selected);
      combined.set(series.overall, series.selected.length);
      const extent = robustExtent(combined) * 1.08;
      const x = (bin: number) => margin.left + (bin / Math.max(1, manifest.bin_count - 1)) * plotWidth;
      const y = (value: number) => margin.top + (0.5 - value / (2 * extent)) * plotHeight;
      context.strokeStyle = line;
      context.lineWidth = 1;
      [-1, -0.5, 0, 0.5, 1].forEach(fraction => {
        const yy = y(fraction * extent);
        context.beginPath(); context.moveTo(margin.left, yy); context.lineTo(width - margin.right, yy); context.stroke();
      });
      const drawSeries = (input: Float32Array, color: string, widthPx: number) => {
        context.strokeStyle = color;
        context.lineWidth = widthPx;
        context.beginPath();
        for (let bin = 0; bin < input.length; bin += 1) {
          const xx = x(bin); const yy = y(input[bin]);
          if (bin === 0) context.moveTo(xx, yy); else context.lineTo(xx, yy);
        }
        context.stroke();
      };
      drawSeries(series.selected, pairColor, 1.35);
      drawSeries(series.overall, overallColor, 2);
      context.fillStyle = faint;
      context.font = "9px ui-monospace, monospace";
      context.textAlign = "right";
      context.fillText(extent.toFixed(metric === "kernel" ? 2 : 3), margin.left - 6, margin.top + 7);
      context.fillText("0", margin.left - 6, y(0) + 3);
      context.fillText((-extent).toFixed(metric === "kernel" ? 2 : 3), margin.left - 6, height - margin.bottom);
      context.textAlign = "left";
      context.fillText("Key 0", margin.left, height - 6);
      context.textAlign = "right";
      context.fillText(`Key ${(manifest.key_length - 1).toLocaleString()}`, width - margin.right, height - 6);
    };
    draw();
    const observer = new ResizeObserver(draw);
    observer.observe(canvas);
    return () => observer.disconnect();
  }, [manifest, metric, series]);

  const moveHover = (event: React.MouseEvent<HTMLCanvasElement>, includePair: boolean) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const bin = Math.max(0, Math.min(manifest.bin_count - 1, Math.floor(((event.clientX - rect.left - 42) / Math.max(1, rect.width - 50)) * manifest.bin_count)));
    const pair = includePair
      ? Math.max(0, Math.min(manifest.pair_count - 1, Math.floor(((event.clientY - rect.top - 6) / Math.max(1, rect.height - 30)) * manifest.pair_count)))
      : selectedPair;
    setHovered({ bin, pair });
  };
  const hoverKeyStart = hovered.bin * manifest.bin_size;
  const hoverKeyEnd = Math.min(manifest.key_length, hoverKeyStart + manifest.bin_size) - 1;
  const hoverValue = values?.[hovered.bin * manifest.pair_count + hovered.pair] ?? Number.NaN;
  const hoverOverall = series?.overall[hovered.bin] ?? Number.NaN;
  const metricLabels: Record<RopePairMetric, string> = {
    post: "实际 post-RoPE pair logit",
    pre: "反旋转后的 pre-RoPE pair logit",
    delta: "RoPE 改变量：post − pre",
    kernel: "纯位置核：cos(Δωᵢ)",
  };

  return <>
    <section className="chart-panel rope-pair-panel" data-testid="rope-pair-panel">
      <div className="panel-heading">
        <div><p className="section-kicker">ROPE PAIR CONTRIBUTION · 64K QUERY</p><h2>Layer {payload?.layer ?? layer} · Head {payload?.head ?? head} · KV Head {payload?.kv_head ?? Math.floor(head / Math.max(1, manifest.num_attention_heads / manifest.num_key_value_heads))}</h2></div>
        <div className="heading-meta"><span>{loading ? "当前 head 读取中" : `${manifest.bin_count} 个位置区间 × ${manifest.pair_count} 个二维对`}</span><b>Query pos {manifest.query_position.toLocaleString()}</b></div>
      </div>
      <div className="rope-head-selectors">
        <label htmlFor="rope-layer-select"><span>Layer</span><select id="rope-layer-select" value={layer} onChange={event => onLayerChange(Number(event.target.value))}>{Array.from({ length: manifest.num_layers }, (_, value) => <option key={value} value={value}>{value}</option>)}</select></label>
        <label htmlFor="rope-head-select"><span>Head</span><select id="rope-head-select" value={head} onChange={event => onHeadChange(Number(event.target.value))}>{Array.from({ length: manifest.num_attention_heads }, (_, value) => <option key={value} value={value}>{value}</option>)}</select></label>
        <span>切换时只读取当前 head；纯位置核不需要重新请求。</span>
      </div>
      <div className="pre-metric-switch rope-pair-switch" role="group" aria-label="RoPE 分维度贡献指标">
        <span>贡献口径</span>
        {(["post", "pre", "delta", "kernel"] as RopePairMetric[]).map(value => <button key={value} className={metric === value ? "active" : ""} onClick={() => setMetric(value)}>{metricLabels[value]}</button>)}
      </div>
      <div className="rope-pair-control">
        <label htmlFor="rope-pair-select"><span>二维频率对</span><b>{selectedPair}</b><small>维度 ({selectedPair}, {selectedPair + manifest.pair_count})</small></label>
        <input id="rope-pair-select" type="range" min="0" max={manifest.pair_count - 1} value={selectedPair} onChange={event => { const value = Number(event.target.value); setSelectedPair(value); setHovered(current => ({ ...current, pair: value })); }} />
        <div className="range-labels"><span>0 · 最高频</span><span>{manifest.pair_count - 1} · 最低频</span></div>
      </div>
      {values ? <>
        <div className="rope-chart-title"><b>64 个二维对 × 前序 Key 位置</b><span>红 = 负贡献 · 金 = 正贡献 · 点击热力图行可选中频率对</span></div>
        <canvas ref={heatRef} className="rope-pair-heatmap" aria-label="64 个 RoPE 二维频率对在所有前序 Key 位置上的贡献热力图" onMouseMove={event => moveHover(event, true)} onClick={event => { moveHover(event, true); const rect = event.currentTarget.getBoundingClientRect(); setSelectedPair(Math.max(0, Math.min(manifest.pair_count - 1, Math.floor(((event.clientY - rect.top - 6) / Math.max(1, rect.height - 30)) * manifest.pair_count)))); }} />
        <div className="rope-chart-title"><b>选中频率对与整体贡献曲线</b><span><i className="rope-overall-line" />整体 {metric === "kernel" ? "64 对均值" : "64 对求和"}<i className="rope-selected-line" />Pair {selectedPair}</span></div>
        <canvas ref={lineRef} className="rope-pair-lines" aria-label={`RoPE 频率对 ${selectedPair} 与整体贡献曲线`} onMouseMove={event => moveHover(event, false)} />
        <div className="rope-pair-readout">
          <div><span>Key 区间</span><b>{hoverKeyStart.toLocaleString()}–{hoverKeyEnd.toLocaleString()}</b></div>
          <div><span>相对距离</span><b>{(manifest.query_position - hoverKeyEnd).toLocaleString()}–{(manifest.query_position - hoverKeyStart).toLocaleString()}</b></div>
          <div><span>Pair {hovered.pair}</span><b>{Number.isFinite(hoverValue) ? hoverValue.toFixed(5) : "—"}</b></div>
          <div><span>整体</span><b>{Number.isFinite(hoverOverall) ? hoverOverall.toFixed(5) : "—"}</b></div>
        </div>
      </> : <div className="trace-empty">{error ? `读取失败：${error}` : "正在读取当前 Layer / Head 的二维贡献数据…"}</div>}
    </section>
    <section className="lower-grid rope-pair-notes">
      <div className="curve-panel">
        <div className="panel-heading compact"><div><p className="section-kicker">PAIR LAYOUT</p><h3>Qwen3 split-half 配对</h3></div></div>
        <p className="panel-note">Pair i 对应维度 <code>(i, i+64)</code>，不是相邻的 <code>(2i, 2i+1)</code>。每个实际曲线点是连续 {manifest.bin_size} 个 Key token 的均值。</p>
      </div>
      <div className="token-confidence pre-explanation">
        <div className="panel-heading compact"><div><p className="section-kicker">REALTIME VS ACTIVATION</p><h3>哪些可以实时计算</h3></div></div>
        <ul>
          <li><b>纯位置核</b><span>只用 <code>inv_freq</code> 与相对距离，浏览器实时计算；所有层/head 相同。</span></li>
          <li><b>实际贡献</b><span>使用服务器捕获的 Q/K 激活；切换 Layer / Head 时只懒加载当前约百 KB 文件。</span></li>
          <li><b>整体曲线</b><span>实际模式对 64 对求和，可与该 head 的 raw attention logit 对齐。</span></li>
        </ul>
      </div>
    </section>
  </>;
}

function traceSpec(token: string, goldCodes: string[]): { kind: TraceKind; roles: string[] } {
  const normalized = token.trim();
  if (normalized.toUpperCase() === "OTHER") return { kind: "other", roles: [] };
  if (normalized === goldCodes[0]) return { kind: "evidence", roles: ["start_key"] };
  if (normalized === goldCodes[1]) return { kind: "evidence", roles: ["hop1_result", "hop2_input"] };
  if (normalized === goldCodes[2]) return { kind: "evidence", roles: ["hop2_result"] };
  return { kind: "top100", roles: [] };
}

function preEvidenceRoles(token: string) {
  const normalized = token.trim().toLocaleLowerCase();
  if (normalized === "river") return [{ role: "start_key", label: "river · 起始证据" }];
  if (normalized === "window") return [
    { role: "hop1_result", label: "window · 第一跳结果" },
    { role: "hop2_input", label: "window · 第二跳输入" },
  ];
  if (normalized === "basket") return [{ role: "hop2_result", label: "basket · 最终证据" }];
  return [];
}

function traceValue(
  data: LengthData,
  token: string,
  scope: Scope,
  layer: number,
  head: number,
  topX: number,
) {
  const distribution = traceDistribution(data, scope, layer, head);
  const spec = traceSpec(token, data.gold_codes);
  if (spec.kind === "other") {
    return {
      value: Math.max(0, 1 - distribution.scores.slice(0, topX).reduce((sum, value) => sum + value, 0)),
      matchedPositions: Math.max(0, data.attention.key_length - topX),
    };
  }
  if (spec.kind === "evidence") {
    const value = spec.roles.reduce((sum, role) => {
      const index = data.attention.role_order.indexOf(role);
      return sum + (index >= 0 ? distribution.roles[index] ?? 0 : 0);
    }, 0);
    const matchedPositions = spec.roles.reduce(
      (sum, role) => sum + (data.spans[role] ?? []).reduce((total, [start, end]) => total + Math.max(0, end - start), 0),
      0,
    );
    return { value, matchedPositions };
  }

  const normalized = token.trim();
  const tokenMap = new Map(data.token_table.map((row) => [row[0], row]));
  let value = 0;
  let matchedPositions = 0;
  distribution.positions.forEach((position, index) => {
    const text = tokenMap.get(position)?.[2] ?? "";
    if (text === normalized || text.trim() === normalized) {
      value += distribution.scores[index] ?? 0;
      matchedPositions += 1;
    }
  });
  return { value, matchedPositions };
}

function TokenTraceCurve({
  points,
  selectedLength,
  token,
  kind,
}: {
  points: TracePoint[];
  selectedLength: number;
  token: string;
  kind: TraceKind;
}) {
  if (!points.length) return <div className="trace-empty">输入 token 后开始读取全部长度点。</div>;
  const width = 900;
  const height = 230;
  const left = 58;
  const right = 18;
  const top = 16;
  const bottom = 34;
  const maxLength = Math.max(...points.map((point) => point.length), 1);
  const maxValue = Math.max(...points.map((point) => point.value), 1e-9);
  const xy = (point: TracePoint) => ({
    x: left + (point.length / maxLength) * (width - left - right),
    y: height - bottom - (point.value / maxValue) * (height - top - bottom),
  });
  const path = points.map((point, index) => {
    const current = xy(point);
    return `${index === 0 ? "M" : "L"}${current.x.toFixed(2)},${current.y.toFixed(2)}`;
  }).join(" ");
  const active = points.find((point) => point.length === selectedLength) ?? points[0];
  const activePosition = xy(active);
  const color = kind === "other" ? ROLE_COLORS.other : kind === "evidence" ? ROLE_COLORS.hop1_result : ROLE_COLORS.hop2_input;
  const xTicks = Array.from({ length: 5 }, (_, index) => Math.round((maxLength * index) / 4));
  const yTicks = [0, maxValue / 2, maxValue];
  return (
    <div className="token-trace-chart" style={{ "--trace-color": color } as React.CSSProperties}>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${token} attention 随序列长度变化`}>
        <title>{token} attention trace</title>
        <desc>当前范围内从 short 到 {Math.round(maxLength / 1000)}K 的 attention softmax 质量。</desc>
        {yTicks.map((value) => {
          const y = height - bottom - (value / maxValue) * (height - top - bottom);
          return <g key={value}><line x1={left} y1={y} x2={width - right} y2={y} className="trace-grid" /><text x={left - 8} y={y + 3} textAnchor="end">{formatPercent(value)}</text></g>;
        })}
        {xTicks.map((value) => {
          const x = left + (value / maxLength) * (width - left - right);
          return <g key={value}><line x1={x} y1={top} x2={x} y2={height - bottom} className="trace-grid vertical" /><text x={x} y={height - 12} textAnchor="middle">{value === 0 ? "short" : `${value / 1000}K`}</text></g>;
        })}
        <path d={path} className="trace-line" />
        {points.map((point) => {
          const position = xy(point);
          return <circle key={point.length} cx={position.x} cy={position.y} r="2" className="trace-point"><title>{`${point.length.toLocaleString()} tokens · ${formatPercent(point.value)}`}</title></circle>;
        })}
        <circle cx={activePosition.x} cy={activePosition.y} r="5" className="trace-active"><title>{`${active.length.toLocaleString()} tokens · ${formatPercent(active.value)}`}</title></circle>
      </svg>
    </div>
  );
}

function PplCurve({ summaries, selectedLength }: { summaries: Summary[]; selectedLength: number }) {
  const width = 620;
  const height = 128;
  const values = summaries.map((row) => row.gold_ppl);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const useLogScale = max / Math.max(min, 1e-9) > 10;
  const scale = (value: number) => useLogScale ? Math.log(Math.max(value, 1e-9)) : value;
  const scaledMin = scale(min);
  const scaledMax = scale(max);
  const maxLength = Math.max(...summaries.map((row) => row.length), 1);
  const point = (row: Summary) => ({
    x: 12 + (row.length / maxLength) * (width - 24),
    y: height - 18 - ((scale(row.gold_ppl) - scaledMin) / Math.max(scaledMax - scaledMin, 0.001)) * (height - 36),
  });
  const path = summaries.map((row, index) => {
    const p = point(row);
    return `${index === 0 ? "M" : "L"}${p.x.toFixed(1)},${p.y.toFixed(1)}`;
  }).join(" ");
  const selected = summaries.find((row) => row.length === selectedLength) ?? summaries[0];
  const active = point(selected);
  return (
    <div className="curve-wrap" aria-label={`Gold final PPL 随长度变化曲线${useLogScale ? "（对数纵轴）" : ""}`}>
      <svg viewBox={`0 0 ${width} ${height}`} role="img">
        <line x1="12" y1={height - 18} x2={width - 12} y2={height - 18} className="axis-line" />
        <path d={path} className="ppl-line" />
        <circle cx={active.x} cy={active.y} r="5" className="ppl-dot" />
      </svg>
      <div className="curve-labels"><span>short</span><span>{Math.round(maxLength / 2000)}K</span><span>{Math.round(maxLength / 1000)}K{useLogScale ? " · log y" : ""}</span></div>
    </div>
  );
}

function PreSoftmaxLogitCurve({ rows, selectedLength }: { rows: PreSoftmaxRow[]; selectedLength: number }) {
  if (!rows.length) return <div className="trace-empty">正在读取 pre-softmax 诊断。</div>;
  const width = 900;
  const height = 230;
  const roles = ["hop1_result", "hop2_input", "hop2_result"] as const;
  const values = rows.flatMap((row) => roles.map((role) => row.roles[role].mean_logit));
  const minValue = Math.min(0, ...values);
  const maxValue = Math.max(0, ...values);
  const span = Math.max(0.001, maxValue - minValue);
  const maxLength = Math.max(1, rows.at(-1)?.length ?? 1);
  const point = (length: number, value: number) => ({
    x: 28 + (length / maxLength) * (width - 48),
    y: 12 + ((maxValue - value) / span) * (height - 42),
  });
  const zeroY = point(0, 0).y;
  const selected = rows.find((row) => row.length === selectedLength) ?? rows[0];
  return (
    <div className="pre-logit-curve" aria-label="三类证据的 pre-softmax QK logit 随长度变化">
      <svg viewBox={`0 0 ${width} ${height}`} role="img">
        <line x1="28" y1={zeroY} x2={width - 20} y2={zeroY} className="pre-zero-line" />
        {[minValue, maxValue].map((value) => {
          const y = point(0, value).y;
          return <g key={value}><line x1="28" y1={y} x2={width - 20} y2={y} className="trace-grid" /><text x="2" y={y + 3}>{value.toFixed(1)}</text></g>;
        })}
        {roles.map((role) => {
          const path = rows.map((row, index) => {
            const current = point(row.length, row.roles[role].mean_logit);
            return `${index === 0 ? "M" : "L"}${current.x.toFixed(2)},${current.y.toFixed(2)}`;
          }).join(" ");
          const active = point(selected.length, selected.roles[role].mean_logit);
          return <g key={role}>
            <path d={path} className="pre-logit-line" style={{ stroke: ROLE_COLORS[role] }} />
            <circle cx={active.x} cy={active.y} r="4" className="pre-logit-dot" style={{ stroke: ROLE_COLORS[role] }} />
          </g>;
        })}
      </svg>
      <div className="pre-curve-footer">
        <span>short</span>
        <div>{roles.map((role) => <b key={role}><i style={{ background: ROLE_COLORS[role] }} />{ROLE_LABELS[role]}</b>)}</div>
        <span>{Math.round(maxLength / 1000)}K</span>
      </div>
    </div>
  );
}

function SelectedPreSoftmaxTokenCurve({
  rows,
  selectedLength,
  token,
}: {
  rows: PreSoftmaxRow[];
  selectedLength: number;
  token: string;
}) {
  const normalized = token.trim().toLocaleLowerCase();
  const series = normalized === "window"
    ? ([
        { role: "hop1_result" as const, label: "window · 第一跳结果" },
        { role: "hop2_input" as const, label: "window · 第二跳输入" },
      ])
    : normalized === "basket"
      ? [{ role: "hop2_result" as const, label: "basket · 最终证据" }]
      : [];
  if (!series.length) {
    return <div className="trace-empty compact">现有跨长度实验只保存了 <code>window</code> 和 <code>basket</code> 的证据位置 QK logit；“{token || "—"}”只有 128K 全量点。</div>;
  }
  if (!rows.length) return <div className="trace-empty">正在读取跨长度 pre-softmax 数据。</div>;
  const width = 900;
  const height = 230;
  const values = rows.flatMap((row) => series.map(({ role }) => row.roles[role].mean_logit));
  const minValue = Math.min(0, ...values);
  const maxValue = Math.max(0, ...values);
  const valueSpan = Math.max(0.001, maxValue - minValue);
  const maxLength = Math.max(1, rows.at(-1)?.length ?? 1);
  const point = (length: number, value: number) => ({
    x: 54 + (length / maxLength) * (width - 74),
    y: 15 + ((maxValue - value) / valueSpan) * (height - 48),
  });
  const selected = rows.find((row) => row.length === selectedLength) ?? rows[0];
  const yTicks = [minValue, (minValue + maxValue) / 2, maxValue];
  return <div className="selected-pre-token-curve" aria-label={`${token} 的 pre-softmax QK logit 随长度变化`}>
    <svg viewBox={`0 0 ${width} ${height}`} role="img">
      {yTicks.map((value) => {
        const y = point(0, value).y;
        return <g key={value}><line x1="54" y1={y} x2={width - 20} y2={y} className="trace-grid" /><text x="46" y={y + 3} textAnchor="end">{value.toFixed(2)}</text></g>;
      })}
      {series.map(({ role, label }) => {
        const path = rows.map((row, index) => {
          const current = point(row.length, row.roles[role].mean_logit);
          return `${index === 0 ? "M" : "L"}${current.x.toFixed(2)},${current.y.toFixed(2)}`;
        }).join(" ");
        const active = point(selected.length, selected.roles[role].mean_logit);
        return <g key={role}>
          <path d={path} className="pre-logit-line" style={{ stroke: ROLE_COLORS[role] }} />
          <circle cx={active.x} cy={active.y} r="5" className="pre-logit-dot" style={{ stroke: ROLE_COLORS[role] }}><title>{`${selected.length.toLocaleString()} tokens · ${label} · ${selected.roles[role].mean_logit.toFixed(4)}`}</title></circle>
        </g>;
      })}
      <text x="54" y={height - 9}>short</text>
      <text x={width - 20} y={height - 9} textAnchor="end">{Math.round(maxLength / 1000)}K</text>
    </svg>
    <div className="selected-pre-legend">
      {series.map(({ role, label }) => <span key={role}><i style={{ background: ROLE_COLORS[role] }} />{label}<b>{selected.roles[role].mean_logit.toFixed(4)}</b></span>)}
    </div>
  </div>;
}

function ScopedPreMetricCurve({
  series,
  selectedLength,
  metric,
}: {
  series: PreMetricSeries[];
  selectedLength: number;
  metric: PreMetric;
}) {
  if (!series.length) return <div className="trace-empty compact">这个 token 没有跨长度逐 head 诊断；可在 128K 查看它的全量 token-type 热力图。</div>;
  const width = 900;
  const height = 230;
  const readValue = (point: PreMetricSeries["points"][number]) => metric === "logit" ? point.logit : point.share;
  const values = series.flatMap((row) => row.points.map(readValue));
  const minValue = metric === "logit" ? Math.min(0, ...values) : 0;
  const maxValue = Math.max(...values, metric === "logit" ? 0 : 1e-12);
  const valueSpan = Math.max(1e-12, maxValue - minValue);
  const maxLength = Math.max(1, ...series.flatMap((row) => row.points.map((point) => point.length)));
  const xy = (length: number, value: number) => ({
    x: 58 + (length / maxLength) * (width - 78),
    y: 14 + ((maxValue - value) / valueSpan) * (height - 48),
  });
  const yTicks = [minValue, (minValue + maxValue) / 2, maxValue];
  return <div className="selected-pre-token-curve" aria-label={`pre-softmax ${metric} 随长度变化`}>
    <svg viewBox={`0 0 ${width} ${height}`} role="img">
      {yTicks.map((value, index) => {
        const y = xy(0, value).y;
        return <g key={`${value}-${index}`}><line x1="58" y1={y} x2={width - 20} y2={y} className="trace-grid" /><text x="50" y={y + 3} textAnchor="end">{metric === "logit" ? value.toFixed(2) : formatPercent(value)}</text></g>;
      })}
      {series.map((row) => {
        const path = row.points.map((point, index) => {
          const position = xy(point.length, readValue(point));
          return `${index === 0 ? "M" : "L"}${position.x.toFixed(2)},${position.y.toFixed(2)}`;
        }).join(" ");
        const activePoint = row.points.find((point) => point.length === selectedLength);
        const active = activePoint ? xy(activePoint.length, readValue(activePoint)) : null;
        return <g key={row.key}>
          <path d={path} className="pre-logit-line" style={{ stroke: row.color }} />
          {activePoint && active && <circle cx={active.x} cy={active.y} r="5" className="pre-logit-dot" style={{ stroke: row.color }}><title>{`${activePoint.length.toLocaleString()} tokens · ${row.label} · ${metric === "logit" ? activePoint.logit.toFixed(4) : formatPercent(activePoint.share)}`}</title></circle>}
        </g>;
      })}
      <text x="58" y={height - 9}>short</text>
      <text x={width - 20} y={height - 9} textAnchor="end">{Math.round(maxLength / 1000)}K</text>
    </svg>
    <div className="selected-pre-legend">
      {series.map((row) => {
        const active = row.points.find((point) => point.length === selectedLength);
        return <span key={row.key}><i style={{ background: row.color }} />{row.label}<b>{active ? metric === "logit" ? active.logit.toFixed(4) : formatPercent(active.share) : "当前长度无"}</b></span>;
      })}
    </div>
  </div>;
}

const RELATIVE_METRICS: Record<RelativeMetric, {
  label: string;
  short: string;
  value: (row: RelativePointMetrics) => number;
  transform: (value: number) => number;
  format: (value: number) => string;
}> = {
  ppl: {
    label: "Gold final PPL",
    short: "PPL",
    value: (row) => row.ppl,
    transform: (value) => Math.log10(Math.max(value, 1e-6)),
    format: (value) => value >= 100 ? value.toFixed(0) : value.toFixed(2),
  },
  logit: {
    label: "证据 raw QK logit",
    short: "RAW LOGIT",
    value: (row) => row.evidenceLogit,
    transform: (value) => value,
    format: (value) => value.toFixed(3),
  },
  cosine: {
    label: "证据 Q/K cosine",
    short: "COSINE",
    value: (row) => row.evidenceCosine,
    transform: (value) => value,
    format: (value) => value.toFixed(4),
  },
  mass: {
    label: "证据 attention mass",
    short: "ATTN MASS",
    value: (row) => row.evidenceMass,
    transform: (value) => value,
    format: (value) => formatPercent(value),
  },
};

function RelativeComparisonCurve({
  rows,
  selectedLength,
  metric,
}: {
  rows: RelativePositionRow[];
  selectedLength: number;
  metric: RelativeMetric;
}) {
  if (!rows.length) return <div className="trace-empty">固定相对距离数据读取中。</div>;
  const spec = RELATIVE_METRICS[metric];
  const width = 980;
  const height = 280;
  const pad = { left: 62, right: 22, top: 22, bottom: 34 };
  const plotWidth = width - pad.left - pad.right;
  const plotHeight = height - pad.top - pad.bottom;
  const maxLength = Math.max(...rows.map((row) => row.fillerTokens), 1);
  const rawValues = rows.flatMap((row) => [spec.transform(spec.value(row.fixed)), spec.transform(spec.value(row.middle))]);
  let minValue = Math.min(...rawValues);
  let maxValue = Math.max(...rawValues);
  const margin = Math.max((maxValue - minValue) * 0.08, metric === "cosine" ? 0.005 : 0.02);
  minValue -= margin;
  maxValue += margin;
  const x = (value: number) => pad.left + (value / maxLength) * plotWidth;
  const y = (value: number) => pad.top + (1 - (spec.transform(value) - minValue) / Math.max(1e-9, maxValue - minValue)) * plotHeight;
  const path = (placement: "fixed" | "middle") => rows.map((row, index) => `${index ? "L" : "M"}${x(row.fillerTokens).toFixed(2)},${y(spec.value(row[placement])).toFixed(2)}`).join(" ");
  const selected = rows.find((row) => row.fillerTokens === selectedLength) ?? rows[0];
  const gridFractions = [0, .25, .5, .75, 1];
  return <div className="relative-curve" data-testid="fixed-relative-curve">
    <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${spec.label}，固定相对距离与中部放置对照`}>
      {gridFractions.map((fraction) => {
        const gridY = pad.top + plotHeight * fraction;
        const transformed = maxValue - (maxValue - minValue) * fraction;
        const labelValue = metric === "ppl" ? 10 ** transformed : transformed;
        return <g key={`y-${fraction}`}><line className="relative-grid" x1={pad.left} x2={width - pad.right} y1={gridY} y2={gridY} /><text x={pad.left - 10} y={gridY + 3} textAnchor="end">{spec.format(labelValue)}</text></g>;
      })}
      {[0, 32000, 64000, 96000, 128000].map((tick) => <g key={`x-${tick}`}><line className="relative-grid vertical" x1={x(tick)} x2={x(tick)} y1={pad.top} y2={height - pad.bottom} /><text x={x(tick)} y={height - 10} textAnchor="middle">{tick === 0 ? "0" : `${tick / 1000}K`}</text></g>)}
      <path className="relative-line fixed" d={path("fixed")} />
      <path className="relative-line middle" d={path("middle")} />
      <line className="relative-selected-line" x1={x(selected.fillerTokens)} x2={x(selected.fillerTokens)} y1={pad.top} y2={height - pad.bottom} />
      <circle className="relative-dot fixed" cx={x(selected.fillerTokens)} cy={y(spec.value(selected.fixed))} r="5" />
      <circle className="relative-dot middle" cx={x(selected.fillerTokens)} cy={y(spec.value(selected.middle))} r="5" />
    </svg>
    <div className="relative-curve-legend">
      <span><i className="fixed" />相对距离固定 328 <b>{spec.format(spec.value(selected.fixed))}</b></span>
      <span><i className="middle" />证据放在中部 <b>{spec.format(spec.value(selected.middle))}</b></span>
      <small>{metric === "ppl" ? "纵轴为 log10 尺度" : "纵轴为实际数值"}</small>
    </div>
  </div>;
}

function FixedRelativeDashboard({
  payload,
  selectedLength,
}: {
  payload: FixedRelativePayload | null;
  selectedLength: number;
}) {
  const [metric, setMetric] = useState<RelativeMetric>("ppl");
  if (!payload) return <div className="relative-dashboard"><div className="trace-empty">正在读取固定相对距离实验结果。</div></div>;
  const row = payload.rows.find((item) => item.fillerTokens === selectedLength) ?? payload.rows[0];
  const fixed = payload.fixed;
  const middle = payload.middle;
  const decomposition = fixed.attention_log_mass_decomposition;
  const currentPplRatio = row.middle.ppl > 0 ? row.fixed.ppl / row.middle.ppl : 0;
  const currentMassRatio = row.middle.evidenceMass > 0 ? row.fixed.evidenceMass / row.middle.evidenceMass : 0;
  const binMax = Math.max(...middle.ppl_bins.map((bin) => Math.log10(bin.gold_ppl_median + 1)), 1);
  const correlationRows = [
    ["attention mass", fixed.correlations_with_log_ppl.mean_evidence_mass.length_residual_spearman, "证据获得的概率质量"],
    ["Q/K cosine", fixed.correlations_with_log_ppl.mean_evidence_cosine.length_residual_spearman, "证据方向匹配"],
    ["raw QK logit", fixed.correlations_with_log_ppl.mean_evidence_logit.length_residual_spearman, "softmax 前证据分数"],
  ] as const;
  return <div className="relative-dashboard" data-testid="fixed-relative-dashboard">
    <section className="relative-hero">
      <div className="relative-verdict">
        <p className="section-kicker">CONTROLLED POSITION EXPERIMENT · 257 LENGTHS</p>
        <h2>固定相对距离后，QK 方向没有退化；但 PPL 仍变坏 <em>{fixed.median_ppl_factor_long_over_short.toFixed(2)}×</em></h2>
        <p>把证据到查询的距离恒定为 328 tokens，隔离原实验里不断增长的 RoPE 相对位置效应。长上下文的证据 logit 与 cosine 反而上升，剩余退化主要来自 softmax 竞争与后续 Value / residual / readout。</p>
      </div>
      <div className="relative-layout-diagram" aria-label="固定相对距离数据布局">
        <div className="prefix-block"><span>可变 prefix filler</span><b>{row.fillerTokens.toLocaleString()}</b></div>
        <div className="evidence-block"><span>证据链</span><b>34</b></div>
        <div className="gap-block"><span>固定 filler</span><b>256</b></div>
        <div className="query-block"><span>query</span><b>1</b></div>
        <i>evidence → query = 328 tokens，所有长度不变</i>
      </div>
    </section>

    <section className="relative-current-grid">
      <article><span>当前 filler</span><strong>{row.fillerTokens.toLocaleString()}</strong><small>prompt {row.promptTokens.toLocaleString()}</small></article>
      <article><span>固定距离 PPL</span><strong>{row.fixed.ppl.toFixed(3)}</strong><small>gold prob {formatPercent(row.fixed.goldProbability ?? 0)}</small></article>
      <article><span>中部放置 PPL</span><strong>{row.middle.ppl.toFixed(3)}</strong><small>固定 / 中部 = {currentPplRatio.toFixed(3)}×</small></article>
      <article><span>方向优势</span><strong>{(row.fixed.evidenceLogit - row.middle.evidenceLogit >= 0 ? "+" : "") + (row.fixed.evidenceLogit - row.middle.evidenceLogit).toFixed(3)}</strong><small>fixed − middle raw logit</small></article>
      <article><span>证据质量比</span><strong>{currentMassRatio.toFixed(2)}×</strong><small>fixed / middle attention mass</small></article>
    </section>

    <section className="relative-chart-panel">
      <div className="panel-heading">
        <div><p className="section-kicker">FIXED 328 vs MIDDLE</p><h2>随 filler 长度变化的逐点对照</h2></div>
        <div className="relative-metric-tabs" role="group" aria-label="固定相对距离对照指标">
          {(Object.keys(RELATIVE_METRICS) as RelativeMetric[]).map((value) => <button key={value} className={metric === value ? "active" : ""} onClick={() => setMetric(value)}>{RELATIVE_METRICS[value].short}</button>)}
        </div>
      </div>
      <RelativeComparisonCurve rows={payload.rows} selectedLength={selectedLength} metric={metric} />
      {row.fillerTokens === 0 && <p className="relative-outlier-note"><b>0-token 结构异常点：</b>固定布局在 filler=0 时 PPL={row.fixed.ppl.toFixed(1)}，而 filler=500 时迅速恢复到 6.93。它反映 prompt 起始边界结构，不代表长度趋势。</p>}
    </section>

    <section className="relative-mechanism-grid">
      <article className="mechanism-card direction">
        <p className="section-kicker">NUMERATOR · QK DIRECTION</p>
        <strong>×{decomposition.numerator_factor.toFixed(2)}</strong>
        <span>短 → 长 raw logit <b>+{decomposition.delta_evidence_logit.toFixed(3)}</b></span>
        <p>证据分子的 exp(logit) 变强，不支持“证据 key 或 query 方向在固定距离下系统性转错”。</p>
      </article>
      <article className="mechanism-card competition">
        <p className="section-kicker">DENOMINATOR · COMPETITION</p>
        <strong>×{decomposition.competition_factor.toFixed(3)}</strong>
        <span>logsumexp <b>+{decomposition.delta_logsumexp.toFixed(3)}</b></span>
        <p>候选 token 增多把同一证据的 softmax 份额稀释约 {(1 / decomposition.competition_factor).toFixed(1)}×。</p>
      </article>
      <article className="mechanism-card net">
        <p className="section-kicker">NET ATTENTION MASS</p>
        <strong>×{decomposition.combined_factor.toFixed(3)}</strong>
        <span>几何平均质量 <b>{((decomposition.combined_factor - 1) * 100).toFixed(1)}%</b></span>
        <p>更强的 QK 分子抵消了大部分竞争，但没有完全抵消；随后还会经历 V、残差流与答案 readout。</p>
      </article>
    </section>

    <section className="relative-lower-grid">
      <article className="relative-bin-panel">
        <div className="panel-heading compact"><div><p className="section-kicker">PPL BY LENGTH BIN</p><h3>固定距离显著缓解，但没有消除长度退化</h3></div><span>中位数 · 对数宽度</span></div>
        <div className="relative-bins">
          {fixed.ppl_bins.map((bin, index) => {
            const middleBin = middle.ppl_bins[index];
            return <div className="relative-bin-row" key={bin.label}>
              <b>{bin.label}</b>
              <div><span className="fixed" style={{ width: `${Math.log10(bin.gold_ppl_median + 1) / binMax * 100}%` }} /><em>{bin.gold_ppl_median.toFixed(2)}</em></div>
              <div><span className="middle" style={{ width: `${Math.log10(middleBin.gold_ppl_median + 1) / binMax * 100}%` }} /><em>{middleBin.gold_ppl_median.toFixed(2)}</em></div>
            </div>;
          })}
        </div>
        <div className="relative-bin-legend"><span><i className="fixed" />固定 328</span><span><i className="middle" />中部放置</span></div>
      </article>
      <article className="relative-correlation-panel">
        <div className="panel-heading compact"><div><p className="section-kicker">LENGTH-RESIDUAL SPEARMAN</p><h3>控制长度趋势后，什么和 PPL 同步变化</h3></div></div>
        <div className="relative-correlations">
          {correlationRows.map(([label, value, detail]) => <div key={label}><div><b>{label}</b><strong>{value.toFixed(3)}</strong></div><span><i style={{ width: `${Math.abs(value) * 100}%` }} /></span><small>{detail}越强，PPL 越低</small></div>)}
        </div>
      </article>
    </section>

    <section className="relative-conclusion">
      <p className="section-kicker">ANSWER TO THE DIRECTION QUESTION</p>
      <div><b>原 middle 实验的方向退化</b><span>主要是 evidence–query 相对距离不断变化带来的系统性位置项；固定到 328 后，长段 raw logit 比 middle 高 {payload.comparison.long_logit_fixed_minus_middle_mean.toFixed(2)}，cosine 高 {payload.comparison.long_cosine_fixed_minus_middle_mean.toFixed(3)}。</span></div>
      <div><b>固定距离后的剩余退化</b><span>不是“证据方向越来越错”这一条机制可以解释的。logsumexp 随 log key length 的斜率为 {fixed.logsumexp_vs_log_key_length_ge_8k.slope.toFixed(3)}（R²={fixed.logsumexp_vs_log_key_length_ge_8k.r_squared.toFixed(3)}），说明竞争池增长仍然非常稳定。</span></div>
      <div><b>最严谨的当前结论</b><span>相对位置增长是 QK 方向退化的主因；softmax 竞争与下游状态传播共同决定固定距离下仍有 {fixed.median_ppl_factor_long_over_short.toFixed(2)}× PPL 恶化。</span></div>
    </section>
  </div>;
}

function Metric({ label, value, detail, accent }: { label: string; value: string; detail: string; accent?: string }) {
  return (
    <div className="metric" style={{ "--metric-accent": accent ?? "#55c2a5" } as React.CSSProperties}>
      <span className="metric-label">{label}</span>
      <strong>{value}</strong>
      <span className="metric-detail">{detail}</span>
    </div>
  );
}

export default function Home() {
  const [experimentMode, setExperimentMode] = useState<ExperimentMode>("english_single_token");
  const [scoreSpace, setScoreSpace] = useState<ScoreSpace>("failure_boundary");
  const [manifest, setManifest] = useState<Manifest>(() => makeDemoManifest());
  const [isDemo, setIsDemo] = useState(true);
  const [failureBoundary, setFailureBoundary] = useState<FailureBoundaryPayload | null>(null);
  const [failureBoundaryError, setFailureBoundaryError] = useState("");
  const [failureLengthIndex, setFailureLengthIndex] = useState(14);
  const [relativePosition, setRelativePosition] = useState<FixedRelativePayload | null>(null);
  const [relativePositionError, setRelativePositionError] = useState("");
  const [preSoftmax, setPreSoftmax] = useState<PreSoftmaxPayload | null>(null);
  const [preSoftmaxError, setPreSoftmaxError] = useState("");
  const [ropePairManifest, setRopePairManifest] = useState<RopePairManifest | null>(null);
  const [ropePairHead, setRopePairHead] = useState<RopePairHeadPayload | null>(null);
  const [ropePairLoading, setRopePairLoading] = useState(false);
  const [ropePairError, setRopePairError] = useState("");
  const [fullPreManifest, setFullPreManifest] = useState<FullPreSoftmaxManifest | null>(null);
  const [fullPreTokens, setFullPreTokens] = useState<FullPreSoftmaxTokens | null>(null);
  const [fullPreScope, setFullPreScope] = useState<FullPreSoftmaxScope | null>(null);
  const [tokenHeatmap, setTokenHeatmap] = useState<TokenTypeHeatmapPayload | null>(null);
  const [tokenHeatmapError, setTokenHeatmapError] = useState("");
  const [preHeadLength, setPreHeadLength] = useState<PreHeadLengthPayload | null>(null);
  const [preHeadLengthError, setPreHeadLengthError] = useState("");
  const [tokenTypeLengthManifest, setTokenTypeLengthManifest] = useState<TokenTypeLengthManifest | null>(null);
  const [tokenTypeLength, setTokenTypeLength] = useState<TokenTypeLengthPayload | null>(null);
  const [tokenTypeLengthLoading, setTokenTypeLengthLoading] = useState(false);
  const [tokenTypeLengthError, setTokenTypeLengthError] = useState("");
  const [fullPreLoading, setFullPreLoading] = useState(false);
  const [fullPreError, setFullPreError] = useState("");
  const [preMetric, setPreMetric] = useState<PreMetric>("logit");
  const [lengthIndex, setLengthIndex] = useState(0);
  const [data, setData] = useState<LengthData | null>(null);
  const [loading, setLoading] = useState(false);
  const [scope, setScope] = useState<Scope>("overall");
  const [layer, setLayer] = useState(31);
  const [head, setHead] = useState(30);
  const [topX, setTopX] = useState(20);
  const [tokenDraft, setTokenDraft] = useState("");
  const [trackedToken, setTrackedToken] = useState("");
  const [trace, setTrace] = useState<TracePoint[]>([]);
  const [traceLoading, setTraceLoading] = useState(false);
  const [traceProgress, setTraceProgress] = useState(0);
  const [traceError, setTraceError] = useState("");
  const selectedLength = manifest.completed_lengths[Math.min(lengthIndex, manifest.completed_lengths.length - 1)] ?? 0;
  const requestedPreToken = (trackedToken || manifest.gold_codes[1] || "window").trim();
  const fullPreLengthSelected = Boolean(
    fullPreManifest && selectedLength === fullPreManifest.target_context_tokens,
  );
  const failurePoint = failureBoundary?.points[
    Math.min(failureLengthIndex, Math.max(0, failureBoundary.points.length - 1))
  ];

  useEffect(() => {
    let live = true;
    fetch(FAILURE_BOUNDARY_FILE, { cache: "no-store" })
      .then((response) => {
        if (!response.ok) throw new Error(`失败边界数据读取失败（HTTP ${response.status}）`);
        return readGzipJson<FailureBoundaryPayload>(response, FAILURE_BOUNDARY_FILE);
      })
      .then((payload) => {
        if (live) {
          setFailureBoundary(payload);
          setFailureBoundaryError("");
          const firstFailure = payload.points.findIndex((point) => !point.fullVocabCorrect);
          if (firstFailure >= 0) setFailureLengthIndex(firstFailure);
        }
      })
      .catch((error) => {
        if (live) {
          setFailureBoundaryError(
            error instanceof Error ? error.message : "失败边界数据读取失败",
          );
        }
      });
    return () => {
      live = false;
    };
  }, []);

  useEffect(() => {
    let live = true;
    if (experimentMode !== "english_single_token") {
      setRelativePosition(null);
      setRelativePositionError("");
      return () => { live = false; };
    }
    setRelativePositionError("");
    fetch("/data/english_single_token/fixed_relative_328.json", { cache: "no-store" })
      .then((response) => {
        if (!response.ok) throw new Error(`固定相对距离结果读取失败（HTTP ${response.status}）`);
        return response.json() as Promise<FixedRelativePayload>;
      })
      .then((payload) => { if (live) setRelativePosition(payload); })
      .catch((error) => {
        if (live) setRelativePositionError(error instanceof Error ? error.message : "固定相对距离结果读取失败");
      });
    return () => { live = false; };
  }, [experimentMode]);

  useEffect(() => {
    let live = true;
    if (experimentMode !== "english_single_token") {
      setRopePairManifest(null);
      setRopePairHead(null);
      setRopePairError("");
      return () => { live = false; };
    }
    fetch(`${ROPE_PAIR_ROOT}/manifest.json`, { cache: "no-store" })
      .then(response => {
        if (!response.ok) throw new Error(`RoPE pair manifest 尚未同步（HTTP ${response.status}）`);
        return response.json() as Promise<RopePairManifest>;
      })
      .then(payload => { if (live) { setRopePairManifest(payload); setRopePairError(""); } })
      .catch(error => { if (live) setRopePairError(error instanceof Error ? error.message : "RoPE pair manifest 尚未同步"); });
    return () => { live = false; };
  }, [experimentMode]);

  useEffect(() => {
    let live = true;
    if (scoreSpace !== "rope_pairs" || !ropePairManifest) {
      setRopePairHead(null);
      setRopePairLoading(false);
      return () => { live = false; };
    }
    const file = applyScopePattern(ropePairManifest.files.head_pattern, layer, head);
    setRopePairLoading(true);
    setRopePairError("");
    fetch(`${ROPE_PAIR_ROOT}/${file}`, { cache: "no-store" })
      .then(response => {
        if (!response.ok) throw new Error(`${file} 读取失败（HTTP ${response.status}）`);
        return readGzipJson<RopePairHeadPayload>(response, file);
      })
      .then(payload => { if (live) setRopePairHead(payload); })
      .catch(error => { if (live) { setRopePairHead(null); setRopePairError(error instanceof Error ? error.message : "RoPE pair head 读取失败"); } })
      .finally(() => { if (live) setRopePairLoading(false); });
    return () => { live = false; };
  }, [head, layer, ropePairManifest, scoreSpace]);

  useEffect(() => {
    let live = true;
    if (experimentMode !== "english_single_token") {
      setScoreSpace("post_softmax");
      setPreSoftmax(null);
      setPreSoftmaxError("");
      return () => { live = false; };
    }
    setPreSoftmaxError("");
    fetch("/data/english_single_token/pre_softmax_summary.json", { cache: "no-store" })
      .then((response) => {
        if (!response.ok) throw new Error("pre-softmax summary unavailable");
        return response.json() as Promise<PreSoftmaxPayload>;
      })
      .then((payload) => { if (live) setPreSoftmax(payload); })
      .catch((error) => {
        if (live) setPreSoftmaxError(error instanceof Error ? error.message : "pre-softmax summary unavailable");
      });
    return () => { live = false; };
  }, [experimentMode]);

  useEffect(() => {
    let live = true;
    let found = false;
    let timer: ReturnType<typeof setInterval> | undefined;
    if (experimentMode !== "english_single_token") {
      setTokenTypeLengthManifest(null);
      setTokenTypeLength(null);
      setTokenTypeLengthError("");
      return () => { live = false; };
    }
    const loadManifest = async () => {
      try {
        const response = await fetch(`${TOKEN_TYPE_LENGTH_ROOT}/manifest.json`, { cache: "no-store" });
        if (!response.ok) throw new Error(`全 token 长度包尚未完成（HTTP ${response.status}）`);
        const payload = await response.json() as TokenTypeLengthManifest;
        if (live) {
          found = true;
          setTokenTypeLengthManifest(payload);
          setTokenTypeLengthError("");
        }
      } catch (error) {
        if (live && !found) {
          setTokenTypeLengthError(error instanceof Error ? error.message : "全 token 长度包尚未完成");
        }
      }
    };
    void loadManifest();
    timer = setInterval(() => { void loadManifest(); }, 15000);
    return () => {
      live = false;
      if (timer) clearInterval(timer);
    };
  }, [experimentMode]);

  useEffect(() => {
    let live = true;
    const canonical = requestedPreToken.toLocaleLowerCase();
    if (!tokenTypeLengthManifest || !canonical) {
      setTokenTypeLength(null);
      setTokenTypeLengthLoading(false);
      return () => { live = false; };
    }
    const entry = tokenTypeLengthManifest.tokens[canonical];
    setTokenTypeLength(null);
    if (!entry) {
      setTokenTypeLengthLoading(false);
      setTokenTypeLengthError(`“${requestedPreToken}”不是当前实验文本中的完整 tokenizer token`);
      return () => { live = false; };
    }
    setTokenTypeLengthLoading(true);
    setTokenTypeLengthError("");
    fetch(`${TOKEN_TYPE_LENGTH_ROOT}/${entry.file}`, { cache: "no-store" })
      .then((response) => {
        if (!response.ok) throw new Error(`${entry.file} 读取失败（HTTP ${response.status}）`);
        return readGzipJson<TokenTypeLengthPayload>(response, entry.file);
      })
      .then((payload) => { if (live) setTokenTypeLength(payload); })
      .catch((error) => {
        if (live) setTokenTypeLengthError(error instanceof Error ? error.message : "token 长度数据读取失败");
      })
      .finally(() => { if (live) setTokenTypeLengthLoading(false); });
    return () => { live = false; };
  }, [requestedPreToken, tokenTypeLengthManifest]);

  useEffect(() => {
    let live = true;
    if (experimentMode !== "english_single_token") {
      setPreHeadLength(null);
      setPreHeadLengthError("");
      return () => { live = false; };
    }
    const file = "pre_softmax_head_length_summary.json.gz";
    setPreHeadLengthError("");
    fetch(`/data/english_single_token/${file}`, { cache: "no-store" })
      .then((response) => {
        if (!response.ok) throw new Error(`逐 head 长度摘要读取失败（HTTP ${response.status}）`);
        return readGzipJson<PreHeadLengthPayload>(response, file);
      })
      .then((payload) => { if (live) setPreHeadLength(payload); })
      .catch((error) => { if (live) setPreHeadLengthError(error instanceof Error ? error.message : "逐 head 长度摘要读取失败"); });
    return () => { live = false; };
  }, [experimentMode]);

  useEffect(() => {
    let live = true;
    let found = false;
    let timer: ReturnType<typeof setInterval> | undefined;
    if (experimentMode !== "english_single_token") {
      setFullPreManifest(null);
      setFullPreTokens(null);
      setFullPreScope(null);
      setTokenHeatmap(null);
      setTokenHeatmapError("");
      setFullPreError("");
      return () => { live = false; };
    }
    const loadManifest = async () => {
      try {
        const response = await fetch(`${FULL_PRE_ROOT}/manifest.json`, { cache: "no-store" });
        if (!response.ok) throw new Error(`全量 128K 数据尚未到达（HTTP ${response.status}）`);
        const payload = await response.json() as FullPreSoftmaxManifest;
        if (live) {
          found = true;
          setFullPreManifest(payload);
          setFullPreError("");
        }
      } catch (error) {
        if (live && !found) {
          setFullPreError(error instanceof Error ? error.message : "全量 128K 数据尚未到达");
        }
      }
    };
    void loadManifest();
    timer = setInterval(() => { void loadManifest(); }, 15000);
    return () => {
      live = false;
      if (timer) clearInterval(timer);
    };
  }, [experimentMode]);

  useEffect(() => {
    let live = true;
    if (!fullPreManifest) {
      setFullPreTokens(null);
      return () => { live = false; };
    }
    const file = fullPreManifest.files.tokens;
    fetch(`${FULL_PRE_ROOT}/${file}`, { cache: "no-store" })
      .then((response) => {
        if (!response.ok) throw new Error(`token 表读取失败（HTTP ${response.status}）`);
        return readGzipJson<FullPreSoftmaxTokens>(response, file);
      })
      .then((payload) => { if (live) setFullPreTokens(payload); })
      .catch((error) => { if (live) setFullPreError(error instanceof Error ? error.message : "token 表读取失败"); });
    return () => { live = false; };
  }, [fullPreManifest]);

  useEffect(() => {
    let live = true;
    if (!fullPreManifest) {
      setTokenHeatmap(null);
      return () => { live = false; };
    }
    const file = fullPreManifest.files.token_type_heatmap ?? "token_type_heatmap.json.gz";
    fetch(`${FULL_PRE_ROOT}/${file}`, { cache: "no-store" })
      .then((response) => {
        if (!response.ok) throw new Error(`热力图摘要读取失败（HTTP ${response.status}）`);
        return readGzipJson<TokenTypeHeatmapPayload>(response, file);
      })
      .then((payload) => {
        if (live) {
          setTokenHeatmap(payload);
          setTokenHeatmapError("");
        }
      })
      .catch((error) => {
        if (live) setTokenHeatmapError(error instanceof Error ? error.message : "热力图摘要读取失败");
      });
    return () => { live = false; };
  }, [fullPreManifest]);

  useEffect(() => {
    let live = true;
    if (scoreSpace !== "pre_softmax" || !fullPreManifest || !fullPreLengthSelected) {
      setFullPreScope(null);
      return () => { live = false; };
    }
    const file = fullPreScopeFile(fullPreManifest, scope, layer, head);
    setFullPreLoading(true);
    setFullPreError("");
    fetch(`${FULL_PRE_ROOT}/${file}`, { cache: "no-store" })
      .then((response) => {
        if (!response.ok) throw new Error(`${file} 读取失败（HTTP ${response.status}）`);
        return readGzipJson<FullPreSoftmaxScope>(response, file);
      })
      .then((payload) => { if (live) setFullPreScope(payload); })
      .catch((error) => {
        if (live) {
          setFullPreScope(null);
          setFullPreError(error instanceof Error ? error.message : "pre-softmax scope 读取失败");
        }
      })
      .finally(() => { if (live) setFullPreLoading(false); });
    return () => { live = false; };
  }, [fullPreLengthSelected, fullPreManifest, head, layer, scope, scoreSpace]);

  useEffect(() => {
    let live = true;
    setIsDemo(true);
    setData(null);
    fetch(EXPERIMENT_MODES[experimentMode].manifest, { cache: "no-store" })
      .then((response) => {
        if (!response.ok) throw new Error("manifest unavailable");
        return response.json() as Promise<Manifest>;
      })
      .then((payload) => {
        if (!live) return;
        setManifest(payload);
        setIsDemo(false);
        setLengthIndex(0);
        setTokenDraft(payload.gold_codes[0] ?? "");
        // Do not start a full-sweep trace while the selected length is fetching
        // the same gzip payload. The user can submit the prefilled token or use
        // a preset once the mode has settled.
        setTrackedToken("");
      })
      .catch(() => {
        if (live) setIsDemo(true);
      });
    return () => { live = false; };
  }, [experimentMode]);

  useEffect(() => {
    let live = true;
    setLoading(true);
    const summary = manifest.summaries.find((row) => row.length === selectedLength);
    if (isDemo || !summary) {
      const demo = makeDemoLengthData(selectedLength, manifest);
      setData(demo);
      setLoading(false);
      return () => { live = false; };
    }
    fetch(lengthDataUrl(experimentMode, summary), { cache: "no-store" })
      .then((response) => {
        if (!response.ok) throw new Error(`length ${selectedLength} unavailable`);
        return readLengthPayload(response, summary.file);
      })
      .then((payload) => { if (live) setData(payload); })
      .catch(() => { if (live) setData(makeDemoLengthData(selectedLength, manifest)); })
      .finally(() => { if (live) setLoading(false); });
    return () => { live = false; };
  }, [experimentMode, isDemo, manifest, selectedLength]);

  useEffect(() => {
    setLayer((value) => Math.min(value, manifest.model_config.num_layers - 1));
    setHead((value) => Math.min(value, manifest.model_config.num_attention_heads - 1));
  }, [manifest]);

  const currentTraceSpec = useMemo(
    () => traceSpec(trackedToken, manifest.gold_codes),
    [trackedToken, manifest.gold_codes],
  );
  const traceTopX = currentTraceSpec.kind === "other" ? topX : 100;

  useEffect(() => {
    const normalized = trackedToken.trim();
    if (scoreSpace !== "post_softmax" || !normalized || isDemo) {
      setTrace([]);
      setTraceLoading(false);
      setTraceProgress(0);
      setTraceError("");
      return;
    }
    const controller = new AbortController();
    const scopeKey = scope === "overall" ? "overall" : scope === "layer" ? `layer-${layer}` : `layer-${layer}-head-${head}`;
    const cacheKey = `${experimentMode}|${scopeKey}|top-${traceTopX}|${normalized}`;
    const cached = TRACE_CACHE.get(cacheKey);
    if (cached) {
      setTrace(cached);
      setTraceProgress(cached.length);
      setTraceLoading(false);
      setTraceError("");
      return () => controller.abort();
    }

    setTraceLoading(true);
    setTraceProgress(0);
    setTraceError("");
    setTrace([]);
    const load = async () => {
      const output: TracePoint[] = [];
      // Chromium can reject several large gzip/DecompressionStream pipelines
      // running at once. Local files are fast enough to read serially, and the
      // Lower peak memory keeps all experiment modes reliable through 128K.
      const batchSize = 1;
      for (let start = 0; start < manifest.summaries.length; start += batchSize) {
        const batch = manifest.summaries.slice(start, start + batchSize);
        const loaded = await Promise.all(batch.map(async (summary) => {
          const url = lengthDataUrl(experimentMode, summary);
          try {
            const response = await fetch(url, {
              // Vite serves the pre-compressed files with response metadata
              // that is not safe to reuse through Chromium's force-cache.
              // The normal selected-length reader already uses no-store.
              cache: "no-store",
              signal: controller.signal,
            });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const payload = await readLengthPayload(response, summary.file);
            const result = traceValue(payload, normalized, scope, layer, head, traceTopX);
            return { length: summary.length, ...result };
          } catch (error) {
            if (controller.signal.aborted) throw error;
            const detail = error instanceof Error ? error.message : "unknown error";
            throw new Error(`${summary.length.toLocaleString()} tokens (${url}): ${detail}`);
          }
        }));
        output.push(...loaded);
        setTraceProgress(output.length);
      }
      output.sort((a, b) => a.length - b.length);
      TRACE_CACHE.set(cacheKey, output);
      setTrace(output);
    };
    load()
      .catch((error) => {
        if (!controller.signal.aborted) setTraceError(error instanceof Error ? error.message : "trace load failed");
      })
      .finally(() => {
        if (!controller.signal.aborted) setTraceLoading(false);
      });
    return () => controller.abort();
  }, [experimentMode, head, isDemo, layer, manifest.summaries, scope, scoreSpace, traceTopX, trackedToken]);

  const selectedSummary = manifest.summaries.find((row) => row.length === selectedLength) ?? manifest.summaries[0];
  const selectedRelative = relativePosition?.rows.find((row) => row.fillerTokens === selectedLength) ?? relativePosition?.rows[0];
  const selectedPreSoftmax = preSoftmax?.rows.find((row) => row.length === selectedLength) ?? null;
  const fullPreActive = scoreSpace === "pre_softmax" && fullPreLengthSelected && Boolean(fullPreManifest && fullPreTokens && fullPreScope);
  const fullPreDecoded = useMemo<FullPreDecoded | null>(() => {
    if (!fullPreTokens || !fullPreScope) return null;
    const logits = decodeF16Base64(fullPreScope.logits_f16_b64);
    const tokenIds = decodeU32Base64(fullPreTokens.token_ids_u32_b64);
    let probabilities: Float32Array;
    if (fullPreScope.probabilities_f16_b64) {
      probabilities = decodeF16Base64(fullPreScope.probabilities_f16_b64);
    } else {
      probabilities = new Float32Array(logits.length);
      const denominator = fullPreScope.logsumexp;
      if (denominator === undefined) throw new Error("head scope 缺少 logsumexp，无法重建占比");
      for (let index = 0; index < logits.length; index += 1) {
        probabilities[index] = Math.exp(logits[index] - denominator);
      }
    }
    if (logits.length !== tokenIds.length || probabilities.length !== logits.length) {
      throw new Error(`全量数组长度不一致：logits=${logits.length}, probabilities=${probabilities.length}, tokens=${tokenIds.length}`);
    }
    return { logits, probabilities, tokenIds };
  }, [fullPreScope, fullPreTokens]);
  const fullPreBars = useMemo<FullPreBar[]>(() => {
    if (!fullPreDecoded || !fullPreScope || !fullPreTokens) return [];
    const selected: FullPreBar[] = fullPreScope.top_logit_positions.slice(0, topX).map((position) => {
      const tokenId = fullPreDecoded.tokenIds[position];
      return {
        position,
        text: compactToken(fullPreTokens.token_text[String(tokenId)] ?? `token_${tokenId}`),
        role: roleAtPosition(position, fullPreTokens),
        logit: fullPreDecoded.logits[position],
        share: fullPreDecoded.probabilities[position],
      };
    });
    const shownShare = selected.reduce((sum, row) => sum + row.share, 0);
    selected.push({ position: -1, text: "OTHER", role: "other", logit: null, share: Math.max(0, 1 - shownShare) });
    return selected;
  }, [fullPreDecoded, fullPreScope, fullPreTokens, topX]);
  const fullTrackedToken = requestedPreToken;
  const tokenTypeLengthDecoded = useMemo<TokenTypeLengthDecoded | null>(() => {
    if (!tokenTypeLength || tokenTypeLength.token !== fullTrackedToken.toLocaleLowerCase()) return null;
    const occurrenceCounts = decodeU32Base64(tokenTypeLength.occurrence_counts_u32_b64);
    const meanLogits = decodeF16Base64(tokenTypeLength.mean_logits_f16_b64);
    const probabilityMass = decodeF32Base64(tokenTypeLength.probability_mass_f32_b64);
    const expectedCells = tokenTypeLength.shape.reduce((product, value) => product * value, 1);
    if (occurrenceCounts.length !== tokenTypeLength.shape[0] || meanLogits.length !== expectedCells || probabilityMass.length !== expectedCells) {
      throw new Error("全 token 长度数组与 shape 不一致");
    }
    return { occurrenceCounts, meanLogits, probabilityMass };
  }, [fullTrackedToken, tokenTypeLength]);
  const preHeadLengthDecoded = useMemo<PreHeadLengthDecoded | null>(() => {
    if (!preHeadLength) return null;
    const roleLogits = decodeF16Base64(preHeadLength.role_logits_f16_b64);
    const roleMass = decodeF32Base64(preHeadLength.role_mass_f32_b64);
    const roleBestRank = decodeU32Base64(preHeadLength.role_best_rank_u32_b64);
    const headLogsumexp = decodeF16Base64(preHeadLength.head_logsumexp_f16_b64);
    const headMaxLogit = decodeF16Base64(preHeadLength.head_max_logit_f16_b64);
    const roleCells = preHeadLength.shape.reduce((product, value) => product * value, 1);
    const headCells = preHeadLength.shape[0] * preHeadLength.num_layers * preHeadLength.num_attention_heads;
    if (roleLogits.length !== roleCells || roleMass.length !== roleCells || roleBestRank.length !== roleCells || headLogsumexp.length !== headCells || headMaxLogit.length !== headCells) {
      throw new Error("逐 head 长度摘要数组与 shape 不一致");
    }
    return { roleLogits, roleMass, roleBestRank, headLogsumexp, headMaxLogit };
  }, [preHeadLength]);
  const selectedPreEvidenceRoles = preEvidenceRoles(fullTrackedToken);
  const preHeadSeries = useMemo<PreMetricSeries[]>(() => {
    if (tokenTypeLength && tokenTypeLengthDecoded) {
      const [lengthCount, layerCount, headCount] = tokenTypeLength.shape;
      const cellsPerLength = layerCount * headCount;
      const points = tokenTypeLength.lengths.flatMap((lengthValue, lengthPosition) => {
        if (!tokenTypeLengthDecoded.occurrenceCounts[lengthPosition]) return [];
        let logit = 0;
        let share = 0;
        let count = 0;
        const addCell = (layerIndex: number, headIndex: number) => {
          const offset = lengthPosition * cellsPerLength + layerIndex * headCount + headIndex;
          logit += tokenTypeLengthDecoded.meanLogits[offset];
          share += tokenTypeLengthDecoded.probabilityMass[offset];
          count += 1;
        };
        if (scope === "head") addCell(layer, head);
        else if (scope === "layer") {
          for (let headIndex = 0; headIndex < headCount; headIndex += 1) addCell(layer, headIndex);
        } else {
          for (let layerIndex = 0; layerIndex < layerCount; layerIndex += 1) {
            for (let headIndex = 0; headIndex < headCount; headIndex += 1) addCell(layerIndex, headIndex);
          }
        }
        return [{ length: lengthValue, logit: logit / Math.max(1, count), share: share / Math.max(1, count) }];
      });
      return [{
        key: `token-type-${tokenTypeLength.token}`,
        label: `${tokenTypeLength.display} · 所有同 token 位置`,
        color: ROLE_COLORS.hop1_result,
        points,
      }];
    }
    if (!preHeadLength || !preHeadLengthDecoded || !selectedPreEvidenceRoles.length) return [];
    const [, roleCount, layerCount, headCount] = preHeadLength.shape;
    const roleStride = layerCount * headCount;
    const lengthStride = roleCount * roleStride;
    return selectedPreEvidenceRoles.flatMap(({ role, label }) => {
      const roleIndex = preHeadLength.role_order.indexOf(role);
      if (roleIndex < 0) return [];
      const points = preHeadLength.lengths.map((lengthValue, lengthPosition) => {
        let logit = 0;
        let share = 0;
        let count = 0;
        const addCell = (layerIndex: number, headIndex: number) => {
          const offset = lengthPosition * lengthStride + roleIndex * roleStride + layerIndex * headCount + headIndex;
          logit += preHeadLengthDecoded.roleLogits[offset];
          share += preHeadLengthDecoded.roleMass[offset];
          count += 1;
        };
        if (scope === "head") addCell(layer, head);
        else if (scope === "layer") {
          for (let headIndex = 0; headIndex < headCount; headIndex += 1) addCell(layer, headIndex);
        } else {
          for (let layerIndex = 0; layerIndex < layerCount; layerIndex += 1) {
            for (let headIndex = 0; headIndex < headCount; headIndex += 1) addCell(layerIndex, headIndex);
          }
        }
        return { length: lengthValue, logit: logit / Math.max(1, count), share: share / Math.max(1, count) };
      });
      return [{ key: role, label, color: ROLE_COLORS[role] ?? ROLE_COLORS.filler, points }];
    });
  }, [head, layer, preHeadLength, preHeadLengthDecoded, scope, selectedPreEvidenceRoles, tokenTypeLength, tokenTypeLengthDecoded]);
  const preHeadHeatmapCells = useMemo(() => {
    if (tokenTypeLength && tokenTypeLengthDecoded) {
      const lengthPosition = tokenTypeLength.lengths.indexOf(selectedLength);
      if (lengthPosition < 0 || !tokenTypeLengthDecoded.occurrenceCounts[lengthPosition]) return [];
      const [, layerCount, headCount] = tokenTypeLength.shape;
      const cellsPerLength = layerCount * headCount;
      const lengthOffset = lengthPosition * cellsPerLength;
      return Array.from({ length: cellsPerLength }, (_, cellIndex) => ({
        layer: Math.floor(cellIndex / headCount),
        head: cellIndex % headCount,
        logit: tokenTypeLengthDecoded.meanLogits[lengthOffset + cellIndex],
        share: tokenTypeLengthDecoded.probabilityMass[lengthOffset + cellIndex],
        rank: null as number | null,
      }));
    }
    if (!preHeadLength || !preHeadLengthDecoded || !selectedPreEvidenceRoles.length) return [];
    const lengthPosition = preHeadLength.lengths.indexOf(selectedLength);
    if (lengthPosition < 0) return [];
    const [, roleCount, layerCount, headCount] = preHeadLength.shape;
    const roleStride = layerCount * headCount;
    const lengthStride = roleCount * roleStride;
    const roleIndexes = selectedPreEvidenceRoles.map(({ role }) => preHeadLength.role_order.indexOf(role)).filter((index) => index >= 0);
    return Array.from({ length: layerCount * headCount }, (_, cellIndex) => {
      let logit = 0;
      let share = 0;
      let rank = 0;
      for (const roleIndex of roleIndexes) {
        const offset = lengthPosition * lengthStride + roleIndex * roleStride + cellIndex;
        logit += preHeadLengthDecoded.roleLogits[offset];
        share += preHeadLengthDecoded.roleMass[offset];
        rank += preHeadLengthDecoded.roleBestRank[offset];
      }
      return {
        layer: Math.floor(cellIndex / headCount),
        head: cellIndex % headCount,
        logit: logit / Math.max(1, roleIndexes.length),
        share,
        rank: rank / Math.max(1, roleIndexes.length) as number | null,
      };
    });
  }, [preHeadLength, preHeadLengthDecoded, selectedLength, selectedPreEvidenceRoles, tokenTypeLength, tokenTypeLengthDecoded]);
  const preHeadCurrent = useMemo(() => {
    if (!preHeadHeatmapCells.length) return null;
    const cells = scope === "head"
      ? preHeadHeatmapCells.filter((cell) => cell.layer === layer && cell.head === head)
      : scope === "layer"
        ? preHeadHeatmapCells.filter((cell) => cell.layer === layer)
        : preHeadHeatmapCells;
    return {
      logit: cells.reduce((sum, cell) => sum + cell.logit, 0) / cells.length,
      share: cells.reduce((sum, cell) => sum + cell.share, 0) / cells.length,
      rank: cells.every((cell) => cell.rank !== null)
        ? cells.reduce((sum, cell) => sum + (cell.rank ?? 0), 0) / cells.length
        : null,
    };
  }, [head, layer, preHeadHeatmapCells, scope]);
  const selectedTokenTypeLengthIndex = tokenTypeLength?.lengths.indexOf(selectedLength) ?? -1;
  const selectedTokenTypeOccurrences = selectedTokenTypeLengthIndex >= 0 && tokenTypeLengthDecoded
    ? tokenTypeLengthDecoded.occurrenceCounts[selectedTokenTypeLengthIndex]
    : 0;
  const tokenTypePresentLengthCount = tokenTypeLengthDecoded
    ? Array.from(tokenTypeLengthDecoded.occurrenceCounts).filter((count) => count > 0).length
    : 0;
  const preHeadHeatmapMaxAbsLogit = Math.max(1e-9, ...preHeadHeatmapCells.map((cell) => Math.abs(cell.logit)));
  const preHeadHeatmapMaxShare = Math.max(1e-12, ...preHeadHeatmapCells.map((cell) => cell.share));
  const preScopedAvailable = fullPreLengthSelected || Boolean(tokenTypeLengthDecoded) || Boolean(preHeadLengthDecoded && selectedPreEvidenceRoles.length);
  const tokenHeatmapDecoded = useMemo<TokenTypeHeatmapDecoded | null>(() => {
    if (!tokenHeatmap) return null;
    const tokenIds = decodeU32Base64(tokenHeatmap.token_ids_u32_b64);
    const tokenCounts = decodeU32Base64(tokenHeatmap.token_counts_u32_b64);
    const meanLogits = decodeF16Base64(tokenHeatmap.mean_logits_f16_b64);
    const probabilityMass = decodeF16Base64(tokenHeatmap.probability_mass_f16_b64);
    const expectedCells = tokenHeatmap.shape[0] * tokenHeatmap.shape[1] * tokenHeatmap.shape[2];
    if (tokenIds.length !== tokenHeatmap.shape[0] || tokenCounts.length !== tokenIds.length || meanLogits.length !== expectedCells || probabilityMass.length !== expectedCells) {
      throw new Error("token heatmap 数组长度与 manifest 不一致");
    }
    return { tokenIds, tokenCounts, meanLogits, probabilityMass };
  }, [tokenHeatmap]);
  const tokenHeatmapCells = useMemo(() => {
    if (!tokenHeatmap || !tokenHeatmapDecoded || !fullTrackedToken) return [];
    const wanted = fullTrackedToken.toLocaleLowerCase();
    const tokenIndexes: number[] = [];
    for (let index = 0; index < tokenHeatmapDecoded.tokenIds.length; index += 1) {
      const tokenId = tokenHeatmapDecoded.tokenIds[index];
      const text = tokenHeatmap.token_text[String(tokenId)] ?? "";
      if (text.trim().toLocaleLowerCase() === wanted) tokenIndexes.push(index);
    }
    const cellCount = tokenHeatmap.num_layers * tokenHeatmap.num_attention_heads;
    const totalOccurrences = tokenIndexes.reduce((sum, index) => sum + tokenHeatmapDecoded.tokenCounts[index], 0);
    if (!tokenIndexes.length || totalOccurrences === 0) return [];
    return Array.from({ length: cellCount }, (_, cellIndex) => {
      let weightedLogit = 0;
      let share = 0;
      for (const tokenIndex of tokenIndexes) {
        const offset = tokenIndex * cellCount + cellIndex;
        weightedLogit += tokenHeatmapDecoded.meanLogits[offset] * tokenHeatmapDecoded.tokenCounts[tokenIndex];
        share += tokenHeatmapDecoded.probabilityMass[offset];
      }
      return {
        layer: Math.floor(cellIndex / tokenHeatmap.num_attention_heads),
        head: cellIndex % tokenHeatmap.num_attention_heads,
        logit: weightedLogit / totalOccurrences,
        share,
        occurrences: totalOccurrences,
      };
    });
  }, [fullTrackedToken, tokenHeatmap, tokenHeatmapDecoded]);
  const tokenHeatmapMaxAbsLogit = Math.max(1e-9, ...tokenHeatmapCells.map((cell) => Math.abs(cell.logit)));
  const tokenHeatmapMaxShare = Math.max(1e-12, ...tokenHeatmapCells.map((cell) => cell.share));
  const fullPreMatches = useMemo(() => {
    if (!fullPreDecoded || !fullPreTokens || !fullTrackedToken) return [];
    const wanted = fullTrackedToken.toLocaleLowerCase();
    const positions: number[] = [];
    for (let position = 0; position < fullPreDecoded.tokenIds.length; position += 1) {
      const tokenId = fullPreDecoded.tokenIds[position];
      const text = fullPreTokens.token_text[String(tokenId)] ?? "";
      if (text.trim().toLocaleLowerCase() === wanted) positions.push(position);
      if (positions.length >= 100) break;
    }
    return positions.map((position) => {
      const logit = fullPreDecoded.logits[position];
      let rank = 1;
      for (let index = 0; index < fullPreDecoded.logits.length; index += 1) {
        if (fullPreDecoded.logits[index] > logit) rank += 1;
      }
      return {
        position,
        role: roleAtPosition(position, fullPreTokens),
        logit,
        share: fullPreDecoded.probabilities[position],
        rank,
      };
    });
  }, [fullPreDecoded, fullPreTokens, fullTrackedToken]);
  const fullPreMatchedShare = fullPreMatches.reduce((sum, row) => sum + row.share, 0);
  const fullPreBarMax = Math.max(
    1e-12,
    ...fullPreBars.map((row) => preMetric === "share" ? row.share : Math.abs(row.logit ?? 0)),
  );
  const preRoleEntries = selectedPreSoftmax
    ? (["hop1_result", "hop2_input", "hop2_result"] as const).map((role) => ({
        role,
        ...selectedPreSoftmax.roles[role],
      }))
    : [];
  const selectedTrackedPreLogit = selectedPreSoftmax
    ? fullTrackedToken.toLocaleLowerCase() === "window"
      ? (selectedPreSoftmax.roles.hop1_result.mean_logit + selectedPreSoftmax.roles.hop2_input.mean_logit) / 2
      : fullTrackedToken.toLocaleLowerCase() === "basket"
        ? selectedPreSoftmax.roles.hop2_result.mean_logit
        : null
    : null;
  const preLogitExtent = Math.max(
    1,
    Math.abs(selectedPreSoftmax?.mean_head_max_logit ?? 0),
    ...preRoleEntries.map((row) => Math.abs(row.mean_logit)),
  );
  const maxCompletedLength = manifest.completed_lengths.at(-1) ?? 0;
  const lengthGuides = Array.from({ length: 5 }, (_, index) => {
    const value = Math.round((maxCompletedLength * index) / 4);
    if (index === 0) return "short";
    return value >= 1000 ? `${Math.round(value / 1000)}K` : `${value}`;
  });
  const failureLengthGuides = ["34", "48", "60", "80", "100"];
  const displayedLength = scoreSpace === "failure_boundary"
    ? (failurePoint?.length ?? 48)
    : selectedLength;
  const displayedChain = scoreSpace === "failure_boundary" && failureBoundary
    ? failureBoundary.chain
    : manifest.gold_codes;
  const failureHeadMasses = failurePoint
    ? failurePoint.roleMass.flatMap((heads) =>
        heads.map((roles) => roles.slice(0, 4).reduce((sum, value) => sum + value, 0)),
      )
    : [];
  const failureGlobalEvidenceMass = failureHeadMasses.length
    ? failureHeadMasses.reduce((sum, value) => sum + value, 0) / failureHeadMasses.length
    : 0;
  const failureLateMasses = failurePoint
    ? failurePoint.roleMass.slice(30, 34).flatMap((heads) =>
        heads.map((roles) => roles.slice(0, 4).reduce((sum, value) => sum + value, 0)),
      )
    : [];
  const failureLateEvidenceMass = failureLateMasses.length
    ? failureLateMasses.reduce((sum, value) => sum + value, 0) / failureLateMasses.length
    : 0;
  const tokenMap = useMemo(() => new Map((data?.token_table ?? []).map((row) => [row[0], row])), [data]);

  const distribution = useMemo(() => {
    if (!data) return { positions: [] as number[], scores: [] as number[], entropy: 0, roles: [] as number[], recent: 0, sink: 0 };
    if (scope === "head") {
      return {
        positions: data.attention.head_positions[layer]?.[head] ?? [],
        scores: data.attention.head_scores[layer]?.[head] ?? [],
        entropy: data.attention.head_entropy[layer]?.[head] ?? 0,
        roles: data.attention.head_role_mass[layer]?.[head] ?? [],
        recent: data.attention.head_recent512_mass[layer]?.[head] ?? 0,
        sink: data.attention.head_sink16_mass[layer]?.[head] ?? 0,
      };
    }
    if (scope === "layer") {
      return {
        positions: data.attention.layer_positions[layer] ?? [],
        scores: data.attention.layer_scores[layer] ?? [],
        entropy: data.attention.layer_entropy[layer] ?? 0,
        roles: data.attention.layer_role_mass[layer] ?? [],
        recent: data.attention.head_recent512_mass[layer]?.reduce((sum, value) => sum + value, 0) / manifest.model_config.num_attention_heads || 0,
        sink: data.attention.head_sink16_mass[layer]?.reduce((sum, value) => sum + value, 0) / manifest.model_config.num_attention_heads || 0,
      };
    }
    return {
      positions: data.attention.overall_positions,
      scores: data.attention.overall_scores,
      entropy: data.attention.overall_entropy,
      roles: data.attention.overall_role_mass,
      recent: data.attention.overall_recent512_mass,
      sink: data.attention.overall_sink16_mass,
    };
  }, [data, scope, layer, head, manifest.model_config.num_attention_heads]);

  const bars = useMemo(() => {
    const count = Math.min(topX, distribution.positions.length);
    const selected = distribution.positions.slice(0, count).map((position, index) => {
      const token = tokenMap.get(position);
      return {
        position,
        text: compactToken(token?.[2] ?? `tok_${position}`),
        role: token?.[3] ?? "filler",
        score: distribution.scores[index] ?? 0,
      };
    });
    const shownMass = selected.reduce((sum, row) => sum + row.score, 0);
    selected.push({ position: -1, text: "OTHER", role: "other", score: Math.max(0, 1 - shownMass) });
    return selected;
  }, [distribution, tokenMap, topX]);

  const roleIndex = (role: string) => data?.attention.role_order.indexOf(role) ?? -1;
  const roleMass = (role: string) => {
    const index = roleIndex(role);
    return index >= 0 ? distribution.roles[index] ?? 0 : 0;
  };
  const roleEnrichment = (role: string) => {
    if (!data) return 0;
    const spanTokens = (data.spans?.[role] ?? []).reduce((sum, [start, end]) => sum + Math.max(0, end - start), 0);
    const uniformMass = spanTokens / Math.max(1, data.attention.key_length);
    return uniformMass > 0 ? roleMass(role) / uniformMass : 0;
  };
  const barMax = Math.max(...bars.map((row) => row.score), 0.000001);
  const scopeTitle = scope === "overall" ? "模型整体 · 36 层 × 32 heads 均值" : scope === "layer" ? `Layer ${layer} · 32 heads 均值` : `Layer ${layer} · Head ${head}`;
  const traceScopeTitle = scope === "overall" ? "模型整体" : scope === "layer" ? `Layer ${layer}` : `Layer ${layer} · Head ${head}`;
  const currentTracePoint = trace.find((point) => point.length === selectedLength);
  const traceMax = trace.length ? Math.max(...trace.map((point) => point.value)) : 0;
  const traceMin = trace.length ? Math.min(...trace.map((point) => point.value)) : 0;
  const visibleTracePoints = trace.filter((point) => point.matchedPositions > 0).length;
  const traceNote = currentTraceSpec.kind === "evidence"
    ? "精确 evidence span attention；即使未进入 Top-100 也不会丢失。"
    : currentTraceSpec.kind === "other"
      ? `精确计算 1 − Top-${topX} attention 总和。`
      : `普通 token 仅统计当前范围 Top-100 中可见的匹配；0 表示未进入 Top-100，不代表真实 attention 为 0。`;

  return (
    <main>
      <header className="topbar">
        <div className="brand-mark">A<span>2</span></div>
        <div>
          <p className="eyebrow">INTERNAL ATTENTION LAB</p>
          <h1>Clean 两跳链 · Attention 与失败边界</h1>
        </div>
        <div className="mode-switch" role="group" aria-label="证据编码模式">
          {(Object.keys(EXPERIMENT_MODES) as ExperimentMode[]).map((value) => (
            <button
              key={value}
              className={experimentMode === value ? "active" : ""}
              onClick={() => setExperimentMode(value)}
            >
              <b>{EXPERIMENT_MODES[value].label}</b>
              <small>{EXPERIMENT_MODES[value].detail}</small>
            </button>
          ))}
        </div>
        <div className="status-cluster">
          <span className={scoreSpace === "failure_boundary" ? failureBoundary ? "status-dot live" : "status-dot waiting" : isDemo ? "status-dot waiting" : "status-dot live"} />
          <div>
            <b>{scoreSpace === "failure_boundary"
              ? failureBoundary ? "Qwen3-8B · 密集边界已加载" : "正在读取失败边界"
              : isDemo ? "实验运行中 · 等待数据" : "Qwen3-8B · 真实结果"}</b>
            <small>{scoreSpace === "failure_boundary"
              ? `${failureBoundary?.points.length ?? 0} 个长度点 · 每 1 token`
              : `${manifest.completed_lengths.length} 个长度点 · 每 500 tokens`}</small>
          </div>
        </div>
      </header>

      <section className="control-deck">
        <div className="length-control">
          <div className="control-heading"><span>FILLER 长度</span><strong>{displayedLength.toLocaleString()} <small>tokens</small></strong></div>
          <input
            data-testid="length-slider"
            aria-label="Filler 长度"
            type="range"
            min="0"
            max={scoreSpace === "failure_boundary"
              ? Math.max(0, (failureBoundary?.points.length ?? 1) - 1)
              : Math.max(0, manifest.completed_lengths.length - 1)}
            value={scoreSpace === "failure_boundary" ? failureLengthIndex : lengthIndex}
            onInput={(event) => {
              const value = Number(event.currentTarget.value);
              if (scoreSpace === "failure_boundary") setFailureLengthIndex(value);
              else setLengthIndex(value);
            }}
            onChange={(event) => {
              const value = Number(event.target.value);
              if (scoreSpace === "failure_boundary") setFailureLengthIndex(value);
              else setLengthIndex(value);
            }}
          />
          <div className="range-labels">
            {(scoreSpace === "failure_boundary" ? failureLengthGuides : lengthGuides)
              .map((label, index) => <span key={`${label}-${index}`}>{label}</span>)}
          </div>
        </div>
        <div className="chain-readout">
          <span>{experimentMode === "legacy" ? "多-token clean 链" : experimentMode === "english_single_token" ? "英文单-token clean 链" : "中文单-token clean 链"}</span>
          <div><code>{displayedChain[0]}</code><i>→</i><code>{displayedChain[1]}</code><i>→</i><code>{displayedChain[2]}</code></div>
        </div>
        <div className="score-space-control">
          <span>SCORE SPACE</span>
          <div role="group" aria-label="Attention 分数空间">
            <button className={scoreSpace === "post_softmax" ? "active" : ""} onClick={() => setScoreSpace("post_softmax")}>POST-SOFTMAX</button>
            <button
              data-testid="failure-boundary-mode"
              className={scoreSpace === "failure_boundary" ? "active" : ""}
              disabled={!failureBoundary}
              title={failureBoundaryError || "查看 34–100 token 的密集失败边界"}
              onClick={() => {
                setScoreSpace("failure_boundary");
                setScope("overall");
              }}
            >失败边界</button>
            <button
              className={scoreSpace === "pre_softmax" ? "active" : ""}
              disabled={experimentMode !== "english_single_token" || !preSoftmax}
              title={experimentMode !== "english_single_token" ? "pre-softmax 诊断仅为英文 128K 实验保存" : preSoftmaxError || "查看 softmax 前的 QK logit"}
              onClick={() => {
                setScoreSpace("pre_softmax");
                setScope("overall");
                setLengthIndex(Math.max(0, manifest.completed_lengths.length - 1));
                if (!trackedToken) {
                  const token = manifest.gold_codes[1] ?? "window";
                  setTokenDraft(token);
                  setTrackedToken(token);
                }
              }}
            >PRE-SOFTMAX QK</button>
            <button
              data-testid="fixed-relative-mode"
              className={scoreSpace === "relative_position" ? "active" : ""}
              disabled={experimentMode !== "english_single_token" || !relativePosition}
              title={experimentMode !== "english_single_token" ? "请先选择英文单-token · 128K" : relativePositionError || "固定 evidence-query 距离与 middle 放置对照"}
              onClick={() => {
                setScoreSpace("relative_position");
                setScope("overall");
                if (selectedLength === 0) setLengthIndex(1);
              }}
            >固定距离对照</button>
            <button
              data-testid="rope-pair-mode"
              className={scoreSpace === "rope_pairs" ? "active" : ""}
              disabled={experimentMode !== "english_single_token" || !ropePairManifest}
              title={experimentMode !== "english_single_token" ? "请先选择英文单-token · 128K" : ropePairError || "查看 64K Query 的逐 RoPE 二维对贡献"}
              onClick={() => {
                setScoreSpace("rope_pairs");
                setScope("head");
                const index = manifest.completed_lengths.indexOf(64000);
                if (index >= 0) setLengthIndex(index);
              }}
            >ROPE 维度分解</button>
          </div>
        </div>
        <div className="run-readout">
          <span>{scoreSpace === "failure_boundary" ? "当前 prompt" : scoreSpace === "relative_position" ? "相对距离" : scoreSpace === "rope_pairs" ? "QUERY POSITION" : "当前 prompt"}</span>
          <strong>{scoreSpace === "failure_boundary"
            ? (failurePoint?.promptTokens ?? 0).toLocaleString()
            : scoreSpace === "relative_position"
              ? "328"
              : scoreSpace === "rope_pairs"
                ? (ropePairManifest?.query_position.toLocaleString() ?? "64K")
                : (data?.prompt_tokens ?? selectedSummary?.prompt_tokens ?? 0).toLocaleString()}</strong>
          <small>{scoreSpace === "failure_boundary"
            ? "34–100 · 每个有效长度"
            : scoreSpace === "relative_position"
              ? "evidence → query · 恒定"
              : scoreSpace === "rope_pairs"
                ? "扫描前序全部 Key"
                : "evidence 固定在中部"}</small>
        </div>
      </section>

      <section className="metric-strip">
        {scoreSpace === "failure_boundary" && failurePoint ? <>
          <Metric label="GOLD PPL" value={failurePoint.goldPpl.toFixed(3)} detail={`basket ${formatPercent(failurePoint.goldProbability)}`} accent="#f0746e" />
          <Metric label="全模型证据 MASS" value={formatPercent(failureGlobalEvidenceMass)} detail="四个证据 token · 36×32 均值" accent="#55c2a5" />
          <Metric label="L30–33 证据 MASS" value={formatPercent(failureLateEvidenceMass)} detail="与输出 margin 最相关的晚层" accent="#7ab8ff" />
          <Metric label="完整词表 MARGIN" value={`${failurePoint.fullVocabMargin >= 0 ? "+" : ""}${failurePoint.fullVocabMargin.toFixed(3)}`} detail={`top-1: ${failurePoint.topToken}`} accent={failurePoint.fullVocabCorrect ? "#55c2a5" : "#f0746e"} />
          <Metric label="候选答案 MARGIN" value={`+${failurePoint.candidateMargin.toFixed(3)}`} detail={`${failurePoint.candidatePrediction} · ${failurePoint.candidateCorrect ? "正确" : "错误"}`} accent="#e6a95d" />
        </> : scoreSpace === "relative_position" && selectedRelative ? <>
          <Metric label="FIXED GOLD PPL" value={selectedRelative.fixed.ppl.toFixed(3)} detail={`${selectedLength.toLocaleString()} filler tokens`} accent="#55c2a5" />
          <Metric label="MIDDLE GOLD PPL" value={selectedRelative.middle.ppl.toFixed(3)} detail={`距离 ${Math.round(selectedRelative.middleDistance).toLocaleString()} tokens`} accent="#f0746e" />
          <Metric label="FIXED RAW LOGIT" value={selectedRelative.fixed.evidenceLogit.toFixed(3)} detail={`middle ${selectedRelative.middle.evidenceLogit.toFixed(3)}`} accent="#e6a95d" />
          <Metric label="FIXED Q/K COSINE" value={selectedRelative.fixed.evidenceCosine.toFixed(4)} detail={`middle ${selectedRelative.middle.evidenceCosine.toFixed(4)}`} accent="#7ab8ff" />
          <Metric label="FIXED EVIDENCE MASS" value={formatPercent(selectedRelative.fixed.evidenceMass)} detail={`${(selectedRelative.fixed.evidenceMass / Math.max(selectedRelative.middle.evidenceMass, 1e-12)).toFixed(2)}× middle`} accent="#c4a7e7" />
        </> : scoreSpace === "rope_pairs" && ropePairManifest ? <>
          <Metric label="QUERY POSITION" value={ropePairManifest.query_position.toLocaleString()} detail={`${ropePairManifest.key_length.toLocaleString()} 个前序 Key`} accent="#55c2a5" />
          <Metric label="当前 HEAD" value={`L${layer} H${head}`} detail={`共享 KV head ${Math.floor(head / Math.max(1, ropePairManifest.num_attention_heads / ropePairManifest.num_key_value_heads))}`} accent="#7ab8ff" />
          <Metric label="二维频率对" value={ropePairManifest.pair_count.toString()} detail="split-half (i, i+64)" accent="#e6a95d" />
          <Metric label="位置分辨率" value={`${ropePairManifest.bin_size} tokens`} detail={`${ropePairManifest.bin_count} 个区间`} accent="#c4a7e7" />
          <Metric label="数据状态" value={ropePairLoading ? "读取中" : ropePairHead ? "已加载" : "等待"} detail={ropePairError || "纯位置核实时计算"} accent="#f0746e" />
        </> : fullPreActive && fullPreScope ? <>
          <Metric label="GOLD FINAL PPL" value={(data?.answer.gold_ppl ?? selectedSummary?.gold_ppl ?? 0).toFixed(3)} detail={`128K · prompt ${fullPreScope.key_length.toLocaleString()}`} accent="#f0746e" />
          <Metric label="当前 SCOPE" value={scope === "overall" ? "整体" : scope === "layer" ? `L${layer}` : `L${layer} H${head}`} detail={`${fullPreScope.key_length.toLocaleString()} 个 token`} accent="#7ab8ff" />
          <Metric label="MAX RAW LOGIT" value={fullPreScope.max_logit.toFixed(3)} detail={`min ${fullPreScope.min_logit.toFixed(3)}`} accent="#e6a95d" />
          <Metric label={`${fullTrackedToken} 占比`} value={formatPercent(fullPreMatchedShare)} detail={`${fullPreMatches.length} 个匹配位置`} accent={ROLE_COLORS.hop1_result} />
          <Metric label="显示范围" value={`Top-${topX}`} detail="按 raw logit 排序 + Other" accent="#c4a7e7" />
        </> : scoreSpace === "pre_softmax" && preHeadCurrent ? <>
          <Metric label="GOLD FINAL PPL" value={(selectedPreSoftmax?.gold_ppl ?? selectedSummary?.gold_ppl ?? 0).toFixed(3)} detail={`${selectedLength.toLocaleString()} filler tokens`} accent="#f0746e" />
          <Metric label="当前 SCOPE" value={scope === "overall" ? "整体" : scope === "layer" ? `L${layer}` : `L${layer} H${head}`} detail={tokenTypeLengthDecoded ? "完整 token-type 诊断" : "逐 head 证据诊断"} accent="#7ab8ff" />
          <Metric label={`${fullTrackedToken} RAW LOGIT`} value={preHeadCurrent.logit.toFixed(4)} detail={tokenTypeLengthDecoded ? "所有同 token 位置均值" : "证据角色位置均值"} accent="#e6a95d" />
          <Metric label={`${fullTrackedToken} 占比`} value={formatPercent(preHeadCurrent.share)} detail="当前 scope 的 softmax mass" accent={ROLE_COLORS.hop1_result} />
          <Metric label={preHeadCurrent.rank === null ? "TOKEN 出现次数" : "平均排名"} value={preHeadCurrent.rank === null ? selectedTokenTypeOccurrences.toLocaleString() : `#${Math.round(preHeadCurrent.rank).toLocaleString()}`} detail={preHeadCurrent.rank === null ? "当前 prompt 内所有匹配位置" : `候选长度 ${selectedPreSoftmax?.key_length.toLocaleString() ?? "—"}`} accent="#c4a7e7" />
        </> : scoreSpace === "pre_softmax" && selectedPreSoftmax ? <>
          <Metric label="GOLD FINAL PPL" value={selectedPreSoftmax.gold_ppl.toFixed(3)} detail={`prompt ${selectedPreSoftmax.prompt_tokens.toLocaleString()} tokens`} accent="#f0746e" />
          <Metric label="第一跳结果 QK LOGIT" value={selectedPreSoftmax.roles.hop1_result.mean_logit.toFixed(3)} detail={`cos ${selectedPreSoftmax.roles.hop1_result.mean_cosine.toFixed(3)}`} accent={ROLE_COLORS.hop1_result} />
          <Metric label="第二跳输入 QK LOGIT" value={selectedPreSoftmax.roles.hop2_input.mean_logit.toFixed(3)} detail={`cos ${selectedPreSoftmax.roles.hop2_input.mean_cosine.toFixed(3)}`} accent={ROLE_COLORS.hop2_input} />
          <Metric label="第二跳结果 QK LOGIT" value={selectedPreSoftmax.roles.hop2_result.mean_logit.toFixed(3)} detail={`cos ${selectedPreSoftmax.roles.hop2_result.mean_cosine.toFixed(3)}`} accent={ROLE_COLORS.hop2_result} />
          <Metric label="MEAN LOGSUMEXP" value={selectedPreSoftmax.mean_head_logsumexp.toFixed(3)} detail={`max logit ${selectedPreSoftmax.mean_head_max_logit.toFixed(3)}`} accent="#e6a95d" />
        </> : <>
          <Metric label="GOLD FINAL PPL" value={(data?.answer.gold_ppl ?? selectedSummary?.gold_ppl ?? 0).toFixed(3)} detail={`NLL ${(data?.answer.gold_mean_nll ?? 0).toFixed(3)}`} accent="#f0746e" />
          <Metric label="第一跳结果 attention" value={formatPercent(roleMass("hop1_result"))} detail={`uniform enrichment ${roleEnrichment("hop1_result").toFixed(1)}×`} accent={ROLE_COLORS.hop1_result} />
          <Metric label="第二跳输入 attention" value={formatPercent(roleMass("hop2_input"))} detail={`uniform enrichment ${roleEnrichment("hop2_input").toFixed(1)}×`} accent={ROLE_COLORS.hop2_input} />
          <Metric label="第二跳结果 attention" value={formatPercent(roleMass("hop2_result"))} detail={`uniform enrichment ${roleEnrichment("hop2_result").toFixed(1)}×`} accent={ROLE_COLORS.hop2_result} />
          <Metric label="有效关注 token 数" value={Math.exp(distribution.entropy || 0).toFixed(0)} detail={`entropy ${(distribution.entropy || 0).toFixed(2)}`} accent="#e6a95d" />
        </>}
      </section>

      {scoreSpace === "failure_boundary"
        ? failureBoundary
          ? <FailureBoundaryDashboard
              payload={failureBoundary}
              selectedIndex={failureLengthIndex}
              scope={scope}
              layer={layer}
              head={head}
              onScopeChange={setScope}
              onLayerChange={setLayer}
              onHeadChange={setHead}
            />
          : <div className="failure-loading">{failureBoundaryError || "正在读取密集失败边界数据…"}</div>
        : scoreSpace === "relative_position"
          ? <FixedRelativeDashboard payload={relativePosition} selectedLength={selectedLength} />
          : <div className="workspace-grid">
        <aside className="side-panel">
          <section>
            <p className="section-kicker">VIEW SCOPE</p>
            <div className="segmented" role="group" aria-label="统计范围">
              {(["overall", "layer", "head"] as Scope[]).map((value) => (
                <button key={value} disabled={(scoreSpace === "rope_pairs" && value !== "head") || (scoreSpace === "pre_softmax" && !preScopedAvailable && value !== "overall")} className={scope === value ? "active" : ""} onClick={() => setScope(value)}>
                  {value === "overall" ? "整体" : value === "layer" ? "按层" : "单 Head"}
                </button>
              ))}
            </div>
            {scoreSpace === "pre_softmax" && <p className="scope-note">{fullPreLengthSelected ? "128K 全量位置包：可切换 36 层 × 32 heads，并查看任意完整 token。" : tokenTypeLengthDecoded ? `完整 token-type 包：${tokenTypePresentLengthCount} 个存在该 token 的长度点均有 36×32 数据。` : tokenTypeLengthManifest ? tokenTypeLengthLoading ? "正在读取这个 token 的全部长度数据。" : tokenTypeLengthError || "请输入实验文本里存在的完整 tokenizer token。" : selectedPreEvidenceRoles.length ? "全 token 实验运行中；当前先显示证据位置的 257 点旧诊断。" : "全 token 的 0–128K 实验正在运行并等待同步。"}</p>}
            {scoreSpace === "rope_pairs" && <p className="scope-note">实际 Q/K 二维贡献只定义在单 Head；纯位置核在所有层/head 使用相同 RoPE 参数。</p>}
          </section>
          <section className={scope === "overall" || (scoreSpace === "pre_softmax" && !preScopedAvailable) ? "disabled-control" : ""}>
            <label htmlFor="layer-select"><span>Layer</span><b>{layer}</b></label>
            <input id="layer-select" type="range" min="0" max={manifest.model_config.num_layers - 1} value={layer} disabled={scope === "overall" || (scoreSpace === "pre_softmax" && !preScopedAvailable)} onChange={(event) => setLayer(Number(event.target.value))} />
            <div className="range-labels"><span>0</span><span>{manifest.model_config.num_layers - 1}</span></div>
          </section>
          <section className={scope !== "head" || (scoreSpace === "pre_softmax" && !preScopedAvailable) ? "disabled-control" : ""}>
            <label htmlFor="head-select"><span>Head</span><b>{head}</b></label>
            <input id="head-select" type="range" min="0" max={manifest.model_config.num_attention_heads - 1} value={head} disabled={scope !== "head" || (scoreSpace === "pre_softmax" && !preScopedAvailable)} onChange={(event) => setHead(Number(event.target.value))} />
            <div className="range-labels"><span>0</span><span>{manifest.model_config.num_attention_heads - 1}</span></div>
          </section>
          <section className={(scoreSpace === "rope_pairs" || (scoreSpace === "pre_softmax" && !fullPreLengthSelected)) ? "disabled-control" : ""}>
            <label htmlFor="top-select"><span>展示 Top-X</span><b>{topX}</b></label>
            <input data-testid="top-slider" id="top-select" type="range" min="5" max="100" step="5" value={topX} disabled={scoreSpace === "rope_pairs" || (scoreSpace === "pre_softmax" && !fullPreLengthSelected)} onChange={(event) => setTopX(Number(event.target.value))} />
            <div className="range-labels"><span>5</span><span>100 + Other</span></div>
          </section>
          <section className="legend">
            <p className="section-kicker">TOKEN ROLE</p>
            {["hop1_result", "hop2_input", "hop2_result", "query", "filler", "other"].map((role) => (
              <div key={role}><i style={{ background: ROLE_COLORS[role] }} /><span>{ROLE_LABELS[role]}</span></div>
            ))}
          </section>
          <section className="small-stats">
            {scoreSpace === "rope_pairs" && ropePairManifest ? <>
              <div><span>Query position</span><b>{ropePairManifest.query_position.toLocaleString()}</b></div>
              <div><span>Key bins</span><b>{ropePairManifest.bin_count.toLocaleString()}</b></div>
              <div><span>Pair layout</span><b>(i, i+64)</b></div>
              <div><span>当前文件</span><b>{ropePairHead ? `L${ropePairHead.layer} H${ropePairHead.head}` : "—"}</b></div>
            </> : fullPreActive && fullPreScope ? <>
              <div><span>Scope max logit</span><b>{fullPreScope.max_logit.toFixed(3)}</b></div>
              <div><span>Scope min logit</span><b>{fullPreScope.min_logit.toFixed(3)}</b></div>
              <div><span>匹配 token 占比</span><b>{formatPercent(fullPreMatchedShare)}</b></div>
              <div><span>数组长度</span><b>{fullPreScope.key_length.toLocaleString()}</b></div>
            </> : scoreSpace === "pre_softmax" && preHeadCurrent ? <>
              <div><span>Token raw logit</span><b>{preHeadCurrent.logit.toFixed(4)}</b></div>
              <div><span>Token 占比</span><b>{formatPercent(preHeadCurrent.share)}</b></div>
              <div><span>{preHeadCurrent.rank === null ? "Token 出现次数" : "Token 平均排名"}</span><b>{preHeadCurrent.rank === null ? selectedTokenTypeOccurrences.toLocaleString() : `#${Math.round(preHeadCurrent.rank).toLocaleString()}`}</b></div>
              <div><span>可用长度点</span><b>{tokenTypeLengthDecoded ? `${tokenTypePresentLengthCount} / ${tokenTypeLength?.lengths.length ?? 257}` : "257 / 257"}</b></div>
            </> : scoreSpace === "pre_softmax" && selectedPreSoftmax ? <>
              <div><span>Mean logsumexp</span><b>{selectedPreSoftmax.mean_head_logsumexp.toFixed(3)}</b></div>
              <div><span>Mean max logit</span><b>{selectedPreSoftmax.mean_head_max_logit.toFixed(3)}</b></div>
              <div><span>Mean query norm</span><b>{selectedPreSoftmax.mean_query_norm.toFixed(3)}</b></div>
              <div><span>Top-2% budget</span><b>{selectedPreSoftmax.top2pct_budget.toLocaleString()}</b></div>
            </> : <>
              <div><span>Recent-512 mass</span><b>{formatPercent(distribution.recent)}</b></div>
              <div><span>Sink-16 mass</span><b>{formatPercent(distribution.sink)}</b></div>
              <div><span>Query 计算</span><b>{(data?.timing.query_seconds ?? 0).toFixed(2)}s</b></div>
            </>}
          </section>
        </aside>

        <div className="analysis-column">
          {scoreSpace === "rope_pairs" && ropePairManifest ? <RopePairDashboard manifest={ropePairManifest} payload={ropePairHead} loading={ropePairLoading} error={ropePairError} layer={layer} head={head} onLayerChange={setLayer} onHeadChange={setHead} /> : scoreSpace === "pre_softmax" ? fullPreLengthSelected && fullPreManifest ? <>
            <section className="chart-panel full-pre-panel" data-testid="full-pre-softmax-panel">
              <div className="panel-heading">
                <div><p className="section-kicker">FULL PRE-SOFTMAX QK · 128K</p><h2>{scopeTitle}</h2></div>
                <div className="heading-meta"><span>{fullPreLoading ? "scope 读取中" : `${Math.max(0, fullPreBars.length - 1)} tokens + Other`}</span><b>{fullPreScope?.key_length.toLocaleString() ?? "128K"} positions</b></div>
              </div>
              <div className="pre-metric-switch" role="group" aria-label="pre-softmax 图表指标">
                <span>柱长指标</span>
                <button className={preMetric === "logit" ? "active" : ""} onClick={() => setPreMetric("logit")}>实际数值 · RAW LOGIT</button>
                <button className={preMetric === "share" ? "active" : ""} onClick={() => setPreMetric("share")}>占比 · SOFTMAX SHARE</button>
              </div>
              {fullPreScope && fullPreDecoded && fullPreTokens ? <>
                <div className="full-pre-columns"><span>排名 / TOKEN</span><span>{preMetric === "logit" ? "按 |RAW LOGIT| 缩放" : "按 SOFTMAX 占比缩放"}</span><span>实际数值</span><span>占比</span></div>
                <div className="bars full-pre-bars" data-testid="full-pre-bars">
                  {fullPreBars.map((bar, index) => {
                    const metricValue = preMetric === "share" ? bar.share : Math.abs(bar.logit ?? 0);
                    return <div className={`bar-row full-pre-row ${bar.role === "other" ? "other-row" : ""}`} key={`${bar.position}-${index}`}>
                      <div className="bar-rank">{bar.role === "other" ? "Σ" : String(index + 1).padStart(2, "0")}</div>
                      <div className="bar-token"><code>{bar.text}</code><span>{bar.position >= 0 ? `pos ${bar.position.toLocaleString()} · ${ROLE_LABELS[bar.role] ?? bar.role}` : "其余 token 的 attention 占比"}</span></div>
                      <div className="bar-track"><div className="bar-fill" style={{ width: `${Math.max(bar.role === "other" && preMetric === "logit" ? 0 : 0.25, (metricValue / fullPreBarMax) * 100)}%`, background: bar.logit !== null && bar.logit < 0 ? ROLE_COLORS.hop2_result : ROLE_COLORS[bar.role] ?? ROLE_COLORS.filler }} /></div>
                      <div className={`pre-raw-value ${bar.logit !== null && bar.logit < 0 ? "negative" : ""}`}>{bar.logit === null ? "—" : bar.logit.toFixed(4)}</div>
                      <div className="bar-value">{formatPercent(bar.share)}</div>
                    </div>;
                  })}
                </div>
                <div className="full-pre-note">
                  <b>口径</b>
                  <p>单 Head 占比由 <code>exp(raw logit − logsumexp)</code> 精确重建；层/整体占比是各 head 的 post-softmax 概率均值。Top-X 只控制展示，128K 所有位置的 raw logit 都已保存。</p>
                </div>
              </> : <div className="trace-empty">{fullPreError ? `等待全量数据：${fullPreError}` : "正在读取当前层 / Head 的 128K 数组…"}</div>}
            </section>

            <section className="token-trace-panel full-pre-token-panel" data-testid="full-pre-token-panel">
              <div className="panel-heading trace-heading">
                <div><p className="section-kicker">EXACT TOKEN LOOKUP · CURRENT SCOPE</p><h2><code>{fullTrackedToken || "—"}</code> · {traceScopeTitle}</h2></div>
                <div className="trace-current"><span>匹配位置合计占比</span><strong>{formatPercent(fullPreMatchedShare)}</strong></div>
              </div>
              <form className="trace-controls" onSubmit={(event) => {
                event.preventDefault();
                const next = tokenDraft.trim();
                if (next) setTrackedToken(next);
              }}>
                <label htmlFor="full-pre-token-input">关注的 token</label>
                <input id="full-pre-token-input" data-testid="full-pre-token-input" value={tokenDraft} onChange={(event) => setTokenDraft(event.target.value)} placeholder="例如：river、window、basket" autoComplete="off" />
                <button type="submit">查询实际值</button>
                <div className="trace-presets" aria-label="英文证据 token">
                  {Array.from(new Set([...(fullPreManifest.gold_codes ?? manifest.gold_codes), "river", "window", "basket"])).map((token) => <button type="button" key={token} className={fullTrackedToken === token ? "active" : ""} onClick={() => { setTokenDraft(token); setTrackedToken(token); }}>{token}</button>)}
                </div>
              </form>
              {fullPreMatches.length ? <div className="full-pre-match-table">
                <div className="full-pre-match-head"><span>位置</span><span>角色</span><span>RAW LOGIT</span><span>SOFTMAX 占比</span><span>全序列排名</span></div>
                {fullPreMatches.map((row) => <div key={row.position}><code>{row.position.toLocaleString()}</code><span>{ROLE_LABELS[row.role] ?? row.role}</span><b className={row.logit < 0 ? "negative" : ""}>{row.logit.toFixed(4)}</b><b>{formatPercent(row.share)}</b><b>#{row.rank.toLocaleString()}</b></div>)}
              </div> : <div className="trace-empty compact">{fullPreDecoded ? `没有找到完整 token “${fullTrackedToken}”；注意这里只匹配 tokenizer 拆分后的单 token。` : "scope 数据读取中…"}</div>}
            </section>

            <section className="token-trace-panel" data-testid="pre-token-length-curve">
              <div className="panel-heading trace-heading">
                <div><p className="section-kicker">TOKEN SCORE TRACE · 257 LENGTHS</p><h2><code>{fullTrackedToken || "—"}</code> · {traceScopeTitle}</h2></div>
                <div className="trace-current"><span>{selectedLength.toLocaleString()} tokens</span><strong>{preHeadCurrent ? preMetric === "logit" ? preHeadCurrent.logit.toFixed(4) : formatPercent(preHeadCurrent.share) : "—"}</strong></div>
              </div>
              <ScopedPreMetricCurve series={preHeadSeries} selectedLength={selectedLength} metric={preMetric} />
              <p className="panel-note">曲线跟随左侧整体 / Layer / Head 选择；完整 token-type 数据到达后会汇总 prompt 中所有同 token 位置。当前指标为 {preMetric === "logit" ? "softmax 前 raw logit" : "softmax 后 attention 占比"}。</p>
            </section>

            <section className="head-map-panel pre-heatmap-panel" data-testid="pre-softmax-heatmap">
              <div className="panel-heading compact">
                <div><p className="section-kicker">TOKEN × LAYER × HEAD HEATMAP · 128K</p><h3><code>{fullTrackedToken || "—"}</code> · {preMetric === "logit" ? "平均 raw logit" : "所有匹配位置的 softmax 占比总和"}</h3></div>
                <span>{tokenHeatmapCells.length ? `${tokenHeatmapCells[0].occurrences} 个 prompt 位置 · 点击格子进入该 Head` : tokenHeatmapError || "正在读取热力图"}</span>
              </div>
              {tokenHeatmapCells.length ? <div className="head-map pre-token-heatmap" style={{ gridTemplateColumns: `42px repeat(${manifest.model_config.num_attention_heads}, minmax(10px, 1fr))` }}>
                <div />
                {Array.from({ length: manifest.model_config.num_attention_heads }, (_, index) => <span className="head-axis" key={`pre-h-${index}`}>{index % 4 === 0 ? index : ""}</span>)}
                {Array.from({ length: manifest.model_config.num_layers }, (_, layerIndex) => <div className="head-map-row" key={`pre-layer-${layerIndex}`} style={{ display: "contents" }}>
                  <span className="layer-axis">L{layerIndex}</span>
                  {Array.from({ length: manifest.model_config.num_attention_heads }, (_, headIndex) => {
                    const cell = tokenHeatmapCells[layerIndex * manifest.model_config.num_attention_heads + headIndex];
                    const metricValue = preMetric === "logit" ? cell.logit : cell.share;
                    const intensity = preMetric === "logit"
                      ? Math.min(1, Math.abs(cell.logit) / tokenHeatmapMaxAbsLogit)
                      : Math.min(1, Math.sqrt(cell.share / tokenHeatmapMaxShare));
                    const color = preMetric === "logit" && metricValue < 0 ? "240,116,110" : preMetric === "logit" ? "230,169,93" : "85,194,165";
                    return <button
                      aria-label={`Layer ${layerIndex} Head ${headIndex}, raw logit ${cell.logit.toFixed(4)}, share ${formatPercent(cell.share)}`}
                      title={`L${layerIndex} H${headIndex} · raw ${cell.logit.toFixed(4)} · share ${formatPercent(cell.share)}`}
                      key={`${layerIndex}-${headIndex}`}
                      className={scope === "head" && layer === layerIndex && head === headIndex ? "selected" : ""}
                      style={{ backgroundColor: `rgba(${color},${(0.08 + intensity * 0.92).toFixed(3)})` }}
                      onClick={() => { setLayer(layerIndex); setHead(headIndex); setScope("head"); }}
                    />;
                  })}
                </div>)}
              </div> : <div className="trace-empty compact">{tokenHeatmapError ? `热力图读取失败：${tokenHeatmapError}` : `“${fullTrackedToken}”不是当前 prompt 中的完整单 token，或摘要仍在读取。`}</div>}
              <div className="heatmap-scale"><span>低</span><i className={preMetric === "logit" ? "logit-scale" : "share-scale"} /><span>高</span><b>{preMetric === "logit" ? "颜色方向：红 = 负值，金 = 正值" : "颜色强度：该 head 分给该 token 的 attention 占比"}</b></div>
            </section>

            <section className="lower-grid">
              <div className="curve-panel">
                <div className="panel-heading compact"><div><p className="section-kicker">CONFIDENCE TRACE</p><h3>Gold final PPL</h3></div><strong>{selectedSummary?.gold_ppl.toFixed(3)}</strong></div>
                <PplCurve summaries={manifest.summaries} selectedLength={selectedLength} />
                <p className="panel-note">全量 QK 数组对应 128K 点；上方 scope 选择不会改变答案 PPL，只改变正在观察的层/head。</p>
              </div>
              <div className="token-confidence pre-explanation">
                <div className="panel-heading compact"><div><p className="section-kicker">STORAGE</p><h3>按 scope 懒加载</h3></div></div>
                <ul>
                  <li><b>Head</b><span>一个文件保存该 head 的全部 128K raw logits 和 logsumexp。</span></li>
                  <li><b>Layer / Overall</b><span>同时保存平均 raw logit 与准确的平均 attention 占比。</span></li>
                  <li><b>切换即读取</b><span>浏览器一次只载入当前 scope，不会一次下载 36×32 份数据。</span></li>
                </ul>
              </div>
            </section>
          </> : <>
            <section className="chart-panel pre-softmax-panel" data-testid="pre-softmax-panel">
              <div className="panel-heading">
                <div><p className="section-kicker">PRE-SOFTMAX QK LOGIT · QUERY POSITION</p><h2>模型整体 · 36 层 × 32 heads 均值</h2></div>
                <div className="heading-meta"><span>{selectedLength.toLocaleString()} filler tokens</span><b>QK / √d · softmax 之前</b></div>
              </div>
              {selectedPreSoftmax ? <>
                <div className="pre-reference-grid">
                  <div><span>Mean max competitor</span><strong>{selectedPreSoftmax.mean_head_max_logit.toFixed(3)}</strong></div>
                  <div><span>Mean logsumexp</span><strong>{selectedPreSoftmax.mean_head_logsumexp.toFixed(3)}</strong></div>
                  <div><span>Mean query norm</span><strong>{selectedPreSoftmax.mean_query_norm.toFixed(3)}</strong></div>
                  <div><span>Dynamic Top-2%</span><strong>{selectedPreSoftmax.top2pct_budget.toLocaleString()}</strong></div>
                </div>
                <div className="pre-role-list">
                  {preRoleEntries.map((row) => {
                    const width = (Math.abs(row.mean_logit) / preLogitExtent) * 50;
                    const left = row.mean_logit >= 0 ? 50 : 50 - width;
                    return <article key={row.role} className="pre-role-row">
                      <div className="pre-role-name"><i style={{ background: ROLE_COLORS[row.role] }} /><div><strong>{ROLE_LABELS[row.role]}</strong><span>{row.role === "hop1_result" ? "window · rule 1 consequent" : row.role === "hop2_input" ? "window · rule 2 antecedent" : "basket · final result"}</span></div></div>
                      <div className="pre-signed-track"><i className="pre-zero" /><span style={{ left: `${left}%`, width: `${Math.max(0.4, width)}%`, background: ROLE_COLORS[row.role] }} /></div>
                      <strong className={row.mean_logit < 0 ? "negative" : ""}>{row.mean_logit.toFixed(3)}</strong>
                      <dl>
                        <div><dt>Q/K cosine</dt><dd>{row.mean_cosine.toFixed(4)}</dd></div>
                        <div><dt>mean rank</dt><dd>{Math.round(row.mean_rank).toLocaleString()}</dd></div>
                        <div><dt>rank percentile</dt><dd>{formatPercent(row.mean_rank_percentile)}</dd></div>
                        <div><dt>Top-2% heads</dt><dd>{formatPercent(row.top2pct_head_fraction)}</dd></div>
                        <div><dt>Top-100 heads</dt><dd>{formatPercent(row.top100_head_fraction)}</dd></div>
                        <div><dt>max competitor gap</dt><dd>{row.mean_max_logit_gap.toFixed(3)}</dd></div>
                      </dl>
                    </article>;
                  })}
                </div>
              </> : <div className="trace-empty">{preSoftmaxError ? `读取失败：${preSoftmaxError}` : "正在读取 pre-softmax 数据。"}</div>}
            </section>

            <section className="token-trace-panel pre-trace-panel">
              <div className="panel-heading trace-heading">
                <div><p className="section-kicker">TOKEN SCORE TRACE · ALL 257 LENGTHS</p><h2><code>{fullTrackedToken || "—"}</code> · {traceScopeTitle}</h2></div>
                <div className="trace-current"><span>{selectedLength.toLocaleString()} tokens</span><strong>{preHeadCurrent ? preMetric === "logit" ? preHeadCurrent.logit.toFixed(4) : formatPercent(preHeadCurrent.share) : selectedTrackedPreLogit?.toFixed(4) ?? "—"}</strong></div>
              </div>
              <div className="pre-metric-switch" role="group" aria-label="跨长度曲线指标">
                <span>曲线与热力图指标</span>
                <button className={preMetric === "logit" ? "active" : ""} onClick={() => setPreMetric("logit")}>实际数值 · RAW LOGIT</button>
                <button className={preMetric === "share" ? "active" : ""} onClick={() => setPreMetric("share")}>占比 · SOFTMAX SHARE</button>
              </div>
              <form className="trace-controls" onSubmit={(event) => {
                event.preventDefault();
                const next = tokenDraft.trim();
                if (next) setTrackedToken(next);
              }}>
                <label htmlFor="pre-length-token-input">关注的 token</label>
                <input id="pre-length-token-input" data-testid="pre-length-token-input" value={tokenDraft} onChange={(event) => setTokenDraft(event.target.value)} placeholder="window 或 basket" autoComplete="off" />
                <button type="submit">绘制长度曲线</button>
                <div className="trace-presets">
                  {["window", "basket", "river"].map((token) => <button type="button" key={token} className={fullTrackedToken === token ? "active" : ""} onClick={() => { setTokenDraft(token); setTrackedToken(token); }}>{token}</button>)}
                </div>
              </form>
              <ScopedPreMetricCurve series={preHeadSeries} selectedLength={selectedLength} metric={preMetric} />
              <div className="pre-limit-note">
                <strong>{tokenTypeLengthDecoded ? "完整 token-type 数据" : "实验补采状态"}</strong>
                <p>{tokenTypeLengthDecoded ? <><code>{fullTrackedToken}</code> 在每个长度都按 tokenizer token id 汇总；raw logit 是所有匹配位置的均值，softmax share 是所有匹配位置的概率和。</> : tokenTypeLengthLoading ? "正在读取这个 token 的全部长度数据。" : tokenTypeLengthError ? `完整包暂不可用：${tokenTypeLengthError}` : "0–128K、每 500 tokens、36×32 heads 的完整 token-type 实验正在远程运行；完成并同步后这里会自动替换旧的证据专用曲线。"}</p>
              </div>
            </section>

            <section className="head-map-panel pre-heatmap-panel" data-testid="pre-head-length-heatmap">
              <div className="panel-heading compact">
                <div><p className="section-kicker">TOKEN × LAYER × HEAD</p><h3><code>{fullTrackedToken || "—"}</code> · {selectedLength.toLocaleString()} tokens · {preMetric === "logit" ? "raw logit" : "softmax 占比"}</h3></div>
                <span>{preHeadHeatmapCells.length ? `36 × 32 · ${selectedTokenTypeOccurrences ? `${selectedTokenTypeOccurrences} 个匹配位置 · ` : ""}点击格子切换到该 Head` : tokenTypeLengthLoading ? "正在读取 token 数据" : tokenTypeLengthError || preHeadLengthError || "该长度没有这个 token"}</span>
              </div>
              {preHeadHeatmapCells.length ? <div className="head-map pre-token-heatmap" style={{ gridTemplateColumns: `42px repeat(${manifest.model_config.num_attention_heads}, minmax(10px, 1fr))` }}>
                <div />
                {Array.from({ length: manifest.model_config.num_attention_heads }, (_, index) => <span className="head-axis" key={`length-h-${index}`}>{index % 4 === 0 ? index : ""}</span>)}
                {Array.from({ length: manifest.model_config.num_layers }, (_, layerIndex) => <div className="head-map-row" key={`length-layer-${layerIndex}`} style={{ display: "contents" }}>
                  <span className="layer-axis">L{layerIndex}</span>
                  {Array.from({ length: manifest.model_config.num_attention_heads }, (_, headIndex) => {
                    const cell = preHeadHeatmapCells[layerIndex * manifest.model_config.num_attention_heads + headIndex];
                    const metricValue = preMetric === "logit" ? cell.logit : cell.share;
                    const intensity = preMetric === "logit"
                      ? Math.min(1, Math.abs(cell.logit) / preHeadHeatmapMaxAbsLogit)
                      : Math.min(1, Math.sqrt(cell.share / preHeadHeatmapMaxShare));
                    const color = preMetric === "logit" && metricValue < 0 ? "240,116,110" : preMetric === "logit" ? "230,169,93" : "85,194,165";
                    const rankText = cell.rank === null ? "" : ` · rank #${Math.round(cell.rank).toLocaleString()}`;
                    return <button
                      aria-label={`Layer ${layerIndex} Head ${headIndex}, raw logit ${cell.logit.toFixed(4)}, share ${formatPercent(cell.share)}${cell.rank === null ? "" : `, rank ${Math.round(cell.rank)}`}`}
                      title={`L${layerIndex} H${headIndex} · raw ${cell.logit.toFixed(4)} · share ${formatPercent(cell.share)}${rankText}`}
                      key={`${layerIndex}-${headIndex}`}
                      className={scope === "head" && layer === layerIndex && head === headIndex ? "selected" : ""}
                      style={{ backgroundColor: `rgba(${color},${(0.08 + intensity * 0.92).toFixed(3)})` }}
                      onClick={() => { setLayer(layerIndex); setHead(headIndex); setScope("head"); }}
                    />;
                  })}
                </div>)}
              </div> : <div className="trace-empty compact">{tokenTypeLengthLoading ? "正在读取完整 token-type 热力图。" : tokenTypeLengthError ? `完整包暂不可用：${tokenTypeLengthError}` : preHeadLengthError ? `读取失败：${preHeadLengthError}` : `当前长度没有完整 token “${fullTrackedToken}”，或全量实验仍在同步。`}</div>}
              <div className="heatmap-scale"><span>低</span><i className={preMetric === "logit" ? "logit-scale" : "share-scale"} /><span>高</span><b>{preMetric === "logit" ? "红 = 负 raw logit，金 = 正 raw logit" : "绿色越亮，该 head 分给所有同 token 位置的 attention 占比越大"}</b></div>
            </section>

            <section className="lower-grid">
              <div className="curve-panel">
                <div className="panel-heading compact"><div><p className="section-kicker">CONFIDENCE TRACE</p><h3>Gold final PPL</h3></div><strong>{selectedSummary?.gold_ppl.toFixed(3)}</strong></div>
                <PplCurve summaries={manifest.summaries} selectedLength={selectedLength} />
                <p className="panel-note">把输出置信度和同一长度点的 QK logit 对齐，区分检索方向退化与 softmax 竞争增长。</p>
              </div>
              <div className="token-confidence pre-explanation">
                <div className="panel-heading compact"><div><p className="section-kicker">HOW TO READ</p><h3>logit、rank 与 logsumexp</h3></div></div>
                <ul>
                  <li><b>target logit / cosine 下降</b><span>query 与证据 key 的方向匹配本身变弱。</span></li>
                  <li><b>logsumexp 上升</b><span>候选数量或强竞争 token 增加，softmax 分母变大。</span></li>
                  <li><b>rank 与 max gap 变差</b><span>越来越多无关 token 超过目标证据。</span></li>
                </ul>
              </div>
            </section>
          </> : <>
          <section className="chart-panel">
            <div className="panel-heading">
              <div><p className="section-kicker">SOFTMAX ATTENTION · QUERY POSITION</p><h2>{scopeTitle}</h2></div>
              <div className="heading-meta"><span>{loading ? "读取中" : `${bars.length - 1} tokens + Other`}</span><b>Σ = 1.000</b></div>
            </div>
            <div className="bars" data-testid="attention-bars">
              {bars.map((bar, index) => (
                <div className={`bar-row ${bar.role === "other" ? "other-row" : ""}`} key={`${bar.position}-${index}`}>
                  <div className="bar-rank">{bar.role === "other" ? "Σ" : String(index + 1).padStart(2, "0")}</div>
                  <div className="bar-token"><code>{bar.text}</code><span>{bar.position >= 0 ? `pos ${bar.position.toLocaleString()}` : "remaining mass"}</span></div>
                  <div className="bar-track"><div className="bar-fill" style={{ width: `${Math.max(0.3, (bar.score / barMax) * 100)}%`, background: ROLE_COLORS[bar.role] ?? ROLE_COLORS.filler }} /></div>
                  <div className="bar-value">{formatPercent(bar.score)}</div>
                </div>
              ))}
            </div>
          </section>

          <section className="token-trace-panel" data-testid="token-trace-panel">
            <div className="panel-heading trace-heading">
              <div>
                <p className="section-kicker">TOKEN ATTENTION TRACE · ALL LENGTHS</p>
                <h2><code>{trackedToken || "—"}</code> · {traceScopeTitle}</h2>
              </div>
              <div className="trace-current">
                <span>{selectedLength.toLocaleString()} tokens</span>
                <strong>{currentTracePoint ? formatPercent(currentTracePoint.value) : "—"}</strong>
              </div>
            </div>
            <form
              className="trace-controls"
              onSubmit={(event) => {
                event.preventDefault();
                const next = tokenDraft.trim();
                if (next) setTrackedToken(next);
              }}
            >
              <label htmlFor="trace-token-input">关注的 token</label>
              <input
                id="trace-token-input"
                data-testid="trace-token-input"
                value={tokenDraft}
                onChange={(event) => setTokenDraft(event.target.value)}
                placeholder={
                  experimentMode === "english_single_token"
                    ? "例如：river、window、basket、OTHER"
                    : "例如：佑、丝、须、OTHER"
                }
                autoComplete="off"
              />
              <button type="submit">绘制曲线</button>
              <div className="trace-presets" aria-label="常用 token">
                {Array.from(new Set([...manifest.gold_codes, "OTHER"])).map((token) => (
                  <button
                    type="button"
                    key={token}
                    className={trackedToken === token ? "active" : ""}
                    onClick={() => { setTokenDraft(token); setTrackedToken(token); }}
                  >
                    {token}
                  </button>
                ))}
              </div>
            </form>
            {traceLoading && (
              <div className="trace-progress">
                <span>正在读取 {traceProgress} / {manifest.summaries.length} 个长度点</span>
                <progress value={traceProgress} max={manifest.summaries.length} />
              </div>
            )}
            {traceError && <p className="trace-error">读取失败：{traceError}</p>}
            <TokenTraceCurve
              points={trace}
              selectedLength={selectedLength}
              token={trackedToken}
              kind={currentTraceSpec.kind}
            />
            <div className="trace-summary">
              <span>最小值 <b>{formatPercent(traceMin)}</b></span>
              <span>最大值 <b>{formatPercent(traceMax)}</b></span>
              <span>可见长度点 <b>{visibleTracePoints} / {trace.length || manifest.summaries.length}</b></span>
              <p>{traceNote}</p>
            </div>
          </section>

          <section className="lower-grid">
            <div className="curve-panel">
              <div className="panel-heading compact"><div><p className="section-kicker">CONFIDENCE TRACE</p><h3>Gold final PPL</h3></div><strong>{selectedSummary?.gold_ppl.toFixed(3)}</strong></div>
              <PplCurve summaries={manifest.summaries} selectedLength={selectedLength} />
              <p className="panel-note">当前长度点沿完整 0–{Math.round(maxCompletedLength / 1000)}K 扫描定位。PPL 越高，模型给正确答案 token 的平均概率越低。</p>
            </div>
            <div className="token-confidence">
              <div className="panel-heading compact"><div><p className="section-kicker">GOLD TOKEN CONFIDENCE</p><h3>{data?.answer.gold_answer ?? manifest.gold_codes[2]}</h3></div></div>
              <div className="token-probs">
                {(data?.answer.gold_token_scores ?? []).map((row) => (
                  <div key={row.index}><code>{compactToken(row.token)}</code><span><i style={{ width: `${Math.min(100, row.probability * 100)}%` }} /></span><b>{formatPercent(row.probability)}</b></div>
                ))}
              </div>
              <p className="panel-note">Teacher-forced 分解：可以直接看是答案的哪一个 sub-token 最先失去信心。</p>
            </div>
          </section>

          <section className="head-map-panel">
            <div className="panel-heading compact"><div><p className="section-kicker">HEAD MAP · EVIDENCE MASS</p><h3>第一跳结果 + 第二跳结果</h3></div><span>点击格子进入单 Head</span></div>
            <div className="head-map" style={{ gridTemplateColumns: `42px repeat(${manifest.model_config.num_attention_heads}, minmax(7px, 1fr))` }}>
              <div />
              {Array.from({ length: manifest.model_config.num_attention_heads }, (_, index) => <span className="head-axis" key={`h-${index}`}>{index % 4 === 0 ? index : ""}</span>)}
              {Array.from({ length: manifest.model_config.num_layers }, (_, layerIndex) => (
                <div className="head-map-row" key={`layer-${layerIndex}`} style={{ display: "contents" }}>
                  <span className="layer-axis">L{layerIndex}</span>
                  {Array.from({ length: manifest.model_config.num_attention_heads }, (_, headIndex) => {
                    const roles = data?.attention.head_role_mass[layerIndex]?.[headIndex] ?? [];
                    const hop1Index = data?.attention.role_order.indexOf("hop1_result") ?? -1;
                    const hop2Index = data?.attention.role_order.indexOf("hop2_result") ?? -1;
                    const mass = (roles[hop1Index] ?? 0) + (roles[hop2Index] ?? 0);
                    const intensity = Math.min(1, mass / 0.13);
                    return <button
                      aria-label={`Layer ${layerIndex} Head ${headIndex}, evidence mass ${formatPercent(mass)}`}
                      title={`L${layerIndex} H${headIndex} · ${formatPercent(mass)}`}
                      key={`${layerIndex}-${headIndex}`}
                      className={scope === "head" && layer === layerIndex && head === headIndex ? "selected" : ""}
                      style={{ backgroundColor: `color-mix(in srgb, #55c2a5 ${Math.max(5, intensity * 100)}%, #202733)` }}
                      onClick={() => { setLayer(layerIndex); setHead(headIndex); setScope("head"); }}
                    />;
                  })}
                </div>
              ))}
            </div>
          </section>
          </>}
        </div>
      </div>}

      <footer>
        <span>Qwen3-8B · seed 0 · clean · {scoreSpace === "relative_position" ? "fixed distance 328 vs middle" : "middle placement"} · full2 query · {EXPERIMENT_MODES[experimentMode].label} · {scoreSpace === "failure_boundary" ? "dense failure boundary" : scoreSpace === "pre_softmax" ? "pre-softmax QK" : scoreSpace === "relative_position" ? "position-controlled analysis" : scoreSpace === "rope_pairs" ? "RoPE pair decomposition" : "post-softmax attention"}</span>
        <span>{scoreSpace === "failure_boundary"
          ? `密集数据已对接 · ${failureBoundary?.points.length ?? 0} 个长度点 · 34–100`
          : isDemo
            ? "预览数据会在实验 JSON 到达后自动替换"
            : `数据已对接 · ${manifest.completed_lengths.length} 个长度点 · 最高 ${maxCompletedLength.toLocaleString()} tokens`}</span>
      </footer>
    </main>
  );
}
