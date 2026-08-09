import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { gunzipSync } from "node:zlib";

const root = new URL("../", import.meta.url);

async function render(pathname = "/") {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);
  return worker.fetch(
    new Request(`http://localhost${pathname}`, { headers: { accept: "text/html" } }),
    { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
    { waitUntil() {}, passThroughOnException() {} },
  );
}

test("server-renders the attention confidence dashboard", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);
  const html = await response.text();
  assert.match(html, /<title>Attention Confidence Lab · Qwen3-8B<\/title>/i);
  assert.match(html, /Clean 两跳链 · Attention 与失败边界/);
  assert.match(html, /POST-SOFTMAX/);
  assert.match(html, /PRE-SOFTMAX QK/);
  assert.match(html, /失败边界/);
  assert.match(html, /固定距离对照/);
  assert.match(html, /data-testid="length-slider"/);
  assert.match(html, /正在读取密集失败边界数据/);
});

test("server-renders the fixed-300 age distractor dashboard", async () => {
  const response = await render("/age-distractor");
  assert.equal(response.status, 200);
  const html = await response.text();
  assert.match(html, /固定 300-token 年龄干扰实验/);
  assert.match(html, /QWEN3-8B CONTROLLED RETRIEVAL LAB/);
});

test("keeps the experiment data contract explicit and lazy-loaded", async () => {
  const page = await readFile(new URL("app/page.tsx", root), "utf8");
  assert.match(page, /manifest: "\/data\/manifest\.json"/);
  assert.match(page, /manifest: "\/data\/single_token\/manifest\.json"/);
  assert.match(page, /manifest: "\/data\/english_single_token\/manifest\.json"/);
  assert.match(page, /\/data\/english_single_token\/pre_softmax_summary\.json/);
  assert.match(page, /EXPERIMENT_MODES\[experimentMode\]\.manifest/);
  assert.match(page, /DecompressionStream\("gzip"\)/);
  assert.match(page, /const fallback = response\.clone\(\)/);
  assert.match(page, /return await response\.json\(\) as T/);
  assert.match(page, /head_positions/);
  assert.match(page, /layer_positions/);
  assert.match(page, /overall_positions/);
  assert.match(page, /Math\.max\(0, 1 - shownMass\)/);
  assert.match(page, /function traceValue\(/);
  assert.match(page, /1 - distribution\.scores\.slice\(0, topX\)/);
  assert.match(page, /\.\.\.manifest\.gold_codes, "OTHER"/);
  assert.match(page, /mean_head_logsumexp/);
  assert.match(page, /mean_max_logit_gap/);
  assert.match(page, /full_token_logits_saved/);
  assert.match(page, /full_pre_softmax_128k/);
  assert.match(page, /\$\{FULL_PRE_ROOT\}\/manifest\.json/);
  assert.match(page, /function decodeF16Base64\(/);
  assert.match(page, /function decodeU32Base64\(/);
  assert.match(page, /probabilities_f16_b64/);
  assert.match(page, /exp\(raw logit − logsumexp\)/);
  assert.match(page, /data-testid="full-pre-bars"/);
  assert.match(page, /function SelectedPreSoftmaxTokenCurve\(/);
  assert.match(page, /fullPreLengthSelected/);
  assert.match(page, /token_type_heatmap\.json\.gz/);
  assert.match(page, /data-testid="pre-token-length-curve"/);
  assert.match(page, /data-testid="pre-softmax-heatmap"/);
  assert.match(page, /pre_softmax_head_length_summary\.json\.gz/);
  assert.match(page, /token_type_all_lengths/);
  assert.match(page, /probability_mass_f32_b64/);
  assert.match(page, /完整 token-type 数据/);
  assert.match(page, /function ScopedPreMetricCurve\(/);
  assert.match(page, /data-testid="pre-head-length-heatmap"/);
  assert.match(page, /role_best_rank_u32_b64/);
  assert.match(page, /fixed_relative_328\.json/);
  assert.match(page, /failure_boundary_dense\/payload\.json\.gz/);
  assert.match(page, /data-testid="failure-boundary-mode"/);
  assert.match(page, /FailureBoundaryDashboard/);
  assert.match(page, /data-testid="fixed-relative-dashboard"/);
  assert.match(page, /function RelativeComparisonCurve\(/);
  assert.match(page, /固定相对距离后，QK 方向没有退化/);
});

test("packages the complete dense failure-boundary tensor", async () => {
  const compressed = await readFile(
    new URL("public/data/failure_boundary_dense/payload.json.gz", root),
  );
  const payload = JSON.parse(gunzipSync(compressed).toString("utf8"));
  assert.equal(payload.points.length, 67);
  assert.deepEqual([payload.lengths.at(0), payload.lengths.at(-1)], [34, 100]);
  assert.deepEqual(
    [
      payload.points[0].roleMass.length,
      payload.points[0].roleMass[0].length,
      payload.points[0].roleMass[0][0].length,
    ],
    [36, 32, 4],
  );
  assert.equal(
    payload.points.filter((point) => point.candidateCorrect).length,
    67,
  );
  assert.equal(
    payload.points.find((point) => !point.fullVocabCorrect).length,
    48,
  );
});

test("packages all ten fixed-300 age distractor tensors", async () => {
  const page = await readFile(new URL("app/age-distractor/page.tsx", root), "utf8");
  assert.match(page, /每个 token 的平均 Attention/);
  assert.match(page, /categoryMeanAttentionValues/);
  assert.match(page, /平均 Attention = Mass ÷ token 数/);
  const compressed = await readFile(
    new URL("public/data/age_distractor_fixed300/payload.json.gz", root),
  );
  const payload = JSON.parse(gunzipSync(compressed).toString("utf8"));
  assert.equal(payload.points.length, 10);
  assert.deepEqual(
    payload.points.map((point) => point.distractorCount),
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
  );
  assert.ok(payload.points.every((point) => point.totalTokens === 300));
  assert.deepEqual(
    [
      payload.points[0].headCategoryMass.length,
      payload.points[0].headCategoryMass[0].length,
      payload.points[0].headCategoryMass[0][0].length,
    ],
    [36, 32, 5],
  );
  assert.deepEqual(
    [
      payload.points[0].headCategoryMaxLogit.length,
      payload.points[0].headCategoryMaxLogit[0].length,
      payload.points[0].headCategoryMaxLogit[0][0].length,
    ],
    [36, 32, 5],
  );
  assert.ok(payload.points.every((point) => point.candidatePrediction === "nine"));
});
