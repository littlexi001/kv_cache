import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { gzipSync } from "node:zlib";

const scriptRoot = dirname(fileURLToPath(import.meta.url));
const siteRoot = resolve(scriptRoot, "..");
const defaultInput = resolve(
  siteRoot,
  "..",
  "artifacts",
  "20260724_fixed300_age_distractor_qk_qwen3_8b",
);
const inputRoot = resolve(process.argv[2] ?? defaultInput);
const outputPath = resolve(
  process.argv[3] ??
    resolve(siteRoot, "public", "data", "age_distractor_fixed300", "payload.json.gz"),
);

function average(values) {
  return values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length);
}

function pearson(xs, ys) {
  const mx = average(xs);
  const my = average(ys);
  let numerator = 0;
  let xx = 0;
  let yy = 0;
  for (let index = 0; index < xs.length; index += 1) {
    const dx = xs[index] - mx;
    const dy = ys[index] - my;
    numerator += dx * dy;
    xx += dx * dx;
    yy += dy * dy;
  }
  return numerator / Math.max(1e-30, Math.sqrt(xx * yy));
}

function overallCategory(headTensor, categoryIndex) {
  return average(
    headTensor.flatMap((layer) => layer.map((head) => head[categoryIndex])),
  );
}

const manifest = JSON.parse(await readFile(resolve(inputRoot, "manifest.json"), "utf8"));
if (manifest.completed_count !== 10) {
  throw new Error(`Expected 10 completed cases, got ${manifest.completed_count}`);
}

const rawCases = await Promise.all(
  Array.from({ length: 10 }, async (_, count) =>
    JSON.parse(
      await readFile(resolve(inputRoot, `case_${String(count).padStart(2, "0")}.json`), "utf8"),
    ),
  ),
);

const points = rawCases.map((raw) => {
  const { case: experimentCase, answer, attention, timing } = raw;
  return {
    distractorCount: experimentCase.distractor_count,
    totalTokens: experimentCase.total_tokens,
    fillerCount: experimentCase.filler_count,
    fillerGapCounts: experimentCase.filler_gap_counts,
    categoryCounts: experimentCase.category_counts,
    categorySpans: experimentCase.category_spans,
    goldSpan: experimentCase.gold_span,
    goldAgeSpan: experimentCase.gold_age_span,
    querySpan: experimentCase.query_span,
    goldText: experimentCase.gold_text,
    queryText: experimentCase.query_text,
    distractors: experimentCase.distractors,
    promptText: experimentCase.prompt_text,
    decodedTokens: experimentCase.decoded_tokens,
    tokenCategories: experimentCase.token_categories,
    goldProbability: answer.gold_probability,
    goldPpl: answer.gold_ppl,
    fullVocabMargin: answer.full_vocab_margin,
    fullVocabCorrect: answer.full_vocab_correct,
    candidateMargin: answer.candidate_margin,
    candidateCorrect: answer.candidate_correct,
    candidatePrediction: answer.candidate_prediction,
    topToken: answer.top_token,
    topProbability: answer.top_probability,
    strongestNonGold: answer.strongest_non_gold,
    strongestWrongCandidate: answer.strongest_wrong_candidate,
    nextTokenTop10: answer.next_token_top10,
    candidateScores: answer.candidate_scores,
    headCategoryMass: attention.head_category_mass,
    headCategoryMeanAttention: attention.head_category_mean_attention,
    headCategoryEnrichment: attention.head_category_enrichment,
    headCategoryMeanLogit: attention.head_category_mean_logit,
    headCategoryMaxLogit: attention.head_category_max_logit,
    headCategoryLogsumexp: attention.head_category_logsumexp,
    headCategoryBestRank: attention.head_category_best_rank,
    headEntropy: attention.head_entropy,
    headEffectiveTokens: attention.head_effective_tokens,
    headLogsumexp: attention.head_logsumexp,
    headMaxLogit: attention.head_max_logit,
    timing,
  };
});

const goldAgeIndex = manifest.category_order.indexOf("gold_age");
const goldLineIndex = manifest.category_order.indexOf("gold_line");
const distractorAgeIndex = manifest.category_order.indexOf("distractor_ages");
const distractorLineIndex = manifest.category_order.indexOf("distractor_lines");
const irrelevantIndex = manifest.category_order.indexOf("irrelevant_periods");
const goldAgeMass = points.map((point) =>
  overallCategory(point.headCategoryMass, goldAgeIndex),
);
const goldLineMass = points.map((point) =>
  overallCategory(point.headCategoryMass, goldLineIndex),
);
const distractorAgeMass = points.map((point) =>
  overallCategory(point.headCategoryMass, distractorAgeIndex),
);
const distractorLineMass = points.map((point) =>
  overallCategory(point.headCategoryMass, distractorLineIndex),
);
const irrelevantMass = points.map((point) =>
  overallCategory(point.headCategoryMass, irrelevantIndex),
);
const margins = points.map((point) => point.fullVocabMargin);
const ppls = points.map((point) => point.goldPpl);

const payload = {
  schemaVersion: 1,
  experiment: manifest.experiment,
  model: manifest.model_name_or_path,
  totalTokens: manifest.total_tokens,
  goldEvidence: manifest.gold_evidence,
  query: manifest.query,
  goldAnswer: manifest.gold_answer,
  numberWords: manifest.number_words,
  answerTokenIds: manifest.answer_token_ids,
  categoryOrder: manifest.category_order,
  categoryLabels: manifest.category_labels,
  numLayers: points[0].headCategoryMass.length,
  numHeads: points[0].headCategoryMass[0].length,
  summary: {
    fullVocabCorrectCount: points.filter((point) => point.fullVocabCorrect).length,
    candidateCorrectCount: points.filter((point) => point.candidateCorrect).length,
    goldAgeMassVsMarginPearson: pearson(goldAgeMass, margins),
    goldLineMassVsMarginPearson: pearson(goldLineMass, margins),
    distractorAgeMassVsMarginPearson: pearson(distractorAgeMass, margins),
    distractorLineMassVsMarginPearson: pearson(distractorLineMass, margins),
    irrelevantMassVsMarginPearson: pearson(irrelevantMass, margins),
    goldAgeMassVsPplPearson: pearson(goldAgeMass, ppls),
    firstFullVocabFailure:
      points.find((point) => !point.fullVocabCorrect)?.distractorCount ?? null,
    firstCandidateFailure:
      points.find((point) => !point.candidateCorrect)?.distractorCount ?? null,
  },
  points,
};

await mkdir(dirname(outputPath), { recursive: true });
const json = JSON.stringify(payload);
await writeFile(outputPath, gzipSync(Buffer.from(json), { level: 9 }));
await writeFile(outputPath.replace(/\.gz$/, ""), json);
console.log(
  JSON.stringify({
    inputRoot,
    outputPath,
    points: points.length,
    uncompressedBytes: Buffer.byteLength(json),
  }),
);
