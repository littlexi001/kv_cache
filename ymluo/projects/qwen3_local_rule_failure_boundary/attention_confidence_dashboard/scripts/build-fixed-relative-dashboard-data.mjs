import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const dashboardRoot = resolve(here, "..");
const artifactRoot = resolve(
  dashboardRoot,
  "../artifacts/20260719_fixed_relative_328_128k/analysis",
);
const outputFile = resolve(
  dashboardRoot,
  "public/data/english_single_token/fixed_relative_328.json",
);

function parseNumericCsv(text) {
  const [headerLine, ...lines] = text.trim().split(/\r?\n/);
  const headers = headerLine.split(",");
  return lines.filter(Boolean).map((line) => {
    const values = line.split(",");
    return Object.fromEntries(headers.map((header, index) => [header, Number(values[index])]));
  });
}

const [summaryText, fixedText, comparisonText] = await Promise.all([
  readFile(resolve(artifactRoot, "fixed_relative_summary.json"), "utf8"),
  readFile(resolve(artifactRoot, "fixed_relative_rows.csv"), "utf8"),
  readFile(resolve(artifactRoot, "fixed_vs_middle_rows.csv"), "utf8"),
]);

const summary = JSON.parse(summaryText);
const fixedRows = parseNumericCsv(fixedText);
const comparisonRows = parseNumericCsv(comparisonText);

if (fixedRows.length !== comparisonRows.length) {
  throw new Error(`Row mismatch: fixed=${fixedRows.length}, comparison=${comparisonRows.length}`);
}

const rows = fixedRows.map((fixed, index) => {
  const comparison = comparisonRows[index];
  if (fixed.filler_tokens !== comparison.filler_tokens) {
    throw new Error(`Filler mismatch at row ${index}`);
  }
  return {
    fillerTokens: fixed.filler_tokens,
    promptTokens: fixed.prompt_tokens,
    keyLength: fixed.key_length,
    evidencePosition: fixed.evidence_position,
    queryPosition: fixed.query_position,
    fixedDistance: fixed.relative_distance,
    middleDistance: comparison.middle_relative_distance,
    fixed: {
      ppl: fixed.gold_ppl,
      goldProbability: fixed.gold_probability,
      evidenceMass: fixed.mean_evidence_mass,
      evidenceLogit: fixed.mean_evidence_logit,
      evidenceCosine: fixed.mean_evidence_cosine,
      evidenceKeyNorm: fixed.mean_evidence_key_norm,
      queryNorm: fixed.mean_query_norm,
      evidenceRank: fixed.mean_evidence_rank,
      logsumexp: fixed.mean_head_logsumexp,
      maxLogit: fixed.mean_head_max_logit,
      top2HeadFraction: fixed.target_top2pct_head_fraction,
    },
    middle: {
      ppl: comparison.middle_gold_ppl,
      evidenceMass: comparison.middle_evidence_mass,
      evidenceLogit: comparison.middle_evidence_logit,
      evidenceCosine: comparison.middle_evidence_cosine,
    },
  };
});

const payload = {
  schemaVersion: 1,
  experiment: "fixed-relative-distance-328-vs-middle",
  model: summary.model,
  condition: summary.condition,
  chain: ["river", "window", "basket"],
  queryMode: "full2",
  seed: 0,
  fillerStep: 500,
  fixedBodyOverhead: summary.fixed_body_overhead,
  rowCount: rows.length,
  fixed: summary.fixed,
  middle: summary.middle,
  comparison: summary.fixed_vs_middle,
  rows,
};

await mkdir(dirname(outputFile), { recursive: true });
await writeFile(outputFile, `${JSON.stringify(payload)}\n`, "utf8");
console.log(`Wrote ${rows.length} rows to ${outputFile}`);
