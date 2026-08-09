import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { gzipSync } from "node:zlib";

const here = dirname(fileURLToPath(import.meta.url));
const dashboardRoot = resolve(here, "..");
const artifactRoot = resolve(
  dashboardRoot,
  "../artifacts/20260724_candidate_margin_dense_34_100",
);
const rawRoot = resolve(artifactRoot, "data");
const outputFile = resolve(
  dashboardRoot,
  "public/data/failure_boundary_dense/payload.json.gz",
);

const round = (value, digits = 8) =>
  Number(Number(value).toFixed(digits));

const points = [];
for (let length = 34; length <= 100; length += 1) {
  const payload = JSON.parse(
    await readFile(resolve(rawRoot, `length_${length}.json`), "utf8"),
  );
  const { answer, attention } = payload;
  const gold = answer.gold_answer;
  const goldProbability = Number(answer.gold_token_scores[0].probability);
  const strongestWrong = answer.next_token_top5
    .filter((row) => row.token.trim() !== gold)
    .sort((left, right) => right.probability - left.probability)[0];
  const letRow = answer.next_token_top5.find(
    (row) => row.token.trim() === "Let",
  );
  const roleMass = attention.head_role_mass.map((layer) =>
    layer.map((head) => head.slice(0, 4).map((value) => round(value, 10))),
  );
  const roleLogit = attention.head_role_logit_mean.map((layer) =>
    layer.map((head) => head.slice(0, 4).map((value) => round(value, 7))),
  );
  points.push({
    length,
    promptTokens: payload.prompt_tokens,
    goldPpl: round(answer.gold_ppl, 8),
    goldProbability: round(goldProbability, 10),
    fullVocabMargin: round(
      Math.log(goldProbability) - Math.log(strongestWrong.probability),
      8,
    ),
    fullVocabCorrect:
      answer.next_token_top5[0].token.trim() === answer.gold_answer,
    candidateMargin: round(answer.candidate_margin, 8),
    candidateCorrect: Boolean(answer.candidate_correct),
    candidatePrediction: answer.candidate_prediction,
    topToken: answer.next_token_top5[0].token.trim(),
    topProbability: round(answer.next_token_top5[0].probability, 10),
    letProbability: round(letRow?.probability ?? 0, 10),
    roleMass,
    roleLogit,
    headLogsumexp: attention.head_logsumexp.map((layer) =>
      layer.map((value) => round(value, 7)),
    ),
    headEntropy: attention.head_entropy.map((layer) =>
      layer.map((value) => round(value, 7)),
    ),
  });
}

const layerHeadSummary = JSON.parse(
  await readFile(
    resolve(artifactRoot, "dense_analysis/layer_head_summary.json"),
    "utf8",
  ),
);

const output = {
  schemaVersion: 1,
  experiment: "single-sample-dense-failure-boundary",
  model: "Qwen3-8B",
  condition: "clean two-hop",
  chain: ["river", "window", "basket"],
  placement: "middle",
  lengths: points.map((point) => point.length),
  numLayers: points[0].roleMass.length,
  numHeads: points[0].roleMass[0].length,
  roleOrder: ["start_key", "hop1_result", "hop2_input", "hop2_result"],
  roleLabels: {
    start_key: "起始键 river",
    hop1_result: "第一跳结果 window",
    hop2_input: "第二跳输入 window",
    hop2_result: "最终结果 basket",
  },
  summary: layerHeadSummary,
  points,
};

await mkdir(dirname(outputFile), { recursive: true });
await writeFile(outputFile, gzipSync(`${JSON.stringify(output)}\n`));
console.log(
  `Wrote ${points.length} dense points (${output.numLayers} layers × ${output.numHeads} heads) to ${outputFile}`,
);
