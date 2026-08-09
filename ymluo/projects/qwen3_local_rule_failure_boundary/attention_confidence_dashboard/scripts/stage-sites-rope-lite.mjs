import { cp, mkdir, readFile, readdir, stat, writeFile } from "node:fs/promises";
import path from "node:path";

const [projectArg, stageArg] = process.argv.slice(2);
if (!projectArg || !stageArg) {
  throw new Error("usage: node scripts/stage-sites-rope-lite.mjs PROJECT_DIR STAGE_DIR");
}

const project = path.resolve(projectArg);
const stage = path.resolve(stageArg);
const dist = path.join(project, "dist");
const stageDist = path.join(stage, "dist");
const sourceData = path.join(dist, "client", "data", "english_single_token");
const targetData = path.join(stageDist, "client", "data", "english_single_token");
const anchorLengths = new Set([0, 8_000, 32_000, 64_000, 128_000]);

await mkdir(path.join(stage, ".openai"), { recursive: true });
await cp(path.join(project, ".openai", "hosting.json"), path.join(stage, ".openai", "hosting.json"));
await cp(dist, stageDist, {
  recursive: true,
  filter(source) {
    return !source.startsWith(path.join(dist, "client", "data"));
  },
});

await mkdir(targetData, { recursive: true });
await cp(path.join(sourceData, "rope_pair_64k"), path.join(targetData, "rope_pair_64k"), { recursive: true });

for (const name of [
  "fixed_relative_328.json",
  "pre_softmax_summary.json",
  "pre_softmax_head_length_summary.json.gz",
]) {
  await cp(path.join(sourceData, name), path.join(targetData, name));
}

const manifest = JSON.parse(await readFile(path.join(sourceData, "manifest.json"), "utf8"));
manifest.completed_lengths = manifest.completed_lengths.filter((length) => anchorLengths.has(length));
manifest.summaries = manifest.summaries.filter((row) => anchorLengths.has(row.length));
await writeFile(path.join(targetData, "manifest.json"), `${JSON.stringify(manifest)}\n`, "utf8");

for (const summary of manifest.summaries) {
  const relativeFile = summary.file.replace(/^data\//, "");
  await cp(path.join(sourceData, relativeFile), path.join(targetData, relativeFile));
}

async function directoryBytes(directory) {
  const entries = await readdir(directory, { withFileTypes: true });
  let total = 0;
  for (const entry of entries) {
    const absolute = path.join(directory, entry.name);
    total += entry.isDirectory() ? await directoryBytes(absolute) : (await stat(absolute)).size;
  }
  return total;
}

const bytes = await directoryBytes(stageDist);
process.stdout.write(JSON.stringify({ stage, bytes, megabytes: bytes / 1024 / 1024 }));
