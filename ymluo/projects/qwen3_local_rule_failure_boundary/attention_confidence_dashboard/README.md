# Attention Confidence Lab

Interactive frontend for the single-case Qwen3-8B clean two-hop length sweep.

The dashboard supports:

- switching between legacy multi-token identifiers, the matched Chinese
  single-token control, and ordinary-English single-token evidence up to 128K;
- a data-driven filler-length slider with 500-token steps;
- model-wide, per-layer, and per-head softmax attention views;
- Top-X selection from 5 to 100 plus the exact remaining attention mass;
- a token attention trace across every available sequence length, following the
  current overall, layer, or head scope;
- separate tracking of the first-hop result, second-hop input, and final result;
- a `POST-SOFTMAX` / `PRE-SOFTMAX QK` switch for the English 128K run; the
  pre-softmax view aligns exact evidence logits, Q/K cosine, rank, competitor
  gap, per-head logsumexp, and all 257 length points;
- per-answer-token confidence, PPL curve, attention entropy, recent-token mass,
  sink-token mass, and a clickable layer/head evidence heatmap.
- a 64K RoPE-pair decomposition view with selectable layer/head, all 64
  split-half dimension pairs, the pair-summed raw-logit curve, the post-RoPE,
  inverse-RoPE, and post-minus-pre views, plus a parameter-only position kernel
  computed live in the browser.

## Data contract

The legacy mode loads `public/data/manifest.json`; the two single-token modes
load `public/data/single_token/manifest.json` and
`public/data/english_single_token/manifest.json`. It then lazily fetches the selected
length file. Both plain `.json` and gzip-compressed `.json.gz` files are
supported. If no data bundle exists, the UI displays a clearly marked preview
dataset so frontend work can continue while the remote run is active.

The production experiment creates a compact bundle under:

```text
outputs/attention_confidence_qwen3_8b_20260717/site_data/
```

Copy the contents of that directory into `public/data/` to connect the complete
result without changing frontend code.

From this repository on Windows, the provided helper performs that copy:

```powershell
..\scripts\pull_attention_confidence_dashboard_data.ps1
..\scripts\pull_attention_confidence_dashboard_data.ps1 -Mode english_single_token
..\scripts\pull_attention_confidence_dashboard_data.ps1 -Mode rope_pair_64k
```

## Local development

Node.js 22.13 or newer is required.

```bash
pnpm install
pnpm dev
pnpm build
```

On Windows, the repository launcher uses the bundled Codex Node runtime and
starts the dashboard on localhost. Keep that PowerShell window open while using
the dashboard:

```powershell
powershell -ExecutionPolicy Bypass -File ..\scripts\serve_dashboard_local.ps1
```

Open `http://localhost:3000`. Enter an evidence code such as `佑`, `丝`, `须`,
`river`, `window`, or `basket` to read its exact evidence-span attention. Enter
`OTHER` for the exact
`1 - Top-X` attention mass. Other token strings are matched against the saved
Top-100 positions, so a zero means "not visible in Top-100", not necessarily
zero model attention.

The pre-softmax payload is explicit about its limit: the raw run saved exact
layer/head diagnostics for the marked evidence roles, but did not persist every
token's full QK-logit vector. The pre-softmax page therefore shows the three
evidence positions and competition statistics, not a fabricated all-token
Top-100 list.

On Windows, a deeply nested checkout can exceed Node's path limit with pnpm's
default virtual store. Install with a short virtual-store directory if needed:

```powershell
pnpm install --virtual-store-dir "$HOME/.pnpm-vstore/attention-confidence"
```
