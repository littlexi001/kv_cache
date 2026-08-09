#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
CHECKOUT="${QKSIEVE_RETROINFER_CHECKOUT:-$ROOT/external/RetrievalAttention}"
REPORT="${QKSIEVE_RETROINFER_AUDIT:-$ROOT/results/20260728_retroinfer_official_checkout_audit.json}"
REPOSITORY=https://github.com/microsoft/RetrievalAttention.git
COMMIT=6b1228c346836769da0ed525dadf05bb7010e96b

mkdir -p "$(dirname "$CHECKOUT")" "$(dirname "$REPORT")"
if [[ ! -d "$CHECKOUT/.git" ]]; then
  git clone --filter=blob:none "$REPOSITORY" "$CHECKOUT"
fi

if [[ -n "$(git -C "$CHECKOUT" status --short --untracked-files=no)" ]]; then
  echo "Refusing to change a tracked-dirty RetroInfer checkout: $CHECKOUT" >&2
  exit 2
fi

git -C "$CHECKOUT" fetch origin "$COMMIT"
git -C "$CHECKOUT" checkout --detach "$COMMIT"
actual="$(git -C "$CHECKOUT" rev-parse HEAD)"
if [[ "$actual" != "$COMMIT" ]]; then
  echo "RetroInfer commit mismatch: expected $COMMIT, got $actual" >&2
  exit 2
fi

python -u "$ROOT/src/audit_retroinfer_official_checkout_20260728.py" \
  --checkout "$CHECKOUT" \
  --output "$REPORT"

cat <<'EOF'
Pinned source audit passed. No Python packages or CUDA kernels were installed.
Use a dedicated Python 3.10.16 / CUDA 12.4 environment for the native run.
Dependency commits recorded by the audit must be validated before installation.
EOF
