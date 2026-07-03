#!/usr/bin/env bash
set -euo pipefail

source /home/fdong/miniconda3/bin/activate moe

EXTERNAL_ROOT="${EXTERNAL_ROOT:-/home/fdong/ymluo/external}"
REPO_DIR="${REPO_DIR:-$EXTERNAL_ROOT/KVCache-Factory}"
PYDEPS_DIR="${PYDEPS_DIR:-/home/fdong/ymluo/pydeps/kvcache_factory_tf444}"
REPO_URL="${REPO_URL:-https://github.com/Zefan-Cai/KVCache-Factory.git}"

mkdir -p "$EXTERNAL_ROOT" "$(dirname "$PYDEPS_DIR")"

if [[ ! -d "$REPO_DIR/.git" ]]; then
  git clone --depth 1 "$REPO_URL" "$REPO_DIR"
fi

if [[ ! -d "$PYDEPS_DIR/transformers" ]]; then
  python -m pip install --target "$PYDEPS_DIR" "transformers==4.44.2" "tokenizers==0.19.1"
  rm -rf "$PYDEPS_DIR"/numpy "$PYDEPS_DIR"/numpy-* "$PYDEPS_DIR"/fsspec "$PYDEPS_DIR"/fsspec-*
fi

python - "$REPO_DIR" <<'PY'
from pathlib import Path
import sys

repo = Path(sys.argv[1])

patches = {
    repo / "pyramidkv" / "llama_model.py": (
        "from transformers.generation.utils import Cache, DynamicCache, StaticCache",
        "try:\n"
        "    from transformers.generation.utils import Cache, DynamicCache, StaticCache\n"
        "except ImportError:\n"
        "    from transformers.cache_utils import Cache, DynamicCache, StaticCache",
    ),
    repo / "pyramidkv" / "llama_model_think.py": (
        "from transformers.generation.utils import Cache, DynamicCache, StaticCache",
        "try:\n"
        "    from transformers.generation.utils import Cache, DynamicCache, StaticCache\n"
        "except ImportError:\n"
        "    from transformers.cache_utils import Cache, DynamicCache, StaticCache",
    ),
    repo / "pyramidkv" / "cache_utils_think.py": (
        "from transformers.utils import is_torchdynamo_compiling, is_quanto_available",
        "try:\n"
        "    from transformers.utils import is_torchdynamo_compiling, is_quanto_available\n"
        "except ImportError:\n"
        "    from transformers.utils import is_torchdynamo_compiling\n"
        "    def is_quanto_available():\n"
        "        return False",
    ),
}

for path, (old, new) in patches.items():
    text = path.read_text(encoding="utf-8")
    if new in text:
        continue
    if old not in text:
        print(f"[warn] patch anchor not found: {path}")
        continue
    path.write_text(text.replace(old, new), encoding="utf-8")
    print(f"[patched] {path}")
PY

PYTHONPATH="$PYDEPS_DIR:$REPO_DIR:${PYTHONPATH:-}" python - <<'PY'
import numpy
import fsspec
import transformers
from transformers.models.llama import modeling_llama

print("transformers", transformers.__version__)
print("numpy", numpy.__version__)
print("fsspec", fsspec.__version__)
print("has LlamaFlashAttention2", hasattr(modeling_llama, "LlamaFlashAttention2"))
print("has LlamaSdpaAttention", hasattr(modeling_llama, "LlamaSdpaAttention"))
PY
