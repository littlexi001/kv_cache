from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
YMLUO_UTILS_DIR = REPO_ROOT / "ymluo" / "utils"
if str(YMLUO_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(YMLUO_UTILS_DIR))

from moe_selectivity_experiment import run_experiment  # noqa: E402


if __name__ == "__main__":
    run_experiment("forced_warmup")
