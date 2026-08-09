from __future__ import annotations

import argparse
from pathlib import Path

from data_pipeline import ensure_manifests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dclm-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-files", type=int, default=4096)
    parser.add_argument("--validation-files", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1701)
    args = parser.parse_args()
    metadata = ensure_manifests(
        args.dclm_root,
        args.output_dir,
        args.train_files,
        args.validation_files,
        args.seed,
    )
    print(metadata)


if __name__ == "__main__":
    main()

