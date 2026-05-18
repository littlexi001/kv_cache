from __future__ import annotations

import argparse
from pathlib import Path

from random_quadruple_data import (
    QuadrupleTableSpec,
    load_quadruple_table,
    save_quadruple_table,
    validate_quadruple_table,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_file",
        default=str(PROJECT_DIR / "data" / "random_quadruples_1000_100000.pt"),
    )
    parser.add_argument("--token_min", type=int, default=1)
    parser.add_argument("--token_max", type=int, default=1000)
    parser.add_argument("--quadruple_len", type=int, default=4)
    parser.add_argument("--num_quadruples", type=int, default=100_000)
    parser.add_argument("--quadruple_seed", type=int, default=20_260_518)
    parser.add_argument("--force", type=str2bool, default=False)
    args = parser.parse_args()

    output_file = Path(args.output_file)
    if not output_file.is_absolute():
        output_file = PROJECT_DIR / output_file
    args.output_file = str(output_file)
    return args


def main() -> None:
    args = parse_args()
    spec = QuadrupleTableSpec(
        token_min=args.token_min,
        token_max=args.token_max,
        quadruple_len=args.quadruple_len,
        num_quadruples=args.num_quadruples,
        seed=args.quadruple_seed,
    )
    output_file = Path(args.output_file)
    save_quadruple_table(output_file, spec, overwrite=args.force)
    table, metadata = load_quadruple_table(output_file)
    validate_quadruple_table(table, spec, metadata)
    print(f"wrote quadruple table: {output_file}", flush=True)
    print(f"shape={tuple(table.shape)}", flush=True)
    print(f"token_range=[{int(table.min().item())}, {int(table.max().item())}]", flush=True)
    print(f"seed={metadata.get('seed', args.quadruple_seed)}", flush=True)


if __name__ == "__main__":
    main()
