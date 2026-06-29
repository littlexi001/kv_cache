#!/usr/bin/env python3
"""Post-hoc lazy pairwise-gate analysis for PCIC-CR.

This script uses existing blockwise CSV files only. It estimates whether a
calibration-only rule can skip pairwise sentinel probes and directly keep the
risk-memory anchor without losing the quality of conffast_s8.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path


OFFSETS = ("8192", "16384", "24576", "32768")
DATASETS = ("war", "monte")


@dataclass(frozen=True)
class Block:
    dataset: str
    offset: str
    block: int
    s8_delta_ppl: float
    memory_delta_ppl: float
    selected_combo: str
    memory_combo: str
    min_loss_combo: str
    triggered: bool
    candidate_count: int
    min_loss_delta_loss: float
    memory_delta_loss: float
    pairwise_delta_loss: float
    best_margin: float
    sentinel_route: str
    fast_route: str

    @property
    def calib_gap(self) -> float:
        return self.memory_delta_loss - self.min_loss_delta_loss

    @property
    def memory_selected(self) -> bool:
        return self.selected_combo == self.memory_combo


def read_eval_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="") as handle:
        return [row for row in csv.DictReader(handle) if row.get("kind") == "pcic_r_eval"]


def load_blocks(root: Path) -> list[Block]:
    blocks: list[Block] = []
    for dataset in DATASETS:
        for offset in OFFSETS:
            conffast = (
                root
                / f"server_pcic_r3_{dataset}_off{offset}_b4_conffast_s8_seed64_allm01_pairm005_slack03_delta005_eager"
                / "pcic_r_blockwise_results.csv"
            )
            memory = (
                root
                / f"server_pcic_r3_{dataset}_off{offset}_b4_riskmemory_monogate_eager"
                / "pcic_r_blockwise_results.csv"
            )
            conffast_rows = read_eval_rows(conffast)
            memory_rows = read_eval_rows(memory)
            memory_by_block = {int(row["block"]): row for row in memory_rows}
            for row in conffast_rows:
                block_idx = int(row["block"])
                rule = json.loads(row.get("rescue_rule") or "{}")
                memory_row = memory_by_block[block_idx]
                blocks.append(
                    Block(
                        dataset=dataset,
                        offset=offset,
                        block=block_idx,
                        s8_delta_ppl=float(row["delta_ppl"]),
                        memory_delta_ppl=float(memory_row["delta_ppl"]),
                        selected_combo=row["combo"],
                        memory_combo=str(rule.get("memory_combo") or memory_row["combo"]),
                        min_loss_combo=str(rule.get("min_loss_combo") or ""),
                        triggered=bool(int(rule.get("triggered") or 0)),
                        candidate_count=len(rule.get("candidate_combos") or []),
                        min_loss_delta_loss=float(rule.get("min_loss_delta_loss") or 0.0),
                        memory_delta_loss=float(rule.get("memory_delta_loss") or 0.0),
                        pairwise_delta_loss=float(rule.get("sentinel_pairwise_delta_loss") or 0.0),
                        best_margin=float(rule.get("sentinel_best_margin") or 0.0),
                        sentinel_route=str(rule.get("sentinel_route") or ""),
                        fast_route=str(rule.get("fast_route") or ""),
                    )
                )
    return blocks


def summarize(name: str, values: list[float], saved: int, total_pairwise: int, wrong_skips: int) -> str:
    return (
        f"| {name} | {sum(values) / len(values):.6f} | {max(values):.6f} | "
        f"{saved}/{total_pairwise} | {wrong_skips} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs_root", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    blocks = load_blocks(args.outputs_root)
    pairwise = [block for block in blocks if block.triggered and block.candidate_count == 2]
    total_pairwise = len(pairwise)
    base_values = [block.s8_delta_ppl for block in blocks]
    memory_values = [block.memory_delta_ppl for block in blocks]

    print("## dataset")
    print("| item | value |")
    print("|---|---:|")
    print(f"| blocks | {len(blocks)} |")
    print(f"| triggered_pairwise_blocks | {total_pairwise} |")
    print(f"| s8_memory_selected_pairwise | {sum(block.memory_selected for block in pairwise)} |")
    print(f"| s8_minloss_selected_pairwise | {sum(not block.memory_selected for block in pairwise)} |")
    print()

    print("## block_features")
    print(
        "| dataset | offset | block | s8_combo | mem_combo | min_combo | s8_route | "
        "min_dl | mem_dl | calib_gap | pair_delta | memory_selected | s8_delta | memory_delta |"
    )
    print("|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for block in pairwise:
        print(
            f"| {block.dataset} | {block.offset} | {block.block} | {block.selected_combo} | "
            f"{block.memory_combo} | {block.min_loss_combo} | {block.sentinel_route} | "
            f"{block.min_loss_delta_loss:.4f} | {block.memory_delta_loss:.4f} | "
            f"{block.calib_gap:.4f} | {block.pairwise_delta_loss:.4f} | "
            f"{int(block.memory_selected)} | {block.s8_delta_ppl:.6f} | {block.memory_delta_ppl:.6f} |"
        )
    print()

    print("## policy_grid")
    print("| policy | mean_delta_ppl | worst_delta_ppl | saved_pairwise_probes | wrong_skips |")
    print("|---|---:|---:|---:|---:|")
    print(summarize("conffast_s8", base_values, 0, total_pairwise, 0))
    print(summarize("always_memory", memory_values, total_pairwise, total_pairwise, sum(not b.memory_selected for b in pairwise)))

    candidates: list[tuple[str, list[float], int, int]] = []
    for min_dl_threshold in (-0.10, -0.075, -0.05, -0.025, 0.0, 0.025, 0.05):
        for gap_threshold in (0.0, 0.02, 0.05, 0.08, 0.12, 0.16):
            for mem_dl_threshold in (-0.08, -0.04, 0.0, 0.04, 0.08):
                values: list[float] = []
                saved = 0
                wrong = 0
                for block in blocks:
                    skip = (
                        block.triggered
                        and block.candidate_count == 2
                        and block.min_loss_delta_loss >= min_dl_threshold
                        and block.calib_gap <= gap_threshold
                        and block.memory_delta_loss <= mem_dl_threshold
                    )
                    if skip:
                        saved += 1
                        wrong += int(not block.memory_selected)
                        values.append(block.memory_delta_ppl)
                    else:
                        values.append(block.s8_delta_ppl)
                name = (
                    f"min_dl>={min_dl_threshold:g},gap<={gap_threshold:g},"
                    f"mem_dl<={mem_dl_threshold:g}"
                )
                candidates.append((name, values, saved, wrong))

    candidates.sort(key=lambda item: (item[3], sum(item[1]) / len(item[1]), -item[2]))
    for name, values, saved, wrong in candidates[:20]:
        print(summarize(name, values, saved, total_pairwise, wrong))

    print()
    print("## best_safe")
    safe = [item for item in candidates if item[3] == 0 and item[2] > 0]
    if not safe:
        print("No zero-wrong skip policy found in this grid.")
    else:
        safe.sort(key=lambda item: (-item[2], sum(item[1]) / len(item[1])))
        for name, values, saved, wrong in safe[:10]:
            print(summarize(name, values, saved, total_pairwise, wrong))

    print()
    print("## best_nonzero_savings")
    nonzero = [item for item in candidates if item[2] > 0]
    nonzero.sort(key=lambda item: (sum(item[1]) / len(item[1]), item[3], -item[2]))
    for name, values, saved, wrong in nonzero[:20]:
        print(summarize(name, values, saved, total_pairwise, wrong))

    print()
    print("## best_by_saved_probe_count")
    for min_saved in (1, 2, 4, 8, 12):
        eligible = [item for item in candidates if item[2] >= min_saved]
        if not eligible:
            continue
        eligible.sort(key=lambda item: (sum(item[1]) / len(item[1]), item[3], -item[2]))
        name, values, saved, wrong = eligible[0]
        print(summarize(f"saved>={min_saved}: {name}", values, saved, total_pairwise, wrong))


if __name__ == "__main__":
    main()
