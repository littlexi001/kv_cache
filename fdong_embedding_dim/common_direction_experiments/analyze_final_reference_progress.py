#!/usr/bin/env python3
"""Test final-reference progress of singular vectors vs singular values.

For each seed, train the same toy attention LM used by
run_two_phase_singular_dynamics.py.  Store small matrices at checkpoints, take
the final checkpoint as reference, and compare:

  vector progress:
      (cos^2(v_t, v_T) - cos^2(v_0, v_T)) / (1 - cos^2(v_0, v_T))

  sigma progress:
      (sigma_t - sigma_0) / (sigma_T - sigma_0)

Both are clipped only in the summary pass for threshold timing.  Raw progress is
kept in the CSV because overshoot/non-monotonicity is itself diagnostic.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from run_two_phase_singular_dynamics import (
    AttnLM,
    Config,
    build_init_and_data,
    compute_grad,
    set_seed,
    weighted_loss_metrics,
)


def centered(x: torch.Tensor) -> torch.Tensor:
    return x.detach().float() - x.detach().float().mean(dim=0, keepdim=True)


def top_svd(matrix: torch.Tensor) -> Tuple[torch.Tensor, float, torch.Tensor]:
    u, s, vh = torch.linalg.svd(matrix.detach().float(), full_matrices=False)
    return u[:, 0].detach(), float(s[0]), vh[0].detach()


def sqcos(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = a.norm() * b.norm()
    if float(denom) <= 1e-12:
        return 0.0
    return float(((a @ b) / denom).square())


def snapshot(model: AttnLM) -> Dict[str, torch.Tensor]:
    return {
        "E_centered": centered(model.E),
        "Wq": model.Wq.detach().clone(),
        "Wk": model.Wk.detach().clone(),
        "Wv": model.Wv.detach().clone(),
        "Wo": model.Wo.detach().clone(),
        "Bqk": model.Wq.detach().T @ model.Wk.detach(),
    }


def progress(numer: float, denom: float) -> float:
    if abs(denom) <= 1e-12:
        return 1.0
    return numer / denom


def train_collect(cfg: Config, condition: str, seed: int) -> List[Dict[str, object]]:
    set_seed(seed)
    data = build_init_and_data(cfg, condition)
    model = AttnLM(data["E0"], cfg.dim, seed, cfg.init_noise, cfg.residual_alpha, cfg.use_o_proj)
    checkpoints: List[Dict[str, object]] = []
    for step in range(cfg.steps + 1):
        if step % cfg.record_every == 0 or step == cfg.steps:
            checkpoints.append(
                {
                    "step": step,
                    "matrices": snapshot(model),
                    "metrics": weighted_loss_metrics(model, data),
                }
            )
        if step == cfg.steps:
            break
        grads = compute_grad(model, data)
        with torch.no_grad():
            for name in ["E", "Wq", "Wk", "Wv", "Wo"]:
                getattr(model, name).sub_(cfg.lr * grads[name])

    modules = ["E_centered", "Wq", "Wk", "Wv", "Wo", "Bqk"]
    final = checkpoints[-1]["matrices"]
    first = checkpoints[0]["matrices"]
    refs: Dict[str, Dict[str, object]] = {}
    for module in modules:
        u0, sigma0, v0 = top_svd(first[module])
        uT, sigmaT, vT = top_svd(final[module])
        refs[module] = {
            "u0": u0,
            "v0": v0,
            "uT": uT,
            "vT": vT,
            "sigma0": sigma0,
            "sigmaT": sigmaT,
            "right_c0": sqcos(v0, vT),
            "left_c0": sqcos(u0, uT),
        }

    rows: List[Dict[str, object]] = []
    for ckpt in checkpoints:
        step = int(ckpt["step"])
        for module in modules:
            u, sigma, v = top_svd(ckpt["matrices"][module])
            ref = refs[module]
            right_closeness = sqcos(v, ref["vT"])
            left_closeness = sqcos(u, ref["uT"])
            sigma_prog = progress(sigma - float(ref["sigma0"]), float(ref["sigmaT"]) - float(ref["sigma0"]))
            right_prog = progress(right_closeness - float(ref["right_c0"]), 1.0 - float(ref["right_c0"]))
            left_prog = progress(left_closeness - float(ref["left_c0"]), 1.0 - float(ref["left_c0"]))
            rows.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "step": step,
                    "module": module,
                    "loss": ckpt["metrics"]["loss"],
                    "tail_loss": ckpt["metrics"]["tail_loss"],
                    "common_loss": ckpt["metrics"]["common_loss"],
                    "sigma1": sigma,
                    "sigma0": ref["sigma0"],
                    "sigmaT": ref["sigmaT"],
                    "right_closeness_to_final": right_closeness,
                    "right_initial_closeness_to_final": ref["right_c0"],
                    "left_closeness_to_final": left_closeness,
                    "left_initial_closeness_to_final": ref["left_c0"],
                    "sigma_progress": sigma_prog,
                    "right_vector_progress": right_prog,
                    "left_vector_progress": left_prog,
                    "right_minus_sigma_progress": right_prog - sigma_prog,
                    "left_minus_sigma_progress": left_prog - sigma_prog,
                    "abs_right_minus_sigma_progress": abs(right_prog - sigma_prog),
                    "abs_left_minus_sigma_progress": abs(left_prog - sigma_prog),
                }
            )
    return rows


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str, int], List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["condition"]), str(row["module"]), int(row["step"])), []).append(row)
    out: List[Dict[str, object]] = []
    numeric = [
        k
        for k, v in rows[0].items()
        if k not in {"condition", "seed", "step", "module"} and isinstance(v, (int, float))
    ]
    for (condition, module, step), items in sorted(grouped.items()):
        rec: Dict[str, object] = {"condition": condition, "module": module, "step": step, "num_seeds": len(items)}
        for key in numeric:
            vals = np.array([float(x[key]) for x in items], dtype=np.float64)
            rec[f"{key}_mean"] = float(vals.mean())
            rec[f"{key}_std"] = float(vals.std(ddof=0))
        out.append(rec)
    return out


def first_reach(rows: List[Dict[str, object]], key: str, threshold: float) -> int | None:
    ordered = sorted(rows, key=lambda r: int(r["step"]))
    for row in ordered:
        if float(row[key]) >= threshold:
            return int(row["step"])
    return None


def summarize(rows: List[Dict[str, object]], agg: List[Dict[str, object]]) -> Dict[str, object]:
    modules = sorted({str(r["module"]) for r in rows})
    conditions = sorted({str(r["condition"]) for r in rows})
    summary: Dict[str, object] = {"conditions": {}}
    for condition in conditions:
        summary["conditions"][condition] = {"modules": {}}
        for module in modules:
            ars = [r for r in agg if r["condition"] == condition and r["module"] == module]
            final = sorted(ars, key=lambda r: int(r["step"]))[-1]
            raw = [r for r in rows if r["condition"] == condition and r["module"] == module]
            by_seed: Dict[int, List[Dict[str, object]]] = {}
            for row in raw:
                by_seed.setdefault(int(row["seed"]), []).append(row)
            threshold_stats: Dict[str, object] = {}
            for th in [0.5, 0.8, 0.9, 0.95]:
                sigma_steps = []
                right_steps = []
                left_steps = []
                for items in by_seed.values():
                    sigma_step = first_reach(items, "sigma_progress", th)
                    right_step = first_reach(items, "right_vector_progress", th)
                    left_step = first_reach(items, "left_vector_progress", th)
                    if sigma_step is not None:
                        sigma_steps.append(sigma_step)
                    if right_step is not None:
                        right_steps.append(right_step)
                    if left_step is not None:
                        left_steps.append(left_step)
                threshold_stats[f"t{int(th * 100)}_sigma_step_mean"] = float(np.mean(sigma_steps)) if sigma_steps else None
                threshold_stats[f"t{int(th * 100)}_right_step_mean"] = float(np.mean(right_steps)) if right_steps else None
                threshold_stats[f"t{int(th * 100)}_left_step_mean"] = float(np.mean(left_steps)) if left_steps else None
            summary["conditions"][condition]["modules"][module] = {
                "mean_abs_right_minus_sigma_progress": float(
                    np.mean([float(r["abs_right_minus_sigma_progress_mean"]) for r in ars])
                ),
                "max_abs_right_minus_sigma_progress": float(
                    np.max([float(r["abs_right_minus_sigma_progress_mean"]) for r in ars])
                ),
                "mean_abs_left_minus_sigma_progress": float(
                    np.mean([float(r["abs_left_minus_sigma_progress_mean"]) for r in ars])
                ),
                "max_abs_left_minus_sigma_progress": float(
                    np.max([float(r["abs_left_minus_sigma_progress_mean"]) for r in ars])
                ),
                "final_sigma1_mean": final["sigma1_mean"],
                "initial_right_closeness_to_final_mean": ars[0]["right_initial_closeness_to_final_mean"],
                "initial_left_closeness_to_final_mean": ars[0]["left_initial_closeness_to_final_mean"],
                **threshold_stats,
            }
    return summary


def plot(path: Path, agg: List[Dict[str, object]], condition: str) -> None:
    modules = ["E_centered", "Wq", "Wk", "Wv", "Wo", "Bqk"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharex=True, sharey=True)
    for ax, module in zip(axes.flat, modules):
        rows = sorted(
            [r for r in agg if r["condition"] == condition and r["module"] == module],
            key=lambda r: int(r["step"]),
        )
        xs = [int(r["step"]) for r in rows]
        ax.plot(xs, [float(r["right_vector_progress_mean"]) for r in rows], label="right vector progress")
        ax.plot(xs, [float(r["left_vector_progress_mean"]) for r in rows], label="left vector progress")
        ax.plot(xs, [float(r["sigma_progress_mean"]) for r in rows], label="sigma progress")
        ax.set_title(module)
        ax.grid(alpha=0.25)
        ax.set_ylim(-0.05, 1.05)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Progress to final checkpoint, normalized to [0,1]: {condition}")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--dim", type=int, default=4)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--record_every", type=int, default=20)
    parser.add_argument("--theta_deg", type=float, default=12.0)
    parser.add_argument("--init_noise", type=float, default=0.005)
    parser.add_argument("--residual_alpha", type=float, default=0.0)
    parser.add_argument("--no_o_proj", action="store_true")
    parser.add_argument("--conditions", default="withK_zipf")
    args = parser.parse_args()
    cfg = Config(
        outdir=args.outdir,
        seeds=args.seeds,
        dim=args.dim,
        steps=args.steps,
        lr=args.lr,
        record_every=args.record_every,
        theta_deg=args.theta_deg,
        init_noise=args.init_noise,
        residual_alpha=args.residual_alpha,
        use_o_proj=not args.no_o_proj,
    )
    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    for condition in [x.strip() for x in args.conditions.split(",") if x.strip()]:
        for seed in [int(x) for x in cfg.seeds.split(",") if x.strip()]:
            print(f"running final-reference progress condition={condition} seed={seed}", flush=True)
            rows.extend(train_collect(cfg, condition, seed))
    agg = aggregate(rows)
    summary = summarize(rows, agg)
    write_csv(outdir / "history.csv", rows)
    write_csv(outdir / "aggregate.csv", agg)
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (outdir / "config.json").write_text(json.dumps({**asdict(cfg), "conditions": args.conditions}, indent=2) + "\n")
    for condition in [x.strip() for x in args.conditions.split(",") if x.strip()]:
        plot(outdir / f"final_reference_progress_{condition}.png", agg, condition)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
