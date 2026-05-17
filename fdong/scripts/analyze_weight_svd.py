import argparse
import json
import math
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch


DEFAULT_RUNS = ",".join(
    [
        "inverse-kv-local-h128-l3-top1",
        "inverse-kv-attn-output-router",
        "inverse-kv-head-moe-hidden-router",
        "inverse-kv-attn-output-head-moe",
    ]
)


ATTN_RE = re.compile(r"model\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.weight$")
EXPERT_RE = re.compile(
    r"model\.layers\.(\d+)\.mlp\.experts\.(.+)\.(gate_proj|up_proj|down_proj)\.weight$"
)
ROUTER_RE = re.compile(r"model\.layers\.(\d+)\.mlp\.routers?\.(\d+\.)?weight$")


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze singular-value spectra of trained weights.")
    parser.add_argument("--checkpoint_root", type=str, default="../checkpoints")
    parser.add_argument("--runs", type=str, default=DEFAULT_RUNS)
    parser.add_argument("--checkpoint_step", type=int, default=5000)
    parser.add_argument("--output_dir", type=str, default="../experiments/weight_svd_step5000")
    parser.add_argument("--include_router", action="store_true", default=True)
    parser.add_argument("--no_include_router", action="store_false", dest="include_router")
    parser.add_argument("--plot", action="store_true", default=True)
    parser.add_argument("--no_plot", action="store_false", dest="plot")
    return parser.parse_args()


def find_checkpoint(run_dir: str, preferred_step: int) -> Tuple[Optional[str], Optional[int]]:
    candidates = []
    if not os.path.isdir(run_dir):
        return None, None
    for name in os.listdir(run_dir):
        if not name.endswith(".pth"):
            continue
        step_text = name[:-4]
        if not step_text.isdigit():
            continue
        step = int(step_text)
        if step <= preferred_step:
            candidates.append((step, os.path.join(run_dir, name)))
    if not candidates:
        return None, None
    step, path = max(candidates, key=lambda item: item[0])
    return path, step


def classify_weight(name: str, include_router: bool) -> Optional[Dict]:
    match = ATTN_RE.match(name)
    if match:
        layer, proj = match.groups()
        return {
            "family": "attention",
            "group": f"attention.{proj}",
            "layer": int(layer),
            "subtype": proj,
        }

    match = EXPERT_RE.match(name)
    if match:
        layer, expert_path, proj = match.groups()
        return {
            "family": "moe_expert",
            "group": f"moe_expert.{proj}",
            "layer": int(layer),
            "subtype": proj,
            "expert_path": expert_path,
        }

    if include_router:
        match = ROUTER_RE.match(name)
        if match:
            layer, head = match.groups()
            return {
                "family": "moe_router",
                "group": "moe_router",
                "layer": int(layer),
                "subtype": "router",
                "head": None if head is None else int(head[:-1]),
            }

    return None


def svd_metrics(weight: torch.Tensor) -> Dict:
    matrix = weight.detach().float().cpu()
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2-D weight, got shape={tuple(matrix.shape)}")
    s = torch.linalg.svdvals(matrix)
    s = torch.sort(s, descending=True).values
    energy = s.square()
    energy_sum = energy.sum().item()
    if energy_sum <= 0:
        probs = torch.zeros_like(energy)
    else:
        probs = energy / energy_sum
    entropy = float(-(probs * torch.log(probs + 1e-12)).sum().item())
    rank = int(s.numel())
    eff_rank = float(math.exp(entropy))
    s0 = float(s[0].item()) if rank else 0.0
    slast = float(s[-1].item()) if rank else 0.0
    eps = 1e-12

    def topk_energy(k: int) -> float:
        if rank == 0 or energy_sum <= 0:
            return 0.0
        return float(energy[: min(k, rank)].sum().item() / energy_sum)

    return {
        "shape": list(matrix.shape),
        "rank": rank,
        "singular_values": [float(x) for x in s.tolist()],
        "normalized_singular_values": [float((x / (s0 + eps)).item()) for x in s],
        "spectral_norm": s0,
        "min_singular_value": slast,
        "condition_number": float(s0 / max(slast, eps)),
        "frobenius_norm": float(torch.linalg.vector_norm(matrix).item()),
        "stable_rank": float(energy_sum / max(s0 * s0, eps)),
        "effective_rank": eff_rank,
        "effective_rank_ratio": float(eff_rank / max(rank, 1)),
        "top1_energy": topk_energy(1),
        "top5_energy": topk_energy(5),
        "top10_energy": topk_energy(10),
        "top20_energy": topk_energy(20),
        "spectral_entropy": entropy,
    }


def mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def summarize_group(items: List[Dict]) -> Dict:
    metric_names = [
        "effective_rank_ratio",
        "stable_rank",
        "top1_energy",
        "top5_energy",
        "top10_energy",
        "top20_energy",
        "condition_number",
        "spectral_norm",
    ]
    summary = {"num_matrices": len(items)}
    for metric in metric_names:
        values = [item[metric] for item in items if item.get(metric) is not None]
        summary[f"mean_{metric}"] = mean(values)
        if values:
            summary[f"min_{metric}"] = float(min(values))
            summary[f"max_{metric}"] = float(max(values))
    return summary


def interpolate_curve(curve: List[float], points: int = 128) -> List[float]:
    if not curve:
        return []
    if len(curve) == 1:
        return [curve[0]] * points
    result = []
    for i in range(points):
        pos = i * (len(curve) - 1) / (points - 1)
        left = int(math.floor(pos))
        right = min(left + 1, len(curve) - 1)
        frac = pos - left
        result.append(float(curve[left] * (1 - frac) + curve[right] * frac))
    return result


def plot_group_curves(results: Dict, output_dir: str):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"matplotlib is unavailable, writing SVG plots instead: {exc}")
        plot_group_curves_svg(results, output_dir)
        return

    groups = sorted(
        {
            matrix["group"]
            for run in results["runs"]
            for matrix in run.get("matrices", [])
        }
    )
    for group in groups:
        plt.figure(figsize=(7, 4.5))
        for run in results["runs"]:
            matrices = [m for m in run.get("matrices", []) if m["group"] == group]
            if not matrices:
                continue
            curves = [interpolate_curve(m["normalized_singular_values"]) for m in matrices]
            mean_curve = [
                sum(curve[i] for curve in curves) / len(curves)
                for i in range(len(curves[0]))
            ]
            xs = [i / (len(mean_curve) - 1) for i in range(len(mean_curve))]
            plt.plot(xs, mean_curve, label=run["run"])
        plt.yscale("log")
        plt.xlabel("normalized singular-value rank")
        plt.ylabel("singular value / largest singular value")
        plt.title(group)
        plt.grid(True, alpha=0.25)
        plt.legend(fontsize=8)
        plt.tight_layout()
        safe_group = group.replace(".", "_")
        plt.savefig(os.path.join(output_dir, f"{safe_group}_svd.png"), dpi=180)
        plt.close()


def plot_group_curves_svg(results: Dict, output_dir: str):
    groups = sorted(
        {
            matrix["group"]
            for run in results["runs"]
            for matrix in run.get("matrices", [])
        }
    )
    colors = [
        "#1f77b4",
        "#d62728",
        "#2ca02c",
        "#9467bd",
        "#ff7f0e",
        "#17becf",
    ]

    for group in groups:
        run_curves = []
        for run_idx, run in enumerate(results["runs"]):
            matrices = [m for m in run.get("matrices", []) if m["group"] == group]
            if not matrices:
                continue
            curves = [interpolate_curve(m["normalized_singular_values"]) for m in matrices]
            mean_curve = [
                sum(curve[i] for curve in curves) / len(curves)
                for i in range(len(curves[0]))
            ]
            run_curves.append((run["run"], mean_curve, colors[run_idx % len(colors)]))

        if not run_curves:
            continue

        width, height = 820, 520
        left, right, top, bottom = 80, 30, 45, 75
        plot_w = width - left - right
        plot_h = height - top - bottom
        y_min_log, y_max_log = -4.0, 0.0

        def xy(points, idx, value):
            x = left + idx * plot_w / max(len(points) - 1, 1)
            value = max(float(value), 1e-4)
            y_log = math.log10(value)
            y_log = min(max(y_log, y_min_log), y_max_log)
            y = top + (y_max_log - y_log) * plot_h / (y_max_log - y_min_log)
            return x, y

        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            f'<text x="{left}" y="28" font-family="Arial" font-size="18">{group}</text>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333"/>',
            f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333"/>',
            f'<text x="{left + plot_w / 2 - 85}" y="{height - 25}" font-family="Arial" font-size="13">normalized singular-value rank</text>',
            f'<text x="18" y="{top + plot_h / 2 + 80}" transform="rotate(-90 18,{top + plot_h / 2 + 80})" font-family="Arial" font-size="13">singular value / largest</text>',
        ]
        for power in range(0, -5, -1):
            y = top + (y_max_log - power) * plot_h / (y_max_log - y_min_log)
            label = f"1e{power}" if power < 0 else "1"
            parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#ddd"/>')
            parts.append(f'<text x="{left - 46}" y="{y + 4:.1f}" font-family="Arial" font-size="11">{label}</text>')

        legend_y = top + 8
        for curve_idx, (run_name, curve, color) in enumerate(run_curves):
            coords = []
            for i, value in enumerate(curve):
                x, y = xy(curve, i, value)
                coords.append(f"{x:.1f},{y:.1f}")
            parts.append(
                f'<polyline fill="none" stroke="{color}" stroke-width="2.2" points="{" ".join(coords)}"/>'
            )
            y = legend_y + curve_idx * 18
            parts.append(f'<line x1="{left + plot_w - 260}" y1="{y}" x2="{left + plot_w - 240}" y2="{y}" stroke="{color}" stroke-width="3"/>')
            parts.append(f'<text x="{left + plot_w - 234}" y="{y + 4}" font-family="Arial" font-size="11">{run_name}</text>')

        parts.append("</svg>")
        safe_group = group.replace(".", "_")
        with open(os.path.join(output_dir, f"{safe_group}_svd.svg"), "w", encoding="utf-8") as f:
            f.write("\n".join(parts))


def analyze_run(args, run_name: str) -> Dict:
    run_dir = os.path.join(args.checkpoint_root, run_name)
    ckpt_path, step = find_checkpoint(run_dir, args.checkpoint_step)
    if ckpt_path is None:
        return {
            "run": run_name,
            "error": f"No checkpoint <= {args.checkpoint_step} found in {run_dir}",
        }

    state = torch.load(ckpt_path, map_location="cpu")
    matrices = []
    for name, tensor in state.items():
        info = classify_weight(name, args.include_router)
        if info is None:
            continue
        metrics = svd_metrics(tensor)
        matrices.append({"name": name, **info, **metrics})

    groups = defaultdict(list)
    families = defaultdict(list)
    for matrix in matrices:
        groups[matrix["group"]].append(matrix)
        families[matrix["family"]].append(matrix)

    return {
        "run": run_name,
        "checkpoint_step": step,
        "checkpoint_path": ckpt_path,
        "num_matrices": len(matrices),
        "summary_by_group": {group: summarize_group(items) for group, items in sorted(groups.items())},
        "summary_by_family": {family: summarize_group(items) for family, items in sorted(families.items())},
        "matrices": matrices,
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    runs = [run.strip() for run in args.runs.split(",") if run.strip()]
    results = {
        "config": vars(args),
        "runs": [analyze_run(args, run) for run in runs],
    }
    output_path = os.path.join(args.output_dir, "weight_svd_summary.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    compact_path = os.path.join(args.output_dir, "weight_svd_compact.json")
    compact = {
        "config": vars(args),
        "runs": [
            {
                "run": run["run"],
                "checkpoint_step": run.get("checkpoint_step"),
                "summary_by_group": run.get("summary_by_group"),
                "summary_by_family": run.get("summary_by_family"),
            }
            for run in results["runs"]
        ],
    }
    with open(compact_path, "w", encoding="utf-8") as f:
        json.dump(compact, f, indent=2)

    if args.plot:
        plot_group_curves(results, args.output_dir)

    print(json.dumps(compact, indent=2))
    print(f"Wrote {output_path}")
    print(f"Wrote {compact_path}")


if __name__ == "__main__":
    main()
