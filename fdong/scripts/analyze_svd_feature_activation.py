import argparse
import gc
import json
import math
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze SVD-defined input-side feature activation for Q/K/V/O "
            "projection matrices. PyTorch Linear uses y=xW^T, so the input-side "
            "basis is the right singular-vector basis of Linear.weight."
        )
    )
    parser.add_argument("--model_dir", type=str, default="../Qwen3-0.6B")
    parser.add_argument("--text_dir", type=str, default="../../fdong_seq_compress/data/synthetic_texts")
    parser.add_argument("--text_file", action="append", default=None)
    parser.add_argument("--output_dir", type=str, default="../experiments/svd_feature_activation_qwen3_0p6b")
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--max_sequences", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--tau", type=float, default=0.9)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--layers", type=str, default="all", help="'all' or comma-separated layer ids.")
    parser.add_argument("--projections", type=str, default="q,k,v,o", help="subset of q,k,v,o")
    parser.add_argument("--no_plot", action="store_true")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    # Avoid selecting MPS by default: several PyTorch ops used by hooks and
    # accumulation either lack float64 support or are OS-version sensitive.
    # Users can still pass `--device mps` explicitly if they want to try it.
    return torch.device("cpu")


def choose_dtype(name: str):
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise ValueError(name)


def parse_layers(text: str, num_layers: int) -> List[int]:
    if text == "all":
        return list(range(num_layers))
    result = []
    for item in text.split(","):
        item = item.strip()
        if item:
            result.append(int(item))
    for layer in result:
        if layer < 0 or layer >= num_layers:
            raise ValueError(f"layer {layer} out of range [0, {num_layers})")
    return result


def parse_projections(text: str) -> List[str]:
    mapping = {"q": "q_proj", "k": "k_proj", "v": "v_proj", "o": "o_proj"}
    result = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if item not in mapping:
            raise ValueError(f"Unsupported projection {item}; expected subset of q,k,v,o")
        result.append(mapping[item])
    return result


def iter_texts(args) -> Iterable[str]:
    paths = []
    if args.text_file:
        paths.extend(args.text_file)
    if args.text_dir and os.path.isdir(args.text_dir):
        for root, _, files in os.walk(args.text_dir):
            for name in sorted(files):
                if name.endswith((".txt", ".jsonl")):
                    paths.append(os.path.join(root, name))
    if not paths:
        raise FileNotFoundError("No text files found. Pass --text_file or --text_dir.")

    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if not text:
            continue
        if path.endswith(".jsonl"):
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    value = json.loads(line)
                    if isinstance(value, str):
                        yield value
                    elif isinstance(value, dict):
                        for key in ("text", "content", "prompt"):
                            if isinstance(value.get(key), str):
                                yield value[key]
                                break
                except json.JSONDecodeError:
                    yield line
        else:
            yield text


def make_batches(args, tokenizer) -> List[torch.Tensor]:
    sequences = []
    for text in iter_texts(args):
        token_ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[0]
        if token_ids.numel() == 0:
            continue
        if token_ids.numel() < args.seq_len:
            pad_id = tokenizer.pad_token_id
            if pad_id is None:
                pad_id = tokenizer.eos_token_id
            padded = torch.full((args.seq_len,), int(pad_id), dtype=torch.long)
            padded[: token_ids.numel()] = token_ids
            sequences.append(padded)
        else:
            for start in range(0, token_ids.numel() - args.seq_len + 1, args.stride):
                sequences.append(token_ids[start : start + args.seq_len].clone())
                if len(sequences) >= args.max_sequences:
                    break
        if len(sequences) >= args.max_sequences:
            break
    if not sequences:
        raise RuntimeError("No token sequences built from input text.")
    batches = []
    for start in range(0, len(sequences), args.batch_size):
        batches.append(torch.stack(sequences[start : start + args.batch_size], dim=0))
    return batches


def get_layers(model):
    base = getattr(model, "model", model)
    layers = getattr(base, "layers", None)
    if layers is None:
        raise ValueError("Could not find model.layers on loaded model.")
    return layers


def get_projection_module(layer, projection: str):
    attn = getattr(layer, "self_attn")
    return getattr(attn, projection)


def compute_input_svd(linear: torch.nn.Linear) -> Tuple[torch.Tensor, torch.Tensor]:
    # Linear computes y = x @ weight.T. For weight = U S Vh, input-side
    # feature basis is rows of Vh, and x @ Vh.T gives feature projections.
    weight = linear.weight.detach().float().cpu()
    _, singular_values, vh = torch.linalg.svd(weight, full_matrices=False)
    return vh.contiguous(), singular_values.contiguous()


class FeatureAccumulator:
    def __init__(self, layer: int, projection: str, vh: torch.Tensor, singular_values: torch.Tensor, tau: float):
        self.layer = layer
        self.projection = projection
        self.vh_cpu = vh
        self.s_cpu = singular_values
        self.tau = tau
        self.rank = int(singular_values.numel())
        self.input_dim = int(vh.shape[1])
        self.count = torch.zeros(self.rank, dtype=torch.float64)
        self.abs_sum = torch.zeros(self.rank, dtype=torch.float64)
        self.energy_sum_on_active = torch.zeros(self.rank, dtype=torch.float64)
        self.energy_sum_all = torch.zeros(self.rank, dtype=torch.float64)
        self.num_items = 0
        self.total_active_features = 0
        self._device_cache = {}

    def _basis_on(self, device):
        key = str(device)
        cached = self._device_cache.get(key)
        if cached is None:
            vh = self.vh_cpu.to(device=device, dtype=torch.float32)
            s = self.s_cpu.to(device=device, dtype=torch.float32)
            cached = (vh, s)
            self._device_cache[key] = cached
        return cached

    @torch.no_grad()
    def update(self, inputs: torch.Tensor):
        x = inputs.detach().float().reshape(-1, inputs.shape[-1])
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Input dim mismatch for layer={self.layer} projection={self.projection}: "
                f"got {x.shape[-1]}, expected {self.input_dim}"
            )
        vh, s = self._basis_on(x.device)
        activation = (x @ vh.T) * s
        abs_activation = activation.abs()
        energy = activation.square()
        total = energy.sum(dim=-1)
        valid = total > 0
        if not bool(valid.any()):
            return
        energy = energy[valid]
        abs_activation = abs_activation[valid]
        total = total[valid]
        n = int(energy.shape[0])

        sorted_energy, sorted_idx = torch.sort(energy, dim=-1, descending=True)
        cumsum = torch.cumsum(sorted_energy, dim=-1)
        threshold = self.tau * total.unsqueeze(-1)
        active_sorted = cumsum <= threshold
        first_ge = torch.argmax((cumsum >= threshold).to(torch.int64), dim=-1)
        active_sorted[torch.arange(n, device=energy.device), first_ge] = True
        active = torch.zeros_like(active_sorted, dtype=torch.bool)
        active.scatter_(1, sorted_idx, active_sorted)

        active_f = active.to(torch.float32)
        active_count_per_item = active_f.sum(dim=-1)
        self.num_items += n
        self.total_active_features += int(active_count_per_item.sum().item())

        self.count += active_f.sum(dim=0).cpu().double()
        self.abs_sum += (abs_activation * active_f).sum(dim=0).cpu().double()
        self.energy_sum_on_active += (energy * active_f).sum(dim=0).cpu().double()
        self.energy_sum_all += energy.sum(dim=0).cpu().double()

    def result(self) -> Dict:
        eps = 1e-12
        count = self.count
        frequency = count / max(self.num_items, 1)
        avg_abs_on_active = self.abs_sum / torch.clamp(count, min=eps)
        avg_energy_on_active = self.energy_sum_on_active / torch.clamp(count, min=eps)
        weighted_energy_share = self.energy_sum_all / max(float(self.energy_sum_all.sum().item()), eps)
        sorted_freq, sorted_idx = torch.sort(frequency, descending=True)
        sorted_avg_abs = avg_abs_on_active[sorted_idx]
        sorted_energy_share = weighted_energy_share[sorted_idx]
        return {
            "layer": self.layer,
            "projection": self.projection.replace("_proj", ""),
            "rank": self.rank,
            "input_dim": self.input_dim,
            "num_items": self.num_items,
            "avg_active_features_per_item": float(self.total_active_features / max(self.num_items, 1)),
            "mean_activation_frequency": float(frequency.mean().item()),
            "max_activation_frequency": float(frequency.max().item()),
            "min_activation_frequency": float(frequency.min().item()),
            "top1_frequency": float(sorted_freq[0].item()),
            "top5_frequency_sum": float(sorted_freq[: min(5, self.rank)].sum().item()),
            "top10_frequency_sum": float(sorted_freq[: min(10, self.rank)].sum().item()),
            "top20_frequency_sum": float(sorted_freq[: min(20, self.rank)].sum().item()),
            "feature_activation_frequency": [float(x) for x in frequency.tolist()],
            "avg_abs_activation_on_active": [float(x) for x in avg_abs_on_active.tolist()],
            "avg_energy_on_active": [float(x) for x in avg_energy_on_active.tolist()],
            "weighted_energy_share": [float(x) for x in weighted_energy_share.tolist()],
            "feature_rank_by_frequency": [int(x) for x in sorted_idx.tolist()],
            "frequency_sorted": [float(x) for x in sorted_freq.tolist()],
            "avg_abs_activation_sorted_by_frequency": [float(x) for x in sorted_avg_abs.tolist()],
            "weighted_energy_share_sorted_by_frequency": [float(x) for x in sorted_energy_share.tolist()],
        }


def register_hooks(model, accumulators: Dict[Tuple[int, str], FeatureAccumulator]):
    handles = []
    layers = get_layers(model)
    for (layer_idx, projection), acc in accumulators.items():
        module = get_projection_module(layers[layer_idx], projection)

        def make_hook(accumulator):
            def hook(_module, module_inputs, _module_output):
                accumulator.update(module_inputs[0])
            return hook

        handles.append(module.register_forward_hook(make_hook(acc)))
    return handles


def plot_results(results: Dict, output_dir: str):
    mpl_config_dir = os.path.join(output_dir, ".mplconfig")
    os.makedirs(mpl_config_dir, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", mpl_config_dir)
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        plot_results_svg(results, output_dir)
        return

    matrices = results["matrices"]
    projections = [p for p in PROJECTIONS if any(m["projection"] == p.replace("_proj", "") for m in matrices)]
    short_to_items = defaultdict(list)
    for item in matrices:
        short_to_items[item["projection"]].append(item)
    for items in short_to_items.values():
        items.sort(key=lambda x: x["layer"])

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.reshape(-1)
    for ax, short in zip(axes, [p.replace("_proj", "") for p in projections]):
        for item in short_to_items.get(short, []):
            y = item["frequency_sorted"]
            x = list(range(1, len(y) + 1))
            ax.plot(x, y, alpha=0.55, linewidth=1.0, label=f"L{item['layer']}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{short}: feature activation frequency")
        ax.set_xlabel("feature rank by activation frequency")
        ax.set_ylabel("activation frequency")
        ax.grid(True, alpha=0.25)
    axes[0].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "feature_activation_frequency_by_layer.png"), dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.reshape(-1)
    for ax, short in zip(axes, [p.replace("_proj", "") for p in projections]):
        for item in short_to_items.get(short, []):
            y = item["avg_abs_activation_sorted_by_frequency"]
            x = list(range(1, len(y) + 1))
            ax.plot(x, y, alpha=0.55, linewidth=1.0, label=f"L{item['layer']}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{short}: avg |scaled activation| on active tokens")
        ax.set_xlabel("feature rank by activation frequency")
        ax.set_ylabel("avg |a_i| when active")
        ax.grid(True, alpha=0.25)
    axes[0].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "avg_activation_strength_by_layer.png"), dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for short in [p.replace("_proj", "") for p in projections]:
        items = short_to_items.get(short, [])
        if not items:
            continue
        xs = [item["layer"] for item in items]
        ys = [item["avg_active_features_per_item"] for item in items]
        ax.plot(xs, ys, marker="o", label=short)
    ax.set_title(f"Average number of active features per token (tau={results['config']['tau']})")
    ax.set_xlabel("layer")
    ax.set_ylabel("avg active features")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "avg_active_feature_count_by_layer.png"), dpi=180)
    plt.close(fig)


def _svg_polyline(points, color="#1f77b4", width=1.3, opacity=0.75):
    coords = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return f'<polyline fill="none" stroke="{color}" stroke-width="{width}" opacity="{opacity}" points="{coords}"/>'


def _plot_curves_svg(curves, title, xlabel, ylabel, path, logx=True, logy=True):
    width, height = 980, 660
    left, right, top, bottom = 82, 24, 54, 72
    plot_w = width - left - right
    plot_h = height - top - bottom
    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
        "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]

    xs_all, ys_all = [], []
    prepared = []
    for label, xs, ys in curves:
        pts = []
        for x, y in zip(xs, ys):
            if y <= 0:
                continue
            xx = math.log10(max(x, 1e-12)) if logx else x
            yy = math.log10(max(y, 1e-12)) if logy else y
            pts.append((xx, yy))
            xs_all.append(xx)
            ys_all.append(yy)
        if pts:
            prepared.append((label, pts))
    if not prepared:
        return
    min_x, max_x = min(xs_all), max(xs_all)
    min_y, max_y = min(ys_all), max(ys_all)
    if min_x == max_x:
        max_x = min_x + 1
    if min_y == max_y:
        max_y = min_y + 1
    pad_y = 0.05 * (max_y - min_y)
    min_y -= pad_y
    max_y += pad_y

    def sx(x):
        return left + (x - min_x) / (max_x - min_x) * plot_w

    def sy(y):
        return top + (max_y - y) / (max_y - min_y) * plot_h

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2:.0f}" y="28" text-anchor="middle" font-family="Arial" font-size="18">{title}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#333"/>',
        f'<text x="{width/2:.0f}" y="{height-22}" text-anchor="middle" font-family="Arial" font-size="13">{xlabel}</text>',
        f'<text transform="translate(20,{height/2:.0f}) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="13">{ylabel}</text>',
    ]
    for frac in [0, 0.25, 0.5, 0.75, 1.0]:
        x = left + frac * plot_w
        y = top + frac * plot_h
        lines.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+plot_h}" stroke="#ddd" stroke-width="0.8"/>')
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#ddd" stroke-width="0.8"/>')

    for idx, (label, pts) in enumerate(prepared):
        color = colors[idx % len(colors)]
        scaled = [(sx(x), sy(y)) for x, y in pts]
        lines.append(_svg_polyline(scaled, color=color))
        if idx < 14:
            lx = left + plot_w - 120
            ly = top + 18 + idx * 18
            lines.append(f'<line x1="{lx}" y1="{ly-4}" x2="{lx+22}" y2="{ly-4}" stroke="{color}" stroke-width="2"/>')
            lines.append(f'<text x="{lx+28}" y="{ly}" font-family="Arial" font-size="11">{label}</text>')
    lines.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def plot_results_svg(results: Dict, output_dir: str):
    matrices = results["matrices"]
    short_to_items = defaultdict(list)
    for item in matrices:
        short_to_items[item["projection"]].append(item)
    for items in short_to_items.values():
        items.sort(key=lambda x: x["layer"])

    for short, items in sorted(short_to_items.items()):
        freq_curves = []
        act_curves = []
        for item in items:
            x = list(range(1, len(item["frequency_sorted"]) + 1))
            freq_curves.append((f"L{item['layer']}", x, item["frequency_sorted"]))
            act_curves.append((f"L{item['layer']}", x, item["avg_abs_activation_sorted_by_frequency"]))
        _plot_curves_svg(
            freq_curves,
            f"{short}: feature activation frequency",
            "feature rank by activation frequency",
            "activation frequency",
            os.path.join(output_dir, f"{short}_feature_activation_frequency.svg"),
        )
        _plot_curves_svg(
            act_curves,
            f"{short}: avg |scaled activation| on active tokens",
            "feature rank by activation frequency",
            "avg |a_i| when active",
            os.path.join(output_dir, f"{short}_avg_activation_strength.svg"),
        )

    count_curves = []
    for short, items in sorted(short_to_items.items()):
        xs = [item["layer"] + 1 for item in items]
        ys = [item["avg_active_features_per_item"] for item in items]
        count_curves.append((short, xs, ys))
    _plot_curves_svg(
        count_curves,
        f"Average number of active features per token (tau={results['config']['tau']})",
        "layer + 1",
        "avg active features",
        os.path.join(output_dir, "avg_active_feature_count_by_layer.svg"),
        logx=False,
        logy=False,
    )


def main():
    args = parse_args()
    if not (0.0 < args.tau <= 1.0):
        raise ValueError("--tau must be in (0, 1].")
    os.makedirs(args.output_dir, exist_ok=True)

    device = choose_device(args.device)
    dtype = choose_dtype(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    batches = make_batches(args, tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        torch_dtype=dtype,
        local_files_only=True,
    )
    model.eval().to(device)
    model.config.use_cache = False
    if hasattr(model.config, "_attn_implementation"):
        model.config._attn_implementation = "eager"

    layers = get_layers(model)
    selected_layers = parse_layers(args.layers, len(layers))
    selected_projections = parse_projections(args.projections)

    accumulators = {}
    for layer_idx in selected_layers:
        for projection in selected_projections:
            module = get_projection_module(layers[layer_idx], projection)
            vh, singular_values = compute_input_svd(module)
            accumulators[(layer_idx, projection)] = FeatureAccumulator(
                layer=layer_idx,
                projection=projection,
                vh=vh,
                singular_values=singular_values,
                tau=args.tau,
            )

    handles = register_hooks(model, accumulators)
    try:
        with torch.no_grad():
            for batch in batches:
                input_ids = batch.to(device)
                model(input_ids=input_ids, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    matrices = [acc.result() for _, acc in sorted(accumulators.items())]
    for acc in accumulators.values():
        acc._device_cache.clear()
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    summary = {
        "config": {
            "model_dir": args.model_dir,
            "text_dir": args.text_dir,
            "text_file": args.text_file,
            "seq_len": args.seq_len,
            "stride": args.stride,
            "max_sequences": args.max_sequences,
            "batch_size": args.batch_size,
            "tau": args.tau,
            "device": str(device),
            "dtype": args.dtype,
            "layers": selected_layers,
            "projections": [p.replace("_proj", "") for p in selected_projections],
        },
        "num_sequences": sum(batch.shape[0] for batch in batches),
        "matrices": matrices,
    }

    compact = defaultdict(dict)
    for item in matrices:
        compact[item["projection"]][str(item["layer"])] = {
            "avg_active_features_per_item": item["avg_active_features_per_item"],
            "top1_frequency": item["top1_frequency"],
            "top5_frequency_sum": item["top5_frequency_sum"],
            "top10_frequency_sum": item["top10_frequency_sum"],
            "top20_frequency_sum": item["top20_frequency_sum"],
            "max_activation_frequency": item["max_activation_frequency"],
            "mean_activation_frequency": item["mean_activation_frequency"],
        }
    summary["compact_by_projection"] = compact

    output_path = os.path.join(args.output_dir, "svd_feature_activation.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if not args.no_plot:
        plot_results(summary, args.output_dir)

    print(json.dumps({
        "output_path": output_path,
        "output_dir": args.output_dir,
        "num_matrices": len(matrices),
        "num_sequences": summary["num_sequences"],
    }, indent=2))


if __name__ == "__main__":
    main()
