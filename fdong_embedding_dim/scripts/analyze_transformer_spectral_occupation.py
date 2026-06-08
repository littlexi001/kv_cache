#!/usr/bin/env python3
import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict

import torch
import torch.nn.functional as F

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FDONG_SCRIPTS = os.path.join(REPO_ROOT, "fdong", "scripts")
if FDONG_SCRIPTS not in sys.path:
    sys.path.insert(0, FDONG_SCRIPTS)

import analyze_frequency_width_dynamics as awd  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Analyze head/middle/tail spectral occupation in transformer checkpoints.")
    p.add_argument("--config_dir", default="fdong/Qwen3-0.6B")
    p.add_argument("--checkpoint_root", default="fdong/checkpoints")
    p.add_argument("--run_specs", required=True, help="alias:run_name:train_distribution[,..]")
    p.add_argument("--checkpoint_steps", default="all")
    p.add_argument("--output_path", default="fdong_embedding_dim/outputs/transformer_spectral_occupation.json")
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--num_samples", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--synthetic_block_size", type=int, default=4)
    p.add_argument("--synthetic_num_hierarchy_layers", type=int, default=2)
    p.add_argument("--synthetic_content_token_count", type=int, default=256)
    p.add_argument("--synthetic_num_units_per_layer", type=int, default=64)
    p.add_argument("--synthetic_seed", type=int, default=0)
    p.add_argument("--synthetic_min_token_id", type=int, default=1)
    p.add_argument("--synthetic_zipf_alpha", type=float, default=1.3)
    p.add_argument("--synthetic_zipf_shuffle_ranks", action="store_true", default=True)
    p.add_argument("--eval_sampling_distribution", choices=["uniform", "zipf"], default="uniform")
    p.add_argument("--frequency_feature_layer", type=int, default=1)
    p.add_argument("--head_fraction", type=float, default=0.2)
    p.add_argument("--tail_fraction", type=float, default=0.4)
    p.add_argument("--top_dims", default="1,2,5,10,20")
    p.add_argument("--head_subspace_dim", type=int, default=3)
    return p.parse_args()


def parse_ints(text):
    return [int(x) for x in text.split(",") if x.strip()]


def pearson(xs, ys):
    if len(xs) < 2:
        return None
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def effective_rank(vals: torch.Tensor):
    vals = vals.float().clamp_min(0)
    total = vals.sum()
    if float(total.item()) <= 0:
        return 0.0
    p = vals / total
    return float(torch.exp(-(p * p.clamp_min(1e-12).log()).sum()).item())


def pairwise_cos(vectors):
    if len(vectors) < 2:
        return {"mean": None, "min": None, "max": None}
    x = F.normalize(torch.stack(vectors).float(), dim=-1)
    mat = x @ x.T
    vals = mat[torch.triu(torch.ones_like(mat, dtype=torch.bool), diagonal=1)]
    return {"mean": float(vals.mean().item()), "min": float(vals.min().item()), "max": float(vals.max().item())}


def centroids_by_feature(x, labels):
    centroids, counts = {}, {}
    for feature in torch.unique(labels).tolist():
        if feature < 0:
            continue
        mask = labels == feature
        centroids[int(feature)] = x[mask].float().mean(dim=0)
        counts[int(feature)] = int(mask.sum().item())
    return centroids, counts


def bucket_name(feature, bucket_ids):
    return ["head", "middle", "tail"][int(bucket_ids[int(feature)].item())]


def summarize_feature_space(x, labels, bucket_ids, scores, top_dims, head_subspace_dim):
    x = x.float()
    x_centered = x - x.mean(dim=0, keepdim=True)
    _, s, vh = torch.linalg.svd(x_centered, full_matrices=False)
    pcs = vh
    eigen = s.pow(2) / max(x_centered.shape[0] - 1, 1)
    centroids, counts = centroids_by_feature(x_centered, labels)

    head_vectors = [v for f, v in centroids.items() if bucket_name(f, bucket_ids) == "head"]
    if len(head_vectors) >= 1:
        head_mat = torch.stack(head_vectors)
        head_centered = head_mat - head_mat.mean(dim=0, keepdim=True)
        _, _, head_vh = torch.linalg.svd(head_centered, full_matrices=False)
        head_basis = head_vh[: min(head_subspace_dim, head_vh.shape[0])]
    else:
        head_basis = torch.zeros(0, x.shape[-1])

    bucket_stats = {}
    freq_rows = []
    for bucket in ["head", "middle", "tail"]:
        feats = [f for f in centroids if bucket_name(f, bucket_ids) == bucket]
        vecs = [centroids[f] for f in feats]
        if not vecs:
            bucket_stats[bucket] = {"num_features": 0}
            continue
        mat = torch.stack(vecs)
        norms = mat.norm(dim=-1).clamp_min(1e-12)
        stats = {
            "num_features": len(vecs),
            "centroid_norm_mean": float(norms.mean().item()),
            "pairwise_centroid_cosine": pairwise_cos(vecs),
        }
        for k in top_dims:
            kk = min(k, pcs.shape[0])
            proj = mat @ pcs[:kk].T
            stats[f"global_top{k}_projection_energy_fraction"] = float((proj.pow(2).sum(dim=-1) / norms.pow(2)).mean().item())
        if head_basis.numel() > 0:
            proj_head = mat @ head_basis.T
            head_frac = proj_head.pow(2).sum(dim=-1) / norms.pow(2)
            stats["head_subspace_energy_fraction"] = float(head_frac.mean().item())
            stats["residual_after_head_energy_fraction"] = float((1.0 - head_frac).mean().item())
            residual = mat - proj_head @ head_basis
            if residual.shape[0] >= 2:
                _, rs, _ = torch.linalg.svd(residual - residual.mean(dim=0, keepdim=True), full_matrices=False)
                stats["residual_effective_rank"] = effective_rank(rs.pow(2))
        bucket_stats[bucket] = stats

    for f, vec in centroids.items():
        norm = float(vec.norm().item())
        score = float(scores[int(f)].item())
        freq_rows.append(
            {
                "feature": int(f),
                "bucket": bucket_name(f, bucket_ids),
                "score": score,
                "log_score": math.log(score + 1e-12),
                "centroid_norm": norm,
                "pc1": float(torch.dot(vec, pcs[0]).item()) if pcs.shape[0] else 0.0,
            }
        )

    return {
        "hidden_dim": int(x.shape[-1]),
        "effective_rank": effective_rank(eigen),
        "explained_top": {
            f"top{k}": float(eigen[: min(k, eigen.numel())].sum().item() / max(eigen.sum().item(), 1e-12))
            for k in top_dims
        },
        "bucket_stats": bucket_stats,
        "frequency_correlations": {
            "log_score_vs_centroid_norm": pearson([r["log_score"] for r in freq_rows], [r["centroid_norm"] for r in freq_rows]),
            "log_score_vs_pc1": pearson([r["log_score"] for r in freq_rows], [r["pc1"] for r in freq_rows]),
        },
    }


def collect_token_matrix_labels(args, dataset, bucket_ids):
    token_bucket_counts = defaultdict(Counter)
    for idx in range(len(dataset)):
        _, target, _, _ = dataset[idx]
        meta = dataset.get_metadata(idx)["unit_ids_by_layer"][1:]
        feature = meta[:, args.frequency_feature_layer]
        valid = feature.ge(0)
        for tok, feat in zip(target[valid].tolist(), feature[valid].tolist()):
            token_bucket_counts[int(tok)][int(bucket_ids[int(feat)].item())] += 1
    labels = {}
    for tok, counts in token_bucket_counts.items():
        if counts:
            labels[int(tok)] = counts.most_common(1)[0][0]
    return labels


def summarize_token_rows(weight, token_to_bucket):
    rows = {}
    for bucket_name, bucket_idx in [("head", 0), ("middle", 1), ("tail", 2)]:
        ids = [tok for tok, b in token_to_bucket.items() if b == bucket_idx and tok < weight.shape[0]]
        if not ids:
            rows[bucket_name] = {"count": 0}
            continue
        mat = weight[ids].float()
        centered = mat - mat.mean(dim=0, keepdim=True)
        _, s, _ = torch.linalg.svd(centered, full_matrices=False)
        rows[bucket_name] = {
            "count": len(ids),
            "norm_mean": float(mat.norm(dim=-1).mean().item()),
            "pairwise_cosine": pairwise_cos([row for row in mat]),
            "effective_rank": effective_rank(s.pow(2)),
        }
    return rows


def main():
    args = parse_args()
    args.config_dir = os.path.join(REPO_ROOT, args.config_dir) if not os.path.isabs(args.config_dir) else args.config_dir
    args.checkpoint_root = os.path.join(REPO_ROOT, args.checkpoint_root) if not os.path.isabs(args.checkpoint_root) else args.checkpoint_root
    device = awd.choose_device()
    top_dims = parse_ints(args.top_dims)
    results = {"config": vars(args), "device": str(device), "runs": []}
    for spec in awd.parse_run_specs(args.run_specs):
        run_dir = os.path.join(args.checkpoint_root, spec["run_name"])
        bucket_ids, bucket_meta = awd.make_bucket_ids(args, spec["train_distribution"])
        scores = awd.frequency_scores(args, spec["train_distribution"]).float()
        eval_dataset = awd.make_dataset(args, args.eval_sampling_distribution, num_samples=args.num_samples)
        token_to_bucket = collect_token_matrix_labels(args, eval_dataset, bucket_ids)
        run_row = {**spec, "bucket_meta": bucket_meta, "checkpoints": []}
        for step, checkpoint_path in awd.list_checkpoints(run_dir, args.checkpoint_steps):
            print(f"analyze {spec['alias']} step={step}", flush=True)
            model, config = awd.load_model(args, run_dir, checkpoint_path, device)
            reps, labels = awd.collect_hidden(args, model, eval_dataset, device, args.num_samples)
            ckpt_row = {
                "step": step,
                "checkpoint": checkpoint_path,
                "model": {
                    "hidden_size": int(config.hidden_size),
                    "num_hidden_layers": int(config.num_hidden_layers),
                    "num_attention_heads": int(config.num_attention_heads),
                },
                "embedding_rows": summarize_token_rows(model.model.embed_tokens.weight.detach().cpu(), token_to_bucket),
                "lm_head_rows": summarize_token_rows(model.lm_head.weight.detach().cpu(), token_to_bucket),
                "hidden_layers": [],
            }
            for layer_idx, x in enumerate(reps):
                ckpt_row["hidden_layers"].append(
                    {
                        "layer": layer_idx,
                        **summarize_feature_space(
                            x,
                            labels["higher_unit"],
                            bucket_ids,
                            scores,
                            top_dims,
                            args.head_subspace_dim,
                        ),
                    }
                )
            run_row["checkpoints"].append(ckpt_row)
            del model
        results["runs"].append(run_row)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {args.output_path}")


if __name__ == "__main__":
    main()
