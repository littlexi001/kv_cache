#!/usr/bin/env python3
"""Evaluate K/R accuracy for all d=32 experiments from saved checkpoints."""
import json, os, sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))

OUTDIR = '/Users/bytedance/kv_cache/fdong_embedding_dim/outputs/synthetic_crs'
STEPS = [50, 100, 200, 500, 1000, 2000]
K_IDS = set(range(10))


def load_model(ckpt_path, use_crs=False, alpha=0.3, p=4):
    from synthetic_crs_experiment import SmallLM
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model = SmallLM(vocab_size=500, d_model=32, n_layers=1, n_heads=2, n_kv_heads=1,
                    intermediate_size=96, max_seq_len=12,
                    use_crs=use_crs, alpha=alpha, p=p)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


def eval_accuracy(model, patterns, K_ids):
    """Compute per-position accuracy on all patterns."""
    K_correct = 0
    K_total = 0
    R_correct = 0
    R_total = 0

    for seq in patterns:
        ids = torch.tensor(seq, dtype=torch.long).unsqueeze(0)  # (1, T)
        with torch.no_grad():
            logits = model(ids)  # (1, T, V)
            preds = logits.argmax(dim=-1).squeeze(0)  # (T,)

        for t in range(len(seq)):
            if seq[t] in K_ids:
                K_correct += (preds[t].item() == seq[t])
                K_total += 1
            else:
                R_correct += (preds[t].item() == seq[t])
                R_total += 1

    return K_correct / max(K_total, 1), R_correct / max(R_total, 1)


def main():
    from synthetic_crs_experiment import generate_synthetic_patterns
    data = generate_synthetic_patterns(n_patterns=200, seed=42)
    patterns = data['patterns']

    configs = [
        ('baseline', 'baseline', False, 0.0, 4),
        ('CRS α=0.3', 'crs_a0.3_p4', True, 0.3, 4),
        ('CRS α=0.5', 'crs_a0.5_p4', True, 0.5, 4),
        ('CRS α=0.8', 'crs_a0.8_p4', True, 0.8, 4),
        ('CRS α=1.0', 'crs_a1.0_p4', True, 1.0, 4),
    ]

    # ---- Print header ----
    header = f"{'Step':>5s}"
    for _, short, _, _, _ in configs:
        header += f"  {short:>10s}K    {short:>10s}R"
    print(header)
    print("-" * len(header))

    for step in STEPS:
        row = f"{step:>5d}"
        for _, subdir, use_crs, alpha, p in configs:
            ckpt_path = os.path.join(OUTDIR, subdir, f'ckpt_step{step}.pt')
            if os.path.exists(ckpt_path):
                model = load_model(ckpt_path, use_crs=use_crs, alpha=alpha, p=p)
                k_acc, r_acc = eval_accuracy(model, patterns, K_IDS)
                row += f"  {k_acc*100:>9.1f}%  {r_acc*100:>9.1f}%"
            else:
                row += f"  {'N/A':>9s}  {'N/A':>9s}"
        print(row)

    # ---- Also check if CRS catches up at convergence ----
    print(f"\n{'='*75}")
    print(f"  Best R accuracy vs Baseline at same step")
    print(f"{'='*75}")
    for step in STEPS:
        base_k, base_r = 0, 0
        best_label = ""
        best_r = 0
        for _, subdir, use_crs, alpha, p in configs:
            ckpt_path = os.path.join(OUTDIR, subdir, f'ckpt_step{step}.pt')
            if os.path.exists(ckpt_path):
                model = load_model(ckpt_path, use_crs=use_crs, alpha=alpha, p=p)
                k_acc, r_acc = eval_accuracy(model, patterns, K_IDS)
                if subdir == 'baseline':
                    base_k, base_r = k_acc, r_acc
                elif r_acc > best_r:
                    best_r = r_acc
                    best_label = f"α={alpha}"
        print(f"  step {step:>4d}: baseline R={base_r*100:.1f}%  best CRS R={best_r*100:.1f}%  "
              f"({best_label} {'+' if best_r > base_r else ''}{best_r-base_r:+.4f})")


if __name__ == '__main__':
    main()
