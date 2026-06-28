#!/usr/bin/env python3
"""Compare CRS vs Baseline spectral evolution at key checkpoints."""
import json, math, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

OUTDIR = '/Users/bytedance/kv_cache/fdong_embedding_dim/outputs/synthetic_crs'
STEPS = [50, 100, 200, 500, 1000, 2000, 3000]


def load_model(ckpt_path, use_crs=False):
    from synthetic_crs_experiment import SmallLM
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model = SmallLM(vocab_size=500, d_model=128, n_layers=1, n_heads=4, n_kv_heads=2,
                    intermediate_size=384, max_seq_len=12,
                    use_crs=use_crs, alpha=0.3, p=8)
    model.load_state_dict(ckpt['model_state_dict'])
    return model


def svd_stats(W):
    U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
    S = S.numpy()
    total = np.sum(S ** 2)
    effrank = np.sum(S ** 2) ** 2 / max(np.sum(S ** 4), 1e-30)
    return float(S[0]), float(effrank), [float(S[i] ** 2 / total) for i in range(min(5, len(S)))]


def main():
    base_dir = os.path.join(OUTDIR, 'baseline')
    crs_dir = os.path.join(OUTDIR, 'crs_a0.3_p8')

    print(f"{'Step':>6s}  {'Base σ1':>8s}  {'Base effR':>8s}  "
          f"{'CRS σ1_r':>9s}  {'CRS effR_r':>10s}  "
          f"{'CRS σ1_qb':>10s}  {'CRS effR_qb':>10s}  {'σ1_r/σ1':>8s}")
    print("-" * 85)

    for step in STEPS:
        base_ckpt = os.path.join(base_dir, f'ckpt_step{step}.pt')
        crs_ckpt = os.path.join(crs_dir, f'ckpt_step{step}.pt')

        base_line = ""
        crs_line = ""

        if os.path.exists(base_ckpt):
            m = load_model(base_ckpt, use_crs=False)
            s1, er, _ = svd_stats(m.layers[0].attn.Wq.weight.data)
            base_line = f"{s1:8.4f}  {er:8.1f}"
        else:
            base_line = f"{'N/A':>8s}  {'N/A':>8s}"

        if os.path.exists(crs_ckpt):
            m = load_model(crs_ckpt, use_crs=True)
            s1_r, er_r, _ = svd_stats(m.layers[0].attn.Wq_r.weight.data)
            # Effective matrix of q_branch
            W_up = m.layers[0].attn.q_branch.W_up.weight.data.float()
            W_dn = m.layers[0].attn.q_branch.W_down.weight.data.float()
            s1_qb, er_qb, _ = svd_stats(W_up @ W_dn)

            # Compare residual vs baseline
            if os.path.exists(base_ckpt):
                m_base = load_model(base_ckpt, use_crs=False)
                s1_base, _, _ = svd_stats(m_base.layers[0].attn.Wq.weight.data)
                ratio = f"{s1_r/s1_base:7.2%}"
            else:
                ratio = "N/A"

            crs_line = f"{s1_r:9.4f}  {er_r:10.1f}  {s1_qb:10.4f}  {er_qb:10.1f}  {ratio:>8s}"
        else:
            crs_line = f"{'N/A':>9s}  {'N/A':>10s}  {'N/A':>10s}  {'N/A':>10s}  {'N/A':>8s}"

        print(f"  {step:>3d}  {base_line}  {crs_line}")

    # --- Also compare K_loss vs R_loss at early steps ---
    print(f"\n{'='*75}")
    print(f"  K_loss vs R_loss early dynamics")
    print(f"{'='*75}")
    print(f"{'Step':>6s}  {'Base K':>9s}  {'Base R':>9s}  {'Base R/K':>9s}  "
          f"{'CRS K':>8s}  {'CRS R':>8s}  {'CRS R/K':>8s}")
    print("-" * 70)

    base_metrics = os.path.join(base_dir, 'metrics.jsonl')
    crs_metrics = os.path.join(crs_dir, 'metrics.jsonl')

    base_by_step = {}
    if os.path.exists(base_metrics):
        with open(base_metrics) as f:
            for line in f:
                r = json.loads(line.strip())
                base_by_step[r['step']] = r

    crs_by_step = {}
    if os.path.exists(crs_metrics):
        with open(crs_metrics) as f:
            for line in f:
                r = json.loads(line.strip())
                crs_by_step[r['step']] = r

    for step in [50, 100, 150, 200, 250, 300, 400, 500]:
        b = base_by_step.get(step, {})
        c = crs_by_step.get(step, {})
        b_k = b.get('K_loss', float('nan'))
        b_r = b.get('R_loss', float('nan'))
        c_k = c.get('K_loss', float('nan'))
        c_r = c.get('R_loss', float('nan'))
        b_rk = f"{b_r/b_k:.1f}" if b_k > 0 else "inf"
        c_rk = f"{c_r/c_k:.1f}" if c_k > 0 else "inf"
        print(f"  {step:>3d}  {b_k:9.4f}  {b_r:9.4f}  {b_rk:>9s}  "
              f"{c_k:8.4f}  {c_r:8.4f}  {c_rk:>8s}")


if __name__ == '__main__':
    main()
