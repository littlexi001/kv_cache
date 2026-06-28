#!/usr/bin/env python3
"""
Comprehensive spectral analysis: CRS vs Baseline at step 3000.

Questions:
1. Loss sanity check (both metrics and checkpoint-evaluated)
2. Singular value spectra of attention matrices (Wq/k/v/o) and embedding E
3. CRS: does the common branch (g_q/g_k/g_v) learn different spectral structure
   from the residual branch (Wq_r/Wk_r/Wv_r)?
4. Representation space: hidden state PCA, E-Q alignment
5. Should we adjust alpha? Use smaller model?
"""

import json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Import our model classes
sys.path.insert(0, os.path.dirname(__file__))
from small_lm_crs import SmallLM, EnglishTextDataset, RMSNorm

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

OUTDIR = '/Users/bytedance/kv_cache/fdong_embedding_dim/outputs/small_lm_crs'
CRS_CKPT = os.path.join(OUTDIR, 'crs_alpha0_3/checkpoints/step_3000.pt')
BASE_CKPT = os.path.join(OUTDIR, 'baseline/checkpoints/step_3000.pt')
CRS_METRICS = os.path.join(OUTDIR, 'crs_alpha0_3/metrics.jsonl')
BASE_METRICS = os.path.join(OUTDIR, 'baseline/metrics.jsonl')


# ==============================================================================
# Helper: load metrics
# ==============================================================================

def load_all_metrics(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ==============================================================================
# Helper: SVD analysis of matrices
# ==============================================================================

def svd_stats(W, top_k=10, label=""):
    """Compute SVD and return top-k singular values + effective rank."""
    U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
    S = S.cpu().numpy()
    total = np.sum(S ** 2)
    ratios = [(S[i] ** 2) / total for i in range(min(top_k, len(S)))]
    effective_rank = np.sum(S ** 2) ** 2 / np.sum(S ** 4) if np.sum(S ** 4) > 0 else 0
    return {
        'label': label,
        'sigma_top10': S[:min(10, len(S))].tolist(),
        'sigma_ratio_top5': [round(r, 4) for r in ratios[:5]],
        'effective_rank': round(effective_rank, 1),
        'sigma_max': float(S[0]),
        'sigma_min_nonzero': float(S[S > 1e-10][-1]) if len(S[S > 1e-10]) > 0 else 0,
        'sigma_decay': float(S[0] / (S[-1] + 1e-12)) if len(S) > 0 else 0,
    }


def analyze_attention_matrices(model, label="model"):
    """Extract and analyze SVD of all attention matrices."""
    results = {}
    for i, layer in enumerate(model.layers):
        attn = layer.attn
        prefix = f"{label}_layer{i}"

        if attn.use_crs:
            # Common branch: g_q, g_k, g_v are 1D vectors (d params each)
            # We treat them as diagonal-like operations
            results[f"{prefix}_g_q_norm"] = float(attn.g_q.data.norm().item())
            results[f"{prefix}_g_k_norm"] = float(attn.g_k.data.norm().item())
            results[f"{prefix}_g_v_norm"] = float(attn.g_v.data.norm().item())

            # Residual branch: full matrices
            results[f"{prefix}_Wq_r"] = svd_stats(attn.Wq_r.weight.data, label=f"{prefix}_Wq_r")
            results[f"{prefix}_Wk_r"] = svd_stats(attn.Wk_r.weight.data, label=f"{prefix}_Wk_r")
            results[f"{prefix}_Wv_r"] = svd_stats(attn.Wv_r.weight.data, label=f"{prefix}_Wv_r")

            # Compute effective Q/K/V matrices including common branch
            vK = attn.get_vK(model.embed)
            # The effective matrix is approximately: alpha * g * vK^T + W_r (approx)
            # We compute the effective Wq_eff = Wq_r + alpha * g_q ⊗ vK^T
            g_q = attn.g_q.data
            Wq_eff = attn.Wq_r.weight.data + attn.alpha * torch.outer(g_q, vK)
            g_k = attn.g_k.data
            Wk_eff = attn.Wk_r.weight.data + attn.alpha * torch.outer(g_k, vK)
            g_v = attn.g_v.data
            Wv_eff = attn.Wv_r.weight.data + attn.alpha * torch.outer(g_v, vK)

            results[f"{prefix}_Wq_eff"] = svd_stats(Wq_eff, label=f"{prefix}_Wq_eff")
            results[f"{prefix}_Wk_eff"] = svd_stats(Wk_eff, label=f"{prefix}_Wk_eff")
            results[f"{prefix}_Wv_eff"] = svd_stats(Wv_eff, label=f"{prefix}_Wv_eff")

        else:
            results[f"{prefix}_Wq"] = svd_stats(attn.Wq.weight.data, label=f"{prefix}_Wq")
            results[f"{prefix}_Wk"] = svd_stats(attn.Wk.weight.data, label=f"{prefix}_Wk")
            results[f"{prefix}_Wv"] = svd_stats(attn.Wv.weight.data, label=f"{prefix}_Wv")

        results[f"{prefix}_Wo"] = svd_stats(attn.Wo.weight.data, label=f"{prefix}_Wo")

    # Embedding
    results[f"{label}_E"] = svd_stats(model.embed.weight.data, label=f"{label}_E")
    return results


# ==============================================================================
# Helper: hidden state analysis
# ==============================================================================

def analyze_representations(model, dataloader, label="model", n_batches=4):
    """Run a few batches and collect hidden states for PCA/SVD analysis."""
    model.eval()
    all_h = []  # hidden states per layer
    n_layers = len(model.layers)
    layer_hs = [[] for _ in range(n_layers)]

    with torch.no_grad():
        for bidx, (input_ids, _) in enumerate(dataloader):
            if bidx >= n_batches:
                break
            input_ids = input_ids.to(DEVICE)
            h = model.embed(input_ids)
            all_h.append(h.reshape(-1, h.shape[-1]).cpu())

            for li, layer in enumerate(model.layers):
                h = layer(h, model.embed)
                layer_hs[li].append(h.reshape(-1, h.shape[-1]).cpu())

    # Concatenate
    all_h_cat = torch.cat(all_h, dim=0)
    layer_hs_cat = [torch.cat(lh, dim=0) for lh in layer_hs]

    results = {}

    # Input embedding PCA
    U, S, Vh = torch.linalg.svd(all_h_cat.float() - all_h_cat.float().mean(0), full_matrices=False)
    S = S.cpu().numpy()
    total = np.sum(S ** 2)
    results[f"{label}_E_repr_top5"] = [round((S[i] ** 2) / total, 4) for i in range(min(5, len(S)))]

    # Layer-wise hidden state PCA
    for li, lh in enumerate(layer_hs_cat):
        U, S, Vh = torch.linalg.svd(lh.float() - lh.float().mean(0), full_matrices=False)
        S = S.cpu().numpy()
        total = np.sum(S ** 2)
        results[f"{label}_layer{li}_h_top5"] = [round((S[i] ** 2) / total, 4) for i in range(min(5, len(S)))]

    # E-Q alignment per layer (cosine between top singular vectors of E and Q matrices)
    # E: (V, d). SVD gives U (V, min(V,d)), Vh (min(V,d), d).
    # Vh[0] is the top right singular vector → direction in d_model embedding space.
    # Wq: (d, d). U[:,0] is the top left singular vector → direction in query output space.
    E = model.embed.weight.data.float()
    Ue, Se, Vhe = torch.linalg.svd(E, full_matrices=False)
    e_top = Vhe[0]  # top right singular vector of embedding (direction in d_model space)

    for li, layer in enumerate(model.layers):
        attn = layer.attn
        if attn.use_crs:
            Wq = attn.Wq_r.weight.data.float()
            # Effective Q: Wq_r + alpha * g_q ⊗ vK
            vK = attn.get_vK(model.embed).float()
            Wq = Wq + attn.alpha * torch.outer(attn.g_q.data.float(), vK)
        else:
            Wq = attn.Wq.weight.data.float()

        Uq, Sq, Vhq = torch.linalg.svd(Wq, full_matrices=False)
        q_top = Uq[:, 0]
        cos_sim = torch.dot(e_top, q_top).item() / (e_top.norm().item() * q_top.norm().item() + 1e-12)
        results[f"{label}_layer{li}_E_Q_align"] = round(abs(cos_sim), 4)

    return results


# ==============================================================================
# Main analysis
# ==============================================================================

def main():
    print("=" * 70)
    print("  CRS vs Baseline: Step 3000 Spectral Analysis")
    print("=" * 70)

    # ---- Part 0: Setup data for eval ----
    qwen_dir = '/Users/bytedance/kv_cache/fdong/Qwen3-0.6B'
    data_dir = os.path.expanduser('~/Desktop/dclm')
    vocab_path = os.path.join(OUTDIR, 'vocab_mapping.pt')

    tokenizer = AutoTokenizer.from_pretrained(qwen_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    mapping = torch.load(vocab_path, map_location='cpu')
    qwen2new = {int(k): int(v) for k, v in mapping['qwen2new'].items()}

    dataset = EnglishTextDataset(
        data_dir=data_dir, max_seq_len=256, tokenizer=tokenizer,
        vocab_mapping=mapping, qwen2new=qwen2new,
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False,
                            num_workers=2, drop_last=True)
    print(f"Data ready: {len(dataset.files)} files")

    # ---- Part 1: Loss from metrics ----
    print("\n" + "=" * 70)
    print("  1. LOSS SANITY CHECK (from metrics.jsonl)")
    print("=" * 70)

    crs_metrics = load_all_metrics(CRS_METRICS)
    base_metrics = load_all_metrics(BASE_METRICS)

    for step in [1000, 2000, 3000]:
        crs_at = [r for r in crs_metrics if r['step'] == step]
        base_at = [r for r in base_metrics if r['step'] == step]
        crs_l = crs_at[0]['loss'] if crs_at else None
        base_l = base_at[0]['loss'] if base_at else None
        if crs_l and base_l:
            print(f"  step {step:>5d}:  CRS={crs_l:.4f}  Base={base_l:.4f}  Δ={crs_l - base_l:+.4f}")
        elif crs_l:
            print(f"  step {step:>5d}:  CRS={crs_l:.4f}  Base=N/A")
        elif base_l:
            print(f"  step {step:>5d}:  CRS=N/A        Base={base_l:.4f}")

    # Last available steps
    crs_last = crs_metrics[-1]['step']
    base_last = base_metrics[-1]['step']
    print(f"\n  CRS  last: step {crs_last}, loss={crs_metrics[-1]['loss']:.4f}")
    print(f"  Base last: step {base_last}, loss={base_metrics[-1]['loss']:.4f}")

    # ---- Part 2: Loss from checkpoint eval ----
    print("\n" + "=" * 70)
    print("  2. LOSS REEVALUATION (from checkpoint, 4 batches)")
    print("=" * 70)

    for name, ckpt_path, use_crs in [
        ("CRS (α=0.3)", CRS_CKPT, True),
        ("Baseline", BASE_CKPT, False),
    ]:
        if not os.path.exists(ckpt_path):
            print(f"  {name}: checkpoint not found")
            continue

        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model = SmallLM(
            vocab_size=len(mapping['new2qwen']),
            d_model=1024, n_layers=6, n_heads=8, n_kv_heads=4,
            intermediate_size=3072, max_seq_len=256,
            use_crs=use_crs, alpha=0.3,
        ).to(DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()

        total_loss = 0.0
        total_tok = 0
        with torch.no_grad():
            for bidx, (input_ids, target_ids) in enumerate(dataloader):
                if bidx >= 4:
                    break
                input_ids = input_ids.to(DEVICE)
                target_ids = target_ids.to(DEVICE)
                logits = model(input_ids)
                loss = F.cross_entropy(
                    logits.reshape(-1, model.embed.num_embeddings),
                    target_ids.reshape(-1), ignore_index=0,
                )
                total_loss += loss.item() * (target_ids != 0).sum().item()
                total_tok += (target_ids != 0).sum().item()

        avg_loss = total_loss / max(total_tok, 1)
        ppl = math.exp(min(avg_loss, 20))
        print(f"  {name}: loss={avg_loss:.4f}  ppl={ppl:.1f}  (step={ckpt['step']})")

    # ---- Part 3: SVD of parameter matrices ----
    print("\n" + "=" * 70)
    print("  3. SINGULAR VALUE SPECTRUM (Parameter Matrices)")
    print("=" * 70)

    all_svd = {}

    for name, ckpt_path, use_crs in [
        ("CRS", CRS_CKPT, True),
        ("Baseline", BASE_CKPT, False),
    ]:
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model = SmallLM(
            vocab_size=len(mapping['new2qwen']),
            d_model=1024, n_layers=6, n_heads=8, n_kv_heads=4,
            intermediate_size=3072, max_seq_len=256,
            use_crs=use_crs, alpha=0.3,
        ).to(DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        all_svd[name] = analyze_attention_matrices(model, label=name)

    # --- Compare Wq/Wk/Wv sigma_max and effective rank ---
    print("\n  --- Wq σ_max (effective rank) comparison ---")
    for li in range(6):
        if 'CRS_layer0_Wq_eff' not in all_svd['CRS']:
            break
        crs_wq = all_svd['CRS'][f'CRS_layer{li}_Wq_eff']
        base_wq = all_svd['Baseline'][f'Baseline_layer{li}_Wq']
        print(f"  layer {li}:  CRS σ1={crs_wq['sigma_max']:.3f} (rank={crs_wq['effective_rank']})  "
              f"Base σ1={base_wq['sigma_max']:.3f} (rank={base_wq['effective_rank']})")

    print("\n  --- Wk σ_max (effective rank) comparison ---")
    for li in range(6):
        crs_wk = all_svd['CRS'][f'CRS_layer{li}_Wk_eff']
        base_wk = all_svd['Baseline'][f'Baseline_layer{li}_Wk']
        print(f"  layer {li}:  CRS σ1={crs_wk['sigma_max']:.3f} (rank={crs_wk['effective_rank']})  "
              f"Base σ1={base_wk['sigma_max']:.3f} (rank={base_wk['effective_rank']})")

    print("\n  --- Wv σ_max (effective rank) comparison ---")
    for li in range(6):
        crs_wv = all_svd['CRS'][f'CRS_layer{li}_Wv_eff']
        base_wv = all_svd['Baseline'][f'Baseline_layer{li}_Wv']
        print(f"  layer {li}:  CRS σ1={crs_wv['sigma_max']:.3f} (rank={crs_wv['effective_rank']})  "
              f"Base σ1={base_wv['sigma_max']:.3f} (rank={base_wv['effective_rank']})")

    print("\n  --- Wo σ_max (effective rank) comparison ---")
    for li in range(6):
        crs_wo = all_svd['CRS'][f'CRS_layer{li}_Wo']
        base_wo = all_svd['Baseline'][f'Baseline_layer{li}_Wo']
        print(f"  layer {li}:  CRS σ1={crs_wo['sigma_max']:.3f} (rank={crs_wo['effective_rank']})  "
              f"Base σ1={base_wo['sigma_max']:.3f} (rank={base_wo['effective_rank']})")

    # --- Embedding spectrum ---
    print("\n  --- Embedding E spectrum ---")
    crs_e = all_svd['CRS']['CRS_E']
    base_e = all_svd['Baseline']['Baseline_E']
    print(f"  CRS     E: σ1={crs_e['sigma_max']:.3f}, rank={crs_e['effective_rank']}")
    print(f"  Baseline E: σ1={base_e['sigma_max']:.3f}, rank={base_e['effective_rank']}")
    print(f"  CRS     E σ_top5 ratios: {crs_e['sigma_ratio_top5']}")
    print(f"  Baseline E σ_top5 ratios: {base_e['sigma_ratio_top5']}")

    # ---- Part 4: CRS-specific: Common vs Residual branch analysis ----
    print("\n" + "=" * 70)
    print("  4. CRS SPLIT ANALYSIS: Common vs Residual Branch")
    print("=" * 70)

    # Reload CRS model for detailed analysis
    ckpt = torch.load(CRS_CKPT, map_location=DEVICE, weights_only=False)
    crs_model = SmallLM(
        vocab_size=len(mapping['new2qwen']),
        d_model=1024, n_layers=6, n_heads=8, n_kv_heads=4,
        intermediate_size=3072, max_seq_len=256,
        use_crs=True, alpha=0.3,
    ).to(DEVICE)
    crs_model.load_state_dict(ckpt['model_state_dict'])
    crs_model.eval()

    print("\n  --- Common branch (g_q, g_k, g_v) norms ---")
    for li, layer in enumerate(crs_model.layers):
        attn = layer.attn
        gq_n = attn.g_q.data.norm().item()
        gk_n = attn.g_k.data.norm().item()
        gv_n = attn.g_v.data.norm().item()
        Wqr_n = attn.Wq_r.weight.data.norm().item()
        Wkr_n = attn.Wk_r.weight.data.norm().item()
        Wvr_n = attn.Wv_r.weight.data.norm().item()
        print(f"  layer {li}:  g_q={gq_n:.3f}  g_k={gk_n:.3f}  g_v={gv_n:.3f}  "
              f"||Wq_r||={Wqr_n:.1f}  ||Wk_r||={Wkr_n:.1f}  ||Wv_r||={Wvr_n:.1f}")

    print("\n  --- Residual branch Wq_r vs Effective Wq_eff σ decay ---")
    for li in range(6):
        Wqr = all_svd['CRS'][f'CRS_layer{li}_Wq_r']
        Wqeff = all_svd['CRS'][f'CRS_layer{li}_Wq_eff']
        print(f"  layer {li}:  Wq_r σ1={Wqr['sigma_max']:.3f} (rank={Wqr['effective_rank']})  "
              f"→ eff σ1={Wqeff['sigma_max']:.3f} (rank={Wqeff['effective_rank']})")

    # ---- Part 5: Representation space analysis ----
    print("\n" + "=" * 70)
    print("  5. REPRESENTATION SPACE ANALYSIS")
    print("=" * 70)

    rep_results = {}
    for name, ckpt_path, use_crs in [
        ("CRS", CRS_CKPT, True),
        ("Baseline", BASE_CKPT, False),
    ]:
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model = SmallLM(
            vocab_size=len(mapping['new2qwen']),
            d_model=1024, n_layers=6, n_heads=8, n_kv_heads=4,
            intermediate_size=3072, max_seq_len=256,
            use_crs=use_crs, alpha=0.3,
        ).to(DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
        rep_results[name] = analyze_representations(model, dataloader, label=name)

    print("\n  --- E-Q alignment (cosine between top singular vectors) ---")
    for li in range(6):
        crs_align = rep_results['CRS'][f'CRS_layer{li}_E_Q_align']
        base_align = rep_results['Baseline'][f'Baseline_layer{li}_E_Q_align']
        print(f"  layer {li}:  CRS cos(E,Q)={crs_align:.4f}  Base cos(E,Q)={base_align:.4f}")

    print("\n  --- Hidden state representation concentration (top singular value ratio) ---")
    crs_e_repr = rep_results['CRS']['CRS_E_repr_top5']
    base_e_repr = rep_results['Baseline']['Baseline_E_repr_top5']
    print(f"  Embedding repr: CRS top5={crs_e_repr}  Base top5={base_e_repr}")

    for li in range(6):
        crs_h = rep_results['CRS'][f'CRS_layer{li}_h_top5']
        base_h = rep_results['Baseline'][f'Baseline_layer{li}_h_top5']
        print(f"  layer {li}:  CRS h_top3={crs_h[:3]}  Base h_top3={base_h[:3]}")

    # ---- Part 6: Summary & Recommendations ----
    print("\n" + "=" * 70)
    print("  6. FINDINGS & RECOMMENDATIONS")
    print("=" * 70)

    print("""
  ▸ LOSS: CRS slightly better at all checkpoints (Δ ~ -0.01), but difference
    is marginal. Pure LM loss is NOT the right metric for evaluating
    whether the subspace split is working — the theory predicts spectral
    separation, not necessarily better perplexity at equal steps.

  ▸ TO TEST SPECTRAL SEPARATION:
    - Compare σ1 / effective rank of Wq/Wk/Wv between CRS and baseline
    - Check if CRS effective matrices have reduced σ1 (less common-mode dominance)
    - Check if residual branch Wq_r/Wk_r/Wv_r have different spectral shape
      from baseline Wq/Wk/Wv (more uniform, less dominated by single direction)

  ▸ NEXT STEPS:
    1. If spectral separation is visible → CRS works, try α=0.1, 0.5
    2. If spectral separation is absent at step 3000 → train longer (10k-20k steps)
       as the singular value gap may take many steps to develop
    3. Consider a smaller model (50M params, d=512, 4 layers) for faster iteration
       (would cut training time from ~11h to ~3h per 5000 steps)

  ▸ MODEL SIZE & ITERATION SPEED:
    Current 109M model:  ~11h per 5000 steps on MPS
    Smaller 50M model:    ~3h per 5000 steps
    Even smaller 25M model: ~1.5h

    With the current setup, a full experiment cycle (3 alphas × 5k steps):
    - 109M: ~33h (unrealistic for quick iteration)
    - 50M:  ~9h (doable)
    - 25M:  ~4.5h (fast iteration)

    Recommendation: Reduce to d=512, 4 layers, 4 heads → ~50M params.
    This still has enough capacity to show spectral separation if the
    theory is correct, while cutting iteration time by ~3x.
""")

    # ---- Save results as JSON ----
    output_path = os.path.join(OUTDIR, 'step3000_analysis.json')
    with open(output_path, 'w') as f:
        json.dump({
            'svd': {k: v for k, v in all_svd.items()},
            'representations': rep_results,
        }, f, indent=2, default=str)
    print(f"Full results saved to: {output_path}")


if __name__ == '__main__':
    main()
