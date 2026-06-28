#!/usr/bin/env python3
"""
Synthetic CRS experiment.

Data:  500 tokens (10 K common + 490 R rare), 50 fixed patterns of length 10,
       K tokens collectively ~30% of positions, each individual K ~3%.

Model: 1 TransformerBlock, d_model=128, 4 heads, SwiGLU MLP.
       CRS: attention + MLP both split into common (rank-p bottleneck) +
       residual (full matrix) branches.

       h_common[t] = cummean(h[:, :t+1], dim=1).detach()
       branch_out  = W_up(W_down(h_common))          ← rank-p, p=8 default
       residual    = full_matrix(h)                  ← d×d, with gradient
       output      = α * branch_out + residual

Usage:
    python3 workbuddy_scripts/synthetic_crs_experiment.py --mode crs --alpha 0.3 --p 8
    python3 workbuddy_scripts/synthetic_crs_experiment.py --mode baseline
"""

import argparse, json, math, os, sys, time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ==============================================================================
# Synthetic data generation
# ==============================================================================

def generate_synthetic_patterns(n_patterns=200, seed=42):
    """
    Generate n_patterns fixed patterns.
    - 500 tokens: 0-9 are K, 10-499 are R.
    - Each pattern: 10 positions, 3 K + 7 R.
    - Each K token appears in n_patterns * 3 / 10 patterns (3% of all positions).
    - Deterministic from seed.
    """
    rng = np.random.RandomState(seed)
    vocab_size = 500
    n_K = 10
    n_R = vocab_size - n_K  # 490
    seq_len = 10
    n_K_per_pattern = 3
    n_R_per_pattern = seq_len - n_K_per_pattern  # 7

    # --- Assign K positions per pattern at random ---
    K_positions = []  # list of list of 3 positions per pattern
    for _ in range(n_patterns):
        pos = sorted(rng.choice(seq_len, size=n_K_per_pattern, replace=False).tolist())
        K_positions.append(pos)

    # --- Assign which K token goes to which K slot ---
    # Total K slots = n_patterns * n_K_per_pattern = 150
    # Each of 10 K tokens should appear 15 times.
    total_K_slots = n_patterns * n_K_per_pattern
    K_ids_pool = []
    for k in range(n_K):
        K_ids_pool.extend([k] * (total_K_slots // n_K))
    rng.shuffle(K_ids_pool)

    # --- Assign R tokens to R slots ---
    # Total R slots = n_patterns * n_R_per_pattern
    # If more slots than available R tokens, allow replacement.
    total_R_slots = n_patterns * n_R_per_pattern
    R_ids_pool = rng.choice(n_R, size=total_R_slots, replace=(total_R_slots > n_R)).tolist()
    R_ids_pool = [r + n_K for r in R_ids_pool]  # offset to actual IDs

    # --- Build patterns ---
    patterns = []
    kidx = 0
    ridx = 0
    for pi in range(n_patterns):
        seq = [0] * seq_len
        for pos in K_positions[pi]:
            seq[pos] = K_ids_pool[kidx]
            kidx += 1
        for pos in range(seq_len):
            if pos not in K_positions[pi]:
                seq[pos] = R_ids_pool[ridx]
                ridx += 1
        patterns.append(seq)

    # --- Compute actual statistics ---
    K_counts = defaultdict(int)
    R_counts = defaultdict(int)
    for seq in patterns:
        for tok in seq:
            if tok < n_K:
                K_counts[tok] += 1
            else:
                R_counts[tok] += 1

    return {
        'patterns': patterns,
        'K_positions': K_positions,
        'n_K': n_K,
        'vocab_size': vocab_size,
        'seq_len': seq_len,
        'n_patterns': n_patterns,
        'K_ids': list(range(n_K)),
        'R_ids': list(range(n_K, vocab_size)),
        'K_counts': dict(K_counts),
        'R_counts': dict(R_counts),
    }


class SyntheticPatternDataset(Dataset):
    """Wraps fixed patterns into a dataset for DataLoader."""
    def __init__(self, patterns, n_samples=10000):
        self.patterns = patterns
        self.seq_len = len(patterns[0])
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        rng = np.random.RandomState(index)
        pi = rng.randint(0, len(self.patterns))
        seq = self.patterns[pi]
        input_ids = torch.tensor(seq[:self.seq_len], dtype=torch.long)
        target_ids = torch.tensor(seq[:self.seq_len], dtype=torch.long)
        return input_ids, target_ids


# ==============================================================================
# Model components
# ==============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


class RankPBottleneck(nn.Module):
    """Rank-p projection: W_down (d→p) → W_up (p→d_out), no bias, no activation."""
    def __init__(self, d_in, d_out, p):
        super().__init__()
        self.W_down = nn.Linear(d_in, p, bias=False)
        self.W_up = nn.Linear(p, d_out, bias=False)

    def forward(self, x):
        return self.W_up(self.W_down(x))


def apply_rotary_emb(x, freqs_cis):
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs = freqs_cis[:x.shape[2]].unsqueeze(0).unsqueeze(0)
    x_rotated = x_complex * freqs
    return torch.view_as_real(x_rotated).flatten(-2).type_as(x)


def precompute_freqs_cis(dim, max_seq_len, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[:dim//2].float() / dim))
    t = torch.arange(max_seq_len)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


class CausalAttention(nn.Module):
    """Multi-head causal attention with optional rank-p CRS split."""

    def __init__(self, d_model, n_heads, n_kv_heads, max_seq_len, use_crs=False, alpha=0.3, p=8, residual_beta=1.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.use_crs = use_crs
        self.alpha = alpha
        self.residual_beta = residual_beta
        self.scale = math.sqrt(self.head_dim)

        q_dim = n_heads * self.head_dim   # = d_model
        kv_dim = n_kv_heads * self.head_dim

        if use_crs:
            self.q_branch = RankPBottleneck(d_model, q_dim, p)
            self.k_branch = RankPBottleneck(d_model, kv_dim, p)
            self.v_branch = RankPBottleneck(d_model, kv_dim, p)
            self.Wq_r = nn.Linear(d_model, q_dim, bias=False)
            self.Wk_r = nn.Linear(d_model, kv_dim, bias=False)
            self.Wv_r = nn.Linear(d_model, kv_dim, bias=False)
        else:
            self.Wq = nn.Linear(d_model, q_dim, bias=False)
            self.Wk = nn.Linear(d_model, kv_dim, bias=False)
            self.Wv = nn.Linear(d_model, kv_dim, bias=False)

        self.Wo = nn.Linear(q_dim, d_model, bias=False)

        freqs = precompute_freqs_cis(self.head_dim, max_seq_len)
        self.register_buffer('freqs_cis', freqs)

        mask = torch.triu(torch.full((max_seq_len, max_seq_len), float('-inf')), diagonal=1)
        self.register_buffer('causal_mask', mask)

    def forward(self, h):
        B, T, D = h.shape

        if self.use_crs:
            # h_common: per-position running mean of hidden states, detached
            denom = torch.arange(1, T + 1, device=h.device, dtype=h.dtype).view(1, -1, 1)
            h_cum = torch.cumsum(h, dim=1)
            h_common = (h_cum / denom).detach()

            q_c = self.q_branch(h_common).reshape(B, T, self.n_heads, self.head_dim)
            k_c = self.k_branch(h_common).reshape(B, T, self.n_kv_heads, self.head_dim)
            v_c = self.v_branch(h_common).reshape(B, T, self.n_kv_heads, self.head_dim)

            q_r = self.Wq_r(h).reshape(B, T, self.n_heads, self.head_dim)
            k_r = self.Wk_r(h).reshape(B, T, self.n_kv_heads, self.head_dim)
            v_r = self.Wv_r(h).reshape(B, T, self.n_kv_heads, self.head_dim)

            q = self.alpha * q_c + self.residual_beta * q_r
            k = self.alpha * k_c + self.residual_beta * k_r
            v = self.alpha * v_c + self.residual_beta * v_r
        else:
            q = self.Wq(h).reshape(B, T, self.n_heads, self.head_dim)
            k = self.Wk(h).reshape(B, T, self.n_kv_heads, self.head_dim)
            v = self.Wv(h).reshape(B, T, self.n_kv_heads, self.head_dim)

        q = apply_rotary_emb(q.permute(0, 2, 1, 3), self.freqs_cis)
        k = apply_rotary_emb(k.permute(0, 2, 1, 3), self.freqs_cis)
        v = v.permute(0, 2, 1, 3)

        if self.n_heads > self.n_kv_heads:
            r = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(r, dim=1)
            v = v.repeat_interleave(r, dim=1)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn_scores = attn_scores + self.causal_mask[:T, :T]
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_weights, v)
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, T, self.n_heads * self.head_dim)
        return self.Wo(attn_out)


class MLP(nn.Module):
    """SwiGLU MLP with optional rank-p CRS split."""

    def __init__(self, d_model, intermediate_size, use_crs=False, alpha=0.3, p=8, residual_beta=1.0):
        super().__init__()
        self.use_crs = use_crs
        self.alpha = alpha
        self.residual_beta = residual_beta
        self.W1 = nn.Linear(d_model, intermediate_size, bias=False)
        self.W2 = nn.Linear(intermediate_size, d_model, bias=False)
        self.W3 = nn.Linear(d_model, intermediate_size, bias=False)
        if use_crs:
            self.mlp_branch = RankPBottleneck(d_model, d_model, p)

    def forward(self, x):
        if self.use_crs:
            B, T, D = x.shape
            denom = torch.arange(1, T + 1, device=x.device, dtype=x.dtype).view(1, -1, 1)
            h_cum = torch.cumsum(x, dim=1)
            h_common = (h_cum / denom).detach()
            mlp_c = self.mlp_branch(h_common)                       # (B, T, d)
            mlp_r = self.W2(F.silu(self.W1(x)) * self.W3(x))       # (B, T, d)
            return self.alpha * mlp_c + self.residual_beta * mlp_r
        else:
            return self.W2(F.silu(self.W1(x)) * self.W3(x))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, intermediate_size, max_seq_len,
                 use_crs=False, alpha=0.3, p=8, residual_beta=1.0):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = CausalAttention(d_model, n_heads, n_kv_heads, max_seq_len,
                                     use_crs=use_crs, alpha=alpha, p=p, residual_beta=residual_beta)
        self.mlp_norm = RMSNorm(d_model)
        self.mlp = MLP(d_model, intermediate_size, use_crs=use_crs, alpha=alpha, p=p, residual_beta=residual_beta)

    def forward(self, h):
        h = h + self.attn(self.attn_norm(h))
        h = h + self.mlp(self.mlp_norm(h))
        return h


class SmallLM(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_layers=1, n_heads=4, n_kv_heads=2,
                 intermediate_size=384, max_seq_len=16, use_crs=False, alpha=0.3, p=8, residual_beta=1.0):
        super().__init__()
        self.d_model = d_model
        self.use_crs = use_crs

        self.embed = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embed.weight, std=1.0 / math.sqrt(d_model))
        self.norm = RMSNorm(d_model)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, n_kv_heads, intermediate_size, max_seq_len,
                             use_crs=use_crs, alpha=alpha, p=p, residual_beta=residual_beta)
            for _ in range(n_layers)
        ])

    def forward(self, input_ids):
        h = self.embed(input_ids)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        logits = h @ self.embed.weight.T
        return logits

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


# ==============================================================================
# Training
# ==============================================================================

def svd_analysis(model, label=""):
    """Return SVD stats for all attention and MLP matrices."""
    results = {}
    for li, layer in enumerate(model.layers):
        attn = layer.attn
        mlp = layer.mlp
        prefix = f"{label}_layer{li}"

        # Attention
        if attn.use_crs:
            for mat_name, module in [('Wq_r', attn.Wq_r), ('Wk_r', attn.Wk_r),
                                      ('Wv_r', attn.Wv_r), ('Wo', attn.Wo)]:
                W = module.weight.data.float()
                U, S_svd, Vh = torch.linalg.svd(W, full_matrices=False)
                S_svd = S_svd.cpu().numpy()
                total = np.sum(S_svd ** 2)
                results[f"{prefix}_{mat_name}_sigma1"] = float(S_svd[0])
                results[f"{prefix}_{mat_name}_effrank"] = float(np.sum(S_svd ** 2) ** 2 / max(np.sum(S_svd ** 4), 1e-30))

            # Common branch effective matrices
            for br_name, br in [('q_branch', attn.q_branch), ('k_branch', attn.k_branch),
                                 ('v_branch', attn.v_branch)]:
                W_up = br.W_up.weight.data.float()   # (d_out, p)
                W_dn = br.W_down.weight.data.float()  # (p, d_in)
                W_eff = W_up @ W_dn                  # (d_out, d_in)
                U, S_svd, Vh = torch.linalg.svd(W_eff, full_matrices=False)
                S_svd = S_svd.cpu().numpy()
                results[f"{prefix}_{br_name}_sigma1"] = float(S_svd[0])
                results[f"{prefix}_{br_name}_effrank"] = float(min(np.sum(S_svd ** 2) ** 2 / max(np.sum(S_svd ** 4), 1e-30), W_eff.shape[0]))
        else:
            for mat_name in ['Wq', 'Wk', 'Wv', 'Wo']:
                module = getattr(attn, mat_name)
                W = module.weight.data.float()
                U, S_svd, Vh = torch.linalg.svd(W, full_matrices=False)
                S_svd = S_svd.cpu().numpy()
                total = np.sum(S_svd ** 2)
                results[f"{prefix}_{mat_name}_sigma1"] = float(S_svd[0])
                results[f"{prefix}_{mat_name}_effrank"] = float(np.sum(S_svd ** 2) ** 2 / max(np.sum(S_svd ** 4), 1e-30))

        # MLP common branch
        if mlp.use_crs:
            W_up = mlp.mlp_branch.W_up.weight.data.float()   # (d_model, p)
            W_dn = mlp.mlp_branch.W_down.weight.data.float()  # (p, d_model)
            W_eff = W_up @ W_dn                              # (d_model, d_model)
            U, S_svd, Vh = torch.linalg.svd(W_eff, full_matrices=False)
            S_svd = S_svd.cpu().numpy()
            results[f"{prefix}_mlp_branch_sigma1"] = float(S_svd[0])
            results[f"{prefix}_mlp_branch_effrank"] = float(min(np.sum(S_svd ** 2) ** 2 / max(np.sum(S_svd ** 4), 1e-30), W_eff.shape[0]))

        # Embedding
        E = model.embed.weight.data.float()
        U, S_svd, Vh = torch.linalg.svd(E, full_matrices=False)
        S_svd = S_svd.cpu().numpy()
        total = np.sum(S_svd ** 2)
        results[f"{label}_E_sigma1"] = float(S_svd[0])
        results[f"{label}_E_top5ratio"] = [round(float(S_svd[i] ** 2 / total), 4) for i in range(min(5, len(S_svd)))]

    return results


def train(args):
    device = torch.device('mps' if torch.backends.mps.is_available() else
                          'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Mode: {args.mode}, alpha={args.alpha}, p={args.p}")

    # --- Synthetic data ---
    data = generate_synthetic_patterns(n_patterns=args.n_patterns, seed=42)
    patterns = data['patterns']
    vocab_size = data['vocab_size']
    K_ids = set(data['K_ids'])  # {0,...,9}
    seq_len = data['seq_len']
    K_positions = data['K_positions']

    print(f"\nData: {data['n_patterns']} patterns × {seq_len} tokens")
    print(f"  K tokens: {data['n_K']} (IDs {data['K_ids']})")
    print(f"  R tokens: {vocab_size - data['n_K']} (IDs {data['n_K']}-{vocab_size-1})")
    print(f"  K counts: {dict(sorted(data['K_counts'].items()))}")
    print(f"  Total R tokens used: {len(data['R_counts'])}")

    # --- Generate TEST patterns (different seed, not seen during training) ---
    test_data = generate_synthetic_patterns(n_patterns=args.n_patterns, seed=12345)
    test_patterns = test_data['patterns']

    train_dataset = SyntheticPatternDataset(patterns, n_samples=5000)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # For evaluation, use test patterns (unseen)
    eval_loader = DataLoader(
        SyntheticPatternDataset(test_patterns, n_samples=args.batch_size * 5),
        batch_size=args.batch_size, shuffle=False, drop_last=True
    )

    # --- Model ---
    use_crs = (args.mode == 'crs')
    d = args.d_model
    n_heads = 4 if d >= 64 else 2
    n_kv_heads = 2 if d >= 64 else 1
    intermediate = d * 3
    # Scale p with d if not explicitly different from default
    p = args.p

    model = SmallLM(
        vocab_size=vocab_size, d_model=d, n_layers=1, n_heads=n_heads, n_kv_heads=n_kv_heads,
        intermediate_size=intermediate, max_seq_len=seq_len + 2,
        use_crs=use_crs, alpha=args.alpha, p=p, residual_beta=args.residual_beta,
    ).to(device)
    print(f"\nModel: {model.param_count():,} params (d={d}, heads={n_heads}/{n_kv_heads})")
    print(f"  CRS: {use_crs}, α={args.alpha}, bottleneck rank p={p}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                                   weight_decay=args.weight_decay)

    # --- Output dir ---
    run_name = f"{args.mode}_a{args.alpha}_p{args.p}" if use_crs else 'baseline'
    out_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)
    metrics_path = os.path.join(out_dir, 'metrics.jsonl')

    # --- Training loop ---
    model.train()
    global_step = 0
    step_start = time.time()
    total_start = step_start

    print(f"\nTraining: {args.max_steps} steps, lr={args.lr}, bs={args.batch_size}")
    print(f"  Metrics → {metrics_path}")

    for epoch in range(100):
        for input_ids, target_ids in train_loader:
            if global_step >= args.max_steps:
                break

            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size), target_ids.reshape(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            global_step += 1

            # --- Ablation: freeze common branch at specified step ---
            if args.freeze_common_at > 0 and global_step == args.freeze_common_at and use_crs:
                print(f"\n  == Freezing common branch at step {global_step} ==")
                for layer in model.layers:
                    for module in [layer.attn.q_branch, layer.attn.k_branch,
                                   layer.attn.v_branch, layer.mlp.mlp_branch]:
                        for p in module.parameters():
                            p.requires_grad = False
                # Re-create optimizer with reduced param set (only trainable params remain)
                optimizer = torch.optim.AdamW(
                    [p for p in model.parameters() if p.requires_grad],
                    lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)

            if global_step % args.log_every == 0:
                # --- Evaluate loss by K/R ---
                model.eval()
                K_loss_total = 0.0
                K_count = 0
                R_loss_total = 0.0
                R_count = 0
                with torch.no_grad():
                    for eid, (e_ids, e_tgt) in enumerate(eval_loader):
                        if eid >= 10:
                            break
                        e_ids = e_ids.to(device)
                        e_tgt = e_tgt.to(device)
                        e_logits = model(e_ids)
                        e_loss = F.cross_entropy(
                            e_logits.reshape(-1, vocab_size),
                            e_tgt.reshape(-1), reduction='none'
                        ).reshape(args.batch_size, seq_len)

                        # Need K positions per pattern — match eval data to patterns
                        for bi in range(e_ids.shape[0]):
                            # For eval, just use all patterns (not pattern-specific)
                            # K positions are global: any K token position
                            for t in range(seq_len):
                                tok = e_tgt[bi, t].item()
                                if tok in K_ids:
                                    K_loss_total += e_loss[bi, t].item()
                                    K_count += 1
                                else:
                                    R_loss_total += e_loss[bi, t].item()
                                    R_count += 1
                model.train()

                K_loss = K_loss_total / max(K_count, 1)
                R_loss = R_loss_total / max(R_count, 1)
                elapsed = time.time() - step_start
                tok_per_sec = (args.batch_size * seq_len * args.log_every) / max(elapsed, 1)

                print(f"step {global_step:>6d}/{args.max_steps}  "
                      f"loss={loss.item():.4f}  test_K={K_loss:.4f}  test_R={R_loss:.4f}  "
                      f"tok/s={tok_per_sec:,.0f}")

                with open(metrics_path, 'a') as f:
                    f.write(json.dumps({
                        'step': global_step,
                        'loss': round(loss.item(), 6),
                        'K_loss': round(K_loss, 6),
                        'R_loss': round(R_loss, 6),
                    }) + '\n')

                step_start = time.time()

            # Save checkpoints at key steps for spectral analysis
            ckpt_steps = {50, 100, 200, 500, 1000, 2000, 3000, 5000}
            if global_step in ckpt_steps:
                ckpt_path = os.path.join(out_dir, f'ckpt_step{global_step}.pt')
                torch.save({
                    'step': global_step,
                    'model_state_dict': model.state_dict(),
                }, ckpt_path)

        if global_step >= args.max_steps:
            break

    # --- Final analysis ---
    model.eval()
    results = svd_analysis(model, label=args.mode)

    # Also evaluate final loss breakdown
    K_loss_total = 0.0
    K_count = 0
    R_loss_total = 0.0
    R_count = 0
    with torch.no_grad():
        for eid, (e_ids, e_tgt) in enumerate(eval_loader):
            if eid >= 10:
                break
            e_ids = e_ids.to(device)
            e_tgt = e_tgt.to(device)
            e_logits = model(e_ids)
            e_loss = F.cross_entropy(
                e_logits.reshape(-1, vocab_size),
                e_tgt.reshape(-1), reduction='none'
            ).reshape(args.batch_size, seq_len)
            for bi in range(e_ids.shape[0]):
                for t in range(seq_len):
                    tok = e_tgt[bi, t].item()
                    if tok in K_ids:
                        K_loss_total += e_loss[bi, t].item()
                        K_count += 1
                    else:
                        R_loss_total += e_loss[bi, t].item()
                        R_count += 1

    results['final_K_loss'] = round(K_loss_total / max(K_count, 1), 6)
    results['final_R_loss'] = round(R_loss_total / max(R_count, 1), 6)
    results['final_loss'] = round((K_loss_total + R_loss_total) / max(K_count + R_count, 1), 6)

    result_path = os.path.join(out_dir, 'final_analysis.json')
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nAnalysis saved → {result_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"  Final: K_loss={results['final_K_loss']:.4f}  "
          f"R_loss={results['final_R_loss']:.4f}  "
          f"total={results['final_loss']:.4f}")
    if results.get('baseline_layer0_Wq_sigma1') or results.get('crs_layer0_Wq_r_sigma1'):
        print(f"\n  SVD (layer 0):")
        if 'baseline_layer0_Wq_sigma1' in results:
            print(f"    Baseline Wq: σ1={results['baseline_layer0_Wq_sigma1']:.3f}  "
                  f"effrank={results['baseline_layer0_Wq_effrank']:.1f}")
        if 'crs_layer0_Wq_r_sigma1' in results:
            print(f"    CRS Wq_r:    σ1={results['crs_layer0_Wq_r_sigma1']:.3f}  "
                  f"effrank={results['crs_layer0_Wq_r_effrank']:.1f}")
            if 'crs_layer0_q_branch_sigma1' in results:
                print(f"    CRS q_branch: σ1={results['crs_layer0_q_branch_sigma1']:.3f}  "
                      f"effrank={results['crs_layer0_q_branch_effrank']:.1f}")
    print(f"{'='*60}")


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='crs', choices=['baseline', 'crs'])
    parser.add_argument('--alpha', type=float, default=0.3)
    parser.add_argument('--p', type=int, default=8, help='Common branch bottleneck rank')
    parser.add_argument('--d_model', type=int, default=128,
                        help='Model dimension (64 for smaller test)')
    parser.add_argument('--freeze_common_at', type=int, default=0,
                        help='If >0, freeze common branch params at this step (ablation)')
    parser.add_argument('--residual_beta', type=float, default=1.0,
                        help='Amplification factor for residual branch (β), default 1.0')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_steps', type=int, default=2000)
    parser.add_argument('--n_patterns', type=int, default=200,
                        help='Number of fixed patterns (more = harder to memorize)')
    parser.add_argument('--log_every', type=int, default=50)
    parser.add_argument('--output_dir', type=str,
                        default='/Users/bytedance/kv_cache/fdong_embedding_dim/outputs/synthetic_crs')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)
