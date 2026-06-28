#!/usr/bin/env python3
"""
Iterative SVD Decomposition Training.

每个 Transformer 内的线性矩阵（Wq, Wk, Wv, Wo, W1, W2, W3）独立执行
迭代奇异方向提取。训练过程中定期检查各矩阵的 top 奇异方向是否收敛，
收敛后将其从参数矩阵和输入 hidden state 中同时拆出并冻结。

Forward:
    frozen_out = 0
    for each frozen (σᵢ, uᵢ, vᵢ):
        sᵢ = h · vᵢ                    ← 投影到 vᵢ 方向
        frozen_out += σᵢ · sᵢ · uᵢ    ← 累加 frozen 分支输出
        h = h - sᵢ · vᵢ               ← 从输入中移除该方向
    residual_out = W^(k) · h           ← 残差矩阵处理正交分量
    total = frozen_out + residual_out

Usage:
    python3 workbuddy_scripts/iterative_svd_experiment.py --mode svd --extract_every 200
    python3 workbuddy_scripts/iterative_svd_experiment.py --mode baseline
"""

import argparse, json, math, os, sys, time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


# ==============================================================================
# Synthetic data (reused)
# ==============================================================================

def generate_synthetic_patterns(n_patterns=200, seed=42):
    rng = np.random.RandomState(seed)
    vocab_size = 500
    n_K = 10
    n_R = vocab_size - n_K
    seq_len = 10
    n_K_per_pattern = 3
    n_R_per_pattern = seq_len - n_K_per_pattern

    K_positions = []
    for _ in range(n_patterns):
        pos = sorted(rng.choice(seq_len, size=n_K_per_pattern, replace=False).tolist())
        K_positions.append(pos)

    total_K_slots = n_patterns * n_K_per_pattern
    K_ids_pool = []
    for k in range(n_K):
        K_ids_pool.extend([k] * (total_K_slots // n_K))
    rng.shuffle(K_ids_pool)

    total_R_slots = n_patterns * n_R_per_pattern
    R_ids_pool = rng.choice(n_R, size=total_R_slots, replace=(total_R_slots > n_R)).tolist()
    R_ids_pool = [r + n_K for r in R_ids_pool]

    patterns = []
    kidx = 0
    ridx = 0
    for pi in range(n_patterns):
        seq = [0] * seq_len
        for pos in K_positions[pi]:
            seq[pos] = K_ids_pool[kidx]; kidx += 1
        for pos in range(seq_len):
            if pos not in K_positions[pi]:
                seq[pos] = R_ids_pool[ridx]; ridx += 1
        patterns.append(seq)

    K_counts = defaultdict(int)
    for seq in patterns:
        for tok in seq:
            if tok < n_K: K_counts[tok] += 1

    return {
        'patterns': patterns, 'K_positions': K_positions, 'n_K': n_K,
        'vocab_size': vocab_size, 'seq_len': seq_len, 'n_patterns': n_patterns,
        'K_ids': list(range(n_K)), 'R_ids': list(range(n_K, vocab_size)),
        'K_counts': dict(K_counts),
    }


class SyntheticPatternDataset(torch.utils.data.Dataset):
    def __init__(self, patterns, n_samples=10000):
        self.patterns = patterns
        self.seq_len = len(patterns[0])
        self.n_samples = n_samples

    def __len__(self):        return self.n_samples

    def __getitem__(self, index):
        rng = np.random.RandomState(index)
        pi = rng.randint(0, len(self.patterns))
        seq = self.patterns[pi]
        return (torch.tensor(seq[:self.seq_len], dtype=torch.long),
                torch.tensor(seq[:self.seq_len], dtype=torch.long))


# ==============================================================================
# DecomposableLinear: 支持迭代奇异方向提取与冻结
# ==============================================================================

class DecomposableLinear(nn.Module):
    """
    可迭代拆分的线性层。

    内部状态：
      - self.W: (d_out, d_in) trainable 权重矩阵（残差部分）
      - self.frozen: list of {"sigma":float, "u":Tensor(d_out), "v":Tensor(d_in)}
                     已冻结的 rank-1 组件。训练时 frozen 参数不做梯度更新。

    Forward:
      1. 计算 h_orth = h - Σ (h·v)·v  （移除所有 frozen v 方向上的投影）
      2. residual_out = self.W @ h_orth
      3. frozen_out  = Σ σᵢ·(h·v)·uᵢ    （h 是原始输入，不是 h_orth）
      4. return residual_out + frozen_out

    extract():  对 self.W 做 SVD 取第一分量，冻结，从 self.W 减去该分量。
    """

    def __init__(self, d_in, d_out):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.W = nn.Parameter(torch.empty(d_out, d_in))
        nn.init.kaiming_uniform_(self.W)
        self._n_extracted = 0

        # Frozen components
        self.register_buffer('_frozen_u', torch.empty(0, d_out))
        self.register_buffer('_frozen_v', torch.empty(0, d_in))
        self.register_buffer('_frozen_sigma', torch.empty(0))

        # For stability-based extraction: remember the top v from the PREVIOUS check
        self.register_buffer('_prev_v', torch.zeros(d_in))      # (d_in,)
        self._has_prev = False  # whether _prev_v has been set yet

    def n_frozen(self):
        return self._frozen_u.shape[0]

    def check_and_extract_if_stable(self, cos_threshold=0.99):
        """
        检查当前 top 奇异方向是否已稳定（相对上次检查的方向变化 < cos_threshold）。
        如果稳定，则冻结该方向并从 W 减去。
        返回 True 表示本次提取了一个方向。
        """
        with torch.no_grad():
            U, S, Vh = torch.linalg.svd(self.W.float(), full_matrices=False)
            sigma = S[0].item()
            if sigma < 1e-10:
                return False

            v_now = Vh[0, :].to(self.W.dtype)  # (d_in,)

            # 对照上次的方向
            extracted = False
            if self._has_prev:
                cos_sim = torch.dot(v_now, self._prev_v).item()
                cos_sim = abs(cos_sim)  # 方向的正负不重要
                if cos_sim >= cos_threshold:
                    # 方向稳定 → 冻结
                    u = U[:, 0].to(self.W.dtype)
                    u_buf = torch.cat([self._frozen_u, u.unsqueeze(0)], dim=0)
                    v_buf = torch.cat([self._frozen_v, v_now.unsqueeze(0)], dim=0)
                    s_buf = torch.cat([self._frozen_sigma,
                                       torch.tensor([sigma], device=self.W.device)])
                    self._frozen_u = u_buf
                    self._frozen_v = v_buf
                    self._frozen_sigma = s_buf
                    self.W.data -= sigma * torch.outer(u, v_now)
                    self._n_extracted += 1
                    extracted = True

            # 无论是否提取，都记录当前方向供下次对比
            self._prev_v = v_now.clone()
            self._has_prev = True

            return extracted

    def forward(self, x):
        """
        x: (B, T, d_in)
        Returns: (B, T, d_out)
        """
        B, T, _ = x.shape

        if self.n_frozen() == 0:
            return F.linear(x, self.W)

        # --- Step 1: 一次性移除所有 frozen v 方向上的投影 ---
        # V: (k, d_in), x: (B, T, d_in)
        # x_orth = x - x @ V^T @ V  (投影到 V 的行空间上)
        V = self._frozen_v                                   # (k, d_in)
        s_all = torch.matmul(x, V.T)                         # (B, T, k) — h 在各 v 上的投影标量
        proj = torch.matmul(s_all, V)                        # (B, T, d_in) — 重构 = 投影分量
        x_orth = x - proj                                    # (B, T, d_in) — 正交于所有 frozen v

        # --- Step 2: frozen 分支输出（用原始 x 的投影标量） ---
        U = self._frozen_u                                   # (k, d_out)
        sigma = self._frozen_sigma.unsqueeze(0).unsqueeze(0)  # (1, 1, k)
        frozen_out = torch.matmul(s_all * sigma, U)           # (B, T, d_out)

        # --- Step 3: 残差矩阵输出 ---
        residual_out = F.linear(x_orth, self.W)              # (B, T, d_out)

        return residual_out + frozen_out

    def forward_no_frozen(self, x):
        """纯残差 forward，不经过 frozen 分支（用于 baseline 对比）。"""
        return F.linear(x, self.W)


# ==============================================================================
# Transformer 组件（使用 DecomposableLinear）
# ==============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


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
    """Multi-head causal attention，所有线性矩阵可迭代 SVD 拆分。"""

    def __init__(self, d_model, n_heads, n_kv_heads, max_seq_len):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.q_dim = n_heads * self.head_dim        # = d_model
        self.kv_dim = n_kv_heads * self.head_dim
        self.scale = math.sqrt(self.head_dim)

        # 四个线性矩阵均使用 DecomposableLinear
        self.Wq = DecomposableLinear(d_model, self.q_dim)
        self.Wk = DecomposableLinear(d_model, self.kv_dim)
        self.Wv = DecomposableLinear(d_model, self.kv_dim)
        self.Wo = DecomposableLinear(self.q_dim, d_model)

        freqs = precompute_freqs_cis(self.head_dim, max_seq_len)
        self.register_buffer('freqs_cis', freqs)
        mask = torch.triu(torch.full((max_seq_len, max_seq_len), float('-inf')), diagonal=1)
        self.register_buffer('causal_mask', mask)

    def extract_stable(self, cos_threshold=0.99):
        """对四个矩阵各检查一次，稳定的才拆。返回提取的总数。"""
        count = 0
        for m in [self.Wq, self.Wk, self.Wv, self.Wo]:
            if m.check_and_extract_if_stable(cos_threshold):
                count += 1
        return count

    def forward(self, h):
        B, T, D = h.shape
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
        attn_out = torch.matmul(F.softmax(attn_scores, dim=-1), v)
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, T, self.q_dim)
        return self.Wo(attn_out)


class MLP(nn.Module):
    """SwiGLU MLP，全部线性矩阵可迭代 SVD 拆分。"""

    def __init__(self, d_model, intermediate_size):
        super().__init__()
        self.W1 = DecomposableLinear(d_model, intermediate_size)
        self.W2 = DecomposableLinear(intermediate_size, d_model)
        self.W3 = DecomposableLinear(d_model, intermediate_size)

    def extract_stable(self, cos_threshold=0.99):
        count = 0
        for m in [self.W1, self.W2, self.W3]:
            if m.check_and_extract_if_stable(cos_threshold):
                count += 1
        return count

    def forward(self, x):
        return self.W2(F.silu(self.W1(x)) * self.W3(x))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, intermediate_size, max_seq_len):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = CausalAttention(d_model, n_heads, n_kv_heads, max_seq_len)
        self.mlp_norm = RMSNorm(d_model)
        self.mlp = MLP(d_model, intermediate_size)

    def extract_stable(self, cos_threshold=0.99):
        return self.attn.extract_stable(cos_threshold) + self.mlp.extract_stable(cos_threshold)

    def forward(self, h):
        h = h + self.attn(self.attn_norm(h))
        h = h + self.mlp(self.mlp_norm(h))
        return h


class SmallLM(nn.Module):
    def __init__(self, vocab_size, d_model=32, n_layers=1, n_heads=2, n_kv_heads=1,
                 intermediate_size=96, max_seq_len=16):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embed.weight, std=1.0 / math.sqrt(d_model))
        self.norm = RMSNorm(d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, n_kv_heads, intermediate_size, max_seq_len)
            for _ in range(n_layers)
        ])

    def extract_stable(self, cos_threshold=0.99):
        """对模型中所有 DecomposableLinear 检查稳定方向，稳定的才拆。"""
        total = 0
        for layer in self.layers:
            total += layer.extract_stable(cos_threshold)
        return total

    def forward(self, input_ids):
        h = self.embed(input_ids)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        return h @ self.embed.weight.T

    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    def frozen_count(self):
        """计算所有矩阵中已冻结的 rank-1 组件总数。"""
        total = 0
        for layer in self.layers:
            for m in [layer.attn.Wq, layer.attn.Wk, layer.attn.Wv, layer.attn.Wo,
                      layer.mlp.W1, layer.mlp.W2, layer.mlp.W3]:
                total += m.n_frozen()
        return total


# ==============================================================================
# 谱分析
# ==============================================================================

def svd_analysis(model, label=""):
    """收集所有矩阵的 σ₁ 和有效秩。"""
    results = {}
    layer = model.layers[0]
    matrices = [
        ('Wq', layer.attn.Wq), ('Wk', layer.attn.Wk),
        ('Wv', layer.attn.Wv), ('Wo', layer.attn.Wo),
        ('W1', layer.mlp.W1), ('W2', layer.mlp.W2),
        ('W3', layer.mlp.W3),
    ]
    for name, m in matrices:
        # 残差矩阵的 SVD
        W = m.W.data.float()
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        S = S.cpu().numpy()
        total = np.sum(S ** 2)
        effrank = np.sum(S ** 2) ** 2 / max(np.sum(S ** 4), 1e-30)
        results[f"{label}_{name}_sigma1"] = float(S[0])
        results[f"{label}_{name}_effrank"] = float(min(effrank, W.shape[0]))
        results[f"{label}_{name}_frozen"] = m.n_frozen()

    # 嵌入谱
    E = model.embed.weight.data.float()
    U, S, Vh = torch.linalg.svd(E, full_matrices=False)
    S = S.cpu().numpy()
    total = np.sum(S ** 2)
    results[f"{label}_E_sigma1"] = float(S[0])
    results[f"{label}_E_top5ratio"] = [round(float(S[i] ** 2 / total), 4) for i in range(min(5, len(S)))]

    return results


# ==============================================================================
# 训练
# ==============================================================================

def train(args):
    device = torch.device('mps' if torch.backends.mps.is_available() else
                          'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # --- 数据 ---
    data = generate_synthetic_patterns(n_patterns=args.n_patterns, seed=42)
    patterns = data['patterns']
    vocab_size = data['vocab_size']
    K_ids = set(data['K_ids'])
    seq_len = data['seq_len']

    train_dataset = SyntheticPatternDataset(patterns, n_samples=5000)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    test_data = generate_synthetic_patterns(n_patterns=args.n_patterns, seed=12345)
    test_loader = DataLoader(
        SyntheticPatternDataset(test_data['patterns'], n_samples=args.batch_size * 5),
        batch_size=args.batch_size, shuffle=False, drop_last=True
    )

    print(f"Data: {data['n_patterns']} patterns × {seq_len}, 10 K + 490 R")

    # --- 模型 ---
    d = args.d_model
    n_heads = 4 if d >= 64 else 2
    n_kv_heads = 2 if d >= 64 else 1
    model = SmallLM(
        vocab_size=vocab_size, d_model=d, n_layers=1,
        n_heads=n_heads, n_kv_heads=n_kv_heads,
        intermediate_size=d * 3, max_seq_len=seq_len + 2,
    ).to(device)
    print(f"Model: {model.param_count():,} params (d={d})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                                   weight_decay=args.weight_decay)

    # --- 输出 ---
    run_name = f"{args.mode}_extract{args.extract_every}"
    out_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)
    metrics_path = os.path.join(out_dir, 'metrics.jsonl')

    # --- 训练 ---
    model.train()
    global_step = 0
    step_start = time.time()
    total_start = step_start

    print(f"\nTraining: {args.max_steps} steps, extract every {args.extract_every}")

    for epoch in range(100):
        for input_ids, target_ids in train_loader:
            if global_step >= args.max_steps:
                break

            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            logits = model(input_ids)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), target_ids.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            global_step += 1

            # --- 每 N 步检查稳定方向，稳定则拆 ---
            if args.mode == 'svd' and args.extract_every > 0 and global_step % args.extract_every == 0:
                n = model.extract_stable(args.cos_threshold)
                if n > 0:
                    frozen_total = model.frozen_count()
                    print(f"\n  == step {global_step}: {n} stable directions extracted, "
                          f"total frozen = {frozen_total} ==")

            if global_step % args.log_every == 0:
                # --- eval ---
                model.eval()
                K_loss_total = R_loss_total = 0.0
                K_count = R_count = 0
                with torch.no_grad():
                    for e_ids, e_tgt in test_loader:
                        e_ids = e_ids.to(device); e_tgt = e_tgt.to(device)
                        e_logits = model(e_ids)
                        e_loss = F.cross_entropy(
                            e_logits.reshape(-1, vocab_size),
                            e_tgt.reshape(-1), reduction='none'
                        ).reshape(args.batch_size, seq_len)
                        for bi in range(e_ids.shape[0]):
                            for t in range(seq_len):
                                if e_tgt[bi, t].item() in K_ids:
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
                frozen = model.frozen_count()

                print(f"step {global_step:>6d}/{args.max_steps}  "
                      f"loss={loss.item():.4f}  test_K={K_loss:.4f}  test_R={R_loss:.4f}  "
                      f"frozen={frozen:>3d}  tok/s={tok_per_sec:,.0f}")

                with open(metrics_path, 'a') as f:
                    f.write(json.dumps({
                        'step': global_step, 'loss': round(loss.item(), 6),
                        'K_loss': round(K_loss, 6), 'R_loss': round(R_loss, 6),
                        'frozen': frozen,
                    }) + '\n')

                step_start = time.time()

        if global_step >= args.max_steps:
            break

    # --- 终局分析 ---
    model.eval()
    results = svd_analysis(model, label=args.mode)
    results['final_step'] = global_step
    results['total_frozen'] = model.frozen_count()

    # 终局 test loss
    K_loss_total = R_loss_total = K_count = R_count = 0
    with torch.no_grad():
        for e_ids, e_tgt in test_loader:
            e_ids = e_ids.to(device); e_tgt = e_tgt.to(device)
            e_logits = model(e_ids)
            e_loss = F.cross_entropy(
                e_logits.reshape(-1, vocab_size),
                e_tgt.reshape(-1), reduction='none'
            ).reshape(args.batch_size, seq_len)
            for bi in range(e_ids.shape[0]):
                for t in range(seq_len):
                    if e_tgt[bi, t].item() in K_ids:
                        K_loss_total += e_loss[bi, t].item(); K_count += 1
                    else:
                        R_loss_total += e_loss[bi, t].item(); R_count += 1
    results['final_K_loss'] = round(K_loss_total / max(K_count, 1), 6)
    results['final_R_loss'] = round(R_loss_total / max(R_count, 1), 6)

    result_path = os.path.join(out_dir, 'final_analysis.json')
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Step {global_step}: total_frozen={results['total_frozen']}")
    print(f"  test_K={results['final_K_loss']:.6f}  test_R={results['final_R_loss']:.6f}")
    for name in ['Wq', 'Wk', 'Wv', 'Wo', 'W1', 'W2', 'W3']:
        s1 = results.get(f"{args.mode}_{name}_sigma1", 0)
        er = results.get(f"{args.mode}_{name}_effrank", 0)
        fr = results.get(f"{args.mode}_{name}_frozen", 0)
        print(f"  {name}: σ₁={s1:.4f}  effrank={er:.1f}  frozen={fr}")
    print(f"{'='*60}")


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='svd', choices=['baseline', 'svd'])
    parser.add_argument('--d_model', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_steps', type=int, default=2000)
    parser.add_argument('--log_every', type=int, default=100)
    parser.add_argument('--extract_every', type=int, default=25,
                        help='SVD mode: check stability every N steps, extract only if direction stable')
    parser.add_argument('--cos_threshold', type=float, default=0.99,
                        help='Cosine similarity threshold for direction stability')
    parser.add_argument('--n_patterns', type=int, default=200)
    parser.add_argument('--output_dir', type=str,
                        default='/Users/bytedance/kv_cache/fdong_embedding_dim/outputs/iterative_svd')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)
