#!/usr/bin/env python3
"""
Small LM (100M) with CRS Subspace Split, trained on real English text.

CRS (Common-Residual Split): Each attention layer's Q/K/V projections are
decomposed into a "common" branch (1D vector, processes the projection of
the hidden state onto the vocabulary centroid direction) and a "residual"
branch (full matrix, processes everything orthogonal to the centroid).
This prevents the common (high-frequency token) direction from dominating
the spectral structure of the attention parameter matrices.

Usage:
    # Build reduced vocab first:
    python3 workbuddy_scripts/small_lm_crs.py --build_vocab

    # Train CRS model:
    python3 workbuddy_scripts/small_lm_crs.py --mode crs --alpha 0.3

    # Train baseline model:
    python3 workbuddy_scripts/small_lm_crs.py --mode baseline
"""

import argparse, json, math, os, time, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from transformers import AutoTokenizer


# ==============================================================================
# TeeLogger: duplicate print output to both console and a log file
# ==============================================================================

class TeeLogger:
    """Duplicates writes to both the original stream and a log file."""
    def __init__(self, log_path, stream):
        self.log_file = open(log_path, 'a', buffering=1)  # line-buffered
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.log_file.write(data)

    def flush(self):
        self.stream.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()


# ==============================================================================
# Reduced Vocabulary Builder
# ==============================================================================

def build_reduced_vocab(qwen_tokenizer_dir, output_dir, max_vocab_size=32768):
    """Filter Qwen3 vocab to English-only tokens, build remapping."""
    tokenizer = AutoTokenizer.from_pretrained(qwen_tokenizer_dir, trust_remote_code=True)
    vocab = tokenizer.get_vocab()
    print(f"Original vocab size: {len(vocab)}")

    # Keep special tokens always
    special_ids = set()
    for name in ['<|endoftext|>', '<|im_start|>', '<|im_end|>', '<unk>', '<s>', '</s>', '<pad>']:
        tok_id = tokenizer.convert_tokens_to_ids(name)
        if tok_id != tokenizer.unk_token_id:
            special_ids.add(tok_id)

    # Score each token: prefer ASCII + short + common in English
    id_to_token = {v: k for k, v in vocab.items()}

    def is_english_token(token_str):
        """Keep tokens that contain mostly ASCII printable chars."""
        if not token_str:
            return False
        ascii_count = sum(1 for c in token_str if 32 <= ord(c) < 127)
        return ascii_count / max(len(token_str), 1) > 0.7

    def token_score(token_id):
        if token_id in special_ids:
            return (True, float('inf'))
        tok = id_to_token.get(token_id, '')
        if is_english_token(tok):
            return (True, len(tok))  # shorter is better (more common subword)
        return (False, len(tok))

    # Score all tokens
    valid_tokens = []
    for tid in range(len(vocab)):
        keep, score = token_score(tid)
        if keep or tid in special_ids:
            valid_tokens.append((tid, score if keep else float('inf')))

    # Sort by score (ascending = shorter/better first), then take top max_vocab_size
    valid_tokens.sort(key=lambda x: (x[0] in special_ids, x[1]), reverse=False)
    # Special tokens first, then shortest valid tokens
    special_first = [(tid, s) for tid, s in valid_tokens if tid in special_ids]
    normal_sorted = sorted([(tid, s) for tid, s in valid_tokens if tid not in special_ids],
                           key=lambda x: x[1])
    selected = special_first + normal_sorted[:max_vocab_size - len(special_first)]

    # Build mapping
    qwen2new = {}  # qwen token ID → our token ID
    new2qwen = []  # our token ID → qwen token ID
    for new_id, (qwen_id, _) in enumerate(selected):
        qwen2new[qwen_id] = new_id
        new2qwen.append(qwen_id)

    os.makedirs(output_dir, exist_ok=True)
    torch.save({'qwen2new': qwen2new, 'new2qwen': new2qwen},
               os.path.join(output_dir, 'vocab_mapping.pt'))

    print(f"Reduced vocab size: {len(new2qwen)}")
    print(f"Special tokens kept: {len(special_first)}")
    print(f"English tokens kept: {len(normal_sorted[:max_vocab_size - len(special_first)])}")

    # Show a few examples
    print("\nSample kept tokens:")
    for i in range(min(20, len(new2qwen))):
        qwen_id = new2qwen[i]
        tok = id_to_token.get(qwen_id, '<UNK>')
        print(f"  new_id={i:5d} ← qwen_id={qwen_id:6d}  '{tok}'")

    return qwen2new, new2qwen


# ==============================================================================
# Dataset
# ==============================================================================

class EnglishTextDataset(Dataset):
    """Loads dclm text files, tokenizes with Qwen + remaps to reduced vocab."""

    def __init__(self, data_dir, max_seq_len, tokenizer, vocab_mapping, qwen2new):
        super().__init__()
        self.data_dir = data_dir
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.qwen2new = qwen2new
        self.unk_id = qwen2new.get(tokenizer.unk_token_id, 0)

        # Collect all text files
        self.files = []
        for root, _, filenames in os.walk(data_dir):
            for fn in filenames:
                if fn.endswith('.txt'):
                    self.files.append(os.path.join(root, fn))
        self.files = sorted(self.files)

        # Each file contains lines of ~1024 tokens. We chunk each line into
        # segments of max_seq_len to maximize data utilization.
        self._load_file(0)
        self.total_segments = len(self.files) * self.segments_per_file

    def _load_file(self, file_idx):
        self.cur_file_idx = file_idx
        with open(self.files[file_idx], 'r', encoding='utf-8') as f:
            self.file_lines = [line.strip() for line in f if line.strip()]
        self.lines_per_file = len(self.file_lines)

        # Pre-compute segments per line: each ~1024-token line → floor(1024/seq_len) chunks
        self.segments_per_file = self.lines_per_file  # placeholder; recomputed on first access

    def __len__(self):
        # Approximate; actual length depends on tokenization
        return len(self.files) * self.lines_per_file * 4

    def _remap(self, token_ids):
        """Remap Qwen token IDs to reduced vocab IDs, UNK for OOV."""
        result = []
        for tid in token_ids:
            result.append(self.qwen2new.get(tid, self.unk_id))
        return result

    def __getitem__(self, index):
        # Map flat index to (file, line, segment)
        # Approximate: each line produces ~4 segments
        segments_per_line = 4
        segments_per_file = self.lines_per_file * segments_per_line
        total_segments = len(self.files) * segments_per_file

        file_idx = index // segments_per_file
        if file_idx != self.cur_file_idx:
            self._load_file(min(file_idx, len(self.files) - 1))

        offset = index % segments_per_file
        line_idx = offset // segments_per_line
        seg_idx = offset % segments_per_line

        if line_idx >= len(self.file_lines):
            line_idx = len(self.file_lines) - 1

        text = self.file_lines[line_idx]
        try:
            text = json.loads(text)
        except json.JSONDecodeError:
            pass

        # Tokenize full line (truncate to ~4*seq_len to save time)
        token_ids = self.tokenizer(
            text, truncation=True,
            max_length=self.max_seq_len * segments_per_line + 1,
            return_tensors='pt'
        ).input_ids[0].tolist()

        # Slice to the target segment
        start = seg_idx * self.max_seq_len
        end = start + self.max_seq_len + 1
        if start >= len(token_ids):
            start = 0  # fallback
            end = min(self.max_seq_len + 1, len(token_ids))
        segment = token_ids[start:end]

        remapped = self._remap(segment)

        # Pad if needed
        if len(remapped) < self.max_seq_len + 1:
            remapped = remapped + [0] * (self.max_seq_len + 1 - len(remapped))

        input_ids = torch.tensor(remapped[:-1], dtype=torch.long)
        target_ids = torch.tensor(remapped[1:], dtype=torch.long)
        return input_ids, target_ids


# ==============================================================================
# Model Components
# ==============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


def precompute_freqs_cis(dim, max_seq_len, theta=10000.0):
    """RoPE frequency precomputation."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(x, freqs_cis):
    """Apply RoPE to query/key tensors. x: (batch, heads, seq, head_dim)."""
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs = freqs_cis[:x.shape[2]].unsqueeze(0).unsqueeze(0)
    x_rotated = x_complex * freqs
    return torch.view_as_real(x_rotated).flatten(-2).type_as(x)


class CausalAttention(nn.Module):
    """Multi-head causal attention, optionally with CRS split on Q/K/V.

    CRS mode:
      h_common[t] = cummean(h[:, :t+1])         ← per-position running mean of hidden states
      q_c = (h_common · g_q) * g_q              ← rank-1 projection along g_q
      h_common is DETACHED — the cumulative-mean operation carries no gradient.
      Common branch sees only the "what's typical in this sequence so far" component.
      Residual branch (full W matrices) sees the full hidden state with gradient.
    """

    def __init__(self, d_model, n_heads, n_kv_heads, max_seq_len, use_crs=False, alpha=0.3):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.use_crs = use_crs
        self.alpha = alpha
        self.scale = math.sqrt(self.head_dim)

        if use_crs:
            # Common branch: g vectors are d_model-dim for dot(h_common, g).
            # Q uses full dim; K/V use first kv_dim entries after dot.
            self.q_dim = n_heads * self.head_dim          # = d_model
            self.kv_dim = n_kv_heads * self.head_dim
            self.g_q = nn.Parameter(torch.zeros(d_model))
            self.g_k = nn.Parameter(torch.zeros(d_model))
            self.g_v = nn.Parameter(torch.zeros(d_model))
            # Residual branch: full matrices
            self.Wq_r = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
            self.Wk_r = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
            self.Wv_r = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        else:
            self.Wq = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
            self.Wk = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
            self.Wv = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)

        self.Wo = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

        # RoPE
        freqs_cis = precompute_freqs_cis(self.head_dim, max_seq_len)
        self.register_buffer('freqs_cis', freqs_cis)

        # Causal mask
        mask = torch.full((max_seq_len, max_seq_len), float('-inf'))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer('causal_mask', mask)

    def forward(self, h):
        B, T, D = h.shape

        if self.use_crs:
            # --- h_common: per-position cumulative mean of hidden states, DETACH ---
            # At position t: h_common[t] = mean(h[:, :t+1, :], dim=1)
            # Detach ensures gradient does NOT flow back from common branch
            # through the cumulative-mean computation.
            denom = torch.arange(1, T + 1, device=h.device, dtype=h.dtype).view(1, -1, 1)
            h_cum = torch.cumsum(h, dim=1)                          # (B, T, d)
            h_common = (h_cum / denom).detach()                     # (B, T, d), no grad

            # Common branch: rank-1 projection of h_common onto g direction
            s_q = (h_common * self.g_q).sum(dim=-1, keepdim=True)   # (B, T, 1)
            q_c = (s_q * self.g_q).reshape(B, T, self.n_heads, self.head_dim)
            s_k = (h_common * self.g_k).sum(dim=-1, keepdim=True)
            k_c = (s_k * self.g_k[:self.kv_dim]).reshape(B, T, self.n_kv_heads, self.head_dim)
            s_v = (h_common * self.g_v).sum(dim=-1, keepdim=True)
            v_c = (s_v * self.g_v[:self.kv_dim]).reshape(B, T, self.n_kv_heads, self.head_dim)

            # Residual branch: full matrices on h (with gradient)
            q_r = self.Wq_r(h).reshape(B, T, self.n_heads, self.head_dim)
            k_r = self.Wk_r(h).reshape(B, T, self.n_kv_heads, self.head_dim)
            v_r = self.Wv_r(h).reshape(B, T, self.n_kv_heads, self.head_dim)

            q = self.alpha * q_c + q_r
            k = self.alpha * k_c + k_r
            v = self.alpha * v_c + v_r
        else:
            q = self.Wq(h).reshape(B, T, self.n_heads, self.head_dim)
            k = self.Wk(h).reshape(B, T, self.n_kv_heads, self.head_dim)
            v = self.Wv(h).reshape(B, T, self.n_kv_heads, self.head_dim)

        # RoPE
        q = apply_rotary_emb(q.permute(0, 2, 1, 3), self.freqs_cis)
        k = apply_rotary_emb(k.permute(0, 2, 1, 3), self.freqs_cis)
        v = v.permute(0, 2, 1, 3)  # (B, n_kv_heads, T, head_dim)

        # GQA: repeat KV heads to match Q heads
        if self.n_heads > self.n_kv_heads:
            r = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(r, dim=1)
            v = v.repeat_interleave(r, dim=1)

        # Attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn_scores = attn_scores + self.causal_mask[:T, :T]
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_weights, v)

        # Merge heads
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, T, self.n_heads * self.head_dim)
        return self.Wo(attn_out)


class MLP(nn.Module):
    """SwiGLU MLP, optionally with CRS split.

    CRS mode: same decomposition as attention —
      h_common = cummean(x, dim=1).detach()
      mlp_c = (h_common · g_mlp) * g_mlp     ← rank-1 common branch
      mlp_r = SwiGLU(x)                       ← full residual branch
      output = α * mlp_c + mlp_r
    """

    def __init__(self, d_model, intermediate_size, use_crs=False, alpha=0.3):
        super().__init__()
        self.use_crs = use_crs
        self.alpha = alpha
        self.W1 = nn.Linear(d_model, intermediate_size, bias=False)
        self.W2 = nn.Linear(intermediate_size, d_model, bias=False)
        self.W3 = nn.Linear(d_model, intermediate_size, bias=False)  # gate
        if use_crs:
            self.g_mlp = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        if self.use_crs:
            B, T, D = x.shape
            denom = torch.arange(1, T + 1, device=x.device, dtype=x.dtype).view(1, -1, 1)
            h_cum = torch.cumsum(x, dim=1)
            h_common = (h_cum / denom).detach()

            # Common branch: rank-1
            s = (h_common * self.g_mlp).sum(dim=-1, keepdim=True)  # (B, T, 1)
            mlp_c = s * self.g_mlp                                  # (B, T, d)

            # Residual branch: full SwiGLU
            mlp_r = self.W2(F.silu(self.W1(x)) * self.W3(x))

            return self.alpha * mlp_c + mlp_r
        else:
            return self.W2(F.silu(self.W1(x)) * self.W3(x))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, intermediate_size, max_seq_len,
                 use_crs=False, alpha=0.3):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = CausalAttention(d_model, n_heads, n_kv_heads, max_seq_len,
                                     use_crs=use_crs, alpha=alpha)
        self.mlp_norm = RMSNorm(d_model)
        self.mlp = MLP(d_model, intermediate_size, use_crs=use_crs, alpha=alpha)

    def forward(self, h):
        h = h + self.attn(self.attn_norm(h))
        h = h + self.mlp(self.mlp_norm(h))
        return h


class SmallLM(nn.Module):
    """Qwen-style small LM with optional CRS split."""

    def __init__(self, vocab_size, d_model=1024, n_layers=6, n_heads=8, n_kv_heads=4,
                 intermediate_size=3072, max_seq_len=512, use_crs=False, alpha=0.3):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.use_crs = use_crs

        self.embed = nn.Embedding(vocab_size, d_model)
        # Tied embedding → initial logit variance = d_model × Var(embed).
        # Default N(0,1) init gives Var≈1024 → initial loss ~140 (expected but noisy).
        # Shrink to σ=1/√d so initial loss ≈ ln(vocab_size) ≈ 10.
        nn.init.normal_(self.embed.weight, std=1.0 / math.sqrt(d_model))
        self.norm = RMSNorm(d_model)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, n_kv_heads, intermediate_size, max_seq_len,
                             use_crs=use_crs, alpha=alpha)
            for _ in range(n_layers)
        ])

        # Tied embedding: output logits = h @ embed^T
        # Use the embedding weight for output (no separate lm_head)

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

def save_checkpoint(path, model, optimizer, global_step, total_tokens, accum_count,
                     scaler=None):
    """Save full training state for exact resume."""
    ckpt = {
        'step': global_step,
        'total_tokens': total_tokens,
        'accum_count': accum_count,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'rng_state': torch.get_rng_state(),
    }
    if scaler is not None:
        ckpt['scaler_state_dict'] = scaler.state_dict() if hasattr(scaler, 'state_dict') else scaler
    torch.save(ckpt, path)


def load_checkpoint(path, model, optimizer, device, scaler=None):
    """Load full training state. Returns (global_step, total_tokens, accum_count)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    torch.set_rng_state(ckpt.get('rng_state', torch.get_rng_state()))
    if scaler is not None and 'scaler_state_dict' in ckpt:
        if hasattr(scaler, 'load_state_dict'):
            scaler.load_state_dict(ckpt['scaler_state_dict'])
    return ckpt['step'], ckpt.get('total_tokens', 0), ckpt.get('accum_count', 0)


def train(args):
    device = torch.device('mps' if torch.backends.mps.is_available() else
                          'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Resolve run name and output directory ---
    if args.run_name is None:
        if args.mode == 'crs':
            args.run_name = f"crs_alpha{args.alpha}".replace('.', '_')
        else:
            args.run_name = 'baseline'
    run_dir = os.path.join(args.output_dir, args.run_name)
    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"Run name: {args.run_name}")
    print(f"Output dir: {run_dir}")
    print(f"  Batch size: {args.batch_size}, Seq len: {args.max_seq_len}")
    print(f"  Gradient accumulation: {args.grad_accum}, "
          f"effective batch: {args.batch_size * args.grad_accum}")

    # --- Metrics log file ---
    metrics_path = os.path.join(run_dir, 'metrics.jsonl')

    # --- Redirect stdout/stderr to log file (in addition to console) ---
    console_log = os.path.join(run_dir, 'console.log')
    _orig_stdout = sys.stdout
    _orig_stderr = sys.stderr
    sys.stdout = TeeLogger(console_log, _orig_stdout)
    sys.stderr = TeeLogger(console_log, _orig_stderr)

    # --- Load tokenizer ---
    qwen_dir = args.qwen_dir
    tokenizer = AutoTokenizer.from_pretrained(qwen_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Load vocab mapping ---
    mapping = torch.load(args.vocab_mapping_path, map_location='cpu')
    qwen2new = {int(k): int(v) for k, v in mapping['qwen2new'].items()}
    new2qwen = [int(x) for x in mapping['new2qwen']]
    vocab_size = len(new2qwen)
    print(f"Reduced vocab size: {vocab_size}")

    # --- Dataset ---
    dataset = EnglishTextDataset(
        data_dir=args.data_dir,
        max_seq_len=args.max_seq_len,
        tokenizer=tokenizer,
        vocab_mapping=mapping,
        qwen2new=qwen2new,
    )
    print(f"Dataset: {len(dataset.files)} files, ~{len(dataset)} samples")

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=False,
        drop_last=True
    )

    # --- Model ---
    use_crs = (args.mode == 'crs')
    model = SmallLM(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        intermediate_size=args.intermediate_size,
        max_seq_len=args.max_seq_len,
        use_crs=use_crs,
        alpha=args.alpha,
    ).to(device)
    print(f"Model params: {model.param_count():,}")
    print(f"  CRS mode: {use_crs}, alpha: {args.alpha}")

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                                   weight_decay=args.weight_decay)

    # --- Resume or fresh start ---
    global_step = 0
    total_tokens = 0
    accum_count = 0

    if args.resume_from:
        print(f"\n=== Resuming from checkpoint: {args.resume_from} ===")
        global_step, total_tokens, accum_count = load_checkpoint(
            args.resume_from, model, optimizer, device)
        print(f"  Resumed at step {global_step}, total_tokens={total_tokens:,}")
        if global_step >= args.max_steps:
            print(f"  Already at or past max_steps ({args.max_steps}). Nothing to do.")
            return
    else:
        print(f"\n=== Starting fresh training ===")

    # --- Training loop ---
    model.train()
    running_loss = 0.0
    log_steps = 0  # micro-steps accumulated for loss averaging

    print(f"\nTraining config: {args.max_steps} total steps "
          f"(will do {args.max_steps - global_step} more)")
    print(f"  Micro batch: {args.batch_size} × {args.max_seq_len} = "
          f"{args.batch_size * args.max_seq_len:,} tokens")
    print(f"  Grad accum: {args.grad_accum}")
    print(f"  Effective batch: {args.batch_size * args.grad_accum} × {args.max_seq_len} = "
          f"{args.batch_size * args.grad_accum * args.max_seq_len:,} tokens")
    print(f"  Checkpoints every {args.save_every} steps → {ckpt_dir}/")
    print(f"  Metrics → {metrics_path}")

    if accum_count == 0:
        optimizer.zero_grad()
    step_start = time.time()
    total_start = step_start

    for epoch in range(100):
        for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
            if global_step >= args.max_steps:
                break

            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            logits = model(input_ids)
            loss_raw = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                target_ids.reshape(-1),
                ignore_index=0,
            )
            loss = loss_raw / args.grad_accum
            loss.backward()

            accum_count += 1
            running_loss += loss_raw.item()
            total_tokens += input_ids.numel()
            log_steps += 1

            if accum_count >= args.grad_accum:
                # Record grad norm BEFORE clipping
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5

                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                accum_count = 0
                global_step += 1

                if global_step % args.log_every == 0:
                    avg_loss = running_loss / log_steps
                    ppl = math.exp(min(avg_loss, 20.0))
                    elapsed = time.time() - step_start
                    tok_per_sec = (args.batch_size * args.max_seq_len * args.grad_accum
                                   * args.log_every) / max(elapsed, 1)

                    print(f"step {global_step:6d}/{args.max_steps}  "
                          f"loss={avg_loss:.4f}  ppl={ppl:.1f}  "
                          f"grad_norm={total_norm:.2f}  "
                          f"tok/s={tok_per_sec:,.0f}  "
                          f"tokens={total_tokens:,}")

                    # Write to metrics file
                    with open(metrics_path, 'a') as f:
                        f.write(json.dumps({
                            'step': global_step,
                            'loss': round(avg_loss, 6),
                            'ppl': round(ppl, 2),
                            'grad_norm': round(total_norm, 4),
                            'lr': args.lr,
                            'tokens': total_tokens,
                            'elapsed_sec': round(time.time() - total_start, 1),
                        }) + '\n')

                    running_loss = 0.0
                    log_steps = 0
                    step_start = time.time()

                if global_step % args.save_every == 0:
                    save_path = os.path.join(ckpt_dir, f'step_{global_step}.pt')
                    save_checkpoint(save_path, model, optimizer, global_step,
                                    total_tokens, accum_count)
                    print(f"  → saved checkpoint: {save_path}")

        if global_step >= args.max_steps:
            break

    # --- Final save ---
    final_ckpt = os.path.join(ckpt_dir, f'step_{global_step}_final.pt')
    save_checkpoint(final_ckpt, model, optimizer, global_step, total_tokens, accum_count)
    print(f"\nTraining done ({global_step} steps). Final checkpoint: {final_ckpt}")
    print(f"  Metrics saved to: {metrics_path}")
    print(f"  Console log saved to: {console_log}")

    # Cleanup TeeLogger
    sys.stdout.close()
    sys.stderr.close()
    sys.stdout = _orig_stdout
    sys.stderr = _orig_stderr


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--build_vocab', action='store_true',
                        help='Build reduced vocabulary mapping')

    # Paths
    parser.add_argument('--qwen_dir', type=str,
                        default='/Users/bytedance/kv_cache/fdong/Qwen3-0.6B')
    parser.add_argument('--data_dir', type=str,
                        default=os.path.expanduser('~/Desktop/dclm'))
    parser.add_argument('--output_dir', type=str,
                        default='/Users/bytedance/kv_cache/fdong_embedding_dim/outputs/small_lm_crs')
    parser.add_argument('--vocab_mapping_path', type=str,
                        default='/Users/bytedance/kv_cache/fdong_embedding_dim/outputs/small_lm_crs/vocab_mapping.pt')

    # Model
    parser.add_argument('--mode', type=str, default='baseline',
                        choices=['baseline', 'crs'])
    parser.add_argument('--d_model', type=int, default=1024)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_kv_heads', type=int, default=4)
    parser.add_argument('--intermediate_size', type=int, default=3072)
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--alpha', type=float, default=0.3,
                        help='Common branch suppression coefficient (CRS only)')

    # Training
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--grad_accum', type=int, default=4,
                        help='Gradient accumulation steps (effective bs = batch_size × grad_accum)')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--max_steps', type=int, default=5000)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--log_every', type=int, default=25)
    parser.add_argument('--save_every', type=int, default=200)

    # Checkpointing & resume
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to checkpoint.pt to resume training from (full state: model+optimizer+step)')
    parser.add_argument('--run_name', type=str, default=None,
                        help='Run name for subdirectory. Auto-generated from mode/alpha if not set.')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.build_vocab:
        build_reduced_vocab(args.qwen_dir, args.output_dir)
    else:
        if not os.path.exists(args.vocab_mapping_path):
            print("Vocab mapping not found. Run with --build_vocab first.")
            sys.exit(1)
        train(args)
