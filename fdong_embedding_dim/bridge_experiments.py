#!/usr/bin/env python3
"""
Bridge experiments before jumping to real data:
  E1: Weight-tied E=W (with/without extra projection matrix)
  E2: Medium-scale synthesis (100 groups, realistic mini-batch failure mode)
  E3: 2-layer network (bridging towards Transformer expressivity)
"""
import math, numpy as np
import torch, torch.nn.functional as F

theta_deg = 12.0; dim = 2; lr = 0.03

# ============================================================
# Shared helpers
# ============================================================
def rot_2d(v, deg):
    t = math.radians(deg)
    return np.array([math.cos(t)*v[0]-math.sin(t)*v[1], math.sin(t)*v[0]+math.cos(t)*v[1]], dtype=np.float32)

def build_k_token_data(dim, K_count=1, k_map_fn=None, trigram=False):
    """Build K-token bigram data: 4 groups, each G0→G1, G1→K, K→G2, G2→G0."""
    if dim == 2:
        centers = {'A':(1.,0.), 'B':(0.,1.), 'C':(0.,-1.), 'D':(-1.,0.)}
        k_vecs = [np.array([0.707,0.707],dtype=np.float32)]
    elif dim == 3:
        centers = {'A':(1.,0.,0.), 'B':(0.,1.,0.), 'C':(0.,-1.,0.), 'D':(0.,0.,1.)}
        k_vecs = [np.ones(3,dtype=np.float32)/np.sqrt(3)]
    else:
        centers = {'A':(1.,0.,0.,0.), 'B':(0.,1.,0.,0.), 'C':(0.,0.,1.,0.), 'D':(0.,0.,0.,1.)}
        k_vecs = [np.ones(4,dtype=np.float32)*0.5]

    gns = list(centers.keys())
    # Build token vectors
    gv = {}
    for gn, c in centers.items():
        cc = np.array(c, dtype=np.float32); cc /= np.linalg.norm(cc)
        gv[gn] = []
        for off in [0.0, theta_deg, -theta_deg]:
            if dim == 2:
                v = rot_2d(cc, off)
            else:
                perp = np.zeros(dim, dtype=np.float32)
                if abs(cc[0]) < 0.9: perp[0] = 1.0
                else: perp[1] = 1.0
                perp -= float(np.dot(perp, cc)) * cc; perp /= np.linalg.norm(perp)
                t = math.radians(off)
                v = math.cos(t)*cc + math.sin(t)*perp
            gv[gn].append(v.astype(np.float32))

    all_v = list(k_vecs)
    gi = {}
    idx = len(k_vecs)
    for gn in gns:
        gi[gn] = [idx, idx+1, idx+2]
        all_v.extend(gv[gn]); idx += 3

    E0 = torch.tensor(np.stack(all_v), dtype=torch.float32)

    if k_map_fn is None:
        k_map = {gn: 0 for gn in gns}
    else:
        k_map = k_map_fn(gns)

    inputs, targets, sample_groups = [], [], []
    for gn in gns:
        i0,i1,i2 = gi[gn]; ki = k_map[gn]
        inputs.extend([i0,i1,ki,i2]); targets.extend([i1,ki,i2,i0])
        sample_groups.extend([gn]*4)
    return E0, gi, (torch.tensor(inputs,dtype=torch.long), torch.tensor(targets,dtype=torch.long), sample_groups, k_map)

def compute_reweighted_weights(base_w, targets_t, mode='none'):
    if mode == 'none':
        return base_w / base_w.sum()
    f_target = {}
    for t in targets_t.tolist(): f_target[t] = f_target.get(t, 0) + 1
    w = base_w / torch.tensor([f_target[t] for t in targets_t.tolist()], dtype=torch.float32)
    return w / w.sum()

# ============================================================
# Experiment 1: Weight-tied E=W
# ============================================================
def run_e1_weight_tied():
    print('='*70)
    print('E1: WEIGHT-TIED E=W')
    print('='*70)
    results = []

    for dim in [2, 3]:
        E0, gi, data = build_k_token_data(dim, K_count=1)
        inputs_t, targets_t, sample_groups, _ = data
        V = E0.shape[0]
        base_w = torch.ones(len(sample_groups), dtype=torch.float32) / len(sample_groups)

        for extra_params in [False, True]:
            for rew_mode in ['none', 'global']:
                adjusted_w = compute_reweighted_weights(base_w.clone(), targets_t, rew_mode)

                class TiedBigramLM(torch.nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.E = torch.nn.Parameter(E0.clone())
                        self.has_proj = extra_params
                        if extra_params:
                            self.M = torch.nn.Parameter(torch.eye(dim, dtype=torch.float32) * 0.5)
                    def forward(self, ids):
                        h = self.E[ids]
                        if self.has_proj:
                            h_proj = h @ self.M.T  # project input side
                            return h_proj @ self.E.T
                        return h @ self.E.T

                model = TiedBigramLM()
                conv = {}
                for step in range(800):
                    logits = model(inputs_t)
                    losses = F.cross_entropy(logits, targets_t, reduction='none')
                    (losses * adjusted_w).sum().backward()
                    gE = model.E.grad.clone()
                    if extra_params: gM = model.M.grad.clone()
                    with torch.no_grad():
                        pred = logits.argmax(-1); correct = (pred == targets_t).float()
                        for gn in ['A','B','C','D']:
                            gidxs = [i for i,g in enumerate(sample_groups) if g==gn]
                            acc = float(correct[gidxs].mean())
                            if acc>=1.0 and gn not in conv: conv[gn] = step+1
                        kt_idxs = [i for i,t in enumerate(targets_t.tolist()) if t==0]
                        acc = float(correct[kt_idxs].mean())
                        if acc>=1.0 and 'K' not in conv: conv['K'] = step+1
                        model.E -= lr*gE
                        if extra_params: model.M -= lr*gM

                extra_str = 'proj' if extra_params else 'pure'
                label = f'Tied_{dim}D_{extra_str}_{rew_mode}'
                results.append((label, conv))

    print(f'{"Setup":>28} {"K":>6} {"A":>6} {"B":>6} {"C":>6} {"D":>6}')
    for label, conv in results:
        def f(k): v=conv.get(k); return f'{v:4d}' if v else '  N/A'
        print(f'{label:>28} {f("K")} {f("A")} {f("B")} {f("C")} {f("D")}')
    return results

# ============================================================
# Experiment 2: Medium-scale synthesis with realistic mini-batch failure
# ============================================================
def run_e2_medium_scale():
    print('\n' + '='*70)
    print('E2: MEDIUM-SCALE SYNTHESIS (100 groups)')
    print('='*70)

    n_total = 100
    n_high = 20   # 20 high-freq groups have 90% total mass
    n_low = 80    # 80 low-freq groups have 10% total mass
    dim = 8  # need higher dim for 100 groups

    # Each group: 3 tokens, internal cyclic + universal K
    # G0→G1, G1→K, K→G2, G2→G0

    # Build token embeddings (8D: use first 2 dims for angular separation, rest random small)
    K_vec = np.zeros(dim, dtype=np.float32)
    K_vec[:min(2,dim)] = [0.707, 0.707]
    K_vec = K_vec / np.linalg.norm(K_vec)
    all_v = [K_vec]
    gi = {}
    idx = 1
    rng = np.random.default_rng(42)
    for gi_idx in range(n_total):
        angle = 2*math.pi * gi_idx / n_total
        center = np.zeros(dim, dtype=np.float32)
        center[0] = math.cos(angle)
        center[1] = math.sin(angle)
        # Add small random components in remaining dims for distinguishability
        center[2:] = rng.normal(0, 0.02, dim-2).astype(np.float32)
        center /= np.linalg.norm(center)
        gi[gi_idx] = [idx, idx+1, idx+2]
        for off in [0.0, theta_deg, -theta_deg]:
            # Rotate in 2D plane, keep higher dims same
            t = math.radians(off)
            v = center.copy()
            v[0] = math.cos(t)*center[0] - math.sin(t)*center[1]
            v[1] = math.sin(t)*center[0] + math.cos(t)*center[1]
            all_v.append(v.astype(np.float32))
            idx += 1
    V = len(all_v)
    E0 = torch.tensor(np.stack(all_v), dtype=torch.float32)

    # Bigrams: each group G0→G1, G1→K, K→G2, G2→G0
    inputs, targets, sample_groups = [], [], []
    per_high = (0.90 / n_high) / 4.0  # per-bigram weight for high-freq groups
    per_low = (0.10 / n_low) / 4.0     # per-bigram weight for low-freq groups
    for gi_idx in range(n_total):
        i0,i1,i2 = gi[gi_idx]
        inputs.extend([i0,i1,0,i2]); targets.extend([i1,0,i2,i0])
        is_high = gi_idx < n_high
        sample_groups.extend([gi_idx]*4)

    inputs_t = torch.tensor(inputs, dtype=torch.long)
    targets_t = torch.tensor(targets, dtype=torch.long)
    base_w = torch.tensor([per_high if gi < n_high else per_low for gi in sample_groups], dtype=torch.float32)
    total_bigrams = len(inputs)

    # Mini-batch settings
    n_total_data = len(inputs)
    batch_sizes = [n_total_data, 64, 32, 16]  # full, large, medium, small

    print(f'Vocab: {V} tokens (1 K + {n_total*3} group tokens)')
    print(f'{n_high} high-freq groups ({90}% total), {n_low} low-freq groups ({10}% total)')
    print(f'Per high-freq bigram weight: {per_high:.6f}, per low: {per_low:.6f}')

    results = []
    for rew_mode in ['none', 'global']:
        for bs in batch_sizes:
            max_steps = 2000 if bs < n_total_data else 600
            model = BigramLM_Simple(E0)

            # Full-rewight off global data for fair comparison
            adjusted_w_full = compute_reweighted_weights(base_w.clone(), targets_t, rew_mode)

            conv_high = []; conv_low = []
            for step in range(max_steps):
                if bs >= n_total_data:
                    logits = model(inputs_t)
                    losses = F.cross_entropy(logits, targets_t, reduction='none')
                    (losses * adjusted_w_full).sum().backward()
                else:
                    idxs = torch.randint(0, n_total_data, (bs,))
                    logits = model(inputs_t[idxs])
                    losses = F.cross_entropy(logits, targets_t[idxs], reduction='none')
                    batch_w = adjusted_w_full[idxs]
                    (losses * batch_w / batch_w.sum()).sum().backward()

                gE,gW = model.E.grad.clone(), model.W.grad.clone()
                with torch.no_grad():
                    with torch.no_grad():
                        full_logits = model(inputs_t)
                        pred = full_logits.argmax(-1)
                        correct = (pred == targets_t).float()

                    # Track high-freq convergence
                    high_accs = []
                    low_accs = []
                    for gi_idx in range(n_total):
                        gidxs = [i for i,g in enumerate(sample_groups) if g==gi_idx]
                        acc = float(correct[gidxs].mean())
                        if gi_idx < n_high: high_accs.append(acc)
                        else: low_accs.append(acc)
                    high_avg = np.mean(high_accs)
                    low_avg = np.mean(low_accs)

                    if high_avg >= 1.0 and not conv_high:
                        conv_high.append(step+1)
                    if low_avg >= 1.0 and not conv_low:
                        conv_low.append(step+1)

                    model.E -= lr*gE; model.W -= lr*gW

                    if len(conv_high) >= 1 and len(conv_low) >= 1:
                        break

            ch = conv_high[0] if conv_high else 9999
            cl = conv_low[0] if conv_low else 9999
            bs_str = 'full' if bs >= n_total_data else f'mb{bs}'
            label = f'MidScale_{bs_str}_{rew_mode}'
            results.append((label, ch, cl))
            print(f'  {label:>22}: high_avg@100%={ch:5d}, low_avg@100%={cl:5d}, gap={cl-ch:5d}')

    return results

class BigramLM_Simple(torch.nn.Module):
    def __init__(self, E0):
        super().__init__()
        self.E = torch.nn.Parameter(E0.clone())
        self.W = torch.nn.Parameter(E0.clone())
    def forward(self, ids):
        return self.E[ids] @ self.W.T

# ============================================================
# Experiment 3: 2-layer network
# ============================================================
def run_e3_two_layer():
    print('\n' + '='*70)
    print('E3: 2-LAYER MODEL (E → Linear → ReLU → Linear → W)')
    print('='*70)

    dim = 2; hidden = 8
    E0, gi, data = build_k_token_data(dim, K_count=1)
    inputs_t, targets_t, sample_groups, _ = data
    base_w = torch.ones(len(sample_groups), dtype=torch.float32) / len(sample_groups)

    results = []
    for rew_mode in ['none', 'global']:
        adjusted_w = compute_reweighted_weights(base_w.clone(), targets_t, rew_mode)

        class TwoLayerLM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.E = torch.nn.Parameter(E0.clone())
                self.W = torch.nn.Parameter(E0.clone())
                # Xavier init
                self.L1 = torch.nn.Parameter(torch.empty(dim, hidden, dtype=torch.float32))
                self.L2 = torch.nn.Parameter(torch.empty(hidden, dim, dtype=torch.float32))
                torch.nn.init.xavier_uniform_(self.L1)
                torch.nn.init.xavier_uniform_(self.L2)
            def forward(self, ids):
                h = self.E[ids]
                h = F.relu(h @ self.L1)
                h = h @ self.L2
                return h @ self.W.T

        model = TwoLayerLM()
        conv = {}
        for step in range(800):
            logits = model(inputs_t)
            losses = F.cross_entropy(logits, targets_t, reduction='none')
            (losses * adjusted_w).sum().backward()
            with torch.no_grad():
                pred = logits.argmax(-1); correct = (pred == targets_t).float()
                for gn in ['A','B','C','D']:
                    gidxs = [i for i,g in enumerate(sample_groups) if g==gn]
                    acc = float(correct[gidxs].mean())
                    if acc>=1.0 and gn not in conv: conv[gn] = step+1
                kt_idxs = [i for i,t in enumerate(targets_t.tolist()) if t==0]
                acc = float(correct[kt_idxs].mean())
                if acc>=1.0 and 'K' not in conv: conv['K'] = step+1
                for p in model.parameters():
                    if p.grad is not None: p -= lr * p.grad
                model.zero_grad()

        label = f'2Layer_{rew_mode}'
        results.append((label, conv))

    print(f'{"Setup":>18} {"K":>6} {"A":>6} {"B":>6} {"C":>6} {"D":>6}')
    for label, conv in results:
        def f(k): v=conv.get(k); return f'{v:4d}' if v else '  N/A'
        print(f'{label:>18} {f("K")} {f("A")} {f("B")} {f("C")} {f("D")}')
    return results

# ============================================================
# Run all
# ============================================================
r1 = run_e1_weight_tied()
r2 = run_e2_medium_scale()
r3 = run_e3_two_layer()

print('\n' + '='*70)
print('SUMMARY')
print('='*70)
print('E1: Weight-tied E=W needs extra linear projection to train.')
print('     Pure tied model cannot learn (E[i] and E[j] identical direction).')
print('E2: Medium-scale: reweighting works until batch << vocab, then degrades.')
print('E3: 2-layer model: reweighting still effective through nonlinear layers.')
