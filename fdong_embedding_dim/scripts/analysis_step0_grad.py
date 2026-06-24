#!/usr/bin/env python3
"""Step-0 gradient analysis: trace exactly what the gradient does to parameter space."""
import math
import numpy as np
import torch
import torch.nn.functional as F

dim = 2
theta_deg = 12.0
lr = 0.03

centers = {
    'common': np.array([1.0, 0.0], dtype=np.float32),
    'tail1':  np.array([0.0, 1.0], dtype=np.float32),
    'tail2':  np.array([0.0, -1.0], dtype=np.float32),
    'tail3':  np.array([-1.0, 0.0], dtype=np.float32),
}

def rotate_in_plane(center, angle_deg):
    th = math.radians(angle_deg)
    return np.array([
        math.cos(th) * center[0] - math.sin(th) * center[1],
        math.sin(th) * center[0] + math.cos(th) * center[1],
    ], dtype=np.float32)

group_tokens = {}
all_vecs = []
for gname, center in centers.items():
    toks = []
    for angle in [0.0, theta_deg, -theta_deg]:
        toks.append(rotate_in_plane(center, angle))
        all_vecs.append(toks[-1])
    group_tokens[gname] = toks

E0 = torch.tensor(np.stack(all_vecs), dtype=torch.float32)
W0 = E0.clone()

# Cyclic bigrams within each group
inputs = []
targets = []
for gidx, (gname, toks) in enumerate(group_tokens.items()):
    base = gidx * 3
    for t in range(3):
        inputs.append(base + t)
        targets.append(base + (t + 1) % 3)

targets = torch.tensor(targets, dtype=torch.long)

group_names = ['common', 'tail1', 'tail2', 'tail3']
group_probs = {'common': 0.70, 'tail1': 0.10, 'tail2': 0.10, 'tail3': 0.10}

def flatten_grad(model):
    gE = model.E.grad.detach().reshape(-1).clone()
    gW = model.W.grad.detach().reshape(-1).clone()
    return torch.cat([gE, gW]).numpy()

print('=' * 80)
print('STEP 0 GRADIENT TRACE — What exactly does the gradient modify?')
print('=' * 80)

# --- 1. Conditional gradients per group ---
print('\n' + '─' * 60)
print('1. CONDITIONAL GRADIENTS PER GROUP (if only that group data existed)')
print('─' * 60)

q = {}
norms = {}
for gname in group_names:
    class BigramLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.E = torch.nn.Parameter(E0.clone())
            self.W = torch.nn.Parameter(W0.clone())
        def forward(self, ids):
            return self.E[ids] @ self.W.T
    
    model = BigramLM()
    gidx = group_names.index(gname)
    idxs = torch.tensor([gidx * 3 + t for t in range(3)], dtype=torch.long)
    targs = torch.tensor([gidx * 3 + (t + 1) % 3 for t in range(3)], dtype=torch.long)
    
    logits = model(idxs)
    loss = F.cross_entropy(logits, targs, reduction='mean')
    loss.backward()
    vec = flatten_grad(model)
    norms[gname] = float(np.linalg.norm(vec))
    q[gname] = vec

print('\nConditional gradient pairwise cosines:')
header = ''.join(f'{g:>10}' for g in group_names)
print(f'{"":>10}{header}')
for g1 in group_names:
    row = f'{g1:>10}'
    for g2 in group_names:
        cos = float(np.dot(q[g1], q[g2]) / max(norms[g1] * norms[g2], 1e-12))
        row += f'{cos:10.4f}'
    print(row)

print('\n--- SIR for each tail (conditional gradient) ---')
for gname in ['tail1', 'tail2', 'tail3']:
    unit = q[gname] / max(norms[gname], 1e-12)
    signal = group_probs[gname] * norms[gname]
    common_interf = abs(group_probs['common'] * float(np.dot(q['common'], unit)))
    other_interf = 0.0
    for other in ['tail1', 'tail2', 'tail3']:
        if other != gname:
            other_interf += abs(group_probs[other] * float(np.dot(q[other], unit)))
    sir = signal / (common_interf + other_interf + 1e-12)
    print(f'  {gname}: signal={signal:.3f}, common_interf={common_interf:.3f}, '
          f'other_interf={other_interf:.3f} → SIR={sir:.3f}')

# --- 2. Full gradient (all data, weighted) ---
print('\n' + '─' * 60)
print('2. FULL-BATCH GRADIENT (weighted sum of all groups)')
print('─' * 60)

class BigramLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.E = torch.nn.Parameter(E0.clone())
        self.W = torch.nn.Parameter(W0.clone())
    def forward(self, ids):
        return self.E[ids] @ self.W.T

model = BigramLM()
logits_all = model(torch.tensor(inputs, dtype=torch.long))
# Weighted loss to match experiment: each bigram weighted by group_prob/3
sample_weights = torch.zeros(12, dtype=torch.float32)
for i, (inp, targ) in enumerate(zip(inputs, targets)):
    gidx = inp // 3
    sample_weights[i] = group_probs[group_names[gidx]] / 3.0
loss_all = F.cross_entropy(logits_all, targets, weight=sample_weights, reduction='sum')
loss_all.backward()

E_grad = model.E.grad.detach().numpy()
W_grad = model.W.grad.detach().numpy()

print('\nPer-token gradient and one-step update (lr=0.03):')
print(f'{"Token":>16} | {"E_init":>20} | {"E_grad":>24} | {"E_new":>20} | {"W_grad":>24} | {"W_new":>20}')
print('-' * 140)
for i in range(12):
    gname = group_names[i // 3]
    label = f'{gname}_{i%3}'
    e0 = E0[i].numpy()
    ge = E_grad[i]
    gw = W_grad[i]
    en = e0 - lr * ge
    wn = E0[i].numpy() - lr * gw
    print(f'{label:>16} | ({e0[0]:+7.4f},{e0[1]:+7.4f}) | '
          f'({ge[0]:+8.4f},{ge[1]:+8.4f}) | ({en[0]:+7.4f},{en[1]:+7.4f}) | '
          f'({gw[0]:+8.4f},{gw[1]:+8.4f}) | ({wn[0]:+7.4f},{wn[1]:+7.4f})')

# --- 3. Decompose: what drives each token's gradient? ---
print('\n' + '─' * 60)
print('3. GRADIENT DECOMPOSITION for E[common_0] (target: common_1)')
print('─' * 60)

print('\n∂L/∂E[common_0] = Σ_m (softmax_m - δ_{m,target}) · W[m]')
print(f'Initial E[common_0] = ({E0[0,0]:+.4f},{E0[0,1]:+.4f})')
print(f'Target token: common_1 (index 1), W[1] = ({W0[1,0]:+.4f},{W0[1,1]:+.4f})')

logit = E0[0] @ W0.T
sm = torch.softmax(logit, dim=0).detach().numpy()
print('\nSoftmax probabilities for common_0 → ?:')
for m in range(12):
    marker = ' ← TARGET' if m == 1 else ''
    print(f'  token {m:2d} ({group_names[m//3]:>6}_{m%3}): sm={sm[m]:.6f}{marker}')

print('\nContributions to ∂L/∂E[common_0]:')
print(f'{"m":>3} {"group":>8} {"softmax":>10} {"coeff":>10} {"W_x":>8} {"W_y":>8} {"contrib_x":>10} {"contrib_y":>10}')
print('-' * 80)

contribs = []
for m in range(12):
    coef = sm[m] - (1.0 if m == 1 else 0.0)
    cx = coef * W0[m, 0].item()
    cy = coef * W0[m, 1].item()
    contribs.append((cx, cy))
    gname = group_names[m // 3]
    print(f'{m:3d} {gname:>8} {sm[m]:10.6f} {coef:+10.6f} '
          f'{W0[m,0].item():+8.4f} {W0[m,1].item():+8.4f} {cx:+10.6f} {cy:+10.6f}')

total = (sum(c[0] for c in contribs), sum(c[1] for c in contribs))
print(f'{"":>3} {"TOTAL":>8} {"":>10} {"":>10} {"":>8} {"":>8} {total[0]:+10.6f} {total[1]:+10.6f}')

# Break down by group
for gidx, gname in enumerate(group_names):
    gc = [contribs[gidx*3 + t] for t in range(3)]
    gsum = (sum(c[0] for c in gc), sum(c[1] for c in gc))
    print(f'{"":>3} {gname:>8} (sum) {"":>10} {"":>10} {"":>8} {"":>8} {gsum[0]:+10.6f} {gsum[1]:+10.6f}')

# --- 4. Symmetry analysis: uniform vs zipf ---
print('\n' + '─' * 60)
print('4. UNIFORM vs ZIPF: the asymmetry')
print('─' * 60)

print('\nIn uniform case (each group 25%):')
print('  g = 0.25 · q_common + 0.25 · q_tail1 + 0.25 · q_tail2 + 0.25 · q_tail3')
print('  symmetric → all groups contribute equally → symmetric update')

print('\nIn Zipf case (common 70%, tails 10% each):')
print('  g = 0.70 · q_common + 0.10 · (q_tail1 + q_tail2 + q_tail3)')
print('  q_common dominates → parameter update is mostly common direction')

# numerical check
for ti, tname in enumerate(['tail1', 'tail2', 'tail3'], 1):
    u = q[tname] / norms[tname]
    proj_common = float(np.dot(q['common'], u)) / norms['common']
    print(f'\n  q_common projected onto {tname}\'s gradient direction: '
          f'cos = {proj_common:.4f}, |proj| = {abs(proj_common):.4f}')
    
    # Uniform: net gradient in tail direction
    uniform_net = (0.25 * norms[tname] + 0.25 * np.dot(q['common'], u) 
                   + 0.25 * sum(np.dot(q[o], u) for o in ['tail1','tail2','tail3'] if o != tname))
    # Zipf: net gradient in tail direction
    zipf_net = (0.10 * norms[tname] + 0.70 * np.dot(q['common'], u)
                + 0.10 * sum(np.dot(q[o], u) for o in ['tail1','tail2','tail3'] if o != tname))
    print(f'    Uniform: net effective gradient along {tname} dir = {uniform_net:.4f}')
    print(f'    Zipf:    net effective gradient along {tname} dir = {zipf_net:.4f}')
    print(f'    Ratio (zipf/uniform): {zipf_net/max(uniform_net,1e-12):.4f}')

# --- 5. Where does the representation space go? ---
print('\n' + '─' * 60)
print('5. WHERE DOES REPRESENTATION SPACE GO after 1 step?')
print('─' * 60)

print('\nBefore step (initial E):')
for i in range(12):
    gname = group_names[i // 3]
    v = E0[i].numpy()
    print(f'  {gname:>6}_{i%3}: ({v[0]:+.4f}, {v[1]:+.4f}), norm={np.linalg.norm(v):.4f}')

print('\nAfter 1 GD step (E_new = E0 - lr * grad):')
for i in range(12):
    gname = group_names[i // 3]
    en = E0[i].numpy() - lr * E_grad[i]
    angle = math.degrees(math.atan2(en[1], en[0]))
    print(f'  {gname:>6}_{i%3}: ({en[0]:+.4f}, {en[1]:+.4f}), norm={np.linalg.norm(en):.4f}, angle={angle:+.2f}°')

print('\nKey observation:')
# Compute per-group motion
for gidx, gname in enumerate(group_names):
    v0 = E0[gidx*3:(gidx+1)*3].numpy()
    v1 = E0[gidx*3:(gidx+1)*3].numpy() - lr * E_grad[gidx*3:(gidx+1)*3]
    centroid_old = v0.mean(axis=0)
    centroid_new = v1.mean(axis=0)
    motion = centroid_new - centroid_old
    print(f'  {gname}: centroid moves ({motion[0]:+.6f}, {motion[1]:+.6f})')
    # Is the motion aligned with the group's intended direction?
    target_dir = centers[gname]
    cos_motion = np.dot(motion, target_dir) / max(np.linalg.norm(motion) * np.linalg.norm(target_dir), 1e-12)
    print(f'    Motion vs target direction cosine: {cos_motion:.4f}')
