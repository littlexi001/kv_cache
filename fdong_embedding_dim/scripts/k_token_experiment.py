#!/usr/bin/env python3
"""K-token as 'mean bias': K acts as a common connector token (like 'and').
   Groups keep internal cycles, K bridges between them.
   
   V2: each group has 3 tokens, 2 regular + 1 transition through K.
   A0→A1, A1→K, K→A2, A2→A0  (A1→K→A2, two steps through K)
   Similarly for B, C, D.
   
   Zipf: A=70%, B=10%, C=10%, D=10%
   → K appears in 2/4 bigrams of EACH group → always 50% probability as target/input
"""
import math, numpy as np
import torch, torch.nn.functional as F

theta_deg = 12.0; dim = 2; lr = 0.03

def rot(v, d):
    t = math.radians(d)
    return np.array([math.cos(t)*v[0]-math.sin(t)*v[1], math.sin(t)*v[0]+math.cos(t)*v[1]], dtype=np.float32)

group_centers = {'A':(1.,0.), 'B':(0.,1.), 'C':(0.,-1.), 'D':(-1.,0.)}
group_probs  = {'A':0.25, 'B':0.25, 'C':0.25, 'D':0.25}  # UNIFORM: K's pull is symmetric from all groups

# K + 4 groups × 3 tokens = 13 tokens
token_vecs = []
# K at a distinctive position (not on any group axis, ~45°)
k_init = np.array([0.707, 0.707], dtype=np.float32)
token_vecs.append(k_init)

group_indices = {}  # group -> [i0, i1, i2]
idx = 1
for gname, center in group_centers.items():
    i0 = idx; token_vecs.append(rot(center, 0.0)); idx += 1
    i1 = idx; token_vecs.append(rot(center, theta_deg)); idx += 1
    i2 = idx; token_vecs.append(rot(center, -theta_deg)); idx += 1
    group_indices[gname] = [i0, i1, i2]

E0 = torch.tensor(np.stack(token_vecs), dtype=torch.float32)
W0 = E0.clone()
V = len(token_vecs)

# Bigrams: each group has 4 bigrams → G0→G1, G1→K, K→G2, G2→G0
inputs, targets, sample_weights = [], [], []
for gname, (i0, i1, i2) in group_indices.items():
    w = group_probs[gname] / 4.0  # 4 bigrams per group
    # internal: G0→G1
    inputs.append(i0); targets.append(i1); sample_weights.append(w)
    # bridge: G1→K
    inputs.append(i1); targets.append(0); sample_weights.append(w)
    # bridge: K→G2
    inputs.append(0); targets.append(i2); sample_weights.append(w)
    # internal: G2→G0
    inputs.append(i2); targets.append(i0); sample_weights.append(w)

targets_t = torch.tensor(targets, dtype=torch.long)
sample_weights_t = torch.tensor(sample_weights, dtype=torch.float32)
inp_t = torch.tensor(inputs, dtype=torch.long)

print('=' * 70)
print('K-TOKEN AS MEAN BIAS (v2)')
print('=' * 70)
print(f'\n13 tokens: K + 4 groups × 3 tokens')
print(f'K initial: ({k_init[0]:+.3f}, {k_init[1]:+.3f})')
print(f'Each group: G0→G1, G1→K, K→G2, G2→G0 (2 internal + 2 through K)')
print(f'K freq: target in 4/16 bigrams, input in 4/16 bigrams')
print(f'UNIFORM: all groups 25%')
print()

class BigramLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.E = torch.nn.Parameter(E0.clone())
        self.W = torch.nn.Parameter(W0.clone())
    def forward(self, ids):
        return self.E[ids] @ self.W.T

model = BigramLM()
for step in range(600):
    logits = model(inp_t)
    losses = F.cross_entropy(logits, targets_t, reduction='none')
    (losses * sample_weights_t).sum().backward()
    gE, gW = model.E.grad.clone(), model.W.grad.clone()
    with torch.no_grad():
        model.E -= lr * gE
        model.W -= lr * gW

E_final = model.E.detach().numpy()

# Analysis
k_pos = E_final[0]
k_dir = k_pos / max(np.linalg.norm(k_pos), 1e-12)
centroid_all = E_final[1:].mean(axis=0)
c_dir = centroid_all / max(np.linalg.norm(centroid_all), 1e-12)

print(f'K  final |norm| = {np.linalg.norm(k_pos):.4f}')
print(f'K  initial  dir  = ({k_init[0]:+.4f}, {k_init[1]:+.4f})')
print(f'K  final    dir  = ({k_dir[0]:+.4f}, {k_dir[1]:+.4f})')
print(f'K  deviation     = {math.degrees(math.acos(np.clip(np.dot(k_init,k_dir),-1,1))):.2f}°')
print()
print(f'Mean of non-K tokens: norm={np.linalg.norm(centroid_all):.4f}')
print(f'Mean of non-K dir:    ({c_dir[0]:+.4f}, {c_dir[1]:+.4f})')
print(f'Angle(K init, mean):  {math.degrees(math.acos(np.clip(np.dot(k_init/np.linalg.norm(k_init),c_dir),-1,1))):.2f}°')
print(f'Angle(K final, mean): {math.degrees(math.acos(np.clip(np.dot(k_dir,c_dir),-1,1))):.2f}°')
print()

print('Per-group analysis:')
for gname, (i0,i1,i2) in group_indices.items():
    gc = E_final[[i0,i1,i2]].mean(axis=0)
    gc_dir = gc / max(np.linalg.norm(gc), 1e-12)
    init_dir = np.array(group_centers[gname])
    # Deviation from init
    dev = math.degrees(math.acos(np.clip(np.dot(gc_dir, init_dir/np.linalg.norm(init_dir)), -1, 1)))
    # Distance to K
    dk = np.linalg.norm(gc - k_pos)
    cos_k = float(np.dot(gc_dir, k_dir))
    print(f'  {gname} (freq={group_probs[gname]:.0%}): |c|={np.linalg.norm(gc):.2f}, '
          f'dev={dev:.1f}°, cos(K)={cos_k:+.3f}, dist(K)={dk:.2f}')

print('\nCross-group centroid cosines:')
gnames = list(group_centers.keys())
for i, g1 in enumerate(gnames):
    c1 = E_final[group_indices[g1]].mean(axis=0)
    c1 = c1 / max(np.linalg.norm(c1), 1e-12)
    for g2 in gnames[i+1:]:
        c2 = E_final[group_indices[g2]].mean(axis=0)
        c2 = c2 / max(np.linalg.norm(c2), 1e-12)
        print(f'  {g1}·{g2} = {float(np.dot(c1,c2)):+.4f}')

print('\n--- CONTRAST WITH ORIGINAL TOY (no K) ---')
print('Original (step 200): common deviated 2°, tails deviated 41°')
print('With K connector:    K now plays the \"and\" role -')
print('  common goes through K → affected by K\'s pulling')
print('  tails go through K → also affected')
print('  K acts as gravity center that all groups orbit')
