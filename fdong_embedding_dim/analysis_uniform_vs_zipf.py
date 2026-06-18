#!/usr/bin/env python3
"""Uniform vs Zipf: trace how parameter geometry evolves differently.
Fixed: proper per-sample weighting."""
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
group_names = ['common', 'tail1', 'tail2', 'tail3']

def rotate_in_plane(center, angle_deg):
    th = math.radians(angle_deg)
    return np.array([
        math.cos(th) * center[0] - math.sin(th) * center[1],
        math.sin(th) * center[0] + math.cos(th) * center[1],
    ], dtype=np.float32)

all_vecs = []
for gname in group_names:
    center = centers[gname]
    for angle in [0.0, theta_deg, -theta_deg]:
        all_vecs.append(rotate_in_plane(center, angle))
E0 = torch.tensor(np.stack(all_vecs), dtype=torch.float32)
W0 = E0.clone()

inputs = []
targets = []
sample_groups = []
for gidx, gname in enumerate(group_names):
    base = gidx * 3
    for t in range(3):
        inputs.append(base + t)
        targets.append(base + (t + 1) % 3)
        sample_groups.append(gname)

inputs_t = torch.tensor(inputs, dtype=torch.long)
targets_t = torch.tensor(targets, dtype=torch.long)

probs_uniform = {'common': 0.25, 'tail1': 0.25, 'tail2': 0.25, 'tail3': 0.25}
probs_zipf    = {'common': 0.70, 'tail1': 0.10, 'tail2': 0.10, 'tail3': 0.10}

class BigramLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.E = torch.nn.Parameter(E0.clone())
        self.W = torch.nn.Parameter(W0.clone())
    def forward(self, ids):
        return self.E[ids] @ self.W.T

def run_sim(probs, label, max_steps=200):
    torch.manual_seed(0)
    model = BigramLM()
    sample_weights = torch.tensor([probs[g] / 3.0 for g in sample_groups], dtype=torch.float32)
    
    history = []
    for s in range(max_steps):
        logits = model(inputs_t)
        losses = F.cross_entropy(logits, targets_t, reduction='none')
        weighted_loss = (losses * sample_weights).sum()
        
        model.E.grad = None
        model.W.grad = None
        weighted_loss.backward()
        
        E_grad = model.E.grad.clone()
        W_grad = model.W.grad.clone()
        
        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            correct = (pred == targets_t).float()
            group_acc = {}
            for gname in group_names:
                gidxs = [i for i, g in enumerate(sample_groups) if g == gname]
                group_acc[gname] = float(correct[gidxs].mean())
            E_np = model.E.detach().numpy()
            _, sv, _ = np.linalg.svd(E_np, full_matrices=False)
            centroids = {}
            for gidx, gname in enumerate(group_names):
                centroids[gname] = E_np[gidx*3:(gidx+1)*3].mean(axis=0)
            norms = {}
            for gidx, gname in enumerate(group_names):
                norms[gname] = float(np.linalg.norm(E_np[gidx*3:(gidx+1)*3].mean(axis=0)))
            
            model.E -= lr * E_grad
            model.W -= lr * W_grad
        
        sv_ratio = sv[0] / (sv[1] + 1e-12) if len(sv) >= 2 and sv[1] > 1e-8 else 1.0
        history.append({
            'step': s+1,
            'loss': float(weighted_loss.item()),
            'acc': group_acc.copy(),
            'sv': sv.copy(),
            'sv_ratio': sv_ratio,
            'centroids': {k: v.copy() for k, v in centroids.items()},
            'norms': norms.copy(),
        })
    return history

print("Running Uniform  ...", end=" ", flush=True)
hist_u = run_sim(probs_uniform, "Uniform")
print(f"done ({len(hist_u)} steps)")
print("Running Zipf     ...", end=" ", flush=True)
hist_z = run_sim(probs_zipf, "Zipf")
print(f"done ({len(hist_z)} steps)")

# ========================
print("\n" + "=" * 80)
print("COMPARISON: UNIFORM (1:1:1:1) vs ZIPF (7:1:1:1)")
print("=" * 80)

# Step 1: motion per centroid
print("\n--- Step 1: Per-group centroid motion ---")
print(f'{"Group":>10} {"Uniform Δx":>14} {"Uniform Δy":>14} {"|Δ|":>12} | {"Zipf Δx":>14} {"Zipf Δy":>14} {"|Δ|":>12}')
for gname in group_names:
    cu = hist_u[0]['centroids'][gname] - centers[gname]
    cz = hist_z[0]['centroids'][gname] - centers[gname]
    nu = np.linalg.norm(cu)
    nz = np.linalg.norm(cz)
    print(f'{gname:>10} {cu[0]:+14.8f} {cu[1]:+14.8f} {nu:12.6f} | {cz[0]:+14.8f} {cz[1]:+14.8f} {nz:12.6f}')

# Step 1: per-group norm
print("\n--- Step 1: Per-group centroid norm ---")
print(f'{"Group":>10} {"Uniform":>12} {"Zipf":>12} {"Z/U":>10}')
for gname in group_names:
    nu = hist_u[0]['norms'][gname]
    nz = hist_z[0]['norms'][gname]
    print(f'{gname:>10} {nu:12.8f} {nz:12.8f} {nz/nu:10.6f}')

# Singular value ratio evolution
print("\n--- Singular value ratio σ1/σ2 ---")
print(f'{"Step":>6} {"Uniform":>14} {"Zipf":>14} {"Comment":>30}')
key_steps = [1, 5, 10, 15, 20, 25, 30, 40, 50, 200]
for s in key_steps:
    if s <= len(hist_u):
        u_r = hist_u[s-1]['sv_ratio']
        z_r = hist_z[s-1]['sv_ratio']
        comment = ""
        if s <= len(hist_u) and s <= len(hist_z):
            ug = hist_u[s-1]['acc']
            zg = hist_z[s-1]['acc']
            u_all = all(v>=1.0 for v in ug.values())
            z_all = all(v>=1.0 for v in zg.values())
            if u_all and z_all:
                comment = "both converged"
            elif u_all:
                comment = "uniform converged, zipf NOT"
            elif z_all:
                comment = "zipf converged, uniform NOT"
        print(f'{s:6d} {u_r:14.8f} {z_r:14.8f} {comment:>30}')

# Convergence: first step each group hits acc=1.0
print("\n--- First step accuracy=1.0 ---")
print(f'{"Group":>10} {"Uniform":>10} {"Zipf":>10} {"Z/U":>10}')
for gname in group_names:
    try:
        u_s = next(i+1 for i, h in enumerate(hist_u) if h['acc'][gname] >= 1.0)
    except StopIteration:
        u_s = 999
    try:
        z_s = next(i+1 for i, h in enumerate(hist_z) if h['acc'][gname] >= 1.0)
    except StopIteration:
        z_s = 999
    print(f'{gname:>10} {u_s:10d} {z_s:10d} {z_s/max(u_s,1):10.2f}')

# Per-step centroid speed early
print("\n--- Per-step centroid speed (first 10 steps) ---")
for gname in group_names:
    print(f'\n{gname}:')
    print(f'  {"Step":>6} {"Uniform |Δ|":>14} {"Zipf |Δ|":>14} {"Z/U ratio":>12} {"U com/tail":>12} {"Z com/tail":>12}')
    u_prev = centers[gname].copy()
    z_prev = centers[gname].copy()
    for s in range(min(10, min(len(hist_u), len(hist_z)))):
        uc = hist_u[s]['centroids'][gname]
        zc = hist_z[s]['centroids'][gname]
        ud = np.linalg.norm(uc - u_prev)
        zd = np.linalg.norm(zc - z_prev)
        # Also compute common/tail speed ratio
        if gname == 'common':
            u_ct = '-'
            z_ct = '-'
        else:
            u_common_spd = np.linalg.norm(hist_u[s]['centroids']['common'] - (centers['common'] if s == 0 else hist_u[s-1]['centroids']['common']))
            z_common_spd = np.linalg.norm(hist_z[s]['centroids']['common'] - (centers['common'] if s == 0 else hist_z[s-1]['centroids']['common']))
            u_ct = f'{u_common_spd/(ud+1e-12):.2f}'
            z_ct = f'{z_common_spd/(zd+1e-12):.2f}'
        print(f'  {s+1:6d} {ud:14.10f} {zd:14.10f} {zd/(ud+1e-12):12.4f} {u_ct:>12} {z_ct:>12}')
        u_prev = uc.copy()
        z_prev = zc.copy()

# Final geometry
print("\n--- Final geometry ---")
uf = hist_u[-1]
zf = hist_z[-1]
print(f'Uniform (step {len(hist_u)}): σ=[{uf["sv"][0]:.2f}, {uf["sv"][1]:.2f}], σ1/σ2={uf["sv_ratio"]:.4f}')
print(f'Zipf    (step {len(hist_z)}): σ=[{zf["sv"][0]:.2f}, {zf["sv"][1]:.2f}], σ1/σ2={zf["sv_ratio"]:.4f}')
print(f'\n{"Group":>10} {"Uni |c|":>12} {"Zpf |c|":>12} {"Z/U":>10}')
for gname in group_names:
    print(f'{gname:>10} {uf["norms"][gname]:12.4f} {zf["norms"][gname]:12.4f} {zf["norms"][gname]/uf["norms"][gname]:10.4f}')
