#!/usr/bin/env python3
"""
Extended experiments:
  E1: Universal K vs Subset K (3D, uniform & Zipf, bigram & trigram)
  E2: K frequency hierarchy (3D, uniform)
  E3: Dimension sweep (2D/3D/4D, Universal K, uniform)
  E4: Mini-batch effect (partial data per batch, test reweighting robustness)

All with/without inverse target frequency reweighting.
"""
import math, numpy as np
import torch, torch.nn.functional as F

theta_deg = 12.0
lr = 0.03

# ============================================================
# Geometry helpers
# ============================================================
def rot_2d(v, deg):
    t = math.radians(deg)
    return np.array([math.cos(t)*v[0]-math.sin(t)*v[1], math.sin(t)*v[0]+math.cos(t)*v[1]], dtype=np.float32)

def spread_init_groups(dim, group_centers):
    """Build token vectors for groups at given center directions."""
    vectors = {}
    for gn, center in group_centers.items():
        c = np.array(center, dtype=np.float32)
        c /= np.linalg.norm(c)
        vectors[gn] = []
        for offset in [0.0, theta_deg, -theta_deg]:
            if dim == 2:
                v = rot_2d(c, offset)
            else:
                # For dim>2, rotate in the plane of (c, perp) where perp is a random orthogonal direction
                perp = np.zeros(dim, dtype=np.float32)
                if abs(c[0]) < 0.9:
                    perp[0] = 1.0
                else:
                    perp[1] = 1.0
                perp -= float(np.dot(perp, c)) * c
                perp /= np.linalg.norm(perp)
                t = math.radians(offset)
                v = math.cos(t) * c + math.sin(t) * perp
            vectors[gn].append(v.astype(np.float32))
    return vectors

def build_token_table(K_vectors, group_vectors, group_names):
    """Build flat token table: [K0, K1, ..., G0_tok0, G0_tok1, G0_tok2, G1...]"""
    all_vecs = list(K_vectors)
    idx = len(K_vectors)
    group_indices = {}  # group -> [idx0, idx1, idx2]
    for gn in group_names:
        vs = group_vectors[gn]
        group_indices[gn] = [idx, idx+1, idx+2]
        all_vecs.extend(vs)
        idx += 3
    return torch.tensor(np.stack(all_vecs), dtype=torch.float32), group_indices

# ============================================================
# Data builders
# ============================================================
def build_4group_centers(dim):
    if dim == 2:
        return {'A': (1.,0.), 'B': (0.,1.), 'C': (0.,-1.), 'D': (-1.,0.)}
    elif dim == 3:
        return {'A': (1.,0.,0.), 'B': (0.,1.,0.), 'C': (0.,-1.,0.), 'D': (0.,0.,1.)}
    elif dim == 4:
        return {'A': (1.,0.,0.,0.), 'B': (0.,1.,0.,0.), 'C': (0.,0.,1.,0.), 'D': (0.,0.,0.,1.)}
    raise ValueError(f"dim={dim} not supported")

def K_vector_neutral(dim):
    """Neutral K position, equidistant from all group directions."""
    if dim <= 2:
        return np.array([0.707, 0.707], dtype=np.float32)
    v = np.ones(dim, dtype=np.float32)
    return v / np.linalg.norm(v)

def K_vectors_subset(dim):
    """K1 connecting A+B, K2 connecting C+D."""
    if dim == 2:
        k1 = np.array([0.707, 0.707], dtype=np.float32)  # between A(1,0) and B(0,1)
        k2 = np.array([-0.707, -0.707], dtype=np.float32)  # between C(0,-1) and D(-1,0)
    elif dim == 3:
        k1 = np.array([0.5, 0.5, 0.707], dtype=np.float32)  # biased toward x-y plane
        k2 = np.array([0.0, -0.707, 0.707], dtype=np.float32)  # biased toward y-z plane
    else:
        k1 = np.zeros(dim, dtype=np.float32); k1[0]=1.0; k1[1]=0.5
        k2 = np.zeros(dim, dtype=np.float32); k2[1]=-0.5; k2[2]=1.0
    k1 /= np.linalg.norm(k1)
    k2 /= np.linalg.norm(k2)
    return [k1, k2]

def K_vectors_hierarchy(dim):
    """K1 (high freq, A+B+C), K2 (mid freq, B+C+D), K3 (low freq, D)."""
    if dim <= 2:
        k1 = np.array([0.5, 0.5], dtype=np.float32)
        k2 = np.array([-0.5, -0.5], dtype=np.float32)
        k3 = np.array([-0.866, -0.5], dtype=np.float32)
    elif dim == 3:
        k1 = np.array([0.4, 0.0, 0.0], dtype=np.float32)  # near A, between A,B
        k2 = np.array([0.0, -0.3, 0.3], dtype=np.float32)  # between B,C,D
        k3 = np.array([0.0, 0.0, 0.7], dtype=np.float32)  # near D
    else:
        k1 = np.zeros(dim, dtype=np.float32); k1[0]=1.0
        k2 = np.zeros(dim, dtype=np.float32); k2[1]=-0.5; k2[2]=0.5
        k3 = np.zeros(dim, dtype=np.float32); k3[3]=1.0
    for ki in [k1, k2, k3]:
        n = np.linalg.norm(ki); ki /= max(n, 1e-12)
    return [k1, k2, k3]

def build_data_universal_K(dim, trigram=False):
    """1 K connecting all 4 groups. G0→G1, G1→K, K→G2, G2→G0."""
    centers = build_4group_centers(dim)
    gns = list(centers.keys())
    gv = spread_init_groups(dim, centers)
    Ks = [K_vector_neutral(dim)]
    E0, gi = build_token_table(Ks, gv, gns)
    
    if not trigram:
        inputs, targets, sample_groups = [], [], []
        for gn in gns:
            i0,i1,i2 = gi[gn]
            inputs.extend([i0,i1,0,i2]); targets.extend([i1,0,i2,i0])
            sample_groups.extend([gn]*4)
        return E0, gi, (torch.tensor(inputs,dtype=torch.long), torch.tensor(targets,dtype=torch.long), sample_groups, None, None)
    else:
        return None  # not implementing trigram for extended experiments, keep it simple

def build_data_subset_K(dim, trigram=False):
    """K1 connects A+B, K2 connects C+D. G0→G1, G1→Ki, Ki→G2, G2→G0."""
    centers = build_4group_centers(dim)
    gns = list(centers.keys())
    gv = spread_init_groups(dim, centers)
    Ks = K_vectors_subset(dim)
    E0, gi = build_token_table(Ks, gv, gns)
    
    # K1 (index 0) for A+B, K2 (index 1) for C+D
    k_map = {'A': 0, 'B': 0, 'C': 1, 'D': 1}
    
    if not trigram:
        inputs, targets, sample_groups = [], [], []
        for gn in gns:
            i0,i1,i2 = gi[gn]; ki = k_map[gn]
            inputs.extend([i0,i1,ki,i2]); targets.extend([i1,ki,i2,i0])
            sample_groups.extend([gn]*4)
        return E0, gi, (torch.tensor(inputs,dtype=torch.long), torch.tensor(targets,dtype=torch.long), sample_groups, k_map, None)
    else:
        return None

def build_data_hierarchy_K(dim):
    """K1 (A+B+C), K2 (B+C+D), K3 (D). G0→G1, G1→K, K→G2, G2→G0."""
    centers = build_4group_centers(dim)
    gns = list(centers.keys())
    gv = spread_init_groups(dim, centers)
    Ks = K_vectors_hierarchy(dim)
    E0, gi = build_token_table(Ks, gv, gns)
    
    # Mapping: which K each group connects to
    # K1(0): A, B, C; K2(1): B, C, D; K3(2): D
    # But each group must go through exactly 1 K. Let's assign:
    # A→K1, B→K1, C→K2, D→K3
    k_map = {'A': 0, 'B': 0, 'C': 1, 'D': 2}
    
    inputs, targets, sample_groups = [], [], []
    for gn in gns:
        i0,i1,i2 = gi[gn]; ki = k_map[gn]
        inputs.extend([i0,i1,ki,i2]); targets.extend([i1,ki,i2,i0])
        sample_groups.extend([gn]*4)
    return E0, gi, (torch.tensor(inputs,dtype=torch.long), torch.tensor(targets,dtype=torch.long), sample_groups, k_map)

# ============================================================
# Models and training
# ============================================================
class BigramLM(torch.nn.Module):
    def __init__(self, E0):
        super().__init__(); self.E=torch.nn.Parameter(E0.clone()); self.W=torch.nn.Parameter(E0.clone())
    def forward(self, ids):
        return self.E[ids] @ self.W.T

def compute_reweighting(sample_weights, targets_t, mode='per_target'):
    """Reweight loss by inverse target frequency, then rescale to sum=1."""
    if mode == 'none':
        w = sample_weights / sample_weights.sum()
    elif mode == 'global':
        f_target = {}
        for t in targets_t.tolist(): f_target[t] = f_target.get(t, 0) + 1
        w = sample_weights / torch.tensor([f_target[t] for t in targets_t.tolist()], dtype=torch.float32)
        w = w / w.sum()
    elif mode == 'per_target':
        # Each target type gets equal total weight
        unique_targets = list(set(targets_t.tolist()))
        f_target = {}
        for t in targets_t.tolist(): f_target[t] = f_target.get(t, 0) + 1
        w = sample_weights / torch.tensor([f_target[t] for t in targets_t.tolist()], dtype=torch.float32)
        w = w / w.sum()
    return w

def train_and_eval(E0, data, probs, reweight='none', max_steps=600, minibatch_size=None):
    """Train bigram model, return convergence steps and final geometry."""
    inputs_t, targets_t, sample_groups, k_map, *_ = data
    V = E0.shape[0]
    
    base_w = torch.tensor([probs[g]/4.0 for g in sample_groups], dtype=torch.float32)
    adjusted_w = compute_reweighting(base_w, targets_t, mode=reweight)
    
    model = BigramLM(E0)
    conv = {}  # first step group reaches acc=100%
    
    K_indices = list(range(k_map is not None and len(set(k_map.values())) or 0))
    # Actually, K indices are 0..num_K-1
    num_K = 0
    for t in targets_t.tolist():
        if t < (E0.shape[0] - 4*3):  # Ks are at the beginning
            num_K = max(num_K, t+1)
    
    if minibatch_size:
        # Mini-batch: sample subset each step
        n_data = len(inputs_t)
        
    for step in range(max_steps):
        if minibatch_size:
            idxs = torch.randint(0, n_data, (minibatch_size,))
            logits = model(inputs_t[idxs])
            losses = F.cross_entropy(logits, targets_t[idxs], reduction='none')
            batch_w = adjusted_w[idxs]
            (losses * batch_w / batch_w.sum()).sum().backward()
        else:
            logits = model(inputs_t)
            losses = F.cross_entropy(logits, targets_t, reduction='none')
            (losses * adjusted_w).sum().backward()
        
        gE,gW = model.E.grad.clone(), model.W.grad.clone()
        with torch.no_grad():
            # Full-batch metrics (always evaluate on all data for fairness)
            with torch.no_grad():
                full_logits = model(inputs_t)
                pred = full_logits.argmax(-1)
                correct = (pred == targets_t).float()
            
            # Per-group accuracy
            for gn in ['A','B','C','D']:
                if gn in probs:
                    gidxs = [i for i,g in enumerate(sample_groups) if g==gn]
                    acc = float(correct[gidxs].mean())
                    if acc>=1.0 and gn not in conv: conv[gn] = step+1
            
            # Per-K accuracy (as target)
            for ki in range(num_K):
                kt_idxs = [i for i,t in enumerate(targets_t.tolist()) if t==ki]
                if kt_idxs:
                    acc = float(correct[kt_idxs].mean())
                    if acc>=1.0 and f'K{ki}_target' not in conv: conv[f'K{ki}_target'] = step+1
            
            model.E -= lr*gE; model.W -= lr*gW
            
            if len(conv) >= 4 and all('K' not in k or k in conv for k in conv):
                if not any(f'K{ki}_target' not in conv for ki in range(num_K)):
                    break  # All groups and Ks converged
    
    # Final geometry
    E_final = model.E.detach().numpy()
    _, sv, _ = np.linalg.svd(E_final, full_matrices=False)
    sv_ratio = sv[0] / max(sv[1], 1e-12)
    
    return conv, {'sv': sv, 'sv_ratio': sv_ratio, 'E': E_final}

# ============================================================
# Main experiments
# ============================================================
def fmt(v):
    if v is None: return ' N/A'
    return f'{v:4d}'

print('='*80)
print('EXTENDED EXPERIMENTS')
print('='*80)

results = []

# ----- E1: Universal K, 3D, uniform & Zipf, bigram -----
for dist_name, probs in [('Uniform', {'A':0.25,'B':0.25,'C':0.25,'D':0.25}),
                          ('Zipf',    {'A':0.70,'B':0.10,'C':0.10,'D':0.10})]:
    for rew_name, rew_mode in [('no_rew', 'none'), ('rew', 'global')]:
        E0, gi, data = build_data_universal_K(dim=3)
        conv, geo = train_and_eval(E0, data, probs, reweight=rew_mode, max_steps=800)
        label = f'UniK_3D_{dist_name}_{rew_name}'
        results.append((label, conv, geo))

# ----- E2: Subset K, 3D, uniform & Zipf, bigram -----
for dist_name, probs in [('Uniform', {'A':0.25,'B':0.25,'C':0.25,'D':0.25}),
                          ('Zipf',    {'A':0.70,'B':0.10,'C':0.10,'D':0.10})]:
    for rew_name, rew_mode in [('no_rew', 'none'), ('rew', 'global')]:
        E0, gi, data = build_data_subset_K(dim=3)
        conv, geo = train_and_eval(E0, data, probs, reweight=rew_mode, max_steps=800)
        label = f'SubK_3D_{dist_name}_{rew_name}'
        results.append((label, conv, geo))

# ----- E3: K hierarchy, 3D, uniform -----
for rew_name, rew_mode in [('no_rew', 'none'), ('rew', 'global')]:
    E0, gi, data = build_data_hierarchy_K(dim=3)
    conv, geo = train_and_eval(E0, data, {'A':0.25,'B':0.25,'C':0.25,'D':0.25}, reweight=rew_mode, max_steps=800)
    label = f'KHier_3D_Uniform_{rew_name}'
    results.append((label, conv, geo))

# ----- E4: Dimension sweep, Universal K, uniform, 2D/3D/4D -----
for dim in [2, 3, 4]:
    centers = build_4group_centers(dim)
    gns = list(centers.keys())
    gv = spread_init_groups(dim, centers)
    Ks = [K_vector_neutral(dim)]
    E0, gi = build_token_table(Ks, gv, gns)
    
    inputs, targets, sample_groups = [], [], []
    for gn in gns:
        i0,i1,i2 = gi[gn]
        inputs.extend([i0,i1,0,i2]); targets.extend([i1,0,i2,i0])
        sample_groups.extend([gn]*4)
    data = (torch.tensor(inputs,dtype=torch.long), torch.tensor(targets,dtype=torch.long), sample_groups, None)
    
    for rew_name, rew_mode in [('no_rew', 'none'), ('rew', 'global')]:
        conv, geo = train_and_eval(E0, data, {'A':0.25,'B':0.25,'C':0.25,'D':0.25}, reweight=rew_mode, max_steps=600)
        label = f'DimSweep_{dim}D_Uniform_{rew_name}'
        results.append((label, conv, geo))

# ----- E5: Mini-batch effect, 2D Universal K, uniform, with/without reweighting -----
print('\n--- Mini-batch experiment ---')
for mb_size in [16, 8, 4]:  # 16=full batch, 8=half, 4=quarter
    E0, gi, data = build_data_universal_K(dim=2)
    for rew_name, rew_mode in [('no_rew', 'none'), ('rew', 'global')]:
        conv, geo = train_and_eval(E0, data, {'A':0.25,'B':0.25,'C':0.25,'D':0.25},
                                    reweight=rew_mode, max_steps=600, minibatch_size=mb_size)
        label = f'MiniBatch_mb{mb_size}_2D_{rew_name}'
        results.append((label, conv, geo))

# ============================================================
# Report
# ============================================================
print('\n' + '='*80)
print('RESULTS SUMMARY')
print('='*80)

# --- E1: Universal K 3D ---
print('\n--- E1: Universal K, 3D ---')
print(f'{"Setup":>25} {"K0_tgt":>8} {"A":>6} {"B":>6} {"C":>6} {"D":>6} {"sv_r":>8}')
for label, conv, geo in results:
    if label.startswith('UniK_3D'):
        r = [fmt(conv.get('K0_target',None)), fmt(conv.get('A',None)), fmt(conv.get('B',None)),
             fmt(conv.get('C',None)), fmt(conv.get('D',None)), f'{geo["sv_ratio"]:8.4f}']
        print(f'{label:>25} ' + ' '.join(r))

# --- E2: Subset K 3D ---
print('\n--- E2: Subset K, 3D ---')
print(f'{"Setup":>25} {"K0_tgt":>8} {"K1_tgt":>8} {"A":>6} {"B":>6} {"C":>6} {"D":>6} {"sv_r":>8}')
for label, conv, geo in results:
    if label.startswith('SubK_3D'):
        r = [fmt(conv.get('K0_target',None)), fmt(conv.get('K1_target',None)),
             fmt(conv.get('A',None)), fmt(conv.get('B',None)),
             fmt(conv.get('C',None)), fmt(conv.get('D',None)), f'{geo["sv_ratio"]:8.4f}']
        print(f'{label:>25} ' + ' '.join(r))

# --- E3: K Hierarchy ---
print('\n--- E3: K Hierarchy, 3D, Uniform ---')
print(f'{"Setup":>25} {"K0_tgt":>8} {"K1_tgt":>8} {"K2_tgt":>8} {"A":>6} {"B":>6} {"C":>6} {"D":>6} {"sv_r":>8}')
for label, conv, geo in results:
    if label.startswith('KHier'):
        r = [fmt(conv.get('K0_target',None)), fmt(conv.get('K1_target',None)), fmt(conv.get('K2_target',None)),
             fmt(conv.get('A',None)), fmt(conv.get('B',None)),
             fmt(conv.get('C',None)), fmt(conv.get('D',None)), f'{geo["sv_ratio"]:8.4f}']
        print(f'{label:>25} ' + ' '.join(r))

# --- E4: Dimension sweep ---
print('\n--- E4: Dimension Sweep, Universal K, Uniform ---')
print(f'{"Setup":>25} {"K0_tgt":>8} {"A":>6} {"B":>6} {"C":>6} {"D":>6} {"sv_r":>8}')
for label, conv, geo in results:
    if label.startswith('DimSweep'):
        r = [fmt(conv.get('K0_target',None)), fmt(conv.get('A',None)), fmt(conv.get('B',None)),
             fmt(conv.get('C',None)), fmt(conv.get('D',None)), f'{geo["sv_ratio"]:8.4f}']
        print(f'{label:>25} ' + ' '.join(r))

# --- E5: Mini-batch ---
print('\n--- E5: Mini-batch effect, 2D ---')
print(f'{"Setup":>25} {"K0_tgt":>8} {"A":>6} {"B":>6} {"C":>6} {"D":>6} {"sv_r":>8}')
for label, conv, geo in results:
    if label.startswith('MiniBatch'):
        r = [fmt(conv.get('K0_target',None)), fmt(conv.get('A',None)), fmt(conv.get('B',None)),
             fmt(conv.get('C',None)), fmt(conv.get('D',None)), f'{geo["sv_ratio"]:8.4f}']
        print(f'{label:>25} ' + ' '.join(r))
