#!/usr/bin/env python3
"""
Tied+Proj+Trigram: The most realistic toy setting.
  Model: shared E (V×d), projection M (d×d), trigram context (sum of 2 embeddings).
  Questions:
    1. Is K's embedding direction the mean of all other tokens?
    2. What's the norm relationship between K and low-freq tokens?
    3. How fast does each bigram type converge? Gradient interference?
  Settings:
    A. Uniform vs Zipf group frequencies
    B. Multi-K with frequency hierarchy
"""
import math, numpy as np
import torch, torch.nn.functional as F

theta_deg = 12.0; lr = 0.03; dim = 3  # 3D for realistic separation

# ============================================================
# Data builders (trigram)
# ============================================================
def spread_init_3d():
    centers = {'A':(1.,0.,0.), 'B':(0.,1.,0.), 'C':(0.,-1.,0.), 'D':(0.,0.,1.)}
    gv = {}
    for gn, c in centers.items():
        cc = np.array(c, dtype=np.float32); cc /= np.linalg.norm(cc)
        perp = np.zeros(3, dtype=np.float32)
        if abs(cc[0]) < 0.9: perp[0] = 1.0
        else: perp[1] = 1.0
        perp -= float(np.dot(perp, cc)) * cc; perp /= np.linalg.norm(perp)
        gv[gn] = []
        for off in [0., theta_deg, -theta_deg]:
            t = math.radians(off)
            v = math.cos(t)*cc + math.sin(t)*perp
            gv[gn].append(v.astype(np.float32))
    return centers, gv

def build_trigram_data(K_vectors, k_map):
    """K_vectors: list of K embedding init vectors. k_map: group->K index."""
    centers, gv = spread_init_3d()
    gns = list(centers.keys())
    
    all_v = list(K_vectors)
    gi = {}; idx = len(K_vectors)
    for gn in gns:
        gi[gn] = [idx, idx+1, idx+2]
        all_v.extend(gv[gn]); idx += 3
    E0 = torch.tensor(np.stack(all_v), dtype=torch.float32)
    
    # Trigram triples: (t-2, t-1) -> t
    # Per group: (G0,G1)->K, (G1,K)->G2, (K,G2)->G0, (G2,G0)->G1
    c1_list, c2_list, target_list, bigram_types, sample_groups = [], [], [], [], []
    for gn in gns:
        i0,i1,i2 = gi[gn]; ki = k_map[gn]
        # (G0, G1) -> K
        c1_list.append(i0); c2_list.append(i1); target_list.append(ki)
        bigram_types.append('to_K'); sample_groups.append(gn)
        # (G1, K) -> G2
        c1_list.append(i1); c2_list.append(ki); target_list.append(i2)
        bigram_types.append('from_K'); sample_groups.append(gn)
        # (K, G2) -> G0
        c1_list.append(ki); c2_list.append(i2); target_list.append(i0)
        bigram_types.append('from_K2'); sample_groups.append(gn)
        # (G2, G0) -> G1
        c1_list.append(i2); c2_list.append(i0); target_list.append(i1)
        bigram_types.append('internal'); sample_groups.append(gn)
    
    c1_t = torch.tensor(c1_list, dtype=torch.long)
    c2_t = torch.tensor(c2_list, dtype=torch.long)
    targ_t = torch.tensor(target_list, dtype=torch.long)
    return E0, gi, c1_t, c2_t, targ_t, bigram_types, sample_groups

def K_init_universal():
    v = np.ones(3, dtype=np.float32); return [v / np.linalg.norm(v)]

def k_init_subset():
    k1 = np.array([0.5, 0.5, 0.0], dtype=np.float32)
    k2 = np.array([0.0, -0.5, 0.5], dtype=np.float32)
    k1 /= np.linalg.norm(k1); k2 /= np.linalg.norm(k2)
    return [k1, k2]

def k_init_hierarchy():
    k1 = np.array([0.5, -0.3, 0.0], dtype=np.float32)
    k2 = np.array([0.0, 0.3, 0.5], dtype=np.float32)
    k3 = np.array([0.7, 0.0, 0.0], dtype=np.float32)
    for k in [k1,k2,k3]: k /= np.linalg.norm(k)
    return [k1, k2, k3]

# ============================================================
# Model: Tied E + Projection M
# ============================================================
class TiedProjTrigramLM(torch.nn.Module):
    def __init__(self, E0):
        super().__init__()
        self.E = torch.nn.Parameter(E0.clone())
        self.M = torch.nn.Parameter(torch.eye(E0.shape[1], dtype=torch.float32) * 0.5)
    def forward(self, c1, c2):
        h = self.E[c1] + self.E[c2]
        h_proj = h @ self.M.T
        return h_proj @ self.E.T

# ============================================================
# Training + analysis
# ============================================================
def train_and_analyze(E0, c1, c2, targ, types, groups, probs, reweight_alpha, max_steps=1500, label=''):
    """Train tied trigram model, return comprehensive metrics."""
    base_w = torch.tensor([probs[g]/4.0 for g in groups], dtype=torch.float32)
    
    if reweight_alpha > 0:
        f_target = {}
        for t in targ.tolist(): f_target[t] = f_target.get(t, 0) + 1
        w = base_w.clone() / torch.tensor([f_target[t]**reweight_alpha for t in targ.tolist()], dtype=torch.float32)
        w = w / w.sum()
    else:
        w = base_w / base_w.sum()
    
    model = TiedProjTrigramLM(E0)
    conv = {}
    history = []
    
    for step in range(max_steps):
        logits = model(c1, c2)
        losses = F.cross_entropy(logits, targ, reduction='none')
        (losses * w).sum().backward()
        
        gE = model.E.grad.clone(); gM = model.M.grad.clone()
        with torch.no_grad():
            model.E -= lr * gE; model.M -= lr * gM
            model.E.grad = None; model.M.grad = None
        
        # Sample metrics
        if step % 100 == 0 or step < 5 or step == max_steps - 1:
            with torch.no_grad():
                pred = model(c1, c2).argmax(-1); correct = (pred == targ).float()
                loss_val = float(F.cross_entropy(model(c1,c2), targ, reduction='mean'))
                
                # Per-group accuracy
                group_accs = {}
                for gn in set(groups):
                    gidxs = [i for i,g in enumerate(groups) if g == gn]
                    group_accs[gn] = float(correct[gidxs].mean())
                
                # Per-bigram-type accuracy
                type_accs = {}
                for bt in set(types):
                    tidxs = [i for i,t in enumerate(types) if t == bt]
                    type_accs[bt] = float(correct[tidxs].mean())
                
                history.append({'step': step, 'loss': loss_val, 'group_acc': group_accs, 'type_acc': type_accs})
                
                # Convergence tracking
                for gn in set(groups):
                    if group_accs[gn] >= 1.0 and gn not in conv: conv[gn] = step+1
                for bt in set(types):
                    if type_accs[bt] >= 1.0 and bt not in conv: conv[bt] = step+1
    
    # Final geometry
    E_final = model.E.detach().numpy()
    _, sv, _ = np.linalg.svd(E_final, full_matrices=False)
    
    return conv, E_final, sv, history

# ============================================================
# Run experiments
# ============================================================
print('='*80)
print('TIED+PROJ+TRIGRAM: Comprehensive Experiments')
print('='*80)

all_results = []

# --- Exp A: Universal K, Uniform vs Zipf, with/without soft reweighting ---
print('\n--- A: Universal K, Uniform vs Zipf ---')
for dist_name, probs in [('Uniform', {'A':0.25,'B':0.25,'C':0.25,'D':0.25}),
                          ('Zipf',    {'A':0.70,'B':0.10,'C':0.10,'D':0.10})]:
    for alpha, rew_name in [(0.0, 'no_rew'), (0.5, 'soft_rew')]:
        E0, gi, c1, c2, targ, types, groups = build_trigram_data(K_init_universal(), {'A':0,'B':0,'C':0,'D':0})
        conv, E, sv, hist = train_and_analyze(E0, c1, c2, targ, types, groups, probs, alpha, label=f'UniK_{dist_name}_{rew_name}')
        
        # Geometry
        k_dir = E[0] / np.linalg.norm(E[0])
        nonk_centroid = E[1:].mean(axis=0)
        nonk_dir = nonk_centroid / max(np.linalg.norm(nonk_centroid), 1e-12)
        k_init = K_init_universal()[0]; k_init /= np.linalg.norm(k_init)
        angle = math.degrees(math.acos(np.clip(np.dot(k_init, nonk_dir), -1, 1)))
        
        all_results.append({
            'label': f'UniK_{dist_name}_{rew_name}',
            'conv': conv, 'sv': sv, 'angle': angle,
            'k_norm': float(np.linalg.norm(E[0])),
            'avg_grp_norm': float(np.mean([np.linalg.norm(E[gi[gn]].mean(0)) for gn in gi])),
        })
        
        print(f'  {dist_name:>8} {rew_name:>8}: K_norm={np.linalg.norm(E[0]):.2f}, avg_grp_norm={np.mean([np.linalg.norm(E[gi[gn]].mean(0)) for gn in gi]):.2f}, '
              f'K-init vs centroid angle={angle:.1f}°, σ1/σ2={sv[0]/max(sv[1],1e-12):.3f}, '
              f'K→G2_acc={hist[-1]["type_acc"].get("from_K",0):.0%}')

# --- Exp B: Multi-K with frequency hierarchy, Uniform ---
print('\n--- B: Multi-K Hierarchy, Uniform ---')
k_map_hier = {'A':0, 'B':0, 'C':1, 'D':2}  # K0=A+B, K1=C, K2=D
for alpha, rew_name in [(0.0, 'no_rew'), (0.5, 'soft_rew')]:
    E0, gi, c1, c2, targ, types, groups = build_trigram_data(k_init_hierarchy(), k_map_hier)
    conv, E, sv, hist = train_and_analyze(E0, c1, c2, targ, types, groups, 
                                           {'A':0.25,'B':0.25,'C':0.25,'D':0.25}, alpha, label=f'Hier_{rew_name}')
    all_results.append({'label': f'Hier_Uni_{rew_name}', 'conv': conv, 'sv': sv, 'k_norms': [float(np.linalg.norm(E[i])) for i in range(3)]})
    k0_acc = hist[-1]['type_acc'].get('to_K', 0)
    f_target_0 = sum(1 for t in targ.tolist() if t==0)
    print(f'  {rew_name:>8}: K0(tgt_freq={f_target_0})_to_K_acc={k0_acc:.0%}, σ1/σ2={sv[0]/max(sv[1],1e-12):.3f}')

# --- Summary table ---
print('\n' + '='*80)
print('SUMMARY')
print('='*80)
print(f'{"Exp":>30} {"K_norm":>8} {"g_norm":>8} {"angle":>7} {"sv1/sv2":>8} {"conv":>20}')
for r in all_results:
    if 'k_norm' in r:  # Universal K results
        conv_str = '; '.join(f'{k}={v}' for k,v in r['conv'].items())
        print(f'{r["label"]:>30} {r["k_norm"]:8.2f} {r["avg_grp_norm"]:8.2f} {r["angle"]:7.1f}° {r["sv"][0]/max(r["sv"][1],1e-12):8.3f} {conv_str[:40]:>40}')
    elif 'k_norms' in r:  # Hierarchy results
        k_norms_str = ','.join(f'{n:.1f}' for n in r['k_norms'])
        conv_str = '; '.join(f'{k}={v}' for k,v in r['conv'].items())
        print(f'{r["label"]:>30} Ks=[{k_norms_str}], σ1/σ2={r["sv"][0]/max(r["sv"][1],1e-12):.3f} {conv_str[:40]:>40}')
