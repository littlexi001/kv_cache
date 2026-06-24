#!/usr/bin/env python3
"""Three experiments:
   1. Trigram baseline (normal loss)
   2. Bigram + inverse target frequency reweighting  
   3. Trigram + inverse target frequency reweighting

   All on K-token setup: G0→G1→K→G2→G0 per group, uniform groups.
"""
import math, numpy as np, torch, torch.nn.functional as F

theta_deg=12.0; dim=2; lr=0.03
group_centers={'A':(1.,0.),'B':(0.,1.),'C':(0.,-1.),'D':(-1.,0.)}

# --- Build tokens (shared) ---
token_vecs=[np.array([0.707,0.707],dtype=np.float32)]  # K
group_indices={}; idx=1
for gn,c in group_centers.items():
    def rot(v,d):
        t=math.radians(d)
        return np.array([math.cos(t)*v[0]-math.sin(t)*v[1],math.sin(t)*v[0]+math.cos(t)*v[1]],dtype=np.float32)
    group_indices[gn]=[idx,idx+1,idx+2]
    token_vecs.extend([rot(c,0.),rot(c,theta_deg),rot(c,-theta_deg)]); idx+=3
E0=torch.tensor(np.stack(token_vecs)); W0=E0.clone()

# ============================================================
# Experiment 2: Bigram + inverse frequency reweighting
# ============================================================
def run_bigram_reweighted(probs, label, max_steps=800):
    # Data: bigrams G0→G1, G1→K, K→G2, G2→G0
    inputs, targets, sample_groups = [], [], []
    for gn,(i0,i1,i2) in group_indices.items():
        inputs.extend([i0,i1,0,i2]); targets.extend([i1,0,i2,i0])
        sample_groups.extend([gn]*4)
    targets_t = torch.tensor(targets, dtype=torch.long)
    inp_t = torch.tensor(inputs, dtype=torch.long)
    
    # Base weights: prob / 4 per bigram
    base_w = torch.tensor([probs[g]/4.0 for g in sample_groups], dtype=torch.float32)
    # Compute target frequency
    f_target = {}
    for t in targets: f_target[t] = f_target.get(t, 0) + 1
    # Reweight: divide by target frequency, then rescale to sum=1
    adjusted_w = base_w / torch.tensor([f_target[t] for t in targets], dtype=torch.float32)
    adjusted_w = adjusted_w / adjusted_w.sum()  # rescale
    
    class BigramLM(torch.nn.Module):
        def __init__(self): super().__init__(); self.E=torch.nn.Parameter(E0.clone()); self.W=torch.nn.Parameter(W0.clone())
        def forward(self,ids): return self.E[ids] @ self.W.T
    
    model = BigramLM()
    conv = {}; history = []
    for step in range(max_steps):
        logits = model(inp_t)
        losses = F.cross_entropy(logits, targets_t, reduction='none')
        (losses * adjusted_w).sum().backward()
        gE,gW = model.E.grad.clone(), model.W.grad.clone()
        with torch.no_grad():
            pred = logits.argmax(-1); correct = (pred == targets_t).float()
            # Per-group
            for gn,(i0,i1,i2) in group_indices.items():
                gidxs=[i for i,g in enumerate(sample_groups) if g==gn]
                acc=float(correct[gidxs].mean())
                if acc>=1.0 and gn not in conv: conv[gn]=step+1
            # K as target / input
            to_K_idxs=[i for i,(inp,targ) in enumerate(zip(inputs,targets)) if targ==0]
            from_K_idxs=[i for i,(inp,targ) in enumerate(zip(inputs,targets)) if inp==0]
            kt_acc=float(correct[to_K_idxs].mean()); ki_acc=float(correct[from_K_idxs].mean())
            if kt_acc>=1.0 and 'K_target' not in conv: conv['K_target']=step+1
            if ki_acc>=1.0 and 'K_input' not in conv: conv['K_input']=step+1
            model.E -= lr*gE; model.W -= lr*gW
    return model, conv

# ============================================================
# Experiments 1 & 3: Trigram
# ============================================================
def build_trigram_data():
    """Build trigram triples: (t-2, t-1, t) for cyclic chain G0,G1,K,G2 per group."""
    contexts, targets, sample_groups = [], [], []
    for gn,(i0,i1,i2) in group_indices.items():
        k_idx = 0  # K is always index 0
        # (G0, G1) → K
        contexts.append((i0,i1)); targets.append(k_idx); sample_groups.append(gn)
        # (G1, K) → G2
        contexts.append((i1,k_idx)); targets.append(i2); sample_groups.append(gn)
        # (K, G2) → G0
        contexts.append((k_idx,i2)); targets.append(i0); sample_groups.append(gn)
        # (G2, G0) → G1
        contexts.append((i2,i0)); targets.append(i1); sample_groups.append(gn)
    c1 = torch.tensor([c[0] for c in contexts], dtype=torch.long)
    c2 = torch.tensor([c[1] for c in contexts], dtype=torch.long)
    targ = torch.tensor(targets, dtype=torch.long)
    return c1, c2, targ, sample_groups

def run_trigram(probs, label, reweight=False, max_steps=800):
    c1, c2, targ, sample_groups = build_trigram_data()
    
    # Base weights
    base_w = torch.tensor([probs[g]/4.0 for g in sample_groups], dtype=torch.float32)
    
    if reweight:
        f_target = {}
        for t in targ.tolist(): f_target[t] = f_target.get(t, 0) + 1
        adjusted_w = base_w / torch.tensor([f_target[t] for t in targ.tolist()], dtype=torch.float32)
        adjusted_w = adjusted_w / adjusted_w.sum()
    else:
        adjusted_w = base_w / base_w.sum()
    
    class TrigramLM(torch.nn.Module):
        def __init__(self): super().__init__(); self.E=torch.nn.Parameter(E0.clone()); self.W=torch.nn.Parameter(W0.clone())
        def forward(self, c1, c2):
            h = self.E[c1] + self.E[c2]
            return h @ self.W.T
    
    model = TrigramLM()
    conv = {}; history = []
    for step in range(max_steps):
        logits = model(c1, c2)
        losses = F.cross_entropy(logits, targ, reduction='none')
        (losses * adjusted_w).sum().backward()
        gE,gW = model.E.grad.clone(), model.W.grad.clone()
        with torch.no_grad():
            pred = logits.argmax(-1); correct = (pred == targ).float()
            for gn,(i0,i1,i2) in group_indices.items():
                gidxs=[i for i,g in enumerate(sample_groups) if g==gn]
                acc=float(correct[gidxs].mean())
                if acc>=1.0 and gn not in conv: conv[gn]=step+1
            # K as target (G0,G1→K)
            kt_idxs=[i for i,t in enumerate(targ.tolist()) if t==0]
            ki_idxs=[i for i,(x1,x2) in enumerate(zip(c1.tolist(),c2.tolist())) if x2==0]  # K is c2
            kt_acc=float(correct[kt_idxs].mean()); ki_acc=float(correct[ki_idxs].mean())
            if kt_acc>=1.0 and 'K_target' not in conv: conv['K_target']=step+1
            if ki_acc>=1.0 and 'K_input' not in conv: conv['K_input']=step+1
            model.E -= lr*gE; model.W -= lr*gW
    return model, conv

# ============================================================
# Run all three
# ============================================================
probs = {'A':0.25,'B':0.25,'C':0.25,'D':0.25}
results = {}

print('='*70)
print('EXPERIMENT 1: Trigram baseline (normal loss)')
print('='*70)
m1, c1 = run_trigram(probs, 'trigram', reweight=False)
results['trigram'] = c1
print(f'  K_target={c1.get("K_target","N/A")}, K_input={c1.get("K_input","N/A")}')
for gn in group_centers: print(f'  {gn}={c1.get(gn,"N/A")}')

print('\n' + '='*70)
print('EXPERIMENT 2: Bigram + inverse frequency reweighting')
print('='*70)
m2, c2 = run_bigram_reweighted(probs, 'bigram+rew')
results['bigram+rew'] = c2
print(f'  K_target={c2.get("K_target","N/A")}, K_input={c2.get("K_input","N/A")}')
for gn in group_centers: print(f'  {gn}={c2.get(gn,"N/A")}')

print('\n' + '='*70)
print('EXPERIMENT 3: Trigram + inverse frequency reweighting')
print('='*70)
m3, c3 = run_trigram(probs, 'trigram+rew', reweight=True)
results['trigram+rew'] = c3
print(f'  K_target={c3.get("K_target","N/A")}, K_input={c3.get("K_input","N/A")}')
for gn in group_centers: print(f'  {gn}={c3.get(gn,"N/A")}')

# ============================================================
# Comparison table
# ============================================================
print('\n' + '='*70)
print('COMPARISON: All three experiments')
print('='*70)
print(f'{"":>16} {"Bigram (orig)":>14} {"Bigram+rew":>14} {"Trigram":>14} {"Trigram+rew":>14}')
# Load original bigram results from earlier run
orig = {'K_target':43, 'K_input':'N/A', 'A':703, 'B':695, 'C':671, 'D':569}
for key in ['K_target','K_input','A','B','C','D']:
    o = orig[key]
    b = c2.get(key,'N/A')
    t = c1.get(key,'N/A')
    tr = c3.get(key,'N/A')
    print(f'{key:>16}: {str(o):>14} {str(b):>14} {str(t):>14} {str(tr):>14}')

print('\n--- KEY TAKEAWAYS ---')
print('1. Trigram: K_input should CONVERGE (context disambiguates it)')
print('2. Bigram+rew: K_target weight reduced → should slow K_target convergence relative to original bigram')
print('3. Trigram+rew: ideal case — model CAN learn K, AND reweighting prevents mean bias')
