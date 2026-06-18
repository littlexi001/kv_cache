#!/usr/bin/env python3
"""K-token: separate accuracy for K-as-target vs K-as-input."""
import math, numpy as np, torch, torch.nn.functional as F

theta_deg=12.0; dim=2; lr=0.03
def rot(v,d):
    t=math.radians(d);return np.array([math.cos(t)*v[0]-math.sin(t)*v[1],math.sin(t)*v[0]+math.cos(t)*v[1]],dtype=np.float32)

group_centers={'A':(1.,0.),'B':(0.,1.),'C':(0.,-1.),'D':(-1.,0.)}
token_vecs=[np.array([0.707,0.707],dtype=np.float32)]
group_indices={}
idx=1
for gn,center in group_centers.items():
    group_indices[gn]=[idx, idx+1, idx+2]
    token_vecs.extend([rot(center,0.),rot(center,theta_deg),rot(center,-theta_deg)])
    idx+=3
E0=torch.tensor(np.stack(token_vecs));W0=E0.clone()

# Bigrams: G0→G1, G1→K, K→G2, G2→G0
inputs,targets,sample_groups=[],[],[]
# Also tag each bigram by type
bigram_labels=[]  # 'internal', 'to_K', 'from_K'
for gn,(i0,i1,i2) in group_indices.items():
    inputs.extend([i0,i1,0,i2]); targets.extend([i1,0,i2,i0])
    sample_groups.extend([gn,gn,gn,gn])
    bigram_labels.extend(['internal','to_K','from_K','internal'])

targets_t=torch.tensor(targets,dtype=torch.long)
inp_t=torch.tensor(inputs,dtype=torch.long)

to_K_idxs=[i for i,lab in enumerate(bigram_labels) if lab=='to_K']
from_K_idxs=[i for i,lab in enumerate(bigram_labels) if lab=='from_K']
internal_idxs=[i for i,lab in enumerate(bigram_labels) if lab=='internal']

def run(probs,label,steps=500):
    sw=torch.tensor([probs[g]/4.0 for g in sample_groups],dtype=torch.float32)
    class M(torch.nn.Module):
        def __init__(self):super().__init__();self.E=torch.nn.Parameter(E0.clone());self.W=torch.nn.Parameter(W0.clone())
        def forward(self,ids):return self.E[ids]@self.W.T
    m=M()
    conv={}  # first step each group hits 100%
    for step in range(steps):
        logits=m(inp_t);losses=F.cross_entropy(logits,targets_t,reduction='none')
        (losses*sw).sum().backward()
        gE,gW=m.E.grad.clone(),m.W.grad.clone()
        with torch.no_grad():
            pred=logits.argmax(-1);correct=(pred==targets_t).float()
            # K as TARGET accuracy (G1→K)
            k_target_acc=float(correct[to_K_idxs].mean())
            # K as INPUT accuracy (K→G2)
            k_input_acc=float(correct[from_K_idxs].mean())
            # Internal accuracy (G0→G1, G2→G0)
            internal_acc=float(correct[internal_idxs].mean())
            # Per-group
            for gn,(i0,i1,i2) in group_indices.items():
                gidxs=[i for i,g in enumerate(sample_groups) if g==gn]
                acc=float(correct[gidxs].mean()); loss_gn=float(losses[gidxs].mean())
                if acc>=1.0 and gn not in conv: conv[gn]=step+1
            if k_target_acc>=1.0 and 'K_target' not in conv: conv['K_target']=step+1
            if k_input_acc>=1.0 and 'K_input' not in conv: conv['K_input']=step+1
            m.E-=lr*gE;m.W-=lr*gW
            if all(v in conv for v in list(group_centers.keys())+['K_target','K_input']): break
    kt = conv.get('K_target', 'N/A')
    ki = conv.get('K_input', 'N/A')
    fg = min(conv.get(gn, 999) for gn in group_centers)
    print(f'\n{label}:')
    print(f'  K as TARGET (G1->K)      first acc=100% at step {kt}')
    print(f'  K as INPUT  (K->G2)      first acc=100% at step {ki}')
    print(f'  Internal  (G0->G1,G2->G0) first acc=100% at step {fg} (first group)')
    for gn in group_centers:
        cv = conv.get(gn, 'N/A')
        print(f'  Group {gn} (all 4 bigrams)  first acc=100% at step {cv}')
    return conv,k_target_acc,k_input_acc

print('='*70)
print('K-TOKEN: K-as-TARGET vs K-as-INPUT convergence')
print('='*70)

ru,_,_ = run({'A':0.25,'B':0.25,'C':0.25,'D':0.25},'Uniform',steps=800)
print()
rz,_,_ = run({'A':0.70,'B':0.10,'C':0.10,'D':0.10},'Zipf',steps=800)
