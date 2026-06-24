#!/usr/bin/env python3
"""Simplified medium-scale test: track learning trajectory, not just convergence."""
import math, numpy as np
import torch, torch.nn.functional as F

theta_deg=12.0; lr=0.03

# ---- Setup: 50 groups, 2D universal K ----
n_total=50; n_high=10; n_low=40; dim=2
K_vec=np.array([0.707,0.707],dtype=np.float32)
all_v=[K_vec]; gi={}; idx=1
for gi_idx in range(n_total):
    angle=2*math.pi*gi_idx/n_total; c=np.array([math.cos(angle),math.sin(angle)],dtype=np.float32)
    gi[gi_idx]=[idx,idx+1,idx+2]
    for off in [0.,theta_deg,-theta_deg]:
        t=math.radians(off)
        all_v.append(np.array([math.cos(t)*c[0]-math.sin(t)*c[1],math.sin(t)*c[0]+math.cos(t)*c[1]],dtype=np.float32))
        idx+=1
E0=torch.tensor(np.stack(all_v))

inputs,targets,sample_groups=[],[],[]
for gi_idx in range(n_total):
    i0,i1,i2=gi[gi_idx]
    inputs.extend([i0,i1,0,i2]); targets.extend([i1,0,i2,i0])
    sample_groups.extend([gi_idx]*4)
inputs_t=torch.tensor(inputs,dtype=torch.long); targets_t=torch.tensor(targets,dtype=torch.long)
n_data=len(inputs)

per_high=0.90/n_high/4; per_low=0.10/n_low/4
base_w=torch.tensor([per_high if g<n_high else per_low for g in sample_groups],dtype=torch.float32)

def reweight(w):
    f={}; [f.update({t:f.get(t,0)+1}) for t in targets_t.tolist()]
    w2=w/torch.tensor([f[t] for t in targets_t.tolist()],dtype=torch.float32)
    return w2/w2.sum()

print('='*70)
print('MEDIUM-SCALE: 50 groups (10 high=90%, 40 low=10%), 200 bigrams')
print('Tracking accuracy quartiles at fixed steps (not full convergence)')
print('='*70)

for rew_name, rew_w in [('no_rew', base_w/base_w.sum()), ('rew', reweight(base_w))]:
    print(f'\n--- {rew_name} ---')
    print(f'{">":>5} {"loss":>8} {"K_acc":>7} {"highQ1":>8} {"highmed":>8} {"lowQ1":>8} {"lows4":>8}')
    
    class M(torch.nn.Module):
        def __init__(s): super().__init__(); s.E=torch.nn.Parameter(E0.clone()); s.W=torch.nn.Parameter(E0.clone())
        def forward(s,ids): return s.E[ids]@s.W.T
    m=M()
    
    for step in range(2000):
        logits=m(inputs_t); losses=F.cross_entropy(logits,targets_t,reduction='none')
        (losses*rew_w).sum().backward(); gE,gW=m.E.grad.clone(),m.W.grad.clone()
        with torch.no_grad():
            m.E-=lr*gE; m.W-=lr*gW
        
        if step in [0,9,49,99,199,399,799,1499,1999]:
            with torch.no_grad():
                pred=m(inputs_t).argmax(-1); correct=(pred==targets_t).float()
                # Per-group accuracy
                accs_high=[float(correct[[i for i,g in enumerate(sample_groups) if g==gi]].mean()) for gi in range(n_high)]
                accs_low=[float(correct[[i for i,g in enumerate(sample_groups) if g==gi]].mean()) for gi in range(n_high,n_total)]
                # K accuracy (target)
                k_acc=float(correct[[i for i,t in enumerate(targets_t.tolist()) if t==0]].mean())
                # Quartiles
                ah=sorted(accs_high); al=sorted(accs_low)
                hq1=ah[len(ah)//4] if ah else 0; hm=ah[len(ah)//2] if ah else 0
                lq1=al[len(al)//4] if al else 0; ls4=al[3*len(al)//4] if al else 0
                print(f'{step+1:5d} {float(losses.mean()):8.4f} {k_acc:7.2%} {hq1:8.2%} {hm:8.2%} {lq1:8.2%} {ls4:8.2%}')

# ---- Quick E3 check ----
print('\n\n' + '='*70)
print('E3 QUICK CHECK: 2-layer model loss trajectory')
print('='*70)

# Just inline the 4-group setup for E3
centers2d={'A':(1.,0.),'B':(0.,1.),'C':(0.,-1.),'D':(-1.,0.)}
def rot(v,d):t=math.radians(d);return np.array([math.cos(t)*v[0]-math.sin(t)*v[1],math.sin(t)*v[0]+math.cos(t)*v[1]],dtype=np.float32)
all_v2=[np.array([0.707,0.707],dtype=np.float32)]
gi2={}; idx2=1
for gn,c in centers2d.items():
    cc=np.array(c,dtype=np.float32); cc/=np.linalg.norm(cc)
    gi2[gn]=[idx2,idx2+1,idx2+2]; idx2+=3
    for off in [0.,theta_deg,-theta_deg]: all_v2.append(rot(cc,off))
E0_2d=torch.tensor(np.stack(all_v2))

ins,tars,sgs=[],[],[]
for gn,(i0,i1,i2) in gi2.items():
    ins.extend([i0,i1,0,i2]); tars.extend([i1,0,i2,i0]); sgs.extend([gn]*4)
ins_t=torch.tensor(ins,dtype=torch.long); tars_t=torch.tensor(tars,dtype=torch.long)
bw=torch.ones(len(ins),dtype=torch.float32)/len(ins)

for rew_name, rew_w in [('no_rew',bw/bw.sum()), ('rew',reweight(bw.clone()))]:
    class TL(torch.nn.Module):
        def __init__(s):super().__init__();s.E=torch.nn.Parameter(E0_2d.clone());s.W=torch.nn.Parameter(E0_2d.clone());s.L1=torch.nn.Parameter(torch.empty(2,8));s.L2=torch.nn.Parameter(torch.empty(8,2));torch.nn.init.xavier_uniform_(s.L1);torch.nn.init.xavier_uniform_(s.L2)
        def forward(s,ids):h=s.E[ids];h=F.relu(h@s.L1);h=h@s.L2;return h@s.W.T
    m=TL()
    print(f'\n2Layer {rew_name}:')
    for step in range(500):
        logits=m(ins_t);losses=F.cross_entropy(logits,tars_t,reduction='none');(losses*rew_w).sum().backward()
        with torch.no_grad():
            for p in m.parameters():
                if p.grad is not None:p-=lr*p.grad
            m.zero_grad()
        if step%50==0 or step<5:
            with torch.no_grad():
                p=m(ins_t).argmax(-1);acc=float((p==tars_t).float().mean())
            print(f'  step{step+1:5d}: loss={float(losses.mean()):.4f}, acc={acc:.2%}')
