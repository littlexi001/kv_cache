#!/usr/bin/env python3
"""Medium-scale with realistic K frequency (~3%), not 25%."""
import math, numpy as np, torch, torch.nn.functional as F

theta_deg=12.0; dim=2; lr=0.03; n_total=50

# Groups 0-5: go through K. Groups 6-49: internal cycle only.
n_with_K = 6

# Build tokens
K_vec=np.array([0.707,0.707],dtype=np.float32)
all_v=[K_vec]; gi={}; idx=1
for gi_idx in range(n_total):
    angle=2*math.pi*gi_idx/n_total
    c=np.array([math.cos(angle),math.sin(angle)],dtype=np.float32)
    gi[gi_idx]=[idx,idx+1,idx+2]; idx+=3
    for off in [0.,theta_deg,-theta_deg]:
        t=math.radians(off)
        all_v.append(np.array([math.cos(t)*c[0]-math.sin(t)*c[1],math.sin(t)*c[0]+math.cos(t)*c[1]],dtype=np.float32))
E0=torch.tensor(np.stack(all_v))

# Bigrams
inputs,targets,sample_groups=[],[],[]
for gi_idx in range(n_total):
    i0,i1,i2=gi[gi_idx]
    if gi_idx < n_with_K:
        # With K: G0→G1, G1→K, K→G2, G2→G0
        inputs.extend([i0,i1,0,i2]); targets.extend([i1,0,i2,i0])
        sample_groups.extend([gi_idx]*4)
    else:
        # Without K: G0→G1, G1→G2, G2→G0
        inputs.extend([i0,i1,i2]); targets.extend([i1,i2,i0])
        sample_groups.extend([gi_idx]*3)

inputs_t=torch.tensor(inputs,dtype=torch.long); targets_t=torch.tensor(targets,dtype=torch.long)
n_data=len(inputs)
print(f'Total bigrams: {n_data}, K as target: {n_with_K}/{n_data}={n_with_K/n_data:.2%}')

# Uniform weight + reweight
base_w=torch.ones(n_data,dtype=torch.float32)/n_data
# Global reweight
f_target={}
for t in targets_t.tolist(): f_target[t]=f_target.get(t,0)+1
rew_w=base_w/torch.tensor([f_target[t] for t in targets_t.tolist()],dtype=torch.float32)
rew_w=rew_w/rew_w.sum()

print(f'f_target(K)={f_target.get(0,"N/A")}, K reweight factor: 1/{f_target.get(0,"N/A")}')
print(f'K total weight: no_rew={base_w[[i for i,t in enumerate(targets_t.tolist()) if t==0]].sum():.3%}, rew={rew_w[[i for i,t in enumerate(targets_t.tolist()) if t==0]].sum():.3%}')

print(f'\n{"step":>5} {"no_rew loss":>10} {"no_rew K%":>8} {"no_rew hi50":>9} {"no_rew lo75":>9} | {"rew loss":>10} {"rew K%":>8} {"rew hi50":>9} {"rew lo75":>9}')

class M(torch.nn.Module):
    def __init__(s):super().__init__();s.E=torch.nn.Parameter(E0.clone());s.W=torch.nn.Parameter(E0.clone())
    def forward(s,ids):return s.E[ids]@s.W.T

m_no=M(); m_rew=M()
for step in range(2000):
    for m,w,name in [(m_no,base_w,'no'),(m_rew,rew_w,'yes')]:
        logits=m(inputs_t); losses=F.cross_entropy(logits,targets_t,reduction='none')
        (losses*w).sum().backward()
        gE,gW=m.E.grad.clone(),m.W.grad.clone()
        with torch.no_grad(): m.E-=lr*gE; m.W-=lr*gW
    
    if step%100==0 or step<10:
        with torch.no_grad():
            pred_no=m_no(inputs_t).argmax(-1); corr_no=(pred_no==targets_t).float()
            pred_rew=m_rew(inputs_t).argmax(-1); corr_rew=(pred_rew==targets_t).float()
            k_no=float(corr_no[[i for i,t in enumerate(targets_t.tolist()) if t==0]].mean())
            k_rew=float(corr_rew[[i for i,t in enumerate(targets_t.tolist()) if t==0]].mean())
            # Group accuracies
            accs_no=[float(corr_no[[i for i,g in enumerate(sample_groups) if g==gi]].mean()) for gi in range(n_total)]
            accs_rew=[float(corr_rew[[i for i,g in enumerate(sample_groups) if g==gi]].mean()) for gi in range(n_total)]
            a_no=sorted(accs_no); a_rew=sorted(accs_rew)
            hi50_no=a_no[len(a_no)//2] if a_no else 0; lo75_no=a_no[3*len(a_no)//4] if a_no else 0
            hi50_rew=a_rew[len(a_rew)//2] if a_rew else 0; lo75_rew=a_rew[3*len(a_rew)//4] if a_rew else 0
            loss_no=float(F.cross_entropy(m_no(inputs_t),targets_t,reduction='mean'))
            loss_rew=float(F.cross_entropy(m_rew(inputs_t),targets_t,reduction='mean'))
            print(f'{step+1:5d} {loss_no:10.4f} {k_no:8.2%} {hi50_no:9.2%} {lo75_no:9.2%} | {loss_rew:10.4f} {k_rew:8.2%} {hi50_rew:9.2%} {lo75_rew:9.2%}')
