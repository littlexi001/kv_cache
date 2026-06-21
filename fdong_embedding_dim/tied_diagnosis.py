#!/usr/bin/env python3
"""Diagnosis: M-output mean bias, to_K slowness, depth effect on norm & mean bias."""
import math, numpy as np, torch, torch.nn.functional as F

theta_deg=12.0; dim=3; lr=0.03

centers={'A':(1.,0.,0.),'B':(0.,1.,0.),'C':(0.,-1.,0.),'D':(0.,0.,1.)}
def rot3(v,d):
    cc=np.array(v,dtype=np.float32); cc/=np.linalg.norm(cc)
    perp=np.zeros(3,dtype=np.float32)
    idx=0 if abs(cc[0])<0.9 else 1
    perp[idx]=1.0
    perp-=float(np.dot(perp,cc))*cc; perp/=np.linalg.norm(perp)
    t=math.radians(d)
    return (math.cos(t)*cc+math.sin(t)*perp).astype(np.float32)

all_v=[(np.ones(3,dtype=np.float32)/np.sqrt(3)).astype(np.float32)]
gi={}; idx=1
for gn,c in centers.items():
    cc=np.array(c,dtype=np.float32); cc/=np.linalg.norm(cc)
    gi[gn]=[idx,idx+1,idx+2]; idx+=3
    for off in [0.,theta_deg,-theta_deg]: all_v.append(rot3(c,off))
all_v_np=np.stack(all_v).astype(np.float32)
E0=torch.from_numpy(all_v_np)

c1,t2,targ,types,grps=[],[],[],[],[]
for gn,(i0,i1,i2) in gi.items():
    c1+=[i0,i1,0,i2]; t2+=[i1,0,i2,i0]; targ+=[0,i2,i0,i1]
    types+=['to_K','from_K','from_K2','internal']; grps+=[gn]*4
c1_t=torch.tensor(c1,dtype=torch.long); c2_t=torch.tensor(t2,dtype=torch.long)
targ_t=torch.tensor(targ,dtype=torch.long)
sw=torch.ones(len(c1),dtype=torch.float32)/len(c1)

# ======= Train tied+M =======
E=torch.nn.Parameter(E0.clone())
M=torch.nn.Parameter(torch.eye(3,dtype=torch.float32)*0.5)
for step in range(2000):
    h=E[c1_t]+E[c2_t]; hp=h@M.T; logits=hp@E.T
    losses=F.cross_entropy(logits,targ_t,reduction='none'); (losses*sw).sum().backward()
    with torch.no_grad(): E-=lr*E.grad; M-=lr*M.grad; E.grad=None; M.grad=None
En=E.detach().numpy().astype(np.float32)
k_init=np.array([0.57735,0.57735,0.57735],dtype=np.float32)

# 1) Per-type
with torch.no_grad():
    h=E[c1_t]+E[c2_t]; hp=h@M.T; logits=hp@E.T
    losses=F.cross_entropy(logits,targ_t,reduction='none')
    pred=logits.argmax(-1); correct=(pred==targ_t).float()
    print('--- Per-bigram-type ---')
    for bt in ['to_K','from_K','from_K2','internal']:
        idxs=[i for i,t in enumerate(types) if t==bt]
        print(f'  {bt:>10}: acc={float(correct[idxs].mean()):.0%}, loss={float(losses[idxs].mean()):.4f}')

# 2) M-output (h_proj) mean direction
print('\n--- M-output mean direction ---')
h_all=[]
for i in range(len(c1)):
    ei=En[c1[i]]; ej=En[t2[i]]
    h=(torch.from_numpy(ei)+torch.from_numpy(ej)).numpy().astype(np.float32)
    hp=(torch.from_numpy(h)@M.T).detach().numpy().astype(np.float32)
    h_all.append(hp)
h_all=np.stack(h_all)
centroid_all=h_all.mean(0)
print(f'  h_proj centroid norm={np.linalg.norm(centroid_all):.3f}')

# K context (G1,K) representation
k_ctx=[]
for gn,(i0,i1,i2) in gi.items():
    ei=En[i1]; ej=En[0]; h=(torch.from_numpy(ei)+torch.from_numpy(ej)).numpy().astype(np.float32)
    hp=(torch.from_numpy(h)@M.T).detach().numpy().astype(np.float32)
    k_ctx.append(hp)
k_ctx=np.stack(k_ctx).mean(0)
cos_hp=float(np.dot(k_ctx/np.linalg.norm(k_ctx),centroid_all/np.linalg.norm(centroid_all)))
print(f'  K_in_context h_proj vs h_proj centroid cos={cos_hp:+.4f}')

# E space
k_e=En[0]; nonk_e=En[1:].mean(0); k_e_dir=k_e/np.linalg.norm(k_e)
nonk_e_dir=nonk_e/np.linalg.norm(nonk_e)
print(f'\n  E space: |K|={np.linalg.norm(k_e):.3f}, cos(K,centroid)={float(np.dot(k_e_dir,nonk_e_dir)):+.4f}')

# 3) Why to_K slow? margin
print('\n--- to_K margin ---')
for gn,(i0,i1,i2) in gi.items():
    ei=En[i0]; ej=En[i1]; h=torch.from_numpy(ei)+torch.from_numpy(ej)
    hp=(h@M.T).detach().numpy().astype(np.float32)
    lK=float(hp@En[0]); l0=float(hp@En[i0]); l1=float(hp@En[i1]); l2=float(hp@En[i2])
    mc=max(l0,l1,l2)
    print(f'  {gn}: logit_K={lK:.2f}, best_comp={mc:.2f}, margin={lK-mc:+.2f}')

# 4) Depth: norm growth + mean bias
print('\n--- Depth vs norm & mean bias ---')
for n_layers in [1,2,4]:
    E2=torch.nn.Parameter(E0.clone())
    Ms=[torch.nn.Parameter(torch.eye(3,dtype=torch.float32)*0.5) for _ in range(n_layers)]
    params=[E2]+Ms
    for step in range(1500):
        h=E2[c1_t]+E2[c2_t]
        for M_i in Ms: h=torch.relu(h@M_i.T)
        logits=h@E2.T
        losses=F.cross_entropy(logits,targ_t,reduction='none'); (losses*sw).sum().backward()
        with torch.no_grad():
            for p in params:
                if p.grad is not None: p-=lr*p.grad; p.grad=None
    E2n=E2.detach().numpy().astype(np.float32)
    k_n=float(np.linalg.norm(E2n[0]))
    g_n=float(np.mean([np.linalg.norm(E2n[gi[gn]].mean(0)) for gn in gi]))
    kd=E2n[0]/k_n; nd=E2n[1:].mean(0); nd/=np.linalg.norm(nd)
    angle=math.degrees(math.acos(np.clip(np.dot(k_init,nd),-1,1)))
    print(f'  {n_layers} layers: |K|={k_n:.2f}, |grp_avg|={g_n:.2f}, '
          f'cos(K,centroid)={float(np.dot(kd,nd)):+.4f}, ang(K_init,c)={angle:.1f}°')
