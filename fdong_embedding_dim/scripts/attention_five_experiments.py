#!/usr/bin/env python3
"""
Five extended experiments on dim=4 attention model.
  E1: Multi-K spectral occupation (E space & attention output)
  E2: Multi-K + group convergence vs K frequency
  E3: Single K + uniform/zipf group convergence comparison  
  E4: Soft reweighting alpha sweep in high-dim attention
  E5: Learning rate sweep (with/without reweighting)
"""
import math, numpy as np, torch, torch.nn.functional as F

theta_deg=12.0; dim=4; lr=0.03; max_steps=1500

def rot4(v,d):
    cc=np.array(v,dtype=np.float32); cc/=np.linalg.norm(cc)
    perp=np.zeros(dim,dtype=np.float32); perp[0 if abs(cc[0])<0.9 else 1]=1.0
    perp-=float(np.dot(perp,cc))*cc; perp/=np.linalg.norm(perp)
    t=math.radians(d); return (math.cos(t)*cc+math.sin(t)*perp).astype(np.float32)

centers={'A':(1.,0.,0.,0.),'B':(0.,1.,0.,0.),'C':(0.,0.,1.,0.),'D':(0.,0.,0.,1.)}
gns=list(centers.keys())

def make_E0_and_gi(K_vectors, k_map):
    """K_vectors: list of K init vectors. k_map: group->K_idx."""
    all_v=list(K_vectors)
    gi={}; idx=len(K_vectors)
    for gn,c in centers.items():
        cc=np.array(c,dtype=np.float32); cc/=np.linalg.norm(cc)
        gi[gn]=[idx,idx+1,idx+2]; idx+=3
        for off in [0.,theta_deg,-theta_deg]: all_v.append(rot4(c,off))
    return torch.tensor(np.stack(all_v).astype(np.float32)), gi, k_map

def make_trigram_data(gi, k_map):
    c1,c2,targ,names,grps=[],[],[],[],[]
    for gn,(i0,i1,i2) in gi.items():
        ki=k_map[gn]
        c1+=[i0,i1,ki,i2]; c2+=[i1,ki,i2,i0]; targ+=[ki,i2,i0,i1]
        names+=['G0G1_K','G1K_G2','KG2_G0','G2G0_G1']; grps+=[gn]*4
    return (torch.tensor(c1,dtype=torch.long),torch.tensor(c2,dtype=torch.long),
            torch.tensor(targ,dtype=torch.long),names,grps)

class AttnLM(torch.nn.Module):
    def __init__(self,E0):
        super().__init__(); self.E=torch.nn.Parameter(E0.clone())
        self.Wq=torch.nn.Parameter(torch.eye(dim,dtype=torch.float32)*0.1)
        self.Wk=torch.nn.Parameter(torch.eye(dim,dtype=torch.float32)*0.1)
        self.Wv=torch.nn.Parameter(torch.eye(dim,dtype=torch.float32)*0.1)
        self.scale=math.sqrt(dim)
    def forward(self,c1,c2):
        h1=self.E[c1]; h2=self.E[c2]
        K=torch.stack([h1@self.Wk.T,h2@self.Wk.T],dim=1)
        V=torch.stack([h1@self.Wv.T,h2@self.Wv.T],dim=1)
        Q=(h2@self.Wq.T).unsqueeze(1)
        s=torch.bmm(Q,K.transpose(1,2))/self.scale
        return torch.bmm(F.softmax(s,dim=-1),V).squeeze(1)@self.E.T

def g(v): return str(v) if v else '-'

def train(E0,c1,c2,targ,names,grps,probs,alpha=0.0,steps=max_steps,override_lr=None):
    use_lr=override_lr if override_lr else lr
    base_w=torch.tensor([probs[g]/len([x for x in grps if x==g]) for g in grps],dtype=torch.float32)
    if alpha>0:
        ft={}
        for t in targ.tolist(): ft[t]=ft.get(t,0)+1
        w=base_w.clone()/torch.tensor([ft[t]**alpha for t in targ.tolist()],dtype=torch.float32)
        w=w/w.sum()
    else: w=base_w/base_w.sum()
    m=AttnLM(E0); conv={}
    for step in range(steps):
        logits=m(c1,c2); losses=F.cross_entropy(logits,targ,reduction='none')
        (losses*w).sum().backward()
        with torch.no_grad():
            for p in m.parameters():
                if p.grad is not None: p-=use_lr*p.grad; p.grad=None
        if step%100==0:
            with torch.no_grad():
                pred=m(c1,c2).argmax(-1); correct=(pred==targ).float()
                for nm in set(names):
                    idxs=[i for i,n in enumerate(names) if n==nm]
                    if float(correct[idxs].mean())>=1.0 and nm not in conv: conv[nm]=step+1
                for gn in set(grps):
                    idxs=[i for i,g in enumerate(grps) if g==gn]
                    if float(correct[idxs].mean())>=1.0 and gn not in conv: conv[gn]=step+1
    En=m.E.detach().numpy(); _,sv,_=np.linalg.svd(En,full_matrices=False)
    return conv,En,sv

probs_uni={'A':0.25,'B':0.25,'C':0.25,'D':0.25}
probs_zipf={'A':0.70,'B':0.10,'C':0.10,'D':0.10}

# ================================================================
# E1: Multi-K spectral occupation
# ================================================================
print('='*80)
print('E1: MULTI-K SPECTRAL OCCUPATION')
print('='*80)
K_init=np.ones(dim,dtype=np.float32)/math.sqrt(dim)
# 2 Ks: K0 connects A+B, K1 connects C+D
K0=np.array([0.5,0.5,0.0,0.0],dtype=np.float32); K0/=np.linalg.norm(K0)
K1=np.array([0.0,0.0,0.5,0.5],dtype=np.float32); K1/=np.linalg.norm(K1)
for n_ks,Ks,k_map in [(2,[K0,K1],{'A':0,'B':0,'C':1,'D':1})]:
    E0,gi,km=make_E0_and_gi(Ks,k_map)
    c1,c2,targ,names,grps=make_trigram_data(gi,km)
    conv,En,sv=train(E0,c1,c2,targ,names,grps,probs_uni,alpha=0.5)
    _,S,_=np.linalg.svd(En,full_matrices=False)
    print(f'  2 Ks (K0=A+B, K1=C+D): σ=[{S[0]:.2f},{S[1]:.2f},{S[2]:.2f},{S[3]:.2f}]')
    
    # Project each K onto top singular directions (use Vh = right singular vectors)
    U,S,Vh=np.linalg.svd(En,full_matrices=False)  # Vh shape (4,4) for 4D embeddings
    for ki in range(2):
        k_vec=En[ki]; proj_top1=float(np.dot(k_vec,Vh[0,:])); proj_top2=float(np.dot(k_vec,Vh[1,:]))
        print(f'    K{ki}: |K|={np.linalg.norm(k_vec):.2f}, proj on σ1={proj_top1:.3f}, σ2={proj_top2:.3f}')
    
    # Convergence
    for nm in sorted(set(names)):
        print(f'    {nm}={g(conv.get(nm))}')

# ================================================================
# E2: Multi-K + group convergence vs K frequency
# ================================================================
print('\n'+'='*80)
print('E2: MULTI-K: GROUP CONVERGENCE vs K FREQUENCY')
print('='*80)
# K0 (high freq): connects A+B+C, K1 (mid): connects D
K_hi=np.array([0.6,0.4,0.3,0.0],dtype=np.float32); K_hi/=np.linalg.norm(K_hi)
K_lo=np.array([0.0,0.0,0.3,0.6],dtype=np.float32); K_lo/=np.linalg.norm(K_lo)
k_hier={'A':0,'B':0,'C':0,'D':1}  # K0=hi freq (A+B+C), K1=lo freq (D)

E0,gi,km=make_E0_and_gi([K_hi,K_lo],k_hier)
c1,c2,targ,names,grps=make_trigram_data(gi,km)
for alpha,rew_name in [(0.0,'no_rew'),(0.5,'soft_rew')]:
    conv,En,sv=train(E0,c1,c2,targ,names,grps,probs_uni,alpha=alpha)
    svr=sv[0]/max(sv[1],1e-12)
    print(f'  {rew_name}: G0G1_K={g(conv.get("G0G1_K"))}, G1K_G2={g(conv.get("G1K_G2"))}, '
          f'G2G0_G1={g(conv.get("G2G0_G1"))}, KG2_G0={g(conv.get("KG2_G0"))} | '
          f'A={g(conv.get("A"))},B={g(conv.get("B"))},C={g(conv.get("C"))},D={g(conv.get("D"))} s1/s2={svr:.3f}')
    k0_n=float(np.linalg.norm(En[0])); k1_n=float(np.linalg.norm(En[1]))
    print(f'    |K0(hi_freq)|={k0_n:.2f}, |K1(lo_freq)|={k1_n:.2f}')

# ================================================================
# E3: Single K, uniform vs zipf, group convergence
# ================================================================
print('\n'+'='*80)
print('E3: GROUP CONVERGENCE: Uniform vs Zipf (single K)')
print('='*80)
K_single=[np.ones(dim,dtype=np.float32)/math.sqrt(dim)]
k_single={'A':0,'B':0,'C':0,'D':0}
E0,gi,km=make_E0_and_gi(K_single,k_single)
c1,c2,targ,names,grps=make_trigram_data(gi,km)

for dist_name,probs in [('Uniform',probs_uni),('Zipf',probs_zipf)]:
    for alpha,rew_name in [(0.0,'no_rew'),(0.5,'soft_rew')]:
        conv,En,sv=train(E0,c1,c2,targ,names,grps,probs,alpha=alpha)
        svr=sv[0]/max(sv[1],1e-12)
        grp_cols=' '.join(f'{gn}={g(conv.get(gn))}' for gn in gns)
        print(f'  {dist_name:>7} {rew_name:>7}: A={g(conv.get("A"))},B={g(conv.get("B"))},'
              f'C={g(conv.get("C"))},D={g(conv.get("D"))} s1/s2={svr:.3f}')

# ================================================================
# E4: Alpha sweep
# ================================================================
print('\n'+'='*80)
print('E4: SOFT REWEIGHTING ALPHA SWEEP (Single K, Uniform)')
print('='*80)
for alpha in [0.0,0.2,0.3,0.5,0.7,1.0]:
    conv,En,sv=train(E0,c1,c2,targ,names,grps,probs_uni,alpha=alpha)
    svr=sv[0]/max(sv[1],1e-12)
    n_conv=sum(1 for v in conv.values() if v)
    ke=En[0]; nk=En[1:].mean(0)
    cos_k=float(np.dot(ke/np.linalg.norm(ke),nk/np.linalg.norm(nk)))
    print(f'  α={alpha:.1f}: G2G0_G1={g(conv.get("G2G0_G1"))}, KG2_G0={g(conv.get("KG2_G0"))}, '
          f'G0G1_K={g(conv.get("G0G1_K"))}, |K|={np.linalg.norm(ke):.2f}, '
          f'cos(K,c)={cos_k:+.4f}, s1/s2={svr:.3f}, #converged={n_conv}')

# ================================================================
# E5: Learning rate sweep
# ================================================================
print('\n'+'='*80)
print('E5: LEARNING RATE SWEEP (with/without reweighting)')
print('='*80)
print(f'  {"lr":>6} | {"no_rew converged":>30} | {"soft_rew converged":>30}')
for test_lr in [0.01,0.03,0.05,0.08,0.12,0.18]:
    conv_no,_,_=train(E0,c1,c2,targ,names,grps,probs_uni,alpha=0.0,steps=1000,override_lr=test_lr)
    conv_rw,_,_=train(E0,c1,c2,targ,names,grps,probs_uni,alpha=0.5,steps=1000,override_lr=test_lr)
    no_str=' '.join(f'{nm}={g(conv_no.get(nm))}' for nm in sorted(set(names))[:4])
    rw_str=' '.join(f'{nm}={g(conv_rw.get(nm))}' for nm in sorted(set(names))[:4])
    print(f'  {test_lr:6.2f} | {no_str:>30} | {rw_str:>30}')
