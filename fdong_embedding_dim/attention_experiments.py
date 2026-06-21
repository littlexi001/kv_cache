#!/usr/bin/env python3
"""
Single-head attention toy: dim=4, 4 orthogonal groups.
  Clean comparison: WithK vs NoK, Uniform vs Zipf, with/without reweighting.
"""
import math, numpy as np, torch, torch.nn.functional as F

theta_deg=12.0; dim=4; lr=0.03; max_steps=2000

def rot4(v,d):
    cc=np.array(v,dtype=np.float32); cc/=np.linalg.norm(cc)
    perp=np.zeros(dim,dtype=np.float32); perp[0 if abs(cc[0])<0.9 else 1]=1.0
    perp-=float(np.dot(perp,cc))*cc; perp/=np.linalg.norm(perp)
    t=math.radians(d); return (math.cos(t)*cc+math.sin(t)*perp).astype(np.float32)

centers={'A':(1.,0.,0.,0.),'B':(0.,1.,0.,0.),'C':(0.,0.,1.,0.),'D':(0.,0.,0.,1.)}
gns=list(centers.keys())

# ---- Build E0 for NoK (12 tokens) and WithK (1 K + 12 = 13 tokens) ----
E0_noK_tokens=[]
for gn,c in centers.items():
    cc=np.array(c,dtype=np.float32); cc/=np.linalg.norm(cc)
    for off in [0.,theta_deg,-theta_deg]: E0_noK_tokens.append(rot4(c,off))
E0_noK=torch.tensor(np.stack(E0_noK_tokens).astype(np.float32))
gi_noK={gn:[list(centers.keys()).index(gn)*3+i for i in range(3)] for gn in gns}

K_init=np.ones(dim,dtype=np.float32)/math.sqrt(dim)
E0_wK_tokens=[K_init]
for gn,c in centers.items():
    cc=np.array(c,dtype=np.float32); cc/=np.linalg.norm(cc)
    for off in [0.,theta_deg,-theta_deg]: E0_wK_tokens.append(rot4(c,off))
E0_wK=torch.tensor(np.stack(E0_wK_tokens).astype(np.float32))
gi_wK={gn:[1+list(centers.keys()).index(gn)*3+i for i in range(3)] for gn in gns}

# ---- Data builders ----
def make_data_noK():
    c1,c2,targ,names,grps=[],[],[],[],[]
    for gn,(i0,i1,i2) in gi_noK.items():
        c1+=[i0,i1,i2]; c2+=[i1,i2,i0]; targ+=[i2,i0,i1]
        names+=['G0G1_G2','G1G2_G0','G2G0_G1']; grps+=[gn]*3
    return (torch.tensor(c1,dtype=torch.long),torch.tensor(c2,dtype=torch.long),
            torch.tensor(targ,dtype=torch.long),names,grps)

def make_data_withK():
    c1,c2,targ,names,grps=[],[],[],[],[]
    for gn,(i0,i1,i2) in gi_wK.items():
        ki=0
        c1+=[i0,i1,ki,i2]; c2+=[i1,ki,i2,i0]; targ+=[ki,i2,i0,i1]
        names+=['G0G1_K','G1K_G2','KG2_G0','G2G0_G1']; grps+=[gn]*4
    return (torch.tensor(c1,dtype=torch.long),torch.tensor(c2,dtype=torch.long),
            torch.tensor(targ,dtype=torch.long),names,grps)

# ---- Model ----
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

def train(E0,c1,c2,targ,names,grps,probs,alpha=0.0):
    base_w=torch.tensor([probs[g]/len([x for x in grps if x==g]) for g in grps],dtype=torch.float32)
    if alpha>0:
        ft={}
        for t in targ.tolist(): ft[t]=ft.get(t,0)+1
        w=base_w.clone()/torch.tensor([ft[t]**alpha for t in targ.tolist()],dtype=torch.float32)
        w=w/w.sum()
    else: w=base_w/base_w.sum()
    m=AttnLM(E0); conv={}
    for step in range(max_steps):
        logits=m(c1,c2); losses=F.cross_entropy(logits,targ,reduction='none')
        (losses*w).sum().backward()
        with torch.no_grad():
            for p in m.parameters():
                if p.grad is not None: p-=lr*p.grad; p.grad=None
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

print(f"dim={dim}, 4 groups on 4 orthogonal axes, attention (Wq/Wk/Wv)")
print()

for wK in [True, False]:
    if wK:
        E0=E0_wK; c1,c2,targ,names,grps=make_data_withK(); mode='WithK'
    else:
        E0=E0_noK; c1,c2,targ,names,grps=make_data_noK(); mode='NoK  '
    
    for dist_name,probs in [('Uniform',{'A':0.25,'B':0.25,'C':0.25,'D':0.25}),
                              ('Zipf',   {'A':0.70,'B':0.10,'C':0.10,'D':0.10})]:
        for alpha,rew_name in [(0.0,'no_rew'),(0.5,'soft_rew')]:
            conv,En,sv=train(E0,c1,c2,targ,names,grps,probs,alpha)
            label=f"{mode} {dist_name:>7} {rew_name:>7}"
            svr=sv[0]/max(sv[1],1e-12)
            pts=[f"{nm}={g(conv.get(nm))}" for nm in sorted(set(names))]
            grps_=[f"{gn}={g(conv.get(gn))}" for gn in sorted(set(grps))]
            print(f"{label}: [{', '.join(pts)}] | [{', '.join(grps_)}]  s1/s2={svr:.3f}")
            if wK:
                ke=En[0]; nk=En[1:].mean(0)
                cos_k=float(np.dot(ke/np.linalg.norm(ke),nk/np.linalg.norm(nk)))
                print(f"  K: |K|={np.linalg.norm(ke):.2f}, |c|={np.linalg.norm(nk):.2f}, cos(K,c)={cos_k:+.4f}")
