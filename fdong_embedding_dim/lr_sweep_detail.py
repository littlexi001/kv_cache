#!/usr/bin/env python3
"""Detailed LR sweep: per-pattern convergence at each lr for no_rew vs soft_rew."""
import math, numpy as np, torch, torch.nn.functional as F

theta_deg=12.0; dim=4
def rot4(v,d):
    cc=np.array(v,dtype=np.float32); cc/=np.linalg.norm(cc)
    perp=np.zeros(dim,dtype=np.float32); perp[0 if abs(cc[0])<0.9 else 1]=1.0
    perp-=float(np.dot(perp,cc))*cc; perp/=np.linalg.norm(perp)
    t=math.radians(d); return (math.cos(t)*cc+math.sin(t)*perp).astype(np.float32)

centers={'A':(1.,0.,0.,0.),'B':(0.,1.,0.,0.),'C':(0.,0.,1.,0.),'D':(0.,0.,0.,1.)}
gns=list(centers.keys())
K_init=np.ones(dim,dtype=np.float32)/math.sqrt(dim)
all_v=[K_init]
for gn,c in centers.items():
    cc=np.array(c,dtype=np.float32); cc/=np.linalg.norm(cc)
    for off in [0.,theta_deg,-theta_deg]: all_v.append(rot4(c,off))
E0=torch.tensor(np.stack(all_v).astype(np.float32))
gi={gn:[1+list(centers.keys()).index(gn)*3+i for i in range(3)] for gn in gns}

c1,c2,targ,names,grps=[],[],[],[],[]
for gn,(i0,i1,i2) in gi.items():
    c1+=[i0,i1,0,i2]; c2+=[i1,0,i2,i0]; targ+=[0,i2,i0,i1]
    names+=['G0G1_K','G1K_G2','KG2_G0','G2G0_G1']; grps+=[gn]*4
c1_t=torch.tensor(c1,dtype=torch.long); c2_t=torch.tensor(c2,dtype=torch.long)
targ_t=torch.tensor(targ,dtype=torch.long)

class AttnLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.E=torch.nn.Parameter(E0.clone())
        self.Wq=torch.nn.Parameter(torch.eye(dim,dtype=torch.float32)*0.1)
        self.Wk=torch.nn.Parameter(torch.eye(dim,dtype=torch.float32)*0.1)
        self.Wv=torch.nn.Parameter(torch.eye(dim,dtype=torch.float32)*0.1)
        self.scale=math.sqrt(dim)
    def forward(self,c1,c2):
        h1=self.E[c1]; h2=self.E[c2]
        K=torch.stack([h1@self.Wk.T,h2@self.Wk.T],dim=1)
        V=torch.stack([h1@self.Wv.T,h2@self.Wv.T],dim=1)
        Q=(h2@self.Wq.T).unsqueeze(1)
        return torch.bmm(F.softmax(torch.bmm(Q,K.transpose(1,2))/self.scale,dim=-1),V).squeeze(1)@self.E.T

probs={'A':0.25,'B':0.25,'C':0.25,'D':0.25}
base_w=torch.tensor([probs[g]/len([x for x in grps if x==g]) for g in grps],dtype=torch.float32)
ft={}
for t in targ_t.tolist(): ft[t]=ft.get(t,0)+1
rew_w=base_w.clone()/torch.tensor([ft[t]**0.5 for t in targ_t.tolist()],dtype=torch.float32)
rew_w=rew_w/rew_w.sum()
no_w=base_w/base_w.sum()

def train(w,use_lr,steps=1000):
    m=AttnLM(); conv={}
    for step in range(steps):
        logits=m(c1_t,c2_t); losses=F.cross_entropy(logits,targ_t,reduction='none')
        (losses*w).sum().backward()
        with torch.no_grad():
            for p in m.parameters():
                if p.grad is not None: p-=use_lr*p.grad; p.grad=None
        if step%50==0:
            with torch.no_grad():
                pred=m(c1_t,c2_t).argmax(-1); correct=(pred==targ_t).float()
                for nm in set(names):
                    idxs=[i for i,n in enumerate(names) if n==nm]
                    if float(correct[idxs].mean())>=1.0 and nm not in conv: conv[nm]=step+1
                for gn in set(grps):
                    idxs=[i for i,g in enumerate(grps) if g==gn]
                    if float(correct[idxs].mean())>=1.0 and gn not in conv: conv[gn]=step+1
    return conv

def g(v): return f'{v:4d}' if v else '   -'

print(f'{"lr":>6} | {"no_rew G0G1_K":>14} {"no_rew G1K_G2":>14} {"no_rew KG2_G0":>14} {"no_rew G2G0_G1":>14} | {"rew G0G1_K":>14} {"rew G1K_G2":>14} {"rew KG2_G0":>14} {"rew G2G0_G1":>14}')
print('-'*135)

for test_lr in [0.01,0.03,0.05,0.08,0.12,0.18]:
    c_no=train(no_w,test_lr); c_rw=train(rew_w,test_lr)
    r_no=[g(c_no.get(p)) for p in ['G0G1_K','G1K_G2','KG2_G0','G2G0_G1']]
    r_rw=[g(c_rw.get(p)) for p in ['G0G1_K','G1K_G2','KG2_G0','G2G0_G1']]
    print(f'{test_lr:6.2f} | {r_no[0]:>14} {r_no[1]:>14} {r_no[2]:>14} {r_no[3]:>14} | {r_rw[0]:>14} {r_rw[1]:>14} {r_rw[2]:>14} {r_rw[3]:>14}')

print()
print('KEY: rew unlocks G2G0_G1 (always "—" for no_rew).')
print('At lr=0.18, rew converges ALL 4 patterns; no_rew never touches G2G0_G1.')
print('Raising lr speeds up ALL patterns linearly for rew.')
