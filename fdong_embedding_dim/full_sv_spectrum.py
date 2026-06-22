#!/usr/bin/env python3
"""Full singular value spectrum: no_rew vs soft_rew vs hard_rew, 3D & 4D."""
import math, numpy as np, torch, torch.nn.functional as F

theta_deg=12.0; lr=0.03; max_steps=1500

def run_sv_analysis(dim):
    def rot(v,d):
        cc=np.array(v,dtype=np.float32); cc/=np.linalg.norm(cc)
        perp=np.zeros(dim,dtype=np.float32)
        perp[0 if abs(cc[0])<0.9 else 1]=1.0
        perp-=float(np.dot(perp,cc))*cc; perp/=np.linalg.norm(perp)
        t=math.radians(d); return (math.cos(t)*cc+math.sin(t)*perp).astype(np.float32)
    
    if dim==4:
        ct=((1.,0.,0.,0.),(0.,1.,0.,0.),(0.,0.,1.,0.),(0.,0.,0.,1.))
    else:
        ct=((1.,0.,0.),(0.,1.,0.),(0.,-1.,0.),(0.,0.,1.))
    centers={'A':ct[0][:dim],'B':ct[1][:dim],'C':ct[2][:dim],'D':ct[3][:dim]}
    gns=list(centers.keys())
    
    K_init=np.ones(dim,dtype=np.float32)/math.sqrt(dim)
    all_v=[K_init]
    for gn,c in centers.items():
        cc=np.array(c,dtype=np.float32); cc/=np.linalg.norm(cc)
        for off in [0.,theta_deg,-theta_deg]: all_v.append(rot(c,off))
    E0=torch.tensor(np.stack(all_v).astype(np.float32))
    
    gi={gn:[1+list(centers.keys()).index(gn)*3+i for i in range(3)] for gn in gns}
    c1,c2,targ=[],[],[]
    for gn,(i0,i1,i2) in gi.items():
        c1+=[i0,i1,0,i2]; c2+=[i1,0,i2,i0]; targ+=[0,i2,i0,i1]
    c1_t=torch.tensor(c1,dtype=torch.long); c2_t=torch.tensor(c2,dtype=torch.long)
    targ_t=torch.tensor(targ,dtype=torch.long)
    n_data=len(c1)
    
    ft={}
    for t in targ_t.tolist(): ft[t]=ft.get(t,0)+1
    base_w=torch.ones(n_data,dtype=torch.float32)/n_data
    no_w=base_w/base_w.sum()
    soft_w=base_w.clone()/torch.tensor([ft[t]**0.5 for t in targ_t.tolist()],dtype=torch.float32)
    soft_w=soft_w/soft_w.sum()
    hard_w=base_w.clone()/torch.tensor([ft[t] for t in targ_t.tolist()],dtype=torch.float32)
    hard_w=hard_w/hard_w.sum()
    
    sv_results={}
    for w,w_name in [(no_w,'no_rew'),(soft_w,'soft_rew'),(hard_w,'hard_rew')]:
        class A(torch.nn.Module):
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
                s=torch.bmm(Q,K.transpose(1,2))/self.scale
                return torch.bmm(F.softmax(s,dim=-1),V).squeeze(1)@self.E.T
        m=A()
        for step in range(max_steps):
            logits=m(c1_t,c2_t); losses=F.cross_entropy(logits,targ_t,reduction='none')
            (losses*w).sum().backward()
            with torch.no_grad():
                for p in m.parameters():
                    if p.grad is not None: p-=lr*p.grad; p.grad=None
        En=m.E.detach().numpy()
        _,sv,_=np.linalg.svd(En,full_matrices=False)
        sv_results[w_name]=sv
    return sv_results

print('='*70)
print('FULL SINGULAR VALUE SPECTRUM: no_rew vs soft_rew vs hard_rew')
print('='*70)

for dim in [3,4]:
    svs=run_sv_analysis(dim)
    print(f'\ndim={dim}:')
    print(f'{"":>10} {"σ₁":>8} {"σ₂":>8} {"σ₃":>8}',end='')
    if dim==4: print(' {"σ₄":>8}',end='')
    print(' {"σ₁/σ₂":>10} {"σ₁/σ₃":>10}',end='')
    if dim==4: print(' {"σ₁/σ₄":>10}',end='')
    print()
    print('-'*(30+dim*10))
    for name in ['no_rew','soft_rew','hard_rew']:
        s=svs[name]
        sig_str=f'{s[0]:8.2f} {s[1]:8.2f} {s[2]:8.2f}'
        rat_str=f'{s[0]/max(s[1],1e-12):10.3f} {s[0]/max(s[2],1e-12):10.3f}'
        if dim==4:
            sig_str+=f' {s[3]:8.2f}'
            rat_str+=f' {s[0]/max(s[3],1e-12):10.3f}'
        print(f'{name:>10} {sig_str} {rat_str}')
