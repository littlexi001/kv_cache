#!/usr/bin/env python3
"""Gradient analysis: does the untied-bigram mechanism still hold in tied+attention?"""
import math, numpy as np, torch, torch.nn.functional as F

theta_deg=12.0; dim=4; lr=0.03

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

# Build data per group for conditional gradient
group_data={}
for gn,(i0,i1,i2) in gi.items():
    # Group data: (G0,G1)->K, (G1,K)->G2, (K,G2)->G0, (G2,G0)->G1
    group_data[gn]=(torch.tensor([i0,i1,0,i2],dtype=torch.long),
                    torch.tensor([i1,0,i2,i0],dtype=torch.long),
                    torch.tensor([0,i2,i0,i1],dtype=torch.long))
# Also K-group data
c1_full=torch.tensor([],dtype=torch.long); c2_full=torch.tensor([],dtype=torch.long); targ_full=torch.tensor([],dtype=torch.long)
for gn in gns: c1_full=torch.cat([c1_full,group_data[gn][0]]); c2_full=torch.cat([c2_full,group_data[gn][1]]); targ_full=torch.cat([targ_full,group_data[gn][2]])

class AttnLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.E = torch.nn.Parameter(E0.clone())
        self.Wq = torch.nn.Parameter(torch.eye(dim, dtype=torch.float32) * 0.1)
        self.Wk = torch.nn.Parameter(torch.eye(dim, dtype=torch.float32) * 0.1)
        self.Wv = torch.nn.Parameter(torch.eye(dim, dtype=torch.float32) * 0.1)
        self.scale = math.sqrt(dim)
    def forward(self,c1,c2): 
        h1=self.E[c1]; h2=self.E[c2]
        K=torch.stack([h1@self.Wk.T,h2@self.Wk.T],dim=1); V=torch.stack([h1@self.Wv.T,h2@self.Wv.T],dim=1)
        Q=(h2@self.Wq.T).unsqueeze(1)
        return torch.bmm(F.softmax(torch.bmm(Q,K.transpose(1,2))/self.scale,dim=-1),V).squeeze(1)@self.E.T

def flatten_grad(model):
    parts=[]
    for p in model.parameters():
        if p.grad is not None: parts.append(p.grad.detach().reshape(-1).clone())
        else: parts.append(torch.zeros_like(p).reshape(-1))
    return torch.cat(parts).numpy()

# Train with soft rew (alpha=0.5) to reach early stage
probs={'A':0.25,'B':0.25,'C':0.25,'D':0.25}
base_w=torch.ones(16,dtype=torch.float32)/16
ft={}
for t in targ_full.tolist(): ft[t]=ft.get(t,0)+1
rew_w=base_w.clone()/torch.tensor([ft[t]**0.5 for t in targ_full.tolist()],dtype=torch.float32)
rew_w=rew_w/rew_w.sum()

model=AttnLM()

# Check gradient structure at step 0 and step 200
for checkpoint in [0, 200]:
    if checkpoint>0:
        for step in range(checkpoint):
            logits=model(c1_full,c2_full); losses=F.cross_entropy(logits,targ_full,reduction='none')
            (losses*rew_w).sum().backward()
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None: p-=lr*p.grad; p.grad=None
    
    # Conditional gradients
    q={}; norms={}
    for gn in gns:
        m_copy=AttnLM()
        m_copy.load_state_dict(model.state_dict())
        c1_g,c2_g,targ_g=group_data[gn]
        logits=m_copy(c1_g,c2_g); loss=F.cross_entropy(logits,targ_g,reduction='mean')
        loss.backward()
        q[gn]=flatten_grad(m_copy)
        norms[gn]=float(np.linalg.norm(q[gn]))
    
    # K conditional: all bigrams involving K (to_K + from_K + from_K2 = 3 per group × 4 = 12)
    # Actually just do: K as target (G0G1->K) for all groups, and K as context
    k_mask=(c1_full==0)|(c2_full==0)|(targ_full==0)  # any involvement of K
    k_idxs=torch.where(k_mask)[0]
    m_copy=AttnLM(); m_copy.load_state_dict(model.state_dict())
    logits_k=m_copy(c1_full[k_idxs],c2_full[k_idxs])
    loss_k=F.cross_entropy(logits_k,targ_full[k_idxs],reduction='mean')
    loss_k.backward()
    q['K']=flatten_grad(m_copy); norms['K']=float(np.linalg.norm(q['K']))
    
    print(f'\n{"="*60}')
    print(f'Step {checkpoint}: Gradient Structure Analysis')
    print(f'{"="*60}')
    
    # Cosine matrix (include K, A, B, C, D)
    entities=list(gns)+['K']
    print(f'{"":>8}',end='')
    for e in entities: print(f'{e:>10}',end='')
    print()
    for e1 in entities:
        print(f'{e1:>8}',end='')
        for e2 in entities:
            cos=float(np.dot(q[e1],q[e2])/max(norms[e1]*norms[e2],1e-12))
            print(f'{cos:10.4f}',end='')
        print()
    
    # SIR: for each tail group (B,C,D), compute SIR vs common (A) and K
    print(f'\n  SIR (tail vs common interference):')
    common_name='A'
    for tail in ['B','C','D']:
        unit=q[tail]/norms[tail]
        signal=0.25*norms[tail]  # uniform group weight
        common_interf=abs(0.25*float(np.dot(q[common_name],unit)))
        other_interf=sum(abs(0.25*float(np.dot(q[o],unit))) for o in ['B','C','D'] if o!=tail)
        sir=signal/(common_interf+other_interf+1e-12)
        print(f'    {tail}: signal={signal:.4f}, common_interf={common_interf:.4f}, sir={sir:.3f}')
    
    # K gradient dominance
    print(f'\n  Gradient norm comparison:')
    for e in entities:
        arrow=' ← HIGH' if norms[e]==max(norms.values()) else ''
        print(f'    {e}: |q|={norms[e]:.4f}{arrow}')
    
    # Decompose: where does K's gradient pull go?
    if checkpoint==200:
        print(f'\n  K gradient decomposition (what drives |q_K|?):')
        # E[K] receives gradient from being target and being in context
        # ∂L/∂E[K] from being target: (softmax_K-1) * h_proj_context
        # ∂L/∂E[K] from being in context: through attention mechanism
        Eg=model.E.grad  # just zero the grad and recompute
        # Compute E[K] gradient contribution from target-only
        m_copy=AttnLM(); m_copy.load_state_dict(model.state_dict())
        k_as_target_idxs=[i for i in range(len(targ_full)) if targ_full[i]==0]
        logits_t=m_copy(c1_full[k_as_target_idxs],c2_full[k_as_target_idxs])
        loss_t=F.cross_entropy(logits_t,targ_full[k_as_target_idxs],reduction='mean')
        loss_t.backward()
        g_EK_target=float(torch.norm(m_copy.E.grad[0]))
        
        # E[K] gradient from being in context (K appears as c1 or c2)
        k_as_ctx_idxs=[i for i in range(len(targ_full)) if c1_full[i]==0 or c2_full[i]==0 and i not in k_as_target_idxs]
        m_copy2=AttnLM(); m_copy2.load_state_dict(model.state_dict())
        if len(k_as_ctx_idxs)>0:
            logits_c=m_copy2(c1_full[k_as_ctx_idxs],c2_full[k_as_ctx_idxs])
            loss_c=F.cross_entropy(logits_c,targ_full[k_as_ctx_idxs],reduction='mean')
            loss_c.backward()
            g_EK_ctx=float(torch.norm(m_copy2.E.grad[0]))
        else:
            g_EK_ctx=0.0
        print(f'    |∂L/∂E[K]| from being TARGET: {g_EK_target:.4f}')
        print(f'    |∂L/∂E[K]| from being in CONTEXT: {g_EK_ctx:.4f}')
