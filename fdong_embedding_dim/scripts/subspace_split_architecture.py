#!/usr/bin/env python3
"""Subspace Split Architecture: Decoupling Common and Residual Attention.
Reproduces all results in subspace_split_architecture.md.
Run: python3 subspace_split_architecture.py
"""

import math, numpy as np, torch, torch.nn.functional as F
from collections import Counter

dim=4; theta_deg=12.0

# ── Token geometry ──
def rot4(v,d):
    cc=np.array(v,dtype=np.float32); cc/=np.linalg.norm(cc)
    perp=np.zeros(dim,dtype=np.float32); perp[0 if abs(cc[0])<0.9 else 1]=1.0
    perp-=float(np.dot(perp,cc))*cc; perp/=np.linalg.norm(perp)
    t=math.radians(d); return (math.cos(t)*cc+math.sin(t)*perp).astype(np.float32)

K_init=np.ones(dim,dtype=np.float32)/math.sqrt(dim)
centers={"A":(1.,0.,0.,0.),"B":(0.,1.,0.,0.),"C":(0.,0.,1.,0.),"D":(0.,0.,0.,1.)}
all_v=[K_init]; gi={}; idx=1
for gn in ["A","B","C","D"]:
    c=np.array(centers[gn],dtype=np.float32); c/=np.linalg.norm(c)
    gi[gn]=[idx,idx+1,idx+2]; idx+=3
    for off in [0.,theta_deg,-theta_deg]: all_v.append(rot4(c,off))
E0=torch.tensor(np.stack(all_v).astype(np.float32))

# ── Training data: 16 trigrams ──
c1k,c2k,tark,namesk=[],[],[],[]
for gn,(i0,i1,i2) in gi.items():
    c1k+=[i0,i1,0,i2]; c2k+=[i1,0,i2,i0]; tark+=[0,i2,i0,i1]
    namesk+=['G0G1_K','G1K_G2','KG2_G0','G2G0_G1']
c1_t=torch.tensor(c1k,dtype=torch.long); c2_t=torch.tensor(c2k,dtype=torch.long)
tar_t=torch.tensor(tark,dtype=torch.long)

# ── Loss weights ──
bw=torch.ones(16,dtype=torch.float32)/16
ft=Counter(tark)
w_no  = bw / bw.sum()
w_hard = bw.clone() / torch.tensor([ft[t]**1.0 for t in tark],dtype=torch.float32)
w_hard = w_hard / w_hard.sum()

def rms_norm(x):
    return x / torch.sqrt((x**2).mean(dim=-1,keepdim=True) + 1e-6)

# ── Models ──
class Baseline(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.E = torch.nn.Parameter(E0.clone())
        self.Wq = torch.nn.Parameter(torch.eye(dim,dtype=torch.float32)*0.1)
        self.Wk = torch.nn.Parameter(torch.eye(dim,dtype=torch.float32)*0.1)
        self.Wv = torch.nn.Parameter(torch.eye(dim,dtype=torch.float32)*0.1)
        self.scale = math.sqrt(dim)
    def forward(self, c1, c2):
        h1 = self.E[c1]; h2 = self.E[c2]
        hn1 = rms_norm(h1); hn2 = rms_norm(h2)
        q1 = hn1 @ self.Wq.T; q2 = hn2 @ self.Wq.T
        k1 = hn1 @ self.Wk.T; k2 = hn2 @ self.Wk.T
        v1 = hn1 @ self.Wv.T; v2 = hn2 @ self.Wv.T
        K = torch.stack([k1, k2], dim=1); V = torch.stack([v1, v2], dim=1)
        attn = torch.bmm(F.softmax(torch.bmm(q2.unsqueeze(1), K.transpose(1,2)) / self.scale, dim=-1), V).squeeze(1)
        return (h2 + attn) @ self.E.T

class SubspaceSplit(torch.nn.Module):
    def __init__(self, alpha_common=0.3):
        super().__init__()
        self.E = torch.nn.Parameter(E0.clone())
        self.Wq_c = torch.nn.Parameter(torch.eye(dim,dtype=torch.float32)*0.1)
        self.Wk_c = torch.nn.Parameter(torch.eye(dim,dtype=torch.float32)*0.1)
        self.Wv_c = torch.nn.Parameter(torch.eye(dim,dtype=torch.float32)*0.1)
        self.Wq_r = torch.nn.Parameter(torch.eye(dim,dtype=torch.float32)*0.1)
        self.Wk_r = torch.nn.Parameter(torch.eye(dim,dtype=torch.float32)*0.1)
        self.Wv_r = torch.nn.Parameter(torch.eye(dim,dtype=torch.float32)*0.1)
        self.scale = math.sqrt(dim)
        self.alpha = alpha_common
    def get_vK(self):
        with torch.no_grad():
            c = self.E.mean(dim=0)
            return c / (c.norm() + 1e-12)
    def forward(self, c1, c2):
        h1 = self.E[c1]; h2 = self.E[c2]
        hn1 = rms_norm(h1); hn2 = rms_norm(h2)
        vK = self.get_vK()
        proj1 = (hn1 @ vK).unsqueeze(-1) * vK.unsqueeze(0)
        proj2 = (hn2 @ vK).unsqueeze(-1) * vK.unsqueeze(0)
        res1  = hn1 - proj1; res2  = hn2 - proj2
        q1c = proj1 @ self.Wq_c.T; q2c = proj2 @ self.Wq_c.T
        k1c = proj1 @ self.Wk_c.T; k2c = proj2 @ self.Wk_c.T
        v1c = proj1 @ self.Wv_c.T; v2c = proj2 @ self.Wv_c.T
        Kc = torch.stack([k1c, k2c], dim=1); Vc = torch.stack([v1c, v2c], dim=1)
        attn_c = torch.bmm(F.softmax(torch.bmm(q2c.unsqueeze(1), Kc.transpose(1,2)) / self.scale, dim=-1), Vc).squeeze(1)
        q1r = res1 @ self.Wq_r.T; q2r = res2 @ self.Wq_r.T
        k1r = res1 @ self.Wk_r.T; k2r = res2 @ self.Wk_r.T
        v1r = res1 @ self.Wv_r.T; v2r = res2 @ self.Wv_r.T
        Kr = torch.stack([k1r, k2r], dim=1); Vr = torch.stack([v1r, v2r], dim=1)
        attn_r = torch.bmm(F.softmax(torch.bmm(q2r.unsqueeze(1), Kr.transpose(1,2)) / self.scale, dim=-1), Vr).squeeze(1)
        return (h2 + self.alpha * attn_c + attn_r) @ self.E.T

# ── Training ──
def train_track(ModelClass, w, lr_val, max_steps, **kwargs):
    m = ModelClass(**kwargs); rec = []
    for step in range(max_steps):
        logits = m(c1_t, c2_t)
        l_all = F.cross_entropy(logits, tar_t, reduction='none')
        (l_all * w).sum().backward()
        with torch.no_grad():
            for p in m.parameters():
                if p.grad is not None: p -= lr_val * p.grad; p.grad = None
        if step % 100 == 0 or step == max_steps - 1:
            with torch.no_grad():
                pred = logits.argmax(-1); correct = (pred == tar_t).float()
                pl = {nm: float(l_all[[i for i,n in enumerate(namesk) if n==nm]].mean()) for nm in set(namesk)}
                pa = {nm: float(correct[[i for i,n in enumerate(namesk) if n==nm]].mean()) for nm in set(namesk)}
                rec.append({'step': step, 'loss': float(l_all.mean()), 'acc': float(correct.mean()),
                            'per_loss': pl, 'per_acc': pa})
    return rec

lr = 0.05; steps = 3000
rec_b = train_track(Baseline, w_no, lr, steps)
rec_h = train_track(Baseline, w_hard, lr*1.6, steps)
rec_s = train_track(SubspaceSplit, w_no, lr, steps, alpha_common=0.3)

# ── Report ──
def find_conv(rec, p):
    for r in rec:
        if r['per_acc'][p] >= 1.0: return r['step'] + 1
    return None

print("="*80); print("PER-PATTERN CONVERGENCE")
print("="*80)
print(f"{'Model':>20} {'G0G1→K':>10} {'G1K→G2':>10} {'KG2→G0':>10} {'G2G0→G1':>10} {'Overall':>10}")
print("-"*65)
for nm, rec in [("Baseline no_rew", rec_b), ("Baseline hard_rew", rec_h), ("Split α=0.3", rec_s)]:
    c = [find_conv(rec, p) for p in ['G0G1_K','G1K_G2','KG2_G0','G2G0_G1']]
    print(f"{nm:>20} {str(c[0]):>10} {str(c[1]):>10} {str(c[2]):>10} {str(c[3]):>10} {str(max(c) if all(x is not None for x in c) else 'None'):>10}")

print(f"\n{'='*80}"); print(f"FINAL PARAMETER SPECTRUM (σ₁/σ₄)")
print(f"{'='*80}")
def get_sv(m, w, lr, steps, **kw):
    mdl = m(**kw)
    for _ in range(steps):
        logits = mdl(c1_t, c2_t)
        (F.cross_entropy(logits, tar_t, reduction='none')*w).sum().backward()
        with torch.no_grad():
            for p in mdl.parameters():
                if p.grad is not None: p -= lr * p.grad; p.grad = None
    with torch.no_grad():
        logits = mdl(c1_t, c2_t); acc = float((logits.argmax(-1) == tar_t).float().mean())
        En = mdl.E.detach().numpy(); svE = np.linalg.svd(En)[1]
        if hasattr(mdl, 'Wq_c'):
            svQc = np.linalg.svd(mdl.Wq_c.detach().numpy())[1]
            svQr = np.linalg.svd(mdl.Wq_r.detach().numpy())[1]
            return acc, svE, svQc, svQr
        else:
            svQ = np.linalg.svd(mdl.Wq.detach().numpy())[1]
            return acc, svE, svQ

a_b, sE_b, sQ_b = get_sv(Baseline, w_no, lr, steps)
a_h, sE_h, sQ_h = get_sv(Baseline, w_hard, lr*1.6, steps)
a_s, sE_s, sQc, sQr = get_sv(SubspaceSplit, w_no, lr, steps, alpha_common=0.3)

def r(x): return x[0]/max(x[-1],1e-12)
print(f"{'Model':>20} {'acc':>8} {'E σ₁/σ₄':>10} {'Wq_c σ₁/σ₄':>12} {'Wq_r σ₁/σ₄':>12} {'Wq σ₁/σ₄':>12}")
print("-"*70)
print(f"{'Baseline no_rew':>20} {a_b:8.1%} {r(sE_b):10.1f} {'-':>12} {'-':>12} {r(sQ_b):12.1f}")
print(f"{'Baseline hard_rew':>20} {a_h:8.1%} {r(sE_h):10.1f} {'-':>12} {'-':>12} {r(sQ_h):12.1f}")
print(f"{'Split α=0.3':>20} {a_s:8.1%} {r(sE_s):10.1f} {r(sQc):12.1f} {r(sQr):12.1f} {'-':>12}")
