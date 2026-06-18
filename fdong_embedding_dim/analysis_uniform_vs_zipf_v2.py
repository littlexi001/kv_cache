#!/usr/bin/env python3
"""Uniform vs Zipf: clean speed/geometry comparison."""
import math
import numpy as np
import torch
import torch.nn.functional as F

dim = 2; theta_deg = 12.0; lr = 0.03
centers = {'common': (1.,0.), 'tail1': (0.,1.), 'tail2': (0.,-1.), 'tail3': (-1.,0.)}
group_names = ['common','tail1','tail2','tail3']

def rot(v,d):
    t=math.radians(d)
    return np.array([math.cos(t)*v[0]-math.sin(t)*v[1], math.sin(t)*v[0]+math.cos(t)*v[1]], dtype=np.float32)

all_v = []
for gn in group_names:
    for a in [0., theta_deg, -theta_deg]:
        all_v.append(rot(centers[gn], a))
E0 = torch.tensor(np.stack(all_v)); W0 = E0.clone()

ins, tars, grps = [], [], []
for gi, gn in enumerate(group_names):
    b = gi*3
    for t in range(3):
        ins.append(b+t); tars.append(b+(t+1)%3); grps.append(gn)
in_t = torch.tensor(ins, dtype=torch.long)
tar_t = torch.tensor(tars, dtype=torch.long)

class BigramLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.E = torch.nn.Parameter(E0.clone())
        self.W = torch.nn.Parameter(W0.clone())
    def forward(self, ids):
        return self.E[ids] @ self.W.T

def sim(probs, steps=200):
    sw = torch.tensor([probs[g]/3.0 for g in grps], dtype=torch.float32)
    m = BigramLM()
    hist = []
    for s in range(steps):
        logits = m(in_t)
        losses = F.cross_entropy(logits, tar_t, reduction='none')
        (losses * sw).sum().backward()
        gE, gW = m.E.grad.clone(), m.W.grad.clone()
        with torch.no_grad():
            pred = logits.argmax(-1)
            correct = (pred == tar_t).float()
            E_np = m.E.detach().numpy()
            _, sv, _ = np.linalg.svd(E_np, full_matrices=False)
            vv, nrm, acc = {}, {}, {}
            for gi, gn in enumerate(group_names):
                vv[gn] = E_np[gi*3:(gi+1)*3].mean(0)
                nrm[gn] = float(np.linalg.norm(vv[gn]))
                ix = [i for i,g in enumerate(grps) if g==gn]
                acc[gn] = float(correct[ix].mean())
            m.E -= lr * gE
            m.W -= lr * gW
        hist.append({'svr': sv[0]/max(sv[1],1e-12), 'nrm': nrm, 'acc': acc, 'v': vv})
    return hist

print("Simulating Uniform ...", end=" ")
pu = sim({'common':0.25,'tail1':0.25,'tail2':0.25,'tail3':0.25})
print(f"done ({len(pu)} steps)")
print("Simulating Zipf    ...", end=" ")
pz = sim({'common':0.70,'tail1':0.10,'tail2':0.10,'tail3':0.10})
print(f"done ({len(pz)} steps)")

# E0 centroid (NOT the theoretical center)
e0_c = {}
for gi, gn in enumerate(group_names):
    e0_c[gn] = E0[gi*3:(gi+1)*3].detach().numpy().mean(0)

print()
print("=" * 80)
print("Q1: Per-group centroid speed under Uniform")
print("=" * 80)
for s in range(3):
    prev = e0_c if s == 0 else pu[s-1]['v']
    speeds = {gn: float(np.linalg.norm(pu[s]['v'][gn] - prev[gn])) for gn in group_names}
    vals = list(speeds.values())
    same = max(vals) - min(vals) < 1e-8
    print(f"  Step {s+1}: common={speeds['common']:.8f}, tail1={speeds['tail1']:.8f}, "
          f"tail2={speeds['tail2']:.8f}, tail3={speeds['tail3']:.8f}  {'[ALL EQUAL]' if same else ''}")

print()
print("=" * 80)
print("Q2: Per-group centroid speed under Zipf")
print("=" * 80)
for s in range(3):
    prev = e0_c if s == 0 else pz[s-1]['v']
    speeds = {gn: float(np.linalg.norm(pz[s]['v'][gn] - prev[gn])) for gn in group_names}
    ratio = speeds['common'] / max(speeds['tail1'], 1e-12)
    print(f"  Step {s+1}: common={speeds['common']:.8f}, tail1={speeds['tail1']:.8f}, "
          f"tail2={speeds['tail2']:.8f}, tail3={speeds['tail3']:.8f}  "
          f"[ratio common/tail = {ratio:.1f}]")

print()
print("=" * 80)
print("Q3: Singular value ratio σ1/σ2 evolution")
print("=" * 80)
print(f"  {'Step':>6}  {'Uniform':>12}  {'Zipf':>12}  {'Comment'}")
for s in [0, 1, 2, 4, 9, 19, 29, 49, 99, 199]:
    ur = pu[s]['svr']
    zr = pz[s]['svr']
    comment = ""
    if ur > 1.0001:
        comment += "Uniform >1! "
    if zr > 1.01:
        comment += "Zipf diverging"
    print(f"  {s+1:6d}  {ur:12.8f}  {zr:12.8f}  {comment}")

print()
print("=" * 80)
print("Q4: Final geometry (step 200)")
print("=" * 80)
print(f"  Uniform: σ1/σ2 = {pu[-1]['svr']:.6f}")
print(f"  Zipf:    σ1/σ2 = {pz[-1]['svr']:.6f}")
for gn in group_names:
    un = pu[-1]['nrm'][gn]
    zn = pz[-1]['nrm'][gn]
    print(f"  {gn:>8}: Uni |c|={un:.4f}, Zpf |c|={zn:.4f}, Z/U={zn/un:.4f}")

print()
print("=" * 80)
print("Q5: Accuracy comparison")
print("=" * 80)
for s in [10, 20, 30, 40, 50, 100, 200]:
    uall = all(pu[s-1]['acc'][gn] >= 1.0 for gn in group_names)
    zall = all(pz[s-1]['acc'][gn] >= 1.0 for gn in group_names)
    print(f"  Step {s:3d}: Uniform all=1.0: {uall},  Zipf all=1.0: {zall}")
