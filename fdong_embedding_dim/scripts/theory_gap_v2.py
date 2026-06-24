#!/usr/bin/env python3
"""
E2 v2: Measure gradient projection onto singular directions.
  Instead of artificial perturbation, look at where the actual gradient goes.
  Measure: projection of ∂L/∂E onto v₁ vs v₄, and the resultant loss decrease.

E4 v2: Train longer, track dσ₁/dt every step, correlate with loss saturation.
"""
import math, numpy as np, torch, torch.nn.functional as F

dim = 4; theta_deg = 12.0; lr = 0.03

def rot4(v, d):
    cc = np.array(v, dtype=np.float32); cc /= np.linalg.norm(cc)
    perp = np.zeros(dim, dtype=np.float32)
    perp[0 if abs(cc[0]) < 0.9 else 1] = 1.0
    perp -= float(np.dot(perp, cc)) * cc; perp /= np.linalg.norm(perp)
    t = math.radians(d)
    return (math.cos(t) * cc + math.sin(t) * perp).astype(np.float32)

class AttnModel(torch.nn.Module):
    def __init__(self, E0):
        super().__init__()
        self.E = torch.nn.Parameter(E0.clone())
        d = E0.shape[1]
        self.Wq = torch.nn.Parameter(torch.eye(d, dtype=torch.float32) * 0.1)
        self.Wk = torch.nn.Parameter(torch.eye(d, dtype=torch.float32) * 0.1)
        self.Wv = torch.nn.Parameter(torch.eye(d, dtype=torch.float32) * 0.1)
        self.scale = math.sqrt(d)
    def forward(self, c1, c2):
        h1 = self.E[c1]; h2 = self.E[c2]
        K_ = torch.stack([h1 @ self.Wk.T, h2 @ self.Wk.T], dim=1)
        V = torch.stack([h1 @ self.Wv.T, h2 @ self.Wv.T], dim=1)
        Q = (h2 @ self.Wq.T).unsqueeze(1)
        s = torch.bmm(Q, K_.transpose(1, 2)) / self.scale
        return torch.bmm(F.softmax(s, dim=-1), V).squeeze(1) @ self.E.T

# ---- Build nested data (AB, ABC, ABCD) ----
centers = {"A":(1.,0.,0.,0.),"B":(0.,1.,0.,0.),"C":(0.,0.,1.,0.),"D":(0.,0.,0.,1.)}
all_v = []
for gn in ["A","B","C","D"]:
    c = np.array(centers[gn], dtype=np.float32); c /= np.linalg.norm(c)
    for off in [0.,theta_deg,-theta_deg]: all_v.append(rot4(c, off))
E0 = torch.tensor(np.stack(all_v).astype(np.float32))

A0,A1,A2=0,1,2; B0,B1,B2=3,4,5; C0,C1,C2=6,7,8; D0,D1,D2=9,10,11
c1 = [A0,A1,B0,  A0,A1,B0,B1,C0,  A0,A1,B0,B1,C0,C1,D0]
c2 = [A1,B0,B1,  A1,B0,B1,C0,C1,  A1,B0,B1,C0,C1,D0,D1]
targ= [A1,B0,B1,  A1,B0,B1,C0,C1,  A1,B0,B1,C0,C1,D0,D1]
c1_t = torch.tensor(c1, dtype=torch.long)
c2_t = torch.tensor(c2, dtype=torch.long)
targ_t = torch.tensor(targ, dtype=torch.long)
w = torch.ones(len(c1), dtype=torch.float32) / len(c1)

# ============================================================
# E2 v2: Gradient projection onto singular directions
# ============================================================
print("=" * 70)
print("E2 v2: GRADIENT PROJECTION onto singular directions")
print("  Question: Does the optimizer spend more effort along σ₁ than σ₄?")
print("  Method: Train model, then compute |proj(∂L/∂E, v_k)| for each k")
print("=" * 70)

model = AttnModel(E0)
proj_history = []

for step in range(1200):
    logits = model(c1_t, c2_t)
    losses = F.cross_entropy(logits, targ_t, reduction="none")
    (losses * w).sum().backward()

    # Before update: compute gradient projections
    if step % 50 == 0 or step < 5:
        with torch.no_grad():
            En = model.E.detach().numpy()
            sv = np.linalg.svd(En, full_matrices=False)[1]
            Vh = np.linalg.svd(En, full_matrices=False)[2]  # V^T

            # Flatten gradient of E
            gE = model.E.grad.detach().numpy().reshape(-1)  # (V*d,) = (12*4,)=48

            # Project gE onto each singular direction
            # E has shape (12, 4). Each singular direction is a 4D vector v_k.
            # The gradient component along v_k: sum over all tokens of (∂L/∂E_tok · v_k)^2
            projections = []
            for k in range(4):
                vk = Vh[k, :]  # (4,)
                # For each token, project its gradient onto vk
                total_proj = 0.0
                for tok in range(12):
                    g_tok = model.E.grad.detach().numpy()[tok]  # (4,)
                    total_proj += float(np.dot(g_tok, vk)) ** 2
                projections.append(np.sqrt(total_proj))
            proj_history.append((step, sv.copy(), projections, float(losses.mean())))

    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                p -= lr * p.grad
                p.grad = None

print(f'{"Step":>6} {"σ₁":>7} {"σ₂":>7} {"σ₃":>7} {"σ₄":>7} {"|proj(v₁)|":>12} {"|proj(v₂)|":>12} {"|proj(v₃)|":>12} {"|proj(v₄)|":>12} {"pref(v₁/v₄)":>14} {"σ₁/σ₄":>8}')
print("-" * 130)

for step, sv, proj, loss_val in proj_history:
    v1v4 = proj[0] / max(proj[3], 1e-12)
    s1s4 = sv[0] / max(sv[3], 1e-12)
    print(f"{step+1:6d} {sv[0]:7.2f} {sv[1]:7.2f} {sv[2]:7.2f} {sv[3]:7.2f} {proj[0]:12.4f} {proj[1]:12.4f} {proj[2]:12.4f} {proj[3]:12.4f} {v1v4:14.3f} {s1s4:8.3f}")

# Final analysis
final = proj_history[-1]
v1v4_final = final[2][0] / max(final[2][3], 1e-12)
s1s4_final = final[1][0] / max(final[1][3], 1e-12)
print(f"\nE2 v2 VERDICT:")
print(f"  Final gradient preference |proj(v₁)|/|proj(v₄)| = {v1v4_final:.2f}")
print(f"  Final spectrum σ₁/σ₄ = {s1s4_final:.2f}")
if v1v4_final > 1.5:
    print(f"  ✓ Optimizer strongly prefers σ₁ direction ({v1v4_final:.1f}× more gradient power)")
elif v1v4_final > 1.1:
    print(f"  △ Modest preference ({v1v4_final:.1f}×). Spectrum alone doesn't dominate.")
else:
    print(f"  ✗ No strong preference. Optimizer distributes effort equally across all σ directions.")

# ============================================================
# E4 v2: LONGER training with fine-grained σ tracking
# ============================================================
print("\n\n" + "=" * 70)
print("E4 v2: LONG-TERM σ₁ GROWTH RATE TRACKING")
print("  Question: When does dσ₁/dt decay, and does it align with loss plateau?")
print("=" * 70)

# Use K-token model for more interesting dynamics
K_init = np.ones(dim, dtype=np.float32) / math.sqrt(dim)
centers_k = {"A":(1.,0.,0.,0.),"B":(0.,1.,0.,0.),"C":(0.,0.,1.,0.),"D":(0.,0.,0.,1.)}
all_v_k = [K_init]
gi_k = {}; idx_k = 1
for gn in ["A","B","C","D"]:
    c = np.array(centers_k[gn], dtype=np.float32); c /= np.linalg.norm(c)
    gi_k[gn] = [idx_k, idx_k+1, idx_k+2]; idx_k += 3
    for off in [0.,theta_deg,-theta_deg]: all_v_k.append(rot4(c, off))
E0_k = torch.tensor(np.stack(all_v_k).astype(np.float32))

c1k,c2k,targk = [],[],[]
for gn,(i0,i1,i2) in gi_k.items():
    c1k += [i0,i1,0,i2]; c2k += [i1,0,i2,i0]; targk += [0,i2,i0,i1]
c1k_t = torch.tensor(c1k, dtype=torch.long)
c2k_t = torch.tensor(c2k, dtype=torch.long)
targk_t = torch.tensor(targk, dtype=torch.long)
wk = torch.ones(len(c1k), dtype=torch.float32) / len(c1k)

model_k = AttnModel(E0_k)
sigma_hist = []
loss_hist = []
grad_norm_hist = []

# More steps, higher LR for K-token
lr_k = 0.05
max_steps_k = 3000

for step in range(max_steps_k):
    logits = model_k(c1k_t, c2k_t)
    losses = F.cross_entropy(logits, targk_t, reduction="none")
    total_loss = (losses * wk).sum()
    total_loss.backward()

    with torch.no_grad():
        # Track gradient norm
        gn = 0.0
        for p in model_k.parameters():
            if p.grad is not None:
                gn += float((p.grad ** 2).sum())
        grad_norm_hist.append(np.sqrt(gn))

        # Apply update
        for p in model_k.parameters():
            if p.grad is not None:
                p -= lr_k * p.grad
                p.grad = None

    # Track SV every 10 steps
    if step % 10 == 0 or step < 5:
        with torch.no_grad():
            En = model_k.E.detach().numpy()
            sv = np.linalg.svd(En, full_matrices=False)[1]
            sigma_hist.append((step, sv.copy()))
            pred = logits.argmax(-1)
            acc = float((pred == targk_t).float().mean())
            loss_hist.append((step, float(total_loss), acc))

# Compute growth rates
sv_arr = np.array([s for _, s in sigma_hist])
sigma1 = sv_arr[:, 0]
sigma4 = sv_arr[:, 3]
steps_arr = np.array([s for s, _ in sigma_hist])

# Growth rate with step normalization
d_sigma1 = np.diff(sigma1) / np.diff(steps_arr)
d_sigma4 = np.diff(sigma4) / np.diff(steps_arr)

# Find decay point: where smoothed growth rate drops below 50% of max
window = 20
if len(d_sigma1) > window:
    d1_smooth = np.convolve(d_sigma1, np.ones(window)/window, mode='valid')
    max_growth = np.max(d1_smooth[:50]) if len(d1_smooth) > 50 else np.max(d1_smooth)
    half_max = max_growth * 0.5
    decay_idx = None
    for i in range(window, len(d1_smooth)):
        if d1_smooth[i] < half_max and d1_smooth[max(0,i-10)] < half_max:
            decay_idx = i + window
            break
else:
    decay_idx = None

print(f"  total training steps: {max_steps_k}")
print(f"  Max dσ₁/dt (smoothed): {np.max(d1_smooth) if 'd1_smooth' in dir() else 'N/A'}")
if decay_idx:
    decay_step = steps_arr[decay_idx] + 1
    print(f"  Half-decay at step {decay_step}")
    # Find accuracy at decay
    decay_acc = 0.0
    for step, _, acc in loss_hist:
        if step >= decay_step:
            decay_acc = acc
            break
    print(f"  Accuracy at decay: {decay_acc:.2%}")

# Show evolution
print(f'\n{"Step":>6} {"σ₁":>7} {"σ₄":>7} {"dσ₁/dt":>10} {"dσ₄/dt":>10} {"loss":>8} {"acc":>6} {"σ₁/σ₄":>8}')
print("-" * 65)
# Show key checkpoints
n_show = min(30, len(sigma_hist))
indices = np.linspace(0, len(sigma_hist)-1, n_show, dtype=int)
for i in indices:
    step = sigma_hist[i][0]
    s1 = sigma1[i]; s4 = sigma4[i]
    d1 = d_sigma1[i-1] if i > 0 else 0
    d4 = d_sigma4[i-1] if i > 0 else 0
    loss_v = loss_hist[min(i, len(loss_hist)-1)][1]
    acc_v = loss_hist[min(i, len(loss_hist)-1)][2]
    r = s1 / max(s4, 1e-12)
    print(f"{step+1:6d} {s1:7.2f} {s4:7.2f} {d1:10.5f} {d4:10.5f} {loss_v:8.4f} {acc_v:6.2%} {r:8.3f}")

# Final verdict
final_idx = len(sigma_hist) - 1
final_s1, final_s4 = sigma1[-1], sigma4[-1]
final_loss, final_acc = loss_hist[-1][1], loss_hist[-1][2]
print(f"\nE4 v2 VERDICT:")
print(f"  Final: σ₁={final_s1:.2f}, σ₄={final_s4:.2f}, loss={final_loss:.4f}, acc={final_acc:.2%}")
if final_acc > 0.9:
    avg_last_d1 = np.mean(d_sigma1[-50:]) if len(d_sigma1) > 50 else np.mean(d_sigma1[-10:])
    avg_early_d1 = np.mean(d_sigma1[10:60]) if len(d_sigma1) > 60 else np.mean(d_sigma1[:20])
    print(f"  Early avg dσ₁/dt = {avg_early_d1:.5f}")
    print(f"  Late avg dσ₁/dt  = {avg_last_d1:.5f}")
    print(f"  Decay ratio = {avg_last_d1/max(avg_early_d1,1e-12):.2%}")
    if avg_last_d1 < 0.3 * avg_early_d1:
        print(f"  ✓ Theory confirmed: dσ₁/dt decays as model converges")
    else:
        print(f"  △ Partial support: some decay but not dramatic")
else:
    print(f"  △ Model did not fully converge. Cannot test decay hypothesis.")
    print(f"  Try: more steps or different lr.")
