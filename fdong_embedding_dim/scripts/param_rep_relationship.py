#!/usr/bin/env python3
"""
Analyze: parameter space singular directions — origin, meaning, and
relationship to representation space.

Questions:
  Q1: Do Wq, Wk, Wv, E all develop spectral concentration? In what order?
  Q2: Are their top singular directions aligned across matrices?
  Q3: How does gradient flow couple parameter sing directions to rep directions?
"""
import math, numpy as np, torch, torch.nn.functional as F

dim = 4
theta_deg = 12.0
lr = 0.03
max_steps = 1200


def rot4(v, d):
    cc = np.array(v, dtype=np.float32)
    cc /= np.linalg.norm(cc)
    perp = np.zeros(dim, dtype=np.float32)
    perp[0 if abs(cc[0]) < 0.9 else 1] = 1.0
    perp -= float(np.dot(perp, cc)) * cc
    perp /= np.linalg.norm(perp)
    t = math.radians(d)
    return (math.cos(t) * cc + math.sin(t) * perp).astype(np.float32)


class AttnModel(torch.nn.Module):
    def __init__(self, E0):
        super().__init__()
        self.E = torch.nn.Parameter(E0.clone())
        self.Wq = torch.nn.Parameter(torch.eye(dim, dtype=torch.float32) * 0.1)
        self.Wk = torch.nn.Parameter(torch.eye(dim, dtype=torch.float32) * 0.1)
        self.Wv = torch.nn.Parameter(torch.eye(dim, dtype=torch.float32) * 0.1)
        self.scale = math.sqrt(dim)

    def forward(self, c1, c2):
        h1 = self.E[c1]
        h2 = self.E[c2]
        K = torch.stack([h1 @ self.Wk.T, h2 @ self.Wk.T], dim=1)
        V = torch.stack([h1 @ self.Wv.T, h2 @ self.Wv.T], dim=1)
        Q = (h2 @ self.Wq.T).unsqueeze(1)
        s = torch.bmm(Q, K.transpose(1, 2)) / self.scale
        attn_out = torch.bmm(F.softmax(s, dim=-1), V).squeeze(1)
        return attn_out @ self.E.T, attn_out


# ---- Build K-token data ----
K_init = np.ones(dim, dtype=np.float32) / math.sqrt(dim)
centers = {
    "A": (1.0, 0.0, 0.0, 0.0),
    "B": (0.0, 1.0, 0.0, 0.0),
    "C": (0.0, 0.0, 1.0, 0.0),
    "D": (0.0, 0.0, 0.0, 1.0),
}
all_v = [K_init]
gi = {}
idx = 1
for gn in ["A", "B", "C", "D"]:
    c = np.array(centers[gn], dtype=np.float32)
    c /= np.linalg.norm(c)
    gi[gn] = [idx, idx + 1, idx + 2]
    idx += 3
    for off in [0.0, theta_deg, -theta_deg]:
        all_v.append(rot4(c, off))
E0 = torch.tensor(np.stack(all_v).astype(np.float32))

c1k, c2k, tark, namesk = [], [], [], []
for gn, (i0, i1, i2) in gi.items():
    c1k += [i0, i1, 0, i2]
    c2k += [i1, 0, i2, i0]
    tark += [0, i2, i0, i1]
    namesk += ["G0G1_K", "G1K_G2", "KG2_G0", "G2G0_G1"]
c1_t = torch.tensor(c1k, dtype=torch.long)
c2_t = torch.tensor(c2k, dtype=torch.long)
tar_t = torch.tensor(tark, dtype=torch.long)
w = torch.ones(len(c1k), dtype=torch.float32) / len(c1k)

# ---- Train and track ----
model = AttnModel(E0)
records = []


def svd_all():
    """SVD on all 4 parameter matrices. Return {name: (U,σ,Vh)} and σ₁/σ₄."""
    result = {}
    with torch.no_grad():
        for pname in ["E", "Wq", "Wk", "Wv"]:
            P = getattr(model, pname).detach().numpy()
            U, sv, Vh = np.linalg.svd(P, full_matrices=False)
            result[pname] = {
                "sv": sv.copy(),
                "Vh": Vh.copy(),  # right singular vectors (input-side)
                "U": U.copy(),  # left singular vectors (output-side)
                "svr": sv[0] / max(sv[-1], 1e-12),
            }
    return result


def cross_matrix_alignment(svd_dict):
    """Pairwise cosines between top-k right singular vectors of each matrix."""
    pairs = []
    names = ["E", "Wq", "Wk", "Wv"]
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            if i >= j:
                continue
            # Compare top right singular vectors (input-side, d-dim)
            v_i = svd_dict[ni]["Vh"][0, :]  # top right singular vector
            v_j = svd_dict[nj]["Vh"][0, :]
            cos = float(np.dot(v_i, v_j))
            pairs.append((f"{ni}↔{nj}", abs(cos)))
    return pairs


def gradient_decomposition():
    """
    Compute: for each param matrix, what fraction of its gradient
    norm projects onto E's top singular direction?
    """
    result = {}
    # Get E's top right singular direction (in hidden/dim space)
    En = model.E.detach().numpy()
    _, _, Vh_E = np.linalg.svd(En, full_matrices=False)
    v1_E = Vh_E[0, :]  # top right singular vector of E (d-dim)

    logits, _ = model(c1_t, c2_t)
    losses = F.cross_entropy(logits, tar_t, reduction="none")
    (losses * w).sum().backward()

    for pname in ["E", "Wq", "Wk", "Wv"]:
        gP = getattr(model, pname).grad.detach().numpy()
        gP_flat = gP.reshape(-1)
        total_norm_sq = float(np.dot(gP_flat, gP_flat))

        if pname == "E":
            # E: (V, d). Right multiply by v1_E projects rows onto v1_E.
            # gradient of E: for each token, the grad points in d-dim space.
            # proj of grad onto v1_E direction, summed over all tokens
            proj_sq = 0.0
            for tok in range(gP.shape[0]):
                proj_sq += float(np.dot(gP[tok], v1_E)) ** 2
            result[pname] = np.sqrt(proj_sq) / max(np.sqrt(total_norm_sq), 1e-12)
        else:
            # Wq, Wk, Wv: (d, d). Right singular vectors of these matrices
            # tell us which input (hidden) directions get amplified.
            # The gradient gP has shape (d,d). We want: how much of gP
            # corresponds to gradient flow along v1_E direction.
            #
            # For a weight matrix W, ∂L/∂W = (∂L/∂out) ⊗ input
            # The right singular vectors are the input-space directions.
            # Gradient contribution from v1_E: project ∂L/∂W onto outer
            # product span(v1_E) direction.
            # Approx: look at gradient rows (which correspond to output
            # dimensions) and see how much they point toward v1_E.
            #
            # Rough measure: ||gP @ v1_E||^2 / ||gP||^2
            gP_on_v1 = gP @ v1_E  # (d,)
            proj_sq = float(np.dot(gP_on_v1, gP_on_v1))
            result[pname] = np.sqrt(proj_sq) / max(np.sqrt(total_norm_sq), 1e-12)

        getattr(model, pname).grad = None

    model.Wq.grad = None
    model.Wk.grad = None
    model.Wv.grad = None
    return result


def hidden_rep_alignment():
    """cos(hidden, v1_E) for each pattern type."""
    logits, hidden = model(c1_t, c2_t)
    En = model.E.detach().numpy()
    _, _, Vh_E = np.linalg.svd(En, full_matrices=False)
    v1_E = Vh_E[0, :]
    result = {}
    for bt in set(namesk):
        idxs = [i for i, n in enumerate(namesk) if n == bt]
        cos_vals = []
        for i in idxs:
            h = hidden[i].detach().numpy()
            cos = float(
                np.dot(h, v1_E) / max(np.linalg.norm(h) * np.linalg.norm(v1_E), 1e-12)
            )
            cos_vals.append(abs(cos))
        result[bt] = np.mean(cos_vals)
    return result


print("TRAINING + TRACKING (every 50 steps)...")
print()

for step in range(max_steps):
    logits, _ = model(c1_t, c2_t)
    losses = F.cross_entropy(logits, tar_t, reduction="none")
    loss_val = float(losses.mean())
    (losses * w).sum().backward()
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                p -= lr * p.grad
                p.grad = None

    if step % 50 == 0 or step == max_steps - 1:
        pred = logits.argmax(-1)
        acc = float((pred == tar_t).float().mean())
        svd_d = svd_all()
        align = cross_matrix_alignment(svd_d)
        grad_d = gradient_decomposition()
        hid_d = hidden_rep_alignment()
        records.append(
            {
                "step": step,
                "loss": loss_val,
                "acc": acc,
                "svd": svd_d,
                "align": align,
                "grad_proj": grad_d,
                "hidden_cos": hid_d,
            }
        )

# ================================================================
# REPORT 1: Per-matrix spectral evolution
# ================================================================
print("=" * 90)
print("Q1: PER-MATRIX SPECTRAL CONCENTRATION OVER TIME")
print(f"{'Step':>6} {'E_σ₁₄':>8} {'Wq_σ₁₄':>8} {'Wk_σ₁₄':>8} {'Wv_σ₁₄':>8} {'acc':>7}  {'Eσ':>35}")
print("-" * 90)
n_show = min(15, len(records))
indices = np.linspace(0, len(records) - 1, n_show, dtype=int)
for i in indices:
    r = records[i]
    sv = r["svd"]
    e_sv = sv["E"]["sv"]
    print(
        f"{r['step']+1:6d} {sv['E']['svr']:8.3f} {sv['Wq']['svr']:8.3f} {sv['Wk']['svr']:8.3f} {sv['Wv']['svr']:8.3f} {r['acc']:7.2%}  [{' '.join(f'{x:.2f}' for x in e_sv)}]"
    )

# Final per-matrix full spectra
print(f"\nFINAL STATE (step {max_steps}) — full singular spectra:")
for pname in ["E", "Wq", "Wk", "Wv"]:
    sv = records[-1]["svd"][pname]["sv"]
    print(
        f"  {pname}: σ=[{' '.join(f'{x:.3f}' for x in sv)}], σ₁/σ₄={sv[0]/max(sv[-1],1e-12):.3f}"
    )

# ================================================================
# REPORT 2: Cross-matrix singular direction alignment
# ================================================================
print(f"\n{'='*90}")
print(
    f"Q2: CROSS-MATRIX ALIGNMENT (cos between top right singular vectors)"
)
print(f"{'Step':>6} | {'E↔Wq':>8} {'E↔Wk':>8} {'E↔Wv':>8} {'Wq↔Wk':>8} {'Wq↔Wv':>8} {'Wk↔Wv':>8}")
print("-" * 75)
for i in indices:
    r = records[i]
    align_dict = {a[0]: a[1] for a in r["align"]}
    print(
        f"{r['step']+1:6d} | "
        + " ".join(
            f"{align_dict.get(p,0):8.4f}"
            for p in ["E↔Wq", "E↔Wk", "E↔Wv", "Wq↔Wk", "Wq↔Wv", "Wk↔Wv"]
        )
    )

# ================================================================
# REPORT 3: Gradient decomposition — coupling strength
# ================================================================
print(f"\n{'='*90}")
print(
    f"Q3: GRADIENT DECOMPOSITION — fraction of each param's gradient along v1(E)"
)
print(
    f"{'Step':>6} {'E_g':>8} {'Wq_g':>8} {'Wk_g':>8} {'Wv_g':>8} {'G0G1_K':>10} {'G1K_G2':>10} {'KG2_G0':>10} {'G2G0_G1':>10}"
)
print("-" * 90)
for i in indices:
    r = records[i]
    g = r["grad_proj"]
    h = r["hidden_cos"]
    print(
        f"{r['step']+1:6d} {g['E']:8.4f} {g['Wq']:8.4f} {g['Wk']:8.4f} {g['Wv']:8.4f} "
        + " ".join(
            f"{h.get(p,0):10.4f}" for p in ["G0G1_K", "G1K_G2", "KG2_G0", "G2G0_G1"]
        )
    )

# ================================================================
# REPORT 4: Summary — what drives what?
# ================================================================
print(f"\n{'='*90}")
print("Q4: CAUSAL INTERPRETATION")
print("-" * 90)

final = records[-1]
svd_f = final["svd"]
align_f = {a[0]: a[1] for a in final["align"]}
grad_f = final["grad_proj"]

print()
print("1. WHICH MATRICES CONCENTRATE MOST?")
for pname in ["E", "Wq", "Wk", "Wv"]:
    print(
        f"   {pname}: σ₁/σ₄={svd_f[pname]['svr']:.3f}, σ=[{' '.join(f'{x:.3f}' for x in svd_f[pname]['sv'])}]"
    )

print()
print("2. WHICH PARAMETERS SHARE SINGULAR DIRECTIONS? (pairwise cos)")
for name, val in sorted(align_f.items(), key=lambda x: -x[1]):
    marker = " ← STRONG" if val > 0.7 else (" ← MODERATE" if val > 0.3 else "")
    print(f"   {name}: {val:.4f}{marker}")

print()
print("3. WHOSE GRADIENT IS MOST COUPLED TO E's COMMON DIRECTION?")
for pname in ["E", "Wq", "Wk", "Wv"]:
    print(
        f"   ∂L/∂{pname} projected onto v1(E): {grad_f[pname]:.1%}"
    )

print()
print("4. WHICH PATTERNS HAVE HIDDEN STATES ALIGNED WITH v1(E)?")
for bt in ["G0G1_K", "G1K_G2", "KG2_G0", "G2G0_G1"]:
    print(
        f"   {bt}: cos(hidden, v1_E) = {final['hidden_cos'][bt]:.4f}"
    )

print()
print("5. KEY INSIGHT:")
wv_conc = svd_f["Wv"]["svr"]
wq_conc = svd_f["Wq"]["svr"]
if wv_conc < wq_conc * 0.8:
    print(
        f"   ✓ Wv spectrum ({wv_conc:.2f}) is flatter than Wq ({wq_conc:.2f})"
    )
    print(f"     → Content (Wv) ≠ Routing (Wq/Wk). Confirms boss doc §4 prediction.")
else:
    print(
        f"   △ Wv concentration ({wv_conc:.2f}) similar to Wq ({wq_conc:.2f})"
    )
    print(f"     → Both routing and content share concentrated directions.")

e_wq_align = align_f.get("E↔Wq", 0)
e_wk_align = align_f.get("E↔Wk", 0)
if e_wq_align > 0.5 and e_wk_align > 0.5:
    print(
        f"   ✓ E's top direction strongly aligns with Wq ({e_wq_align:.2f}) and Wk ({e_wk_align:.2f})"
    )
    print(f"     → Representation common direction IS the query/key common direction.")
    print(f"     → MoE should gate at the attention routing level (Wq/Wk), not just E.")
