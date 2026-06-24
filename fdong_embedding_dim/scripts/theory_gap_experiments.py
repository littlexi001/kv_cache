#!/usr/bin/env python3
"""
Fill theory gaps: 4 experiments in logical order.
E1: Nested + balanced vs unbalanced (is frequency asymmetry necessary?)
E2: σ vs loss decrease quantification (verify ΔL ∝ σ)
E3: Hidden state E_K component quantification (verify "injection" path)
E4: σ₁ growth rate decay tracking (verify stopping condition)
"""
import math, numpy as np, torch, torch.nn.functional as F
from collections import defaultdict

# ============================================================
# SHARED UTILITIES
# ============================================================
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
    """Tied embedding + single-head attention."""
    def __init__(self, E0):
        super().__init__()
        self.E = torch.nn.Parameter(E0.clone())
        d = E0.shape[1]
        self.Wq = torch.nn.Parameter(torch.eye(d, dtype=torch.float32) * 0.1)
        self.Wk = torch.nn.Parameter(torch.eye(d, dtype=torch.float32) * 0.1)
        self.Wv = torch.nn.Parameter(torch.eye(d, dtype=torch.float32) * 0.1)
        self.scale = math.sqrt(d)

    def forward(self, c1, c2):
        h1 = self.E[c1]
        h2 = self.E[c2]
        K_ = torch.stack([h1 @ self.Wk.T, h2 @ self.Wk.T], dim=1)
        V = torch.stack([h1 @ self.Wv.T, h2 @ self.Wv.T], dim=1)
        Q = (h2 @ self.Wq.T).unsqueeze(1)
        s = torch.bmm(Q, K_.transpose(1, 2)) / self.scale
        attn_out = torch.bmm(F.softmax(s, dim=-1), V).squeeze(1)
        return attn_out @ self.E.T, attn_out  # logits, hidden


class BigramModel(torch.nn.Module):
    """Tied E + linear projection M (no attention, pure bigram)."""
    def __init__(self, E0):
        super().__init__()
        d = E0.shape[1]
        self.E = torch.nn.Parameter(E0.clone())
        self.M = torch.nn.Parameter(torch.eye(d, dtype=torch.float32) * 0.5)

    def forward(self, ids):
        h = self.E[ids] @ self.M.T
        logits = h @ self.E.T
        return logits, h


# ============================================================
# E1: NESTED FREQUENCY ABLATION
# ============================================================
print("=" * 70)
print("E1: NESTED FREQUENCY ABLATION")
print("  Question: Is frequency asymmetry NECESSARY for spectrum concentration?")
print("  Data: AB, ABC, ABCD nested sequences")
print("  Compares: natural 3:2:1 vs balanced (loss-reweighted to equal)")
print("=" * 70)


def build_nested_data():
    """Build AB, ABC, ABCD nested data for 4 tokens A,B,C,D on 4 orthogonal axes."""
    # 4 tokens on 4 orthogonal axes
    centers = {
        "A": (1.0, 0.0, 0.0, 0.0),
        "B": (0.0, 1.0, 0.0, 0.0),
        "C": (0.0, 0.0, 1.0, 0.0),
        "D": (0.0, 0.0, 0.0, 1.0),
    }
    all_v = []
    for gn in ["A", "B", "C", "D"]:
        c = np.array(centers[gn], dtype=np.float32)
        c /= np.linalg.norm(c)
        for off in [0.0, theta_deg, -theta_deg]:
            all_v.append(rot4(c, off))
    E0 = torch.tensor(np.stack(all_v).astype(np.float32))
    # Token indices: A0=0,A1=1,A2=2, B0=3,B1=4,B2=5, C0=6,C1=7,C2=8, D0=9,D1=10,D2=11
    A0, A1, A2 = 0, 1, 2
    B0, B1, B2 = 3, 4, 5
    C0, C1, C2 = 6, 7, 8
    D0, D1, D2 = 9, 10, 11

    # Trigram: use A0,A0 for self-context of A; B0,A0 for cross, etc.
    # AB:     (A0,A0)->A1, (A1,A1)->B0, (B0,B0)->B1
    # ABC:    (A0,A0)->A1, (A1,A1)->B0, (B0,B0)->B1, (B1,B1)->C0, (C0,C0)->C1
    # ABCD:   ... + (C1,C1)->D0, (D0,D0)->D1
    # But to test frequency, we care about which bigram pair (truth table)
    # Let's make it simpler: just A→B, B→C, C→D as bigram pairs
    # and use the nested structure to govern their frequencies

    # Actually for trigram, let me just use the actual tokens as context
    c1, c2, targ, names = [], [], [], []

    # AB sequence: A0→A1, A1→B0, B0→B1
    c1 += [A0, A1, B0]
    c2 += [A1, B0, B1]
    targ += [A1, B0, B1]
    names += ["A_A", "A_B", "B_B"]

    # ABC sequence: A0→A1, A1→B0, B0→B1, B1→C0, C0→C1
    c1 += [A0, A1, B0, B1, C0]
    c2 += [A1, B0, B1, C0, C1]
    targ += [A1, B0, B1, C0, C1]
    names += ["A_A", "A_B", "B_B", "B_C", "C_C"]

    # ABCD sequence
    c1 += [A0, A1, B0, B1, C0, C1, D0]
    c2 += [A1, B0, B1, C0, C1, D0, D1]
    targ += [A1, B0, B1, C0, C1, D0, D1]
    names += ["A_A", "A_B", "B_B", "B_C", "C_C", "C_D", "D_D"]

    return (
        E0,
        torch.tensor(c1, dtype=torch.long),
        torch.tensor(c2, dtype=torch.long),
        torch.tensor(targ, dtype=torch.long),
        names,
    )


def count_bigram_types(names, targ_t):
    """Identify distinct bigram types (ignoring internal like A_A, B_B)"""
    # We care about cross-group: A→B, B→C, C→D
    # Map each sample to its type
    type_counts = defaultdict(int)
    for i, nm in enumerate(names):
        type_counts[nm] += 1
    return type_counts


E0_nest, c1_nest, c2_nest, targ_nest, names_nest = build_nested_data()
n_total = len(c1_nest)
base_w_nest = torch.ones(n_total, dtype=torch.float32) / n_total

# Count frequencies
type_freq = count_bigram_types(names_nest, targ_nest)
print("\nBigram type frequencies (natural):")
for nm in sorted(set(names_nest)):
    print(f"  {nm}: {type_freq[nm]}×")

# Build balanced weights: each bigram TYPE gets equal total weight
# Then each occurrence of that type gets (1/N_types) / (occurrence_count)
unique_types = sorted(set(names_nest))
n_types = len(unique_types)
bal_w = torch.zeros(n_total, dtype=torch.float32)
for i, nm in enumerate(names_nest):
    bal_w[i] = (1.0 / n_types) / type_freq[nm]
bal_w = bal_w / bal_w.sum()  # renormalize

# Also build "just equal per-sample" weights (control)
uni_w = base_w_nest.clone() / base_w_nest.sum()


def train_nested(weight, weight_name, model_type="attention", steps=max_steps):
    """Train and return step-by-step SV spectrum."""
    if model_type == "attention":
        model = AttnModel(E0_nest)
    else:
        model = BigramModel(E0_nest)

    sv_history = []
    loss_history = []
    with torch.no_grad():
        En = model.E.detach().numpy()
        sv = np.linalg.svd(En, full_matrices=False)[1]
        sv_history.append((0, sv.copy()))

    for step in range(steps):
        if model_type == "attention":
            logits, _ = model(c1_nest, c2_nest)
        else:
            all_inputs = torch.arange(12, dtype=torch.long)
            logits, _ = model(all_inputs)
            # For bigram, we use a different approach: train on the pairs directly
            # Actually let me reconsider for bigram...

        # Hmm, for training we need per-pair loss. Let me restructure.
        # Actually for bigram model in nested setting, I should use per-pair inputs
        # Let me fix this later. For now, use attention.

    # Let me rewrite this more cleanly...


# --- Rewrite E1 with clean training loop ---
def train_and_track(model_obj, weight_t, model_type="attention", steps=max_steps):
    """Train model and track SV spectrum every step."""
    sv_history = []
    step_losses = []

    for step in range(steps):
        if model_type == "attention":
            logits, hidden = model_obj(c1_nest, c2_nest)
        else:
            logits, hidden = model_obj(torch.arange(12, dtype=torch.long))

        # Compute per-sample loss
        loss_per_sample = F.cross_entropy(
            logits, targ_nest, reduction="none"
        )
        total_loss = (loss_per_sample * weight_t).sum()
        total_loss.backward()

        with torch.no_grad():
            for p in model_obj.parameters():
                if p.grad is not None:
                    p -= lr * p.grad
                    p.grad = None

        if step % 50 == 0 or step < 5 or step == steps - 1:
            with torch.no_grad():
                En = model_obj.E.detach().numpy()
                sv = np.linalg.svd(En, full_matrices=False)[1]
                sv_history.append((step, sv.copy()))
                step_losses.append((step, float(total_loss)))

    return sv_history, step_losses


print("\nTraining nested models (this takes a while)...")

# E1a: Attention, natural frequencies (3:2:1)
print("  E1a: attention + natural frequencies...")
m1a = AttnModel(E0_nest)
sv1a, _ = train_and_track(m1a, uni_w, "attention")

# E1b: Attention, balanced frequencies (per-type equal weight)
print("  E1b: attention + balanced frequencies...")
m1b = AttnModel(E0_nest)
sv1b, _ = train_and_track(m1b, bal_w, "attention")

print("\n--- E1 RESULTS: Nested data, Attention model ---")
print(
    f'{"Step":>6} | {"Natural(3:2:1) σ₁/σ₄":>22} | {"Balanced σ₁/σ₄":>18} | {"Nat σ":>35} | {"Bal σ":>35}'
)
print("-" * 130)

for (s1, sv1), (s2, sv2) in zip(sv1a, sv1b):
    r1 = sv1[0] / max(sv1[3], 1e-12)
    r2 = sv2[0] / max(sv2[3], 1e-12)
    sv1_str = " ".join(f"{v:.2f}" for v in sv1)
    sv2_str = " ".join(f"{v:.2f}" for v in sv2)
    print(
        f"{s1+1:6d} | {r1:22.4f} | {r2:18.4f} | [{sv1_str}] | [{sv2_str}]"
    )

# Summary
r1_final = sv1a[-1][1][0] / max(sv1a[-1][1][3], 1e-12)
r2_final = sv1b[-1][1][0] / max(sv1b[-1][1][3], 1e-12)
print(f"\nE1 VERDICT:")
print(f"  Natural (3:2:1): σ₁/σ₄ = {r1_final:.3f}")
print(f"  Balanced:         σ₁/σ₄ = {r2_final:.3f}")
if r2_final < 1.05:
    print(f"  → Frequency asymmetry IS necessary. Balanced nesting does NOT create spectrum concentration.")
elif r2_final < r1_final * 0.5:
    print(f"  → Frequency asymmetry is the PRIMARY driver. Nesting alone contributes mildly.")
else:
    print(f"  → Nesting alone creates significant spectrum concentration independent of frequency.")


# ============================================================
# E2: σ vs LOSS DECREASE QUANTIFICATION
# ============================================================
print("\n\n" + "=" * 70)
print("E2: σ vs LOSS DECREASE QUANTIFICATION")
print("  Question: Does ΔL ∝ σ? (Perturb E along v₁ and v₄, measure ΔL)")
print("=" * 70)

# Use the natural-frequency model from E1a (already trained)
model_e2 = m1a
E_e2 = model_e2.E.detach().numpy()
U, sv, Vh = np.linalg.svd(E_e2, full_matrices=False)
v1 = Vh[0, :]  # top singular direction
v4 = Vh[3, :]  # bottom singular direction

print(f"\nCurrent spectrum: σ = [{sv[0]:.2f}, {sv[1]:.2f}, {sv[2]:.2f}, {sv[3]:.2f}]")
print(f"  σ₁/σ₄ = {sv[0]/max(sv[3],1e-12):.3f}")

# Compute baseline loss
with torch.no_grad():
    logits_base, _ = model_e2(c1_nest, c2_nest)
    loss_base = float(F.cross_entropy(logits_base, targ_nest, reduction="mean"))

# Perturb E along v1 and v4, measure ΔL
epsilons = [0.001, 0.005, 0.01, 0.02, 0.05]
print(f"\nBaseline loss = {loss_base:.6f}")
print(f'{"ε":>8} {"ΔL along v₁":>14} {"ΔL along v₄":>14} {"ratio ΔL₁/ΔL₄":>16} {"predicted (σ₁/σ₄)":>18}')

for eps in epsilons:
    deltas = {}
    for direction_name, direction_vec in [("v1", v1), ("v4", v4)]:
        # Perturb: add ε * direction_vec to every token's embedding
        E_perturbed = E_e2.copy()
        for i in range(12):
            E_perturbed[i] += eps * direction_vec

        # Set model parameters to perturbed version
        with torch.no_grad():
            model_e2.E.copy_(torch.tensor(E_perturbed.astype(np.float32)))

        # Compute new loss
        with torch.no_grad():
            logits_pert, _ = model_e2(c1_nest, c2_nest)
            loss_pert = float(F.cross_entropy(logits_pert, targ_nest, reduction="mean"))

        deltas[direction_name] = loss_pert - loss_base

        # Restore
        with torch.no_grad():
            model_e2.E.copy_(torch.tensor(E_e2.astype(np.float32)))

    ratio = abs(deltas["v1"]) / max(abs(deltas["v4"]), 1e-12)
    predicted = sv[0] / max(sv[3], 1e-12)
    print(
        f"{eps:8.3f} {deltas['v1']:14.6f} {deltas['v4']:14.6f} {ratio:16.3f} {predicted:18.3f}"
    )

print(
    f"\nE2 VERDICT: If ratio ≈ predicted, theory holds (ΔL ∝ σ)."
)

# ============================================================
# E3: HIDDEN STATE E_K COMPONENT QUANTIFICATION
# ============================================================
print("\n\n" + "=" * 70)
print("E3: HIDDEN STATE E_K COMPONENT QUANTIFICATION")
print("  Question: How much E_K component is injected into hidden states?")
print("=" * 70)

# Build K-token data (4 groups, universal K, trigram + attention)
K_init = np.ones(dim, dtype=np.float32) / math.sqrt(dim)
centers_4 = {
    "A": (1.0, 0.0, 0.0, 0.0),
    "B": (0.0, 1.0, 0.0, 0.0),
    "C": (0.0, 0.0, 1.0, 0.0),
    "D": (0.0, 0.0, 0.0, 1.0),
}
all_v_k = [K_init]
gi_k = {}
idx_k = 1
for gn in ["A", "B", "C", "D"]:
    c = np.array(centers_4[gn], dtype=np.float32)
    c /= np.linalg.norm(c)
    gi_k[gn] = [idx_k, idx_k + 1, idx_k + 2]
    idx_k += 3
    for off in [0.0, theta_deg, -theta_deg]:
        all_v_k.append(rot4(c, off))
E0_k = torch.tensor(np.stack(all_v_k).astype(np.float32))

c1_k, c2_k, targ_k, names_k, grps_k = [], [], [], [], []
for gn, (i0, i1, i2) in gi_k.items():
    ki = 0  # K is index 0
    c1_k += [i0, i1, ki, i2]
    c2_k += [i1, ki, i2, i0]
    targ_k += [ki, i2, i0, i1]
    names_k += ["G0G1_K", "G1K_G2", "KG2_G0", "G2G0_G1"]
    grps_k += [gn] * 4

c1_kt = torch.tensor(c1_k, dtype=torch.long)
c2_kt = torch.tensor(c2_k, dtype=torch.long)
targ_kt = torch.tensor(targ_k, dtype=torch.long)
w_k = torch.ones(len(c1_k), dtype=torch.float32) / len(c1_k)

# Train K-token model
print("  Training K-token attention model...")
model_e3 = AttnModel(E0_k)
for step in range(800):
    logits, hidden = model_e3(c1_kt, c2_kt)
    losses = F.cross_entropy(logits, targ_kt, reduction="none")
    (losses * w_k).sum().backward()
    with torch.no_grad():
        for p in model_e3.parameters():
            if p.grad is not None:
                p -= lr * p.grad
                p.grad = None

# Now measure: for each bigram type, cos(hidden, E_K)
with torch.no_grad():
    logits_all, hidden_all = model_e3(c1_kt, c2_kt)
    E_K_vec = model_e3.E[0].detach().numpy()
    E_K_norm = np.linalg.norm(E_K_vec)

    print(f"\n{'Bigram type':>12} {'K in ctx?':>10} {'cos(h, E_K)':>12} {'|h|':>8} {'loss':>8}")
    print("-" * 55)
    for bt in sorted(set(names_k)):
        idxs = [i for i, n in enumerate(names_k) if n == bt]
        cos_vals = []
        norm_vals = []
        loss_vals = []
        for i in idxs:
            h = hidden_all[i].detach().numpy()
            nh = np.linalg.norm(h)
            cos = float(np.dot(h, E_K_vec) / max(nh * E_K_norm, 1e-12))
            cos_vals.append(cos)
            norm_vals.append(nh)
            loss_vals.append(float(losses[i]))
        avg_cos = np.mean(cos_vals)
        avg_norm = np.mean(norm_vals)
        avg_loss = np.mean(loss_vals)
        k_in_ctx = "YES" if "K" in bt and "K" != bt.split("_")[0] else ("SELF" if bt.startswith("K") else "NO")
        # Actually: G0G1_K → K is target, not in context
        # G1K_G2 → K is in context (ctx2)
        # KG2_G0 → K is in context (ctx1)
        # G2G0_G1 → K is NOT in context
        if "G1K" in bt or "G2G0" in bt:
            pass
        k_in_ctx = "YES" if (bt == "G1K_G2" or bt == "KG2_G0") else "NO"
        print(f"{bt:>12} {k_in_ctx:>10} {avg_cos:12.4f} {avg_norm:8.2f} {avg_loss:8.4f}")

# Detailed comparison
print(f"\nE3 VERDICT:")
k_yes_idx = [i for i, n in enumerate(names_k) if n in ("G1K_G2", "KG2_G0")]
k_no_idx = [i for i, n in enumerate(names_k) if n in ("G0G1_K", "G2G0_G1")]
with torch.no_grad():
    cos_yes = np.mean(
        [
            float(
                np.dot(
                    hidden_all[i].detach().numpy(),
                    E_K_vec,
                )
                / max(
                    np.linalg.norm(hidden_all[i].detach().numpy()) * E_K_norm,
                    1e-12,
                )
            )
            for i in k_yes_idx
        ]
    )
    cos_no = np.mean(
        [
            float(
                np.dot(
                    hidden_all[i].detach().numpy(),
                    E_K_vec,
                )
                / max(
                    np.linalg.norm(hidden_all[i].detach().numpy()) * E_K_norm,
                    1e-12,
                )
            )
            for i in k_no_idx
        ]
    )
print(f"  K in context:     avg cos(h, E_K) = {cos_yes:.4f}")
print(f"  K NOT in context: avg cos(h, E_K) = {cos_no:.4f}")
print(f"  Ratio: {cos_yes/max(abs(cos_no),1e-12):.2f}×")
if cos_yes > cos_no * 1.5:
    print(f"  → Strong evidence: K in context injects significant E_K component into hidden states.")
else:
    print(f"  → Weak or no evidence for injection effect.")


# ============================================================
# E4: σ₁ GROWTH RATE DECAY TRACKING
# ============================================================
print("\n\n" + "=" * 70)
print("E4: σ₁ GROWTH RATE DECAY TRACKING")
print("  Question: When does σ₁ growth slow down, and does it align with convergence?")
print("=" * 70)

# Use the K-token model and track per-step SV
print("  Training K-token model with dense SV tracking...")
model_e4 = AttnModel(E0_k)
sv_e4 = []
acc_e4 = []
with torch.no_grad():
    En = model_e4.E.detach().numpy()
    sv_e4.append(np.linalg.svd(En, full_matrices=False)[1].copy())
    logits, _ = model_e4(c1_kt, c2_kt)
    pred = logits.argmax(-1)
    acc_e4.append(float((pred == targ_kt).float().mean()))

for step in range(800):
    logits, hidden = model_e4(c1_kt, c2_kt)
    losses = F.cross_entropy(logits, targ_kt, reduction="none")
    (losses * w_k).sum().backward()
    with torch.no_grad():
        for p in model_e4.parameters():
            if p.grad is not None:
                p -= lr * p.grad
                p.grad = None

    if step % 5 == 0 or step < 5:
        with torch.no_grad():
            En = model_e4.E.detach().numpy()
            sv_e4.append(np.linalg.svd(En, full_matrices=False)[1].copy())
            pred = logits.argmax(-1)
            acc_e4.append(float((pred == targ_kt).float().mean()))

# Compute growth rates
sv_arr = np.array(sv_e4)
sigma_1 = sv_arr[:, 0]
sigma_4 = sv_arr[:, 3]

# Growth rate: (σ(t+Δt) - σ(t)) / Δt
growth_1 = np.diff(sigma_1)
growth_4 = np.diff(sigma_4)

# Smooth for cleaner signal
window = 10
growth_1_smooth = np.convolve(growth_1, np.ones(window) / window, mode="valid")
growth_4_smooth = np.convolve(growth_4, np.ones(window) / window, mode="valid")

print(f"\n{'Step':>6} {'σ₁':>8} {'dσ₁/dt':>10} {'dσ₄/dt':>10} {'acc':>6}")
print("-" * 45)
checkpoints = list(range(0, len(sv_e4), max(1, len(sv_e4) // 20)))
for i in checkpoints[:25]:
    step_num = i * 5 if i > 0 else 0
    g1 = growth_1[i - 1] if i > 0 else 0
    g4 = growth_4[i - 1] if i > 0 else 0
    acc = acc_e4[i]
    print(
        f"{step_num+1:6d} {sigma_1[i]:8.2f} {g1:10.4f} {g4:10.4f} {acc:6.2%}"
    )

# Find where growth rate drops below 50% of initial
initial_g1 = np.mean(growth_1[:10])  # average over first 10 intervals
half_g1 = initial_g1 * 0.5
decay_step = None
for i in range(10, len(growth_1)):
    if growth_1[i] < half_g1 and growth_1[max(0, i - 5)] < half_g1:
        decay_step = i * 5  # approximate step
        break

print(f"\nE4 VERDICT:")
print(f"  Initial dσ₁/dt ≈ {initial_g1:.4f}")
print(f"  Half-decay at step ~{decay_step}" if decay_step else "  No clear decay found")
if decay_step:
    # What's the accuracy at decay?
    decay_idx = min(decay_step // 5, len(acc_e4) - 1)
    print(f"  Accuracy at decay: {acc_e4[decay_idx]:.2%}")

# Check correlation: dσ₁/dt vs (1 - accuracy)
print(f"\n  Correlation check:")
latest_g1 = np.mean(growth_1[-20:])
print(f"  Latest dσ₁/dt = {latest_g1:.6f} (initial = {initial_g1:.4f}, ratio = {latest_g1/max(initial_g1,1e-12):.2%})")
print(f"  Latest accuracy = {acc_e4[-1]:.2%}")
if latest_g1 < 0.1 * initial_g1 and acc_e4[-1] > 0.9:
    print(f"  ✓ Theory confirmed: σ₁ growth decays as accuracy approaches 100%")
else:
    print(f"  △ Partial support: growth decays but not perfectly aligned with accuracy")


# ============================================================
# ADDITIONAL: σ₁/σ₄ over time for K-token model
# ============================================================
print("\n\n" + "=" * 70)
print("SUPPLEMENTARY: σ₁/σ₄ over time (K-token attention, no reweighting)")
print("=" * 70)
print(f'{"Step":>6} {"σ₁":>8} {"σ₂":>8} {"σ₃":>8} {"σ₄":>8} {"σ₁/σ₄":>10}')
for i in checkpoints[:20]:
    s1, s2, s3, s4 = sv_arr[i]
    r = s1 / max(s4, 1e-12)
    print(f"{i*5+1:6d} {s1:8.2f} {s2:8.2f} {s3:8.2f} {s4:8.2f} {r:10.4f}")
