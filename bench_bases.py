"""
Why neural bases fail on this Helmholtz problem -- measured, without training.

A physics-informed network minimises the PDE residual over some basis of
functions. The usual explanation for poor results is spectral bias, and the
usual remedy is a better basis: Fourier features, SIREN, Gabor atoms, domain
decomposition. This script tests that explanation directly, and it needs no
training to do so.

For each family, at random initialisation, it measures two things:

  1. PROJECTION ERROR -- the best possible least-squares fit of the verified
     FDM field onto the basis. This is capacity: can the basis represent P at
     all?

  2. RESIDUAL OF THAT SAME PROJECTION -- does that best-possible field satisfy
     the equation? Normalised so that 100 % is exactly as bad as the zero
     field.

The second number is the one that matters, and it is the one nobody reports.

--------------------------------------------------------------------------
THE RESULT
--------------------------------------------------------------------------
Every family represents the field to better than 0.4 %. Every family leaves a
PDE residual of hundreds of thousands of percent. The best of them, Gabor, is
still some 2500 times worse than writing down zero.

The reason is not capacity and not conditioning. In this geometry the
Laplacian amplifies a representation error by 1/L^2 = 625, where L is the neck
length, and the true solution lives on a near-cancellation: lap(P) and -k^2 P
are both enormous and their difference is small. Representing P to 0.15 % is
nowhere near enough to represent lap(P) at all.

What the basis does change is the conditioning of the system actually being
solved, kappa(A_phys), and there the spread is eleven orders of magnitude --
a real result, and a real reason to prefer sine activations. It is simply not
the binding constraint.

Trefftz is the exception that proves the point: its functions satisfy
lap(phi) + k^2 phi = 0 exactly, so A_phys is identically zero. It cannot
misrepresent the operator because it never approximates it -- and for the same
reason it cannot carry the source, which is why its residual sits at 100 %.
A Trefftz method needs a particular solution supplied separately.

Runs in about two minutes on a CPU. No training, no weights, no checkpoints.

Output: data/bench_bases.npz, plots/bench_bases.png
Env: BB_HID (96)  BB_F (209.84)  BB_SEED (0)
"""
import os

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
torch.set_default_dtype(torch.float64)

SEED = int(os.environ.get("BB_SEED", 0))
HID = int(os.environ.get("BB_HID", 96))
FREQ = float(os.environ.get("BB_F", 209.84))

C, RHO = 343.0, 1.204
R_NECK, R_CAV, L_NECK, H_CAV = 0.01, 0.04, 0.04, 0.08
Z_TOP = L_NECK + H_CAV
SRC_Z, SRC_W, SRC_A = L_NECK + 0.5 * H_CAV, 0.01, 1.0e4
K2 = (2 * np.pi * FREQ / C) ** 2


# --------------------------------------------------------------------------
# The verified reference: the same harmonic solver as fdm_sweep.py
# --------------------------------------------------------------------------
def fdm(freq, h=1e-3):
    w = 2 * np.pi * freq
    k2 = (w / C) ** 2
    ka = w / C * R_NECK
    zr = RHO * C * (0.5 * ka ** 2 + 1j * (8 / (3 * np.pi)) * ka)
    al = 1j * w * RHO / zr
    nr = int(round(R_CAV / h)) + 1
    nz = int(round(Z_TOP / h)) + 1
    r = np.arange(nr) * h
    z = np.arange(nz) * h
    rr, zz = np.meshgrid(r, z, indexing="ij")
    fl = (((zz < L_NECK - 1e-12) & (rr <= R_NECK + 1e-12))
          | ((zz >= L_NECK - 1e-12) & (rr <= R_CAV + 1e-12)))
    mo = fl & (np.abs(zz) < 1e-12) & (rr <= R_NECK + 1e-12)

    def idx(i, j):
        return i + j * nr

    n, ih2 = nr * nz, 1.0 / h ** 2
    forcing = SRC_A * np.exp(-(rr ** 2 + (zz - SRC_Z) ** 2) / (2 * SRC_W ** 2))
    rows, cols, dat = [], [], []
    b = np.zeros(n, complex)
    for i in range(nr):
        for j in range(nz):
            ci = idx(i, j)
            if not fl[i, j]:
                rows += [ci]; cols += [ci]; dat += [1.0]
                continue
            diag = k2 + 0j
            if i == 0:
                diag += -4 * ih2
                rows += [ci]; cols += [idx(1, j)]; dat += [4 * ih2]
            else:
                lc, rc = ih2 * (1 - 0.5 / i), ih2 * (1 + 0.5 / i)
                diag += -2 * ih2
                if fl[i - 1, j]:
                    rows += [ci]; cols += [idx(i - 1, j)]; dat += [lc]
                else:
                    diag += lc
                if i + 1 < nr and fl[i + 1, j]:
                    rows += [ci]; cols += [idx(i + 1, j)]; dat += [rc]
                else:
                    diag += rc
            if mo[i, j]:
                diag += -2 * ih2 - 2 * al / h
                rows += [ci]; cols += [idx(i, j + 1)]; dat += [2 * ih2]
            else:
                diag += -2 * ih2
                for jj in (j - 1, j + 1):
                    if 0 <= jj < nz and fl[i, jj]:
                        rows += [ci]; cols += [idx(i, jj)]; dat += [ih2]
                    else:
                        diag += ih2
            rows += [ci]; cols += [ci]; dat += [diag]
            b[ci] = -forcing[i, j]
    a = sp.coo_matrix((dat, (rows, cols)), shape=(n, n), dtype=complex).tocsr()
    return r, z, fl, spsolve(a, b).reshape((nz, nr)).T


r_ref, z_ref, FL, P_REF = fdm(FREQ)
ii, jj = np.where(FL)
RG, ZG = r_ref[ii], z_ref[jj]
PREF = P_REF[FL]
print(f"reference FDM: {FL.sum()} nodes, max|P| = {np.abs(PREF).max():.1f} Pa, f = {FREQ} Hz")
print(f"{HID} basis functions, random initialisation, NO training\n")

X = torch.tensor(np.stack([RG / R_CAV, ZG / Z_TOP], 1))     # normalised inputs
XR = torch.tensor(np.stack([RG, ZG], 1))                    # physical coordinates


# ==========================================================================
# The families of bases
# ==========================================================================
def mlp(act, layers=5, w0=1.0, siren=False):
    """phi(x): (N,2) -> (N,HID), at random weights."""
    torch.manual_seed(SEED)
    mods, d = [], 2
    for i in range(layers):
        lin = nn.Linear(d, HID)
        if siren:
            with torch.no_grad():
                b = (1.0 / d) if i == 0 else (np.sqrt(6.0 / d) / w0)
                lin.weight.uniform_(-b, b)
                lin.bias.uniform_(-b, b)
        mods += [lin, act()]
        d = HID
    net = nn.Sequential(*mods)
    return lambda x: net(x)


class Sine(nn.Module):
    def __init__(self, w0=30.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


def gabor_basis(nb, sig_lo, sig_hi, kmax):
    """Gabor atoms: a localised Gaussian times an oscillation."""
    rng = np.random.default_rng(SEED)
    cr = torch.tensor(rng.uniform(0, R_CAV, nb))
    cz = torch.tensor(rng.uniform(0, Z_TOP, nb))
    sg = torch.tensor(rng.uniform(sig_lo, sig_hi, nb))
    kr = torch.tensor(rng.normal(0, kmax, nb))
    kz = torch.tensor(rng.normal(0, kmax, nb))
    ph = torch.tensor(rng.uniform(0, 2 * np.pi, nb))

    def f(xr):
        d2 = (xr[:, 0:1] - cr) ** 2 + (xr[:, 1:2] - cz) ** 2
        return torch.exp(-d2 / (2 * sg ** 2)) * torch.cos(
            kr * xr[:, 0:1] + kz * xr[:, 1:2] + ph)
    return f


def local_basis(nb, w0=8.0):
    """DOMAIN DECOMPOSITION: one small network per zone, blended by a partition
    of unity. The zones overlap, so no interface constraint is needed -- this
    is the FBPINN construction."""
    torch.manual_seed(SEED)
    zc = [0.5 * L_NECK, L_NECK, L_NECK + 0.55 * H_CAV]     # neck, junction, cavity
    zs = [0.9 * L_NECK, 0.5 * L_NECK, 0.75 * H_CAV]        # widths, overlapping
    per = nb // 3
    nets = []
    for _ in range(3):
        mods, d = [], 2
        for i in range(3):
            lin = nn.Linear(d, per)
            with torch.no_grad():
                b = (1.0 / d) if i == 0 else (np.sqrt(6.0 / d) / w0)
                lin.weight.uniform_(-b, b)
                lin.bias.uniform_(-b, b)
            mods += [lin, Sine(w0)]
            d = per
        nets.append(nn.Sequential(*mods))

    def f(x):
        z = x[:, 1:2] * Z_TOP                              # normalised -> physical z
        ws = [torch.exp(-0.5 * ((z - c) / sg) ** 2) for c, sg in zip(zc, zs)]
        tot = sum(ws)
        return torch.cat([(w / tot) * n(x) for w, n in zip(ws, nets)], 1)
    return f


def trefftz_basis(nb):
    """TREFFTZ: every function satisfies lap(phi) + k^2 phi = 0 EXACTLY.

    In axisymmetry, separation gives phi = J0(kappa r) Z(z) with
    Z'' = (kappa^2 - k^2) Z. Taking kappa_m = j'_(0,m)/R_CAV, the zeros of
    J0', also satisfies the lateral cavity wall by construction. The PDE
    residual can then only come from the source and the other boundaries --
    never from the approximation.
    """
    from scipy.special import jn_zeros, j0
    kap = np.concatenate([[0.0], jn_zeros(1, nb // 2) / R_CAV])
    fns = []
    for km in kap:
        d = km ** 2 - K2
        if d < 0:
            b = np.sqrt(-d)
            zf = [lambda z, b=b: np.cos(b * z), lambda z, b=b: np.sin(b * z)]
        else:
            b = np.sqrt(d)
            zf = [lambda z, b=b: np.cosh(b * np.clip(z, 0, Z_TOP)) / np.cosh(b * Z_TOP),
                  lambda z, b=b: np.sinh(b * np.clip(z, 0, Z_TOP)) / np.cosh(b * Z_TOP)]
        for zz in zf:
            fns.append((km, zz))
        if len(fns) >= nb:
            break
    fns = fns[:nb]

    def f(xr):
        r = xr[:, 0].detach().numpy()
        z = xr[:, 1].detach().numpy()
        return torch.stack([torch.tensor(j0(km * r) * zz(z)) for km, zz in fns], 1)
    return f


BASES = [
    ("tanh (reference)",      lambda: mlp(nn.Tanh),                             "norm"),
    ("SIREN, w0 = 5",         lambda: mlp(lambda: Sine(5.), siren=True, w0=5.), "norm"),
    ("SIREN, w0 = 8",         lambda: mlp(lambda: Sine(8.), siren=True, w0=8.), "norm"),
    ("Gabor atoms",           lambda: gabor_basis(HID, 0.005, 0.02, 60.0),      "phys"),
    ("domain decomposition",  lambda: local_basis(HID, 8.0),                    "norm"),
    ("Trefftz (Bessel)",      lambda: trefftz_basis(HID),                    "trefftz"),
]


def laplacian_of(fn, xr_np, mode):
    """Axisymmetric lap(phi) by autodiff, for all HID functions at once."""
    xr = torch.tensor(xr_np, requires_grad=True)
    inp = xr if mode == "phys" else torch.stack([xr[:, 0] / R_CAV, xr[:, 1] / Z_TOP], 1)
    out = fn(inp)
    lap = torch.zeros_like(out)
    for j in range(out.shape[1]):
        g = torch.autograd.grad(out[:, j].sum(), xr, create_graph=True)[0]
        gr, gz = g[:, 0], g[:, 1]
        grr = torch.autograd.grad(gr.sum(), xr, create_graph=True)[0][:, 0]
        gzz = torch.autograd.grad(gz.sum(), xr, create_graph=True)[0][:, 1]
        r = xr[:, 0]
        inv = torch.where(r < 1e-6, torch.zeros_like(r), 1.0 / torch.clamp(r, min=1e-6))
        lap[:, j] = grr + torch.where(r < 1e-6, grr, gr * inv) + gzz
    return out.detach(), lap.detach()


# subsample for the physics matrix (autodiff over HID outputs is the cost)
sel = np.random.default_rng(0).choice(RG.size, size=min(900, RG.size), replace=False)
XSEL = np.stack([RG[sel], ZG[sel]], 1)
FSEL = SRC_A * np.exp(-(XSEL[:, 0] ** 2 + (XSEL[:, 1] - SRC_Z) ** 2) / (2 * SRC_W ** 2))

print(f"{'basis':22s} {'field error':>12s} {'residual of that fit':>22s} "
      f"{'kappa(A_phys)':>14s} {'rank A':>9s}")
print("-" * 84)
print("  The residual is the number that matters: it says whether the basis")
print("  represents lap(P), not merely P. 100 % is as bad as the zero field.")
print("-" * 84)

rows = []
for name, make, mode in BASES:
    fn = make()
    with torch.no_grad():
        phi = fn(X if mode == "norm" else XR).numpy()

    # capacity: the best possible projection of the FDM solution
    thr, *_ = np.linalg.lstsq(phi, PREF.real, rcond=None)
    thi, *_ = np.linalg.lstsq(phi, PREF.imag, rcond=None)
    fit = phi @ thr + 1j * (phi @ thi)
    err = np.linalg.norm(fit - PREF) / np.linalg.norm(PREF)

    # the PHYSICS matrix: this is what is actually being solved
    if mode == "trefftz":
        # lap(phi) + k^2 phi = 0 by construction, so A is identically zero.
        with torch.no_grad():
            m_mat = fn(torch.tensor(XSEL))
        lap = -K2 * m_mat
    else:
        m_mat, lap = laplacian_of(fn, XSEL, mode)
    a_phys = (lap + K2 * m_mat).numpy()
    sa = np.linalg.svd(a_phys, compute_uv=False)
    kappa_a = sa[0] / max(sa[-1], 1e-300)
    rank_a = int((sa > sa[0] * 1e-12).sum())

    # THE test: does the best-represented field produce an acceptable residual?
    t2r, *_ = np.linalg.lstsq(m_mat.numpy(), PREF[sel].real, rcond=None)
    t2i, *_ = np.linalg.lstsq(m_mat.numpy(), PREF[sel].imag, rcond=None)
    resid = np.sqrt(np.mean((a_phys @ t2r + FSEL) ** 2 + (a_phys @ t2i) ** 2)) \
        / np.sqrt(np.mean(FSEL ** 2))

    rows.append((name, err, resid, kappa_a, rank_a))
    print(f"{name:22s} {err * 100:11.2f}% {resid * 100:21.1f}% "
          f"{kappa_a:14.2e} {rank_a:6d}/{HID:<3d}")

print()
print("Reading the table:")
print("  field error   can the basis represent P at all?          (small = good)")
print("  residual      does that same field satisfy the equation? (small = good)")
print("  kappa(A_phys) conditioning of the system actually solved.")
print()
neural = [r for r in rows if not r[0].startswith("Trefftz")]
best = min(neural, key=lambda r: r[2])
print(f"Best neural basis: {best[0]}, field error {best[1] * 100:.2f} %, "
      f"residual {best[2] * 100:.0f} %")
print(f"  -- that is {best[2]:.0f}x worse than writing down the zero field, while")
print(f"     representing the field itself to better than {best[1] * 100:.2f} %.")
print("  The obstacle is not capacity. The Laplacian amplifies representation")
print("  error by 1/L^2 = 625, and the solution lives on a near-cancellation.")

# --------------------------------------------------------------------------
os.makedirs("data", exist_ok=True)
os.makedirs("plots", exist_ok=True)
np.savez("data/bench_bases.npz",
         names=np.array([r[0] for r in rows]),
         field_error=np.array([r[1] for r in rows]),
         residual=np.array([r[2] for r in rows]),
         kappa=np.array([r[3] for r in rows]),
         rank=np.array([r[4] for r in rows]),
         freq=FREQ, hidden=HID)

fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
names = [r[0] for r in rows]
pos = np.arange(len(rows))
colours = ["#a8442a" if n.startswith("tanh") else
           "#6b7f99" if n.startswith("Trefftz") else "#1d5480" for n in names]

ax[0].barh(pos, [r[1] * 100 for r in rows], color=colours)
ax[0].set_yticks(pos); ax[0].set_yticklabels(names, fontsize=9)
ax[0].invert_yaxis()
ax[0].set_xlabel("field error (%)")
ax[0].set_title("Capacity: every basis represents $P$ well")
ax[0].grid(ls=":", alpha=0.6, axis="x")

ax[1].barh(pos, [max(r[2] * 100, 1e-1) for r in rows], color=colours)
ax[1].axvline(100, color="k", ls="--", lw=1.4)
ax[1].annotate("the zero field", (100, len(rows) - 0.4), fontsize=9,
               rotation=90, va="bottom", ha="right")
ax[1].set_xscale("log")
ax[1].set_yticks(pos); ax[1].set_yticklabels([])
ax[1].invert_yaxis()
ax[1].set_xlabel("PDE residual of that same field (%), log scale")
ax[1].set_title("Reality: none of them represents $\\nabla^2 P$")
ax[1].grid(ls=":", alpha=0.6, axis="x", which="both")

fig.tight_layout()
fig.savefig("plots/bench_bases.png", dpi=140)
plt.close(fig)
print("\nFigure: plots/bench_bases.png | Data: data/bench_bases.npz")
