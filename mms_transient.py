"""
Manufactured solution for the TRANSIENT solver (code verification).

Fills a gap in the project: the MMS of the harmonic part runs on a rectangular
domain with DIRICHLET edges, so it tests neither the mask nor the treatment of
the rigid walls. What is verified explicitly here:
  * the leapfrog time scheme;
  * the axisymmetric Laplacian in MASKED FINITE VOLUMES;
  * the treatment of the AXIS r = 0;
  * the RIGID WALLS obtained by cancelling the flux (face radius = 0).

--------------------------------------------------------------------------
WHAT THIS TEST REVEALED
--------------------------------------------------------------------------
In masked finite volumes the wall is NOT on the last node: it lies HALF A CELL
beyond it, on the face of the last cell. The domain actually simulated is
therefore

        radius  R_eff = R + h/2          (one lateral wall)
        height  H_eff = H + h            (two walls, one at each end)

The consequence, measured below: if the manufactured solution makes its
derivative vanish at R (the nominal geometry), the observed order drops to 1 --
the h/2 offset is a first-order geometric error. Matching it to R + h/2
restores the expected order 2, with errors about 27 times smaller.

This is not a defect of the solver, it is its convention. But it implies a
GEOMETRIC BIAS of order h on the effective dimensions, which has to be
accounted for when a natural frequency is compared with an analytic theory.

--------------------------------------------------------------------------
Manufactured solution, solid cylinder, all walls rigid:

    p(r,z,t) = J0(lambda*r) * cos(kz*(z + h/2)) * T(t)
    lambda = j / R_eff   with J1(j) = 0     ->  dp/dr = 0 at r = R_eff
    kz     = n*pi / H_eff                   ->  dp/dz = 0 at z = -h/2 and H+h/2

    laplacian(p) = -(lambda^2 + kz^2) * p          (exact)
    source F     = p_tt - c^2 * laplacian(p)

T(t) = u^3 exp(-u), u = t/tau, with tau = 1/(c*K): matched to the time scale of
the problem so that T'' and c^2 K^2 T are of the same order. Otherwise the
solution is a near-cancellation of two large terms and the error constant
explodes. T(0) = T'(0) = T''(0) = 0, so the leapfrog start from rest is exact
to third order, well below the error of the scheme.

Output: data/mms_transient.npz, plots/mms_transient.png
"""
import os, time
import numpy as np
from scipy.special import j0, jn_zeros
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)

C = 343.0
R_DOM, H_DOM = 0.04, 0.12
N_MODE = 2
CFL = 0.40
J1Z = jn_zeros(1, 1)[0]


def run(h, half_cell):
    """One run. half_cell=True: solution matched to the EFFECTIVE geometry."""
    R_eff = R_DOM + (0.5*h if half_cell else 0.0)
    H_eff = H_DOM + (1.0*h if half_cell else 0.0)
    z_off = 0.5*h if half_cell else 0.0

    LAM = J1Z / R_eff
    KZ = N_MODE*np.pi / H_eff
    K2 = LAM**2 + KZ**2
    TAU = 1.0/(C*np.sqrt(K2))
    T_END = 4.0*TAU

    Nr = int(round(R_DOM/h)) + 1
    Nz = int(round(H_DOM/h)) + 1
    r = np.arange(Nr)*h; z = np.arange(Nz)*h
    RR, ZZ = np.meshgrid(r, z, indexing="ij")
    dom = np.ones((Nr, Nz), bool)

    # masked-flux machinery, identical to the transient solver
    in_rp = np.zeros_like(dom); in_rp[:-1, :] = dom[1:, :] & dom[:-1, :]
    in_rm = np.zeros_like(dom); in_rm[1:, :]  = dom[:-1, :] & dom[1:, :]
    in_zp = np.zeros_like(dom); in_zp[:, :-1] = dom[:, 1:] & dom[:, :-1]
    in_zm = np.zeros_like(dom); in_zm[:, 1:]  = dom[:, :-1] & dom[:, 1:]
    rp_face = np.where(in_rp, (np.arange(Nr)[:, None] + 0.5)*h, 0.0)
    rm_face = np.where(in_rm, (np.arange(Nr)[:, None] - 0.5)*h, 0.0)
    r_col = np.where(r > 0, r, 1.0)[:, None]

    def lap(p):
        a = np.zeros_like(p); a[:-1, :] = p[1:, :] - p[:-1, :]
        b = np.zeros_like(p); b[1:, :]  = p[1:, :] - p[:-1, :]
        rad = (rp_face*a - rm_face*b) / (r_col*h*h)
        rad[0, :] = np.where(in_rp[0, :], 4.0*(p[1, :] - p[0, :])/(h*h), 0.0)
        ap = np.where(in_zp, np.roll(p, -1, axis=1) - p, 0.0)
        am = np.where(in_zm, p - np.roll(p, 1, axis=1), 0.0)
        return (rad + (ap - am)/(h*h)) * dom

    shape = j0(LAM*RR) * np.cos(KZ*(ZZ + z_off))
    T   = lambda t: (t/TAU)**3 * np.exp(-t/TAU)
    Tpp = lambda t: (6*(t/TAU) - 6*(t/TAU)**2 + (t/TAU)**3)*np.exp(-t/TAU)/TAU**2

    dt = CFL*h/(C*np.sqrt(2.0))
    Nt = int(np.ceil(T_END/dt)); dt = T_END/Nt
    p_old = np.zeros((Nr, Nz)); p_cur = np.zeros((Nr, Nz))
    for n in range(Nt):
        tn = n*dt
        p_new = 2.0*p_cur - p_old + (C*dt)**2*lap(p_cur) \
                + dt*dt*shape*(Tpp(tn) + C**2*K2*T(tn))
        p_old, p_cur = p_cur, p_new

    ex = shape*T(T_END)
    w = np.maximum(RR, 0.5*h)
    err = np.sqrt(np.sum(w*(p_cur - ex)**2)/np.sum(w))
    return Nt, err


HS = [2.0e-3, 1.0e-3, 5.0e-4, 2.5e-4]
res = {}
print("=== TRANSIENT MMS: the role of the wall convention ===\n")
for half_cell in (False, True):
    errs = []
    for h in HS:
        t0 = time.time()
        Nt, e = run(h, half_cell)
        errs.append(e)
    errs = np.array(errs)
    orders = np.log(errs[:-1]/errs[1:])/np.log(2.0)
    res["eff" if half_cell else "nom"] = (errs, orders)
    lab = "EFFECTIVE geometry (wall at R + h/2)" if half_cell else \
          "NOMINAL geometry   (wall at R)"
    print(f"{lab}")
    for h, e in zip(HS, errs):
        print(f"    h = {h*1e3:5.2f} mm   L2 error = {e:.4e}")
    print(f"    observed orders : {'  '.join(f'{o:.3f}' for o in orders)}"
          f"   ->  mean {orders.mean():.3f}\n")

en, on_ = res["nom"]; ee, oe = res["eff"]
print("=== CONCLUSION ===")
print(f"  nominal   : order {on_.mean():.2f}  -> the h/2 offset dominates (first-order error)")
print(f"  effective : order {oe.mean():.2f}  -> scheme VERIFIED at order 2")
print(f"  error reduction at h = 2 mm : factor {en[0]/ee[0]:.0f}")

os.makedirs("data", exist_ok=True); os.makedirs("plots", exist_ok=True)
np.savez("data/mms_transient.npz", h=np.array(HS), err_nominal=en,
         err_effective=ee, orders_nominal=on_, orders_effective=oe)

fig, ax = plt.subplots(figsize=(6.8, 4.6))
hh = np.array(HS)*1e3
ax.loglog(hh, en, "s--", color="#a8442a", label=f"wall at R (nominal) - order {on_.mean():.2f}")
ax.loglog(hh, ee, "o-", color="#1d5480", label=f"wall at R + h/2 - order {oe.mean():.2f}")
ax.loglog(hh, ee[0]*(hh/hh[0])**2, "k:", lw=1, label="slope 2")
ax.loglog(hh, en[0]*(hh/hh[0])**1, ":", color="0.6", lw=1, label="slope 1")
ax.set_xlabel("mesh step h (mm)"); ax.set_ylabel("L2 error (Pa)")
ax.set_title("Transient MMS - the wall convention sets the order")
ax.grid(True, which="both", ls=":"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig("plots/mms_transient.png", dpi=110)
print("\nFigure: plots/mms_transient.png | Data: data/mms_transient.npz")
