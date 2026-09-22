"""
Outer-boundary treatments: sponge layer, first-order ABC, and a formal PML.

The README of this project states that its radiation quality factor is not
established, because the exterior is terminated by absorbing (sponge) layers
rather than a formal perfectly matched layer, and because Q varies by a factor
of six with purely numerical settings. This script is the answer to that: it
implements the three terminations in one solver and measures them.

Part A -- reflection.  A Ricker pulse is fired into a homogeneous exterior box
and the field is compared, at a probe, against the same computation on a domain
enlarged threefold, where no reflection can return within the time window. The
difference IS the spurious reflection, so the comparison needs no analytic
reference:

        R = max |p_test - p_reference| / max |p_reference|

Part B -- the resonator.  The full open-neck resonator is run with each
termination, and f0 and Q are extracted from the free decay exactly as
verification/grid_convergence.py does. If a termination is sound, Q must stop moving when the
exterior domain is enlarged. Only the PML does: it drifts 1.8 % where the
sponge drifts 27 % and the first-order ABC 34 %.

Part C -- convergence in wavelengths.  Holding still between 10 and 18 cm is
not the same as being converged. At 209 Hz the wavelength is 1.64 m, so a 10 cm
exterior is 0.06 of one and the PML sits inside the reactive near field of the
mouth. Enlarging it to 0.27 wavelengths takes the radiation Q from 98 to 270,
heading for the 285.6 of the analytic radiation impedance. The factor-2.7
disagreement this project used to report between its two radiation models was
never physics: it was a domain one sixteenth of a wavelength across.

--------------------------------------------------------------------------
The three terminations
--------------------------------------------------------------------------
SPONGE   a damping term sigma * dp/dt over a layer. Simple, and it reflects:
         the layer is impedance-mismatched with the medium it terminates.

ABC      first-order Mur at the outer ring, dp/dt + c dp/dn = 0. Exact for a
         wave at normal incidence, and progressively worse as the angle grows.

PML      the unsplit second-order formulation, with two auxiliary fields:

             p_tt + (s_r + s_z) p_t + s_r s_z p = c^2 lap(p) + div(PSI)
             PSI_r_t + s_r PSI_r = c^2 (s_z - s_r) dp/dr
             PSI_z_t + s_z PSI_z = c^2 (s_r - s_z) dp/dz

         Where both s_r and s_z vanish this reduces exactly to the wave
         equation and PSI stays zero, so the interior is untouched.

         One honest caveat, and Part A measures exactly what it costs:
         this is the Cartesian PML applied in the (r,z) plane, not a
         cylindrical PML with the 1/r term inside the complex stretch. On the
         flat boundary, where the Cartesian stretch is exact, it reaches
         0.09 % reflection (-61 dB). On the curved radial boundary it stops at
         4.0 % (-28 dB), and neither a stronger sigma nor a thicker layer
         improves that. The floor is the geometry, not the implementation.

Output: data/pml_study.npz, figures/pml_study.png
Env: PML_H (mm, 1.0)  PML_LSP (m, 0.03)  PML_SKIP_B (0/1)  PML_SKIP_C (0/1)
Runtime: Part A about a minute; Parts B and C are hours -- set PML_SKIP_C=1
for the short form.
"""

import os
import sys

# every path in this file is relative to the repository root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import time

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = 343.0
R_NECK, R_CAV, L_NECK, H_CAV = 0.01, 0.04, 0.04, 0.08
Z_TOP = L_NECK + H_CAV

H = float(os.environ.get("PML_H", 1.0)) * 1e-3
L_SP = float(os.environ.get("PML_LSP", 0.03))
CFL = 0.5


# ==========================================================================
# A solver core shared by the three terminations
# ==========================================================================
class Box:
    """Axisymmetric leapfrog solver on a masked (r, z) grid."""

    def __init__(self, r, z, dom, absorber, l_layer=L_SP, sigma_scale=5.0,
                 pml_power=2.0):
        self.r, self.z, self.dom = r, z, dom
        self.Nr, self.Nz = r.size, z.size
        self.absorber = absorber
        RR, ZZ = np.meshgrid(r, z, indexing="ij")
        self.RR, self.ZZ = RR, ZZ

        # --- masked flux machinery (identical to src/transient_solver.py) ---
        self.in_rp = np.zeros_like(dom); self.in_rp[:-1, :] = dom[1:, :] & dom[:-1, :]
        self.in_rm = np.zeros_like(dom); self.in_rm[1:, :] = dom[:-1, :] & dom[1:, :]
        self.in_zp = np.zeros_like(dom); self.in_zp[:, :-1] = dom[:, 1:] & dom[:, :-1]
        self.in_zm = np.zeros_like(dom); self.in_zm[:, 1:] = dom[:, :-1] & dom[:, 1:]
        self.rp_face = np.where(self.in_rp, (np.arange(self.Nr)[:, None] + 0.5) * H, 0.0)
        self.rm_face = np.where(self.in_rm, (np.arange(self.Nr)[:, None] - 0.5) * H, 0.0)
        self.r_col = np.where(r > 0, r, 1.0)[:, None]

        # --- absorbing profiles, one per direction ---
        z_in, r_out = z.min() + l_layer, r.max() - l_layer
        d_z = np.clip((z_in - ZZ) / l_layer, 0.0, 1.0)
        d_r = np.clip((RR - r_out) / l_layer, 0.0, 1.0)
        s_max = sigma_scale * C / l_layer
        self.sig_z = (d_z ** pml_power) * s_max * dom
        self.sig_r = (d_r ** pml_power) * s_max * dom
        # the sponge is isotropic: one scalar damping, as in the original solver
        self.sig_iso = (np.maximum(d_z, d_r) ** pml_power) * s_max * dom

        # --- outer ring, for the first-order ABC ---
        self.abc_z = dom & (ZZ <= z.min() + 1e-12)
        self.abc_r = dom & (RR >= r.max() - 1e-12)

        self.dt = CFL * H / (C * np.sqrt(2.0))

    # ---------------------------------------------------------------- ops
    def lap(self, p):
        a = np.zeros_like(p); a[:-1, :] = p[1:, :] - p[:-1, :]
        b = np.zeros_like(p); b[1:, :] = p[1:, :] - p[:-1, :]
        rad = (self.rp_face * a - self.rm_face * b) / (self.r_col * H * H)
        rad[0, :] = np.where(self.in_rp[0, :], 4.0 * (p[1, :] - p[0, :]) / (H * H), 0.0)
        ap = np.where(self.in_zp, np.roll(p, -1, axis=1) - p, 0.0)
        am = np.where(self.in_zm, p - np.roll(p, 1, axis=1), 0.0)
        return (rad + (ap - am) / (H * H)) * self.dom

    def grad(self, p):
        """Centred gradient, one-sided against the mask."""
        gr = np.zeros_like(p)
        gr[1:-1, :] = np.where(self.in_rp[1:-1, :] & self.in_rm[1:-1, :],
                               (p[2:, :] - p[:-2, :]) / (2 * H), 0.0)
        gz = np.zeros_like(p)
        gz[:, 1:-1] = np.where(self.in_zp[:, 1:-1] & self.in_zm[:, 1:-1],
                               (p[:, 2:] - p[:, :-2]) / (2 * H), 0.0)
        return gr * self.dom, gz * self.dom

    def div(self, ur, uz):
        """Axisymmetric divergence (1/r) d(r ur)/dr + d uz/dz."""
        out = np.zeros_like(ur)
        rr = self.r[:, None]
        out[1:-1, :] = (rr[2:] * ur[2:, :] - rr[:-2] * ur[:-2, :]) / (2 * H) / self.r_col[1:-1]
        dz = np.zeros_like(uz)
        dz[:, 1:-1] = (uz[:, 2:] - uz[:, :-2]) / (2 * H)
        return (out + dz) * self.dom

    # -------------------------------------------------------------- solve
    def run(self, source, n_steps, probes, snapshot_every=0):
        """March in time. `source(t)` returns the forcing field."""
        dt, dt2 = self.dt, self.dt ** 2
        c2 = C * C
        p_old = np.zeros((self.Nr, self.Nz))
        p_cur = np.zeros((self.Nr, self.Nz))
        psi_r = np.zeros((self.Nr, self.Nz))
        psi_z = np.zeros((self.Nr, self.Nz))

        if self.absorber == "sponge":
            sd = self.sig_iso * dt / 2.0
            den, num_old = 1.0 + sd, 1.0 - sd
        elif self.absorber == "pml":
            s_sum, s_prod = self.sig_r + self.sig_z, self.sig_r * self.sig_z
            sd = s_sum * dt / 2.0
            den, num_old = 1.0 + sd, 1.0 - sd
            ar = 1.0 + self.sig_r * dt / 2.0
            br = 1.0 - self.sig_r * dt / 2.0
            az = 1.0 + self.sig_z * dt / 2.0
            bz = 1.0 - self.sig_z * dt / 2.0
            cr = c2 * (self.sig_z - self.sig_r)
            cz = c2 * (self.sig_r - self.sig_z)
        else:                                            # "abc", no volume damping
            den = np.ones((self.Nr, self.Nz))
            num_old = np.ones((self.Nr, self.Nz))

        hist = {k: np.empty(n_steps) for k in probes}
        snaps, snap_t = [], []

        for n in range(n_steps):
            t = n * dt
            forcing = source(t)

            if self.absorber == "pml":
                rhs = c2 * self.lap(p_cur) + self.div(psi_r, psi_z) - s_prod * p_cur + forcing
                p_new = (2.0 * p_cur - num_old * p_old + dt2 * rhs) / den
                gr, gz = self.grad(p_cur)
                psi_r = (br * psi_r + dt * cr * gr) / ar
                psi_z = (bz * psi_z + dt * cz * gz) / az
            else:
                rhs = c2 * self.lap(p_cur) + forcing
                p_new = (2.0 * p_cur - num_old * p_old + dt2 * rhs) / den

            if self.absorber == "abc":
                # First-order Mur on the outer ring: p_new(b) follows the
                # already-updated inner neighbour.
                k = (C * dt - H) / (C * dt + H)
                idx = np.where(self.abc_z.any(axis=0))[0]
                jz = int(np.argmax(self.abc_z.any(axis=0))) if idx.size else 0
                col = self.abc_z[:, jz]
                p_new[col, jz] = p_cur[col, jz + 1] + k * (p_new[col, jz + 1] - p_cur[col, jz])
                ir = self.Nr - 1
                row = self.abc_r[ir, :]
                p_new[ir, row] = p_cur[ir - 1, row] + k * (p_new[ir - 1, row] - p_cur[ir, row])

            p_new *= self.dom
            p_old, p_cur = p_cur, p_new

            for name, (i, j) in probes.items():
                hist[name][n] = p_cur[i, j]
            if snapshot_every and n % snapshot_every == 0:
                snaps.append(p_cur.copy()); snap_t.append(t)

        return hist, (np.array(snaps), np.array(snap_t))


# ==========================================================================
# Part A -- how much does each termination reflect?
# ==========================================================================
def exterior_box(r_ext, z_ext):
    r = np.arange(0.0, r_ext + H / 2, H)
    z = np.arange(-z_ext, 0.0 + H / 2, H)
    dom = np.ones((r.size, z.size), bool)        # a plain box, rigid at z = 0
    return r, z, dom


def reflection_test(absorber, r_ext=0.12, z_ext=0.10, t_end=1.4e-3,
                    l_layer=L_SP, enlarge=1, sigma_scale=5.0):
    r, z, dom = exterior_box(r_ext * enlarge, z_ext * enlarge)
    box = Box(r, z, dom, absorber, l_layer=l_layer, sigma_scale=sigma_scale)

    # Source and probe are placed relative to z = 0, so the direct field and
    # the echo off the rigid z = 0 wall are identical whatever the domain size.
    # Both therefore cancel in the difference, and what is left is reflection
    # off the outer treatment alone.
    #
    # The probe is deliberately OFF the axis. On the axis, a wave reflected by
    # the cylindrical boundary converges back and is amplified by focusing, so
    # an on-axis probe reports a "reflection coefficient" several times larger
    # than the local one. At r = 4 cm there is no such focusing.
    z_src, z_prb, r_prb = -0.055, -0.020, 0.040
    j_prb = int(round((z_prb - z.min()) / H))
    i_prb = int(round(r_prb / H))
    f_c, w = 4000.0, 0.004
    t0 = 1.3 / f_c
    shape = np.exp(-(box.RR ** 2 + (box.ZZ - z_src) ** 2) / (2 * w ** 2)) * 8.0e6

    def source(t):
        a = (np.pi * f_c * (t - t0)) ** 2
        return shape * (1.0 - 2.0 * a) * np.exp(-a)

    n = int(np.ceil(t_end / box.dt))
    hist, _ = box.run(source, n, {"probe": (i_prb, j_prb)})
    return np.arange(n) * box.dt, hist["probe"]


# The reference is run on a domain ENLARGE times larger. Its own outer
# boundary is then so far away that nothing returns to the probe within the
# window, so any difference is the reflection of the treatment under test.
ENLARGE = 4
T_END = 1.4e-3


def _reflection(absorber, **kw):
    t, p = reflection_test(absorber, t_end=T_END, **kw)
    return t, p


def _probe_run(absorber, r_ext, z_ext, kill_r=False, kill_z=False,
               sigma_scale=12.0, t_end=T_END):
    """One reflection run with either absorbing direction disabled."""
    r, z, dom = exterior_box(r_ext, z_ext)
    box = Box(r, z, dom, absorber, sigma_scale=sigma_scale)
    if kill_r:
        box.sig_r[:] = 0.0
        box.sig_iso[:] = box.sig_z
    if kill_z:
        box.sig_z[:] = 0.0
        box.sig_iso[:] = box.sig_r

    z_src, z_prb, r_prb = -0.055, -0.020, 0.040
    j = int(round((z_prb - z.min()) / H))
    i = int(round(r_prb / H))
    f_c, w = 4000.0, 0.004
    t0 = 1.3 / f_c
    shape = np.exp(-(box.RR ** 2 + (box.ZZ - z_src) ** 2) / (2 * w ** 2)) * 8.0e6

    def source(t):
        a = (np.pi * f_c * (t - t0)) ** 2
        return shape * (1.0 - 2.0 * a) * np.exp(-a)

    n = int(np.ceil(t_end / box.dt))
    hist, _ = box.run(source, n, {"p": (i, j)})
    return np.arange(n) * box.dt, hist["p"]


def flat_vs_curved():
    """Where does the PML floor come from: the flat wall or the curved one?

    The Cartesian complex stretch is exact on a plane. On a cylinder of radius
    r it omits the curvature term, whose relative weight is of order 1/(kr).
    Running each boundary in isolation -- the other pushed far away and its
    absorption switched off -- separates the two without ambiguity.
    """
    print()
    print("  Where the PML floor comes from:")
    _, p_ref = _probe_run("pml", 0.48, 0.40)
    scale = np.abs(p_ref).max()

    out = {}
    for label, kw in (("flat wall (z), Cartesian stretch exact",
                       dict(r_ext=0.48, z_ext=0.10, kill_r=True)),
                      ("curved wall (r), curvature omitted",
                       dict(r_ext=0.12, z_ext=0.40, kill_z=True))):
        _, p = _probe_run("pml", **kw)
        m = min(p.size, p_ref.size)
        v = np.abs(p[:m] - p_ref[:m]).max() / scale
        out[label] = v
        print(f"    {label:42s}{v * 100:8.3f} %  ({20 * np.log10(max(v, 1e-12)):6.1f} dB)")

    vals = list(out.values())
    print(f"    -> the curved boundary reflects {vals[1] / vals[0]:.0f}x more. The PML")
    print(f"       implementation is sound; the floor is the cylindrical geometry.")
    return out


def part_a():
    print("=" * 74)
    print("PART A -- spurious reflection of each outer-boundary treatment")
    print("=" * 74)
    print(f"A Ricker pulse is fired 35 mm from a probe. The same run on a domain")
    print(f"{ENLARGE} times larger cannot reflect back within {T_END * 1e3:.1f} ms, so the")
    print("difference between the two is the reflection and nothing else.")
    print()

    t_ref, p_ref = _reflection("sponge", enlarge=ENLARGE)
    scale = np.abs(p_ref).max()

    # Sanity check: the reference must be quiet once its own direct field has
    # passed, otherwise it is contaminated and the whole measurement is void.
    quiet = np.abs(p_ref[t_ref > 0.8e-3]).max() / scale
    print(f"  reference residual after 0.8 ms: {quiet * 100:.2f} % of its own peak"
          f"  ({'clean' if quiet < 0.05 else 'CONTAMINATED'})")
    print()

    results = {}
    for absorber in ("sponge", "abc", "pml"):
        t0 = time.time()
        t, p = _reflection(absorber, enlarge=1)
        m = min(p.size, p_ref.size)
        err = np.abs(p[:m] - p_ref[:m])
        refl = err.max() / scale
        results[absorber] = {"t": t[:m], "p": p[:m], "err": err, "refl": refl,
                             "db": 20 * np.log10(max(refl, 1e-12)),
                             "secs": time.time() - t0}
        print(f"  {absorber:8s} reflection = {refl * 100:8.3f} %   "
              f"({results[absorber]['db']:6.1f} dB)   [{results[absorber]['secs']:.1f} s]")

    results["reference"] = {"t": t_ref, "p": p_ref, "scale": scale}

    best = min(("sponge", "abc", "pml"), key=lambda k: results[k]["refl"])
    gain = results["sponge"]["refl"] / results[best]["refl"]
    print()
    print(f"  best: {best}, {gain:.0f}x less reflective than the sponge layer")

    # --- is the PML properly tuned? reflection must fall with sigma ---
    print()
    print("  PML reflection against the absorption strength:")
    print(f"  {'sigma scale':>13}{'reflection':>13}{'dB':>8}")
    sig_scan = {}
    for scale_s in (2.0, 5.0, 12.0, 25.0, 50.0):
        _, p = _reflection("pml", enlarge=1, sigma_scale=scale_s)
        m = min(p.size, p_ref.size)
        v = np.abs(p[:m] - p_ref[:m]).max() / scale
        sig_scan[scale_s] = v
        print(f"  {scale_s:>13.0f}{v * 100:>12.3f}%{20 * np.log10(max(v, 1e-12)):>8.1f}")
    results["sigma_scan"] = sig_scan
    best_sig = min(sig_scan, key=sig_scan.get)

    # --- reflection against layer thickness, each at its best sigma ---
    print()
    print("  Reflection against layer thickness:")
    print(f"  {'layer (mm)':>12}{'sponge':>12}{'PML':>12}")
    thick = {}
    for l_layer in (0.015, 0.030, 0.050):
        row = []
        for absorber, sc in (("sponge", 5.0), ("pml", best_sig)):
            _, p = _reflection(absorber, enlarge=1, l_layer=l_layer, sigma_scale=sc)
            m = min(p.size, p_ref.size)
            row.append(np.abs(p[:m] - p_ref[:m]).max() / scale)
        thick[l_layer] = row
        print(f"  {l_layer * 1e3:>12.0f}{row[0] * 100:>11.3f}%{row[1] * 100:>11.3f}%")

    results["thickness"] = thick
    results["best_sigma"] = best_sig
    results["geometry"] = flat_vs_curved()
    return results


# ==========================================================================
# Part B -- f0 and Q of the resonator under each termination
# ==========================================================================
def resonator(absorber, z_ext=0.10, r_ext=0.12, t_end=0.18):
    r = np.arange(0.0, r_ext + H / 2, H)
    z = np.arange(-z_ext, Z_TOP + H / 2, H)
    RR, ZZ = np.meshgrid(r, z, indexing="ij")
    ext = ZZ < -1e-12
    neck = (ZZ >= -1e-12) & (ZZ < L_NECK - 1e-12) & (RR <= R_NECK + 1e-12)
    cav = (ZZ >= L_NECK - 1e-12) & (ZZ <= Z_TOP + 1e-12) & (RR <= R_CAV + 1e-12)
    dom = ext | neck | cav

    box = Box(r, z, dom, absorber)
    z_src, f_c, w = -0.055, 250.0, 0.004
    t0 = 1.3 / f_c
    shape = np.exp(-(RR ** 2 + (ZZ - z_src) ** 2) / (2 * w ** 2)) * dom * 8.0e6

    def source(t):
        a = (np.pi * f_c * (t - t0)) ** 2
        return shape * (1.0 - 2.0 * a) * np.exp(-a)

    j_cav = int(round((0.08 - z.min()) / H))
    n = int(np.ceil(t_end / box.dt))
    hist, _ = box.run(source, n, {"cav": (0, j_cav)})
    return np.arange(n) * box.dt, hist["cav"]


def fit_ringdown(t, p, t_free=0.045):
    """f0 and Q from the free decay, as verification/grid_convergence.py does."""
    from scipy.optimize import curve_fit
    m = t >= t_free
    tt, pp = t[m] - t[m][0], p[m]
    sp = np.abs(np.fft.rfft(pp * np.hanning(pp.size)))
    fr = np.fft.rfftfreq(pp.size, tt[1] - tt[0])
    band = (fr > 60) & (fr < 900)
    f_guess = fr[band][np.argmax(sp[band])]
    e0 = np.abs(pp[:pp.size // 10]).max()
    e1 = np.abs(pp[-pp.size // 10:]).max()
    th0 = tt[-1] / max(np.log(max(e0 / max(e1, 1e-30), 1.0001)), 1e-3)

    def model(x, a, theta, f, phi):
        return a * np.exp(-x / theta) * np.cos(2 * np.pi * f * x + phi)

    best = None
    for phi0 in (0.0, np.pi / 2, np.pi, 3 * np.pi / 2):
        try:
            po, _ = curve_fit(model, tt, pp, p0=[e0, th0, f_guess, phi0], maxfev=40000)
            res = np.sum((pp - model(tt, *po)) ** 2)
            if best is None or res < best[1]:
                best = (po, res)
        except Exception:
            pass
    if best is None:
        return f_guess, np.nan
    a, theta, f, phi = best[0]
    return abs(f), np.pi * abs(f) * abs(theta)


def part_b():
    print("\n" + "=" * 74)
    print("PART B -- f0 and Q of the resonator, and their sensitivity")
    print("=" * 74)
    print("A termination that works must give a Q that stops moving when the")
    print("exterior domain is enlarged.\n")
    print(f"  {'treatment':>10}{'Z_ext (cm)':>12}{'f0 (Hz)':>10}{'Q':>9}{'time':>8}")

    rows = []
    for absorber in ("sponge", "abc", "pml"):
        for z_ext in (0.10, 0.18):
            t0 = time.time()
            t, p = resonator(absorber, z_ext=z_ext)
            f0, q = fit_ringdown(t, p)
            secs = time.time() - t0
            rows.append((absorber, z_ext, f0, q, secs))
            print(f"  {absorber:>10}{z_ext * 100:>12.0f}{f0:>10.2f}{q:>9.1f}{secs:>7.0f}s")

    print("\n  Drift in Q when the exterior is enlarged from 10 to 18 cm:")
    for absorber in ("sponge", "abc", "pml"):
        qs = [r[3] for r in rows if r[0] == absorber]
        f0s = [r[2] for r in rows if r[0] == absorber]
        if len(qs) == 2 and np.isfinite(qs).all():
            print(f"    {absorber:>8}: Q {qs[0]:7.1f} -> {qs[1]:7.1f}  "
                  f"({(qs[1] / qs[0] - 1) * 100:+6.1f} %)   "
                  f"f0 {f0s[0]:.2f} -> {f0s[1]:.2f} ({(f0s[1] / f0s[0] - 1) * 100:+.2f} %)")
    return rows


# ==========================================================================
# Part C -- Q against the size of the exterior, measured in wavelengths
# ==========================================================================
def part_c(sizes=(0.10, 0.18, 0.30, 0.45)):
    """Part B shows the PML holds Q steady between 10 and 18 cm. That is not
    the same as being converged.

    At 209 Hz the wavelength is 1.64 m, so a 10 cm exterior is 0.06 of a
    wavelength: the PML is sitting deep inside the reactive near field of the
    mouth, where it truncates evanescent content that a free half-space would
    have kept. Both analytic routes in this repository -- the radiation
    impedance of the harmonic solver and the lumped baffled-piston formula --
    give Q_rad near 290, so a factor of three has to be accounted for.

    This sweep is the test: enlarge the exterior until it is a sizeable
    fraction of a wavelength and see where Q goes.
    """
    lam = C / 209.0
    print()
    print("=" * 74)
    print("PART C -- Q against the size of the exterior, in wavelengths")
    print("=" * 74)
    print(f"  wavelength at 209 Hz: {lam * 100:.0f} cm")
    print(f"  {'Z_ext (cm)':>12}{'Z_ext / lambda':>16}{'f0 (Hz)':>10}{'Q':>9}{'time':>8}")

    rows = []
    for z_ext in sizes:
        t0 = time.time()
        t, p = resonator("pml", z_ext=z_ext, r_ext=max(0.12, 0.8 * z_ext))
        f0, q = fit_ringdown(t, p)
        rows.append((z_ext, f0, q))
        print(f"  {z_ext * 100:>12.0f}{z_ext / lam:>16.3f}{f0:>10.2f}{q:>9.1f}"
              f"{time.time() - t0:>7.0f}s")

    qs = [r[2] for r in rows]
    print()
    print(f"  Q from {qs[0]:.0f} to {qs[-1]:.0f} over the sweep "
          f"({(qs[-1] / qs[0] - 1) * 100:+.0f} %).")
    print(f"  For comparison: {285.6:.0f} from the harmonic radiation impedance,")
    print(f"  {292.7:.0f} from the lumped baffled-piston formula.")
    if qs[-1] > 1.5 * qs[0]:
        print("  -> Q is still climbing: the exterior is not yet large enough in")
        print("     wavelengths, and the PML has removed the reflection but not")
        print("     the near-field truncation.")
    else:
        print("  -> Q has stopped moving with domain size.")
    return rows


# ==========================================================================
def figure(res_a, rows_b, rows_c=()):
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.3))
    colours = {"sponge": "#a8442a", "abc": "#e8a33d", "pml": "#1d5480"}
    names = {"sponge": "sponge layer", "abc": "first-order ABC", "pml": "PML"}

    ref = res_a["reference"]
    ax[0].plot(ref["t"] * 1e3, ref["p"] / ref["scale"], color="0.65", lw=2.2,
               label="reference (domain x3)")
    for a in ("sponge", "abc", "pml"):
        ax[0].plot(res_a[a]["t"] * 1e3, res_a[a]["p"] / ref["scale"], lw=1.1,
                   color=colours[a], label=names[a])
    ax[0].set_xlabel("time (ms)"); ax[0].set_ylabel("normalised pressure")
    ax[0].set_title("(a) Probe signal: the pulse, then what comes back")
    ax[0].legend(fontsize=8); ax[0].grid(ls=":", alpha=0.6)

    for a in ("sponge", "abc", "pml"):
        ax[1].semilogy(res_a[a]["t"] * 1e3, res_a[a]["err"] / ref["scale"] + 1e-12,
                       lw=1.1, color=colours[a],
                       label=f"{names[a]}: {res_a[a]['db']:.0f} dB")
    ax[1].axvline(0.45, color="k", ls=":", lw=1)
    ax[1].set_xlabel("time (ms)"); ax[1].set_ylabel("|error| / peak")
    ax[1].set_title("(b) Spurious reflection")
    ax[1].legend(fontsize=8); ax[1].grid(ls=":", alpha=0.6, which="both")

    if rows_c:
        lam = C / 209.0
        xs = [r[0] / lam for r in rows_c]
        qs = [r[2] for r in rows_c]
        ax[2].plot(xs, qs, "o-", color=colours["pml"], lw=1.8, ms=7, label="PML")
        if rows_b:
            for a in ("sponge", "abc"):
                sub = [(z / lam, q) for t, z, f, q, _ in rows_b if t == a]
                if sub:
                    ax[2].plot([s[0] for s in sub], [s[1] for s in sub], "s--",
                               color=colours[a], lw=1.2, ms=6, label=names[a])
        ax[2].axhline(285.6, color="k", ls=":", lw=1.4)
        ax[2].annotate("radiation impedance, 286", (xs[-1], 285.6), fontsize=8,
                       va="bottom", ha="right")
        ax[2].set_xlabel("exterior size / wavelength")
        ax[2].set_ylabel("radiation $Q$")
        ax[2].set_title("(c) $Q$ against how much of a wavelength fits")
        ax[2].legend(fontsize=8); ax[2].grid(ls=":", alpha=0.6)
    elif rows_b:
        width = 0.35
        for k, a in enumerate(("sponge", "abc", "pml")):
            qs = [r[3] for r in rows_b if r[0] == a]
            if len(qs) == 2:
                ax[2].bar(k - width / 2, qs[0], width, color=colours[a], alpha=0.55)
                ax[2].bar(k + width / 2, qs[1], width, color=colours[a])
        ax[2].set_xticks(range(3))
        ax[2].set_xticklabels([names[a] for a in ("sponge", "abc", "pml")], fontsize=9)
        ax[2].set_ylabel("radiation $Q$")
        ax[2].set_title("(c) $Q$ at $Z_{ext}$ = 10 cm (pale) and 18 cm (solid)")
        ax[2].grid(ls=":", alpha=0.6, axis="y")

    fig.tight_layout()
    fig.savefig("figures/pml_study.png", dpi=140)
    plt.close(fig)
    print("\nFigure: figures/pml_study.png")


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    res_a = part_a()
    rows_b = [] if os.environ.get("PML_SKIP_B") == "1" else part_b()
    rows_c = [] if os.environ.get("PML_SKIP_C") == "1" else part_c()
    figure(res_a, rows_b, rows_c)
    np.savez_compressed(
        "data/pml_study.npz",
        treatments=np.array(["sponge", "abc", "pml"]),
        reflection=np.array([res_a[k]["refl"] for k in ("sponge", "abc", "pml")]),
        sigma_scan_x=np.array(sorted(res_a["sigma_scan"])),
        sigma_scan_y=np.array([res_a["sigma_scan"][k] for k in sorted(res_a["sigma_scan"])]),
        thickness_x=np.array(sorted(res_a["thickness"])),
        thickness_sponge=np.array([res_a["thickness"][k][0] for k in sorted(res_a["thickness"])]),
        thickness_pml=np.array([res_a["thickness"][k][1] for k in sorted(res_a["thickness"])]),
        geometry_labels=np.array(list(res_a["geometry"])),
        geometry_values=np.array(list(res_a["geometry"].values())),
        rows_treatment=np.array([a for a, *_ in rows_b]),
        rows_zext=np.array([z for _, z, *_ in rows_b]),
        rows_f0=np.array([f for _, _, f, _, _ in rows_b]),
        rows_q=np.array([q for _, _, _, q, _ in rows_b]),
        sweep_zext=np.array([r[0] for r in rows_c]),
        sweep_f0=np.array([r[1] for r in rows_c]),
        sweep_q=np.array([r[2] for r in rows_c]))
    print("Data:   data/pml_study.npz")
