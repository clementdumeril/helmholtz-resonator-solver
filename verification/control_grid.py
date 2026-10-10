"""
Fourth-grid control of the zero-mesh extrapolations, for both solvers.

grid_convergence.py and the harmonic sweeps extrapolate f0 from three grids
(2, 1 and 0.5 mm). Three grids give exactly one observed order, so they cannot
show whether the solution is in the asymptotic range. A fourth grid at 0.25 mm
gives two overlapping triplets, (2, 1, 0.5) and (1, 0.5, 0.25): if the observed
order and the extrapolated value are stable from one triplet to the next, the
extrapolation can be trusted at that level.

Inputs (produced beforehand):
  time domain       data/open_resonator_h2.npz, data/open_resonator.npz,
                    data/open_resonator_h05.npz, data/open_resonator_h025.npz
                    (src/transient_solver.py with OR_TMS=250 and OR_H = 2, 1,
                    0.5, 0.25; the 0.25 mm run takes about 8 h on one core)
  frequency domain  data/fdm_sweep_h{2.0,1.0,0.5,0.25}.npz
                    (studies/frequency_sweep.py in 0.05 Hz steps, over
                    195-220 Hz for 2, 1 and 0.5 mm and over 203.5-207 Hz for
                    0.25 mm: 71 solves of about 4 s each)

Output: data/control_grid.npz (small, versioned). Missing inputs are skipped.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import numpy as np

from verification.grid_convergence import fit_f0

FS = 1.25          # safety factor of the GCI for three-grid studies (Roache)


def triplet(f_coarse, f_mid, f_fine, r=2.0):
    """Observed order, Richardson extrapolation and fine-grid GCI."""
    p = np.log(abs((f_coarse - f_mid) / (f_mid - f_fine))) / np.log(r)
    f_ext = f_fine + (f_fine - f_mid) / (r ** p - 1)
    gci = FS * abs((f_mid - f_fine) / f_fine) / (r ** p - 1)
    return p, f_ext, gci


def report(name, hs, fs):
    print(f"\n{name}")
    for h, f in zip(hs, fs):
        print(f"  h = {h*1e3:5.2f} mm   f0 = {f:9.3f} Hz")
    out = []
    for k in range(len(fs) - 2):
        p, fe, g = triplet(*fs[k:k + 3])
        out.append((p, fe, g))
        print(f"  triplet {hs[k]*1e3:.2f}/{hs[k+1]*1e3:.2f}/{hs[k+2]*1e3:.2f} mm: "
              f"p = {p:.3f}, f_ext = {fe:.2f} Hz, GCI = {g*100:.2f} %")
    if len(out) == 2:
        d = abs(out[1][1] - out[0][1]) / out[0][1]
        print(f"  change of the extrapolated value between triplets: {d*100:.2f} %")
    return np.array(out) if out else np.zeros((0, 3))


def main():
    # ---- time domain: damped-sinusoid fit of the free decay ----
    t_cases = [(2.0e-3, "data/open_resonator_h2.npz"),
               (1.0e-3, "data/open_resonator.npz"),
               (0.5e-3, "data/open_resonator_h05.npz"),
               (0.25e-3, "data/open_resonator_h025.npz")]
    th, tf, tq = [], [], []
    for h, path in t_cases:
        if not os.path.exists(path):
            print(f"time domain, h = {h*1e3:.2f} mm: {path} missing, skipped")
            continue
        d = np.load(path)
        f0, q, _ = fit_f0(d["t"], d["p_cav"])
        th.append(h); tf.append(f0); tq.append(q)

    # ---- frequency domain: three-point parabola on the sweep peak ----
    f_cases = [(2.0e-3, "data/fdm_sweep_h2.0.npz"), (1.0e-3, "data/fdm_sweep_h1.0.npz"),
               (0.5e-3, "data/fdm_sweep_h0.5.npz"), (0.25e-3, "data/fdm_sweep_h0.25.npz")]
    fh, ff, fq = [], [], []
    for h, path in f_cases:
        if not os.path.exists(path):
            print(f"frequency domain, h = {h*1e3:.2f} mm: {path} missing, skipped")
            continue
        d = np.load(path)
        fh.append(h); ff.append(float(d["f_pk"])); fq.append(float(d["Q"]))

    t_tri = report("TIME DOMAIN (meshed exterior, sponge layers)", th, tf)
    f_tri = report("FREQUENCY DOMAIN (radiation impedance)", fh, ff)

    if len(t_tri) and len(f_tri):
        for k in range(min(len(t_tri), len(f_tri))):
            gap = abs(t_tri[k][1] - f_tri[k][1]) / t_tri[k][1]
            print(f"\nsolver gap, triplet {k+1}: {t_tri[k][1]:.2f} vs {f_tri[k][1]:.2f} Hz"
                  f" -> {gap*100:.2f} %")

    np.savez("data/control_grid.npz",
             t_h=np.array(th), t_f0=np.array(tf), t_Q=np.array(tq), t_triplets=t_tri,
             f_h=np.array(fh), f_f0=np.array(ff), f_Q=np.array(fq), f_triplets=f_tri)
    print("\nData: data/control_grid.npz  (triplet rows: p, f_ext, GCI)")


if __name__ == "__main__":
    main()
