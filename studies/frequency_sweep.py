"""
Full frequency sweep of the open-neck Helmholtz resonator.

Solves the Helmholtz equation in the harmonic regime at EVERY frequency from
1 to 1000 Hz, with a radiation-impedance condition at the mouth (baffled
piston), and plots the amplitude response. The peak is the resonance.

The matrix CHANGES at every frequency: k^2 = (2 pi f / c)^2 sits on the
diagonal, and the impedance coefficient alpha = i w rho / Z_r also depends on
omega, so the mouth rows change too. There is no shortcut -- each frequency is
a complete linear solve (~0.1 s), which is precisely why the harmonic route is
worth taking: 1000 solves here cost 77 s, against the 7 to 17 hours an
equivalent transient sweep would need.

Output: data/fdm_sweep.npz, figures/fdm_sweep.png
Env: SW_FMIN (1) SW_FMAX (1000) SW_DF (1.0) SW_H (mm, 1.0) SW_TAG ("")
"""

import os
import sys

# every path in this file is relative to the repository root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import time
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.harmonic_solver import HarmonicSolver


FMIN = float(os.environ.get("SW_FMIN", 1.0))
FMAX = float(os.environ.get("SW_FMAX", 1000.0))
DF   = float(os.environ.get("SW_DF", 1.0))
H    = float(os.environ.get("SW_H", 1.0))*1e-3
TAG  = os.environ.get("SW_TAG", "")

SOLVER = HarmonicSolver(h=H)
FLUID = SOLVER.fluid
r, z = SOLVER.r, SOLVER.z
J_CAV = SOLVER.cavity_probe_index(0.08)     # probe: cavity bottom, on the axis
solve = SOLVER.solve

print(f"grid {SOLVER.nr}x{SOLVER.nz} = {SOLVER.n} nodes, {FLUID.sum()} fluid "
      f"| h = {H*1e3:.2f} mm")
print(f"sweep {FMIN:.0f} -> {FMAX:.0f} Hz in steps of {DF:g} Hz "
      f"= {int(round((FMAX-FMIN)/DF))+1} solves")


freqs = np.arange(FMIN, FMAX + 0.5*DF, DF)
amp_cav = np.empty(freqs.size)          # |P| at the cavity bottom
amp_max = np.empty(freqs.size)          # peak |P| over the domain
phase   = np.empty(freqs.size)
t0 = time.time()
for n, f in enumerate(freqs):
    P = solve(f)
    amp_cav[n] = abs(P[0, J_CAV])
    amp_max[n] = np.abs(np.where(FLUID, P, 0)).max()
    phase[n] = np.angle(P[0, J_CAV])
    if (n+1) % 100 == 0 or n == freqs.size-1:
        print(f"  {n+1:4d}/{freqs.size}  f = {f:7.1f} Hz  |P|cav = {amp_cav[n]:10.2f} "
              f"| {time.time()-t0:5.0f}s")

# ---- resonance peak: parabola through the three points at the top ----
kpk = int(np.argmax(amp_cav))
if 0 < kpk < freqs.size-1:
    y0, y1, y2 = amp_cav[kpk-1], amp_cav[kpk], amp_cav[kpk+1]
    d = 0.5*(y0 - y2)/(y0 - 2*y1 + y2)
    f_pk = freqs[kpk] + d*DF
else:
    f_pk = freqs[kpk]
# -3 dB width -> quality factor
half = amp_cav[kpk]/np.sqrt(2.0)
lo = np.where(amp_cav[:kpk] < half)[0]
hi = np.where(amp_cav[kpk:] < half)[0]
if lo.size and hi.size:
    f_lo = np.interp(half, amp_cav[lo[-1]:kpk+1], freqs[lo[-1]:kpk+1])
    seg = slice(kpk, kpk+hi[0]+1)
    f_hi = np.interp(half, amp_cav[seg][::-1], freqs[seg][::-1])
    Q = f_pk/(f_hi - f_lo)
else:
    f_lo = f_hi = Q = np.nan

print(f"\n=== RESONANCE ===")
print(f"peak (3-point parabola)  : {f_pk:.2f} Hz")
print(f"amplitude at the peak    : {amp_cav[kpk]:.1f} Pa")
print(f"-3 dB band               : {f_lo:.2f} - {f_hi:.2f} Hz")
print(f"quality factor Q         : {Q:.1f}")
print(f"gain at resonance        : {amp_cav[kpk]/amp_cav[0]:.1f}x the amplitude at {FMIN:.0f} Hz")

np.savez_compressed(f"data/fdm_sweep{TAG}.npz", freqs=freqs, amp_cav=amp_cav,
                    amp_max=amp_max, phase=phase, f_pk=f_pk, Q=Q,
                    f_lo=f_lo, f_hi=f_hi, h=H, r=r, z=z, fluid=FLUID)

fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                       gridspec_kw={"height_ratios": [2, 1]})
ax[0].semilogy(freqs, amp_cav, "C0", lw=1.4, label="cavity bottom")
ax[0].semilogy(freqs, amp_max, "C1", lw=0.9, alpha=.6, label="domain maximum")
ax[0].axvline(f_pk, color="C3", ls="--", lw=1.1)
ax[0].annotate(f"f₀ = {f_pk:.1f} Hz\nQ = {Q:.0f}", (f_pk, amp_cav[kpk]),
               xytext=(f_pk+70, amp_cav[kpk]*0.7), color="C3", fontsize=10,
               arrowprops=dict(arrowstyle="->", color="C3", lw=1))
ax[0].set(ylabel="|P| (Pa)", title=f"Frequency sweep - open-neck Helmholtz resonator "
                                  f"(h = {H*1e3:.1f} mm)")
ax[0].legend(); ax[0].grid(alpha=.3, which="both")
ax[1].plot(freqs, np.degrees(np.unwrap(phase)), "C2", lw=1.2)
ax[1].axvline(f_pk, color="C3", ls="--", lw=1.1)
ax[1].set(xlabel="frequency (Hz)", ylabel="phase (deg)")
ax[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig(f"figures/fdm_sweep{TAG}.png", dpi=120)
print(f"Figure: figures/fdm_sweep{TAG}.png | Data: data/fdm_sweep{TAG}.npz")
