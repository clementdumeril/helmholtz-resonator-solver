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

Output: data/fdm_sweep.npz, plots/fdm_sweep.png
Env: SW_FMIN (1) SW_FMAX (1000) SW_DF (1.0) SW_H (mm, 1.0) SW_TAG ("")
"""
import os, time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)

C, RHO = 343.0, 1.204
R_NECK, R_CAV, L_NECK, H_CAV = 0.01, 0.04, 0.04, 0.08
Z_TOP = L_NECK + H_CAV
SRC_Z, SRC_W, SRC_A = L_NECK + 0.5*H_CAV, 0.01, 1.0e4

FMIN = float(os.environ.get("SW_FMIN", 1.0))
FMAX = float(os.environ.get("SW_FMAX", 1000.0))
DF   = float(os.environ.get("SW_DF", 1.0))
H    = float(os.environ.get("SW_H", 1.0))*1e-3
TAG  = os.environ.get("SW_TAG", "")

# ---- geometry and mask: constant, computed once ----
Nr = int(round(R_CAV/H)) + 1
Nz = int(round(Z_TOP/H)) + 1
r = np.arange(Nr)*H; z = np.arange(Nz)*H
RR, ZZ = np.meshgrid(r, z, indexing="ij")
FLUID = ((ZZ < L_NECK-1e-12) & (RR <= R_NECK+1e-12)) | ((ZZ >= L_NECK-1e-12) & (RR <= R_CAV+1e-12))
MOUTH = FLUID & (np.abs(ZZ) < 1e-12) & (RR <= R_NECK+1e-12)
FORC = SRC_A*np.exp(-(RR**2 + (ZZ-SRC_Z)**2)/(2*SRC_W**2))
N = Nr*Nz
J_CAV = int(round(0.08/H))          # probe: cavity bottom, on the axis

print(f"grid {Nr}x{Nz} = {N} nodes, {FLUID.sum()} fluid | h = {H*1e3:.2f} mm")
print(f"sweep {FMIN:.0f} -> {FMAX:.0f} Hz in steps of {DF:g} Hz "
      f"= {int(round((FMAX-FMIN)/DF))+1} solves")


def solve(freq):
    """One complete solve at the given frequency -> complex field P."""
    w = 2*np.pi*freq; k2 = (w/C)**2
    ka = w/C*R_NECK
    Zr = RHO*C*(0.5*ka**2 + 1j*(8/(3*np.pi))*ka)     # baffled piston
    alpha = 1j*w*RHO/Zr
    idx = lambda i, j: i + j*Nr
    ih2 = 1.0/H**2
    rows, cols, dat = [], [], []
    b = np.zeros(N, complex)
    for i in range(Nr):
        for j in range(Nz):
            ci = idx(i, j)
            if not FLUID[i, j]:
                rows += [ci]; cols += [ci]; dat += [1.0]; continue
            diag = k2 + 0j
            if i == 0:                                   # limiting form on the axis
                diag += -4*ih2; rows += [ci]; cols += [idx(1, j)]; dat += [4*ih2]
            else:
                lc = ih2*(1-0.5/i); rc = ih2*(1+0.5/i); diag += -2*ih2
                if FLUID[i-1, j]: rows += [ci]; cols += [idx(i-1, j)]; dat += [lc]
                else:             diag += lc             # ghost node folded in
                if i+1 < Nr and FLUID[i+1, j]: rows += [ci]; cols += [idx(i+1, j)]; dat += [rc]
                else:                          diag += rc
            if MOUTH[i, j]:                              # Robin condition
                diag += -2*ih2 - 2*alpha/H
                rows += [ci]; cols += [idx(i, j+1)]; dat += [2*ih2]
            else:
                diag += -2*ih2
                for jj in (j-1, j+1):
                    if 0 <= jj < Nz and FLUID[i, jj]:
                        rows += [ci]; cols += [idx(i, jj)]; dat += [ih2]
                    else:
                        diag += ih2
            rows += [ci]; cols += [ci]; dat += [diag]
            b[ci] = -FORC[i, j]
    A = sp.coo_matrix((dat, (rows, cols)), shape=(N, N), dtype=complex).tocsr()
    return spsolve(A, b).reshape((Nz, Nr)).T


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
fig.tight_layout(); fig.savefig(f"plots/fdm_sweep{TAG}.png", dpi=120)
print(f"Figure: plots/fdm_sweep{TAG}.png | Data: data/fdm_sweep{TAG}.npz")
