"""
OPEN-NECK Helmholtz resonator, with a meshed exterior domain.
-> exhibits RESONANCE in the transient regime, which the all-rigid-wall model
   cannot do: a closed box only produces a ramp.

Configuration (axisymmetric r-z):
  * cavity  : z in [L_neck, L_neck+H_cav], r <= R_cav   (rigid walls)
  * neck    : z in [0, L_neck],            r <= R_neck  (rigid walls)
  * MOUTH   : z = 0, r <= R_neck  -> OPEN to the exterior
  * baffle  : z = 0, r >  R_neck  -> rigid wall (baffled half-space)
  * exterior: z in [-Z_ext, 0],   r <= R_ext, terminated by absorbing
              (sponge) layers on z = -Z_ext and r = R_ext.

Excitation: a BROADBAND Ricker pulse emitted by a small source in the exterior.
The wave propagates, reaches the mouth, and the resonator *selects* its own
natural frequency -> it rings at f0 once the pulse has passed. That is the
proof of resonance, and it does not depend on the content of the excitation.

Verification: FFT of the cavity signal in free decay -> f0, compared with the
Helmholtz formula and with the harmonic study of this project (~235 Hz).

Outputs: data/open_resonator.npz, plots/open_resonator_diag.png
Env: OR_TMS (duration in ms, default 180), OR_H (step in mm, default 1.0)
"""
import os, time
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)

# --- geometry and physics ---
R_NECK, R_CAV, L_NECK, H_CAV = 0.01, 0.04, 0.04, 0.08
Z_TOP = L_NECK + H_CAV          # 0.12 m (cavity bottom)
C = 343.0

H       = float(os.environ.get("OR_H", 1.0)) * 1e-3     # spatial step
R_EXT   = float(os.environ.get("OR_REXT", 0.12))        # radius of the exterior domain
Z_EXT   = float(os.environ.get("OR_ZEXT", 0.10))        # exterior depth
L_SP    = float(os.environ.get("OR_LSP", 0.03))         # thickness of the absorbing layers
T_MAX   = float(os.environ.get("OR_TMS", 180.0)) * 1e-3

# --- grid ---
r = np.arange(0.0, R_EXT + H/2, H)
z = np.arange(-Z_EXT, Z_TOP + H/2, H)
Nr, Nz = r.size, z.size
RR, ZZ = np.meshgrid(r, z, indexing="ij")

ext  = (ZZ < -1e-12)
neck = (ZZ >= -1e-12) & (ZZ < L_NECK - 1e-12) & (RR <= R_NECK + 1e-12)
cav  = (ZZ >= L_NECK - 1e-12) & (ZZ <= Z_TOP + 1e-12) & (RR <= R_CAV + 1e-12)
dom  = ext | neck | cav
print(f"Grid {Nr}x{Nz} (h={H*1e3:.1f} mm) | active cells {dom.sum()} "
      f"| exterior {ext.sum()}, neck {neck.sum()}, cavity {cav.sum()}")

# --- in-domain neighbours (zero flux across the mask = Neumann) ---
in_rp = np.zeros_like(dom); in_rp[:-1, :] = dom[1:, :] & dom[:-1, :]
in_rm = np.zeros_like(dom); in_rm[1:, :]  = dom[:-1, :] & dom[1:, :]
in_zp = np.zeros_like(dom); in_zp[:, :-1] = dom[:, 1:] & dom[:, :-1]
in_zm = np.zeros_like(dom); in_zm[:, 1:]  = dom[:, :-1] & dom[:, 1:]
rp_face = np.where(in_rp, (np.arange(Nr)[:, None] + 0.5) * H, 0.0)
rm_face = np.where(in_rm, (np.arange(Nr)[:, None] - 0.5) * H, 0.0)
r_col = np.where(r > 0, r, 1.0)[:, None]

def laplacian(p):
    pr_p = np.zeros_like(p); pr_p[:-1, :] = p[1:, :] - p[:-1, :]
    pr_m = np.zeros_like(p); pr_m[1:, :]  = p[1:, :] - p[:-1, :]
    rad = (rp_face*pr_p - rm_face*pr_m) / (r_col * H * H)
    rad[0, :] = np.where(in_rp[0, :], 4.0*(p[1, :] - p[0, :])/(H*H), 0.0)   # axis r=0
    az_p = np.where(in_zp, np.roll(p, -1, axis=1) - p, 0.0)
    az_m = np.where(in_zm, p - np.roll(p, 1, axis=1), 0.0)
    return (rad + (az_p - az_m)/(H*H)) * dom

# --- absorbing (sponge) layers on the outer edges ---
d_z = np.clip((-Z_EXT + L_SP - ZZ)/L_SP, 0.0, 1.0)
d_r = np.clip((RR - (R_EXT - L_SP))/L_SP, 0.0, 1.0)
SIG_MAX = float(os.environ.get("OR_SIG", 5.0)) * C / L_SP
sigma = (np.maximum(d_z, d_r)**2) * SIG_MAX * dom

# --- source: broadband Ricker pulse, in the exterior ---
F_C   = 250.0                      # centre frequency of the wavelet
Z_SRC = -0.055
W_SRC = 0.004
T0    = 1.3/F_C
src_spatial = np.exp(-(RR**2 + (ZZ - Z_SRC)**2)/(2*W_SRC**2)) * dom * 8.0e6
def ricker(t):
    a = (np.pi*F_C*(t - T0))**2
    return (1.0 - 2.0*a)*np.exp(-a)

# --- time step ---
dt = 0.5 * H / (C*np.sqrt(2.0))
Nt = int(np.ceil(T_MAX/dt)); dt = T_MAX/Nt
print(f"dt = {dt*1e6:.3f} us | Nt = {Nt} | T = {T_MAX*1e3:.0f} ms")

# --- probes ---
j_cav  = int(round((0.08 - (-Z_EXT))/H))       # z = +0.08 (cavity)
j_neck = int(round((0.02 - (-Z_EXT))/H))       # z = +0.02 (neck)
j_out  = int(round((-0.02 - (-Z_EXT))/H))      # z = -0.02 (exterior, in front of the mouth)

p_old = np.zeros((Nr, Nz)); p_cur = np.zeros((Nr, Nz))
c2dt2, dt2 = (C*dt)**2, dt*dt
sd2 = sigma*dt/2.0
den = 1.0 + sd2; num_old = 1.0 - sd2

NFRAMES = int(os.environ.get("OR_NFRAMES", 260))
frame_every = max(1, Nt//NFRAMES)
frames, frame_t = [], []
pc, pn, po, thist = [], [], [], []

t0 = time.time()
for n in range(Nt):
    tn = n*dt
    F = src_spatial * ricker(tn)
    p_new = (2.0*p_cur - num_old*p_old + dt2*(C*C*laplacian(p_cur)/1.0 + F)) / den
    # NB: c^2 * lap written as C*C*lap to stay readable
    p_old, p_cur = p_cur, p_new*dom
    pc.append(p_cur[0, j_cav]); pn.append(p_cur[0, j_neck]); po.append(p_cur[0, j_out])
    thist.append(tn + dt)
    if n % frame_every == 0:
        frames.append(p_cur.copy()); frame_t.append(tn + dt)
print(f"FDM finished in {time.time()-t0:.1f} s ({len(frames)} frames)")

thist = np.array(thist); pc = np.array(pc); pn = np.array(pn); po = np.array(po)

# --- analysis: FFT of the free decay, after the pulse has passed ---
t_free = 0.045
m = thist >= t_free
seg = pc[m] - pc[m].mean()
win = np.hanning(seg.size)
spec = np.abs(np.fft.rfft(seg*win))
freqs = np.fft.rfftfreq(seg.size, dt)
band = (freqs > 60) & (freqs < 900)
f0_meas = freqs[band][np.argmax(spec[band])]

# Helmholtz formula, with and without end corrections
S = np.pi*R_NECK**2; V = np.pi*R_CAV**2*H_CAV
f_helm = lambda Leff: C/(2*np.pi)*np.sqrt(S/(V*Leff))
f0_noc = f_helm(L_NECK)
f0_cor = f_helm(L_NECK + 0.85*R_NECK + 0.66*R_NECK)

# quality factor from the envelope decay (Hilbert-like, via a running maximum)
def envelope(x, t, win_ms=6.0):
    k = max(3, int(win_ms*1e-3/dt))
    n = len(x)//k
    tt = np.array([t[i*k:(i+1)*k].mean() for i in range(n)])
    ee = np.array([np.abs(x[i*k:(i+1)*k]).max() for i in range(n)])
    return tt, ee
te, ee = envelope(pc, thist)
sel = (te > 0.06) & (ee > 0)
Q_meas = np.nan
if sel.sum() > 4:
    sl = np.polyfit(te[sel], np.log(ee[sel]), 1)[0]      # ln E = -pi f0 t / Q
    if sl < 0: Q_meas = -np.pi*f0_meas/sl

print("\n=== RESONANCE ===")
print(f"f0 measured (FFT of the free decay)     : {f0_meas:6.1f} Hz")
print(f"Helmholtz without end correction        : {f0_noc:6.1f} Hz")
print(f"Helmholtz with corrections (0.85+0.66)a : {f0_cor:6.1f} Hz")
print(f"harmonic study of this project (L=4 cm) :  235   Hz")
print(f"Q (radiation only, fitted)              : {Q_meas:6.1f}")
print(f"cavity/exterior amplification (peak)    : {np.abs(pc).max()/max(np.abs(po).max(),1e-12):.2f}")

TAG = os.environ.get("OR_TAG", "")
os.makedirs("data", exist_ok=True); os.makedirs("plots", exist_ok=True)
np.savez_compressed(f"data/open_resonator{TAG}.npz",
    r=r, z=z, dom=dom, t=thist, p_cav=pc, p_neck=pn, p_out=po,
    frames=np.array(frames, dtype=np.float32), frame_t=np.array(frame_t),
    freqs=freqs, spec=spec, f0_meas=f0_meas, f0_noc=f0_noc, f0_cor=f0_cor,
    Q_meas=Q_meas, H=H, dt=dt, T_MAX=T_MAX, F_C=F_C)

fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
ax[0].plot(thist*1e3, po, color="0.6", lw=0.7, label="exterior (in front of the mouth)")
ax[0].plot(thist*1e3, pc, "C0", lw=0.8, label="cavity")
ax[0].set(xlabel="t (ms)", ylabel="p (Pa)", title="Broadband pulse -> the cavity RINGS")
ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
ax[1].plot(freqs[band], spec[band]/spec[band].max(), "C0")
ax[1].axvline(f0_meas, color="C3", ls="--", label=f"f0 = {f0_meas:.0f} Hz")
ax[1].axvline(235, color="C2", ls=":", label="harmonic study, 235 Hz")
ax[1].set(xlabel="f (Hz)", ylabel="normalised |P|", title="Spectrum of the free decay (cavity)")
ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
ax[2].semilogy(te*1e3, ee, "o-", ms=3)
ax[2].set(xlabel="t (ms)", ylabel="envelope |p| (Pa)",
          title=f"Decay -> Q ~ {Q_meas:.0f}"); ax[2].grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"plots/open_resonator_diag{TAG}.png", dpi=110)
print(f"\nFigure: plots/open_resonator_diag{TAG}.png | Data: data/open_resonator{TAG}.npz")
