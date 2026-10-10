"""
Figures and numbers of the report (paper/main.tex).

Everything here is read from versioned results in data/; nothing is recomputed
except elementary arithmetic (ratios, percentages). The script writes
  paper/figures/*.pdf   vector figures
  paper/numbers.tex     LaTeX macros for the numbers quoted in the text,
so a number in the report can be traced to the line below that produces it.

Run from anywhere:  python paper/make_figures.py   (a few seconds)
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT = os.path.join("paper", "figures")
os.makedirs(OUT, exist_ok=True)

BLUE, RED, AMBER, GREY = "#1d5480", "#a8442a", "#d99a2b", "0.55"
TEXTW = 6.3                                   # inches, A4 with 2.5 cm margins
plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 8.5, "axes.labelsize": 8.5,
    "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "0.88", "grid.linewidth": 0.6,
    "lines.linewidth": 1.3, "savefig.bbox": "tight", "pdf.fonttype": 42,
})

C = 343.0
R_NECK, R_CAV, L_NECK, H_CAV = 0.01, 0.04, 0.04, 0.08
NUM = {}                                       # macro name -> formatted value


def num(name, value, fmt):
    NUM[name] = format(value, fmt)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)
    print("figure", os.path.join(OUT, name))


def richardson(fc, fm, ff, r=2.0, fs=1.25):
    p = np.log(abs((fc - fm) / (fm - ff))) / np.log(r)
    fe = ff + (ff - fm) / (r ** p - 1)
    return p, fe, fs * abs((fm - ff) / ff) / (r ** p - 1)


# ===========================================================================
# Figure 1: the two computational domains
# ===========================================================================
d_or = np.load("data/open_resonator.npz")
fig, ax = plt.subplots(1, 2, figsize=(TEXTW, 2.55),
                       gridspec_kw={"width_ratios": [1.25, 1.0]})
r, z, dom = d_or["r"] * 1e3, d_or["z"] * 1e3, d_or["dom"]
a = ax[0]
a.pcolormesh(z, r, np.where(dom, 1.0, np.nan), cmap="Blues", vmin=0, vmax=4,
             shading="nearest", rasterized=True)
L_SP = 30.0
a.add_patch(Rectangle((z[0], 0), L_SP, r[-1], color=AMBER, alpha=0.35, lw=0))
a.add_patch(Rectangle((z[0], r[-1] - L_SP), -z[0], L_SP, color=AMBER, alpha=0.35, lw=0))
a.plot([0, 0], [10, r[-1]], color="k", lw=2.0)
a.plot(-55, 0, marker="*", color=RED, ms=9, clip_on=False)
for zp, lab in ((80, "cavity probe"), (20, "neck probe"), (-20, "exterior probe")):
    a.plot(zp, 0, "o", color="k", ms=3, clip_on=False)
a.annotate("source", (-55, 4), color=RED, ha="center", fontsize=7.5)
a.annotate("absorbing layer", (-97, 104), fontsize=7.5, color="0.25")
a.annotate("rigid baffle", (3, 70), fontsize=7.5, color="0.25")
a.annotate("neck", (20, 13), fontsize=7.5, ha="center", color="0.25")
a.annotate("cavity", (80, 44), fontsize=7.5, ha="center", color="0.25")
a.annotate("exterior (meshed)", (-50, 70), fontsize=7.5, ha="center", color="0.25")
a.set(xlabel="z (mm)", ylabel="r (mm)", title="(a) time-domain solver", aspect="equal")
a.grid(False)

md = np.load("data/helmholtz_mode.npz")
b = ax[1]
Pm = np.abs(md["P"]) / np.nanmax(np.abs(md["P"]))
Pm = np.where(md["fluid"], Pm, np.nan)
im = b.pcolormesh(md["z"] * 1e3, md["r"] * 1e3, Pm, cmap="Blues", vmin=0, vmax=1.05,
                  shading="nearest", rasterized=True)
b.plot([0, 0], [0, 10], color=RED, lw=2.4)
b.annotate("mouth: $Z_r$", (3, 15), color=RED, fontsize=7.5)
b.annotate("source $F$", (80, 3), fontsize=7.5, ha="center", color="w")
b.set(xlabel="z (mm)", ylabel="r (mm)", aspect="equal", xlim=(-2, 122), ylim=(0, 42),
      title="(b) frequency-domain solver,\n$|p|$ at " + f"{float(md['freq']):.2f} Hz")
b.grid(False)
cb = fig.colorbar(im, ax=b, orientation="horizontal", fraction=0.06, pad=0.32, aspect=30)
cb.set_label(r"$|p| \,/\, \max |p|$", fontsize=7)
cb.ax.tick_params(labelsize=6.5)
fig.tight_layout()
save(fig, "fig_domains.pdf")

# ===========================================================================
# Figure 2: code verification by manufactured solutions
# ===========================================================================
mh = np.load("data/mms_results.npz")
mt = np.load("data/mms_transient.npz")
p_h = np.polyfit(np.log(mh["h"]), np.log(mh["err"]), 1)[0]
num("mmsHarmonicOrder", p_h, ".2f")
num("mmsTransientOrder", mt["orders_effective"].mean(), ".2f")
num("mmsNominalOrder", mt["orders_nominal"].mean(), ".2f")
num("mmsErrorRatio", mt["err_nominal"][0] / mt["err_effective"][0], ".0f")

fig, ax = plt.subplots(1, 2, figsize=(TEXTW, 2.35))
h = mh["h"] * 1e3
ax[0].loglog(h, mh["err"], "o-", color=BLUE, label=f"observed, order {p_h:.2f}")
ax[0].loglog(h, mh["err"][0] * (h / h[0]) ** 2, ":", color="k", lw=1, label="slope 2")
ax[0].set(xlabel="h (mm)", ylabel="RMS error", title="(a) frequency domain, $k$ at 235 Hz")
ax[0].legend()
h = mt["h"] * 1e3
ax[1].loglog(h, mt["err_nominal"], "s--", color=RED,
             label=f"solution matched to wall at $R$: order {mt['orders_nominal'].mean():.2f}")
ax[1].loglog(h, mt["err_effective"], "o-", color=BLUE,
             label=f"matched to $R+h/2$: order {mt['orders_effective'].mean():.2f}")
ax[1].loglog(h, mt["err_effective"][0] * (h / h[0]) ** 2, ":", color="k", lw=1)
ax[1].set(xlabel="h (mm)", ylabel="weighted $L_2$ error (Pa)", title="(b) time domain")
from matplotlib.ticker import FixedLocator, NullLocator, FixedFormatter
for a, ticks in ((ax[0], [0.5, 1, 2, 4]), (ax[1], [0.25, 0.5, 1, 2])):
    a.xaxis.set_major_locator(FixedLocator(ticks))
    a.xaxis.set_major_formatter(FixedFormatter([f"{t:g}" for t in ticks]))
    a.xaxis.set_minor_locator(NullLocator())
    a.grid(True, which="major")
ax[1].set_ylim(1e-5, 0.2)
ax[1].legend(loc="lower right", fontsize=7)
fig.tight_layout()
save(fig, "fig_mms.pdf")

# ===========================================================================
# Figure 3: solution verification, both solvers on the reference geometry
# ===========================================================================
cg = np.load("data/control_grid.npz") if os.path.exists("data/control_grid.npz") else None
cv = np.load("data/convergence_f0.npz")
th, tf = cv["h"], cv["f0"]
fh = np.array([2e-3, 1e-3, 5e-4])
ff = np.array([float(np.load(f"data/fdm_sweep_h{x}.npz")["f_pk"]) for x in ("2.0", "1.0", "0.5")])
if cg is not None:
    th, tf, fh, ff = cg["t_h"], cg["t_f0"], cg["f_h"], cg["f_f0"]

pt, ft_ext, gt = richardson(*tf[:3])
pf, ff_ext, gf = richardson(*ff[:3])
f_theo = float(cv["f_theo"])
num("tOrder", pt, ".2f"); num("tExt", ft_ext, ".1f"); num("tGCI", gt * 100, ".1f")
num("tExtPrec", ft_ext, ".2f")
num("fOrder", pf, ".2f"); num("fExt", ff_ext, ".1f"); num("fGCI", gf * 100, ".1f")
num("fExtPrec", ff_ext, ".2f")
num("solverGap", abs(ft_ext - ff_ext) / ft_ext * 100, ".1f")
num("solverGapPrec", abs(ft_ext - ff_ext) / ft_ext * 100, ".2f")
num("fTheo", f_theo, ".1f")
num("theoGap", abs(ft_ext - f_theo) / f_theo * 100, ".2f")
for k, (hh, f) in enumerate(zip(th, tf)):
    num("tF" + "ABCD"[k], f, ".2f")
for k, (hh, f) in enumerate(zip(fh, ff)):
    num("fF" + "ABCD"[k], f, ".2f")
four_t = len(tf) >= 4
four_f = len(ff) >= 4
if four_t:
    p2, fe2, g2 = richardson(*tf[1:4])
    num("tOrderB", p2, ".2f"); num("tExtB", fe2, ".2f"); num("tGCIB", g2 * 100, ".2f")
    num("tExtShift", abs(fe2 - ft_ext) / ft_ext * 100, ".2f")
if four_f:
    p2, fe2, g2 = richardson(*ff[1:4])
    num("fOrderB", p2, ".2f"); num("fExtB", fe2, ".2f"); num("fGCIB", g2 * 100, ".1f")
    num("fExtShift", abs(fe2 - ff_ext) / ff_ext * 100, ".2f")
    ff_ext2, gf2 = fe2, g2
    num("fRatioA", (ff[0] - ff[1]) / (ff[1] - ff[2]), ".2f")
    num("fRatioB", (ff[1] - ff[2]) / (ff[2] - ff[3]), ".2f")
num("tRatioA", (tf[0] - tf[1]) / (tf[1] - tf[2]), ".2f")
if four_t and four_f:
    num("solverGapB", abs(richardson(*tf[1:4])[1] - richardson(*ff[1:4])[1])
        / richardson(*tf[1:4])[1] * 100, ".2f")

fig, a = plt.subplots(figsize=(TEXTW * 0.62, 2.6))
a.plot(th * 1e3, tf, "o-", color=BLUE, label="time domain (meshed exterior)")
a.plot(fh * 1e3, ff, "s-", color=RED, label="frequency domain (impedance)")
a.errorbar([-0.03], [ft_ext], yerr=[gt * ft_ext], fmt="o", color=BLUE, mfc="white", capsize=3)
a.errorbar([0.03], [ff_ext], yerr=[gf * ff_ext], fmt="s", color=RED, mfc="white", capsize=3)
if four_f:
    a.errorbar([0.09], [ff_ext2], yerr=[gf2 * ff_ext2], fmt="D", color=RED, ms=4, capsize=2,
               label="frequency domain, triplet 1/0.5/0.25 mm")
a.axhline(f_theo, color="k", ls=":", lw=1, label=f"lumped, $\\Delta L = 1.51a$: {f_theo:.1f} Hz")
a.set(xlabel="h (mm)", ylabel="$f_0$ (Hz)", xlim=(-0.12, 2.1))
a.annotate("extrapolated to $h\\to 0$\n(Richardson, bars = GCI)", (0.12, 211.0), fontsize=7)
a.legend(loc="lower right", fontsize=6.5)
fig.tight_layout()
save(fig, "fig_convergence.pdf")

# ===========================================================================
# Figure 4: the resonance seen by both solvers
# ===========================================================================
num("quadrature", np.degrees(float(md["dphi"])), ".1f")
fig = plt.figure(figsize=(TEXTW, 2.3))
g = fig.add_gridspec(1, 2, width_ratios=[1.3, 1.0])
a = fig.add_subplot(g[0])
t = d_or["t"] * 1e3
m = t <= 80
k = max(1, int(m.sum() // 6000))
a.plot(t[m][::k], d_or["p_out"][m][::k] * 1e3, color=GREY, lw=0.8, label="exterior probe")
a.plot(t[m][::k], d_or["p_cav"][m][::k] * 1e3, color=BLUE, lw=0.8, label="cavity probe")
a.set(xlabel="t (ms)", ylabel="p (mPa)", ylim=(-0.36, 0.48),
      title="(a) time domain, 1 mm grid: pulse, then free ringing")
a.legend(loc="upper right", ncol=2)
a = fig.add_subplot(g[1])
cols = {"2.0": "0.7", "1.0": "0.45", "0.5": BLUE, "0.25": RED}
ngrid = 0
for x in ("2.0", "1.0", "0.5", "0.25"):
    fn = f"data/fdm_sweep_h{x}.npz"
    if not os.path.exists(fn):
        continue
    s = np.load(fn)
    ngrid += 1
    a.plot(s["freqs"], s["amp_cav"] / s["amp_cav"].max(), color=cols[x], lw=1.0,
           label=f"h = {x} mm")
a.set(xlabel="f (Hz)", ylabel="$|p_{cav}|$ (normalised)", xlim=(202, 219),
      title=f"(b) frequency domain, {ngrid} grids")
a.legend(loc="upper left", bbox_to_anchor=(0.5, 1.0), fontsize=7)
fig.tight_layout()
save(fig, "fig_resonance.pdf")

# ===========================================================================
# Figure 5: outer boundaries
# ===========================================================================
pm = np.load("data/pml_study.npz")
refl = dict(zip(pm["treatments"], pm["reflection"]))
num("reflSponge", refl["sponge"] * 100, ".0f")
num("reflAbc", refl["abc"] * 100, ".1f")
num("reflPmlDefault", refl["pml"] * 100, ".1f")
num("reflPml", pm["sigma_scan_y"].min() * 100, ".1f")
gv = dict(zip(pm["geometry_labels"], pm["geometry_values"]))
flat = [v for k_, v in gv.items() if k_.startswith("flat")][0]
curved = [v for k_, v in gv.items() if k_.startswith("curved")][0]
num("reflFlat", flat * 100, ".3f"); num("reflFlatdB", 20 * np.log10(flat), ".0f")
num("reflCurved", curved * 100, ".1f"); num("reflCurveddB", 20 * np.log10(curved), ".0f")
drift = {}
for tr in ("sponge", "abc", "pml"):
    q = pm["rows_q"][pm["rows_treatment"] == tr]
    drift[tr] = (q[1] / q[0] - 1) * 100
num("driftSponge", drift["sponge"], "+.1f"); num("driftAbc", drift["abc"], "+.1f")
num("driftPml", drift["pml"], "+.1f")
f0b = pm["rows_f0"]
num("fZeroSpread", (f0b.max() - f0b.min()) / f0b.min() * 100, ".2f")
lam = C / 209.0
xs = pm["sweep_zext"] / lam
num("wavelength", lam, ".2f")
for k, (x, q) in enumerate(zip(xs, pm["sweep_q"])):
    num("extFrac" + "ABCD"[k], x, ".3f"); num("extQ" + "ABCD"[k], q, ".1f")
    num("extF" + "ABCD"[k], pm["sweep_f0"][k], ".2f")
num("extFspread", (pm["sweep_f0"].max() - pm["sweep_f0"].min()) / pm["sweep_f0"].min() * 100, ".2f")
q_imp = float(np.load("data/fdm_sweep_fin.npz")["Q"])
num("qImpedance", q_imp, ".1f")
num("qImpHalf", float(np.load("data/fdm_sweep_h0.5.npz")["Q"]), ".1f")
if os.path.exists("data/fdm_sweep_h0.25.npz"):
    num("qImpQuarter", float(np.load("data/fdm_sweep_h0.25.npz")["Q"]), ".1f")
qb = pm["rows_q"][pm["rows_treatment"] == "pml"]
num("pmlQshallow", qb[0], ".1f"); num("pmlQdeepB", qb[1], ".1f")
num("pmlQdeepC", pm["sweep_q"][1], ".1f")
num("pmlRadiusJump", (pm["sweep_q"][1] / qb[1] - 1) * 100, "+.0f")
rc = np.maximum(0.12, 0.8 * pm["sweep_zext"]) * 100
for k, v in enumerate(rc):
    num("extR" + "ABCD"[k], v, ".3g")
ss = np.load("data/sponge_sensitivity.npz")
num("spongeQmin", ss["Q"].min(), ".0f"); num("spongeQmax", ss["Q"].max(), ".0f")

fig, ax = plt.subplots(1, 2, figsize=(TEXTW, 2.8))
a = ax[0]
bars = [("sponge layer (as used)", refl["sponge"], RED),
        ("first-order Mur", refl["abc"], AMBER),
        ("PML, default strength", refl["pml"], "0.6"),
        ("PML, tuned strength", pm["sigma_scan_y"].min(), BLUE),
        ("PML, flat wall only", flat, BLUE)]
yb = np.arange(len(bars))
a.barh(yb, [b_[1] * 100 for b_ in bars], color=[b_[2] for b_ in bars])
for y_, b_ in zip(yb, bars):
    a.annotate(f"{b_[1]*100:.3g}%", (b_[1] * 100 * 1.15, y_), va="center", fontsize=7)
a.set_xscale("log"); a.set_xlim(0.05, 300)
a.set_yticks(yb); a.set_yticklabels([b_[0] for b_ in bars]); a.invert_yaxis()
a.set(xlabel="spurious reflection (%)", title="(a) reflection against a 4$\\times$ larger domain")
a = ax[1]
a.plot(xs, pm["sweep_q"], "o-", color=BLUE, label="PML, radius $0.8\\times$depth (C)")
for tr, col, mk in (("pml", BLUE, "o"), ("sponge", RED, "s"), ("abc", AMBER, "^")):
    sel = pm["rows_treatment"] == tr
    a.plot(pm["rows_zext"][sel] / lam, pm["rows_q"][sel], mk + "--", color=col, mfc="white",
           label={"pml": "PML", "sponge": "sponge", "abc": "Mur"}[tr] + ", radius 12 cm (B)")
a.axhline(q_imp, color="k", ls=":", lw=1)
a.annotate(f"frequency domain, 1 mm: Q = {q_imp:.0f}", (0.27, q_imp + 5), fontsize=7, ha="right")
a.set(ylim=(50, 312), xlabel="exterior depth / wavelength", ylabel="radiation $Q$",
      title="(b) $Q$ as the exterior grows")
a.legend(loc="upper center", bbox_to_anchor=(0.45, -0.3), ncol=2, fontsize=6.5, frameon=False)
fig.tight_layout()
save(fig, "fig_boundaries.pdf")

# ===========================================================================
# Figure 6: comparison with published measurements
# ===========================================================================
lv = np.load("data/literature_validation.npz")
num("selLumped", float(lv["selamet_lumped"][0]), ".1f")
dl = (lv["selamet_lumped"] - lv["selamet_measured"]) / lv["selamet_measured"] * 100
num("selLumpedDevA", dl[0], "+.1f"); num("selLumpedDevB", dl[1], "+.0f")
num("flaskLumped", float(lv["flask_lumped"]), ".1f")
num("flaskMin", lv["flask_fdm"].min(), ".0f"); num("flaskMax", lv["flask_fdm"].max(), ".0f")
fdev = (lv["flask_fdm"] - lv["flask_measured"]) / lv["flask_measured"] * 100
num("flaskDevMin", fdev.min(), ".0f"); num("flaskDevMax", fdev.max(), ".0f")
num("qFlaskLo", lv["q_measured_range"][0], ".0f"); num("qFlaskHi", lv["q_measured_range"][1], ".0f")
num("qFlaskPred", float(lv["q_predicted"]), ".0f")

fig, ax = plt.subplots(1, 2, figsize=(TEXTW, 2.35), gridspec_kw={"width_ratios": [1.3, 1]})
a = ax[0]
labels = ["Selamet\n$l/d = 1$", "Selamet\n$l/d = 10$", "flask\n(Indenbom)"]
lump = list(dl) + [(float(lv["flask_lumped"]) - 351) / 351 * 100]
fdmv = [2.2, 0.9, (lv["flask_fdm"].mean() - 351) / 351 * 100]   # Selamet: notebook section 5
x = np.arange(3)
a.bar(x - 0.18, lump, 0.36, color=GREY, label="lumped formula")
a.bar(x + 0.18, fdmv, 0.36, color=BLUE, label="finite differences")
a.errorbar([2.18], [fdmv[2]], yerr=[[fdmv[2] - fdev.min()], [fdev.max() - fdmv[2]]],
           fmt="none", color="k", capsize=3, lw=0.9)
a.axhline(0, color="k", lw=0.8)
a.set_xticks(x); a.set_xticklabels(labels)
a.set(ylabel="deviation from measurement (%)", title="(a) resonance frequency")
a.legend(loc="upper left")
a = ax[1]
a.plot(lv["flask_fdm_hcav"] * 1e3, lv["flask_fdm"], "o-", color=BLUE, label="finite differences")
a.axhline(351, color=RED, lw=1.2, label="measured, 351 Hz")
a.axhline(float(lv["flask_lumped"]), color=GREY, ls="--", lw=1, label="lumped")
a.set(xlabel="height of the equivalent cylinder (mm)", ylabel="$f_0$ (Hz)", ylim=(340, 405),
      title="(b) flask: no cavity shape closes the gap")
a.legend(loc="center right")
fig.tight_layout()
save(fig, "fig_validation.pdf")

# ===========================================================================
# Figure 7: neural bases, field error against PDE residual
# ===========================================================================
bb = np.load("data/bench_bases.npz")
names = [str(n) for n in bb["names"]]
neural = [i for i, n in enumerate(names) if not n.startswith("Trefftz")]
num("basesErrMax", bb["field_error"][neural].max() * 100, ".2f")
num("basesErrMin", bb["field_error"][neural].min() * 100, ".2f")
num("basesResMin", bb["residual"][neural].min(), ",.0f")
num("basesResMax", bb["residual"][neural].max(), ",.0f")
it = names.index("Trefftz (Bessel)")
num("trefftzErr", bb["field_error"][it] * 100, ".1f")
num("trefftzRes", bb["residual"][it] * 100, ".0f")
kap = bb["kappa"][neural]
sci = lambda v: r"{:.0f}\times10^{{{:d}}}".format(v / 10 ** int(np.floor(np.log10(v))), int(np.floor(np.log10(v))))
NUM["kappaMin"] = sci(kap.min()); NUM["kappaMax"] = sci(kap.max())
num("basesHidden", int(bb["hidden"]), "d")
rmin = bb["residual_min"][neural]; emin = bb["field_error_minres"][neural]
num("minResLo", rmin.min() * 100, ".1f"); num("minResHi", rmin.max() * 100, ".0f")
num("minResErrLo", emin.min() * 100, ".0f")
NUM["minResErrHi"] = sci(emin.max() * 100)
num("kTwoPoverF", float(bb["k2p_over_f"]), ".0f")
# length scale on which a field error eps*|P| must vary to produce the residual R*|F|:
# R |F| ~ eps |P| / l^2  and  |k^2 P| = K2P |F|  =>  (k l)^2 ~ eps K2P / R
kk = 2 * np.pi * float(bb["freq"]) / C
ig = names.index("Gabor atoms")
ell = np.sqrt(bb["field_error"][ig] * float(bb["k2p_over_f"]) / bb["residual"][ig]) / kk
num("errScale", ell * 1e3, ".1f")
num("waveK", kk, ".2f")
# the target field is the 1 mm solution at 209.84 Hz, that grid's own resonance; on the
# 0.5 mm grid the same frequency is off resonance and the cavity response is much weaker
s05, s10 = np.load("data/fdm_sweep_h0.5.npz"), np.load("data/fdm_sweep_h1.0.npz")
i05 = np.argmin(abs(s05["freqs"] - 209.84)); i10 = np.argmin(abs(s10["freqs"] - 209.84))
num("offResRatio", s10["amp_cav"][i10] / s05["amp_cav"][i05], ".0f")

pretty = [n.replace("w0", "$\\omega_0$") for n in names]
fig, ax = plt.subplots(1, 2, figsize=(TEXTW, 2.7), sharey=True)
y = np.arange(len(names))
colsb = [RED if n.startswith("tanh") else GREY if n.startswith("Trefftz") else BLUE for n in names]
ax[0].barh(y, bb["field_error"] * 100, color=colsb, label="projection of the field")
ax[0].plot(bb["field_error_minres"] * 100, y, "kx", ms=5,
           label="minimum-residual fit")
ax[0].set_xscale("log"); ax[0].set_xlim(0.05, 3e8)
ax[0].set_yticks(y); ax[0].set_yticklabels(pretty); ax[0].invert_yaxis()
ax[0].set(xlabel="field error (%)", title="(a) error on $p$")
ax[1].barh(y, np.maximum(bb["residual"], 1e-3), color=colsb)
ax[1].plot(bb["residual_min"], y, "kx", ms=5)
ax[1].set_xscale("log")
ax[1].axvline(1, color="k", ls="--", lw=1)
ax[1].annotate("zero field", (1.15, len(names) - 0.6), fontsize=7)
ax[1].set(xlabel="PDE residual (multiples of the zero field)",
          title="(b) residual of $\\nabla^2 p + k^2 p + F$")
h_, l_ = ax[0].get_legend_handles_labels()
fig.legend(h_[::-1] if l_[0].startswith("minimum") else h_, ["projection of the field (bars)", "minimum-residual fit (crosses)"],
           loc="lower center", ncol=2, fontsize=7, frameon=False)
fig.tight_layout(rect=(0, 0.08, 1, 1))
save(fig, "fig_bases.pdf")

# ===========================================================================
# Numbers taken directly from versioned files (no computation)
# ===========================================================================
sc = np.load("data/extended_scaling.npz")
num("aTheory", float(sc["a_theory"]), ".2f")
num("dAICcExt", float(sc["dAICc"]), ".1f")
sw = np.load("data/fdm_sweep.npz"); swf = np.load("data/fdm_sweep_fin.npz")
num("sweepPeakCoarse", float(sw["f_pk"]), ".2f"); num("sweepPeakFine", float(swf["f_pk"]), ".2f")
num("sweepAmpCoarse", sw["amp_cav"].max() / 1e3, ".2f"); num("sweepAmpFine", swf["amp_cav"].max() / 1e3, ".2f")
num("sweepBand", float(swf["f_hi"] - swf["f_lo"]), ".2f")
d_or1 = np.load("data/open_resonator.npz")
S = np.pi * R_NECK ** 2; V = np.pi * R_CAV ** 2 * H_CAV
leff = lambda f: S / (V * (2 * np.pi * f / C) ** 2)
dl_raw = (leff(float(tf[1])) - L_NECK) / R_NECK          # damped-sinusoid fit, 1 mm
dl_ext = (leff(ft_ext) - L_NECK) / R_NECK
num("dLraw", dl_raw, ".2f"); num("dLext", dl_ext, ".2f")
num("dLrawGap", (1.51 - dl_raw) / 1.51 * 100, ".0f"); num("dLextGap", (dl_ext - 1.51) / 1.51 * 100, ".0f")
# Ingard's interior correction 0.48 sqrt(S) (1 - 1.25 a/R_cav) + flanged exterior 0.85a
dl_ing = 0.85 + 0.85 * (1 - 1.25 * R_NECK / R_CAV)
num("dLIngardInt", 0.85 * (1 - 1.25 * R_NECK / R_CAV), ".2f"); num("dLIngard", dl_ing, ".2f")
f_ing = C / (2 * np.pi) * np.sqrt(S / (V * (L_NECK + dl_ing * R_NECK)))
num("fIngard", f_ing, ".1f")
num("dLextGapIngard", (dl_ext - dl_ing) / dl_ing * 100, ".0f")
num("tGapIngard", (f_ing - ft_ext) / f_ing * 100, ".1f"); num("fGapIngard", (f_ing - ff_ext) / f_ing * 100, ".1f")

with open(os.path.join("paper", "numbers.tex"), "w", encoding="utf-8") as fh_:
    fh_.write("% Generated by paper/make_figures.py from data/*.npz. Do not edit.\n")
    for k_, v in sorted(NUM.items()):
        fh_.write(f"\\newcommand{{\\{k_}}}{{{v}}}\n")
print(f"numbers: paper/numbers.tex ({len(NUM)} macros)")
