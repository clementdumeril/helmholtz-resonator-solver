"""
Regenerate the four figures of paper/paper.pdf from stored data.

Until now these figures had no generating script in the repository: they were
produced by exploratory code that has since been removed, which meant the paper
could not be rebuilt from the repository alone. This script closes that gap. It
reads only `data/*.npz` and rewrites:

    figures/doe_resonance.png        frequency response at L = 4 cm, h = 2 mm
    figures/scaling_law_gci.png      weighted fit of the two scaling models
    figures/phase1_losses.png        ring-down, logarithmic decrement, spectrum
    figures/phase2_verification.png  comparison with the literature

The fits of `scaling_law_gci` are recomputed here rather than read back, so the
figure, the stored results and the tables of the paper all come from one place.

Run: python tools/make_paper_figures.py
"""

import os
import sys

# every path in this file is relative to the repository root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import hilbert

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
os.makedirs("figures", exist_ok=True)

C = 343.0
R_NECK, R_CAV, H_CAV = 0.01, 0.04, 0.08
A_H_TH = (C / (2 * np.pi)) * (R_NECK / R_CAV) / np.sqrt(H_CAV)

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
                     "axes.labelweight": "bold", "figure.dpi": 120})


# --------------------------------------------------------------------------
def fig_doe_resonance():
    d = np.load("data/doe_resonance.npz")
    freqs, amp, f0 = d["freqs"], d["amp"], float(d["f0"])

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(freqs, amp, color="#1d5480", lw=1.6)
    ax.axvline(f0, color="#a8442a", ls="--", lw=1.2,
               label=f"peak, parabolic fit: {f0:.1f} Hz")
    ax.axvline(220.6, color="0.45", ls=":", lw=1.4,
               label="extrapolated to zero mesh: 220.6 Hz")
    ax.set_xlabel("Frequency $f$ (Hz)")
    ax.set_ylabel("Mean cavity amplitude (Pa)")
    ax.set_title("Frequency response, $L = 4$ cm, $h = 2$ mm")
    ax.grid(ls=":", alpha=0.7)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig("figures/doe_resonance.png", dpi=150)
    plt.close(fig)
    print("  figures/doe_resonance.png")


# --------------------------------------------------------------------------
def fig_scaling_law():
    """Weighted fit of the two scaling models, recomputed from the frequencies."""
    d = np.load("data/scaling_law_gci_results.npz")
    L, f_ext, sigma = d["L"], d["f_extrap"], d["sigma"]

    def power(x, a, b):
        return a * x ** b

    def efflen(x, a, dl):
        return a / np.sqrt(x + dl)

    def weighted_fit(model, p0):
        popt, pcov = curve_fit(model, L, f_ext, p0=p0, sigma=sigma,
                               absolute_sigma=True, maxfev=30000)
        chi2 = float(np.sum(((f_ext - model(L, *popt)) / sigma) ** 2))
        aicc = chi2 + 2 * 2 + 2 * 2 * 3 / (len(L) - 2 - 1)
        loo = []
        for i in range(len(L)):
            m = np.ones(len(L), bool)
            m[i] = False
            pp, _ = curve_fit(model, L[m], f_ext[m], p0=popt, sigma=sigma[m],
                              absolute_sigma=True, maxfev=30000)
            loo.append(f_ext[i] - model(L[i], *pp))
        return (popt, np.sqrt(np.diag(pcov)), chi2 / (len(L) - 2), aicc,
                float(np.sqrt(np.mean(np.array(loo) ** 2))))

    pa, ea, chi_a, aic_a, loo_a = weighted_fit(power, [A_H_TH, -0.5])
    pb, eb, chi_b, aic_b, loo_b = weighted_fit(efflen, [A_H_TH, 0.006])

    print(f"    (A) power law      A = {pa[0]:.1f} +/- {ea[0]:.1f}, "
          f"b = {pa[1]:.3f} +/- {ea[1]:.3f} | chi2/dof {chi_a:.2f} | "
          f"AICc {aic_a:.1f} | LOO {loo_a:.2f} Hz")
    print(f"    (B) effective len. A_H = {pb[0]:.1f} +/- {eb[0]:.1f}, "
          f"dL = {pb[1] * 1e3:.1f} +/- {eb[1] * 1e3:.1f} mm "
          f"({pb[1] / R_NECK:.2f} R) | chi2/dof {chi_b:.3f} | "
          f"AICc {aic_b:.1f} | LOO {loo_b:.2f} Hz")
    print(f"    dAICc (A-B) = {aic_a - aic_b:.2f}")

    grid = np.linspace(L.min() * 0.88, L.max() * 1.12, 300)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6),
                                 gridspec_kw={"width_ratios": [1.45, 1]})

    a1.plot(grid * 100, power(grid, *pa), "r-", lw=1.8,
            label=f"(A) $aL^b$, $b = {pa[1]:.2f}$ (LOO {loo_a:.1f} Hz)")
    a1.plot(grid * 100, efflen(grid, *pb), "b--", lw=1.8,
            label=f"(B) $a/\\sqrt{{L+\\Delta L}}$, "
                  f"$\\Delta L = {pb[1] / R_NECK:.2f}R_{{neck}}$ (LOO {loo_b:.1f} Hz)")
    a1.errorbar(L * 100, f_ext, yerr=sigma, fmt="ko", ms=7, capsize=3,
                label="FDM extrapolated (Richardson) $\\pm$ GCI", zorder=5)
    a1.set_xlabel("Neck length $L$ (cm)")
    a1.set_ylabel("Natural frequency $f_0$ (Hz)")
    a1.set_title(f"(a) Weighted fit ($\\Delta$AICc = {aic_a - aic_b:.1f}, not decisive)")
    a1.grid(ls=":", alpha=0.7)
    a1.legend(fontsize=9)

    for model, popt, colour, lab in [(power, pa, "r", "(A)"), (efflen, pb, "b", "(B)")]:
        a2.errorbar(L * 100, f_ext - model(L, *popt), yerr=sigma, fmt="o", ms=7,
                    color=colour, capsize=3, label=lab)
    a2.axhline(0, color="k", lw=0.9)
    a2.set_xlabel("Neck length $L$ (cm)")
    a2.set_ylabel("Residual $f_0 -$ model (Hz)")
    a2.set_title("(b) Residuals against the GCI uncertainty")
    a2.grid(ls=":", alpha=0.7)
    a2.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig("figures/scaling_law_gci.png", dpi=150)
    plt.close(fig)

    np.savez("data/scaling_law_gci_results.npz", L=L, f_fine=d["f_fine"],
             f_extrap=f_ext, sigma=sigma, popt_power=pa, perr_power=ea,
             aicc_power=aic_a, loo_power=loo_a, popt_efflen=pb, perr_efflen=eb,
             aicc_efflen=aic_b, loo_efflen=loo_b, dAICc=aic_a - aic_b,
             a_theory=A_H_TH)
    print("  figures/scaling_law_gci.png  (and refreshed data/scaling_law_gci_results.npz)")


# --------------------------------------------------------------------------
def fig_losses():
    """Ring-down of the reduced damping model: signal, decrement, spectrum.

    The dissipation map of the earlier version of this figure is dropped on
    purpose: it only reproduced the localisation of gamma that was imposed by
    hand, so it illustrated an input rather than a result.

    Note on the value of Q. The damping amplitude was calibrated so that the
    decay would reproduce Q_visc = 47.4. Measuring the stored trace again, with
    a logarithmic decrement on the Hilbert envelope, returns 45.8 -- a 3.3 %
    spread between two estimators of the same decay, and a reminder that the
    agreement here is a calibration check rather than an independent result.
    """
    d = np.load("data/phase1_losses_results.npz")
    t = d["t"]
    none_, bulk, visc = d["cav_none"], d["cav_bulk"], d["cav_visc"]
    f0 = float(d["f0"])
    q_analytic = float(d["Q_visc_analytic"])
    q_bulk = float(d["Q_bulk_analytic"])
    dt = float(t[1] - t[0])

    # Logarithmic decrement: the same estimator as experiments/ringdown_analysis.py.
    env = np.abs(hilbert(visc))
    keep = (t > 10e-3) & (t < 55e-3)          # avoid the Hilbert edge effects
    slope = np.polyfit(t[keep], np.log(env[keep]), 1)[0]
    q_measured = -np.pi * f0 / slope

    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.4))

    ax[0].plot(t * 1e3, none_, color="0.6", lw=1.0, label="lossless ($Q\\to\\infty$)")
    ax[0].plot(t * 1e3, bulk, color="#2e7d32", lw=1.0,
               label=f"Stokes bulk ($Q\\approx{q_bulk:.0e}$)")
    ax[0].plot(t * 1e3, visc, color="#1d5480", lw=1.2,
               label=f"viscothermal neck ($Q={q_analytic:.0f}$)")
    ax[0].set_xlabel("Time $t$ (ms)")
    ax[0].set_ylabel("Mean cavity pressure $p(t)$ (Pa)")
    ax[0].set_title("(a) Only the neck model damps")
    ax[0].legend(fontsize=8.5, loc="upper right")
    ax[0].grid(ls=":", alpha=0.7)

    ax[1].semilogy(t * 1e3, env, color="0.6", lw=0.9, label="Hilbert envelope")
    ax[1].semilogy(t[keep] * 1e3, np.exp(np.polyval(np.polyfit(t[keep], np.log(env[keep]), 1), t[keep])),
                   "r--", lw=1.8, label=f"decrement: $Q = {q_measured:.1f}$"
                   f"  (calibration target {q_analytic:.1f})")
    ax[1].set_xlabel("Time $t$ (ms)")
    ax[1].set_ylabel("Envelope of $p(t)$ (Pa)")
    ax[1].set_title("(b) Logarithmic decrement $\\Rightarrow Q$")
    ax[1].legend(fontsize=8.5)
    ax[1].grid(ls=":", alpha=0.7, which="both")

    win = np.hanning(visc.size)
    freq = np.fft.rfftfreq(visc.size, dt)
    band = (freq > 150) & (freq < 325)
    sp_v = np.abs(np.fft.rfft(visc * win))
    sp_n = np.abs(np.fft.rfft(none_ * win))
    ax[2].plot(freq[band], sp_n[band] / sp_n[band].max(), color="0.6", lw=1.4,
               label="lossless")
    ax[2].plot(freq[band], sp_v[band] / sp_v[band].max(), color="#1d5480", lw=1.6,
               label="viscothermal")
    ax[2].fill_between(freq[band], sp_v[band] / sp_v[band].max(), alpha=0.18,
                       color="#1d5480")
    ax[2].axvline(f0, color="#a8442a", ls=":", lw=1.2)
    ax[2].set_xlabel("Frequency $f$ (Hz)")
    ax[2].set_ylabel("Normalised spectral amplitude")
    ax[2].set_title("(c) Ring-down spectrum")
    ax[2].legend(fontsize=8.5)
    ax[2].grid(ls=":", alpha=0.7)

    fig.tight_layout()
    fig.savefig("figures/phase1_losses.png", dpi=150)
    plt.close(fig)
    print(f"  figures/phase1_losses.png    (calibration target Q = {q_analytic:.1f}, "
          f"decrement on the stored trace {q_measured:.1f}, "
          f"{abs(q_measured - q_analytic) / q_analytic * 100:.1f} % apart)")
    return q_measured


# --------------------------------------------------------------------------
def fig_verification(q_solver):
    d = np.load("data/phase2_verification_results.npz")
    L, f0_fdm = d["L_vals"], d["f0_fdm"]
    q_geo = d["Q_geo"]
    dl_fit = float(d["dL_fit"])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.4))

    grid = np.linspace(0.014, 0.066, 300)
    helm = lambda x, dl: A_H_TH / np.sqrt(x + dl)
    a1.fill_between(grid * 100, helm(grid, 0.85 * R_NECK), helm(grid, 0.60 * R_NECK),
                    color="#1d5480", alpha=0.18,
                    label="Ingard band $\\Delta L \\in [0.60; 0.85]\\,R_{neck}$")
    a1.plot(grid * 100, helm(grid, dl_fit), "b--", lw=1.8,
            label=f"Helmholtz, $\\Delta L = {dl_fit / R_NECK:.2f}\\,R_{{neck}}$ (fitted)")
    a1.plot(grid * 100, helm(grid, 0.0), ":", color="#2e7d32", lw=1.6,
            label="pure 1-D theory ($\\Delta L = 0$)")
    a1.plot(L * 100, f0_fdm, "ko", ms=8, label="FDM (fine grid)", zorder=5)
    a1.set_xlabel("Neck length $L$ (cm)")
    a1.set_ylabel("Natural frequency $f_0$ (Hz)")
    a1.set_title("(a) The FDM falls inside the Rayleigh--Ingard band")
    a1.grid(ls=":", alpha=0.7)
    a1.legend(fontsize=9)

    a2.axhspan(30, 70, color="#e8a33d", alpha=0.20,
               label="literature (Moloney 2004): $Q \\sim 30$--$70$")
    fgrid = np.linspace(150, 320, 300)
    a2.plot(fgrid, q_geo[2] * np.sqrt(fgrid / f0_fdm[2]), "b-", lw=1.8,
            label="$Q_{visc} = a/(F_t\\delta_v) \\propto \\sqrt{\\omega_0}$")
    a2.plot(f0_fdm, q_geo, "ko", ms=8, label="$Q_{visc}$ at the five FDM geometries")
    a2.plot(f0_fdm[2], q_solver, "D", color="#c62828", ms=11,
            label=f"transient solver, decrement ($Q = {q_solver:.1f}$)")
    a2.set_xlabel("Natural frequency $f_0$ (Hz)")
    a2.set_ylabel("Quality factor $Q$")
    a2.set_title("(b) Boundary-layer prediction against the literature")
    a2.set_xlim(150, 325)
    a2.grid(ls=":", alpha=0.7)
    a2.legend(fontsize=9, loc="lower right")

    fig.tight_layout()
    fig.savefig("figures/phase2_verification.png", dpi=150)
    plt.close(fig)
    print("  figures/phase2_verification.png")


if __name__ == "__main__":
    print("Regenerating the figures of paper/paper.pdf")
    fig_doe_resonance()
    fig_scaling_law()
    q_measured = fig_losses()
    fig_verification(q_measured)
    print("Done. Rebuild the paper with:  latexmk -pdf paper/paper.tex")
