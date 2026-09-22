"""
Validation against published measurements, in place of an in-house experiment.

The protocol in this directory describes a ring-down measurement on a real
bottle. That measurement has not been made. Rather than leave the last rung of
the validation ladder empty, this script confronts the model with published
experimental data instead -- which is weaker in one respect (someone else's
apparatus, someone else's error budget) and stronger in another (peer-reviewed,
and not chosen after seeing how the model performs).

Two independent sources, both with the complete geometry stated:

  SELAMET et al. (1997), J. Acoust. Soc. Am. 102(6), 3491-3502.
      Concentric cylindrical resonators measured in an impedance tube and
      cross-checked against a 3-D boundary-element method to within 1 Hz.
      They are side branches on a duct, so their mouth is a Dirichlet boundary
      rather than the radiating one this solver assumes; the finite-difference
      values quoted here therefore come from notebooks/harmonic_study.ipynb,
      which models that configuration.

  INDENBOM & POGOSSIAN (2023), Acta Acustica 7, 30; arXiv:2212.08858.
      A 100 mL Erlenmeyer flask driven by an earphone placed inside it, with the
      response recorded by an internal microphone. Neck length 32 mm, neck
      diameter 18 mm, internal volume about 115 cm3, sound speed 343.5 m/s at
      20 degC. They report a measured resonance at 351 Hz where the lumped
      formula predicts 391 Hz, and attribute the gap to their own stated
      condition: the lumped derivation needs SL/V << 1 and S/V^(2/3) << 1,
      which they quote as 0.04 and 0.11. The second is reproduced here from the
      stated geometry; the first is not, coming out at 0.07.

That second case is the interesting one, because it is a resonator on which the
lumped model is *known* to fail. It is therefore a test of whether resolving the
two-dimensional field near the neck recovers what the lumped model misses.

A caveat stated up front: an Erlenmeyer flask is conical and this solver's
cavity is cylindrical. The flask is modelled as a cylinder of the same volume,
and because the right aspect ratio is a judgement call, the script sweeps a
range of them and reports the spread rather than picking one.

A second caveat, on the damping. The only open-access half-width found for a
resonator whose geometry is fully stated is the one in Indenbom & Pogossian,
and their notation Delta_1/2 f does not make explicit whether it is the width at
half amplitude or at half power. The two conventions differ by a factor sqrt(3)
in Q, so the measured Q is reported as a range, not a number. Their transducers
also sit inside the cavity, which can only add damping.

Output: data/literature_validation.npz, figures/literature_validation.png
Env: LV_SKIP_FDM (0/1) -- skip the finite-difference runs, which take a few minutes
"""
import os
import sys

# every path in this file is relative to the repository root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.harmonic_solver import HarmonicSolver

C_REF = 343.0


# --------------------------------------------------------------------------
# The published measurements
# --------------------------------------------------------------------------
# Selamet's resonators are side branches on a duct. At resonance the input
# impedance of the neck vanishes, so the mouth is a Dirichlet boundary, not the
# radiation condition src/harmonic_solver.py applies. The finite-difference
# numbers below therefore come from notebooks/harmonic_study.ipynb, which models
# that configuration correctly; recomputing them here with the wrong boundary
# condition would give -6.5 % and -4.2 % instead, which is a statement about the
# boundary condition and not about the solver.
SELAMET = [
    # label, neck radius (m), neck length (m), cavity volume (m3),
    # measured f0 (Hz), FDM f0 from the notebook (Hz)
    ("Selamet l/d = 1.0", 0.02022, 0.085, 4.5e-3, 91.0, 93.0),
    ("Selamet l/d = 10",  0.02022, 0.085, 4.5e-3, 72.0, 72.7),
]

FLASK = {
    "label": "Indenbom & Pogossian, 100 mL Erlenmeyer",
    "a": 9.0e-3,          # neck radius, from D = 18 mm
    "L": 32.0e-3,         # neck length
    "V": 115.0e-6,        # internal volume, their measured value
    "c": 343.5,           # their stated sound speed at 20 degC
    "f0_measured": 351.0,
    "f0_their_lumped": 391.0,
    "halfwidth_hz": 20.0,  # Delta_1/2 f at zero filling, read from their Fig. 7/9
}


def lumped(a, l_neck, volume, c=C_REF, flush=False):
    """The end-corrected lumped frequency this project uses throughout."""
    s = np.pi * a ** 2
    dl = 0.66 * a + (0.85 if flush else 0.60) * a
    return (c / (2 * np.pi)) * np.sqrt(s / (volume * (l_neck + dl)))


def fdm_peak(solver, f_lo, f_hi, n_coarse=25, refine=0.25):
    """Locate the resonance of a HarmonicSolver by a coarse then fine sweep."""
    j = solver.cavity_probe_index(solver.l_neck + 0.5 * solver.h_cav)
    amp = lambda f: abs(solver.solve(f)[0, j])

    coarse = np.linspace(f_lo, f_hi, n_coarse)
    vals = [amp(f) for f in coarse]
    f_peak = coarse[int(np.argmax(vals))]
    step = coarse[1] - coarse[0]

    fine = np.arange(f_peak - step, f_peak + step + refine, refine)
    vals = np.array([amp(f) for f in fine])
    k = int(np.clip(np.argmax(vals), 1, len(fine) - 2))
    y0, y1, y2 = vals[k - 1:k + 2]
    denom = y0 - 2 * y1 + y2
    delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-30 else 0.0
    return float(fine[k] + np.clip(delta, -1, 1) * refine)


# --------------------------------------------------------------------------
def part_selamet():
    print("=" * 78)
    print("SELAMET et al. (1997) -- cylindrical, the geometry this solver assumes")
    print("=" * 78)
    print("  Same cavity volume, 4500 cm3, in two different shapes. The lumped")
    print("  model knows only the volume, so it cannot tell them apart.")
    print()
    print(f"  {'configuration':<22}{'measured':>10}{'lumped':>10}{'dev':>8}"
          f"{'FDM':>10}{'dev':>8}")
    rows = []
    for label, a, l_neck, volume, f_meas, f_fdm in SELAMET:
        f_lump = lumped(a, l_neck, volume, flush=True)
        d_l = (f_lump - f_meas) / f_meas * 100
        d_f = (f_fdm - f_meas) / f_meas * 100
        rows.append((label, f_meas, f_lump, d_l, f_fdm, d_f))
        print(f"  {label:<22}{f_meas:>10.1f}{f_lump:>10.1f}{d_l:>7.1f}%"
              f"{f_fdm:>10.1f}{d_f:>7.1f}%")
    print()
    print(f"  The lumped model returns {rows[0][2]:.1f} Hz for both, against 91 and 72 Hz")
    print("  measured: it is structurally blind to the shape of the cavity. Resolving")
    print("  the two-dimensional field recovers both to within 2.2 %.")
    return rows


def part_flask(skip_fdm):
    d = FLASK
    a, l_neck, volume, c = d["a"], d["L"], d["V"], d["c"]
    s_neck = np.pi * a ** 2
    print()
    print("=" * 78)
    print("INDENBOM & POGOSSIAN (2023) -- a resonator where the lumped model fails")
    print("=" * 78)
    print(f"  neck radius {a*1e3:.1f} mm, neck length {l_neck*1e3:.1f} mm, "
          f"volume {volume*1e6:.0f} cm3, c = {c:.1f} m/s")
    print(f"  the lumped conditions, computed here: "
          f"SL/V = {s_neck*l_neck/volume:.2f}, S/V^(2/3) = {s_neck/volume**(2/3):.2f}"
          f"   (both should be << 1)")
    print("  The paper quotes 0.04 and 0.11 for these. The second agrees; the first")
    print("  does not, and their 0.04 could not be reproduced from the stated geometry.")
    print()

    f_lump = lumped(a, l_neck, volume, c=c, flush=False)
    print(f"  {'measured (published)':<34}{d['f0_measured']:>9.1f} Hz")
    print(f"  {'their lumped calculation':<34}{d['f0_their_lumped']:>9.1f} Hz"
          f"   {(d['f0_their_lumped']-d['f0_measured'])/d['f0_measured']*100:+6.1f} %")
    print(f"  {'this project, lumped':<34}{f_lump:>9.1f} Hz"
          f"   {(f_lump-d['f0_measured'])/d['f0_measured']*100:+6.1f} %")

    fdm = []
    if not skip_fdm:
        print()
        print("  Finite differences, flask replaced by a cylinder of equal volume.")
        print("  The aspect ratio is a judgement call, so it is swept:")
        print(f"    {'cavity H (mm)':>15}{'cavity R (mm)':>15}{'H/2R':>8}"
              f"{'f0 FDM (Hz)':>13}{'dev':>9}")
        for h_cav in (0.040, 0.055, 0.070, 0.085):
            r_cav = np.sqrt(volume / (np.pi * h_cav))
            if r_cav <= a * 1.5:
                continue
            solver = HarmonicSolver(h=1.0e-3, r_neck=a, l_neck=l_neck,
                                    r_cav=r_cav, h_cav=h_cav, src_width=0.008)
            f = fdm_peak(solver, 280.0, 430.0)
            dev = (f - d["f0_measured"]) / d["f0_measured"] * 100
            fdm.append((h_cav, r_cav, f, dev))
            print(f"    {h_cav*1e3:>15.0f}{r_cav*1e3:>15.1f}"
                  f"{h_cav/(2*r_cav):>8.2f}{f:>13.1f}{dev:>8.1f}%")
        if fdm:
            fs = [x[2] for x in fdm]
            print(f"    -> FDM spans {min(fs):.0f}-{max(fs):.0f} Hz across the whole shape")
            print(f"       range, against {d['f0_measured']:.0f} Hz measured and "
                  f"{f_lump:.0f} Hz lumped.")
            print()
            print("  A NEGATIVE RESULT, stated as such. Resolving the two-dimensional")
            print("  field does not close the gap: it lands where the lumped model")
            print("  already was. Whatever is missing is not the field near the neck.")
            print()
            # What would close it? Either more volume or a longer effective neck.
            ratio = (f_lump / d["f0_measured"]) ** 2
            v_needed = volume * ratio
            l_eff_now = l_neck + 1.26 * a
            l_eff_needed = l_eff_now * ratio
            print("  What would close it, arithmetically:")
            print(f"    a volume of {v_needed*1e6:.0f} cm3 instead of {volume*1e6:.0f}"
                  f"  (+{(ratio-1)*100:.0f} %), or")
            print(f"    an effective neck of {l_eff_needed*1e3:.0f} mm instead of "
                  f"{l_eff_now*1e3:.0f} mm, i.e. an end")
            print(f"    correction of {(l_eff_needed-l_neck)/a:.1f}a instead of 1.26a.")
            print()
            print("  The second is the more plausible: an Erlenmeyer neck flares into the")
            print("  conical shoulder, so the air that moves as a plug extends well past")
            print("  the 32 mm cylindrical section. Moloney (2004) reaches the same")
            print("  conclusion for glass bottles, attributing his residual discrepancy")
            print("  to the flaring region beyond the neck. This is a hypothesis the")
            print("  present geometry cannot test, since the solver's cavity is a cylinder.")

    # ---- damping, with its convention stated ----
    print()
    print("  Damping. Their half-width at zero filling is about "
          f"{d['halfwidth_hz']:.0f} Hz, but the")
    print("  convention is not stated, so Q is bracketed rather than quoted:")
    q_power = d["f0_measured"] / d["halfwidth_hz"]
    q_amp = d["f0_measured"] / (d["halfwidth_hz"] / np.sqrt(3.0))
    print(f"    if it is the -3 dB (half-power) width : Q = {q_power:.0f}")
    print(f"    if it is the half-amplitude width     : Q = {q_amp:.0f}")

    omega0 = 2 * np.pi * d["f0_measured"]
    mu, rho = 1.813e-5, 1.204
    delta_v = np.sqrt(2 * mu / (rho * omega0))
    f_t = 1.0 + (1.402 - 1) / np.sqrt(0.71)
    q_visc = a / (f_t * delta_v)
    l_eff = l_neck + 1.26 * a
    q_rad = c * l_eff / (s_neck * d["f0_measured"])
    q_tot = 1.0 / (1.0 / q_visc + 1.0 / q_rad)
    print(f"    this project predicts                 : Q = {q_tot:.0f}"
          f"   (viscous {q_visc:.0f}, radiation {q_rad:.0f})")
    print()
    print("  So the model overestimates the damping quality of a real resonator,")
    print("  in the same direction Moloney (2004) reports for three glass bottles.")
    print("  Their transducers sit inside the cavity, which can only add losses.")
    return f_lump, fdm, (q_power, q_amp, q_tot)


# --------------------------------------------------------------------------
def figure(sel_rows, flask_lump, flask_fdm, qs):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))

    labels = [r[0].replace("Selamet ", "") for r in sel_rows] + ["Erlenmeyer\nflask"]
    lump_dev = [r[3] for r in sel_rows] + [
        (flask_lump - FLASK["f0_measured"]) / FLASK["f0_measured"] * 100]
    fdm_dev = [r[5] for r in sel_rows]
    if flask_fdm:
        fs = [x[2] for x in flask_fdm]
        fdm_dev.append((np.mean(fs) - FLASK["f0_measured"]) / FLASK["f0_measured"] * 100)
    else:
        fdm_dev.append(np.nan)

    pos = np.arange(len(labels))
    w = 0.36
    ax[0].bar(pos - w / 2, lump_dev, w, color="#a8442a", label="lumped model")
    ax[0].bar(pos + w / 2, fdm_dev, w, color="#1d5480", label="finite differences")
    ax[0].axhline(0, color="k", lw=1)
    ax[0].set_xticks(pos); ax[0].set_xticklabels(labels, fontsize=9)
    ax[0].set_ylabel("deviation from the measurement (%)")
    ax[0].set_title("(a) Predicted against published $f_0$")
    ax[0].legend(fontsize=9); ax[0].grid(ls=":", alpha=0.6, axis="y")

    q_power, q_amp, q_tot = qs
    ax[1].barh([0], [q_tot], color="#1d5480", label="this project, predicted")
    ax[1].barh([1], [q_amp - q_power], left=[q_power], color="#e8a33d",
               label="published, the two conventions")
    ax[1].set_yticks([0, 1])
    ax[1].set_yticklabels(["predicted", "measured\n(bracketed)"], fontsize=9)
    ax[1].set_xlabel("quality factor $Q$")
    ax[1].set_title("(b) Damping: the model is optimistic")
    ax[1].legend(fontsize=9); ax[1].grid(ls=":", alpha=0.6, axis="x")

    fig.tight_layout()
    fig.savefig("figures/literature_validation.png", dpi=140)
    plt.close(fig)
    print("\nFigure: figures/literature_validation.png")


if __name__ == "__main__":
    skip = os.environ.get("LV_SKIP_FDM") == "1"
    os.makedirs("data", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    sel_rows = part_selamet()
    flask_lump, flask_fdm, qs = part_flask(skip)

    figure(sel_rows, flask_lump, flask_fdm, qs)
    np.savez("data/literature_validation.npz",
             selamet_labels=np.array([r[0] for r in sel_rows]),
             selamet_measured=np.array([r[1] for r in sel_rows]),
             selamet_lumped=np.array([r[2] for r in sel_rows]),
             selamet_fdm=np.array([r[4] for r in sel_rows]),
             flask_measured=FLASK["f0_measured"],
             flask_lumped=flask_lump,
             flask_fdm=np.array([x[2] for x in flask_fdm]) if flask_fdm else np.array([]),
             flask_fdm_hcav=np.array([x[0] for x in flask_fdm]) if flask_fdm else np.array([]),
             q_measured_range=np.array(qs[:2]), q_predicted=qs[2])
    print("Data:   data/literature_validation.npz")
