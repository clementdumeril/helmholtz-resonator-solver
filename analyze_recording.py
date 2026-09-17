"""
Ring-down analysis of a real Helmholtz resonator recorded with a phone.

Closes the loop of this project: the two numerical solvers predict a natural
frequency f0 and a quality factor Q, and this script measures both on a
physical object (a glass bottle, a hand slap, a phone microphone).

Method -- deliberately the SAME estimator the transient solver uses, so that
the measured and the computed Q are directly comparable:

  1. locate the impulse, keep the free decay that follows it;
  2. windowed FFT of the decay -> f0 (dominant peak, parabolic interpolation
     of the log-magnitude for sub-bin resolution);
  3. zero-phase Butterworth band-pass around f0 -> isolates the mode;
  4. Hilbert envelope -> log-envelope -> least-squares slope over a window
     bounded by the -3 dB point and the noise floor;
  5. logarithmic decrement -> Q = pi f0 tau, with a standard error taken from
     the residuals of that fit.

The lumped predictions it is compared against come from the measured geometry
alone, with no adjustable parameter:

     f0 = (c / 2 pi) sqrt( S / (V (L + dL)) ),        S = pi a^2
     Q_visc = a / (F_t delta_v),   delta_v = sqrt(2 mu / (rho omega0))
     Q_rad  = c (L + dL) / (S f0)          (baffled piston, lumped)
     1 / Q_total = 1 / Q_visc + 1 / Q_rad

Usage
-----
    python analyze_recording.py rec1.wav [rec2.wav ...] [options]
    python analyze_recording.py --self-test

Geometry, in millimetres and millilitres (caliper and kitchen scale):
    --neck-radius 10.5 --neck-length 25 --volume 500 [--protruding]

The self-test synthesises a decay of known f0 and Q, runs the whole pipeline on
it and checks that both are recovered -- so the analysis chain is verified
before any bottle is recorded.

Dependencies: numpy, scipy, matplotlib. No audio library: WAV is read by
scipy.io.wavfile, so convert M4A/MP3 to WAV first (for instance with ffmpeg).

Outputs: data/experiment_<tag>.npz, plots/experiment_<tag>_<file>.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, hilbert, sosfiltfilt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

# --------------------------------------------------------------------------
# Air properties
# --------------------------------------------------------------------------
P_ATM = 101325.0        # Pa
R_AIR = 287.05          # J/(kg K)
PRANDTL = 0.71
GAMMA = 1.402
F_T = 1.0 + (GAMMA - 1.0) / np.sqrt(PRANDTL)   # ~1.48: viscous + thermal layer


def air_properties(t_celsius):
    """Speed of sound, density and dynamic viscosity of air at 1 atm."""
    t_kelvin = t_celsius + 273.15
    c = 331.3 * np.sqrt(1.0 + t_celsius / 273.15)
    rho = P_ATM / (R_AIR * t_kelvin)
    mu = 1.458e-6 * t_kelvin ** 1.5 / (t_kelvin + 110.4)      # Sutherland
    return {"T_C": t_celsius, "c": c, "rho": rho, "mu": mu}


# --------------------------------------------------------------------------
# Lumped prediction from the measured geometry
# --------------------------------------------------------------------------
def predict(neck_radius, neck_length, volume, t_celsius=20.0, flush=True):
    """Lumped-element prediction of f0 and Q from geometry only.

    neck_radius, neck_length in metres; volume in cubic metres. `flush`
    selects the exterior end correction: 0.85 a for a mouth flush with a
    surface (baffled), 0.60 a for a neck protruding into free space.
    """
    air = air_properties(t_celsius)
    c, rho, mu = air["c"], air["rho"], air["mu"]

    a = neck_radius
    s_neck = np.pi * a ** 2
    dl_inner = 0.66 * a                          # cavity side (Ingard)
    dl_outer = (0.85 if flush else 0.60) * a     # radiating side
    l_eff = neck_length + dl_inner + dl_outer

    f0 = (c / (2 * np.pi)) * np.sqrt(s_neck / (volume * l_eff))
    omega0 = 2 * np.pi * f0

    delta_v = np.sqrt(2 * mu / (rho * omega0))   # viscous boundary layer
    q_visc = a / (F_T * delta_v)                 # neck boundary-layer losses
    q_rad = c * l_eff / (s_neck * f0)            # baffled-piston radiation
    q_total = 1.0 / (1.0 / q_visc + 1.0 / q_rad)

    return {**air, "a": a, "L": neck_length, "V": volume, "S_neck": s_neck,
            "dL_inner": dl_inner, "dL_outer": dl_outer, "L_eff": l_eff,
            "flush": flush, "f0": f0, "delta_v": delta_v,
            "Q_visc": q_visc, "Q_rad": q_rad, "Q_total": q_total,
            "f0_uncorrected": (c / (2 * np.pi)) * np.sqrt(s_neck / (volume * neck_length))}


# --------------------------------------------------------------------------
# Signal handling
# --------------------------------------------------------------------------
def load_wav(path):
    """Read a WAV file as a mono float64 signal in [-1, 1], plus its rate."""
    rate, data = wavfile.read(path)
    data = np.asarray(data)
    if data.ndim > 1:                            # stereo -> mono
        data = data.mean(axis=1)
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float64) / float(np.iinfo(data.dtype).max)
    else:
        data = data.astype(np.float64)
    return data - data.mean(), float(rate)


def find_decay(sig, rate, f_guess=200.0, skip_periods=5.0):
    """Bracket the free decay: from just after the impulse to the noise floor.

    The excitation is broadband and short, so the envelope peaks on the slap
    itself, which is not the mode. A few periods are skipped so that the click
    and the band-pass transient have died out before the fit starts. Returns
    the decay bracket and the end of the quiet pre-roll, which is where the
    noise floor is measured.
    """
    env = np.abs(hilbert(sig))
    i_peak = int(np.argmax(env))
    i_start = min(i_peak + int(skip_periods * rate / f_guess), len(sig) - 2)
    i_quiet = max(i_peak - int(0.02 * rate), 0)          # pure noise, if any

    pre = env[:i_quiet]
    noise = float(np.median(pre)) if pre.size > 100 else float(np.percentile(env, 5))

    tail = env[i_start:]
    above = np.where(tail > 4.0 * max(noise, 1e-12))[0]
    i_end = i_start + int(above[-1]) if above.size else len(sig) - 1
    if i_end - i_start < int(0.02 * rate):               # keep at least 20 ms
        i_end = min(i_start + int(0.2 * rate), len(sig) - 1)
    return i_start, i_end, i_quiet


def spectral_peak(sig, rate, f_lo=40.0, f_hi=2000.0):
    """Dominant frequency by FFT, refined by a parabola on the log-magnitude."""
    n = int(2 ** np.ceil(np.log2(max(len(sig), 1024))) * 4)      # zero-pad x4
    spec = np.abs(np.fft.rfft(sig * np.hanning(len(sig)), n))
    freq = np.fft.rfftfreq(n, 1.0 / rate)

    band = (freq >= f_lo) & (freq <= f_hi)
    idx = int(np.argmax(np.where(band, spec, 0.0)))
    if 0 < idx < len(spec) - 1:
        y0, y1, y2 = np.log(spec[idx - 1:idx + 2] + 1e-30)
        denom = y0 - 2 * y1 + y2
        delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-30 else 0.0
        delta = float(np.clip(delta, -0.5, 0.5))
    else:
        delta = 0.0
    return float(freq[idx] + delta * (freq[1] - freq[0])), freq, spec


def measure_q(sig, rate, f0, i_start, i_quiet, rel_bandwidth=0.20, order=4,
              floor_db=-20.0, top_db=-3.0):
    """Logarithmic decrement on the Hilbert envelope -> Q and its standard error.

    The band-pass has to be much wider than the resonance it isolates, or the
    decay measured is the filter's rather than the resonator's. The default
    relative bandwidth of 0.20 gives the filter a Q of 5; `q_ratio` below
    reports how far the measurement sits from that limit.

    The fit window is a single contiguous stretch running from `top_db` below
    the peak down to whichever comes first: `floor_db`, or six times the noise
    floor measured on the quiet pre-roll. Stopping at the noise floor matters
    -- an envelope that has flattened into noise biases Q upwards without
    ever looking wrong on a linear plot.
    """
    nyq = 0.5 * rate
    lo = f0 * (1.0 - rel_bandwidth / 2) / nyq
    hi = f0 * (1.0 + rel_bandwidth / 2) / nyq
    # Second-order sections, not transfer-function form: at 44.1 kHz the band
    # is ~0.009 in normalised frequency, where a 4th-order (b, a) Butterworth
    # is numerically unstable and the "envelope" diverges instead of decaying.
    sos = butter(order, [max(lo, 1e-4), min(hi, 0.99)], btype="band", output="sos")
    narrow = sosfiltfilt(sos, sig)               # whole record: clean edges

    env = np.abs(hilbert(narrow))
    noise = (float(np.median(env[:i_quiet])) if i_quiet > 100
             else float(np.percentile(env, 5)))

    seg = env[i_start:]
    t = np.arange(len(seg)) / rate
    e_max = float(seg.max())
    e_top = 10 ** (top_db / 20) * e_max
    e_low = max(10 ** (floor_db / 20) * e_max, 6.0 * noise)

    below_top = np.where(seg <= e_top)[0]
    ia = int(below_top[0]) if below_top.size else 0
    under = np.where(seg[ia:] < e_low)[0]        # first crossing, not the last
    ib = ia + int(under[0]) if under.size else len(seg)
    if ib - ia < 20:
        ib = min(ia + max(20, int(0.05 * rate)), len(seg))

    t_fit, e_fit = t[ia:ib], np.log(seg[ia:ib] + 1e-30)
    slope, intercept = np.polyfit(t_fit, e_fit, 1)
    decay = -float(slope)                        # 1 / tau, in s^-1
    resid = e_fit - (slope * t_fit + intercept)
    dof = max(len(t_fit) - 2, 1)
    var_t = float(np.sum((t_fit - t_fit.mean()) ** 2))
    se_slope = float(np.sqrt(np.sum(resid ** 2) / dof / max(var_t, 1e-30)))
    r2 = 1.0 - float(np.var(resid) / max(np.var(e_fit), 1e-30))

    q = np.pi * f0 / decay if decay > 0 else np.nan
    q_se = q * se_slope / decay if decay > 0 else np.nan
    q_filter = 1.0 / rel_bandwidth               # Q of the band-pass itself
    span_db = 20.0 * np.log10(seg[ia] / max(seg[ib - 1], 1e-30))

    return {"Q": float(q), "Q_se": float(q_se),
            "tau": 1.0 / decay if decay > 0 else np.nan, "decay": decay,
            "r2": r2, "n_fit": int(ib - ia), "span_db": float(span_db),
            "snr_db": float(20 * np.log10(e_max / max(noise, 1e-30))),
            "Q_filter": q_filter, "q_ratio": float(q / q_filter),
            "t_env": t, "env": seg, "narrow": narrow[i_start:],
            "t_fit": t_fit, "fit_line": np.exp(slope * t_fit + intercept)}


# --------------------------------------------------------------------------
# One recording
# --------------------------------------------------------------------------
def analyse(sig, rate, pred, label, rel_bandwidth=0.20, plot=True):
    f_guess = pred["f0"] if pred else 200.0
    i0, i1, i_quiet = find_decay(sig, rate, f_guess=f_guess)

    f0, freq, spec = spectral_peak(sig[i0:i1], rate)
    qres = measure_q(sig, rate, f0, i0, i_quiet, rel_bandwidth=rel_bandwidth)

    result = {"label": label, "rate": rate, "f0_measured": f0,
              "Q_measured": qres["Q"], "Q_se": qres["Q_se"], "tau": qres["tau"],
              "r2": qres["r2"], "n_fit": qres["n_fit"], "q_ratio": qres["q_ratio"],
              "span_db": qres["span_db"], "snr_db": qres["snr_db"],
              "t_start": i0 / rate, "t_end": i1 / rate}

    if plot:
        _figure(sig, rate, i0, i1, freq, spec, f0, qres, pred, label)
    return result, qres


def _figure(sig, rate, i0, i1, freq, spec, f0, qres, pred, label):
    os.makedirs("plots", exist_ok=True)
    t = np.arange(len(sig)) / rate
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))

    ax[0].plot(t, sig, lw=0.5, color="0.55")
    ax[0].plot(t[i0:i1], sig[i0:i1], lw=0.5, color="tab:blue")
    ax[0].axvspan(i0 / rate, i1 / rate, color="tab:blue", alpha=0.10)
    ax[0].set_xlabel("time (s)")
    ax[0].set_ylabel("amplitude")
    ax[0].set_title("waveform: impulse, then free decay")

    band = (freq > 0.3 * f0) & (freq < 4.0 * f0)
    ax[1].semilogy(freq[band], spec[band] + 1e-12, lw=0.8)
    ax[1].axvline(f0, color="tab:red", ls="--", lw=1, label=f"measured {f0:.1f} Hz")
    if pred:
        ax[1].axvline(pred["f0"], color="k", ls=":", lw=1,
                      label=f"lumped {pred['f0']:.1f} Hz")
    ax[1].set_xlabel("frequency (Hz)")
    ax[1].set_ylabel("|P| (a.u.)")
    ax[1].set_title("spectrum of the decay")
    ax[1].legend(fontsize=8)

    ax[2].semilogy(qres["t_env"], qres["env"] + 1e-12, lw=0.7, color="0.55",
                   label="Hilbert envelope")
    ax[2].semilogy(qres["t_fit"], qres["fit_line"], "r-", lw=1.6,
                   label="fit: Q = {:.1f} $\\pm$ {:.1f}".format(qres["Q"], qres["Q_se"]))
    ax[2].set_xlabel("time after onset (s)")
    ax[2].set_ylabel("envelope")
    ax[2].set_title("logarithmic decrement ($R^2$ = {:.3f})".format(qres["r2"]))
    ax[2].legend(fontsize=8)

    for a in ax:
        a.grid(ls=":", alpha=0.6)
    fig.suptitle("Helmholtz ring-down -- " + label, fontsize=11)
    fig.tight_layout()
    path = "plots/experiment_{}.png".format(label)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print("  figure -> " + path)


# --------------------------------------------------------------------------
# Self-test: synthesise a known decay, check the estimator recovers it
# --------------------------------------------------------------------------
def _synth(f0_true, q_true, rate, duration, snr_db, seed):
    """A damped sinusoid behind a broadband slap, plus a fast high partial."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration * rate)) / rate
    t0 = 0.10                                    # impulse at 100 ms
    tau = q_true / (np.pi * f0_true)

    sig = np.zeros_like(t)
    live = t >= t0
    sig[live] = np.exp(-(t[live] - t0) / tau) * np.sin(2 * np.pi * f0_true * (t[live] - t0))
    sig += 0.8 * np.exp(-((t - t0) / 1.5e-3) ** 2) * rng.normal(0.0, 1.0, t.size)
    sig[live] += 0.25 * np.exp(-(t[live] - t0) / 0.010) * np.sin(2 * np.pi * 5.75 * f0_true * (t[live] - t0))
    sig += rng.normal(0.0, 10 ** (-snr_db / 20), t.size)
    return sig


def self_test(rate=44100.0, duration=2.5):
    """Verify both halves of the script before any bottle is recorded.

    Part 1 runs the whole estimator on synthetic decays whose f0 and Q are
    known by construction, across the range a real bottle can produce.
    Part 2 checks the lumped predictions against this project's own reference
    geometry, for which the two FDM solvers give an independent answer.
    """
    print("Part 1 -- estimator on synthetic ring-downs of known f0 and Q")
    print("  {:>9}{:>8}{:>11}{:>9}{:>9}{:>9}{:>8}".format(
        "f0 (Hz)", "Q", "SNR (dB)", "f0 est", "Q est", "err f0", "err Q"))
    cases = [(205.0, 42.0, 40.0), (205.0, 42.0, 25.0), (128.0, 25.0, 35.0),
             (330.0, 70.0, 45.0), (480.0, 55.0, 30.0)]
    worst_f, worst_q = 0.0, 0.0
    for i, (f_true, q_true, snr) in enumerate(cases):
        sig = _synth(f_true, q_true, rate, duration, snr, seed=i)
        res, _ = analyse(sig, rate, None, "selftest", plot=(i == 0))
        e_f = abs(res["f0_measured"] - f_true) / f_true * 100
        e_q = abs(res["Q_measured"] - q_true) / q_true * 100
        worst_f, worst_q = max(worst_f, e_f), max(worst_q, e_q)
        print("  {:>9.1f}{:>8.1f}{:>11.0f}{:>9.2f}{:>9.1f}{:>8.3f}%{:>7.1f}%".format(
            f_true, q_true, snr, res["f0_measured"], res["Q_measured"], e_f, e_q))
    ok_est = worst_f < 0.5 and worst_q < 12.0
    print("  worst case: f0 {:.3f} %, Q {:.1f} %  ->  {}".format(
        worst_f, worst_q, "PASS" if ok_est else "FAIL"))

    # ----------------------------------------------------------------------
    # The reference configuration of this project: neck radius 10 mm,
    # neck length 40 mm, cavity radius 40 mm x height 80 mm.
    print()
    print("Part 2 -- lumped predictions on this project's reference geometry")
    volume = np.pi * 0.040 ** 2 * 0.080
    ref = predict(0.010, 0.040, volume, t_celsius=20.0, flush=True)
    # Q_visc scales as sqrt(f0), so the notebook's 47 at 234.6 Hz becomes
    # 47 * sqrt(205.7 / 234.6) = 44.0 at the frequency computed here.
    checks = [("f0 (Hz)", ref["f0"], 205.56, 1.0, "corrected Helmholtz formula"),
              ("Q_rad", ref["Q_rad"], 285.6, 10.0, "harmonic sweep, radiation impedance"),
              ("Q_visc", ref["Q_visc"], 44.0, 10.0, "notebook 1, rescaled from 234.6 Hz")]
    print("  {:>9}{:>12}{:>12}{:>10}   {}".format(
        "quantity", "this script", "reference", "deviation", "reference is"))
    ok_pred = True
    for name, got, ref_val, tol, src in checks:
        dev = abs(got - ref_val) / ref_val * 100
        ok_pred &= dev < tol
        print("  {:>9}{:>12.2f}{:>12.2f}{:>9.2f}%   {}".format(name, got, ref_val, dev, src))
    print("  ->  " + ("PASS" if ok_pred else "FAIL"))

    ok = ok_est and ok_pred
    print()
    print("Self-test passed: estimator and predictions are both verified."
          if ok else "Self-test FAILED.")
    return ok


# --------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(
        description="Measure f0 and Q of a real Helmholtz resonator from ring-down recordings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("wav", nargs="*", help="WAV recordings, one ring-down each")
    p.add_argument("--self-test", action="store_true",
                   help="verify the estimator on a synthetic decay of known f0 and Q")
    p.add_argument("--neck-radius", type=float, default=None, metavar="mm",
                   help="inner radius of the neck, in millimetres")
    p.add_argument("--neck-length", type=float, default=None, metavar="mm",
                   help="length of the neck, in millimetres")
    p.add_argument("--volume", type=float, default=None, metavar="mL",
                   help="cavity volume in millilitres (fill with water and weigh)")
    p.add_argument("--temperature", type=float, default=20.0, metavar="degC",
                   help="ambient temperature")
    p.add_argument("--protruding", action="store_true",
                   help="neck sticks out (unbaffled, dL_outer = 0.60 a) instead of flush (0.85 a)")
    p.add_argument("--rel-bandwidth", type=float, default=0.20,
                   help="band-pass width as a fraction of f0")
    p.add_argument("--tag", default="bottle", help="name used for the output files")
    args = p.parse_args(argv)

    if args.self_test:
        return 0 if self_test() else 1
    if not args.wav:
        p.error("give at least one WAV file, or --self-test")

    pred = None
    if None not in (args.neck_radius, args.neck_length, args.volume):
        pred = predict(args.neck_radius * 1e-3, args.neck_length * 1e-3,
                       args.volume * 1e-6, args.temperature, flush=not args.protruding)
        print("Lumped prediction from the measured geometry")
        print("  air        : T = {:.1f} C -> c = {:.1f} m/s, rho = {:.3f} kg/m3".format(
            pred["T_C"], pred["c"], pred["rho"]))
        print("  end corr.  : dL = {:.2f} + {:.2f} mm = {:.2f} a".format(
            pred["dL_inner"] * 1e3, pred["dL_outer"] * 1e3,
            (pred["L_eff"] - pred["L"]) / pred["a"]))
        print("  f0         : {:.1f} Hz  (without end correction: {:.1f} Hz)".format(
            pred["f0"], pred["f0_uncorrected"]))
        print("  Q          : viscous {:.0f} | radiation {:.0f} | total {:.0f}".format(
            pred["Q_visc"], pred["Q_rad"], pred["Q_total"]))
    else:
        print("No geometry given (--neck-radius/--neck-length/--volume): measuring only.")
    print()

    runs = []
    for path in args.wav:
        if not os.path.exists(path):
            print("!! missing file: " + path)
            continue
        sig, rate = load_wav(path)
        label = "{}_{}".format(args.tag, os.path.splitext(os.path.basename(path))[0])
        print("{}  ({:.2f} s at {:.1f} kHz)".format(path, len(sig) / rate, rate / 1000))
        res, qres = analyse(sig, rate, pred, label, rel_bandwidth=args.rel_bandwidth)
        print("  f0 = {:.2f} Hz | Q = {:.1f} +/- {:.1f} | tau = {:.1f} ms | R2 = {:.3f}".format(
            res["f0_measured"], res["Q_measured"], res["Q_se"],
            res["tau"] * 1e3, res["r2"]))
        if qres["q_ratio"] < 3.0:
            print("  !! Q is close to the band-pass Q ({:.0f}): the filter contributes to the"
                  " decay. Widen --rel-bandwidth.".format(qres["Q_filter"]))
        runs.append(res)

    if not runs:
        print("Nothing analysed.")
        return 1

    f0s = np.array([r["f0_measured"] for r in runs])
    qs = np.array([r["Q_measured"] for r in runs])
    summary = {"n": len(runs), "f0_mean": float(f0s.mean()), "Q_mean": float(qs.mean()),
               "f0_std": float(f0s.std(ddof=1)) if len(runs) > 1 else 0.0,
               "Q_std": float(qs.std(ddof=1)) if len(runs) > 1 else 0.0}

    print("\n{} recording(s)".format(len(runs)))
    print("  f0 measured : {:.2f} +/- {:.2f} Hz".format(summary["f0_mean"], summary["f0_std"]))
    print("  Q  measured : {:.1f} +/- {:.1f}".format(summary["Q_mean"], summary["Q_std"]))
    if pred:
        d_f = (summary["f0_mean"] - pred["f0"]) / pred["f0"] * 100
        d_q = (summary["Q_mean"] - pred["Q_total"]) / pred["Q_total"] * 100
        print("\n  {:<12}{:>12}{:>12}{:>12}".format("quantity", "measured", "predicted", "deviation"))
        print("  {:<12}{:>12.2f}{:>12.2f}{:>11.1f} %".format("f0 (Hz)", summary["f0_mean"], pred["f0"], d_f))
        print("  {:<12}{:>12.1f}{:>12.1f}{:>11.1f} %".format("Q", summary["Q_mean"], pred["Q_total"], d_q))
        summary["f0_deviation_pct"] = d_f
        summary["Q_deviation_pct"] = d_q

    os.makedirs("data", exist_ok=True)
    out = "data/experiment_{}.npz".format(args.tag)
    np.savez(out, runs=json.dumps(runs), summary=json.dumps(summary),
             prediction=json.dumps({k: v for k, v in (pred or {}).items()
                                    if isinstance(v, (int, float, bool))}))
    print("\n-> " + out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
