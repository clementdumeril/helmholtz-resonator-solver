# Experimental protocol — measuring f₀ and Q on a real resonator

**Objective.** Confront the numerical predictions of this project — a natural frequency and a
quality factor — with a physical measurement, using a bottle and a phone. This is the one step no
calculation replaces.

**Why it matters here.** The two solvers in this repository agree on the natural frequency to
within 0.6 %, and both cross-check against the corrected Helmholtz formula. Neither settles the
**damping**: the harmonic solver returns Q ≈ 286 from an analytic radiation impedance, the
transient solver returns anything between 44 and 267 depending on how the outer boundary is
treated, and the estimated viscothermal losses of the neck (Q ≈ 47) dominate both. The project
measures a natural frequency well and a damping badly — and only an experiment closes that gap.

The analysis script is [`analyze_recording.py`](../analyze_recording.py), at the repository root.

---

## 0. Verify the analysis chain first

Before recording anything, check that the estimator works:

```bash
python analyze_recording.py --self-test
```

It synthesises ring-downs whose f₀ and Q are known by construction, runs the complete pipeline on
them, and reports the recovery error. It then checks the lumped predictions against this project's
own reference geometry, for which the two FDM solvers give an independent answer. Both parts must
print PASS. Typical output: f₀ recovered to better than 0.01 %, Q to better than 5 %.

## 1. Equipment

- A **rigid** bottle or vessel with as cylindrical a neck as possible. Glass is best; soft plastic
  adds wall losses and shifts Q.
- A phone with a WAV/M4A recording app at a sampling rate of at least 44.1 kHz, or a USB
  microphone. M4A must be converted to WAV before analysis (`ffmpeg -i in.m4a out.wav`).
- A caliper or ruler, and a kitchen scale.
- A thermometer: the speed of sound changes by about 0.6 m/s per °C, which is 0.17 % on f₀.

## 2. Measuring the geometry

| Quantity | Symbol | How |
|---|---|---|
| Inner radius of the neck | a | caliper, mean of two perpendicular diameters, halved |
| Length of the neck | L | from the plane of the mouth to where the neck starts to flare |
| Cavity volume | V | fill with water up to the base of the neck and weigh; 1 g = 1 cm³ |

These three numbers, plus the temperature, determine every prediction. No parameter is fitted.

**End correction.** Air spills out at both ends of the neck and takes part in the oscillating mass,
so the effective length is L + ΔL with

- ΔL_inner ≈ 0.66 a (cavity side, Ingard);
- ΔL_outer ≈ 0.85 a for a mouth **flush** with a surface (baffled), or ≈ 0.60 a for a neck that
  **protrudes** into free space — pass `--protruding` in that case.

A typical bottle neck protrudes, so ΔL ≈ 1.26 a.

**Predictions.**

    f₀     = (c / 2π) · √( S / (V (L + ΔL)) ),        S = π a²
    Q_visc = a / (F_t · δ_v),   δ_v = √(2μ / (ρ ω₀)),   F_t ≈ 1.48
    Q_rad  = c (L + ΔL) / (S f₀)                       (baffled piston, lumped)
    1/Q    = 1/Q_visc + 1/Q_rad

For common vessels this gives, at 20 °C:

| Vessel | a (mm) | L (mm) | V (mL) | f₀ | Q_visc | Q_rad | Q total | decay time |
|---|---|---|---|---|---|---|---|---|
| Wine bottle | 9.5 | 60 | 750 | 125 Hz | 33 | 696 | **31** | 80 ms |
| Beer bottle | 9.0 | 55 | 330 | 186 Hz | 38 | 480 | **35** | 60 ms |
| Wide-mouth jar | 20 | 15 | 500 | 432 Hz | 129 | 25 | **21** | 16 ms |

Note how the balance flips: a narrow neck is limited by viscous friction, a wide one by radiation.
A bottle is the better object — its ring-down lasts long enough to fit cleanly.

## 3. Excitation and recording — the ring-down method

1. Quiet room. Microphone 5–10 cm from the mouth, out of any airflow.
2. **Start recording before exciting**: the script measures its noise floor on the quiet pre-roll,
   and stops the envelope fit where the signal reaches it. Half a second of silence is enough.
3. Excite with an impulse: slap the flat of your palm onto the mouth and pull away immediately, or
   flick the neck sharply and tangentially. **Do not blow** — the airflow adds a frequency shift
   and noise.
4. Record 3–5 seconds. Repeat five times, in separate files, for a statistic.
5. Export as WAV and run:

```bash
python analyze_recording.py rec1.wav rec2.wav rec3.wav rec4.wav rec5.wav \
    --neck-radius 9.5 --neck-length 60 --volume 750 --temperature 21 --protruding
```

## 4. What the script does

- **f₀**: FFT of the free decay, with parabolic interpolation of the log-magnitude to locate the
  peak well below one bin width.
- **Q**: zero-phase Butterworth band-pass around f₀ (second-order sections — in transfer-function
  form a filter this narrow is numerically unstable at 44.1 kHz), then the Hilbert envelope, then a
  least-squares slope on the log-envelope between −3 dB and the noise floor. Q = π f₀ τ.
  This is **the same estimator the transient solver uses**, so the two numbers are directly
  comparable rather than merely similar.
- A standard error on Q from the residuals of that fit, and a spread across the repeated files.
- A three-panel figure per recording — waveform with the analysis window, spectrum, log-envelope
  with the fit — written to `plots/`, and a summary to `data/experiment_<tag>.npz`.

It warns if the measured Q approaches the band-pass filter's own Q, which would mean the decay
being measured is partly the filter's.

## 5. Expected sources of discrepancy — to be discussed, not hidden

- The real geometry is not cylindrical: the neck has a shoulder and a flare, so L is ambiguous at
  the few-millimetre level. This is usually the dominant uncertainty on f₀.
- ΔL_outer: a real mouth is neither perfectly baffled nor perfectly free, so the truth sits between
  0.60 a and 0.85 a. Running the script both ways brackets it.
- Wall losses in thin glass, and leakage around the hand.
- Temperature: measure it, do not assume 20 °C.
- The slap excites more than the Helmholtz mode; the band-pass removes the rest, but a very short
  ring-down leaves little to fit.

## 6. What a successful result looks like

Agreement within about 5 % on f₀, and a measured Q between 20 and 60, would be consistent with the
literature (Moloney 2004, *Quality factors and conductances in Helmholtz resonators*, where the lumped theory slightly overestimates Q). A measured Q in
that range would confirm that the viscothermal estimate of the harmonic study, not the radiation
Q of either solver, is what governs a real resonator.

Report the measurement as: f₀ measured ± spread, Q measured ± spread, against predicted, with the
deviation in percent — which is exactly the table the script prints.
