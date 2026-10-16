# Experimental protocol: measuring f₀ and Q on a real resonator

The goal is to compare the two numbers this project predicts, a natural frequency and a quality factor, with a measurement made with a bottle and a phone.

The two solvers agree on f₀ to 0.6%. Damping is the open question. Viscothermal friction in the neck (Q ≈ 47) is about three times stronger than radiation, and no solver here resolves the boundary layer that produces it. A real bottle should ring at Q ≈ 40, and only a measurement can check that.

Status: I have not made this measurement. [`literature_validation.py`](literature_validation.py) compares the model with published data in the meantime. The analysis script, [`ringdown_analysis.py`](ringdown_analysis.py), is built and self-tested.

## 0. Check the analysis chain

```bash
python experiments/ringdown_analysis.py --self-test
```

It runs the pipeline on synthetic ring-downs with known f₀ and Q, then checks the lumped predictions against the reference geometry. Both parts must print PASS. Typical output: f₀ recovered to better than 0.01%, Q to better than 5%.

## 1. Equipment

- A rigid glass bottle with a neck as cylindrical as possible. Soft plastic adds wall losses.
- A phone recording at 44.1 kHz or more. Convert M4A to WAV (`ffmpeg -i in.m4a out.wav`).
- A caliper, a kitchen scale and a thermometer (1 °C is 0.17% on f₀).

## 2. Geometry

| Quantity | Symbol | How |
|---|---|---|
| Inner radius of the neck | a | caliper, mean of two perpendicular diameters, halved |
| Length of the neck | L | from the mouth to where the neck starts to flare |
| Cavity volume | V | fill with water to the base of the neck and weigh; 1 g = 1 cm³ |

No parameter is fitted. The end correction is ΔL ≈ 0.66 a on the cavity side plus 0.85 a for a baffled mouth or 0.60 a for a protruding neck (`--protruding`). A typical bottle neck protrudes, so ΔL ≈ 1.26 a.

    f₀     = (c / 2π) · √( S / (V (L + ΔL)) ),        S = π a²
    Q_visc = a / (F_t · δ_v),   δ_v = √(2μ / (ρ ω₀)),   F_t ≈ 1.48
    Q_rad  = c (L + ΔL) / (S f₀)                       (baffled piston, lumped)
    1/Q    = 1/Q_visc + 1/Q_rad

At 20 °C:

| Vessel | a (mm) | L (mm) | V (mL) | f₀ | Q_visc | Q_rad | Q total | decay time |
|---|---|---|---|---|---|---|---|---|
| Wine bottle | 9.5 | 60 | 750 | 125 Hz | 33 | 696 | 31 | 80 ms |
| Beer bottle | 9.0 | 55 | 330 | 186 Hz | 38 | 480 | 35 | 60 ms |
| Wide-mouth jar | 20 | 15 | 500 | 432 Hz | 129 | 25 | 21 | 16 ms |

A narrow neck is limited by viscous friction, a wide one by radiation. A bottle rings longer, so it is the better object to measure.

## 3. Recording

1. Quiet room, microphone 5–10 cm from the mouth, out of any airflow.
2. Start recording half a second before exciting. The script uses that pre-roll as its noise floor.
3. Slap the flat of your palm onto the mouth and pull away, or flick the neck. Don't blow.
4. Record 3–5 seconds. Repeat five times, in separate files.
5. Run:

```bash
python experiments/ringdown_analysis.py rec1.wav rec2.wav rec3.wav rec4.wav rec5.wav \
    --neck-radius 9.5 --neck-length 60 --volume 750 --temperature 21 --protruding
```

## 4. What the script does

f₀ comes from the FFT peak of the free decay, refined by parabolic interpolation. Q comes from a band-pass around f₀, the Hilbert envelope, and a fit of the log-envelope slope between −3 dB and the noise floor (Q = π f₀ τ), the same estimator as the transient solver. It reports the spread across files, writes a figure per recording to `figures/` and a summary to `data/experiment_<tag>.npz`, and warns if the measured Q gets close to the filter's own Q.

## 5. Expected discrepancies

- The neck has a shoulder and a flare, so L is uncertain by a few millimeters. This is usually the largest error on f₀.
- A real mouth is neither fully baffled nor fully free. Running the script both ways brackets it.
- Wall losses, leakage around the hand, and temperature.

## 6. What a good result looks like

Within about 5% on f₀ and a measured Q between 20 and 60, consistent with Moloney (2004), where lumped theory slightly overestimates Q. That would confirm that viscothermal loss, not radiation, governs a real resonator. The script prints measured ± spread against predicted, with the deviation in percent.
