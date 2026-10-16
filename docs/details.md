# Details

Supporting material for the [README](../README.md): the axis treatment, the frequency-domain extrapolation, the outer-boundary study, the validation cases, and secondary results from the harmonic notebook. The full report is [`paper/main.pdf`](../paper/main.pdf).

## Axis treatment

Both solvers work on the meridian plane (r, z). On the axis the term (1/r)·∂p/∂r is 0/0; by L'Hôpital's rule it becomes ∂²p/∂r², so the Laplacian there is 2∂²p/∂r² + ∂²p/∂z², and its radial part is discretized as 4(p₁ − p₀)/h² (mirror symmetry about the axis supplies one factor of two, the limit the other).

## Frequency-domain f0 extrapolated to zero mesh (203.4 Hz, GCI 2.0%)

The three sweeps were run with `studies/frequency_sweep.py` (radiation impedance at the mouth) over 195–220 Hz in 0.05 Hz steps on 2, 1 and 0.5 mm grids, each peak located by a three-point parabola. The extrapolation uses the versioned results:

```python
import numpy as np
f2, f1, f05 = (float(np.load(f"data/fdm_sweep_h{h}.npz")["f_pk"]) for h in ("2.0", "1.0", "0.5"))
p = np.log((f2 - f1) / (f1 - f05)) / np.log(2)    # observed order: 0.94
f_ext = f05 + (f05 - f1) / (2**p - 1)             # 203.39 Hz
gci = 1.25 * abs(f1 - f05) / f05 / (2**p - 1)     # 2.0 %
```

Peaks: 215.74, 209.84 and 206.75 Hz. The time-domain solver extrapolates to 204.60 Hz (`data/convergence_f0.npz`, observed order 0.86, GCI 1.5%), a gap of 0.59%.

Fourth grid (added in October 2026, `verification/control_grid.py`): a 0.25 mm sweep over 203.5–207 Hz puts the peak at 205.18 Hz. The triplet 1 / 0.5 / 0.25 mm gives an observed order of 0.96 and extrapolates to 203.52 Hz with a GCI of 1.0%, 0.06% from the three-grid value. The time-domain extrapolation still rests on three grids; a 0.25 mm time-domain run would take about 8 h on one core.

## Outer-boundary study

Source: [`studies/radiation_boundaries.py`](../studies/radiation_boundaries.py), data in `data/pml_study.npz`.

With the original sponge layer, the radiation Q of the time-domain solver ranged from 44 to 267 depending on layer strength, thickness and domain size (`data/sponge_sensitivity.npz`). The f0 recorded in that study came from an FFT with a 4 Hz bin, so it cannot show whether the frequency moved. I implemented three terminations in one solver and measured each against the same run on a domain four times larger, where no reflection can return within the time window:

- sponge: a damping term σ·∂p/∂t over a layer. It reflects 44% of the incident amplitude, which explains the factor of six on Q.
- first-order Mur: ∂p/∂t + c ∂p/∂n = 0 on the outer ring, exact at normal incidence only. 5.6%.
- PML: the unsplit second-order formulation with two auxiliary fields. 7.1% at the default strength, 4.0% after scanning σ.

The PML reaches 0.093% (−61 dB) on the flat boundary, where the Cartesian complex stretch is exact, but stays at 4.0% (−28 dB) on the curved radial boundary, because I used the Cartesian stretch in (r, z) and left out the 1/r term of a cylindrical PML. Neither a stronger σ nor a thicker layer lowers that floor.

The resonator was then run in two configurations. In part B only the depth changes, from 10 to 18 cm, at a fixed radius of 12 cm: the PML Q moves by −1.8% (sponge +26.8%, Mur +33.5%), and f0 spreads by 0.46% across the three treatments. That stability only concerns the flat boundary, where the PML is nearly exact. In part C the radius grows with the depth (0.8 × depth, at least 12 cm). At 18 cm depth, moving the curved boundary from 12 to 14.4 cm raises the PML Q from 96.4 to 116.6 (+21%), so the curved boundary, which carries the 4% floor, still controls the damping. At 209 Hz the wavelength is 1.64 m, so a 10 cm exterior is 0.06 of a wavelength and the PML sits in the reactive near field of the mouth. Part C:

| Depth | Radius | Depth in wavelengths | Radiation Q | f0 (Hz) |
|---|---|---|---|---|
| 10 cm | 12 cm | 0.061 | 98.1 | 208.32 |
| 18 cm | 14.4 cm | 0.110 | 116.6 | 208.42 |
| 30 cm | 24 cm | 0.183 | 187.2 | 208.53 |
| 45 cm | 36 cm | 0.274 | 269.6 | 208.49 |

Q climbs toward the 285.6 of the frequency-domain solver with the analytic impedance on the 1 mm grid (289.7 and 290.5 on the 0.5 and 0.25 mm grids), while f0 stays within 0.1%. The largest run is consistent with convergence there but does not demonstrate it.

![Outer-boundary treatments](../figures/pml_study.png)

*(a) Probe signal against the large-domain reference (the legend says ×3; the reference domain is in fact four times larger, `ENLARGE = 4`). (b) Error over time, at the default layer strength (PML −23 dB). (c) Radiation Q against exterior size in wavelengths.*

## Validation cases

Source: [`experiments/literature_validation.py`](../experiments/literature_validation.py) and section 5 of [`notebooks/harmonic_study.ipynb`](../notebooks/harmonic_study.ipynb).

Selamet et al. (1997, J. Acoust. Soc. Am. 101(1), 41–51) measured concentric cylindrical resonators as side branches on a duct. At resonance the input impedance of the neck vanishes, so the mouth is a pressure-release (Dirichlet) boundary rather than a radiating one; the notebook models that configuration. Results at h = 1 mm: 93.0 Hz against 91 Hz (2.2%) and 72.7 Hz against 72 Hz (0.9%), with no fitted parameter.

Indenbom & Pogossian (2023, Acta Acustica 7, 51) measured a 100 mL Erlenmeyer flask at 351 Hz where their lumped formula gives 391 Hz. My lumped model reproduces their 391 Hz to 0.1%. The finite-difference solver, with the flask replaced by a cylinder of equal volume, gives 387–396 Hz across every aspect ratio tried, so resolving the field near the neck does not close the gap. The discrepancy corresponds to an effective neck about 54 mm long instead of 43 mm, which would fit a neck flaring into a conical shoulder; my cylindrical cavity cannot represent that.

For damping, the only open-access half-width I found for a resonator with fully stated geometry is in the same paper. It gives Q between 18 and 30, a range because the paper does not state whether the width is taken at half amplitude or half power. I predict about 40. The model is optimistic about damping, as Moloney (2004) also reports. The script's loss estimate for the flask gives Q_visc ≈ 52 for viscothermal friction in the neck and Q_rad ≈ 167 for radiation, so the viscothermal loss is about three times the radiation loss, and neither solver resolves it.

## Other results from the harmonic notebook

- Frequency sweep: 1,000 harmonic solves from 1 to 1000 Hz take 77 s. At a 1 Hz step the peak read 209.95 Hz and 6.62 kPa; at 0.005 Hz it is 209.84 Hz and 7.25 kPa, because the −3 dB bandwidth (0.73 Hz) is narrower than the first step.
- Scaling law over neck length: a power law and the effective-length model f0 = A/√(L + ΔL) are statistically indistinguishable on the extrapolated frequencies (ΔAICc 0.3 with 5 lengths, 1.8 with 12). I keep the physical model: A matches the theoretical 48.25 m^1/2/s, ΔL ≈ 0.67a, and its leave-one-out error is 5 to 10 times smaller.
- End-correction decomposition: the radiation impedance shifts ΔL by 0.823a, against 8a/(3π) = 0.849a (3.1%). This is a consistency check, since that reactance is imposed.
- Losses: viscothermal friction in the neck gives Q ≈ 47; bulk absorption is negligible (Q ~ 10⁶).

Full report, with the formulation, the verification and the comparison with published measurements: [`paper/main.pdf`](../paper/main.pdf).
