# Details

Supporting material for the [README](../README.md). The full report is [`paper/main.pdf`](../paper/main.pdf).

## Axis treatment

Both solvers work on the meridian plane (r, z). On the axis, (1/r)·∂p/∂r is 0/0 and becomes ∂²p/∂r² by L'Hôpital's rule, so the Laplacian there is 2∂²p/∂r² + ∂²p/∂z². Its radial part is discretized as 4(p₁ − p₀)/h².

## Frequency-domain f0 extrapolated to zero mesh (203.4 Hz, GCI 2.0%)

Sweeps with `studies/frequency_sweep.py` over 195–220 Hz in 0.05 Hz steps on 2, 1 and 0.5 mm grids, each peak located by a three-point parabola:

```python
import numpy as np
f2, f1, f05 = (float(np.load(f"data/fdm_sweep_h{h}.npz")["f_pk"]) for h in ("2.0", "1.0", "0.5"))
p = np.log((f2 - f1) / (f1 - f05)) / np.log(2)    # observed order: 0.94
f_ext = f05 + (f05 - f1) / (2**p - 1)             # 203.39 Hz
gci = 1.25 * abs(f1 - f05) / f05 / (2**p - 1)     # 2.0 %
```

Peaks: 215.74, 209.84 and 206.75 Hz. The time-domain solver extrapolates to 204.60 Hz (observed order 0.86, GCI 1.5%), a gap of 0.59%.

A fourth grid at 0.25 mm (`verification/control_grid.py`) puts the peak at 205.18 Hz. The 1 / 0.5 / 0.25 mm triplet extrapolates to 203.52 Hz, 0.06% from the three-grid value. The time-domain extrapolation still rests on three grids.

## Outer-boundary study

Source: [`studies/radiation_boundaries.py`](../studies/radiation_boundaries.py), data in `data/pml_study.npz`.

With the original sponge layer, the time-domain radiation Q ranged from 44 to 267 depending on numerical settings. I measured three terminations against the same run on a domain four times larger:

- sponge layer: 44% reflection, which explains the spread in Q;
- first-order Mur: 5.6%;
- PML (unsplit, second order): 4.0% after tuning σ.

The PML reaches 0.093% on the flat boundary but stays at 4.0% on the curved one, because I used the Cartesian stretch in (r, z) without the 1/r term of a cylindrical PML.

With only the depth changing (10 to 18 cm, radius 12 cm), the PML Q moves by −1.8% and f0 spreads by 0.46% across the three treatments. Growing the whole exterior:

| Depth | Radius | Depth in wavelengths | Radiation Q | f0 (Hz) |
|---|---|---|---|---|
| 10 cm | 12 cm | 0.061 | 98.1 | 208.32 |
| 18 cm | 14.4 cm | 0.110 | 116.6 | 208.42 |
| 30 cm | 24 cm | 0.183 | 187.2 | 208.53 |
| 45 cm | 36 cm | 0.274 | 269.6 | 208.49 |

Q climbs toward the 285.6 of the frequency-domain solver while f0 stays within 0.1%. This is consistent with convergence but does not demonstrate it.

![Outer-boundary treatments](../figures/pml_study.png)

*(a) Probe signal against the large-domain reference (the legend says ×3; the reference domain is four times larger). (b) Error over time. (c) Radiation Q against exterior size in wavelengths.*

## Validation cases

Source: [`experiments/literature_validation.py`](../experiments/literature_validation.py) and section 5 of [`notebooks/harmonic_study.ipynb`](../notebooks/harmonic_study.ipynb).

Selamet et al. (1997) measured cylindrical resonators as side branches on a duct, so the mouth is modeled as a pressure-release boundary. At h = 1 mm: 93.0 Hz against 91 Hz (2.2%) and 72.7 Hz against 72 Hz (0.9%), with no fitted parameter.

Indenbom & Pogossian (2023) measured a 100 mL Erlenmeyer flask at 351 Hz. My solver, with the flask replaced by a cylinder of equal volume, gives 387–396 Hz for every aspect ratio tried. The gap matches an effective neck of about 54 mm instead of 43 mm, which would fit a neck flaring into a conical shoulder that my cylinder can't represent. For damping, the same paper gives Q between 18 and 30 and I predict about 40, which is optimistic, as Moloney (2004) also reports.

## Other results from the harmonic notebook

- A 1–1000 Hz sweep (1,000 solves) takes 77 s. The −3 dB bandwidth is 0.73 Hz, so the peak needs a finer step than 1 Hz.
- Over neck length, the model f0 = A/√(L + ΔL) fits as well as a power law, with A matching the theoretical 48.25 m^1/2/s and ΔL ≈ 0.67a.
- Viscothermal friction in the neck gives Q ≈ 47; bulk absorption is negligible.
