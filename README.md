# Helmholtz resonator solver

Two axisymmetric finite-difference solvers I wrote for an open-neck Helmholtz resonator, one in the frequency domain and one in the time domain. I verified them with manufactured solutions and grid convergence. They agree with each other to 0.6%. The frequency-domain model matches two published resonators (Selamet et al. 1997) to 0.9% and 2.2%, and overpredicts an Erlenmeyer flask (Indenbom & Pogossian 2023) by 10–13%.

![Transient simulation of the open resonator](figures/helmholtz_resonance.gif)

*Time-domain solver on a 1 mm grid. A broadband pulse enters the neck and the cavity keeps ringing at its own frequency.*

It started in 2023 as my TIPE (the research project of the French classes préparatoires), and I rebuilt it in 2026. The full write-up is [`paper/main.pdf`](paper/main.pdf), a 13-page preprint.

Reference geometry: a neck of radius 1 cm and length 4 cm on a cylindrical cavity of radius 4 cm and height 8 cm, air at 20 °C.

## Results

| Quantity | Value | Source |
|---|---|---|
| f0, time domain, extrapolated to h → 0 | 204.6 Hz, GCI 1.5% | [`verification/grid_convergence.py`](verification/grid_convergence.py) |
| f0, frequency domain, extrapolated to h → 0 | 203.4 Hz, GCI 2.0% | [`docs/details.md`](docs/details.md) |
| Lumped formula with end corrections | 205.6 Hz | `data/convergence_f0.npz` |
| Selamet et al. ([1997](https://doi.org/10.1121/1.417986)), two resonators | 0.9% and 2.2% off, no fitted parameter | [`notebooks/harmonic_study.ipynb`](notebooks/harmonic_study.ipynb) §5 (Dirichlet-mouth variant, h = 1 mm) |
| Indenbom & Pogossian ([2023](https://doi.org/10.1051/aacus/2023039)), Erlenmeyer flask | 10–13% off (387–396 Hz vs 351 Hz) | [`experiments/literature_validation.py`](experiments/literature_validation.py) |

![Validation against published measurements](figures/literature_validation.png)

*Left: deviation from published frequencies. Right: predicted Q for the flask (about 40) against the published 18–30.*

## What I built

- [`src/harmonic_solver.py`](src/harmonic_solver.py): frequency-domain solver, sparse complex assembly, radiation impedance at the mouth.
- [`src/transient_solver.py`](src/transient_solver.py): time-domain solver, explicit leapfrog, meshed exterior with a rigid baffle.
- [`studies/radiation_boundaries.py`](studies/radiation_boundaries.py): sponge layer, first-order Mur and PML compared in the same solver.
- [`verification/`](verification/): time-domain manufactured solution and grid convergence (the frequency-domain one is in [`notebooks/harmonic_study.ipynb`](notebooks/harmonic_study.ipynb) §3).
- [`experiments/ringdown_analysis.py`](experiments/ringdown_analysis.py): analysis for a future bottle measurement ([protocol](experiments/protocol.md)).

Manufactured solutions give an observed order of 2.03 (frequency domain) and 1.98 (time domain). On real geometries the staircase walls lower it, so I only compare frequencies extrapolated to h → 0.

## Neural bases (a negative result)

The repo began as a PINN experiment. I dropped it because I couldn't judge a network's output without a reference I trusted, so I wrote that reference instead. What is left is a test with no trained network: I projected the 1 mm field onto tanh, SIREN and Gabor bases by least squares. They fit the field to better than 0.4%, but the PDE residual of the fit is 2,500 to 10,000 times that of the zero field. Script: [`studies/neural_basis_benchmark.py`](studies/neural_basis_benchmark.py).

## Limitations

- Both solvers are lossless, so Q is overpredicted (about 40 for the flask against 18–30 published).
- Cavities are cylinders. The flask, modeled as a cylinder of equal volume, stays at 387–396 Hz against 351 Hz measured.
- The time-domain radiation Q is not converged, and I have no measurement of my own yet. Details in [`docs/details.md`](docs/details.md).

## Reproduce

`pip install -r requirements.txt` (PyTorch only for the neural benchmark). Times are from my machine.

```bash
python verification/mms_transient.py                  # ~15 s
python experiments/ringdown_analysis.py --self-test   # ~5 s
python studies/frequency_sweep.py                     # 1-1000 Hz sweep, ~77 s
python studies/neural_basis_benchmark.py              # ~1 min
python experiments/literature_validation.py           # ~4 min
jupyter lab notebooks/harmonic_study.ipynb            # harmonic MMS, GCI, Selamet
```

The time-domain grid convergence needs three solver runs first:

```bash
OR_TMS=250 OR_H=2   OR_TAG=_h2  python src/transient_solver.py
OR_TMS=250 OR_H=1               python src/transient_solver.py
OR_TMS=250 OR_H=0.5 OR_TAG=_h05 python src/transient_solver.py
python verification/grid_convergence.py
```

## Credits

Clément Dumeril ([clement.dumeril.net](https://clement.dumeril.net)). MIT license, see [LICENSE](LICENSE).

The physics, the GCI method, the PML and the measurements come from the literature in [`paper/references.bib`](paper/references.bib). Built on NumPy, SciPy, Matplotlib and PyTorch.
