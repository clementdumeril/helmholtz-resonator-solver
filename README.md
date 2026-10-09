# Helmholtz resonator: two finite-difference solvers, verified and checked against published data

Two independent axisymmetric finite-difference solvers (frequency domain and time domain) for an open-neck Helmholtz resonator, verified by manufactured solutions and grid convergence, agreeing with each other to 0.6% and with published measurements to 0.9–2.2%.

![Transient simulation of the open resonator](figures/helmholtz_resonance.gif)

*Time-domain solver, 1 mm grid. A broadband pulse enters the neck; once it has passed, the cavity keeps oscillating at its own frequency. No frequency is imposed.*

This started as my TIPE, the research project of the French classes préparatoires, in 2023. In 2026 I reopened it and rebuilt it properly. The repository began as a physics-informed neural network experiment, hence its old name (`PINN-Helmholtz`), but I could not judge a network's output without a reference I trusted, so I wrote and verified that reference first. The question: can a finite-difference model predict the natural frequency of an open resonator with a stated uncertainty, and where does it stop matching reality?

Reference geometry: neck radius 1 cm, neck length 4 cm, cylindrical cavity of radius 4 cm and height 8 cm, air at 20 °C.

Skills: PDE discretisation, sparse linear algebra (SciPy), explicit time integration, PML, verification & validation (MMS, Richardson/GCI).

## Results

| Quantity | Value | Source |
|---|---|---|
| f0, time-domain solver, extrapolated to h → 0 | 204.6 Hz, GCI 1.5% | [`verification/grid_convergence.py`](verification/grid_convergence.py) |
| f0, frequency-domain solver, extrapolated to h → 0 | 203.4 Hz, GCI 2.0% | `data/fdm_sweep_h*.npz`, computation in [`docs/details.md`](docs/details.md) |
| Lumped formula with end corrections | 205.6 Hz | 0.47% from the time-domain value |
| Selamet et al. (1997), two published resonators | 0.9% and 2.2% off, no fitted parameter | [`notebooks/harmonic_study.ipynb`](notebooks/harmonic_study.ipynb) §5 |
| Indenbom & Pogossian (2023), Erlenmeyer flask | missed by about 11% | [`experiments/literature_validation.py`](experiments/literature_validation.py) |

<details>
<summary>Why there are two GCI values (1.5% and "about 2%")</summary>

1.5% is the grid uncertainty of the time-domain f0 on the reference geometry. "About 2%" is the frequency-domain uncertainty: 2.0% on the reference geometry, and 2.1% on average over the five neck lengths (2 to 6 cm) of the harmonic convergence study.
</details>

![Validation against published measurements](figures/literature_validation.png)

*Left: deviation from published frequencies. Selamet's two resonators have the same cavity volume in two shapes; the lumped formula gives 85.8 Hz for both (−5.7% and +19%), the 2D solver recovers both. Right: predicted Q for the flask (about 40) against the published 18–30.*

## What I built

I wrote all the code here. The physics (lumped model, end corrections, baffled-piston impedance), the GCI method, the PML and the measurements come from the literature in Credits.

- [`src/harmonic_solver.py`](src/harmonic_solver.py): frequency-domain solver for ∇²p + k²p = −F on the (r, z) half-plane. Sparse complex assembly, rigid walls by folding the ghost node onto the diagonal, Robin radiation-impedance condition at the mouth. About 0.1 s per frequency on the 1 mm grid.
- [`src/transient_solver.py`](src/transient_solver.py): time-domain solver for ∂²p/∂t² = c²∇²p + F. Explicit leapfrog, axisymmetric Laplacian in masked finite volumes, the neck open onto a meshed exterior half-space with a rigid baffle, excited by a broadband Ricker pulse.
- [`studies/radiation_boundaries.py`](studies/radiation_boundaries.py): sponge layer, first-order Mur absorbing condition and an unsplit second-order PML in the same solver, so they can be compared on equal terms.
- Verification: [`verification/mms_transient.py`](verification/mms_transient.py), [`verification/grid_convergence.py`](verification/grid_convergence.py), and [`notebooks/harmonic_study.ipynb`](notebooks/harmonic_study.ipynb) (self-contained frequency-domain scheme with its MMS, GCI and Selamet runs).
- Validation: [`experiments/literature_validation.py`](experiments/literature_validation.py), and a self-tested ring-down analysis for a future bottle measurement ([`experiments/ringdown_analysis.py`](experiments/ringdown_analysis.py)).
- [`studies/neural_basis_benchmark.py`](studies/neural_basis_benchmark.py): a training-free test of neural bases against the verified field.

## How it works

The resonator is a mass on a spring: the air plug in the neck oscillates against the compressibility of the air in the cavity, so f0 = (c/2π)·√(S / (V·(L + ΔL))). The end correction ΔL accounts for the air moving with the plug just outside the neck. The time-domain solver computes it from the meshed exterior; the frequency-domain solver imposes it through the radiation impedance.

Both solvers work on the meridian plane. On the axis the term (1/r)·∂p/∂r is 0/0; by L'Hôpital's rule it becomes ∂²p/∂r², so the radial part of the Laplacian there is discretised as 4(p₁ − p₀)/h².

![Resonant mode from the frequency-domain solver](figures/helmholtz_mode.gif)

*Frequency-domain solution near resonance (0.5 mm grid). Neck velocity and cavity pressure are in quadrature (+90.2° measured), as expected for a mass-spring system.*

## How I checked it

| Step | Question | Evidence |
|---|---|---|
| Code verification | Does the code solve the equations it claims to? | Manufactured solutions: observed order 2.03 (frequency domain) and 1.98 (time domain), 4 grids each |
| Solution verification | Is the grid fine enough? | Richardson extrapolation and GCI on 2 / 1 / 0.5 mm grids; a 0.25 mm control grid moves the extrapolated f0 by 0.2–0.4% |
| Cross-check | Does an independent solver agree? | 203.4 vs 204.6 Hz (0.6%), different discretisation and radiation model |
| Validation | Does it match measurements? | Selamet et al. within 0.9% and 2.2%; one flask missed by 11% |

**The manufactured solution found a first-order bias.** Matched to the nominal wall position, the transient test gave order 1.00 instead of 2. The masked finite-volume scheme puts a rigid wall half a cell beyond the last node, so the simulated geometry is R + h/2. Matched to R + h/2, the same test gives order 1.98 and errors 27 times smaller at h = 2 mm. The scheme is second order; the O(h) error comes from the wall position. From then on I only compared frequencies extrapolated to h → 0.

![Transient MMS](figures/mms_transient.png)

Extrapolating also fixed the end correction: 1.29a from the raw 1 mm run, 1.56a after extrapolation (expected 1.51a, so about 3% off).

![Grid convergence of the time-domain f0](figures/convergence_f0.png)

### Radiation damping and the outer boundary

With the original sponge layer, the radiation Q of the time-domain solver varied by a factor of six with purely numerical settings. Measuring each outer boundary against a domain three times larger showed why:

| Outer boundary | Spurious reflection | Q drift, exterior 10 → 18 cm |
|---|---|---|
| sponge layer | 44% | +26.8% |
| first-order Mur | 5.6% | +33.5% |
| PML | 4.0% | −1.8% |

Why the PML floors at 4%, and how Q approaches the analytic value as the exterior grows: [`docs/details.md`](docs/details.md).

### Neural bases: a measured negative result

Without training anything, I fitted the verified field onto each neural basis by least squares and measured the PDE residual of that best fit. tanh, SIREN, Gabor atoms and domain decomposition all represent the field to better than 0.4%, yet leave a residual 2,500 to 10,000 times that of the zero field. The cause is a near-cancellation between ∇²p and −k²p: a 0.15% error on p is a large error on ∇²p. A Trefftz basis of Bessel functions solves the homogeneous equation exactly, so its residual equals the zero field's, but it cannot carry the source and fits the field only to 4%. I did not train a PINN on this problem.

![Neural basis benchmark](figures/bench_bases.png)

## Limitations

- Both solvers are lossless; the dominant viscothermal loss in the neck is only estimated analytically, so Q is overpredicted (about 40 for the flask against 18–30 published).
- Staircase walls on a Cartesian (r, z) grid bring the observed order on real geometries down to about 0.9, hence the extrapolation.
- Cavities are cylinders. Modelled as a cylinder of equal volume, the flask stays at 387–396 Hz against 351 Hz measured for every aspect ratio tried; its conical shoulder is the likely cause, not demonstrated.
- The PML has a 4% reflection floor on the curved boundary, and the largest exterior run (Q = 270 against 286 analytic) does not demonstrate convergence.
- No measurement of my own: validation relies on published data.

## Reproduce

`pip install -r requirements.txt` (PyTorch only for the neural benchmark, CPU build). Times measured on my machine.

```bash
python verification/mms_transient.py                  # transient MMS (~15 s)
python experiments/ringdown_analysis.py --self-test   # ~5 s
python studies/frequency_sweep.py                     # 1-1000 Hz harmonic sweep (~77 s)
python studies/neural_basis_benchmark.py              # no training (~1 min)
python experiments/literature_validation.py           # Selamet + flask (~4 min)
PML_SKIP_B=1 PML_SKIP_C=1 python studies/radiation_boundaries.py   # part A only (~3 min)
jupyter lab notebooks/harmonic_study.ipynb            # harmonic MMS, GCI, Selamet
```

The short PML run rewrites `data/pml_study.npz` and its figure with part A only; parts B and C take hours. The time-domain grid convergence needs three solver runs first (the 0.5 mm one costs about eight times the 1 mm one):

```bash
OR_TMS=250 OR_H=2   OR_TAG=_h2  python src/transient_solver.py
OR_TMS=250 OR_H=1               python src/transient_solver.py
OR_TMS=250 OR_H=0.5 OR_TAG=_h05 python src/transient_solver.py
python verification/grid_convergence.py
```

The notebooks are committed with their outputs.

## Credits

Clément Dumeril ([clement.dumeril.net](https://clement.dumeril.net), [github.com/clementdumeril](https://github.com/clementdumeril)). MIT licence, see [LICENSE](LICENSE).

Theory and data: Helmholtz (1860), Rayleigh (1896), Crandall (1926), Ingard (1953), Morse & Ingard (1968), Bérenger (1994), Roache (1994, 1998), Selamet et al. (1997), Peters et al. (2003), Moloney (2004), Indenbom & Pogossian (2023), and Mur's first-order absorbing boundary condition. Entries in [`paper/references.bib`](paper/references.bib). Built on NumPy, SciPy, Matplotlib and PyTorch.
