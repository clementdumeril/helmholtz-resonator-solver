# Helmholtz resonator solver

Two finite-difference solvers I wrote for an open-neck Helmholtz resonator, one in the frequency domain and one in the time domain, both axisymmetric. I verified them with manufactured solutions and grid convergence. They agree with each other to 0.6% and with published measurements to 0.9–2.2%.

![Transient simulation of the open resonator](figures/helmholtz_resonance.gif)

*Time-domain solver on a 1 mm grid. A broadband pulse enters the neck, and once it has passed, the cavity keeps oscillating at its own frequency. No frequency is imposed.*

This started in 2023 as my TIPE, the individual research project of the French classes préparatoires. In 2026 I reopened it and rebuilt it properly. The repository began in August 2026 as a physics-informed neural network experiment, hence its old name (`PINN-Helmholtz`). I removed that part: I couldn't judge a network's output without a reference I trusted, so I wrote and verified that reference instead. The question I ended up asking: can a finite-difference model predict the natural frequency of an open resonator with a stated uncertainty, and where does it stop matching reality?

The reference geometry is a neck of radius 1 cm and length 4 cm on a cylindrical cavity of radius 4 cm and height 8 cm, with air at 20 °C.

The full write-up is [`paper/main.pdf`](paper/main.pdf), a 13-page preprint with the formulation, the verification, the comparison with published measurements and the limitations.

## Results

| Quantity | Value | Source |
|---|---|---|
| f0, time-domain solver, extrapolated to h → 0 | 204.6 Hz, GCI 1.5% | [`verification/grid_convergence.py`](verification/grid_convergence.py) |
| f0, frequency-domain solver, extrapolated to h → 0 | 203.4 Hz, GCI 2.0% | `data/fdm_sweep_h*.npz`, computation in [`docs/details.md`](docs/details.md) |
| Lumped formula with end corrections | 205.6 Hz (0.5% above the time-domain value) | `data/convergence_f0.npz` |
| Selamet et al. (1997), two published resonators | 0.9% and 2.2% off, no fitted parameter | [`notebooks/harmonic_study.ipynb`](notebooks/harmonic_study.ipynb) §5 |
| Indenbom & Pogossian (2023), Erlenmeyer flask | missed by about 11% | [`experiments/literature_validation.py`](experiments/literature_validation.py) |

<details>
<summary>Why there are two GCI values (1.5% and "about 2%")</summary>

1.5% is the grid uncertainty of the time-domain f0 on the reference geometry. "About 2%" is the frequency-domain uncertainty: 2.0% on the reference geometry, and 2.1% on average over the five neck lengths (2 to 6 cm) of the harmonic convergence study.
</details>

![Validation against published measurements](figures/literature_validation.png)

*Left: deviation from published frequencies. Selamet's two resonators have the same cavity volume in two shapes. The lumped formula gives 85.8 Hz for both (−5.7% and +19%), while the 2D solver is within 0.9% and 2.2% of both. That is on the raw 1 mm grid and without the duct-side end correction, two effects that both raise the computed frequency, so the agreement may be partly fortuitous. Right: predicted Q for the flask (about 40) against the published 18–30.*

## What I built

I wrote all the code here. The physics (lumped model, end corrections, baffled-piston impedance), the GCI method, the PML and the measurements come from the literature listed under Credits.

The frequency-domain solver, [`src/harmonic_solver.py`](src/harmonic_solver.py), solves ∇²p + k²p = −F on the (r, z) half-plane. It uses sparse complex assembly, handles rigid walls by folding the ghost node onto the diagonal, and puts a Robin radiation-impedance condition at the mouth. It takes about 0.1 s per frequency on the 1 mm grid.

The time-domain solver, [`src/transient_solver.py`](src/transient_solver.py), solves ∂²p/∂t² = c²∇²p + F with explicit leapfrog and an axisymmetric Laplacian in masked finite volumes. The neck opens onto a meshed exterior half-space with a rigid baffle, and a broadband Ricker pulse excites it.

[`studies/radiation_boundaries.py`](studies/radiation_boundaries.py) puts a sponge layer, a first-order Mur condition and an unsplit second-order PML in that solver so they can be compared on equal terms.

For verification there is [`verification/mms_transient.py`](verification/mms_transient.py), [`verification/grid_convergence.py`](verification/grid_convergence.py), and [`notebooks/harmonic_study.ipynb`](notebooks/harmonic_study.ipynb), a self-contained frequency-domain scheme with its MMS, GCI and Selamet runs. Validation is in [`experiments/literature_validation.py`](experiments/literature_validation.py). I also wrote a self-tested ring-down analysis for a future bottle measurement ([`experiments/ringdown_analysis.py`](experiments/ringdown_analysis.py)) and a training-free test of neural bases against the 1 mm frequency-domain field ([`studies/neural_basis_benchmark.py`](studies/neural_basis_benchmark.py)).

## How it works

The resonator behaves like a mass on a spring. The air plug in the neck oscillates against the compressibility of the air in the cavity, so f0 = (c/2π)·√(S / (V·(L + ΔL))). The end correction ΔL accounts for the air just outside the neck that moves with the plug. The time-domain solver gets it from the meshed exterior, and the frequency-domain solver imposes it through the radiation impedance.

Both solvers work on the meridian plane of the axisymmetric geometry (the axis treatment is in [`docs/details.md`](docs/details.md)).

![Resonant mode from the frequency-domain solver](figures/helmholtz_mode.gif)

*Frequency-domain solution at 209.84 Hz on the 0.5 mm grid. Neck velocity and cavity pressure are in quadrature (+90.2° measured), as you would expect for a mass-spring system.*

## How I checked it

| Step | Question | Evidence |
|---|---|---|
| Code verification | Does the code solve the equations it claims to? | Manufactured solutions: observed order 2.03 (frequency domain) and 1.98 (time domain), 4 grids each |
| Solution verification | Is the grid fine enough? | Richardson extrapolation and GCI on 2 / 1 / 0.5 mm grids; a 0.25 mm control grid moves the extrapolated f0 by 0.2–0.4% |
| Cross-check | Does an independent solver agree? | 203.4 vs 204.6 Hz (0.6%), different discretization and radiation model |
| Validation | Does it match measurements? | Selamet et al. within 0.9% and 2.2%; one flask missed by 11% |

The manufactured solution turned up a first-order bias. Matched to the nominal wall position, the transient test gave order 1.00 instead of 2. The masked finite-volume scheme puts a rigid wall half a cell beyond the last node, so the simulated geometry is really R + h/2. Matched to R + h/2, the same test gives order 1.98 and errors 27 times smaller at h = 2 mm. So the scheme is second order and the O(h) error comes from the wall position. After that I only compared frequencies extrapolated to h → 0.

![Transient MMS](figures/mms_transient.png)

Extrapolating also fixed the end correction: 1.32a from the raw 1 mm run, 1.56a after extrapolation, between the semi-empirical 1.51a (+3%) and Ingard's 1.43a (+9%).

![Grid convergence of the time-domain f0](figures/convergence_f0.png)

### Radiation damping and the outer boundary

With the original sponge layer, the radiation Q of the time-domain solver varied by a factor of six depending on purely numerical settings. Measuring each outer boundary against a domain four times larger showed why:

| Outer boundary | Spurious reflection | Q drift, depth 10 → 18 cm, radius fixed at 12 cm |
|---|---|---|
| sponge layer | 44% | +26.8% |
| first-order Mur | 5.6% | +33.5% |
| PML | 4.0% | −1.8% |

The PML drift only tests the flat boundary, where the PML is nearly exact. Moving the curved boundary out as well raises Q by about 20%, and growing the whole exterior takes it from 98 to 270, toward the 286 of the frequency-domain solver. So the time-domain radiation Q is not converged. Across the three treatments f0 moves by 0.46%, which I add to the time-domain uncertainty. More in [`docs/details.md`](docs/details.md).

### Neural bases (a negative result)

Without training anything, I projected the 1 mm field onto several neural bases by least squares and measured the PDE residual of each projection. tanh, SIREN, Gabor atoms and domain decomposition all fit the field to better than 0.4%, yet the residual of the fit is 2,500 to 10,000 times that of the zero field, and the fit that minimizes the residual instead misses the field completely. The residual is dominated by the Laplacian of millimeter-scale error, about the grid step, so this says as much about fitting a discrete solution as about the bases (the full analysis is in the report). A Trefftz basis of Bessel functions solves the homogeneous equation exactly, so its residual equals the zero field's, but it cannot carry the source and only fits the field to 4%. No network is trained in this benchmark.

![Neural basis benchmark](figures/bench_bases.png)

## Limitations

Both solvers are lossless. The dominant loss, viscothermal friction in the neck, is only estimated analytically, so Q is overpredicted (about 40 for the flask against 18–30 published).

Staircase walls on a Cartesian (r, z) grid bring the observed order on real geometries down to about 0.9, which is why I extrapolate.

Cavities are cylinders. Modeled as a cylinder of equal volume, the flask stays at 387–396 Hz against 351 Hz measured for every aspect ratio I tried. Its conical shoulder is the likely cause, but I haven't demonstrated that.

The PML has a 4% reflection floor on the curved boundary, and the largest exterior run (Q = 270 against 286 analytic) does not demonstrate convergence.

I have no measurement of my own yet, so validation relies on published data.

## Reproduce

`pip install -r requirements.txt` (PyTorch is only needed for the neural benchmark, CPU build). Times are from my machine.

```bash
python verification/mms_transient.py                  # transient MMS (~15 s)
python experiments/ringdown_analysis.py --self-test   # ~5 s
python studies/frequency_sweep.py                     # 1-1000 Hz harmonic sweep (~77 s)
python studies/neural_basis_benchmark.py              # no training (~1 min)
python experiments/literature_validation.py           # Selamet + flask (~4 min)
PML_SKIP_B=1 PML_SKIP_C=1 python studies/radiation_boundaries.py   # part A only (~3 min)
jupyter lab notebooks/harmonic_study.ipynb            # harmonic MMS, GCI, Selamet
```

The short PML run writes `data/pml_study_partA.npz` and `figures/pml_study_partA.png` and leaves the versioned full results alone; parts B and C take hours. The time-domain grid convergence needs three solver runs first (the 0.5 mm one costs about eight times the 1 mm one):

```bash
OR_TMS=250 OR_H=2   OR_TAG=_h2  python src/transient_solver.py
OR_TMS=250 OR_H=1               python src/transient_solver.py
OR_TMS=250 OR_H=0.5 OR_TAG=_h05 python src/transient_solver.py
python verification/grid_convergence.py
```

The notebooks are committed with their outputs.

## Credits

Clément Dumeril ([clement.dumeril.net](https://clement.dumeril.net), [github.com/clementdumeril](https://github.com/clementdumeril)). MIT license, see [LICENSE](LICENSE).

Theory and data: Helmholtz (1860), Rayleigh (1896), Ingard (1953), Morse & Ingard (1968), Mur (1981), Bérenger (1994), Roache (1994, 1998, 2002), Selamet et al. (1997), Moloney (2004), Grote & Sim (2010), Indenbom & Pogossian (2023). Full entries, each checked against its DOI or arXiv record, are in [`paper/references.bib`](paper/references.bib). Built on NumPy, SciPy, Matplotlib and PyTorch.
