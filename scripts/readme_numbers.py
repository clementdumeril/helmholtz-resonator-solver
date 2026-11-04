"""Recompute the numbers quoted in README.md from the committed data/*.npz files.

Usage: python scripts/readme_numbers.py   (numpy only, under a second)
"""
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[1] / "data"


def load(name):
    return np.load(DATA / f"{name}.npz", allow_pickle=True)


mms, mmt = load("mms_results"), load("mms_transient")
print(f"MMS order, frequency domain: {float(mms['order']):.2f} (h = {mms['h'][0]*1e3:g} to {mms['h'][-1]*1e3:g} mm)")
print(f"MMS order, time domain: {mmt['orders_effective'].mean():.2f} "
      f"(wall at R: {mmt['orders_nominal'].mean():.2f}); error ratio at h = 2 mm: "
      f"{mmt['err_nominal'][0] / mmt['err_effective'][0]:.0f}")

t = load("convergence_f0")
print(f"time domain: f0 = {', '.join(f'{f:.2f}' for f in t['f0'])} Hz on h = 2, 1, 0.5 mm -> "
      f"{float(t['f_ext']):.2f} Hz, order {float(t['order']):.2f}, GCI {float(t['gci'])*100:.1f} %")
print(f"lumped formula with 0.85a + 0.66a end correction: {float(t['f_theo']):.2f} Hz")

f2, f1, f05 = (float(load(f"fdm_sweep_h{h}")["f_pk"]) for h in ("2.0", "1.0", "0.5"))
p = np.log((f2 - f1) / (f1 - f05)) / np.log(2)
f_ext = f05 + (f05 - f1) / (2**p - 1)
gci = 1.25 * abs(f1 - f05) / f05 / (2**p - 1)
print(f"frequency domain: f0 = {f2:.2f}, {f1:.2f}, {f05:.2f} Hz -> {f_ext:.2f} Hz, order {p:.2f}, GCI {gci*100:.1f} %")
print(f"gap between the two extrapolations: {abs(float(t['f_ext']) - f_ext) / f_ext * 100:.2f} %")

c = load("control_grid")
o, fe, g = c["f_triplets"][1]
print(f"fourth grid 0.25 mm: peak {c['f_f0'][3]:.2f} Hz; 1/0.5/0.25 mm triplet -> {fe:.2f} Hz (GCI {g*100:.1f} %), "
      f"{abs(fe - f_ext) / f_ext * 100:.2f} % from the three-grid value")

lit = load("literature_validation")
print("Selamet (stored, rounded to 0.1 Hz):",
      ", ".join(f"{fd:.1f} vs {m:.0f} Hz" for fd, m in zip(lit["selamet_fdm"], lit["selamet_measured"])),
      "(notebook section 5 gives 2.2 % and 0.9 % from unrounded values)")
dev = (lit["flask_fdm"] / float(lit["flask_measured"]) - 1) * 100
print(f"flask: {lit['flask_fdm'].min():.1f} to {lit['flask_fdm'].max():.1f} Hz vs {float(lit['flask_measured']):.0f} Hz "
      f"(+{dev.min():.1f} to +{dev.max():.1f} %); Q predicted {float(lit['q_predicted']):.1f}, "
      f"published {lit['q_measured_range'][0]:.1f} to {lit['q_measured_range'][1]:.1f}")

b = load("bench_bases")
nn = slice(0, 5)  # tanh, SIREN x2, Gabor, domain decomposition; the last entry is the Trefftz basis
print(f"neural bases: field error {b['field_error'][nn].min()*100:.2f} to {b['field_error'][nn].max()*100:.2f} %, "
      f"PDE residual {b['residual'][nn].min():,.0f} to {b['residual'][nn].max():,.0f} x that of the zero field")

pml = load("pml_study")
print("outer-boundary reflection:", ", ".join(f"{n} {r*100:.1f} %" for n, r in zip(pml["treatments"], pml["reflection"])),
      f"(PML after sigma scan: {pml['sigma_scan_y'].min()*100:.1f} %)")
