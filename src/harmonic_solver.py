"""
Axisymmetric harmonic finite-difference solver for the Helmholtz equation.

Solves, on the meridian plane (r, z) of an open-neck Helmholtz resonator,

    d2p/dr2 + (1/r) dp/dr + d2p/dz2 + k^2 p = -F,     k = 2 pi f / c

with a volume source F in the cavity, rigid walls, and a radiation-impedance
condition at the mouth. Three things deserve comment.

THE AXIS.  The term (1/r) dp/dr is 0/0 at r = 0. L'Hopital's rule turns it into
d2p/dr2, so the Laplacian there is 2 d2p/dr2 + d2p/dz2, discretised as
4(p1 - p0)/h^2. Two factors of two meet in that expression: mirror symmetry
about the axis, and the limit itself.

RIGID WALLS.  Neumann is imposed by folding the ghost node onto the diagonal --
the neighbour's coefficient is added to the diagonal instead of appearing
off-diagonal. No extra unknowns and no extra equations. The hidden cost is that
this places the wall half a cell beyond the last node, a first-order geometric
bias; verification/mms_transient.py measures it.

THE MOUTH.  A Robin condition carrying the low-frequency radiation impedance of
a baffled piston,

    Z_r = rho c [ (ka)^2 / 2 + i 8ka / (3 pi) ],      alpha = i omega rho / Z_r

whose reactance encodes the exterior end correction 8a/(3 pi) analytically.

The matrix changes at every frequency -- k^2 sits on the diagonal and alpha
depends on omega -- so a sweep is one complete linear solve per frequency, about
0.1 s at h = 1 mm. That is still three to four orders of magnitude cheaper than
resolving the same band in the time domain.

Used by studies/frequency_sweep.py, studies/neural_basis_benchmark.py and
tools/make_mode_anim.py, which previously each carried their own copy.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

# Air at 20 degC, and the reference geometry of the study.
C, RHO = 343.0, 1.204
R_NECK, R_CAV, L_NECK, H_CAV = 0.01, 0.04, 0.04, 0.08
Z_TOP = L_NECK + H_CAV
SRC_Z, SRC_W, SRC_A = L_NECK + 0.5 * H_CAV, 0.01, 1.0e4


class HarmonicSolver:
    """The geometry is built once; only the frequency-dependent parts are
    reassembled per solve."""

    def __init__(self, h=1e-3, src_amp=SRC_A, src_width=SRC_W, src_z=None,
                 r_neck=R_NECK, l_neck=L_NECK, r_cav=R_CAV, h_cav=H_CAV):
        self.h = h
        self.r_neck, self.l_neck = r_neck, l_neck
        self.r_cav, self.h_cav = r_cav, h_cav
        self.z_top = l_neck + h_cav
        if src_z is None:                       # mid-height of the cavity
            src_z = l_neck + 0.5 * h_cav

        self.nr = int(round(r_cav / h)) + 1
        self.nz = int(round(self.z_top / h)) + 1
        self.r = np.arange(self.nr) * h
        self.z = np.arange(self.nz) * h
        rr, zz = np.meshgrid(self.r, self.z, indexing="ij")
        self.rr, self.zz = rr, zz

        self.fluid = (((zz < l_neck - 1e-12) & (rr <= r_neck + 1e-12))
                      | ((zz >= l_neck - 1e-12) & (rr <= r_cav + 1e-12)))
        self.mouth = self.fluid & (np.abs(zz) < 1e-12) & (rr <= r_neck + 1e-12)
        self.forcing = src_amp * np.exp(
            -(rr ** 2 + (zz - src_z) ** 2) / (2 * src_width ** 2))
        self.n = self.nr * self.nz

    # ------------------------------------------------------------------
    def solve(self, freq):
        """One complete solve at `freq` -> complex field P of shape (nr, nz)."""
        h, nr, nz = self.h, self.nr, self.nz
        w = 2 * np.pi * freq
        k2 = (w / C) ** 2
        ka = w / C * self.r_neck
        zr = RHO * C * (0.5 * ka ** 2 + 1j * (8 / (3 * np.pi)) * ka)   # baffled piston
        alpha = 1j * w * RHO / zr

        fl, mo, forcing = self.fluid, self.mouth, self.forcing
        ih2 = 1.0 / h ** 2

        def idx(i, j):
            return i + j * nr

        rows, cols, dat = [], [], []
        b = np.zeros(self.n, complex)
        for i in range(nr):
            for j in range(nz):
                ci = idx(i, j)
                if not fl[i, j]:
                    rows += [ci]; cols += [ci]; dat += [1.0]
                    continue
                diag = k2 + 0j
                if i == 0:                                   # limiting form on the axis
                    diag += -4 * ih2
                    rows += [ci]; cols += [idx(1, j)]; dat += [4 * ih2]
                else:
                    lc, rc = ih2 * (1 - 0.5 / i), ih2 * (1 + 0.5 / i)
                    diag += -2 * ih2
                    if fl[i - 1, j]:
                        rows += [ci]; cols += [idx(i - 1, j)]; dat += [lc]
                    else:
                        diag += lc                           # ghost node folded in
                    if i + 1 < nr and fl[i + 1, j]:
                        rows += [ci]; cols += [idx(i + 1, j)]; dat += [rc]
                    else:
                        diag += rc
                if mo[i, j]:                                 # Robin condition
                    diag += -2 * ih2 - 2 * alpha / h
                    rows += [ci]; cols += [idx(i, j + 1)]; dat += [2 * ih2]
                else:
                    diag += -2 * ih2
                    for jj in (j - 1, j + 1):
                        if 0 <= jj < nz and fl[i, jj]:
                            rows += [ci]; cols += [idx(i, jj)]; dat += [ih2]
                        else:
                            diag += ih2
                rows += [ci]; cols += [ci]; dat += [diag]
                b[ci] = -forcing[i, j]

        a = sp.coo_matrix((dat, (rows, cols)), shape=(self.n, self.n),
                          dtype=complex).tocsr()
        return spsolve(a, b).reshape((nz, nr)).T

    # ------------------------------------------------------------------
    def cavity_probe_index(self, z_probe=0.08):
        """Index of the on-axis probe at the cavity bottom."""
        return int(round(z_probe / self.h))
