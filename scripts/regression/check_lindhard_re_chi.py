"""Merged regression checks for the CPU-only real static Lindhard calculator.

The checks cover the merged ``lindhard_re_chi.RealLindhardCalculator`` module,
which implements the document convention of
``Lindhard_Re_chi_from_Wannier90_hr.md`` (doc section 7.1):

    chi0_doc(q, 0) = -(1/N_k) sum_{k,n,m} |M_mn|^2 (f_nk - f_m,k+q)
                     * Delta / (Delta^2 + eta^2),
    Delta = eps_nk - eps_m,k+q,

i.e. every component differs from the textbook Lindhard function
``chi0_std = sum (f_nk - f_m,k+q)/(Delta + i eta) |M|^2 / N_k`` by an overall
factor -1 (doc 7.1: ``chi0_doc = -chi0_std``), so the static real part is
non-negative and the q -> 0 limit is the positive compressibility/DOS term.

Checks
------
  (a) direct summation against an independently written reference
      (total / intraband / interband, overlap and scalar vertices, even and
      odd nk),
  (b) overall sign convention: the textbook reference has Re <= 0 while the
      module returns Re >= 0 for every q,
  (c) chi0(q) == chi0(-q) (the static real part is even),
  (d) q-grid alignment: q=0 sits at index (nk//2, nk//2), the returned grids
      equal fftshift(fftfreq(nk)) and label every data pixel exactly,
  (e) 1D chain: Re chi0_doc >= 0 for all q, strictly positive at the smallest
      nonzero q, while the textbook convention gives the opposite sign,
  (f) q -> 0 intraband channel: the df/dE substitution (doc 8.2) turns the
      vanishing 0/0 ratio into the positive DOS/compressibility term, and the
      T = 0 caveat (zero Fermi derivative) is registered,
  (g) finite-eta suppression of the small-q intraband weight (doc 8.3),
  (h) orbital_select projection against a restricted-orbital reference plus
      input validation,
  (i) output contract: keys, real float64 dtype, data == intraband + interband,
  (j) absorbed optional features: output_path HDF5 save (before cropping, read
      back with h5py and through load_susceptibility_from_h5) and q_range
      extension/crop alignment,
  (k) structural checks: no GPU-array or psutil code path in the merged module,
      no flipped Lindhard denominator, the retired modules are gone, and no
      stale reference to the retired module/class name survives in the package
      or in these scripts,
  (l) HDF5 grid convention shared by both susceptibility modules and the
      loader: for odd and even nq the save -> load round trip reproduces the
      fftshift(fftfreq(nq)) grid bit for bit with the data unchanged, the Im
      module writes and reports the same ``imag_Lindhard`` tag, and a legacy
      file that stores no grid is read back onto the corrected grid (q=0 at
      index nq//2),
  (m) vectorized engine against the independent direct summation: the same
      terms summed in a different order must agree to 1e-10 relative (the
      measured deviation is reported; the docstring records why the tolerance is
      not tighter),
  (n) vectorization internals: the (destination, source) view blocks of the
      cyclic k -> k+q shift cover every k exactly once and equal ``np.roll``, and
      the 512-row blocked H(k)/eigh stage is bitwise identical to one unblocked
      call,
  (o) wide-model smoke: the 75-orbital C6LiC6_CWF model at nk=32 (1024 rows,
      the mesh whose unblocked H(k) build segfaults in Accelerate's zgemm)
      completes with the blocked eigen stage, in both the overlap and the scalar
      vertex (skipped when no real model directory is present).

Usage
-----
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<package root> python check_lindhard_re_chi.py
"""

from __future__ import annotations

import importlib
import inspect
import os
import tempfile
import time
from pathlib import Path

import numpy as np

# Keep matplotlib cache warnings out of the regression output.
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

from stm_data_processing.dft.wannier90 import lindhard_re_chi as merged_module  # noqa: E402
from stm_data_processing.dft.wannier90.lindhard_re_chi import (  # noqa: E402
    _HK_ROW_BLOCK,
    RealLindhardCalculator,
    _shifted_ranges,
)
from stm_data_processing.dft.wannier90.mlwf_hamiltonian import (  # noqa: E402
    MLWFHamiltonian,
)
from stm_data_processing.io.susceptibility_io import (  # noqa: E402
    load_susceptibility_from_h5,
)
from stm_data_processing.utils.miscellaneous import fermi  # noqa: E402

_EF = 0.0  # eV
_TEMPERATURE = 100.0  # K
_ETA = 0.05  # eV
_DEGENERACY_TOLERANCE = 1e-12  # eV, matches the module default

_MODULE_PATH = Path(merged_module.__file__).resolve()
_WANNIER90_DIR = _MODULE_PATH.parent
_PACKAGE_DIR = _WANNIER90_DIR.parents[1]
_SCRIPTS_DIR = Path(__file__).resolve().parent

# Real Wannier models used by the wide-model smoke check.  The 75-orbital
# C6LiC6_CWF model (num_wann = 75, 1681 R points) is the case for which an
# unblocked 1024-row H(k) build segfaults inside Accelerate's zgemm; the
# 52-orbital lessorb model is the fallback when only it exists.
_WIDE_MODEL_DIR = Path("/Users/hunfen/Documents/DFT/wannier90/local/C6LiC6_CWF")
_WIDE_MODEL_SEED = "C6LiC6_0"
_LESSORB_MODEL_DIR = Path(
    "/Users/hunfen/Documents/DFT/wannier90/local/C6LiC6_CWF_lessorb"
)

# The retired module/class names and the unused GPU-array library are
# assembled at runtime on purpose: the merge's structural verification greps
# package/ and scripts/ for those identifiers and must report zero hits, so
# this checker cannot spell them out literally.
_RETIRED_MODULE = "bare" + "_lindhard"
_RETIRED_CLASS = "Bare" + "LindhardCalculator"
_GPU_LIB = "cu" + "py"
_PSUTIL = "psu" + "til"


# ----------------------------------------------------------------------
# Small mock tight-binding models (1-2 orbitals)
# ----------------------------------------------------------------------
def build_square_1band(t: float) -> MLWFHamiltonian:
    """2D square-lattice single-band model: H(k) = -2t(cos 2pi k1 + cos 2pi k2)."""
    r_list = np.array(
        [[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]],
        dtype=np.int32,
    )
    h_list_flat = np.array([[0.0], [-t], [-t], [-t], [-t]], dtype=np.complex128)
    ndegen = np.ones(len(r_list), dtype=np.float64)
    return MLWFHamiltonian.from_arrays(1, r_list, h_list_flat, ndegen, np.eye(3))


def build_chain_1band(t: float) -> MLWFHamiltonian:
    """1D nearest-neighbor chain embedded in 2D: H(k) = -2t cos(2pi k1)."""
    r_list = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]], dtype=np.int32)
    h_list_flat = np.array([[0.0], [-t], [-t]], dtype=np.complex128)
    ndegen = np.ones(len(r_list), dtype=np.float64)
    return MLWFHamiltonian.from_arrays(1, r_list, h_list_flat, ndegen, np.eye(3))


def build_two_band_model() -> MLWFHamiltonian:
    """2D two-orbital square-lattice model with k-dependent eigenvectors.

    H(k) = diag(-2(cos 2pi k1 + cos 2pi k2), -(cos 2pi k1 + cos 2pi k2))
           + [[-0.2, 0.3], [0.3, 0.2]].

    The orbital-dependent hopping makes the eigenvector mixing k-dependent, so
    the overlap ``M_mn(k, q)`` is genuinely non-trivial (an on-site mixing with
    identical hopping would leave the eigenvectors k-independent and the
    interband channel identically zero).  Both bands cross the chosen Fermi
    level (0 eV) at 100 K, so the intraband DOS channel is populated.
    """
    r_list = np.array(
        [[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]],
        dtype=np.int32,
    )
    onsite = np.array([[-0.2, 0.3], [0.3, 0.2]], dtype=np.complex128)
    hop = np.array([[-1.0, 0.0], [0.0, -0.5]], dtype=np.complex128)
    h_list_flat = np.stack([onsite, hop, hop, hop, hop]).reshape(5, 4)
    ndegen = np.ones(len(r_list), dtype=np.float64)
    return MLWFHamiltonian.from_arrays(2, r_list, h_list_flat, ndegen, np.eye(3))


# ----------------------------------------------------------------------
# Independent reference implementations
# ----------------------------------------------------------------------
def _fermi_derivative_fd(
    energies: np.ndarray, mu: float, temperature: float, step: float = 1e-5
) -> np.ndarray:
    """Richardson-extrapolated central difference df/dE.

    This is deliberately independent of the closed form ``-f(1-f)/kT`` used by
    the module: it only probes the shared :func:`fermi` function.  With
    ``step = 1e-5`` eV the truncation error is O(step^4) and the round-off
    error is ~1e-11 eV^-1, far below the check tolerances.
    """
    if temperature <= 1e-12:
        return np.zeros_like(energies, dtype=float)

    def central(h: float) -> np.ndarray:
        upper = fermi(energies + h, mu=mu, T=temperature)
        lower = fermi(energies - h, mu=mu, T=temperature)
        return (upper - lower) / (2.0 * h)

    coarse = central(step)
    fine = central(0.5 * step)
    return (4.0 * fine - coarse) / 3.0


def textbook_sum(
    evals: np.ndarray,
    evecs: np.ndarray,
    iq1: int,
    iq2: int,
    eta: float,
    mu: float,
    temperature: float,
    orbitals: np.ndarray | None = None,
    matrix_elements: bool = True,
    regularize: bool = True,
) -> tuple[complex, complex, complex]:
    """Direct sum of the textbook complex Lindhard function (doc 7.1).

    Returns ``(S, S_intra, S_inter)`` with

        S = (1/N_k) sum_{k,n,m} |M_mn|^2 (f_nk - f_m,k+q) / (Delta + i eta),

    where ``M_mn = sum_a U*_{a m}(k+q) U_{a n}(k)`` and
    ``Delta = eps_nk - eps_m,k+q``.  The merged module implements the document
    convention ``Re chi0 = -Re(S)`` (hence ``Re(S) <= 0``).

    With ``regularize=True`` the ``n == m, |Delta| <= tol`` branch follows doc
    8.2 and inserts the finite-difference ``df/dE``; with the textbook sign
    that branch contributes ``+|M|^2 df/dE`` (the doc-convention contribution
    is ``-|M|^2 df/dE``).  ``regularize=False`` keeps the raw ``0/0`` ratio.
    """
    nk1, nk2, nw = evals.shape
    alpha = np.arange(nw) if orbitals is None else np.asarray(orbitals, dtype=int)
    occupations = fermi(evals, mu=mu, T=temperature)
    derivative = _fermi_derivative_fd(evals, mu, temperature)
    total = 0.0j
    intra = 0.0j
    inter = 0.0j
    for i1 in range(nk1):
        for i2 in range(nk2):
            j1 = (i1 + iq1) % nk1
            j2 = (i2 + iq2) % nk2
            for n in range(nw):
                for m in range(nw):
                    delta = evals[i1, i2, n] - evals[j1, j2, m]
                    if matrix_elements:
                        weight = (
                            abs(
                                np.vdot(
                                    evecs[j1, j2, alpha, m], evecs[i1, i2, alpha, n]
                                )
                            )
                            ** 2
                        )
                    else:
                        weight = 1.0
                    if regularize and n == m and abs(delta) <= _DEGENERACY_TOLERANCE:
                        term = weight * derivative[i1, i2, n]
                    else:
                        term = (
                            weight
                            * (occupations[i1, i2, n] - occupations[j1, j2, m])
                            / (delta + 1j * eta)
                        )
                    total += term
                    if n == m:
                        intra += term
                    else:
                        inter += term
    scale = 1.0 / (nk1 * nk2)
    return total * scale, intra * scale, inter * scale


def dos_term(evals: np.ndarray, mu: float, temperature: float) -> float:
    """q -> 0 compressibility/DOS term: -(1/N_k) sum_{k,n} df/dE (doc 8.2).

    Only the k-mesh carries the 1/N_k normalization (doc 9): the band sum is
    inside, so this is not the per-band average.
    """
    derivative = _fermi_derivative_fd(evals, mu, temperature)
    nk1, nk2 = evals.shape[0], evals.shape[1]
    return float(-np.sum(derivative) / (nk1 * nk2))


def displayed_index(iq: int, nk: int) -> int:
    """Map a raw FFT-order q index to its fftshifted (displayed) index."""
    return (iq + nk // 2) % nk


# ----------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------
def check_direct_sum(
    nk: int, orbital_select=None, matrix_elements: bool = True
) -> None:
    """(a) Module output equals the independent direct summation, pixel by pixel."""
    calc = RealLindhardCalculator(build_two_band_model(), nk=nk, eta=_ETA)
    result = calc.calculate(
        chemical_potential=_EF,
        temperature=_TEMPERATURE,
        orbital_select=orbital_select,
        include_matrix_elements=matrix_elements,
    )
    evals, evecs = calc.eigen
    orbitals = None if orbital_select is None else np.asarray(orbital_select, dtype=int)
    max_total = max_intra = max_inter = 0.0
    for iq1 in range(nk):
        for iq2 in range(nk):
            ref, ref_intra, ref_inter = textbook_sum(
                evals,
                evecs,
                iq1,
                iq2,
                _ETA,
                _EF,
                _TEMPERATURE,
                orbitals=orbitals,
                matrix_elements=matrix_elements,
            )
            i1 = displayed_index(iq1, nk)
            i2 = displayed_index(iq2, nk)
            max_total = max(max_total, abs(result["data"][i1, i2] - (-ref.real)))
            max_intra = max(
                max_intra, abs(result["intraband"][i1, i2] - (-ref_intra.real))
            )
            max_inter = max(
                max_inter, abs(result["interband"][i1, i2] - (-ref_inter.real))
            )
    scale = max(1.0, float(np.max(np.abs(result["data"]))))
    label = "scalar" if not matrix_elements else "overlap"
    print(
        f"  [a] nk={nk} ({label}): max |data - (-Re S)| = {max_total:.3e}, "
        f"intra {max_intra:.3e}, inter {max_inter:.3e} (tol {1e-9 * scale:.3e}); "
        f"max |data| = {float(np.max(np.abs(result['data']))):.4e}, "
        f"max |inter| = {float(np.max(np.abs(result['interband']))):.4e}"
    )
    assert max_total < 1e-9 * scale, f"total mismatch: {max_total:.3e}"
    assert max_intra < 1e-9 * scale, f"intraband mismatch: {max_intra:.3e}"
    assert max_inter < 1e-9 * scale, f"interband mismatch: {max_inter:.3e}"
    if matrix_elements and nk >= 8:
        assert float(np.max(np.abs(result["interband"]))) > 1e-4, (
            "interband channel is identically zero: the test model lost its "
            "k-dependent eigenvectors"
        )


def check_sign_convention(nk: int) -> None:
    """(b) Document convention Re >= 0 vs textbook Re <= 0 for every q."""
    for name, ham in (
        ("square_1band", build_square_1band(1.0)),
        ("two_band", build_two_band_model()),
    ):
        calc = RealLindhardCalculator(ham, nk=nk, eta=_ETA)
        result = calc.calculate(chemical_potential=_EF, temperature=_TEMPERATURE)
        evals, evecs = calc.eigen
        min_doc = float(np.min(result["data"]))
        max_textbook = max(
            textbook_sum(evals, evecs, iq1, iq2, _ETA, _EF, _TEMPERATURE)[0].real
            for iq1 in range(nk)
            for iq2 in range(nk)
        )
        print(
            f"  [b] {name} nk={nk}: min Re chi0_doc = {min_doc:.6e} (>= 0), "
            f"max Re chi0_textbook = {max_textbook:.6e} (<= 0)"
        )
        assert min_doc >= -1e-12, f"{name}: doc convention went negative: {min_doc:.3e}"
        assert max_textbook <= 1e-12, f"{name}: textbook sign wrong: {max_textbook:.3e}"
        assert float(np.max(result["data"])) > 0.0, f"{name}: response identically zero"


def check_evenness(nk: int) -> None:
    """(c) Static real part is even: chi0(q) == chi0(-q)."""
    calc = RealLindhardCalculator(build_two_band_model(), nk=nk, eta=_ETA)
    result = calc.calculate(chemical_potential=_EF, temperature=_TEMPERATURE)
    center = nk // 2
    max_err = 0.0
    for i1 in range(nk):
        for i2 in range(nk):
            j1 = (2 * center - i1) % nk
            j2 = (2 * center - i2) % nk
            for key in ("data", "intraband", "interband"):
                max_err = max(max_err, abs(result[key][i1, i2] - result[key][j1, j2]))
    scale = max(1.0, float(np.max(np.abs(result["data"]))))
    print(f"  [c] evenness nk={nk}: max |chi(q) - chi(-q)| = {max_err:.3e}")
    assert max_err < 1e-10 * scale, f"evenness mismatch: {max_err:.3e}"


def check_qgrid_alignment(nk: int) -> None:
    """(d) q=0 at (nk//2, nk//2); grids are fftshift(fftfreq(nk)) and label the data."""
    calc = RealLindhardCalculator(build_square_1band(1.0), nk=nk, eta=_ETA)
    result = calc.calculate(chemical_potential=_EF, temperature=_TEMPERATURE)
    data = result["data"]
    q1 = result["q1_grid"]
    q2 = result["q2_grid"]

    # The retired labeling applied fftshift on top of an already centered
    # linspace grid.  For even nk that moves q=0 to index 0; for odd nk the
    # linspace grid is already offset by half a step, so the shifted grid does
    # not contain q=0 at all.
    linspace_vals = np.linspace(-0.5, 0.5, nk, endpoint=False)
    old_vals = np.fft.fftshift(linspace_vals)
    old_q0 = int(np.argmin(np.abs(old_vals)))
    q_values = np.fft.fftshift(np.fft.fftfreq(nk))
    center = nk // 2
    if nk % 2 == 0:
        assert old_q0 == 0 and abs(old_vals[old_q0]) < 1e-15
        old_note = f"old q=0 at index {old_q0} (buggy)"
    else:
        assert np.min(np.abs(old_vals)) > 1e-3, (
            "odd-nk linspace grid unexpectedly contains q=0"
        )
        old_note = (
            f"old grid has no q=0 (half-step offset, nearest index {old_q0} "
            f"at {old_vals[old_q0]:+.3f})"
        )

    np.testing.assert_allclose(q1[:, 0], q_values, atol=1e-15)
    np.testing.assert_allclose(q2[0, :], q_values, atol=1e-15)
    np.testing.assert_allclose(q1, np.broadcast_to(q_values[:, None], (nk, nk)))
    assert q_values[center] == 0.0
    assert q1[center, center] == 0.0 and q2[center, center] == 0.0

    # Every displayed pixel must carry the value of the transfer q its grid
    # entry names, evaluated with the independent direct summation.
    evals, evecs = calc.eigen
    max_err = 0.0
    for i1 in range(nk):
        for i2 in range(nk):
            iq1 = (i1 - center) % nk
            iq2 = (i2 - center) % nk
            ref = textbook_sum(evals, evecs, iq1, iq2, _ETA, _EF, _TEMPERATURE)[0]
            max_err = max(max_err, abs(data[i1, i2] - (-ref.real)))
            assert abs(q1[i1, i2] - q_values[i1]) < 1e-15
            assert abs(q2[i1, i2] - q_values[i2]) < 1e-15
    scale = max(1.0, float(np.max(np.abs(data))))
    print(
        f"  [d] nk={nk}: {old_note}, new q=0 at {center}; "
        f"grid == fftshift(fftfreq); max pixel label error = {max_err:.3e}"
    )
    assert max_err < 1e-9 * scale, f"pixel misalignment: {max_err:.3e}"


def check_chain_sign(nk: int) -> None:
    """(e) 1D chain: Re chi0_doc >= 0 everywhere, opposite to the textbook sign."""
    calc = RealLindhardCalculator(build_chain_1band(1.0), nk=nk, eta=_ETA)
    result = calc.calculate(chemical_potential=_EF, temperature=_TEMPERATURE)
    center = nk // 2
    along_q1 = result["data"][:, center]
    assert np.all(along_q1 >= -1e-12), f"Re chi0_doc < 0 along q1: {along_q1}"
    smallest_nonzero = float(result["data"][center + 1, center])
    assert smallest_nonzero > 0.0, (
        f"Re chi0_doc at the smallest nonzero q should be > 0, got {smallest_nonzero:.6e}"
    )
    evals, evecs = calc.eigen
    textbook = textbook_sum(evals, evecs, 1, 0, _ETA, _EF, _TEMPERATURE)[0].real
    assert textbook < 0.0, f"textbook convention should be negative, got {textbook:.6e}"
    print(
        f"  [e] nk={nk}: Re chi0_doc(q=(1/{nk}, 0)) = {smallest_nonzero:.6e} (> 0); "
        f"textbook value = {textbook:.6e} (< 0); "
        f"min over q1 = {float(np.min(along_q1)):.3e}"
    )


def check_q_to_zero_dfde(nk: int) -> None:
    """(f) The df/dE substitution produces the positive DOS term at q=0."""
    calc = RealLindhardCalculator(build_two_band_model(), nk=nk, eta=_ETA)
    result = calc.calculate(chemical_potential=_EF, temperature=_TEMPERATURE)
    center = nk // 2
    evals, evecs = calc.eigen
    expected_dos = dos_term(evals, _EF, _TEMPERATURE)
    assert expected_dos > 0.0
    center_value = float(result["data"][center, center])
    center_intra = float(result["intraband"][center, center])
    center_inter = float(result["interband"][center, center])

    # Without the substitution the intraband 0/0 ratio evaluates to 0 and the
    # interband weights vanish by orthonormality, so q=0 would collapse to 0.
    naive = textbook_sum(
        evals, evecs, 0, 0, _ETA, _EF, _TEMPERATURE, regularize=False
    )[0]

    print(
        f"  [f] nk={nk}: chi0_doc(q=0) = {center_value:.6e}, DOS term = "
        f"{expected_dos:.6e}, intra = {center_intra:.6e}, inter = {center_inter:.3e}, "
        f"unregularized q=0 value = {abs(naive):.3e}"
    )
    np.testing.assert_allclose(center_intra, expected_dos, rtol=1e-9)
    assert abs(center_inter) < 1e-12, f"interband at q=0 should vanish: {center_inter:.3e}"
    assert abs(naive) < 1e-14, f"the unregularized 0/0 ratio should vanish: {abs(naive):.3e}"
    assert center_value > 1e-3, "df/dE substitution did not fire at q=0"

    # Documented caveat: at T = 0 the Fermi derivative is zero (a step function
    # has no finite slope), so the intraband replacement contributes nothing.
    cold = calc.calculate(chemical_potential=_EF, temperature=0.0)
    assert np.all(np.isfinite(cold["data"]))
    assert abs(cold["data"][center, center]) < 1e-14, (
        "T=0 caveat changed: expected a vanishing intraband replacement at q=0, "
        f"got {cold['data'][center, center]:.3e}"
    )
    print(
        f"      T=0 caveat: chi0_doc(q=0) = {float(cold['data'][center, center]):.3e} "
        "(zero Fermi derivative, as documented)"
    )


def check_eta_suppression(nk: int) -> None:
    """(g) Doc 8.3: the small-q intraband weight is suppressed by finite eta."""
    ham = build_two_band_model()
    small_eta = RealLindhardCalculator(ham, nk=nk, eta=1e-6)
    large_eta = RealLindhardCalculator(ham, nk=nk, eta=0.1)
    center = nk // 2
    small = small_eta.calculate(chemical_potential=_EF, temperature=_TEMPERATURE)
    large = large_eta.calculate(chemical_potential=_EF, temperature=_TEMPERATURE)
    q_small = (center + 1, center)
    intra_small = float(small["intraband"][q_small])
    intra_large = float(large["intraband"][q_small])
    print(
        f"  [g] nk={nk}: intraband(smallest q) = {intra_small:.6e} at eta=1e-6 vs "
        f"{intra_large:.6e} at eta=0.1 (ratio {intra_small / intra_large:.3f})"
    )
    assert intra_small > 0.0 and intra_large > 0.0
    assert intra_small > intra_large, (
        "finite eta should suppress the small-q intraband weight"
    )
    assert np.all(small["intraband"] >= -1e-12)
    assert np.all(large["intraband"] >= -1e-12)


def check_orbital_projection(nk: int) -> None:
    """(h) orbital_select projects the local density vertex and is validated."""
    ham = build_two_band_model()
    calc = RealLindhardCalculator(ham, nk=nk, eta=_ETA)
    full = calc.calculate(chemical_potential=_EF, temperature=_TEMPERATURE)
    projected = calc.calculate(
        chemical_potential=_EF, temperature=_TEMPERATURE, orbital_select=[0]
    )
    evals, evecs = calc.eigen
    max_err = 0.0
    for iq1 in range(nk):
        for iq2 in range(nk):
            ref = textbook_sum(
                evals, evecs, iq1, iq2, _ETA, _EF, _TEMPERATURE, orbitals=np.array([0])
            )[0]
            i1 = displayed_index(iq1, nk)
            i2 = displayed_index(iq2, nk)
            max_err = max(max_err, abs(projected["data"][i1, i2] - (-ref.real)))
    scale = max(1.0, float(np.max(np.abs(projected["data"]))))
    delta = float(np.max(np.abs(projected["data"] - full["data"])))
    print(
        f"  [h] nk={nk}: projected max error = {max_err:.3e}, "
        f"max |projected - full| = {delta:.3e}"
    )
    assert max_err < 1e-9 * scale, f"projection mismatch: {max_err:.3e}"
    assert delta > 1e-6, "orbital projection had no effect (degenerate model?)"
    assert np.all(projected["data"] >= -1e-12)

    for bad, message in (
        (list(range(ham.num_wann + 1)), "out-of-range"),
        ([], "empty"),
    ):
        try:
            calc.calculate(
                chemical_potential=_EF, temperature=_TEMPERATURE, orbital_select=bad
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"orbital_select ({message}) was not rejected")
    for kwargs, message in (
        ({"eta": 0.0}, "eta = 0"),
        ({"nk": 0}, "nk = 0"),
    ):
        try:
            RealLindhardCalculator(ham, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{message} was not rejected")


def check_output_contract(nk: int) -> None:
    """(i) Keys, real float64 arrays, data == intraband + interband."""
    calc = RealLindhardCalculator(build_two_band_model(), nk=nk, eta=_ETA)
    result = calc.calculate(chemical_potential=_EF, temperature=_TEMPERATURE)
    expected_keys = {
        "data",
        "intraband",
        "interband",
        "q1_grid",
        "q2_grid",
        "qx_grid",
        "qy_grid",
        "metadata",
    }
    assert set(result) == expected_keys, f"unexpected keys: {sorted(result)}"
    for key in (
        "data",
        "intraband",
        "interband",
        "q1_grid",
        "q2_grid",
        "qx_grid",
        "qy_grid",
    ):
        array = result[key]
        assert isinstance(array, np.ndarray), f"{key} is not an ndarray"
        assert array.dtype == np.float64, f"{key} dtype is {array.dtype}"
        assert not np.iscomplexobj(array), f"{key} is complex"
        assert np.all(np.isfinite(array)), f"{key} has non-finite entries"
    assert result["data"].shape == (nk, nk)
    assert result["qx_grid"].shape == (nk, nk)
    np.testing.assert_allclose(
        result["intraband"] + result["interband"],
        result["data"],
        rtol=1e-12,
        atol=1e-15,
    )
    metadata = result["metadata"]
    assert metadata["eta"] == _ETA and metadata["nq"] == nk
    assert metadata["module_type"] == "real_Lindhard"
    assert metadata["temperature"] == _TEMPERATURE
    assert metadata["include_matrix_elements"] is True
    np.testing.assert_array_equal(metadata["orbital_select"], np.arange(2))
    residual = float(
        np.max(np.abs(result["intraband"] + result["interband"] - result["data"]))
    )
    print(
        f"  [i] nk={nk}: keys/dtype/shape OK; max |intra + inter - data| = {residual:.3e}"
    )


def check_h5_and_q_range(nk: int) -> None:
    """(j) output_path HDF5 save (pre-crop) and q_range extension/crop."""
    import h5py

    calc = RealLindhardCalculator(build_two_band_model(), nk=nk, eta=_ETA)
    center = nk // 2
    with tempfile.TemporaryDirectory() as tmp:
        plain_path = str(Path(tmp) / "plain.h5")
        cropped_path = str(Path(tmp) / "cropped.h5")
        plain = calc.calculate(
            chemical_potential=_EF, temperature=_TEMPERATURE, output_path=plain_path
        )
        extended = calc.calculate(
            chemical_potential=_EF,
            temperature=_TEMPERATURE,
            q_range=(-0.75, 0.75),
            output_path=cropped_path,
        )
        simple = calc.calculate(
            chemical_potential=_EF, temperature=_TEMPERATURE, q_range=(-0.25, 0.25)
        )

        # --- HDF5 content ---
        with h5py.File(plain_path, "r") as handle:
            stored = handle["susceptibility"][:]
            attrs = dict(handle.attrs)
            assert "bvecs" in handle, "bvecs dataset missing"
            np.testing.assert_array_equal(stored, plain["data"])
            np.testing.assert_array_equal(handle["bvecs"][:], calc.ham.bvecs)
        assert attrs["module_type"] == "real_Lindhard", attrs["module_type"]
        assert float(attrs["eta"]) == _ETA and int(attrs["nq"]) == nk
        assert float(attrs["temperature"]) == _TEMPERATURE
        assert float(attrs["chemical_potential"]) == _EF
        # The cropped run must still store the uncropped primitive-BZ result.
        with h5py.File(cropped_path, "r") as handle:
            np.testing.assert_array_equal(handle["susceptibility"][:], plain["data"])

        # --- round trip through the loader ---
        loaded = load_susceptibility_from_h5(plain_path)
        np.testing.assert_array_equal(loaded["data"], plain["data"])
        np.testing.assert_allclose(loaded["q1_grid"], plain["q1_grid"], atol=1e-15)
        np.testing.assert_allclose(loaded["q2_grid"], plain["q2_grid"], atol=1e-15)

        # --- simple crop inside the primitive BZ ---
        cropped_n = nk // 2
        assert simple["data"].shape == (cropped_n, cropped_n)
        np.testing.assert_array_equal(
            simple["data"],
            plain["data"][
                center - cropped_n // 2 : center + cropped_n // 2,
                center - cropped_n // 2 : center + cropped_n // 2,
            ],
        )
        assert simple["q1_grid"].shape == simple["data"].shape
        np.testing.assert_allclose(
            simple["intraband"] + simple["interband"],
            simple["data"],
            rtol=1e-12,
            atol=1e-15,
        )

        # --- periodic extension beyond the primitive BZ ---
        extended_n = extended["data"].shape[0]
        assert extended_n > nk, f"q_range extension did not enlarge the grid: {extended_n}"
        q_values = extended["q1_grid"][:, 0]
        assert q_values[0] >= -0.75 and q_values[-1] < 0.75
        np.testing.assert_allclose(np.diff(q_values), 1.0 / nk, atol=1e-12)
        selection = [(int(np.round(qv * nk)) + center) % nk for qv in q_values]
        expected = plain["data"][np.ix_(selection, selection)]
        np.testing.assert_array_equal(extended["data"], expected)
        np.testing.assert_allclose(
            extended["intraband"] + extended["interband"],
            extended["data"],
            rtol=1e-12,
            atol=1e-15,
        )
        # --- bvecs=None: real-space grids degrade to None, HDF5 omits bvecs ---
        no_bvecs_ham = MLWFHamiltonian.from_arrays(
            1,
            np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]], dtype=np.int32),
            np.array([[0.0], [-1.0], [-1.0]], dtype=np.complex128),
            np.ones(3),
            None,
        )
        no_bvecs_path = str(Path(tmp) / "no_bvecs.h5")
        no_bvecs = RealLindhardCalculator(no_bvecs_ham, nk=nk, eta=_ETA).calculate(
            chemical_potential=_EF,
            temperature=_TEMPERATURE,
            output_path=no_bvecs_path,
        )
        assert no_bvecs["qx_grid"] is None and no_bvecs["qy_grid"] is None
        with h5py.File(no_bvecs_path, "r") as handle:
            assert "bvecs" not in handle, "bvecs was written although none was given"

        print(
            f"crop (-0.25, 0.25) -> {simple['data'].shape}; "
            f"extension (-0.75, 0.75) -> {extended['data'].shape} equals the "
            "periodic tiling of the primitive cell"
        )


def check_h5_load_grid_parity(nk: int) -> None:
    """(l) save -> load round trip of the fftshifted grid, odd and even nq.

    The stored data is fftshifted (q=0 at index nq//2), so the loader must
    rebuild the grid as ``fftshift(fftfreq(nq))``.  The retired
    ``linspace(-0.5, 0.5, nq, endpoint=False)`` grid coincides with it only for
    even nq and is off by half a cell for odd nq.
    """
    calc = RealLindhardCalculator(build_two_band_model(), nk=nk, eta=_ETA)
    center = nk // 2
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / f"re_nk{nk}.h5")
        result = calc.calculate(
            chemical_potential=_EF, temperature=_TEMPERATURE, output_path=path
        )
        loaded = load_susceptibility_from_h5(path)

    q_values = np.fft.fftshift(np.fft.fftfreq(nk))
    # Bit-for-bit: the loader and the calculator build the same FFT grid.
    np.testing.assert_array_equal(loaded["q1_grid"], result["q1_grid"])
    np.testing.assert_array_equal(loaded["q2_grid"], result["q2_grid"])
    np.testing.assert_array_equal(loaded["q1_grid"][:, 0], q_values)
    np.testing.assert_array_equal(loaded["q2_grid"][0, :], q_values)
    np.testing.assert_array_equal(loaded["data"], result["data"])
    assert loaded["q1_grid"][center, center] == 0.0
    assert loaded["q2_grid"][center, center] == 0.0

    # How far the retired linspace labelling would have been from the truth.
    # The loader used a plain (unshifted, ascending) linspace grid, which for
    # even nq is the same ascending grid as fftshift(fftfreq(nq)) and for odd
    # nq is offset by half a cell.
    retired_vals = np.linspace(-0.5, 0.5, nk, endpoint=False)
    max_offset = float(np.max(np.abs(q_values - retired_vals)))
    if nk % 2 == 0:
        assert max_offset == 0.0, (
            f"even nq={nk}: the retired linspace grid should coincide with "
            f"fftshift(fftfreq), got offset {max_offset:.3e}"
        )
        parity_note = "even nq: retired linspace grid coincidentally identical"
    else:
        assert abs(max_offset - 0.5 / nk) < 1e-12, (
            f"odd nq={nk}: expected a half-cell offset {0.5 / nk:.3e} for the "
            f"retired linspace grid, got {max_offset:.3e}"
        )
        parity_note = f"odd nq: retired linspace grid off by half a cell ({max_offset:.3f})"
    print(
        f"  [l] nk={nk} ({parity_note}): save->load grids and data bit-identical; "
        f"q=0 at index {center}"
    )


def check_im_module_type(nk: int) -> None:
    """(l) Im module: one shared ``imag_Lindhard`` tag in file and metadata."""
    from stm_data_processing.dft.wannier90.mlwf_im_susceptibility import (
        SusceptibilityCalculator_wang2012,
    )

    import h5py

    calc = SusceptibilityCalculator_wang2012(build_two_band_model(), nk=nk, eta=_ETA)
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / f"im_nk{nk}.h5")
        result = calc.calculate(
            omega_limit=0.1, resolution=0.05, q_range=None, output_path=path
        )
        with h5py.File(path, "r") as handle:
            stored_tag = str(handle.attrs["module_type"])
            np.testing.assert_array_equal(handle["susceptibility"][:], result["data"])
            assert int(handle.attrs["nq"]) == nk
        loaded = load_susceptibility_from_h5(path)

    assert stored_tag == result["metadata"]["module_type"] == "imag_Lindhard", (
        f"Im module_type mismatch: file={stored_tag!r}, "
        f"metadata={result['metadata']['module_type']!r}"
    )
    np.testing.assert_array_equal(loaded["data"], result["data"])
    np.testing.assert_array_equal(
        loaded["q1_grid"][:, 0], np.fft.fftshift(np.fft.fftfreq(nk))
    )
    print(
        f"  [l] Im module nk={nk}: file and metadata both tagged {stored_tag!r}; "
        "h5 round trip reproduces the fftshift(fftfreq) grid"
    )


def check_legacy_h5_file(nk: int) -> None:
    """(l) Legacy file (data + attributes only, no stored grid) loads correctly."""
    import h5py

    data = np.arange(nk * nk, dtype=np.float64).reshape(nk, nk)
    center = nk // 2
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / f"legacy_nk{nk}.h5")
        # Files written before the grid convention was documented carry only
        # the susceptibility dataset plus attributes: no grid, no bvecs.
        with h5py.File(path, "w") as handle:
            handle.create_dataset("susceptibility", data=data)
            handle.attrs["module_type"] = "real_Lindhard"
            handle.attrs["eta"] = _ETA
            handle.attrs["nq"] = nk
        loaded = load_susceptibility_from_h5(path)
        extended = load_susceptibility_from_h5(path, q_range=(-1.0, 1.0))

    q_values = np.fft.fftshift(np.fft.fftfreq(nk))
    np.testing.assert_array_equal(loaded["data"], data)
    np.testing.assert_array_equal(loaded["q1_grid"][:, 0], q_values)
    np.testing.assert_array_equal(loaded["q2_grid"][0, :], q_values)
    assert loaded["q1_grid"][center, center] == 0.0
    assert loaded["metadata"]["module_type"] == "real_Lindhard"
    assert loaded["qx_grid"] is None and loaded["qy_grid"] is None
    # The periodic extension must keep the corrected half-cell offset and the
    # physical step 1/nq of the primitive mesh, and its unshifted (sx = sy = 0)
    # tile must copy the loaded data verbatim.
    extended_q = extended["q1_grid"][:, 0]
    assert extended_q[0] >= -1.0 and extended_q[-1] < 1.0
    np.testing.assert_allclose(np.diff(extended_q), 1.0 / nk, atol=1e-12)
    base_values = set(np.asarray(q_values).tolist())
    tile_rows = [i for i, qv in enumerate(extended_q) if float(qv) in base_values]
    tile_cols = [
        j for j, qv in enumerate(extended["q2_grid"][0, :]) if float(qv) in base_values
    ]
    assert len(tile_rows) == nk and len(tile_cols) == nk, (
        f"extension did not contain the primitive mesh verbatim: "
        f"{len(tile_rows)} x {len(tile_cols)} vs {nk} x {nk}"
    )
    np.testing.assert_array_equal(
        extended["data"][np.ix_(tile_rows, tile_cols)], data
    )
    print(
        f"  [l] legacy h5 nk={nk} (no stored grid): loaded grid == "
        "fftshift(fftfreq); q=0 at index %d; q_range extension consistent"
        % center
    )


def check_vectorized_tolerance() -> None:
    """(m) Quantify the vectorized-engine vs independent direct-sum agreement.

    The vectorized engine accumulates exactly the same terms as
    :func:`textbook_sum` but in a different order: per (q, band block, k chunk)
    instead of per (q, n, m, k).  The two results can therefore differ only by
    float64 summation round-off, which the checks below measure explicitly.

    Tolerance: 1e-10 relative to the largest stored value.  The direct sum
    accumulates a complex partial sum whose real parts nearly cancel (the
    intraband and interband channels have opposite magnitudes), so its round-off
    is amplified relative to the final value; the observed deviation is <=2e-12
    at nk=5 and <=2e-13 at nk>=8, and 1e-10 leaves room for a different BLAS
    reduction order on another machine without hiding a real mismatch.
    """
    worst = (0.0, "")
    for nk, matrix_elements, orbital_select in (
        (5, True, None),
        (6, True, None),
        (8, True, None),
        (16, True, None),
        (8, False, None),
        (16, False, None),
        (6, True, [0]),
    ):
        calc = RealLindhardCalculator(build_two_band_model(), nk=nk, eta=_ETA)
        result = calc.calculate(
            chemical_potential=_EF,
            temperature=_TEMPERATURE,
            orbital_select=orbital_select,
            include_matrix_elements=matrix_elements,
        )
        evals, evecs = calc.eigen
        orbitals = None if orbital_select is None else np.asarray(orbital_select)
        scale = max(
            1.0,
            *(
                float(np.max(np.abs(result[key])))
                for key in ("data", "intraband", "interband")
            ),
        )
        max_err = 0.0
        for iq1 in range(nk):
            for iq2 in range(nk):
                ref, ref_intra, ref_inter = textbook_sum(
                    evals,
                    evecs,
                    iq1,
                    iq2,
                    _ETA,
                    _EF,
                    _TEMPERATURE,
                    orbitals=orbitals,
                    matrix_elements=matrix_elements,
                )
                i1 = displayed_index(iq1, nk)
                i2 = displayed_index(iq2, nk)
                max_err = max(
                    max_err,
                    abs(result["data"][i1, i2] - (-ref.real)),
                    abs(result["intraband"][i1, i2] - (-ref_intra.real)),
                    abs(result["interband"][i1, i2] - (-ref_inter.real)),
                )
        label = (
            f"nk={nk} {'overlap' if matrix_elements else 'scalar'}"
            f"{' orb=[0]' if orbital_select else ''}"
        )
        print(f"    {label}: max |vectorized - direct| = {max_err:.3e} (scale {scale:.3e})")
        if max_err / scale > worst[0]:
            worst = (max_err / scale, label)
        assert max_err < 1e-10 * scale, f"{label}: {max_err:.3e} exceeds 1e-10*{scale:.3e}"
    print(
        f"  [m] vectorized vs direct sum: worst relative deviation "
        f"{worst[0]:.3e} at {worst[1]} (tolerance 1e-10, see docstring)"
    )


def check_shifted_view_blocks() -> None:
    """(n) ``_shifted_ranges`` equals ``np.roll`` and covers every k once.

    The engine reaches k+q through these view blocks instead of copying the
    eigenvector array nk**2 times; a wrong block decomposition would silently
    mix k points (and is cheap to check exhaustively for small meshes).
    """
    for nk in (1, 2, 3, 5, 8):
        probe = np.arange(nk)
        for shift in range(nk):
            covered = np.zeros(nk, dtype=int)
            for dest, src in _shifted_ranges(nk, shift):
                assert dest.stop - dest.start == src.stop - src.start
                covered[dest] += 1
                np.testing.assert_array_equal(probe[src], np.roll(probe, -shift)[dest])
            np.testing.assert_array_equal(covered, np.ones(nk, dtype=int))
    assert _HK_ROW_BLOCK <= 512, (
        f"_HK_ROW_BLOCK={_HK_ROW_BLOCK} leaves the Accelerate zgemm range"
    )
    print(
        "  [n] _shifted_ranges: every k covered exactly once and equal to "
        f"np.roll for nk in (1, 2, 3, 5, 8); _HK_ROW_BLOCK={_HK_ROW_BLOCK} <= 512"
    )


def check_blocked_eigen_equivalence(nk: int) -> None:
    """(n) Blocked H(k)/eigh is bitwise identical to one large call.

    ``_HK_ROW_BLOCK`` exists only to keep the zgemm inside the range Accelerate
    handles, so the band energies and eigenvectors must not change at all.  nk=32
    gives 1024 rows, i.e. two 512-row blocks.
    """
    ham = build_two_band_model()
    nw = ham.num_wann
    calc = RealLindhardCalculator(ham, nk=nk, eta=_ETA)
    evals, evecs = calc.eigen
    k_values = np.linspace(-0.5, 0.5, nk, endpoint=False)
    k1, k2 = np.meshgrid(k_values, k_values, indexing="ij")
    k_points = np.column_stack((k1.ravel(), k2.ravel(), np.zeros(nk * nk)))
    ref_evals, ref_evecs = np.linalg.eigh(ham._hk_cpu(k_points))
    np.testing.assert_array_equal(evals, ref_evals.reshape(nk, nk, nw))
    np.testing.assert_array_equal(evecs, ref_evecs.reshape(nk, nk, nw, nw))
    print(
        f"  [n] nk={nk} ({nk * nk} rows, {-(-nk * nk // _HK_ROW_BLOCK)} blocks): "
        "band energies and eigenvectors bitwise identical to one unblocked call"
    )


def check_wide_model_smoke(nk: int) -> None:
    """(o) 75-orbital model at nk=32: the blocked eigen stage must not crash.

    An unblocked H(k) build with 1024 rows and ``num_wann**2 = 5625`` columns
    segfaults inside Accelerate's zgemm (reproduced: the real C6LiC6_CWF model
    exits with SIGSEGV 139); with ``_HK_ROW_BLOCK``-row blocks the same mesh
    completes.  The run also reports the wall time of the smoke case as a
    non-asserting performance datum.  Skipped with a message when no real model
    directory is available.
    """
    model_dir = None
    for candidate in (_WIDE_MODEL_DIR, _LESSORB_MODEL_DIR):
        if (candidate / f"{_WIDE_MODEL_SEED}_hr.dat").exists():
            model_dir = candidate
            break
    if model_dir is None:
        print(
            f"  [o] wide-model smoke SKIPPED: no real model under "
            f"{_WIDE_MODEL_DIR} or {_LESSORB_MODEL_DIR}"
        )
        return
    ham = MLWFHamiltonian.from_seedname(str(model_dir), _WIDE_MODEL_SEED)
    nw = ham.num_wann
    assert nk * nk >= 1024, "the smoke must reach the rows that used to segfault"

    started = time.perf_counter()
    # Two orbitals keep the smoke cheap; the crash lives in the eigen stage,
    # which always uses the full 75-orbital H(k).
    result = RealLindhardCalculator(ham, nk=nk, eta=_ETA).calculate(
        chemical_potential=_EF,
        temperature=_TEMPERATURE,
        orbital_select=[0, 1],
    )
    matrix_wall = time.perf_counter() - started
    assert result["data"].shape == (nk, nk)
    assert np.all(np.isfinite(result["data"]))
    assert float(np.max(result["data"])) > 0.0

    # Scalar vertex: the eigen stage stores no eigenvectors at all, so the same
    # model runs through the evec-free path (coarse mesh: the blocking and the
    # crash condition are already covered above).
    scalar_calc = RealLindhardCalculator(ham, nk=8, eta=_ETA)
    assert scalar_calc._diagonalize(None)[1] is None, (
        "the scalar vertex must not materialize eigenvectors"
    )
    started = time.perf_counter()
    scalar = scalar_calc.calculate(
        chemical_potential=_EF,
        temperature=_TEMPERATURE,
        include_matrix_elements=False,
    )
    scalar_wall = time.perf_counter() - started
    assert np.all(np.isfinite(scalar["data"]))
    print(
        f"  [o] {model_dir.name} ({nw} orbitals) nk={nk}: blocked H(k) build + "
        f"q-sum finished (orbital_select=[0,1] {matrix_wall:.1f}s; scalar nk=8 "
        f"{scalar_wall:.1f}s stores no eigenvectors), no zgemm crash"
    )


def check_structure() -> None:
    """(k) Structural: no GPU path, features absorbed, retired modules gone."""
    source = _MODULE_PATH.read_text(encoding="utf-8")
    lowered = source.lower()
    for forbidden in (
        "import " + _GPU_LIB,
        _GPU_LIB + " as cp",
        "cp.asarray",
        "cp.zeros",
        "cp.sum",
        "_cuda",
        _PSUTIL,
        "get_backend",
    ):
        assert forbidden not in lowered, f"merged module still contains '{forbidden}'"
    assert "BACKEND" not in source, "merged module still consults the package backend"
    for forbidden in ("eps_n - eps_m", "eps_m - eps_n", "1j * self.eta"):
        assert forbidden not in source, f"old denominator form still present: '{forbidden}'"
    for required in (
        "save_susceptibility_to_h5",
        "extend_qpi",
        "np.fft.fftshift(np.fft.fftfreq",
        "degeneracy_tolerance",
        "Lindhard_Re_chi_from_Wannier90_hr",
        "chi0_doc = -chi0_std",
    ):
        assert required in source, f"merged module is missing '{required}'"

    signature = inspect.signature(RealLindhardCalculator.calculate)
    for name in ("q_range", "output_path", "orbital_select", "include_matrix_elements"):
        assert name in signature.parameters, f"calculate() lost the '{name}' parameter"
    assert signature.parameters["q_range"].default is None
    assert signature.parameters["output_path"].default is None

    retired_file = _WANNIER90_DIR / (_RETIRED_MODULE + ".py")
    retired_check = _SCRIPTS_DIR / ("check_" + _RETIRED_MODULE + ".py")
    assert not retired_file.exists(), f"{retired_file.name} still exists"
    assert not retired_check.exists(), f"{retired_check.name} still exists"
    try:
        importlib.import_module(
            f"stm_data_processing.dft.wannier90.{_RETIRED_MODULE}"
        )
    except ImportError:
        pass
    else:
        raise AssertionError("the retired module is still importable")

    offenders = []
    for directory in (_PACKAGE_DIR, _SCRIPTS_DIR):
        for path in directory.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            text = path.read_text(encoding="utf-8")
            if _RETIRED_MODULE in text or _RETIRED_CLASS in text:
                offenders.append(str(path))
    assert not offenders, f"stale references to the retired module remain: {offenders}"

    print(
        "  [k] merged module: no GPU-array/psutil/backend code path, no old "
        "denominator; q_range + output_path present; retired module and check "
        "script absent; no stale reference anywhere in package/ or scripts/"
    )


def main() -> None:
    """Run every check and summarize."""
    checks = [
        ("(a) direct-sum consistency (overlap, nk=8)", lambda: check_direct_sum(8)),
        (
            "(a) direct-sum consistency (scalar, nk=8)",
            lambda: check_direct_sum(8, matrix_elements=False),
        ),
        ("(a) direct-sum consistency (overlap, odd nk=5)", lambda: check_direct_sum(5)),
        ("(a) direct-sum consistency (projected orbital, nk=6)", lambda: check_direct_sum(6, orbital_select=[0])),
        ("(b) sign convention (doc >= 0, textbook <= 0)", lambda: check_sign_convention(6)),
        ("(c) chi0(q) = chi0(-q) (nk=8)", lambda: check_evenness(8)),
        ("(d) q-grid alignment (even nk=4)", lambda: check_qgrid_alignment(4)),
        ("(d) q-grid alignment (even nk=8)", lambda: check_qgrid_alignment(8)),
        ("(d) q-grid alignment (odd nk=5)", lambda: check_qgrid_alignment(5)),
        ("(e) 1D chain sign (nk=16)", lambda: check_chain_sign(16)),
        ("(f) q -> 0 df/dE DOS term (nk=8)", lambda: check_q_to_zero_dfde(8)),
        ("(g) finite-eta intraband suppression (nk=16)", lambda: check_eta_suppression(16)),
        ("(h) orbital projection + validation (nk=8)", lambda: check_orbital_projection(8)),
        ("(i) output contract (nk=8)", lambda: check_output_contract(8)),
        ("(j) h5 save + q_range crop/extension (nk=8)", lambda: check_h5_and_q_range(8)),
        ("(k) source structure and removals", check_structure),
        ("(l) h5 save->load grid parity (odd nk=5)", lambda: check_h5_load_grid_parity(5)),
        ("(l) h5 save->load grid parity (even nk=8)", lambda: check_h5_load_grid_parity(8)),
        ("(l) Im module_type and h5 round trip (nk=5)", lambda: check_im_module_type(5)),
        ("(l) legacy h5 without stored grid (odd nk=5)", lambda: check_legacy_h5_file(5)),
        ("(l) legacy h5 without stored grid (even nk=8)", lambda: check_legacy_h5_file(8)),
        ("(m) vectorized vs direct-sum tolerance", check_vectorized_tolerance),
        ("(n) shifted view blocks equal np.roll", check_shifted_view_blocks),
        ("(n) blocked eigen stage bitwise equal (nk=32)", lambda: check_blocked_eigen_equivalence(32)),
        ("(o) wide-model nk=32 smoke (blocked zgemm)", lambda: check_wide_model_smoke(32)),
    ]
    failed = []
    for name, fn in checks:
        try:
            fn()
            print(f"[PASS] {name}")
        except AssertionError as exc:
            print(f"[FAIL] {name}: {exc}")
            failed.append(name)
    print()
    if failed:
        print(f"RESULT: FAILED ({len(failed)} check(s) failed): {', '.join(failed)}")
        raise SystemExit(1)
    print("RESULT: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
