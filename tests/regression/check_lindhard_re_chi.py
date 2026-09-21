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
      vertex (skipped when no real model directory is present),
  (p) q row slices: ``calculate(q_index_range=(start, stop))`` returns the raw
      FFT-order slabs of the full-mesh result bit for bit (np.array_equal) for
      nk in (5, 8, 16) split into >= 2 contiguous segments, with the raw
      ``fftfreq`` grids and the ``q_index_range``/``fft_order`` metadata, and
      rejects every illegal range and the slice + output_path / slice + q_range
      combinations with ValueError,
  (q) slice assembly: ``lindhard_re_chi_parallel.assemble_slices`` reproduces
      ``calculate()`` bit for bit for one segment, three contiguous segments and
      a shuffled segment list, and rejects overlapping, missing or wrongly
      shaped slice lists,
  (r) mirror assembly: the ``chi0(q) = chi0(-q)`` mirror of
      ``lindhard_re_chi_parallel`` stays within 1e-12 of the single-process
      reference for nk in (4, 5, 6, 8, 16) (measured value printed), is
      symmetric under the reflection permutation of the frozen
      ``fftshift(fftfreq)`` grid to the same order, gives the same map whether
      the canonical half is one slice or several, and is off unless asked for,
  (s) parallel execution: a two-worker ``spawn`` run on a tiny synthetic model
      reproduces the single-process product bit for bit, lands the
      ``rows_<start>_<stop>.npz`` + ``.json`` + ``.done`` checkpoint triple,
      re-dispatches exactly the slice whose ``npz`` was deleted or corrupted
      when ``resume=True``, returns non-zero and writes no h5 when a worker
      exits non-zero, and the CLI pins the four BLAS thread variables before
      NumPy is imported (``--dry-run`` exits 0),
  (t) logging contract: configuration echo, BLAS thread advisory (WARNING when a
      thread variable is unset or > 1), one timed ``stage=`` record per stage
      (diagonalize, occupations, q_sum, fftshift, h5_write), throttled progress
      records carrying rows/rate/px per s/elapsed/eta/rss_peak and the
      machine-readable ``lindhard_progress`` extra, and the closing summary with
      the three array digests and max|data-(intra+inter)|; the two modules use
      the standard library only for the peak RSS (no GPU-array, psutil or
      backend code) and ``_HK_ROW_BLOCK`` stays <= 512,
  (u) real-model performance and end-to-end: the 75-orbital C6LiC6_CWF model at
      nk=32 (Li projection) single-process vs two spawned workers, with the wall
      clocks and the parallel efficiency printed as non-asserting performance
      data, plus the bit-for-bit equality of the two HDF5 products, their
      ``max|data-(intra+inter)| <= 1e-13`` and ``min(data) >= -1e-12``, the
      loader round trip and the handover plotting script rendering the file
      (skipped when no real model directory is present).
  (v) broken checkpoints: an empty and a truncated ``rows_*.npz`` are treated as
      missing slices (WARNING with the reader's exception type, no exception
      escaping ``run_parallel(resume=True)``), the affected slice is recomputed,
      the final h5 stays byte-identical to the clean control, and
      ``assemble_from_checkpoints`` refuses a directory holding such a slice,
  (w) checkpoint-resume log file: ``run_parallel(log_file=...)`` really writes
      the parent records to that file (non-empty, one plan/summary record), the
      CLI's own handler is not duplicated, and no public parameter of
      ``run_parallel`` is silently ignored,
  (x) ``q_index_range`` validation: three-entry tuples, length 1/4 sequences,
      string/float/None entries, a 2-character string, a 2-D array and
      non-iterable numbers all raise ValueError, while None and the legal
      two-entry ranges keep the behaviour of the full run,
  (y) worker RSS estimate: ``read_model_shape``/``model_identity`` report the
      real ``(num_wann, nrpts)`` of the model, the estimate is an upper bound of
      the measured peak RSS for nk=8/16/32 in the full and Li projections, and
      the nk=256 full eight-worker plan is printed through ``--dry-run`` together
      with the memory-guard decision (a refusal is only accepted when the
      eigenvector arrays alone already exceed the budget),
  (z) checkpoint signature: a slice produced for another model (different
      ``num_wann`` or different ``bvecs``, including a tampered sidecar) is
      rejected on resume and recomputed instead of being reused silently, and
      the same model still reuses its slices.

Usage
-----
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<package root> python check_lindhard_re_chi.py
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np

# Keep matplotlib cache warnings out of the regression output.
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

from stm_data_processing.dft.wannier90 import (
    lindhard_re_chi as merged_module,
)
from stm_data_processing.dft.wannier90 import (
    lindhard_re_chi_parallel as parallel_module,
)
from stm_data_processing.dft.wannier90.lindhard_re_chi import (
    _HK_ROW_BLOCK,
    RealLindhardCalculator,
    _shifted_ranges,
)
from stm_data_processing.dft.wannier90.lindhard_re_chi_parallel import (
    _memory_guard,
    assemble_from_checkpoints,
    assemble_slices,
    available_memory_bytes,
    estimate_worker_rss_bytes,
    model_identity,
    plan_row_slices,
    read_model_shape,
    run_parallel,
    scan_checkpoints,
)
from stm_data_processing.dft.wannier90.mlwf_hamiltonian import (
    MLWFHamiltonian,
)
from stm_data_processing.io.susceptibility_io import (
    load_susceptibility_from_h5,
)
from stm_data_processing.utils.miscellaneous import fermi

_EF = 0.0  # eV
_TEMPERATURE = 100.0  # K
_ETA = 0.05  # eV
_DEGENERACY_TOLERANCE = 1e-12  # eV, matches the module default

_MODULE_PATH = Path(merged_module.__file__).resolve()
_PARALLEL_MODULE_PATH = Path(parallel_module.__file__).resolve()
_WANNIER90_DIR = _MODULE_PATH.parent
_PACKAGE_DIR = _WANNIER90_DIR.parents[1]
_REGRESSION_DIR = Path(__file__).resolve().parent
_CLI_PATH = _REGRESSION_DIR.parents[1] / "scripts" / "run_lindhard_re_chi_parallel.py"
_PLOT_SCRIPT = (
    _REGRESSION_DIR.parents[1] / "tmp_verify/dl/serverpkg/scripts/plot_lindhard_rechi_cwf53.py"
)

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
# the package and tests/regression for those identifiers and must report
# zero hits, so this checker cannot spell them out literally.
_RETIRED_MODULE = "bare" + "_lindhard"
_RETIRED_CLASS = "Bare" + "LindhardCalculator"
_GPU_LIB = "cu" + "py"
_PSUTIL = "psu" + "til"
_BACKEND_FN = "get_" + "backend"

# BLAS thread variables the parallel CLI must pin to one thread before NumPy is
# imported (the pools are created during that import, so a later assignment has
# no effect).
_BLAS_THREAD_VARS = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

# Li projection (orbitals 48-52) of the 75-orbital C6LiC6_CWF model: the
# production configuration of the multi-process driver, and the cheapest
# configuration that still runs the full 75-orbital eigen stage.
_WIDE_ORBITALS = [48, 49, 50, 51, 52]


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
    import h5py

    from stm_data_processing.dft.wannier90.mlwf_im_susceptibility import (
        SusceptibilityCalculator_wang2012,
    )

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
    retired_check = _REGRESSION_DIR / ("check_" + _RETIRED_MODULE + ".py")
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
    for directory in (_PACKAGE_DIR, _REGRESSION_DIR):
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


def _split_ranges(nk: int, parts: int) -> list[tuple[int, int]]:
    """Split ``[0, nk)`` into ``parts`` contiguous, balanced row ranges."""
    base, remainder = divmod(nk, parts)
    ranges = []
    start = 0
    for index in range(parts):
        stop = start + base + (1 if index < remainder else 0)
        ranges.append((start, stop))
        start = stop
    return ranges


def _display_rows(raw_rows, nk: int) -> np.ndarray:
    """Displayed (fftshift) index of a raw FFT-order q row/column."""
    return (np.asarray(raw_rows, dtype=int) + nk // 2) % nk


def _mirror_permutation(nk: int) -> np.ndarray:
    """Displayed-index permutation of the reflection ``q -> -q``.

    On the frozen grid ``fftshift(fftfreq(nk))`` the reflection maps the raw FFT
    index ``iq`` to ``(nk - iq) % nk``, which in displayed indices is ``j ->
    (-j) % nk``, i.e. ``[0, nk-1, nk-2, ..., 1]``: a plain index reversal rolled
    by one.  A literal ``m[::-1, ::-1]`` is a *different* permutation for every
    nk (for even nk it sends the Nyquist entry q = -0.5 to q = 0.25), so it is
    not the mirror symmetry of this grid; (r) prints that residual explicitly.
    """
    shifted_raw = (np.arange(nk) - nk // 2) % nk
    return ((nk - shifted_raw) % nk + nk // 2) % nk


def _dummy_blocks(slices, nk: int) -> list[dict[str, np.ndarray]]:
    """Zero blocks with the shape ``assemble_slices`` expects for each slice."""
    return [
        {
            key: np.zeros((stop - start, nk), dtype=float)
            for key in ("data", "intraband", "interband")
        }
        for start, stop in slices
    ]


def _pinned_env() -> dict[str, str]:
    """Environment with the four BLAS thread pools pinned to one thread."""
    env = dict(os.environ)
    for name in _BLAS_THREAD_VARS:
        env[name] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


@contextmanager
def _quiet_engine() -> Iterator[None]:
    """Silence the engine logger while an expected ValueError is provoked."""
    logger = logging.getLogger("stm_data_processing.dft.wannier90.lindhard_re_chi")
    previous = logger.level
    logger.setLevel(logging.CRITICAL + 1)
    try:
        yield
    finally:
        logger.setLevel(previous)


def _first_existing_model() -> Path | None:
    """Real wide model used by the checks that need one, or None."""
    for candidate in (_WIDE_MODEL_DIR, _LESSORB_MODEL_DIR):
        if (candidate / f"{_WIDE_MODEL_SEED}_hr.dat").exists():
            return candidate
    return None


def write_mock_model(
    folder: Path,
    num_wann: int,
    seedname: str = "mock",
    bvecs: tuple[tuple[float, float, float], ...] | None = None,
) -> Path:
    """Write a tiny Wannier90 ``<seedname>_hr.dat`` (+ ``.wout``) model pair.

    The multi-process checks need a model that loads through the same
    ``MLWFHamiltonian.from_seedname`` path as the production model but costs
    nothing to build, so they write their own three-R-point model instead of
    touching the 472 MB C6LiC6_CWF file.  ``bvecs`` overrides the reciprocal
    vectors of the ``.wout`` file, which is what the checkpoint-signature check
    uses to build a second model with the *same* orbital count.
    """
    folder.mkdir(parents=True, exist_ok=True)
    r_points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    onsite = np.zeros((num_wann, num_wann), dtype=np.complex128)
    hopping = np.zeros((num_wann, num_wann), dtype=np.complex128)
    for index in range(num_wann):
        onsite[index, index] = -1.0 + 0.6 * index
        hopping[index, index] = -(0.3 + 0.05 * index)
    for index in range(num_wann - 1):
        onsite[index, index + 1] = onsite[index + 1, index] = 0.2
    lines = [
        "written on mock model for the parallel regression checks",
        f"      {num_wann}",
        f"      {len(r_points)}",
        "".join("    1" for _ in r_points),
    ]
    for index, (r1, r2, r3) in enumerate(r_points):
        matrix = onsite if index == 0 else hopping
        for m in range(num_wann):
            for n in range(num_wann):
                value = matrix[m, n]
                lines.append(
                    f"{r1:5d}{r2:5d}{r3:5d}{m + 1:5d}{n + 1:5d}"
                    f"{value.real:22.12f}{value.imag:22.12f}"
                )
    hr_path = folder / f"{seedname}_hr.dat"
    hr_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if bvecs is None:
        bvecs = (
            (1.474634, 0.851380, 0.0),
            (0.0, 1.702760, 0.0),
            (0.0, 0.0, 1.0),
        )
    (folder / f"{seedname}.wout").write_text(
        "mock wout for the parallel regression checks\n\n"
        " Reciprocal-Space Vectors (Ang^-1)\n"
        + "".join(
            f"        b_{index + 1}   {row[0]:.6f}   {row[1]:.6f}   {row[2]:.6f}\n"
            for index, row in enumerate(bvecs)
        ),
        encoding="utf-8",
    )
    return hr_path


def _stage_seconds(message: str) -> float:
    """Elapsed seconds of a ``stage=... s=<seconds>`` record.

    The scan is token based on purpose: ``q_sum`` also carries a ``px_per_s=``
    throughput field, so a naive ``rsplit("s=")`` would read that rate instead of
    the stage duration.
    """
    for token in message.split():
        if token.startswith("s=") and token != "s=":
            return float(token[2:])
    raise AssertionError(f"no s=<seconds> field in {message!r}")


class _LogCollector(logging.Handler):
    """Collect the records of one logger for the (t) logging contract."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def messages(self) -> list[str]:
        return [record.getMessage() for record in self.records]


def _attach_log_collector(name: str) -> _LogCollector:
    """Attach a collector to ``name`` and make sure INFO records are created."""
    logger = logging.getLogger(name)
    collector = _LogCollector()
    logger.addHandler(collector)
    if logger.level == logging.NOTSET or logger.level > logging.INFO:
        logger.setLevel(logging.INFO)
    return collector


def check_row_slices(nk: int) -> None:
    """(p) q row slices equal the raw rows of the full-mesh result bit for bit."""
    calc = RealLindhardCalculator(build_two_band_model(), nk=nk, eta=_ETA)
    arguments = {"chemical_potential": _EF, "temperature": _TEMPERATURE}
    full = calc.calculate(**arguments)
    display = _display_rows(np.arange(nk), nk)
    raw_q = np.fft.fftfreq(nk)
    slabs = 0
    for parts in (2, 3):
        for start, stop in _split_ranges(nk, parts):
            part = calc.calculate(q_index_range=(start, stop), **arguments)
            rows = display[start:stop]
            assert part["data"].shape == (stop - start, nk), (
                f"nk={nk}: slice rows=[{start}, {stop}) has shape {part['data'].shape}"
            )
            for key in ("data", "intraband", "interband"):
                expected = full[key][np.ix_(rows, display)]
                assert np.array_equal(part[key], expected), (
                    f"nk={nk}: slice rows=[{start}, {stop}) {key} is not bitwise equal "
                    f"to the single-process raw rows (max |diff| = "
                    f"{float(np.max(np.abs(part[key] - expected))):.3e})"
                )
                assert part[key].shape == (stop - start, nk)
            np.testing.assert_array_equal(part["q1_grid"][:, 0], raw_q[start:stop])
            np.testing.assert_array_equal(part["q2_grid"][0, :], raw_q)
            np.testing.assert_array_equal(
                part["q1_grid"],
                np.broadcast_to(raw_q[start:stop, None], (stop - start, nk)),
            )
            np.testing.assert_array_equal(
                part["q2_grid"], np.broadcast_to(raw_q[None, :], (stop - start, nk))
            )
            metadata = part["metadata"]
            assert metadata["q_index_range"] == (start, stop), (
                f"nk={nk}: metadata q_index_range = {metadata['q_index_range']!r}"
            )
            assert metadata["fft_order"] is True, f"nk={nk}: fft_order is not True"
            assert metadata["nk"] == nk
            slabs += 1

    rejected = 0
    with tempfile.TemporaryDirectory() as tmp:
        invalid = [
            ({"q_index_range": (-1, 2)}, "start < 0"),
            ({"q_index_range": (0, nk + 1)}, "stop > nk"),
            ({"q_index_range": (nk, nk)}, "start == stop"),
            ({"q_index_range": (2, 1)}, "start > stop"),
            (
                {"q_index_range": (0, nk), "output_path": str(Path(tmp) / "slice.h5")},
                "slice + output_path",
            ),
            ({"q_index_range": (0, nk), "q_range": (-0.5, 0.5)}, "slice + q_range"),
        ]
        with _quiet_engine():
            for kwargs, message in invalid:
                try:
                    calc.calculate(**arguments, **kwargs)
                except ValueError:
                    rejected += 1
                else:
                    raise AssertionError(
                        f"nk={nk}: q_index_range with {message} was not rejected"
                    )
    assert slabs == 5, f"nk={nk}: unexpected slab count {slabs}"
    print(
        f"  [p] nk={nk}: {slabs} slabs from 2- and 3-segment splits are bitwise equal "
        "(np.array_equal) to the raw rows of calculate(); grids are raw fftfreq and "
        f"metadata carries q_index_range/fft_order; {rejected} invalid calls rejected "
        "with ValueError (start < 0, stop > nk, start >= stop, slice + output_path, "
        "slice + q_range)"
    )


def check_slice_assembly(nk: int) -> None:
    """(q) ``assemble_slices`` reproduces ``calculate()`` bit for bit."""
    calc = RealLindhardCalculator(build_two_band_model(), nk=nk, eta=_ETA)
    arguments = {"chemical_potential": _EF, "temperature": _TEMPERATURE}
    full = calc.calculate(**arguments)
    plans = (
        ("one segment", [(0, nk)]),
        ("three contiguous segments", _split_ranges(nk, 3)),
        ("shuffled segment list", list(reversed(_split_ranges(nk, 4)))),
    )
    comparisons = 0
    for label, slices in plans:
        blocks = [
            calc.calculate(q_index_range=row_range, **arguments) for row_range in slices
        ]
        assembled = assemble_slices(slices, blocks, nk)
        for key in ("data", "intraband", "interband", "q1_grid", "q2_grid"):
            assert np.array_equal(assembled[key], full[key]), (
                f"nk={nk} {label}: assembled {key} differs from calculate() (max |diff| "
                f"= {float(np.max(np.abs(assembled[key] - full[key]))):.3e})"
            )
            comparisons += 1

    half = [(0, nk // 2)]
    rejected = 0
    for slices, blocks, kwargs, message in (
        ([(0, nk // 2), (0, nk)], _dummy_blocks([(0, nk // 2), (0, nk)], nk), {}, "overlapping rows"),
        (half, _dummy_blocks(half, nk), {}, "missing rows"),
        (half, _dummy_blocks(half, nk), {"mirror": True}, "rows the mirror cannot complete"),
        (
            [(0, nk)],
            [
                {
                    key: np.zeros((1, nk), dtype=float)
                    for key in ("data", "intraband", "interband")
                }
            ],
            {},
            "a wrongly shaped block",
        ),
        ([(0, nk)], [], {}, "a slice/block count mismatch"),
    ):
        try:
            assemble_slices(slices, blocks, nk, **kwargs)
        except ValueError:
            rejected += 1
        else:
            raise AssertionError(f"nk={nk}: assemble_slices accepted {message}")
    print(
        f"  [q] nk={nk}: one / three contiguous / shuffled segment lists all reproduce "
        f"calculate() bitwise ({comparisons} np.array_equal comparisons over "
        "data/intraband/interband/q1_grid/q2_grid); "
        f"{rejected} invalid slice lists rejected with ValueError (overlap, missing "
        "rows, mirror gap, wrong block shape, count mismatch)"
    )


def check_mirror_assembly(nk: int) -> None:
    """(r) Mirror assembly vs the single-process reference (<= 1e-12)."""
    calc = RealLindhardCalculator(build_two_band_model(), nk=nk, eta=_ETA)
    arguments = {"chemical_potential": _EF, "temperature": _TEMPERATURE}
    full = calc.calculate(**arguments)
    permutation = _mirror_permutation(nk)
    assemblies: dict[str, dict] = {}
    worst = 0.0
    for n_workers, label in ((1, "1 slice"), (3, "3 slices")):
        slices = plan_row_slices(nk, n_workers, mirror=True)
        blocks = [
            calc.calculate(q_index_range=row_range, **arguments) for row_range in slices
        ]
        mirrored = assemble_slices(slices, blocks, nk, mirror=True)
        for key in ("data", "intraband", "interband"):
            delta = float(np.max(np.abs(mirrored[key] - full[key])))
            worst = max(worst, delta)
            assert delta <= 1e-12, (
                f"nk={nk} mirror ({label}) {key}: max |mirror - reference| = {delta:.3e} "
                "exceeds 1e-12"
            )
        symmetry = max(
            float(
                np.max(
                    np.abs(mirrored[key] - mirrored[key][np.ix_(permutation, permutation)])
                )
            )
            for key in ("data", "intraband", "interband")
        )
        assert symmetry <= 1e-12, (
            f"nk={nk} mirror ({label}): reflection residual {symmetry:.3e} exceeds 1e-12"
        )
        reversal = float(np.max(np.abs(mirrored["data"] - mirrored["data"][::-1, ::-1])))
        assemblies[label] = {
            "mirrored": mirrored,
            "slices": slices,
            "symmetry": symmetry,
            "reversal": reversal,
            "reversal_equal": bool(
                np.array_equal(mirrored["data"], mirrored["data"][::-1, ::-1])
            ),
        }

    for key in ("data", "intraband", "interband", "q1_grid", "q2_grid"):
        assert np.array_equal(
            assemblies["1 slice"]["mirrored"][key], assemblies["3 slices"]["mirrored"][key]
        ), f"nk={nk}: the mirror changed when the canonical half was split"

    assert inspect.signature(assemble_slices).parameters["mirror"].default is False
    assert inspect.signature(run_parallel).parameters["mirror"].default is False
    full_plan = plan_row_slices(nk, 2)
    assert sum(stop - start for start, stop in full_plan) == nk, (
        f"nk={nk}: the default plan does not cover every row: {full_plan}"
    )
    partial = plan_row_slices(nk, 2, mirror=True)
    try:
        assemble_slices(partial, _dummy_blocks(partial, nk), nk)
    except ValueError:
        pass
    else:
        raise AssertionError(
            f"nk={nk}: a partial slice list was assembled although mirror defaulted to off"
        )

    single = assemblies["1 slice"]
    print(
        f"  [r] nk={nk}: mirror vs single process max|Δ| = {worst:.3e} (np.array_equal "
        f"bitwise=False, tol 1e-12); reflection residual = {single['symmetry']:.3e} "
        f"(<= 1e-12); mirror + 3 slices == mirror + 1 slice bitwise; default mirror off; "
        f"literal m == m[::-1, ::-1] -> {single['reversal_equal']} "
        f"(max {single['reversal']:.3e}: the reflection of the fftshift(fftfreq) grid "
        f"is {permutation.tolist()}, not a plain reversal)"
    )


def check_parallel_execution() -> None:
    """(s) Two spawned workers, checkpoints, resume and failure containment."""
    nk = 6
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        model_dir = root / "model"
        write_mock_model(model_dir, 4, "mock")
        ham = MLWFHamiltonian.from_seedname(str(model_dir), "mock")
        reference = RealLindhardCalculator(ham, nk=nk, eta=_ETA).calculate(
            chemical_potential=_EF, temperature=_TEMPERATURE
        )
        ckpt = root / "ckpt"
        h5 = root / "parallel.h5"

        def run_cli(*extra: str, output: Path = h5, checkpoint: Path = ckpt, model: Path = model_dir):
            command = [
                sys.executable,
                str(_CLI_PATH),
                "--model-dir",
                str(model),
                "--seedname",
                "mock",
                "--nk",
                str(nk),
                "--eta",
                repr(_ETA),
                "--temperature",
                repr(_TEMPERATURE),
                "--mu",
                repr(_EF),
                "--workers",
                "2",
                "--output",
                str(output),
                "--checkpoint-dir",
                str(checkpoint),
                "--progress-interval",
                "0",
                *extra,
            ]
            return subprocess.run(
                command,
                cwd=str(_REGRESSION_DIR.parents[1]),
                env=_pinned_env(),
                capture_output=True,
                text=True,
            )

        completed = run_cli()
        assert completed.returncode == 0, (
            f"two-worker run failed (exit {completed.returncode}):\n"
            f"{completed.stderr[-3000:]}"
        )
        assert h5.exists(), "the successful run wrote no h5"
        loaded = load_susceptibility_from_h5(str(h5))
        assert np.array_equal(loaded["data"], reference["data"]), (
            "the two-worker h5 differs from the single-process reference (max |diff| = "
            f"{float(np.max(np.abs(loaded['data'] - reference['data']))):.3e})"
        )

        slices = plan_row_slices(nk, 2)
        assert slices == [(0, 3), (3, 6)], f"unexpected plan for nk={nk}: {slices}"
        for start, stop in slices:
            stem = f"rows_{start}_{stop}"
            for suffix in (".npz", ".json", ".done"):
                assert (ckpt / f"{stem}{suffix}").exists(), f"missing {stem}{suffix}"
            assert (ckpt / f"{stem}.done").read_text(encoding="utf-8").strip() == "ok"
        assert scan_checkpoints(ckpt) == slices, (
            f"scan_checkpoints returned {scan_checkpoints(ckpt)} instead of {slices}"
        )

        reference_bytes = h5.read_bytes()
        (ckpt / "rows_3_6.npz").unlink()
        completed = run_cli("--resume")
        assert completed.returncode == 0, f"resume failed:\n{completed.stderr[-3000:]}"
        dispatched = [
            line for line in completed.stderr.splitlines() if "dispatched worker=" in line
        ]
        reused = [
            line for line in completed.stderr.splitlines() if "already complete" in line
        ]
        assert len(dispatched) == 1 and "rows=[3, 6)" in dispatched[0], (
            f"resume dispatched {len(dispatched)} slice(s) instead of the deleted one: "
            f"{dispatched}"
        )
        assert len(reused) == 1 and "rows=[0, 3)" in reused[0], (
            f"resume did not reuse the intact slice: {reused}"
        )
        assert h5.read_bytes() == reference_bytes, "resume changed the h5 product"
        assert np.array_equal(load_susceptibility_from_h5(str(h5))["data"], reference["data"])

        np.savez(
            ckpt / "rows_0_3.npz",
            data=np.zeros((1, nk)),
            intraband=np.zeros((1, nk)),
            interband=np.zeros((1, nk)),
        )
        completed = run_cli("--resume")
        assert completed.returncode == 0, f"resume failed:\n{completed.stderr[-3000:]}"
        dispatched = [
            line for line in completed.stderr.splitlines() if "dispatched worker=" in line
        ]
        assert len(dispatched) == 1 and "rows=[0, 3)" in dispatched[0], (
            f"resume with a corrupted shard dispatched {len(dispatched)} slice(s): {dispatched}"
        )
        assert h5.read_bytes() == reference_bytes, "the corrupted shard was not repaired"

        fail_h5 = root / "failed.h5"
        completed = run_cli(output=fail_h5, checkpoint=root / "ckpt_fail", model=root / "missing")
        assert completed.returncode != 0, "a failing worker did not make the parent fail"
        assert not fail_h5.exists(), "the failed run still wrote the final h5"
        assert "exited with code" in completed.stderr, (
            "the parent did not report the failing worker exit code"
        )
        worker_exit = completed.returncode

        dry = subprocess.run(
            [
                sys.executable,
                str(_CLI_PATH),
                "--model-dir",
                str(model_dir),
                "--seedname",
                "mock",
                "--nk",
                str(nk),
                "--workers",
                "2",
                "--mirror",
                "--dry-run",
            ],
            cwd=str(_REGRESSION_DIR.parents[1]),
            env=_pinned_env(),
            capture_output=True,
            text=True,
        )
        assert dry.returncode == 0, f"--dry-run failed:\n{dry.stderr[-2000:]}"
        assert "DRY-RUN" in dry.stdout and "segments=" in dry.stdout and "mirror=True" in dry.stdout
        pinned = [
            line
            for line in dry.stderr.splitlines()
            if "pinned before importing NumPy" in line
        ]
        assert pinned and all(name in pinned[-1] for name in _BLAS_THREAD_VARS), (
            f"the CLI did not pin all four BLAS thread variables: {pinned}"
        )

    source = _CLI_PATH.read_text(encoding="utf-8")
    for name in _BLAS_THREAD_VARS:
        assert f'"{name}"' in source, f"the CLI does not mention {name}"
    assert "os.environ.setdefault" in source, "the CLI does not setdefault the BLAS threads"
    assert source.index("setdefault") < min(
        source.index("import argparse"), source.index("from stm_data_processing")
    ), "the CLI must pin the BLAS threads before the first NumPy import"
    signature = inspect.signature(run_parallel)
    assert signature.parameters["start_method"].default == "spawn", (
        "the driver no longer defaults to the spawn start method"
    )
    assert signature.parameters["mirror"].default is False

    print(
        f"  [s] mock model nk={nk}, workers=2 spawn: h5 bitwise equal to the single "
        "process (np.array_equal True); checkpoint triples rows_0_3/rows_3_6 present "
        "with .done=ok; resume after deleting rows_3_6.npz re-dispatched only [3, 6) "
        "and reproduced the identical h5; resume after corrupting rows_0_3.npz "
        f"re-dispatched only [0, 3); a failing worker -> exit {worker_exit}, no h5; "
        "CLI --dry-run exit 0 with the plan; the four BLAS thread variables are "
        "pinned before the first NumPy import"
    )


def check_logging_contract(nk: int) -> None:
    """(t) The engine logging contract and the structural re-checks."""
    logger_name = "stm_data_processing.dft.wannier90.lindhard_re_chi"
    collector = _attach_log_collector(logger_name)
    saved = {name: os.environ.pop(name, None) for name in _BLAS_THREAD_VARS}
    try:
        calc = RealLindhardCalculator(build_two_band_model(), nk=nk, eta=_ETA)
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / f"logs_nk{nk}.h5")
            calc.calculate(
                chemical_potential=_EF,
                temperature=_TEMPERATURE,
                output_path=path,
                progress_interval_s=1e-6,
            )
        records = list(collector.records)
        messages = collector.messages()
    finally:
        logging.getLogger(logger_name).removeHandler(collector)
        for name, value in saved.items():
            if value is not None:
                os.environ[name] = value

    assert any("config nk=" in message for message in messages), "no configuration echo"
    config = next(message for message in messages if "config nk=" in message)
    for field in ("q_index_range=full", "band_block=", "block_entries=", "degeneracy_tolerance=1.000e-12"):
        assert field in config, f"the configuration echo is missing {field!r}: {config}"

    assert any("threads OMP_NUM_THREADS=" in message for message in messages), (
        "no BLAS thread advisory"
    )
    assert any(
        record.levelno >= logging.WARNING and "unset or > 1" in record.getMessage()
        for record in records
    ), "no WARNING for the unset BLAS thread variables"

    stage_times = {}
    for stage in ("diagonalize", "occupations", "q_sum", "fftshift", "h5_write"):
        timed = [
            message
            for message in messages
            if f"stage={stage} " in message and " s=" in message
        ]
        assert timed, f"no stage={stage} timing record"
        stage_times[stage] = _stage_seconds(timed[-1])
        assert stage_times[stage] >= 0.0

    progress = [
        record for record in records if getattr(record, "lindhard_progress", None) is not None
    ]
    assert progress, "no throttled progress record (progress_interval_s was tiny)"
    for record in progress:
        message = record.getMessage()
        for field in ("rows=", "rate=", "px/s=", "elapsed=", "eta=", "rss_peak="):
            assert field in message, f"the progress record is missing {field!r}: {message}"
        rows_done, rows_total, pixels = record.lindhard_progress
        assert 0 < rows_done <= rows_total and pixels == rows_done * nk

    assert any("summary rows=" in message for message in messages), "no closing summary"
    summary = next(message for message in messages if "summary rows=" in message)
    assert "max|data-(intra+inter)|=" in summary, f"the summary lacks the residual: {summary}"
    for name in ("data", "intraband", "interband"):
        digest = [message for message in messages if f"digest {name} sum=" in message]
        assert digest, f"no digest record for {name}"
        assert " max=" in digest[-1] and " min=" in digest[-1]
    assert any("px_per_s=" in message for message in messages), "no throughput record"

    for path in (_MODULE_PATH, _PARALLEL_MODULE_PATH):
        source = path.read_text(encoding="utf-8")
        lowered = source.lower()
        # The GPU-array library may only be *named* in prose (the engine
        # docstring says it has no such path); what must not exist is a code
        # path, which is exactly the token set check (k) uses.
        for token in (
            "import " + _GPU_LIB,
            _GPU_LIB + " as cp",
            "cp.asarray",
            "cp.zeros",
            "cp.sum",
            "_cuda",
            _PSUTIL,
            _BACKEND_FN,
        ):
            assert token not in lowered, f"{path.name} contains '{token}'"
        assert "BACKEND" not in source, f"{path.name} consults the package backend"
    assert _HK_ROW_BLOCK <= 512, (
        f"_HK_ROW_BLOCK={_HK_ROW_BLOCK} left the range Accelerate's zgemm handles"
    )
    engine_source = _MODULE_PATH.read_text(encoding="utf-8")
    assert "import resource" in engine_source, "the peak RSS no longer uses the stdlib"

    print(
        f"  [t] nk={nk}: configuration echo, BLAS advisory + WARNING for the four unset "
        f"thread variables, timed stages "
        + ", ".join(f"{name}={stage_times[name]:.3f}s" for name in stage_times)
        + f", {len(progress)} progress records with rows/rate/px per s/elapsed/eta/"
        f"rss_peak and the lindhard_progress extra, closing summary with "
        "max|data-(intra+inter)| and the three digests; no psutil/GPU-array/backend "
        f"token in either module; _HK_ROW_BLOCK={_HK_ROW_BLOCK} <= 512"
    )


_SINGLE_RUN_CODE = """
import json
import time

import numpy as np

from stm_data_processing.dft.wannier90.lindhard_re_chi import RealLindhardCalculator
from stm_data_processing.dft.wannier90.mlwf_hamiltonian import MLWFHamiltonian

ham = MLWFHamiltonian.from_seedname({model!r}, {seed!r})
calc = RealLindhardCalculator(ham, nk={nk}, eta={eta!r})
started = time.perf_counter()
result = calc.calculate(
    chemical_potential={mu!r},
    temperature={temperature!r},
    orbital_select={orbitals!r},
    output_path={h5!r},
)
wall = time.perf_counter() - started
np.savez(
    {npz!r},
    **{{key: result[key] for key in ("data", "intraband", "interband")}},
)
print("SINGLE_WALL", json.dumps({{"wall_s": wall}}))
"""


def check_real_model_parallel(nk: int = 32) -> None:
    """(u) Real wide model: single process vs two workers, and the product checks."""
    model_dir = _first_existing_model()
    if model_dir is None:
        print(
            f"  [u] real-model performance SKIPPED: no model under {_WIDE_MODEL_DIR} or "
            f"{_LESSORB_MODEL_DIR}"
        )
        return
    eta = 5e-3
    temperature = 4.2
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        single_h5 = root / "single.h5"
        single_npz = root / "single.npz"
        code = _SINGLE_RUN_CODE.format(
            model=str(model_dir),
            seed=_WIDE_MODEL_SEED,
            nk=nk,
            eta=eta,
            mu=_EF,
            temperature=temperature,
            orbitals=_WIDE_ORBITALS,
            h5=str(single_h5),
            npz=str(single_npz),
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(_REGRESSION_DIR.parents[1]),
            env=_pinned_env(),
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, (
            f"the single-process run failed (exit {completed.returncode}):\n"
            f"{completed.stderr[-2000:]}"
        )
        reported = [
            line for line in completed.stdout.splitlines() if line.startswith("SINGLE_WALL")
        ]
        assert reported, f"the single-process run printed no wall clock:\n{completed.stdout}"
        wall_single = json.loads(reported[-1].split(" ", 1)[1])["wall_s"]

        parallel_h5 = root / "parallel.h5"
        started = time.perf_counter()
        completed = subprocess.run(
            [
                sys.executable,
                str(_CLI_PATH),
                "--model-dir",
                str(model_dir),
                "--seedname",
                _WIDE_MODEL_SEED,
                "--nk",
                str(nk),
                "--eta",
                repr(eta),
                "--temperature",
                repr(temperature),
                "--mu",
                repr(_EF),
                "--orbitals",
                ",".join(str(orbital) for orbital in _WIDE_ORBITALS),
                "--projection-label",
                "li",
                "--workers",
                "2",
                "--output",
                str(parallel_h5),
                "--checkpoint-dir",
                str(root / "ckpt"),
                "--progress-interval",
                "5",
            ],
            cwd=str(_REGRESSION_DIR.parents[1]),
            env=_pinned_env(),
            capture_output=True,
            text=True,
        )
        wall_parallel = time.perf_counter() - started
        assert completed.returncode == 0, (
            f"the two-worker run failed (exit {completed.returncode}):\n"
            f"{completed.stderr[-3000:]}"
        )
        assert parallel_h5.exists(), "the two-worker run wrote no h5"

        with np.load(single_npz) as stored:
            arrays = {
                key: np.asarray(stored[key]) for key in ("data", "intraband", "interband")
            }
        residual = float(
            np.max(np.abs(arrays["data"] - (arrays["intraband"] + arrays["interband"])))
        )
        minimum = float(np.min(arrays["data"]))
        assert residual <= 1e-13, (
            f"nk={nk}: max|data-(intraband+interband)| = {residual:.3e} exceeds 1e-13"
        )
        assert minimum >= -1e-12, f"nk={nk}: min(data) = {minimum:.6e} < -1e-12"

        loaded_single = load_susceptibility_from_h5(str(single_h5))
        loaded_parallel = load_susceptibility_from_h5(str(parallel_h5))
        equal = np.array_equal(loaded_single["data"], loaded_parallel["data"])
        assert equal, (
            "the two-worker h5 differs from the single-process h5 (max |diff| = "
            f"{float(np.max(np.abs(loaded_single['data'] - loaded_parallel['data']))):.3e})"
        )

        plotted = "plot script absent"
        if _PLOT_SCRIPT.exists():
            spec = importlib.util.spec_from_file_location("handover_plot_cwf53", _PLOT_SCRIPT)
            plot_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(plot_module)
            png = root / f"li_nk{nk}_parallel.png"
            plot_module.plot_one(
                str(parallel_h5), str(png), f"cwf53, nk={nk}, Re chi0, Li(48-52)"
            )
            size = png.stat().st_size if png.exists() else 0
            assert size > 10_000, f"the handover plot script produced {size} bytes"
            plotted = f"{_PLOT_SCRIPT.name} -> {png.name} ({size} bytes)"

    speedup = wall_single / wall_parallel if wall_parallel > 0 else float("nan")
    print(
        f"  [u] {model_dir.name} nk={nk} Li(48-52), workers=2 spawn: wall single = "
        f"{wall_single:.1f}s, parallel = {wall_parallel:.1f}s -> speedup {speedup:.2f}x, "
        f"parallel efficiency {speedup / 2.0:.2f} of 2 workers (performance data, no "
        f"assertion); h5 bitwise equal (np.array_equal True); "
        f"max|data-(intra+inter)| = {residual:.3e} <= 1e-13; min(data) = {minimum:.6e} "
        f">= -1e-12; loader round trip OK; {plotted}"
    )


_CLI_SCRIPT_PATH = _CLI_PATH
_PARALLEL_LOGGER = "stm_data_processing.dft.wannier90.lindhard_re_chi_parallel"


def _run_parallel_cli(
    model_dir: Path,
    seedname: str,
    nk: int,
    *,
    output: Path,
    checkpoint: Path,
    workers: int = 2,
    resume: bool = False,
    orbitals: list[int] | None = None,
    extra: tuple[str, ...] = (),
) -> subprocess.CompletedProcess:
    """Run ``scripts/run_lindhard_re_chi_parallel.py`` on the pinned-BLAS env."""
    command = [
        sys.executable,
        str(_CLI_SCRIPT_PATH),
        "--model-dir",
        str(model_dir),
        "--seedname",
        seedname,
        "--nk",
        str(nk),
        "--eta",
        repr(_ETA),
        "--temperature",
        repr(_TEMPERATURE),
        "--mu",
        repr(_EF),
        "--workers",
        str(workers),
        "--output",
        str(output),
        "--checkpoint-dir",
        str(checkpoint),
        "--progress-interval",
        "0",
    ]
    if orbitals is not None:
        command += ["--orbitals", ",".join(str(orbital) for orbital in orbitals)]
    if resume:
        command.append("--resume")
    command += list(extra)
    return subprocess.run(
        command,
        cwd=str(_REGRESSION_DIR.parents[1]),
        env=_pinned_env(),
        capture_output=True,
        text=True,
    )


def _cli_report(completed: subprocess.CompletedProcess) -> dict[str, list[str]]:
    """Parent-side log lines of one CLI run, grouped by what they prove."""
    lines = completed.stderr.splitlines()
    return {
        "dispatched": [line for line in lines if "dispatched worker=" in line],
        "reused": [line for line in lines if "already complete" in line],
        "incompatible": [
            line for line in lines if "does not match the requested parameters" in line
        ],
        "unreadable": [line for line in lines if "unreadable checkpoint" in line],
    }


@contextmanager
def _isolated_root_logging() -> Iterator[None]:
    """Undo the root handlers ``run_parallel(log_file=...)`` installs."""
    root = logging.getLogger()
    before = list(root.handlers)
    level = root.level
    try:
        yield
    finally:
        for handler in list(root.handlers):
            if handler not in before:
                root.removeHandler(handler)
                handler.close()
        root.setLevel(level)


def check_bad_shard_recovery() -> None:
    """(v) A truncated checkpoint slice is discarded and recomputed (R1).

    A worker killed mid-write leaves a ``.npz`` that is empty or cut short; the
    resume path must treat it exactly like a missing slice instead of letting
    the reader's ``EOFError``/``BadZipFile`` escape.  The check drives the
    library API for the empty slice (so the "does not raise" claim is about
    ``run_parallel`` itself), the CLI for the truncated one, and finally asserts
    that ``assemble_from_checkpoints`` refuses an incomplete slice.
    """
    nk = 6
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        model_dir = root / "model"
        write_mock_model(model_dir, 4, "mock")
        ham = MLWFHamiltonian.from_seedname(str(model_dir), "mock")
        reference = RealLindhardCalculator(ham, nk=nk, eta=_ETA).calculate(
            chemical_potential=_EF, temperature=_TEMPERATURE
        )
        ckpt = root / "ckpt"
        h5 = root / "out.h5"
        options = {
            "eta": _ETA,
            "temperature": _TEMPERATURE,
            "chemical_potential": _EF,
            "n_workers": 2,
            "output_path": str(h5),
            "checkpoint_dir": str(ckpt),
            "progress_interval_s": 0.0,
        }
        collector = _attach_log_collector(_PARALLEL_LOGGER)
        observed = []
        try:
            code = run_parallel(str(model_dir), "mock", nk, **options)
            assert code == 0, f"the control run returned {code} instead of 0"
            control = h5.read_bytes()
            assert np.array_equal(
                load_susceptibility_from_h5(str(h5))["data"], reference["data"]
            )

            # --- empty .npz through the library API ---
            shard = ckpt / "rows_0_3.npz"
            shard.write_bytes(b"")
            collector.records.clear()
            code = run_parallel(str(model_dir), "mock", nk, resume=True, **options)
            assert code == 0, f"resume with an empty slice returned {code} instead of 0"
            messages = collector.messages()
            warnings = [message for message in messages if "unreadable checkpoint" in message]
            dispatched = [message for message in messages if "dispatched worker=" in message]
            reused = [message for message in messages if "already complete" in message]
            assert warnings, "the empty slice was discarded without a WARNING"
            assert any("rows_0_3.npz" in message for message in warnings)
            assert any(
                "EOFError" in message or "BadZipFile" in message for message in warnings
            ), f"unexpected read failure for an empty slice: {warnings}"
            assert len(dispatched) == 1 and "rows=[0, 3)" in dispatched[0], (
                f"the empty slice was not recomputed: {dispatched}"
            )
            assert len(reused) == 1 and "rows=[3, 6)" in reused[0]
            assert h5.read_bytes() == control, "the empty slice changed the product"
            observed.append((0, warnings[-1]))

            # --- truncated .npz (valid shard cut in half) through the CLI ---
            shard = ckpt / "rows_3_6.npz"
            valid = shard.read_bytes()
            shard.write_bytes(valid[: max(1, len(valid) // 2)])
            completed = _run_parallel_cli(
                model_dir, "mock", nk, output=h5, checkpoint=ckpt, resume=True
            )
            assert completed.returncode == 0, (
                f"resume with a truncated slice failed:\n{completed.stderr[-2000:]}"
            )
            report = _cli_report(completed)
            assert report["unreadable"], "the truncated slice was not reported"
            assert len(report["dispatched"]) == 1 and "rows=[3, 6)" in report["dispatched"][0]
            assert len(report["reused"]) == 1 and "rows=[0, 3)" in report["reused"][0]
            assert h5.read_bytes() == control, "the truncated slice changed the product"
            observed.append((len(valid) // 2, report["unreadable"][-1]))

            # --- the same broken slice makes the assembly refuse ---
            (ckpt / "rows_0_3.npz").write_bytes(b"")
            try:
                assemble_from_checkpoints(ckpt, nk=nk, orbital_select=[0, 1, 2, 3])
            except ValueError as exc:
                assert "incomplete" in str(exc), f"unexpected assembly error: {exc}"
            else:
                raise AssertionError(
                    "assemble_from_checkpoints assembled a directory with a broken slice"
                )
        finally:
            logging.getLogger(_PARALLEL_LOGGER).removeHandler(collector)

    empty_type = observed[0][1].split("(")[-1].split(")")[0]
    print(
        f"  [v] mock model nk={nk}, workers=2 spawn, resume=True: an empty "
        f"rows_0_3.npz ({empty_type}) and a rows_3_6.npz cut to {observed[1][0]} bytes "
        "were both discarded with a WARNING, each slice was recomputed, and the h5 "
        "stayed byte-identical to the clean control; assemble_from_checkpoints "
        "refuses an incomplete slice with ValueError('checkpoint rows=[0, 3) is "
        "incomplete')"
    )


def check_log_file_parameter() -> None:
    """(w) ``run_parallel(log_file=...)`` really writes the parent log (R2)."""
    nk = 6
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        model_dir = root / "model"
        write_mock_model(model_dir, 4, "mock")
        ham = MLWFHamiltonian.from_seedname(str(model_dir), "mock")
        reference = RealLindhardCalculator(ham, nk=nk, eta=_ETA).calculate(
            chemical_potential=_EF, temperature=_TEMPERATURE
        )
        api_log = root / "api.log"
        api_h5 = root / "api.h5"
        with _isolated_root_logging():
            code = run_parallel(
                str(model_dir),
                "mock",
                nk,
                eta=_ETA,
                temperature=_TEMPERATURE,
                chemical_potential=_EF,
                n_workers=2,
                output_path=str(api_h5),
                checkpoint_dir=str(root / "ckpt_api"),
                log_file=str(api_log),
                progress_interval_s=0.0,
            )
        assert code == 0, f"the API run with log_file returned {code}"
        assert api_log.exists(), "run_parallel(log_file=...) wrote no file"
        api_text = api_log.read_text(encoding="utf-8")
        assert api_log.stat().st_size > 0 and api_text.strip(), "the API log file is empty"
        for expected in ("plan nk=", "summary rows=", "worker digests verified"):
            assert expected in api_text, f"the API log file lacks {expected!r}"
        assert api_text.count("plan nk=") == 1, (
            "the parent log line was duplicated: " f"{api_text.count('plan nk=')} plan lines"
        )
        api_bytes = api_log.stat().st_size
        assert np.array_equal(
            load_susceptibility_from_h5(str(api_h5))["data"], reference["data"]
        )

        # The CLI installs the handler itself: no duplicated records either.
        cli_log = root / "cli.log"
        completed = _run_parallel_cli(
            model_dir,
            "mock",
            nk,
            output=root / "cli.h5",
            checkpoint=root / "ckpt_cli",
            extra=("--log-file", str(cli_log)),
        )
        assert completed.returncode == 0, f"the CLI run failed:\n{completed.stderr[-2000:]}"
        cli_text = cli_log.read_text(encoding="utf-8")
        assert cli_text.count("plan nk=") == 1 and cli_text.count("summary rows=") == 1, (
            "the CLI log has duplicated parent records (handler added twice)"
        )
        cli_bytes = cli_log.stat().st_size

    source = inspect.getsource(run_parallel)
    ignored = [
        name
        for name in inspect.signature(run_parallel).parameters
        if source.count(name) < 2
    ]
    assert not ignored, f"run_parallel ignores the parameter(s) {ignored}"
    docstring = inspect.getdoc(run_parallel) or ""
    assert "log_file" in docstring, "run_parallel does not document log_file"

    # The command line entry point must not ignore a public option either: every
    # destination its parser produces has to be read by main().
    spec = importlib.util.spec_from_file_location("cli_entry_point", _CLI_SCRIPT_PATH)
    cli_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli_module)
    parser = cli_module.build_parser()
    cli_source = _CLI_SCRIPT_PATH.read_text(encoding="utf-8")
    cli_ignored = [
        action.dest
        for action in parser._actions
        if action.dest != "help" and f"args.{action.dest}" not in cli_source
    ]
    assert not cli_ignored, f"the CLI parser produces unused destination(s) {cli_ignored}"
    options = len([action for action in parser._actions if action.option_strings])

    print(
        f"  [w] run_parallel(log_file=...): parent log really written ({api_bytes} B) and "
        f"contains the plan/summary/verification records exactly once; the CLI "
        f"--log-file path (handler already installed) stays at one record per line "
        f"({cli_bytes} B); every parameter of run_parallel is used in its body, "
        f"log_file is documented, and all {options} CLI options are consumed by "
        "main() (no silently ignored public parameter)"
    )


def check_q_index_range_validation(nk: int) -> None:
    """(x) Exactly two integer entries are accepted, everything else raises (R3)."""
    calc = RealLindhardCalculator(build_two_band_model(), nk=nk, eta=_ETA)
    arguments = {"chemical_potential": _EF, "temperature": _TEMPERATURE}
    full = calc.calculate(**arguments)
    invalid = (
        ((0, 1, 2), "a 3-entry tuple (start, stop, step)"),
        ((0,), "a 1-entry tuple"),
        ((0, 1, 2, 3), "a 4-entry tuple"),
        (["0", "1"], "string entries"),
        ((0, 1.5), "a float entry"),
        ((0, None), "a None entry"),
        ("ab", "a 2-character string"),
        (np.zeros((2, 2)), "a 2-D array of length 2"),
        (np.array([0.0, 3.0]), "a float array of length 2"),
        (3, "a non-iterable int"),
        (2.5, "a non-iterable float"),
        (None, "an explicit None"),
    )
    rejected = 0
    with _quiet_engine():
        for value, message in invalid:
            if value is None:
                # None is the documented default, not an error: it must behave
                # exactly like omitting the argument.  Asserted below, not here.
                continue
            try:
                calc.calculate(q_index_range=value, **arguments)
            except ValueError as exc:
                assert "q_index_range" in str(exc), (
                    f"unclear error for {message}: {exc}"
                )
                rejected += 1
            else:
                raise AssertionError(f"q_index_range with {message} was not rejected")

    explicit_none = calc.calculate(q_index_range=None, **arguments)
    for key in ("data", "intraband", "interband", "q1_grid", "q2_grid"):
        assert np.array_equal(explicit_none[key], full[key]), (
            f"q_index_range=None changed {key} against the default call"
        )

    rows = _display_rows(np.arange(nk), nk)
    whole = calc.calculate(q_index_range=(0, nk), **arguments)
    for key in ("data", "intraband", "interband"):
        assert np.array_equal(whole[key], full[key][np.ix_(rows, rows)]), (
            f"q_index_range=(0, {nk}) changed {key} against the full run"
        )
    middle = calc.calculate(q_index_range=(2, nk - 1), **arguments)
    assert middle["data"].shape == (nk - 3, nk)
    assert np.array_equal(middle["data"], full["data"][np.ix_(rows[2 : nk - 1], rows)])
    print(
        f"  [x] nk={nk}: {rejected} illegal q_index_range forms rejected with "
        "ValueError ((0,1,2), length 1/4, string/float/None entries, a 2-character "
        "string, a 2-D array, non-iterable numbers); None behaves exactly like the "
        f"default and (0, {nk}) / (2, {nk - 1}) still return the raw rows of the full run"
    )


_PEAK_RUN_CODE = """
import json
import time

import numpy as np

from stm_data_processing.dft.wannier90.lindhard_re_chi import (
    RealLindhardCalculator,
    peak_rss_bytes,
)
from stm_data_processing.dft.wannier90.mlwf_hamiltonian import MLWFHamiltonian

ham = MLWFHamiltonian.from_seedname({model!r}, {seed!r})
calc = RealLindhardCalculator(ham, nk={nk}, eta={eta!r})
started = time.perf_counter()
result = calc.calculate(
    chemical_potential={mu!r},
    temperature={temperature!r},
    orbital_select={orbitals!r},
)
print("PEAK", json.dumps({{"rss_peak": peak_rss_bytes(), "wall_s": time.perf_counter() - started,
                          "max_data": float(np.max(result["data"]))}}))
"""


def check_estimate_upper_bound(nk_list: tuple[int, ...] = (8, 16, 32)) -> None:
    """(y) The worker RSS estimate is an upper bound of the measured peak (R4)."""
    model_dir = _first_existing_model()
    if model_dir is None:
        print(
            f"  [y] estimate upper bound SKIPPED: no model under {_WIDE_MODEL_DIR} or "
            f"{_LESSORB_MODEL_DIR}"
        )
        return
    ham = MLWFHamiltonian.from_seedname(str(model_dir), _WIDE_MODEL_SEED)
    shape = read_model_shape(str(model_dir), _WIDE_MODEL_SEED)
    assert shape == (ham.num_wann, len(ham.r_list)), (
        f"read_model_shape returned {shape} but the model has "
        f"{ham.num_wann} orbitals and {len(ham.r_list)} R points"
    )
    num_wann, nrpts = shape
    identity_wann, identity_bvecs = model_identity(str(model_dir), _WIDE_MODEL_SEED)
    assert identity_wann == num_wann, (
        f"the checkpoint signature reads num_wann={identity_wann}, expected {num_wann}"
    )
    assert identity_bvecs is not None and np.array_equal(
        np.asarray(identity_bvecs, dtype=float), np.asarray(ham.bvecs, dtype=float)
    ), "the checkpoint signature must read the bvecs through the worker's loader"

    table = []
    for nk in nk_list:
        for label, orbitals in (("full", None), ("li", _WIDE_ORBITALS)):
            code = _PEAK_RUN_CODE.format(
                model=str(model_dir),
                seed=_WIDE_MODEL_SEED,
                nk=nk,
                eta=5e-3,
                mu=_EF,
                temperature=4.2,
                orbitals=orbitals,
            )
            completed = subprocess.run(
                [sys.executable, "-c", code],
                cwd=str(_REGRESSION_DIR.parents[1]),
                env=_pinned_env(),
                capture_output=True,
                text=True,
            )
            assert completed.returncode == 0, (
                f"the nk={nk} {label} peak run failed:\n{completed.stderr[-1500:]}"
            )
            reported = [
                line for line in completed.stdout.splitlines() if line.startswith("PEAK ")
            ]
            assert reported, f"the nk={nk} {label} run printed no peak"
            measured = json.loads(reported[-1].split(" ", 1)[1])
            n_orb = num_wann if orbitals is None else len(orbitals)
            estimate = estimate_worker_rss_bytes(nk, num_wann, n_orb, nrpts)
            assert estimate >= measured["rss_peak"], (
                f"nk={nk} {label}: estimate {estimate / 1024**2:.1f} MB is below the "
                f"measured peak {measured['rss_peak'] / 1024**2:.1f} MB"
            )
            table.append((nk, label, estimate, measured["rss_peak"], measured["wall_s"]))

    assert num_wann != 75 or nrpts != 0  # the estimate really uses the model numbers
    production_nk, production_workers = 256, 8
    estimate_production = estimate_worker_rss_bytes(
        production_nk, num_wann, num_wann, nrpts
    )
    total = estimate_production * production_workers
    available = available_memory_bytes()
    budget = None if available is None else 0.8 * available
    structural = (
        production_nk * production_nk * num_wann * num_wann * 16 * production_workers
    )
    completed = _run_parallel_cli(
        model_dir,
        _WIDE_MODEL_SEED,
        production_nk,
        output=Path(tempfile.gettempdir()) / "t6_dry_run.h5",
        checkpoint=Path(tempfile.gettempdir()) / "t6_dry_run_ckpt",
        workers=production_workers,
        extra=("--dry-run",),
    )
    assert completed.returncode == 0, f"--dry-run failed:\n{completed.stderr[-2000:]}"
    dry = completed.stdout.strip().splitlines()[-1]
    assert (
        f"DRY-RUN workers={production_workers} nk={production_nk}" in dry
        and f"num_wann={num_wann}" in dry
        and f"nrpts={nrpts}" in dry
    ), f"the dry-run plan does not report the real model size: {dry}"
    refused = budget is not None and total > budget
    # The guard must decide exactly on the printed estimate: a budget just above
    # the estimate is admitted, one just below is refused.  That rules out both a
    # hidden over-refusal and a silent admission of an undersized plan.
    _memory_guard(estimate_production, production_workers, 1.01 * total / 1024**3)
    try:
        _memory_guard(estimate_production, production_workers, 0.99 * total / 1024**3)
    except MemoryError:
        pass
    else:
        raise AssertionError(
            "the memory guard admitted a plan whose estimate exceeds the budget"
        )

    worst = max(table, key=lambda row: row[3] / row[2])
    print(
        "  [y] estimate vs measured peak RSS on "
        f"{model_dir.name} (num_wann={num_wann}, nrpts={nrpts} read from the model "
        "header): "
        + "; ".join(
            f"nk={nk} {label} est={estimate / 1024**2:.0f}MB >= measured="
            f"{measured / 1024**2:.0f}MB"
            for nk, label, estimate, measured, _wall in table
        )
    )
    print(
        f"      worst ratio measured/estimate = {worst[3] / worst[2]:.3f} "
        f"(nk={worst[0]} {worst[1]}); nk={production_nk} full x {production_workers} "
        f"workers: estimate {estimate_production / 1024**3:.2f} GB/worker, total "
        f"{total / 1024**3:.1f} GB, default budget "
        + ("unknown" if budget is None else f"{budget / 1024**3:.1f} GB")
        + (
            (
                f" -> the guard refuses; the printed estimate total {total / 1024**3:.1f} GB"
                f" exceeds the budget, and the eigenvector floor "
                f"{structural / 1024**3:.1f} GB is "
                + (
                    "also above it, so the real per-worker arrays do not fit here"
                    if structural > budget
                    else "below it, i.e. the refusal comes from the estimate's "
                    "safety margin (use --max-mem-gb when the real footprint fits)"
                )
            )
            if refused
            else " -> the guard admits the plan"
        )
        + "; the decision boundary sits exactly at the estimate (a budget 1% above "
        "is admitted, 1% below is refused), so an operator on a machine whose budget "
        "lies between the measured need and the estimate can use the documented "
        "--max-mem-gb. Plan: " + dry
    )


def check_foreign_shard_rejection() -> None:
    """(z) A checkpoint slice from another model is never silently reused (R5)."""
    nk = 6
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        model_a = root / "model_a"
        model_c = root / "model_c"
        model_b = root / "model_b"
        write_mock_model(model_a, 4, "mock")
        write_mock_model(
            model_c,
            4,
            "mock",
            bvecs=((1.30, 0.85, 0.0), (0.0, 1.70, 0.0), (0.0, 0.0, 1.0)),
        )
        write_mock_model(model_b, 6, "mock")
        ham_a = MLWFHamiltonian.from_seedname(str(model_a), "mock")
        reference_a = RealLindhardCalculator(ham_a, nk=nk, eta=_ETA).calculate(
            chemical_potential=_EF, temperature=_TEMPERATURE
        )
        ham_b = MLWFHamiltonian.from_seedname(str(model_b), "mock")
        reference_b = RealLindhardCalculator(ham_b, nk=nk, eta=_ETA).calculate(
            chemical_potential=_EF, temperature=_TEMPERATURE
        )
        assert not np.array_equal(reference_a["data"], reference_b["data"]), (
            "the two mock models must differ numerically for this check"
        )
        ckpt = root / "ckpt"
        h5 = root / "out.h5"
        options = {"output": h5, "checkpoint": ckpt, "workers": 2}

        completed = _run_parallel_cli(model_a, "mock", nk, **options)
        assert completed.returncode == 0, f"the model A run failed:\n{completed.stderr[-1500:]}"
        assert np.array_equal(
            load_susceptibility_from_h5(str(h5))["data"], reference_a["data"]
        )
        bytes_a = h5.read_bytes()

        # (1) same orbital count, different bvecs -> incompatible signature
        completed = _run_parallel_cli(model_c, "mock", nk, resume=True, **options)
        assert completed.returncode == 0, f"the model C resume failed:\n{completed.stderr[-1500:]}"
        report = _cli_report(completed)
        assert len(report["dispatched"]) == 2 and not report["reused"], (
            "the shards of model A were reused for model C: " f"{report}"
        )
        assert len(report["incompatible"]) == 2, (
            f"the incompatible shards were not reported: {report['incompatible']}"
        )

        # (2) resuming the same model again reuses both slices (no false alarm)
        completed = _run_parallel_cli(model_c, "mock", nk, resume=True, **options)
        assert completed.returncode == 0, f"the second model C resume failed:\n{completed.stderr[-1500:]}"
        report = _cli_report(completed)
        assert not report["dispatched"] and len(report["reused"]) == 2, (
            "the bvecs comparison produces a false incompatibility: " f"{report}"
        )
        bytes_c = h5.read_bytes()

        # (3) one tampered shard is recomputed, the other one is reused
        meta_path = ckpt / "rows_3_6.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["num_wann"] = 999
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        completed = _run_parallel_cli(model_c, "mock", nk, resume=True, **options)
        assert completed.returncode == 0, f"the tampered-shard resume failed:\n{completed.stderr[-1500:]}"
        report = _cli_report(completed)
        assert len(report["dispatched"]) == 1 and "rows=[3, 6)" in report["dispatched"][0], (
            f"the tampered shard was not recomputed alone: {report}"
        )
        assert len(report["reused"]) == 1 and "rows=[0, 3)" in report["reused"][0]
        assert len(report["incompatible"]) == 1
        assert h5.read_bytes() == bytes_c, "recomputing an identical slice changed the h5"

        # (4) another orbital count: every shard is rejected and replaced
        completed = _run_parallel_cli(model_b, "mock", nk, resume=True, **options)
        assert completed.returncode == 0, f"the model B resume failed:\n{completed.stderr[-1500:]}"
        report = _cli_report(completed)
        assert len(report["dispatched"]) == 2 and not report["reused"], (
            f"the shards of model C were reused for model B: {report}"
        )
        final = load_susceptibility_from_h5(str(h5))["data"]
        assert np.array_equal(final, reference_b["data"]), (
            "the final product does not carry model B's numbers"
        )
        assert not np.array_equal(final, reference_a["data"]) and h5.read_bytes() != bytes_a

    print(
        f"  [z] mock models nk={nk}, workers=2 spawn, shared checkpoint dir: a model "
        "with the same 4 orbitals but different bvecs re-dispatched both slices "
        "(2 incompatible WARNINGs) and a second resume of the same model reused both "
        "(exactly 2 'already complete' records, no false incompatibility); tampering "
        "one shard's num_wann recomputed only that slice; switching to the 6-orbital "
        "model re-dispatched both and the final h5 equals the 6-orbital reference "
        "bitwise and differs from the 4-orbital product"
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
        ("(p) q row slices bitwise equal (odd nk=5)", lambda: check_row_slices(5)),
        ("(p) q row slices bitwise equal (nk=8)", lambda: check_row_slices(8)),
        ("(p) q row slices bitwise equal (nk=16)", lambda: check_row_slices(16)),
        ("(q) slice assembly bitwise equal (odd nk=5)", lambda: check_slice_assembly(5)),
        ("(q) slice assembly bitwise equal (nk=8)", lambda: check_slice_assembly(8)),
        ("(r) mirror assembly (nk=4)", lambda: check_mirror_assembly(4)),
        ("(r) mirror assembly (odd nk=5)", lambda: check_mirror_assembly(5)),
        ("(r) mirror assembly (nk=6)", lambda: check_mirror_assembly(6)),
        ("(r) mirror assembly (nk=8)", lambda: check_mirror_assembly(8)),
        ("(r) mirror assembly (nk=16)", lambda: check_mirror_assembly(16)),
        ("(s) parallel workers/checkpoints/resume (mock model)", check_parallel_execution),
        ("(t) logging contract + structural re-checks", lambda: check_logging_contract(8)),
        ("(u) real-model nk=32 parallel performance + end-to-end", check_real_model_parallel),
        ("(v) truncated/empty checkpoint self-heals (R1)", check_bad_shard_recovery),
        ("(w) run_parallel(log_file=...) really logs (R2)", check_log_file_parameter),
        ("(x) q_index_range validation (R3)", lambda: check_q_index_range_validation(8)),
        ("(y) worker RSS estimate is an upper bound (R4)", check_estimate_upper_bound),
        ("(z) foreign-model checkpoint rejection (R5)", check_foreign_shard_rejection),
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
