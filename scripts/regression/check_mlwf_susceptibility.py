"""Regression checks for the corrected mlwf_susceptibility paths.

Both backends now implement the same zero-temperature Lindhard result

    Im chi(q, omega) = -pi * d_eps * sum_eps sum_k
        Tr[M_init A(k, eps) M_fin A(k+q, eps+omega)]

with the operator spectral function A = i (G^R - G^R^dag) / (2 pi) (bug M8:
the elementwise -Im[G]/pi differs for complex H(k)), the FFT k-reversal that
turns the convolution sum_k B(k) C(q-k) into the correlation
sum_k B(k) C(k+q) (bug M9), and the -pi * d_eps prefactor (bug M10: the
legacy -d_eps/(2 pi) was off by 2 pi^2).  The energy-integration weight is
the actual grid spacing d_eps = |omega| / (n_eps - 1), not the requested
resolution (bug M7).  The CUDA path _compute_imag_chi_cuda uses exactly these
conventions; this machine has no CUDA device, so that branch is validated
structurally by a NumPy replica of its tensor pipeline (check (e)).

Check list:
  (a) CPU path equals an exact direct-sum zero-T Lindhard reference on the
      discrete grid; the unprojected (identity-selection) reference differs
      (orbital projection matters).
  (b) the integration weight equals |omega| / (n_eps - 1); a hypothetical
      |resolution| weighting would scale the result by |resolution| / d_eps.
  (c) H3 regression: the pyFFTW backward plan matches np.fft.ifftn.
  (d) n_eps == 1 raises a clear ValueError.
  (e) the NumPy replica of the fixed CUDA pipeline (cupy swapped for numpy,
      same tensor-operation order) equals the CPU result, while a replica of
      the legacy pipeline differs (M8/M9/M10 are load-bearing).
  (f) the real _compute_imag_chi_cuda body, executed with cupy swapped for a
      NumPy shim, equals the CPU result (indexing/order/prefactor of the
      shipped code, not of a copy).
"""

from __future__ import annotations

import os
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

# Keep matplotlib cache warnings out of the regression output.
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

from stm_data_processing.config import set_backend

# No CUDA device on this machine: force the deterministic CPU path.
set_backend("cpu")

import stm_data_processing.dft.wannier90.mlwf_susceptibility as ms  # noqa: E402
from stm_data_processing.dft.wannier90.mlwf_hamiltonian import (  # noqa: E402
    MLWFHamiltonian,
)
from stm_data_processing.dft.wannier90.mlwf_susceptibility import (  # noqa: E402
    SusceptibilityCalculator_wang2012,
)

_NK = 8
_NW = 3
_ETA = 0.02
_OMEGA_LIMIT = 0.37
_RESOLUTION = 0.1


def build_random_hermitian_hamiltonian(seed: int = 42) -> MLWFHamiltonian:
    """Small random Hermitian tight-binding Hamiltonian.

    H(k) = H0 + exp(2*pi*i*k1) A + exp(-2*pi*i*k1) A^dagger
              + exp(2*pi*i*k2) B + exp(-2*pi*i*k2) B^dagger,
    which is Hermitian for all k.  A random (non-cosine) band structure makes
    the orbital projection non-trivially change the susceptibility.
    """
    rng = np.random.default_rng(seed)
    nw = _NW
    x = rng.standard_normal((nw, nw)) + 1j * rng.standard_normal((nw, nw))
    h0 = (x + x.conj().T) / 2
    a = rng.standard_normal((nw, nw)) + 1j * rng.standard_normal((nw, nw))
    b = rng.standard_normal((nw, nw)) + 1j * rng.standard_normal((nw, nw))
    r_list = np.array(
        [[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]],
        dtype=np.int32,
    )
    h_list_flat = np.stack([h0, a, a.conj().T, b, b.conj().T]).reshape(5, nw * nw)
    ndegen = np.ones(len(r_list), dtype=np.float64)
    bvecs = np.eye(3)
    return MLWFHamiltonian.from_arrays(nw, r_list, h_list_flat, ndegen, bvecs)


def make_calculator() -> SusceptibilityCalculator_wang2012:
    """Calculator with random non-identity orbital selection matrices."""
    ham = build_random_hermitian_hamiltonian()
    rng = np.random.default_rng(7)
    minit = rng.random((_NW, _NW))  # real, non-identity
    mfin = rng.random((_NW, _NW))  # real, non-identity
    calc = SusceptibilityCalculator_wang2012(
        ham, nk=_NK, eta=_ETA, minit=minit, mfin=mfin
    )
    # Redirect FFTW wisdom files out of the source tree.
    wisdom_dir = Path(tempfile.mkdtemp(prefix="fftw_wisdom_"))
    calc._get_fftw_wisdom_path = (
        lambda nk, nw, direction: str(wisdom_dir / f"w_{direction}.json")
    )
    return calc


def _a_op(calc: SusceptibilityCalculator_wang2012, hk_grid: np.ndarray, omega: float):
    """Operator spectral function i (G^R - G^R^dag) / (2 pi)."""
    gr = np.asarray(calc.gf.compute_green(hk_grid, omega))
    return 1j * (gr - np.conj(np.swapaxes(gr, -1, -2))) / (2.0 * np.pi)


def physics_reference(
    calc: SusceptibilityCalculator_wang2012,
    omega_limit: float,
    resolution: float,
    weight: float,
    minit: np.ndarray | None = None,
    mfin: np.ndarray | None = None,
) -> np.ndarray:
    """Exact zero-T Lindhard direct sum on the discrete grid.

    Im chi(q, omega) = -weight * sum_eps sum_k
    Tr[M_init A(k, eps) M_fin A(k+q, eps+omega)], evaluated with explicit
    q/k loops, then fftshifted to match the code output layout.
    """
    nw = calc.num_wann
    nk = calc.nk
    mi = calc.minit if minit is None else np.asarray(minit)
    mf = calc.mfin if mfin is None else np.asarray(mfin)
    n_eps = int(np.round(np.abs(omega_limit) / resolution)) + 1
    eps_occ = np.linspace(-np.abs(omega_limit), 0.0, n_eps)
    H = np.asarray(calc.ham.hk(calc.k_points)).reshape(nk, nk, nw, nw)

    acc = np.zeros((nk, nk))
    for e in eps_occ:
        A1 = np.einsum("ac,ijcb->ijab", mi, _a_op(calc, H, e))
        A2 = np.einsum("ac,ijcb->ijab", mf, _a_op(calc, H, e + omega_limit))
        for q1 in range(nk):
            for q2 in range(nk):
                s = 0.0
                for k1 in range(nk):
                    for k2 in range(nk):
                        s += np.trace(
                            A1[k1, k2] @ A2[(k1 + q1) % nk, (k2 + q2) % nk]
                        ).real
                acc[q1, q2] += s
    return -weight * np.fft.fftshift(acc, axes=(0, 1))


def cuda_pipeline_replica(
    calc: SusceptibilityCalculator_wang2012,
    omega_limit: float,
    resolution: float,
    legacy: bool = False,
) -> np.ndarray:
    """NumPy replica of the fixed _compute_imag_chi_cuda tensor pipeline.

    There is no CUDA device on this machine, so the cupy branch cannot be
    executed.  This replica mirrors _compute_imag_chi_cuda operation for
    operation with cupy swapped for numpy: per-energy operator spectral
    function from the shared _compute_single_particle_spectra, minit/mfin
    einsum("ac,ijcb->ijab") projection, k-reversal of the occupied spectrum,
    fftn + fftshift, einsum("ijab,ijba->ij"), ifftshift + ifftn,
    accumulation, fftshift, and the -pi * d_eps weight.  With legacy=True the
    three pre-fix operations are reconstructed instead (elementwise
    -Im[G]/pi, no k-reversal, -d_eps/(2 pi)), which makes the check sensitive
    to a regression of M8/M9/M10.
    """
    nw = calc.num_wann
    nk = calc.nk
    _, eps_occ, eps_unocc, d_eps = calc._energy_grid(omega_limit, resolution)
    neg_idx = np.concatenate(([0], np.arange(nk - 1, 0, -1)))
    minit = np.asarray(calc.minit)
    mfin = np.asarray(calc.mfin)
    hk_grid = calc.hk_grid

    def spectral(omega: float) -> np.ndarray:
        if not legacy:
            return np.asarray(calc._compute_single_particle_spectra(omega))
        # Legacy M8: elementwise -Im[G]/pi (not Hermitian for complex H(k)).
        gr = np.asarray(calc.gf.compute_green(hk_grid, omega))
        return -np.imag(gr) / np.pi

    chi_q_accum = np.zeros((nk, nk), dtype=np.float64)
    for i in range(len(eps_occ)):
        spectra_occ = spectral(eps_occ[i])
        spectra_occ = np.ascontiguousarray(spectra_occ.reshape(nk, nk, nw, nw))
        spectra_occ = np.einsum("ac,ijcb->ijab", minit, spectra_occ)
        if not legacy:
            # Bug M9: k-reversal of the first factor only.
            spectra_occ = spectra_occ[neg_idx[:, None], neg_idx[None, :], :, :]

        spectra_unocc = spectral(eps_unocc[i])
        spectra_unocc = np.ascontiguousarray(
            spectra_unocc.reshape(nk, nk, nw, nw)
        )
        spectra_unocc = np.einsum("ac,ijcb->ijab", mfin, spectra_unocc)

        b_occ = np.fft.fftn(spectra_occ, axes=(0, 1))
        b_occ_shifted = np.fft.fftshift(b_occ, axes=(0, 1))
        b_unocc = np.fft.fftn(spectra_unocc, axes=(0, 1))
        b_unocc_shifted = np.fft.fftshift(b_unocc, axes=(0, 1))

        b_prod = np.einsum("ijab,ijba->ij", b_occ_shifted, b_unocc_shifted)
        conv_q = np.fft.ifftn(
            np.fft.ifftshift(b_prod, axes=(0, 1)), axes=(0, 1)
        ).real
        chi_q_accum += conv_q

    chi_q_accum = np.fft.fftshift(chi_q_accum)
    if legacy:
        # Legacy M10: -d_eps/(2 pi) instead of -pi * d_eps.
        return -d_eps / (2 * np.pi) * chi_q_accum
    # Bug M10 fix: zero-T Lindhard prefactor.
    return -np.pi * d_eps * chi_q_accum


def check_cpu_vs_physics_reference() -> None:
    """(a) CPU path equals the exact zero-T Lindhard direct sum (M8/M9/M10)."""
    calc = make_calculator()
    n_eps = int(np.round(np.abs(_OMEGA_LIMIT) / _RESOLUTION)) + 1
    d_eps = np.abs(_OMEGA_LIMIT) / (n_eps - 1)
    chi_cpu = calc._compute_imag_chi(_OMEGA_LIMIT, _RESOLUTION)
    chi_ref = physics_reference(
        calc, _OMEGA_LIMIT, _RESOLUTION, weight=np.pi * d_eps
    )
    scale = max(1.0, float(np.max(np.abs(chi_ref))))
    max_err = float(np.max(np.abs(chi_cpu - chi_ref)))
    print(
        f"  [a] nk={_NK}, nw={_NW}: max |CPU - exact zero-T Lindhard| = "
        f"{max_err:.3e} (tol {1e-9 * scale:.3e})"
    )
    assert max_err < 1e-9 * scale, f"CPU vs physics reference mismatch: {max_err:.3e}"

    # The orbital projection must matter: the identity-selection reference
    # differs from the projected result (M6).
    chi_unprojected = physics_reference(
        calc,
        _OMEGA_LIMIT,
        _RESOLUTION,
        weight=np.pi * d_eps,
        minit=np.eye(calc.num_wann),
        mfin=np.eye(calc.num_wann),
    )
    diff_unproj = float(np.max(np.abs(chi_unprojected - chi_cpu)))
    print(
        f"  [a] unprojected reference differs by {diff_unproj:.3e} "
        f"(relative {diff_unproj / scale:.3e}) -> projection matters"
    )
    assert diff_unproj > 1e-3 * scale, "unprojected result should differ from CPU"


def check_cuda_pipeline_replica() -> None:
    """(e) NumPy replica of the fixed CUDA pipeline equals the CPU result."""
    calc = make_calculator()
    chi_cpu = calc._compute_imag_chi(_OMEGA_LIMIT, _RESOLUTION)
    chi_replica = cuda_pipeline_replica(calc, _OMEGA_LIMIT, _RESOLUTION)

    scale = max(1.0, float(np.max(np.abs(chi_cpu))))
    max_err = float(np.max(np.abs(chi_replica - chi_cpu)))
    print(
        f"  [e] max |NumPy replica of _compute_imag_chi_cuda - CPU| = "
        f"{max_err:.3e} (tol {1e-9 * scale:.3e})"
    )
    assert max_err < 1e-9 * scale, (
        f"CUDA pipeline replica does not match the CPU path: {max_err:.3e}"
    )

    # Sensitivity: a replica of the legacy pipeline must NOT match, otherwise
    # this check would not detect an M8/M9/M10 regression in the CUDA path.
    chi_legacy = cuda_pipeline_replica(
        calc, _OMEGA_LIMIT, _RESOLUTION, legacy=True
    )
    diff_legacy = float(np.max(np.abs(chi_legacy - chi_cpu)))
    print(
        f"  [e] legacy-pipeline replica differs by {diff_legacy:.3e} "
        f"(relative {diff_legacy / scale:.3e}) -> M8/M9/M10 are load-bearing"
    )
    assert diff_legacy > 1e-2 * scale, (
        "legacy CUDA pipeline replica should differ from the CPU result"
    )


class _ShimMemoryPool:
    """Stand-in for the cupy memory pool used by the CUDA path."""

    def __init__(self) -> None:
        self.limit = None

    def set_limit(self, size: int) -> None:
        self.limit = size

    def used_bytes(self) -> int:
        return 0

    def free_unused_blocks(self) -> None:
        pass

    def free_all_blocks(self) -> None:
        pass


def _build_cupy_shim() -> types.ModuleType:
    """NumPy-backed cupy stand-in covering the API the CUDA method uses."""
    shim = types.ModuleType("cupy")
    shim.asarray = np.asarray
    shim.asnumpy = lambda a: np.asarray(a)
    shim.ascontiguousarray = np.ascontiguousarray
    shim.einsum = np.einsum
    shim.zeros = np.zeros
    shim.float64 = np.float64
    shim.pi = np.pi
    shim.conj = np.conj
    shim.swapaxes = np.swapaxes
    shim.fft = np.fft

    pool = _ShimMemoryPool()
    shim.get_default_memory_pool = lambda: pool

    class _Device:
        # cupy exposes mem_info as a property returning (free, total).
        @property
        def mem_info(self):
            return (1 << 32, 1 << 34)

    shim.cuda = types.SimpleNamespace(Device=_Device)
    return shim


def check_cuda_method_with_numpy_shim() -> None:
    """(f) the real _compute_imag_chi_cuda body with cupy swapped for numpy."""
    calc = make_calculator()
    chi_cpu = calc._compute_imag_chi(_OMEGA_LIMIT, _RESOLUTION)

    shim = _build_cupy_shim()
    previous_cupy = sys.modules.get("cupy")
    previous_get_xp = ms.get_xp
    sys.modules["cupy"] = shim
    ms.get_xp = lambda: shim
    try:
        chi_cuda = calc._compute_imag_chi_cuda(_OMEGA_LIMIT, _RESOLUTION)
    finally:
        ms.get_xp = previous_get_xp
        if previous_cupy is None:
            sys.modules.pop("cupy", None)
        else:
            sys.modules["cupy"] = previous_cupy

    scale = max(1.0, float(np.max(np.abs(chi_cpu))))
    max_err = float(np.max(np.abs(chi_cuda - chi_cpu)))
    print(
        f"  [f] real _compute_imag_chi_cuda (numpy cupy shim) vs CPU = "
        f"{max_err:.3e} (tol {1e-9 * scale:.3e})"
    )
    assert max_err < 1e-9 * scale, (
        f"CUDA method output differs from the CPU path: {max_err:.3e}"
    )


def check_weight() -> None:
    """(b) M7: integration weight equals |omega| / (n_eps - 1)."""
    calc = make_calculator()
    n_eps = int(np.round(np.abs(_OMEGA_LIMIT) / _RESOLUTION)) + 1
    d_eps = np.abs(_OMEGA_LIMIT) / (n_eps - 1)

    chi_cpu = calc._compute_imag_chi(_OMEGA_LIMIT, _RESOLUTION)
    # If the code weighted the sum by |resolution| instead of d_eps, the
    # result would be the physics reference with weight pi * |resolution|,
    # i.e. chi_cpu scaled by |resolution| / d_eps.
    chi_old_weight = physics_reference(
        calc, _OMEGA_LIMIT, _RESOLUTION, weight=np.pi * np.abs(_RESOLUTION)
    )
    chi_expected_old = chi_cpu * (np.abs(_RESOLUTION) / d_eps)
    scale = max(1.0, float(np.max(np.abs(chi_cpu))))
    max_err = float(np.max(np.abs(chi_old_weight - chi_expected_old)))
    rel_dev = (np.abs(_RESOLUTION) - d_eps) / d_eps
    print(
        f"  [b] n_eps={n_eps}, d_eps={d_eps:.6f} vs |resolution|={_RESOLUTION:.6f}; "
        f"relative deviation if resolution were used = {rel_dev * 100:.2f}%"
    )
    print(f"  [b] old-weight reference == CPU result * |res|/d_eps: max err {max_err:.3e}")
    assert abs(rel_dev) > 0.01, "expected a >1% weight deviation for this parameter set"
    assert max_err < 1e-8 * scale, f"weight scaling mismatch: {max_err:.3e}"


def check_h3_fftw_backward() -> None:
    """(c) H3 regression: pyFFTW backward plan matches np.fft.ifftn."""
    calc = make_calculator()
    wisdom_dir = Path(tempfile.mkdtemp(prefix="fftw_wisdom_"))
    plan, in_arr = calc._init_fftw_plan(
        shape=(_NK, _NK),
        fft_axes=(0, 1),
        num_threads=2,
        wisdom_path=str(wisdom_dir / "ifft.json"),
        direction="FFTW_BACKWARD",
    )
    rng = np.random.default_rng(3)
    x = rng.standard_normal((_NK, _NK)) + 1j * rng.standard_normal((_NK, _NK))
    in_arr[:] = x
    y_plan = np.array(plan(), copy=True)
    y_ref = np.fft.ifftn(x)
    max_err = float(np.max(np.abs(y_plan - y_ref)))
    print(f"  [c] max |pyfftw(BACKWARD) - np.fft.ifftn| = {max_err:.3e}")
    assert max_err < 1e-10 * max(1.0, float(np.max(np.abs(y_ref)))), (
        f"H3 regression failed: {max_err:.3e}"
    )


def check_neps_one() -> None:
    """(d) n_eps == 1 raises a clear ValueError (explicit handling)."""
    calc = make_calculator()
    try:
        calc._energy_grid(0.04, 0.1)  # |omega|/resolution = 0.4 -> n_eps = 1
    except ValueError as exc:
        print(f"  [d] n_eps=1 raises ValueError: {exc}")
        return
    raise AssertionError("n_eps=1 should raise ValueError")


def main() -> None:
    """Run every check and summarize."""
    checks = [
        ("(a) CPU vs exact zero-T Lindhard reference (M8/M9/M10)", check_cpu_vs_physics_reference),
        ("(b) integration weight d_eps = |omega|/(n_eps-1) (M7)", check_weight),
        ("(c) pyFFTW backward plan == ifftn (H3)", check_h3_fftw_backward),
        ("(d) n_eps=1 explicit ValueError", check_neps_one),
        ("(e) CUDA pipeline NumPy replica == CPU path (M8/M9/M10)", check_cuda_pipeline_replica),
        ("(f) real _compute_imag_chi_cuda (numpy cupy shim) == CPU path", check_cuda_method_with_numpy_shim),
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
