"""Regression checks for mlwf_susceptibility fixes (bugs M6 and M7).

Bug M6 (CUDA path ignores the orbital selection matrices)
---------------------------------------------------------
The CPU path projects the spectral functions with
einsum("ac,ijcb->ijab", minit, A) before the FFT, but the CUDA path used to
FFT the raw spectra directly, silently giving the identity-selection result
for any non-identity minit/mfin.  This script replicates the fixed CUDA
algorithm in NumPy and compares it with the real CPU path on a small random
Hamiltonian with non-identity selection matrices.

Bug M7 (energy-integration weight inconsistent with the grid)
-------------------------------------------------------------
n_eps = round(|omega|/resolution) + 1 and eps = linspace(-|omega|, 0, n_eps)
give an actual grid spacing d_eps = |omega| / (n_eps - 1), but the final
normalization used |resolution|, which differs by a few percent whenever
|omega|/resolution is not an integer.  The fix uses d_eps as the weight.

Check list:
  (a) CPU projected path vs NumPy replica of the (fixed) CUDA path agree;
      the unprojected replica differs (bug M6 demonstration).
  (b) the new weight equals |omega| / (n_eps - 1); old-weight result equals
      new result scaled by |resolution| / d_eps (relative deviation printed).
  (c) H3 regression: the pyFFTW backward plan matches np.fft.ifftn.
  (d) n_eps == 1 raises a clear ValueError.

Note: this machine has no CUDA device, so the CUDA branch is validated by a
structural NumPy replica (identical tensor-operation order with cupy swapped
for numpy); the real cupy code path cannot be executed here.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

# Keep matplotlib cache warnings out of the regression output.
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

from stm_data_processing.config import set_backend

# No CUDA device on this machine: force the deterministic CPU path.
set_backend("cpu")

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


def cuda_replica(
    calc: SusceptibilityCalculator_wang2012,
    omega_limit: float,
    resolution: float,
    apply_projection: bool,
    use_deps_weight: bool,
) -> np.ndarray:
    """NumPy replica of the fixed CUDA path (cupy swapped for numpy).

    Mirrors _compute_imag_chi_cuda step by step: per-energy spectral
    function, optional minit/mfin einsum projection, fftn + fftshift,
    einsum("ijab,ijba->ij"), ifftshift + ifftn, and the final weight.
    """
    nw = calc.num_wann
    nk = calc.nk
    n_eps = int(np.round(np.abs(omega_limit) / resolution)) + 1
    eps_occ = np.linspace(-np.abs(omega_limit), 0.0, n_eps)
    eps_unocc = eps_occ + omega_limit
    d_eps = np.abs(omega_limit) / (n_eps - 1)

    chi_q_accum = np.zeros((nk, nk), dtype=np.float64)
    for i in range(n_eps):
        spectra_occ = np.asarray(
            calc._compute_single_particle_spectra(eps_occ[i])
        ).reshape(nk, nk, nw, nw)
        spectra_unocc = np.asarray(
            calc._compute_single_particle_spectra(eps_unocc[i])
        ).reshape(nk, nk, nw, nw)
        if apply_projection:
            spectra_occ = np.einsum("ac,ijcb->ijab", calc.minit, spectra_occ)
            spectra_unocc = np.einsum("ac,ijcb->ijab", calc.mfin, spectra_unocc)
        b_occ = np.fft.fftshift(np.fft.fftn(spectra_occ, axes=(0, 1)), axes=(0, 1))
        b_unocc = np.fft.fftshift(
            np.fft.fftn(spectra_unocc, axes=(0, 1)), axes=(0, 1)
        )
        b_prod = np.einsum("ijab,ijba->ij", b_occ, b_unocc)
        conv_q = np.fft.ifftn(
            np.fft.ifftshift(b_prod, axes=(0, 1)), axes=(0, 1)
        ).real
        chi_q_accum += conv_q

    weight = d_eps if use_deps_weight else np.abs(resolution)
    return np.fft.fftshift(chi_q_accum) * (-weight / (2 * np.pi))


def check_cpu_vs_cuda_replica() -> None:
    """(a) CPU projected path equals the fixed CUDA replica (M6)."""
    calc = make_calculator()
    chi_cpu = calc._compute_imag_chi(_OMEGA_LIMIT, _RESOLUTION)
    chi_replica = cuda_replica(
        calc, _OMEGA_LIMIT, _RESOLUTION, apply_projection=True, use_deps_weight=True
    )
    scale = max(1.0, float(np.max(np.abs(chi_cpu))))
    max_err = float(np.max(np.abs(chi_cpu - chi_replica)))
    print(
        f"  [a] nk={_NK}, nw={_NW}: max |CPU - CUDA-replica| = {max_err:.3e} "
        f"(tol {1e-8 * scale:.3e})"
    )
    assert max_err < 1e-8 * scale, f"CPU vs CUDA replica mismatch: {max_err:.3e}"

    # Bug M6 demonstration: without the projection the result differs.
    chi_unprojected = cuda_replica(
        calc,
        _OMEGA_LIMIT,
        _RESOLUTION,
        apply_projection=False,
        use_deps_weight=True,
    )
    diff_unproj = float(np.max(np.abs(chi_unprojected - chi_cpu)))
    print(
        f"  [a] unprojected replica differs by {diff_unproj:.3e} "
        f"(relative {diff_unproj / scale:.3e}) -> projection matters"
    )
    assert diff_unproj > 1e-3 * scale, "unprojected result should differ from CPU"


def check_weight() -> None:
    """(b) M7: integration weight equals |omega| / (n_eps - 1)."""
    calc = make_calculator()
    n_eps = int(np.round(np.abs(_OMEGA_LIMIT) / _RESOLUTION)) + 1
    d_eps = np.abs(_OMEGA_LIMIT) / (n_eps - 1)
    assert abs(d_eps - np.abs(_OMEGA_LIMIT) / (n_eps - 1)) < 1e-15

    chi_cpu = calc._compute_imag_chi(_OMEGA_LIMIT, _RESOLUTION)
    chi_old_weight = cuda_replica(
        calc,
        _OMEGA_LIMIT,
        _RESOLUTION,
        apply_projection=True,
        use_deps_weight=False,
    )
    # The only difference between the old and new normalization is the
    # weight factor, so the old-weight result must equal the new result
    # scaled by |resolution| / d_eps.
    chi_expected_old = chi_cpu * (np.abs(_RESOLUTION) / d_eps)
    scale = max(1.0, float(np.max(np.abs(chi_cpu))))
    max_err = float(np.max(np.abs(chi_old_weight - chi_expected_old)))
    rel_dev = (np.abs(_RESOLUTION) - d_eps) / d_eps
    print(
        f"  [b] n_eps={n_eps}, d_eps={d_eps:.6f} vs |resolution|={_RESOLUTION:.6f}; "
        f"relative deviation before fix = {rel_dev * 100:.2f}%"
    )
    print(f"  [b] old-weight result == new result * |res|/d_eps: max err {max_err:.3e}")
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
        ("(a) CPU vs CUDA replica equivalence (M6)", check_cpu_vs_cuda_replica),
        ("(b) integration weight d_eps = |omega|/(n_eps-1) (M7)", check_weight),
        ("(c) pyFFTW backward plan == ifftn (H3)", check_h3_fftw_backward),
        ("(d) n_eps=1 explicit ValueError", check_neps_one),
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
