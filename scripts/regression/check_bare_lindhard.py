"""Regression checks for the bare_lindhard fixes (bugs M4 and M5).

Bug M4 (q-grid double fftshift)
-------------------------------
calculate() used to apply np.fft.fftshift to q_vals although
np.linspace(-0.5, 0.5, nk, endpoint=False) already centers q=0 at index
nk//2.  The raw chi_q array is in FFT order (q=0 at index 0); after the data
fftshift, q=0 sits at index nk//2, exactly matching the unshifted linspace
grid.  The extra fftshift moved q=0 to index 0 and mislabeled the data by
half a Brillouin zone.

Bug M5 (denominator sign convention)
------------------------------------
The Lindhard denominator was eps_n(k+q) - eps_m(k) + i*eta, the negative of
the standard retarded static convention eps_m(k) - eps_n(k+q) + i*eta.  The
imaginary part is unchanged but the real part flips sign, so Re chi0 was
positive where the standard static Lindhard function is negative.

Checks run on small mock tight-binding Hamiltonians (1-2 orbitals, nk 4..16):
  (a) library chi0(q) agrees with an independent direct summation,
  (b) symmetry chi0(-q) == conj(chi0(q)),
  (c) q-grid labeling: the q=0 index of the returned grid equals the q=0
      index of the fftshifted data (old vs new labeling reproduced inline),
  (d) 1D free-electron limit: Re chi0(q -> 0) < 0 (standard static Lindhard
      sign), while the old convention gives Re > 0.

Note: this machine has no CUDA device, so only the CPU (NumPy) path is
executed; the GPU path is covered structurally (both denominator assignments
in the source are checked to use the standard convention).
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

from stm_data_processing.dft.wannier90.bare_lindhard import (  # noqa: E402
    BareLindhardCalculator,
)
from stm_data_processing.dft.wannier90.mlwf_hamiltonian import (  # noqa: E402
    MLWFHamiltonian,
)
from stm_data_processing.utils.miscellaneous import fermi  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]

_EF = 0.0  # eV
_TEMPERATURE = 100.0  # K
_ETA = 0.05  # eV


def build_square_1band(t: float) -> MLWFHamiltonian:
    """2D square-lattice single-band tight-binding model.

    H(k) = -2t (cos(2*pi*k1) + cos(2*pi*k2)).
    """
    r_list = np.array(
        [[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]],
        dtype=np.int32,
    )
    h_list_flat = np.array([[0.0], [-t], [-t], [-t], [-t]], dtype=np.complex128)
    ndegen = np.ones(len(r_list), dtype=np.float64)
    bvecs = np.eye(3)
    return MLWFHamiltonian.from_arrays(1, r_list, h_list_flat, ndegen, bvecs)


def build_chain_1band(t: float) -> MLWFHamiltonian:
    """1D nearest-neighbor chain embedded in 2D (dispersion along k1 only).

    H(k) = -2t cos(2*pi*k1).
    """
    r_list = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]], dtype=np.int32)
    h_list_flat = np.array([[0.0], [-t], [-t]], dtype=np.complex128)
    ndegen = np.ones(len(r_list), dtype=np.float64)
    bvecs = np.eye(3)
    return MLWFHamiltonian.from_arrays(1, r_list, h_list_flat, ndegen, bvecs)


def build_square_2band(t: float, delta: float, mixing: float) -> MLWFHamiltonian:
    """2D two-band model with on-site splitting and inter-orbital hopping.

    H(k) = -2t (cos(2*pi*k1) + cos(2*pi*k2)) * I2
           + [[-delta, mixing], [mixing, delta]].
    """
    r_list = np.array(
        [[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]],
        dtype=np.int32,
    )
    onsite = np.array([[-delta, mixing], [mixing, delta]], dtype=np.complex128)
    hop = np.array([[-t, 0.0], [0.0, -t]], dtype=np.complex128)
    h_list_flat = np.stack([onsite, hop, hop, hop, hop]).reshape(5, 4)
    ndegen = np.ones(len(r_list), dtype=np.float64)
    bvecs = np.eye(3)
    return MLWFHamiltonian.from_arrays(2, r_list, h_list_flat, ndegen, bvecs)


def direct_sum(evals, evecs, iq1, iq2, eta, sign) -> complex:
    """Direct double-loop evaluation of chi0(q) (independent reference).

    chi0(q) = (1/N) sum_k sum_{m,n} |<u_m(k)|u_n(k+q)>|^2
              * (f_m(k) - f_n(k+q)) / (sign * (eps_m(k) - eps_n(k+q)) + i*eta)

    sign = +1 gives the standard convention; sign = -1 reproduces the
    pre-fix (M5) denominator.  evals has shape (nk1, nk2, nband) and evecs
    (nk1, nk2, nband, nband); q is an integer index and k+q wraps with
    mod nk, matching the FFT-order layout of the library.
    """
    nk1, nk2, nband = evals.shape
    f = fermi(evals, mu=_EF, T=_TEMPERATURE)
    chi = 0.0j
    for i1 in range(nk1):
        for i2 in range(nk2):
            j1 = (i1 + iq1) % nk1
            j2 = (i2 + iq2) % nk2
            for m in range(nband):
                for n in range(nband):
                    num = f[i1, i2, m] - f[j1, j2, n]
                    den = sign * (evals[i1, i2, m] - evals[j1, j2, n]) + 1j * eta
                    ov = np.vdot(evecs[i1, i2, :, m], evecs[j1, j2, :, n])
                    chi += num * abs(ov) ** 2 / den
    return chi / (nk1 * nk2)


def check_direct_sum(nk: int) -> None:
    """(a) Library chi0(q) agrees with the independent direct summation."""
    calc = BareLindhardCalculator(build_square_2band(1.0, 1.0, 0.5), nk=nk, eta=_ETA)
    evals, evecs = calc.eigen
    chi_lib = calc._compute_bare_lindhard(
        ef=_EF, temperature=_TEMPERATURE, q_chunk_size=8
    )
    max_err = 0.0
    for iq1 in range(nk):
        for iq2 in range(nk):
            ref = direct_sum(evals, evecs, iq1, iq2, _ETA, +1)
            max_err = max(max_err, abs(chi_lib[iq1, iq2] - ref))
    scale = max(1.0, float(np.max(np.abs(chi_lib))))
    # chi0(q=0) vanishes in the bare formula: f(k) - f(k) = 0 for each band
    # pair and the m != n overlaps vanish by orthonormality of the evecs.
    q0_val = abs(chi_lib[0, 0])
    print(
        f"  [a] nk={nk}: max |chi_lib - chi_direct| = {max_err:.3e} "
        f"(tol {1e-9 * scale:.3e}); |chi_lib(q=0)| = {q0_val:.3e}"
    )
    assert max_err < 1e-9 * scale, f"direct-sum mismatch: {max_err:.3e}"
    assert q0_val < 1e-10, f"chi0(q=0) should vanish, got {q0_val:.3e}"


def check_symmetry(nk: int) -> None:
    """(b) chi0(-q) == conj(chi0(q)) on the raw FFT-order grid."""
    calc = BareLindhardCalculator(build_square_2band(1.0, 1.0, 0.5), nk=nk, eta=_ETA)
    chi = calc._compute_bare_lindhard(
        ef=_EF, temperature=_TEMPERATURE, q_chunk_size=8
    )
    max_err = 0.0
    for iq1 in range(nk):
        for iq2 in range(nk):
            lhs = chi[iq1, iq2]
            rhs = np.conj(chi[(-iq1) % nk, (-iq2) % nk])
            max_err = max(max_err, abs(lhs - rhs))
    scale = max(1.0, float(np.max(np.abs(chi))))
    print(f"  [b] nk={nk}: max |chi(q) - conj(chi(-q))| = {max_err:.3e}")
    assert max_err < 1e-8 * scale, f"symmetry mismatch: {max_err:.3e}"


def check_qgrid_alignment(nk: int) -> None:
    """(c) M4: q=0 index of the grid equals the q=0 index of the data."""
    calc = BareLindhardCalculator(build_square_1band(1.0), nk=nk, eta=_ETA)
    result = calc.calculate(temperature=_TEMPERATURE)
    data = result["data"]
    q1 = result["q1_grid"]
    q2 = result["q2_grid"]

    # Old (buggy) labeling: linspace then fftshift -> q=0 at index 0.
    new_q_vals = np.linspace(-0.5, 0.5, nk, endpoint=False)
    old_q_vals = np.fft.fftshift(new_q_vals)
    old_q0 = int(np.argmin(np.abs(old_q_vals)))
    new_q0 = int(np.argmin(np.abs(new_q_vals)))

    assert old_q0 == 0, f"old grid q=0 index should be 0, got {old_q0}"
    assert new_q0 == nk // 2, f"new grid q=0 index should be {nk // 2}, got {new_q0}"
    assert abs(q1[nk // 2, nk // 2]) < 1e-12 and abs(q2[nk // 2, nk // 2]) < 1e-12

    # chi(q=0) = 0 in the bare formula, so the fftshifted data must vanish at
    # the q=0 pixel (nk//2, nk//2) -- the same pixel the grid labels q=0.
    assert abs(data[nk // 2, nk // 2]) < 1e-10

    # Pixel alignment: displayed[j] must equal chi(q = -0.5 + j/nk (mod 1)),
    # i.e. the raw value at integer index (j + nk//2) mod nk, evaluated with
    # the independent direct summation.
    evals, evecs = calc.eigen
    max_err = 0.0
    for i1 in range(nk):
        for i2 in range(nk):
            iq1 = (i1 + nk // 2) % nk
            iq2 = (i2 + nk // 2) % nk
            ref = direct_sum(evals, evecs, iq1, iq2, _ETA, +1)
            max_err = max(max_err, abs(data[i1, i2] - ref))
    scale = max(1.0, float(np.max(np.abs(data))))
    print(
        f"  [c] nk={nk}: old grid q=0 index = {old_q0} (buggy), "
        f"new grid q=0 index = {new_q0}, fftshifted data q=0 index = {nk // 2}; "
        f"max |displayed - chi(q_displayed)| = {max_err:.3e}"
    )
    assert max_err < 1e-9 * scale, f"pixel misalignment: {max_err:.3e}"


def check_1d_sign(nk: int) -> None:
    """(d) 1D free-electron limit: Re chi0 < 0 (M5 fix direction)."""
    calc = BareLindhardCalculator(build_chain_1band(1.0), nk=nk, eta=_ETA)
    chi = calc._compute_bare_lindhard(
        ef=_EF, temperature=_TEMPERATURE, q_chunk_size=8
    )

    # q = (q1, 0): for a chain dispersive only along k1 this is the 1D
    # Lindhard response.  With the standard convention every term is
    # non-positive in the real part (f is monotone in eps), so
    # Re chi0(q1, 0) <= 0 for all q1.
    re_q = chi[:, 0].real
    assert np.all(re_q <= 1e-10), f"Re chi0(q1, 0) > 0 at some q1: {re_q}"
    assert re_q[1] < 0.0, f"Re chi0 at smallest nonzero q should be < 0, got {re_q[1]:.6e}"

    # The old (pre-fix) denominator flips the real part to positive.
    evals, evecs = calc.eigen
    old_val = direct_sum(evals, evecs, 1, 0, _ETA, -1).real
    print(
        f"  [d] nk={nk}: Re chi0(q=(1/{nk}, 0)) = {re_q[1]:.6e} "
        f"(standard, < 0 OK); old convention would give {old_val:.6e} (> 0)"
    )
    assert old_val > 0.0, "old convention should flip Re chi0 to positive"


def check_source_convention() -> None:
    """(e) Structural: CPU and GPU denominators use the standard convention."""
    src = (
        REPO_ROOT / "src/stm_data_processing/dft/wannier90/bare_lindhard.py"
    ).read_text(encoding="utf-8")
    standard = src.count("eps_m[None, :] - eps_n + 1j * self.eta")
    flipped = src.count("eps_n - eps_m[None, :] + 1j * self.eta")
    print(
        f"  [e] source: standard denominator occurrences = {standard} "
        f"(expect 2: CPU + GPU), flipped occurrences = {flipped} (expect 0)"
    )
    assert standard == 2, f"expected standard denominator twice, found {standard}"
    assert flipped == 0, f"old flipped denominator still present ({flipped} x)"


def main() -> None:
    """Run every check and summarize."""
    checks = [
        ("(a) direct-sum consistency (2-band, nk=8)", lambda: check_direct_sum(8)),
        ("(b) chi(-q) = conj(chi(q)) (2-band, nk=8)", lambda: check_symmetry(8)),
        ("(c) q-grid alignment (1-band, nk=4)", lambda: check_qgrid_alignment(4)),
        ("(c) q-grid alignment (1-band, nk=8)", lambda: check_qgrid_alignment(8)),
        ("(d) 1D static Lindhard sign (chain, nk=16)", lambda: check_1d_sign(16)),
        ("(e) CPU/GPU convention (structural)", check_source_convention),
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
