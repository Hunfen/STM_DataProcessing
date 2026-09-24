"""Regression checks for the Usadel module (anchors A1-A5 + numerical path).

Run from the repository root::

    .venv/bin/python tests/regression/check_usadel.py

Covers the numerical anchors of ``docs/design/usadel_gap_model.md`` section 8:

  - A1 bulk BCS density of states (analytic path);
  - A2 weak-coupling gap equation (``Delta0 = 1.764 kB Tc``);
  - A3 self-consistent gap iteration and DOS consistency (analytic path);
  - A4 tunneling spectrum ``dI/dV`` and convolution / broadening (analytic path);
  - A5 S/N bilayer minigap and bulk-BCS recovery in deep S (numerical path);
  - numerical bulk reproduction (numerical Riccati BVP path, V1b).

The ``geometry="bulk"`` branch of ``Usadel1D.dos`` returns the analytic
``bcs_dos`` by design (document section 3.2 / R-2), so A1, A3 and A4 are
labeled *analytic-path* checks: they validate the bulk formula and the
convolution APIs, not the BVP solver.  The BVP solver itself is exercised by
A5 and by ``test_numerical_bulk_reproduction``, which drives the retarded
Riccati solver in a deep-S region and compares it against the analytic BCS
form (mirrors verification report V1b).

All identifiers, comments and strings are in English; no local absolute path is
referenced (the script must stay data-independent for CI).
"""

from __future__ import annotations

import numpy as np

from stm_data_processing.utils.usadel import (
    KB,
    WEAK_COUPLING_RATIO,
    Usadel1D,
    bcs_dos,
    bcs_gap,
    fermi_derivative,
)

DELTA0 = 1e-3  # eV (1 meV)
D = 1e16  # nm^2 / s
TC = DELTA0 / (WEAK_COUPLING_RATIO * KB)


def _rel(a, b):
    return np.abs(a - b) / np.maximum(np.abs(b), 1e-30)


def test_a1_bulk_bcs_dos_analytic_path() -> None:
    """A1 (analytic path): bulk DOS equals the analytic BCS form.

    ``dos(geometry="bulk")`` returns the analytic ``bcs_dos``, so this check
    validates the bulk formula against its own closed form; it does not
    exercise the BVP solver (see ``test_numerical_bulk_reproduction``).
    """
    u = Usadel1D(DELTA0, D, T=0.0)
    E = np.linspace(-3 * DELTA0, 3 * DELTA0, 1201)
    N = u.dos(E, geometry="bulk", Gamma=0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ana = np.where(np.abs(E) > DELTA0, np.abs(E) / np.sqrt(E**2 - DELTA0**2), 0.0)

    above = np.abs(E) - DELTA0 >= 0.05 * DELTA0
    err_above = _rel(N[above], ana[above])
    assert np.max(err_above) < 1e-3, f"A1 above-gap rel err {np.max(err_above)}"

    below = np.abs(E) < DELTA0
    assert np.max(N[below]) < 1e-6, f"A1 inside-gap DOS {np.max(N[below])}"


def test_a2_gap_equation() -> None:
    """A2: T->0 ratio 1.764, Delta(T) monotonic, Delta(Tc)=0."""
    T_small = 1e-3 * TC
    ratio = bcs_gap(TC, T_small) / (KB * TC)
    assert abs(ratio - 1.764) < 0.01, f"A2 ratio {ratio}"

    ts = np.array([0.05, 0.2, 0.4, 0.6, 0.8, 0.95])
    gaps = np.array([bcs_gap(TC, t * TC) for t in ts])
    assert np.all(np.diff(gaps) <= 0.0), "A2 Delta(T) not monotonic non-increasing"

    assert bcs_gap(TC, TC) < 1e-3 * (WEAK_COUPLING_RATIO * KB * TC), (
        "A2 Delta(Tc) not vanishing"
    )


def test_a3_self_consistency_analytic_path() -> None:
    """A3 (analytic path): uniform self-consistency converges and stays uniform.

    The uniform gap iteration uses the bulk Matsubara sum (``_gap_map_uniform``,
    not the BVP), and the DOS sub-check uses ``dos(geometry="bulk")`` which
    returns the analytic ``bcs_dos``; neither exercises the BVP solver.
    """
    u = Usadel1D(DELTA0, D, T=0.0, Tc=TC)
    T_check = 1e-3 * TC
    _, delta = u.solve_gap("bulk", T=T_check)

    # Uniform: no spurious spatial structure.
    assert np.max(np.abs(delta - delta[0])) < 1e-12 * DELTA0, "A3 not uniform"
    # Stays near the uniform initial guess Delta0.
    assert abs(delta[0] - DELTA0) / DELTA0 < 1e-4, "A3 drifted from Delta0"
    # One more fixed-point step changes the gap by < 1e-6 * Delta0.
    new = u._gap_map_uniform(delta[0], T_check, TC)
    assert abs(new - delta[0]) < 1e-6 * DELTA0, "A3 iteration not converged"

    # Real-time (retarded) branch on the uniform profile matches analytic BCS
    # (analytic path: geometry="bulk" returns bcs_dos).
    E = np.linspace(-3 * DELTA0, 3 * DELTA0, 601)
    mask = np.abs(E) - DELTA0 >= 0.05 * DELTA0
    N = u.dos(E, geometry="bulk", Gamma=0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ana = np.abs(E) / np.sqrt(E**2 - DELTA0**2)
    assert np.max(_rel(N[mask], ana[mask])) < 1e-3, "A3 real/analytic mismatch"


def test_a4_tunneling_spectrum_analytic_path() -> None:
    """A4 (analytic path): dI/dV reduces to N(eV) at T=0 and equals the convolution.

    All sub-checks here operate on the bulk geometry, so ``dos`` returns the
    analytic ``bcs_dos``; this validates the convolution / broadening APIs, not
    the BVP solver.
    """
    u = Usadel1D(DELTA0, D, T=0.0)
    V = np.linspace(-3 * DELTA0, 3 * DELTA0, 300)
    mask = np.abs(V) - DELTA0 >= 0.05 * DELTA0

    # Gamma -> 0 and T -> 0: dI/dV(V) == N(eV).
    _, d0 = u.tunnel_spectrum(V, T=0.0, Gamma=1e-9)
    N = u.dos(V, geometry="bulk", Gamma=1e-9)
    assert np.max(_rel(d0[mask], N[mask])) < 1e-3, "A4 T=0 spectrum != N(eV)"

    # Finite T: the module's own two APIs (tunnel_spectrum vs dos + fermi
    # derivative) must agree within 1e-3 on the convolution.
    T_fin = 4.0  # K
    _, dT = u.tunnel_spectrum(V, T=T_fin, Gamma=1e-9)
    half = 20.0 * KB * T_fin
    ref = np.empty_like(V)
    for i, Vi in enumerate(V):
        integrand = lambda E, Vi=Vi: (  # noqa: E731
            u.dos(E, geometry="bulk", Gamma=1e-9) * fermi_derivative(E - Vi, T_fin)
        )
        ref[i] = _quad(integrand, Vi - half, Vi + half)
    assert np.max(_rel(dT[mask], ref[mask])) < 1e-3, "A4 finite-T convolution mismatch"

    # Increasing Gamma: V=0 conductance rises monotonically and the sub-gap
    # density of states becomes nonzero.
    gammas = [1e-6, 1e-5, 1e-4, 1e-3]
    v0 = np.array([u.tunnel_spectrum([0.0], T=0.0, Gamma=g)[1][0] for g in gammas])
    assert np.all(np.diff(v0) > 0.0), "A4 dI/dV(0) not monotonic in Gamma"
    assert np.all(v0 > 0.0), "A4 sub-gap dI/dV(0) not nonzero"
    assert bcs_dos(0.5 * DELTA0, DELTA0, 1e-4) > 0.0, "A4 sub-gap DOS stays zero"


def test_a5_sn_bilayer() -> None:
    """A5 (numerical path): S/N bilayer minigap, N->1 far from gap, bulk BCS in deep S."""
    u = Usadel1D(DELTA0, D, T=0.0)
    xi = u.xi
    x_surf = -10.0 * xi  # deep S

    def minigap(d):
        x_n = d  # outer N edge
        Es = np.linspace(0.0, 0.25 * DELTA0, 101)
        N = np.array([u.dos(E, position=x_n, geometry="sn", d=d) for E in Es])
        edge = np.where(N > 0.05)[0]
        if len(edge) == 0:
            return np.nan, Es, N
        return Es[edge[0]], Es, N

    d_small, d_large = 200.0, 400.0
    eg_small, Es_small, N_small = minigap(d_small)
    eg_large, Es_large, N_large = minigap(d_large)

    assert np.isfinite(eg_small) and np.isfinite(eg_large), "A5 no minigap found"

    # Below the minigap the DOS is exponentially small.
    for d, Es, N, eg in [
        (d_small, Es_small, N_small, eg_small),
        (d_large, Es_large, N_large, eg_large),
    ]:
        below = Es < eg
        assert np.max(N[below]) < 0.05, f"A5 minigap not clean for d={d}"
        # Sample points well inside the gap.
        for frac in (0.0, 0.5):
            N_mid = u.dos(frac * eg, position=d, geometry="sn", d=d)
            assert N_mid < 0.05, f"A5 N({frac}*Eg)={N_mid} for d={d}"

    # Eg increases as d decreases (d ratio = 2, direction must be correct).
    assert eg_small > eg_large, (
        f"A5 Eg(d={d_small})={eg_small} <= Eg(d={d_large})={eg_large}"
    )

    # Far above the gap the DOS returns to the normal metal value 1 (+-2%).
    for d in (d_small, d_large):
        N_far = u.dos(2.0 * DELTA0, position=d, geometry="sn", d=d)
        assert abs(N_far - 1.0) < 0.02, f"A5 N(E=2D0)={N_far} for d={d}"

    # Deep S recovers the bulk BCS form: relative error outside the gap,
    # absolute agreement (both ~0) inside the gap.
    for E in (1.1 * DELTA0, 2.0 * DELTA0, 3.0 * DELTA0):
        N_s = u.dos(E, position=x_surf, geometry="sn", d=d_small)
        N_bulk = bcs_dos(E, DELTA0, 1e-9)
        assert _rel(N_s, N_bulk) < 1e-2, f"A5 deep-S mismatch at E={E}"
    for E in (0.3 * DELTA0, 0.5 * DELTA0, 0.8 * DELTA0):
        N_s = u.dos(E, position=x_surf, geometry="sn", d=d_small)
        N_bulk = bcs_dos(E, DELTA0, 1e-9)
        assert abs(N_s - N_bulk) < 1e-3, f"A5 deep-S sub-gap mismatch at E={E}"


def test_numerical_bulk_reproduction() -> None:
    """Numerical path: the retarded Riccati BVP reproduces the bulk BCS DOS.

    Drives ``solve_retarded_sn`` (via ``dos(geometry="sn")``) in a deep S region
    of an S/N bilayer (``L_S = 30*xi``, evaluated at ``x = -15*xi``) and compares
    against the analytic BCS form.  Mirrors verification report V1b: gates
    ``|E-Delta| >= 0.02*Delta`` above the gap (rel err < 2e-3) plus the sub-gap
    point ``E = 0.5*Delta`` (abs err < 1e-3).  This is the check that turns red
    if the Riccati solver is broken (see the t4 delivery note).
    """
    u = Usadel1D(DELTA0, D, T=0.0)
    xi = u.xi
    L_S = 30.0 * xi
    pos = -15.0 * xi
    d = 200.0

    for r in (1.02, 1.05, 1.1, 1.2, 1.5, 2.0, 3.0):
        E = r * DELTA0
        Nn = u.dos(
            E, position=pos, geometry="sn", d=d, L_S=L_S, Gamma=1e-9, n_nodes=600
        )
        Na = bcs_dos(E, DELTA0, 1e-9)
        err = _rel(Nn, Na)
        assert err < 2e-3, f"numerical bulk err {err} at E={r}*Delta0"

    Nn = u.dos(
        0.5 * DELTA0, position=pos, geometry="sn", d=d, L_S=L_S, Gamma=1e-9, n_nodes=600
    )
    Na = bcs_dos(0.5 * DELTA0, DELTA0, 1e-9)
    assert abs(Nn - Na) < 1e-3, "numerical bulk sub-gap mismatch"


def _quad(f, lo, hi):
    """Tiny wrapper around scipy quad (kept local to this check)."""
    from scipy.integrate import quad as _scipy_quad

    return _scipy_quad(f, lo, hi, epsabs=1e-10, epsrel=1e-8, limit=400)[0]


def main() -> None:
    tests = [
        test_a1_bulk_bcs_dos_analytic_path,
        test_a2_gap_equation,
        test_a3_self_consistency_analytic_path,
        test_a4_tunneling_spectrum_analytic_path,
        test_a5_sn_bilayer,
        test_numerical_bulk_reproduction,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"\nAll {len(tests)} Usadel regression checks passed.")


if __name__ == "__main__":
    main()
