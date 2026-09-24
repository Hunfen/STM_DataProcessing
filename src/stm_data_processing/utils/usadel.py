"""One-dimensional steady-state Usadel equation solver (dirty limit).

This module implements the diffusive-limit (Usadel) superconducting gap and
tunneling-spectroscopy model described in ``docs/design/usadel_gap_model.md``
(v1.1).  It follows the *symmetric tau1* convention of that document:

    Delta_hat = Delta * tau1
    G_hat     = cos(theta) * tau3 + sin(theta) * tau1

so that the normalization ``g^2 + f^2 = 1`` holds and the Matsubara ``theta``
is real.  The two solved branches are

* Matsubara (real ``theta``), for the self-consistent gap::

      hbar*D * theta'' = 2*omega_n*sin(theta) - 2*Delta(x)*cos(theta)

  with the weak-coupling ``ln(Tc/T)`` regularization of document section 4.1.
* Real retarded (complex ``theta``), for the local density of states::

      hbar*D * theta'' + 2i*E*sin(theta) + 2*Delta(x)*cos(theta) = 0

  with ``N(E, x) = Re cos(theta^R(E, x))`` and the retarded branch
  ``E -> E + i0+`` handled explicitly.

Supported geometries:

* ``"bulk"`` -- a homogeneous superconductor (``Delta(x) = Delta0``).
* ``"sn"``   -- a 1D S/N bilayer: ``Delta = Delta0`` in S (``x < 0``) and
  ``Delta = 0`` in N (``0 < x < d``); the S/N interface sits at ``x = 0`` and
  the outer N edge ``x = d`` carries a no-current boundary condition
  (``grad G . n = 0``).

The retarded branch is solved with a Riccati parametrization
``gamma = tan(theta/2)``, which keeps the anomalous amplitude bounded
(``|gamma| <= 1``) and avoids the ``cosh/sinh`` overflow of the naive complex
``theta`` collocation near ``|E| < Delta``.  The S/N interface is represented
as a narrow but smooth ``Delta`` transition (width ``0.05 * xi``); this is
numerically equivalent to the ideal step and lets ``solve_bvp`` place the
collocation points across the jump.  Known limitations: the theta/Riccati
parametrization degenerates at ``theta -> 0`` (normal state) in a *spatially
self-consistent* S/N solve, so that path is kept separate from the bulk gap
equation; no spin, no phase/current, no non-equilibrium distribution function.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad, solve_bvp

__all__ = [
    "HBAR",
    "KB",
    "WEAK_COUPLING_RATIO",
    "Usadel1D",
    "bcs_dos",
    "bcs_gap",
    "fermi_derivative",
]

KB = 8.617333262e-5  # eV / K, matches utils/btk.py
HBAR = 6.582119569e-16  # eV * s

# Weak-coupling BCS ratio Delta0/(kB*Tc) = pi/e^gamma ~= 1.764.
WEAK_COUPLING_RATIO = np.pi / np.e**np.euler_gamma

# Matsubara-frequency cutoff in units of kB*Tc for the gap sums.  The
# ln(Tc/T)-regularized sum converges only logarithmically; 1000 is cheap for a
# flat numpy sum and keeps the T->0 truncation error around 1e-6.
_GAP_OMEGAC = 1000.0


def bcs_dos(energy, Delta, Gamma=0.0):
    """Bulk BCS density of states with optional Dynes broadening.

    Parameters
    ----------
    energy : array_like
        Energy in eV.
    Delta : float
        Superconducting gap in eV.
    Gamma : float, optional
        Dynes broadening in eV (``E -> E + i*Gamma``).

    Returns
    -------
    ndarray
        ``N(E) = Re( E / sqrt(E^2 - Delta^2) )`` with the retarded branch:
        ``|E|/sqrt(E^2-Delta^2)`` above the gap and ``0`` below (``Gamma=0``).
    """
    E = np.asarray(energy, dtype=float)
    Ec = E + 1j * Gamma
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        # Retarded branch of sqrt: negate the principal root for negative real
        # energy so that N(E) = |E|/sqrt(E^2-Delta^2) stays positive.
        root = np.sqrt(Ec**2 - Delta**2)
        root = np.where(np.real(Ec) >= 0, root, -root)
        dos = np.real(Ec / root)
    return np.nan_to_num(dos, nan=0.0, posinf=0.0, neginf=0.0)


def _bcs_gap_ratio(t, omegac=_GAP_OMEGAC):
    """Solve the uniform gap equation for ``Delta/(kB*Tc)`` at ``t = T/Tc``."""
    if t >= 1.0:
        return 0.0
    n_max = max(100, int(np.ceil((omegac / (np.pi * t) - 1.0) / 2.0)))
    n = np.arange(0, n_max + 1, dtype=float)
    an = (2 * n + 1) * np.pi * t  # omega_n / (kB*Tc)
    inv_sum = np.sum(1.0 / an)

    def eq(delta):
        denom = np.sqrt(an**2 + delta**2)
        return 2.0 * np.pi * t * (inv_sum - np.sum(1.0 / denom)) - np.log(1.0 / t)

    lo, hi = 0.0, 2.0
    while eq(hi) < 0 and hi < 4.0:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if eq(mid) < 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def bcs_gap(Tc, T):
    """Bulk weak-coupling self-consistent gap ``Delta(T)`` in eV.

    The ``T -> 0`` limit reproduces ``Delta0 = 1.764 * kB * Tc``.
    """
    Tc = float(Tc)
    T = float(T)
    if T <= 0.0:
        return WEAK_COUPLING_RATIO * KB * Tc
    if Tc <= T:
        return 0.0
    return _bcs_gap_ratio(T / Tc) * KB * Tc


def _tc_from_delta0(Delta0):
    """Weak-coupling ``Tc`` from the T=0 gap ``Delta0``."""
    return float(Delta0) / (WEAK_COUPLING_RATIO * KB)


def fermi_derivative(E, T):
    """Negative energy derivative of the Fermi function ``-df/dE``.

    ``E`` in eV, ``T`` in K; the result integrates to 1.
    """
    E = np.asarray(E, dtype=float)
    if T <= 0:
        raise ValueError("finite temperature required for a smooth Fermi derivative")
    beta = 1.0 / (KB * T)
    x = beta * E
    # -df/dE = beta * e^{beta E} / (1 + e^{beta E})^2 = beta / (4 cosh^2(x/2))
    return beta / (4.0 * np.cosh(x / 2.0) ** 2)


def _gamma_bulk(E, Delta, Gamma=0.0):
    """Bulk retarded Riccati amplitude ``gamma = i*Delta/(root + Ec)``."""
    Ec = np.asarray(E, dtype=complex) + 1j * Gamma
    root = np.sqrt(Ec**2 - Delta**2)
    root = np.where(np.real(Ec) >= 0, root, -root)
    return 1j * Delta / (root + Ec)


def _sn_delta_tilde(x, width):
    """Smooth S/N step profile ``Delta(x)/Delta0`` (1 in S, 0 in N)."""
    return 0.5 * (1.0 - np.tanh(x / width))


def solve_retarded_sn(
    d,
    E,
    D,
    Delta0,
    Gamma=0.0,
    L_S=None,
    interface_width=None,
    n_nodes=300,
):
    """Solve the real retarded branch for the S/N bilayer.

    Parameters
    ----------
    d : float
        Thickness of the normal layer (nm).
    E : float
        Energy in eV.
    D : float
        Diffusion constant (nm^2/s).
    Delta0 : float
        Superconducting gap in S (eV).
    Gamma : float, optional
        Dynes broadening in eV.
    L_S : float, optional
        Superconducting thickness (nm); defaults to ``10 * xi``.
    interface_width : float, optional
        Width of the smooth S/N transition (nm); defaults to ``0.05 * xi``.

    Returns
    -------
    (x, N) : tuple of ndarray
        Positions (nm) and the local density of states ``N(E, x)``.
    """
    xi = np.sqrt(HBAR * D / Delta0)
    if L_S is None:
        L_S = 10.0 * xi
    if interface_width is None:
        interface_width = 0.05 * xi
    X = np.linspace(-L_S, d, n_nodes) / xi
    W = interface_width / xi
    Et = (E + 1j * Gamma) / Delta0
    gb = _gamma_bulk(E, Delta0, Gamma)

    def rhs(X, y):
        g = y[0] + 1j * y[2]
        w = y[1] + 1j * y[3]
        dt = _sn_delta_tilde(X, W)
        gp = w
        wp = 2.0 * g * w**2 / (1.0 + g**2) - 2j * Et * g - dt * (1.0 - g**2)
        return np.vstack([gp.real, wp.real, gp.imag, wp.imag])

    # Initial guess: bulk in S, exponential decay toward the normal state in N.
    guess = np.where(X < 0, gb, gb * np.exp(-X / 3.0))
    y0 = np.vstack([guess.real, np.zeros_like(X), guess.imag, np.zeros_like(X)])

    def bc(ya, yb):
        # Dirichlet bulk at the deep-S end, no-current at the N outer end.
        return np.array([ya[0] - gb.real, ya[2] - gb.imag, yb[1], yb[3]])

    sol = solve_bvp(rhs, bc, X, y0, tol=1e-9, max_nodes=30000)
    if not sol.success:
        raise RuntimeError(f"Usadel retarded solve failed at E={E}: {sol.message}")
    g = sol.y[0] + 1j * sol.y[2]
    cos_theta = (1.0 - g**2) / (1.0 + g**2)
    return sol.x * xi, np.real(cos_theta)


def _matsubara_bulk_theta(omega_n, Delta):
    """Bulk Matsubara ``theta``: ``tan(theta) = Delta/omega_n``."""
    return np.arctan(Delta / omega_n)


def solve_matsubara_sn(
    d,
    omega_n,
    D,
    Delta0,
    delta_x=None,
    L_S=None,
    interface_width=None,
    n_nodes=300,
):
    """Solve the Matsubara branch for a given ``Delta(x)`` profile.

    ``theta`` is real.  Returns ``(x, theta)``.  When ``delta_x`` is None a
    step profile (``Delta0`` in S, 0 in N) is used.
    """
    xi = np.sqrt(HBAR * D / Delta0)
    if L_S is None:
        L_S = 10.0 * xi
    if interface_width is None:
        interface_width = 0.05 * xi
    X = np.linspace(-L_S, d, n_nodes) / xi
    W = interface_width / xi
    wt = omega_n / Delta0

    if delta_x is None:
        delta_vals = _sn_delta_tilde(X, W)
        delta_S = Delta0
    else:
        x_nm = X * xi
        delta_vals = (
            np.interp(x_nm, np.linspace(-L_S, d, len(delta_x)), delta_x) / Delta0
        )
        delta_S = float(delta_x[0])

    # Deep-S Dirichlet value follows the current (possibly self-consistent)
    # gap at the S end, not the fixed Delta0.
    tb = _matsubara_bulk_theta(omega_n, delta_S)

    def dt_of_x(Xq):
        return np.interp(Xq, X, delta_vals)

    def rhs(X, y):
        th, p = y
        dt = dt_of_x(X)
        dth = p
        dp = 2.0 * (wt * np.sin(th) - dt * np.cos(th))
        return np.vstack([dth, dp])

    guess = np.where(X < 0, tb, tb * np.exp(-X / 3.0))
    y0 = np.vstack([guess, np.zeros_like(X)])

    def bc(ya, yb):
        return np.array([ya[0] - tb, yb[1]])

    sol = solve_bvp(rhs, bc, X, y0, tol=1e-9, max_nodes=30000)
    if not sol.success:
        raise RuntimeError(
            f"Usadel Matsubara solve failed at omega_n={omega_n}: {sol.message}"
        )
    return sol.x * xi, sol.y[0]


class Usadel1D:
    """One-dimensional steady-state Usadel solver.

    Follows the draft interface of document section 10.  Units are eV, K and
    nm (consistent with ``utils/btk.py``).  ``D`` may be given in nm^2/s or in
    m^2/s; values below ``1e12`` are interpreted as m^2/s and converted.
    """

    def __init__(self, Delta0, D, T=0.0, Tc=None, Gamma=0.0):
        self.Delta0 = float(Delta0)
        self.D = float(D)
        if self.D < 1e12:  # m^2/s -> nm^2/s
            self.D *= 1e18
        self.T = float(T)
        self.Gamma = float(Gamma)
        self.Tc = float(Tc) if Tc is not None else _tc_from_delta0(self.Delta0)

    # ------------------------------------------------------------------
    # Characteristic scales
    # ------------------------------------------------------------------
    @property
    def xi(self):
        """Dirty-limit coherence length (nm)."""
        return np.sqrt(HBAR * self.D / self.Delta0)

    def thouless(self, L):
        """Thouless energy ``hbar*D/L^2`` (eV) for length ``L`` (nm)."""
        return HBAR * self.D / (float(L) ** 2)

    # ------------------------------------------------------------------
    # Self-consistent gap (Matsubara branch)
    # ------------------------------------------------------------------
    def _gap_map_uniform(self, Delta, T, Tc):
        """One fixed-point step of the uniform gap equation."""
        t = T / Tc
        if t >= 1.0:
            return 0.0
        n_max = max(100, int(np.ceil((_GAP_OMEGAC / (np.pi * t) - 1.0) / 2.0)))
        n = np.arange(0, n_max + 1, dtype=float)
        omega = (2 * n + 1) * np.pi * KB * T
        sum_inv = np.sum(1.0 / omega)
        denom = 2.0 * np.pi * KB * T * sum_inv - np.log(Tc / T)
        f = Delta / np.sqrt(omega**2 + Delta**2)
        return 2.0 * np.pi * KB * T * np.sum(f) / denom

    def solve_gap(
        self,
        geometry="bulk",
        d=None,
        L_S=None,
        T=None,
        n_nodes=300,
        max_iter=500,
        tol=1e-6,
        damping=0.5,
    ):
        """Self-consistently solve the gap profile ``Delta(x)``.

        Parameters
        ----------
        geometry : {"bulk", "sn"}
            ``"bulk"`` is a homogeneous superconductor; ``"sn"`` is the S/N
            bilayer with a self-consistent S region and ``Delta = 0`` in N.
        d : float, optional
            N thickness (nm) for ``geometry="sn"``.
        L_S : float, optional
            S thickness (nm); defaults to ``10 * xi``.
        T : float, optional
            Temperature (K); defaults to ``self.T``.
        max_iter, tol, damping : iteration controls.

        Returns
        -------
        (x, delta_x) : tuple of ndarray
            Positions (nm) and the self-consistent gap profile (eV).
        """
        T = self.T if T is None else float(T)
        Tc = self.Tc
        xi = self.xi

        if geometry == "bulk":
            L = L_S if L_S is not None else 10.0 * xi
            x = np.linspace(0.0, L, n_nodes)
            delta = np.full(n_nodes, self.Delta0)
            for _ in range(max_iter):
                new = self._gap_map_uniform(delta[0], T, Tc)
                new = (1.0 - damping) * delta[0] + damping * new
                change = abs(new - delta[0])
                delta[:] = new
                if change < tol * self.Delta0:
                    break
            return x, delta

        if geometry == "sn":
            if d is None:
                raise ValueError("d (N thickness in nm) is required for geometry='sn'")
            if L_S is None:
                L_S = 10.0 * xi
            x = np.linspace(-L_S, d, n_nodes)
            delta = np.where(x < 0.0, self.Delta0, 0.0)
            # Matsubara frequencies used for the spatial self-consistency.
            t = T / Tc
            omegac = 20.0  # in units of kB*Tc (document section 7.1)
            n_freq = max(4, int(np.ceil((omegac / (np.pi * t) - 1.0) / 2.0)))
            omega_ns = (2 * np.arange(n_freq) + 1) * np.pi * KB * T
            sum_inv = np.sum(1.0 / omega_ns)
            denom = 2.0 * np.pi * KB * T * sum_inv - np.log(Tc / T)
            for _ in range(max_iter):
                f_sum = np.zeros_like(delta)
                for omega_n in omega_ns:
                    x_bvp, theta = solve_matsubara_sn(
                        d,
                        omega_n,
                        self.D,
                        self.Delta0,
                        delta_x=delta,
                        L_S=L_S,
                        n_nodes=n_nodes,
                    )
                    f_sum += np.interp(x, x_bvp, np.sin(theta))
                new = 2.0 * np.pi * KB * T * f_sum / denom
                new = np.where(x < 0.0, new, 0.0)
                change = np.max(np.abs(new - delta))
                delta = (1.0 - damping) * delta + damping * new
                if change < tol * self.Delta0:
                    break
            return x, delta

        raise ValueError(f"unknown geometry {geometry!r}")

    # ------------------------------------------------------------------
    # Local density of states (retarded branch)
    # ------------------------------------------------------------------
    def dos(
        self,
        energy,
        position=0.0,
        geometry="bulk",
        d=None,
        L_S=None,
        Gamma=None,
        n_nodes=300,
    ):
        """Local density of states ``N(E, x)``.

        ``energy`` may be scalar or array-like.  ``position`` is the spatial
        point (nm).  For ``geometry="bulk"`` the analytic bulk result is used;
        for ``geometry="sn"`` the retarded branch is solved on the S/N grid.
        """
        Gamma = self.Gamma if Gamma is None else float(Gamma)
        E = np.asarray(energy, dtype=float)
        scalar = E.ndim == 0
        E = np.atleast_1d(E)

        if geometry == "bulk":
            N = bcs_dos(E, self.Delta0, Gamma)
            return float(N[0]) if scalar else N

        if geometry == "sn":
            if d is None:
                raise ValueError("d (N thickness in nm) is required for geometry='sn'")
            L_S = L_S if L_S is not None else 10.0 * self.xi
            N = np.empty_like(E)
            for i, Ei in enumerate(E):
                x, Nx = solve_retarded_sn(
                    d,
                    Ei,
                    self.D,
                    self.Delta0,
                    Gamma=Gamma,
                    L_S=L_S,
                    n_nodes=n_nodes,
                )
                N[i] = np.interp(position, x, Nx)
            return float(N[0]) if scalar else N

        raise ValueError(f"unknown geometry {geometry!r}")

    # ------------------------------------------------------------------
    # Tunneling spectrum
    # ------------------------------------------------------------------
    def tunnel_spectrum(
        self, V, position=0.0, geometry="bulk", d=None, L_S=None, T=None, Gamma=None
    ):
        """Tunneling spectrum ``dI/dV(V)`` (document section 6.2).

        ``dI/dV(V) ~ int dE N(E, x_surf) * (-df/dE)(E - eV)``.  At ``T = 0``
        this reduces to ``N(eV)``.  For finite ``T`` the convolution is done
        with adaptive quadrature, which resolves the ``1/sqrt`` singularity of
        the gap edge to high accuracy.  Returns ``(V, dIdV)``.
        """
        V = np.asarray(V, dtype=float)
        scalar = V.ndim == 0
        V = np.atleast_1d(V)
        T = self.T if T is None else float(T)
        Gamma = self.Gamma if Gamma is None else float(Gamma)

        if T <= 0:
            dIdV = self.dos(V, position, geometry, d, L_S, Gamma)
            if scalar:
                return float(V[0]), float(dIdV[0])
            return V, dIdV

        half = 20.0 * KB * T
        dIdV = np.empty_like(V)

        def dos_scalar(E):
            return float(self.dos(E, position, geometry, d, L_S, Gamma))

        for i, Vi in enumerate(V):
            integrand = lambda E, Vi=Vi: dos_scalar(E) * fermi_derivative(E - Vi, T)  # noqa: E731
            dIdV[i] = quad(
                integrand, Vi - half, Vi + half, epsabs=1e-10, epsrel=1e-8, limit=400
            )[0]

        if scalar:
            return float(V[0]), float(dIdV[0])
        return V, dIdV
