from itertools import pairwise

import numpy as np
from scipy.integrate import quad

# Fundamental constants (CODATA 2018)
HBAR_Js = 1.054571817e-34  # Reduced Planck constant, J·s
ME_kg = 9.1093837015e-31  # Electron rest mass, kg
EV_TO_J = 1.602176634e-19  # 1 eV in joules
ANGSTROM_TO_M = 1e-10  # 1 Å = 1e-10 m

# Precomputed factor: ħ² / (2 m_e) in eV·Å²
H2OVER2M_EVA2 = (HBAR_Js**2 * 1e20) / (2 * ME_kg * EV_TO_J)

# Boltzmann constant in eV/K
KB_EVK = 8.617333262159999e-5


def free_electron_energy(k: float) -> float:
    """Compute kinetic energy E(k) = ħ²k²/(2m) in eV for k in Å⁻¹."""
    return H2OVER2M_EVA2 * k * k


def fermi_dirac_from_energy(e: float, mu: float, t: float) -> float:
    """Numerically stable Fermi-Dirac distribution f(E)."""
    if t <= 0.0:
        return 1.0 if e < mu else 0.0
    x = (e - mu) / (KB_EVK * t)
    if x > 50.0:
        return np.exp(-x)
    elif x < -50.0:
        return 1.0
    else:
        return 1.0 / (1.0 + np.exp(x))


class Lindhard1DFreeElectron:
    """
    Compute the 1D Lindhard function χ(q, ω+iη) for a free electron gas.

    The response function is defined as:
        χ(q, ω+iη) = ∫ dk [f(E_k) - f(E_{k+q})] / [E_k - E_{k+q} + ω + iη]

    Parameters
    ----------
    q_max : float
        Maximum q value (q ∈ [0, q_max]) in Å⁻¹.
    k_f : float
        Fermi wavevector in Å⁻¹.
    temperature : float
        Temperature in Kelvin.
    omega : float
        Energy transfer in eV.
    eta : float
        Broadening parameter in eV (η > 0).
    q_points : int, optional
        Number of q points (default: 200).
    """

    def __init__(
        self,
        q_max: float,
        k_f: float,
        temperature: float,
        omega: float,
        eta: float,
        q_points: int = 200,
    ):
        if q_max <= 0:
            raise ValueError("q_max must be positive.")
        if eta <= 0:
            raise ValueError("eta must be positive.")
        if q_points < 2:
            raise ValueError("q_points must be at least 2.")

        self.q_max = q_max
        self.k_f = k_f
        self.temperature = temperature
        self.omega = omega
        self.eta = eta
        self.q_points = q_points

        # Generate q array
        self.q_array = np.linspace(0.0, q_max, q_points)

        # Precompute Lindhard function values
        self.chi_array = np.array(
            [self._compute_lindhard_q(q) for q in self.q_array], dtype=complex
        )

    def _compute_lindhard_q(self, q: float) -> complex:
        """Compute χ(q, ω+iη) for a single q."""
        a = H2OVER2M_EVA2
        mu = a * self.k_f * self.k_f

        if abs(q) < 1e-14:
            q = 1e-14

        def integrand(k_val: float) -> complex:
            ek = a * k_val * k_val
            ekq = a * (k_val + q) * (k_val + q)
            fk = fermi_dirac_from_energy(ek, mu, self.temperature)
            fkq = fermi_dirac_from_energy(ekq, mu, self.temperature)
            denom = ek - ekq + self.omega + 1j * self.eta
            return (fk - fkq) / denom

        dk_scale = np.sqrt(max(KB_EVK * max(self.temperature, 1e-12), self.eta) / a)
        k_max = max(2.0, 6.0 * (abs(self.k_f) + abs(q)) + 20.0 * dk_scale)

        k_pole = self.omega / (2.0 * a * q) - q / 2.0
        points = [
            -k_max,
            -self.k_f,
            self.k_f,
            -q - self.k_f,
            -q + self.k_f,
            k_pole,
            k_max,
        ]

        # Use set comprehension and unpacking
        points = sorted({p for p in points if -k_max < p < k_max})
        points = [-k_max, *points, k_max]

        # Remove duplicates within tolerance
        cleaned = [points[0]]
        for p in points[1:]:
            if abs(p - cleaned[-1]) > 1e-12:
                cleaned.append(p)
        points = cleaned

        real_part = 0.0
        imag_part = 0.0

        # Use itertools.pairwise
        for a_seg, b_seg in pairwise(points):
            if b_seg - a_seg < 1e-12:
                continue
            r, _ = quad(
                lambda x: np.real(integrand(x)),
                a_seg,
                b_seg,
                limit=400,
                epsabs=1e-10,
                epsrel=1e-8,
            )
            im, _ = quad(
                lambda x: np.imag(integrand(x)),
                a_seg,
                b_seg,
                limit=400,
                epsabs=1e-10,
                epsrel=1e-8,
            )
            real_part += r
            imag_part += im

        return real_part + 1j * imag_part
