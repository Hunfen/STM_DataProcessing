import numpy as np

kB = 8.617333262e-5  # eV/K


class BTK:
    """
    Blonder-Tinkham-Klapwijk (BTK) model for tunneling spectroscopy.

    This class provides methods to calculate differential conductance spectra
    at both zero and finite temperatures.
    """

    def __init__(self, Delta, Z, Gamma=1e-6):
        """
        Initialize BTK model with physical parameters.

        Parameters
        ----------
        Delta : float
            Superconducting gap energy in eV.
        Z : float
            Barrier strength parameter (dimensionless).
        Gamma : float, optional
            Broadening parameter in eV (default: 1e-6).
        """
        self.Delta = Delta
        self.Z = Z
        self.Gamma = Gamma

    def sigma_zero_T(self, E):
        """
        Calculate zero-temperature BTK conductance.

        Parameters
        ----------
        E : array_like
            Energy values in eV.

        Returns
        -------
        ndarray
            Normalized conductance sigma(E) at T=0.
        """
        E = np.asarray(E, dtype=float)
        sigma = np.empty_like(E, dtype=float)

        sub = np.abs(E) < self.Delta
        if np.any(sub):
            Es = E[sub]
            denom = Es**2 + (self.Delta**2 - Es**2) * (1 + 2 * self.Z**2) ** 2
            A = self.Delta**2 / denom
            sigma[sub] = 2.0 * A

        sup = ~sub
        if np.any(sup):
            Et = E[sup].astype(np.complex128) + 1j * self.Gamma
            root = np.sqrt(Et**2 - self.Delta**2)
            root = np.where(np.real(Et) >= 0, root, -root)

            u2 = 0.5 * (1 + root / Et)
            v2 = 0.5 * (1 - root / Et)

            gamma = u2 + (u2 - v2) * self.Z**2

            A = (u2 * v2) / (gamma**2)
            B = ((u2 - v2) ** 2 * self.Z**2 * (1 + self.Z**2)) / (gamma**2)

            sigma[sup] = np.real(1 + A - B)

        return sigma

    def sigma_finite_T(self, E, T):
        """
        Calculate finite-temperature BTK conductance via thermal broadening.

        Parameters
        ----------
        E : array_like
            Energy values in eV.
        T : float
            Temperature in Kelvin.

        Returns
        -------
        ndarray
            Normalized conductance sigma(E, T).
        """
        E = np.asarray(E)
        sigma0 = self.sigma_zero_T(E)

        beta = 1 / (kB * T)
        x = beta * E / 2

        exp_term = np.exp(-2 * np.abs(x))
        sech2 = 4 * exp_term / (1 + exp_term) ** 2

        dfdE = (beta / 4) * sech2

        dE = E[1] - E[0]
        dfdE /= np.sum(dfdE) * dE

        sigmaT = np.convolve(sigma0, dfdE, mode="same") * dE

        return sigmaT

    def spectrum(self, E_min=-5, E_max=5, n_points=500, T=0):
        """
        Generate a complete conductance spectrum.

        Parameters
        ----------
        E_min : float, optional
            Minimum energy in eV (default: -5).
        E_max : float, optional
            Maximum energy in eV (default: 5).
        n_points : int, optional
            Number of energy points (default: 500).
        T : float, optional
            Temperature in Kelvin (default: 0).

        Returns
        -------
        tuple
            (E, sigma) arrays for plotting.
        """
        E = np.linspace(E_min, E_max, n_points)

        sigma = self.sigma_zero_T(E) if T == 0 else self.sigma_finite_T(E, T)

        return E, sigma

    def update_params(self, Delta=None, Z=None, Gamma=None):
        """
        Update model parameters.

        Parameters
        ----------
        Delta : float, optional
            New superconducting gap energy.
        Z : float, optional
            New barrier strength.
        Gamma : float, optional
            New broadening parameter.
        """
        if Delta is not None:
            self.Delta = Delta
        if Z is not None:
            self.Z = Z
        if Gamma is not None:
            self.Gamma = Gamma
