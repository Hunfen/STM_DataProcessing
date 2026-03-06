import numpy as np

from .mlwf_hamiltonian import MLWFHamiltonian


class GreenFunction:
    """
    Green function calculator based on MLWF Hamiltonian.

    G(k, ω) = [(ω + iη)I - H(k)]^{-1}
    """

    def __init__(self, hamiltonian: MLWFHamiltonian):
        """
        Parameters
        ----------
        hamiltonian : MLWFHamiltonian
            Instance used to compute H(k)
        """
        self.ham = hamiltonian
        self.num_wann = hamiltonian.num_wann

    def G0(self, k_frac, omega, eta):
        """
        Compute the bare retarded Green's function at a single k-point.

        G₀(k, ω) = [(ω + iη)I - H(k)]⁻¹

        Parameters
        ----------
        k_frac : tuple(float,float,float)
            Fractional k coordinate
        omega : float
            Energy (real part of the complex frequency)
        eta : float
            Infinitesimal broadening (positive, gives imaginary part iη)

        Returns
        -------
        np.ndarray
            Complex-valued array of shape (num_wann, num_wann)
        """

        Hk = self.ham.hk(k_frac)

        identity_matrix = np.eye(self.num_wann)

        return np.linalg.inv((omega + 1j * eta) * identity_matrix - Hk)
