"""Green's function computation module for Wannier90 MLWF Hamiltonian."""

from typing import Any

from stm_data_processing.config import get_xp
from stm_data_processing.dft.wannier90.mlwf_hamiltonian import MLWFHamiltonian


class GreenFunction:
    """
    Green's function calculator for MLWF Hamiltonian.

    Computes G(k, ω) = (ω + iη - H(k))⁻¹ using CPU (NumPy) or GPU (CuPy).

    Notes
    -----
    - Backend is determined by current backend setting from config.
    - GPU path keeps data on device unless explicitly requested otherwise.
    - Supports batched k-point evaluation.
    """

    def __init__(
        self,
        mlwf_hamiltonian: MLWFHamiltonian,
        eta: float = 0.001,
    ) -> None:
        """
        Initialize GreenFunction with a Hamiltonian.

        Parameters
        ----------
        mlwf_hamiltonian : MLWFHamiltonian
            The Hamiltonian to compute Green's function from.
        eta : float
            Broadening parameter (imaginary part of energy).
        """
        self.ham: MLWFHamiltonian = mlwf_hamiltonian
        self.eta: float = float(eta)
        self.num_wann: int = self.ham.num_wann

    @property
    def xp(self):
        return get_xp()

    # ============================================================
    # Core Green's function computation
    # ============================================================
    def compute_green(
        self,
        hk: Any,
        omega: float,
    ) -> Any:
        """
        Compute Retarded Green's function from Hamiltonian.

        Parameters
        ----------
        hk : ndarray
            Hamiltonian with shape (N, nw, nw) or (nw, nw)
        omega : float
            Energy value

        Returns
        -------
        ndarray
            Green's function with same shape and backend as hk
        """
        xp = get_xp()

        nw = self.num_wann
        z = omega + 1j * self.eta
        eye = xp.eye(nw, dtype=xp.complex128)

        hk = xp.asarray(hk, dtype=xp.complex128)

        if hk.ndim == 2:
            return xp.linalg.solve(z * eye - hk, eye)

        eye_batch = xp.broadcast_to(eye, hk.shape)
        return xp.linalg.solve(z * eye_batch - hk, eye_batch)
