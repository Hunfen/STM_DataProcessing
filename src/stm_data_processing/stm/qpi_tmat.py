"""T-matrix QPI calculation for a single scalar (s-wave) impurity.

Formalism
---------
For a scalar s-wave impurity of strength V0 (impurity potential V = V0 * I):

    G_loc(omega) = (1/N) sum_k G(k, omega)
    T(omega)     = V * [1 - G_loc(omega) * V]^-1
    rho(q,omega) = -(1/pi) * Im sum_k Tr[G(k, omega) * T(omega) * G(k + q, omega)]

The CPU implementation mirrors stm/qpi_born.py: G(k, omega) is obtained from
dft.wannier90.mlwf_gk.GreenFunction and the q-space sum is evaluated for all
q-points at once with the FFT correlation trick used there. The GPU branch is
intentionally not implemented yet; it raises a descriptive NotImplementedError
while keeping the same method signatures.
"""

import logging
from typing import Any

import numpy as np

from stm_data_processing.config import BACKEND
from stm_data_processing.dft.wannier90.mlwf_gk import GreenFunction
from stm_data_processing.dft.wannier90.mlwf_hamiltonian import MLWFHamiltonian
from stm_data_processing.utils.miscellaneous import extend_qpi, frac_to_real_2d

logger = logging.getLogger(__name__)


class TmatQPI:
    """QPI calculator for a single scalar (s-wave) impurity.

    Uses the T-matrix (full impurity resummation) approximation. ``V`` is the
    scalar s-wave impurity strength in eV; the impurity potential matrix is
    V0 * I_{num_wann x num_wann}.
    """

    def __init__(
        self,
        hamiltonian: MLWFHamiltonian,
        nk: int = 128,
        eta: float = 0.001,
        V: float = 0.1,
    ) -> None:
        """Initialize the T-matrix QPI calculator.

        Parameters
        ----------
        hamiltonian : MLWFHamiltonian
            Hamiltonian object used to evaluate H(k).
        nk : int
            Number of k-points per axis (nk x nk grid).
        eta : float
            Broadening parameter (imaginary part of the energy).
        V : float
            Scalar s-wave impurity strength in eV.
        """
        self._validate_hamiltonian(hamiltonian)
        self.ham: MLWFHamiltonian = hamiltonian
        self.num_wann: int = hamiltonian.num_wann
        self.nk: int = int(nk)
        self.eta: float = float(eta)
        self.V0: complex = complex(V)
        self.V: np.ndarray = self.V0 * np.eye(self.num_wann, dtype=np.complex128)
        self.gf: GreenFunction = GreenFunction(hamiltonian, eta=eta)

        k_vals = np.linspace(-0.5, 0.5, self.nk, endpoint=False)
        self.k1_grid, self.k2_grid = np.meshgrid(k_vals, k_vals, indexing="ij")
        self.q1_grid, self.q2_grid = self.k1_grid.copy(), self.k2_grid.copy()

        k_points = np.column_stack(
            [
                self.k1_grid.ravel(),
                self.k2_grid.ravel(),
                np.zeros(self.nk * self.nk, dtype=np.float64),
            ]
        )
        self.hk_grid: np.ndarray = self.ham.hk(k_points).reshape(
            self.nk, self.nk, self.num_wann, self.num_wann
        )

    def _validate_hamiltonian(self, hamiltonian: MLWFHamiltonian) -> None:
        """Validate that the Hamiltonian is properly initialized.

        Parameters
        ----------
        hamiltonian : MLWFHamiltonian
            Hamiltonian object to validate.

        Raises
        ------
        ValueError
            If the Hamiltonian is missing or has an invalid num_wann.
        """
        if not hasattr(hamiltonian, "num_wann") or hamiltonian.num_wann is None:
            raise ValueError("Invalid MLWFHamiltonian: num_wann is not initialized.")
        if hamiltonian.num_wann <= 0:
            raise ValueError(
                f"Invalid MLWFHamiltonian: num_wann must be positive, "
                f"got {hamiltonian.num_wann}."
            )

    # ============================================================
    # Compute core math (CPU)
    # ============================================================

    def _compute_tmat(self, omega: float) -> np.ndarray:
        """Compute the T-matrix QPI map on the full (nk, nk) q grid (CPU).

        Parameters
        ----------
        omega : float
            Energy value in eV.

        Returns
        -------
        np.ndarray
            QPI map of shape (nk, nk), FFT-shifted so q = 0 is at the centre.
            The 1/pi prefactor of rho(q, omega) is applied by ``calculate``.
        """
        nk = self.nk
        nw = self.num_wann

        logger.info(
            f"  [CPU] Computing T-matrix QPI at omega = {omega:.4f} eV (nk={nk})..."
        )

        g_k = self.gf.compute_green(self.hk_grid, omega)  # (nk, nk, nw, nw)

        # G_loc(omega) = (1/N) sum_k G(k, omega)
        g_loc = g_k.mean(axis=(0, 1))  # (nw, nw)

        # T(omega) = V [1 - G_loc(omega) V]^-1 with V = V0 * I
        v_mat = self.V0 * np.eye(nw, dtype=np.complex128)
        t_mat = v_mat @ np.linalg.inv(np.eye(nw, dtype=np.complex128) - g_loc @ v_mat)

        # rho(q, omega) = -(1/pi) Im sum_k Tr[G(k) T G(k+q)]
        # FFT correlation over orbital blocks (same approach as BornQPI):
        #   IFFT( FFT(A*)* * FFT(B) ) = sum_k A[k] * B[k+q]
        # with A = G*T and B = G.
        g_t = np.einsum("ijab,bc->ijac", g_k, t_mat, optimize=True)
        qpi: np.ndarray = np.zeros((nk, nk), dtype=np.float64)
        for c in range(nw):
            fft_g_t_c = np.conj(np.fft.fftn(np.conj(g_t[:, :, :, c]), axes=(0, 1)))
            fft_g_c = np.fft.fftn(g_k[:, :, c, :], axes=(0, 1))
            corr_c = np.fft.ifftn(fft_g_t_c * fft_g_c, axes=(0, 1))
            qpi += -np.imag(np.sum(corr_c, axis=2))

        logger.info(f"  [CPU] Done (nk={nk}).")
        return np.fft.fftshift(qpi)

    def _compute_tmat_cuda(self, omega: float) -> np.ndarray:
        """CUDA path placeholder.

        Parameters
        ----------
        omega : float
            Energy value in eV.

        Raises
        ------
        NotImplementedError
            Always; the GPU branch is not implemented yet.
        """
        raise NotImplementedError(
            "TmatQPI GPU path is not implemented yet. "
            "Use the CPU backend (config.set_backend('cpu')) or call "
            "_compute_tmat directly."
        )

    # ============================================================
    # Public API
    # ============================================================

    def calculate(
        self,
        energy_range: float | np.ndarray | list[float],
        q_range: tuple[float, float] | None = (-0.5, 0.5),
        V: float | complex | None = None,
        output_path: str | None = None,
    ) -> dict[str, Any]:
        """Compute T-matrix QPI maps over an energy range.

        Parameters
        ----------
        energy_range : float or array-like
            Energy value(s) in eV.
        q_range : tuple[float, float] | None
            Fractional q-range used for grid extension/cropping, as in
            ``BornQPI.calculate``.
        V : float or complex, optional
            Scalar s-wave impurity strength in eV. Overrides the constructor
            value when given.
        output_path : str | None
            Accepted for interface parity with ``BornQPI``; not used yet.

        Returns
        -------
        dict
            Dictionary with keys:
            - 'qpi_layers': (n_omega, nk, nk) float64 array
            - 'q1_grid', 'q2_grid': fractional q grids
            - 'qx_grid', 'qy_grid': real-space q grids in 1/Angstrom
              (None when the Hamiltonian has no bvecs)
            - 'metadata': dict with calculation settings
        """
        nk = self.nk
        energy_array: np.ndarray = np.asarray(energy_range, dtype=np.float64).ravel()

        if V is not None:
            if not np.isscalar(V):
                raise ValueError(
                    f"V must be a scalar impurity strength, got {V!r}"
                )
            self.V0 = complex(V)
            self.V = self.V0 * np.eye(self.num_wann, dtype=np.complex128)

        logger.info(
            "[INFO] Starting T-matrix QPI calculation on %s (nk=%d, n_omega=%d)",
            BACKEND,
            nk,
            len(energy_array),
        )

        compute_func = self._compute_tmat_cuda if BACKEND == "gpu" else self._compute_tmat

        qpi_layers: np.ndarray = np.empty((len(energy_array), nk, nk), dtype=np.float64)
        for ie, omega in enumerate(energy_array):
            logger.info(
                "[%d/%d] Energy: %.4f eV", ie + 1, len(energy_array), float(omega)
            )
            qpi_layers[ie] = compute_func(float(omega)) / np.pi

        if q_range is not None:
            qpi_layers_ext, q1_grid_ext, q2_grid_ext = extend_qpi(
                qpi_layers,
                self.q1_grid,
                self.q2_grid,
                q_range[0],
                q_range[1],
            )
        else:
            qpi_layers_ext = qpi_layers
            q1_grid_ext = self.q1_grid
            q2_grid_ext = self.q2_grid

        qx_grid, qy_grid = frac_to_real_2d(q1_grid_ext, q2_grid_ext, self.ham.bvecs)

        metadata = {
            "module_type": "tmat",
            "eta": self.eta,
            "normalize": True,
            "nq": nk,
            "energy_range": energy_array,
            "bands": None,
            "bvecs": self.ham.bvecs,
            "V": self.V,
            "mask": None,
        }

        result: dict[str, Any] = {
            "qpi_layers": qpi_layers_ext,
            "q1_grid": q1_grid_ext,
            "q2_grid": q2_grid_ext,
            "qx_grid": qx_grid,
            "qy_grid": qy_grid,
            "metadata": metadata,
        }

        logger.info("T-matrix QPI calculation completed.")
        return result

