"""Green's function computation module for Wannier90 MLWF Hamiltonian."""

import logging
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from .mlwf_hamiltonian import _CUPY_USABLE, MLWFHamiltonian, cp


class GreenFunction:
    """
    Green's function calculator for MLWF Hamiltonian.

    Computes G(k, ω) = (ω + iη - H(k))⁻¹ using CPU (NumPy) or GPU (CuPy).

    Notes
    -----
    - Backend is inherited from the associated MLWFHamiltonian.
    - GPU path keeps data on device unless explicitly requested otherwise.
    - Supports batched k-point evaluation.
    """

    def __init__(
        self,
        hamiltonian: MLWFHamiltonian,
        eta: float = 0.001,
    ) -> None:
        """
        Initialize GreenFunction with a Hamiltonian.

        Parameters
        ----------
        hamiltonian : MLWFHamiltonian
            The Hamiltonian to compute Green's function from.
        eta : float
            Broadening parameter (imaginary part of energy).
        """
        self.ham: MLWFHamiltonian = hamiltonian
        self.eta: float = float(eta)
        self.num_wann: int | None = hamiltonian.num_wann
        self._backend: Literal["cpu", "gpu"] = hamiltonian._backend

        if self.num_wann is None:
            raise RuntimeError("Hamiltonian not loaded: num_wann is None.")

        self.logger: logging.Logger = logging.getLogger(__name__)

    # ============================================================
    # Backend helpers
    # ============================================================

    def _use_cuda(self) -> bool:
        """Check if CUDA backend is active."""
        return _CUPY_USABLE and cp is not None and self._backend == "gpu"

    # ============================================================
    # Core Green's function computation
    # ============================================================

    def _green_from_hk_cpu(
        self,
        hk: NDArray[np.complex128],
        omega: float,
    ) -> NDArray[np.complex128]:
        """
        Compute Green's function from Hamiltonian on CPU.

        Parameters
        ----------
        hk : NDArray[np.complex128]
            Hamiltonian with shape (..., nw, nw)
        omega : float
            Energy value

        Returns
        -------
        NDArray[np.complex128]
            Green's function with same shape as hk
        """
        nw = self.num_wann
        z = omega + 1j * self.eta
        eye = np.eye(nw, dtype=np.complex128)

        hk = np.asarray(hk, dtype=np.complex128)

        if hk.ndim == 2:
            return np.linalg.solve(z * eye - hk, eye)

        eye_batch = np.broadcast_to(eye, hk.shape)
        return np.linalg.solve(z * eye_batch - hk, eye_batch)

    def _green_from_hk_gpu(
        self,
        hk_gpu: Any,
        omega: float,
    ) -> Any:
        """
        Compute Green's function from Hamiltonian on GPU.

        Parameters
        ----------
        hk_gpu : cupy.ndarray
            Hamiltonian with shape (..., nw, nw)
        omega : float
            Energy value

        Returns
        -------
        cupy.ndarray
            Green's function with same shape as hk_gpu
        """
        if not _CUPY_USABLE or cp is None:
            raise RuntimeError("GPU path requested but CuPy/CUDA is not available.")

        nw = self.num_wann
        z = omega + 1j * self.eta
        eye = cp.eye(nw, dtype=cp.complex128)

        hk_gpu = cp.asarray(hk_gpu, dtype=cp.complex128)

        if hk_gpu.ndim == 2:
            return cp.linalg.solve(z * eye - hk_gpu, eye)

        eye_batch = cp.broadcast_to(eye, hk_gpu.shape)
        return cp.linalg.solve(z * eye_batch - hk_gpu, eye_batch)

    # ============================================================
    # Grid-based Green's function (for QPI)
    # ============================================================

    def get_g0_grid(
        self,
        k_points: NDArray[np.float64],
        omega: float,
    ) -> NDArray[np.complex128] | Any:
        """
        Compute G(k, ω) on a k-point grid.

        Parameters
        ----------
        k_points : NDArray[np.float64]
            k-points with shape (N, 3)
        omega : float
            Energy value

        Returns
        -------
        NDArray[np.complex128] | cupy.ndarray
            Green's function grid with shape (nk, nk, nw, nw) for 2D grid,
            or (N, nw, nw) for arbitrary k-points.
            Returns CuPy array if GPU backend is active.
        """
        nk_sqrt = int(np.sqrt(k_points.shape[0]))
        is_2d_grid = nk_sqrt * nk_sqrt == k_points.shape[0]

        self.logger.info(
            "  [%s] Computing G(k, ω=%.4f) for %d k-points...",
            "GPU" if self._use_cuda() else "CPU",
            omega,
            k_points.shape[0],
        )

        if self._use_cuda():
            return self._get_g0_grid_gpu(
                k_points, omega, nk_sqrt if is_2d_grid else None
            )
        return self._get_g0_grid_cpu(k_points, omega, nk_sqrt if is_2d_grid else None)

    def _get_g0_grid_cpu(
        self,
        k_points: NDArray[np.float64],
        omega: float,
        nk_sqrt: int | None = None,
    ) -> NDArray[np.complex128]:
        """CPU implementation of grid-based Green's function."""
        hk = self.ham.hk(k_points)
        hk = np.asarray(hk, dtype=np.complex128)

        if nk_sqrt is not None:
            hk = hk.reshape(nk_sqrt, nk_sqrt, self.num_wann, self.num_wann)

        return self._green_from_hk_cpu(hk, omega)

    def _get_g0_grid_gpu(
        self,
        k_points: NDArray[np.float64],
        omega: float,
        nk_sqrt: int | None = None,
    ) -> Any:
        """GPU implementation of grid-based Green's function."""
        if not _CUPY_USABLE or cp is None:
            raise RuntimeError("GPU path requested but CuPy/CUDA is not available.")

        hk = self.ham.hk(k_points)
        hk_gpu = cp.asarray(hk, dtype=cp.complex128)

        if nk_sqrt is not None:
            hk_gpu = hk_gpu.reshape(nk_sqrt, nk_sqrt, self.num_wann, self.num_wann)

        return self._green_from_hk_gpu(hk_gpu, omega)
