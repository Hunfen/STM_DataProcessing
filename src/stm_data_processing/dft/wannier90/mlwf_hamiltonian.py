"""Wannier90 HR Hamiltonian computation module."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from stm_data_processing.config import get_backend
from stm_data_processing.io.w90hr_loader import Wannier90HRLoader

logger = logging.getLogger(__name__)


class MLWFHamiltonian:
    """
    Wannier90 HR Hamiltonian handler.

    This class focuses on computing H(k) from pre-loaded Wannier data.
    For loading data from files, use `from_seedname()`.

    Public API
    ----------
    from_seedname(folder, seedname) : class method to load and construct
    from_arrays(num_wann, r_list, h_list_flat, ndegen, bvecs) : construct from arrays
    hk(k_frac) : compute H(k) for single or batched k-points

    Notes
    -----
    H(k) = sum_R exp(2π i R·k) / ndegen(R) * H(R)

    Implementation uses a flattened contraction:
      weights(k,R) @ H_flat(R, nw*nw)  -> H_flat(k, nw*nw) -> reshape (nw,nw)
    On GPU this becomes a single GEMM-style operation (plus phase building).
    """

    def __init__(
        self,
        num_wann: int,
        r_list: np.ndarray,
        h_list_flat: np.ndarray,
        ndegen: np.ndarray,
        bvecs: np.ndarray | None = None,
    ):
        """
        Initialize MLWFHamiltonian with pre-loaded data.

        Parameters
        ----------
        num_wann : int
            Number of Wannier functions.
        r_list : np.ndarray
            Lattice vector indices, shape (nrpts, 3), dtype int32.
        h_list_flat : np.ndarray
            Flattened Hamiltonian matrices, shape (nrpts, num_wann*num_wann),
            dtype complex128.
        ndegen : np.ndarray
            Degeneracy factors, shape (nrpts,), dtype float64.
        bvecs : np.ndarray | None
            Reciprocal lattice vectors, shape (3, 3), optional.
        """
        self._validate_data(num_wann, r_list, h_list_flat, ndegen)

        self._num_wann = num_wann
        self._r_list = r_list
        self._h_list_flat = h_list_flat
        self._ndegen = ndegen
        self._bvecs = bvecs

        # ✅ GPU cache - lazily initialized
        self._r_list_gpu: Any | None = None
        self._ndegen_gpu: Any | None = None
        self._h_flat_gpu: Any | None = None

        self._folder: str | None = None
        self._seedname: str | None = None

        logger.info(f"[MLWFHamiltonian] Initialized with backend={get_backend()}")

    # ============================================================
    # Read-only properties
    # ============================================================
    @property
    def num_wann(self) -> int:
        """Number of Wannier functions (read-only)."""
        return self._num_wann

    @property
    def r_list(self) -> np.ndarray:
        """Lattice vector indices, shape (nrpts, 3) (read-only)."""
        return self._r_list

    @property
    def h_list_flat(self) -> np.ndarray:
        """Flattened Hamiltonian matrices (read-only)."""
        return self._h_list_flat

    @property
    def ndegen(self) -> np.ndarray:
        """Degeneracy factors, shape (nrpts,) (read-only)."""
        return self._ndegen

    @property
    def bvecs(self) -> np.ndarray | None:
        """Reciprocal lattice vectors, shape (3, 3) (read-only)."""
        return self._bvecs

    @property
    def folder(self) -> str | None:
        """Source folder (if loaded from file, read-only)."""
        return self._folder

    @property
    def seedname(self) -> str | None:
        """Source seedname (if loaded from file, read-only)."""
        return self._seedname

    @classmethod
    def from_seedname(cls, folder: str, seedname: str) -> MLWFHamiltonian:
        """
        Construct MLWFHamiltonian by loading data from Wannier90 files.
        """
        data = Wannier90HRLoader.load(folder, seedname)
        instance = cls(
            num_wann=data["num_wann"],
            r_list=data["r_list"],
            h_list_flat=data["h_list_flat"],
            ndegen=data["ndegen"],
            bvecs=data["bvecs"],
        )
        instance._folder = folder
        instance._seedname = seedname
        return instance

    @classmethod
    def from_arrays(
        cls,
        num_wann: int,
        r_list: np.ndarray,
        h_list_flat: np.ndarray,
        ndegen: np.ndarray,
        bvecs: np.ndarray | None = None,
    ) -> MLWFHamiltonian:
        """
        Construct MLWFHamiltonian from raw arrays.
        """
        return cls(
            num_wann=num_wann,
            r_list=r_list,
            h_list_flat=h_list_flat,
            ndegen=ndegen,
            bvecs=bvecs,
        )

    # ============================================================
    # Public API: Computation
    # ============================================================
    def hk(self, k_frac: np.ndarray) -> Any:
        """
        Compute H(k) in Wannier basis.

        Parameters
        ----------
        k_frac : np.ndarray
            Fractional k-point coordinates, shape (N, 3).
            For single k-point, use shape (1, 3).

        Returns
        -------
        np.ndarray | cupy.ndarray
            Hamiltonian matrix at k, shape (N, num_wann, num_wann).
            Type matches current backend (numpy for CPU, cupy for GPU).
        """
        if k_frac.ndim != 2 or k_frac.shape[-1] != 3:
            raise ValueError(f"k_frac must have shape (N, 3), got {k_frac.shape}")

        num_k = k_frac.shape[0]
        backend = get_backend()
        logger.info(
            f"[MLWFHamiltonian] Computing H(k) for {num_k} k-point(s) using backend='{backend}'"
        )

        if backend == "gpu":
            return self._hk_gpu(k_frac)
        return self._hk_cpu(k_frac)

    def clear_gpu_cache(self) -> None:
        """Clear GPU cache. Call this after switching backend."""
        self._r_list_gpu = None
        self._ndegen_gpu = None
        self._h_flat_gpu = None
        logger.info("[MLWFHamiltonian] GPU cache cleared.")

    # ============================================================
    # Core math (CPU/GPU vectorized)
    # ============================================================
    def _hk_cpu(self, k_frac: np.ndarray) -> np.ndarray:
        """
        CPU vectorized H(k) computation.
        """
        r = self.r_list.astype(np.float64, copy=False)
        dot = k_frac @ r.T
        phases = np.exp(2j * np.pi * dot)
        weights = phases / self.ndegen[None, :]

        hk_flat = weights @ self.h_list_flat
        nw = int(self.num_wann)
        return hk_flat.reshape(-1, nw, nw)

    def _hk_gpu(self, k_frac: np.ndarray) -> Any:
        """
        GPU vectorized H(k) computation with GEMM-style contraction.
        """
        # ✅ 动态获取 cupy
        import cupy as cp

        self._ensure_gpu_cache()

        k2d = cp.asarray(k_frac)

        dot = k2d @ self._r_list_gpu.T
        phases = cp.exp(2j * cp.pi * dot)
        weights = phases / self._ndegen_gpu[None, :]

        hk_flat = weights @ self._h_flat_gpu
        nw = int(self.num_wann)
        return hk_flat.reshape(-1, nw, nw)

    # ============================================================
    # Helper
    # ============================================================
    def _ensure_gpu_cache(self) -> None:
        """Ensure GPU caches are populated."""
        import cupy as cp

        if self._r_list_gpu is None:
            self._r_list_gpu = cp.asarray(self.r_list)
        if self._ndegen_gpu is None:
            self._ndegen_gpu = cp.asarray(self.ndegen, dtype=cp.complex128)
        if self._h_flat_gpu is None:
            self._h_flat_gpu = cp.asarray(self.h_list_flat)

    @staticmethod
    def _validate_data(
        num_wann: int,
        r_list: np.ndarray,
        h_list_flat: np.ndarray,
        ndegen: np.ndarray,
    ) -> None:
        """
        Validate input data shapes and types.
        """
        if num_wann is None or num_wann <= 0:
            raise ValueError(f"num_wann must be positive, got {num_wann}.")
        if r_list is None:
            raise ValueError("r_list cannot be None.")
        if h_list_flat is None:
            raise ValueError("h_list_flat cannot be None.")
        if ndegen is None:
            raise ValueError("ndegen cannot be None.")

        nrpts = len(r_list)
        expected_flat_size = num_wann * num_wann

        if r_list.shape != (nrpts, 3):
            raise ValueError(f"r_list shape mismatch: {r_list.shape}")
        if h_list_flat.shape != (nrpts, expected_flat_size):
            raise ValueError(
                f"h_list_flat shape mismatch: {h_list_flat.shape}, "
                f"expected ({nrpts}, {expected_flat_size})"
            )
        if ndegen.shape != (nrpts,):
            raise ValueError(f"ndegen shape mismatch: {ndegen.shape}")

    @staticmethod
    def _as_k_array(
        k_frac: tuple[float, float, float] | np.ndarray,
    ) -> tuple[np.ndarray, bool, tuple[int, ...]]:
        """
        Normalize k input to (N, 3) float64 array.

        Parameters
        ----------
        k_frac : tuple[float, float, float] | np.ndarray
            Input k-point coordinates.

        Returns
        -------
        tuple[np.ndarray, bool, tuple[int, ...]]
            (k2d, is_single, prefix) where:
            - k2d : np.ndarray (N, 3) reshaped array
            - is_single : bool True if input was a single k-point
            - prefix : tuple Original batch shape (excluding last dimension)
        """
        k = np.asarray(k_frac, dtype=np.float64)
        if k.shape == (3,):
            return k.reshape(1, 3), True, ()
        if k.ndim >= 2 and k.shape[-1] == 3:
            prefix = k.shape[:-1]
            return k.reshape(-1, 3), False, prefix
        raise ValueError("k_frac must have shape (3,) or (..., 3)")
