"""Wannier90 HR Hamiltonian computation module."""

import logging
import os
from typing import Literal

import numpy as np

from stm_data_processing.io.w90hr_loader import Wannier90HRLoader

logger = logging.getLogger(__name__)

# ============================================================
# Optional CuPy (only used if backend involves GPU)
# ============================================================
try:
    import cupy as cp

    _CUPY_IMPORT_OK = True
except ImportError:
    cp = None
    _CUPY_IMPORT_OK = False


# ============================================================
# Backend configuration
# ============================================================


def _env(key: str, default: str) -> str:
    """Get environment variable with default value."""
    return os.environ.get(key, default).strip()


def _cupy_usable() -> bool:
    """Return True if CuPy is importable AND a CUDA device seems usable."""
    if not _CUPY_IMPORT_OK or cp is None:
        return False
    try:
        ndev = int(cp.cuda.runtime.getDeviceCount())  # type: ignore[union-attr]
        return ndev > 0
    except Exception:
        return False


_CUPY_USABLE = _cupy_usable()


def _detect_backend() -> Literal["cpu", "gpu"]:
    """
    Decide backend at import time.

    MLWF_BACKEND = cpu | gpu | auto (default auto)

    auto rule:
      - gpu if CuPy usable
      - else cpu
    """
    mode = _env("MLWF_BACKEND", "auto").lower()
    if mode not in {"auto", "cpu", "gpu"}:
        raise ValueError("MLWF_BACKEND must be one of: auto|cpu|gpu")

    if mode == "cpu":
        return "cpu"
    if mode == "gpu":
        if not _CUPY_USABLE:
            raise RuntimeError(
                "MLWF_BACKEND=gpu requested but CuPy/CUDA device is not usable."
            )
        return "gpu"

    return "gpu" if _CUPY_USABLE else "cpu"


_BACKEND = _detect_backend()


def _gpu_return_mode() -> Literal["numpy", "cupy"]:
    """
    Get GPU return mode from environment.

    MLWF_GPU_RETURN = numpy | cupy (default numpy)
    Controls return type when backend uses GPU.
    """
    mode = _env("MLWF_GPU_RETURN", "numpy").lower()
    if mode not in {"numpy", "cupy"}:
        raise ValueError("MLWF_GPU_RETURN must be one of: numpy|cupy")
    return mode


def _maybe_print_backend_once() -> None:
    """Print backend once for debug if MLWF_BACKEND_VERBOSE=1."""
    verbose = _env("MLWF_BACKEND_VERBOSE", "0")
    if verbose in {"1", "true", "t", "yes", "y"}:
        msg = f"[MLWFHamiltonian] backend={_BACKEND} (cupy_usable={_CUPY_USABLE})"
        logger.info(msg)


_maybe_print_backend_once()


# ============================================================
# Hamiltonian computation class
# ============================================================
class MLWFHamiltonian:
    """
    Wannier90 HR Hamiltonian handler.

    This class focuses on computing H(k) from pre-loaded Wannier data.
    For loading data from files, use `from_seedname()`.

    Public API
    ----------
    from_seedname(folder, seedname) : class method to load and construct
    hk(k_frac) : compute H(k) for single or batched k-points

    Backend selection (import-time)
    --------------------------------
    MLWF_BACKEND = cpu | gpu | auto (default auto)
      auto: gpu if CuPy usable; else cpu

    GPU return type
    ---------------
    MLWF_GPU_RETURN = numpy | cupy (default numpy)

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
        backend: Literal["cpu", "gpu"] | None = None,
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
        backend : Literal["cpu", "gpu"] | None
            Compute backend. If None, uses globally detected backend.
        """
        self._backend = backend if backend is not None else _BACKEND

        self.num_wann = num_wann
        self.r_list = r_list
        self.h_list_flat = h_list_flat
        self.ndegen = ndegen
        self.bvecs = bvecs

        self._r_list_gpu: cp.ndarray | None = None
        self._ndegen_gpu: cp.ndarray | None = None
        self._h_flat_gpu: cp.ndarray | None = None

        self._validate_data()

    @classmethod
    def from_seedname(
        cls, folder: str, seedname: str, backend: Literal["cpu", "gpu"] | None = None
    ) -> "MLWFHamiltonian":
        """
        Construct MLWFHamiltonian by loading data from Wannier90 files.

        Parameters
        ----------
        folder : str
            Directory containing the HR file.
        seedname : str
            Base name of the Wannier90 files (without extension).
        backend : Literal["cpu", "gpu"] | None
            Compute backend. If None, uses globally detected backend.

        Returns
        -------
        MLWFHamiltonian
            Initialized Hamiltonian handler.
        """
        data = Wannier90HRLoader.load(folder, seedname)
        instance = cls(
            num_wann=data["num_wann"],
            r_list=data["r_list"],
            h_list_flat=data["h_list_flat"],
            ndegen=data["ndegen"],
            bvecs=data["bvecs"],
            backend=backend,
        )
        # Store metadata for downstream I/O
        instance.folder = folder
        instance.seedname = seedname
        return instance

    def _validate_data(self) -> None:
        """Validate internal data shapes and types."""
        nrpts = len(self.r_list)
        expected_flat_size = self.num_wann * self.num_wann

        if self.r_list.shape != (nrpts, 3):
            raise ValueError(f"r_list shape mismatch: {self.r_list.shape}")
        if self.h_list_flat.shape != (nrpts, expected_flat_size):
            raise ValueError(
                f"h_list_flat shape mismatch: {self.h_list_flat.shape}, "
                f"expected ({nrpts}, {expected_flat_size})"
            )
        if self.ndegen.shape != (nrpts,):
            raise ValueError(f"ndegen shape mismatch: {self.ndegen.shape}")

    # ============================================================
    # Public API: Computation
    # ============================================================

    def hk(
        self, k_frac: tuple[float, float, float] | np.ndarray
    ) -> np.ndarray | cp.ndarray:
        """
        Compute H(k) in Wannier basis.

        Parameters
        ----------
        k_frac : tuple[float, float, float] | np.ndarray
            Fractional k-point coordinates. Can be:
            - shape (3,) for single k-point
            - shape (N, 3) for batch
            - shape (..., 3) for arbitrary batch dimensions

        Returns
        -------
        np.ndarray | cp.ndarray
            Hamiltonian matrix at k. Shape:
            - (num_wann, num_wann) for single k-point
            - (..., num_wann, num_wann) for batched input

        Raises
        ------
        RuntimeError
            If GPU backend is requested but CuPy is not available.
        """
        k2d, _, _ = self._as_k_array(k_frac)
        num_k = k2d.shape[0]

        logger.info(
            "[MLWFHamiltonian] Computing H(k) for %d k-point(s) using backend='%s'",
            num_k,
            self._backend,
        )

        if self._backend == "gpu":
            return self._hk_gpu(k_frac)

        return self._hk_cpu(k_frac)

    # ============================================================
    # Core math (CPU/GPU vectorized)
    # ============================================================

    def _hk_cpu(self, k_frac: tuple[float, float, float] | np.ndarray) -> np.ndarray:
        """
        CPU vectorized H(k) computation.

        phases = exp(2j*pi * (k @ r^T))          (N,nr)
        weights = phases / ndegen[None,:]        (N,nr)
        hk_flat = weights @ h_flat               (N,nw*nw)
        hk = hk_flat.reshape(N,nw,nw)
        """
        k2d, is_single, prefix = self._as_k_array(k_frac)

        r = self.r_list.astype(np.float64, copy=False)
        dot = k2d @ r.T
        phases = np.exp(2j * np.pi * dot)
        weights = phases / self.ndegen[None, :]

        hk_flat = weights @ self.h_list_flat
        nw = int(self.num_wann)
        hk = hk_flat.reshape(-1, nw, nw)

        if is_single:
            return hk[0]
        if prefix:
            return hk.reshape(*prefix, nw, nw)
        return hk

    def _hk_gpu(
        self, k_frac: tuple[float, float, float] | np.ndarray
    ) -> np.ndarray | cp.ndarray:
        """
        GPU vectorized H(k) computation with GEMM-style contraction.

        Return type controlled by MLWF_GPU_RETURN.
        """
        self._ensure_gpu_cache()

        k2d_cpu, is_single, prefix = self._as_k_array(k_frac)
        k2d = cp.asarray(k2d_cpu)  # type: ignore[union-attr]

        dot = k2d @ self._r_list_gpu.T  # type: ignore[union-attr]
        phases = cp.exp(2j * cp.pi * dot)
        weights = phases / self._ndegen_gpu[None, :]  # type: ignore[union-attr]

        hk_flat = weights @ self._h_flat_gpu  # type: ignore[union-attr]
        nw = int(self.num_wann)
        hk = hk_flat.reshape(-1, nw, nw)

        if is_single:
            hk = hk[0]
        elif prefix:
            hk = hk.reshape(*prefix, nw, nw)

        if _gpu_return_mode() == "cupy":
            return hk
        return cp.asnumpy(hk)  # type: ignore[union-attr]

    def _ensure_gpu_cache(self) -> None:
        """Ensure GPU caches are populated."""
        if not _CUPY_USABLE or cp is None:
            raise RuntimeError("GPU path requested but CuPy/CUDA device is not usable.")
        if self._r_list_gpu is None:
            self._r_list_gpu = cp.asarray(self.r_list)  # type: ignore[union-attr]
        if self._ndegen_gpu is None:
            self._ndegen_gpu = cp.asarray(  # type: ignore[union-attr]
                self.ndegen, dtype=cp.complex128
            )
        if self._h_flat_gpu is None:
            self._h_flat_gpu = cp.asarray(self.h_list_flat)  # type: ignore[union-attr]

    def _clear_gpu_cache(self) -> None:
        """Clear GPU-side cached arrays."""
        self._r_list_gpu = None
        self._ndegen_gpu = None
        self._h_flat_gpu = None

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
