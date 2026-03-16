import logging
from typing import Literal, Optional

# ============================================================
# Module-level state
# ============================================================
try:
    import cupy as _cupy_module

    _CUPY_IMPORT_OK = True
except ImportError:
    _cupy_module = None
    _CUPY_IMPORT_OK = False

# Always import numpy as fallback
import numpy as _numpy_module


def _cupy_usable() -> bool:
    """Return True if CuPy is importable AND a CUDA device seems usable."""
    if not _CUPY_IMPORT_OK or _cupy_module is None:
        return False
    try:
        ndev = int(_cupy_module.cuda.runtime.getDeviceCount())
        return ndev > 0
    except Exception:
        return False


def _detect_backend() -> Literal["cpu", "gpu"]:
    """Decide backend at import time. Prefer GPU if available."""
    return "gpu" if _cupy_usable() else "cpu"


# Global backend state
BACKEND: Literal["cpu", "gpu"] = _detect_backend()

logger = logging.getLogger(__name__)
logger.info(f"[Config] backend={BACKEND} (cupy_usable={_cupy_usable()})")


# ============================================================
# Backend switching API
# ============================================================
def set_backend(backend: Literal["cpu", "gpu", "auto"]) -> None:
    """
    Manually set the computation backend.

    ⚠️ IMPORTANT: Call this BEFORE importing any computation modules
    (e.g., mlwf_hamiltonian, qpi_jdos, etc.)

    Parameters
    ----------
    backend : str
        'cpu' - Force CPU backend (NumPy)
        'gpu' - Force GPU backend (CuPy), falls back to CPU if unavailable
        'auto' - Auto-detect based on CuPy availability

    Examples
    --------
    >>> from stm_data_processing.config import set_backend
    >>> set_backend("cpu")  # Force CPU
    >>> from stm_data_processing.dft.wannier90 import MLWFHamiltonian
    >>> # Now MLWFHamiltonian will use CPU

    >>> set_backend("gpu")  # Try GPU
    >>> from stm_data_processing.stm import qpi_jdos
    >>> # Now qpi_jdos will use GPU (if available)
    """
    global BACKEND

    if backend == "auto":
        BACKEND = _detect_backend()
        logger.info(f"[Config] backend set to auto -> {BACKEND}")
        return

    if backend not in ("cpu", "gpu"):
        raise ValueError(f"backend must be 'cpu', 'gpu', or 'auto', got '{backend}'")

    if backend == "gpu" and not _cupy_usable():
        logger.warning(
            "[Config] GPU requested but CuPy is not usable. Falling back to CPU."
        )
        BACKEND = "cpu"
    else:
        BACKEND = backend

    logger.info(f"[Config] backend manually set to {BACKEND}")


def get_backend() -> Literal["cpu", "gpu"]:
    """Get current backend setting."""
    return BACKEND


def get_xp():
    """
    Get the computation backend module (cupy or numpy).

    ✅ This is the RECOMMENDED way to get the backend module.
    It always returns the correct module based on current BACKEND setting.

    Returns
    -------
    module
        cupy if backend is 'gpu' and available, otherwise numpy

    Examples
    --------
    >>> from stm_data_processing.config import get_xp
    >>> xp = get_xp()
    >>> array = xp.array([1, 2, 3])  # Works on both CPU and GPU
    """
    if BACKEND == "gpu" and _cupy_usable():
        return _cupy_module
    return _numpy_module


def get_cupy() -> Optional:
    """
    Get CuPy module if available and backend is GPU.

    ⚠️ Deprecated: Use get_xp() instead for new code.

    Returns
    -------
    cupy module or None
        cupy if backend is 'gpu' and usable, None otherwise
    """
    if BACKEND == "gpu" and _cupy_usable():
        return _cupy_module
    return None


def is_gpu_available() -> bool:
    """Check if GPU backend is currently usable."""
    return BACKEND == "gpu" and _cupy_usable()


def get_backend_status() -> dict:
    """
    Get detailed backend status information.

    Returns
    -------
    dict
        Dictionary containing:
        - 'backend': Current backend ('cpu' or 'gpu')
        - 'cupy_importable': Whether CuPy can be imported
        - 'cuda_devices': Number of available CUDA devices
        - 'cupy_version': CuPy version string (if available)
    """
    status = {
        "backend": BACKEND,
        "cupy_importable": _CUPY_IMPORT_OK,
        "cuda_devices": 0,
        "cupy_version": None,
    }

    if _CUPY_IMPORT_OK and _cupy_module is not None:
        try:
            status["cuda_devices"] = int(_cupy_module.cuda.runtime.getDeviceCount())
            status["cupy_version"] = _cupy_module.__version__
        except Exception:
            pass

    return status


# ============================================================
# Backend-aware computation helpers
# ============================================================
class BackendArray:
    """
    Helper class for backend-aware array operations.

    This class provides a unified interface for array operations
    that work on both CPU (NumPy) and GPU (CuPy).

    Examples
    --------
    >>> from stm_data_processing.config import BackendArray
    >>> ba = BackendArray()
    >>> arr = ba.array([1, 2, 3])
    >>> result = ba.sum(arr)
    """

    def __init__(self, backend: Literal["cpu", "gpu", "auto"] = "auto"):
        if backend == "auto":
            self.backend = BACKEND
        else:
            self.backend = backend
        self.xp = get_xp() if self.backend == "gpu" else _numpy_module

    def array(self, data, **kwargs):
        """Create an array on the current backend."""
        return self.xp.array(data, **kwargs)

    def asarray(self, data, **kwargs):
        """Convert data to array on the current backend."""
        return self.xp.asarray(data, **kwargs)

    def to_cpu(self, arr):
        """Convert array to NumPy (no-op if already on CPU)."""
        if _cupy_module is not None and isinstance(arr, _cupy_module.ndarray):
            return _cupy_module.asnumpy(arr)
        return arr

    def to_gpu(self, arr):
        """Convert array to CuPy (no-op if already on GPU)."""
        if _cupy_module is None:
            raise RuntimeError("CuPy not available")
        return _cupy_module.asarray(arr)

    def __getattr__(self, name):
        """Delegate unknown attributes to the backend module."""
        return getattr(self.xp, name)
