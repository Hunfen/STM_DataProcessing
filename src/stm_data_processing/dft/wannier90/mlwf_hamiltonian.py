from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from stm_data_processing.utils.lattice_loader import LatticeLoader

# ============================================================
# Optional CuPy (only used if backend involves GPU)
# ============================================================
try:
    import cupy as cp  # type: ignore

    _CUPY_IMPORT_OK = True
except Exception:
    cp = None  # type: ignore
    _CUPY_IMPORT_OK = False


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default).strip()


def _cupy_usable() -> bool:
    """Return True if CuPy is importable AND a CUDA device seems usable."""
    if not _CUPY_IMPORT_OK or cp is None:
        return False
    try:
        # This can fail if driver is missing/blocked.
        ndev = int(cp.cuda.runtime.getDeviceCount())
        return ndev > 0
    except Exception:
        return False


_CUPY_USABLE = _cupy_usable()


def _detect_backend() -> str:
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

    # auto
    return "gpu" if _CUPY_USABLE else "cpu"


_BACKEND = _detect_backend()


def _gpu_return_mode() -> str:
    """
    MLWF_GPU_RETURN = numpy | cupy (default numpy)
    Controls return type when backend uses GPU.
    """
    mode = _env("MLWF_GPU_RETURN", "numpy").lower()
    if mode not in {"numpy", "cupy"}:
        raise ValueError("MLWF_GPU_RETURN must be one of: numpy|cupy")
    return mode


def _maybe_print_backend_once() -> None:
    """
    Print backend once for debug if MLWF_BACKEND_VERBOSE=1.
    """
    verbose = _env("MLWF_BACKEND_VERBOSE", "0")
    if verbose in {"1", "true", "t", "yes", "y"}:
        msg = f"[MLWFHamiltonian] backend={_BACKEND} (cupy_usable={_CUPY_USABLE})"
        print(msg)


_maybe_print_backend_once()


class MLWFHamiltonian:
    """
    Wannier90 HR Hamiltonian handler.

    Public API
    ----------
    load_from_seedname(folder, seedname)
    hk(k_frac)
    hk_batch(k_list)

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

    def __init__(self, folder: str | None = None, seedname: str | None = None):
        # Backend (fixed at import)
        self._backend = _BACKEND

        # Model data (CPU canonical storage)
        self.num_wann: int | None = None
        self.r_list: np.ndarray | None = None  # (nrpts, 3) int32
        self.h_list: np.ndarray | None = None  # (nrpts, nw, nw) complex128
        self.ndegen: np.ndarray | None = None  # (nrpts,) float64
        self.h_list_flat: np.ndarray | None = None  # (nrpts, nw*nw) complex128

        # Lattice info (optional)
        self.folder: str | None = None
        self.seedname: str | None = None
        self.bvecs: np.ndarray | None = None

        # GPU caches (optional)
        self._r_list_gpu = None
        self._ndegen_gpu = None
        self._h_flat_gpu = None  # (nrpts, nw*nw)

        if folder is not None and seedname is not None:
            self.load_from_seedname(folder, seedname)

    # ============================================================
    # Public API
    # ============================================================

    def load_from_seedname(self, folder: str, seedname: str) -> None:
        """Load {seedname}_hr.dat and lattice info (bvecs) if available."""
        folder_p = Path(folder)
        hr_file = folder_p / f"{seedname}_hr.dat"
        if not hr_file.exists():
            raise FileNotFoundError(f"hr.dat file not found: {hr_file}")

        print(f"[MLWFHamiltonian] Loading HR data from: {hr_file}")

        self.folder = str(folder_p)
        self.seedname = seedname

        # Load lattice (optional)
        self.bvecs = self._try_load_bvecs(folder_p, seedname)

        # Load HR (CPU canonical)
        self._load_hr_file(hr_file)

        # Reset GPU cache
        self._clear_gpu_cache()

        print(
            f"[MLWFHamiltonian] Loaded HR with num_wann={self.num_wann}, "
            f"nrpts={len(self.r_list)} (backend={self._backend})"
        )

    def hk(
        self, k_frac: tuple[float, float, float] | np.ndarray
    ) -> np.ndarray | cp.ndarray:
        """
        Compute H(k) in Wannier basis.

        k_frac can be:
          - shape (3,)
          - shape (N,3)
          - shape (...,3)

        Returns:
          - if input is single k: (nw, nw)
          - if input is batched: (..., nw, nw)
        """
        k2d, _, _ = self._as_k_array(k_frac)
        num_k = k2d.shape[0]

        print(
            f"[MLWFHamiltonian] Computing H(k) for {num_k} k-point(s) "
            f"using backend='{self._backend}'"
        )

        if self._backend == "gpu":
            return self._hk_gpu(k_frac)

        return self._hk_cpu(k_frac)

    # ============================================================
    # I/O
    # ============================================================

    def _try_load_bvecs(self, folder: Path, seedname: str) -> np.ndarray | None:
        wout_file = folder / f"{seedname}.wout"
        out_file = folder / f"{seedname}.out"

        lattice_obj = None
        if wout_file.exists():
            lattice_obj = LatticeLoader.create_lattice(filename=wout_file)
        elif out_file.exists():
            lattice_obj = LatticeLoader.create_lattice(filename=out_file)

        return None if lattice_obj is None else lattice_obj.bvecs

    def _load_hr_file(self, filename: str | Path) -> None:
        """
        Parse Wannier90 seedname_hr.dat.

        Stores:
          - num_wann
          - r_list: (nrpts, 3)
          - h_list: (nrpts, nw, nw)
          - ndegen: (nrpts,) float64
          - h_list_flat: (nrpts, nw*nw)
        """
        filename = Path(filename)

        with filename.open("r") as f:
            first_line = f.readline()
            first_stripped = first_line.strip().lower()

            # wannier90 may write a header like: "written on ..."
            if first_stripped.startswith("written on"):
                num_wann = int(f.readline().split()[0])
                nrpts = int(f.readline().split()[0])
            else:
                num_wann = int(first_line.split()[0])
                nrpts = int(f.readline().split()[0])

            ndegen: list[int] = []
            while len(ndegen) < nrpts:
                line = f.readline()
                if not line:
                    raise RuntimeError("Unexpected EOF while reading ndegen list.")
                ndegen.extend([int(x) for x in line.split()])
            ndegen_arr = np.array(ndegen[:nrpts], dtype=np.float64)

            r_list: list[tuple[int, int, int]] = []
            h_list: list[np.ndarray] = []

            current_r: tuple[int, int, int] | None = None
            current_h: np.ndarray | None = None

            nlines = nrpts * num_wann * num_wann
            for _ in range(nlines):
                parts = f.readline().split()
                if len(parts) < 7:
                    raise RuntimeError("Unexpected EOF or malformed hr.dat line.")

                r1, r2, r3 = map(int, parts[:3])
                m, n = map(int, parts[3:5])
                re, im = map(float, parts[5:7])

                r = (r1, r2, r3)
                if current_r is None or r != current_r:
                    if current_r is not None and current_h is not None:
                        r_list.append(current_r)
                        h_list.append(current_h)

                    current_r = r
                    current_h = np.zeros((num_wann, num_wann), dtype=np.complex128)

                # hr.dat uses 1-based indices
                i = m - 1
                j = n - 1
                current_h[i, j] = re + 1j * im

            if current_r is not None and current_h is not None:
                r_list.append(current_r)
                h_list.append(current_h)

        if len(r_list) != nrpts:
            raise RuntimeError(
                f"nrpts mismatch: read {len(r_list)} r-points, expected {nrpts}"
            )

        self.num_wann = num_wann
        self.r_list = np.array(r_list, dtype=np.int32)  # (nrpts, 3)
        self.h_list = np.stack(h_list, axis=0).astype(np.complex128)  # (nrpts, nw, nw)
        self.ndegen = ndegen_arr

        # Flatten H(R) to use GEMM-style contraction
        self.h_list_flat = self.h_list.reshape(nrpts, num_wann * num_wann)

    def _check_loaded(self) -> None:
        if (
            self.num_wann is None
            or self.r_list is None
            or self.ndegen is None
            or self.h_list_flat is None
        ):
            raise RuntimeError("Hamiltonian not loaded. Call load_from_seedname first.")

    # ============================================================
    # Core math (CPU/GPU vectorized)
    # ============================================================

    def _hk_cpu(self, k_frac: tuple[float, float, float] | np.ndarray) -> np.ndarray:
        """
        CPU vectorized:
          phases = exp(2j*pi * (k @ r^T))          (N,nr)
          weights = phases / ndegen[None,:]        (N,nr)
          hk_flat = weights @ h_flat               (N,nw*nw)
          hk = hk_flat.reshape(N,nw,nw)
        """
        self._check_loaded()

        k2d, is_single, prefix = self._as_k_array(k_frac)

        r = self.r_list.astype(np.float64, copy=False)  # (nr,3)
        dot = k2d @ r.T  # (N,nr)
        phases = np.exp(2j * np.pi * dot)  # (N,nr)
        weights = phases / self.ndegen[None, :]  # (N,nr)

        hk_flat = weights @ self.h_list_flat  # (N, nw*nw)
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
        GPU vectorized with GEMM-style contraction.
        Return type controlled by MLWF_GPU_RETURN.
        """
        self._ensure_gpu_cache()

        # Normalize input on CPU then move to GPU
        k2d_cpu, is_single, prefix = self._as_k_array(k_frac)
        k2d = cp.asarray(k2d_cpu)  # (N,3) float64

        # dot: (N,nr)
        # r_list_gpu: (nr,3)
        dot = k2d @ self._r_list_gpu.T  # (N,nr)
        phases = cp.exp(2j * cp.pi * dot)  # (N,nr) complex
        weights = phases / self._ndegen_gpu[None, :]  # (N,nr) complex

        # "single big op": GEMM (N,nr) @ (nr,nw*nw) -> (N,nw*nw)
        hk_flat = weights @ self._h_flat_gpu  # (N,nw*nw)
        nw = int(self.num_wann)
        hk = hk_flat.reshape(-1, nw, nw)

        if is_single:
            hk = hk[0]
        elif prefix:
            hk = hk.reshape(*prefix, nw, nw)

        if _gpu_return_mode() == "cupy":
            return hk
        return cp.asnumpy(hk)

    def _ensure_gpu_cache(self) -> None:
        if not _CUPY_USABLE or cp is None:
            raise RuntimeError("GPU path requested but CuPy/CUDA device is not usable.")
        self._check_loaded()

        if self._r_list_gpu is None:
            self._r_list_gpu = cp.asarray(self.r_list)  # int32 ok
        if self._ndegen_gpu is None:
            # use complex for division safety
            self._ndegen_gpu = cp.asarray(self.ndegen, dtype=cp.complex128)
        if self._h_flat_gpu is None:
            self._h_flat_gpu = cp.asarray(self.h_list_flat)  # complex128

    def _clear_gpu_cache(self) -> None:
        self._r_list_gpu = None
        self._ndegen_gpu = None
        self._h_flat_gpu = None

    @staticmethod
    def _as_k_array(
        k_frac: tuple[float, float, float] | np.ndarray,
    ) -> tuple[np.ndarray, bool, tuple[int, ...]]:
        """
        Normalize k input to (N,3) float64 array.
        Returns (k2d, is_single, batch_shape_prefix).
        """
        k = np.asarray(k_frac, dtype=np.float64)
        if k.shape == (3,):
            return k.reshape(1, 3), True, ()
        if k.ndim >= 2 and k.shape[-1] == 3:
            prefix = k.shape[:-1]
            return k.reshape(-1, 3), False, prefix
        raise ValueError("k_frac must have shape (3,) or (...,3)")
