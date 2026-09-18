"""FFT stage: windowed, plane-subtracted, fftshifted complex spectrum."""

from __future__ import annotations

import numpy as np

from .preprocess import fill_nan_with_plane, validate_image, window_array
from .preprocess import subtract_plane as _subtract_plane

__all__ = ["compute_fft2", "magnitude", "q_axis_limits", "radius_map"]


def compute_fft2(
    image: np.ndarray,
    size_nm: float,
    *,
    window: str | np.ndarray | None = "hann",
    subtract_plane: bool = True,
    nan_policy: str = "plane",
) -> np.ndarray:
    """Return ``fftshift(fft2(windowed, plane-subtracted image))``."""
    arr = validate_image(image, size_nm)
    arr = fill_nan_with_plane(arr, nan_policy)
    if subtract_plane:
        arr = _subtract_plane(arr)
    arr = arr * window_array(window, arr.shape[0])
    return np.fft.fftshift(np.fft.fft2(arr))


def magnitude(spectrum: np.ndarray) -> np.ndarray:
    """Modulus of a complex fftshifted spectrum as float64."""
    return np.abs(np.asarray(spectrum))


def radius_map(n: int) -> np.ndarray:
    """Radius in FFT pixels of every sample, measured from the DC centre."""
    axis = np.arange(n) - n // 2
    qx, qy = np.meshgrid(axis, axis)
    return np.hypot(qx, qy)


def q_axis_limits(n: int, size_nm: float) -> tuple[float, float]:
    """Return ``(dq_nm_inv, q_nyquist_nm_inv)`` for an n x n image of size_nm."""
    dq_nm_inv = 2.0 * np.pi / float(size_nm)
    return dq_nm_inv, np.pi * n / float(size_nm)
