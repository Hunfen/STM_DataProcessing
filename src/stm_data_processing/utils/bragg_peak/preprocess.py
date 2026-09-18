"""Image loading and real-space pre-processing (stage 0)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

__all__ = [
    "fill_nan_with_plane",
    "load_image",
    "subtract_plane",
    "validate_image",
    "window_array",
]

logger = logging.getLogger(__name__)


def validate_image(image: np.ndarray, size_nm: float) -> np.ndarray:
    """Return ``image`` as a float64 (n, n) array, validating shape and size."""
    arr = np.asarray(image)
    if arr.ndim != 2:
        raise ValueError(f"image must be 2-D, got {arr.ndim} dimension(s)")
    if arr.shape[0] != arr.shape[1]:
        raise ValueError(f"image must be square, got shape {arr.shape}")
    if not np.isfinite(size_nm) or size_nm <= 0:
        raise ValueError(f"size_nm must be a positive number, got {size_nm!r}")
    return np.ascontiguousarray(arr, dtype=np.float64)


def load_image(path: str | Path) -> np.ndarray:
    """Read a plain-text matrix (tab, comma or whitespace separated)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"image file not found: {path}")
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                delimiter = "\t" if "\t" in line else ("," if "," in line else None)
                break
        else:
            raise ValueError(f"{path} contains no data rows")
    data = np.loadtxt(path, delimiter=delimiter, comments="#")
    return np.ascontiguousarray(np.atleast_2d(data), dtype=np.float64)


def _plane_coefficients(data: np.ndarray) -> np.ndarray | None:
    """Least-squares plane ``c0*x + c1*y + c2`` over the finite pixels, or None."""
    finite = np.isfinite(data)
    if int(np.count_nonzero(finite)) < 3:
        return None
    rows, cols = np.nonzero(finite)
    design = np.column_stack([cols, rows, np.ones(rows.size)])
    coeffs, *_ = np.linalg.lstsq(design, data[rows, cols], rcond=None)
    return coeffs


def fill_nan_with_plane(image: np.ndarray, nan_policy: str) -> np.ndarray:
    """Replace non-finite pixels by the best-fit plane value."""
    bad = ~np.isfinite(image)
    n_bad = int(np.count_nonzero(bad))
    if n_bad == 0:
        return image
    if nan_policy == "raise":
        raise ValueError(f"image contains {n_bad} non-finite pixel(s)")
    if nan_policy != "plane":
        raise ValueError(f"nan_policy must be 'plane' or 'raise', got {nan_policy!r}")
    coeffs = _plane_coefficients(image)
    filled = image.copy()
    if coeffs is None:
        logger.warning("bragg_peak: fewer than 3 finite pixels; NaN set to zero")
        filled[bad] = 0.0
        return filled
    logger.warning(
        "bragg_peak: filled %d non-finite pixel(s) with the best-fit plane", n_bad
    )
    rows, cols = np.nonzero(bad)
    filled[rows, cols] = coeffs[0] * cols + coeffs[1] * rows + coeffs[2]
    return filled


def subtract_plane(image: np.ndarray) -> np.ndarray:
    """Subtract the best-fit plane (removes the linear background ramp)."""
    coeffs = _plane_coefficients(image)
    if coeffs is None:
        logger.warning("bragg_peak: fewer than 3 finite pixels; plane not subtracted")
        return image.copy()
    rows, cols = np.meshgrid(
        np.arange(image.shape[0]), np.arange(image.shape[1]), indexing="ij"
    )
    return image - (coeffs[0] * cols + coeffs[1] * rows + coeffs[2])


def window_array(window: str | np.ndarray | None, n: int) -> np.ndarray:
    """Return the (n, n) apodisation window requested by ``window``."""
    if window is None:
        return np.ones((n, n), dtype=np.float64)
    if isinstance(window, np.ndarray):
        arr = np.asarray(window, dtype=np.float64)
        if arr.shape != (n, n):
            raise ValueError(f"window array must have shape {(n, n)}, got {arr.shape}")
        return arr
    name = str(window).lower()
    if name == "hann":
        return np.outer(np.hanning(n), np.hanning(n))
    if name == "hamming":
        return np.outer(np.hamming(n), np.hamming(n))
    if name == "blackman":
        return np.outer(np.blackman(n), np.blackman(n))
    raise ValueError(
        "window must be 'hann', 'hamming', 'blackman', a (n, n) array or None, "
        f"got {window!r}"
    )
