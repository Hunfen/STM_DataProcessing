"""Candidate detection on |FFT| (stage 1).

The detection statistic is the **radial** signal-to-noise ratio

    z(r) = (|F(r)| - level(r)) / sigma(r)

where ``level`` and ``sigma`` are the running median and the MAD-derived Rayleigh
scale of the magnitude in 2 px-wide radial bins.  A radial normalisation matters
on real data: the |FFT| background of an STM topograph falls off steeply with |q|
and carries a dense forest of 1/f ridges, so one global noise scale would both
hide the weak outer Bragg spots and let the low-|q| ridges dominate the ranking.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import numpy as np
from scipy.ndimage import maximum_filter

__all__ = ["Candidate", "detect_candidates", "radial_background", "robust_rayleigh_scale"]

logger = logging.getLogger(__name__)

# MAD / sigma for a Rayleigh-distributed |FFT| background.
_MAD_OVER_RAYLEIGH = 0.448641
# Radial bin width (FFT pixels), sub-sampling stride and minimum bin population.
_BACKGROUND_BIN_PX = 2.0
_BACKGROUND_STEP = 4
_MIN_BIN_SAMPLES = 8


class Candidate(NamedTuple):
    """One non-maximum-suppressed |FFT| maximum above ``min_snr``."""

    row: int
    col: int
    snr: float


def robust_rayleigh_scale(magnitude: np.ndarray, band: np.ndarray) -> float:
    """MAD-based Rayleigh scale ``sigma = MAD / 0.448641`` over a boolean band."""
    values = magnitude[band]
    if values.size < 16:
        values = magnitude.ravel()
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad <= 0.0:
        mad = float(np.std(values))
    return 0.0 if mad <= 0.0 else mad / _MAD_OVER_RAYLEIGH


def radial_background(
    magnitude: np.ndarray,
    radius: np.ndarray,
    *,
    bin_px: float = _BACKGROUND_BIN_PX,
    step: int = _BACKGROUND_STEP,
) -> tuple[np.ndarray, np.ndarray]:
    """Robust radial background ``(level, sigma)`` of a |FFT| magnitude."""
    sub_mag = magnitude[::step, ::step].ravel()
    sub_radius = radius[::step, ::step].ravel()
    n_bins = int(np.ceil(float(sub_radius.max()) / bin_px)) + 1
    bins = np.clip((sub_radius / bin_px).astype(int), 0, n_bins - 1)
    order = np.argsort(bins, kind="stable")
    sorted_bins, sorted_mag = bins[order], sub_mag[order]
    counts = np.bincount(sorted_bins, minlength=n_bins)
    starts = np.concatenate([[0], np.cumsum(counts)])
    level, sigma = np.full(n_bins, np.nan), np.full(n_bins, np.nan)
    for index in range(n_bins):
        values = sorted_mag[starts[index] : starts[index + 1]]
        if values.size < _MIN_BIN_SAMPLES:
            continue
        median = float(np.median(values))
        level[index] = median
        sigma[index] = float(np.median(np.abs(values - median))) / _MAD_OVER_RAYLEIGH
    ok = np.isfinite(level) & (sigma > 0.0)
    if not np.any(ok):
        return np.zeros_like(radius), np.zeros_like(radius)
    centres = np.nonzero(ok)[0] * bin_px
    flat = radius.ravel()
    return (
        np.interp(flat, centres, level[ok]).reshape(radius.shape),
        np.interp(flat, centres, sigma[ok]).reshape(radius.shape),
    )


def _half_width(values: np.ndarray, centre: int) -> float:
    """Largest half width at half maximum of a 1-D profile around ``centre``."""
    peak, base = float(values[centre]), float(values.min())
    if peak <= base:
        return 1.0
    half = base + 0.5 * (peak - base)
    widths = []
    for step in (-1, 1):
        previous, distance, index = peak, 0, centre
        while 0 <= index + step < values.size:
            index += step
            distance += 1
            current = float(values[index])
            if current <= half:
                widths.append(distance - 1.0 + (previous - half) / max(previous - current, 1e-30))
                break
            previous = current
        else:
            widths.append(float(distance))
    return float(max(max(widths, default=1.0), 0.5))


def _adaptive_footprint(magnitude: np.ndarray, seeds: list[tuple[int, int]]) -> int:
    """Non-maximum-suppression footprint from the median peak half width."""
    n = magnitude.shape[0]
    widths = []
    for row, col in seeds[:20]:
        window = magnitude[
            max(0, row - 2) : min(n, row + 3), max(0, col - 2) : min(n, col + 3)
        ]
        if window.size < 9:
            continue
        iy, ix = np.unravel_index(np.argmax(window), window.shape)
        widths.append(max(_half_width(window[iy, :], ix), _half_width(window[:, ix], iy)))
    return 3 if not widths else int(np.clip(round(1.5 * float(np.median(widths))), 3, 11))


def detect_candidates(
    magnitude: np.ndarray,
    snr_map: np.ndarray,
    radius: np.ndarray,
    *,
    min_snr: float,
    dc_radius_px: float,
    q_max_px: float,
    max_candidates: int | None,
    footprint: int | None = None,
) -> tuple[list[Candidate], int, int]:
    """Return the +/-q half-plane candidate local maxima, sorted by SNR."""
    seeds = np.nonzero(
        (magnitude == maximum_filter(magnitude, size=3, mode="nearest"))
        & (snr_map >= min_snr)
        & (radius > dc_radius_px)
    )
    seed_list = sorted(
        zip(seeds[0].tolist(), seeds[1].tolist(), strict=True),
        key=lambda rc: -float(magnitude[rc[0], rc[1]]),
    )
    used = int(footprint) if footprint else _adaptive_footprint(magnitude, seed_list)
    mask = (
        (magnitude == maximum_filter(magnitude, size=used, mode="nearest"))
        & (snr_map >= min_snr)
        & (radius > dc_radius_px)
        & (radius <= q_max_px)
    )
    rows, cols = np.nonzero(mask)
    n_total = int(rows.size)
    half_plane = (rows > magnitude.shape[0] // 2) | (
        (rows == magnitude.shape[0] // 2) & (cols > magnitude.shape[0] // 2)
    )
    rows, cols = rows[half_plane], cols[half_plane]
    order = np.argsort(-snr_map[rows, cols], kind="stable")
    candidates = [
        Candidate(int(rows[i]), int(cols[i]), float(snr_map[rows[i], cols[i]])) for i in order
    ]
    if max_candidates is not None:
        candidates = candidates[: int(max_candidates)]
    return candidates, used, n_total
