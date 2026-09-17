"""FFT Bragg peak detection, sub-pixel localization and lattice refinement.

This module implements the algorithm stack defined in
``docs/design/bragg_peak_detection.md``:

1. **Detection** -- Hann-windowed ``|FFT|`` with a robust Rayleigh noise scale,
   an SNR threshold, a peak-width-adaptive non-maximum-suppression footprint
   and no hard cap on the number of peaks.
2. **Sub-pixel localization** -- log-magnitude 3x3 parabola seeding followed by
   a bounded rotated 2-D Gaussian least-squares fit with a plane background;
   every peak carries a 1-sigma uncertainty and a 2x2 covariance.
3. **Lattice-constrained refinement** -- a weighted (GLS) fit of the 2x2
   reciprocal basis, a ``predict -> verify -> localize -> refit`` loop, integer
   ``(h, k)`` labels and the affine distortion matrix with its 4x4 covariance.

Conventions (do not mix them up):

* All 2-D vectors are ``(qx, qy)``; array indices are ``[row=y, col=x]``.
* Reciprocal-space units are nm^-1 unless a name ends with ``_px``; ``_px``
  positions are signed ``fftshift`` offsets from the array centre
  ``(n // 2, n // 2)`` with ``q_nm_inv = q_px * 2*pi/size_nm`` and
  ``q_nyquist = (n/2) * 2*pi/size_nm``.
* ``cov_q_px`` is the **total** position covariance of a peak: the least-squares
  fit covariance plus ``sigma_model_floor_px**2`` times the identity, i.e. the
  very matrix the weighted lattice fit uses as its ``Sigma_i``.  The identity
  ``sigma_q_px = sqrt(diag(cov_q_px))`` therefore holds by construction, and the
  floor is never added twice (design spec, sections 4.4(b) and 4.5).
* Only one of each ``+/-q`` pair is fitted independently: ``independent=True``
  marks the half-plane representative and its partner is obtained by mirroring.
* An **oblique** lattice has no canonical first ring, so its index labelling is
  fixed by a gauge choice: with a caller-supplied basis the module enumerates the
  candidate two-point seeds and keeps the **least distorted** one
  (``d(M) = max(|lambda - 1|)`` relative to that basis, recorded together with
  its margin in ``meta['oblique_gauge']`` / ``meta['oblique_gauge_margin']``);
  without a reference basis it seeds from the two shortest strong peaks.  The
  strain of an inferred oblique lattice is a tautology (its reference is built
  from the seed vectors), so ``strain_gauge`` is ``"undefined"`` there and
  ``principal_stretches`` / ``linearized_strain`` / ``rotation_deg`` are None.
* Every peak reports which localization tier produced it through ``method``
  (``gaussian`` / ``parabola_log`` / ``max_pixel``) and its status through
  ``quality`` (``ok`` / ``coarse`` / ``edge`` / ``unmatched`` / ``failed``); no
  guarantee is made that every peak has a sub-pixel solution -- steep-background
  or boundary candidates legitimately degrade to the integer maximum, and their
  sigma is then the 1/sqrt(12) quantization floor.

Example
-------
>>> from stm_data_processing.utils.bragg_peak_detection import detect_bragg_peaks
>>> result = detect_bragg_peaks(image, size_nm=30.0)          # doctest: +SKIP
>>> for peak in result.peaks[:3]:                              # doctest: +SKIP
...     print(peak.index_hk, peak.q_nm_inv, peak.sigma_q_nm_inv)
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.linalg import polar
from scipy.ndimage import maximum_filter
from scipy.optimize import curve_fit

from ..io.lattice_loader import LatticeLoader

__all__ = [
    "BraggDetectionResult",
    "BraggPeak",
    "BraggPeakDetector",
    "LatticeFit",
    "LatticeSpec",
    "compute_fft2",
    "detect_bragg_peaks",
]

logger = logging.getLogger(__name__)

# MAD / sigma for a Rayleigh-distributed |FFT| background (design spec 4.2).
_MAD_OVER_RAYLEIGH = 0.448641
# Variance of a uniform position on a one-pixel grid.
_QUANTIZED_SIGMA_PX = 1.0 / np.sqrt(12.0)
# Per-axis sigma when a 3x3 parabola has no usable curvature.
_PARABOLA_FALLBACK_SIGMA_PX = 0.15
# Half-plane selection: keep qy > 0, or qy == 0 with qx > 0.
_MIN_DUPLICATE_DISTANCE_PX = 1.0
_HALF_PLANE_SIGN = 1
# A |FFT| noise scale below this fraction of the spectral peak means the
# background is float round-off rather than signal (constant / all-zero image).
_DEGENERATE_SCALE_FRACTION = 1e-12
# Same test against the float64 round-off floor of the transform,
# eps * max|image| * n, with a safety factor: the background of a plane
# subtracted constant image sits exactly there.
_DEGENERATE_FLOOR_FACTOR = 1e3


@dataclass(frozen=True)
class LatticeSpec:
    """Ideal reference reciprocal lattice used to label detected peaks.

    a_nm            real-space lattice constant in nm (hexagonal/square).
    symmetry        "hexagonal" | "square" | "oblique".
    bvecs_nm_inv    (2, 2) rows b1, b2 in nm^-1; overrides a_nm/symmetry.
    orientation_deg counter-clockwise rotation of the reference basis, in
                    degrees (the standard right-handed convention:
                    ``rot = [[cos, -sin], [sin, cos]]`` applied as
                    ``basis @ rot.T``).  None = unknown, in which case the index
                    labelling is canonical (see the design spec) and the affine
                    rotation is defined only modulo the point group angle
                    reported in LatticeFit.rotation_mod_deg.  A wrong sign or a
                    mismatch beyond the seed tolerance ``0.3 * |b1|`` (+/-17.4 deg
                    for a hexagonal first ring) cannot label the ideal pool
                    directly; the module falls back to the orientation-free
                    two-point seed and logs a warning instead of failing
                    silently.
    h_max           maximum |h|, |k| used to build the prediction pool.
    """

    a_nm: float | None = None
    symmetry: str = "hexagonal"
    bvecs_nm_inv: np.ndarray | None = None
    orientation_deg: float | None = None
    h_max: int = 8


@dataclass(frozen=True)
class BraggPeak:
    """One detected Bragg peak.

    q_px, q_nm_inv              sub-pixel position, (qx, qy).
    sigma_q_px, sigma_q_nm_inv  per-axis 1-sigma, equal to the square root of
                                the diagonal of the covariance below.
    cov_q_px, cov_q_nm_inv      (2, 2) **total** position covariance (fit
                                covariance plus the model floor), i.e. exactly
                                the matrix the lattice GLS weights with.
    amplitude, snr              fitted amplitude and |F|_peak / noise_sigma.
    index_hk                    (h, k) once consistent with the fitted lattice.
    q_model_px, sigma_q_model_px  lattice-model position and its 1-sigma, the
                                latter propagated from LatticeFit.cov_bvecs_nm_inv
                                through the (h, k) Jacobian of ``(h, k) @ B``.
    residual_px                 |q_measured - q_model| in px.
    chi2_reduced, n_pixels, method, quality
    conjugate_index             index of the -q partner inside result.peaks.
    independent                 True for the half-plane representative.
    """

    q_px: tuple[float, float]
    q_nm_inv: tuple[float, float]
    sigma_q_px: tuple[float, float]
    sigma_q_nm_inv: tuple[float, float]
    cov_q_px: np.ndarray
    cov_q_nm_inv: np.ndarray
    amplitude: float
    snr: float
    index_hk: tuple[int, int] | None = None
    q_model_px: tuple[float, float] | None = None
    sigma_q_model_px: tuple[float, float] | None = None
    residual_px: float | None = None
    chi2_reduced: float = float("nan")
    n_pixels: int = 0
    method: str = "gaussian"
    quality: str = "ok"
    conjugate_index: int | None = None
    independent: bool = True


@dataclass(frozen=True)
class LatticeFit:
    """Fitted reciprocal basis and (optionally) the affine distortion.

    bvecs_nm_inv         (2, 2) rows b1, b2 in nm^-1.
    cov_bvecs_nm_inv     (4, 4), vec order (b1x, b1y, b2x, b2y); the source of
                         every sigma_q_model_px through J_B cov J_B^T with
                         J_B = [[h, 0, k, 0], [0, h, 0, k]].
    affine               (2, 2) M with q_observed = q_ideal @ M; None when no
                         reference lattice was supplied (an inferred reference
                         still yields bvecs, rotation and the strain below).
    cov_affine           (4, 4) row-major covariance of M.
    rotation_deg         polar-decomposition rotation angle of M, degrees; with
                         an inferred reference it is the canonical labelling and
                         is only defined modulo rotation_mod_deg.
    rotation_sigma_deg   1-sigma of rotation_deg (numerical Jacobian).
    rotation_mod_deg     point-group modulus of rotation_deg (60/90/180/360).
    rotation_is_absolute True when LatticeSpec.orientation_deg was supplied.
    principal_stretches  singular values of M (ascending); invariant under the
                         labelling ambiguity of design D15.
    linearized_strain    (2, 2) polar(M)[1] - I, the right stretch tensor minus
                         the identity; invariant under a left point-group
                         relabelling (0.5 (M + M.T) - I would not be).  When no
                         reference lattice was supplied the fit is gauge-fixed
                         to |det M| = 1, so only the deviatoric shape distortion
                         is reported and the absolute scale lives in bvecs.
    n_independent, chi2_reduced (weighted), rms_residual_px,
    n_iterations, symmetry

    Acceptance gates (design ruling A-0..A-9, all three must hold):

    fit_ok              True when the fit passed every gate below *and* the
                        rank/|det B|/finiteness checks; when False the fit is
                        reported for diagnosis only: every peak loses its
                        index_hk/q_model_px/sigma_q_model_px/residual_px and
                        affine/cov_affine/rotation*/stretch/strain are None.
    quality             "ok", or the failed gates joined by "+"
                        (spacing_below_min / residual_above_max / chi2_above_max).
    The trailing fields split into **measured quantities** -- chi2_reduced,
    rms_residual_px, pool_spacing_px, consistent_fraction, n_independent -- and
    the **threshold actually applied** by G-2 (residual_max_px).  The other two
    gate thresholds are caller inputs and are echoed in ``meta`` instead of being
    copied into the frozen result: ``meta["min_pool_spacing_px"]`` and
    ``meta["chi2_red_max"]``.  Together with ``meta["pool_spacing_px"]`` and
    ``meta["residual_max_px"]`` this lets every verdict be recomputed from
    ``(result.lattice, result.meta)`` alone.

    pool_spacing_px     measured G-1 quantity: ``min_{hk != 0, |h|, |k| <= 2}
                        |(h, k) @ B_fit| / dq`` in FFT pixels; identical to
                        ``meta["pool_spacing_px"]``.
    rms_residual_px     measured G-2 quantity, in **pixels** (the nm^-1 model
                        prediction is converted with ``dq`` before the RMS).
    residual_max_px     G-2 threshold actually applied (explicit, or derived as
                        ``min(1.0 px, 0.25 * pool_spacing_px)``).
    consistent_fraction fraction of labelled peaks with
                        ``residual_px <= lattice_tolerance * hypot(sigma_q_px)``
                        -- measurement sigma only, in pixels, on the labelled set
                        the gates see (so a rejected fit still reports it).
                        Reported for auditing only; it never gates the fit.
                        The model-prediction sigma is excluded on purpose: it
                        scales with ``cov_bvecs``, which the GLS inflates by
                        ``max(1, chi2_reduced)``, so including it would make a
                        worse fit look *more* consistent (measured 0.86-1.00 on
                        fits that the gates reject).
    strain_gauge        which reference fixes the strain scale: "absolute"
                        (caller-supplied reference lattice), "deviatoric" (an
                        inferred hexagonal/square reference, gauge-fixed to
                        |det M| = 1 so only the shape distortion is reported) or
                        "undefined" (no reference with a meaningful ideal shape
                        -- an inferred oblique lattice -- or a rejected fit; the
                        shape fields are then None).  It is independent of
                        rotation_is_absolute: a caller-supplied basis without an
                        orientation has an absolute strain but a modulo-point-
                        group rotation.
    """

    bvecs_nm_inv: np.ndarray
    cov_bvecs_nm_inv: np.ndarray
    affine: np.ndarray | None
    cov_affine: np.ndarray | None
    rotation_deg: float | None
    rotation_sigma_deg: float | None
    rotation_mod_deg: float
    rotation_is_absolute: bool
    principal_stretches: tuple[float, float] | None
    linearized_strain: np.ndarray | None
    n_independent: int
    chi2_reduced: float
    rms_residual_px: float
    n_iterations: int
    symmetry: str
    fit_ok: bool = True
    quality: str = "ok"
    pool_spacing_px: float = 0.0
    residual_max_px: float = 0.0
    consistent_fraction: float = 0.0
    strain_gauge: str = "absolute"


@dataclass(frozen=True)
class BraggDetectionResult:
    """Output of :func:`detect_bragg_peaks`.

    peaks             tuple of BraggPeak, sorted by descending snr.
    lattice           LatticeFit, or None when the lattice stage did not converge.
    size_nm           real-space field of view in nm (square).
    n_px              image side length in pixels.
    dq_nm_inv         2*pi/size_nm, the reciprocal-space pixel size.
    q_nyquist_nm_inv  pi*n/size_nm.
    noise_sigma       robust Rayleigh scale of the |FFT| background.
    n_candidates      number of non-maximum-suppressed candidates before the
                      half-plane (+/-q) selection.
    fft2              the complex fftshifted spectrum when return_fft2=True.
    meta              diagnostic dictionary (footprint, inferred symmetry, ...).
    """

    peaks: tuple[BraggPeak, ...]
    lattice: LatticeFit | None
    size_nm: float
    n_px: int
    dq_nm_inv: float
    q_nyquist_nm_inv: float
    noise_sigma: float
    n_candidates: int
    fft2: np.ndarray | None = None
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# Stage 0: pre-processing and FFT
# --------------------------------------------------------------------------


def _validate_image(image: np.ndarray, size_nm: float) -> np.ndarray:
    """Return ``image`` as a float64 (n, n) array, validating shape and size."""
    arr = np.asarray(image)
    if arr.ndim != 2:
        raise ValueError(f"image must be 2-D, got {arr.ndim} dimension(s)")
    if arr.shape[0] != arr.shape[1]:
        raise ValueError(f"image must be square, got shape {arr.shape}")
    if not np.isfinite(size_nm) or size_nm <= 0:
        raise ValueError(f"size_nm must be a positive number, got {size_nm!r}")
    return np.ascontiguousarray(arr, dtype=np.float64)


def _plane_coefficients(data: np.ndarray) -> np.ndarray | None:
    """Least-squares plane ``c0*x + c1*y + c2`` over the finite pixels."""
    finite = np.isfinite(data)
    if int(np.count_nonzero(finite)) < 3:
        return None
    rows, cols = np.nonzero(finite)
    design = np.column_stack([cols, rows, np.ones(rows.size)])
    coeffs, *_ = np.linalg.lstsq(design, data[rows, cols], rcond=None)
    return coeffs


def _fill_nan_with_plane(image: np.ndarray, nan_policy: str) -> np.ndarray:
    """Replace NaN pixels by the best-fit plane value.

    This mirrors the plane fit of ``utils.plot_funcs.subtractMeanPlane`` but is
    inlined so that the module keeps its numpy/scipy-only import surface.
    """
    bad = ~np.isfinite(image)
    n_bad = int(np.count_nonzero(bad))
    if n_bad == 0:
        return image
    if nan_policy == "raise":
        raise ValueError(f"image contains {n_bad} non-finite pixel(s)")
    if nan_policy != "plane":
        raise ValueError(f"nan_policy must be 'plane' or 'raise', got {nan_policy!r}")
    logger.warning(
        "bragg_peak_detection: filled %d non-finite pixel(s) with the best-fit "
        "plane (nan_policy='plane'); the reported peaks and uncertainties rest "
        "on interpolated data there.",
        n_bad,
    )
    coeffs = _plane_coefficients(image)
    filled = image.copy()
    if coeffs is None:
        logger.warning(
            "bragg_peak_detection: fewer than 3 finite pixels; NaN set to zero."
        )
        filled[bad] = 0.0
        return filled
    rows, cols = np.nonzero(bad)
    filled[rows, cols] = coeffs[0] * cols + coeffs[1] * rows + coeffs[2]
    return filled


def _subtract_plane(image: np.ndarray) -> np.ndarray:
    """Subtract the best-fit plane (equivalent to plot_funcs.subtractMeanPlane)."""
    coeffs = _plane_coefficients(image)
    if coeffs is None:
        logger.warning(
            "bragg_peak_detection: fewer than 3 finite pixels; plane not subtracted."
        )
        return image.copy()
    rows, cols = np.meshgrid(
        np.arange(image.shape[0]), np.arange(image.shape[1]), indexing="ij"
    )
    return image - (coeffs[0] * cols + coeffs[1] * rows + coeffs[2])


def _window_array(window: str | np.ndarray | None, n: int) -> np.ndarray:
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


def compute_fft2(
    image: np.ndarray,
    size_nm: float,
    *,
    window: str | np.ndarray | None = "hann",
    subtract_plane: bool = True,
    nan_policy: str = "plane",
) -> np.ndarray:
    """Return ``fftshift(fft2(windowed, plane-subtracted image))`` as complex128.

    Parameters
    ----------
    image : np.ndarray
        Square 2-D real-space image (STM topograph or any scalar field).
    size_nm : float
        Physical side length of the (square) field of view, in nm.
    window : {"hann", "hamming", "blackman"} or np.ndarray or None
        Apodisation window; ``None`` disables windowing.
    subtract_plane : bool
        Subtract the best-fit plane before windowing.  This mirrors
        ``utils.plot_funcs.subtractMeanPlane`` without importing matplotlib.
    nan_policy : {"plane", "raise"}
        ``"plane"`` fills NaN pixels with the best-fit plane value; ``"raise"``
        turns any non-finite pixel into a ``ValueError``.

    Returns
    -------
    np.ndarray
        ``(n, n)`` complex128 spectrum, fftshifted so that the DC term sits at
        ``(n // 2, n // 2)``.  No zero padding is applied, by design.
    """
    arr = _validate_image(image, size_nm)
    arr = _fill_nan_with_plane(arr, nan_policy)
    if subtract_plane:
        arr = _subtract_plane(arr)
    arr = arr * _window_array(window, arr.shape[0])
    return np.fft.fftshift(np.fft.fft2(arr))


# --------------------------------------------------------------------------
# Stage 1: candidate detection
# --------------------------------------------------------------------------


def _radius_map(n: int) -> np.ndarray:
    """Radius in FFT pixels of every sample, measured from the DC centre."""
    axis = np.arange(n) - n // 2
    qx, qy = np.meshgrid(axis, axis)
    return np.hypot(qx, qy)


def _robust_rayleigh_scale(magnitude: np.ndarray, band: np.ndarray) -> float:
    """MAD-based Rayleigh scale ``sigma = MAD / 0.448641`` over a boolean band."""
    values = magnitude[band]
    if values.size < 16:
        values = magnitude.ravel()
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad <= 0.0:
        mad = float(np.std(values))
    if mad <= 0.0:
        # Degenerate band (constant image, all-zero image): there is no
        # measurable background scale, so the caller must bail out instead of
        # dividing by a denormal number.
        return 0.0
    return mad / _MAD_OVER_RAYLEIGH


def _noise_scale_two_pass(
    magnitude: np.ndarray, annulus: np.ndarray, min_snr: float
) -> float:
    """Two-pass robust |FFT| noise scale that excludes the peak neighbourhoods.

    The first pass measures the scale on the annulus; the second pass removes
    every +/-2 px neighbourhood of the peaks found with that first estimate so
    that bright peaks cannot inflate the background statistics.
    """
    sigma = _robust_rayleigh_scale(magnitude, annulus)
    if sigma <= 0.0:
        return 0.0
    median = float(np.median(magnitude[annulus]))
    snr_map = (magnitude - median) / sigma
    first = (magnitude == maximum_filter(magnitude, size=3, mode="nearest")) & (
        snr_map >= min_snr
    )
    if np.any(first[annulus]):
        neighbours = maximum_filter(first.astype(np.uint8), size=5, mode="nearest") > 0
        cleaned = annulus & ~neighbours
        if int(np.count_nonzero(cleaned)) >= 16:
            sigma = _robust_rayleigh_scale(magnitude, cleaned)
    return sigma


def _half_width_axis(profile: np.ndarray, centre: int) -> float:
    """Largest half width at half maximum of a 1-D profile around ``centre``."""
    peak = float(profile[centre])
    base = float(profile.min())
    if peak <= base:
        return 1.0
    half = base + 0.5 * (peak - base)
    widths = []
    for step in (-1, 1):
        previous = peak
        distance = 0
        index = centre
        while 0 <= index + step < profile.size:
            index += step
            distance += 1
            current = float(profile[index])
            if current <= half:
                fraction = (previous - half) / max(previous - current, 1e-30)
                widths.append(distance - 1.0 + fraction)
                break
            previous = current
        else:
            widths.append(float(distance))
    return float(max(max(widths, default=1.0), 0.5))


def _adaptive_footprint(
    magnitude: np.ndarray, candidates: list[tuple[int, int]]
) -> int:
    """Non-maximum-suppression footprint from the median peak half width."""
    if not candidates:
        return 3
    n = magnitude.shape[0]
    widths = []
    for row, col in candidates[:20]:
        r0, r1 = max(0, row - 2), min(n, row + 3)
        c0, c1 = max(0, col - 2), min(n, col + 3)
        window = magnitude[r0:r1, c0:c1]
        if window.size < 9:
            continue
        iy, ix = np.unravel_index(np.argmax(window), window.shape)
        widths.append(
            max(_half_width_axis(window[iy, :], ix), _half_width_axis(window[:, ix], iy))
        )
    if not widths:
        return 3
    return int(np.clip(round(1.5 * float(np.median(widths))), 3, 11))


def _detect_candidates(
    magnitude: np.ndarray,
    noise_sigma: float,
    radius: np.ndarray,
    dc_radius_px: float,
    q_max_px: float,
    min_snr: float,
    footprint: int | None,
) -> tuple[list[tuple[int, int]], int]:
    """Return the candidate local maxima and the footprint actually used.

    ``dc_radius_px`` and ``q_max_px`` are **candidate-level** filters applied to
    the integer bins of the non-maximum-suppressed mask -- they are not hard
    boundaries of the reported spectrum: sub-pixel localization can move a peak
    by up to the patch radius afterwards, so a reported ``|q|`` may fall slightly
    inside the DC mask or beyond ``q_max_px``.  The only hard guarantee is
    ``|q| <= q_nyquist`` (design F3), which follows from the mask radius being
    below ``n // 2``; the ``quality="edge"`` marker covers candidates whose patch
    does not fit the array.
    """
    median = float(np.median(magnitude))
    snr_map = (magnitude - median) / noise_sigma
    coarse = magnitude == maximum_filter(magnitude, size=3, mode="nearest")
    seed_rows, seed_cols = np.nonzero(coarse & (snr_map >= min_snr))
    seeds = sorted(
        zip(seed_rows.tolist(), seed_cols.tolist(), strict=True),
        key=lambda rc: -float(magnitude[rc[0], rc[1]]),
    )
    used_footprint = (
        int(footprint) if footprint else _adaptive_footprint(magnitude, seeds)
    )
    local_max = magnitude == maximum_filter(
        magnitude, size=used_footprint, mode="nearest"
    )
    mask = (
        local_max
        & (snr_map >= min_snr)
        & (radius > dc_radius_px)
        & (radius <= q_max_px)
    )
    rows, cols = np.nonzero(mask)
    order = np.argsort(-magnitude[rows, cols], kind="stable")
    candidates = [(int(rows[i]), int(cols[i])) for i in order]
    return candidates, used_footprint


# --------------------------------------------------------------------------
# Stage 2: sub-pixel localization
# --------------------------------------------------------------------------


def _gaussian_model(xy, amp, x0, y0, sx, sy, rho, b0, bx, by):
    """Rotated 2-D Gaussian on a plane background (9 parameters)."""
    xg, yg = xy
    dx = xg - x0
    dy = yg - y0
    one_minus_rho2 = 1.0 - rho * rho
    quad = (dx / sx) ** 2 - 2.0 * rho * (dx / sx) * (dy / sy) + (dy / sy) ** 2
    return amp * np.exp(-0.5 * quad / one_minus_rho2) + b0 + bx * dx + by * dy


def _parabola_axis(zm: float, z0: float, zp: float, sigma_log: float):
    """3-point log-parabola peak offset and its 1-sigma along one axis.

    ``x* = -a / (2 c)`` with ``a = (z+ - z-)/2`` and ``c = (z+ - 2 z0 + z-)/2``,
    so the partial derivatives w.r.t. the three samples are
    ``(-1/(4c) + a/(4c^2), -a/(2c^2), 1/(4c) + a/(4c^2))``.
    """
    a = 0.5 * (zp - zm)
    c = 0.5 * (zp - 2.0 * z0 + zm)
    if c >= -1e-12:
        return 0.0, None
    offset = -a / (2.0 * c)
    jp = -1.0 / (4.0 * c) + a / (4.0 * c * c)
    j0 = -a / (2.0 * c * c)
    jm = 1.0 / (4.0 * c) + a / (4.0 * c * c)
    variance = (jp * jp + j0 * j0 + jm * jm) * sigma_log * sigma_log
    return float(np.clip(offset, -1.0, 1.0)), float(np.sqrt(variance))


def _log_parabola_3x3(patch: np.ndarray, iy: int, ix: int, sigma_log: float):
    """Return ``(dx, dy, sigma_x, sigma_y)`` of the 3x3 log-parabola fit."""
    log_patch = np.log(np.maximum(patch, np.finfo(float).tiny))
    dx, sigma_x = _parabola_axis(
        log_patch[iy, ix - 1], log_patch[iy, ix], log_patch[iy, ix + 1], sigma_log
    )
    dy, sigma_y = _parabola_axis(
        log_patch[iy - 1, ix], log_patch[iy, ix], log_patch[iy + 1, ix], sigma_log
    )
    return dx, dy, sigma_x, sigma_y


def _second_moment_sigma(patch: np.ndarray):
    """Second-moment width ``(sx, sy)`` of a patch, or None if it is flat."""
    floor = float(patch.min())
    weights = np.maximum(patch - floor, 0.0)
    total = float(weights.sum())
    if total <= 0.0:
        return None
    xs = np.arange(patch.shape[1], dtype=float)
    ys = np.arange(patch.shape[0], dtype=float)
    mx = float((weights * xs[None, :]).sum() / total)
    my = float((weights * ys[:, None]).sum() / total)
    dx = xs[None, :] - mx
    dy = ys[:, None] - my
    sx = float(np.sqrt(max((weights * dx * dx).sum() / total, 1e-12)))
    sy = float(np.sqrt(max((weights * dy * dy).sum() / total, 1e-12)))
    return sx, sy


def _adaptive_patch_half(magnitude: np.ndarray, row: int, col: int) -> int:
    """``clip(ceil(1.5 * sigma_hat), 3, 7)`` from a 7x7 second-moment width."""
    n = magnitude.shape[0]
    half = 3
    r0, r1 = max(0, row - half), min(n, row + half + 1)
    c0, c1 = max(0, col - half), min(n, col + half + 1)
    moments = _second_moment_sigma(magnitude[r0:r1, c0:c1])
    if moments is None:
        return 3
    return int(np.clip(np.ceil(1.5 * max(moments)), 3, 7))


def _edge_peak(magnitude, row, col, noise_sigma):
    """Peak record for a candidate whose patch does not fit inside the array."""
    qx = col - magnitude.shape[0] // 2
    qy = row - magnitude.shape[0] // 2
    return {
        "row": row,
        "col": col,
        "qx_px": float(qx),
        "qy_px": float(qy),
        "sigma_px": np.array([_QUANTIZED_SIGMA_PX, _QUANTIZED_SIGMA_PX]),
        "cov_px": np.eye(2) * (1.0 / 12.0),
        "amplitude": float(magnitude[row, col]),
        "snr": float(magnitude[row, col] / noise_sigma),
        "chi2_reduced": float("nan"),
        "n_pixels": 1,
        "method": "max_pixel",
        "base_quality": "failed",
        "edge": True,
        "index_hk": None,
        "q_model_px": None,
        "sigma_q_model_px": None,
        "residual_px": None,
    }


def _localize_peak(
    magnitude: np.ndarray,
    row: int,
    col: int,
    noise_sigma: float,
    *,
    patch_half: int | None,
    bounds_half: float,
    sigma_model_floor_px: float,
) -> dict:
    """Sub-pixel localization of one candidate on the |FFT| magnitude.

    Chain: snap to the local maximum, 3x3 log-parabola seed, bounded rotated
    Gaussian LSQ with a plane background, then a documented fallback to the
    log-parabola result and finally to the integer maximum pixel.  Failures are
    never silent: ``method`` and ``base_quality`` always record what was used.
    """
    n = magnitude.shape[0]
    r0, r1 = max(0, row - 2), min(n, row + 3)
    c0, c1 = max(0, col - 2), min(n, col + 3)
    window = magnitude[r0:r1, c0:c1]
    iy, ix = np.unravel_index(np.argmax(window), window.shape)
    row, col = r0 + int(iy), c0 + int(ix)

    half = int(patch_half) if patch_half else _adaptive_patch_half(magnitude, row, col)
    if row - half < 0 or row + half + 1 > n or col - half < 0 or col + half + 1 > n:
        return _edge_peak(magnitude, row, col, noise_sigma)

    patch = magnitude[row - half : row + half + 1, col - half : col + half + 1]
    iy, ix = np.unravel_index(np.argmax(patch), patch.shape)
    amplitude_guess = float(patch.max() - patch.min())
    sigma_log = noise_sigma / max(amplitude_guess, np.finfo(float).tiny)
    if 1 <= iy <= patch.shape[0] - 2 and 1 <= ix <= patch.shape[1] - 2:
        dx, dy, sigma_x_par, sigma_y_par = _log_parabola_3x3(patch, iy, ix, sigma_log)
    else:
        dx, dy, sigma_x_par, sigma_y_par = 0.0, 0.0, None, None
    x0_seed, y0_seed = ix + dx, iy + dy

    moments = _second_moment_sigma(patch)
    if moments is None:
        sx_seed = sy_seed = min(max(0.5 * half, 0.7), half + 0.5)
    else:
        sx_seed = float(np.clip(moments[0], 0.7, half + 0.5))
        sy_seed = float(np.clip(moments[1], 0.7, half + 0.5))

    yy, xx = np.mgrid[0 : patch.shape[0], 0 : patch.shape[1]]
    xy = (xx.ravel().astype(float), yy.ravel().astype(float))
    p0 = [
        amplitude_guess, x0_seed, y0_seed, sx_seed, sy_seed,
        0.0, float(patch.min()), 0.0, 0.0,
    ]
    lo = [0.0, x0_seed - bounds_half, y0_seed - bounds_half, 0.4, 0.4,
          -0.9, -np.inf, -np.inf, -np.inf]
    hi = [np.inf, x0_seed + bounds_half, y0_seed + bounds_half, half + 0.5,
          half + 0.5, 0.9, np.inf, np.inf, np.inf]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, pcov = curve_fit(
                _gaussian_model,
                xy,
                patch.ravel(),
                p0=p0,
                bounds=(lo, hi),
                sigma=np.full(patch.size, noise_sigma),
                absolute_sigma=True,
                maxfev=4000,
            )
    except Exception:
        return _parabola_or_pixel(
            magnitude, row, col, n, patch, iy, ix, half, noise_sigma,
            dx, dy, sigma_x_par, sigma_y_par, sigma_model_floor_px,
        )

    variance = np.diag(pcov).astype(float)
    dof = max(patch.size - 9, 1)
    residual = patch.ravel() - _gaussian_model(xy, *popt)
    chi2_reduced = float(np.sum((residual / noise_sigma) ** 2) / dof)
    at_bound = (
        min(abs(popt[1] - lo[1]), abs(popt[1] - hi[1])) < 1e-6
        or min(abs(popt[2] - lo[2]), abs(popt[2] - hi[2])) < 1e-6
        or min(abs(popt[3] - lo[3]), abs(popt[3] - hi[3])) < 1e-9
        or min(abs(popt[4] - lo[4]), abs(popt[4] - hi[4])) < 1e-9
    )
    sigma_q = np.sqrt(np.maximum(variance[1:3], 0.0))
    centre_ok = (
        0.0 <= popt[1] <= patch.shape[1] - 1.0
        and 0.0 <= popt[2] <= patch.shape[0] - 1.0
    )
    if (
        not np.all(np.isfinite(variance))
        or np.any(variance[1:3] < 0.0)
        or not np.all(np.isfinite(sigma_q))
        or float(np.max(sigma_q)) >= 1.0
        or at_bound
        or not centre_ok
    ):
        return _parabola_or_pixel(
            magnitude, row, col, n, patch, iy, ix, half, noise_sigma,
            dx, dy, sigma_x_par, sigma_y_par, sigma_model_floor_px,
        )

    cov = np.array([[variance[1], pcov[1, 2]], [pcov[1, 2], variance[2]]], dtype=float)
    cov = cov + np.eye(2) * sigma_model_floor_px**2
    sigma_reported = np.sqrt(np.maximum(np.diag(cov), 0.0))
    return {
        "row": row,
        "col": col,
        "qx_px": float(col - half + popt[1] - n // 2),
        "qy_px": float(row - half + popt[2] - n // 2),
        "sigma_px": sigma_reported,
        "cov_px": cov,
        "amplitude": float(popt[0]),
        "snr": float(magnitude[row, col] / noise_sigma),
        "chi2_reduced": chi2_reduced,
        "n_pixels": int(patch.size),
        "method": "gaussian",
        "base_quality": "ok",
        "edge": False,
        "index_hk": None,
        "q_model_px": None,
        "sigma_q_model_px": None,
        "residual_px": None,
    }


def _parabola_or_pixel(
    magnitude, row, col, n, patch, iy, ix, half, noise_sigma,
    dx, dy, sigma_x_par, sigma_y_par, sigma_model_floor_px,
):
    """Second localization tier: 3x3 log parabola, else the integer maximum."""
    if sigma_x_par is not None and sigma_y_par is not None:
        cov = np.diag([sigma_x_par**2, sigma_y_par**2]) + (
            np.eye(2) * sigma_model_floor_px**2
        )
        return {
            "row": row,
            "col": col,
            "qx_px": float(col - half + ix + dx - n // 2),
            "qy_px": float(row - half + iy + dy - n // 2),
            "sigma_px": np.sqrt(np.diag(cov)),
            "cov_px": cov,
            "amplitude": float(patch.max() - patch.min()),
            "snr": float(magnitude[row, col] / noise_sigma),
            "chi2_reduced": float("nan"),
            "n_pixels": 9,
            "method": "parabola_log",
            "base_quality": "coarse",
            "edge": False,
            "index_hk": None,
            "q_model_px": None,
            "sigma_q_model_px": None,
            "residual_px": None,
        }
    logger.debug("localize: integer-maximum fallback at (%d, %d)", row, col)
    return {
        "row": row,
        "col": col,
        "qx_px": float(col - n // 2),
        "qy_px": float(row - n // 2),
        "sigma_px": np.array([_PARABOLA_FALLBACK_SIGMA_PX, _PARABOLA_FALLBACK_SIGMA_PX]),
        "cov_px": np.eye(2) * _PARABOLA_FALLBACK_SIGMA_PX**2,
        "amplitude": float(magnitude[row, col]),
        "snr": float(magnitude[row, col] / noise_sigma),
        "chi2_reduced": float("nan"),
        "n_pixels": 1,
        "method": "max_pixel",
        "base_quality": "failed",
        "edge": False,
        "index_hk": None,
        "q_model_px": None,
        "sigma_q_model_px": None,
        "residual_px": None,
    }


# --------------------------------------------------------------------------
# Stage 3: lattice-constrained refinement (all fits in nm^-1)
# --------------------------------------------------------------------------


def _point_group_mod_deg(symmetry: str) -> float:
    """Rotation modulus (degrees) imposed by the point group of ``symmetry``."""
    if symmetry == "hexagonal":
        return 60.0
    if symmetry == "square":
        return 90.0
    if symmetry == "oblique":
        return 180.0
    raise ValueError(
        f"symmetry must be 'hexagonal', 'square' or 'oblique', got {symmetry!r}"
    )


def _ideal_basis(
    a_nm: float | None, symmetry: str, bvecs_nm_inv: np.ndarray | None
) -> np.ndarray:
    """(2, 2) rows b1, b2 in nm^-1 for the requested ideal lattice.

    Hexagonal uses ``|b| = 4*pi/(sqrt(3)*a)`` (60 degrees between b1 and b2);
    square uses ``|b| = 2*pi/a`` along x and y.  ``bvecs_nm_inv`` overrides both.
    """
    if bvecs_nm_inv is not None:
        basis = np.asarray(bvecs_nm_inv, dtype=float)
        if basis.shape != (2, 2):
            raise ValueError(f"bvecs_nm_inv must have shape (2, 2), got {basis.shape}")
        if abs(float(np.linalg.det(basis))) < 1e-12:
            raise ValueError("bvecs_nm_inv is singular")
        return basis
    if a_nm is None:
        raise ValueError("LatticeSpec requires either a_nm or bvecs_nm_inv")
    if a_nm <= 0:
        raise ValueError(f"a_nm must be positive, got {a_nm!r}")
    if symmetry == "hexagonal":
        b = 4.0 * np.pi / (np.sqrt(3.0) * a_nm)
        return np.array(
            [[b, 0.0], [b * np.cos(np.pi / 3.0), b * np.sin(np.pi / 3.0)]],
            dtype=float,
        )
    if symmetry == "square":
        b = 2.0 * np.pi / a_nm
        return np.array([[b, 0.0], [0.0, b]], dtype=float)
    if symmetry == "oblique":
        raise ValueError("symmetry='oblique' requires explicit bvecs_nm_inv")
    raise ValueError(
        f"symmetry must be 'hexagonal', 'square' or 'oblique', got {symmetry!r}"
    )


def _rotate_basis(basis: np.ndarray, orientation_deg: float | None) -> np.ndarray:
    """Rotate the rows of ``basis`` by ``orientation_deg`` about the origin."""
    if orientation_deg is None:
        return basis
    theta = np.radians(float(orientation_deg))
    rot = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    return basis @ rot.T


def _first_ring_labels(symmetry: str) -> list[tuple[int, int]]:
    """Ideal (h, k) labels of the first ring, ordered by increasing polar angle."""
    if symmetry == "hexagonal":
        return [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    if symmetry == "square":
        return [(1, 0), (0, 1), (-1, 0), (0, -1)]
    raise ValueError(f"no canonical first ring for symmetry {symmetry!r}")


def _ideal_pool(b_ideal: np.ndarray, q_max_nm_inv: float, h_max: int):
    """Ideal reciprocal points up to ``q_max_nm_inv`` as ``(hk, q, used_ops)``.

    The pool is generated with ``lattice_operations`` (through
    ``LatticeLoader.create_lattice`` + ``get_bragg_points_in_circle``); if that
    fails, a local five-line enumeration is used and the flag is set to False.
    ``(h, k)`` are recovered with ``round(q @ inv(B_ideal))``.
    """
    try:
        lattice = LatticeLoader.create_lattice(
            bvecs_array=np.asarray(b_ideal, dtype=float)
        )
        from .lattice_operations import LatticeOperations

        points = LatticeOperations(lattice).get_bragg_points_in_circle(
            q_max_nm_inv, include_origin=False
        )
        q = np.asarray(points, dtype=float).T
        if q.size:
            hk_float = q @ np.linalg.inv(b_ideal)
            hk = np.round(hk_float).astype(int)
            if np.abs(hk_float - hk).max() <= 1e-6:
                order = np.argsort(np.hypot(q[:, 0], q[:, 1]), kind="stable")
                return hk[order], q[order], True
    except Exception as exc:
        logger.warning(
            "bragg_peak_detection: lattice_operations pool failed (%s); "
            "using the local enumeration.",
            exc,
        )
    pairs = []
    for h in range(-h_max, h_max + 1):
        for k in range(-h_max, h_max + 1):
            if (h, k) == (0, 0):
                continue
            point = np.array([h, k], dtype=float) @ b_ideal
            if np.hypot(point[0], point[1]) <= q_max_nm_inv:
                pairs.append((h, k, point))
    pairs.sort(key=lambda item: float(np.hypot(item[2][0], item[2][1])))
    if not pairs:
        return np.zeros((0, 2), int), np.zeros((0, 2)), False
    hk = np.array([[h, k] for h, k, _ in pairs], dtype=int)
    q = np.array([point for _, _, point in pairs], dtype=float)
    return hk, q, False


def _whitener(cov: np.ndarray) -> np.ndarray:
    """Symmetric inverse square root of a 2x2 covariance matrix."""
    values, vectors = np.linalg.eigh(cov)
    values = np.clip(values, 1e-12, None)
    return vectors @ np.diag(1.0 / np.sqrt(values)) @ vectors.T


def _gls_fit(
    q_obs: np.ndarray,
    covs: list[np.ndarray],
    assign: list[tuple[int, tuple[int, int]]],
    b_ideal: np.ndarray,
):
    """Weighted least-squares fit of the 2x2 affine matrix ``M``.

    Solves ``min sum_i || (q_i - (h_i, k_i) @ B_ideal @ M) W_i ||^2`` with
    ``W_i = Cov_i^(-1/2)``, where ``Cov_i`` (nm^2) is the *total* peak position
    covariance of :class:`BraggPeak` -- the model floor is already inside it, so
    it is not added again here.  Returns ``(M, cov_M (4x4, row-major),
    chi2_reduced)`` or ``None`` when the design is rank deficient (fewer than
    three points, or collinear).
    """
    if len(assign) < 3:
        return None
    rows, rhs = [], []
    for index, hk in assign:
        q_ideal = np.asarray(hk, dtype=float) @ b_ideal
        weight = _whitener(np.asarray(covs[index], dtype=float))
        design = np.array(
            [[q_ideal[0], 0.0, q_ideal[1], 0.0], [0.0, q_ideal[0], 0.0, q_ideal[1]]]
        )
        rows.append(weight @ design)
        rhs.append(weight @ np.asarray(q_obs[index], dtype=float))
    design = np.vstack(rows)
    target = np.concatenate(rhs)
    if np.linalg.matrix_rank(design, tol=1e-9) < 4:
        return None
    solution, *_ = np.linalg.lstsq(design, target, rcond=None)
    residual = target - design @ solution
    dof = max(target.size - 4, 1)
    chi2_reduced = float(residual @ residual) / dof
    cov_m = np.linalg.pinv(design.T @ design) * max(1.0, chi2_reduced)
    return solution.reshape(2, 2), cov_m, chi2_reduced


def _predict_sigma(cov_bvecs: np.ndarray, h: int, k: int) -> np.ndarray:
    """Per-axis 1-sigma of ``(h, k) @ B_fit`` from ``cov_bvecs``, in nm^-1.

    ``J_B = [[h, 0, k, 0], [0, h, 0, k]]`` is the Jacobian with respect to the
    row-major flattening ``(b1x, b1y, b2x, b2y)`` of the **fitted basis**, so it
    must be applied to ``cov_bvecs_nm_inv``.  Applying it to the M-space
    covariance would require the explicit ``(B_ideal (x) I2)`` transform first
    (``cov_bvecs = (B_ideal (x) I2) cov_affine (B_ideal (x) I2)^T``).
    """
    jac = np.array([[h, 0.0, k, 0.0], [0.0, h, 0.0, k]])
    return np.sqrt(np.clip(np.diag(jac @ cov_bvecs @ jac.T), 0.0, None))


class _LabelledSet:
    """Peaks that carry an integer (h, k) label, plus their GLS fit."""

    def __init__(self, b_ideal, dq_nm_inv, candidates):
        self.b_ideal = b_ideal
        self.dq_nm_inv = dq_nm_inv
        self.candidates = candidates
        self.entries: list[tuple[dict, tuple[int, int]]] = []

    def rescale_basis(self, scale: float) -> None:
        """Scale the reference basis; the fitted affine absorbs ``1/scale``.

        Used for the deviatoric gauge of an inferred (reference-free) lattice:
        ``q = (h, k) @ B_ideal @ M`` is invariant under ``B -> s B, M -> M/s``,
        so the fitted basis and every predicted position stay unchanged while
        ``|det M|`` is renormalised to 1.
        """
        assert scale > 0.0
        self.b_ideal = self.b_ideal * scale

    def add(self, peak: dict, label: tuple[int, int]) -> None:
        """Store the ``(peak, label)`` pair.

        The label is deliberately *not* written into the peak here: candidate
        seeds are tried in turn and most of them are discarded, and a label that
        survived such a rollback would contradict the documented meaning of
        ``BraggPeak.index_hk`` ("once consistent with the fitted lattice").
        ``_fill_model_fields`` writes the labels of the converged fit only.
        """
        self.entries.append((peak, (int(label[0]), int(label[1]))))

    def q_obs(self) -> np.ndarray:
        """Observed positions of the labelled peaks, in nm^-1."""
        return np.array(
            [[p["qx_px"] * self.dq_nm_inv, p["qy_px"] * self.dq_nm_inv]
             for p, _ in self.entries],
            dtype=float,
        )

    def covs(self) -> list[np.ndarray]:
        """Position covariances of the labelled peaks, in nm^2."""
        return [p["cov_px"] * self.dq_nm_inv**2 for p, _ in self.entries]

    def assign(self):
        """``(index, (h, k))`` pairs for :func:`_gls_fit`."""
        return [(i, hk) for i, (_, hk) in enumerate(self.entries)]

    def fit(self):
        """Run the GLS fit over the labelled set."""
        return _gls_fit(self.q_obs(), self.covs(), self.assign(), self.b_ideal)

    def expand(
        self,
        pool_hk: np.ndarray,
        pool_q: np.ndarray,
        max_distance_px: float,
        model: np.ndarray | None = None,
    ):
        """Label unlabelled peaks by their nearest model prediction.

        ``model`` supplies the seed affine when fewer than three points are
        labelled (an oblique seed has exactly two); otherwise the current GLS
        fit is used.  Returns the added ``(peak, label)`` pairs.
        """
        if not self.entries:
            return []
        if model is None:
            fit = self.fit()
            if fit is None:
                return []
            model = fit[0]
        predicted = pool_q @ model
        taken = {id(peak) for peak, _ in self.entries}
        added = []
        for peak in self.candidates:
            if id(peak) in taken:
                continue
            distances = (
                np.hypot(
                    predicted[:, 0] - peak["qx_px"] * self.dq_nm_inv,
                    predicted[:, 1] - peak["qy_px"] * self.dq_nm_inv,
                )
                / self.dq_nm_inv
            )
            best = int(np.argmin(distances))
            if distances[best] <= max_distance_px:
                label = (int(pool_hk[best, 0]), int(pool_hk[best, 1]))
                self.add(peak, label)
                added.append((peak, label))
        return added

    def expand_and_refit(
        self, pool_hk, pool_q, max_rounds=3, max_distance_px=1.5, model=None
    ):
        """``fit -> nearest-label expansion -> refit`` up to ``max_rounds``.

        ``model`` is the seed affine used by the first expansion round when the
        labelled seed holds fewer than three points (oblique two-point seed).
        """
        fit = self.fit() if model is None else None
        if fit is None and model is None:
            return None
        if model is not None:
            added = self.expand(pool_hk, pool_q, max_distance_px, model=model)
            if not added:
                return None
            fit = self.fit()
            if fit is None:
                del self.entries[-len(added) :]
                return None
        for _ in range(max_rounds):
            added = self.expand(pool_hk, pool_q, max_distance_px)
            if not added:
                break
            refit = self.fit()
            if refit is None:
                del self.entries[-len(added) :]
                _clear_model_fields(self.candidates)
                break
            fit = refit
        return fit


def _full_pool_assignments(q_obs, b_ideal, pool_hk, pool_q, tolerance_nm_inv):
    """Label every peak by its nearest ideal point of the *whole* pool.

    This is the design section 4.4(a) rule for a known orientation: the ideal
    pool and the observations share that orientation, so a nearest-neighbour
    match with a ``0.3 * |b1|`` tolerance is unambiguous even when only two
    first-ring points are visible (square case).  Returns ``[(pairs, None)]``
    once at least three labelled points span the plane, otherwise ``[]``.
    """
    if pool_q.size == 0 or tolerance_nm_inv <= 0.0 or q_obs.shape[0] == 0:
        return []
    pairs = []
    for index in range(q_obs.shape[0]):
        distances = np.hypot(
            pool_q[:, 0] - q_obs[index, 0], pool_q[:, 1] - q_obs[index, 1]
        )
        best = int(np.argmin(distances))
        if distances[best] <= tolerance_nm_inv:
            pairs.append((index, (int(pool_hk[best, 0]), int(pool_hk[best, 1]))))
    if len(pairs) < 3:
        return []
    ideal = np.array([np.asarray(hk, dtype=float) @ b_ideal for _, hk in pairs])
    if np.linalg.matrix_rank(ideal, tol=1e-9 * float(np.max(np.abs(ideal)))) < 2:
        return []
    return [(pairs, None)]


def _two_point_model(b_ideal, q_obs, pairs):
    """Closed-form affine of a two-point seed: ``[g1; g2] @ M = [q1; q2]``.

    With ``g_i = (h, k)_i @ B_ideal`` the model reproduces both seed peaks
    exactly, which is what lets the expansion label the remaining points before
    the GLS has the three points it needs (oblique seeds, and square seeds whose
    half-plane first ring holds only two points).
    """
    if len(pairs) < 2:
        return None
    (first_index, first_hk), (second_index, second_hk) = pairs[0], pairs[1]
    ideal = np.vstack(
        [
            np.asarray(first_hk, dtype=float) @ b_ideal,
            np.asarray(second_hk, dtype=float) @ b_ideal,
        ]
    )
    if abs(float(np.linalg.det(ideal))) <= 1e-12:
        return None
    return np.linalg.inv(ideal) @ np.vstack(
        [q_obs[first_index], q_obs[second_index]]
    )


def _affine_distortion(model: np.ndarray) -> float:
    """``max(|lambda_i - 1|)`` of an affine: its distortion from the identity."""
    stretches = np.linalg.svd(model, compute_uv=False)
    return float(np.max(np.abs(np.asarray(stretches, dtype=float) - 1.0)))


def _oblique_candidate_peaks(q_obs, snr, b_ideal, top_n: int = 6, per_direction: int = 3):
    """Peaks that may carry the images of the reference vectors.

    Union of the ``top_n`` strongest peaks (the candidate set of the design
    addendum A-10) and, for each reference row, the ``per_direction`` peaks whose
    radius is closest to ``|b_i|``.  The second part is required because the
    strongest peaks are *not* reliably the images of ``b1``/``b2`` -- measured on
    the ruling's own asymmetric case, the image of ``b1`` ranked 11th by SNR and
    the least-distortion gauge was therefore unreachable.  Radius matching is
    rotation invariant (a small distortion moves ``|q|`` by a few percent only).
    """
    order = [int(index) for index in np.argsort(-np.asarray(snr, dtype=float))[: max(int(top_n), 2)]]
    radii = np.hypot(q_obs[:, 0], q_obs[:, 1])
    for row in range(2):
        target = float(np.hypot(b_ideal[row, 0], b_ideal[row, 1]))
        nearest = np.argsort(np.abs(radii - target))[: max(int(per_direction), 1)]
        order.extend(int(index) for index in nearest)
    seen = set()
    unique = []
    for index in order:
        if index not in seen:
            seen.add(index)
            unique.append(index)
    return unique


def _oblique_seed_candidates(q_obs, snr, b_ideal, top_n: int = 6):
    """Closed-form two-point seeds over the candidate peaks, least distorted first.

    Every pair of :func:`_oblique_candidate_peaks` is labelled ``(1, 0)``/
    ``(0, 1)`` in both orders, giving a closed-form affine each (see
    :func:`_two_point_model`).  Each carries its distortion
    ``d(M) = max(|lambda - 1|)`` relative to ``b_ideal``, which is the key the
    caller uses to pick the gauge (design addendum A-10): an oblique labelling is
    only defined up to GL(2, Z), so the residual cannot choose it.
    """
    order = _oblique_candidate_peaks(q_obs, snr, b_ideal, top_n=top_n)
    candidates = []
    for position, first in enumerate(order):
        for second in order[position + 1 :]:
            for pairs in (
                [(int(first), (1, 0)), (int(second), (0, 1))],
                [(int(second), (1, 0)), (int(first), (0, 1))],
            ):
                model = _two_point_model(b_ideal, q_obs, pairs)
                if model is None:
                    continue
                candidates.append((pairs, model, _affine_distortion(model)))
    candidates.sort(key=lambda item: item[2])
    return candidates


def _oblique_gauge_solutions(usable, q_obs, snr, b_ideal, pool_hk, pool_q, dq_nm_inv):
    """Expand and refit every oblique gauge candidate, best gauge first.

    Ranking: ``d(M)`` of the converged fit (distortion relative to the caller's
    reference basis), then more labelled peaks, then the weighted chi-square.
    Returns ``(distortion, labelled, fit)`` triples.
    """
    scored = []
    for pairs, model, _seed_distortion in _oblique_seed_candidates(
        q_obs, snr, b_ideal
    ):
        labelled = _LabelledSet(b_ideal, dq_nm_inv, usable)
        for index, label in pairs:
            labelled.add(usable[index], label)
        fit = labelled.expand_and_refit(pool_hk, pool_q, model=model)
        if fit is None:
            continue
        if abs(float(np.linalg.det(b_ideal @ fit[0]))) <= 1e-9:
            continue
        scored.append(
            (
                _affine_distortion(fit[0]),
                -len(labelled.entries),
                float(fit[2]),
                labelled,
                fit,
            )
        )
    scored.sort(key=lambda item: item[:3])
    return [(item[0], item[3], item[4]) for item in scored]


def _oblique_seeds(q_obs, snr, b_ideal):
    """Two-point oblique seeds (labels (1,0) and (0,1)) with their closed model."""
    seed_pairs = _oblique_seed(q_obs, snr)
    if seed_pairs is None:
        return []
    return [(seed_pairs, _two_point_model(b_ideal, q_obs, seed_pairs))]


def _seed_assignments(q_obs, b_ideal, symmetry, orientation_deg):
    """Candidate first-ring label assignments.

    Only half of a ring survives the ``+/-q`` half-plane rule of section 4.2-5
    (3 of 6 hexagonal points, 2 of 4 square points), so the quota is per class:
    ``square >= 2``, ``hexagonal >= 3``.  With a known orientation the ring
    points are matched to the nearest ideal first-ring label; without one they
    are sorted by polar angle and aligned with the ideal first-ring labels
    through every cyclic offset and both handednesses (the resulting 60/90/180
    degree ambiguity is reported in ``LatticeFit.rotation_mod_deg`` and
    ``rotation_is_absolute``).  A class without a canonical first ring
    (oblique) or too few ring points yields ``[]``.

    Every seed is a ``(pairs, model)`` tuple: the labelled ``(index, (h, k))``
    entries plus an optional closed-form affine that the first expansion round
    uses when fewer than three points are labelled.
    """
    if symmetry not in ("hexagonal", "square"):
        return []
    labels = _first_ring_labels(symmetry)
    quota = 3 if symmetry == "hexagonal" else 2
    b1_norm = float(np.hypot(b_ideal[0, 0], b_ideal[0, 1]))
    radii = np.hypot(q_obs[:, 0], q_obs[:, 1])
    ring = np.nonzero(np.abs(radii / b1_norm - 1.0) < 0.15)[0]
    if ring.size < quota:
        return []
    order = ring[np.argsort(np.arctan2(q_obs[ring, 1], q_obs[ring, 0]))]
    if orientation_deg is not None:
        ideal_ring = np.array(labels, dtype=float) @ b_ideal
        assignment = []
        for index in order[: len(labels)]:
            vector = q_obs[index]
            distances = np.hypot(
                ideal_ring[:, 0] - vector[0], ideal_ring[:, 1] - vector[1]
            )
            best = int(np.argmin(distances))
            if distances[best] <= 0.3 * b1_norm:
                assignment.append((int(index), labels[best]))
        if len(assignment) < quota:
            return []
        ideal = np.array([np.asarray(hk, dtype=float) @ b_ideal for _, hk in assignment])
        if np.linalg.matrix_rank(ideal, tol=1e-9 * float(np.max(np.abs(ideal)))) < 2:
            return []
        return [(assignment, _two_point_model(b_ideal, q_obs, assignment))]
    selected = order[: len(labels)]
    candidates = []
    for offset in range(len(labels)):
        for handedness in (1, -1):
            pairs = [
                (int(index), labels[(offset + handedness * position) % len(labels)])
                for position, index in enumerate(selected)
            ]
            candidates.append((pairs, None))
    return candidates

def _infer_symmetry(q_obs: np.ndarray, snr: np.ndarray, min_snr_seed: float):
    """Best-effort symmetry class from the ring multiplicity.

    ``q_obs`` holds the half-plane representatives only (design section 4.2-5),
    so the ring multiplicity is doubled before it is compared with the class
    thresholds of the design spec: for a real image |F| is exactly centro-
    symmetric, so every mirrored ring point exists with the same amplitude and
    counting it is not a second independent measurement.  The doubled count is
    always even, hence exactly three classes can be returned -- hexagonal
    (>= 6), square (4) and oblique (<= 2); the odd "ambiguous" case of the spec
    text cannot occur and is not implemented.

    Returns ``(symmetry, first_ring_radius_nm_inv)``; the radius is None for an
    oblique lattice.
    """
    strong = np.nonzero(np.asarray(snr, dtype=float) >= min_snr_seed)[0]
    if strong.size < 3:
        return None, None
    radii = np.hypot(q_obs[strong, 0], q_obs[strong, 1])
    ratio = radii / max(float(radii.min()), 1e-12)
    first_ring = strong[ratio < 1.15]
    multiplicity = 2 * int(first_ring.size)
    radius = float(np.median(np.hypot(q_obs[first_ring, 0], q_obs[first_ring, 1])))
    if multiplicity >= 5:
        return "hexagonal", radius
    if multiplicity == 4:
        return "square", radius
    return "oblique", None


def _inferred_basis(symmetry: str, radius_nm_inv: float) -> np.ndarray:
    """Ideal basis for an inferred hexagonal/square lattice."""
    if symmetry == "hexagonal":
        return np.array(
            [
                [radius_nm_inv, 0.0],
                [
                    radius_nm_inv * np.cos(np.pi / 3.0),
                    radius_nm_inv * np.sin(np.pi / 3.0),
                ],
            ]
        )
    if symmetry == "square":
        return np.array([[radius_nm_inv, 0.0], [0.0, radius_nm_inv]])
    raise ValueError(f"cannot build an inferred basis for {symmetry!r}")


def _rotation_angle_deg(matrix: np.ndarray) -> float:
    """Polar-decomposition rotation angle of ``matrix`` in degrees."""
    rot, _ = polar(matrix)
    return float(np.degrees(np.arctan2(rot[0, 1], rot[0, 0])))


def _rotation_sigma_deg(matrix: np.ndarray, cov_m: np.ndarray) -> float:
    """1-sigma of :func:`_rotation_angle_deg` from a numerical Jacobian."""
    flat = np.asarray(matrix, dtype=float).ravel()
    jac = np.zeros(4)
    for index in range(4):
        step = 1e-6 * max(1.0, abs(flat[index]))
        plus = flat.copy()
        minus = flat.copy()
        plus[index] += step
        minus[index] -= step
        jac[index] = (
            _rotation_angle_deg(plus.reshape(2, 2))
            - _rotation_angle_deg(minus.reshape(2, 2))
        ) / (2.0 * step)
    variance = float(jac @ cov_m @ jac)
    if not np.isfinite(variance) or variance < 0.0:
        return float("nan")
    return float(np.sqrt(variance))


def _lattice_stage(
    reps: list[dict],
    *,
    lattice: LatticeSpec | None,
    n: int,
    dq_nm_inv: float,
    q_max_px: float,
    dc_radius_px: float,
    noise_sigma: float,
    magnitude: np.ndarray,
    min_snr_seed: float,
    min_snr_verify: float,
    lattice_tolerance: float,
    max_iterations: int,
    patch_half: int | None,
    sigma_model_floor_px: float,
    chi2_red_max: float,
    residual_max_px: float | None,
    min_pool_spacing_px: float,
):
    """Fit the lattice, run the prediction loop and fill the model fields.

    Returns ``(LatticeFit | None, meta_dict)``.  Every peak that ends up
    consistent with the fitted lattice gets ``index_hk``, ``q_model_px``,
    ``sigma_q_model_px`` and ``residual_px`` filled in; unexplained peaks stay in
    the output with ``index_hk = None``.

    Seed construction follows design section 4.4(a): with a known orientation
    the *whole* ideal pool is matched by nearest neighbour (tolerance
    ``0.3 |b1|``); without one the canonical first-ring enumeration is used, and
    a class without a canonical first ring (oblique, or a caller-supplied basis)
    falls back to a two-point seed that is labelled, expanded and refitted.

    When no reference lattice is given the reference basis is inferred from the
    data, so its absolute scale is not observable; the fit is gauge-fixed to
    ``|det M| = 1`` (a deviatoric convention) which leaves ``B_fit`` and every
    predicted position untouched while making the reported principal stretches
    and linearised strain physically meaningful.
    """
    meta = {
        "inferred_symmetry": None,
        "lattice_error": None,
        "n_added_by_loop": 0,
        "pool_from_lattice_operations": None,
        "n_lattice_points": 0,
        "reference_gauge": "user" if lattice is not None else "deviatoric_det_M_1",
        "oblique_gauge": None,
        "oblique_gauge_margin": None,
    }
    usable = [
        peak
        for peak in reps
        if peak["base_quality"] in ("ok", "coarse") and not peak["edge"]
    ]
    if len(usable) < 3:
        meta["lattice_error"] = (
            f"only {len(usable)} usable half-plane peak(s); at least 3 independent "
            "points are required for a lattice fit"
        )
        logger.warning("bragg_peak_detection: %s", meta["lattice_error"])
        _clear_model_fields(reps)
        return None, meta

    q_obs = np.array(
        [[p["qx_px"] * dq_nm_inv, p["qy_px"] * dq_nm_inv] for p in usable]
    )
    snr = np.array([p["snr"] for p in usable])
    q_max_nm_inv = q_max_px * dq_nm_inv
    has_reference = lattice is not None
    orientation_is_absolute = bool(
        lattice is not None and lattice.orientation_deg is not None
    )

    if lattice is not None:
        symmetry = lattice.symmetry
        h_max = lattice.h_max
        if lattice.bvecs_nm_inv is not None:
            b_ideal = _ideal_basis(None, symmetry, lattice.bvecs_nm_inv)
        else:
            b_ideal = _rotate_basis(
                _ideal_basis(lattice.a_nm, symmetry, None), lattice.orientation_deg
            )
        pool_hk, pool_q, used_ops = _ideal_pool(b_ideal, q_max_nm_inv * 1.5, h_max)
        b1_norm = float(np.hypot(b_ideal[0, 0], b_ideal[0, 1]))
        if lattice.orientation_deg is not None:
            seeds = _full_pool_assignments(
                q_obs, b_ideal, pool_hk, pool_q, 0.3 * b1_norm
            )
            if not seeds:
                seeds = _seed_assignments(
                    q_obs, b_ideal, symmetry, lattice.orientation_deg
                )
        else:
            seeds = _seed_assignments(q_obs, b_ideal, symmetry, None)
        if seeds:
            solutions = _solve_seeds(
                usable, q_obs, seeds, b_ideal, pool_hk, pool_q, dq_nm_inv
            )
            if not solutions:
                meta["lattice_error"] = (
                    "no candidate seed assignment produced a rank-4 lattice fit "
                    "(at least 3 non-collinear labelled peaks are required)"
                )
                logger.warning("bragg_peak_detection: %s", meta["lattice_error"])
                _clear_model_fields(reps)
                meta["pool_from_lattice_operations"] = used_ops
                return None, meta
            labelled, fit = solutions[0]
        else:
            if lattice.orientation_deg is not None:
                logger.warning(
                    "bragg_peak_detection: orientation_deg=%g could not label the "
                    "ideal pool (nearest-neighbour tolerance is 0.3*|b1|, about "
                    "+/-17.4 deg for a hexagonal first ring); falling back to the "
                    "orientation-free two-point seed.",
                    float(lattice.orientation_deg),
                )
            gauge_solutions = _oblique_gauge_solutions(
                usable, q_obs, snr, b_ideal, pool_hk, pool_q, dq_nm_inv
            )
            if not gauge_solutions:
                meta["lattice_error"] = (
                    "no two-point seed produced a rank-4 lattice fit for this "
                    "symmetry class"
                )
                logger.warning("bragg_peak_detection: %s", meta["lattice_error"])
                _clear_model_fields(reps)
                meta["pool_from_lattice_operations"] = used_ops
                return None, meta
            distortions = [entry[0] for entry in gauge_solutions]
            meta["oblique_gauge"] = "least_distortion"
            if len(distortions) > 1 and distortions[0] > 0.0:
                meta["oblique_gauge_margin"] = float(
                    distortions[1] / distortions[0]
                )
            else:
                meta["oblique_gauge_margin"] = float("inf")
            labelled, fit = gauge_solutions[0][1], gauge_solutions[0][2]
    else:
        inferred, radius = _infer_symmetry(q_obs, snr, min_snr_seed)
        meta["inferred_symmetry"] = inferred
        if inferred is None:
            meta["lattice_error"] = "no symmetry class could be inferred from the peaks"
            logger.warning("bragg_peak_detection: %s", meta["lattice_error"])
            _clear_model_fields(reps)
            return None, meta
        symmetry = inferred
        h_max = 8
        if inferred == "oblique":
            seed_pairs = _oblique_seed(q_obs, snr)
            if seed_pairs is None:
                meta["lattice_error"] = (
                    "the oblique seed needs two strong non-collinear peaks"
                )
                return None, meta
            b_ideal = _oblique_basis(q_obs, seed_pairs)
            seeds = [(seed_pairs, None)]
        else:
            b_ideal = _inferred_basis(inferred, radius)
            seeds = _seed_assignments(q_obs, b_ideal, inferred, None)
        pool_hk, pool_q, used_ops = _ideal_pool(b_ideal, q_max_nm_inv * 1.5, h_max)
        solutions = _solve_seeds(
            usable, q_obs, seeds, b_ideal, pool_hk, pool_q, dq_nm_inv
        )
        if not solutions:
            meta["lattice_error"] = "the inferred lattice could not be fitted"
            logger.warning("bragg_peak_detection: %s", meta["lattice_error"])
            _clear_model_fields(reps)
            meta["pool_from_lattice_operations"] = used_ops
            return None, meta
        labelled, fit = solutions[0]
        # Deviatoric gauge: B -> s B, M -> M / s leaves B_fit and every model
        # position invariant while forcing |det M| = 1.
        scale = float(np.sqrt(abs(np.linalg.det(fit[0]))))
        if np.isfinite(scale) and scale > 0.0:
            model, cov_m, chi2 = fit
            labelled.rescale_basis(scale)
            b_ideal = labelled.b_ideal
            fit = (model / scale, cov_m / scale**2, chi2)
            pool_hk, pool_q, used_ops = _ideal_pool(
                b_ideal, q_max_nm_inv * 1.5, h_max
            )
    meta["pool_from_lattice_operations"] = used_ops

    model, cov_m, chi2 = fit
    b_fit, cov_b = _basis_and_covariance(b_ideal, model, cov_m)
    n_iterations = 0
    n_added = 0
    for iteration in range(max_iterations):
        added = _prediction_round(
            reps, labelled, model, cov_b, b_ideal, n, dq_nm_inv, q_max_nm_inv,
            dc_radius_px * dq_nm_inv, noise_sigma, magnitude, min_snr_verify,
            lattice_tolerance, patch_half, sigma_model_floor_px, lattice,
        )
        if not added:
            break
        n_added += added
        refit = labelled.fit()
        if refit is None:
            break
        model, cov_m, chi2 = refit
        b_fit, cov_b = _basis_and_covariance(b_ideal, model, cov_m)
        n_iterations = iteration + 1
    meta["n_added_by_loop"] = int(n_added)
    meta["n_lattice_points"] = len(labelled.entries)

    final = labelled.fit()
    if final is None:
        meta["lattice_error"] = "the final weighted lattice fit is rank deficient"
        logger.warning("bragg_peak_detection: %s", meta["lattice_error"])
        _clear_model_fields(reps)
        return None, meta
    model, cov_m, chi2 = final
    if abs(float(np.linalg.det(b_ideal @ model))) <= 1e-9:
        meta["lattice_error"] = "the fitted reciprocal basis is singular (|det B| <= 1e-9)"
        logger.warning("bragg_peak_detection: %s", meta["lattice_error"])
        _clear_model_fields(reps)
        return None, meta
    if not np.isfinite(chi2):
        meta["lattice_error"] = "the weighted chi-square of the lattice fit is not finite"
        logger.warning("bragg_peak_detection: %s", meta["lattice_error"])
        _clear_model_fields(reps)
        return None, meta
    if not has_reference:
        # Re-apply the deviatoric gauge to the *final* affine so that
        # |det M| == 1 holds exactly for the published stretches; the rescaling
        # leaves B_fit and every model position invariant.
        final_scale = float(np.sqrt(abs(np.linalg.det(model))))
        if np.isfinite(final_scale) and final_scale > 0.0:
            labelled.rescale_basis(final_scale)
            b_ideal = labelled.b_ideal
            model = model / final_scale
            cov_m = cov_m / final_scale**2
    b_fit, cov_b = _basis_and_covariance(b_ideal, model, cov_m)
    rms_residual_px, consistent_fraction = _residual_and_consistency(
        labelled, b_fit, dq_nm_inv, lattice_tolerance
    )
    spacing_px = _min_pool_spacing_px(b_fit, dq_nm_inv)
    residual_max = (
        float(residual_max_px)
        if residual_max_px is not None
        else min(1.0, 0.25 * spacing_px)
    )
    failures = []
    if min_pool_spacing_px > 0.0 and spacing_px < float(min_pool_spacing_px):
        failures.append(
            f"spacing_below_min (spacing_px={spacing_px:.3f} < "
            f"{float(min_pool_spacing_px):g})"
        )
    if np.isfinite(rms_residual_px) and rms_residual_px > residual_max:
        failures.append(
            f"residual_above_max (rms_residual_px={rms_residual_px:.4f} > "
            f"{residual_max:.4f})"
        )
    if np.isfinite(chi2) and chi2 > float(chi2_red_max):
        failures.append(
            f"chi2_above_max (chi2_reduced={chi2:.4g} > {float(chi2_red_max):g})"
        )
    fit_ok = not failures
    quality = "ok" if fit_ok else "+".join(
        failure.split(" ")[0] for failure in failures
    )
    if fit_ok:
        _fill_model_fields(labelled, b_fit, cov_b, dq_nm_inv)
    else:
        _clear_model_fields(reps)
        meta["lattice_error"] = "lattice fit rejected: " + "; ".join(failures)
        logger.warning("bragg_peak_detection: %s", meta["lattice_error"])
    meta["fit_ok"] = bool(fit_ok)
    meta["lattice_quality"] = quality
    meta["pool_spacing_px"] = float(spacing_px)
    meta["residual_max_px"] = float(residual_max)
    meta["consistent_fraction"] = float(consistent_fraction)
    meta["min_pool_spacing_px"] = float(min_pool_spacing_px)
    meta["chi2_red_max"] = float(chi2_red_max)
    lattice_fit = _build_lattice_fit(
        model, cov_m, b_ideal, labelled, chi2, n_iterations, symmetry,
        orientation_is_absolute, has_reference, rms_residual_px,
        consistent_fraction, spacing_px, (fit_ok, quality, residual_max),
    )
    return lattice_fit, meta


def _solve_seeds(usable, q_obs, seeds, b_ideal, pool_hk, pool_q, dq_nm_inv):
    """Try every seed, expand it and rank the results by weighted chi-square.

    A seed is ``(pairs, model)``; a ``None`` model is completed with the
    closed-form two-point affine when the seed labels fewer than three points.
    Ranking uses the weighted ``chi2_reduced`` of the expanded fit (design
    section 4.4(a)), with the larger ``|det B|`` breaking exact ties.
    """
    solutions = []
    for pairs, model in seeds:
        if not pairs:
            continue
        if model is None and len(pairs) < 3:
            model = _two_point_model(b_ideal, q_obs, pairs)
            if model is None:
                continue
        labelled = _LabelledSet(b_ideal, dq_nm_inv, usable)
        for index, label in pairs:
            labelled.add(usable[index], label)
        fit = labelled.expand_and_refit(pool_hk, pool_q, model=model)
        if fit is None:
            continue
        determinant = abs(float(np.linalg.det(b_ideal @ fit[0])))
        if determinant <= 1e-9:
            continue
        solutions.append(((float(fit[2]), -determinant), labelled, fit))
    solutions.sort(key=lambda item: item[0])
    return [(labelled, fit) for _, labelled, fit in solutions]


def _oblique_seed(q_obs, snr):
    """Seed labels for an oblique lattice: the two shortest strong peaks.

    The two shortest non-collinear peaks above the median SNR are the most
    likely images of the reference rows ``b1``/``b2``.  Ordering by SNR instead
    (the first repair round) tends to pick higher-order vectors, which generates
    a *sublattice* and leaves most observed peaks unlabelled -- measured on a
    synthetic oblique case: 0.541 of the usable representatives versus 1.000
    with the radius-first order.
    """
    radii = np.hypot(q_obs[:, 0], q_obs[:, 1])
    strong = np.asarray(snr, dtype=float) >= float(np.median(snr))
    order = [int(index) for index in np.argsort(radii) if strong[index]]
    if len(order) < 2:
        order = [int(index) for index in np.argsort(radii)]
    for position, first in enumerate(order[:4]):
        for second in order[position + 1 :][:4]:
            cross = float(
                q_obs[first, 0] * q_obs[second, 1] - q_obs[first, 1] * q_obs[second, 0]
            )
            norm = float(np.hypot(*q_obs[first]) * np.hypot(*q_obs[second]))
            if norm > 0.0 and abs(cross) / norm > 0.2:
                return [(int(first), (1, 0)), (int(second), (0, 1))]
    return None


def _oblique_basis(q_obs, seed_pairs):
    """Basis rows for the inference-path oblique seed: the two seed peak vectors."""
    rows = [q_obs[index] for index, _ in seed_pairs]
    return np.vstack(rows)


def _clear_model_fields(peaks) -> None:
    """Drop every label and model quantity from ``peaks``.

    Called on a hard lattice failure (no seed solution, rank-deficient final
    fit), on a candidate-seed rollback and on a quality-gate rejection, so that
    ``BraggPeak.index_hk`` can never outlive the fit that produced it (design
    S-1/S-2).  ``quality`` then falls back to the existing "unmatched".
    """
    for peak in peaks:
        peak["index_hk"] = None
        peak["q_model_px"] = None
        peak["sigma_q_model_px"] = None
        peak["residual_px"] = None


def _min_pool_spacing_px(b_fit, dq_nm_inv: float) -> float:
    """Smallest distance between predicted points, ``min |(h,k)@B_fit|`` in px.

    The minimum is taken over ``(h, k) != (0, 0)`` with ``|h|, |k| <= 2``, i.e.
    over the neighbourhood of the origin whose spacing decides whether the
    detector could ever resolve two lattice points separately (G-1).
    """
    smallest = np.inf
    for h in range(-2, 3):
        for k in range(-2, 3):
            if (h, k) == (0, 0):
                continue
            point = np.asarray([h, k], dtype=float) @ b_fit
            smallest = min(smallest, float(np.hypot(point[0], point[1])))
    return smallest / dq_nm_inv


def _residual_and_consistency(labelled, b_fit, dq_nm_inv, lattice_tolerance):
    """``(rms_residual_px, consistent_fraction)`` of a labelled set, in pixels.

    The model prediction ``(h, k) @ B_fit`` is in nm^-1, so every residual is
    divided by ``dq_nm_inv`` before it is reported: both returned quantities are
    in FFT pixels (design A-15; the nm^-1 form of ``rms_residual_px`` used to
    leak into the G-2 gate and relaxed it by ``1/dq``).

    ``consistent_fraction`` is the share of labelled peaks with

        ``residual_px <= lattice_tolerance * hypot(sigma_q_px)``

    using the **measurement sigma only** (design A-15).  The model-prediction
    sigma is deliberately excluded: it follows ``cov_bvecs``, which the GLS
    inflates by ``max(1, chi2_reduced)``, so a worse fit would widen the window
    and score a *higher* consistency -- a self-fulfilling statistic.  The fitted
    inflation is dominated by that model term (medians 1.14 / 3.90 / 0.34 px on
    the real scans versus 0.15 / 0.12 / 0.24 px for the point sigma), not by the
    ``1/sqrt(12)`` quantisation floor, and neither is clamped.  The prediction
    loop's acceptance test keeps the *combined* sigma on purpose; the two are
    related but separate definitions and must not be conflated (see
    :func:`_prediction_round`).

    The statistics are computed on the labelled set the gates see, i.e. before
    any label clearing, so a **rejected** fit still reports an auditable value.
    """
    residuals = []
    consistent = 0
    for peak, hk in labelled.entries:
        predicted = np.asarray(hk, dtype=float) @ b_fit
        residual_px = float(
            np.hypot(
                predicted[0] - peak["qx_px"] * dq_nm_inv,
                predicted[1] - peak["qy_px"] * dq_nm_inv,
            )
            / dq_nm_inv
        )
        residuals.append(residual_px)
        sigma_q_px = np.asarray(peak["sigma_px"], dtype=float)
        if residual_px <= float(lattice_tolerance) * float(np.hypot(*sigma_q_px)):
            consistent += 1
    if not residuals:
        return float("nan"), 0.0
    rms = float(np.sqrt(np.mean(np.square(residuals))))
    return rms, consistent / len(residuals)


def _basis_and_covariance(b_ideal, model, cov_m):
    """``(B_fit, cov(vec_row B_fit))`` with ``cov_B = (B_ideal x I2) cov_M (...)``."""
    transform = np.kron(b_ideal, np.eye(2))
    return b_ideal @ model, transform @ cov_m @ transform.T
def _prediction_round(
    reps, labelled, model, cov_b, b_ideal, n, dq_nm_inv, q_max_nm_inv,
    dc_radius_nm_inv, noise_sigma, magnitude, min_snr_verify, lattice_tolerance,
    patch_half, sigma_model_floor_px, lattice,
):
    """One ``predict -> verify -> localize -> accept`` pass of the refinement loop.

    A proposed peak is accepted when its distance to the lattice prediction is
    within ``lattice_tolerance * hypot(sigma_q_px, sigma_pred_px)``.  Both sigma
    terms already include their own model-floor contribution (the peak through
    ``cov_q_px``, the prediction through the GLS covariance), so the floor is
    never added a third time, and the prediction sigma is converted from nm^-1
    to pixels before it is combined.
    """
    h_max = lattice.h_max if lattice is not None else 8
    known = np.array([[p["qx_px"] * dq_nm_inv, p["qy_px"] * dq_nm_inv] for p in reps])
    added = 0
    for h in range(-h_max, h_max + 1):
        for k in range(-h_max, h_max + 1):
            if (h, k) == (0, 0):
                continue
            point = np.array([h, k], dtype=float) @ b_ideal @ model
            radius_nm_inv = float(np.hypot(point[0], point[1]))
            if radius_nm_inv <= dc_radius_nm_inv or radius_nm_inv > q_max_nm_inv:
                continue
            predicted_qx_px = point[0] / dq_nm_inv
            predicted_qy_px = point[1] / dq_nm_inv
            if not (
                predicted_qy_px * _HALF_PLANE_SIGN > 1e-9
                or (
                    abs(predicted_qy_px) <= 1e-9
                    and predicted_qx_px * _HALF_PLANE_SIGN > 0.0
                )
            ):
                continue
            if (
                known.size
                and np.min(
                    np.hypot(known[:, 0] - point[0], known[:, 1] - point[1])
                )
                < 1.5 * dq_nm_inv
            ):
                continue
            row = round(point[1] / dq_nm_inv + n // 2)
            col = round(point[0] / dq_nm_inv + n // 2)
            if not (0 <= row < n and 0 <= col < n):
                continue
            r0, r1 = max(0, row - 1), min(n, row + 2)
            c0, c1 = max(0, col - 1), min(n, col + 2)
            if float(magnitude[r0:r1, c0:c1].max()) / noise_sigma < min_snr_verify:
                continue
            peak = _localize_peak(
                magnitude, row, col, noise_sigma,
                patch_half=patch_half, bounds_half=1.5,
                sigma_model_floor_px=sigma_model_floor_px,
            )
            if peak["edge"] or peak["base_quality"] == "failed":
                continue
            localized = np.array(
                [peak["qx_px"] * dq_nm_inv, peak["qy_px"] * dq_nm_inv]
            )
            if (
                known.size
                and np.min(
                    np.hypot(known[:, 0] - localized[0], known[:, 1] - localized[1])
                )
                < _MIN_DUPLICATE_DISTANCE_PX * dq_nm_inv
            ):
                continue
            sigma_pred_px = _predict_sigma(cov_b, h, k) / dq_nm_inv
            sigma_total_px = np.hypot(
                np.asarray(peak["sigma_px"], dtype=float), sigma_pred_px
            )
            offset = np.hypot(
                peak["qx_px"] * dq_nm_inv - point[0],
                peak["qy_px"] * dq_nm_inv - point[1],
            ) / dq_nm_inv
            if offset > lattice_tolerance * float(np.hypot(*sigma_total_px)):
                continue
            labelled.add(peak, (h, k))
            reps.append(peak)
            known = np.vstack(
                [known, [peak["qx_px"] * dq_nm_inv, peak["qy_px"] * dq_nm_inv]]
            )
            added += 1
    return added
def _fill_model_fields(labelled, b_fit, cov_b, dq_nm_inv):
    """Attach model position, sigma and residual to every labelled peak.

    ``sigma_q_model_px`` is the propagated 1-sigma of ``(h, k) @ B_fit``, i.e.
    ``sqrt(diag(J_B @ cov_bvecs_nm_inv @ J_B^T))`` converted to pixels.
    """
    for peak, hk in labelled.entries:
        point = np.asarray(hk, dtype=float) @ b_fit
        sigma_nm_inv = _predict_sigma(cov_b, int(hk[0]), int(hk[1]))
        peak["index_hk"] = (int(hk[0]), int(hk[1]))
        peak["q_model_px"] = (float(point[0] / dq_nm_inv), float(point[1] / dq_nm_inv))
        peak["sigma_q_model_px"] = (
            float(sigma_nm_inv[0] / dq_nm_inv),
            float(sigma_nm_inv[1] / dq_nm_inv),
        )
        peak["residual_px"] = float(
            np.hypot(
                peak["qx_px"] - point[0] / dq_nm_inv,
                peak["qy_px"] - point[1] / dq_nm_inv,
            )
        )
def _build_lattice_fit(
    model, cov_m, b_ideal, labelled, chi2_reduced, n_iterations, symmetry,
    orientation_is_absolute, has_reference, rms_residual_px, consistent_fraction,
    spacing_px, gated,
):
    """Assemble a :class:`LatticeFit` from a converged GLS solution.

    ``linearized_strain`` is the right stretch tensor of the affine,
    ``polar(M)[1] - I``, which is invariant under the point-group relabelling
    ambiguity of design D15 (``0.5 (M + M^T) - I`` is not).  ``rotation_deg`` is
    always reported from the polar factor of ``M``: with a reference lattice it
    is the distortion rotation, and on the inference path it is the rotation of
    the canonical labelling, meaningful modulo ``rotation_mod_deg``.

    ``gated`` carries the acceptance outcome ``(fit_ok, quality, residual_max)``
    (the considered thresholds); ``spacing_px`` is the **measured** G-1 quantity
    that the caller also echoes into ``meta``.  A rejected fit keeps its basis,
    covariance, chi2, rms, spacing and consistency diagnostics, but every derived
    orientation/strain quantity is withheld (None) because it is not trustworthy.
    """
    fit_ok, quality, residual_max = gated
    b_fit, cov_b = _basis_and_covariance(b_ideal, model, cov_m)
    stretches = tuple(
        float(value) for value in sorted(np.linalg.svd(model, compute_uv=False))
    )
    _, stretch_tensor = polar(model)
    if not fit_ok:
        strain_gauge = "undefined"
    elif has_reference:
        strain_gauge = "absolute"
    elif symmetry in ("hexagonal", "square"):
        # Inferred reference: |det M| = 1 leaves only the shape (deviatoric)
        # distortion observable, which is exactly what is published.
        strain_gauge = "deviatoric"
    else:
        # An inferred oblique reference is built from the seed peak vectors
        # themselves, so its affine is the identity by construction and a
        # "strain" there would be a meaningless tautology (design A-11).
        strain_gauge = "undefined"
    publish_shape = fit_ok and strain_gauge in ("absolute", "deviatoric")
    if publish_shape:
        rotation_deg = _rotation_angle_deg(model)
        rotation_sigma_deg = _rotation_sigma_deg(model, cov_m)
        linearized_strain = stretch_tensor - np.eye(2)
    else:
        rotation_deg = None
        rotation_sigma_deg = None
        linearized_strain = None
    if has_reference and fit_ok:
        affine = model
        cov_affine = cov_m
    else:
        affine = None
        cov_affine = None
    return LatticeFit(
        bvecs_nm_inv=b_fit,
        cov_bvecs_nm_inv=cov_b,
        affine=affine,
        cov_affine=cov_affine,
        rotation_deg=rotation_deg,
        rotation_sigma_deg=rotation_sigma_deg,
        rotation_mod_deg=_point_group_mod_deg(symmetry),
        rotation_is_absolute=bool(has_reference and orientation_is_absolute and fit_ok),
        principal_stretches=stretches if publish_shape else None,
        linearized_strain=linearized_strain,
        n_independent=len(labelled.entries),
        chi2_reduced=float(chi2_reduced),
        rms_residual_px=float(rms_residual_px),
        n_iterations=int(n_iterations),
        symmetry=symmetry,
        fit_ok=bool(fit_ok),
        quality=quality,
        pool_spacing_px=float(spacing_px),
        residual_max_px=float(residual_max),
        consistent_fraction=float(consistent_fraction),
        strain_gauge=strain_gauge,
    )


class BraggPeakDetector:
    """Reusable detector with the same keyword arguments as :func:`detect_bragg_peaks`.

    The reference orientation follows the same convention as
    :func:`detect_bragg_peaks`: ``LatticeSpec.orientation_deg`` rotates the
    reference basis counter-clockwise and the known-orientation seed match uses a
    ``0.3 * |b1|`` tolerance (about ``+/- 17.4 deg`` for a hexagonal first ring).
    """

    _KNOWN = frozenset(
        {
            "lattice",
            "min_snr",
            "min_snr_seed",
            "min_snr_verify",
            "dc_mask_frac",
            "footprint",
            "patch_half",
            "max_peaks",
            "q_max_px",
            "lattice_tolerance",
            "max_iterations",
            "sigma_model_floor_px",
            "chi2_red_max",
            "residual_max_px",
            "min_pool_spacing_px",
            "subtract_plane",
            "nan_policy",
            "window",
        }
    )

    def __init__(self, **kwargs) -> None:
        unknown = set(kwargs) - self._KNOWN
        if unknown:
            raise TypeError(f"unexpected keyword argument(s): {sorted(unknown)}")
        self.options = dict(kwargs)

    def detect(self, image: np.ndarray, size_nm: float) -> BraggDetectionResult:
        """Detect peaks in a real-space image (see :func:`detect_bragg_peaks`)."""
        array = np.asarray(image)
        nan_filled = (
            int(np.count_nonzero(~np.isfinite(array))) if array.ndim == 2 else 0
        )
        image_scale = _finite_scale(array)
        fft2 = compute_fft2(
            image,
            size_nm,
            window=self.options.get("window", "hann"),
            subtract_plane=self.options.get("subtract_plane", True),
            nan_policy=self.options.get("nan_policy", "plane"),
        )
        options = dict(self.options)
        options["_nan_filled"] = nan_filled
        options["_image_scale"] = image_scale
        return _run_pipeline(fft2, float(size_nm), options, return_fft2=False)

    def detect_from_fft2(
        self, fft2: np.ndarray, size_nm: float, *, return_fft2: bool = False
    ) -> BraggDetectionResult:
        """Detect peaks in an already fftshifted complex spectrum.

        The spectrum is used as-is: no windowing and no plane subtraction are
        applied here (use :func:`compute_fft2` for those).
        """
        spectrum = np.asarray(fft2)
        if spectrum.ndim != 2 or spectrum.shape[0] != spectrum.shape[1]:
            raise ValueError(
                f"fft2 must be a square 2-D array, got shape {spectrum.shape}"
            )
        if not np.isfinite(size_nm) or size_nm <= 0:
            raise ValueError(f"size_nm must be a positive number, got {size_nm!r}")
        return _run_pipeline(
            spectrum, float(size_nm), self.options, return_fft2=return_fft2
        )


def detect_bragg_peaks(
    image: np.ndarray,
    size_nm: float,
    *,
    lattice: LatticeSpec | None = None,
    min_snr: float = 5.0,
    min_snr_seed: float = 8.0,
    min_snr_verify: float = 3.5,
    dc_mask_frac: float = 0.03,
    footprint: int | None = None,
    patch_half: int | None = None,
    max_peaks: int | None = None,
    q_max_px: float | None = None,
    lattice_tolerance: float = 3.0,
    max_iterations: int = 3,
    sigma_model_floor_px: float = 0.006,
    chi2_red_max: float = 25.0,
    residual_max_px: float | None = None,
    min_pool_spacing_px: float = 3.0,
    return_fft2: bool = False,
) -> BraggDetectionResult:
    """Detect, localize and lattice-refine the Bragg peaks of an STM topograph.

    Parameters
    ----------
    image : np.ndarray
        Square real-space image, in the same units as ``size_nm``.
    size_nm : float
        Physical side length of the field of view in nm.
    lattice : LatticeSpec or None
        Ideal reference lattice.  ``None`` triggers a best-effort symmetry
        inference and disables the affine matrix (no reference orientation).
        ``LatticeSpec.orientation_deg`` is a **counter-clockwise** rotation of
        the reference basis (``basis @ rot.T``, the standard right-handed
        convention); a known orientation seeds the labelling by nearest
        neighbour over the whole ideal pool with a ``0.3 * |b1|`` tolerance
        (about ``+/- 17.4 deg`` for a hexagonal first ring) and falls back to the
        orientation-free labelling, with a warning, when nothing matches.
    min_snr, min_snr_seed, min_snr_verify : float
        SNR thresholds for candidates, for the first-ring seed and for peaks
        proposed by the prediction loop.
    dc_mask_frac : float
        DC mask radius as a fraction of the image side.  Dropping candidates
        whose *integer* bin falls inside it is equivalent to a long-period
        cutoff of ``size_nm / (dc_mask_frac * n)`` (1.95 nm for a 512 px /
        30 nm image, 234 nm for a 128 px / 900 nm image), so a large real-space
        superstructure can be masked away by this one number.
    footprint, patch_half : int or None
        Explicit non-maximum-suppression footprint / localization half patch;
        ``None`` selects the adaptive rules of the design spec.
    max_peaks : int or None
        Optional cap on the number of candidates; ``None`` means no cap.
    q_max_px : float or None
        Radial cut in FFT pixels; ``None`` means ``0.95 * (n // 2)``.  It bounds
        both the candidate mask and the prediction pool.
    lattice_tolerance : float
        Acceptance threshold of the prediction loop, in predicted 1-sigma.
    max_iterations : int
        Maximum number of predict/verify/refit iterations.
    sigma_model_floor_px : float
        Model-floor sigma added to every reported per-axis uncertainty.
    chi2_red_max : float
        G-3 gate: reject the lattice fit when its weighted chi-square exceeds
        this value (``inf`` disables the gate).  The default 25.0 corresponds to
        a typical residual of 5 sigma, which tolerates the sigma under-estimate
        that a real background causes.
    residual_max_px : float or None
        G-2 gate: reject the fit when ``rms_residual_px`` exceeds this value.
        ``None`` derives ``min(1.0 px, 0.25 * spacing_px)`` from the fitted pool
        spacing (below one spectral bin and below half the Voronoi half-spacing).
    min_pool_spacing_px : float
        G-1 gate: reject the fit when the smallest predicted peak spacing
        ``min |(h, k) @ B_fit|`` is below this many pixels -- two lattice points
        closer than the non-maximum-suppression floor could never be resolved
        (``0`` disables the gate).
    return_fft2 : bool
        Also return the complex fftshifted spectrum.

    Returns
    -------
    BraggDetectionResult
        Peaks sorted by descending SNR, each with sub-pixel q in px and nm^-1,
        1-sigma and 2x2 covariance, SNR, quality/method flags and, when the
        lattice stage converged, the (h, k) label, model position and residual.
    """
    fft2 = compute_fft2(image, size_nm)
    options = {
        "lattice": lattice,
        "min_snr": min_snr,
        "min_snr_seed": min_snr_seed,
        "min_snr_verify": min_snr_verify,
        "dc_mask_frac": dc_mask_frac,
        "footprint": footprint,
        "patch_half": patch_half,
        "max_peaks": max_peaks,
        "q_max_px": q_max_px,
        "lattice_tolerance": lattice_tolerance,
        "max_iterations": max_iterations,
        "sigma_model_floor_px": sigma_model_floor_px,
        "chi2_red_max": chi2_red_max,
        "residual_max_px": residual_max_px,
        "min_pool_spacing_px": min_pool_spacing_px,
    }
    array = np.asarray(image)
    options["_nan_filled"] = (
        int(np.count_nonzero(~np.isfinite(array))) if array.ndim == 2 else 0
    )
    options["_image_scale"] = _finite_scale(array)
    return _run_pipeline(fft2, float(size_nm), options, return_fft2=return_fft2)


def _run_pipeline(
    spectrum: np.ndarray, size_nm: float, options: dict, *, return_fft2: bool
) -> BraggDetectionResult:
    """Run the three-stage pipeline on an fftshifted complex spectrum."""
    n = spectrum.shape[0]
    magnitude = np.abs(spectrum)
    dq_nm_inv = 2.0 * np.pi / size_nm
    q_nyquist_nm_inv = np.pi * n / size_nm
    dc_radius_px = float(options.get("dc_mask_frac", 0.03)) * n
    q_max_option = options.get("q_max_px")
    q_max_px = 0.95 * (n // 2) if q_max_option is None else float(q_max_option)
    min_snr = float(options.get("min_snr", 5.0))
    min_snr_seed = float(options.get("min_snr_seed", 8.0))
    min_snr_verify = float(options.get("min_snr_verify", 3.5))
    footprint = options.get("footprint")
    patch_half = options.get("patch_half")
    max_peaks = options.get("max_peaks")
    lattice_tolerance = float(options.get("lattice_tolerance", 3.0))
    max_iterations = int(options.get("max_iterations", 3))
    sigma_model_floor_px = float(options.get("sigma_model_floor_px", 0.006))
    chi2_red_max = float(options.get("chi2_red_max", 25.0))
    residual_option = options.get("residual_max_px")
    residual_max_px = None if residual_option is None else float(residual_option)
    min_pool_spacing_px = float(options.get("min_pool_spacing_px", 3.0))

    radius = _radius_map(n)
    high = min(0.97 * (n // 2), q_max_px)
    annulus = (radius >= 3.0 * dc_radius_px) & (radius <= high)
    if int(np.count_nonzero(annulus)) < 16:
        annulus = (radius > 0.0) & (radius <= q_max_px)
    noise_sigma = _noise_scale_two_pass(magnitude, annulus, min_snr)
    spectral_scale = float(np.max(np.abs(magnitude)))
    image_scale = float(options.get("_image_scale", 0.0))
    round_off_floor = np.finfo(float).eps * max(image_scale, 0.0) * float(n)
    floor_scale = max(
        _DEGENERATE_SCALE_FRACTION * max(spectral_scale, 0.0),
        _DEGENERATE_FLOOR_FACTOR * max(round_off_floor, np.finfo(float).tiny),
    )
    if (
        not np.isfinite(noise_sigma)
        or noise_sigma <= 0.0
        or not np.isfinite(spectral_scale)
        or spectral_scale <= 0.0
        or noise_sigma <= floor_scale
    ):
        # A constant image, an all-zero image (or any input whose |FFT|
        # background is float round-off rather than signal) has no measurable
        # noise scale: return an empty peak table for every such input instead
        # of reporting peaks that the numerical noise floor would produce.
        return _degenerate_result(
            spectrum, size_nm, magnitude, noise_sigma, dc_radius_px, q_max_px,
            dq_nm_inv, q_nyquist_nm_inv, return_fft2, options,
        )

    candidates, used_footprint = _detect_candidates(
        magnitude, noise_sigma, radius, dc_radius_px, q_max_px, min_snr, footprint
    )
    n_candidates = len(candidates)
    if max_peaks is not None:
        candidates = candidates[: int(max_peaks)]

    reps = _localize_representatives(
        magnitude, candidates, n, noise_sigma, patch_half, sigma_model_floor_px
    )
    lattice_fit, lattice_meta = _lattice_stage(
        reps,
        lattice=options.get("lattice"),
        n=n,
        dq_nm_inv=dq_nm_inv,
        q_max_px=q_max_px,
        dc_radius_px=dc_radius_px,
        noise_sigma=noise_sigma,
        magnitude=magnitude,
        min_snr_seed=min_snr_seed,
        min_snr_verify=min_snr_verify,
        lattice_tolerance=lattice_tolerance,
        max_iterations=max_iterations,
        patch_half=patch_half,
        sigma_model_floor_px=sigma_model_floor_px,
        chi2_red_max=chi2_red_max,
        residual_max_px=residual_max_px,
        min_pool_spacing_px=min_pool_spacing_px,
    )
    meta = {
        "footprint_px": int(used_footprint),
        "dc_radius_px": float(dc_radius_px),
        "q_max_px": float(q_max_px),
        "nan_filled": int(options.get("_nan_filled", 0)),
    }
    meta.update(lattice_meta)
    if lattice_fit is None:
        # Invariant: without a converged lattice no peak may carry a label or a
        # model position (the fit writes them, no earlier stage does).
        _clear_model_fields(reps)
    peaks = _finalize_peaks(reps, n, dq_nm_inv)
    return BraggDetectionResult(
        peaks=tuple(peaks),
        lattice=lattice_fit,
        size_nm=size_nm,
        n_px=n,
        dq_nm_inv=dq_nm_inv,
        q_nyquist_nm_inv=q_nyquist_nm_inv,
        noise_sigma=float(noise_sigma),
        n_candidates=int(n_candidates),
        fft2=spectrum if return_fft2 else None,
        meta=meta,
    )


def _finite_scale(image: np.ndarray) -> float:
    """Largest finite magnitude of the real-space input (0.0 when there is none)."""
    if image.ndim != 2:
        return 0.0
    finite = np.isfinite(image)
    if not np.any(finite):
        return 0.0
    return float(np.max(np.abs(image[finite])))


def _degenerate_result(
    spectrum, size_nm, magnitude, noise_sigma, dc_radius_px, q_max_px,
    dq_nm_inv, q_nyquist_nm_inv, return_fft2, options,
) -> BraggDetectionResult:
    """Empty result for an input without a measurable |FFT| background scale."""
    n = spectrum.shape[0]
    reason = (
        "no measurable |FFT| background scale (MAD of the annular band is zero or "
        "at the float round-off floor of the spectrum); a constant or all-zero "
        "image has no Bragg peaks"
    )
    logger.warning("bragg_peak_detection: %s", reason)
    return BraggDetectionResult(
        peaks=(),
        lattice=None,
        size_nm=size_nm,
        n_px=n,
        dq_nm_inv=dq_nm_inv,
        q_nyquist_nm_inv=q_nyquist_nm_inv,
        noise_sigma=float(noise_sigma),
        n_candidates=0,
        fft2=spectrum if return_fft2 else None,
        meta={
            "footprint_px": 0,
            "dc_radius_px": float(dc_radius_px),
            "q_max_px": float(q_max_px),
            "nan_filled": int(options.get("_nan_filled", 0)),
            "inferred_symmetry": None,
            "n_added_by_loop": 0,
            "n_lattice_points": 0,
            "pool_from_lattice_operations": None,
            "reference_gauge": "user" if options.get("lattice") is not None else "deviatoric_det_M_1",
            "fit_ok": None,
            "lattice_quality": "no_lattice",
            "oblique_gauge": None,
            "oblique_gauge_margin": None,
            "pool_spacing_px": 0.0,
            "residual_max_px": 0.0,
            "min_pool_spacing_px": float(options.get("min_pool_spacing_px", 3.0)),
            "chi2_red_max": float(options.get("chi2_red_max", 25.0)),
            "consistent_fraction": 0.0,
            "degenerate_input": reason,
            "lattice_error": reason,
            "spectral_scale": float(np.max(np.abs(magnitude))),
        },
    )


def _localize_representatives(
    magnitude, candidates, n, noise_sigma, patch_half, sigma_model_floor_px
) -> list[dict]:
    """Localize the half-plane representatives, removing <1 px duplicates."""
    reps: list[dict] = []
    kept_chi2: list[float] = []
    for row, col in candidates:
        qy = row - n // 2
        qx = col - n // 2
        if not (qy * _HALF_PLANE_SIGN > 0 or (qy == 0 and qx * _HALF_PLANE_SIGN > 0)):
            continue
        peak = _localize_peak(
            magnitude, row, col, noise_sigma,
            patch_half=patch_half, bounds_half=2.0,
            sigma_model_floor_px=sigma_model_floor_px,
        )
        duplicate = None
        for index, existing in enumerate(reps):
            if (
                np.hypot(
                    peak["qx_px"] - existing["qx_px"],
                    peak["qy_px"] - existing["qy_px"],
                )
                < _MIN_DUPLICATE_DISTANCE_PX
            ):
                duplicate = index
                break
        if duplicate is None:
            reps.append(peak)
            kept_chi2.append(float(peak["chi2_reduced"]))
            continue
        new_chi2 = float(peak["chi2_reduced"])
        old_chi2 = kept_chi2[duplicate]
        better = np.isfinite(new_chi2) and (not np.isfinite(old_chi2) or new_chi2 < old_chi2)
        if better:
            reps[duplicate] = peak
            kept_chi2[duplicate] = new_chi2
    return reps


def _finalize_peaks(reps: list[dict], n: int, dq_nm_inv: float) -> list[BraggPeak]:
    """Build the public peak list, adding the -q partner of every half-plane peak."""
    records = [(peak, True) for peak in reps]
    for peak in reps:
        mirror = dict(peak)
        mirror["qx_px"] = -peak["qx_px"]
        mirror["qy_px"] = -peak["qy_px"]
        if peak["q_model_px"] is not None:
            mirror["q_model_px"] = (-peak["q_model_px"][0], -peak["q_model_px"][1])
        if peak["index_hk"] is not None:
            mirror["index_hk"] = (-peak["index_hk"][0], -peak["index_hk"][1])
        mirror["row"] = round(-peak["qy_px"] + n // 2) % n
        mirror["col"] = round(-peak["qx_px"] + n // 2) % n
        records.append((mirror, False))
    order = sorted(range(len(records)), key=lambda i: (-records[i][0]["snr"], i))
    position = {old: new for new, old in enumerate(order)}
    built = [_make_peak(records[old][0], dq_nm_inv) for old in order]
    peaks = []
    for new, old in enumerate(order):
        independent = records[old][1]
        partner_old = old + len(reps) if independent else old - len(reps)
        peaks.append(
            replace(
                built[new],
                conjugate_index=position[partner_old],
                independent=independent,
            )
        )
    return peaks


def _make_peak(record: dict, dq_nm_inv: float) -> BraggPeak:
    """Convert an internal peak record into the public frozen dataclass."""
    qx, qy = float(record["qx_px"]), float(record["qy_px"])
    sigma = np.asarray(record["sigma_px"], dtype=float)
    cov = np.asarray(record["cov_px"], dtype=float)
    model = record.get("q_model_px")
    model_sigma = record.get("sigma_q_model_px")
    method = record["method"]
    if record["edge"]:
        quality = "edge"
    elif method == "max_pixel" and record["base_quality"] == "failed":
        quality = "failed"
    elif record.get("index_hk") is None:
        quality = "unmatched"
    elif method == "parabola_log":
        quality = "coarse"
    else:
        quality = "ok"
    return BraggPeak(
        q_px=(qx, qy),
        q_nm_inv=(qx * dq_nm_inv, qy * dq_nm_inv),
        sigma_q_px=(float(sigma[0]), float(sigma[1])),
        sigma_q_nm_inv=(float(sigma[0]) * dq_nm_inv, float(sigma[1]) * dq_nm_inv),
        cov_q_px=cov,
        cov_q_nm_inv=cov * dq_nm_inv**2,
        amplitude=float(record["amplitude"]),
        snr=float(record["snr"]),
        index_hk=record.get("index_hk"),
        q_model_px=None if model is None else (float(model[0]), float(model[1])),
        sigma_q_model_px=(
            None
            if model_sigma is None
            else (float(model_sigma[0]), float(model_sigma[1]))
        ),
        residual_px=record.get("residual_px"),
        chi2_reduced=float(record.get("chi2_reduced", float("nan"))),
        n_pixels=int(record.get("n_pixels", 0)),
        method=method,
        quality=quality,
        conjugate_index=None,
        independent=True,
    )
