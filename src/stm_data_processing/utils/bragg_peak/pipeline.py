"""Public entry points: the detection pipeline and the detector class (stage 4).

Flow: pre-process + FFT -> radial-background candidate detection -> sub-pixel
localization of the +/-q half-plane representatives -> hexagonal ring clustering
-> reference basis (caller-supplied or inferred) -> GLS affine fit -> labelling.

``result.peaks`` holds the members of every detected ring plus every peak the
fitted lattice labels, each with its mirrored ``-q`` partner
(``independent=False``); background maxima in no ring are not Bragg peaks.  The
candidate list is explicitly bounded (``max_candidates``, 2048 by default), so a
real topograph reports tens of peaks instead of thousands.  No quality gate
rejects real data: ``fit_ok`` reports convergence only, and a failed fit leaves
every position, uncertainty and diagnostic intact.
"""

from __future__ import annotations

import logging

import numpy as np

from .detect import detect_candidates, radial_background, robust_rayleigh_scale
from .fft import compute_fft2, magnitude, q_axis_limits, radius_map
from .lattice_fit import fit_lattice
from .localize import finalize_peaks, localize_peak, merge_duplicates
from .models import BraggDetectionResult, LatticeSpec
from .rings import reference_model, ring_clusters

__all__ = ["BraggPeakDetector", "detect_bragg_peaks"]

logger = logging.getLogger(__name__)

# Relative thresholds that recognise the float round-off floor of the transform
# of a constant or all-zero image (the background of such an input sits exactly
# there, so a "noise scale" at that level is not a measurement).
_DEGENERATE_SCALE_FRACTION = 1e-12
_DEGENERATE_FLOOR_FACTOR = 1e3
_KEYS = frozenset(
    (
        "lattice",
        "min_snr",
        "dc_mask_frac",
        "q_max_px",
        "max_candidates",
        "max_peaks",
        "footprint",
        "patch_half",
        "sigma_model_floor_px",
        "window",
        "subtract_plane",
        "nan_policy",
        "return_fft2",
    )
)


def _empty_result(spectrum, size_nm, options, return_fft2, reason, noise_sigma=0.0):
    """Result without peaks, for a degenerate input or an empty candidate list."""
    logger.warning("bragg_peak: %s", reason)
    n = int(spectrum.shape[0])
    return BraggDetectionResult(
        peaks=(),
        lattice=None,
        size_nm=size_nm,
        n_px=n,
        dq_nm_inv=2.0 * np.pi / size_nm,
        q_nyquist_nm_inv=np.pi * n / size_nm,
        noise_sigma=float(noise_sigma),
        n_candidates=0,
        fft2=spectrum if return_fft2 else None,
        meta={
            "degenerate_input": reason,
            "lattice_error": reason,
            "basis_source": None,
            "n_rings": 0,
            "n_candidates_before_cap": 0,
            "n_peaks_reported": 0,
            "max_candidates": options.get("max_candidates", 2048),
            "ring_radii_px": [],
            "ring_snr": [],
            "ring_members": [],
            "ladder_rings": [],
            "ring_ladder": [],
            "reference_radius_px": None,
        },
    )


def _run_pipeline(spectrum, size_nm, options, *, return_fft2):
    """Run the detection pipeline on an fftshifted complex spectrum."""
    n = int(spectrum.shape[0])
    mag = magnitude(spectrum)
    radius = radius_map(n)
    dq_nm_inv, q_nyquist = q_axis_limits(n, size_nm)
    dc_radius_px = float(options.get("dc_mask_frac", 0.03)) * n
    q_max_option = options.get("q_max_px")
    q_max_px = 0.95 * (n // 2) if q_max_option is None else float(q_max_option)
    min_snr = float(options.get("min_snr", 4.0))
    max_candidates = options.get("max_candidates", 2048)
    max_peaks = options.get("max_peaks")
    max_peaks = None if max_peaks is None else int(max_peaks)

    band = (radius >= 3.0 * dc_radius_px) & (radius <= min(0.97 * (n // 2), q_max_px))
    if int(np.count_nonzero(band)) < 16:
        band = (radius > 0.0) & (radius <= q_max_px)
    noise_sigma = robust_rayleigh_scale(mag, band)
    level, sigma = radial_background(mag, radius)
    spectral_scale = float(np.max(mag)) if mag.size else 0.0
    image_scale = float(options.get("_image_scale", 0.0))
    round_off = np.finfo(float).eps * max(image_scale, 0.0) * float(n)
    floor_scale = max(
        _DEGENERATE_SCALE_FRACTION * max(spectral_scale, 0.0),
        _DEGENERATE_FLOOR_FACTOR * max(round_off, np.finfo(float).tiny),
    )
    if (
        not np.isfinite(noise_sigma)
        or noise_sigma <= 0.0
        or not np.isfinite(spectral_scale)
        or spectral_scale <= 0.0
        or float(np.max(sigma)) <= 0.0
        or noise_sigma <= floor_scale
    ):
        return _empty_result(
            spectrum,
            size_nm,
            options,
            return_fft2,
            "no measurable |FFT| background scale (constant or all-zero image)",
            noise_sigma,
        )

    snr_map = (mag - level) / np.maximum(sigma, np.finfo(float).tiny)
    candidates, footprint, n_candidates = detect_candidates(
        mag,
        snr_map,
        radius,
        min_snr=min_snr,
        dc_radius_px=dc_radius_px,
        q_max_px=q_max_px,
        max_candidates=max_candidates,
        footprint=options.get("footprint"),
    )
    reps = merge_duplicates(
        [
            localize_peak(
                mag,
                candidate.row,
                candidate.col,
                snr=candidate.snr,
                noise_sigma=float(sigma[candidate.row, candidate.col]),
                patch_half=int(options.get("patch_half", 3)),
                sigma_model_floor_px=float(options.get("sigma_model_floor_px", 0.006)),
            )
            for candidate in candidates
        ]
    )
    meta = {
        "footprint_px": int(footprint),
        "n_candidates_before_cap": int(n_candidates),
        "n_half_plane": len(reps),
        "dc_radius_px": float(dc_radius_px),
        "q_max_px": float(q_max_px),
        "nan_filled": int(options.get("_nan_filled", 0)),
        "noise_sigma": float(noise_sigma),
        "max_candidates": max_candidates
        if max_candidates is None
        else int(max_candidates),
    }
    if not reps:
        empty = _empty_result(
            spectrum,
            size_nm,
            options,
            return_fft2,
            "no candidate above min_snr",
            noise_sigma,
        )
        empty.meta.update(meta)
        return empty

    q_obs = np.array([[p["qx_px"], p["qy_px"]] for p in reps], dtype=float)
    covs = [np.asarray(p["cov_px"], dtype=float) for p in reps]
    snr_obs = np.array([p["snr"] for p in reps], dtype=float)
    spec = options.get("lattice")
    rings = ring_clusters(q_obs, snr_obs)
    reference = reference_model(
        q_obs,
        covs,
        snr_obs,
        rings,
        spec=spec,
        h_max=int(spec.h_max) if spec is not None else 4,
        q_max=q_max_px,
        dq_nm_inv=dq_nm_inv,
    )
    meta.update(
        n_rings=len(rings),
        ring_radii_px=[round(ring["radius"], 2) for ring in rings],
        ring_snr=[round(ring["snr"], 2) for ring in rings],
        ring_members=[list(ring["triple"]) for ring in rings],
        ladder_rings=reference["ladder_rings"],
        basis_source=reference["basis_source"],
        ring_ladder=reference["ring_ladder"],
        reference_radius_px=reference["reference_radius_px"],
    )
    if reference["model"] is None:
        meta["lattice_error"] = "no hexagonal ring found in the candidate set"
        lattice, labels = None, [None] * len(reps)
    else:
        lattice, labels, fit_meta = fit_lattice(
            q_obs,
            covs,
            model=reference["model"],
            labels=reference["labels"],
            members=reference["members"],
            symmetry=reference["symmetry"],
            reference_radius_px=reference["reference_radius_px"],
            rotation_is_absolute=reference["rotation_is_absolute"],
            dq_nm_inv=dq_nm_inv,
        )
        meta.update(fit_meta)
        if lattice is None:
            # A failed fit leaves no label: index_hk implies a model position.
            labels = [None] * len(reps)
    peaks, fallback, n_reported = finalize_peaks(
        reps, labels, lattice, meta, dq_nm_inv, max_peaks
    )
    meta["fallback_report"] = fallback
    meta["n_peaks_reported"] = int(n_reported)
    if lattice is None:
        logger.warning("bragg_peak: no lattice fitted (%s)", meta.get("lattice_error"))
    return BraggDetectionResult(
        peaks=peaks,
        lattice=lattice,
        size_nm=size_nm,
        n_px=n,
        dq_nm_inv=dq_nm_inv,
        q_nyquist_nm_inv=q_nyquist,
        noise_sigma=float(noise_sigma),
        n_candidates=int(n_candidates),
        fft2=spectrum if return_fft2 else None,
        meta=meta,
    )


def _image_stats(image: np.ndarray) -> tuple[int, float]:
    """Number of non-finite pixels and the largest finite magnitude."""
    array = np.asarray(image)
    if array.ndim != 2:
        return 0, 0.0
    finite = np.isfinite(array)
    if not np.any(finite):
        return int(array.size), 0.0
    return int(array.size - np.count_nonzero(finite)), float(
        np.max(np.abs(array[finite]))
    )


def detect_bragg_peaks(
    image: np.ndarray,
    size_nm: float,
    *,
    lattice: LatticeSpec | None = None,
    min_snr: float = 4.0,
    dc_mask_frac: float = 0.03,
    q_max_px: float | None = None,
    max_candidates: int | None = 2048,
    max_peaks: int | None = None,
    footprint: int | None = None,
    patch_half: int = 3,
    sigma_model_floor_px: float = 0.006,
    window: str | np.ndarray | None = "hann",
    subtract_plane: bool = True,
    nan_policy: str = "plane",
    return_fft2: bool = False,
) -> BraggDetectionResult:
    """Detect, localize and lattice-fit the Bragg peaks of an STM topograph."""
    nan_filled, image_scale = _image_stats(image)
    fft2 = compute_fft2(
        image,
        size_nm,
        window=window,
        subtract_plane=subtract_plane,
        nan_policy=nan_policy,
    )
    options = {
        "lattice": lattice,
        "min_snr": min_snr,
        "dc_mask_frac": dc_mask_frac,
        "q_max_px": q_max_px,
        "max_candidates": max_candidates,
        "max_peaks": max_peaks,
        "footprint": footprint,
        "patch_half": patch_half,
        "sigma_model_floor_px": sigma_model_floor_px,
        "_nan_filled": nan_filled,
        "_image_scale": image_scale,
    }
    return _run_pipeline(fft2, float(size_nm), options, return_fft2=return_fft2)


class BraggPeakDetector:
    """Reusable detector with the keyword arguments of :func:`detect_bragg_peaks`."""

    def __init__(self, **kwargs) -> None:
        unknown = set(kwargs) - _KEYS
        if unknown:
            raise TypeError(f"unexpected keyword argument(s): {sorted(unknown)}")
        self.options = dict(kwargs)

    def detect(self, image: np.ndarray, size_nm: float) -> BraggDetectionResult:
        """Detect peaks in a real-space image (see :func:`detect_bragg_peaks`)."""
        options = dict(self.options)
        return detect_bragg_peaks(
            image,
            size_nm,
            return_fft2=bool(options.pop("return_fft2", False)),
            **options,
        )

    def detect_from_fft2(
        self, fft2: np.ndarray, size_nm: float, *, return_fft2: bool = False
    ) -> BraggDetectionResult:
        """Detect peaks in an already fftshifted complex spectrum (used as-is)."""
        spectrum = np.asarray(fft2)
        if spectrum.ndim != 2 or spectrum.shape[0] != spectrum.shape[1]:
            raise ValueError(
                f"fft2 must be a square 2-D array, got shape {spectrum.shape}"
            )
        if not np.isfinite(size_nm) or size_nm <= 0:
            raise ValueError(f"size_nm must be a positive number, got {size_nm!r}")
        options = dict(self.options)
        options.pop("return_fft2", None)
        return _run_pipeline(spectrum, float(size_nm), options, return_fft2=return_fft2)
