"""Data containers for the Bragg peak package.

Conventions shared by every module (do not mix them up):

* All 2-D vectors are ``(qx, qy)``; image indices are ``[row=y, col=x]``.
* ``_px`` positions are **signed fftshift offsets** from the array centre:
  ``q_px = index - n // 2`` and ``q_nm_inv = q_px * 2*pi/size_nm``.
* ``cov_q_px`` is the **total** position covariance (Gaussian least-squares
  covariance plus the model floor), so ``sigma_q_px = sqrt(diag(cov_q_px))`` holds
  by construction and the GLS lattice fit weights with that same matrix.
* ``snr`` is the *detection* signal-to-noise ratio against the local radial |FFT|
  background (see :mod:`.detect`), not ``|F| / noise_sigma``.
* ``fit_ok`` reports whether the lattice fit converged; a failed fit never removes
  a peak and leaves no ``index_hk`` behind.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["BraggDetectionResult", "BraggPeak", "LatticeFit", "LatticeSpec"]


@dataclass(frozen=True)
class LatticeSpec:
    """Ideal reference reciprocal lattice used to label detected peaks.

    ``orientation_deg`` is the counter-clockwise rotation of the reference basis
    (``basis @ rot.T``); ``None`` means "unknown orientation", so the basis is
    rotated onto the first detected ring and the reported rotation is defined only
    modulo ``LatticeFit.rotation_mod_deg``.
    """

    a_nm: float | None = None
    symmetry: str = "hexagonal"
    bvecs_nm_inv: np.ndarray | None = None
    orientation_deg: float | None = None
    h_max: int = 4


@dataclass(frozen=True)
class BraggPeak:
    """One detected Bragg peak.

    ``index_hk`` is the (h, k) label, or None when the peak is not consistent with
    the fitted lattice (then ``q_model_px``, ``sigma_q_model_px`` and
    ``residual_px`` are None too).  ``quality`` is ``"ok"``, ``"unmatched"``,
    ``"integer"`` or ``"edge"``, and ``independent`` marks the measured half-plane
    representative whose partner is the mirrored copy (|F(-q)| == |F(q)|).
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
    """Fitted reciprocal basis and its distortion.

    ``cov_bvecs_nm_inv`` is the (4, 4) covariance of the flattened fitted basis
    (row-major vec order b1x, b1y, b2x, b2y) and the source of every
    ``sigma_q_model_px``.  With a caller-supplied ``LatticeSpec`` the affine is the
    distortion relative to that reference; with ``lattice=None`` it is the
    residual distortion relative to the hexagon anchored on the first detected
    ring (``|det M| ~ 1``, not enforced).  ``fit_ok`` is True when the GLS fit
    converged: it is not a data-quality gate, and a fit that did not converge
    keeps every position and diagnostic.
    """

    bvecs_nm_inv: np.ndarray
    cov_bvecs_nm_inv: np.ndarray
    affine: np.ndarray
    cov_affine: np.ndarray
    rotation_deg: float | None
    rotation_sigma_deg: float | None
    rotation_mod_deg: float
    rotation_is_absolute: bool
    principal_stretches: tuple[float, float]
    n_independent: int
    chi2_reduced: float
    rms_residual_px: float
    symmetry: str
    reference_radius_px: float
    fit_ok: bool = True
    quality: str = "ok"


@dataclass(frozen=True)
class BraggDetectionResult:
    """Output of :func:`...bragg_peak.detect_bragg_peaks`.

    ``n_candidates`` counts the non-maximum-suppressed candidates above
    ``min_snr`` before the cap and the +/-q selection; ``meta`` carries the
    diagnostics (rings, selection reason, lattice residuals).
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
