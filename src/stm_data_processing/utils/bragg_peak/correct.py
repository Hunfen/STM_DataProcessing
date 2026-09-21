"""Lattice-distortion correction of a real-space topograph (stage 5).

The detector measures the *observed* reciprocal peaks ``q_obs``; a physical
reference lattice provides the *ideal* positions ``q_ideal = (h, k) @ B_ideal``.
A pure stretch of the sample (no rotation) maps one onto the other,

    q_obs = q_ideal @ M ,   M = [[a, b], [b, c]]  symmetric positive definite,

and this module solves ``M`` from the labelled peaks and resamples the image so
that its FFT peaks move from ``q_obs`` back onto ``q_ideal``.

Why the physical truth and not the fitted affine
------------------------------------------------
With ``lattice=None`` the detector anchors its reference basis on the detected
ring itself, so ``LatticeFit.affine`` is near identity by construction (measured
deviation 3e-5 on the 100 nm scan) while the data really sits +1.16 % (topo0009,
|b1| = 29.837 nm^-1) and -4.43 % (topo4_30nm, |b1| = 28.189 nm^-1) away from the
ideal graphene lattice ``4*pi/(sqrt(3)*0.246) = 29.4946 nm^-1``.  Correcting with
the fitted affine would therefore leave the image about 4.4 % wrong, so the
correction is anchored on the physical reference lattice (``LatticeSpec``,
hexagonal a = 0.246 nm by default) instead.

Geometry of the resampling (pinned by regression R4.2)
------------------------------------------------------
Write the observed image as ``I_obs(r) = sum_q A(q) exp(i (q_ideal @ M) . r)`` and
define ``I_corr(r) = I_obs(r')`` with ``M r' = r`` (M acting on a column vector),
using ``(q_ideal @ M) . r' = q_ideal . (M r')``.  Then
``I_corr(r) = sum_q A(q) exp(i q_ideal . r)``: the corrected image carries every
reflection at its ideal position and the sample point to read is ``r' = M^-1 r``.

``scipy.ndimage.affine_transform`` evaluates ``output[y] = input[A y + b]`` with
array indices ``y = (row, col) = (y_phys, x_phys)`` in pixels, so ``r' = M^-1 r``
becomes ``A = P M^-1 P`` with the axis swap ``P = [[0, 1], [1, 0]]``.  The canvas
grows until it contains every transformed corner plus ``pad`` pixels, the centres
are aligned (``b = c_in - A c_out``) and the corrected field of view is
``size_nm * n_out / n`` since the nm-per-pixel scale is kept.  Passing the
transpose of ``A`` (or ``M`` instead of ``M^-1``) is a different map: R4.2
measures the recovered lattice constant then missing by percent.

Known limitation: usable range (not silent)
-------------------------------------------
The correction needs the lattice to be LABELLED first, and labelling is bounded by
the detector's own tolerances: ring clustering keeps a 60-degree triple only while
its members' radii agree within 3 % (``rings._RING_RADIUS_TOL``), and
``match_labels`` accepts a peak only within 2 % of |q| of its ideal position plus a
2 px floor (``lattice_fit._LABEL_TOL_FRAC`` / ``_LABEL_TOL_MIN_PX``).  The
correction therefore works while the anisotropy stays roughly <= 3 %.

Beyond that no lattice is fitted (``lattice_error = "no hexagonal ring found in the
candidate set"``, ``basis_source = "no_rings"``), no peak carries a label, and the
module falls back to the identity: ``method = "identity_fallback"``,
``fallback = True``, ``n_labelled = 0``, ``affine_q = affine_image = I``, logged as
"no positive-definite stretch; correction skipped".  Nothing is resampled then - a
COPY of the input array comes back verbatim with its own geometry (``n_out = n``,
``offset = 0``, ``size_nm`` unchanged, no NaN added), which regression R4.4 pins on
the 5 % / 3 % synthetic (n = 512, ``max |out - in| = 0``).

Strongly anisotropic scans and distorted rectangular or square lattices need the
wider-tolerance correspondence path (roadmap item L1, section 9 of
``docs/design/bragg_peak_detection.md``).  Never make the condition silent: it is
reported in ``meta`` and in the warning log.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.ndimage import affine_transform

from .fft import compute_fft2, q_axis_limits
from .lattice_fit import ideal_basis, rotate_basis
from .models import CorrectionResult, LatticeSpec
from .pipeline import detect_bragg_peaks
from .preprocess import validate_image

__all__ = ["correct_bragg_peaks"]

logger = logging.getLogger(__name__)

# Physical reference lattice used when the caller does not state one: graphene.
_DEFAULT_LATTICE = LatticeSpec(a_nm=0.246, symmetry="hexagonal")
# Position-uncertainty floor of the weighted fit, in FFT pixels.
_SIGMA_FLOOR_PX = 1e-3
# Maps physical (x, y) order to array (row, col) order and back.
_AXIS_SWAP = np.array([[0.0, 1.0], [1.0, 0.0]])


def _labeled_peaks(result) -> list:
    """Independent peaks that carry an (h, k) label, strongest first."""
    peaks = [
        peak for peak in result.peaks if peak.index_hk is not None and peak.independent
    ]
    return sorted(peaks, key=lambda peak: -float(peak.snr))


def _is_positive_definite(matrix) -> bool:
    """True when the 2x2 matrix is finite with strictly positive eigenvalues."""
    if matrix is None or matrix.shape != (2, 2) or not np.all(np.isfinite(matrix)):
        return False
    return bool(
        float(matrix[0, 0]) > 0.0
        and float(matrix[1, 1]) > 0.0
        and float(np.linalg.det(matrix)) > 0.0
    )


def _reference_orientation(result):
    """Direction of the fitted (1, 0) family of a data-anchored detection.

    With ``lattice=None`` the label gauge (which family is called ``(1, 0)``) is
    fixed by the data and carried by its fitted basis.  A symmetric stretch has no
    rotation, so this direction must be handed to the ideal basis; otherwise an
    image whose lattice is rotated relative to the unrotated ideal basis would be
    "corrected" by a spurious shear.
    """
    lattice = getattr(result, "lattice", None)
    if lattice is None:
        return None
    b1 = np.asarray(lattice.bvecs_nm_inv[0], dtype=float)
    if not np.all(np.isfinite(b1)) or float(np.hypot(*b1)) <= 0.0:
        return None
    return float(np.degrees(np.arctan2(b1[1], b1[0])))


def _weighted_stretch(q_obs, q_ideal, sigma_obs):
    """Weighted least squares for ``M = [[a, b], [b, c]]`` over all peaks.

    One peak contributes two equations, ``q_obs_x = a q_id_x + b q_id_y``
    (row ``[q_id_x, q_id_y, 0]``) and ``q_obs_y = b q_id_x + c q_id_y``
    (row ``[0, q_id_x, q_id_y]``), each weighted by the inverse of the matching
    per-axis position sigma, floored to stay finite.  Returns None when the
    design is rank deficient.
    """
    rows, targets, weights = [], [], []
    for observed, ideal, sigma in zip(q_obs, q_ideal, sigma_obs, strict=True):
        rows.append([float(ideal[0]), float(ideal[1]), 0.0])
        targets.append(float(observed[0]))
        weights.append(1.0 / max(float(sigma[0]), _SIGMA_FLOOR_PX))
        rows.append([0.0, float(ideal[0]), float(ideal[1])])
        targets.append(float(observed[1]))
        weights.append(1.0 / max(float(sigma[1]), _SIGMA_FLOOR_PX))
    design = np.asarray(rows, dtype=float)
    weight = np.asarray(weights, dtype=float)
    weighted = design * weight[:, None]
    target = np.asarray(targets, dtype=float) * weight
    if len(rows) < 3 or np.linalg.matrix_rank(weighted, tol=1e-9) < 3:
        return None
    solution, *_ = np.linalg.lstsq(weighted, target, rcond=None)
    return np.array([[solution[0], solution[1]], [solution[1], solution[2]]])


def _two_vector_stretch(q_obs, q_ideal):
    """Diagonal closed form ``diag(a, c)`` from the two best-aligned peaks.

    ``a`` is read off the labelled peak with the largest ``|q_ideal_x|`` and ``c``
    off the one with the largest ``|q_ideal_y|``, i.e. from the reflections that
    measure each stretch best.
    """
    if len(q_obs) < 2:
        return None
    best_x = max(range(len(q_obs)), key=lambda i: abs(float(q_ideal[i][0])))
    best_y = max(range(len(q_obs)), key=lambda i: abs(float(q_ideal[i][1])))
    if abs(float(q_ideal[best_x][0])) < _SIGMA_FLOOR_PX:
        return None
    if abs(float(q_ideal[best_y][1])) < _SIGMA_FLOOR_PX:
        return None
    return np.array(
        [
            [float(q_obs[best_x][0]) / float(q_ideal[best_x][0]), 0.0],
            [0.0, float(q_obs[best_y][1]) / float(q_ideal[best_y][1])],
        ]
    )


def _image_transform(stretch, n: int, pad: int):
    """Array matrix, canvas side and offset for the correction resampling.

    ``A = P M^-1 P`` implements ``output[y] = input[A y + offset]`` in array
    ``(row, col)`` order; the canvas contains every transformed corner plus
    ``pad`` pixels and the offset aligns the two centres.
    """
    matrix = _AXIS_SWAP @ np.linalg.inv(stretch) @ _AXIS_SWAP
    corners = (
        np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]]) * (n - 1) / 2.0
    )
    extent = np.abs(np.linalg.inv(matrix) @ corners.T).max(axis=1)
    n_out = 2 * (int(np.ceil(float(extent.max()))) + int(pad)) + 1
    offset = (n - 1) / 2.0 - matrix @ np.array([(n_out - 1) / 2.0, (n_out - 1) / 2.0])
    return matrix, n_out, offset


def correct_bragg_peaks(
    image: np.ndarray,
    size_nm: float,
    *,
    lattice: LatticeSpec | None = None,
    result=None,
    order: int = 3,
    pad: int = 10,
    return_fft2: bool = True,
) -> CorrectionResult:
    """Correct the lattice distortion of a real-space image.

    Parameters
    ----------
    image, size_nm : np.ndarray, float
        Square real-space image and the physical side length of its field of view.
    lattice : LatticeSpec or None
        Physical reference lattice used both to detect the peaks and as the ideal
        basis of the fit.  ``None`` detects with the data-anchored inference and
        uses graphene (a = 0.246 nm, hexagonal) as the ideal basis.
    result : BraggDetectionResult or None
        Reuse an existing detection instead of running one.
    order : int
        Spline order of the resampling (0-5).
    pad : int
        Extra margin in pixels around the transformed corners.
    return_fft2 : bool
        Also return the complex spectrum of the corrected image (its NaN padding
        is filled with the best-fit plane for that transform only).

    Returns
    -------
    CorrectionResult
        Corrected NaN-padded image, corrected field of view, fitted q-space
        stretch, array matrix and offset actually used, measured and ideal ring
        radii, and diagnostics in ``meta``.

    Notes
    -----
    The fit needs peaks that carry ``index_hk``, and a label is granted only while
    the ring radii agree within 3 % (``rings._RING_RADIUS_TOL``) and the peak sits
    within 2 % of |q| of its ideal position (``lattice_fit._LABEL_TOL_FRAC``).
    Anisotropy beyond roughly 3 % - and distorted rectangular or square lattices,
    which have no 60-degree first-ring triple - therefore fit no lattice at all:
    ``meta["method"]`` is ``"identity_fallback"``, ``meta["fallback"]`` is True,
    ``meta["n_labelled"]`` is 0 and ``affine_q = affine_image = I``.  Nothing is
    resampled then: a copy of the input comes back verbatim with the input geometry
    (``n_out == n_px``, zero ``offset``, same ``size_nm``, no pixel added or
    invalidated, so ``fft2`` is the spectrum of the input).  Reported, not silent:
    see ``meta`` and the warning log, and inspect ``meta["method"]`` before trusting
    a corrected image.  The wider-tolerance correspondence path is roadmap item L1
    in section 9 of ``docs/design/bragg_peak_detection.md`` (out of scope here).
    """
    arr = validate_image(image, size_nm)
    n = int(arr.shape[0])
    dq_nm_inv, _ = q_axis_limits(n, size_nm)
    detection = (
        result
        if result is not None
        else detect_bragg_peaks(arr, size_nm, lattice=lattice)
    )
    ideal_spec = lattice if lattice is not None else _DEFAULT_LATTICE
    orientation_deg = ideal_spec.orientation_deg
    if lattice is None:
        # The detection inferred the reference from the data, so its label gauge
        # is the data's; a caller-supplied spec already states its own frame.
        orientation_deg = _reference_orientation(detection)
    basis_px = (
        rotate_basis(
            ideal_basis(ideal_spec.a_nm, ideal_spec.symmetry, ideal_spec.bvecs_nm_inv),
            orientation_deg,
        )
        / dq_nm_inv
    )

    peaks = _labeled_peaks(detection)
    q_obs = np.array([peak.q_px for peak in peaks], dtype=float)
    q_ideal = np.array(
        [np.asarray(peak.index_hk, dtype=float) @ basis_px for peak in peaks],
        dtype=float,
    )
    sigma_obs = np.array([peak.sigma_q_px for peak in peaks], dtype=float)

    affine_q = _weighted_stretch(q_obs, q_ideal, sigma_obs) if len(peaks) else None
    method, fallback = "weighted_lsq", False
    if not _is_positive_definite(affine_q):
        affine_q = _two_vector_stretch(q_obs, q_ideal)
        method, fallback = "two_vector_fallback", True
    if not _is_positive_definite(affine_q):
        affine_q = np.eye(2)
        method, fallback = "identity_fallback", True
        logger.warning("bragg_peak: no positive-definite stretch; correction skipped")

    if method == "identity_fallback":
        # Nothing could be solved: return the input verbatim with its own geometry.
        # Resampling with the identity map would still re-pad the canvas and, for an
        # even n, land on a half-pixel offset (regression R4.4 pins the no-op).
        corrected, matrix, n_out = np.array(arr, copy=True), np.eye(2), n
        offset, size_out = np.zeros(2, dtype=float), float(size_nm)
    else:
        matrix, n_out, offset = _image_transform(affine_q, n, pad)
        corrected = affine_transform(
            arr,
            matrix,
            offset=offset,
            output_shape=(n_out, n_out),
            order=int(order),
            mode="constant",
            cval=np.nan,
            prefilter=int(order) > 1,
        )
        size_out = float(size_nm) * n_out / n
    valid_fraction = float(np.count_nonzero(np.isfinite(corrected))) / float(
        corrected.size
    )

    target_radius_px = float(np.hypot(*basis_px[0]))
    radii = np.array([float(np.hypot(*peak.q_px)) for peak in peaks])
    first_ring = (
        radii[radii <= 1.02 * float(radii.min())] if radii.size else np.zeros(0)
    )
    measured_radius_px = float(np.mean(first_ring)) if first_ring.size else float("nan")
    residual_ratio = (
        measured_radius_px / target_radius_px if first_ring.size else float("nan")
    )
    residuals = np.hypot(*(q_obs - q_ideal @ affine_q).T) if len(peaks) else np.zeros(0)
    meta = {
        "method": method,
        "fallback": bool(fallback),
        "n_labelled": len(peaks),
        "order": int(order),
        "pad": int(pad),
        "nan_fraction": float(1.0 - valid_fraction),
        "rms_residual_px": (
            float(np.sqrt(np.mean(residuals**2))) if residuals.size else float("nan")
        ),
        "eigenvalues": tuple(float(v) for v in np.linalg.eigvalsh(affine_q)),
        "dq_nm_inv": float(dq_nm_inv),
        "input_size_nm": float(size_nm),
        "input_n_px": n,
        "target_radius_nm_inv": target_radius_px * dq_nm_inv,
        "measured_radius_nm_inv": (
            measured_radius_px * dq_nm_inv if first_ring.size else float("nan")
        ),
    }
    logger.info(
        "bragg_peak: correction %s over %d labelled peak(s), first ring %.4f nm^-1 "
        "(target %.4f)",
        method,
        len(peaks),
        meta["measured_radius_nm_inv"],
        meta["target_radius_nm_inv"],
    )
    return CorrectionResult(
        image=corrected,
        size_nm=size_out,
        n_out=n_out,
        n_px=n,
        affine_q=affine_q,
        affine_image=matrix,
        offset=offset,
        target_radius_px=target_radius_px,
        measured_radius_px=measured_radius_px,
        residual_ratio=residual_ratio,
        valid_fraction=valid_fraction,
        fft2=compute_fft2(corrected, size_out, nan_policy="plane")
        if return_fft2
        else None,
        meta=meta,
    )
