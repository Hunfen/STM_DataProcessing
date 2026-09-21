"""Reference basis, pool labelling and the GLS affine fit (stage 3a).

This module holds the lattice *primitives*: the ideal basis of a
:class:`~.models.LatticeSpec`, the two-point oblique seed, the integer pool
labelling and the weighted (GLS) fit of the 2x2 affine matrix ``M`` of
``q_observed = (h, k) @ b_ref @ M``.  The ring clustering and the inference of
``b_ref`` from the data live in :mod:`...bragg_peak.rings`.

Conventions (FFT pixels unless a name ends in ``_nm_inv``): ``cov_q_px`` is the
**total** position covariance (fit plus model floor), so the GLS weights with it
directly; ``cov_bvecs_nm_inv`` is the covariance of the flattened fitted basis and
``J = [[h, 0, k, 0], [0, h, 0, k]]`` propagates it to a model position.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import polar

from .models import LatticeFit, LatticeSpec

__all__ = [
    "fit_lattice",
    "gls_fit",
    "hexagon_basis",
    "ideal_basis",
    "match_labels",
    "rotate_basis",
    "spec_model",
    "two_point_basis",
]

# Hexagonal basis in units of |b1| (60 degrees between b1 and b2), so that
# |b1 + b2| = sqrt(3)|b1| is the second ring.
_HEXAGONAL_UNIT = np.array([[1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]])
# Labelling tolerance of the pool match (fraction of |q|, plus a pixel floor).
_LABEL_TOL_FRAC = 0.02
_LABEL_TOL_MIN_PX = 2.0
_POINT_GROUP_MOD_DEG = {"hexagonal": 60.0, "square": 90.0, "oblique": 180.0}


def point_group_mod_deg(symmetry: str) -> float:
    """Rotation modulus (degrees) imposed by the point group."""
    if symmetry not in _POINT_GROUP_MOD_DEG:
        raise ValueError(
            f"symmetry must be 'hexagonal', 'square' or 'oblique', got {symmetry!r}"
        )
    return _POINT_GROUP_MOD_DEG[symmetry]


def ideal_basis(
    a_nm: float | None, symmetry: str, bvecs_nm_inv: np.ndarray | None
) -> np.ndarray:
    """(2, 2) rows b1, b2 in nm^-1 for the requested ideal lattice."""
    if bvecs_nm_inv is not None:
        basis = np.asarray(bvecs_nm_inv, dtype=float)
        if basis.shape != (2, 2) or abs(float(np.linalg.det(basis))) < 1e-12:
            raise ValueError(f"bvecs_nm_inv must be a non-singular (2, 2), got {basis}")
        return basis
    if a_nm is None:
        raise ValueError("LatticeSpec requires either a_nm or bvecs_nm_inv")
    if a_nm <= 0:
        raise ValueError(f"a_nm must be positive, got {a_nm!r}")
    if symmetry == "hexagonal":
        return (4.0 * np.pi / (np.sqrt(3.0) * a_nm)) * _HEXAGONAL_UNIT
    if symmetry == "square":
        return np.eye(2) * (2.0 * np.pi / a_nm)
    if symmetry == "oblique":
        raise ValueError("symmetry='oblique' requires explicit bvecs_nm_inv")
    raise ValueError(
        f"symmetry must be 'hexagonal', 'square' or 'oblique', got {symmetry!r}"
    )


def rotate_basis(basis: np.ndarray, orientation_deg: float | None) -> np.ndarray:
    """Rotate the rows of ``basis`` counter-clockwise by ``orientation_deg``."""
    if orientation_deg is None:
        return basis
    theta = np.radians(float(orientation_deg))
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return basis @ rot.T


def hexagon_basis(radius_px: float, angle_deg: float) -> np.ndarray:
    """Hexagonal basis with ``|b1| = radius_px`` and b1 at ``angle_deg``."""
    return rotate_basis(radius_px * _HEXAGONAL_UNIT, angle_deg)


def two_point_basis(
    q_obs: np.ndarray, snr: np.ndarray, min_angle_deg: float = 15.0
) -> np.ndarray | None:
    """Simple oblique two-point seed: the two shortest strong non-collinear peaks."""
    radius = np.hypot(q_obs[:, 0], q_obs[:, 1])
    strong = np.asarray(snr, dtype=float) >= float(np.median(snr))
    order = [int(i) for i in np.argsort(radius, kind="stable") if strong[i]]
    if len(order) < 2:
        order = [int(i) for i in np.argsort(radius, kind="stable")]
    for position, first in enumerate(order[:4]):
        for second in order[position + 1 :][:4]:
            cross = float(
                q_obs[first, 0] * q_obs[second, 1] - q_obs[first, 1] * q_obs[second, 0]
            )
            norm = float(np.hypot(*q_obs[first]) * np.hypot(*q_obs[second]))
            if norm <= 0.0:
                continue
            if np.degrees(np.arcsin(min(abs(cross) / norm, 1.0))) >= min_angle_deg:
                return np.vstack([q_obs[first], q_obs[second]])
    return None


def match_labels(
    q_obs: np.ndarray,
    basis: np.ndarray,
    h_max: int,
    q_max: float,
    *,
    tol_frac: float = _LABEL_TOL_FRAC,
    tol_min: float = _LABEL_TOL_MIN_PX,
) -> list[tuple[int, int] | None]:
    """Nearest-pool-index labels, or None when the nearest point is too far."""
    hk = [
        (h, k)
        for h in range(-h_max, h_max + 1)
        for k in range(-h_max, h_max + 1)
        if (h, k) != (0, 0)
    ]
    points = np.array([np.asarray(row, dtype=float) @ basis for row in hk])
    hk = np.array(hk, dtype=int)[np.hypot(points[:, 0], points[:, 1]) <= q_max]
    points = points[np.hypot(points[:, 0], points[:, 1]) <= q_max]
    if points.size == 0:
        return [None] * q_obs.shape[0]
    distance = np.hypot(
        q_obs[:, 0, None] - points[None, :, 0], q_obs[:, 1, None] - points[None, :, 1]
    )
    nearest = np.argmin(distance, axis=1)
    best = distance[np.arange(q_obs.shape[0]), nearest]
    tolerance = np.maximum(tol_min, tol_frac * np.hypot(q_obs[:, 0], q_obs[:, 1]))
    return [
        (int(hk[nearest[i], 0]), int(hk[nearest[i], 1]))
        if best[i] <= tolerance[i]
        else None
        for i in range(q_obs.shape[0])
    ]


def _whitener(cov: np.ndarray) -> np.ndarray:
    """Symmetric inverse square root of a 2x2 covariance matrix."""
    values, vectors = np.linalg.eigh(np.asarray(cov, dtype=float))
    return vectors @ np.diag(1.0 / np.sqrt(np.clip(values, 1e-12, None))) @ vectors.T


def gls_fit(
    q_obs: np.ndarray,
    covs: list[np.ndarray],
    labels: list[tuple[int, int]],
    basis: np.ndarray,
):
    """Weighted least-squares fit of the 2x2 affine matrix ``M``."""
    if len(labels) < 3:
        return None
    rows, rhs = [], []
    for index, hk in enumerate(labels):
        q_ideal = np.asarray(hk, dtype=float) @ basis
        weight = _whitener(covs[index])
        design = [
            [q_ideal[0], 0.0, q_ideal[1], 0.0],
            [0.0, q_ideal[0], 0.0, q_ideal[1]],
        ]
        rows.append(weight @ np.asarray(design))
        rhs.append(weight @ np.asarray(q_obs[index], dtype=float))
    design = np.vstack(rows)
    target = np.concatenate(rhs)
    if np.linalg.matrix_rank(design, tol=1e-9) < 4:
        return None
    solution, *_ = np.linalg.lstsq(design, target, rcond=None)
    residual = target - design @ solution
    chi2_reduced = float(residual @ residual) / max(target.size - 4, 1)
    return (
        solution.reshape(2, 2),
        np.linalg.pinv(design.T @ design) * max(1.0, chi2_reduced),
        chi2_reduced,
    )


def basis_covariance(basis: np.ndarray, cov_m: np.ndarray) -> np.ndarray:
    """(4, 4) covariance of ``basis @ M``: ``vec(basis @ M) = (basis (x) I2) vec(M)``."""
    transform = np.kron(np.asarray(basis, dtype=float), np.eye(2))
    return transform @ np.asarray(cov_m, dtype=float) @ transform.T


def rotation_angle_deg(matrix: np.ndarray) -> float:
    """Polar-decomposition rotation angle of ``matrix`` in degrees."""
    rot, _ = polar(matrix)
    return float(np.degrees(np.arctan2(rot[0, 1], rot[0, 0])))


def rotation_sigma_deg(matrix: np.ndarray, cov_m: np.ndarray) -> float:
    """1-sigma of :func:`rotation_angle_deg` from a numerical Jacobian."""
    flat = np.asarray(matrix, dtype=float).ravel()
    jac = np.zeros(4)
    for index in range(4):
        step = 1e-6 * max(1.0, abs(flat[index]))
        plus, minus = flat.copy(), flat.copy()
        plus[index] += step
        minus[index] -= step
        jac[index] = (
            rotation_angle_deg(plus.reshape(2, 2))
            - rotation_angle_deg(minus.reshape(2, 2))
        ) / (2.0 * step)
    variance = float(jac @ np.asarray(cov_m, dtype=float) @ jac)
    if not np.isfinite(variance) or variance < 0.0:
        return float("nan")
    return float(np.sqrt(variance))


def fit_lattice(
    q_obs: np.ndarray,
    covs: list[np.ndarray],
    *,
    model: np.ndarray,
    labels: list[tuple[int, int] | None],
    members: list[int],
    symmetry: str,
    reference_radius_px: float,
    rotation_is_absolute: bool,
    dq_nm_inv: float,
) -> tuple[LatticeFit | None, list[tuple[int, int] | None], dict]:
    """GLS-fit the affine over ``members`` and assemble the public result."""
    meta: dict = {"n_labelled": 0, "lattice_error": None}
    used = sorted({int(i) for i in members if labels[int(i)] is not None})
    labels = [labels[i] if i in set(used) else None for i in range(q_obs.shape[0])]
    meta["n_labelled"] = len(used)
    if len(used) < 3:
        meta["lattice_error"] = (
            f"only {len(used)} labelled ring member(s); at least 3 are required"
        )
        return None, labels, meta
    fit = gls_fit(
        q_obs[used], [covs[i] for i in used], [labels[i] for i in used], model
    )
    if fit is None:
        meta["lattice_error"] = "the weighted lattice fit is rank deficient"
        return None, labels, meta
    affine, cov_m, chi2_reduced = fit
    b_fit = model @ affine
    if abs(float(np.linalg.det(b_fit))) <= 1e-12 or not np.isfinite(chi2_reduced):
        meta["lattice_error"] = "the fitted reciprocal basis is singular"
        return None, labels, meta
    predicted = np.array([np.asarray(labels[i], dtype=float) @ b_fit for i in used])
    residual = np.hypot(
        q_obs[used, 0] - predicted[:, 0], q_obs[used, 1] - predicted[:, 1]
    )
    rms_residual_px = float(np.sqrt(np.mean(residual**2)))
    meta.update(chi2_reduced=float(chi2_reduced), rms_residual_px=rms_residual_px)
    lattice = LatticeFit(
        bvecs_nm_inv=b_fit * dq_nm_inv,
        cov_bvecs_nm_inv=basis_covariance(model, cov_m) * dq_nm_inv**2,
        affine=affine,
        cov_affine=cov_m,
        rotation_deg=rotation_angle_deg(affine),
        rotation_sigma_deg=rotation_sigma_deg(affine, cov_m),
        rotation_mod_deg=point_group_mod_deg(symmetry),
        rotation_is_absolute=bool(rotation_is_absolute),
        principal_stretches=tuple(
            float(value) for value in sorted(np.linalg.svd(affine, compute_uv=False))
        ),
        n_independent=len(used),
        chi2_reduced=float(chi2_reduced),
        rms_residual_px=rms_residual_px,
        symmetry=symmetry,
        reference_radius_px=float(reference_radius_px),
        fit_ok=True,
        quality="ok",
    )
    return lattice, labels, meta


def spec_model(
    spec: LatticeSpec, q_obs: np.ndarray, snr: np.ndarray, pixel_scale: float
):
    """Reference model of a caller-supplied :class:`LatticeSpec`, in FFT pixels.

    ``ideal_basis`` is in nm^-1, so it is multiplied by ``pixel_scale``
    (``1/dq_nm_inv``) to match ``q_obs``; the oblique symmetry without explicit
    vectors falls back to the two-point seed, which is already in pixels.
    Returns ``(model, basis_source)``; a None model means the seed failed.
    """
    if spec.bvecs_nm_inv is not None or spec.symmetry != "oblique":
        basis = rotate_basis(
            ideal_basis(spec.a_nm, spec.symmetry, spec.bvecs_nm_inv),
            spec.orientation_deg,
        )
        return basis * float(pixel_scale), "spec"
    seeded = two_point_basis(q_obs, snr)
    return (None, "no_two_point_seed") if seeded is None else (seeded, "two_point_seed")
