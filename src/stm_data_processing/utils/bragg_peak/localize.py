"""Sub-pixel localization of a candidate on |FFT| (stage 2) and the
public peak list.

Chain: snap to the local maximum of a 5x5 neighbourhood, seed the position with a
3x3 log-parabola fit, then fit a bounded rotated 2-D Gaussian on a plane
background by least squares.  When that fit is unusable the peak degrades to the
integer maximum with the ``1/sqrt(12)`` px quantization sigma; a patch that does
not fit the array is flagged ``edge``.  ``method`` and ``quality`` always record
which tier was used, and no tier drops a peak.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
from scipy.optimize import curve_fit

from .models import BraggPeak

__all__ = ["finalize_peaks", "localize_peak", "merge_duplicates"]

logger = logging.getLogger(__name__)

# Variance of a uniform position on a one-pixel grid.
_QUANTIZED_SIGMA_PX = 1.0 / np.sqrt(12.0)
_FALLBACK_REPORT_PEAKS = 32  # report size when neither a ring nor a lattice is found


def _gaussian_model(xy, amp, x0, y0, sx, sy, rho, b0, bx, by):
    """Rotated 2-D Gaussian on a plane background (9 parameters)."""
    xg, yg = xy
    dx = xg - x0
    dy = yg - y0
    one_minus_rho2 = 1.0 - rho * rho
    quad = (dx / sx) ** 2 - 2.0 * rho * (dx / sx) * (dy / sy) + (dy / sy) ** 2
    return amp * np.exp(-0.5 * quad / one_minus_rho2) + b0 + bx * dx + by * dy


def _parabola_axis(zm: float, z0: float, zp: float, sigma_log: float):
    """3-point log-parabola peak offset and its 1-sigma along one axis."""
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


def _second_moment_sigma(patch: np.ndarray) -> tuple[float, float] | None:
    """Second-moment width ``(sx, sy)`` of a patch, or None if it is flat."""
    weights = np.maximum(patch - float(patch.min()), 0.0)
    total = float(weights.sum())
    if total <= 0.0:
        return None
    xs = np.arange(patch.shape[1], dtype=float)[None, :]
    ys = np.arange(patch.shape[0], dtype=float)[:, None]
    mx = float((weights * xs).sum() / total)
    my = float((weights * ys).sum() / total)
    sx = float(np.sqrt(max((weights * (xs - mx) ** 2).sum() / total, 1e-12)))
    sy = float(np.sqrt(max((weights * (ys - my) ** 2).sum() / total, 1e-12)))
    return sx, sy


def _integer_record(magnitude, row, col, snr, quality):
    """Integer-maximum record with the quantization sigma."""
    n = magnitude.shape[0]
    return {
        "row": row, "col": col,
        "qx_px": float(col - n // 2), "qy_px": float(row - n // 2),
        "sigma_px": np.array([_QUANTIZED_SIGMA_PX, _QUANTIZED_SIGMA_PX]),
        "cov_px": np.eye(2) / 12.0,
        "amplitude": float(magnitude[row, col]), "snr": float(snr),
        "chi2_reduced": float("nan"), "n_pixels": 1,
        "method": "max_pixel", "quality": quality,
    }


def localize_peak(
    magnitude: np.ndarray,
    row: int,
    col: int,
    *,
    snr: float,
    noise_sigma: float,
    patch_half: int = 3,
    bounds_half: float = 2.0,
    sigma_model_floor_px: float = 0.006,
) -> dict:
    """Sub-pixel localization of one candidate on the |FFT| magnitude."""
    n = magnitude.shape[0]
    r0, c0 = max(0, row - 2), max(0, col - 2)
    window = magnitude[r0 : min(n, row + 3), c0 : min(n, col + 3)]
    iy, ix = np.unravel_index(np.argmax(window), window.shape)
    row, col = r0 + int(iy), c0 + int(ix)

    half = int(patch_half)
    if row - half < 0 or row + half + 1 > n or col - half < 0 or col + half + 1 > n:
        return _integer_record(magnitude, row, col, snr, "edge")

    patch = magnitude[row - half : row + half + 1, col - half : col + half + 1]
    iy, ix = np.unravel_index(np.argmax(patch), patch.shape)
    amplitude_guess = float(patch.max() - patch.min())
    sigma_log = noise_sigma / max(amplitude_guess, np.finfo(float).tiny)
    if 1 <= iy <= patch.shape[0] - 2 and 1 <= ix <= patch.shape[1] - 2:
        dx, _ = _parabola_axis(
            patch[iy, ix - 1], patch[iy, ix], patch[iy, ix + 1], sigma_log
        )
        dy, _ = _parabola_axis(
            patch[iy - 1, ix], patch[iy, ix], patch[iy + 1, ix], sigma_log
        )
    else:
        dx = dy = 0.0
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
                sigma=np.full(patch.size, max(noise_sigma, np.finfo(float).tiny)),
                absolute_sigma=True,
                maxfev=4000,
            )
    except Exception:
        logger.debug("bragg_peak: Gaussian fit failed at (%d, %d)", row, col)
        return _integer_record(magnitude, row, col, snr, "integer")

    variance = np.diag(pcov).astype(float)
    sigma_q = np.sqrt(np.maximum(variance[1:3], 0.0))
    centre_ok = (
        0.0 <= popt[1] <= patch.shape[1] - 1.0
        and 0.0 <= popt[2] <= patch.shape[0] - 1.0
    )
    if (
        not np.all(np.isfinite(variance))
        or np.any(variance[1:3] < 0.0)
        or float(np.max(sigma_q)) >= 1.0
        or not centre_ok
    ):
        return _integer_record(magnitude, row, col, snr, "integer")

    residual = patch.ravel() - _gaussian_model(xy, *popt)
    chi2_reduced = float(
        np.sum((residual / max(noise_sigma, np.finfo(float).tiny)) ** 2)
        / max(patch.size - 9, 1)
    )
    cov = np.array([[variance[1], pcov[1, 2]], [pcov[2, 1], variance[2]]], dtype=float)
    cov = cov + np.eye(2) * sigma_model_floor_px**2
    return {
        "row": row,
        "col": col,
        "qx_px": float(col - half + popt[1] - n // 2),
        "qy_px": float(row - half + popt[2] - n // 2),
        "sigma_px": np.sqrt(np.maximum(np.diag(cov), 0.0)),
        "cov_px": cov,
        "amplitude": float(popt[0]),
        "snr": float(snr),
        "chi2_reduced": chi2_reduced,
        "n_pixels": int(patch.size),
        "method": "gaussian",
        "quality": "ok",
    }


def _model_sigma_px(lattice, index_hk, dq_nm_inv) -> tuple[float, float]:
    """Per-axis 1-sigma of ``(h, k) @ B_fit``, in pixels."""
    h, k = index_hk
    jac = np.array([[h, 0.0, k, 0.0], [0.0, h, 0.0, k]])
    covariance = jac @ lattice.cov_bvecs_nm_inv @ jac.T
    return tuple(float(v) for v in np.sqrt(np.clip(np.diag(covariance), 0, None)) / dq_nm_inv)


def merge_duplicates(reps, min_distance_px: float = 1.0):
    """Drop repeated localizations of one peak, keeping the strongest."""
    kept: list[dict] = []
    for record in sorted(reps, key=lambda item: -item["snr"]):
        here = np.array([record["qx_px"], record["qy_px"]])
        if kept:
            positions = np.array([[k["qx_px"], k["qy_px"]] for k in kept])
            if float(np.min(np.hypot(*(positions - here).T))) < min_distance_px:
                continue
        kept.append(record)
    return kept


def _make_peak(record, label, model_px, model_sigma_px, dq_nm_inv, mirrored):
    """Convert a localization record into the public dataclass (mirrored or not)."""
    qx, qy = float(record["qx_px"]), float(record["qy_px"])
    if mirrored:
        qx, qy = -qx, -qy
    sigma = np.asarray(record["sigma_px"], dtype=float)
    cov = np.asarray(record["cov_px"], dtype=float)
    quality = record["quality"]
    if quality == "ok" and label is None:
        quality = "unmatched"
    return {
        "q_px": (qx, qy),
        "q_nm_inv": (qx * dq_nm_inv, qy * dq_nm_inv),
        "sigma_q_px": (float(sigma[0]), float(sigma[1])),
        "sigma_q_nm_inv": (float(sigma[0]) * dq_nm_inv, float(sigma[1]) * dq_nm_inv),
        "cov_q_px": cov,
        "cov_q_nm_inv": cov * dq_nm_inv**2,
        "amplitude": float(record["amplitude"]),
        "snr": float(record["snr"]),
        "index_hk": label,
        "q_model_px": None if model_px is None else (float(model_px[0]), float(model_px[1])),
        "sigma_q_model_px": model_sigma_px,
        "residual_px": None
        if model_px is None
        else float(np.hypot(qx - model_px[0], qy - model_px[1])),
        "chi2_reduced": float(record["chi2_reduced"]),
        "n_pixels": int(record["n_pixels"]),
        "method": record["method"],
        "quality": quality,
        "independent": not mirrored,
    }


def finalize_peaks(reps, labels, lattice, meta, dq_nm_inv, max_peaks):
    """Select, mirror and build the public peak list."""
    selected = list(
        dict.fromkeys(
            [int(index) for ring in meta.get("ring_members", []) for index in ring]
            + [i for i, label in enumerate(labels) if label is not None]
        )
    )
    fallback = not selected
    if fallback:
        limit = _FALLBACK_REPORT_PEAKS if max_peaks is None else int(max_peaks)
        selected = [
            int(i) for i in np.argsort([-p["snr"] for p in reps], kind="stable")[:limit]
        ]
    order = sorted(selected, key=lambda i: -reps[i]["snr"])[: max_peaks or len(selected)]

    b_px = None if lattice is None else lattice.bvecs_nm_inv / dq_nm_inv
    pairs = []
    for index in order:
        label = labels[index] if index < len(labels) else None
        model_px = None if (label is None or b_px is None) else np.asarray(label, float) @ b_px
        sigma_model = (
            None
            if (label is None or lattice is None)
            else _model_sigma_px(lattice, label, dq_nm_inv)
        )
        mirror_label = None if label is None else (-label[0], -label[1])
        mirror_model = None if model_px is None else (-model_px[0], -model_px[1])
        pairs.append(
            (
                _make_peak(reps[index], label, model_px, sigma_model, dq_nm_inv, False),
                _make_peak(
                    reps[index], mirror_label, mirror_model, sigma_model, dq_nm_inv, True
                ),
            )
        )
    records = [peak for pair in pairs for peak in pair]
    perm = sorted(range(len(records)), key=lambda i: (-records[i]["snr"], i))
    position = {old: new for new, old in enumerate(perm)}
    partner = {
        old: position[old + 1 if old % 2 == 0 else old - 1] for old in range(len(records))
    }
    built = tuple(BraggPeak(**records[old], conjugate_index=partner[old]) for old in perm)
    return built, fallback, len(order)
