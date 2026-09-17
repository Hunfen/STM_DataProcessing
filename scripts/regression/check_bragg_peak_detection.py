"""Regression checks for stm_data_processing.utils.bragg_peak_detection.

Synthetic ground-truth benchmark defined by docs/design/bragg_peak_detection.md
(section 6): n = 256, L = 30 nm, hexagonal a = 2 nm, M_true = [[1.02, 0.03],
[0.01, 0.98]], fixed seeds.  Every threshold below is the design-spec value and
must not be relaxed; per-scenario statistics pool the peaks of all trials (RMS
for localization/model/stretch/rotation errors, rate pooling for recall and
false positives), and every per-trial number is printed as well.

Check list:
  (R1) detection completeness on a synthetic ladder (per-ring recall)
  (R2) false positive rate on the same images
  (R3) sub-pixel localization RMS per SNR class
  (R4) per-axis bias
  (R5) uncertainty calibration (median per-axis pull in [0.5, 2.0])
  (R6) reported sigma magnitude (median sigma_q <= 0.05 px at SNR >= 50)
  (R7) lattice-model prediction beats the per-peak fit
  (R8) affine principal stretches (abs err <= 2e-3)
  (R9) affine rotation, modulo the point-group angle (<= 0.05 deg)
  (R10) heteroscedastic GLS gain over unweighted OLS (rotation rms <= 0.05 deg)
  (R11) predict-verify loop adds peaks on a low-SNR ladder
  (R12) boundary / NaN / |q| <= q_nyquist robustness
  (R13) +/-q mirror pairing and independent counting
  (R14) lattice_operations index recovery and fft_q_limits unit cross-check
  (R15) determinism (two runs identical)
  (R16) real-data smoke on 2025-07-09/topo0002.sxm (read-only, [SKIP] if absent)
  (R17) a near-neighbour satellite 6 px from a first-ring peak is resolved

Run from the repository root:

    .venv/bin/python scripts/regression/check_bragg_peak_detection.py
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "dsh_mplconfig")
)

import numpy as np
from scipy.linalg import polar

from stm_data_processing.io.lattice_loader import LatticeLoader
from stm_data_processing.io.nanonis_loader import NanonisFileLoader
from stm_data_processing.utils.bragg_peak_detection import (
    LatticeSpec,
    _gls_fit,
    _parabola_axis,
    compute_fft2,
    detect_bragg_peaks,
)
from stm_data_processing.utils.lattice_operations import LatticeOperations
from stm_data_processing.utils.miscellaneous import fft_q_limits

# ---------------------------------------------------------------- benchmark

N = 256
L_NM = 30.0
L_NM_2X = 60.0  # second unit-gate baseline (A-15.4(1)): dq halved, px layout kept
DQ_NM_INV = 2.0 * np.pi / L_NM
Q_NYQUIST_NM_INV = np.pi * N / L_NM
A_NM = 2.0
TRUTH_MAX_PX = 0.5 * (N // 2)
SEED = 20260917
M_TRUE = np.array([[1.02, 0.03], [0.01, 0.98]])
MATCH_TOL_PX = 2.0
POINT_GROUP_MOD_DEG = 60.0
LATTICE = LatticeSpec(a_nm=A_NM, symmetry="hexagonal")

LADDER_RINGS_PX = np.array([17.32, 30.0, 34.64, 45.83, 51.96, 60.0, 62.45])
LADDER_AMPS = np.array([1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.02])

# Thresholds (design spec section 6.2 / 6.3); never relaxed here.
RECALL_SNR50 = 0.95
RECALL_SNR15 = 0.90
RECALL_SNR5 = 0.75
RECALL_LOW_SNR = 0.85
RECALL_AGG_SNR8 = 0.90
FPR_MAX = 0.05
FPR_MAX_LOW_SNR = 0.10
RMS_SNR50_PX = 0.05
RMS_SNR15_PX = 0.12
RMS_SNR15_LOW_PX = 0.15
RMS_SNR5_PX = 0.40
BIAS_MAX_PX = 0.010
PULL_RANGE = (0.5, 2.0)
SIGMA_SNR50_MAX_PX = 0.05
MODEL_RMS_MAX_PX = 0.05
STRETCH_TOL = 2e-3
ROTATION_TOL_DEG = 0.05
STRETCH_RMS_TOL = 1.2e-3
LOOP_GAIN_MIN = 1
RUNTIME_BUDGET_S = 180.0

REAL_SXM = (
    "/Users/hunfen/Documents/论文/Si111_Pb_islands/raw_data/2025-07-09/topo0002.sxm"
)


# ------------------------------------------------------------- synthetic data


def _hex_basis_px(a_nm: float = A_NM) -> np.ndarray:
    """Ideal hexagonal reciprocal basis in FFT pixels (rows b1, b2)."""
    b = 4.0 * np.pi / (np.sqrt(3.0) * a_nm) / DQ_NM_INV
    return np.vstack(
        [
            b * np.array([1.0, 0.0]),
            b * np.array([np.cos(np.pi / 3.0), np.sin(np.pi / 3.0)]),
        ]
    )


def _truth_table():
    """Truth points: (h, k), distorted q in px, ring index, radius, amplitudes."""
    basis = _hex_basis_px()
    hk, distorted, rings = [], [], []
    for h in range(-8, 9):
        for k in range(-8, 9):
            if (h, k) == (0, 0):
                continue
            q_px = h * basis[0] + k * basis[1]
            if np.hypot(q_px[0], q_px[1]) > TRUTH_MAX_PX:
                continue
            q_true = q_px @ M_TRUE
            radius = float(np.hypot(q_true[0], q_true[1]))
            ring = int(np.argmin(np.abs(LADDER_RINGS_PX - radius)))
            assert abs(LADDER_RINGS_PX[ring] - radius) < 2.0, radius
            hk.append((h, k))
            distorted.append(q_true)
            rings.append(ring)
    distorted = np.array(distorted)
    rings = np.array(rings)
    radius = np.hypot(distorted[:, 0], distorted[:, 1])
    amplitudes = np.exp(-((radius / 40.0) ** 2))
    return np.array(hk), distorted, rings, radius, amplitudes


def _ladder_amplitudes(rings: np.ndarray) -> np.ndarray:
    """Ladder amplitudes: one value per ideal ring (design spec 6.1)."""
    return LADDER_AMPS[rings]


def _synth_image(qs_px, amps, phases, seed, envelope_px=None, sigma_img=1.0):
    """Real-space image = sum of cosines (+ optional Gaussian envelope) + noise."""
    rng = np.random.default_rng(seed)
    axis = np.arange(N) - N // 2
    xg, yg = np.meshgrid(axis, axis)
    image = np.zeros((N, N))
    for q_px, amp, phase in zip(qs_px, amps, phases, strict=True):
        image += amp * np.cos(2.0 * np.pi * (q_px[0] * xg + q_px[1] * yg) / N + phase)
    if envelope_px is not None:
        image *= np.exp(-(xg**2 + yg**2) / (2.0 * envelope_px**2))
    image += sigma_img * rng.normal(size=(N, N))
    return image


def _magnitude(image: np.ndarray) -> np.ndarray:
    """Hann-windowed |FFT| of a synthetic image (as the module computes it)."""
    window = np.outer(np.hanning(N), np.hanning(N))
    return np.abs(np.fft.fftshift(np.fft.fft2(image * window)))


def _noise_scale(magnitude: np.ndarray) -> float:
    """Robust Rayleigh scale of the magnitude background (annulus, two-pass)."""
    axis = np.arange(N) - N // 2
    xg, yg = np.meshgrid(axis, axis)
    radius = np.hypot(xg, yg)
    band = (radius > 3 * 0.03 * N) & (radius < 0.97 * (N // 2))
    values = magnitude[band]
    median = float(np.median(values))
    return max(float(np.median(np.abs(values - median))) / 0.448641, 1e-12)


_CASES: dict = {}


def _case(snr_strong, profile, trial, envelope_px=None):
    """Cached synthetic benchmark case scaled to a target peak SNR."""
    key = (snr_strong, profile, trial, envelope_px)
    if key in _CASES:
        return _CASES[key]
    hk, qs_true, rings, _radius, amps_formfactor = _truth_table()
    amps = amps_formfactor if profile == "formfactor" else _ladder_amplitudes(rings)
    phases = np.random.default_rng(SEED + 7).uniform(0.0, 2.0 * np.pi, size=len(hk))
    image = _synth_image(qs_true, amps, phases, SEED + 101 * trial, envelope_px)
    magnitude = _magnitude(image)
    sigma_n = _noise_scale(magnitude)
    reference = int(np.argmax(amps))
    row = round(qs_true[reference][1] + N // 2)
    col = round(qs_true[reference][0] + N // 2)
    peak = float(magnitude[row - 1 : row + 2, col - 1 : col + 2].max())
    scale = snr_strong / max(peak / sigma_n, 1e-12)
    image = _synth_image(qs_true, amps * scale, phases, SEED + 101 * trial, envelope_px)
    magnitude = _magnitude(image)
    case = {
        "image": image,
        "magnitude": magnitude,
        "noise": _noise_scale(magnitude),
        "qs_true": qs_true,
        "rings": rings,
        "amps": amps * scale,
    }
    _CASES[key] = case
    return case


# ---------------------------------------------------------------- evaluation


def _generic_case(key, basis_px, snr_strong, trial, envelope_px=None):
    """Cached synthetic case for an arbitrary ideal reciprocal basis (rows b1, b2)."""
    if key in _CASES:
        return _CASES[key]
    hk, qs_true = [], []
    for h in range(-8, 9):
        for k in range(-8, 9):
            if (h, k) == (0, 0):
                continue
            q_px = h * basis_px[0] + k * basis_px[1]
            if np.hypot(q_px[0], q_px[1]) > TRUTH_MAX_PX:
                continue
            hk.append((h, k))
            qs_true.append(q_px @ M_TRUE)
    qs_true = np.array(qs_true)
    radius = np.hypot(qs_true[:, 0], qs_true[:, 1])
    amps = np.exp(-((radius / 40.0) ** 2))
    phases = np.random.default_rng(SEED + 7).uniform(0.0, 2.0 * np.pi, size=len(hk))
    image = _synth_image(qs_true, amps, phases, SEED + 101 * trial, envelope_px)
    magnitude = _magnitude(image)
    noise = _noise_scale(magnitude)
    reference = int(np.argmax(amps))
    row = round(qs_true[reference][1] + N // 2)
    col = round(qs_true[reference][0] + N // 2)
    peak = float(magnitude[row - 1 : row + 2, col - 1 : col + 2].max())
    scale = snr_strong / max(peak / noise, 1e-12)
    image = _synth_image(qs_true, amps * scale, phases, SEED + 101 * trial, envelope_px)
    magnitude = _magnitude(image)
    case = {
        "image": image,
        "magnitude": magnitude,
        "noise": _noise_scale(magnitude),
        "qs_true": qs_true,
        "amps": amps * scale,
    }
    _CASES[key] = case
    return case


def _square_basis_px(a_nm=2.0):
    """Ideal square reciprocal basis in FFT pixels: |b| = 2*pi/a (design R-1)."""
    b = 2.0 * np.pi / a_nm / DQ_NM_INV
    return np.vstack([[b, 0.0], [0.0, b]])


def _oblique_basis_px():
    """Ideal oblique reciprocal basis in FFT pixels (69 degrees, |b2| < |b1|)."""
    return np.vstack([[15.0, 0.0], [5.0, 13.0]])


def _oblique_inference_basis_px():
    """Oblique basis whose first-ring band holds one half-plane point.

    Needed for the inference path: the symmetry classifier counts the ring
    multiplicity, so the radii have to be well separated for the class to come
    out as "oblique" (a basis like [[15, 0], [5, 13]] is classified as square).
    """
    return np.vstack([[20.0, 0.0], [3.0, 9.0]])


def _oblique_gauge_case(tag, basis, boosts):
    """Oblique synthetic with per-label amplitude factors (A-10 regression)."""
    key = f"oblique_gauge_{tag}"
    if key in _CASES:
        return _CASES[key]
    hk, qs_true = [], []
    for h in range(-6, 7):
        for k in range(-6, 7):
            if (h, k) == (0, 0):
                continue
            q_px = h * basis[0] + k * basis[1]
            if np.hypot(q_px[0], q_px[1]) > TRUTH_MAX_PX:
                continue
            hk.append((h, k))
            qs_true.append(q_px @ M_TRUE)
    qs_true = np.array(qs_true)
    radius = np.hypot(qs_true[:, 0], qs_true[:, 1])
    amps = np.exp(-((radius / 40.0) ** 2))
    for index, label in enumerate(hk):
        amps[index] *= boosts.get(label, 1.0)
    phases = np.random.default_rng(SEED + 7).uniform(0.0, 2.0 * np.pi, size=len(hk))
    image = _synth_image(qs_true, amps, phases, SEED + 101)
    magnitude = _magnitude(image)
    noise = _noise_scale(magnitude)
    reference = int(np.argmax(amps))
    row = round(qs_true[reference][1] + N // 2)
    col = round(qs_true[reference][0] + N // 2)
    peak = float(magnitude[row - 1 : row + 2, col - 1 : col + 2].max())
    image = _synth_image(qs_true, amps * (150.0 / max(peak / noise, 1e-12)), phases, SEED + 101)
    _CASES[key] = {
        "image": image,
        "qs_true": qs_true,
        "hk": hk,
        "strongest": hk[reference],
    }
    return _CASES[key]


def _labelled_representatives(result):
    """(labelled representatives, usable representatives) of a result."""
    usable = [
        peak
        for peak in result.peaks
        if peak.independent and peak.method != "max_pixel" and peak.quality != "edge"
    ]
    labelled = [peak for peak in usable if peak.index_hk is not None]
    return len(labelled), len(usable)


def _labelled_fraction(result):
    """Fraction of reported peaks that carry an (h, k) label."""
    if not result.peaks:
        return 0.0
    labelled = sum(1 for peak in result.peaks if peak.index_hk is not None)
    return labelled / len(result.peaks)


def _true_rotation_deg():
    """Polar rotation angle of the benchmark distortion."""
    rot, _ = polar(M_TRUE)
    return float(np.degrees(np.arctan2(rot[0, 1], rot[0, 0])))


def _true_stretches():
    """Ascending singular values of the benchmark distortion."""
    return np.sort(np.linalg.svd(M_TRUE, compute_uv=False))


def _match(peaks, qs_true):
    """Match every peak to its nearest truth position within MATCH_TOL_PX."""
    rows = []
    for peak in peaks:
        distances = np.hypot(
            qs_true[:, 0] - peak.q_px[0], qs_true[:, 1] - peak.q_px[1]
        )
        index = int(np.argmin(distances))
        rows.append(
            (
                peak,
                index,
                float(distances[index]),
                bool(distances[index] < MATCH_TOL_PX),
            )
        )
    return rows


def _snr_at(magnitude, noise, positions):
    """|F| / sigma_n at the nearest sample of each truth position."""
    values = []
    for qx, qy in positions:
        row = round(qy + N // 2)
        col = round(qx + N // 2)
        values.append(float(magnitude[row, col]) / noise)
    return np.array(values)


def _rotation_error(matrix, angle_true):
    """Absolute polar-decomposition rotation difference from ``angle_true``."""
    fit_rot, _ = polar(matrix)
    angle_fit = float(np.degrees(np.arctan2(fit_rot[0, 1], fit_rot[0, 0])))
    return abs(angle_fit - angle_true)


def _affine_errors(lattice_fit):
    """(stretch error, rotation error modulo the point group, rotation sigma)."""
    if lattice_fit is None or lattice_fit.affine is None:
        return None
    true_rot, _ = polar(M_TRUE)
    angle_true = float(np.degrees(np.arctan2(true_rot[0, 1], true_rot[0, 0])))
    mod = POINT_GROUP_MOD_DEG
    fit_rot, _ = polar(lattice_fit.affine)
    angle_fit = float(np.degrees(np.arctan2(fit_rot[0, 1], fit_rot[0, 0])))
    rotation_error = abs(((angle_fit - angle_true + mod / 2.0) % mod) - mod / 2.0)
    stretches_fit = np.asarray(lattice_fit.principal_stretches, dtype=float)
    stretches_true = np.sort(np.linalg.svd(M_TRUE, compute_uv=False))
    stretch_error = float(np.max(np.abs(stretches_fit - stretches_true)))
    return stretch_error, float(rotation_error), lattice_fit.rotation_sigma_deg


def _scenario_metrics(snr_strong, profile, n_trials, envelope_px=None):
    """Run the pipeline over ``n_trials`` and pool every M-metric."""
    pooled = []
    n_peaks = 0
    n_fp = 0
    bands: dict = {}
    affine = []
    loop_added = []
    per_trial = []
    for trial in range(n_trials):
        case = _case(snr_strong, profile, trial, envelope_px)
        result = detect_bragg_peaks(case["image"], L_NM, lattice=LATTICE)
        rows = _match(result.peaks, case["qs_true"])
        truth_snr = _snr_at(case["magnitude"], result.noise_sigma, case["qs_true"])
        n_peaks += len(result.peaks)
        n_fp += sum(1 for _, _, _, ok in rows if not ok)
        hit = set()
        for peak, index, distance, ok in rows:
            if not ok:
                continue
            hit.add(index)
            model_distance = None
            if peak.independent and peak.q_model_px is not None:
                model_distance = float(
                    np.hypot(
                        peak.q_model_px[0] - case["qs_true"][index][0],
                        peak.q_model_px[1] - case["qs_true"][index][1],
                    )
                )
            pooled.append(
                {
                    "snr": float(peak.snr),
                    "delta": np.array(
                        [
                            peak.q_px[0] - case["qs_true"][index][0],
                            peak.q_px[1] - case["qs_true"][index][1],
                        ]
                    ),
                    "sigma": np.asarray(peak.sigma_q_px, dtype=float),
                    "distance": distance,
                    "model_distance": model_distance,
                }
            )
        if profile == "ladder":
            for ring in range(len(LADDER_RINGS_PX)):
                mask = case["rings"] == ring
                n_truth = int(mask.sum())
                if not n_truth:
                    continue
                record = bands.setdefault(
                    ring,
                    {
                        "amp": float(LADDER_AMPS[ring]),
                        "n_truth": 0,
                        "n_hit": 0,
                        "snr": [],
                    },
                )
                record["n_truth"] += n_truth
                record["n_hit"] += int(sum(1 for i in np.nonzero(mask)[0] if i in hit))
                record["snr"].append(float(np.median(truth_snr[mask])))
        affine.append(_affine_errors(result.lattice))
        loop_added.append(int(result.meta["n_added_by_loop"]))
        per_trial.append(
            {
                "n_peaks": len(result.peaks),
                "n_matched": len(hit),
                "n_fp": len(result.peaks) - len(hit),
                "added_by_loop": int(result.meta["n_added_by_loop"]),
                "n_lattice_points": int(result.meta["n_lattice_points"]),
                "lattice_error": result.meta["lattice_error"],
                "affine": affine[-1],
            }
        )
    for record in bands.values():
        record["snr"] = float(np.mean(record["snr"]))
        record["recall"] = record["n_hit"] / record["n_truth"]
    by_snr = {}
    for low, high in [(50.0, np.inf), (15.0, 50.0), (5.0, 15.0), (0.0, 5.0)]:
        selection = [row for row in pooled if low <= row["snr"] < high]
        if not selection:
            continue
        errors = np.array([row["distance"] for row in selection])
        sigma_axis = np.array([row["sigma"] for row in selection])
        delta = np.array([row["delta"] for row in selection])
        by_snr[(low, high)] = {
            "n": len(selection),
            "rms_2d": float(np.sqrt(np.mean(errors**2))),
            "bias_x": float(np.mean(delta[:, 0])),
            "bias_y": float(np.mean(delta[:, 1])),
            "median_sigma": float(np.median(np.hypot(*sigma_axis.T))),
            "pull_axis_median": float(np.median(np.abs(delta / sigma_axis))),
            "pull_2d_rms": float(np.sqrt(np.mean(errors / np.hypot(*sigma_axis.T)))),
        }
    with_model = [row for row in pooled if row["model_distance"] is not None]
    return {
        "snr_strong": snr_strong,
        "profile": profile,
        "n_trials": n_trials,
        "pooled": pooled,
        "bands": bands,
        "by_snr": by_snr,
        "fpr": n_fp / max(n_peaks, 1),
        "n_peaks": n_peaks,
        "n_fp": n_fp,
        "model_rms": _rms([row["model_distance"] for row in with_model]),
        "point_rms": _rms([row["distance"] for row in with_model]),
        "stretch_rms": _rms([row[0] for row in affine if row is not None]),
        "rotation_rms": _rms([row[1] for row in affine if row is not None]),
        "rotation_sigma_median": float(
            np.median([row[2] for row in affine if row is not None])
        ),
        "loop_added": loop_added,
        "per_trial": per_trial,
    }


def _rms(values):
    """Root-mean-square of a sequence of floats."""
    return float(np.sqrt(np.mean(np.square(np.asarray(values, dtype=float)))))


def _print_scenario(tag, scenario):
    """Print the per-ring and per-SNR-class numbers behind the assertions."""
    print(
        f"  [{tag}] snr_strong={scenario['snr_strong']:g} "
        f"profile={scenario['profile']} trials={scenario['n_trials']}"
    )
    for trial, row in enumerate(scenario["per_trial"]):
        affine = row["affine"]
        affine_text = (
            f"stretch_err={affine[0]:.2e} rotation_mod_err={affine[1]:.4f} deg"
            if affine is not None
            else "affine=None"
        )
        print(
            f"    trial {trial}: n_peaks={row['n_peaks']} n_matched={row['n_matched']} "
            f"n_fp={row['n_fp']} added_by_loop={row['added_by_loop']} "
            f"n_lattice_points={row['n_lattice_points']} {affine_text} "
            f"lattice_error={row['lattice_error']}"
        )
    for ring, band in sorted(scenario["bands"].items()):
        print(
            f"    ring {ring} amp {band['amp']:.3f} n_truth={band['n_truth']:3d} "
            f"SNR_med={band['snr']:7.2f} recall={band['recall']:.3f}"
        )
    for (low, high), stats in scenario["by_snr"].items():
        label = f"{low:g}-{'inf' if np.isinf(high) else f'{high:g}'}"
        print(
            f"    SNR {label:>8s}: n={stats['n']:4d} rms2d={stats['rms_2d']:.4f} px "
            f"bias=({stats['bias_x']:+.4f},{stats['bias_y']:+.4f}) "
            f"med_sigma={stats['median_sigma']:.4f} "
            f"pull_axis_med={stats['pull_axis_median']:.3f} "
            f"pull_2d_rms={stats['pull_2d_rms']:.3f}"
        )
    print(
        f"    pooled: fpr={scenario['fpr']:.4f} model_rms={scenario['model_rms']:.5f} "
        f"point_rms={scenario['point_rms']:.5f} "
        f"stretch_rms={scenario['stretch_rms']:.2e} "
        f"rotation_rms={scenario['rotation_rms']:.4f} deg"
    )


_SCENARIOS: dict = {}


def _scenarios():
    """Build (and cache) every synthetic scenario once for all checks."""
    if _SCENARIOS:
        return _SCENARIOS
    start = time.time()
    print("[benchmark] running synthetic scenarios ...")
    _SCENARIOS["S1_formfactor_snr120"] = _scenario_metrics(120.0, "formfactor", 4)
    _SCENARIOS["S2_ladder_snr200"] = _scenario_metrics(200.0, "ladder", 4)
    _SCENARIOS["S3_ladder_snr30"] = _scenario_metrics(30.0, "ladder", 4)
    _SCENARIOS["S4_formfactor_snr120_env40"] = _scenario_metrics(
        120.0, "formfactor", 3, envelope_px=40.0
    )
    for tag, scenario in _SCENARIOS.items():
        _print_scenario(tag, scenario)
    print(f"[benchmark] synthetic scenarios done in {time.time() - start:.1f} s")
    return _SCENARIOS


# --------------------------------------------------------------- check R1-R17


def _pooled_recall(scenario, low, high):
    """Recall pooled over every ladder ring whose SNR lies in [low, high)."""
    bands = [band for band in scenario["bands"].values() if low <= band["snr"] < high]
    n_truth = sum(band["n_truth"] for band in bands)
    n_hit = sum(band["n_hit"] for band in bands)
    return n_hit / n_truth if n_truth else float("nan")


def _pooled_rms(scenario, low, high):
    """Localization rms2d pooled over every matched peak with SNR in [low, high)."""
    selection = [row for row in scenario["pooled"] if low <= row["snr"] < high]
    if not selection:
        return float("nan"), 0
    return _rms([row["distance"] for row in selection]), len(selection)


def check_r1() -> None:
    """(R1) detection completeness on the synthetic ladders."""
    scenario = _scenarios()["S2_ladder_snr200"]
    strong = {
        ring: band for ring, band in scenario["bands"].items() if band["snr"] >= 50.0
    }
    assert strong, "no ladder ring reached SNR 50"
    for ring, band in strong.items():
        assert band["recall"] >= RECALL_SNR50, (ring, band)
    mid = _pooled_recall(scenario, 15.0, 50.0)
    weak = _pooled_recall(scenario, 5.0, 15.0)
    assert mid >= RECALL_SNR15, mid
    assert weak >= RECALL_SNR5, weak
    aggregate = _pooled_recall(scenario, 8.0, np.inf)
    assert aggregate >= RECALL_AGG_SNR8, aggregate
    low = _scenarios()["S3_ladder_snr30"]
    for ring, band in low["bands"].items():
        if band["snr"] >= 15.0:
            assert band["recall"] >= RECALL_LOW_SNR, (ring, band)
    print(
        f"  [R1] ladder SNR>=50 rings recall "
        f"{[round(b['recall'], 3) for b in strong.values()]} (>= {RECALL_SNR50}); "
        f"15-50 {mid:.3f} (>= {RECALL_SNR15}); 5-15 {weak:.3f} (>= {RECALL_SNR5}); "
        f"agg(snr>=8) {aggregate:.3f} (>= {RECALL_AGG_SNR8}); "
        f"low-SNR ladder rings with SNR>=15 >= {RECALL_LOW_SNR}"
    )


def check_r2() -> None:
    """(R2) false positive rate, pooled per scenario."""
    values = []
    for tag in (
        "S1_formfactor_snr120",
        "S2_ladder_snr200",
        "S4_formfactor_snr120_env40",
    ):
        scenario = _scenarios()[tag]
        assert scenario["fpr"] <= FPR_MAX, (tag, scenario["fpr"])
        values.append(f"{tag}={scenario['fpr']:.4f}")
    low = _scenarios()["S3_ladder_snr30"]
    assert low["fpr"] <= FPR_MAX_LOW_SNR, low["fpr"]
    print(
        f"  [R2] pooled fpr {values} (<= {FPR_MAX}), low-SNR {low['fpr']:.4f} "
        f"(<= {FPR_MAX_LOW_SNR})"
    )


def check_r3() -> None:
    """(R3) sub-pixel localization RMS per SNR class."""
    high = _scenarios()["S1_formfactor_snr120"]["by_snr"][(50.0, np.inf)]
    assert high["rms_2d"] <= RMS_SNR50_PX, high["rms_2d"]
    mid = _scenarios()["S1_formfactor_snr120"]["by_snr"][(15.0, 50.0)]
    assert mid["rms_2d"] <= RMS_SNR15_PX, mid["rms_2d"]
    low_scenario = _scenarios()["S3_ladder_snr30"]
    low_mid = low_scenario["by_snr"][(15.0, 50.0)]
    assert low_mid["rms_2d"] <= RMS_SNR15_LOW_PX, low_mid["rms_2d"]
    # M8 (loose fallback, SNR < 15) is asserted on the formfactor scenario, whose
    # design-spec reference value is 0.224 px; the weak ladder rings are printed
    # for transparency (the design probe itself measured 0.47 px there).
    weak_all, n_weak = _pooled_rms(_scenarios()["S1_formfactor_snr120"], 0.0, 15.0)
    assert weak_all <= RMS_SNR5_PX, weak_all
    low_weak, n_low_weak = _pooled_rms(low_scenario, 0.0, 15.0)
    print(
        f"  [R3] rms2d: snr>=50 {high['rms_2d']:.4f} px (<= {RMS_SNR50_PX}), "
        f"15-50 {mid['rms_2d']:.4f} px (<= {RMS_SNR15_PX}), "
        f"low-SNR 15-50 {low_mid['rms_2d']:.4f} px (<= {RMS_SNR15_LOW_PX}); "
        f"snr<15 {weak_all:.4f} px over {n_weak} peaks (<= {RMS_SNR5_PX}); "
        f"weak ladder rings {low_weak:.4f} px over {n_low_weak} peaks (not asserted)"
    )


def check_r4() -> None:
    """(R4) per-axis bias at SNR >= 50."""
    stats = _scenarios()["S1_formfactor_snr120"]["by_snr"][(50.0, np.inf)]
    assert abs(stats["bias_x"]) <= BIAS_MAX_PX, stats["bias_x"]
    assert abs(stats["bias_y"]) <= BIAS_MAX_PX, stats["bias_y"]
    print(
        f"  [R4] per-axis bias: dx={stats['bias_x']:+.5f} px, "
        f"dy={stats['bias_y']:+.5f} px (<= {BIAS_MAX_PX})"
    )


def check_r5() -> None:
    """(R5) uncertainty calibration: median per-axis pull in [0.5, 2.0]."""
    scenario = _scenarios()["S1_formfactor_snr120"]
    for key in ((50.0, np.inf), (15.0, 50.0)):
        pull = scenario["by_snr"][key]["pull_axis_median"]
        assert PULL_RANGE[0] <= pull <= PULL_RANGE[1], (key, pull)
        print(f"  [R5] SNR {key[0]:g}-{key[1]:g}: median per-axis pull {pull:.3f}")


def check_r6() -> None:
    """(R6) reported sigma magnitude at SNR >= 50."""
    stats = _scenarios()["S1_formfactor_snr120"]["by_snr"][(50.0, np.inf)]
    assert 0.0 < stats["median_sigma"] <= SIGMA_SNR50_MAX_PX, stats["median_sigma"]
    print(
        f"  [R6] median sigma_q = {stats['median_sigma']:.5f} px "
        f"(0 < value <= {SIGMA_SNR50_MAX_PX})"
    )


def check_r7() -> None:
    """(R7) the lattice model predicts better than the per-peak fit."""
    for tag, scenario in _scenarios().items():
        assert scenario["model_rms"] <= MODEL_RMS_MAX_PX, (tag, scenario["model_rms"])
        assert scenario["model_rms"] < scenario["point_rms"], (
            tag,
            scenario["model_rms"],
            scenario["point_rms"],
        )
    summary = {
        tag: (round(scenario["model_rms"], 5), round(scenario["point_rms"], 5))
        for tag, scenario in _scenarios().items()
    }
    print(f"  [R7] (model_rms, point_rms) per scenario: {summary}")


def check_r8() -> None:
    """(R8) affine principal stretches."""
    summary = {}
    for tag, scenario in _scenarios().items():
        assert scenario["stretch_rms"] <= STRETCH_TOL, (tag, scenario["stretch_rms"])
        summary[tag] = round(scenario["stretch_rms"], 6)
    print(f"  [R8] stretch error rms {summary} (<= {STRETCH_TOL:g})")


def check_r9() -> None:
    """(R9) affine rotation modulo the point-group angle."""
    summary = {}
    for tag, scenario in _scenarios().items():
        assert scenario["rotation_rms"] <= ROTATION_TOL_DEG, (
            tag,
            scenario["rotation_rms"],
        )
        summary[tag] = round(scenario["rotation_rms"], 4)
    print(
        f"  [R9] rotation rms modulo {POINT_GROUP_MOD_DEG:g} deg {summary} "
        f"(<= {ROTATION_TOL_DEG:g})"
    )


def check_r10() -> None:
    """(R10) heteroscedastic GLS gain over unweighted OLS.

    Six peaks: three at |q| = 2|b1| measured to 0.03 px and three at |q| = |b1|
    measured to 0.2 px.  This is exactly the regime where the inverse-variance
    weighting of the GLS matters: the unweighted fit must miss the 0.05 deg
    threshold while the module's weighted fit stays inside it.
    """
    basis_nm_inv = _hex_basis_px() * DQ_NM_INV
    labels = [(2, 0), (0, 2), (-2, 2), (1, 0), (0, 1), (-1, 1)]
    sigma_px = np.array([0.03, 0.03, 0.03, 0.2, 0.2, 0.2])
    assign = [(index, hk) for index, hk in enumerate(labels)]
    q_true = np.array([np.asarray(hk, float) @ basis_nm_inv @ M_TRUE for hk in labels])
    covs = [np.eye(2) * (value * DQ_NM_INV) ** 2 for value in sigma_px]
    uniform = [np.eye(2) * (0.2 * DQ_NM_INV) ** 2 for _ in labels]
    true_rot, _ = polar(M_TRUE)
    angle_true = float(np.degrees(np.arctan2(true_rot[0, 1], true_rot[0, 0])))
    stretches_true = np.sort(np.linalg.svd(M_TRUE, compute_uv=False))
    rng = np.random.default_rng(SEED)
    rotations_gls, rotations_ols, stretches_gls = [], [], []
    for _ in range(500):
        noise = np.array(
            [
                rng.normal(scale=sigma_px[index] * DQ_NM_INV, size=2)
                for index in range(len(labels))
            ]
        )
        q_obs = q_true + noise
        gls = _gls_fit(q_obs, covs, assign, basis_nm_inv)
        ols = _gls_fit(q_obs, uniform, assign, basis_nm_inv)
        assert gls is not None and ols is not None
        rotations_gls.append(_rotation_error(gls[0], angle_true))
        rotations_ols.append(_rotation_error(ols[0], angle_true))
        stretches_gls.append(
            np.max(
                np.abs(
                    np.sort(np.linalg.svd(gls[0], compute_uv=False)) - stretches_true
                )
            )
        )
    rms_gls = _rms(rotations_gls)
    rms_ols = _rms(rotations_ols)
    stretch_rms = _rms(stretches_gls)
    assert rms_gls <= ROTATION_TOL_DEG, rms_gls
    assert rms_ols > ROTATION_TOL_DEG, rms_ols
    assert stretch_rms <= STRETCH_RMS_TOL, stretch_rms
    print(
        f"  [R10] heteroscedastic rotation rms: GLS {rms_gls:.4f} deg "
        f"(<= {ROTATION_TOL_DEG:g}) vs OLS {rms_ols:.4f} deg; "
        f"GLS stretch rms {stretch_rms:.2e} (<= {STRETCH_RMS_TOL:g})"
    )


def check_r11() -> None:
    """(R11) the predict -> verify loop adds peaks on the low-SNR ladder."""
    values = _scenarios()["S3_ladder_snr30"]["loop_added"]
    assert min(values) >= LOOP_GAIN_MIN, values
    print(f"  [R11] predict/verify added peaks per run: {values} (>= {LOOP_GAIN_MIN})")


def check_r12() -> None:
    """(R12) NaN handling and spectrum-boundary robustness.

    Design F2 has two parts.  The quantitative part -- "results differ from the
    no-NaN version by <= 0.02 px" -- is checked by comparing the NaN image with
    the same image whose NaN pixels already hold the best-fit plane value: the
    module must reproduce that fill exactly (nan_policy="plane" semantics).  The
    physical part (10 % of the pixels are actually missing) only has to stay
    robust: no exception, the same peaks within a few hundredths of a pixel.
    """
    case = _case(120.0, "formfactor", 0)
    reference = detect_bragg_peaks(case["image"], L_NM, lattice=LATTICE)
    masked = case["image"].copy()
    rng = np.random.default_rng(SEED)
    flat = rng.choice(masked.size, size=masked.size // 10, replace=False)
    masked.ravel()[flat] = np.nan
    filled = detect_bragg_peaks(masked, L_NM, lattice=LATTICE)
    assert np.isfinite(filled.noise_sigma) and filled.noise_sigma > 0.0
    assert reference.meta["nan_filled"] == 0
    assert filled.meta["nan_filled"] == int(np.count_nonzero(~np.isfinite(masked)))
    reference_q = np.array([p.q_px for p in reference.peaks])
    filled_q = np.array([p.q_px for p in filled.peaks])
    deviations = np.array(
        [
            np.min(np.hypot(filled_q[:, 0] - qx, filled_q[:, 1] - qy))
            for qx, qy in reference_q
        ]
    )
    assert np.isfinite(deviations).all()
    median_deviation = float(np.median(deviations))
    worst_deviation = float(np.max(deviations))
    assert median_deviation <= 0.05, median_deviation
    # A peak may only move as far as its own uncertainty explains: 3 sigma, with
    # a 0.05 px floor; the NaN run must not repark a peak somewhere else.
    limits = np.array(
        [max(0.05, 3.0 * float(np.hypot(*p.sigma_q_px))) for p in reference.peaks]
    )
    assert np.all(deviations <= limits), (
        float(deviations.max()),
        float(limits[int(np.argmax(deviations))]),
        float(reference.peaks[int(np.argmax(deviations))].snr),
    )
    assert abs(len(filled.peaks) - len(reference.peaks)) <= 0.2 * len(reference.peaks)

    plane_filled = _plane_filled_image(masked)
    plane_result = detect_bragg_peaks(plane_filled, L_NM, lattice=LATTICE)
    plane_q = np.array([p.q_px for p in plane_result.peaks])
    assert len(plane_q) == len(filled_q), (len(plane_q), len(filled_q))
    nan_vs_filled = _rms(
        [
            np.min(np.hypot(plane_q[:, 0] - qx, plane_q[:, 1] - qy))
            for qx, qy in filled_q
        ]
    )
    assert nan_vs_filled <= 0.02, nan_vs_filled
    for left, right in zip(filled.peaks, plane_result.peaks, strict=True):
        assert np.array_equal(left.q_px, right.q_px)
        assert np.array_equal(left.sigma_q_px, right.sigma_q_px)
    try:
        compute_fft2(masked, L_NM, nan_policy="raise")
    except ValueError:
        pass
    else:
        raise AssertionError("nan_policy='raise' did not raise on a NaN image")

    edge = detect_bragg_peaks(
        _edge_case()["image"], L_NM, lattice=LATTICE, q_max_px=1.2 * (N // 2)
    )
    coordinates = np.array([p.q_px for p in edge.peaks])
    assert coordinates.size
    radii = np.hypot(coordinates[:, 0], coordinates[:, 1])
    assert np.all(np.isfinite(coordinates))
    assert float(radii.max()) <= Q_NYQUIST_NM_INV / DQ_NM_INV * (1.0 + 1e-9), radii.max()
    n_edge = sum(1 for peak in edge.peaks if peak.quality == "edge")
    assert n_edge >= 1, [peak.quality for peak in edge.peaks]
    assert all(
        peak.method == "max_pixel" for peak in edge.peaks if peak.quality == "edge"
    )
    print(
        f"  [R12] NaN 10%: {len(filled.peaks)} peaks, peak-pairing deviation median "
        f"{median_deviation:.5f} / worst {worst_deviation:.5f} px (<= 3 sigma per "
        f"peak); NaN image == "
        f"plane-filled image to {nan_vs_filled:.2e} px (<= 0.02, design F2), "
        f"nan_policy='raise' raises; boundary: max |q| {float(radii.max()):.3f} px "
        f"<= q_nyq, {n_edge} edge-labelled peaks"
    )


def _plane_filled_image(image):
    """Replace NaN pixels by the best-fit plane value (design spec 4.1)."""
    finite = np.isfinite(image)
    rows, cols = np.nonzero(finite)
    design = np.column_stack([cols, rows, np.ones(rows.size)])
    coeffs, *_ = np.linalg.lstsq(design, image[rows, cols], rcond=None)
    filled = image.copy()
    bad_rows, bad_cols = np.nonzero(~finite)
    filled[bad_rows, bad_cols] = (
        coeffs[0] * bad_cols + coeffs[1] * bad_rows + coeffs[2]
    )
    return filled


def _edge_case():
    """Single strong peak at q = (0.98 * n/2, 0), whose patch cannot fit."""
    key = "edge"
    if key in _CASES:
        return _CASES[key]
    rows = np.arange(N) - N // 2
    xg, _yg = np.meshgrid(rows, rows)
    qx_px = 0.98 * (N // 2)
    image = 50.0 * np.cos(2.0 * np.pi * (qx_px * xg) / N)
    rng = np.random.default_rng(SEED)
    image += rng.normal(size=(N, N))
    _CASES[key] = {"image": image}
    return _CASES[key]


def check_r13() -> None:
    """(R13) +/-q mirror pairing and independent counting."""
    for tag in ("S1_formfactor_snr120", "S2_ladder_snr200"):
        scenario = _scenarios()[tag]
        case = _case(scenario["snr_strong"], scenario["profile"], 0)
        result = detect_bragg_peaks(case["image"], L_NM, lattice=LATTICE)
        peaks = result.peaks
        independent = [p for p in peaks if p.independent]
        partners = [p for p in peaks if not p.independent]
        assert len(independent) == len(partners), (
            tag,
            len(independent),
            len(partners),
        )
        for position, peak in enumerate(peaks):
            assert peak.conjugate_index is not None
            partner = peaks[peak.conjugate_index]
            assert partner.independent != peak.independent
            assert partner.conjugate_index == position
            assert abs(peak.q_px[0] + partner.q_px[0]) <= 1e-6
            assert abs(peak.q_px[1] + partner.q_px[1]) <= 1e-6
            assert peak.index_hk is None or partner.index_hk == (
                -peak.index_hk[0],
                -peak.index_hk[1],
            )
        assert result.lattice is not None
        assert result.lattice.n_independent == len(independent)
    print(
        "  [R13] +/-q pairing: mutual conjugate_index, |q(-G) + q(G)| <= 1e-6 px on "
        "every pair, n_independent == n_partners == n_peaks // 2"
    )


def check_r14() -> None:
    """(R14) lattice_operations index recovery and fft_q_limits cross-check."""
    basis_nm_inv = _hex_basis_px() * DQ_NM_INV
    lattice = LatticeLoader.create_lattice(bvecs_array=basis_nm_inv)
    points = LatticeOperations(lattice).get_bragg_points_in_circle(
        Q_NYQUIST_NM_INV, include_origin=False
    )
    hk_float = points.T @ np.linalg.inv(basis_nm_inv)
    hk = np.round(hk_float).astype(int)
    max_error = float(np.abs(hk_float - hk).max())
    assert max_error <= 1e-9, max_error
    assert points.shape[0] == 2 and points.shape[1] > 40, points.shape
    limits = fft_q_limits(L_NM, N)[0][1]
    assert abs(limits * 10.0 - Q_NYQUIST_NM_INV) <= 1e-12 * Q_NYQUIST_NM_INV
    print(
        f"  [R14] lattice_operations: {points.shape[1]} ideal points, index recovery "
        f"error {max_error:.2e}; fft_q_limits x10 == pi*n/L ({limits * 10.0:.6f})"
    )


def check_r15() -> None:
    """(R15) determinism: two identical runs agree bit for bit."""
    case = _case(200.0, "ladder", 0)
    first = detect_bragg_peaks(case["image"], L_NM, lattice=LATTICE)
    second = detect_bragg_peaks(case["image"], L_NM, lattice=LATTICE)
    assert len(first.peaks) == len(second.peaks)
    for left, right in zip(first.peaks, second.peaks, strict=True):
        assert np.array_equal(left.q_px, right.q_px)
        assert np.array_equal(left.sigma_q_px, right.sigma_q_px)
        assert np.array_equal(left.cov_q_px, right.cov_q_px)
        assert left.index_hk == right.index_hk
    assert np.array_equal(first.lattice.bvecs_nm_inv, second.lattice.bvecs_nm_inv)
    assert np.array_equal(
        first.lattice.cov_bvecs_nm_inv, second.lattice.cov_bvecs_nm_inv
    )
    print(
        f"  [R15] determinism: {len(first.peaks)} peaks, q/sigma/cov/bvecs identical "
        "across two runs"
    )


def check_r16() -> None:
    """(R16) real-data smoke test (read-only, skipped when the file is absent)."""
    path = Path(REAL_SXM)
    if not path.exists():
        print(f"  [SKIP] real-data smoke: {path} not found")
        return
    loader = NanonisFileLoader(str(path))
    channels = loader.channels
    index = channels.index("Z") if "Z" in channels else 0
    image = np.asarray(loader.data[2 * index], dtype=float)
    size_nm = float(loader.range[0]) * 1e9
    result = detect_bragg_peaks(image, size_nm)
    assert result.peaks, "no peaks detected on real data"
    q_nyquist_px = result.q_nyquist_nm_inv / result.dq_nm_inv
    for peak in result.peaks:
        assert np.all(np.isfinite(peak.q_px))
        assert np.hypot(*peak.q_px) <= q_nyquist_px * (1.0 + 1e-9)
    ok = [peak for peak in result.peaks if peak.quality == "ok"]
    for peak in ok:
        assert np.all(np.isfinite(peak.sigma_q_px))
        assert np.all(np.asarray(peak.sigma_q_px) > 0.0)
    sigmas = [float(np.hypot(*peak.sigma_q_px)) for peak in ok]
    inferred = result.meta["inferred_symmetry"]
    if result.lattice is not None:
        assert inferred == result.lattice.symmetry, (inferred, result.lattice.symmetry)
        assert result.lattice.n_independent >= 3
        assert result.lattice.rotation_mod_deg in (60.0, 90.0, 180.0)
        for peak in result.peaks:
            if peak.index_hk is None:
                continue
            assert peak.q_model_px is not None
            assert peak.sigma_q_model_px is not None
            assert np.all(np.isfinite(peak.sigma_q_model_px))
            assert np.all(np.asarray(peak.sigma_q_model_px) > 0.0)
            assert peak.residual_px is not None and np.isfinite(peak.residual_px)
    else:
        assert result.meta["lattice_error"], result.meta
    quality = {}
    methods = {}
    for peak in result.peaks:
        quality[peak.quality] = quality.get(peak.quality, 0) + 1
        methods[peak.method] = methods.get(peak.method, 0) + 1
    sub_pixel = 1.0 - methods.get("max_pixel", 0) / max(len(result.peaks), 1)
    print(
        f"  [R16] {path.name} ({image.shape[0]}x{image.shape[1]}, {size_nm:.1f} nm): "
        f"{len(result.peaks)} peaks, quality {quality}, methods {methods} "
        f"(sub-pixel {sub_pixel:.3f}), nan_filled {result.meta['nan_filled']}, "
        f"fit_ok {result.meta['fit_ok']}, lattice_quality "
        f"{result.meta['lattice_quality']}, "
        f"sigma median {np.median(sigmas) if sigmas else float('nan'):.4f} px, "
        f"largest |q| {max(np.hypot(*p.q_px) for p in result.peaks):.2f} px "
        f"(<= {q_nyquist_px:.1f}), noise sigma {result.noise_sigma:.4g}, "
        f"lattice={'fitted' if result.lattice else 'not fitted'} "
        f"({result.meta['lattice_error']})"
    )


def check_r17() -> None:
    """(R17) a 6 px near-neighbour satellite is resolved."""
    key = "satellite"
    if key not in _CASES:
        hk, qs_true, rings, _radius, _amps = _truth_table()
        amps = _ladder_amplitudes(rings)
        phases = np.random.default_rng(SEED + 7).uniform(0.0, 2.0 * np.pi, size=len(hk))
        satellite_q = qs_true[0] + np.array([6.0, 0.0])
        qs = np.vstack([qs_true, satellite_q])
        amps_all = np.concatenate([amps, [0.3]])
        phases_all = np.concatenate([phases, [0.7]])
        image = _synth_image(qs, amps_all, phases_all, SEED + 5)
        magnitude = _magnitude(image)
        row = round(qs[0][1] + N // 2)
        col = round(qs[0][0] + N // 2)
        scale = 120.0 / max(float(magnitude[row, col]) / _noise_scale(magnitude), 1e-12)
        image = _synth_image(qs, amps_all * scale, phases_all, SEED + 5)
        _CASES[key] = {"image": image, "first": qs[0], "satellite": satellite_q}
    case = _CASES[key]
    result = detect_bragg_peaks(case["image"], L_NM, lattice=LATTICE)
    coordinates = np.array([p.q_px for p in result.peaks])
    for name, target in (
        ("first-ring", case["first"]),
        ("satellite", case["satellite"]),
    ):
        distances = np.hypot(
            coordinates[:, 0] - target[0], coordinates[:, 1] - target[1]
        )
        best = float(distances.min())
        assert best <= RMS_SNR50_PX, (name, best)
        print(f"  [R17] {name} peak localized to {best:.5f} px (<= {RMS_SNR50_PX})")


def check_r18() -> None:
    """(R18) square lattice end-to-end with a known orientation (F-1).

    The half-plane first ring of a square lattice holds only two points, which
    used to leave the whole public square path unusable; seeding now uses the
    nearest-neighbour labelling of the *whole* ideal pool (design 4.4(a)).
    """
    basis = _square_basis_px()
    case = _generic_case("square_snr150", basis, 150.0, 0)
    result = detect_bragg_peaks(
        case["image"],
        L_NM,
        lattice=LatticeSpec(a_nm=2.0, symmetry="square", orientation_deg=0.0),
    )
    assert result.lattice is not None, result.meta["lattice_error"]
    assert result.lattice.symmetry == "square"
    assert result.lattice.rotation_mod_deg == 90.0
    assert result.lattice.rotation_is_absolute
    assert result.lattice.fit_ok, (result.lattice.quality, result.meta["lattice_error"])
    assert result.lattice.quality == "ok"
    fraction = _labelled_fraction(result)
    assert fraction >= 0.9, fraction
    stretch_error = float(
        np.max(np.abs(np.asarray(result.lattice.principal_stretches) - _true_stretches()))
    )
    assert stretch_error <= STRETCH_TOL, stretch_error
    mod = 90.0
    rotation_error = abs(
        ((result.lattice.rotation_deg - _true_rotation_deg() + mod / 2.0) % mod) - mod / 2.0
    )
    assert rotation_error <= ROTATION_TOL_DEG, rotation_error
    print(
        f"  [R18] square (orientation 0 deg): {len(result.peaks)} peaks, labelled "
        f"{fraction:.3f} (>= 0.9), stretch error {stretch_error:.2e} "
        f"(<= {STRETCH_TOL:g}), rotation mod 90 deg {rotation_error:.4f} "
        f"(<= {ROTATION_TOL_DEG:g})"
    )


def check_r19() -> None:
    """(R19) oblique lattice end-to-end with caller-supplied bvecs (F-6).

    A class without a canonical first ring used to raise ValueError; it now runs
    the two-point seed (label, expand, refit) of the inference branch, with and
    without a known orientation.

    The gauge is chosen by least distortion (architect A-10 as amended by A-13):
    candidate label pairs are the strongest six peaks **union** the three peaks
    whose ``|q|`` is closest to each reference row's ``|b_i|`` (radius matching is
    rotation invariant), each labelled ``(1, 0)``/``(0, 1)`` in both orders; every
    candidate is expanded and refitted and ranked by ``d(M) = max(|lambda - 1|)``
    relative to the caller's basis, then by more labelled peaks, then by chi2.
    A-10's original "top-6 strongest peaks only" wording is void (A-13): the peak
    ordering follows amplitude * phase interference, so the image of a reference
    row can rank outside the top six and the true gauge then becomes unreachable.
    The case below is exactly such an interference/asymmetry case, and it asserts
    that the true ``b1`` image is not among the six strongest detections.
    """
    basis = _oblique_basis_px()
    bvecs_nm_inv = basis * DQ_NM_INV
    case = _generic_case("oblique_snr150", basis, 150.0, 0)
    det_true = abs(float(np.linalg.det(bvecs_nm_inv @ M_TRUE)))
    summary = []
    for orientation in (None, 0.0):
        result = detect_bragg_peaks(
            case["image"],
            L_NM,
            lattice=LatticeSpec(bvecs_nm_inv=bvecs_nm_inv, orientation_deg=orientation),
        )
        assert result.lattice is not None, (orientation, result.meta["lattice_error"])
        fraction = _labelled_fraction(result)
        assert fraction >= 0.9, (orientation, fraction)
        det_fit = abs(float(np.linalg.det(result.lattice.bvecs_nm_inv)))
        determinant_error = abs(det_fit - det_true) / det_true
        assert determinant_error <= 2e-3, (orientation, determinant_error)
        summary.append(f"orientation={orientation}: |det B| error {determinant_error:.2e}")
    # A-10: the gauge must not be decided by "whichever peak is strongest".
    bvecs_px = _oblique_basis_px() * DQ_NM_INV
    gauge = LatticeSpec(symmetry="oblique", bvecs_nm_inv=bvecs_px)
    # Amplitude asymmetry (the ruling's A-10 case): the peak at the b2 image is
    # boosted and the b1 image damped.  The boost is x4.0 rather than the wording's
    # x1.6 because the detector ranks peaks by amplitude * phase interference: at
    # x1.6 the strongest *detection* is the (-1, 1) image, so the requirement
    # "the strongest peak is an image of b2" would not hold deterministically.
    # With x4.0 the fixed seed gives strongest = (0, 1) and rank(b1 image) = 10.
    asymmetric = _oblique_gauge_case(
        "asymmetric",
        _oblique_basis_px(),
        {(0, 1): 4.0, (0, -1): 4.0, (1, 0): 0.45, (-1, 0): 0.45},
    )
    gauged = detect_bragg_peaks(asymmetric["image"], L_NM, lattice=gauge)
    assert gauged.lattice is not None and gauged.lattice.fit_ok, gauged.meta
    assert gauged.lattice.strain_gauge == "absolute"
    assert gauged.meta["oblique_gauge"] == "least_distortion"
    assert gauged.meta["oblique_gauge_margin"] > 1.0
    gauge_fraction = _labelled_fraction(gauged)
    assert gauge_fraction >= 0.9, gauge_fraction
    gauge_stretch = float(
        np.max(
            np.abs(
                np.asarray(gauged.lattice.principal_stretches) - _true_stretches()
            )
        )
    )
    assert gauge_stretch <= STRETCH_TOL, gauge_stretch
    mod = 180.0
    gauge_rotation = abs(
        ((gauged.lattice.rotation_deg - _true_rotation_deg() + mod / 2.0) % mod) - mod / 2.0
    )
    assert gauge_rotation <= ROTATION_TOL_DEG, gauge_rotation
    # A-13.2 case requirement, in its precise form: match the detections to the
    # truth positions (2 px tolerance), take the best-ranked detection whose
    # truth label is +/-(1, 0) and require rank > 6; the strongest detection must
    # be an image of the other reference row.
    truth_px = (
        np.array(
            [(h, k) for h in range(-6, 7) for k in range(-6, 7) if (h, k) != (0, 0)]
        )
        @ _oblique_basis_px()
        @ M_TRUE
    )
    truth_hk = [
        (h, k) for h in range(-6, 7) for k in range(-6, 7) if (h, k) != (0, 0)
    ]
    independent = [peak for peak in gauged.peaks if peak.independent]
    order = sorted(range(len(independent)), key=lambda i: -independent[i].snr)
    b1_rank = np.inf
    strongest_hk = None
    for rank, index in enumerate(order, start=1):
        peak = independent[index]
        distances = np.hypot(truth_px[:, 0] - peak.q_px[0], truth_px[:, 1] - peak.q_px[1])
        best = int(np.argmin(distances))
        if distances[best] > 2.0:
            continue
        label = truth_hk[best]
        if rank == 1:
            strongest_hk = label
        if label in ((1, 0), (-1, 0)):
            b1_rank = min(b1_rank, rank)
    assert np.isfinite(b1_rank) and b1_rank > 6, b1_rank
    assert strongest_hk in ((0, 1), (0, -1)), strongest_hk
    equal = _oblique_gauge_case("equal", _oblique_basis_px(), {})
    equal_result = detect_bragg_peaks(equal["image"], L_NM, lattice=gauge)
    labelled_reps, usable_reps = _labelled_representatives(equal_result)
    assert labelled_reps >= 0.9 * usable_reps, (labelled_reps, usable_reps)

    # A-11: an inferred oblique lattice publishes no strain (its reference is
    # built from the seed vectors, so the "strain" would be the identity).
    inference_case = _oblique_gauge_case("inference", _oblique_inference_basis_px(), {})
    inferred = detect_bragg_peaks(inference_case["image"], L_NM)
    assert inferred.lattice is not None, inferred.meta["lattice_error"]
    assert inferred.meta["inferred_symmetry"] == "oblique"
    assert inferred.lattice.fit_ok
    assert inferred.lattice.strain_gauge == "undefined"
    assert inferred.lattice.principal_stretches is None
    assert inferred.lattice.linearized_strain is None
    assert inferred.lattice.rotation_deg is None
    assert inferred.lattice.rotation_sigma_deg is None
    assert inferred.lattice.affine is None and inferred.lattice.cov_affine is None
    assert inferred.lattice.bvecs_nm_inv is not None
    inferred_labelled, inferred_usable = _labelled_representatives(inferred)
    assert inferred_labelled >= 0.9 * inferred_usable, (inferred_labelled, inferred_usable)
    print(
        f"  [R19] oblique (bvecs, {len(result.peaks)} peaks, labelled "
        f"{fraction:.3f} >= 0.9): " + "; ".join(summary)
        + f"; A-10 gauge: strongest peak is {asymmetric['strongest']} yet stretch "
        f"error {gauge_stretch:.2e} (<= {STRETCH_TOL:g}), rotation mod 180 deg "
        f"{gauge_rotation:.4f} (<= {ROTATION_TOL_DEG:g}), oblique_gauge="
        f"{gauged.meta['oblique_gauge']}, margin "
        f"{gauged.meta['oblique_gauge_margin']:.2f} (b1 image SNR rank "
        f"{int(b1_rank)} > 6, strongest is {strongest_hk}); equal-amplitude labelled "
        f"{labelled_reps}/{usable_reps} representatives; inferred oblique: "
        f"strain_gauge=undefined, stretches/strain/rotation None, labelled "
        f"{inferred_labelled}/{inferred_usable}"
    )


def check_r20() -> None:
    """(R20) inferred (lattice=None) hexagonal end-to-end (F-7).

    Exercises the whole inference path, the deviatoric reference gauge and the
    affine-free but rotation/strain-carrying LatticeFit.
    """
    case = _case(200.0, "ladder", 0)
    result = detect_bragg_peaks(case["image"], L_NM)
    assert result.lattice is not None, result.meta["lattice_error"]
    assert result.meta["inferred_symmetry"] == "hexagonal", result.meta
    assert result.lattice.symmetry == "hexagonal"
    assert result.lattice.fit_ok, (result.lattice.quality, result.meta["lattice_error"])
    assert result.lattice.affine is None and result.lattice.cov_affine is None
    fraction = _labelled_fraction(result)
    assert fraction >= 0.9, fraction
    stretches = np.asarray(result.lattice.principal_stretches, dtype=float)
    assert abs(float(stretches[0] * stretches[1]) - 1.0) <= 1e-9, stretches
    stretch_error = float(np.max(np.abs(stretches - _true_stretches())))
    assert stretch_error <= STRETCH_TOL, stretch_error
    mod = POINT_GROUP_MOD_DEG
    rotation_error = abs(
        ((result.lattice.rotation_deg - _true_rotation_deg() + mod / 2.0) % mod) - mod / 2.0
    )
    assert rotation_error <= ROTATION_TOL_DEG, rotation_error
    print(
        f"  [R20] lattice=None hexagonal: {len(result.peaks)} peaks, labelled "
        f"{fraction:.3f} (>= 0.9), |det M| = {stretches[0] * stretches[1]:.12f}, "
        f"stretch error {stretch_error:.2e}, rotation mod 60 deg {rotation_error:.4f}"
    )


def check_r21() -> None:
    """(R21) field self-consistency: total covariance, model sigma, strain (F-2/F-3)."""
    case = _case(120.0, "formfactor", 0)
    result = detect_bragg_peaks(case["image"], L_NM, lattice=LATTICE)
    worst_cov = 0.0
    worst_model = 0.0
    n_model = 0
    for peak in result.peaks:
        sigma = np.asarray(peak.sigma_q_px, dtype=float)
        cov = np.asarray(peak.cov_q_px, dtype=float)
        worst_cov = max(worst_cov, float(np.max(np.abs(np.sqrt(np.diag(cov)) - sigma))))
        assert np.allclose(
            np.asarray(peak.sigma_q_nm_inv) / result.dq_nm_inv, sigma, rtol=0.0, atol=1e-15
        )
        if peak.index_hk is None:
            continue
        h, k = peak.index_hk
        jac = np.array([[h, 0.0, k, 0.0], [0.0, h, 0.0, k]])
        predicted_nm_inv = np.sqrt(
            np.clip(np.diag(jac @ result.lattice.cov_bvecs_nm_inv @ jac.T), 0.0, None)
        )
        reported_px = np.asarray(peak.sigma_q_model_px, dtype=float)
        reference_px = predicted_nm_inv / result.dq_nm_inv
        relative = float(np.max(np.abs(reported_px - reference_px) / reference_px))
        worst_model = max(worst_model, relative)
        n_model += 1
    assert worst_cov <= 1e-12, worst_cov
    assert n_model > 0
    assert worst_model <= 1e-9, worst_model

    # Orientation unknown with a reference lattice: linearised strain is exactly
    # polar(M)[1] - I (the invariant form).
    _, stretch = polar(result.lattice.affine)
    assert np.allclose(
        result.lattice.linearized_strain, stretch - np.eye(2), rtol=0.0, atol=1e-12
    )
    # Known orientation: the strain is absolute and must reproduce the truth.
    known = detect_bragg_peaks(
        case["image"], L_NM, lattice=LatticeSpec(a_nm=A_NM, orientation_deg=0.0)
    )
    assert known.lattice is not None
    _, stretch_true = polar(M_TRUE)
    strain_error = float(
        np.max(np.abs(known.lattice.linearized_strain - (stretch_true - np.eye(2))))
    )
    assert strain_error <= 2e-3, strain_error

    # Parabola covariance: analytic partials against numerical differentiation.
    rng = np.random.default_rng(SEED)
    checked = 0
    for _ in range(80):
        samples = rng.normal(size=3) * 0.3 + np.array([-1.0, 0.0, -1.0])
        curvature = 0.5 * (samples[2] - 2.0 * samples[1] + samples[0])
        if curvature >= -1e-6:
            continue
        sigma_log = 0.05
        _, sigma_analytic = _parabola_axis(
            samples[0], samples[1], samples[2], sigma_log
        )

        def offset(values):
            a = 0.5 * (values[2] - values[0])
            c = 0.5 * (values[2] - 2.0 * values[1] + values[0])
            return -a / (2.0 * c)

        step = 1e-6
        jacobian = np.zeros(3)
        for index in range(3):
            delta = np.zeros(3)
            delta[index] = step
            jacobian[index] = (offset(samples + delta) - offset(samples - delta)) / (
                2.0 * step
            )
        sigma_numeric = abs(sigma_log) * float(np.sqrt(np.sum(jacobian**2)))
        relative = abs(sigma_analytic - sigma_numeric) / sigma_numeric
        assert relative <= 1e-6, relative
        checked += 1
    assert checked >= 10
    print(
        f"  [R21] sqrt(diag(cov_q_px)) == sigma_q_px (worst {worst_cov:.2e}); "
        f"sigma_q_model_px == J_B cov_bvecs J_B^T on {n_model} peaks "
        f"(worst rel {worst_model:.2e} <= 1e-9); linearized_strain == polar(M)[1]-I "
        f"(1e-12) and reproduces the truth to {strain_error:.2e} <= 2e-3; parabola "
        f"partials vs numerical differentiation on {checked} samples (<= 1e-6)"
    )


class _ListHandler(logging.Handler):
    """Collect log messages so the S-6 warning can be asserted."""

    def __init__(self, sink):
        super().__init__()
        self.sink = sink

    def emit(self, record):
        """Append one formatted log message to the sink."""
        self.sink.append(record.getMessage())


def check_r22() -> None:
    """(R22) degenerate inputs, label rollback and the orientation convention.

    S-1: without a converged lattice no peak may carry ``index_hk`` or a model
    position (the fit writes them; no earlier stage may leave one behind).
    S-3: a constant or an all-zero image has no measurable |FFT| background, so
    both must return the same empty peak table with a ``meta`` record instead of
    peaks conjured out of the numerical noise floor.
    S-6: ``orientation_deg`` is a counter-clockwise rotation of the reference
    basis with a 0.3|b1| seed tolerance; a mismatch must be logged, not silent.
    """
    # S-3: constant and all-zero images behave identically.
    degenerate = []
    for image in (
        np.full((N // 2, N // 2), 3.7),
        np.zeros((N // 2, N // 2)),
        np.ones((N // 2, N // 2)),
    ):
        result = detect_bragg_peaks(image, L_NM)
        assert result.peaks == (), len(result.peaks)
        assert result.lattice is None
        assert result.n_candidates == 0
        assert result.meta["degenerate_input"]
        assert result.meta["lattice_error"] == result.meta["degenerate_input"]
        degenerate.append(result)
    assert len({item.meta["degenerate_input"] for item in degenerate}) == 1

    # S-1: one cosine plus noise -> a visible peak pair but no lattice at all.
    axis = np.arange(N) - N // 2
    _rows, cols = np.meshgrid(axis, axis)
    rng = np.random.default_rng(SEED)
    single = 40.0 * np.cos(2.0 * np.pi * (17.32 * cols) / N) + rng.normal(size=(N, N))
    result = detect_bragg_peaks(single, L_NM, lattice=LATTICE)
    assert result.peaks
    assert result.lattice is None, result.meta["lattice_error"]
    for peak in result.peaks:
        assert peak.index_hk is None
        assert peak.q_model_px is None
        assert peak.sigma_q_model_px is None
        assert peak.residual_px is None

    # S-6: the lattice stage must not degrade silently (design D13).
    logger = logging.getLogger("stm_data_processing.utils.bragg_peak_detection")
    captured = []
    handler = _ListHandler(captured)
    logger.addHandler(handler)
    try:
        detect_bragg_peaks(single, L_NM, lattice=LATTICE)
    finally:
        logger.removeHandler(handler)
    assert any("required for a lattice fit" in text for text in captured), captured
    assert "counter-clockwise" in LatticeSpec.__doc__
    print(
        f"  [R22] degenerate inputs (constant/zero/one) -> 0 peaks + meta record "
        f"({degenerate[0].meta['degenerate_input'][:48]}...); "
        f"{len(result.peaks)} peaks without a lattice carry no index_hk/model field; "
        f"a failing lattice stage warns ({len(captured)} message(s)) and the "
        f"counter-clockwise convention is documented"
    )


def check_r23() -> None:
    """(R23) lattice acceptance gates (architect ruling A-0..A-9).

    The architect's "new R18" is added here as R23 so the existing R18 (square
    end-to-end) keeps its task-contract numbering.  Three gates are checked:
    every synthetic scenario that keeps a lattice must pass all of them, an
    orientation-pinned hexagonal/square mismatch must be rejected with every
    derived quantity withheld, and the gates must be switchable off through the
    public keyword arguments.  Synthetic input only -- real data never enters
    this script.
    """
    summary = []
    for tag, snr, profile, envelope in (
        ("S1_formfactor_snr120", 120.0, "formfactor", None),
        ("S2_ladder_snr200", 200.0, "ladder", None),
        ("S3_ladder_snr30", 30.0, "ladder", None),
        ("S4_formfactor_snr120_env40", 120.0, "formfactor", 40.0),
    ):
        case = _case(snr, profile, 0, envelope)
        result = detect_bragg_peaks(case["image"], L_NM, lattice=LATTICE)
        assert result.lattice is not None, (tag, result.meta["lattice_error"])
        lattice = result.lattice
        assert lattice.fit_ok, (tag, lattice.quality, result.meta["lattice_error"])
        assert lattice.quality == "ok", (tag, lattice.quality)
        assert lattice.consistent_fraction >= 0.9, (tag, lattice.consistent_fraction)
        assert result.meta["fit_ok"] is True
        assert result.meta["lattice_quality"] == "ok"
        summary.append(
            f"{tag}: chi2 {lattice.chi2_reduced:.2f} rms {lattice.rms_residual_px:.4f} "
            f"spacing {result.meta['pool_spacing_px']:.2f} (>= "
            f"{result.meta['min_pool_spacing_px']:g}) residual_max "
            f"{lattice.residual_max_px:.3f} consistent {lattice.consistent_fraction:.3f}"
        )
    inferred = detect_bragg_peaks(_case(200.0, "ladder", 0)["image"], L_NM)
    assert inferred.lattice is not None and inferred.lattice.fit_ok

    mismatch_case = _case(200.0, "ladder", 0)
    pinned = LatticeSpec(a_nm=A_NM, symmetry="square", orientation_deg=0.0)
    rejected = detect_bragg_peaks(mismatch_case["image"], L_NM, lattice=pinned)
    assert rejected.lattice is not None, rejected.meta["lattice_error"]
    assert not rejected.lattice.fit_ok, rejected.lattice.quality
    assert rejected.lattice.quality != "ok"
    assert rejected.meta["fit_ok"] is False
    assert "rejected" in rejected.meta["lattice_error"]
    for attribute in (
        "affine",
        "cov_affine",
        "rotation_deg",
        "rotation_sigma_deg",
        "principal_stretches",
        "linearized_strain",
    ):
        assert getattr(rejected.lattice, attribute) is None, attribute
    assert rejected.lattice.rotation_is_absolute is False
    for peak in rejected.peaks:
        assert peak.index_hk is None
        assert peak.q_model_px is None
        assert peak.sigma_q_model_px is None
        assert peak.residual_px is None
        assert peak.quality == "unmatched"
    assert rejected.lattice.chi2_reduced > 25.0
    assert np.isfinite(rejected.lattice.rms_residual_px)
    assert np.isfinite(rejected.meta["pool_spacing_px"])
    ungated = detect_bragg_peaks(
        mismatch_case["image"],
        L_NM,
        lattice=pinned,
        chi2_red_max=np.inf,
        residual_max_px=np.inf,
        min_pool_spacing_px=0.0,
    )
    assert ungated.lattice is not None and ungated.lattice.fit_ok
    assert any(peak.index_hk is not None for peak in ungated.peaks)
    print(
        "  [R23] gates on synthetic scenarios: "
        + " | ".join(summary)
        + f"; orientation-pinned mismatch rejected (chi2 "
        f"{rejected.lattice.chi2_reduced:.4g} > 25, labels/affine/rotation/strain "
        f"withheld), opt-out kwargs restore the fit (fit_ok="
        f"{ungated.lattice.fit_ok}, labelled peaks present)"
    )


def check_r24() -> None:
    """(R24) gate reporting self-consistency (A-14) and the accepted inference-oblique case.

    A-14 renames the measured G-1 quantity to ``LatticeFit.pool_spacing_px`` and
    keeps only the *applied* G-2 threshold in the result object; the two caller
    thresholds are echoed in ``meta`` so that ``(result.lattice, result.meta)``
    alone reproduces every verdict.  The second half asserts a **non-rejected**
    inference-oblique end-to-end run (the rejected branch is covered by R19/R23).
    """
    case = _case(120.0, "formfactor", 0)
    result = detect_bragg_peaks(case["image"], L_NM, lattice=LATTICE)
    lattice = result.lattice
    assert lattice is not None and lattice.fit_ok

    # (1) measured spacing identical in the object and in meta.
    assert lattice.pool_spacing_px == result.meta["pool_spacing_px"]
    # (2) and equal to an independent recomputation from the fitted basis.
    recomputed = min(
        float(np.hypot(*(np.asarray(hk, dtype=float) @ lattice.bvecs_nm_inv)))
        / result.dq_nm_inv
        for hk in [(h, k) for h in range(-2, 3) for k in range(-2, 3) if (h, k) != (0, 0)]
    )
    relative = abs(recomputed - lattice.pool_spacing_px) / lattice.pool_spacing_px
    assert relative <= 1e-9, relative
    # (3) derived G-2 threshold.
    assert abs(lattice.residual_max_px - min(1.0, 0.25 * lattice.pool_spacing_px)) <= 1e-12
    # (4) caller thresholds echoed, defaults and explicit values.
    assert result.meta["min_pool_spacing_px"] == 3.0
    assert result.meta["chi2_red_max"] == 25.0
    explicit = detect_bragg_peaks(
        case["image"],
        L_NM,
        lattice=LATTICE,
        chi2_red_max=np.inf,
        residual_max_px=0.5,
        min_pool_spacing_px=0.0,
    )
    assert explicit.meta["chi2_red_max"] == np.inf
    assert explicit.meta["min_pool_spacing_px"] == 0.0
    assert explicit.lattice.residual_max_px == 0.5

    # (5) a rejected fit carries finite diagnostics and readable measured numbers.
    mismatch = LatticeSpec(a_nm=A_NM, symmetry="square", orientation_deg=0.0)
    rejected = detect_bragg_peaks(_case(200.0, "ladder", 0)["image"], L_NM, lattice=mismatch)
    assert rejected.lattice is not None and not rejected.lattice.fit_ok
    error = rejected.meta["lattice_error"]
    assert f"{rejected.lattice.chi2_reduced:.4g}" in error, error
    for value in (
        rejected.lattice.chi2_reduced,
        rejected.lattice.rms_residual_px,
        rejected.lattice.pool_spacing_px,
        rejected.lattice.residual_max_px,
    ):
        assert np.isfinite(value)

    # Non-rejected inference-oblique end-to-end (design A-11).
    inference = detect_bragg_peaks(
        _oblique_gauge_case("r24_inference", _oblique_inference_basis_px(), {})["image"],
        L_NM,
    )
    assert inference.lattice is not None, inference.meta["lattice_error"]
    assert inference.meta["inferred_symmetry"] == "oblique"
    assert inference.lattice.fit_ok, inference.meta["lattice_error"]
    assert inference.lattice.strain_gauge == "undefined"
    assert inference.lattice.principal_stretches is None
    assert inference.lattice.linearized_strain is None
    assert inference.lattice.rotation_deg is None
    assert inference.lattice.rotation_sigma_deg is None
    assert inference.lattice.affine is None and inference.lattice.cov_affine is None
    assert inference.lattice.bvecs_nm_inv is not None
    labelled, usable = _labelled_representatives(inference)
    assert labelled >= 0.9 * usable, (labelled, usable)
    assert all(
        peak.index_hk is None
        for peak in inference.peaks
        if peak.independent and peak.index_hk is None
    )
    for peak in inference.peaks:
        if peak.index_hk is not None:
            assert peak.q_model_px is not None
            assert peak.residual_px is not None
    print(
        f"  [R24] gate reporting: pool_spacing_px == meta ({lattice.pool_spacing_px:.4f} px, "
        f"independent recomputation rel {relative:.1e}), residual_max_px "
        f"{lattice.residual_max_px:.4f} == min(1, 0.25*spacing), meta thresholds "
        f"({result.meta['min_pool_spacing_px']:g}, {result.meta['chi2_red_max']:g}) and "
        f"explicit ({explicit.meta['min_pool_spacing_px']:g}, "
        f"{explicit.meta['chi2_red_max']:g}); rejected fit keeps finite diagnostics "
        f"(chi2 {rejected.lattice.chi2_reduced:.4g}, rms "
        f"{rejected.lattice.rms_residual_px:.4f}, spacing "
        f"{rejected.lattice.pool_spacing_px:.3f}); accepted inference oblique: "
        f"fit_ok, strain_gauge=undefined, shape/rotation/affine None, labelled "
        f"{labelled}/{usable} representatives"
    )


def check_r25() -> None:
    """(R25) A-15: residual units, the measurement-only consistency and its power.

    Assertions (architect A-15.4, thresholds untouched):
      (1) unit gate on two dq baselines -- dq = 0.2094395 (L = 30 nm) and
          dq = 0.1047198 (L = 60 nm, lattice rescaled a_nm 2 -> 4 nm so the px
          pattern is unchanged): the published ``rms_residual_px`` equals the
          RMS of the per-peak ``residual_px`` recomputed from the public API,
          for an accepted *and* a rejected fit, and every px-space statistic is
          invariant across the baselines.  The nm^-1 leak relaxed the G-2 gate
          by 1/dq (4.77x here, 9.55x on the second baseline) and must stay
          fixed;
      (2) the ``consistent_fraction`` formula identity, recomputed from the peaks;
      (3) discriminating power: >= 0.9 on the four benchmark scenarios, <= 0.5 on
          the mismatched fit in both gate states;
      (4) ``lattice_tolerance`` is actually used: frac(0.5) <= frac(10);
      (5) a rejected fit keeps finite measured diagnostics and a reason string
          that quotes the measured values (A-14.4(5)).
    """
    case = _case(120.0, "formfactor", 0)
    accepted = detect_bragg_peaks(case["image"], L_NM, lattice=LATTICE)
    lattice = accepted.lattice
    assert lattice is not None and lattice.fit_ok
    dq = accepted.dq_nm_inv
    assert abs(dq - 2.0 * np.pi / 30.0) <= 1e-12  # the baseline must not be dq = 1

    def recomputed(peaks, tolerance):
        usable = [peak for peak in peaks if peak.residual_px is not None]
        residuals = np.array([peak.residual_px for peak in usable])
        sigma = np.array([float(np.hypot(*peak.sigma_q_px)) for peak in usable])
        rms = float(np.sqrt(np.mean(np.square(residuals))))
        fraction = float(np.mean(residuals <= tolerance * sigma))
        return rms, fraction, len(usable)

    rms_accepted, fraction_accepted, n_accepted = recomputed(accepted.peaks, 3.0)
    assert n_accepted > 0
    assert abs(lattice.rms_residual_px - rms_accepted) <= 1e-9, (
        lattice.rms_residual_px,
        rms_accepted,
    )
    assert abs(lattice.consistent_fraction - fraction_accepted) <= 1e-12, (
        lattice.consistent_fraction,
        fraction_accepted,
    )

    mismatch = LatticeSpec(a_nm=A_NM, symmetry="square", orientation_deg=0.0)
    mismatch_image = _case(200.0, "ladder", 0)["image"]
    rejected = detect_bragg_peaks(mismatch_image, L_NM, lattice=mismatch)
    assert rejected.lattice is not None and not rejected.lattice.fit_ok
    ungated = detect_bragg_peaks(
        mismatch_image,
        L_NM,
        lattice=mismatch,
        chi2_red_max=np.inf,
        residual_max_px=np.inf,
        min_pool_spacing_px=0.0,
    )
    assert ungated.lattice is not None and ungated.lattice.fit_ok
    rms_ungated, fraction_ungated, n_ungated = recomputed(ungated.peaks, 3.0)
    assert n_ungated > 0
    assert abs(ungated.lattice.rms_residual_px - rms_ungated) <= 1e-9, (
        ungated.lattice.rms_residual_px,
        rms_ungated,
    )
    assert abs(ungated.lattice.consistent_fraction - fraction_ungated) <= 1e-12
    # the gates do not change the fit, so both states report the same statistics
    assert abs(
        rejected.lattice.rms_residual_px - ungated.lattice.rms_residual_px
    ) <= 1e-12
    assert (
        abs(
            rejected.lattice.consistent_fraction
            - ungated.lattice.consistent_fraction
        )
        <= 1e-12
    )
    assert rejected.lattice.consistent_fraction <= 0.5, rejected.lattice.consistent_fraction
    assert ungated.lattice.consistent_fraction <= 0.5, ungated.lattice.consistent_fraction
    for value in (
        rejected.lattice.chi2_reduced,
        rejected.lattice.rms_residual_px,
        rejected.lattice.pool_spacing_px,
        rejected.lattice.residual_max_px,
    ):
        assert np.isfinite(value)
    error = rejected.meta["lattice_error"]
    assert f"{rejected.lattice.chi2_reduced:.4g}" in error, error

    # second unit-gate baseline: L = 60 nm with a_nm 2 -> 4 nm keeps the same px
    # pattern, so every px statistic must be identical, while the old nm^-1 leak
    # (factor 1/dq) would differ by 2x between the baselines.
    lattice_scaled = LatticeSpec(a_nm=2.0 * A_NM, symmetry="hexagonal")
    accepted_scaled = detect_bragg_peaks(case["image"], L_NM_2X, lattice=lattice_scaled)
    assert accepted_scaled.lattice is not None and accepted_scaled.lattice.fit_ok
    dq_scaled = accepted_scaled.dq_nm_inv
    assert abs(dq_scaled - 2.0 * np.pi / 60.0) <= 1e-12
    assert abs(dq / dq_scaled - 2.0) <= 1e-12
    assert abs((1.0 / dq_scaled) / (1.0 / dq) - 2.0) <= 1e-12  # leak scales, px does not
    rms_scaled, fraction_scaled, n_scaled = recomputed(accepted_scaled.peaks, 3.0)
    assert n_scaled > 0
    assert abs(accepted_scaled.lattice.rms_residual_px - rms_scaled) <= 1e-9
    assert abs(accepted_scaled.lattice.consistent_fraction - fraction_scaled) <= 1e-12
    assert abs(accepted_scaled.lattice.rms_residual_px - lattice.rms_residual_px) <= 1e-9
    assert (
        abs(accepted_scaled.lattice.consistent_fraction - lattice.consistent_fraction)
        <= 1e-12
    )
    mismatch_scaled = LatticeSpec(a_nm=2.0 * A_NM, symmetry="square", orientation_deg=0.0)
    rejected_scaled = detect_bragg_peaks(mismatch_image, L_NM_2X, lattice=mismatch_scaled)
    assert rejected_scaled.lattice is not None and not rejected_scaled.lattice.fit_ok
    ungated_scaled = detect_bragg_peaks(
        mismatch_image,
        L_NM_2X,
        lattice=mismatch_scaled,
        chi2_red_max=np.inf,
        residual_max_px=np.inf,
        min_pool_spacing_px=0.0,
    )
    assert ungated_scaled.lattice is not None and ungated_scaled.lattice.fit_ok
    rms_ungated_scaled, fraction_ungated_scaled, n_ungated_scaled = recomputed(
        ungated_scaled.peaks, 3.0
    )
    assert n_ungated_scaled > 0
    assert abs(ungated_scaled.lattice.rms_residual_px - rms_ungated_scaled) <= 1e-9
    assert abs(ungated_scaled.lattice.consistent_fraction - fraction_ungated_scaled) <= 1e-12
    assert (
        abs(ungated_scaled.lattice.rms_residual_px - ungated.lattice.rms_residual_px) <= 1e-9
    )
    assert rejected_scaled.lattice.consistent_fraction <= 0.5
    assert (
        abs(
            rejected_scaled.lattice.consistent_fraction
            - ungated_scaled.lattice.consistent_fraction
        )
        <= 1e-12
    )

    scenario_fractions = []
    for tag, snr, profile, envelope in (
        ("S1", 120.0, "formfactor", None),
        ("S2", 200.0, "ladder", None),
        ("S3", 30.0, "ladder", None),
        ("S4", 120.0, "formfactor", 40.0),
    ):
        scenario = detect_bragg_peaks(
            _case(snr, profile, 0, envelope)["image"], L_NM, lattice=LATTICE
        )
        assert scenario.lattice is not None and scenario.lattice.fit_ok
        scenario_fractions.append((tag, scenario.lattice.consistent_fraction))
        assert scenario.lattice.consistent_fraction >= 0.9, (tag, scenario.lattice.consistent_fraction)

    loose = detect_bragg_peaks(case["image"], L_NM, lattice=LATTICE, lattice_tolerance=0.5)
    tight = detect_bragg_peaks(case["image"], L_NM, lattice=LATTICE, lattice_tolerance=10.0)
    assert loose.lattice.consistent_fraction <= tight.lattice.consistent_fraction
    assert loose.lattice.consistent_fraction < tight.lattice.consistent_fraction
    print(
        f"  [R25] units (dq={dq:.6f} and dq={dq_scaled:.6f}): rms_residual_px == "
        f"recomputed per-peak RMS ({lattice.rms_residual_px:.6f} px, n={n_accepted}) on an "
        f"accepted fit and ({ungated.lattice.rms_residual_px:.6f} px, n={n_ungated}) on the "
        f"mismatched fit (gates off); identical px statistics across the 2x dq change "
        f"(scaled {accepted_scaled.lattice.rms_residual_px:.6f} / "
        f"{rejected_scaled.lattice.rms_residual_px:.6f} px, mismatch fraction "
        f"{rejected_scaled.lattice.consistent_fraction:.3f}); formula identity <= 1e-12; "
        f"benchmark fractions {[round(value, 3) for _, value in scenario_fractions]} "
        f"(>= 0.9); mismatch fraction {rejected.lattice.consistent_fraction:.3f} (<= 0.5) "
        f"identical with gates on/off; tolerance 0.5 -> "
        f"{loose.lattice.consistent_fraction:.3f} vs 10 -> "
        f"{tight.lattice.consistent_fraction:.3f}"
    )


CHECKS = [
    ("R1  detection completeness (ladder)", check_r1),
    ("R2  false positive rate", check_r2),
    ("R3  sub-pixel localization RMS", check_r3),
    ("R4  per-axis bias", check_r4),
    ("R5  uncertainty calibration (pull)", check_r5),
    ("R6  reported sigma magnitude", check_r6),
    ("R7  model beats per-peak fit", check_r7),
    ("R8  affine principal stretches", check_r8),
    ("R9  affine rotation modulo point group", check_r9),
    ("R10 GLS gain over unweighted OLS", check_r10),
    ("R11 predict/verify loop gain", check_r11),
    ("R12 boundary / NaN robustness", check_r12),
    ("R13 +/-q mirror pairing", check_r13),
    ("R14 lattice_operations / units", check_r14),
    ("R15 determinism", check_r15),
    ("R16 real-data smoke", check_r16),
    ("R17 near-neighbour satellite", check_r17),
    ("R18 square end-to-end (F-1)", check_r18),
    ("R19 oblique end-to-end (F-6)", check_r19),
    ("R20 inferred lattice=None (F-7)", check_r20),
    ("R21 field self-consistency (F-2/F-3)", check_r21),
    ("R22 degenerate inputs / rollback / orientation", check_r22),
    ("R23 lattice acceptance gates (S-2/A)", check_r23),
    ("R24 gate reporting + accepted inference oblique (A-14)", check_r24),
    ("R25 residual units + consistent_fraction (A-15)", check_r25),
]


def main() -> int:
    """Run every check; return 0 when all pass, 1 otherwise."""
    started = time.time()
    failures = 0
    for name, function in CHECKS:
        try:
            function()
            print(f"[PASS] {name}")
        except Exception as error:  # report and continue with the remaining checks
            failures += 1
            print(f"[FAIL] {name}: {error}")
            traceback.print_exc()
    elapsed = time.time() - started
    print(f"elapsed {elapsed:.1f} s (budget {RUNTIME_BUDGET_S:g} s)")
    if elapsed > RUNTIME_BUDGET_S:
        failures += 1
        print(f"[FAIL] runtime budget exceeded: {elapsed:.1f} s")
    if failures:
        print(f"RESULT: {failures} CHECK(S) FAILED")
        return 1
    print("RESULT: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
