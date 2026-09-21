"""One-command self-test of the skill: the engine identities, the estimator
properties, the gauge layer, the paper-style pairwise analysis, the ring lookup
branches and the figure atlas contract.

Everything checked here is pure mathematics (angles, weights, Fourier identities,
estimator algebra) or a contract of the delivered files.  No physical statement is
made or needed.

    cd /path/to/STM_DataProcessing
    MPLCONFIGDIR=<writable> PYTHONDONTWRITEBYTECODE=1 \
        .venv/bin/python <this script> --stm-lib /path/to/STM_DataProcessing/src

Options:
    --workdir DIR   scratch directory (default: <tmp>/stm-phase-selftest)
    --size N        canvas side of the synthetic images (default 384)
    --stm-lib DIR   STM_DataProcessing src directory (package detection/colormap)
    --quick         skip the end-to-end pipeline and correction stages
    --keep          keep the scratch directory

Exit code 0 = every check passed; each check prints its acceptance threshold next
to the measured value, so a failure is quantified instead of announced.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import phasepipe as pp  # noqa: E402
import phasemath as pm  # noqa: E402

TWO_PI = pm.TWO_PI
SQRT3 = pp.SQRT3
LADDER = (0.0, 120.0, 240.0)
DEFAULT_STM_LIB = "/Users/hunfen/Documents/GitHub/STM_DataProcessing/src"
RESULTS: list[tuple[str, bool, str]] = []
# The forbidden tokens are assembled from fragments so that this file itself
# contains no occurrence of them: the delivered skill is scanned for exactly these
# strings (figure names and titles, plus the two forward-looking documents
# SKILL.md and README.md) and the checker must not be the one file that trips its
# own scan.
FORBIDDEN = ["".join(parts) for parts in
             [("kek", "ule"), ("kek", "ulé"), ("z", "3"), ("k", "3p"), ("b", "z"),
              ("读", "法"), ("布里", "渊"), ("k", " 点"), ("m", "点"),
              ("bri", "llouin")]]
# The tokens of the removed engine, again assembled from fragments (the scanner
# must not match its own scan list).
REMOVED_ENGINE = ["".join(parts) for parts in
                  [("reflection_", "field"), ("demod_", "phase"),
                   ("circle_", "mask"), ("mask_", "radius"), ("--", "pct")]]
PEAK_FIGURE_KINDS = ("amplitude", "theta_map", "theta_dist")
SUMMARY_FIGURE_KINDS = ("theta_hist_summary", "theta_map_summary", "theta_field",
                        "ring_members_qspace")
PAIR_FIGURE_KINDS = ("phase_diff", "amp_diff", "2dhist")
PEAKS_PER_RING = 6
WITHIN_PAIRS = ((0, 1), (2, 3), (4, 5))
FIGURES_PER_RING = PEAKS_PER_RING * len(PEAK_FIGURE_KINDS) + len(SUMMARY_FIGURE_KINDS) \
    + len(WITHIN_PAIRS) * len(PAIR_FIGURE_KINDS)
CROSS_PAIRS = 6
CROSS_FIGURES = CROSS_PAIRS * len(PAIR_FIGURE_KINDS) + 1
TOTAL_FIGURES = 2 * FIGURES_PER_RING + CROSS_FIGURES
FIGURE_PATTERN = re.compile(
    r"^(ring_1x1|ring_r3)_(p[0-5]_(amplitude|theta_map|theta_dist)"
    r"|theta_hist_summary|theta_map_summary|theta_field|ring_members_qspace"
    r"|pair_[0-5]_[0-5]_(phase_diff|amp_diff|2dhist))\.png$"
    r"|^cross_pair_([0-5]_(phase_diff|amp_diff|2dhist)|phase_diff_grid)\.png$")


def check(name, ok, detail, threshold=""):
    RESULTS.append((name, bool(ok), detail))
    flag = "PASS" if ok else "FAIL"
    limit = f"  [threshold {threshold}]" if threshold else ""
    print(f"  [{flag}] {name}: {detail}{limit}")
    return bool(ok)


def section(title):
    print(f"\n== {title} ==")


def ring_vectors(n, radius_frac=0.30, rotation_deg=30.0):
    """The six ``ring_1x1`` and the six ``ring_r3`` wavevectors of a canvas."""
    r1 = radius_frac * n
    angles = np.arange(6) * np.pi / 3.0
    ref = [(r1 * np.cos(a), r1 * np.sin(a)) for a in angles]
    rotate = np.radians(rotation_deg)
    r3 = [(r1 / SQRT3 * np.cos(a + rotate), r1 / SQRT3 * np.sin(a + rotate))
          for a in angles]
    return ref, r3


def ring_dict(vectors):
    return {"radius": float(np.hypot(*vectors[0])),
            "members": [(v[0], v[1], 1.0, 1.0, float(np.hypot(*v))) for v in vectors]}


def plane_wave(topo_shape, vectors, phases_deg, amplitudes=None):
    """Real image: a sum of plane waves at the given vectors with set phases."""
    n = int(topo_shape[0])
    yy, xx = np.mgrid[:n, :n]
    amplitudes = [1.0] * len(vectors) if amplitudes is None else list(amplitudes)
    total = np.zeros((n, n), dtype=float)
    for (qx, qy), phase, amp in zip(vectors, phases_deg, amplitudes):
        total = total + float(amp) * np.cos((TWO_PI / n) * (qx * xx + qy * yy)
                                            + np.radians(float(phase)))
    return total


def circular_mean_deg(values, weights):
    mean, resultant, _total = pm.circ_mean(values, weights)
    return float(np.degrees(mean) % 360.0), float(resultant)


def roll_canvas(image, shift):
    """Roll the canvas by ``shift = (dx, dy)`` pixels (circular wrap)."""
    return np.roll(np.roll(image, int(shift[1]), axis=0), int(shift[0]), axis=1)


def translation_phase(q_px, shift, n):
    """The phase the engine adds to a translated canvas: ``-(2 pi / N) q.delta``."""
    return -(TWO_PI / n) * (float(q_px[0]) * shift[0] + float(q_px[1]) * shift[1])


def periodic_rings():
    """A fully periodic two-ring geometry: every wavevector is an FFT bin.

    ``b1 = (96, 0)`` and ``b2 = (48, 84)`` are 60.3 deg apart with
    ``|b2| / |b1| = 1.0078``, and the ``ring_r3`` members are the thirds
    ``(b1 + b2)/3``, ``(2 b1 - b2)/3`` and ``(b1 - 2 b2)/3`` with their negatives --
    all integers, so ``T`` is exactly periodic on the canvas and a circular roll is
    an exact translation of the analysed field.
    """
    b1 = np.array([96.0, 0.0])
    b2 = np.array([48.0, 84.0])
    ref = [tuple(b1), tuple(b2), tuple(b1 - b2),
           tuple(-b1), tuple(-b2), tuple(b2 - b1)]
    r3 = [tuple((b1 + b2) / 3.0), tuple((2.0 * b1 - b2) / 3.0),
          tuple((b1 - 2.0 * b2) / 3.0)]
    r3 = r3 + [(-v[0], -v[1]) for v in r3]
    return ref, r3


# --------------------------------------------------------------------------- #
# 1. the delivered engine: static contract of the source files
# --------------------------------------------------------------------------- #
def test_engine_source_contract():
    """The delivered scripts must contain no trace of the removed mask engine."""
    section("engine contract of the delivered sources")
    sources = {name: (HERE / name).read_text()
               for name in ("phasepipe.py", "phasemath.py", "atlas.py",
                            "stm_phase_analysis.py")}
    hits = {name: [token for token in REMOVED_ENGINE if token in text]
            for name, text in sources.items()}
    hits = {name: found for name, found in hits.items() if found}
    check("no removed-engine token in any delivered script", not hits,
          f"scanned {len(sources)} scripts for the removed engine tokens, hits: "
          f"{hits if hits else 'none'}", "0 hits")
    code = sources["phasepipe.py"]
    check("every per-reflection field of phasepipe comes from the local-q-map engine",
          "localqmap.demodulate(" in code and "def gaussian_field(" in code
          and "from phasemath import" in code,
          "phasepipe calls localqmap.demodulate inside gaussian_field(...) and "
          "sigma-free arg(psi) is the only phase convention "
          f"('def gaussian_field(' present: {'def gaussian_field(' in code})",
          "localqmap.demodulate used, no legacy engine")
    analysis = sources["stm_phase_analysis.py"]
    check("the analysis CLI carries --lambda-nm and no --pct",
          '"--lambda-nm"' in analysis and '"--pct"' not in analysis,
          f"--lambda-nm present: {'\"--lambda-nm\"' in analysis}, --pct present: "
          f"{'\"--pct\"' in analysis}", "--lambda-nm yes, --pct no")


# --------------------------------------------------------------------------- #
# 2. circular estimators
# --------------------------------------------------------------------------- #
def test_circular_estimators():
    section("circular estimators on circular samples")
    phi = np.radians([10.0, 20.0, 30.0])
    mean, _r, _total = pm.circ_mean(phi)
    check("circ_mean of 10/20/30 deg", abs(np.degrees(mean) - 20.0) < 1e-9,
          f"mean = {np.degrees(mean):.12f} deg", "|mean - 20| < 1e-9 deg")

    rng = np.random.default_rng(7)
    worst = 0.0
    for kappa, size in ((2.0, 500), (0.5, 2000), (8.0, 300), (0.1, 1000)):
        sample = rng.vonmises(0.0, kappa, size) % TWO_PI
        weights = 0.5 + rng.random(size)
        median, _span, _ = pm.circ_median(sample, weights)
        grid = np.linspace(0.0, TWO_PI, 20001)
        values = np.array([np.sum(weights * np.abs(pm.wrap_pm_pi(g - sample)))
                           for g in grid])
        ours = float(np.sum(weights * np.abs(pm.wrap_pm_pi(median - sample))))
        worst = max(worst, ours - float(values.min()))
    check("circ_median is the exact minimiser (4 weighted samples)",
          worst < 1e-9, f"max excess over the brute-force grid minimum = {worst:.3e}",
          "< 1e-9")

    _median, span, _ = pm.circ_median(np.array([0.0, np.pi]))
    check("circ_median flags a non-unique (flat) L1 objective", span > 350.0,
          f"minimiser-set span = {span:.2f} deg for two antipodal equal weights",
          "> 350 deg")

    sigma = np.radians(8.0)
    sample = np.radians(100.0) + sigma * rng.standard_normal(200000)
    stats = pm.weighted_stats(sample, np.ones_like(sample))
    expected = 2.354820045 * np.degrees(sigma)
    check("FWHM of a Gaussian sample reproduces 2.3548 sigma after deconvolution",
          abs(stats["fwhm_deconv_deg"] - expected) < 0.01 * expected,
          f"FWHM = {stats['fwhm_deconv_deg']:.4f} deg vs {expected:.4f} deg; raw "
          f"{stats['fwhm_deg']:.4f} deg", "1 %")
    folded = 2.354820045 * np.sqrt(np.degrees(sigma) ** 2 + 4.0)
    check("the raw FWHM is the sample width folded with the smoothing kernel (1 %)",
          abs(stats["fwhm_deg"] - folded) < 0.01 * folded,
          f"raw {stats['fwhm_deg']:.4f} deg vs {folded:.4f} deg", "1 %")

    sample = np.concatenate([np.radians(20.0) + np.radians(6.0) * rng.standard_normal(50000),
                             np.radians(140.0) + np.radians(6.0) * rng.standard_normal(30000)])
    weights = np.concatenate([np.ones(50000), 0.6 * np.ones(30000)])
    stats = pm.weighted_stats(sample, weights)
    centres = [cluster["centre_deg"] for cluster in stats["clusters"]]
    fractions = [cluster["weight_fraction"] for cluster in stats["clusters"]]
    check("two-component sample: two clusters at the injected centres",
          len(centres) == 2 and abs(centres[0] - 20.0) < 0.5
          and abs(centres[1] - 140.0) < 0.5,
          f"centres = {['%.3f' % value for value in centres]}, weights = "
          f"{['%.4f' % value for value in fractions]}", "0.5 deg / 0.01")


# --------------------------------------------------------------------------- #
# 3. the engine identities
# --------------------------------------------------------------------------- #
def test_engine_identities(n, lambda_nm=3.0, nm_per_px=0.13):
    section("engine identities: window, theta = +phi (no ramp), Friedel, translation")
    image = pp.synth_image(n, 0.30 * n, [{"kind": "full", "amp": 1.0, "phase_deg": 47.0}])
    ref, r3vec = ring_vectors(n)

    worst = 0.0
    for lambda_test in (0.5, 3.0, 10.0):
        for q in (ref[0], r3vec[1], (0.0, n / 4)):
            psi = pp.gaussian_field(image, q, lambda_test, nm_per_px)
            total = np.sum(psi)
            row = np.arange(n, dtype=float)[None, :]
            column = np.arange(n, dtype=float)[:, None]
            exact = complex(np.sum(np.nan_to_num(image) * np.exp(
                -1j * (TWO_PI / n) * (q[0] * row + q[1] * column))))
            worst = max(worst, abs(total - exact) / max(abs(exact), 1e-30))
    check("sum_r psi = sum_r T exp(-i q.r), independent of the window width",
          worst < 1e-9,
          f"max relative deviation = {worst:.3e} over lambda = 0.5/3/10 nm and three "
          f"wavevectors (the window's k = 0 weight is exactly one, so every per-peak "
          f"phase value is the exact whole-canvas sum arg sum_r T exp(-i q.r))",
          "< 1e-9")

    worst = 0.0
    for q in ((0.0, n // 4), (n // 4, 0.0), (n // 4, n // 4)):
        wave = plane_wave((n, n), [q], [23.0])
        psi = pp.gaussian_field(wave, q, lambda_nm, nm_per_px)
        moved = pp.gaussian_field(roll_canvas(wave, (7, -5)), q, lambda_nm, nm_per_px)
        shift = translation_phase(q, (7, -5), n)
        worst = max(worst, float(np.max(np.abs(
            pm.wrap_pm_pi(np.angle(moved) - np.angle(psi) - shift)))))
    check("translation law: rolling the canvas adds exactly -(2pi/N) q.delta",
          worst < 1e-9,
          f"max deviation = {worst:.3e} rad over three plane waves and a (7, -5) px "
          f"roll (the identity psi'(r) = exp(-i q.delta) psi(r - delta) reduces to a "
          f"constant shift wherever the demodulated field is constant, which is the "
          f"case of a single plane wave at exactly q)", "< 1e-9 rad")

    worst = 0.0
    for qx, qy in ((0.0, n // 4), (n // 4, 0.0), (n // 4, n // 4)):
        for phase_deg in (0.0, 33.0, 181.0):
            wave = plane_wave((n, n), [(qx, qy)], [phase_deg])
            psi = pp.gaussian_field(wave, (qx, qy), lambda_nm, nm_per_px)
            measured = np.degrees(np.angle(np.mean(psi))) % 360.0
            worst = max(worst, abs(pm.ang_diff_deg(measured, phase_deg)))
    check("theta = arg psi = +phi on 9 plane waves, with no per-peak constant",
          worst < 0.01, f"max deviation from the injected phase = {worst:.5f} deg",
          "< 0.01 deg")

    qx, qy = n // 4, 0.0
    wave = plane_wave((n, n), [(qx, qy)], [33.0])
    psi = pp.gaussian_field(wave, (qx, qy), lambda_nm, nm_per_px)
    ramp_x = np.angle(psi[0, 1:] * np.conj(psi[0, :-1]))
    check("the delivered phase field carries no q.r ramp",
          float(np.max(np.abs(ramp_x))) < 1e-9,
          f"max phase step between neighbouring pixels = "
          f"{float(np.max(np.abs(ramp_x))):.3e} rad; a carrier of "
          f"2 pi q_x / N = {(TWO_PI / n) * qx:.6f} rad/px would show here", "< 1e-9 rad")

    # the previous release removed the carrier with phi = angle(psi) - (2 pi / N)
    # q.(r - c) around the canvas centre c = N // 2, which adds +pi (q_x + q_y) to
    # the *value* of a reflection (its identity was arg X(q) + (2 pi / N) q.c).
    legacy_shift = np.mod(180.0 * (qx + qy), 360.0)
    legacy_value = np.mod(np.degrees(np.angle(np.fft.fft2(wave)[int(qy) % n, int(qx) % n]))
                          + 180.0 * (qx + qy), 360.0)
    measured = np.degrees(np.angle(np.mean(psi))) % 360.0
    deviation = abs(pm.ang_diff_deg(legacy_value, measured) - legacy_shift)
    check("v2 -> v3: the raw single-peak phase is shifted by -pi (q_x + q_y)",
          deviation < 1e-6,
          f"legacy value {legacy_value:.6f} deg vs delivered {measured:.6f} deg: the "
          f"measured shift is {pm.ang_diff_deg(legacy_value, measured):.6f} deg against "
          f"the predicted pi (q_x + q_y) = {legacy_shift:.6f} deg", "< 1e-6 deg")

    worst = 0.0
    for q in pp.independent_triple(ref) + pp.independent_triple(r3vec):
        plus = pp.gaussian_field(image, q, lambda_nm, nm_per_px)
        minus = pp.gaussian_field(image, (-q[0], -q[1]), lambda_nm, nm_per_px)
        worst = max(worst, float(np.max(np.abs(minus - np.conj(plus))))
                    / float(np.max(np.abs(plus))))
    check("Friedel identity psi(-q) = conj(psi(q)) is exact", worst < 1e-12,
          f"max relative deviation = {worst:.3e} over six Friedel pairs", "< 1e-12")

    worst_scalar, worst_field = 0.0, 0.0
    for phase_deg in (0.0, 33.0, 100.0, 240.0):
        canvas = pp.synth_image(n, 0.30 * n, [{"kind": "full", "amp": 1.0,
                                               "phase_deg": phase_deg}])
        angles, theta = [], np.zeros((n, n))
        for q in pp.independent_triple(r3vec):
            psi = pp.gaussian_field(canvas, q, lambda_nm, nm_per_px)
            field = np.asarray(pp.theta_field(psi))
            angles.append(np.degrees(np.angle(np.mean(psi))) % 360.0)
            theta = theta + field
        scalar = float(np.sum(angles) % 360.0)
        field_mean = np.degrees(np.angle(np.mean(np.exp(1j * np.mod(theta, TWO_PI))))) % 360.0
        expected = 3.0 * phase_deg % 360.0
        worst_scalar = max(worst_scalar, abs(pm.ang_diff_deg(scalar, expected)))
        worst_field = max(worst_field, abs(pm.ang_diff_deg(field_mean, expected)))
    check("three independent phases of one region sum to 3*Phi",
          worst_scalar < 0.5, f"max deviation = {worst_scalar:.4f} deg", "< 0.5 deg")
    check("per-pixel triple product of one region equals 3*Phi",
          worst_field < 0.5, f"max deviation = {worst_field:.4f} deg", "< 0.5 deg")


# --------------------------------------------------------------------------- #
# 4. the gauge layer
# --------------------------------------------------------------------------- #
def test_gauge_layer(n, lambda_nm=3.0, nm_per_px=0.13):
    """Which quantities drift with the image origin, and by how much.

    *exact* -- three plane waves on integer wavevectors, so the analysed field
    contains nothing but the waves.  A circular roll of the canvas is then an exact
    translation: every phase must move by ``-(2 pi / N) q.delta`` and the fitted
    origin by exactly ``delta``, to machine precision.

    *invariance* -- a fully periodic two-ring canvas (every wavevector an FFT bin),
    rolled by the same integer shift.  The delivered rule is that the gauge layer is
    invariant when the origin follows the image (``r0 -> r0 - delta``); the
    unconstrained re-fit is reported as well, because it can land on another branch
    of the wrapped least-squares minimum (the structural degeneracy documented in
    SKILL.md).

    *practical* -- the synthesised, non-periodic canvas the pipeline really sees, so
    the measured drift rate and its residual against the exact law are reported
    together with the quantities that must not move (the per-pixel triple product,
    R, FWHM, cluster count).
    """
    section("gauge layer: drift law of the origin and the invariant quantities")
    shift = (7, -5)
    waves = [(96.0, 0.0, 10.0), (48.0, 83.0, 45.0), (-48.0, 83.0, 200.0)]
    exact = {}
    for tag, roll in (("base", None), ("rolled", shift)):
        canvas = plane_wave((n, n), [(qx, qy) for qx, qy, _p in waves],
                            [p for _qx, _qy, p in waves])
        if roll is not None:
            canvas = roll_canvas(canvas, roll)
        members = [(qx, qy, 1.0, 1.0, float(np.hypot(qx, qy))) for qx, qy, _p in waves]
        records, _fields, _psi = pp.analyse_ring(
            canvas, np.ones((n, n), dtype=bool), ring_dict([(m[0], m[1]) for m in members]),
            lambda_nm, nm_per_px, prefix="w_", peaks_override=members)
        best, _minima = pm.fit_origin([record["q_px"] for record in records],
                                      [np.radians(record["stats"]["phase_ungated"]["mean_deg"])
                                       for record in records], n)
        exact[tag] = {"raw": np.array([record["stats"]["phase_ungated"]["mean_deg"]
                                       for record in records]),
                      "qs": np.array([record["q_px"] for record in records]),
                      "r0": np.asarray(best["r0_px"]),
                      "c_rad": float(best["c_rad"]),
                      "rms": float(best["rms_deg"])}
    drift = pm.wrap_pm_pi(np.radians(exact["rolled"]["raw"] - exact["base"]["raw"]))
    predicted = -((TWO_PI / n) * (exact["base"]["qs"] @ np.array(shift)))
    worst_phase = float(np.max(np.abs(np.degrees(pm.wrap_pm_pi(drift - predicted)))))
    check("exact: a translated plane wave moves its phase by -(2pi/N) q.delta",
          worst_phase < 1e-9,
          f"max residual = {worst_phase:.3e} deg over three waves for a {shift} px "
          f"translation", "< 1e-9 deg")
    model = (TWO_PI / n) * (exact["rolled"]["qs"] @ (exact["base"]["r0"] - np.array(shift))) \
        + exact["base"]["c_rad"]
    worst_model = float(np.max(np.abs(np.degrees(pm.wrap_pm_pi(
        np.radians(exact["rolled"]["raw"]) - model)))))
    check("exact: the base solution with r0 - delta explains the translated phases",
          worst_model < 1e-9 and exact["base"]["rms"] < 1e-6,
          f"max model residual = {worst_model:.3e} deg with r0 - ({shift[0]}, "
          f"{shift[1]}) px (fit rms of the base {exact['base']['rms']:.2e} deg; a fitted "
          f"origin is defined up to a lattice vector of the reference wavevectors, so "
          f"only this shifted solution, not the raw minimum, is comparable)", "< 1e-9 deg")

    # the documented v2 -> v3 constant: adding pi (q_x + q_y) to every raw phase is
    # exactly the model with r0 -> r0 + (N/2, N/2), hence the gauge layer cannot move.
    shifted_raw = np.mod(exact["base"]["raw"] + 180.0
                         * (exact["base"]["qs"][:, 0] + exact["base"]["qs"][:, 1]), 360.0)
    centre = np.array([n / 2.0, n / 2.0])
    base_fixed = [float(np.degrees(pm.gauge_phase(
        np.radians(value), q, exact["base"]["r0"], exact["base"]["c_rad"], n)) % 360.0)
        for value, q in zip(exact["base"]["raw"], exact["base"]["qs"])]
    shifted_fixed = [float(np.degrees(pm.gauge_phase(
        np.radians(value), q, exact["base"]["r0"] + centre, exact["base"]["c_rad"], n))
        % 360.0) for value, q in zip(shifted_raw, exact["base"]["qs"])]
    model_check = float(np.max(np.abs(np.degrees(pm.wrap_pm_pi(
        np.radians(shifted_raw) - ((TWO_PI / n) * (exact["base"]["qs"]
                                                   @ (exact["base"]["r0"] + centre))
                                   + exact["base"]["c_rad"]))))))
    worst_gauge = float(np.max([abs(pm.ang_diff_deg(a, b))
                                for a, b in zip(base_fixed, shifted_fixed)]))
    check("the v2 -> v3 constant is absorbed by r0 -> r0 + (N/2, N/2): the "
          "gauge-fixed phases are identical",
          worst_gauge < 1e-9 and model_check < 1e-9,
          f"gauge-fixed phases agree to {worst_gauge:.3e} deg and the shifted phases "
          f"satisfy the shifted model to {model_check:.3e} deg (pi (q_x + q_y) = "
          f"(2 pi / N) q.(N/2, N/2) exactly, so the shift is an origin change, not a "
          f"change of the observable)", "< 1e-9 deg")

    # ---- periodic two-ring canvas: the invariance block ------------------- #
    ref, r3vec = periodic_rings()
    valid = np.ones((n, n), dtype=bool)
    periodic = (sum(plane_wave((n, n), [v], [40.0]) for v in pp.independent_triple(ref))
                + 0.8 * sum(plane_wave((n, n), [v], [133.0])
                            for v in pp.independent_triple(r3vec)))
    invariance = {}
    for tag, roll in (("base", None), ("moved", shift)):
        canvas = periodic if roll is None else roll_canvas(periodic, roll)
        ref_records, _f, _p = pp.analyse_ring(
            canvas, valid, ring_dict(ref), lambda_nm, nm_per_px, prefix="ref_",
            peaks_override=ring_dict(ref)["members"])
        r3_records, fields, _p = pp.analyse_ring(
            canvas, valid, ring_dict(r3vec), lambda_nm, nm_per_px, prefix="r3_",
            peaks_override=ring_dict(r3vec)["members"])
        best, minima = pm.fit_origin(
            [record["q_px"] for record in ref_records],
            [np.radians(record["stats"]["phase_ungated"]["mean_deg"])
             for record in ref_records], n)
        theta = np.zeros((n, n))
        for record in r3_records:
            theta = theta + fields[record["name"]][1]
        invariance[tag] = {
            "raw_ref": np.array([record["stats"]["phase_ungated"]["mean_deg"]
                                 for record in ref_records]),
            "raw_r3": np.array([record["stats"]["phase_ungated"]["mean_deg"]
                                for record in r3_records]),
            "qs_r3": np.array([record["q_px"] for record in r3_records]),
            "best": best, "minima": minima,
            "theta": float(np.degrees(np.angle(np.mean(np.exp(
                1j * np.mod(theta, TWO_PI))))) % 360.0),
            "R": float(r3_records[0]["stats"]["phase_ungated"]["resultant_R"]),
            "fwhm": float(r3_records[0]["stats"]["phase_gated"]["fwhm_deg"]),
            "clusters": int(r3_records[0]["stats"]["phase_gated"]["n_clusters"]),
        }
    predicted_r3 = -((TWO_PI / n) * (invariance["base"]["qs_r3"] @ np.array(shift)))
    raw_drift = float(np.max(np.abs(np.degrees(pm.wrap_pm_pi(np.radians(
        invariance["moved"]["raw_r3"] - invariance["base"]["raw_r3"]) - predicted_r3)))))
    peak_move = float(np.max(np.abs(np.degrees(pm.wrap_pm_pi(np.radians(
        invariance["moved"]["raw_r3"] - invariance["base"]["raw_r3"]))))))
    check("periodic canvas: every raw phase of both rings moves by the exact law",
          raw_drift < 1e-9,
          f"max residual = {raw_drift:.3e} deg on the six ring_r3 reflections (the "
          f"canvas is periodic, so the roll is an exact translation and there is no "
          f"border residual); the raw peaks themselves move by up to "
          f"{peak_move:.3f} deg", "< 1e-9 deg")
    origin = np.asarray(invariance["base"]["best"]["r0_px"]) - np.array(shift, dtype=float)
    base_fixed = [float(np.degrees(pm.gauge_phase(
        np.radians(value), q, invariance["base"]["best"]["r0_px"],
        invariance["base"]["best"]["c_rad"], n)) % 360.0)
        for value, q in zip(invariance["base"]["raw_r3"], invariance["base"]["qs_r3"])]
    moved_fixed = [float(np.degrees(pm.gauge_phase(
        np.radians(value), q, origin, invariance["base"]["best"]["c_rad"], n)) % 360.0)
        for value, q in zip(invariance["moved"]["raw_r3"], invariance["moved"]["qs_r3"])]
    delta = np.abs(np.asarray(base_fixed) - np.asarray(moved_fixed))
    delta = np.minimum(delta, 360.0 - delta)
    mod120 = np.minimum(delta % 120.0, 120.0 - delta % 120.0)
    check("periodic canvas: with the origin following the image the gauge-fixed "
          "phases of the other ring do not move",
          float(np.max(delta)) < 1e-9 and float(np.max(mod120)) < 1e-9,
          f"max change of the six gauge-fixed ring_r3 phases = {float(np.max(delta)):.3e} "
          f"deg (mod 120: {float(np.max(mod120)):.3e} deg) for a {shift} px origin shift",
          "< 1e-9 deg")
    fitted = np.asarray(invariance["moved"]["best"]["r0_px"])
    branch_gap = float(np.min([np.hypot(*(np.asarray(row["r0_px"]) - origin))
                               for row in invariance["moved"]["minima"]]))
    induced = [float(np.degrees(np.mod((TWO_PI / n)
                                       * float(np.dot(q, fitted - origin)),
                                       TWO_PI)))
               for q in invariance["moved"]["qs_r3"]]
    induced_120 = all(min(abs(np.radians(value)) % np.radians(120.0),
                          np.radians(120.0) - abs(np.radians(value)) % np.radians(120.0))
                      < 1e-6 for value in induced)
    check("the unconstrained re-fit is reported: it returns another branch of the "
          "wrapped minimum (structural degeneracy, not a numerical error)",
          len(invariance["moved"]["minima"]) >= 1,
          f"the re-fit returned {len(invariance['moved']['minima'])} local minimum(a); "
          f"the branch with r0 - delta is among them within {branch_gap:.3e} px; its own "
          f"raw choice would move the ring_r3 phases by "
          f"{[round(value, 3) for value in induced]} deg "
          f"(multiples of 120 deg here: {induced_120}), which is why every r0 report "
          f"carries its branch and the induced-shift table", "reported")
    theta_delta = abs(pm.ang_diff_deg(invariance["moved"]["theta"],
                                      invariance["base"]["theta"]))
    check("periodic canvas: the per-pixel triple product is invariant",
          theta_delta < 0.5,
          f"theta changed by {theta_delta:.4f} deg while the single peaks move by up to "
          f"{peak_move:.3f} deg", "< 0.5 deg")
    for key, label, tolerance in (("R", "concentration R", 0.05),
                                  ("fwhm", "gated FWHM", 0.05),
                                  ("clusters", "cluster count", None)):
        before, after = invariance["base"][key], invariance["moved"][key]
        same = (before == after) if tolerance is None \
            else (abs(before - after) < tolerance)
        check(f"{label} is invariant under an origin shift", same,
              f"{before} vs {after}", "exact" if tolerance is None else f"< {tolerance}")

    # ---- practical case: synthesised, non-periodic canvas ------------------ #
    ref_p, r3_p = ring_vectors(n, 0.229)
    image = pp.synth_image(n, 0.229 * n, [{"kind": "full", "amp": 1.0, "phase_deg": 33.0}])
    measured = {}
    for tag, roll in (("base", None), ("moved", shift)):
        canvas = image if roll is None else roll_canvas(image, roll)
        records, fields, _p = pp.analyse_ring(
            canvas, valid, ring_dict(r3_p), lambda_nm, nm_per_px, prefix="r3_",
            peaks_override=ring_dict(r3_p)["members"])
        ref_records, _f, _p = pp.analyse_ring(
            canvas, valid, ring_dict(ref_p), lambda_nm, nm_per_px, prefix="ref_",
            peaks_override=ring_dict(ref_p)["members"])
        theta = np.zeros((n, n))
        for record in records:
            theta = theta + fields[record["name"]][1]
        measured[tag] = {
            "qs": np.array([record["q_px"] for record in records]),
            "raw": np.array([record["stats"]["phase_ungated"]["mean_deg"]
                             for record in records]),
            "ref_raw": np.array([record["stats"]["phase_ungated"]["mean_deg"]
                                 for record in ref_records]),
            "theta": float(np.degrees(np.angle(np.mean(np.exp(
                1j * np.mod(theta, TWO_PI))))) % 360.0),
            "R": float(records[0]["stats"]["phase_ungated"]["resultant_R"]),
            "fwhm": float(records[0]["stats"]["phase_gated"]["fwhm_deg"]),
            "clusters": int(records[0]["stats"]["phase_gated"]["n_clusters"]),
        }
    drift = pm.wrap_pm_pi(np.radians(measured["moved"]["raw"] - measured["base"]["raw"]))
    predicted = -((TWO_PI / n) * (measured["base"]["qs"] @ np.array(shift)))
    residual = float(np.max(np.abs(np.degrees(pm.wrap_pm_pi(drift - predicted)))))
    rate = float(np.max(np.abs(np.degrees(drift))) / float(np.hypot(*shift)))
    check("practical: the raw phases follow the same law with a measured residual",
          residual < 30.0,
          f"drift rate up to {rate:.3f} deg/px for a {shift} px origin shift; residual "
          f"against the exact law {residual:.3f} deg, produced by the periodic-FFT "
          f"border of a canvas that is not periodic (the engine applies no "
          f"apodisation, so the analysed field is the exact sum over valid pixels)",
          "reported; < 30 deg")
    theta_delta = abs(pm.ang_diff_deg(measured["moved"]["theta"], measured["base"]["theta"]))
    check("practical: the per-pixel triple-product phase drifts far less than a "
          "single peak",
          theta_delta < 0.5,
          f"theta changed by {theta_delta:.4f} deg (drift rate "
          f"{theta_delta / float(np.hypot(*shift)):.5f} deg/px) while the raw single "
          f"peaks move at up to {rate:.3f} deg/px", "< 0.5 deg")
    for key, label, tolerance in (("R", "concentration R", 0.05),
                                  ("fwhm", "gated FWHM", 0.05),
                                  ("clusters", "cluster count", None)):
        before, after = measured["base"][key], measured["moved"][key]
        same = (before == after) if tolerance is None \
            else (abs(before - after) < tolerance)
        check(f"practical: {label} is invariant under an origin shift", same,
              f"{before} vs {after}", "exact" if tolerance is None else f"< {tolerance}")


# --------------------------------------------------------------------------- #
# 5. the pairwise contract: which pairs, and what their fields obey
# --------------------------------------------------------------------------- #
def test_pairwise_contract(n, lambda_nm=3.0, nm_per_px=0.13):
    section("pairwise contract: within/cross selection and the Friedel identity")
    ref, r3vec = ring_vectors(n)
    members_1x1 = pp.order_ring_members(ring_dict(ref)["members"])
    members_r3 = pp.order_ring_members(ring_dict(r3vec)["members"])
    records_1x1 = [{"name": f"ring_1x1_p{i}", "q_px": (float(m[0]), float(m[1])),
                    "radius_px": float(np.hypot(m[0], m[1]))}
                   for i, m in enumerate(members_1x1)]
    records_r3 = [{"name": f"ring_r3_p{i}", "q_px": (float(m[0]), float(m[1])),
                   "radius_px": float(np.hypot(m[0], m[1]))}
                  for i, m in enumerate(members_r3)]

    pairs = pp.within_ring_pairs(records_1x1)
    angles = [float(np.degrees(np.arctan2(r["q_px"][1], r["q_px"][0])) % 360.0)
              for r in records_1x1]
    steps = [abs(pm.ang_diff_deg(angles[j], angles[k])) for j, k in pairs]
    friedel_steps = [abs(pm.ang_diff_deg(angles[i], angles[(i + 3) % 6])) for i in range(6)]
    check("within-ring pairs are (p0,p1), (p2,p3), (p4,p5): 60 deg apart, never Friedel",
          pairs == [(0, 1), (2, 3), (4, 5)]
          and all(abs(step - 60.0) < 1e-9 for step in steps)
          and all(abs(step - 180.0) < 1e-9 for step in friedel_steps),
          f"pairs {pairs}, separations "
          f"{['%.3f' % value for value in steps]} deg; the avoided Friedel pairs "
          f"(p_i, p_(i+3)) are {['%.3f' % value for value in friedel_steps]} deg apart",
          "(0,1),(2,3),(4,5) at 60 deg")

    cross = pp.cross_ring_pairs(records_1x1, records_r3)
    brute = []
    for j, left in enumerate(records_1x1):
        angle_left = np.arctan2(left["q_px"][1], left["q_px"][0])
        distances = [abs(float(pm.wrap_pm_pi(
            angle_left - np.arctan2(right["q_px"][1], right["q_px"][0]))))
            for right in records_r3]
        nearest = min(distances)
        brute.append(distances.index(
            next(value for value in distances if value <= nearest + pp.ANGLE_TIE)))
    gaps = [abs(pm.ang_diff_deg(
        float(np.degrees(np.arctan2(records_1x1[j]["q_px"][1], records_1x1[j]["q_px"][0]))),
        float(np.degrees(np.arctan2(records_r3[k]["q_px"][1], records_r3[k]["q_px"][0])))))
        for j, k in cross]
    chosen = [k for _j, k in cross]
    check("cross pairs: every ring_1x1 peak with the ring_r3 peak closest in polar angle",
          chosen == brute and [j for j, _k in cross] == list(range(len(records_1x1)))
          and len(cross) == CROSS_PAIRS,
          f"{len(cross)} pairs (one per ring_1x1 peak), chosen ring_r3 indices {chosen} "
          f"equal the documented nearest-angle rule {brute} (smallest ring_r3 index on "
          f"an exact tie); angular gaps "
          f"{['%.3f' % value for value in gaps]} deg (the two rings differ by a 30 deg "
          f"rotation here, so the gaps sit near 30 deg on both sides)", f"{CROSS_PAIRS} pairs")

    image = pp.synth_image(n, 0.40 * n, [{"kind": "full", "amp": 1.0, "phase_deg": 47.0}])
    q = (0.40 * n, 0.0)
    plus = pp.gaussian_field(image, q, lambda_nm, nm_per_px)
    minus = pp.gaussian_field(image, (-q[0], -q[1]), lambda_nm, nm_per_px)
    product = plus * minus
    worst_real = float(np.max(np.abs(np.imag(product))) / np.max(np.abs(product)))
    diff = pp.pair_phase_diff_field(plus, minus)
    worst_trivial = float(np.max(np.abs(pm.wrap_pm_pi(diff - 2.0 * np.angle(plus)))))
    amp_diff = pp.pair_amplitude_diff_field(plus, minus)
    check("the Friedel pair (q, -q) is trivial: its pair sum is exactly 0 and its "
          "amplitude difference is identically 0",
          worst_real < 1e-12 and float(np.max(np.abs(amp_diff))) < 1e-15
          and worst_trivial < 1e-12,
          f"arg psi_j + arg psi_k = arg(psi_j psi_k) is real to {worst_real:.3e} "
          f"(relative), max |a_jk| = {float(np.max(np.abs(amp_diff))):.3e}, and "
          f"D_jk = 2 theta_j to {worst_trivial:.3e} rad -- the difference field of a "
          f"Friedel pair is a function of one field, which is why the within-ring "
          f"pairs avoid them", "< 1e-12")

    swapped = pp.pair_phase_diff_field(minus, plus)
    check("swapping the two members negates D and a, so |D| mod pi is unchanged",
          float(np.max(np.abs(pm.wrap_pm_pi(swapped + diff)))) < 1e-12,
          f"max |D_ba + D_ab| = "
          f"{float(np.max(np.abs(pm.wrap_pm_pi(swapped + diff)))):.3e} rad", "< 1e-12 rad")

    mask = np.ones((n, n), dtype=bool)
    counts, x_edges, y_edges, n_valid = pp.pair_histogram(
        diff, amp_diff, mask, pp.pair_weight_field(plus, minus))
    folded = np.mod(np.abs(diff), np.pi)
    check("the pair histogram spans x = |D| mod pi in [0, pi] and y = a in [-1, 1]",
          counts.shape == (180, 100) and n_valid == n * n
          and abs(x_edges[0]) < 1e-15 and abs(x_edges[-1] - np.pi) < 1e-12
          and abs(y_edges[0] + 1.0) < 1e-15 and abs(y_edges[-1] - 1.0) < 1e-15
          and float(np.max(folded)) <= np.pi
          and float(counts.sum()) > 0.0,
          f"counts shape {counts.shape}, {n_valid} effective pixels, x in "
          f"[{x_edges[0]:.3f}, {x_edges[-1]:.6f}], y in [{y_edges[0]:.1f}, "
          f"{y_edges[-1]:.1f}], max |D| mod pi = {float(np.max(folded)):.6f} rad",
          "180 x 100 bins over [0, pi] x [-1, 1]")


# --------------------------------------------------------------------------- #
# 6. injected recovery through the pairwise fields
# --------------------------------------------------------------------------- #
def test_pairwise_recovery(n, lambda_nm=3.0, nm_per_px=0.13):
    """A known phase difference and a known amplitude ratio must come back."""
    section("pairwise injection: phase difference and amplitude difference")
    r1 = 0.40 * n
    q_j = (r1, 0.0)
    q_k = (r1 * np.cos(np.pi / 3.0), r1 * np.sin(np.pi / 3.0))
    valid = np.ones((n, n), dtype=bool)

    recovered = []
    for injected in (0.0, 40.0):
        canvas = plane_wave((n, n), [q_j, q_k], [10.0, 10.0 + injected])
        psi_j = pp.gaussian_field(canvas, q_j, lambda_nm, nm_per_px)
        psi_k = pp.gaussian_field(canvas, q_k, lambda_nm, nm_per_px)
        diff = pp.pair_phase_diff_field(psi_j, psi_k)
        weight = pp.pair_weight_field(psi_j, psi_k)
        mean, resultant = circular_mean_deg(diff[valid], weight[valid])
        recovered.append((mean, resultant))
    deltas = [pm.ang_diff_deg(recovered[1][0], recovered[0][0])]
    check("an injected 40 deg phase difference is recovered as -40 deg (<= 0.5 deg)",
          abs(deltas[0] + 40.0) <= 0.5,
          f"D mean {recovered[0][0]:.6f} deg -> {recovered[1][0]:.6f} deg, shift "
          f"{deltas[0]:.6f} deg (D = arg psi_j - arg psi_k moves by -Delta phi), "
          f"concentration R {recovered[1][1]:.9f}", "|shift + 40| <= 0.5 deg")

    ratios = []
    for amp_j, amp_k in ((1.0, 1.0), (1.5, 1.0), (1.0, 3.0)):
        canvas = plane_wave((n, n), [q_j, q_k], [10.0, 10.0],
                            amplitudes=[amp_j, amp_k])
        psi_j = pp.gaussian_field(canvas, q_j, lambda_nm, nm_per_px)
        psi_k = pp.gaussian_field(canvas, q_k, lambda_nm, nm_per_px)
        amp_diff = pp.pair_amplitude_diff_field(psi_j, psi_k)
        weight = pp.pair_weight_field(psi_j, psi_k)
        median, _fwhm, _top = pm.linear_median_fwhm(amp_diff[valid], weight[valid])
        ratios.append((median, (amp_j - amp_k) / (amp_j + amp_k)))
    worst = max(abs(value - expected) for value, expected in ratios)
    check("the normalized amplitude difference reproduces the injected ratio "
          "(<= 1e-3)",
          worst <= 1e-3,
          "; ".join(f"a_median {value:+.9f} vs injected {expected:+.9f}"
                    for value, expected in ratios) + f"; max deviation {worst:.3e}",
          "<= 1e-3")

    counts, _xe, _ye, n_valid = pp.pair_histogram(
        np.zeros((n, n)), np.full((n, n), ratios[1][0]), valid,
        np.ones((n, n)))
    check("a constant pair sample lands in a single histogram column",
          int(np.count_nonzero(counts)) == 1 and n_valid == n * n,
          f"non-empty bins: {int(np.count_nonzero(counts))} of {counts.size} "
          f"({n_valid} effective pixels, constant D and a)", "1 bin")


# --------------------------------------------------------------------------- #
# 7. ring lookup and its failure branch
# --------------------------------------------------------------------------- #
def test_ring_lookup():
    section("reference ring selection and the 1/sqrt(3) lookup")
    rings = [{"radius": 117.2, "members": [(0, 0, 1.0, 1.0, 117.2)] * 6,
              "total_amplitude": 6.0},
             {"radius": 67.7, "members": [(0, 0, 1.0, 1.0, 67.7)] * 6,
              "total_amplitude": 6.0},
             {"radius": 40.0, "members": [(0, 0, 1.0, 1.0, 40.0)] * 6,
              "total_amplitude": 1.0}]
    choice = pp.choose_rings(rings, anchor="auto")
    check("auto: the strongest ring with a ring at 1/sqrt(3) is the reference",
          choice["status"] == "ok" and abs(choice["ring_1x1"]["radius"] - 117.2) < 1e-9
          and abs(choice["ring_r3"]["radius"] - 67.7) < 1e-9,
          f"ring_1x1 = {choice['ring_1x1']['radius']:.2f} px, ring_r3 = "
          f"{choice['ring_r3']['radius']:.2f} px, ratio {choice['ratio']:.6f}")
    choice = pp.choose_rings(rings, anchor="radius", reference_radius_px=40.0)
    check("explicit reference radius without a 1/sqrt(3) partner -> not found",
          choice["status"] == "r3_not_found" and choice["ring_1x1"]["radius"] == 40.0,
          f"status = {choice['status']}, method = {choice['method']}")
    broken = [{"radius": 117.2, "members": [(0, 0, 1.0, 1.0, 117.2)] * 6,
               "total_amplitude": 6.0},
              {"radius": 55.0, "members": [(0, 0, 1.0, 1.0, 55.0)] * 6,
               "total_amplitude": 6.0}]
    check("no ring at 1/sqrt(3) or sqrt(3) of another ring -> not found, never a guess",
          pp.choose_rings(broken, anchor="auto")["status"] == "r3_not_found",
          "status = r3_not_found")
    inner_r3 = [{"radius": 135.4, "members": [(0, 0, 1.0, 1.0, 135.4)] * 6,
                 "total_amplitude": 6.0},
                {"radius": 234.5, "members": [(0, 0, 1.0, 1.0, 234.5)] * 6,
                 "total_amplitude": 4.0}]
    choice = pp.choose_rings(inner_r3, anchor="auto")
    check("an innermost r3 ring is reported as the ring above it",
          choice["status"] == "ok" and abs(choice["ring_1x1"]["radius"] - 234.5) < 1e-9,
          f"ring_1x1 = {choice['ring_1x1']['radius']:.2f} px, method = "
          f"{choice['method']}")
    order = pp.order_ring_members(ring_dict(
        [(153.6 * np.cos(a + np.radians(30.0)), 153.6 * np.sin(a + np.radians(30.0)))
         for a in np.arange(6) * np.pi / 3.0])["members"])
    angles = [float(np.degrees(np.arctan2(item[1], item[0])) % 360.0) for item in order]
    steps = [(angles[i] - angles[i + 1]) % 360.0 for i in range(len(angles) - 1)]
    check("peak numbering starts at 12 o'clock and runs clockwise",
          abs(angles[0] - 90.0) < 1e-9 and all(abs(step - 60.0) < 1e-9 for step in steps),
          f"p0 at {angles[0]:.3f} deg, then "
          f"{['%.1f' % value for value in angles[1:]]} (steps "
          f"{['%.1f' % value for value in steps]})", "p0 at 90 deg, -60 deg steps")


# --------------------------------------------------------------------------- #
# 8. discrimination boundary of the phase-sum test
# --------------------------------------------------------------------------- #
def phasor(weights, phases_deg):
    total = float(np.sum(weights))
    z = np.sum(np.asarray(weights, dtype=float)
               * np.exp(1j * np.radians(np.asarray(phases_deg, dtype=float))))
    angle = 3.0 * np.degrees(np.angle(z)) % 360.0
    return {"three_phi_bar_deg": float(angle),
            "ladder_distance_deg": float(pm.dist_to_ladder_deg(angle, LADDER)),
            "coherence": float(abs(z) / total)}


def test_multi_component_boundary(n, lambda_nm=3.0, nm_per_px=0.13):
    """The three-phase sum is a value-range test, never a component counter."""
    section("multi-component discrimination boundary (1 / 2 / 3 components)")
    one = phasor([1.0], [0.0])
    check("1 component: the sum sits on the ladder, coherence 1",
          one["ladder_distance_deg"] < 1e-9 and abs(one["coherence"] - 1.0) < 1e-12,
          f"3 Phi_bar = {one['three_phi_bar_deg']:.4f} deg (ladder distance "
          f"{one['ladder_distance_deg']:.4f}), coherence {one['coherence']:.6f}",
          "< 1e-9 deg / 1.0")
    two = phasor([0.5, 0.5], [0.0, 120.0])
    check("2 equal components 120 deg apart: the sum is 60 deg OFF the ladder",
          abs(two["ladder_distance_deg"] - 60.0) < 1e-9,
          f"3 Phi_bar = {two['three_phi_bar_deg']:.4f} deg, ladder distance "
          f"{two['ladder_distance_deg']:.4f} deg, coherence {two['coherence']:.4f}",
          "60 deg")
    three = phasor([1 / 3, 1 / 3, 1 / 3], [0.0, 120.0, 240.0])
    check("3 equal components on the ladder cancel: the phase is undefined",
          three["coherence"] < 1e-12,
          f"coherence = {three['coherence']:.3e} (any phase number is meaningless)",
          "< 1e-12")

    boundary = []
    for minority in (0.005, 0.01, 0.02, 0.03, 0.05, 0.10, 0.20, 0.35, 0.50):
        mixed = phasor([1.0 - minority, minority], [0.0, 120.0])
        boundary.append((minority, mixed["ladder_distance_deg"], mixed["coherence"]))
    crossing = [row for row in boundary if row[1] >= 5.0]
    if crossing:
        index = boundary.index(crossing[0])
        if index == 0:
            interpolated = crossing[0][0]
        else:
            low, high = boundary[index - 1], crossing[0]
            interpolated = low[0] + (high[0] - low[0]) * (5.0 - low[1]) / (high[1] - low[1])
    else:
        interpolated = float("nan")
    check("2 unequal components: the sum resolves a minority above ~3 %",
          0.02 < interpolated < 0.05,
          f"the 5 deg crossing of the ladder distance is at f = {interpolated:.4f}; "
          + "; ".join(f"f={row[0]:.3f} -> {row[1]:.2f} deg, coherence {row[2]:.3f}"
                      for row in boundary[:5]),
          "2 % < f < 5 %")

    # ---- the same statement through the delivered engine ------------------ #
    r1 = 0.458 * n
    vectors = [(r1 / SQRT3 * np.cos(a + np.radians(30.0)),
                r1 / SQRT3 * np.sin(a + np.radians(30.0)))
               for a in np.arange(6) * np.pi / 3.0]
    members = [(v[0], v[1], 1.0, 1.0, float(np.hypot(*v))) for v in vectors]
    valid = np.ones((n, n), dtype=bool)
    rows = []
    for radius_frac in (0.3697, 0.098):
        domains = [{"kind": "full", "amp": 1.0, "phase_deg": 0.0},
                   {"kind": "disk", "amp": 1.0, "phase_deg": 120.0,
                    "centre": (0.5, 0.5), "radius_frac": radius_frac, "edge": 4.0}]
        image = pp.synth_image(n, r1, domains, ref_amp=0.3, ref_phase_deg=40.0)
        truth = pp.true_mixture(domains, n, r1, window=None)
        records, fields, _psi = pp.analyse_ring(
            image, valid, {"radius": r1 / SQRT3, "members": members}, lambda_nm,
            nm_per_px, prefix="r3_", peaks_override=members)
        triple = pp.triple_summary(records, fields, valid, key="phase_ungated",
                                   quantity="mean_deg")
        clusters = sorted({record["stats"]["phase_gated"]["n_clusters"]
                           for record in records})
        deviation = abs(pm.ang_diff_deg(triple["scalar_sum_deg"],
                                        truth["three_phi_bar_area_deg"]))
        rows.append({"weight": float(truth["area_weight_fractions"][1]),
                     "predicted": truth["three_phi_bar_area_deg"],
                     "measured": triple["scalar_sum_deg"],
                     "ladder": triple["scalar_ladder_dist_deg"],
                     "clusters": clusters, "deviation": deviation})
        print(f"    engine: minority area weight f_w = {rows[-1]['weight']:.4f} "
              f"-> predicted 3 Phi_bar = {rows[-1]['predicted']:.4f} deg, measured "
              f"{rows[-1]['measured']:.4f} deg, ladder distance {rows[-1]['ladder']:.4f} "
              f"deg, cluster counts {clusters}")
    worst = max(row["deviation"] for row in rows)
    check("the whole-canvas phase value reproduces the area-weighted phasor prediction",
          worst < 1.0, f"max deviation = {worst:.4f} deg over {len(rows)} two-component "
                       f"configurations (the engine sums T exp(-i q.r) over every valid "
                       f"pixel, so the region weight of a smooth region is its area "
                       f"integral; the residual is the spectral leakage of the other "
                       f"ring into the demodulation window)", "< 1.0 deg")
    check("a ~3 % minority is invisible to the cluster count but visible to the sum",
          rows[1]["clusters"] == [1] and rows[1]["ladder"] > 3.0,
          f"f_w = {rows[1]['weight']:.4f} -> cluster counts {rows[1]['clusters']}, "
          f"ladder distance {rows[1]['ladder']:.2f} deg",
          "1 cluster and > 3 deg off the ladder")
    check("a ~30 % minority shows up in the cluster count as well",
          rows[0]["clusters"] == [2],
          f"f_w = {rows[0]['weight']:.4f} -> cluster counts {rows[0]['clusters']}",
          "2 clusters")


# --------------------------------------------------------------------------- #
# 9. end-to-end pipeline: atlas contract, determinism, the not-found branch
# --------------------------------------------------------------------------- #
def write_synthetic_csv(path, n, radius_frac=0.458, phases=(0.0, 240.0)):
    r1 = radius_frac * n
    domains = [{"kind": "full", "amp": 1.0, "phase_deg": phases[0]}]
    if len(phases) > 1:
        domains.append({"kind": "disk", "amp": 1.0, "phase_deg": phases[1],
                        "centre": (0.5, 0.5), "radius_frac": 0.20, "edge": 4.0})
    image = pp.synth_image(n, r1, domains, ref_amp=1.0, ref_phase_deg=40.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, image, delimiter=",", fmt="%.10e")
    return r1


def run_script(script, arguments, env):
    return subprocess.run([sys.executable, str(HERE / script), *arguments],
                          capture_output=True, text=True, env=env, check=False)


def test_pipeline_contract(n, workdir, stm_lib):
    section("end-to-end pipeline: atlas contract, determinism, not-found branch")
    env = dict(os.environ)
    env["MPLCONFIGDIR"] = str(workdir / ".mplcache")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    (workdir / ".mplcache").mkdir(parents=True, exist_ok=True)
    csv_path = workdir / "pipeline" / "synthetic.csv"
    field_of_view = 50.0
    write_synthetic_csv(csv_path, n)

    runs = {}
    for tag in ("a", "b"):
        outdir = workdir / "pipeline" / f"out_{tag}"
        runs[tag] = (run_script("stm_phase_analysis.py",
                                [str(csv_path), "-o", str(outdir), "-L", str(field_of_view),
                                 "--detector", "builtin", "--stm-lib", stm_lib], env),
                     outdir)
    completed, outdir = runs["a"]
    second_completed, second_outdir = runs["b"]
    check("the second run also exits 0 (checked before its outputs are read)",
          second_completed.returncode == 0,
          f"exit code {second_completed.returncode}"
          + ("" if second_completed.returncode == 0 else
             f"; stderr tail: {second_completed.stderr.strip().splitlines()[-1]}"
             if second_completed.stderr.strip() else ""),
          "exit code 0")
    check("analysis script exits 0 on a two-ring synthetic", completed.returncode == 0,
          f"exit code {completed.returncode}"
          + ("" if completed.returncode == 0 else
             f"; stderr tail: {completed.stderr.strip().splitlines()[-1]}"
             if completed.stderr.strip() else ""),
          "exit code 0")
    if completed.returncode != 0:
        return

    log_text = (outdir / "phase_stats.log").read_text()
    check("the log states the engine and the lambda of the run",
          "gaussian window via local-q-map" in log_text and "--lambda-nm" not in log_text
          and "lambda = 3 nm" in log_text,
          "log carries the engine line: '"
          + next((line for line in log_text.splitlines() if "gaussian window" in line), "")
          + "'", "engine line present")

    removed = run_script("stm_phase_analysis.py",
                         [str(csv_path), "-o", str(workdir / "pipeline" / "out_pct"),
                          "-L", str(field_of_view), "--detector", "builtin", "--pct", "5"],
                         env)
    check("the removed --pct option is rejected by the CLI",
          removed.returncode != 0 and "--pct" in (removed.stderr + removed.stdout),
          f"exit code {removed.returncode}, message "
          f"{'present' if '--pct' in (removed.stderr + removed.stdout) else 'MISSING'}",
          "non-zero exit and an explicit message")

    figures = sorted(outdir.glob("*.png"))
    check(f"atlas contains {FIGURES_PER_RING} figures per ring and {CROSS_FIGURES} "
          f"cross-ring figures ({TOTAL_FIGURES} in total)",
          len(figures) == TOTAL_FIGURES,
          f"{len(figures)} PNG files found", f"{TOTAL_FIGURES}")

    import atlas as at  # imported after MPLCONFIGDIR has been set
    stats_path = outdir / "phase_stats.json"
    manifest_path = outdir / "atlas_manifest.json"
    ok, failures, _lines = at.check_manifest(manifest_path, stats_path=stats_path,
                                             expected_figures=TOTAL_FIGURES,
                                             expected_per_ring=FIGURES_PER_RING,
                                             expected_cross=CROSS_FIGURES,
                                             verbose=False)
    check("atlas manifest audit: files, sizes, PIL, embedded text, numbers = JSON",
          ok, f"{len(failures)} failure(s)" + (f": {failures[:3]}" if failures else ""),
          "0 failures")
    if not ok:
        return

    manifest = json.loads(manifest_path.read_text())
    bad = [entry["file"] for entry in manifest["figures"]
           if not FIGURE_PATTERN.match(entry["file"])]
    check("every figure name follows the documented pattern", not bad,
          f"{len(manifest['figures'])} names checked, off-pattern: {bad[:3]}", "0")
    text = " ".join(entry["file"] + " " + entry["title"] for entry in manifest["figures"])
    hits = [token for token in FORBIDDEN if token.lower() in text.lower()]
    check("no forbidden token in any figure name or title", not hits,
          f"scanned {len(manifest['figures'])} names/titles, hits: {hits}", "0 hits")
    documents = {name: (HERE.parent / name).read_text()
                 for name in ("SKILL.md", "README.md")}
    document_hits = {name: [token for token in FORBIDDEN if token.lower() in text.lower()]
                     for name, text in documents.items()}
    document_hits = {name: found for name, found in document_hits.items() if found}
    check("no forbidden token in the delivered forward-looking documents",
          not document_hits,
          f"scanned SKILL.md and README.md, hits: "
          f"{document_hits if document_hits else 'none'} (CHANGES.md is the historical "
          f"record and is not part of this naming scan)", "0 hits")

    groups = {}
    for entry in manifest["figures"]:
        groups.setdefault(entry["group"], {}).setdefault(entry["kind"], 0)
        groups[entry["group"]][entry["kind"]] += 1
    check("the manifest carries group and kind for every figure, pairwise with a pair key",
          groups.get("ring_1x1") == {"per_peak": 18, "summary": 4, "pairwise": 9}
          and groups.get("ring_r3") == {"per_peak": 18, "summary": 4, "pairwise": 9}
          and groups.get("cross") == {"summary": 1, "pairwise": 18}
          and all(entry.get("pair") for entry in manifest["figures"]
                  if entry["kind"] == "pairwise")
          and all(entry.get("pair") is None for entry in manifest["figures"]
                  if entry["kind"] != "pairwise"),
          f"per group: {groups}", "18/4/9, 18/4/9, 1/18")
    pair_names = sorted({entry["pair"] for entry in manifest["figures"]
                         if entry["kind"] == "pairwise"})
    check("the manifest names all twelve pairs (3 + 3 within, 6 cross)",
          len(pair_names) == 12
          and sum(1 for name in pair_names if "_pair_" in name) == 0,
          f"{len(pair_names)} pair keys: {pair_names}", "12 pair keys")
    check("every pairwise annotation is declared with a path into the pairwise section",
          all(all(path.startswith("pairwise.") for path in entry["paths"].values())
              for entry in manifest["figures"] if entry["kind"] == "pairwise"),
          "all pairwise paths start at 'pairwise.'", "pairwise paths")

    expected_panels = {
        "amplitude": ["amplitude |psi(r)| (log scale)"],
        "theta_map": ["theta(r) map", "amplitude gate mask"],
        "theta_dist": ["theta distribution (gated)",
                       "theta distribution folded mod 120 deg"],
    }
    bad_panels = None
    for entry in manifest["figures"]:
        if entry["kind"] != "per_peak":
            continue
        match = re.match(r"^(?:ring_1x1|ring_r3)_p(\d)_(amplitude|theta_map|theta_dist)\.png$",
                         entry["file"])
        expected = expected_panels.get(match.group(2)) if match else None
        if expected is None or entry.get("panels") != expected:
            bad_panels = (entry["file"], entry.get("panels"))
            break
    check("every per-peak figure declares its panels", bad_panels is None,
          "18 per-peak figures carry the documented panel lists"
          if bad_panels is None else f"unexpected panels: {bad_panels}",
          "amplitude / theta map+mask / distribution+folded")
    summary_panels = {}
    for entry in manifest["figures"]:
        if entry["kind"] == "summary":
            summary_panels.setdefault(entry["group"], {})[entry["file"]] = entry.get("panels")
    check("each ring has four summary figures and the cross group one, all with panels",
          all(len(kinds) == 4 for group, kinds in summary_panels.items()
              if group in ("ring_1x1", "ring_r3"))
          and len(summary_panels.get("cross", {})) == 1
          and all(all(panels for panels in kinds.values())
                  for kinds in summary_panels.values()),
          "per group: "
          + ", ".join(f"{group}: {len(kinds)} figures"
                      for group, kinds in sorted(summary_panels.items())),
          "4 per ring + 1 cross with panels")

    completed_cli = run_script("atlas.py",
                               ["--check", str(manifest_path), "--stats", str(stats_path),
                                "--expected-figures", str(TOTAL_FIGURES),
                                "--expected-per-ring", str(FIGURES_PER_RING),
                                "--expected-cross", str(CROSS_FIGURES)], env)
    check("the delivered atlas checker really runs as a command line tool",
          completed_cli.returncode == 0 and "ATLAS CHECK PASSED" in completed_cli.stdout,
          f"exit code {completed_cli.returncode}, stdout tail: "
          f"{completed_cli.stdout.strip().splitlines()[-1] if completed_cli.stdout else ''}",
          "exit 0 and ATLAS CHECK PASSED")

    check("the manifest declares the histogram reference-line convention",
          manifest.get("reference_lines_deg") == [0.0, 120.0, 240.0]
          and bool(manifest.get("reference_lines_note")),
          f"reference_lines_deg = {manifest.get('reference_lines_deg')}, note = "
          f"'{manifest.get('reference_lines_note')}'", "2 pi k / 3 (0/120/240 deg)")

    per_group = {group: info["figures"] for group, info in manifest["per_group"].items()}
    check(f"the manifest declares {FIGURES_PER_RING} figures per ring and "
          f"{CROSS_FIGURES} cross figures",
          per_group == {"ring_1x1": FIGURES_PER_RING, "ring_r3": FIGURES_PER_RING,
                        "cross": CROSS_FIGURES},
          f"per group: {per_group}",
          f"{{'ring_1x1': {FIGURES_PER_RING}, 'ring_r3': {FIGURES_PER_RING}, "
          f"'cross': {CROSS_FIGURES}}}")

    stats_a = json.loads(stats_path.read_text())
    pairwise = stats_a.get("pairwise", {})
    groups_json = pairwise.get("groups", {})
    counts_ok = (groups_json.get("within_1x1", {}).get("n_pairs") == 3
                 and groups_json.get("within_r3", {}).get("n_pairs") == 3
                 and groups_json.get("cross", {}).get("n_pairs") == 6)
    entry_ok = True
    for group in groups_json.values():
        for entry in group.get("pairs", []):
            import numpy as np
            hist = np.array(entry.get("hist_counts", []))
            if (not {"j", "k", "q_j", "q_k", "phase_diff_mean", "phase_diff_median",
                     "phase_diff_R", "phase_diff_fwhm_deg", "amp_diff_median",
                     "n_valid", "hist_counts"} <= set(entry)):
                entry_ok = False
            if hist.shape != (180, 100) or not np.all(np.asarray(entry["hist_x_edges"]) >= 0.0):
                entry_ok = False
    check("phase_stats.json carries the pairwise section: 3 + 3 within pairs, 6 cross "
          "pairs, each with the D and a statistics and a 180 x 100 hist_counts",
          counts_ok and entry_ok,
          f"within_1x1 {groups_json.get('within_1x1', {}).get('n_pairs')}, within_r3 "
          f"{groups_json.get('within_r3', {}).get('n_pairs')}, cross "
          f"{groups_json.get('cross', {}).get('n_pairs')}; fields complete: {entry_ok}",
          "3 / 3 / 6 with complete entries")

    if second_completed.returncode != 0 or not (second_outdir / "phase_stats.json").is_file():
        check("determinism and the remaining pipeline checks", False,
              "the second run did not produce phase_stats.json; not reading it",
              "run b completes")
        return
    stats_b = json.loads((second_outdir / "phase_stats.json").read_text())

    def masked_bytes(path, directory):
        """The file bytes with its own output directory replaced by a placeholder."""
        return path.read_bytes().replace(str(directory).encode(), b"<outdir>")

    stats_a_bytes = masked_bytes(stats_path, outdir)
    stats_b_bytes = masked_bytes(second_outdir / "phase_stats.json", second_outdir)
    check("two runs give byte-identical atlas_manifest.json",
          manifest_path.read_bytes() == (second_outdir / "atlas_manifest.json").read_bytes(),
          f"the two manifests are byte identical ({len(manifest_path.read_bytes())} bytes "
          f"each; the manifest carries no path of its own)", "byte identical")
    check("two runs give byte-identical phase_stats.json (its own output directory "
          "masked)",
          stats_a_bytes == stats_b_bytes and b"<outdir>" in stats_a_bytes,
          f"the two files are byte identical after replacing each run's own output "
          f"directory ({len(stats_a_bytes)} bytes each; the only difference is the "
          f"'atlas.dir' field, which names that run's directory by construction)",
          "byte identical modulo the output directory")
    for payload in (stats_a, stats_b):
        payload.pop("atlas", None)
    check("two runs give the same JSON numbers",
          json.dumps(stats_a, sort_keys=True) == json.dumps(stats_b, sort_keys=True),
          "phase_stats.json identical (atlas path field masked)", "identical")

    from PIL import Image
    same_pixels, byte_identical, first_difference = True, True, ""
    for picture in figures:
        other = second_outdir / picture.name
        if not other.is_file():
            same_pixels, first_difference = False, f"{picture.name} missing in run b"
            break
        if picture.read_bytes() != other.read_bytes():
            byte_identical = False
        with Image.open(picture) as image_a, Image.open(other) as image_b:
            if image_a.size != image_b.size or not np.array_equal(np.asarray(image_a),
                                                                  np.asarray(image_b)):
                same_pixels, first_difference = False, f"{picture.name} differs"
                break
    check("two runs give identical figure pixels", same_pixels,
          "all figures pixel-identical" if same_pixels else first_difference, "identical")
    check("two runs give byte-identical PNGs (reported, not required)", byte_identical,
          "byte streams identical" if byte_identical
          else "pixel-identical, byte streams differ (metadata only)", "informational")

    hits = []
    for ring in ("ring_1x1", "ring_r3"):
        block = stats_a["rings_analysis"][ring]
        for field, value in (("radius", block["radius_px"]),
                             ("median", block["peaks"][0]["phase_gated_median_deg"]),
                             ("mean", block["peaks"][0]["phase_ungated_mean_deg"])):
            token = f"{value:.4f}"
            hits.append((f"{ring}.{field}", token, token in log_text))
    pair_token = f"{groups_json['cross']['pairs'][0]['phase_diff_mean']:.4f}"
    hits.append(("cross pair D mean", pair_token, pair_token in log_text))
    check("the log carries the same numbers as the JSON",
          all(hit for _name, _token, hit in hits),
          "; ".join(f"{name}={token}:{'found' if hit else 'MISSING'}"
                    for name, token, hit in hits), "all found")

    outdir_missing = workdir / "pipeline" / "out_not_found"
    completed = run_script("stm_phase_analysis.py",
                           [str(csv_path), "-o", str(outdir_missing), "-L",
                            str(field_of_view), "--detector", "builtin",
                            "--stm-lib", stm_lib, "--anchor", "inner"], env)
    log = (outdir_missing / "phase_stats.log").read_text()
    payload = json.loads((outdir_missing / "phase_stats.json").read_text())
    check("missing r3 partner: exit code 2, an explicit message and no figure",
          completed.returncode == 2 and "r3 ring is NOT reported" in log
          and payload["status"] == "r3_not_found"
          and not list(outdir_missing.glob("*.png")),
          f"exit code {completed.returncode}, status {payload['status']}, "
          f"{len(list(outdir_missing.glob('*.png')))} figures, message "
          f"{'present' if 'r3 ring is NOT reported' in log else 'MISSING'}",
          "exit code 2, message present, 0 figures")


def test_field_of_view_from_log(n, workdir, stm_lib):
    """--size-nm-from-log must read the corrected canvas, not the input canvas."""
    section("field of view from a correction log (corrected canvas, not input canvas)")
    env = dict(os.environ)
    env["MPLCONFIGDIR"] = str(workdir / ".mplcache")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    base = workdir / "fov_from_log"
    base.mkdir(parents=True, exist_ok=True)
    csv_path = base / "synthetic.csv"
    write_synthetic_csv(csv_path, n)
    size_nm, gain = 50.0, 1.03
    log_path = base / "correction.log"
    log_path.write_text(
        "# geometry correction through the bragg_peak package (skill version 2.0)\n"
        f"# canvas {n} x {n} px, field of view {size_nm:g} nm ({size_nm / n:.6f} nm/px)\n"
        f"# corrected canvas: {int(round(n * gain))} x {int(round(n * gain))} px, "
        f"field of view {size_nm * gain:.4f} nm ({size_nm / n:.6f} nm/px)\n")
    outdir = base / "out"
    completed = run_script("stm_phase_analysis.py",
                           [str(csv_path), "-o", str(outdir), "--size-nm-from-log",
                            str(log_path), "--detector", "builtin", "--stm-lib", stm_lib,
                            "--no-figures"], env)
    log = (outdir / "phase_stats.log").read_text() if (outdir / "phase_stats.log").is_file() else ""
    payload = (json.loads((outdir / "phase_stats.json").read_text())
               if (outdir / "phase_stats.json").is_file() else {})
    check("--size-nm-from-log takes the corrected canvas line, not the input canvas one",
          completed.returncode == 0
          and payload.get("field_of_view_nm") == size_nm * gain
          and f"field of view {size_nm * gain:g} nm" in log
          and "corrected canvas line" in (payload.get("field_of_view_source") or ""),
          f"exit code {completed.returncode}, field_of_view_nm = "
          f"{payload.get('field_of_view_nm')} (input canvas {size_nm:g} nm, corrected "
          f"canvas {size_nm * gain:.4f} nm), log header "
          f"{'has' if f'field of view {size_nm * gain:g} nm' in log else 'MISSING'} "
          f"the corrected value",
          f"{size_nm * gain:.4f} nm from the corrected canvas line")
    plain = base / "plain.log"
    plain.write_text(f"# canvas {n} x {n} px, field of view {size_nm:g} nm\n")
    outdir_plain = base / "out_plain"
    completed = run_script("stm_phase_analysis.py",
                           [str(csv_path), "-o", str(outdir_plain), "--size-nm-from-log",
                            str(plain), "--detector", "builtin", "--stm-lib", stm_lib,
                            "--no-figures"], env)
    payload_plain = (json.loads((outdir_plain / "phase_stats.json").read_text())
                     if (outdir_plain / "phase_stats.json").is_file() else {})
    check("--size-nm-from-log falls back to the last 'field of view' match",
          completed.returncode == 0
          and payload_plain.get("field_of_view_nm") == size_nm,
          f"exit code {completed.returncode}, field_of_view_nm = "
          f"{payload_plain.get('field_of_view_nm')} for a log whose only line is "
          f"{size_nm:g} nm", f"{size_nm:.4f} nm")
    empty = base / "empty.log"
    empty.write_text("# no field of view line at all\n")
    outdir_empty = base / "out_empty"
    completed = run_script("stm_phase_analysis.py",
                           [str(csv_path), "-o", str(outdir_empty), "--size-nm-from-log",
                            str(empty), "--detector", "builtin", "--stm-lib", stm_lib,
                            "--no-figures"], env)
    text = completed.stderr + completed.stdout
    needle = "no 'field of view"
    check("--size-nm-from-log without any match is an explicit error",
          completed.returncode != 0 and needle in text,
          f"exit code {completed.returncode}, message "
          f"{'present' if needle in text else 'MISSING'}",
          "non-zero exit with an explicit message")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workdir", default=None,
                        help="scratch directory (default: <tmp>/stm-phase-selftest)")
    parser.add_argument("--size", type=int, default=384,
                        help="canvas side of the synthetic images (default 384)")
    parser.add_argument("--stm-lib", default=DEFAULT_STM_LIB,
                        help="STM_DataProcessing src directory (package detection and "
                             "the gwyddion colormap)")
    parser.add_argument("--lambda-nm", type=float, default=3.0,
                        help="Gaussian window width used by the analytic stages "
                             "(default 3.0 nm, the engine default)")
    parser.add_argument("--nm-per-px", type=float, default=0.13,
                        help="pixel size used by the analytic stages (default 0.13 nm, "
                             "about the 50 nm / 384 px of the scratch canvas)")
    parser.add_argument("--quick", action="store_true",
                        help="skip the end-to-end pipeline and correction stages")
    parser.add_argument("--keep", action="store_true",
                        help="keep the scratch directory")
    args = parser.parse_args(argv)

    workdir = (Path(args.workdir) if args.workdir
               else Path(tempfile.gettempdir()) / "stm-phase-selftest")
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(workdir / ".mplcache"))

    print("# selftest.py - analytic checks of the demodulation engine, the Fourier")
    print("# identities it satisfies, the gauge layer, the paper-style pairwise")
    print("# analysis, the ring lookup branches and the atlas contract of the skill")
    print(f"# scratch directory: {workdir}")

    test_engine_source_contract()
    test_circular_estimators()
    test_engine_identities(args.size, args.lambda_nm, args.nm_per_px)
    test_gauge_layer(args.size, args.lambda_nm, args.nm_per_px)
    test_pairwise_contract(args.size, args.lambda_nm, args.nm_per_px)
    test_pairwise_recovery(args.size, args.lambda_nm, args.nm_per_px)
    test_ring_lookup()
    test_multi_component_boundary(args.size, args.lambda_nm, args.nm_per_px)
    if args.quick:
        check("end-to-end pipeline and correction stages skipped (--quick)", True,
              "informational")
    else:
        test_pipeline_contract(args.size, workdir, args.stm_lib)
        test_field_of_view_from_log(args.size, workdir, args.stm_lib)

    failed = [name for name, ok, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    if failed:
        print("FAILED: " + ", ".join(failed))
        return 1
    if not args.keep and not args.workdir:
        shutil.rmtree(workdir, ignore_errors=True)
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
