"""One-command self-test of the local-q-map skill.

The synthetic is a hexagonal topograph (ring_1x1 and ring_r3 members, each with
its own amplitude and modulation phase) built directly as a sum of plane waves,
so every ground truth is known exactly and no other skill is needed.  All checks
are geometry and signal processing: the amplitude/phase convention of the
demodulated field, the three exact identities, the basis sources and the file
contract.  No physical statement is made anywhere.

The synthetic radius is an integer number of FFT pixels
(``ring_radius_px``, about 0.39 of the canvas side and always even, so it scales
with ``--size`` and stays below the Nyquist radius).  Every ``{b1, b2}`` member
then has ``2 q_px_x`` integral, so its negative frequency copy falls exactly on an
FFT grid column and is exactly absent from the window neighbourhood: the
demodulated field of a single member equals ``(A / 2) exp(+i phi)`` to about 1e-14,
and only the +/-(1/3, -2/3) pair, whose copy lands on the ``k_x = 0`` column, carries
the finite-canvas sampling term at the ~1e-3 level.  What the 12-q checks measure is
therefore mostly the leak of the OTHER five modulations through the window: a
GLOBAL statistic of ~1e-6 (amplitude median) and 0.06 deg (mean phase), plus a
PER-PIXEL value of about ``18.6 / N`` (3.6e-2 at N = 512).  Check 2 asserts both,
because a per-pixel user (phase-difference maps, triple products, per-pixel phase
statistics) must quote the per-pixel floor, never the median one.

    cd /path/to/STM_DataProcessing
    MPLCONFIGDIR=<writable> PYTHONDONTWRITEBYTECODE=1 \
        .venv/bin/python <this script>

Options:
    --workdir DIR   scratch directory (default: a fresh directory under <tmp>)
    --size N        canvas side of the synthetic image (default 512)
    --keep          keep the scratch directory
    --stm-lib DIR   STM_DataProcessing src directory used by the CLI scripts

Exit code 0 = every check passed; each check prints its acceptance threshold next
to the measured value, so a failure is quantified instead of announced.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import localqmap as lq  # noqa: E402

STM_LIB_DEFAULT = "/Users/hunfen/Documents/GitHub/STM_DataProcessing/src"
FIELD_NM = 50.0
LAMBDA_NM = 3.0
# The ring radius in FFT pixels: an integer, so (1, 0) is commensurate.
RING_RADIUS_FRACTION = 0.195


def ring_radius_px(n_px):
    """Radius of the synthetic ring_1x1 in FFT pixels: even and commensurate."""
    return 2.0 * float(round(RING_RADIUS_FRACTION * float(n_px)))
AMPLITUDE_REL_TOL = 1e-3          # check 1: amplitude ~ A / 2
THETA_TOL_DEG = 1e-2              # check 1: theta ~ +phi0
GAUGE_TOL_DEG = 1e-2              # check 5: theta -> theta + phi0
TRANSLATION_REL_TOL = 1e-9        # check 5: maps translate with the image
FRIEDEL_TOL_DEG = 1e-6            # check 6
MEAN_IDENTITY_TOL_DEG = 1e-9      # check 7
NAN_SPECTRUM_REL_TOL = 1e-7       # check 8
NAN_HALO_WINDOWS = 8.0            # check 8: sample the halo at 8 Lambda
MULTI_PEAK_REL_TOL = 1e-3         # check 2: global amplitude of each q
MULTI_PEAK_THETA_TOL_DEG = 0.5    # check 2: global phase of each q
CROSS_TALK_TOL = 2e-2             # check 2: 12-q field against the single modulation
# check 2: the PER-PIXEL floor of the same maps.  Measured worst pointwise leakage
# is 18.6 / N at N = 512 (3.63e-2) and 10.3 / N ... 21.0 / N between N = 128 and
# N = 1024, so the bound is the default-canvas value 5e-2 with the 24 / N scaling
# kept for other sizes: it is the quantity a per-pixel user must quote, not the
# global median statistics above.
RIPPLE_REL_TOL = 5e-2
RIPPLE_NS_CEILING = 24.0
BASIS_PX_REL_TOL = 1e-12          # check 3
RING_1X1 = [(1.0, 0.0), (0.0, 1.0), (1.0, -1.0)]
RING_R3 = [(1.0 / 3.0, 1.0 / 3.0), (2.0 / 3.0, -1.0 / 3.0), (1.0 / 3.0, -2.0 / 3.0)]
# A real image carries one modulation per Friedel pair, so the 12 q's of the
# synthetic are six independent (h, k) and their six negatives; the phase at -q is
# exactly minus the phase at +q.  The translation law of check 5 is exact for the
# sampled array only when the carrier is periodic on the canvas, which forces a
# commensurate (integer fftshift offset) basis there.
LINE_MEMBERS = list(RING_1X1) + list(RING_R3)
COMMENSURATE_BASIS_PX = ((200.0, 0.0), (0.0, 200.0))
COMMENSURATE_MEMBERS = [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, -1.0)]
PEAK_AMPLITUDE_START = 0.90
PEAK_AMPLITUDE_STEP = 0.05
PEAK_PHASE_STEP_DEG = 11.0
PEAK_PHASE_START_DEG = 7.0

SECTIONS: list[dict] = []


def section(title):
    SECTIONS.append({"title": title, "checks": []})
    print(f"\n== {title} ==")


def check(name, ok, detail, threshold=""):
    SECTIONS[-1]["checks"].append((name, bool(ok), detail))
    flag = "PASS" if ok else "FAIL"
    limit = f"  [threshold {threshold}]" if threshold else ""
    print(f"  [{flag}] {name}: {detail}{limit}")
    return bool(ok)


def run_script(script, arguments, env):
    return subprocess.run([sys.executable, str(HERE / script), *arguments],
                          capture_output=True, text=True, env=env, check=False)


# --------------------------------------------------------------------------- #
# synthetic ground truth (independent of any other skill)
# --------------------------------------------------------------------------- #
def synthetic_basis(n_px, radius_px=None):
    """(2, 2) rows b1, b2 in rad/px of the ring_1x1 hexagon."""
    radius = ring_radius_px(n_px) if radius_px is None else float(radius_px)
    return radius * lq.hexagonal_unit() * (2.0 * np.pi / float(n_px))


def coordinates(n_px):
    columns = np.arange(n_px, dtype=float)[None, :]
    rows = np.arange(n_px, dtype=float)[:, None]
    return columns, rows


def peak_table():
    """The six modulations: three ring_1x1 members and three ring_r3 members.

    Each carries its own amplitude and its own modulation phase; the twelve q's of
    the synthetic are these six and their six negatives, and a real image relates
    the two by the Friedel identity (the phase at -q is minus the phase at +q).
    """
    table = []
    for index, (h, k) in enumerate(LINE_MEMBERS):
        amplitude = PEAK_AMPLITUDE_START - PEAK_AMPLITUDE_STEP * index
        phase = np.radians(PEAK_PHASE_START_DEG + PEAK_PHASE_STEP_DEG * index)
        table.append({"h": h, "k": k, "amplitude": amplitude, "phase": phase})
    return table


def plane_wave(columns, rows, q_rad_px, amplitude, phase):
    return amplitude * np.cos(q_rad_px[0] * columns + q_rad_px[1] * rows + phase)


def synthetic_image(columns, rows, basis, table, keep=None):
    image = np.zeros((rows.size, columns.size), dtype=float)
    for index, peak in enumerate(table):
        if keep is not None and index != keep:
            continue
        q = lq.q_vector(peak["h"], peak["k"], basis[0], basis[1])
        image = image + plane_wave(columns, rows, q, peak["amplitude"], peak["phase"])
    return image


def lf_report(basis_rad_px, n_px, size_nm, lambda_nm=30.0):
    """A lawler-fujita-correction style report carrying a 120 degree pair."""
    per_px = size_nm / n_px
    b1 = basis_rad_px[0] / per_px
    b2 = basis_rad_px[1] / per_px
    q_b = b2 - b1
    return {
        "skill": "lawler-fujita-correction", "skill_version": "1.0",
        "canvas_px": n_px, "field_of_view_nm": size_nm, "nm_per_px": per_px,
        "q_a_px": list(lq.rad_px_to_px_offsets(b1, n_px)),
        "q_b_px": list(lq.rad_px_to_px_offsets(q_b, n_px)),
        "q_c_px": list(lq.rad_px_to_px_offsets(-(b1 + q_b), n_px)),
        "q_a_nm_inv": [float(value) for value in b1],
        "q_b_nm_inv": [float(value) for value in q_b],
        "q_c_nm_inv": [float(value) for value in -(b1 + q_b)],
        "lockin": {"lambda_nm": float(lambda_nm)},
        "corrected_field_of_view_nm": float(size_nm),
        "corrected_nm_per_px": float(per_px),
    }


def affine_report(basis_rad_px, n_px, size_nm, measured=True, ideal=False,
                  orientation_offset_deg=0.0):
    """An affine-correction style report carrying orientation plus a radius.

    ``orientation_offset_deg`` rotates the report's ``orientation_deg`` away from
    the canvas' own ring_1x1 direction: that is the real shape of an affine report,
    whose orientation describes the INPUT frame of the correction.
    """
    per_px = size_nm / n_px
    b1 = basis_rad_px[0] / per_px
    radius = float(np.hypot(*b1))
    orientation = (float(np.degrees(np.arctan2(b1[1], b1[0])))
                   + float(orientation_offset_deg))
    a_nm = 2.0 * size_nm / (np.sqrt(3.0) * ring_radius_px(n_px))
    payload = {
        "skill": "topo-correction", "skill_version": "1.0",
        "canvas_px": n_px, "field_of_view_nm": size_nm, "nm_per_px": per_px,
        "orientation_deg": orientation,
        "a_nm": a_nm, "a_ref_nm": a_nm,
        "b1_measured_nm_inv_after": radius if measured else None,
        "b1_ideal_nm_inv": radius if ideal else None,
        "implied_lattice_after": {"a_ref_nm": a_nm, "a_1x1_nm": a_nm},
        "corrected_field_of_view_nm": float(size_nm),
        "corrected_nm_per_px": float(per_px),
    }
    return payload


def affine_r3_report(basis_rad_px, n_px, size_nm, labelled=True):
    """An affine-correction report whose radius keys are the anchored ring_r3.

    ``b1_measured_nm_inv_after`` is the r3 radius (1/sqrt(3) of the ring_1x1
    radius) and ``implied_lattice_after.a_ref_nm`` is sqrt(3) times ``a_1x1_nm``,
    exactly as a real report anchored on r3 carries them.
    """
    per_px = size_nm / n_px
    b1 = basis_rad_px[0] / per_px
    radius_1x1 = float(np.hypot(*b1))
    radius_r3 = radius_1x1 / np.sqrt(3.0)
    orientation = float(np.degrees(np.arctan2(b1[1], b1[0])))
    a_1x1 = 4.0 * np.pi / (np.sqrt(3.0) * radius_1x1)
    return {
        "skill": "stm-topo-phase-analysis", "skill_version": "2.0",
        "canvas_px": n_px, "field_of_view_nm": size_nm, "nm_per_px": per_px,
        "anchor_ring": "r3" if labelled else None,
        "a_nm": 0.246, "a_ref_nm": float(np.sqrt(3.0) * a_1x1),
        "orientation_deg": orientation,
        "b1_measured_nm_inv_after": radius_r3,
        "b1_ideal_nm_inv": radius_r3,
        "implied_lattice_after": {"a_ref_nm": float(np.sqrt(3.0) * a_1x1),
                                  "a_1x1_nm": a_1x1},
        "rings_after": [
            {"radius_px": float(ring_radius_px(n_px)), "radius_nm_inv": radius_1x1,
             "n_members": 12, "total_amplitude": 1.0},
            {"radius_px": float(ring_radius_px(n_px)) / np.sqrt(3.0),
             "radius_nm_inv": radius_r3, "n_members": 6, "total_amplitude": 0.3},
        ],
        "corrected_field_of_view_nm": float(size_nm),
        "corrected_nm_per_px": float(per_px),
    }


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2) + "\n")


def read_json(path):
    return json.loads(Path(path).read_text())


def rel_error(measured, expected):
    scale = abs(float(expected))
    if scale == 0.0:
        return abs(float(measured))
    return abs(float(measured) - float(expected)) / scale


# --------------------------------------------------------------------------- #
# 1. single plane wave: the sign convention of theta
# --------------------------------------------------------------------------- #
def check_single_plane_wave(basis, n_px, nm_per_px, columns, rows):
    section("1. single plane wave (amplitude A/2, theta = +phi0)")
    table = peak_table()
    peak = table[0]                                   # (h, k) = (1, 0)
    q = lq.q_vector(peak["h"], peak["k"], basis[0], basis[1])
    phase = np.radians(37.0)
    image = plane_wave(columns, rows, q, 1.0, phase)
    field = lq.demodulate(image, q, LAMBDA_NM, nm_per_px)
    amplitude = float(np.median(np.abs(field)))
    theta = np.angle(field)
    mean_theta = float(np.angle(np.mean(np.exp(1j * theta))))
    spread_deg = float(np.degrees(theta.max() - theta.min()))
    q_px = lq.rad_px_to_px_offsets(q, n_px)
    check("single plane wave: amplitude is A/2",
          rel_error(amplitude, 0.5) <= AMPLITUDE_REL_TOL,
          f"median |psi| = {amplitude:.12f} vs A/2 = 0.5 "
          f"(relative {rel_error(amplitude, 0.5):.3e})",
          f"relative {AMPLITUDE_REL_TOL:g}")
    check("single plane wave: theta equals +phi0 with the demodulated sign",
          abs(np.degrees(lq.wrap_pm_pi(mean_theta - phase))) <= THETA_TOL_DEG,
          f"theta = {np.degrees(mean_theta):.12f} deg vs +phi0 = "
          f"{np.degrees(phase):.6f} deg (residual "
          f"{np.degrees(lq.wrap_pm_pi(mean_theta - phase)):.3e} deg, map spread "
          f"{spread_deg:.3e} deg, q_px = ({q_px[0]:.3f}, {q_px[1]:.3f}))",
          f"{THETA_TOL_DEG:g} deg")
    check("single plane wave: theta is wrapped inside (-pi, pi]",
          bool(np.all(theta > -np.pi - 1e-12) and np.all(theta <= np.pi + 1e-12)),
          f"min {theta.min():.6f} rad, max {theta.max():.6f} rad",
          "(-pi, pi]")


# --------------------------------------------------------------------------- #
# 2. the 12-peak synthetic: own amplitude and phase, and cross-talk
# --------------------------------------------------------------------------- #
def check_multi_peak(basis, n_px, nm_per_px, columns, rows, lambda_px_value):
    section("2. the 12 q of the synthetic (ring_1x1 + ring_r3): own A and phi, "
            "cross-talk")
    table = peak_table()
    image = synthetic_image(columns, rows, basis, table)
    worst_amplitude, worst_theta, worst_dc, worst_ripple = 0.0, 0.0, 0.0, 0.0
    worst_label = ""
    separations = []
    for peak in table:
        q_line = lq.q_vector(peak["h"], peak["k"], basis[0], basis[1])
        alone_image = plane_wave(columns, rows, q_line, peak["amplitude"], peak["phase"])
        for sign in (1.0, -1.0):
            q = sign * q_line
            field = lq.demodulate(image, q, LAMBDA_NM, nm_per_px)
            alone = lq.demodulate(alone_image, q, LAMBDA_NM, nm_per_px)
            amplitude = float(np.median(np.abs(field)))
            theta = float(np.angle(np.mean(field)))
            error_amplitude = rel_error(amplitude, peak["amplitude"] / 2.0)
            error_theta = abs(np.degrees(lq.wrap_pm_pi(theta - sign * peak["phase"])))
            dc_error = abs(complex(np.mean(field)) - complex(np.mean(alone))) \
                / abs(complex(np.mean(alone)))
            ripple = float(np.max(np.abs(field - alone))) \
                / float(np.median(np.abs(alone)))
            if error_amplitude > worst_amplitude:
                worst_amplitude = error_amplitude
                worst_label = (lq.format_hk(peak["h"], peak["k"])
                               + (" resp. its Friedel partner" if sign < 0 else ""))
            worst_theta = max(worst_theta, error_theta)
            worst_dc = max(worst_dc, float(dc_error))
            worst_ripple = max(worst_ripple, ripple)
        for other in table:
            if other is peak:
                continue
            for sign in (1.0, -1.0):
                delta = (q_line - sign * lq.q_vector(other["h"], other["k"], basis[0],
                                                     basis[1])) / nm_per_px
                separations.append(float(np.hypot(*delta)))
    smallest = min(separations)
    analytic = float(np.exp(-0.5 * (LAMBDA_NM * smallest) ** 2))
    check("every one of the 12 q recovers its own amplitude A / 2",
          worst_amplitude <= MULTI_PEAK_REL_TOL,
          f"worst relative amplitude error {worst_amplitude:.3e} "
          f"(q = {worst_label}); the median over the canvas, i.e. a GLOBAL "
          "statistic of the map, not its per-pixel floor (that one is asserted "
          "separately below)",
          f"relative {MULTI_PEAK_REL_TOL:g}")
    check("every one of the 12 q recovers its own modulation phase (Friedel signed)",
          worst_theta <= MULTI_PEAK_THETA_TOL_DEG,
          f"worst |dtheta| = {worst_theta:.3e} deg from the canvas mean field, i.e. a "
          "GLOBAL statistic; the per-pixel phase floor is the pointwise leakage "
          "asserted below",
          f"{MULTI_PEAK_THETA_TOL_DEG:g} deg")
    check("cross-talk between the six modulations stays at the sampling floor",
          worst_dc <= CROSS_TALK_TOL,
          f"worst relative change of the demodulated mean when the other five "
          f"modulations are present: {worst_dc:.3e}; the analytic window bound is "
          f"exp(-lambda^2 dq^2 / 2) = {analytic:.3e} for the smallest separation "
          f"{smallest:.4f} nm^-1 (lambda = {LAMBDA_NM:g} nm = {lambda_px_value:.4f} px), "
          "so the residual is the finite-canvas sampling term, not the window",
          f"relative {CROSS_TALK_TOL:g}")
    ripple_tolerance = max(RIPPLE_REL_TOL, RIPPLE_NS_CEILING / float(n_px))
    check("the pointwise inter-component leakage stays at the 24 / N floor",
          worst_ripple <= ripple_tolerance,
          f"worst pointwise |psi(12 q) - psi(its own q alone)| / median |psi| = "
          f"{worst_ripple:.3e} = {worst_ripple * n_px:.2f} / N at N = {n_px}, against "
          f"the bound {ripple_tolerance:.3e} = max({RIPPLE_REL_TOL:g}, "
          f"{RIPPLE_NS_CEILING:g} / N); this is the PER-PIXEL floor of the maps (the "
          "other five modulations leaking through the window), about 6000x the "
          f"global median amplitude error ({worst_amplitude:.2e}) and about 36x the "
          f"global mean-phase error ({worst_theta:.2e} deg) asserted above, so the "
          "per-pixel floor must be the quoted one for per-pixel use",
          f"{ripple_tolerance:.3e} at N = {n_px} ({RIPPLE_NS_CEILING:g} / N)")


# --------------------------------------------------------------------------- #
# 3. (h, k) basis coordinates against an explicit --basis-px
# --------------------------------------------------------------------------- #
def check_basis_px_equivalence(env, workdir, csv_path, report_path, size_nm):
    section("3. (h, k) basis coordinates against an explicit --basis-px")
    first = workdir / "basis_hk"
    second = workdir / "basis_px"
    completed = run_script("stm_local_q_map.py",
                           [str(csv_path), "-o", str(first), "--basis-from",
                            str(report_path), "--q", "2/3,1/3", "--no-figures",
                            "-L", f"{size_nm:g}"], env)
    if completed.returncode != 0:
        check("the (h, k) run completes", False,
              f"exit code {completed.returncode}: {completed.stderr.strip()[-300:]}")
        return
    payload = read_json(first / "local_q_map_report.json")
    basis = payload["basis_source"]
    escape = (f"{basis['b1_px'][0]:.17g},{basis['b1_px'][1]:.17g};"
              f"{basis['b2_px'][0]:.17g},{basis['b2_px'][1]:.17g}")
    completed_px = run_script("stm_local_q_map.py",
                              [str(csv_path), "-o", str(second), "--basis-px", escape,
                               "--q", "2/3,1/3", "--no-figures", "-L", f"{size_nm:g}"],
                              env)
    if completed_px.returncode != 0:
        check("the --basis-px run completes", False,
              f"exit code {completed_px.returncode}: "
              f"{completed_px.stderr.strip()[-300:]}")
        return
    name = f"{csv_path.stem}_q0_field.npy"
    field_a = np.load(first / name)
    field_b = np.load(second / name)
    scale = float(np.median(np.abs(field_a)))
    difference = float(np.max(np.abs(field_a - field_b))) / scale
    check("the same vector given as (h, k) or as --basis-px gives the same map",
          difference <= BASIS_PX_REL_TOL,
          f"max |d psi| / median |psi| = {difference:.3e} for (h, k) = (2/3, 1/3); "
          f"basis px {escape[:44]}...",
          f"relative {BASIS_PX_REL_TOL:g}")


# --------------------------------------------------------------------------- #
# 4. the basis sources
# --------------------------------------------------------------------------- #
def check_basis_sources(env, workdir, csv_path, size_nm, basis_rad_px, n_px):
    section("4. basis parsing: lawler-fujita style, affine style, affine anchored on "
            "ring_r3, rotated affine report, missing basis")
    per_px = size_nm / n_px

    lf_path = workdir / "report_lf.json"
    write_json(lf_path, lf_report(basis_rad_px, n_px, size_nm))
    out_lf = workdir / "basis_lf"
    completed = run_script("stm_local_q_map.py",
                           [str(csv_path), "-o", str(out_lf), "--basis-from",
                            str(lf_path), "--q", "1,0", "--no-figures",
                            "-L", f"{size_nm:g}"], env)
    if completed.returncode == 0:
        payload = read_json(out_lf / "local_q_map_report.json")["basis_source"]
        expected = basis_rad_px[0] / per_px
        found = np.asarray(payload["b1_nm_inv"], dtype=float)
        difference = float(np.max(np.abs(found - expected))) / float(np.hypot(*expected))
        check("lawler-fujita style report: b1 = q_a, b2 = q_a + q_b",
              payload["kind"] == "correction_report_lawler_fujita"
              and difference <= 1e-12
              and payload["b2_nm_inv"] is not None,
              f"kind {payload['kind']}, |b1 - q_a| / |q_a| = {difference:.3e}, "
              f"keys {payload['detected_keys']}",
              "kind correction_report_lawler_fujita and relative 1e-12")
        log = (out_lf / "local_q_map.log").read_text()
        check("the log prints the basis report's lockin.lambda_nm for reference only",
              "lockin.lambda_nm (reference only" in log and "30.0000 nm" in log,
              "log line: " + next((line for line in log.splitlines()
                                   if "lockin.lambda_nm" in line), "MISSING"),
              "a reference-only line with 30.0 nm")
    else:
        check("lawler-fujita style report is accepted", False,
              f"exit code {completed.returncode}: {completed.stderr.strip()[-300:]}")

    affine_path = workdir / "report_affine.json"
    write_json(affine_path, affine_report(basis_rad_px, n_px, size_nm))
    out_affine = workdir / "basis_affine"
    completed = run_script("stm_local_q_map.py",
                           [str(csv_path), "-o", str(out_affine), "--basis-from",
                            str(affine_path), "--q", "1,0", "--no-figures",
                            "-L", f"{size_nm:g}"], env)
    if completed.returncode == 0:
        payload = read_json(out_affine / "local_q_map_report.json")["basis_source"]
        expected = basis_rad_px[0] / per_px
        found = np.asarray(payload["b1_nm_inv"], dtype=float)
        difference = float(np.max(np.abs(found - expected))) / float(np.hypot(*expected))
        check("affine style report: b1 = R (cos t, sin t), b2 = b1 rotated +60 deg",
              payload["kind"] == "correction_report_affine" and difference <= 1e-12
              and abs(payload["angle_between_deg"] - 60.0) <= 1e-6,
              f"kind {payload['kind']}, |b1 - R(cos t, sin t)| / R = {difference:.3e}, "
              f"angle between b1 and b2 = {payload['angle_between_deg']:.9f} deg",
              "kind correction_report_affine, relative 1e-12, 60 deg")
    else:
        check("affine style report is accepted", False,
              f"exit code {completed.returncode}: {completed.stderr.strip()[-300:]}")

    fallbacks = []
    fallback_ok = True
    for label, measured, ideal in (("b1_ideal_nm_inv", False, True),
                                   ("implied_lattice_after.a_1x1_nm", False, False)):
        path = workdir / f"report_fallback_{label.split('.')[0]}.json"
        write_json(path, affine_report(basis_rad_px, n_px, size_nm, measured=measured,
                                       ideal=ideal))
        out = workdir / f"basis_fallback_{label.split('.')[0]}"
        completed = run_script("stm_local_q_map.py",
                               [str(csv_path), "-o", str(out), "--basis-from", str(path),
                                "--q", "1,0", "--no-figures", "-L", f"{size_nm:g}"], env)
        if completed.returncode != 0:
            fallbacks.append(f"{label}: exit {completed.returncode}")
            fallback_ok = False
            continue
        payload = read_json(out / "local_q_map_report.json")["basis_source"]
        expected = float(np.hypot(*(basis_rad_px[0] / per_px)))
        found = float(payload["abs_b1_nm_inv"])
        relative = rel_error(found, expected)
        fallback_ok = fallback_ok and relative <= 1e-12
        fallbacks.append(f"{label}: relative {relative:.3e}")
    check("the affine radius falls back to b1_ideal_nm_inv and to a_1x1_nm",
          fallback_ok, "; ".join(fallbacks), "relative 1e-12")

    # the affine report of a correction anchored on ring_r3: its radius keys carry
    # the r3 radius, so the ring_1x1 radius the basis needs is sqrt(3) times them
    expected_1x1 = float(np.hypot(*(basis_rad_px[0] / per_px)))
    expected_r3 = expected_1x1 / np.sqrt(3.0)
    for label, labelled in (("anchor_ring=r3", True), ("no anchor_ring", False)):
        path = workdir / f"report_affine_r3_{int(labelled)}.json"
        write_json(path, affine_r3_report(basis_rad_px, n_px, size_nm,
                                          labelled=labelled))
        out = workdir / f"basis_affine_r3_{int(labelled)}"
        completed = run_script("stm_local_q_map.py",
                               [str(csv_path), "-o", str(out), "--basis-from", str(path),
                                "--q", "1,0", "--no-figures", "-L", f"{size_nm:g}"], env)
        if completed.returncode != 0:
            check(f"the ring_r3-anchored affine report is accepted ({label})", False,
                  f"exit code {completed.returncode}: "
                  f"{completed.stderr.strip()[-300:]}")
            continue
        payload = read_json(out / "local_q_map_report.json")
        source = payload["basis_source"]
        found = float(source["abs_b1_nm_inv"])
        entry = payload["q"][0]
        relative = rel_error(found, expected_1x1)
        check(f"an affine report anchored on ring_r3 resolves the ring_1x1 basis "
              f"({label})",
              relative <= 1e-12
              and rel_error(entry["q_nm_inv_abs"], expected_1x1) <= 1e-12
              and not source["basis_warnings"],
              f"|b1| = {found:.9f} nm^-1 vs the ring_1x1 radius {expected_1x1:.9f} "
              f"(the anchored r3 radius of the report is {expected_r3:.9f}): relative "
              f"{relative:.3e}; the (1,0) member is delivered at |q| = "
              f"{entry['q_nm_inv_abs']:.9f} nm^-1; basis warnings "
              f"{source['basis_warnings'] or 'none'}",
              "relative 1e-12, the (1,0) member on ring_1x1 and no basis warning")
        if labelled:
            log = (out / "local_q_map.log").read_text()
            check("the log and the report notes name the decoded anchored ring",
                  "anchored on ring_r3" in log
                  and any("anchored on ring_r3" in note for note in source["notes"]),
                  "log line: " + next((line for line in log.splitlines()
                                       if "anchored on ring_r3" in line), "MISSING")
                  + f"; {len(source['notes'])} report note(s)",
                  "the decoded ring identity in the log and in basis_source.notes")

    # the rings_after cross-check itself: a basis left on the anchored r3 radius
    # must warn, a basis on the ring_1x1 radius must stay silent
    r3_payload = affine_r3_report(basis_rad_px, n_px, size_nm)
    warned = lq.basis_ring_warning(r3_payload, expected_r3, "r3", expected_r3)
    quiet = lq.basis_ring_warning(r3_payload, expected_1x1, "r3", expected_r3)
    check("the rings_after cross-check warns only when the basis sits on ring_r3",
          bool(warned) and quiet is None,
          f"the r3 radius -> {('a WARNING' if warned else 'MISSING')}; the ring_1x1 "
          f"radius -> {('a WARNING' if quiet else 'none')}",
          "a WARNING on the r3 radius and silence on the ring_1x1 radius")

    # the rotated affine report (review F6): orientation_deg describes the INPUT
    # frame of the correction, here 30.8 deg away from the corrected canvas' own
    # ring_1x1 direction.  The default path measures the canvas itself and re-frames
    # the basis onto its ring; --basis-orientation-from report reproduces the defect,
    # and the unconditional basis-vs-canvas check flags it (WARNING + report flag,
    # and --strict stops before any per-q artifact).
    rotation_deg = 30.8
    rotated_path = workdir / "report_affine_rotated.json"
    write_json(rotated_path, affine_report(basis_rad_px, n_px, size_nm,
                                           orientation_offset_deg=rotation_deg))
    out_rotated = workdir / "basis_affine_rotated"
    completed_rotated = run_script(
        "stm_local_q_map.py",
        [str(csv_path), "-o", str(out_rotated), "--basis-from", str(rotated_path),
         "--q", "1,0", "--no-figures", "-L", f"{size_nm:g}"], env)
    out_report_frame = workdir / "basis_affine_rotated_report_frame"
    completed_report_frame = run_script(
        "stm_local_q_map.py",
        [str(csv_path), "-o", str(out_report_frame), "--basis-from", str(rotated_path),
         "--q", "1,0", "--no-figures", "-L", f"{size_nm:g}",
         "--basis-orientation-from", "report", "--no-strict"], env)
    out_strict = workdir / "basis_affine_rotated_strict"
    completed_strict = run_script(
        "stm_local_q_map.py",
        [str(csv_path), "-o", str(out_strict), "--basis-from", str(rotated_path),
         "--q", "1,0", "--no-figures", "-L", f"{size_nm:g}",
         "--basis-orientation-from", "report"], env)  # --strict is the default
    if completed_rotated.returncode != 0:
        check("a rotated affine report is re-framed on the corrected canvas", False,
              f"exit code {completed_rotated.returncode}: "
              f"{completed_rotated.stderr.strip()[-300:]}")
    else:
        payload = read_json(out_rotated / "local_q_map_report.json")
        source = payload["basis_source"]
        expected_radius = float(np.hypot(*basis_rad_px[0])) / per_px
        found = float(source["abs_b1_nm_inv"])
        folded = float(lq.fold_deg(source["b1_angle_deg"]))
        canvas_angle = float(source["canvas_orientation"]["member_angles_deg"][0])
        median = float(np.median(np.load(
            out_rotated / f"{csv_path.stem}_q0_amplitude.npy")))
        check("a rotated affine report is re-framed on the corrected canvas",
              rel_error(found, expected_radius) <= 1e-12
              and min(folded, 60.0 - folded) <= 1e-2
              and source["orientation_from_canvas"] is True
              and not payload["warnings"]["basis_ring_mismatch"]
              and median > 0.35,
              f"the report's orientation_deg is {rotation_deg:g} deg away from the "
              f"canvas ring (measured canvas member at {canvas_angle:.4f} deg); "
              f"|b1| = {found:.9f} nm^-1 (relative "
              f"{rel_error(found, expected_radius):.3e}), the resolved b1 folds to "
              f"{folded:.4f} deg and |psi| median is {median:.6f} (A / 2 = 0.45, the "
              f"noise floor of the report frame is asserted below)",
              "|b1| relative 1e-12, b1 on a canvas ring member (folded 60 deg), "
              "|psi| > 0.35 and no mismatch")
        if completed_report_frame.returncode != 0 or completed_strict.returncode == 0:
            check("the basis-vs-canvas check flags the report frame and the default "
                  "strict mode stops it", False,
                  f"the report-frame --no-strict run exited "
                  f"{completed_report_frame.returncode} and the default strict run exited "
                  f"{completed_strict.returncode}")
        else:
            forced = read_json(out_report_frame / "local_q_map_report.json")
            warnings = forced["warnings"]
            log = (out_report_frame / "local_q_map.log").read_text()
            forced_median = float(np.median(np.load(
                out_report_frame / f"{csv_path.stem}_q0_amplitude.npy")))
            ratio = float(warnings["basis_ring_strength_ratio"])
            check("the basis-vs-canvas check flags the report frame and the default "
                  "strict mode stops it",
                  warnings["basis_ring_mismatch"] is True
                  and "basis-vs-canvas check: MISMATCH" in log
                  and ratio < lq.BASIS_RING_STRENGTH_FRACTION
                  and completed_strict.returncode == 3
                  and not list(out_strict.glob("*.npy"))
                  and completed_report_frame.returncode == 0
                  and len(list(out_report_frame.glob("*.npy"))) == 4
                  and median > 1e6 * forced_median,
                  f"report frame (--no-strict): the members reach {100.0 * ratio:.4f} % of "
                  f"the annulus peak (threshold "
                  f"{100.0 * lq.BASIS_RING_STRENGTH_FRACTION:.0f} %), worst member offset "
                  f"{float(warnings['basis_ring_worst_offset_px']):.3f} px (tolerance "
                  f"{float(warnings['basis_ring_offset_tolerance_px']):.3f} px), mismatch "
                  f"flag {warnings['basis_ring_mismatch']}, exit "
                  f"{completed_report_frame.returncode} with "
                  f"{len(list(out_report_frame.glob('*.npy')))} npy; |psi| median "
                  f"{forced_median:.3e} there against {median:.6f} on the canvas-derived "
                  f"basis (factor {median / max(forced_median, 1e-300):.3e}); default "
                  f"strict exit {completed_strict.returncode} with "
                  f"{len(list(out_strict.glob('*.npy')))} npy file(s)",
                  "mismatch true, the log line present, ratio < 0.2, |psi| factor > 1e6, "
                  "--no-strict exit 0 and the default strict mode exit 3 without per-q "
                  "artifacts")

    broken_path = workdir / "report_broken.json"
    write_json(broken_path, {"skill": "something-else",
                             "corrected_field_of_view_nm": size_nm,
                             "corrected_nm_per_px": per_px})
    completed = run_script("stm_local_q_map.py",
                           [str(csv_path), "-o", str(workdir / "basis_broken"),
                            "--basis-from", str(broken_path), "--q", "1,0",
                            "--no-figures", "-L", f"{size_nm:g}"], env)
    message = (completed.stderr + completed.stdout)
    check("a report without a recognised basis exits non-zero with a clear message",
          completed.returncode != 0 and "no recognised reciprocal basis" in message,
          f"exit code {completed.returncode}, message: "
          f"{next((line for line in message.splitlines() if 'no recognised' in line), 'MISSING')}",
          "non-zero exit and 'no recognised reciprocal basis'")


# --------------------------------------------------------------------------- #
# 5. the gauge behaviour of the demodulated maps
# --------------------------------------------------------------------------- #
def check_gauge_law(n_px, nm_per_px, columns, rows):
    section("5. gauge law: a constant modulation phase and an image shift")

    # The phi0 law holds for any carrier; the translation law is exact for the
    # sampled array only when the carrier is periodic on the canvas, so this check
    # runs on a commensurate (integer fftshift offset) basis.
    basis = np.asarray(COMMENSURATE_BASIS_PX, dtype=float) * (2.0 * np.pi / n_px)
    table = [{"h": h, "k": k,
              "amplitude": PEAK_AMPLITUDE_START - PEAK_AMPLITUDE_STEP * index,
              "phase": np.radians(PEAK_PHASE_START_DEG + PEAK_PHASE_STEP_DEG * index)}
             for index, (h, k) in enumerate(COMMENSURATE_MEMBERS)]
    base = synthetic_image(columns, rows, basis, table)
    phi0 = np.radians(52.25)
    shifted = synthetic_image(columns, rows, basis,
                              [dict(peak, phase=peak["phase"] + phi0) for peak in table])
    worst_phase = 0.0
    for peak in table:
        q = lq.q_vector(peak["h"], peak["k"], basis[0], basis[1])
        a = np.angle(lq.demodulate(base, q, LAMBDA_NM, nm_per_px))
        b = np.angle(lq.demodulate(shifted, q, LAMBDA_NM, nm_per_px))
        worst_phase = max(worst_phase, float(np.max(np.abs(
            np.degrees(lq.wrap_pm_pi(b - a - phi0))))))
    check("a constant modulation phase phi0 shifts the theta map by exactly phi0",
          worst_phase <= GAUGE_TOL_DEG,
          f"worst |dtheta - phi0| = {worst_phase:.3e} deg over the "
          f"{len(table)} commensurate modulations "
          f"(phi0 = {np.degrees(phi0):.4f} deg)",
          f"{GAUGE_TOL_DEG:g} deg")

    shift = (13, -7)  # (row, col) pixels
    moved = np.roll(base, shift=shift, axis=(0, 1))
    worst_field, worst_amplitude = 0.0, 0.0
    gauge_deg = float("nan")
    for peak in table:
        q = lq.q_vector(peak["h"], peak["k"], basis[0], basis[1])
        before = lq.demodulate(base, q, LAMBDA_NM, nm_per_px)
        after = lq.demodulate(moved, q, LAMBDA_NM, nm_per_px)
        gauge = np.exp(-1j * (q[0] * shift[1] + q[1] * shift[0]))
        expected = gauge * np.roll(before, shift=shift, axis=(0, 1))
        scale = float(np.median(np.abs(before)))
        worst_field = max(worst_field, float(np.max(np.abs(after - expected))) / scale)
        worst_amplitude = max(worst_amplitude, float(np.max(np.abs(
            np.abs(after) - np.roll(np.abs(before), shift=shift, axis=(0, 1))))) / scale)
        gauge_deg = np.degrees(np.angle(gauge))
    check("shifting the image by delta translates the maps by delta",
          worst_field <= TRANSLATION_REL_TOL,
          f"max |psi'(r) - exp(-i q.delta) psi(r - delta)| / median |psi| = "
          f"{worst_field:.3e} for delta = {shift} px (the exact gauge constant of "
          f"the first modulation is {gauge_deg:.4f} deg)",
          f"relative {TRANSLATION_REL_TOL:g}")
    check("shifting the image by delta translates the amplitude map by delta",
          worst_amplitude <= TRANSLATION_REL_TOL,
          f"max | |psi'| - |psi|(r - delta) | / median |psi| = {worst_amplitude:.3e}",
          f"relative {TRANSLATION_REL_TOL:g}")


# --------------------------------------------------------------------------- #
# 6. and 7. the two exact identities
# --------------------------------------------------------------------------- #
def check_identities(basis, n_px, nm_per_px, columns, rows):
    section("6. Friedel identity theta_q + theta_-q = 0 (mod 2 pi)")
    table = peak_table()
    image = synthetic_image(columns, rows, basis, table)
    worst_friedel = 0.0
    for peak in table:
        q = lq.q_vector(peak["h"], peak["k"], basis[0], basis[1])
        theta = np.angle(lq.demodulate(image, q, LAMBDA_NM, nm_per_px))
        theta_minus = np.angle(lq.demodulate(image, -q, LAMBDA_NM, nm_per_px))
        worst_friedel = max(worst_friedel, float(np.max(np.abs(
            np.degrees(lq.wrap_pm_pi(theta + theta_minus))))))
    check("theta_q + theta_-q = 0 mod 2 pi for a real image",
          worst_friedel <= FRIEDEL_TOL_DEG,
          f"worst |wrap(theta_q + theta_-q)| = {worst_friedel:.3e} deg over all 12 "
          "q (6 modulations and their Friedel partners)",
          f"{FRIEDEL_TOL_DEG:g} deg")

    section("7. mean identity arg sum psi_q = arg sum T exp(-i q.r)")
    worst_mean = 0.0
    for peak in table:
        for sign in (1.0, -1.0):
            q = sign * lq.q_vector(peak["h"], peak["k"], basis[0], basis[1])
            field = lq.demodulate(image, q, LAMBDA_NM, nm_per_px)
            left = float(np.angle(np.sum(field)))
            right = float(np.angle(lq.demodulated_sum(image, q)))
            worst_mean = max(worst_mean,
                             abs(float(np.degrees(lq.wrap_pm_pi(left - right)))))
    check("the window cancels from the total phase (the strongest exact identity)",
          worst_mean <= MEAN_IDENTITY_TOL_DEG,
          f"worst |arg sum psi - arg sum T exp(-i q.r)| = {worst_mean:.3e} deg over "
          "all 12 q", f"{MEAN_IDENTITY_TOL_DEG:g} deg")


# --------------------------------------------------------------------------- #
# 8. NaN handling
# --------------------------------------------------------------------------- #
def check_nan_handling(env, workdir, basis, n_px, nm_per_px, columns, rows, size_nm):
    section("8. NaN: fill-0 against fill-plane, and the mask")
    table = peak_table()
    image = synthetic_image(columns, rows, basis, table)
    peak = table[0]
    q = lq.q_vector(peak["h"], peak["k"], basis[0], basis[1])
    patch = (90, 130, 60, 100)  # row0, row1, col0, col1
    damaged = image.copy()
    damaged[patch[0]:patch[1], patch[2]:patch[3]] = np.nan
    finite = np.isfinite(damaged)
    background = float(np.mean(damaged[finite]))
    plane_filled = np.where(finite, damaged, background)

    spectrum_zero = lq.demodulated_spectrum(damaged, q, nan_fill=0.0)
    spectrum_plane = lq.demodulated_spectrum(plane_filled, q, nan_fill=0.0)
    missing = ~finite
    predicted = lq.demodulated_spectrum(missing.astype(float), q, nan_fill=0.0) \
        * (-background)
    residual = float(np.linalg.norm((spectrum_zero - spectrum_plane
                                     - predicted).ravel()))
    reference = float(np.linalg.norm(spectrum_zero.ravel()))
    relative = residual / reference
    check("fill-0 and fill-plane differ in the spectrum only through the gap",
          relative <= NAN_SPECTRUM_REL_TOL,
          f"relative residual between the measured fill difference and the missing "
          f"support weighted by the fill value = {relative:.3e} "
          f"(missing fraction {float(np.mean(missing)):.6f}, fill plane "
          f"{background:.6e})",
          f"relative {NAN_SPECTRUM_REL_TOL:g}")

    field_zero = lq.demodulate(damaged, q, LAMBDA_NM, nm_per_px, nan_fill=0.0)
    field_plane = lq.demodulate(plane_filled, q, LAMBDA_NM, nm_per_px, nan_fill=0.0)
    row_gap = np.maximum(np.maximum(np.arange(n_px)[:, None] - patch[1],
                                    patch[0] - np.arange(n_px)[:, None]), 0.0)
    col_gap = np.maximum(np.maximum(np.arange(n_px)[None, :] - patch[3],
                                    patch[2] - np.arange(n_px)[None, :]), 0.0)
    distance = np.hypot(row_gap, col_gap)
    halo = distance > NAN_HALO_WINDOWS * lq.lambda_px(LAMBDA_NM, nm_per_px)
    scale = float(np.median(np.abs(field_zero)))
    missing_fraction = float(np.mean(missing))
    perturbation = float(np.max(np.abs(field_zero - field_plane))) / scale
    halo_relative = float(np.max(np.abs(field_zero[halo] - field_plane[halo]))) / scale
    near_relative = float(np.max(np.abs(field_zero[~halo] - field_plane[~halo]))) / scale
    check("the fill value perturbs the demodulated maps by less than the gap itself",
          perturbation <= missing_fraction,
          f"max |psi(fill 0) - psi(fill plane)| / median |psi| = {perturbation:.3e} "
          f"against the missing fraction {missing_fraction:.3e} "
          f"(outside {NAN_HALO_WINDOWS:g} lambda = "
          f"{NAN_HALO_WINDOWS * lq.lambda_px(LAMBDA_NM, nm_per_px):.2f} px: "
          f"{halo_relative:.3e}, inside: {near_relative:.3e})",
          f"relative {missing_fraction:.3e}")

    nan_csv = workdir / "nan_topo.csv"
    border = image.copy()
    border[:6, :] = np.nan
    border[-6:, :] = np.nan
    border[patch[0]:patch[1], patch[2]:patch[3]] = np.nan
    np.savetxt(nan_csv, border, delimiter=",", fmt="%.6e")
    outdir = workdir / "nan_run"
    completed = run_script("stm_local_q_map.py",
                           [str(nan_csv), "-o", str(outdir), "--basis-from",
                            str(workdir / "report_affine.json"), "--q", "1,0",
                            "--no-figures", "-L", f"{size_nm:g}"], env)
    if completed.returncode != 0:
        check("the NaN pipeline run completes", False,
              f"exit code {completed.returncode}: {completed.stderr.strip()[-300:]}")
        return
    payload = read_json(outdir / "local_q_map_report.json")
    stem = nan_csv.stem
    products = {name: np.load(outdir / f"{stem}_q0_{name}.npy")
                for name in ("field", "amplitude", "theta", "mask")}
    nan_region = ~np.isfinite(border)
    flagged = bool(np.all(products["mask"][nan_region] == 0.0))
    finite = all(bool(np.all(np.isfinite(array))) for array in products.values())
    dtype_ok = products["field"].dtype == np.complex128
    entry = payload["q"][0]
    nan_fraction = float(np.mean(nan_region))
    check("every NaN pixel of the input is flagged and every npy stays finite",
          flagged and finite and dtype_ok
          and abs(entry["nan_fraction"] - nan_fraction) <= 1e-12,
          f"NaN pixels flagged: {flagged}; all npy finite: {finite}; field dtype "
          f"complex128: {dtype_ok}; reported nan_fraction {entry['nan_fraction']:.6f} "
          f"vs measured {nan_fraction:.6f}; mask coverage "
          f"{entry['mask_coverage_fraction']:.6f}",
          "flagged, finite, complex128, nan_fraction equal")


# --------------------------------------------------------------------------- #
# 9. determinism
# --------------------------------------------------------------------------- #
def check_determinism(env, workdir, csv_path, report_path, size_nm):
    section("9. determinism: two runs give byte-identical artifacts")
    first = workdir / "determinism_a"
    second = workdir / "determinism_b"
    arguments = [str(csv_path), "--basis-from", str(report_path), "--q", "1,0",
                 "--q", "1/3,1/3", "-L", f"{size_nm:g}"]
    completed_a = run_script("stm_local_q_map.py", [*arguments, "-o", str(first)], env)
    completed_b = run_script("stm_local_q_map.py", [*arguments, "-o", str(second)], env)
    if completed_a.returncode != 0 or completed_b.returncode != 0:
        check("both determinism runs complete", False,
              f"exit codes {completed_a.returncode} and {completed_b.returncode}")
        return
    differences = []
    compared = 0
    for path in sorted(first.iterdir()):
        if path.name in ("local_q_map_report.json", "local_q_map.log"):
            continue                      # they carry the output directory path
        other = second / path.name
        compared += 1
        if not other.is_file() or path.read_bytes() != other.read_bytes():
            differences.append(path.name)
    check("two runs produce byte-identical npy and png artifacts",
          not differences and compared == 14,
          f"{compared - len(differences)}/{compared} artifacts byte-identical "
          f"(2 q x [field/amplitude/theta/mask npy + amplitude/theta/mask png])"
          + (f"; differing: {differences}" if differences else ""),
          "14/14 byte-identical")


# --------------------------------------------------------------------------- #
# 10. the field of view from a correction log
# --------------------------------------------------------------------------- #
def check_size_nm_from_log(env, workdir, csv_path, basis_px_spec, n_px, lf_path,
                           size_nm):
    section("10. field of view: --size-nm-from-log branches and the correction-report "
            "branch")
    small = workdir / "fov_small.csv"
    np.savetxt(small, np.zeros((16, 16)), delimiter=",", fmt="%.6e")
    corrected_nm = 103.3691
    log_path = workdir / "correction_two_lines.log"
    log_path.write_text(
        f"# canvas {n_px} x {n_px} px, field of view {FIELD_NM:g} nm "
        f"({FIELD_NM / n_px:.6f} nm/px)\n"
        f"# corrected canvas: {n_px} x {n_px} px, field of view {corrected_nm} nm "
        f"({corrected_nm / n_px:.6f} nm/px)\n")
    outdir = workdir / "fov_corrected"
    completed = run_script("stm_local_q_map.py",
                           [str(small), "-o", str(outdir), "--basis-px", basis_px_spec,
                            "--q", "1,0", "--no-figures",
                            "--size-nm-from-log", str(log_path)], env)
    if completed.returncode == 0:
        payload = read_json(outdir / "local_q_map_report.json")
        check("the corrected-canvas line of the log wins over the input-canvas line",
              abs(payload["field_of_view_nm"] - corrected_nm) <= 1e-9,
              f"field of view {payload['field_of_view_nm']:.6f} nm from "
              f"'{payload['field_of_view_source']}'",
              f"{corrected_nm} nm (not {FIELD_NM:g} nm)")
    else:
        check("the corrected-canvas log branch completes", False,
              f"exit code {completed.returncode}: {completed.stderr.strip()[-300:]}")

    plain = workdir / "correction_one_line.log"
    plain.write_text(f"# canvas {n_px} x {n_px} px, field of view 42.5 nm\n")
    outdir_plain = workdir / "fov_plain"
    completed = run_script("stm_local_q_map.py",
                           [str(small), "-o", str(outdir_plain), "--basis-px",
                            basis_px_spec, "--q", "1,0", "--no-figures",
                            "--size-nm-from-log", str(plain)], env)
    if completed.returncode == 0:
        payload = read_json(outdir_plain / "local_q_map_report.json")
        check("a single-line log falls back to its last field of view",
              abs(payload["field_of_view_nm"] - 42.5) <= 1e-9,
              f"field of view {payload['field_of_view_nm']:.6f} nm from "
              f"'{payload['field_of_view_source']}'", "42.5 nm")
    else:
        check("the single-line log branch completes", False,
              f"exit code {completed.returncode}: {completed.stderr.strip()[-300:]}")

    empty = workdir / "correction_empty.log"
    empty.write_text("# no field of view line at all\n")
    completed = run_script("stm_local_q_map.py",
                           [str(small), "-o", str(workdir / "fov_empty"),
                            "--basis-px", basis_px_spec, "--q", "1,0",
                            "--no-figures", "--size-nm-from-log", str(empty)], env)
    message = completed.stderr + completed.stdout
    check("a log without any field of view line is an explicit error",
          completed.returncode != 0 and "no 'field of view" in message,
          f"exit code {completed.returncode}, message: "
          f"{next((line for line in message.splitlines() if 'field of view' in line), 'MISSING')}",
          "non-zero exit and an explicit message")

    # the correction-report branch: --basis-from with neither -L nor
    # --size-nm-from-log, on a report carrying only corrected_nm_per_px
    per_px = size_nm / n_px
    report_fov = workdir / "report_fov_from_per_px.json"
    fov_payload = read_json(lf_path)
    fov_payload.pop("corrected_field_of_view_nm", None)
    write_json(report_fov, fov_payload)
    out_fov = workdir / "fov_from_report"
    completed = run_script("stm_local_q_map.py",
                           [str(csv_path), "-o", str(out_fov), "--basis-from",
                            str(report_fov), "--q", "1,0", "--no-figures"], env)
    if completed.returncode == 0:
        payload = read_json(out_fov / "local_q_map_report.json")
        check("the report branch times corrected_nm_per_px by the canvas",
              rel_error(payload["field_of_view_nm"], per_px * n_px) <= 1e-12
              and rel_error(payload["nm_per_px"], per_px) <= 1e-12
              and "corrected_nm_per_px" in payload["field_of_view_source"],
              f"field of view {payload['field_of_view_nm']:.6f} nm and nm/px "
              f"{payload['nm_per_px']:.12f} against corrected_nm_per_px "
              f"{per_px:.12f} x canvas {n_px} = {per_px * n_px:.6f} nm; the unscaled "
              f"value would be {per_px:.6f} nm, a factor {n_px} off "
              f"({payload['field_of_view_source']})",
              "field_of_view_nm = corrected_nm_per_px * n and nm_per_px = per_px")
        check("the report branch keeps the window inside the canvas",
              rel_error(payload["lambda_px"], LAMBDA_NM / per_px) <= 1e-12
              and not payload["warnings"]["boundary_warning"],
              f"lambda_px {payload['lambda_px']:.6f} vs lambda_nm / per_px = "
              f"{LAMBDA_NM / per_px:.6f}; boundary warning "
              f"{payload['warnings']['boundary_warning']} (lambda {LAMBDA_NM:g} nm "
              f"against L / 4 = {size_nm / 4.0:g} nm)",
              "lambda_px = lambda_nm / per_px and no boundary warning")
    else:
        check("the report field-of-view branch completes", False,
              f"exit code {completed.returncode}: {completed.stderr.strip()[-300:]}")


# --------------------------------------------------------------------------- #
# 11. end to end
# --------------------------------------------------------------------------- #
def check_end_to_end(env, workdir, csv_path, lf_path, affine_path, n_px, size_nm):
    section("11. end to end: two full pipeline runs and the artifact contract")
    single = workdir / "e2e_single"
    multi = workdir / "e2e_multi"
    completed = run_script("stm_local_q_map.py",
                           [str(csv_path), "-o", str(single), "--basis-from",
                            str(lf_path), "--q", "1,0", "-L", f"{size_nm:g}"], env)
    if completed.returncode != 0:
        check("the single-q pipeline run completes", False,
              f"exit code {completed.returncode}: {completed.stderr.strip()[-300:]}")
        return
    multi_q = ["1,0", "0,1", "1/3,1/3"]
    q_flags = [item for spec in multi_q for item in ("--q", spec)]
    completed_multi = run_script(
        "stm_local_q_map.py",
        [str(csv_path), "-o", str(multi), "--basis-from", str(affine_path),
         *q_flags, "-L", f"{size_nm:g}"], env)
    if completed_multi.returncode != 0:
        check("the multi-q pipeline run completes", False,
              f"exit code {completed_multi.returncode}: "
              f"{completed_multi.stderr.strip()[-300:]}")
        return

    stem = csv_path.stem
    expected_files = []
    for index in range(1):
        for suffix in ("field.npy", "amplitude.npy", "theta.npy", "mask.npy",
                       "amplitude.png", "theta.png", "mask.png"):
            expected_files.append((single, f"{stem}_q{index}_{suffix}"))
    for index in range(3):
        for suffix in ("field.npy", "amplitude.npy", "theta.npy", "mask.npy",
                       "amplitude.png", "theta.png", "mask.png"):
            expected_files.append((multi, f"{stem}_q{index}_{suffix}"))
    missing = [f"{outdir.name}/{name}" for outdir, name in expected_files
               if not (outdir / name).is_file()]
    check("every per-q artifact of both runs exists on disk",
          not missing,
          f"single-q run 7 artifacts (4 npy + 3 png), multi-q run 3 x 7 artifacts; "
          f"missing: {missing if missing else 'none'}",
          "21/21 files present")

    field = np.load(single / f"{stem}_q0_field.npy")
    theta_npy = np.load(single / f"{stem}_q0_theta.npy")
    amplitude_npy = np.load(single / f"{stem}_q0_amplitude.npy")
    mask_npy = np.load(single / f"{stem}_q0_mask.npy")
    check("the npy contract holds (complex128 field, radians, 0/1 mask)",
          field.dtype == np.complex128 and field.shape == (n_px, n_px)
          and theta_npy.dtype == np.float64 and amplitude_npy.dtype == np.float64
          and set(np.unique(mask_npy).tolist()) <= {0.0, 1.0}
          and bool(np.all(np.isfinite(theta_npy)))
          and bool(np.all(np.isfinite(amplitude_npy))),
          f"field {field.dtype} {field.shape}, theta {theta_npy.dtype}, amplitude "
          f"{amplitude_npy.dtype}, mask values {sorted(set(np.unique(mask_npy).tolist()))}",
          "complex128, float64 radians, mask {0, 1}, all finite")

    payload = read_json(multi / "local_q_map_report.json")
    entries = payload["q"]
    required = ("label", "h", "k", "q_px", "q_rad_px", "q_nm_inv", "q_nm_inv_abs",
                "lambda_nm", "lambda_px", "amplitude", "theta", "nan_fraction",
                "mask_coverage_fraction", "artifacts")
    absent = sorted({name for entry in entries for name in required if name not in entry})
    stats_absent = sorted({name for entry in entries
                           for name in ("median", "fwhm") if name not in entry["amplitude"]})
    theta_absent = sorted({name for entry in entries
                           for name in ("circular_mean_deg", "circular_mean_all_pixels_deg",
                                        "mean_identity_error_deg", "circular_median_deg",
                                        "resultant_R") if name not in entry["theta"]})
    check("the multi-q report carries every per-q contract field",
          not absent and not stats_absent and not theta_absent and len(entries) == 3,
          f"{len(entries)} q entries; missing per-q keys {absent or 'none'}; missing "
          f"amplitude keys {stats_absent or 'none'}; missing theta keys "
          f"{theta_absent or 'none'}",
          "3 entries and every field present")
    labels = [entry["label"] for entry in entries]
    coordinates = [(round(entry["h"], 12), round(entry["k"], 12)) for entry in entries]
    check("the q labels and the (h, k) bookkeeping follow the command line",
          labels == ["q0", "q1", "q2"]
          and coordinates == [(1.0, 0.0), (0.0, 1.0),
                              (round(1.0 / 3.0, 12), round(1.0 / 3.0, 12))],
          f"labels {labels}, (h, k) {coordinates}", "q0, q1, q2 and (1,0), (0,1), (1/3,1/3)")

    basis = payload["basis_source"]
    source_absent = sorted({name for name in ("kind", "path", "detected_keys",
                                              "b1_nm_inv", "b2_nm_inv", "b1_rad_px",
                                              "b2_rad_px", "basis_nm_per_px",
                                              "orientation_source",
                                              "orientation_from_canvas",
                                              "orientation_delta_deg",
                                              "canvas_orientation",
                                              "basis_canvas_check")
                            if name not in basis})
    check("the report carries the basis-source and warning fields",
          not source_absent
          and {"cross_talk_pairs", "boundary_warning", "border_band_px"}
          <= set(payload["warnings"])
          and payload["rings"]["ring_1x1_members_hk"][0] == [1.0, 0.0],
          f"missing basis keys {source_absent or 'none'}; warnings "
          f"{sorted(payload['warnings'])}; ring_1x1 members "
          f"{payload['rings']['ring_1x1_members_hk']}",
          "no missing basis key and the warning block present")

    log = (single / "local_q_map.log").read_text()
    needles = ("# canvas ", "# field of view source:", "# basis source:",
               "# window lambda =", "# q0:", "# written:", "boundary: the periodic FFT")
    absent_needles = [needle for needle in needles if needle not in log]
    check("the log carries the contract lines",
          not absent_needles,
          f"{len(needles) - len(absent_needles)}/{len(needles)} lines present; "
          f"missing {absent_needles or 'none'}", "every line present")

    check("no k-space high-symmetry name appears in the products",
          all(token not in log.lower() for token in ("bragg point", "high-symmetry",
                                                     "high symmetry", " k-point")),
          "the log names the rings ring_1x1 and ring_r3 only",
          "no k-space name")

    # the amplitude-weighted circular median of the report: a bimodal-amplitude
    # synthetic where most pixels carry the small amplitude, so the unweighted and
    # the weighted definitions land on the two opposite phases
    bimodal_csv = workdir / "bimodal_amplitude.csv"
    columns = np.arange(n_px, dtype=float)[None, :]
    rows = np.arange(n_px, dtype=float)[:, None]
    peak = peak_table()[0]
    basis_rows = synthetic_basis(n_px)
    q_line = lq.q_vector(peak["h"], peak["k"], basis_rows[0], basis_rows[1])
    cut = round(0.75 * n_px)
    phase_map = np.radians(np.where(columns < cut, 12.0, 83.0))
    amplitude_map = np.where(columns < cut, 0.20, 1.60)
    bimodal = amplitude_map * np.cos(q_line[0] * columns + q_line[1] * rows
                                     + phase_map)
    np.savetxt(bimodal_csv, bimodal, delimiter=",", fmt="%.17g")
    nm_per_px = size_nm / n_px
    field = lq.demodulate(bimodal, q_line, LAMBDA_NM, nm_per_px)
    amplitude = np.abs(field)
    valid, _ = lq.valid_mask(amplitude, ~np.isfinite(bimodal),
                             lq.DEFAULT_AMPLITUDE_FRACTION)
    theta = np.angle(field)
    weighted, _, _ = lq.circular_median_rad(theta[valid], weight=amplitude[valid])
    unweighted, _, _ = lq.circular_median_rad(theta[valid])
    bimodal_out = workdir / "bimodal_run"
    completed = run_script("stm_local_q_map.py",
                           [str(bimodal_csv), "-o", str(bimodal_out), "--basis-from",
                            str(lf_path), "--q", "1,0", "--no-figures",
                            "-L", f"{size_nm:g}"], env)
    if completed.returncode == 0:
        reported = float(read_json(bimodal_out / "local_q_map_report.json")["q"][0]
                         ["theta"]["circular_median_deg"])
        gap = abs(float(np.degrees(lq.wrap_pm_pi(weighted - unweighted))))
        residual = abs(float(np.degrees(lq.wrap_pm_pi(np.radians(reported) - weighted))))
        check("the reported circular median is the amplitude-weighted one",
              residual <= 1e-9 and gap >= 20.0,
              f"reported {reported:.6f} deg, amplitude weighted {np.degrees(weighted) % 360.0:.6f} "
              f"deg, unweighted {np.degrees(unweighted) % 360.0:.6f} deg "
              f"(residual {residual:.3e} deg); the two definitions are {gap:.3f} deg "
              f"apart on this bimodal-amplitude sample",
              "reported = weighted to 1e-9 deg with the definitions >= 20 deg apart")
    else:
        check("the bimodal-amplitude run completes", False,
              f"exit code {completed.returncode}: {completed.stderr.strip()[-300:]}")

    # the warning branches of the contract, on the same pipeline
    matching = read_json(single / "local_q_map_report.json")["warnings"]
    check("a canvas that agrees with the basis source raises no nm/px mismatch",
          matching["nm_per_px_mismatch"] is False
          and abs(float(matching["nm_per_px_ratio"]) - 1.0) <= 1e-12,
          f"ratio basis_nm_per_px / (L / N) = {matching['nm_per_px_ratio']:.12f}, "
          f"mismatch flag {matching['nm_per_px_mismatch']}", "ratio 1 and flag false")
    check("the end-to-end run passes the basis-vs-canvas check",
          matching["basis_ring_mismatch"] is False
          and float(matching["basis_ring_strength_ratio"])
          >= lq.BASIS_RING_STRENGTH_FRACTION
          and float(matching["basis_ring_worst_offset_px"])
          <= float(matching["basis_ring_offset_tolerance_px"]),
          f"the members reach "
          f"{100.0 * float(matching['basis_ring_strength_ratio']):.4f} % of the annulus "
          f"peak (threshold {100.0 * lq.BASIS_RING_STRENGTH_FRACTION:.0f} %), worst "
          f"member offset {float(matching['basis_ring_worst_offset_px']):.4f} px "
          f"(tolerance {float(matching['basis_ring_offset_tolerance_px']):.4f} px), the "
          f"canvas' strongest in-band peak is at "
          f"{float(matching['basis_ring_measured_angle_deg']):.4f} deg, mismatch flag "
          f"{matching['basis_ring_mismatch']}",
          "ratio >= 0.2, the offset inside its tolerance and the flag false")
    cross_out = workdir / "warn_cross_talk"
    completed_cross = run_script(
        "stm_local_q_map.py",
        [str(csv_path), "-o", str(cross_out), "--basis-from", str(lf_path),
         "--q", "1,0", "--q", "1,0.0005", "--no-figures", "-L", f"{size_nm:g}"], env)
    boundary_out = workdir / "warn_boundary"
    completed_boundary = run_script(
        "stm_local_q_map.py",
        [str(csv_path), "-o", str(boundary_out), "--basis-from", str(lf_path),
         "--q", "1,0", "--lambda-nm", f"{size_nm / 2.0:g}", "--no-figures",
         "-L", f"{size_nm:g}"], env)
    mismatch_out = workdir / "warn_nm_per_px"
    completed_mismatch = run_script(
        "stm_local_q_map.py",
        [str(csv_path), "-o", str(mismatch_out), "--basis-from", str(lf_path),
         "--q", "1,0", "--no-figures", "-L", f"{size_nm / 2.0:g}"], env)
    if completed_mismatch.returncode == 0:
        mismatch_payload = read_json(mismatch_out / "local_q_map_report.json")
        mismatch_log = (mismatch_out / "local_q_map.log").read_text()
        warnings = mismatch_payload["warnings"]
        check("a field of view that disagrees with the basis source warns",
              warnings["nm_per_px_mismatch"] is True
              and "nm/px mismatch" in mismatch_log
              and rel_error(warnings["nm_per_px_ratio"], 2.0) <= 1e-12,
              f"L / N = {warnings['canvas_nm_per_px']:.12f} nm/px against the basis "
              f"source's {warnings['basis_nm_per_px']:.12f} nm/px, ratio "
              f"{warnings['nm_per_px_ratio']:.6f}, flag "
              f"{warnings['nm_per_px_mismatch']}, log line: "
              f"{'nm/px mismatch' in mismatch_log}",
              "ratio 2, flag true and the log line present")
    else:
        check("the mismatched field-of-view branch completes", False,
              f"exit code {completed_mismatch.returncode}: "
              f"{completed_mismatch.stderr.strip()[-300:]}")
    if completed_cross.returncode == 0 and completed_boundary.returncode == 0:
        cross_payload = read_json(cross_out / "local_q_map_report.json")
        boundary_payload = read_json(boundary_out / "local_q_map_report.json")
        cross_log = (cross_out / "local_q_map.log").read_text()
        boundary_log = (boundary_out / "local_q_map.log").read_text()
        pair = cross_payload["warnings"]["cross_talk_pairs"][0]
        check("a q pair closer than 2 / lambda raises the cross-talk warning",
              cross_payload["warnings"]["cross_talk"]
              and "CROSS-TALK" in cross_log
              and abs(pair["threshold_rad_per_nm"] - 2.0 / LAMBDA_NM) < 1e-12,
              f"separation {pair['separation_rad_per_nm']:.3e} nm^-1 against the "
              f"2 / lambda threshold {pair['threshold_rad_per_nm']:.6f} nm^-1 "
              "(threshold 2 / 3 nm), warning in the log: "
              f"{'CROSS-TALK' in cross_log}",
              "cross_talk true and the log line present")
        check("lambda above L / 4 raises the boundary warning",
              boundary_payload["warnings"]["boundary_warning"]
              and "exceeds L / 4" in boundary_log,
              f"lambda = {boundary_payload['lambda_nm']:g} nm against L / 4 = "
              f"{size_nm / 4.0:g} nm, warning in the log: "
              f"{'exceeds L / 4' in boundary_log}",
              "boundary_warning true and the log line present")
    else:
        check("the two warning branches complete", False,
              f"exit codes {completed_cross.returncode} and "
              f"{completed_boundary.returncode}")


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workdir", default=None,
                        help="scratch directory (default: a fresh one under <tmp>)")
    parser.add_argument("--size", type=int, default=512,
                        help="canvas side of the synthetic image (default 512)")
    parser.add_argument("--keep", action="store_true", help="keep the scratch directory")
    parser.add_argument("--stm-lib", default=STM_LIB_DEFAULT,
                        help="STM_DataProcessing src directory used by the CLI scripts")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    sys.path.insert(0, args.stm_lib)
    if args.workdir:
        workdir = Path(args.workdir)
    else:
        workdir = Path(tempfile.mkdtemp(prefix="local_q_map_selftest_"))
    workdir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.setdefault("MPLCONFIGDIR", str(workdir / "mpl"))
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    print(f"local-q-map self-test, canvas {args.size} x {args.size}, workdir {workdir}")
    print(f"stm-lib: {args.stm_lib}")

    n_px = int(args.size)
    size_nm = FIELD_NM
    nm_per_px = size_nm / n_px
    basis = synthetic_basis(n_px)
    columns, rows = coordinates(n_px)
    table = peak_table()
    image = synthetic_image(columns, rows, basis, table)
    csv_path = workdir / "synthetic_hex.csv"
    np.savetxt(csv_path, image, delimiter=",", fmt="%.6e")
    lf_path = workdir / "report_lf.json"
    affine_path = workdir / "report_affine.json"
    write_json(lf_path, lf_report(basis, n_px, size_nm))
    write_json(affine_path, affine_report(basis, n_px, size_nm))
    basis_px_spec = ";".join(",".join(f"{value:.17g}" for value in row)
                             for row in lq.rad_px_to_px_offsets(basis, n_px))

    try:
        check_single_plane_wave(basis, n_px, nm_per_px, columns, rows)
        check_multi_peak(basis, n_px, nm_per_px, columns, rows,
                         lq.lambda_px(LAMBDA_NM, nm_per_px))
        check_basis_px_equivalence(env, workdir, csv_path, affine_path, size_nm)
        check_basis_sources(env, workdir, csv_path, size_nm, basis, n_px)
        check_gauge_law(n_px, nm_per_px, columns, rows)
        check_identities(basis, n_px, nm_per_px, columns, rows)
        check_nan_handling(env, workdir, basis, n_px, nm_per_px, columns, rows, size_nm)
        check_determinism(env, workdir, csv_path, affine_path, size_nm)
        check_size_nm_from_log(env, workdir, csv_path, basis_px_spec, n_px, lf_path,
                               size_nm)
        check_end_to_end(env, workdir, csv_path, lf_path, affine_path, n_px, size_nm)
    finally:
        shutil.rmtree(workdir / "mpl", ignore_errors=True)

    print("\n== summary ==")
    total_checks = 0
    failed_checks = 0
    passed_sections = 0
    for _index, item in enumerate(SECTIONS, start=1):
        ok = all(result for _, result, _ in item["checks"])
        passed_sections += int(ok)
        total_checks += len(item["checks"])
        failed_checks += sum(1 for _, result, _ in item["checks"] if not result)
        failures = [name for name, result, _ in item["checks"] if not result]
        print(f"  [{'PASS' if ok else 'FAIL'}] {item['title']}"
              + (f" -- failed: {failures}" if failures else ""))
    print(f"  {passed_sections}/{len(SECTIONS)} checks passed "
          f"({total_checks - failed_checks}/{total_checks} assertions)")
    if not args.keep and not args.workdir:
        shutil.rmtree(workdir, ignore_errors=True)
    elif not args.keep:
        print(f"  scratch kept: {workdir}")
    return 0 if failed_checks == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
