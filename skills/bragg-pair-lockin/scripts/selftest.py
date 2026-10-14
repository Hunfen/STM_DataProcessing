"""Self-test of the bragg-pair-lockin skill.

Four families of checks, all of them executable on this machine:

1. the synthetic identities of the engine the skill reuses -- Friedel
   ``psi_{-q} = conj psi_q``, the exact sum ``sum psi_q = sum T exp(-i q.r)``,
   the circular-roll translation law and the single-plane-wave amplitude
   identity ``B = A cos(q.r + phi)`` (the last one is what pins the factor two
   of the analytic reconstruction);
2. the products and the contract of the command line -- both spellings of a
   point, the candidate table when no point is given, the merge of an explicit
   ``-q``, the self-conjugate warning, the per-point window, the figure set and
   ``--steps-only``;
3. the real-data cross-check: the analytic route and the strict two-sided route
   are compared point by point on the 20251112 r3 points and the comparison is
   recorded in the report of the run;
4. the empty-object gates -- an empty point set, an empty canvas, an
   identically zero field -- each of which must exit with a non-zero status, plus
   the negative controls (drop the factor two, disable the automatic route
   selection, empty the candidate table, empty the module under test), each run
   on a copy in the scratch directory and each required to turn its gate red.

Nothing here mutates the skill: every mutation is applied to a copy inside the
scratch directory, and the copy is removed unless ``--keep`` is given.
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

HERE = Path(__file__).resolve().parent
SKILL_DIR = HERE.parent
SKILLS_DIR = SKILL_DIR.parent
sys.path.insert(0, str(HERE))

import bpl_lib as bpl  # noqa: E402

CLI_NAME = "stm_bragg_pair_lockin.py"
REAL_STEM = "20251112_topo0009_1mV100pA_100nm_corrected"
REAL_REL = Path("data_processing/20251110/20251112_topo0009_1mV100pA_1000nm")
ROUTE_CHECK_CORRELATION_MIN = 0.9
ROUTE_CHECK_RATIO_RANGE = (0.6, 1.4)
R3_Q_NM_INV_ABS = 17.027602458481265


# --------------------------------------------------------------------------- #
# the small report/check machinery of this self-test
# --------------------------------------------------------------------------- #
def check(results, name, ok, detail, expected="to hold"):
    """Record and print one named check."""
    results.append(
        {"name": name, "ok": bool(ok), "detail": str(detail), "expected": expected}
    )
    print(f"#   [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
    return bool(ok)


def run_cli(args, argv, cwd=None):
    """Run the command line of this skill

    in a fresh interpreter.  A fresh process matters: the negative controls
    change the source of the module under test, and an in-process import would
    keep serving the module of the first load.
    """
    command = [sys.executable, str(SKILL_DIR / "scripts" / CLI_NAME)]
    command.extend(str(item) for item in argv)
    environment = dict(os.environ)
    environment.setdefault("MPLCONFIGDIR", str(Path(args.workdir) / ".mplcache"))
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        cwd=str(cwd) if cwd else None,
        env=environment,
    )


def run_selftest(args, script, gate, workdir):
    """Run one gate of a (possibly mutated) copy of this self-test."""
    command = [
        sys.executable,
        str(script),
        "--workdir",
        str(workdir),
    ]
    if gate is not None:
        command.extend(["--gate", gate])
    environment = dict(os.environ)
    environment.setdefault("MPLCONFIGDIR", str(workdir / ".mplcache"))
    return subprocess.run(command, capture_output=True, text=True, env=environment)


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def synthetic_fixture(workdir, n=256, fov=12.0, a_nm=0.246, tilt_deg=12.0, phase=0.4):
    """A hexagonal synthetic canvas plus a lawler-fujita shaped correction report.

    The canvas carries one real cosine per reciprocal lattice class, so every
    peak keeps the phase it was given (adding the explicit ``-q`` partner of a
    cosine would cancel it).
    """
    workdir.mkdir(parents=True, exist_ok=True)
    nm_per_px = fov / float(n)
    abs_b = 4.0 * np.pi / (np.sqrt(3.0) * a_nm)
    angle = np.radians(tilt_deg)
    b1 = abs_b * np.array([np.cos(angle), np.sin(angle)])
    b2 = abs_b * np.array([np.cos(angle + np.pi / 3.0), np.sin(angle + np.pi / 3.0)])
    rows = np.arange(n, dtype=float)[:, None]
    columns = np.arange(n, dtype=float)[None, :]
    canvas = np.full((n, n), 2.0e-10)
    for h, k in ((1, 0), (0, 1), (1, -1)):
        vector = (h * b1 + k * b2) * nm_per_px
        canvas += 1.0e-10 * np.cos(vector[0] * columns + vector[1] * rows)
    envelope = 0.6 + 0.4 * np.cos(2.0 * np.pi * columns / float(n))
    for h, k in ((1 / 3, 1 / 3), (2 / 3, -1 / 3), (1 / 3, -2 / 3)):
        vector = (h * b1 + k * b2) * nm_per_px
        canvas += (
            3.0e-11 * envelope * np.cos(vector[0] * columns + vector[1] * rows + phase)
        )
    csv_path = workdir / "synthetic.csv"
    np.savetxt(csv_path, canvas, delimiter=",", fmt="%.10e")
    report = {
        "skill": "lawler-fujita-correction",
        "skill_version": "1.0",
        "canvas_px": n,
        "field_of_view_nm": fov,
        "nm_per_px": nm_per_px,
        "corrected_field_of_view_nm": fov,
        "corrected_nm_per_px": nm_per_px,
        "q_a_nm_inv": b1.tolist(),
        "q_b_nm_inv": (b2 - b1).tolist(),
        "q_c_nm_inv": (-b2).tolist(),
        "method": "lawler_fujita",
        "fallback": False,
    }
    report_path = workdir / "correction_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {
        "csv": csv_path,
        "report": report_path,
        "canvas": canvas,
        "n": n,
        "fov": fov,
        "nm_per_px": nm_per_px,
        "b1": b1,
        "b2": b2,
    }


def tiny_report(path, n=64):
    """A 64 px report for the empty-canvas fixtures (same basis, smaller canvas)."""
    fov = 4.0
    abs_b = 4.0 * np.pi / (np.sqrt(3.0) * 0.246)
    b1 = abs_b * np.array([1.0, 0.0])
    b2 = abs_b * np.array([np.cos(np.pi / 3.0), np.sin(np.pi / 3.0)])
    payload = {
        "skill": "lawler-fujita-correction",
        "canvas_px": n,
        "field_of_view_nm": fov,
        "nm_per_px": fov / n,
        "corrected_nm_per_px": fov / n,
        "corrected_field_of_view_nm": fov,
        "q_a_nm_inv": b1.tolist(),
        "q_b_nm_inv": (b2 - b1).tolist(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def real_data_paths(args):
    """The real 20251112 canvas and its correction report, or ``(None, None)``."""
    csv_path = args.csv
    report_path = args.report
    if csv_path is None:
        csv_path = os.environ.get("BPL_SELFTEST_CSV")
        report_path = report_path or os.environ.get("BPL_SELFTEST_REPORT")
    if csv_path is None:
        for base in [Path.cwd(), *SKILL_DIR.parents]:
            candidate = base / REAL_REL / f"{REAL_STEM}.csv"
            if candidate.is_file():
                csv_path = candidate
                report_path = report_path or candidate.parent / "correction_report.json"
                break
    if csv_path is None:
        return None, None
    csv_path = Path(csv_path)
    report_path = (
        Path(report_path) if report_path else csv_path.parent / "correction_report.json"
    )
    if not csv_path.is_file() or not report_path.is_file():
        return None, None
    return csv_path, report_path


# --------------------------------------------------------------------------- #
# 1. the synthetic identities
# --------------------------------------------------------------------------- #
def gate_synthetic(args, workdir, results):
    """Friedel, sum identity, translation and the plane-wave amplitude."""
    rng = np.random.default_rng(20260101)
    n = 128
    nm_per_px = 0.05
    lambda_nm = 3.0
    canvas = rng.normal(size=(n, n)) + 2.0 * np.cos(
        2.0 * np.pi * (3 * np.arange(n)[None, :] + 1 * np.arange(n)[:, None]) / n
    )
    q_rad_px = np.array([0.39269908169872414, 0.19634954084936207])  # 8 and 4 q-px
    field = bpl.local_field(canvas, q_rad_px, lambda_nm, nm_per_px)
    mirror = bpl.local_field(canvas, -q_rad_px, lambda_nm, nm_per_px)
    reference = float(np.sqrt(np.mean(np.abs(field) ** 2)))
    error = float(np.max(np.abs(mirror - np.conjugate(field))))
    check(
        results,
        "synthetic: Friedel identity psi(-q) = conj psi(q) on a random canvas",
        error <= 1e-10 * reference,
        f"max |psi(-q) - conj psi(q)| = {error:.3e} (rms |psi| = {reference:.3e})",
        "<= 1e-10 * rms |psi|",
    )
    total = complex(np.sum(field))
    exact = bpl.lq.demodulated_sum(canvas, q_rad_px)
    difference = abs(total - exact)
    check(
        results,
        "synthetic: sum psi_q = sum T exp(-i q.r) for every window width",
        difference <= 1e-10 * abs(exact),
        f"|sum psi - exact sum| = {difference:.3e}, |exact| = {abs(exact):.3e}",
        "<= 1e-10 * |exact|",
    )
    shift = (5, -3)
    rolled = np.roll(np.roll(canvas, shift[0], axis=0), shift[1], axis=1)
    field_rolled = bpl.local_field(rolled, q_rad_px, lambda_nm, nm_per_px)
    expected = np.exp(
        -1j * (q_rad_px[0] * shift[1] + q_rad_px[1] * shift[0])
    ) * np.roll(np.roll(field, shift[0], axis=0), shift[1], axis=1)
    error = float(np.max(np.abs(field_rolled - expected)))
    check(
        results,
        "synthetic: a circular roll acts as the exact translation of psi",
        error <= 1e-9 * reference,
        f"max |psi_rolled - exp(-i q.delta) psi(r - delta)| = {error:.3e}",
        "<= 1e-9 * rms |psi|",
    )

    # one plane wave exactly on the FFT grid: B must reproduce it, which is what
    # fixes the factor two of the analytic reconstruction
    n_wave = 128
    amplitude = 1.0
    phase = 0.7
    grid_q_px = 9
    wave_q_rad_px = 2.0 * np.pi * grid_q_px / n_wave
    rows = np.arange(n_wave, dtype=float)[:, None]
    columns = np.arange(n_wave, dtype=float)[None, :]
    wave = amplitude * np.cos(wave_q_rad_px * columns + 0.0 * rows + phase)
    wave_field = bpl.local_field(
        wave, np.array([wave_q_rad_px, 0.0]), lambda_nm, nm_per_px
    )
    analytic = bpl.analytic_map(wave_field, np.array([wave_q_rad_px, 0.0]))
    two_sided, imaginary = bpl.two_sided_map(
        wave, np.array([wave_q_rad_px, 0.0]), lambda_nm, nm_per_px
    )
    error_a = float(np.max(np.abs(analytic - wave)))
    error_t = float(np.max(np.abs(two_sided - wave)))
    envelope = float(np.median(np.abs(wave_field)))
    check(
        results,
        "synthetic: single plane wave -> analytic B = A cos(q.r + phi) exactly",
        error_a <= 1e-9 * amplitude
        and abs(envelope - amplitude / 2.0) <= 1e-9 * amplitude,
        f"max |2 Re[psi exp(i q.r)] - A cos| = {error_a:.3e}, "
        f"median |psi| = {envelope:.12f} vs A/2 = {amplitude / 2.0:.12f}",
        "<= 1e-9 * A",
    )
    check(
        results,
        "synthetic: the same plane wave through the strict two-sided route",
        error_t <= 1e-9 * amplitude and imaginary <= 1e-12,
        f"max |B_two_sided - A cos| = {error_t:.3e}, imaginary maximum = {imaginary:.3e}",
        "<= 1e-9 * A",
    )


# --------------------------------------------------------------------------- #
# 2. products and command-line contract
# --------------------------------------------------------------------------- #
def h5_datasets(path):
    """The dataset names of one product file, plus the shape of ``field``."""
    import h5py

    with h5py.File(path, "r") as handle:
        return sorted(handle.keys()), tuple(
            handle["field"].shape
        ) if "field" in handle else None


def gate_products(args, workdir, results):
    """Run the CLI on the synthetic canvas and audit every product it promises."""
    fixture = synthetic_fixture(workdir / "products")
    outdir = workdir / "products" / "out"
    proc = run_cli(
        args,
        [
            fixture["csv"],
            "--basis-from",
            fixture["report"],
            "--lambda-nm",
            "3",
            "--q",
            "1/3,1/3",
            "--q",
            "2/3,-1/3",
            "-o",
            outdir,
        ],
    )
    check(
        results,
        "products: the command line exits 0 on the synthetic canvas",
        proc.returncode == 0,
        f"exit {proc.returncode}",
        "exit 0",
    )
    if proc.returncode != 0:
        print(proc.stdout[-2000:])
        print(proc.stderr[-2000:])
        return
    stem = fixture["csv"].stem
    expected = [
        "report.json",
        "bragg_pair_lockin.log",
        f"{stem}_q0.h5",
        f"{stem}_q0_amplitude.png",
        f"{stem}_q0_theta.png",
        f"{stem}_q0_mask.png",
        f"{stem}_q0_B.h5",
        f"{stem}_q0_B.png",
        f"{stem}_q1.h5",
        f"{stem}_q1_B.h5",
        "B_pairs_summary.png",
        "steps_q0/step1_fft_marks.png",
        "steps_q0/step2_q_zoom.png",
        "steps_q0/step3_modulated_field.png",
        "steps_q0/step4_lowpassed_field.png",
        "steps_q0/step5_phase_coherence.png",
        "steps_q1/step1_fft_marks.png",
        "steps_q1/step4_lowpassed_field.png",
        "steps_q1/step5_phase_coherence.png",
    ]
    missing = [name for name in expected if not (outdir / name).is_file()]
    check(
        results,
        "products: per point h5 + previews, per pair B h5 + preview, report, log, "
        "summary and the four step figures of every point exist",
        not missing,
        f"{len(expected) - len(missing)}/{len(expected)} present"
        + (f", missing {missing}" if missing else ""),
        "no missing product",
    )
    names, shape = h5_datasets(outdir / f"{stem}_q0.h5")
    check(
        results,
        "products: the per-point h5 carries field/amplitude/theta/mask of the canvas",
        names == ["amplitude", "field", "mask", "theta"]
        and shape == (fixture["n"], fixture["n"]),
        f"datasets {names}, field shape {shape}",
        "field, amplitude, theta, mask at the canvas shape",
    )
    names, _ = h5_datasets(outdir / f"{stem}_q0_B.h5")
    check(
        results,
        "products: the per-pair h5 carries the real B(r) and its mask",
        names == ["B", "mask"],
        f"datasets {names}",
        "B, mask",
    )
    report = json.loads((outdir / "report.json").read_text(encoding="utf-8"))
    required_top = ["skill", "input", "lambda_nm", "q", "pairs"]
    missing_top = [key for key in required_top if key not in report]
    required_q = [
        "label",
        "h",
        "k",
        "writings",
        "q_nm_inv_abs",
        "route",
        "criteria",
        "amplitude",
    ]
    missing_q = sorted(
        {key for entry in report["q"] for key in required_q if key not in entry}
    )
    check(
        results,
        "report: top level has skill/input/lambda_nm/q/pairs and every q has "
        "label/h/k/writings/|q|/route/criteria/amplitude",
        not missing_top and not missing_q,
        f"missing top level {missing_top}, missing per point {missing_q}",
        "no missing key",
    )
    text = (outdir / "report.json").read_text(encoding="utf-8")
    both = all(
        entry["writings"]["cli_hk"] in text
        and entry["writings"]["cli_n_times_basis"] in text
        for entry in report["q"]
    ) and all(
        entry["writings"]["cli_hk"]
        in (outdir / "bragg_pair_lockin.log").read_text(encoding="utf-8")
        for entry in report["q"]
    )
    check(
        results,
        "report and log print both spellings of every point",
        both,
        "every --q and --g spelling of every point appears in report.json and in the log",
        "both spellings present",
    )
    # the figures are rendered products: a declared caption that never reached an
    # artist would not be recoverable, so the CLI asserts the read-back itself;
    # here the rendered pixels are checked for actual content
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.image as mpimg

    blanks = []
    for name in (
        "steps_q0/step1_fft_marks.png",
        "steps_q0/step2_q_zoom.png",
        "steps_q0/step3_modulated_field.png",
        "steps_q0/step4_lowpassed_field.png",
        "steps_q0/step5_phase_coherence.png",
        "B_pairs_summary.png",
    ):
        image = mpimg.imread(outdir / name)
        if not (image.std() > 0.02):
            blanks.append((name, float(image.std())))
    check(
        results,
        "figures: every step figure and the B summary has real pixel content",
        not blanks,
        f"{len(blanks)} blank figure(s) {blanks}"
        if blanks
        else "all rendered figures have content (std > 0.02)",
        "no blank figure",
    )


def gate_candidate(args, workdir, results):
    """No point given: non-zero exit, a candidate table with every required column."""
    fixture = synthetic_fixture(workdir / "candidate")
    outdir = workdir / "candidate" / "out"
    proc = run_cli(
        args,
        [
            fixture["csv"],
            "--basis-from",
            fixture["report"],
            "--lambda-nm",
            "3",
            "--stm-lib",
            args.stm_lib,
            "-o",
            outdir,
        ],
    )
    check(
        results,
        "candidate: without --q/--g the run exits with a non-zero status",
        proc.returncode != 0,
        f"exit {proc.returncode}",
        "non-zero exit",
    )
    path = outdir / "q_candidates.json"
    if not path.is_file():
        check(
            results,
            "candidate: q_candidates.json was written",
            False,
            f"{path} missing",
            "file present",
        )
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["candidates"]
    check(
        results,
        "candidate: q_candidates.json holds at least one row and at least one ring",
        len(rows) >= 1 and len(payload["rings"]) >= 1,
        f"{len(rows)} row(s), {len(payload['rings'])} ring(s) "
        f"(radii {[round(ring['radius_nm_inv_mean'], 4) for ring in payload['rings']]})",
        ">= 1 row, >= 1 ring",
    )
    required = [
        "h",
        "k",
        "writings",
        "q_nm_inv_abs",
        "f_amplitude",
        "azimuth_deg",
        "friedel_partner",
        "rational_residual_nm_inv",
        "amplitude_residual_ring",
        "cross_talk_with_selected",
        "ring_id",
    ]
    missing = sorted({key for row in rows for key in required if key not in row})
    partner_ok = all(
        {"amplitude", "asymmetry_residual", "detected_peak_found"}
        <= set(row["friedel_partner"])
        for row in rows
    )
    check(
        results,
        "candidate: every row has (h, k), both spellings, |q|, |F|, azimuth, the "
        "Friedel partner strength and residual, the rational residual, the ring "
        "amplitude residual and the cross-talk flag",
        not missing and partner_ok,
        f"missing {missing}" if missing else f"{len(rows)} rows complete",
        "no missing column",
    )
    sample = sorted(rows, key=lambda row: -row["f_amplitude"])[0]
    print(
        "#   sample row: (h, k) = "
        f"({sample['h_k_fraction'][0]}, {sample['h_k_fraction'][1]}) "
        f"[{sample['writings']['cli_hk']} | {sample['writings']['cli_n_times_basis']}]  "
        f"|q| = {sample['q_nm_inv_abs']:.4f} nm^-1  |F| = {sample['f_amplitude']:.6g}  "
        f"azimuth = {sample['azimuth_deg']:.4f} deg  "
        f"Friedel partner |F| = {sample['friedel_partner']['amplitude']:.6g} "
        f"(residual {sample['friedel_partner']['asymmetry_residual']:.4g})  "
        f"rational residual = {sample['rational_residual_nm_inv']:.4g} nm^-1  "
        f"ring amplitude residual = {sample['amplitude_residual_ring']:.4g}"
    )


def gate_list_candidates(args, workdir, results):
    """--list-candidates writes the table of a run that already has a point."""
    fixture = synthetic_fixture(workdir / "list_candidates")
    outdir = workdir / "list_candidates" / "out"
    proc = run_cli(
        args,
        [
            fixture["csv"],
            "--basis-from",
            fixture["report"],
            "--lambda-nm",
            "3",
            "--q",
            "1/3,1/3",
            "--list-candidates",
            "-o",
            outdir,
            "--no-figures",
        ],
    )
    path = outdir / "q_candidates.json"
    if proc.returncode != 0 or not path.is_file():
        check(
            results,
            "candidate: --list-candidates writes the table of a run that has points",
            False,
            f"exit {proc.returncode}, table present {path.is_file()}",
            "exit 0 with a written table",
        )
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    measured = [
        row
        for row in payload["candidates"]
        if row["cross_talk_with_selected"]["min_separation_rad_per_nm"] is not None
    ]
    same = [
        row
        for row in measured
        if row["cross_talk_with_selected"]["same_reflection"]
        or row["cross_talk_with_selected"]["violations"]
    ]
    check(
        results,
        "candidate: with --list-candidates the cross-talk column is measured "
        "against the selected point, and the selected reflection itself is "
        "reported as same_reflection instead of cross-talk",
        len(measured) == len(payload["candidates"])
        and len(same) >= 1
        and all(
            row["cross_talk_with_selected"]["cross_talk"]
            == bool(row["cross_talk_with_selected"]["violations"])
            for row in payload["candidates"]
        ),
        f"{len(measured)}/{len(payload['candidates'])} rows measured against the "
        f"selected point, {len(same)} of them coincide with it",
        "every row measured, at least one coincidence",
    )


def gate_route_selection(args, workdir, results):
    """A commensurate Bragg point stays analytic, a tiny incommensurate one does not."""
    fixture = synthetic_fixture(workdir / "routes")
    base = workdir / "routes"
    analytic_out = base / "analytic"
    proc = run_cli(
        args,
        [
            fixture["csv"],
            "--basis-from",
            fixture["report"],
            "--lambda-nm",
            "3",
            "--q",
            "1/3,1/3",
            "-o",
            analytic_out,
            "--no-figures",
        ],
    )
    report = json.loads((analytic_out / "report.json").read_text(encoding="utf-8"))
    entry = report["q"][0]
    check(
        results,
        "route: the r3-class point of the synthetic canvas takes the analytic route",
        entry["route"] == "analytic",
        f"route = {entry['route']}, 2 lambda^2 |q|^2 = {entry['criteria']['two_lambda2_q2']:.6g}, "
        f"smallest |dq| = {bpl.number_text(entry['criteria']['crosstalk_min_separation_rad_per_nm'])} "
        f"nm^-1 vs 2/lambda = {entry['criteria']['crosstalk_threshold_rad_per_nm']:.6g} nm^-1, "
        f"envelope change per carrier cycle = {entry['criteria']['slow_variation_per_cycle']:.6g}",
        f"route analytic (exit {proc.returncode})",
    )
    tiny_out = base / "tiny"
    proc = run_cli(
        args,
        [
            fixture["csv"],
            "--basis-from",
            fixture["report"],
            "--lambda-nm",
            "3",
            "--q",
            "0.004,0.003",
            "-o",
            tiny_out,
            "--no-figures",
        ],
    )
    report = json.loads((tiny_out / "report.json").read_text(encoding="utf-8"))
    entry = report["q"][0]
    check(
        results,
        "route: the tiny incommensurate point takes the strict two-sided route",
        entry["route"] == "two_sided" and entry["criteria"]["friedel_ok"] is False,
        f"route = {entry['route']}, 2 lambda^2 |q|^2 = {entry['criteria']['two_lambda2_q2']:.6g} "
        f"(needs >= {bpl.FRIEDEL_EXPONENT_MIN:.6g}), Friedel suppression = "
        f"{entry['criteria']['friedel_suppression']:.6g}, envelope change per carrier cycle = "
        f"{bpl.number_text(entry['criteria']['slow_variation_per_cycle'])}",
        "route two_sided with friedel_ok false",
    )


def gate_cli_contract(args, workdir, results):
    """R1, R3, R5, --steps-only and the required lambda."""
    fixture = synthetic_fixture(workdir / "contract")
    base = workdir / "contract"

    # R1 -- both spellings are the same point, bit for bit
    first = base / "hk"
    second = base / "n_basis"
    run_cli(
        args,
        [
            fixture["csv"],
            "--basis-from",
            fixture["report"],
            "--lambda-nm",
            "3",
            "--q",
            "1/3,1/3",
            "-o",
            first,
            "--no-figures",
        ],
    )
    run_cli(
        args,
        [
            fixture["csv"],
            "--basis-from",
            fixture["report"],
            "--lambda-nm",
            "3",
            "--g",
            "3:1,1",
            "-o",
            second,
            "--no-figures",
        ],
    )
    left = json.loads((first / "report.json").read_text(encoding="utf-8"))["q"][0]
    right = json.loads((second / "report.json").read_text(encoding="utf-8"))["q"][0]
    fields = ["h", "k", "q_nm_inv_abs", "q_px", "route"]
    same = all(left[field] == right[field] for field in fields)
    check(
        results,
        "contract: --q 1/3,1/3 and --g 3:1,1 give bit-identical h, k, |q|, q_px and route",
        same,
        "; ".join(f"{field}: {left[field]!r} == {right[field]!r}" for field in fields),
        "identical values",
    )

    # R3 -- an explicit -q is merged into one pair; a self-conjugate point warns
    merged = base / "merged"
    proc = run_cli(
        args,
        [
            fixture["csv"],
            "--basis-from",
            fixture["report"],
            "--lambda-nm",
            "3",
            "--q",
            "1/3,1/3",
            "--q",
            "-1/3,-1/3",
            "-o",
            merged,
            "--no-figures",
        ],
    )
    report = json.loads((merged / "report.json").read_text(encoding="utf-8"))
    log = (merged / "bragg_pair_lockin.log").read_text(encoding="utf-8")
    check(
        results,
        "contract: an explicit -q is merged into a single pair and the merge is announced",
        report["pairing"]["n_pairs"] == 1
        and report["pairing"]["n_points"] == 1
        and len(report["pairs"][0]["aliases"]) == 1
        and "merged into the single pair" in log,
        f"pairs {report['pairing']['n_pairs']}, points {report['pairing']['n_points']}, "
        f"aliases {len(report['pairs'][0]['aliases'])}, merge notice in the log: "
        f"{'yes' if 'merged into the single pair' in log else 'no'}",
        "one pair, one alias, notice printed",
    )
    degenerate = base / "degenerate"
    proc = run_cli(
        args,
        [
            fixture["csv"],
            "--basis-from",
            fixture["report"],
            "--lambda-nm",
            "3",
            "--q",
            "1/2,0",
            "--q",
            "0,0",
            "-o",
            degenerate,
            "--no-figures",
        ],
    )
    report = json.loads((degenerate / "report.json").read_text(encoding="utf-8"))
    log = (degenerate / "bragg_pair_lockin.log").read_text(encoding="utf-8")
    products = [Path(entry["artifacts"]["h5"]).is_file() for entry in report["q"]]
    check(
        results,
        "contract: self-conjugate points (1/2,0) and (0,0) only warn, are flagged "
        "phase_degenerate and still write their products",
        proc.returncode == 0
        and report["phase_degenerate_labels"] == ["q0", "q1"]
        and log.count("self-conjugate") >= 2
        and all(products),
        f"exit {proc.returncode}, flagged {report['phase_degenerate_labels']}, "
        f"warnings {log.count('self-conjugate')}, products {products}",
        "exit 0, both flagged, products written",
    )

    # R5 -- the window is required and can be overridden per point
    per_point = base / "lambda"
    proc = run_cli(
        args,
        [
            fixture["csv"],
            "--basis-from",
            fixture["report"],
            "--lambda-nm",
            "3",
            "--lambda-nm-label",
            "q1=6",
            "--q",
            "1/3,1/3",
            "--q",
            "2/3,-1/3",
            "-o",
            per_point,
            "--no-figures",
        ],
    )
    report = json.loads((per_point / "report.json").read_text(encoding="utf-8"))
    lambdas = report["lambda_nm_per_point"]
    check(
        results,
        "contract: --lambda-nm-label overrides the window of one point only",
        lambdas == {"q0": 3.0, "q1": 6.0}
        and [entry["lambda_nm"] for entry in report["q"]] == [3.0, 6.0],
        f"per point lambda {lambdas}",
        "q0 3.0, q1 6.0",
    )
    missing = run_cli(
        args,
        [
            fixture["csv"],
            "--basis-from",
            fixture["report"],
            "--q",
            "1/3,1/3",
            "-o",
            base / "nolambda",
            "--no-figures",
        ],
    )
    check(
        results,
        "contract: --lambda-nm is required (no default) and its absence is a non-zero exit",
        missing.returncode != 0 and "--lambda-nm" in missing.stderr,
        f"exit {missing.returncode}, stderr mentions --lambda-nm: {'--lambda-nm' in missing.stderr}",
        "non-zero exit naming --lambda-nm",
    )

    # --steps-only narrows the step figures
    narrowed = base / "steps_only"
    run_cli(
        args,
        [
            fixture["csv"],
            "--basis-from",
            fixture["report"],
            "--lambda-nm",
            "3",
            "--q",
            "1/3,1/3",
            "--q",
            "2/3,-1/3",
            "--steps-only",
            "q1",
            "-o",
            narrowed,
        ],
    )
    only = sorted(path.name for path in narrowed.glob("steps_*"))
    check(
        results,
        "contract: --steps-only q1 writes the step set of q1 and not of q0",
        (narrowed / "steps_q1" / "step1_fft_marks.png").is_file()
        and not (narrowed / "steps_q0").exists()
        and (narrowed / "report.json").is_file(),
        f"step directories {only}, report present {(narrowed / 'report.json').is_file()}",
        "only steps_q1",
    )


# --------------------------------------------------------------------------- #
# 3. the real-data cross-check
# --------------------------------------------------------------------------- #
def gate_real_data(args, workdir, results):
    """The three r3 points of the 20251112 canvas: analytic vs strict two-sided."""
    csv_path, report_path = real_data_paths(args)
    if csv_path is None:
        check(
            results,
            "real data: the 20251112 corrected canvas is available",
            False,
            "not found: pass --csv/--report, set BPL_SELFTEST_CSV, or run from the "
            "workspace that holds data_processing/20251110/"
            "20251112_topo0009_1mV100pA_1000nm/",
            "the dataset present",
        )
        return
    outdir = workdir / "real"
    proc = run_cli(
        args,
        [
            csv_path,
            "--basis-from",
            report_path,
            "--lambda-nm",
            "13",
            "--q",
            "1/3,1/3",
            "--q",
            "2/3,-1/3",
            "--q",
            "1/3,-2/3",
            "-o",
            outdir,
            "--no-figures",
        ],
        cwd=csv_path.parent,
    )
    check(
        results,
        "real data: the three r3 points run to completion on the corrected canvas",
        proc.returncode == 0,
        f"exit {proc.returncode}",
        "exit 0",
    )
    if proc.returncode != 0:
        print(proc.stdout[-3000:])
        print(proc.stderr[-3000:])
        return
    report = json.loads((outdir / "report.json").read_text(encoding="utf-8"))
    check(
        results,
        "real data: every r3 point carries |q| = 17.0276 nm^-1 and the analytic route",
        len(report["q"]) == 3
        and all(
            abs(entry["q_nm_inv_abs"] - R3_Q_NM_INV_ABS) < 1e-4 for entry in report["q"]
        )
        and all(entry["route"] == "analytic" for entry in report["q"]),
        " ".join(
            f"{entry['label']}: |q| {entry['q_nm_inv_abs']:.6f}, route {entry['route']}, "
            f"2l^2q^2 {entry['criteria']['two_lambda2_q2']:.6g}"
            for entry in report["q"]
        ),
        "three analytic points at 17.0276 nm^-1",
    )
    checks_ok = True
    details = []
    for entry in report["q"]:
        pair = entry["route_check"]
        inside = (
            pair["correlation"] >= ROUTE_CHECK_CORRELATION_MIN
            and ROUTE_CHECK_RATIO_RANGE[0]
            <= pair["rms_ratio_two_sided_over_analytic"]
            <= ROUTE_CHECK_RATIO_RANGE[1]
        )
        checks_ok = checks_ok and inside
        details.append(
            f"{entry['label']}: corr {pair['correlation']:.6f}, rms ratio "
            f"{pair['rms_ratio_two_sided_over_analytic']:.6f}, max diff "
            f"{pair['max_abs_difference']:.3e}, {pair['n_pixels']} valid pixel(s)"
        )
    check(
        results,
        "real data: the analytic and the strict two-sided route agree point by "
        f"point (correlation >= {ROUTE_CHECK_CORRELATION_MIN}, rms ratio in "
        f"{ROUTE_CHECK_RATIO_RANGE}) and the comparison is in the report",
        checks_ok,
        "; ".join(details),
        "the recorded comparison inside the gate",
    )


# --------------------------------------------------------------------------- #
# 4. the empty objects
# --------------------------------------------------------------------------- #
def empty_canvas_fixture(workdir, kind):
    """A canvas with no finite pixel (nan) or with no contrast at all (zero)."""
    workdir.mkdir(parents=True, exist_ok=True)
    n = 64
    data = np.full((n, n), np.nan) if kind == "nan" else np.zeros((n, n))
    csv_path = workdir / f"{kind}.csv"
    np.savetxt(csv_path, data, delimiter=",", fmt="%.6e")
    report_path = tiny_report(workdir / "report64.json", n)
    return csv_path, report_path


COHERENCE_TEST_N = 512
COHERENCE_TEST_NM_PER_PX = 0.05
COHERENCE_TEST_LAMBDA_NM = 1.0
COHERENCE_TEST_RAMP_RAD_PER_NM = 0.01
COHERENCE_TEST_BLOCK_NM = 1.0
#: Geometry of the profile assertions: the valid region almost fills the canvas,
#: so every lag bin up to ``max_lag`` collects a large number of pairs and the
#: floating-point residue of the circular autocorrelation (which turns ``A / N``
#: into noise once ``N`` itself is a rounding error) cannot leak into the range.
COHERENCE_TEST_MARGIN_PROFILE = 8
#: Geometry of the ramp assertion.  The estimator is a *circular* autocorrelation,
#: so beyond twice the inset of the valid region the shifted copy meets itself
#: across the canvas edge and biases ``arg C``.  This margin keeps the field of
#: view adaptive cut (``(2/3) max_lag``) below ``2 * inset``, i.e. inside the
#: wrap-free regime; the printed check reports both numbers.
COHERENCE_TEST_MARGIN_RAMP = 118
#: Geometry of the ramp-cut assertion: the valid region is inset by enough that
#: the cap (twice the inset, 7.3 nm) is far below ``fit_range`` (12.07 nm), i.e.
#: the uncapped cut would fold the canvas into ``arg C``.
COHERENCE_TEST_MARGIN_CAP = 60
#: Geometry of the pair-floor assertion: the reach of this valid region
#: (``sqrt(2) * extent``) is smaller than ``max_lag``, so the outer lag bins do
#: not contain a single real pixel pair and must be NaN.
COHERENCE_TEST_MARGIN_FLOOR = 140


def coherence_test_inputs(margin):
    """A square canvas with a rectangular valid mask and a constant phase."""
    n = COHERENCE_TEST_N
    mask = np.zeros((n, n))
    mask[margin : n - margin, margin : n - margin] = 1.0
    return np.full((n, n), 0.7), mask, n * COHERENCE_TEST_NM_PER_PX


def coherence_run(theta, mask, size_nm):
    """One estimator call with the default (field of view adaptive) ranges."""
    max_lag_nm = size_nm / np.sqrt(2.0)
    return bpl.phase_coherence(
        theta,
        mask,
        nm_per_px=COHERENCE_TEST_NM_PER_PX,
        lambda_nm=COHERENCE_TEST_LAMBDA_NM,
        max_lag_nm=max_lag_nm,
        fit_range_nm=(2.0 / 3.0) * max_lag_nm,
    )


def linear_pair_count(valid, di, dj):
    """Number of valid pixel pairs at the (non-circular) lag ``(di, dj)``.

    An independent, slice based count: no FFT is involved, so it does not share
    the convention or the floating point residue of the estimator.
    """
    n = int(valid.shape[0])
    first = valid[max(0, -di) : n - max(0, di), max(0, -dj) : n - max(0, dj)]
    second = valid[max(0, di) : n - max(0, -di), max(0, dj) : n - max(0, -dj)]
    return int(np.count_nonzero(first & second))


def eroded_valid(mask):
    """The validity mask the estimator uses: eroded by ``0.65 lambda``."""
    from scipy.ndimage import distance_transform_edt

    valid0 = np.asarray(mask, dtype=float) > 0.5
    return valid0 & (
        distance_transform_edt(valid0)
        > bpl.COHERENCE_BORDER_FACTOR
        * COHERENCE_TEST_LAMBDA_NM
        / COHERENCE_TEST_NM_PER_PX
    )


def ramp_recovery_error(mask, theta, size_nm, cut_nm=None):
    """Worst relative error of the recovered ramp for one or both injections.

    Injects ``+0.01 rad/nm`` along +x and along +y in turn and compares the change
    of ``arg C`` against the un-injected baseline; returns the worst of the four
    numbers (two axes x two injections) relative to the injected 0.01 rad/nm.
    """
    rows, columns = np.mgrid[0 : mask.shape[0], 0 : mask.shape[0]]
    nm_per_px = COHERENCE_TEST_NM_PER_PX
    max_lag_nm = size_nm / np.sqrt(2.0)
    kwargs = {
        "nm_per_px": nm_per_px,
        "lambda_nm": COHERENCE_TEST_LAMBDA_NM,
        "max_lag_nm": max_lag_nm,
        "fit_range_nm": (2.0 / 3.0) * max_lag_nm,
    }
    if cut_nm is not None:
        kwargs["ramp_cut_nm"] = float(cut_nm)
    baseline = bpl.phase_coherence(theta, mask, **kwargs).record["ramp"]
    worst = 0.0
    for injected_axis, expected in (("x", "x"), ("y", "y")):
        coordinate = (columns if injected_axis == "x" else rows) * nm_per_px
        injected = bpl.phase_coherence(
            theta + COHERENCE_TEST_RAMP_RAD_PER_NM * coordinate, mask, **kwargs
        ).record["ramp"]
        for name in ("x", "y"):
            delta = (
                injected[name]["slope_rad_per_nm"] - baseline[name]["slope_rad_per_nm"]
            )
            target = COHERENCE_TEST_RAMP_RAD_PER_NM if name == expected else 0.0
            worst = max(
                worst,
                abs(delta - target) / COHERENCE_TEST_RAMP_RAD_PER_NM,
            )
    return worst


def gate_coherence(args, workdir, results):
    """The three phase-coherence assertions of the frozen design.

    (1) a constant phase on the same mask gives ``|C| = 1``;  (2) a ramp of
    0.01 rad/nm injected along +x and along +y is recovered 1:1 against the
    un-injected baseline;  (3) a phase randomised in blocks of one lambda decays
    where the real field does not.  Each of the three has a mutation that turns
    exactly this assertion red (see MUTANTS).
    """
    theta, mask, size_nm = coherence_test_inputs(COHERENCE_TEST_MARGIN_PROFILE)
    max_lag_nm = size_nm / np.sqrt(2.0)

    # (1) constant phase -> |C| = 1 everywhere
    result = coherence_run(theta, mask, size_nm)
    magnitude = np.abs(result.profile)
    in_range = (
        (result.lag_nm > 0) & (result.lag_nm <= max_lag_nm) & np.isfinite(magnitude)
    )
    smallest_bin = (
        float(np.min(result.bin_pairs[in_range])) if np.any(in_range) else 0.0
    )
    erosion_px = 0.65 * COHERENCE_TEST_LAMBDA_NM / COHERENCE_TEST_NM_PER_PX
    measured = (
        float(np.max(np.abs(1.0 - magnitude[in_range])))
        if np.any(in_range)
        else float("inf")
    )
    check(
        results,
        "coherence: a constant phase field gives |C| = 1 at every lag of the profile",
        int(np.count_nonzero(in_range)) >= 100 and measured <= 1e-12,
        f"max |1 - |C|| = {measured:.3e} over {int(np.count_nonzero(in_range))} lag bin(s) "
        f"up to {max_lag_nm:.4f} nm (profile geometry: margin "
        f"{COHERENCE_TEST_MARGIN_PROFILE} px + {erosion_px:.4f} px erosion, so the "
        f"smallest bin still holds {smallest_bin:.4g} valid pixel pairs)",
        "<= 1e-12 over at least 100 lag bins",
    )

    # (2) injected ramp along +x / +y, read against the un-injected baseline; the
    # wrap-free geometry above is used here (the profile geometry would fold the
    # canvas edge into arg C at these lags)
    ramp_theta, ramp_mask, ramp_size_nm = coherence_test_inputs(
        COHERENCE_TEST_MARGIN_RAMP
    )
    rows, columns = np.mgrid[0:COHERENCE_TEST_N, 0:COHERENCE_TEST_N]
    baseline_result = coherence_run(ramp_theta, ramp_mask, ramp_size_nm)
    baseline = baseline_result.record["ramp"]
    inset_px = (
        COHERENCE_TEST_MARGIN_RAMP
        + 0.65 * COHERENCE_TEST_LAMBDA_NM / COHERENCE_TEST_NM_PER_PX
    )
    recovered = {}
    for axis, coordinate in (
        ("x", columns * COHERENCE_TEST_NM_PER_PX),
        ("y", rows * COHERENCE_TEST_NM_PER_PX),
    ):
        injected = coherence_run(
            ramp_theta + COHERENCE_TEST_RAMP_RAD_PER_NM * coordinate,
            ramp_mask,
            ramp_size_nm,
        ).record["ramp"]
        recovered[axis] = {
            name: float(injected[name]["slope_rad_per_nm"])
            - float(baseline[name]["slope_rad_per_nm"])
            for name in ("x", "y")
        }
    expected = COHERENCE_TEST_RAMP_RAD_PER_NM
    worst = max(
        abs(recovered["x"]["x"] - expected),
        abs(recovered["y"]["y"] - expected),
        abs(recovered["x"]["y"]),
        abs(recovered["y"]["x"]),
    )
    check(
        results,
        "coherence: an injected 0.01 rad/nm ramp along x and along y is recovered 1:1 "
        "(the reading is the slope of arg C against the un-injected baseline)",
        worst <= 1e-9,
        f"x injection -> d slope_x {recovered['x']['x']:+.12f}, d slope_y "
        f"{recovered['x']['y']:+.12f}; y injection -> d slope_x {recovered['y']['x']:+.12f}, "
        f"d slope_y {recovered['y']['y']:+.12f} rad/nm (expected {expected:+.4f} on the "
        f"injected axis and 0 on the other; the sign is the estimator convention "
        f"C(dr) = <exp(i[theta(r+dr) - theta(r)])> of the historical script); "
        f"ramp geometry: {2 * inset_px:.0f} px of mirrored inset against the cap "
        f"(cut {baseline_result.record['ramp_cut_nm']:.4f} nm = "
        f"{baseline_result.record['ramp_cut_px']} px, truncated "
        f"{baseline_result.record['ramp_cut_truncated']}); "
        f"worst |error| = {worst:.3e} rad/nm",
        "<= 1e-9 rad/nm on both axes",
    )

    # (3) block randomisation (block = lambda, offsets 0/+-2pi/3, fixed seed)
    rng = np.random.default_rng(20251112)
    block_px = COHERENCE_TEST_BLOCK_NM / COHERENCE_TEST_NM_PER_PX
    n_blocks = int(np.ceil(COHERENCE_TEST_N / block_px))
    offsets = rng.choice(
        [0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0], size=(n_blocks, n_blocks)
    )
    shift = offsets[
        np.minimum((rows / block_px).astype(int), n_blocks - 1),
        np.minimum((columns / block_px).astype(int), n_blocks - 1),
    ]
    randomised = coherence_run(theta + shift, mask, size_nm)

    def at_max_lag(res):
        return float(np.interp(max_lag_nm, res.lag_nm, np.abs(res.profile)))

    real_value, randomised_value = at_max_lag(result), at_max_lag(randomised)
    check(
        results,
        "coherence: the block-randomised phase (block = lambda) decays where the real "
        "field does not, so |C| is a real test and not a constant",
        real_value >= 0.9 and randomised_value <= 0.5 * real_value,
        f"|C|(max_lag = {max_lag_nm:.4f} nm): real {real_value:.6g}, block-randomised "
        f"{randomised_value:.6g}, ratio {randomised_value / real_value:.6g}; xi = "
        f"{result.record['xi_nm']:.6g} nm vs {randomised.record['xi_nm']:.6g} nm",
        "real >= 0.9 and block-randomised <= 0.5 * real",
    )

    # (4) the ramp cut cap, against a known truth: with the capped cut the
    # injected 0.01 rad/nm comes back to 0.0 %, while the uncapped cut the cap
    # forbids (0.9 L) is off by ~80 %.  The cap is not decoration.
    cap_theta, cap_mask, cap_size_nm = coherence_test_inputs(COHERENCE_TEST_MARGIN_CAP)
    cap_record = coherence_run(cap_theta, cap_mask, cap_size_nm).record
    err_cap = ramp_recovery_error(cap_mask, cap_theta, cap_size_nm)
    err_wide = ramp_recovery_error(
        cap_mask, cap_theta, cap_size_nm, cut_nm=0.9 * cap_size_nm
    )
    check(
        results,
        "coherence: with the capped ramp cut a known 0.01 rad/nm ramp is recovered "
        "within 5 %, while an uncapped cut at 0.9 L breaks it (the cap is not "
        "decoration)",
        err_cap <= 0.05 and err_wide >= max(2.0 * err_cap, 0.05),
        f"capped cut {cap_record['ramp_cut_nm']:.4f} nm ({cap_record['ramp_cut_px']} px, "
        f"truncated {cap_record['ramp_cut_truncated']}: cap = "
        f"{cap_record['ramp_cap_nm']:.4f} nm from inset_x {cap_record['inset_x_px']} px / "
        f"inset_y {cap_record['inset_y_px']} px, wrap free up to "
        f"{cap_record['wrap_free_lag_nm']:.4f} nm, fit_range "
        f"{(2.0 / 3.0) * cap_size_nm / np.sqrt(2.0):.4f} nm) -> worst relative "
        f"recovery error "
        f"{err_cap:.4%}; cut at 0.9 L = {0.9 * cap_size_nm:.4f} nm -> worst relative "
        f"error {err_wide:.4%} ({err_wide / max(err_cap, 1e-12):.3g} x the capped cut)",
        "capped <= 5 % and 0.9 L >= max(2 x capped, 5 %)",
    )

    # (5) the pair count has an integer floor: the FFT of a 0/1 mask leaves a
    # ~1e-10 residue where the true count is zero, so a lag whose real pair count
    # is zero must be NaN and no |C| may leave the unit disc.
    floor_theta, floor_mask, floor_size_nm = coherence_test_inputs(
        COHERENCE_TEST_MARGIN_FLOOR
    )
    floor_result = coherence_run(floor_theta, floor_mask, floor_size_nm)
    floor_magnitude = np.abs(floor_result.profile)
    nan_bins = ~np.isfinite(floor_magnitude)
    finite_bins = np.flatnonzero(np.isfinite(floor_magnitude))
    valid = eroded_valid(floor_mask)
    radius = bpl.coherence_radius_px(int(floor_mask.shape[0]))
    edges = np.arange(
        0.0, floor_result.max_lag_nm / COHERENCE_TEST_NM_PER_PX + 1.0, 1.0
    )
    index = np.digitize(radius, edges) - 1
    centre = int(floor_mask.shape[0]) // 2
    rows, columns = np.nonzero((index >= 0) & (index < len(edges) - 1))
    polluted = []
    sampled = 0
    for bin_index in np.flatnonzero(nan_bins):
        inside = np.flatnonzero(index[rows, columns] == bin_index)
        if inside.size == 0:
            continue
        take = inside[:: max(1, inside.size // 40)][:40]
        counts = [
            linear_pair_count(valid, int(rows[k]) - centre, int(columns[k]) - centre)
            for k in take
        ]
        sampled += len(take)
        if max(counts) >= 1:
            polluted.append(
                (int(bin_index), float(floor_result.lag_nm[bin_index]), max(counts))
            )
    # the finite part of the profile must be supported by real pairs (sampled,
    # independent, non-circular counts); the outermost finite bin can be a single
    # corner pixel pair, so sample the whole finite range
    finite_support = []
    for bin_index in finite_bins:
        inside = np.flatnonzero(index[rows, columns] == bin_index)
        if inside.size == 0:
            continue
        take = inside[:: max(1, inside.size // 8)][:8]
        finite_support.extend(
            linear_pair_count(valid, int(rows[k]) - centre, int(columns[k]) - centre)
            for k in take
        )
    worst_magnitude = (
        float(np.nanmax(floor_magnitude))
        if np.any(np.isfinite(floor_magnitude))
        else float("nan")
    )
    check(
        results,
        "coherence: a small valid region leaves lags without a single real pixel "
        "pair, which must be NaN and keep |C| <= 1 (the pair count has an integer "
        "floor)",
        bool(nan_bins.any())
        and worst_magnitude <= 1.0 + 1e-9
        and not polluted
        and bool(finite_support)
        and max(finite_support) >= 1,
        f"valid region {int(valid.sum())} px (inset {COHERENCE_TEST_MARGIN_FLOOR} px + "
        f"erosion); profile {int(finite_bins.size)} finite / "
        f"{int(np.count_nonzero(nan_bins))} NaN bin(s), last finite lag "
        f"{floor_result.record['profile_last_lag_nm']:.4f} nm of "
        f"max_lag {floor_result.max_lag_nm:.4f} nm, max |C| = {worst_magnitude:.6g}; "
        f"{sampled} pixel(s) sampled from the NaN bins with an independent "
        f"non-circular pair count -> {len(polluted)} NaN bin(s) do contain a real pair "
        f"(worst {polluted[0][2] if polluted else 0} pairs)"
        + (
            f"; the finite bins hold up to {max(finite_support)} real pair(s) over "
            f"{len(finite_support)} sampled pixel(s)"
            if finite_support
            else "; no finite bin at all"
        ),
        "NaN bins exist, no |C| > 1, no NaN bin holds a real pair, "
        "the finite bins hold at least one",
    )

    # (6) the strain constant is a formula, not a typed number: what the report
    # records must be 2 beta / (sqrt(3) a_nm) of the recorded beta and a_nm, and
    # the divisor must be 20 A.  The command line writes this very block into
    # report.json (the schema check of the task reads it back from there).
    config = bpl.coherence_config(
        30.0, COHERENCE_TEST_LAMBDA_NM, COHERENCE_TEST_NM_PER_PX
    )
    recorded_a = float(config["A_per_strain_angstrom_inv"])
    recorded_divisor = float(config["strain_divisor_per_nm"])
    recomputed_a = (
        2.0 * float(config["beta"]) / (np.sqrt(3.0) * float(config["a_nm"]) * 10.0)
    )
    deviation_a = abs(recorded_a - recomputed_a) / abs(recomputed_a)
    deviation_divisor = abs(recorded_divisor - 20.0 * recomputed_a) / (
        20.0 * recomputed_a
    )
    check(
        results,
        "coherence: the recorded strain constant is recomputed from the recorded "
        "beta and a_nm (A = 2 beta / (sqrt(3) a0), divisor = 20 A), not a typed "
        "number",
        deviation_a <= 1e-12 and deviation_divisor <= 1e-12,
        f"a_nm = {config['a_nm']}, beta = {config['beta']}: recorded A = "
        f"{float(recorded_a):.15f} vs recomputed A = {float(recomputed_a):.15f} "
        f"(relative {deviation_a:.3e}); recorded divisor = "
        f"{float(recorded_divisor):.15f} vs 20*A = {float(20.0 * recomputed_a):.15f} "
        f"(relative {deviation_divisor:.3e})",
        "both relative deviations <= 1e-12",
    )


def gate_empty_input(args, workdir, results):
    """A canvas without a single finite pixel must not be analysed."""
    csv_path, report_path = empty_canvas_fixture(workdir / "empty_input", "nan")
    proc = run_cli(
        args,
        [
            csv_path,
            "--basis-from",
            report_path,
            "--lambda-nm",
            "1",
            "--q",
            "1/3,1/3",
            "-o",
            workdir / "empty_input" / "out",
            "--no-figures",
        ],
    )
    check(
        results,
        "empty object: an all-NaN canvas exits with a non-zero status",
        proc.returncode != 0,
        f"exit {proc.returncode}: {proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else proc.stderr.strip()[-160:]}",
        "non-zero exit",
    )


def gate_empty_field(args, workdir, results):
    """A canvas with zero contrast leaves no field to reconstruct."""
    csv_path, report_path = empty_canvas_fixture(workdir / "empty_field", "zero")
    proc = run_cli(
        args,
        [
            csv_path,
            "--basis-from",
            report_path,
            "--lambda-nm",
            "1",
            "--q",
            "1/3,1/3",
            "-o",
            workdir / "empty_field" / "out",
            "--no-figures",
        ],
    )
    check(
        results,
        "empty object: an identically zero canvas exits with a non-zero status",
        proc.returncode != 0,
        f"exit {proc.returncode}: {proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else proc.stderr.strip()[-160:]}",
        "non-zero exit",
    )


def gate_empty_points(args, workdir, results):
    """No point given and no candidate found: the table is empty and the run stops."""
    csv_path, report_path = empty_canvas_fixture(workdir / "empty_points", "zero")
    outdir = workdir / "empty_points" / "out"
    proc = run_cli(
        args, [csv_path, "--basis-from", report_path, "--lambda-nm", "1", "-o", outdir]
    )
    path = outdir / "q_candidates.json"
    rows = (
        json.loads(path.read_text(encoding="utf-8"))["candidates"]
        if path.is_file()
        else None
    )
    check(
        results,
        "empty object: a canvas without a detectable peak exits with a non-zero "
        "status and does not pretend to have candidates",
        proc.returncode != 0
        and rows == []
        and "candidate table is empty" in proc.stdout,
        f"exit {proc.returncode}, q_candidates.json rows {rows}, "
        f"error line {'present' if 'candidate table is empty' in proc.stdout else 'missing'}",
        "non-zero exit with an empty candidate table",
    )


# --------------------------------------------------------------------------- #
# the negative controls (mutation on a copy in the scratch directory)
# --------------------------------------------------------------------------- #
def patch_source(path, old, new, label):
    """Replace one exact occurrence; a missing pattern is an error, not a pass."""
    text = Path(path).read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise RuntimeError(
            f"negative control {label}: the pattern to mutate occurs "
            f"{text.count(old)} time(s), expected exactly one"
        )
    Path(path).write_text(text.replace(old, new), encoding="utf-8")


MUTANTS = {
    "drop-x2": {
        "gate": "synthetic",
        "what": "the factor two of the analytic reconstruction is dropped",
        "apply": lambda lib: patch_source(
            lib, "ANALYTIC_FACTOR = 2.0", "ANALYTIC_FACTOR = 1.0", "drop-x2"
        ),
    },
    "force-analytic": {
        "gate": "route-selection",
        "what": "the automatic route selection is disabled (always analytic)",
        "apply": lambda lib: patch_source(
            lib,
            '    if record["friedel_ok"] and record["crosstalk_ok"] and record["slow_ok"]:\n'
            "        return ROUTE_ANALYTIC\n"
            "    return ROUTE_TWO_SIDED",
            "    return ROUTE_ANALYTIC  # MUTANT: automatic route selection disabled",
            "force-analytic",
        ),
    },
    "empty-candidates": {
        "gate": "candidate",
        "what": "the detector result is emptied, so the candidate table has no row",
        "apply": lambda lib: patch_source(
            lib,
            "    peaks = list(detection.peaks)",
            "    peaks = []  # MUTANT",
            "empty-candidates",
        ),
    },
    "coherence-normalisation": {
        "gate": "coherence",
        "what": "the coherence is normalised by the number of valid pixels instead of "
        "the number of valid pixel pairs, so |C| decays with the lag even for a "
        "constant phase",
        "apply": lambda lib: patch_source(
            lib,
            "        c = np.where(pair_count >= COHERENCE_PAIR_FLOOR, "
            "numerator / pair_count, np.nan)",
            "        c = np.where(\n"
            "            pair_count >= COHERENCE_PAIR_FLOOR,\n"
            "            numerator / float(np.count_nonzero(valid)),\n"
            "            np.nan,\n"
            "        )  # MUTANT: normalised by the pixel count",
            "coherence-normalisation",
        ),
        "expect_fail": "constant phase",
    },
    "coherence-ramp-units": {
        "gate": "coherence",
        "what": "the 1D ramp cut is built in pixels instead of nm (the unit slip the "
        "historical script suffered from)",
        "apply": lambda lib: patch_source(
            lib,
            "    lags = np.arange(length) * float(nm_per_px)",
            "    lags = np.arange(length) * 1.0  # MUTANT: px instead of nm",
            "coherence-ramp-units",
        ),
        "expect_fail": "recovered 1:1",
    },
    "coherence-flat-bins": {
        "gate": "coherence",
        "what": "every pixel lands in the first lag bin, so the radial profile keeps no "
        "lag dependence at all",
        "apply": lambda lib: patch_source(
            lib,
            "    index = np.digitize(radius_px, edges) - 1",
            "    index = np.zeros(radius_px.shape, dtype=int)  # MUTANT",
            "coherence-flat-bins",
        ),
        "expect_fail": "block-randomised",
    },
    "coherence-no-cut-cap": {
        "gate": "coherence",
        "what": "the ramp cut cap is removed, so the cut runs to fit_range and the "
        "circular autocorrelation folds the far side of the canvas into arg C",
        "apply": lambda lib: patch_source(
            lib,
            "        cut_nm = min(float(fit_range_nm), cap_nm)",
            "        cut_nm = float(fit_range_nm)  # MUTANT: cap removed",
            "coherence-no-cut-cap",
        ),
        "expect_fail": "capped ramp cut",
    },
    "coherence-pair-floor": {
        "gate": "coherence",
        "what": "the integer pair-count floor is removed (the floor constant drops "
        "to 0.0, which is the previous N > 0 behaviour), so the ~1e-10 FFT residue "
        "of the mask passes for pairs and the NaN of a zero-count pixel poisons the "
        "whole lag bin",
        "apply": lambda lib: patch_source(
            lib,
            "COHERENCE_PAIR_FLOOR = 0.5",
            "COHERENCE_PAIR_FLOOR = 0.0  # MUTANT: no integer floor",
            "coherence-pair-floor",
        ),
        "expect_fail": "integer floor",
    },
    "coherence-hardcoded-strain-constant": {
        "gate": "coherence",
        "what": "the strain constant is written down as a seven digit number again "
        "(A = 14.081746 nm^-1, the value of the first addendum before its "
        "arithmetic was corrected) instead of being computed from beta and a_nm",
        "apply": lambda lib: patch_source(
            lib,
            "STRAIN_A_NM_INV = 2.0 * STRAIN_BETA / (math.sqrt(3.0) * STRAIN_A0_NM)",
            "STRAIN_A_NM_INV = 14.081746  # MUTANT: hardcoded (wrong) constant",
            "coherence-hardcoded-strain-constant",
        ),
        "expect_fail": "recomputed from the recorded",
    },
    "empty-module": {
        "gate": "synthetic",
        "mutant_gate": None,
        "what": "the module under test is emptied",
        "apply": lambda lib: Path(lib).write_text("", encoding="utf-8"),
    },
}


def negative_controls(args, workdir, results):
    """Every mutation must turn its own gate red, on a copy inside the scratch dir."""
    root = workdir / "mutants"
    root.mkdir(parents=True, exist_ok=True)
    for name, mutant in MUTANTS.items():
        target = root / name / SKILLS_DIR.name
        if target.exists():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(SKILLS_DIR, target, symlinks=True)
        script = target / SKILL_DIR.name / "scripts" / "selftest.py"
        lib = target / SKILL_DIR.name / "scripts" / "bpl_lib.py"
        try:
            mutant["apply"](lib)
        except RuntimeError as error:
            check(
                results,
                f"negative control {name}: the mutation was applied",
                False,
                str(error),
                "mutation applied",
            )
            continue
        pristine = run_selftest(
            args, HERE / "selftest.py", mutant["gate"], root / f"{name}-pristine-work"
        )
        mutant_gate = mutant.get("mutant_gate", mutant["gate"])
        mutant_run = run_selftest(
            args, script, mutant_gate, root / f"{name}-mutant-work"
        )
        gate_name = mutant_gate or "the default suite"
        expect_fail = mutant.get("expect_fail")
        failed_lines = [
            line for line in mutant_run.stdout.splitlines() if "[FAIL]" in line
        ]
        specific = expect_fail is None or any(
            expect_fail in line for line in failed_lines
        )
        detail = (
            f"pristine exit {pristine.returncode}, mutated exit {mutant_run.returncode}"
        )
        if expect_fail is not None:
            detail += f"; the FAIL line names {expect_fail!r}: " + (
                "yes" if specific else "NO"
            )
        check(
            results,
            f"negative control {name}: {mutant['what']} -> {gate_name} goes red "
            "(and the same gate passes on the unmutated tree)"
            + (
                f", and the red check is the one named {expect_fail!r}"
                if expect_fail
                else ""
            ),
            pristine.returncode == 0 and mutant_run.returncode != 0 and specific,
            detail,
            "pristine 0, mutated non-zero"
            + (f", failing check named {expect_fail!r}" if expect_fail else ""),
        )
        shutil.rmtree(target.parent, ignore_errors=True)


# --------------------------------------------------------------------------- #
# runner
# --------------------------------------------------------------------------- #
GATES = {
    "synthetic": gate_synthetic,
    "products": gate_products,
    "candidate": gate_candidate,
    "list-candidates": gate_list_candidates,
    "route-selection": gate_route_selection,
    "cli-contract": gate_cli_contract,
    "coherence": gate_coherence,
    "real-data": gate_real_data,
    "empty-input": gate_empty_input,
    "empty-field": gate_empty_field,
    "empty-points": gate_empty_points,
}
DEFAULT_ORDER = [
    "synthetic",
    "products",
    "candidate",
    "list-candidates",
    "route-selection",
    "cli-contract",
    "coherence",
    "real-data",
    "empty-input",
    "empty-field",
    "empty-points",
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--workdir",
        default=None,
        help="scratch directory (default: <tmp>/stm-bragg-pair-lockin-selftest)",
    )
    parser.add_argument(
        "--gate",
        default=None,
        choices=sorted(GATES),
        help="run a single named gate and stop (used by the negative controls)",
    )
    parser.add_argument(
        "--csv", default=None, help="corrected canvas of the real-data cross-check"
    )
    parser.add_argument(
        "--report", default=None, help="correction report of that canvas"
    )
    parser.add_argument(
        "--no-real-data",
        action="store_true",
        help="SKIP the real-data cross-check (the gate is then not green, it is skipped and reported as skipped)",
    )
    parser.add_argument(
        "--keep", action="store_true", help="keep the scratch directory"
    )
    parser.add_argument(
        "--stm-lib",
        default=bpl.STM_LIB_DEFAULT,
        help="STM_DataProcessing src directory of the candidate detector",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    workdir = (
        Path(args.workdir)
        if args.workdir
        else Path(tempfile.gettempdir()) / "stm-bragg-pair-lockin-selftest"
    )
    if not args.gate and workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(workdir / ".mplcache"))
    args.workdir = workdir
    lib = Path(bpl.lq.__file__)
    print("# selftest.py - analytic identities of the engine this skill reuses, the")
    print(
        "# product and command-line contract, the real-data route cross-check and the"
    )
    print("# empty-object gates with their negative controls")
    print(f"# skill under test: {SKILL_DIR}")
    print(f"# engine: {lib} (exists: {lib.is_file()})")
    print(f"# scratch directory: {workdir}")

    results = []
    skipped = []
    if args.gate:
        GATES[args.gate](args, workdir, results)
    else:
        for name in DEFAULT_ORDER:
            if name == "real-data" and args.no_real_data:
                skipped.append(name)
                print(
                    f"#   [SKIP] {name}: --no-real-data was given on the command line"
                )
                continue
            print(f"# gate {name}")
            GATES[name](args, workdir, results)
        print("# gate negative-controls")
        negative_controls(args, workdir, results)

    failed = [item for item in results if not item["ok"]]
    print("")
    print(
        f"# {len(results) - len(failed)}/{len(results)} check(s) passed"
        + (
            f", {len(skipped)} gate(s) SKIPPED ({', '.join(skipped)})"
            if skipped
            else ""
        )
    )
    for item in failed:
        print(
            f"# FAILED: {item['name']} -- {item['detail']} (expected: {item['expected']})"
        )
    if not failed:
        print("# every check passed")
    if not args.keep and not args.gate:
        shutil.rmtree(workdir, ignore_errors=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
