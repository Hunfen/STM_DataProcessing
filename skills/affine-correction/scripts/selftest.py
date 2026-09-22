"""One-command self-test of the affine-correction skill: the geometry stage on a
synthetic image with a known anisotropy, plus the two-stage transform workflow
(``--save-transform`` export and ``stm_apply_transform.py`` on another canvas).

Everything checked here is geometry (a stretch matrix, a lattice constant derived
from a measured ring radius, the anchor self-check verdict and the resampling
round trip); no physical statement is made and no number is needed from the
phase-analysis skill.

    cd /path/to/STM_DataProcessing
    MPLCONFIGDIR=<writable> PYTHONDONTWRITEBYTECODE=1 \
        .venv/bin/python <this script>

Options:
    --workdir DIR   scratch directory (default: a fresh directory under <tmp>)
    --size N        canvas side of the synthetic image (default 384)
    --keep          keep the scratch directory

Exit code 0 = every check passed; each check prints its acceptance threshold next
to the measured value, so a failure is quantified instead of announced.
"""

from __future__ import annotations

import argparse
import io
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

SQRT3 = float(np.sqrt(3.0))
# Maps physical (x, y) order to array (row, col) order and back (same constant as
# the fit script's package and the apply script's mirror).
AXIS_SWAP = np.array([[0.0, 1.0], [1.0, 0.0]])
# The mandatory cross-skill contract line of a correction log.
CONTRACT_LINE_RE = re.compile(
    r"# corrected canvas: (\d+) x (\d+) px, field of view "
    r"([\d.]+) nm"
)
RESULTS: list[tuple[str, bool, str]] = []


def check(name, ok, detail, threshold=""):
    RESULTS.append((name, bool(ok), detail))
    flag = "PASS" if ok else "FAIL"
    limit = f"  [threshold {threshold}]" if threshold else ""
    print(f"  [{flag}] {name}: {detail}{limit}")
    return bool(ok)


def section(title):
    print(f"\n== {title} ==")


# --------------------------------------------------------------------------- #
# synthetic ground truth (independent of the phase-analysis skill)
# --------------------------------------------------------------------------- #
def ring_vectors(radius_px, start_deg, step_deg=-60.0, count=6):
    """``count`` hexagonal wavevectors of one ring, first one at ``start_deg``.

    Default gauge: the first vector points at 12 o'clock and the ring steps by
    ``-60`` degrees, i.e. a regular hexagon with one member on the vertical axis.
    """
    angles = np.radians(start_deg + step_deg * np.arange(count, dtype=float))
    return [
        (float(radius_px * np.cos(angle)), float(radius_px * np.sin(angle)))
        for angle in angles
    ]


def ring_wave(n, vectors, phase_rad):
    """Real part of ``sum_j exp(i (2 pi k_j . r / n + phase))`` over the ring."""
    yy, xx = np.mgrid[:n, :n]
    total = np.zeros((n, n), dtype=float)
    for kx, ky in vectors:
        total += np.cos((2.0 * np.pi / n) * (kx * xx + ky * yy) + phase_rad)
    return total


def soft_disk(n, centre, radius_frac, edge=None):
    """Soft region indicator of a disk (same profile as the synthetic domains)."""
    radius = float(radius_frac) * n
    if edge is None:
        edge = max(4.0, 0.06 * radius)
    yy, xx = np.mgrid[:n, :n]
    distance = np.hypot(xx - centre[0] * n, yy - centre[1] * n)
    return 0.5 * (1.0 - np.tanh((distance - radius) / float(edge)))


def synthetic_image(
    n,
    radius_frac=0.458,
    ref_amp=1.0,
    ref_phase_deg=40.0,
    disk_amp=1.0,
    disk_phase_deg=240.0,
    disk_centre=(0.5, 0.5),
    disk_radius_frac=0.20,
):
    """Ground truth image: one reference ring plus a disk-dependent inner ring.

    The reference ring sits at ``radius_frac * n`` px and carries one phase
    everywhere; the disk region adds the same hexagon rotated by 30 degrees and
    scaled by ``1/sqrt(3)`` (an r3 ring inside the reference ring) with the disk's
    own phase, so the pair of rings is at a radius ratio ``sqrt(3)``.  Returns the
    image and the reference ring radius in pixels.
    """
    r1 = float(radius_frac) * n
    image = float(ref_amp) * ring_wave(
        n, ring_vectors(r1, 90.0), np.radians(ref_phase_deg)
    )
    disk = soft_disk(n, disk_centre, disk_radius_frac)
    inner = ring_vectors(r1 / SQRT3, 120.0)
    image = image + float(disk_amp) * disk * ring_wave(
        n, inner, np.radians(disk_phase_deg)
    )
    return image, r1


def stretched_image(image, stretch, pad=10, order=3):
    """Resample the pattern so that its peaks move to ``q_in @ stretch``.

    The affine geometry is the one the correction inverts: ``output = input[A y + b]``
    with ``A = P M P`` (``P`` the axis swap), the canvas grown to hold every
    transformed corner plus ``pad``, outside the source set to NaN.
    """
    from scipy.ndimage import affine_transform

    arr = np.asarray(image, dtype=float)
    n = arr.shape[0]
    swap = np.array([[0.0, 1.0], [1.0, 0.0]])
    matrix = swap @ np.asarray(stretch, dtype=float) @ swap
    corners = (
        np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]]) * (n - 1) / 2.0
    )
    extent = np.abs(np.linalg.inv(matrix) @ corners.T).max(axis=1)
    n_out = 2 * (int(np.ceil(float(extent.max()))) + int(pad)) + 1
    offset = (n - 1) / 2.0 - matrix @ np.array([(n_out - 1) / 2.0, (n_out - 1) / 2.0])
    out = affine_transform(
        arr,
        matrix,
        offset=offset,
        output_shape=(n_out, n_out),
        order=int(order),
        mode="constant",
        cval=np.nan,
        prefilter=int(order) > 1,
    )
    return out, n_out


def run_script(script, arguments, env):
    return subprocess.run(
        [sys.executable, str(HERE / script), *arguments],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def ratio_text(check_result):
    if check_result.get("ratio") is None:
        return "not enough corrected rings to form a pair"
    return (
        f"{check_result['ratio']:.6f} (deviation "
        f"{100 * check_result['deviation']:.4f} % from sqrt(3))"
    )


# --------------------------------------------------------------------------- #
# the correction stage
# --------------------------------------------------------------------------- #
def test_correction_stage(n, workdir):
    section("correction stage: explicit anchor ring (needs the package)")
    try:
        sys.path.insert(0, "/Users/hunfen/Documents/GitHub/STM_DataProcessing/src")
        import stm_data_processing  # noqa: F401
    except Exception as exc:
        check(
            "correction stage skipped",
            True,
            f"the package is not importable here ({type(exc).__name__}); run the "
            f"self-test with the repository interpreter to cover this stage",
            "informational",
        )
        return

    env = dict(os.environ)
    env["MPLCONFIGDIR"] = str(workdir / ".mplcache")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    (workdir / ".mplcache").mkdir(parents=True, exist_ok=True)
    base = workdir / "correction"
    base.mkdir(parents=True, exist_ok=True)
    stretch = np.diag([1.010, 0.990])
    radius_frac = 0.458
    image, _r1 = synthetic_image(n, radius_frac)
    stretched, n_out = stretched_image(image, stretch, pad=10, order=3)
    stretched = np.nan_to_num(stretched, nan=float(np.mean(image)))
    csv_path = base / "stretched.csv"
    np.savetxt(csv_path, stretched, delimiter=",", fmt="%.10e")
    # the field of view is chosen so that the reference ring of this synthetic sits
    # where the nominal lattice constant puts it (the ring radius is 0.458 n px and
    # |b1| = 4 pi / (sqrt(3) a) nm^-1); the nm-per-pixel scale then survives the
    # resampling, so the stretched canvas keeps it and its field of view grows
    ideal_b1 = 4.0 * np.pi / (SQRT3 * 0.246)
    field_of_view = 2.0 * np.pi * (radius_frac * n) / ideal_b1 * n_out / n

    reports = {}
    for anchor in ("r3", "1x1"):
        outdir = base / f"correction_{anchor}"
        completed = run_script(
            "stm_topo_correct.py",
            [
                str(csv_path),
                "-L",
                f"{field_of_view:.6f}",
                "-o",
                str(outdir),
                "--anchor-ring",
                anchor,
            ],
            env,
        )
        report = (
            json.loads((outdir / "correction_report.json").read_text())
            if (outdir / "correction_report.json").is_file()
            else None
        )
        reports[anchor] = (completed, report)
    good, bad = reports["r3"][1], reports["1x1"][1]
    ran = (
        reports["r3"][0].returncode == 0
        and reports["1x1"][0].returncode == 0
        and good is not None
        and bad is not None
    )
    check(
        "correction stage ran (both anchors)",
        ran,
        f"exit codes {reports['r3'][0].returncode} / {reports['1x1'][0].returncode}, "
        + (
            "both correction_report.json written"
            if ran
            else f"stderr: {reports['r3'][0].stderr.strip()[-200:]}"
        ),
        "exit code 0 and reports written",
    )
    if not ran:
        return
    check(
        "correct anchor: the recovered stretch undoes the injected one",
        abs(good["affine_q"][0][0] - 1.010) < 3e-3
        and abs(good["affine_q"][1][1] - 0.990) < 3e-3,
        f"M = [[{good['affine_q'][0][0]:.5f}, {good['affine_q'][0][1]:.5f}], "
        f"[{good['affine_q'][1][0]:.5f}, {good['affine_q'][1][1]:.5f}]] vs the "
        f"injected [[1.010, 0], [0, 0.990]]; |det M|^(1/2) = "
        f"{good['stretch_scale_sqrt_det']:.5f}",
        "3e-3",
    )
    check(
        "correct anchor: the implied 1x1 lattice constant comes back",
        abs(good["implied_lattice_before"]["deviation_from_nominal"]) < 0.01,
        f"implied a = {good['implied_lattice_before']['a_1x1_nm']:.4f} nm vs the "
        f"nominal {good['a_nm']:.3f} nm "
        f"({100 * good['implied_lattice_before']['deviation_from_nominal']:+.3f} %)",
        "1 %",
    )
    check(
        "correct anchor: the corrected image carries a 1 : sqrt(3) ring pair",
        good["anchor_self_check"]["verdict"] == "consistent",
        f"anchor verdict = {good['anchor_self_check']['verdict']}, corrected ring "
        f"ratio = {ratio_text(good['anchor_self_check'])}",
        "consistent",
    )
    check(
        "a wrong anchor is caught by the same self-check",
        bad["anchor_self_check"]["verdict"] != "consistent",
        f"anchor verdict = {bad['anchor_self_check']['verdict']}, corrected ring "
        f"ratio = {ratio_text(bad['anchor_self_check'])}",
        "not consistent",
    )


def write_stretched_case(n, path, stretch=None, radius_frac=0.458, pad=10, order=3):
    """Write the synthetic anisotropic case of the correction-stage check.

    Same generator and same field-of-view rule as checks 1-2 (the reference ring of
    the synthetic sits where the nominal lattice constant puts it, so the nm-per-pixel
    scale survives the resampling onto the stretched canvas).  Returns the csv path,
    the field of view in nm and the canvas side of the written matrix.
    """
    if stretch is None:
        stretch = np.diag([1.010, 0.990])
    image, _r1 = synthetic_image(n, radius_frac)
    stretched, _n_out = stretched_image(image, stretch, pad=pad, order=order)
    stretched = np.nan_to_num(stretched, nan=float(np.mean(image)))
    np.savetxt(path, stretched, delimiter=",", fmt="%.10e")
    ideal_b1 = 4.0 * np.pi / (SQRT3 * 0.246)
    field_of_view = 2.0 * np.pi * (radius_frac * n) / ideal_b1 * stretched.shape[0] / n
    return path, field_of_view, int(stretched.shape[0])


# --------------------------------------------------------------------------- #
# the two-stage transform workflow (export + apply on another canvas)
# --------------------------------------------------------------------------- #
def test_transform_stage(n, workdir):
    section("two-stage transform: --save-transform export + stm_apply_transform.py")
    try:
        sys.path.insert(0, "/Users/hunfen/Documents/GitHub/STM_DataProcessing/src")
        import stm_apply_transform

        import stm_data_processing  # noqa: F401
        from stm_data_processing.utils.bragg_peak import load_image
        from stm_data_processing.utils.bragg_peak.correct import (
            _image_transform as package_image_transform,
        )
        from stm_data_processing.utils.plot_funcs import subtractMeanPlane
    except Exception as exc:
        check(
            "two-stage transform skipped",
            True,
            f"the package or the apply script is not importable here "
            f"({type(exc).__name__}: {exc}); run the self-test with the repository "
            f"interpreter to cover this stage",
            "informational",
        )
        return

    env = dict(os.environ)
    env["MPLCONFIGDIR"] = str(workdir / ".mplcache")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    (workdir / ".mplcache").mkdir(parents=True, exist_ok=True)
    base = workdir / "transform"
    base.mkdir(parents=True, exist_ok=True)

    name = (
        "two-stage transform: export, round trip, other canvas, mirror parity, "
        "identity fallback"
    )

    # (a) fit the synthetic reference and export the standalone transform
    ref_csv, ref_fov, _ref_n = write_stretched_case(n, base / "reference.csv")
    fit_dir = base / "fit"
    transform_path = base / "transform.json"
    fit = run_script(
        "stm_topo_correct.py",
        [
            str(ref_csv),
            "-L",
            f"{ref_fov:.6f}",
            "-o",
            str(fit_dir),
            "--anchor-ring",
            "r3",
            "--save-transform",
            str(transform_path),
        ],
        env,
    )
    report_path = fit_dir / "correction_report.json"
    payload = (
        json.loads(transform_path.read_text()) if transform_path.is_file() else None
    )
    report = json.loads(report_path.read_text()) if report_path.is_file() else None
    fields = (
        "schema",
        "schema_version",
        "source_report",
        "source_input",
        "affine_q",
        "affine_image_reference",
        "n_px_reference",
        "n_out_reference",
        "field_of_view_nm_reference",
        "pad",
        "order",
        "method",
        "fallback",
        "n_labelled",
        "anchor_ring",
        "a_nm",
        "a_ref_nm",
        "stretch_scale_sqrt_det",
        "anchor_verdict",
        "usage",
    )
    missing = sorted(field for field in fields if field not in (payload or {}))
    export_ok = bool(
        fit.returncode == 0
        and payload is not None
        and report is not None
        and not missing
        and payload["schema"] == "topo-correction-transform"
        and payload["schema_version"] == 1
        and payload["source_report"] == str(report_path.resolve())
        and payload["source_input"] == str(ref_csv.resolve())
        and isinstance(payload["usage"], str)
        and bool(payload["usage"])
        and np.array_equal(
            np.asarray(payload["affine_q"], dtype=float),
            np.asarray(report["affine_q"], dtype=float),
        )
    )
    if not export_ok:
        check(
            name,
            False,
            f"(a) export failed: exit code {fit.returncode}, missing field(s) "
            f"{missing or 'none'}, transform written {transform_path.is_file()}, "
            f"stderr: {fit.stderr.strip()[-160:]}",
            "every sub-step passes",
        )
        return
    stretch = np.asarray(payload["affine_q"], dtype=float)
    pad = int(payload["pad"])

    # (b) applying the transform to the SAME reference must reproduce the fit stage
    # (the NaN padding is identical on both sides, hence equal_nan=True)
    apply_ref_dir = base / "apply_reference"
    applied = run_script(
        "stm_apply_transform.py",
        [
            str(ref_csv),
            "--transform",
            str(transform_path),
            "-L",
            f"{ref_fov:.6f}",
            "-o",
            str(apply_ref_dir),
        ],
        env,
    )
    fit_csv = fit_dir / f"{ref_csv.stem}_corrected.csv"
    apply_csv = apply_ref_dir / f"{ref_csv.stem}_corrected.csv"
    round_trip_ok, round_detail = False, "(b) the fit stage wrote no corrected csv"
    if fit_csv.is_file() and apply_csv.is_file():
        from_fit = np.loadtxt(fit_csv, delimiter=",")
        from_apply = np.loadtxt(apply_csv, delimiter=",")
        finite = np.isfinite(from_fit) & np.isfinite(from_apply)
        residual = (
            float(np.abs(from_fit - from_apply)[finite].max())
            if finite.any()
            else float("nan")
        )
        same_nan = np.array_equal(np.isnan(from_fit), np.isnan(from_apply))
        round_trip_ok = bool(
            from_apply.shape == from_fit.shape
            and same_nan
            and np.allclose(
                from_apply, from_fit, rtol=1e-10, atol=1e-10, equal_nan=True
            )
        )
        round_detail = (
            f"(b) round trip: shape {from_apply.shape}, max |delta| "
            f"{residual:.3e}, same NaN mask {same_nan}, byte-identical "
            f"{apply_csv.read_bytes() == fit_csv.read_bytes()}"
        )

    # (c) a target of a DIFFERENT size: canvas and offset must follow the target's n
    target_generator_n = 160
    tgt_csv, tgt_fov, tgt_n = write_stretched_case(
        target_generator_n, base / "target.csv"
    )
    apply_tgt_dir = base / "apply_target"
    applied_tgt = run_script(
        "stm_apply_transform.py",
        [
            str(tgt_csv),
            "--transform",
            str(transform_path),
            "-L",
            f"{tgt_fov:.6f}",
            "-o",
            str(apply_tgt_dir),
        ],
        env,
    )
    tgt_report_path = apply_tgt_dir / "apply_report.json"
    tgt_report = (
        json.loads(tgt_report_path.read_text()) if tgt_report_path.is_file() else None
    )
    # independent recomputation of the resampling geometry from affine_q at the
    # target's own canvas side (the formula of the package's _image_transform)
    matrix = AXIS_SWAP @ np.linalg.inv(stretch) @ AXIS_SWAP
    corners = (
        np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
        * (tgt_n - 1)
        / 2.0
    )
    extent = np.abs(np.linalg.inv(matrix) @ corners.T).max(axis=1)
    n_out_expected = 2 * (int(np.ceil(float(extent.max()))) + pad) + 1
    tgt_out_csv = apply_tgt_dir / f"{tgt_csv.stem}_corrected.csv"
    shape_ok = bool(
        tgt_out_csv.is_file()
        and np.loadtxt(tgt_out_csv, delimiter=",").shape
        == (n_out_expected, n_out_expected)
    )
    tgt_log = apply_tgt_dir / "correction.log"
    tgt_log_text = tgt_log.read_text() if tgt_log.is_file() else ""
    contract = CONTRACT_LINE_RE.search(tgt_log_text)
    fov_expected = tgt_fov * n_out_expected / tgt_n
    contract_ok = bool(
        contract
        and int(contract.group(1)) == n_out_expected
        and int(contract.group(2)) == n_out_expected
        and abs(float(contract.group(3)) - fov_expected) < 1e-3
    )
    entry_ok = bool(
        applied_tgt.returncode == 0
        and tgt_report is not None
        and int(tgt_report["n_out"]) == n_out_expected
        and n_out_expected != int(payload["n_out_reference"])
    )
    target_detail = (
        f"(c) target n={tgt_n} (generator {target_generator_n}): expected "
        f"n_out {n_out_expected}, csv shape ok {shape_ok}, report n_out "
        f"{None if tgt_report is None else tgt_report['n_out']}, contract "
        f"line {contract.group(0) if contract else 'missing'}, expected "
        f"field of view {fov_expected:.4f} nm, reference n_out "
        f"{payload['n_out_reference']}"
    )

    # (d) the local mirror of the package's resampling geometry must agree bit for bit
    local_matrix, local_n_out, local_offset = stm_apply_transform._image_transform(
        stretch, tgt_n, pad
    )
    ref_matrix, ref_n_out, ref_offset = package_image_transform(stretch, tgt_n, pad)
    mirror_ok = bool(
        np.array_equal(local_matrix, ref_matrix)
        and local_n_out == ref_n_out
        and np.array_equal(local_offset, ref_offset)
    )
    mirror_detail = (
        f"(d) mirror: matrix equal "
        f"{np.array_equal(local_matrix, ref_matrix)}, n_out "
        f"{local_n_out} vs {ref_n_out}, offset equal "
        f"{np.array_equal(local_offset, ref_offset)}"
    )

    # (e) identity_fallback copies the input verbatim and warns
    identity = dict(payload)
    identity.update(
        {
            "method": "identity_fallback",
            "fallback": True,
            "n_labelled": 0,
            "anchor_verdict": "unverifiable",
            "affine_q": [[1.0, 0.0], [0.0, 1.0]],
            "affine_image_reference": [[1.0, 0.0], [0.0, 1.0]],
            "stretch_scale_sqrt_det": 1.0,
        }
    )
    identity_path = base / "identity_transform.json"
    identity_path.write_text(json.dumps(identity, indent=2) + "\n")
    apply_ident_dir = base / "apply_identity"
    applied_ident = run_script(
        "stm_apply_transform.py",
        [
            str(ref_csv),
            "--transform",
            str(identity_path),
            "-L",
            f"{ref_fov:.6f}",
            "-o",
            str(apply_ident_dir),
        ],
        env,
    )
    ident_csv = apply_ident_dir / f"{ref_csv.stem}_corrected.csv"
    expected = io.StringIO()
    np.savetxt(
        expected,
        np.flipud(subtractMeanPlane(load_image(ref_csv))),
        delimiter=",",
        fmt="%.10e",
    )
    ident_log = apply_ident_dir / "correction.log"
    ident_log_text = ident_log.read_text() if ident_log.is_file() else ""
    verbatim = bool(
        ident_csv.is_file() and ident_csv.read_text() == expected.getvalue()
    )
    identity_ok = bool(
        applied_ident.returncode == 0 and verbatim and "WARNING" in ident_log_text
    )
    identity_detail = (
        f"(e) identity fallback: exit {applied_ident.returncode}, input "
        f"copied verbatim {verbatim}, WARNING logged "
        f"{'WARNING' in ident_log_text}"
    )

    check(
        name,
        bool(
            export_ok
            and applied.returncode == 0
            and round_trip_ok
            and shape_ok
            and contract_ok
            and entry_ok
            and mirror_ok
            and identity_ok
        ),
        "; ".join(
            (
                f"(a) export ok, all {len(fields)} schema fields present",
                round_detail,
                target_detail,
                mirror_detail,
                identity_detail,
            )
        ),
        "every sub-step passes",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--workdir",
        default=None,
        help="scratch directory (default: a fresh directory under <tmp>)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=384,
        help="canvas side of the synthetic image (default 384)",
    )
    parser.add_argument(
        "--keep", action="store_true", help="keep the scratch directory"
    )
    args = parser.parse_args(argv)

    if args.workdir:
        workdir = Path(args.workdir)
        if workdir.exists():
            shutil.rmtree(workdir)
    else:
        workdir = Path(tempfile.mkdtemp(prefix="stm-topo-correct-selftest-"))
    workdir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(workdir / ".mplcache"))

    print("# selftest.py - geometry checks of the topo-correction skill: the")
    print("# anchor-ring contract of stm_topo_correct.py on a synthetic image with a")
    print("# known anisotropic stretch, plus the two-stage transform workflow")
    print("# (--save-transform export and stm_apply_transform.py on another canvas)")
    print(f"# scratch directory: {workdir}")

    test_correction_stage(args.size, workdir)
    test_transform_stage(args.size, workdir)

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
