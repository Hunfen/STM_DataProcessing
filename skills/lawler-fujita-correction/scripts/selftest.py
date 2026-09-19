"""One-command self-test of the lawler-fujita-correction skill.

The synthetic is a hexagonal lattice (a = 0.246 nm, 50 nm field of view) carrying a
**known** smooth displacement field -- a Gaussian bump plus a linear drift -- which
is then recovered by the fit stage and used to resample the image.  Everything
checked here is geometry: the displacement field, the ring quality of the corrected
image, the phase consistency of the three lock-in directions, and the transfer of
the bundle to the same and to a different grid.  No physical statement is made and
no number is needed from the phase-analysis skill.

The injected amplitude is bounded by the visibility condition of the data-anchored
peak detection (see ``BUMP_NM`` below): a strained lattice has Bragg peaks smeared
by ``|grad u| * radius``, so more than roughly two percent strain fragments the ring
that the correction must anchor on.

    cd /path/to/STM_DataProcessing
    MPLCONFIGDIR=<writable> PYTHONDONTWRITEBYTECODE=1 \
        .venv/bin/python <this script>

Options:
    --workdir DIR   scratch directory (default: a fresh directory under <tmp>)
    --size N        canvas side of the synthetic image (default 1024)
    --keep          keep the scratch directory

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

import lf_lib as lf  # noqa: E402

A_NM = 0.246
FIELD_NM = 50.0
# Injected displacement: a Gaussian bump of 1.0 nm plus a linear drift of 0.4 nm
# across the field.  The bump gradient is 1.0 / (30 * sqrt(e)) = 0.020 nm/nm, so the
# local strain stays at the two percent that the ring clustering of the
# data-anchored detection still tolerates (measured: the ring is found up to
# |grad u| ~ 0.025 and fragments at 0.03).
BUMP_NM = 1.0
BUMP_SIGMA_NM = 30.0
DRIFT_NM = 0.4
BUMP_CENTRE_NM = (6.0, -4.0)
R3_AMPLITUDE = 0.5
TARGET_SIZE = 800
NO_LATTICE_SIZE = 512
# The lock-in mask must contain the band of the injected distortion: the analytic
# bound is lambda <= 2 pi / (|K| max|grad u|), and the test uses a third of it.
LAMBDA_BAND_SAFETY = 3.0
LAMBDA_WINDOW_NM = (1.5, 6.0)
RECOVERY_RELATIVE_TOL = 0.10
RECOVERY_ABSOLUTE_TOL_NM = 0.10
PAIR_RATIO_TOL = 0.01
ANISOTROPY_SLACK = 1.05
SQRT3 = float(np.sqrt(3.0))
STM_LIB_DEFAULT = "/Users/hunfen/Documents/GitHub/STM_DataProcessing/src"
THIRD_DIRECTION_INTERIOR_TOL_DEG = 2.0
RESULTS: list[tuple[str, bool, str]] = []


def check(name, ok, detail, threshold=""):
    RESULTS.append((name, bool(ok), detail))
    flag = "PASS" if ok else "FAIL"
    limit = f"  [threshold {threshold}]" if threshold else ""
    print(f"  [{flag}] {name}: {detail}{limit}")
    return bool(ok)


def section(title):
    print(f"\n== {title} ==")


def run_script(script, arguments, env):
    return subprocess.run([sys.executable, str(HERE / script), *arguments],
                          capture_output=True, text=True, env=env, check=False)


# --------------------------------------------------------------------------- #
# synthetic ground truth (independent of any other skill)
# --------------------------------------------------------------------------- #
def ring_vectors(size_nm, a_nm, angles_deg):
    """First-order wave vectors of the hexagonal ring, in FFT pixels."""
    radius = lf.ideal_radius_px(size_nm, a_nm)
    angles = np.radians(np.asarray(angles_deg, dtype=float))
    return np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)


def displacement_field(cols, rows, size_nm):
    """Known displacement: Gaussian bump plus linear drift, zero mean."""
    pixel = size_nm / cols.shape[0]
    x = (cols - cols.shape[1] / 2.0) * pixel
    y = (rows - rows.shape[0] / 2.0) * pixel
    bump = BUMP_NM * np.exp(-((x - BUMP_CENTRE_NM[0]) ** 2 + (y - BUMP_CENTRE_NM[1]) ** 2)
                            / (2.0 * BUMP_SIGMA_NM ** 2))
    u_x = bump + DRIFT_NM * x / size_nm
    u_y = 0.6 * bump - 0.8 * DRIFT_NM * y / size_nm
    return np.stack([u_x - u_x.mean(), u_y - u_y.mean()])


def lattice_image(cols, rows, size_nm, u_nm=None, with_r3=True):
    """Ideal hexagonal lattice, optionally displaced by ``u_nm`` (model r - u)."""
    n = cols.shape[0]
    ring = ring_vectors(size_nm, A_NM, (90.0, 30.0, -30.0))
    if u_nm is None:
        x_px, y_px = cols.astype(float), rows.astype(float)
    else:
        pixel = size_nm / n
        x_px, y_px = cols - u_nm[0] / pixel, rows - u_nm[1] / pixel
    image = np.zeros(cols.shape, dtype=float)
    for index in range(3):
        image += np.cos((2.0 * np.pi / n) * (ring[index][0] * x_px + ring[index][1] * y_px))
    if with_r3:
        for index in range(3):
            kx, ky = ring[index] / SQRT3
            image += R3_AMPLITUDE * np.cos(
                (2.0 * np.pi / n) * (kx * x_px + ky * y_px) + 0.7 * (index + 1))
    return image


def write_case(path, n, size_nm, with_lattice=True, with_r3=True):
    """Write one synthetic CSV and return the injected field (None without lattice)."""
    rows, cols = np.mgrid[:n, :n]
    if not with_lattice:
        x = (cols - n / 2.0) / n
        y = (rows - n / 2.0) / n
        smooth = (0.4 * x + 0.3 * y + 0.2 * x * y
                  + 0.5 * np.exp(-((x - 0.1) ** 2 + (y + 0.2) ** 2) / 0.05))
        np.savetxt(path, smooth, delimiter=",", fmt="%.10e")
        return None
    u_nm = displacement_field(cols, rows, size_nm)
    image = lattice_image(cols, rows, size_nm, u_nm=u_nm, with_r3=with_r3)
    np.savetxt(path, image, delimiter=",", fmt="%.10e")
    return u_nm


def gradient_bound(u_nm, size_nm):
    """Largest |grad u| of the injected field, in nm per nm."""
    pixel = size_nm / u_nm.shape[1]
    return max(float(np.max(np.hypot(*np.gradient(component)))) / pixel
               for component in (u_nm[0], u_nm[1]))


def compare_recovered_field(u_fit, mask, u_injected, size_nm):
    """Compare the recovered field with the injected one in the same frame.

    The fit stage works on ``flipud(subtractMeanPlane(image))`` (first row = top
    scan line), so the injected field is flipped the same way before the
    comparison, and the displacement field is defined up to a rigid translation and
    a uniform strain of the reference hexagon -- the fit stage reads the orientation
    from the data and the radius from ``--a``, and a mean shear of the data biases
    that reference.  Both the raw error and the error after removing that affine
    gauge are returned, so the check can report how much of it is honest gauge.
    """
    n = u_injected.shape[1]
    pixel = size_nm / n
    rows, cols = np.mgrid[:n, :n]
    x = (cols - n / 2.0) * pixel
    y = (rows - n / 2.0) * pixel
    reference = np.stack([np.flipud(u_injected[0]), -np.flipud(u_injected[1])])
    x_flipped, y_flipped = np.flipud(x), -np.flipud(y)
    reference = reference - reference[:, mask].mean(axis=1)[:, None, None]
    error = u_fit - reference
    design = np.column_stack([np.ones(int(np.count_nonzero(mask))), x_flipped[mask],
                              y_flipped[mask]])
    gauge, residual = [], []
    for component in range(2):
        coefficients, *_ = np.linalg.lstsq(design, error[component][mask], rcond=None)
        gauge.append([float(value) for value in coefficients])
        residual.append(error[component][mask] - design @ coefficients)
    return {
        "reference_rms_nm": float(np.sqrt(np.mean(reference[:, mask] ** 2))),
        "error_rms_nm": float(np.sqrt(np.mean(error[:, mask] ** 2))),
        "gauge_removed_rms_nm": float(np.sqrt(np.mean(np.stack(residual) ** 2))),
        "gauge": gauge,
    }


def read_report(outdir):
    path = Path(outdir) / "correction_report.json"
    return json.loads(path.read_text()) if path.is_file() else None


def load_matrix(path):
    return np.loadtxt(path, delimiter=",")


def preprocessed(path, stm_lib):
    """The frame the scripts work in: flipud(subtractMeanPlane(loaded matrix))."""
    sys.path.insert(0, str(stm_lib))
    from stm_data_processing.utils.plot_funcs import subtractMeanPlane

    return np.flipud(subtractMeanPlane(load_matrix(path)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workdir", default=None,
                        help="scratch directory (default: a fresh directory under <tmp>)")
    parser.add_argument("--size", type=int, default=1024,
                        help="canvas side of the synthetic image (default 1024)")
    parser.add_argument("--keep", action="store_true",
                        help="keep the scratch directory")
    parser.add_argument("--stm-lib", default=STM_LIB_DEFAULT,
                        help="STM_DataProcessing src directory")
    args = parser.parse_args(argv)

    if args.workdir:
        workdir = Path(args.workdir)
        if workdir.exists():
            shutil.rmtree(workdir)
    else:
        workdir = Path(tempfile.mkdtemp(prefix="lawler-fujita-correction-selftest-"))
    workdir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(workdir / ".mplcache"))
    (workdir / ".mplcache").mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["MPLCONFIGDIR"] = str(workdir / ".mplcache")
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    print("# selftest.py - geometry checks of the lawler-fujita-correction skill")
    print("# synthetic hexagonal lattice (a = 0.246 nm) with a known smooth displacement")
    print(f"# bump {BUMP_NM:g} nm (sigma {BUMP_SIGMA_NM:g} nm) + drift {DRIFT_NM:g} nm, "
          f"separate 1x1 and r3 content")
    print(f"# scratch directory: {workdir}")

    # ------------------------------------------------------------------ setup #
    section("synthetic ground truth")
    csv_path = workdir / "synthetic_distorted.csv"
    u_injected = write_case(csv_path, args.size, FIELD_NM)
    gradient = gradient_bound(u_injected, FIELD_NM)
    wave = lf.wave_vectors_nm_inv(ring_vectors(FIELD_NM, A_NM, (90.0,))[0], FIELD_NM)
    lambda_band = 2.0 * np.pi / (float(np.hypot(*wave)) * gradient)
    lambda_test = lambda_band / LAMBDA_BAND_SAFETY
    injected_rms = float(np.sqrt(np.mean(u_injected ** 2)))
    print(f"# injected displacement: rms {injected_rms:.4f} nm, "
          f"max {float(np.max(np.hypot(*u_injected))):.4f} nm, "
          f"max |grad u| {gradient:.4f} nm/nm")
    print(f"# lock-in band: lambda <= 2 pi / (|K| max|grad u|) = {lambda_band:.3f} nm; "
          f"the test uses lambda = {lambda_test:.3f} nm")
    check("the injected field is visible to the data-anchored detection",
          LAMBDA_WINDOW_NM[0] <= lambda_test <= LAMBDA_WINDOW_NM[1],
          f"lambda_test = {lambda_test:.3f} nm inside {LAMBDA_WINDOW_NM}",
          "strain below the ring-clustering tolerance")

    # ------------------------------------------------------- fit + artifacts #
    section("fit stage (Lawler-Fujita field, lock-in, warp, artifacts)")
    fit_dir = workdir / "fit"
    bundle = workdir / "bundle.json"
    fit = run_script("stm_lf_correct.py",
                     [str(csv_path), "-L", f"{FIELD_NM:g}", "--a", f"{A_NM:g}",
                      "--lambda-nm", f"{lambda_test:.6f}",
                      "--save-transform", str(bundle), "-o", str(fit_dir)], env)
    report = read_report(fit_dir)
    log_text = ((fit_dir / "correction.log").read_text()
                if (fit_dir / "correction.log").is_file() else "")
    print(f"# fit exit code {fit.returncode} (stdout {len(fit.stdout.splitlines())} lines)")
    if fit.returncode != 0:
        print(fit.stdout[-2000:])
        print(fit.stderr[-2000:], file=sys.stderr)
    check("fit stage ran", fit.returncode == 0 and report is not None,
          f"exit code {fit.returncode}, correction_report.json "
          f"{'written' if report is not None else 'missing'}", "exit code 0 and report")
    if report is None:
        print("\n0/{0} checks passed".format(len(RESULTS)))
        return 1

    required = ("q_a_px", "q_b_px", "q_a_nm_inv", "q_b_nm_inv", "lambda_nm", "gauge",
                "lockin", "displacement", "third_direction", "residual_self_check",
                "lf_artifacts")
    missing = sorted(key for key in required if key not in report)
    check("report carries the Lawler-Fujita fields", not missing,
          ("missing: " + ", ".join(missing)) if missing else
          f"{len(report)} fields, Q_a = {[round(value, 2) for value in report['q_a_px']]} px, "
          f"lambda = {report['lambda_nm']:.3f} nm", "every documented field present")

    contract = [line for line in log_text.splitlines()
                if line.startswith("# canvas ") or line.startswith("# corrected canvas:")]
    check("both log contract lines are present",
          any(line.startswith("# canvas ") for line in contract)
          and any(line.startswith("# corrected canvas:") for line in contract),
          " | ".join(contract) or "no contract line found",
          "'# canvas ... field of view <L> nm' and '# corrected canvas: ...'")

    artifacts = report.get("lf_artifacts", {})
    wanted = ["theta_a", "theta_b", "theta_c", "amplitude_a", "amplitude_b",
              "amplitude_c", "u_x", "u_y", "mask"]
    absent = [name for name in wanted
              if name not in artifacts
              or not Path(artifacts[name]["npy"]).is_file()
              or not Path(artifacts[name]["png"]).is_file()]
    check("LF artifacts (theta / amplitude / u / mask, npy + png) written", not absent,
          ("missing: " + ", ".join(absent)) if absent else f"{len(wanted)} maps, npy + png",
          "every map has both files")
    # The names are pinned down literally, not only through the report: the SKILL.md
    # and README tables promise <stem>_lf_<map>.{npy,png}, and a doc-name drift has to
    # fail here instead of only showing up on a real run.
    documented = [f"{csv_path.stem}_lf_{name}{suffix}" for name in wanted
                  for suffix in (".npy", ".png")]
    missing_files = [name for name in documented if not (fit_dir / name).is_file()]
    check("LF artifacts carry the documented _lf_ filenames on disk", not missing_files,
          ("missing: " + ", ".join(missing_files)) if missing_files
          else f"{len(documented)}/{len(documented)} documented names exist in {fit_dir}",
          "every documented <stem>_lf_<map>.{npy,png} exists")

    products = (fit_dir / f"{csv_path.stem}_corrected.csv",
                fit_dir / f"{csv_path.stem}_corrected_fft2.npy",
                fit_dir / f"{csv_path.stem}_corrected.png",
                fit_dir / f"{csv_path.stem}_corrected_fft.png")
    fft2 = np.load(products[1]) if products[1].is_file() else None
    check("corrected CSV + complex FFT2 + previews written",
          all(path.is_file() for path in products) and fft2 is not None
          and np.iscomplexobj(fft2) and fft2.shape == (report["n_out"], report["n_out"]),
          f"corrected {report['n_out']} x {report['n_out']} px, fft2 "
          f"{None if fft2 is None else fft2.dtype}{None if fft2 is None else fft2.shape}",
          "four products, complex128 fft2 on the corrected canvas")

    # ------------------------------------------------------------ recovery  #
    section("known displacement recovery")
    mask = np.load(artifacts["mask"]["npy"]).astype(bool)
    u_fit = np.stack([np.load(artifacts["u_x"]["npy"]), np.load(artifacts["u_y"]["npy"])])
    comparison = compare_recovered_field(u_fit, mask, u_injected, FIELD_NM)
    reference_rms = comparison["reference_rms_nm"]
    tolerance = max(RECOVERY_RELATIVE_TOL * reference_rms, RECOVERY_ABSOLUTE_TOL_NM)
    check("recovered displacement matches the injected field",
          comparison["error_rms_nm"] < tolerance,
          f"error rms {comparison['error_rms_nm']:.5f} nm vs injected rms "
          f"{reference_rms:.5f} nm "
          f"({100 * comparison['error_rms_nm'] / reference_rms:.2f} %), mask coverage "
          f"{100 * mask.mean():.2f} %", f"< {tolerance:.4f} nm")
    check("the recovery matches the injected field up to the reference gauge",
          comparison["gauge_removed_rms_nm"] < tolerance,
          f"after removing the affine reference gauge (rigid translation + uniform "
          f"strain, slopes {[[round(value, 5) for value in row] for row in comparison['gauge']]}"
          f"): {comparison['gauge_removed_rms_nm']:.5f} nm "
          f"({100 * comparison['gauge_removed_rms_nm'] / reference_rms:.2f} %)",
          f"< {tolerance:.4f} nm")
    check("the recovery test is not vacuous", reference_rms > tolerance,
          f"injected rms {reference_rms:.4f} nm > tolerance {tolerance:.4f} nm",
          "the injected field exceeds the acceptance tolerance")

    # --------------------------------------------------------- ring quality #
    section("ring quality of the corrected image")
    before = report["residual_self_check"]["before"]["ring_1x1"]
    after = report["residual_self_check"]["after"]["ring_1x1"]
    if before is None or after is None:
        check("1x1 ring re-detected on the corrected image", False,
              f"before {before}, after {after}", "both rings present")
    else:
        check("1x1 ring anisotropy is not worse than the input's",
              after["anisotropy"] <= ANISOTROPY_SLACK * before["anisotropy"],
              f"before {100 * before['anisotropy']:.4f} % -> after "
              f"{100 * after['anisotropy']:.4f} % (radius "
              f"{after['radius_px']:.4f} px, spread {after['min_radius_px']:.4f} .. "
              f"{after['max_radius_px']:.4f} px)",
              f"after <= {ANISOTROPY_SLACK:.2f} x before")
        pair = report["residual_self_check"]["after"]["r3_ratio_vs_sqrt3"]
        check("1x1 / r3 ring pair ratio is the ideal sqrt(3)", pair is not None
              and abs(pair["deviation"]) <= PAIR_RATIO_TOL,
              (f"ratio {pair['ratio']:.6f} ({100 * pair['deviation']:+.4f} % from "
               f"sqrt(3))") if pair is not None else "no r3 ring on the corrected image",
              f"within {100 * PAIR_RATIO_TOL:.1f} % of sqrt(3)")

    # ------------------------------------------------------ third direction #
    third = report["third_direction"]
    interior_rms = third.get("wrapped_rms_deg_interior")
    check("third-direction consistency of the three lock-in phases",
          bool(third.get("available")) and interior_rms is not None
          and interior_rms < THIRD_DIRECTION_INTERIOR_TOL_DEG,
          f"theta_a + theta_b + theta_c wrapped rms "
          f"{interior_rms if interior_rms is None else round(interior_rms, 4)} deg inside "
          f"the {third.get('interior_margin_px')} px border margin "
          f"({third.get('interior_margin_rule')}), {round(float(third.get('wrapped_rms_deg', float('nan'))), 4)} deg "
          f"over the full mask; Q_a + Q_b + Q_c = 0 exactly",
          f"< {THIRD_DIRECTION_INTERIOR_TOL_DEG:g} deg (interior)")

    # --------------------------------------------------------- transfer     #
    section("transfer bundle")
    u_field = bundle.with_suffix(".npz")
    bundle_payload = json.loads(bundle.read_text()) if bundle.is_file() else {}
    check("bundle JSON + u-field npz written", bundle.is_file() and u_field.is_file()
          and bundle_payload.get("schema") == "lawler-fujita-correction-transform"
          and bundle_payload.get("schema_version") == 1,
          f"schema {bundle_payload.get('schema')!r} v{bundle_payload.get('schema_version')}, "
          f"u field {u_field.name} "
          f"{'present' if u_field.is_file() else 'missing'}", "schema + npz")

    same_dir = workdir / "apply_same"
    same = run_script("stm_lf_apply.py",
                      [str(csv_path), "--transform", str(bundle), "-L", f"{FIELD_NM:g}",
                       "-o", str(same_dir)], env)
    same_csv = same_dir / f"{csv_path.stem}_corrected.csv"
    fit_csv = fit_dir / f"{csv_path.stem}_corrected.csv"
    ok_same = same.returncode == 0 and same_csv.is_file()
    mask_same = exact = None
    detail = f"apply exit code {same.returncode}"
    if ok_same:
        array_same, array_fit = load_matrix(same_csv), load_matrix(fit_csv)
        mask_same = np.isnan(array_same)
        same_mask = bool(np.array_equal(mask_same, np.isnan(array_fit)))
        exact = bool(np.allclose(array_same, array_fit, rtol=1e-10, atol=1e-10,
                                 equal_nan=True))
        detail = (f"max |delta| {float(np.nanmax(np.abs(array_same - array_fit))):.3e}, "
                  f"identical NaN mask {same_mask}, allclose(1e-10) {exact}")
    check("apply on the reference grid reproduces the fit stage", bool(ok_same and exact),
          detail, "allclose rtol=atol=1e-10 and the same NaN mask")

    target_csv = workdir / f"target_{TARGET_SIZE}.csv"
    write_case(target_csv, TARGET_SIZE, FIELD_NM)
    target_dir = workdir / "apply_target"
    target = run_script("stm_lf_apply.py",
                        [str(target_csv), "--transform", str(bundle), "-L", f"{FIELD_NM:g}",
                         "-o", str(target_dir)], env)
    target_report = (json.loads((target_dir / "apply_report.json").read_text())
                     if (target_dir / "apply_report.json").is_file() else None)
    target_log = ((target_dir / "correction.log").read_text()
                  if (target_dir / "correction.log").is_file() else "")
    rescaled = bool(target_report and target_report.get("u_field_rescaled"))
    contract_ok = any(line.startswith("# corrected canvas:")
                      for line in target_log.splitlines())
    check(f"apply on a different grid ({TARGET_SIZE} px) rescales the field",
          target.returncode == 0 and target_report is not None and rescaled
          and contract_ok and target_report.get("n_px") == TARGET_SIZE,
          (f"exit code {target.returncode}, n_px "
           f"{None if target_report is None else target_report.get('n_px')}, n_out "
           f"{None if target_report is None else target_report.get('n_out')}, "
           f"u field rescaled {rescaled}, contract line {contract_ok}"),
          f"exit code 0, n_px {TARGET_SIZE}, rescaled field, contract line")

    # ---------------------------------------------------------- fallback    #
    section("identity fallback without lattice")
    plain_csv = workdir / "no_lattice.csv"
    write_case(plain_csv, NO_LATTICE_SIZE, FIELD_NM, with_lattice=False)
    plain_dir = workdir / "no_lattice_out"
    plain = run_script("stm_lf_correct.py",
                       [str(plain_csv), "-L", f"{FIELD_NM:g}", "--a", f"{A_NM:g}",
                        "-o", str(plain_dir)], env)
    plain_report = read_report(plain_dir)
    plain_log = ((plain_dir / "correction.log").read_text()
                 if (plain_dir / "correction.log").is_file() else "")
    plain_csv_out = plain_dir / "no_lattice_corrected.csv"
    copied = plain_csv_out.is_file() and np.allclose(
        load_matrix(plain_csv_out), preprocessed(plain_csv, args.stm_lib),
        rtol=1e-9, atol=1e-9, equal_nan=True)
    check("a lattice-free input falls back to the identity",
          plain.returncode == 0 and plain_report is not None
          and plain_report.get("fallback") is True
          and plain_report.get("method") == "identity_fallback"
          and "WARNING" in plain_log and copied,
          f"exit code {plain.returncode}, method "
          f"{None if plain_report is None else plain_report.get('method')}, "
          f"WARNING logged {'WARNING' in plain_log}, input copied verbatim {copied}",
          "fallback=True, method identity_fallback, WARNING, copy unchanged")

    # ---------------------------------------------------- band reality note #
    section("informational: what the default lambda keeps")
    default_dir = workdir / "fit_default_lambda"
    default = run_script("stm_lf_correct.py",
                         [str(csv_path), "-L", f"{FIELD_NM:g}", "--a", f"{A_NM:g}",
                          "-o", str(default_dir)], env)
    default_report = read_report(default_dir)
    detail = f"fit exit code {default.returncode}"
    if default_report is not None and default_report.get("fallback") is False:
        artifacts_default = default_report["lf_artifacts"]
        mask_default = np.load(artifacts_default["mask"]["npy"]).astype(bool)
        u_default = np.stack([np.load(artifacts_default["u_x"]["npy"]),
                              np.load(artifacts_default["u_y"]["npy"])])
        rms_default = float(np.sqrt(np.mean(u_default[:, mask_default] ** 2)))
        lockin = default_report.get("lockin", {})
        detail = (f"lambda 30 nm keeps a displacement rms of {rms_default:.4f} nm "
                  f"against {reference_rms:.4f} nm injected "
                  f"({100 * rms_default / reference_rms:.1f} %), band margin "
                  f"lambda x offset = {lockin.get('band_margin_lambda_times_offset')} "
                  f"(band_warning {lockin.get('band_warning')}): the default low-pass is "
                  "meant for distortions far below one percent strain")
    elif default_report is not None:
        detail = (f"lambda 30 nm falls back to the identity "
                  f"({default_report.get('fallback_reason')})")
    print(f"  [INFO] {detail}")

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
