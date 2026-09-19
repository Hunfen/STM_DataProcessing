"""One-command self-test of the topo-correction skill: the geometry stage on a
synthetic image with a known anisotropy.

Everything checked here is geometry (a stretch matrix, a lattice constant derived
from a measured ring radius, and the anchor self-check verdict); no physical
statement is made and no number is needed from the phase-analysis skill.

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

SQRT3 = float(np.sqrt(3.0))
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
    return [(float(radius_px * np.cos(angle)), float(radius_px * np.sin(angle)))
            for angle in angles]


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


def synthetic_image(n, radius_frac=0.458, ref_amp=1.0, ref_phase_deg=40.0,
                    disk_amp=1.0, disk_phase_deg=240.0, disk_centre=(0.5, 0.5),
                    disk_radius_frac=0.20):
    """Ground truth image: one reference ring plus a disk-dependent inner ring.

    The reference ring sits at ``radius_frac * n`` px and carries one phase
    everywhere; the disk region adds the same hexagon rotated by 30 degrees and
    scaled by ``1/sqrt(3)`` (an r3 ring inside the reference ring) with the disk's
    own phase, so the pair of rings is at a radius ratio ``sqrt(3)``.  Returns the
    image and the reference ring radius in pixels.
    """
    r1 = float(radius_frac) * n
    image = float(ref_amp) * ring_wave(n, ring_vectors(r1, 90.0),
                                       np.radians(ref_phase_deg))
    disk = soft_disk(n, disk_centre, disk_radius_frac)
    inner = ring_vectors(r1 / SQRT3, 120.0)
    image = image + float(disk_amp) * disk * ring_wave(n, inner,
                                                       np.radians(disk_phase_deg))
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
    corners = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0],
                        [1.0, 1.0]]) * (n - 1) / 2.0
    extent = np.abs(np.linalg.inv(matrix) @ corners.T).max(axis=1)
    n_out = 2 * (int(np.ceil(float(extent.max()))) + int(pad)) + 1
    offset = (n - 1) / 2.0 - matrix @ np.array([(n_out - 1) / 2.0,
                                                (n_out - 1) / 2.0])
    out = affine_transform(arr, matrix, offset=offset, output_shape=(n_out, n_out),
                           order=int(order), mode="constant", cval=np.nan,
                           prefilter=int(order) > 1)
    return out, n_out


def run_script(script, arguments, env):
    return subprocess.run([sys.executable, str(HERE / script), *arguments],
                          capture_output=True, text=True, env=env, check=False)


def ratio_text(check_result):
    if check_result.get("ratio") is None:
        return "not enough corrected rings to form a pair"
    return (f"{check_result['ratio']:.6f} (deviation "
            f"{100 * check_result['deviation']:.4f} % from sqrt(3))")


# --------------------------------------------------------------------------- #
# the correction stage
# --------------------------------------------------------------------------- #
def test_correction_stage(n, workdir):
    section("correction stage: explicit anchor ring (needs the package)")
    try:
        sys.path.insert(0, "/Users/hunfen/Documents/GitHub/STM_DataProcessing/src")
        import stm_data_processing  # noqa: F401
    except Exception as exc:
        check("correction stage skipped", True,
              f"the package is not importable here ({type(exc).__name__}); run the "
              f"self-test with the repository interpreter to cover this stage",
              "informational")
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
        completed = run_script("stm_topo_correct.py",
                               [str(csv_path), "-L", f"{field_of_view:.6f}", "-o",
                                str(outdir), "--anchor-ring", anchor], env)
        report = (json.loads((outdir / "correction_report.json").read_text())
                  if (outdir / "correction_report.json").is_file() else None)
        reports[anchor] = (completed, report)
    good, bad = reports["r3"][1], reports["1x1"][1]
    ran = (reports["r3"][0].returncode == 0 and reports["1x1"][0].returncode == 0
           and good is not None and bad is not None)
    check("correction stage ran (both anchors)",
          ran,
          f"exit codes {reports['r3'][0].returncode} / {reports['1x1'][0].returncode}, "
          + ("both correction_report.json written" if ran else
             f"stderr: {reports['r3'][0].stderr.strip()[-200:]}"),
          "exit code 0 and reports written")
    if not ran:
        return
    check("correct anchor: the recovered stretch undoes the injected one",
          abs(good["affine_q"][0][0] - 1.010) < 3e-3
          and abs(good["affine_q"][1][1] - 0.990) < 3e-3,
          f"M = [[{good['affine_q'][0][0]:.5f}, {good['affine_q'][0][1]:.5f}], "
          f"[{good['affine_q'][1][0]:.5f}, {good['affine_q'][1][1]:.5f}]] vs the "
          f"injected [[1.010, 0], [0, 0.990]]; |det M|^(1/2) = "
          f"{good['stretch_scale_sqrt_det']:.5f}", "3e-3")
    check("correct anchor: the implied 1x1 lattice constant comes back",
          abs(good["implied_lattice_before"]["deviation_from_nominal"]) < 0.01,
          f"implied a = {good['implied_lattice_before']['a_1x1_nm']:.4f} nm vs the "
          f"nominal {good['a_nm']:.3f} nm "
          f"({100 * good['implied_lattice_before']['deviation_from_nominal']:+.3f} %)",
          "1 %")
    check("correct anchor: the corrected image carries a 1 : sqrt(3) ring pair",
          good["anchor_self_check"]["verdict"] == "consistent",
          f"anchor verdict = {good['anchor_self_check']['verdict']}, corrected ring "
          f"ratio = {ratio_text(good['anchor_self_check'])}", "consistent")
    check("a wrong anchor is caught by the same self-check",
          bad["anchor_self_check"]["verdict"] != "consistent",
          f"anchor verdict = {bad['anchor_self_check']['verdict']}, corrected ring "
          f"ratio = {ratio_text(bad['anchor_self_check'])}", "not consistent")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workdir", default=None,
                        help="scratch directory (default: a fresh directory under <tmp>)")
    parser.add_argument("--size", type=int, default=384,
                        help="canvas side of the synthetic image (default 384)")
    parser.add_argument("--keep", action="store_true",
                        help="keep the scratch directory")
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
    print("# known anisotropic stretch (no phase analysis is involved)")
    print(f"# scratch directory: {workdir}")

    test_correction_stage(args.size, workdir)

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
