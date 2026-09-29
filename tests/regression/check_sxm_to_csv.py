"""Regression check for scripts/sxm_to_csv.py.

Data-independent end-to-end test: it builds a minimal but legal .sxm in a
temporary directory (no real measurement data anywhere), runs the converter and
verifies the output naming, CSV layout, the plane-fit + min-to-zero correction
and NaN preservation.  The least-squares plane is re-derived independently in
this script so the check does not share the implementation's fitting code.

The payload is written the way Nanonis writes it (``pixels/line`` samples per
line along the x / fast axis, one line per y row), so the layout expectations
below are the real ones: a channel is ``(ny, nx) = (lines, pixels/line)``.
Each layout expectation is paired with a reverse assertion against the
pre-fix layout, and the expected file name is rebuilt from the widget facts
instead of being copied from the implementation, so a regression to either the
old shape or the old name turns this check red.

No local absolute data path (e.g. the user home directory) appears in this
file, so ``pytest -m "not localdata"`` (the CI gate) selects it.

Run from the repository root:

    .venv/bin/python tests/regression/check_sxm_to_csv.py

Exits non-zero when any check fails.
"""

from __future__ import annotations

import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

# Importing plot_funcs pulls in matplotlib; give it a writable config dir
# before it is first imported (see tests/regression/check_nanonis_sxm.py).
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "dsh_mplconfig"),
)

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "sxm_to_csv.py"

# Widget / header facts used to build the minimal .sxm and to predict the name.
REC_DATE = "25.06.2025"  # dd.mm.yyyy
BIAS_V = 0.05  # V  -> 50 mV
SETPOINT_A = 2.0e-11  # A  -> 20 pA
SCAN_RANGE = (1.0e-7, 2.0e-7)  # m  -> longer side 200 nm
NX, NY = 4, 3  # SCAN_PIXELS "4 3" = (nx = pixels/line, ny = lines)
N_CHANNELS = 2  # Z, Current


def _rebuild_expected_name() -> str:
    """Rebuild the expected output name from the widget facts.

    Deliberately imports nothing from scripts/sxm_to_csv.py: the name is
    recomputed from REC_DATE/BIAS/SETPOINT/SCAN_RANGE with the documented
    ``{yyyymmdd}_{stem}_{bias}mV{setpoint}pA_{frame}nm.csv`` template, so a
    change of the template shows up here as an independent expectation.
    """
    day, month, year = (int(v) for v in REC_DATE.split("."))
    date_part = f"{year:04d}{month:02d}{day:02d}"
    bias_mv = BIAS_V * 1000.0
    setpoint_pa = SETPOINT_A * 1e12
    frame_nm = max(SCAN_RANGE) * 1e9
    return f"{date_part}_mini_{bias_mv:g}mV{setpoint_pa:g}pA_{frame_nm:g}nm.csv"


EXPECTED_NAME = _rebuild_expected_name()  # 20250625_mini_50mV20pA_200nm.csv
# The pre-fix template glued the setpoint to the frame (no separator between
# "pA" and the frame length); the reverse assertions must reject that name.
NAME_WITHOUT_UNDERSCORE = EXPECTED_NAME.replace("pA_", "pA")

_CHECK_RESULTS: list[str] = []
_FAILS = 0


def check(name: str, ok: bool, detail: str) -> None:
    """Record one PASS/FAIL result."""
    global _FAILS
    if not ok:
        _FAILS += 1
    _CHECK_RESULTS.append(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")


def expected_z_forward() -> np.ndarray:
    """The Z forward matrix the payload encodes, in loader order.

    Shape ``(NY, NX)`` = (lines, pixels/line) and row-major pointing along x,
    exactly like a real Nanonis payload.  The values are ``2*x + 3*y + 4`` plus
    a little noise with a NaN block, returned in the payload's float32
    precision so the check can compare element-wise without tolerance.
    """
    xs, ys = np.meshgrid(np.arange(NX), np.arange(NY))
    z_forward = 2.0 * xs + 3.0 * ys + 4.0
    z_forward = z_forward + np.random.default_rng(1234).normal(0.0, 1e-3, size=(NY, NX))
    z_forward[1:3, 1:3] = np.nan  # NaN hole stays NaN through correction
    return z_forward.astype(np.float32)


def write_sxm(path: Path) -> None:
    """Write a minimal legal .sxm with a known Z forward scan."""
    # Build the payload the way Nanonis writes it: one scan line per y row,
    # NX = pixels/line floats per line (x = fast axis), row-major.  A load that
    # uses ``lines`` as the row width (the pre-fix layout) therefore fails both
    # the shape and the element-order expectations.
    payload_f = np.zeros(N_CHANNELS * 2 * NX * NY, dtype=np.float32)
    z_forward = expected_z_forward()
    payload_f[: z_forward.size] = z_forward.reshape(-1)
    # Leave the remaining rows (backward + Current) as 0; only Z forward matters.

    header = (
        ":REC_DATE:\n"
        f" {REC_DATE}\n"
        ":SCAN_PIXELS:\n"
        f"{NX} {NY}\n"
        ":SCAN_RANGE:\n"
        f"{SCAN_RANGE[0]} {SCAN_RANGE[1]}\n"
        ":SCAN_DIR:\n"
        "down\n"
        ":BIAS:\n"
        f"{BIAS_V}\n"
        ":Z-CONTROLLER:\n"
        "Name\ton\tSetpoint\tP-gain\tI-gain\tT-const\n"
        f"log Current\t1\t{SETPOINT_A:.3E} A\t1.0e-12 m\t1.0e-8 m/s\t1.5e-5 s\n"
        ":DATA_INFO:\n"
        "Channel\tName\tUnit\tDirection\tCalibration\tOffset\n"
        "30\tZ\tm\tboth\t8.000E-9\t0.000E+0\n"
        "5\tCurrent\tA\tboth\t1.000E-9\t0.000E+0\n"
        ":Scan>channels:\n"
        "Current (A);Z (m)\n"
        ":SCANIT_END:\n"
    )
    tail = b"\x1a\x04" + struct.pack(f">{payload_f.size}f", *payload_f)
    path.write_bytes(header.encode("utf-8") + tail)


def fit_plane_independent(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit a*x + b*y + c to the finite pixels with an independent lstsq.

    Matches plot_funcs.subtractMeanPlane's coordinate convention:
    ``x = arange(rows), y = arange(cols)``, plane = a*x + b*y + c.
    Returns (plane_evaluated, coefficient_vector).
    """
    rows, cols = data.shape
    y, x = np.meshgrid(np.arange(cols), np.arange(rows))
    finite = np.isfinite(data)
    A = np.column_stack([x.ravel(), y.ravel(), np.ones(x.ravel().size)])
    coeffs, _, _, _ = np.linalg.lstsq(
        A[finite.ravel()],
        data.ravel()[finite.ravel()],
        rcond=None,
    )
    plane = coeffs[0] * x + coeffs[1] * y + coeffs[2]
    return plane, coeffs


def run_end_to_end(tmp: Path) -> tuple[Path, list[str]]:
    """Create the .sxm, run the converter and return (sxm, produced names)."""
    sxm = tmp / "mini.sxm"
    write_sxm(sxm)
    out_dir = tmp / "out"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(sxm), "-o", str(out_dir)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return sxm, sorted(p.name for p in out_dir.iterdir())


def main() -> int:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # --- R1: filename (expected name plus the reverse, pre-fix variant) ---
        sxm, produced = run_end_to_end(tmp)
        check(
            "R1 output filename",
            EXPECTED_NAME in produced,
            f"expected {EXPECTED_NAME}, produced {produced}",
        )
        check(
            "R1 filename is not the pre-fix variant (pA glued to the frame)",
            NAME_WITHOUT_UNDERSCORE not in produced,
            f"rejected {NAME_WITHOUT_UNDERSCORE}, produced {produced}",
        )
        csv_path = tmp / "out" / EXPECTED_NAME
        if not csv_path.exists():
            print(f"\n{SCRIPT.name} regression: {_FAILS} failure(s)")
            print("\n".join(_CHECK_RESULTS))
            return 1

        # --- R2: CSV shape / delimiter / no header ----------------------------
        text = csv_path.read_text(encoding="utf-8", errors="replace").splitlines()
        headerless = all(not line.startswith("#") for line in text)
        check("R2 headerless", headerless, f"{len(text)} data line(s)")
        loaded = np.loadtxt(csv_path, delimiter="\t")
        check(
            "R2 shape is the corrected (lines, pixels/line) layout",
            loaded.shape == (NY, NX),
            f"shape = {loaded.shape}, expected {(NY, NX)}",
        )
        check(
            "R2 shape is not the pre-fix (pixels/line, lines) layout",
            loaded.shape != (NX, NY),
            f"rejected {(NX, NY)}, got {loaded.shape}",
        )
        check(
            "R2 tab delimiter",
            text[0].count("\t") == NX - 1,
            f"first line columns = {text[0].count(chr(9)) + 1}",
        )

        # --- R3: payload row order (row = x samples, row width = pixels/line) -
        forward = _read_z_forward_from_payload(sxm)
        expected_forward = expected_z_forward()
        order_match = np.array_equal(forward, expected_forward, equal_nan=True)
        check(
            "R3 payload row order",
            forward.shape == (NY, NX) and order_match,
            f"shape = {forward.shape}, element-wise match = {order_match}",
        )

        # --- R4: content matches an independent plane re-fit ------------------
        finite = np.isfinite(forward)
        plane_indep, _ = fit_plane_independent(forward)
        plane_removed_indep = forward - plane_indep
        plane_removed_indep -= plane_removed_indep[finite].min()

        out_finite = np.isfinite(loaded)
        check("R4 finite min == 0.0", bool(loaded[out_finite].min() == 0.0), "")
        nan_diff = int(np.count_nonzero(np.isnan(loaded) != np.isnan(forward)))
        check("R4 NaN mask preserved", nan_diff == 0, f"mismatched pixels = {nan_diff}")
        residual = np.abs(loaded[finite] - plane_removed_indep[finite]).max()
        # float32 payload storage gives ~1e-7 relative rounding; scale the
        # tolerance to the raw data range so the check accepts numerical noise
        # but still fails loudly if the plane is not actually removed.
        tol = 1e-5 * float(np.abs(forward[finite]).max()) or 1e-10
        check(
            "R4 independent plane residual ≈ 0",
            residual <= tol,
            f"max |diff| = {residual:.3e} (tol {tol:.3e})",
        )

    print(f"\n{SCRIPT.name} regression: {_FAILS} failure(s)")
    print("\n".join(_CHECK_RESULTS))
    return 1 if _FAILS else 0


def _read_z_forward_from_payload(sxm: Path) -> np.ndarray:
    """Re-read the Z forward matrix the way scripts/sxm_to_csv.py sees it."""
    from stm_data_processing.io.nanonis_loader import NanonisFileLoader

    loader = NanonisFileLoader(str(sxm))
    data_info = loader.header["DATA_INFO"]
    i_z = next(
        i
        for i, row in data_info.iterrows()
        if str(row["Name"]).strip() == "Z" and str(row["Unit"]).strip() == "m"
    )
    return loader.data[2 * i_z]


if __name__ == "__main__":
    raise SystemExit(main())
