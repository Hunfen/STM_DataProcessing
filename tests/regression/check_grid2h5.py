"""Regression check for the grid2h5 converter on pure synthetic fixtures.

Data-independent end-to-end test: it builds minimal but legal Nanonis ``.3ds``
grid-spectroscopy files in a temporary directory (no real measurement data
anywhere), runs :func:`stm_data_processing.io.grid2h5.grid_to_h5` on them and
checks the converter's contract.  Every expected value is recomputed here from
the fixture payload this script wrote itself, never copied from the converter,
so a wrong implementation cannot confirm itself.  Each of these naive
implementations turns a named check below red:

* a ``/bias`` rule that always builds ``linspace(sweep_start, sweep_end)``
  instead of preferring the measured multi-segment sweep channel (c1), and one
  that always uses the measured channel even when the pixels disagree (c2);
* a hard-coded or transposed ``/data`` shape (a1);
* a non-generic ``/header`` dump that enumerates known metadata keys instead
  of mirroring every ``loader.header`` key, or that loses a module, a ``/`` in
  a module name, a value spanning two lines or an empty value (e1);
* a converter that refuses a payload the file does not fully carry (f1);
* a converter that drops the per-pixel parameter table, mixes up its column
  names (b1) or loses the unit of a channel (d1);
* a CLI that writes somewhere other than ``<source>.h5`` by default, or that
  exits 0 on a source it cannot convert (g1, i1-i4).

No local absolute data path (e.g. the maintainer's home directory) appears in
this file, so ``pytest -m "not localdata"`` (the CI gate) selects it.

Run from the repository root:

    .venv/bin/python tests/regression/check_grid2h5.py

Exits non-zero when any check fails.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Importing the package pulls in matplotlib transitively; give it a writable
# config dir before matplotlib is first imported, as check_sxm_to_csv.py does.
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "dsh_mplconfig"),
)

import h5py
import numpy as np

from stm_data_processing.io.grid2h5 import Grid2H5Error, grid_to_h5

REPO_ROOT = Path(__file__).resolve().parents[2]

# Non-square grid (nx = 3 samples per line along the fast axis, ny = 2 lines),
# so the axis order of /data (ny, nx, channel, point) is distinguishable from
# an (nx, ny, ...) layout, which a square grid could never reveal.
NX, NY = 3, 2
NCH, NPTS, NPARAM = 3, 5, 4
N_PIXELS = NX * NY

PARAM_NAMES = ("Sweep Start", "Sweep End", "X (m)", "Y (m)")
CHANNELS = ("Current (A)", "Bias (V)", "DSP 7280 X (%)")
CHANNEL_UNITS = ("A", "V", "%")
SWEEP_SIGNAL = "Bias (V)"
SWEEP_START, SWEEP_END = 0.03, -0.03
# Multi-segment measured bias: NOT a linspace between Sweep Start and End
# (that would be [0.03, 0.015, 0.0, -0.015, -0.03]).
MEASURED_BIAS = (0.03, 0.005, 0.0, -0.005, -0.03)
LINSPACE_BIAS = tuple(np.linspace(SWEEP_START, SWEEP_END, NPTS).tolist())
# A deliberately per-pixel-varying sweep row: with this one the measured
# channel is not identical across pixels, so /bias must be the linspace of the
# first pixel's sweep bounds.
VARYING_ROW = (0.09, 0.005, 0.0, -0.005, -0.09)

# Datasets the product must contain (plus the /header group).
EXPECTED_DATASETS = (
    "data",
    "channels",
    "channel_units",
    "bias",
    "params",
    "param_columns",
    "header",
)

_CHECK_RESULTS: list[str] = []
_FAILS = 0
_ASSERTS = 0


def check(name: str, ok: bool, detail: str) -> None:
    """Record one PASS/FAIL result and count it as an assertion."""
    global _FAILS, _ASSERTS
    if not ok:
        _FAILS += 1
    _ASSERTS += 1
    _CHECK_RESULTS.append(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")


def float32(value: float) -> np.float32:
    """Encode ``value`` the way the loader sees it (big-endian float32)."""
    return np.asarray(value, dtype=">f4").astype(np.float32)


def build_fixture(
    *,
    grid: tuple[int, int] = (NX, NY),
    points: int = NPTS,
    sweep_rows: list[tuple[float, ...]] | None = None,
) -> tuple[np.ndarray, dict]:
    """Assemble the per-pixel parameters and channel data (float32 arrays).

    Returns ``(payload, expected)``: the flat big-endian float32 data block in
    Nanonis per-pixel block order (parameters then channel-major channel data)
    and a dict the checks use to rebuild the expected ``/data`` and ``/params``
    without sharing any code with the converter.
    """
    nx, ny = grid
    pixels = nx * ny
    if sweep_rows is None:
        if points != NPTS:
            error_msg = "the default sweep row only fits the default point count"
            raise ValueError(error_msg)
        sweep_rows = [tuple(MEASURED_BIAS)] * pixels
    expected_params = np.empty((pixels, NPARAM), dtype=np.float64)
    for n in range(pixels):
        expected_params[n] = [
            float32(SWEEP_START),
            float32(SWEEP_END),
            float32(n),
            float32(2 * n + 1),
        ]
    expected_grid = np.empty((pixels, NCH, points), dtype=np.float64)
    for n in range(pixels):
        expected_grid[n, 0] = [float32(n * 10 + p) for p in range(points)]
        expected_grid[n, 1] = [float32(v) for v in sweep_rows[n]]
        expected_grid[n, 2] = [float32(100 - n * 5 + p) for p in range(points)]
    blocks = []
    for n in range(pixels):
        blocks.append(expected_params[n].astype(">f4").reshape(-1))
        blocks.append(expected_grid[n].astype(">f4").reshape(-1))
    return np.concatenate(blocks), {"params": expected_params, "grid": expected_grid}


#: Header extras every fixture carries: a module with two leaves, a
#: single-attribute module, a ``/`` in a module name, a brand-new top-level
#: key, a value spanning two lines and an empty value.  A converter that
#: enumerates the keys it knows, or that refuses ``/`` or a newline, turns e1
#: red.
EXTRA_HEADER = (
    "Future Top-Level Key=some brand-new scalar",
    "Ext. VI 1>FutureTool>New Knob=3.14",
    "Ext. VI 1>FutureTool>Future Mode=on",
    "Ext. VI 1>Solo>One leaf=only one",
    "Odd/Module>x=1",
    'Ext. VI 1>Comment="first line',
    'second line"',
)


def write_3ds(
    path: Path,
    payload: np.ndarray,
    *,
    grid: tuple[int, int] = (NX, NY),
    points: int = NPTS,
    fixed_parameters: str = "Sweep Start;Sweep End",
    drop: tuple[str, ...] = (),
    cut_bytes: int = 0,
) -> None:
    """Write a minimal legal .3ds holding ``payload`` as the big-endian data.

    Header lines are CRLF-terminated ``key=value`` pairs; the data block
    follows a line containing ``:HEADER_END:`` plus CRLF.  ``drop`` removes
    header lines by their leading key and ``cut_bytes`` truncates the data
    block, both for the error and truncation cases.
    """
    nx, ny = grid
    header_lines = [
        # Space-separated dims, as real Nanonis headers write them
        # ("304 x 304"); the loader strips the spaces before splitting on "x".
        f"Grid dim={nx} x {ny}",
        f"Points={points}",
        f"# Parameters (4 byte)={NPARAM}",
        f"Experiment size (bytes)={4 * NCH * points}",
        f"Fixed parameters={fixed_parameters}",
        "Experiment parameters=X (m);Y (m)",
        "Channels=" + ";".join(CHANNELS),
        f"Sweep Signal={SWEEP_SIGNAL}",
        "User=",
        *EXTRA_HEADER,
    ]
    dropped = set(drop)
    data = payload.astype(">f4").tobytes()
    if cut_bytes:
        data = data[: len(data) - cut_bytes]
    with path.open("wb") as handle:
        for line in header_lines:
            if line.split("=", 1)[0] in dropped:
                continue
            handle.write(f"{line}\r\n".encode())
        handle.write(b"\x1a:HEADER_END:\r\n")
        handle.write(data)


def fixture(
    tmp: Path,
    name: str = "fixture",
    *,
    convert: bool = True,
    **kwargs,
) -> tuple[Path, Path, dict]:
    """Write a fixture ``.3ds``, convert it and return its paths and values."""
    src = tmp / f"{name}.3ds"
    out = tmp / f"{name}.h5"
    payload, expected = build_fixture(
        grid=kwargs.get("grid", (NX, NY)),
        points=kwargs.get("points", NPTS),
        sweep_rows=kwargs.get("sweep_rows"),
    )
    write_3ds(
        src,
        payload,
        grid=kwargs.get("grid", (NX, NY)),
        points=kwargs.get("points", NPTS),
        fixed_parameters=kwargs.get("fixed_parameters", "Sweep Start;Sweep End"),
        drop=kwargs.get("drop", ()),
        cut_bytes=kwargs.get("cut_bytes", 0),
    )
    if convert:
        grid_to_h5(src, out)
    return src, out, expected


def check_data(handle, expected) -> None:
    """Checks (a): shape, dtype, axis order and elementwise mapping."""
    data = handle["data"]
    check(
        "a1 /data shape is (ny, nx, nch, npts) with ny from Grid dim",
        data.shape == (NY, NX, NCH, NPTS),
        f"shape = {data.shape}, expected {(NY, NX, NCH, NPTS)}",
    )
    check(
        "a1 /data dtype is float32",
        data.dtype == np.float32,
        f"dtype = {data.dtype}",
    )
    check(
        "a1 /data records the source dtype and the axis order",
        data.attrs.get("source_dtype") == ">f4"
        and data.attrs.get("axis_order") == "(ny, nx, nch, npts)",
        f"source_dtype = {data.attrs.get('source_dtype')!r}, "
        f"axis_order = {data.attrs.get('axis_order')!r}",
    )
    # Mutation that turns the next check red: transposing the two pixel axes
    # after a correct reshape, which keeps the shape but reorders elements.
    # Pixel n = iy*nx + ix, x fastest, so /data[iy, ix] must equal
    # expected_grid[n].
    if data.shape == (NY, NX, NCH, NPTS):
        got = np.asarray(data[:])
        want = expected["grid"].astype(np.float32).reshape(NY, NX, NCH, NPTS)
        mismatched = np.argwhere(got != want)
        detail = (
            "every pixel/channel/point matches"
            if mismatched.size == 0
            else f"{mismatched.shape[0]} element(s) differ, "
            f"first at (iy, ix, ch, pt) = {mismatched[0].tolist()}"
        )
        check("a1 elementwise /data pixel-index mapping", mismatched.size == 0, detail)
    else:
        check(
            "a1 elementwise /data pixel-index mapping",
            False,
            f"cannot verify the mapping for shape {data.shape}",
        )


def check_params(handle, expected) -> None:
    """Check (b): /params and /param_columns."""
    params = handle["params"][:]
    want = expected["params"].astype(np.float32).astype(np.float64)
    check(
        "b1 /params has shape (npix, nparam)",
        params.shape == (N_PIXELS, NPARAM),
        f"shape = {params.shape}, expected {(N_PIXELS, NPARAM)}",
    )
    check(
        "b1 /params elementwise equality with the fixture parameter rows",
        params.shape == want.shape and np.array_equal(params, want),
        f"max |diff| = {np.abs(params - want).max():.3e}"
        if params.shape == want.shape
        else f"shape {params.shape} != {want.shape}",
    )
    columns = [s.decode() for s in handle["param_columns"][:]]
    check(
        "b1 /param_columns == Fixed + Experiment names",
        columns == list(PARAM_NAMES),
        f"got {columns}, expected {list(PARAM_NAMES)}",
    )


def check_bias_measured(tmp: Path) -> None:
    """Check (c1): /bias prefers the measured multi-segment sweep channel."""
    _, out, _ = fixture(tmp, "measured")
    with h5py.File(out, "r") as handle:
        dataset = handle["bias"]
        bias = dataset[:]
        want = np.asarray([float32(v) for v in MEASURED_BIAS]).astype(np.float64)
        check(
            "c1 /bias equals the measured sweep channel row",
            np.array_equal(bias, want),
            f"bias = {bias.tolist()}, expected {list(MEASURED_BIAS)}",
        )
        # Mutation that turns this red: building /bias as
        # linspace(Sweep Start, Sweep End, Points), which is what the header
        # parameters alone suggest.  The measured channel is deliberately
        # multi-segment, so the two rules disagree at the interior points.
        check(
            "c1 /bias is NOT linspace(sweep_start, sweep_end, npts)",
            not np.allclose(bias, np.asarray(LINSPACE_BIAS)),
            f"bias {bias.tolist()} must differ from linspace {list(LINSPACE_BIAS)}",
        )
        check(
            "c1 /bias records the measured-channel source",
            dataset.attrs.get("bias_source") == "measured_channel",
            f"bias_source = {dataset.attrs.get('bias_source')!r}",
        )
        check(
            "c1 /bias carries the unit of the sweep channel name",
            dataset.attrs.get("units") == "V",
            f"units = {dataset.attrs.get('units')!r}",
        )


def check_bias_linspace(tmp: Path) -> None:
    """Check (c2): a per-pixel-varying sweep channel falls back to linspace."""
    rows = [tuple(MEASURED_BIAS)] * N_PIXELS
    rows[1] = VARYING_ROW
    _, out, _ = fixture(tmp, "fallback", sweep_rows=rows)
    with h5py.File(out, "r") as handle:
        dataset = handle["bias"]
        bias = dataset[:]
        # Mutation that turns this red: using the measured channel whenever a
        # 'Sweep Signal' channel exists, without checking that the pixels
        # agree - /bias would then be pixel 0's row while pixel 1 recorded a
        # different sweep.
        check(
            "c2 /bias is the linspace of the first pixel's sweep bounds",
            np.allclose(bias, np.asarray(LINSPACE_BIAS)),
            f"bias = {bias.tolist()}, expected linspace {list(LINSPACE_BIAS)}",
        )
        check(
            "c2 /bias records the linspace source",
            dataset.attrs.get("bias_source") == "linspace",
            f"bias_source = {dataset.attrs.get('bias_source')!r}",
        )
        check(
            "c2 /bias still carries a unit",
            dataset.attrs.get("units") == "V",
            f"units = {dataset.attrs.get('units')!r}",
        )


def check_channels(handle) -> None:
    """Check (d): channel names and their parsed units."""
    names = [s.decode() for s in handle["channels"][:]]
    check(
        "d1 /channels keeps the loader channel names and order",
        names == list(CHANNELS),
        f"got {names}, expected {list(CHANNELS)}",
    )
    units = [s.decode() for s in handle["channel_units"][:]]
    check(
        "d1 /channel_units parses 'A', 'V' and '%'",
        units == list(CHANNEL_UNITS),
        f"got {units}, expected {list(CHANNEL_UNITS)}",
    )


def check_header(tmp: Path) -> None:
    """Check (e): /header mirrors loader.header key by key."""
    _, out, _ = fixture(tmp, "header")
    with h5py.File(out, "r") as handle:
        group = handle["header"]
        # Mutation that turns these red: enumerating the handful of keys the
        # module happens to know (so a key added by a newer Nanonis version is
        # silently dropped) instead of mirroring the loader's dict.
        check(
            "e1 a brand-new top-level key appears under /header",
            group.attrs.get("Future Top-Level Key") == "some brand-new scalar",
            f"value = {group.attrs.get('Future Top-Level Key')!r}",
        )
        check(
            "e1 a module becomes a subgroup holding its leaves",
            "FutureTool" in group
            and group["FutureTool"].attrs.get("New Knob") == "3.14"
            and group["FutureTool"].attrs.get("Future Mode") == "on",
            f"FutureTool attrs = "
            f"{dict(group['FutureTool'].attrs) if 'FutureTool' in group else None}",
        )
        check(
            "e1 a single-attribute module keeps its leaf name",
            "Solo" in group and group["Solo"].attrs.get("One leaf") == "only one",
            f"Solo attrs = {dict(group['Solo'].attrs) if 'Solo' in group else None}",
        )
        # Mutation that turns this red: using the module name verbatim in
        # create_group, which makes h5py nest 'Odd' -> 'Module' and loses the
        # module.
        check(
            "e1 '/' in a module name cannot create a nested group",
            "Odd%2FModule" in group
            and "Odd" not in group
            and group["Odd%2FModule"].attrs.get("x") == "1",
            f"groups = {sorted(group.keys())}",
        )
        check(
            "e1 a value spanning two lines survives verbatim",
            group["Comment"].attrs.get("Comment") == "first line\nsecond line",
            f"Comment = {group['Comment'].attrs.get('Comment')!r}",
        )
        check(
            "e1 an empty value stays an empty string",
            group.attrs.get("User") == "",
            f"User = {group.attrs.get('User')!r}",
        )


def check_truncated_payload(tmp: Path) -> None:
    """Check (f): a payload the file does not fully carry still converts."""
    # The loader NaN-pads the missing pixels, so the cube keeps the
    # header-declared shape.  Mutation that turns this red: refusing a short
    # payload instead of converting it.
    cut = NCH * NPTS * 4 * 2  # two pixel blocks' worth of channel data
    _, out, _ = fixture(tmp, "truncated", cut_bytes=cut)
    want = build_fixture()[1]["grid"].astype(np.float32).reshape(NY, NX, NCH, NPTS)
    with h5py.File(out, "r") as handle:
        data = handle["data"]
        values = np.asarray(data[:])
        check(
            "f1 a truncated payload keeps the header-declared shape",
            data.shape == (NY, NX, NCH, NPTS),
            f"shape = {data.shape}",
        )
        check(
            "f1 the pixels the file does not carry are NaN, not zero",
            bool(np.isnan(values).any()),
            f"NaN count = {int(np.isnan(values).sum())}",
        )
        check(
            "f1 the pixels the file does carry keep their values",
            np.array_equal(
                values[0, 0, 0][: np.count_nonzero(~np.isnan(values[0, 0, 0]))],
                want[0, 0, 0][: np.count_nonzero(~np.isnan(values[0, 0, 0]))],
            ),
            f"first pixel channel 0 = {values[0, 0, 0].tolist()}",
        )
        check(
            "f1 /bias is still written for a truncated file",
            handle["bias"].shape == (NPTS,),
            f"bias shape = {handle['bias'].shape}",
        )


def check_default_destination_and_cli(tmp: Path) -> None:
    """Check (g): the default output path and the CLI."""
    src, _, _ = fixture(tmp, "cli", convert=False)
    completed = subprocess.run(
        [sys.executable, "-m", "stm_data_processing.io.grid2h5", str(src)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    default = src.with_suffix(".h5")
    check(
        "g1 the CLI converts and exits 0",
        completed.returncode == 0,
        f"exit = {completed.returncode}, stderr = {completed.stderr.strip()[:120]!r}",
    )
    check(
        "g1 without -o the product is <source>.h5",
        default.exists(),
        f"{default.name} exists = {default.exists()}",
    )
    if default.exists():
        with h5py.File(default, "r") as handle:
            check(
                "g1 the default-path product holds every dataset",
                all(name in handle for name in EXPECTED_DATASETS),
                f"datasets = {sorted(handle.keys())}",
            )


def check_creation_date_preserved(tmp: Path) -> None:
    """Check (h): re-converting a product keeps its creation date."""
    src, out, _ = fixture(tmp, "again")
    with h5py.File(out, "r") as handle:
        first = handle.attrs.get("creation_date")
    grid_to_h5(src, out)
    with h5py.File(out, "r") as handle:
        second = handle.attrs.get("creation_date")
    check(
        "h1 re-converting the same destination keeps its creation date",
        first is not None and first == second,
        f"{first!r} -> {second!r}",
    )


def check_error_paths(tmp: Path) -> None:
    """Check (i): a source that cannot be converted fails loudly, writes nothing."""
    missing = tmp / "absent.3ds"
    refused: Exception | None = None
    try:
        grid_to_h5(missing)
    except Exception as exc:  # reported as a FAIL, never raised
        refused = exc
    check(
        "i1 a missing source is refused by name",
        isinstance(refused, Grid2H5Error) and missing.name in str(refused),
        f"exception = {type(refused).__name__}: {refused}",
    )
    check(
        "i1 a refused source writes no product",
        not missing.with_suffix(".h5").exists(),
        f"product = {missing.with_suffix('.h5').exists()}",
    )

    wrong = tmp / "notagrid.dat"
    wrong.write_bytes(b"nope")
    refused = None
    try:
        grid_to_h5(wrong)
    except Exception as exc:  # reported as a FAIL, never raised
        refused = exc
    check(
        "i2 a non-.3ds source is refused with its suffix named",
        isinstance(refused, Grid2H5Error) and ".3ds" in str(refused),
        f"exception = {type(refused).__name__}: {refused}",
    )

    # A header that does not name the grid: the converter must say which field
    # is missing instead of raising a bare lookup error, and must not leave a
    # half-built product behind.
    src, out, _ = fixture(tmp, "no_grid", convert=False, drop=("Grid dim",))
    refused = None
    try:
        grid_to_h5(src, out)
    except Exception as exc:  # reported as a FAIL, never raised
        refused = exc
    check(
        "i3 a missing header field is refused by name",
        isinstance(refused, Grid2H5Error) and "Grid dim" in str(refused),
        f"exception = {type(refused).__name__}: {refused}",
    )
    check(
        "i3 the failed conversion leaves no product",
        not out.exists(),
        f"product = {out.exists()}",
    )

    # Sweep bounds are what the linspace fallback needs; without them (and
    # without a measured sweep channel) the source must be a named error too.
    src, out, _ = fixture(
        tmp,
        "no_sweep",
        convert=False,
        drop=("Sweep Signal",),
        fixed_parameters="Bias (V);Setpoint (A)",
    )
    refused = None
    try:
        grid_to_h5(src, out)
    except Exception as exc:  # reported as a FAIL, never raised
        refused = exc
    check(
        "i4 missing sweep bound columns are refused by name",
        isinstance(refused, Grid2H5Error)
        and "Sweep Start" in str(refused)
        and "Sweep End" in str(refused),
        f"exception = {type(refused).__name__}: {refused}",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "stm_data_processing.io.grid2h5",
            str(src),
            "-o",
            str(out),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    check(
        "i4 the CLI exits 1 with the message, not a traceback",
        completed.returncode == 1
        and "grid2h5: error:" in completed.stderr
        and "Traceback" not in completed.stderr,
        f"exit = {completed.returncode}, stderr = {completed.stderr.strip()[:140]!r}",
    )


def main() -> int:
    """Run every check; return the number of failures (0 = success)."""
    with tempfile.TemporaryDirectory(prefix="grid2h5_synth_") as workdir:
        tmp = Path(workdir)
        _, out, expected = fixture(tmp, "fixture")
        with h5py.File(out, "r") as handle:
            check(
                "z1 the product holds every expected dataset",
                all(name in handle for name in EXPECTED_DATASETS),
                f"datasets = {sorted(handle.keys())}",
            )
            check_data(handle, expected)
            check_params(handle, expected)
            check_channels(handle)
        check_bias_measured(tmp)
        check_bias_linspace(tmp)
        check_header(tmp)
        check_truncated_payload(tmp)
        check_default_destination_and_cli(tmp)
        check_creation_date_preserved(tmp)
        check_error_paths(tmp)

    print("\n".join(_CHECK_RESULTS))
    print(f"\ngrid2h5 synthetic regression: {_FAILS} failure(s)")
    print(f"assertions executed: {_ASSERTS}")
    return 1 if _FAILS else 0


if __name__ == "__main__":
    with contextlib.suppress(BrokenPipeError):
        sys.exit(main())
