"""End-to-end regression check for grid2h5 on real Nanonis grid files.

Four real ``.3ds`` files are converted (read-only; size and mtime are recorded
before each conversion and asserted unchanged afterwards):

* the c6lic6 grid-spectroscopy example, whose product is checked against
  expectations read off the file by hand (shape, channel names, sweep axis)
  and against the loader's public output for the parts where the loader is the
  declared input, so a converter regression cannot confirm itself;
* a file whose header carries a value spanning physical lines and a cp1252
  byte, so ``/header`` VALUES are checked against literals, not just key names;
* a file whose header entry is a table (the loader reports a DataFrame), so a
  non-string header value has to be stored as text;
* a truncated recording, which must still convert (the loader NaN-pads the
  pixels the file does not carry).

Because this script must reference absolute local data paths, it is
auto-classified ``localdata`` by ``tests/regression/test_regression_suite.py``
(the ``/Users/`` substring) and is therefore deselected in CI; run it on the
maintainer's workstation.

Run from the repository root:

    .venv/bin/python tests/regression/check_grid2h5_real.py

Exits non-zero when any check fails.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Importing the package pulls in matplotlib transitively; give it a writable
# config dir before matplotlib is first imported.
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "dsh_mplconfig"),
)

import h5py
import numpy as np

from stm_data_processing.io import h5_convention
from stm_data_processing.io.grid2h5 import channel_unit, grid_to_h5
from stm_data_processing.io.nanonis_loader import NanonisFileLoader

REPO_ROOT = Path(__file__).resolve().parents[2]

# Real Nanonis grid-spectroscopy examples (read-only).
REAL_3DS = (
    "/Users/hunfen/Documents/论文/c6lic6/data/raw/2025-11-10/Grid Spectroscopy001.3ds"
)
# A value spanning physical lines (a quoted condition closed by a lone '"'
# line) and a cp1252 byte in the sample name.
MULTILINE_3DS = (
    "/Users/hunfen/Documents/论文/c6lic6/data/raw/2025-09-11/Grid Spectroscopy001.3ds"
)
# A truncated recording: the loader NaN-pads the pixels the file lacks.
TRUNCATED_3DS = (
    "/Users/hunfen/Documents/论文/c6lic6/data/raw/2025-10-24/Grid Spectroscopy003.3ds"
)
# A header entry that is a table, so its value is not a string.
FRAME_3DS = (
    "/Users/hunfen/Documents/论文/Si111_Pb_islands/raw_data/2025-06-26/"
    "Grid Spectroscopy002.3ds"
)

# Expectations read off the file by hand, deliberately not computed from
# loader.header/loader.channels, so these checks are not self-referential.
NX, NY, NCH, NPTS = 304, 304, 14, 5
MEASURED_BIAS = (0.03, 0.005, 0.0, -0.005, -0.03)
LINSPACE_BIAS = np.linspace(0.03, -0.03, NPTS)
EXPECTED_CHANNELS = (
    "Current (A)",
    "Bias (AI5) (V)",
    "DSP 7280 X (%)",
    "DSP 7280 Y (%)",
    "Bias (V)",
    "LI Demod 1 X (A)",
    "LI Demod 1 Y (A)",
    "Current [bwd] (A)",
    "Bias (AI5) [bwd] (V)",
    "DSP 7280 X [bwd] (%)",
    "DSP 7280 Y [bwd] (%)",
    "Bias [bwd] (V)",
    "LI Demod 1 X [bwd] (A)",
    "LI Demod 1 Y [bwd] (A)",
)
EXPECTED_UNITS = tuple(channel_unit(name) for name in EXPECTED_CHANNELS)
EXPECTED_MULTILINE = {
    ("Sample", "Sample"): "BLG/SiC - r3\u00d7r3 Yb",
    ("Condition", "Condition"): "20250910 anneal@1.35W for 30mins\n",
}

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


def strings(handle: h5py.File, path: str) -> list[str]:
    """Return a string dataset as a list of ``str``."""
    return [value.decode() for value in handle[path][:]]


def stat_of(path: Path) -> tuple[int, int]:
    """Return ``(size, mtime_ns)`` of ``path``."""
    info = path.stat()
    return info.st_size, info.st_mtime_ns


def check_data(handle: h5py.File) -> None:
    """Check (a): the /data cube of the complete example."""
    data = handle["data"]
    values = np.asarray(data[:])
    check(
        "a1 /data has the header-declared shape (ny, nx, nch, npts)",
        data.shape == (NY, NX, NCH, NPTS),
        f"shape = {data.shape}, expected {(NY, NX, NCH, NPTS)}",
    )
    check(
        "a1 /data is float32 and fully finite",
        data.dtype == np.float32 and not np.isnan(values).any(),
        f"dtype = {data.dtype}, NaN count = {int(np.isnan(values).sum())}",
    )
    check(
        "a1 /data records the source dtype and the axis order",
        data.attrs.get("source_dtype") == ">f4"
        and data.attrs.get("axis_order") == "(ny, nx, nch, npts)",
        f"source_dtype = {data.attrs.get('source_dtype')!r}, "
        f"axis_order = {data.attrs.get('axis_order')!r}",
    )


def check_channels(handle: h5py.File) -> None:
    """Check (b): the channel names and their units."""
    check(
        "b1 /channels holds the file's channel names in order",
        tuple(strings(handle, "channels")) == EXPECTED_CHANNELS,
        f"got {strings(handle, 'channels')}",
    )
    check(
        "b1 /channel_units parses the unit of every channel name",
        tuple(strings(handle, "channel_units")) == EXPECTED_UNITS,
        f"got {strings(handle, 'channel_units')}",
    )


def check_bias(handle: h5py.File, source: Path) -> None:
    """Check (c): /bias is the measured sweep channel, not a linspace."""
    loader = NanonisFileLoader(str(source))
    signal = str(loader.header.get("Sweep Signal", ""))
    index = [str(channel) for channel in loader.channels].index(signal)
    measured = np.asarray(loader.data[:, index, :], dtype=np.float64)[0]
    bias = handle["bias"]
    check(
        "c1 /bias equals the measured sweep channel row of the file",
        np.array_equal(bias[:], measured),
        f"bias = {bias[:].tolist()}, measured = {measured.tolist()}",
    )
    check(
        "c1 /bias is the multi-segment row, not linspace(start, end)",
        not np.allclose(bias[:], LINSPACE_BIAS),
        f"bias {bias[:].tolist()} vs linspace {LINSPACE_BIAS.tolist()}",
    )
    check(
        "c1 /bias records the measured-channel source and its unit",
        bias.attrs.get("bias_source") == "measured_channel"
        and bias.attrs.get("units") == "V",
        f"bias_source = {bias.attrs.get('bias_source')!r}, "
        f"units = {bias.attrs.get('units')!r}",
    )


def check_params(handle: h5py.File, source: Path) -> None:
    """Check (d): the per-pixel parameter table and its column names."""
    loader = NanonisFileLoader(str(source))
    # The table can hold NaN (a parameter the file did not record), so the
    # comparison is NaN-safe.
    want = loader.parameters.values.astype(np.float32).astype(np.float64)
    params = handle["params"][:]
    check(
        "d1 /params matches the loader's per-pixel parameter table",
        params.shape == want.shape and np.array_equal(params, want, equal_nan=True),
        f"shape = {params.shape}, element(s) differing = "
        f"{int(np.count_nonzero(~(params == want))) if params.shape == want.shape else 'n/a'}",
    )
    header = loader.header
    columns = [
        item
        for key in ("Fixed parameters", "Experiment parameters")
        for item in str(header.get(key, "")).split(";")
        if item
    ]
    check(
        "d1 /param_columns matches the header's Fixed + Experiment names",
        strings(handle, "param_columns") == columns,
        f"got {strings(handle, 'param_columns')}, expected {columns}",
    )


def check_header(handle: h5py.File, source: Path) -> None:
    """Check (e): /header mirrors every loader.header key and keeps its values."""
    loader = NanonisFileLoader(str(source))
    group = handle["header"]
    missing: list[str] = []
    wrong: list[str] = []
    for key, value in loader.header.items():
        if isinstance(value, dict):
            if str(key) not in group:
                missing.append(str(key))
                continue
            subgroup = group[str(key)]
            for leaf, leaf_value in value.items():
                text = leaf_value if isinstance(leaf_value, str) else str(leaf_value)
                if str(leaf) not in subgroup.attrs:
                    missing.append(f"{key}>{leaf}")
                elif subgroup.attrs[str(leaf)] != text:
                    wrong.append(f"{key}>{leaf}")
        elif str(key) not in group.attrs:
            missing.append(str(key))
        elif group.attrs[str(key)] != (value if isinstance(value, str) else str(value)):
            wrong.append(str(key))
    check(
        "e1 every loader.header key appears in /header",
        not missing,
        f"missing = {missing[:4]}",
    )
    check(
        "e1 every mirrored value equals the loader's value",
        not wrong,
        f"different = {wrong[:4]}",
    )
    check(
        "e1 a '/' in a leaf name is kept verbatim",
        group["Bias"].attrs.get("Calibration (V/V)") == "10E-3",
        f"Bias attrs = {dict(group['Bias'].attrs)}",
    )
    undecoded = [
        f"{key}/{leaf}"
        for key, value in loader.header.items()
        if isinstance(value, dict)
        for leaf in value
        if "\ufffd" in str(group[str(key)].attrs.get(str(leaf), ""))
    ]
    check(
        "e1 no mirrored header value contains U+FFFD",
        not undecoded,
        f"keys with U+FFFD = {undecoded}",
    )


def check_multiline_values(tmp: Path) -> None:
    """Check (f): values spanning lines and cp1252 bytes survive verbatim.

    Mutation that turns these red: removing more than one quote pair, or
    normalising the text (which would mangle the newline or the cp1252
    character).
    """
    source = Path(MULTILINE_3DS)
    if not source.is_file():
        check("f0 the multi-line header file is present", False, f"missing {source}")
        return
    out = tmp / "multiline_real.h5"
    grid_to_h5(source, out)
    with h5py.File(out, "r") as handle:
        group = handle["header"]
        for (key, leaf), expected in EXPECTED_MULTILINE.items():
            got = group[key].attrs.get(leaf)
            check(
                f"f0 /header value {key}>{leaf} survives verbatim",
                got == expected,
                f"got {got!r}, expected {expected!r}",
            )


def check_frame_leaf(tmp: Path) -> None:
    """Check (f): a non-string header value is stored as its text."""
    source = Path(FRAME_3DS)
    if not source.is_file():
        check("f1 the table-valued header file is present", False, f"missing {source}")
        return
    out = tmp / "frame_real.h5"
    grid_to_h5(source, out)
    loader = NanonisFileLoader(str(source))
    found = [
        (str(module), str(leaf))
        for module, value in loader.header.items()
        if isinstance(value, dict)
        for leaf, leaf_value in value.items()
        if not isinstance(leaf_value, str)
    ]
    with h5py.File(out, "r") as handle:
        group = handle["header"]
        stored = [
            group[module].attrs.get(leaf) for module, leaf in found if module in group
        ]
        check(
            "f1 a table-valued header leaf is stored as text",
            bool(found)
            and len(stored) == len(found)
            and all(isinstance(text, str) and text for text in stored),
            f"leaf(ves) = {found}, stored = {[str(t)[:40] for t in stored]}",
        )
        check(
            "f1 the stored text is the loader value's text form",
            all(
                group[module].attrs.get(leaf) == str(loader.header[module][leaf])
                for module, leaf in found
            ),
            f"stored = {[str(t)[:40] for t in stored]}",
        )


def check_truncated(tmp: Path) -> None:
    """Check (g): a truncated recording still converts."""
    source = Path(TRUNCATED_3DS)
    if not source.is_file():
        check("g1 the truncated real file is present", False, f"missing {source}")
        return
    out = tmp / "truncated_real.h5"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "stm_data_processing.io.grid2h5",
            str(source),
            "-o",
            str(out),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    check(
        "g1 a truncated .3ds converts instead of erroring",
        completed.returncode == 0 and out.exists(),
        f"exit = {completed.returncode}, stderr = {completed.stderr.strip()[:120]!r}",
    )
    if not out.exists():
        return
    loader = NanonisFileLoader(str(source))
    dims = str(loader.header["Grid dim"]).replace(" ", "").split("x")
    declared = (int(dims[1]), int(dims[0]))
    with h5py.File(out, "r") as handle:
        data = handle["data"]
        values = np.asarray(data[:])
        check(
            "g1 the truncated product keeps the header-declared shape",
            data.shape[:2] == declared,
            f"shape = {data.shape}, header grid = {declared}",
        )
        check(
            "g1 the pixels the file does not carry are NaN",
            bool(np.isnan(values).any()),
            f"NaN count = {int(np.isnan(values).sum())}",
        )
        check(
            "g1 the truncated product still carries /bias",
            handle["bias"].shape[0] == data.shape[3],
            f"bias shape = {handle['bias'].shape}",
        )


def check_reconversion_keeps_date(tmp: Path) -> None:
    """Check (h): a second conversion of the same destination keeps its date."""
    source = Path(REAL_3DS)
    out = tmp / "again_real.h5"
    grid_to_h5(source, out)
    first = h5_convention.read_creation_date(out)
    grid_to_h5(source, out)
    second = h5_convention.read_creation_date(out)
    check(
        "h1 re-converting the same destination keeps its creation date",
        first is not None and first == second,
        f"{first!r} -> {second!r}",
    )


def main() -> int:
    """Run every check; return the number of failures (0 = success)."""
    source = Path(REAL_3DS)
    if not source.is_file():
        check("the real example file is present", False, f"missing {source}")
        print("\n".join(_CHECK_RESULTS))
        return 1
    before = stat_of(source)
    with tempfile.TemporaryDirectory(prefix="grid2h5_real_") as workdir:
        tmp = Path(workdir)
        out = tmp / "example_real.h5"
        grid_to_h5(source, out)
        with h5py.File(out, "r") as handle:
            check(
                "z1 the product holds every expected dataset",
                all(
                    name in handle
                    for name in (
                        "data",
                        "channels",
                        "channel_units",
                        "bias",
                        "params",
                        "param_columns",
                        "header",
                    )
                ),
                f"datasets = {sorted(handle.keys())}",
            )
            check_data(handle)
            check_channels(handle)
            check_bias(handle, source)
            check_params(handle, source)
            check_header(handle, source)
        check(
            "z2 the .3ds source is unmodified (size and mtime unchanged)",
            stat_of(source) == before,
            f"before = {before}, after = {stat_of(source)}",
        )
        check_multiline_values(tmp)
        check_frame_leaf(tmp)
        check_truncated(tmp)
        check_reconversion_keeps_date(tmp)

    print("\n".join(_CHECK_RESULTS))
    print(f"\ngrid2h5 real-data regression: {_FAILS} failure(s)")
    print(f"assertions executed: {_ASSERTS}")
    return 1 if _FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
