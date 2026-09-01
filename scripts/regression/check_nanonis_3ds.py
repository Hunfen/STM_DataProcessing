"""Regression checks for the Nanonis 3DS / DAT loader fixes (M3, M16, M19).

Verifies, against two real Nanonis files (read-only) plus synthetic edge
cases, that:

- M3: a .dat file without a "Bias Spectroscopy" module parses its header
  without raising KeyError; a file WITH a "Bias Spectroscopy" module gets
  its "MultiLine Settings" converted to a DataFrame (both .dat and .3ds
  header paths).
- M16: a missing/zero block layout raises a ValueError naming the missing
  header fields; raw data longer than the expected total point count is
  truncated safely with a logger warning; params/grid never go out of
  bounds.
- M19: empty strings in "Fixed parameters" / "Experiment parameters" are
  filtered out, and a column-count mismatch with "# Parameters (4 byte)"
  raises instead of silently writing values into the wrong columns.

Run from the repository root:

    .venv/bin/python scripts/regression/check_nanonis_3ds.py

Exits with a non-zero status when any check fails. The real data files are
only ever opened for reading.
"""

import logging
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from stm_data_processing.io.nanonis_loader import NanonisFileLoader

# Real Nanonis data files (read-only). Optional CLI arguments may override.
LINE_3DS = Path(
    "/Users/hunfen/Documents/论文/Si111_Pb_islands/raw_data/2025-07-09/"
    "Grid Spectroscopy002.3ds",
)
GRID_3DS = Path(
    "/Users/hunfen/Documents/论文/Si111_Pb_islands/raw_data/2025-10-24/"
    "Grid Spectroscopy008.3ds",
)
# Real 3ds file whose header contains 'Bias Spectroscopy>MultiLine
# Settings' (note: 2025-07-09/Grid Spectroscopy002.3ds does NOT carry that
# key; the MultiLine-bearing 002.3ds lives in 2025-06-25).
MULTILINE_3DS = Path(
    "/Users/hunfen/Documents/论文/Si111_Pb_islands/raw_data/2025-06-25/"
    "Grid Spectroscopy002.3ds",
)
# Real 3ds file used to verify the channels property is order-independent
# with respect to header parsing and yields quote-free names.
CHANNELS_3DS = Path(
    "/Users/hunfen/Documents/论文/Si111_Pb_islands/raw_data/2025-07-14/"
    "Grid Spectroscopy001.3ds",
)

HEADER_FIELDS = (
    "Grid dim",
    "Points",
    "Channels",
    "# Parameters (4 byte)",
    "Experiment size (bytes)",
)


def write_3ds(path: Path, header: dict[str, str], floats: list[float]) -> None:
    """Write a synthetic .3ds file (key=value header + big-endian f4 data)."""
    with path.open("wb") as f:
        for key, value in header.items():
            f.write(f"{key}={value}\r\n".encode())
        f.write(b":HEADER_END:\r\n")
        f.write(np.asarray(floats, dtype=">f4").tobytes())


def write_dat(
    path: Path,
    header_lines: list[tuple[str, str]],
    columns: list[str],
    rows: list[list[str]],
) -> None:
    """Write a synthetic .dat file (tab-separated header + data section)."""
    with path.open("wb") as f:
        for key, value in header_lines:
            f.write(f"{key}\t{value}\t\r\n".encode())
        f.write(b"[DATA]\r\n")
        f.write(("\t".join(columns) + "\r\n").encode())
        for row in rows:
            f.write(("\t".join(row) + "\r\n").encode())


def check_real_file(path: Path) -> None:
    """Load a real 3ds file, print key fields and assert self-consistency."""
    print(f"--- real file: {path.name} ---")
    loader = NanonisFileLoader(str(path))
    header = loader.header
    for field in HEADER_FIELDS:
        print(f"  {field}: {header.get(field)!r}")

    params = loader.parameters
    grid = loader.data
    print(f"  params shape: {params.shape}, columns: {list(params.columns)}")
    print(f"  grid shape: {grid.shape}, NaN ratio: {float(np.isnan(grid).mean()):.4f}")

    points = int(header["Points"])
    param_length = int(header["# Parameters (4 byte)"])
    grid_dim = loader.pixels
    total_pixels = grid_dim[0] * grid_dim[1]
    channels = header["Channels"].split(";")

    expected_grid = (total_pixels, len(channels), points)
    assert grid.shape == expected_grid, (
        f"grid shape {grid.shape} != expected {expected_grid}"
    )
    expected_params = (total_pixels, param_length)
    assert params.shape == expected_params, (
        f"params shape {params.shape} != expected {expected_params}"
    )
    assert len(params.columns) == param_length, (
        f"params column count {len(params.columns)} != param_length {param_length}"
    )
    # Real files may legitimately hold NaN in some parameter columns (e.g.
    # "Final Z (m)" on early pixels); require at least one finite value.
    first_row = np.asarray(params.iloc[0].tolist(), dtype=float)
    assert np.isfinite(first_row).any(), "first params row has no finite values"
    assert np.isfinite(grid[0]).all(), "first grid pixel contains NaN"

    nan_ratio = float(np.isnan(grid).mean())
    assert 0.0 <= nan_ratio <= 1.0


def check_dat_without_bias_spec(tmpdir: Path) -> None:
    """M3: a .dat header without 'Bias Spectroscopy' must parse cleanly."""
    dat_path = tmpdir / "z_spectroscopy.dat"
    write_dat(
        dat_path,
        [
            ("#Z Spectroscopy>Setpoint", "0.500"),
            ("#Z Spectroscopy>Time Constant", "0.010"),
            ("Bias (V)", "-0.500"),
            ("Channel", "Current (A)"),
        ],
        ["Bias (V)", "Current (A)"],
        [["-0.500", "1.23"]],
    )
    loader = NanonisFileLoader(str(dat_path))
    header = loader.header  # must not raise KeyError
    assert "Bias Spectroscopy" not in header
    assert "#Z Spectroscopy" in header
    assert header["#Z Spectroscopy"]["Setpoint"] == "0.500"
    print("  .dat header keys (subset):", sorted(header)[:6])


def check_multiline_synthetic_dat(tmpdir: Path) -> None:
    """M3 regression: .dat with Bias Spectroscopy>MultiLine Settings converts."""
    dat_path = tmpdir / "bias_spectroscopy.dat"
    write_dat(
        dat_path,
        [
            (
                "Bias Spectroscopy>MultiLine Settings:V (V),dI/dV (A)",
                "0.01,1e-9;0.005,2e-9",
            ),
            ("Bias Spectroscopy>Sweep Start (V)", "0.01"),
            ("Bias Spectroscopy>Sweep End (V)", "-0.01"),
            ("Bias (V)", "-0.500"),
            ("Channel", "Current (A)"),
        ],
        ["Bias (V)", "Current (A)"],
        [["-0.500", "1.23"]],
    )
    loader = NanonisFileLoader(str(dat_path))
    header = loader.header
    bs = header["Bias Spectroscopy"]
    ml = bs["MultiLine Settings"]
    assert isinstance(ml, pd.DataFrame), (
        f"MultiLine Settings is {type(ml).__name__}, expected DataFrame"
    )
    assert list(ml.columns) == ["V (V)", "dI/dV (A)"], list(ml.columns)
    assert ml.shape == (2, 2), ml.shape
    assert np.allclose(ml.iloc[0].tolist(), [0.01, 1e-9])
    assert np.allclose(ml.iloc[1].tolist(), [0.005, 2e-9])
    assert not any(
        "MultiLine Settings:" in key for key in bs
    ), "raw colon-bearing MultiLine key was not popped"
    print(f"  .dat MultiLine Settings converted to DataFrame {ml.shape}")


def check_multiline_real_3ds() -> None:
    """M3 regression: real 3ds MultiLine Settings must be a DataFrame."""
    loader = NanonisFileLoader(str(MULTILINE_3DS))
    header = loader.header
    bs = header["Bias Spectroscopy"]
    ml = bs["MultiLine Settings"]
    assert isinstance(ml, pd.DataFrame), (
        f"MultiLine Settings is {type(ml).__name__}, expected DataFrame"
    )
    assert ml.shape == (6, 5), f"unexpected shape {ml.shape}"
    for name in ("Segment Start (V)", "Segment End (V)", "Settling (s)"):
        assert any(name in col for col in ml.columns), (
            f"column {name!r} missing from {list(ml.columns)}"
        )
    # row0 must match the raw string '2E-3,0E+0,5E-3,50E-3,2'
    assert np.allclose(ml.iloc[0].tolist(), [2e-3, 0.0, 5e-3, 50e-3, 2.0])
    assert not any(
        "MultiLine Settings :" in key for key in bs
    ), "raw colon-bearing MultiLine key was not popped"
    print(
        f"  real 3ds MultiLine Settings converted to DataFrame {ml.shape}; "
        f"columns: {list(ml.columns)}"
    )


def check_channels_order_independent() -> None:
    """t22: .channels is quote-free and independent of header parse order."""
    # Fresh loader, channels accessed BEFORE header parsing.
    loader_first = NanonisFileLoader(str(CHANNELS_3DS))
    ch_first = loader_first.channels
    # Fresh loader, header parsed before channels.
    loader_second = NanonisFileLoader(str(CHANNELS_3DS))
    _ = loader_second.header
    ch_after = loader_second.channels
    assert ch_first == ch_after, (
        f"channels differ by access order: {ch_first} != {ch_after}"
    )
    for name in ch_first:
        assert name == name.strip('"').strip(), f"dirty channel name {name!r}"
    assert len(ch_first) > 0, "no channels returned"
    print(
        f"  channels order-independent and quote-free "
        f"({len(ch_first)} channels)"
    )


def check_missing_block_fields(tmpdir: Path) -> None:
    """M16: missing block header fields must raise a ValueError naming them."""
    path = tmpdir / "missing_fields.3ds"
    write_3ds(
        path,
        {
            "Grid dim": "1 x 2",
            "Points": "3",
            "Channels": "A;B",
            # "# Parameters (4 byte)" and "Experiment size (bytes)" omitted
        },
        list(range(16)),
    )
    loader = NanonisFileLoader(str(path))
    try:
        _ = loader.data  # triggers _reform_3ds_data
    except ValueError as exc:
        msg = str(exc)
        assert "block_size" in msg, f"message lacks block_size hint: {msg}"
        for field in ("# Parameters (4 byte)", "Experiment size (bytes)", "Points"):
            assert field in msg, f"message does not name {field!r}: {msg}"
        print("  ValueError raised as expected; message:", msg.split(".")[0])
    else:
        raise AssertionError("expected ValueError for missing block fields")


def check_trailing_extra_bytes(tmpdir: Path) -> None:
    """M16: raw data longer than total_pts is truncated with a warning."""
    path = tmpdir / "trailing_extra.3ds"
    write_3ds(
        path,
        {
            "Grid dim": "1 x 2",
            "Points": "3",
            "Channels": "A;B",
            "# Parameters (4 byte)": "1",
            "Experiment size (bytes)": "24",
            "Fixed parameters": "Sweep",
            "Experiment parameters": "",
        },
        [float(i) for i in range(16)],  # 14 expected + 2 trailing extras
    )

    records: list[logging.LogRecord] = []

    class CaptureHandler(logging.Handler):
        """Collect log records for assertion."""

        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    module_logger = logging.getLogger("stm_data_processing.io.nanonis_loader")
    handler = CaptureHandler()
    module_logger.addHandler(handler)
    try:
        loader = NanonisFileLoader(str(path))
        params = loader.parameters
        grid = loader.data
    finally:
        module_logger.removeHandler(handler)

    assert params.shape == (2, 1), f"params shape {params.shape} != (2, 1)"
    assert grid.shape == (2, 2, 3), f"grid shape {grid.shape} != (2, 2, 3)"
    assert params.iloc[0, 0] == 0.0
    assert np.allclose(grid[0], [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert np.allclose(grid[1], [[8.0, 9.0, 10.0], [11.0, 12.0, 13.0]])
    assert any(
        "exceeds expected" in record.getMessage() for record in records
    ), "no warning logged about trailing bytes"
    print(f"  warning logged ({len(records)} record(s)); trailing bytes ignored")


def check_empty_param_fields(tmpdir: Path) -> None:
    """M19: empty param strings are filtered; mismatches raise, never miswrite."""
    # 1) Empty "Fixed parameters" is filtered out; column count matches.
    path_ok = tmpdir / "empty_params_ok.3ds"
    write_3ds(
        path_ok,
        {
            "Grid dim": "1 x 1",
            "Points": "1",
            "Channels": "A",
            "# Parameters (4 byte)": "1",
            "Experiment size (bytes)": "4",
            "Fixed parameters": "",
            "Experiment parameters": "X (m)",
        },
        [7.0, 42.0],
    )
    loader_ok = NanonisFileLoader(str(path_ok))
    params_ok = loader_ok.parameters
    grid_ok = loader_ok.data
    assert list(params_ok.columns) == ["X (m)"], list(params_ok.columns)
    assert params_ok.shape == (1, 1)
    assert grid_ok.shape == (1, 1, 1)
    assert params_ok.iloc[0, 0] == 7.0
    assert grid_ok[0, 0, 0] == 42.0
    print("  empty 'Fixed parameters' filtered; columns:", list(params_ok.columns))

    # 2) Both param fields empty with param_length=1 -> explicit ValueError.
    path_bad = tmpdir / "empty_params_bad.3ds"
    write_3ds(
        path_bad,
        {
            "Grid dim": "1 x 1",
            "Points": "1",
            "Channels": "A",
            "# Parameters (4 byte)": "1",
            "Experiment size (bytes)": "4",
            "Fixed parameters": "",
            "Experiment parameters": "",
        },
        [7.0, 42.0],
    )
    loader_bad = NanonisFileLoader(str(path_bad))
    try:
        _ = loader_bad.data
    except ValueError as exc:
        msg = str(exc)
        assert "parameter layout mismatch" in msg, f"unexpected message: {msg}"
    else:
        raise AssertionError("expected ValueError for param column mismatch")
    print("  empty param fields with mismatch raise ValueError as expected")


def main() -> int:
    """Run every regression check; return a process exit code."""
    line_path = Path(sys.argv[1]) if len(sys.argv) > 1 else LINE_3DS
    grid_path = Path(sys.argv[2]) if len(sys.argv) > 2 else GRID_3DS
    for path in (line_path, grid_path, MULTILINE_3DS, CHANNELS_3DS):
        if not path.is_file():
            print(f"missing real data file: {path}")
            return 2

    tmpdir = Path(tempfile.mkdtemp(prefix="check_nanonis_3ds_"))
    checks = [
        ("real 1D line file", lambda _tmpdir: check_real_file(line_path)),
        ("real 50x50 grid file", lambda _tmpdir: check_real_file(grid_path)),
        ("dat header without Bias Spectroscopy (M3)", check_dat_without_bias_spec),
        ("dat MultiLine Settings converted (M3)", check_multiline_synthetic_dat),
        (
            "real 3ds MultiLine Settings converted (M3)",
            lambda _tmpdir: check_multiline_real_3ds(),
        ),
        (
            "3ds channels order-independent (t22)",
            lambda _tmpdir: check_channels_order_independent(),
        ),
        ("3ds missing block header fields (M16)", check_missing_block_fields),
        ("3ds trailing extra bytes (M16)", check_trailing_extra_bytes),
        ("3ds empty parameter fields (M19)", check_empty_param_fields),
    ]
    failed = 0
    try:
        for name, fn in checks:
            try:
                fn(tmpdir)
                print(f"[PASS] {name}")
            except Exception as exc:  # report every failing check
                failed += 1
                print(f"[FAIL] {name}: {exc}")
                traceback.print_exc()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    total = len(checks)
    print(f"{total - failed}/{total} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
