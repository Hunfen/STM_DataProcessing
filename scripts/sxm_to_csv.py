"""Batch-convert Nanonis .sxm topography files to CSV.

For each input .sxm it loads the Z(m) channel's forward scan via
:class:`stm_data_processing.io.nanonis_loader.NanonisFileLoader`, removes a
NaN-safe best-fit plane with
:func:`stm_data_processing.utils.plot_funcs.subtractMeanPlane`, shifts the
finite minimum to zero, and writes the result as a tab-separated, header-less,
metre-valued ``%g`` CSV (the same layout as the reference product):

    {yyyymmdd}_{sxm 名}_{bias}mV{setpoint}pA_{frame}nm.csv

Run from the repository root::

    .venv/bin/python scripts/sxm_to_csv.py <sxm 路径...> -o <输出目录>
    .venv/bin/python scripts/sxm_to_csv.py --list FILE -o <输出目录>
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

# Importing plot_funcs pulls in matplotlib, which must be told where to put
# its config/cache before it is imported (see tests/regression/check_nanonis_sxm.py).
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "dsh_mplconfig"),
)

import numpy as np

from stm_data_processing.io.nanonis_loader import NanonisFileLoader
from stm_data_processing.utils.plot_funcs import subtractMeanPlane


def _format_g(value: float) -> str:
    """Format a number with %g (6 significant digits, trailing zeros dropped)."""
    return f"{value:g}"


def format_date(rec_date: str) -> str:
    """Convert a Nanonis ``dd.mm.yyyy`` REC_DATE to ``yyyymmdd``."""
    day, month, year = rec_date.strip().split(".")
    return f"{int(year):04d}{int(month):02d}{int(day):02d}"


def format_bias(bias: float) -> str:
    """V to mV, %g."""
    return _format_g(bias * 1000.0)


def format_setpoint(setpoint_amps: float) -> str:
    """A to pA, %g."""
    return _format_g(setpoint_amps * 1e12)


def format_frame(scan_range: tuple[float, float]) -> str:
    """m (longer side) to nm, %g."""
    frame_m = max(scan_range)
    return _format_g(frame_m * 1e9)


def derive_filename(header: dict, sxm_name: str) -> str:
    """Build the output CSV name from the .sxm header.

    date: header REC_DATE (dd.mm.yyyy -> yyyymmdd)
    bias: header BIAS (V -> mV)
    setpoint: header Z-CONTROLLER Setpoint (A -> pA)
    frame: max(SCAN_RANGE) (m -> nm)
    All four numeric values are %g-formatted and glued to their unit letter
    with no space.
    """
    rec_date = header.get("REC_DATE", "")
    if not rec_date:
        raise ValueError("header lacks REC_DATE")
    date_part = format_date(rec_date)

    bias_v = float(header.get("BIAS"))
    bias_part = format_bias(bias_v)

    z_controller = header.get("Z-CONTROLLER")
    try:
        setpoint_str = z_controller["Setpoint"].iloc[0].split()[0]
        setpoint_a = float(setpoint_str)
    except (AttributeError, KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError(f"cannot read Z-CONTROLLER setpoint: {exc}") from exc
    setpoint_part = format_setpoint(setpoint_a)

    scan_range = tuple(float(v) for v in header.get("SCAN_RANGE", "").split())
    if len(scan_range) != 2:
        raise ValueError(f"SCAN_RANGE malformed: {header.get('SCAN_RANGE')!r}")
    frame_part = format_frame(scan_range)

    stem = Path(sxm_name).stem
    return f"{date_part}_{stem}_{bias_part}mV{setpoint_part}pA_{frame_part}nm.csv"


def find_channel_index(header: dict) -> int:
    """Return the DATA_INFO row index of the Z (m) channel.

    Prioritises an exact ``Z (m)`` entry in ``Scan>channels``, falling back to a
    DATA_INFO row whose Name == "Z" and Unit == "m".  Raises ValueError when the
    Z channel is not found either way (the caller skips that file).
    """
    # Preferred source: Scan>channels ("Current (A);Bias (V);Z (m);...").
    scan = header.get("Scan")
    channels_str = (
        scan.get("channels")
        if isinstance(scan, dict) and isinstance(scan.get("channels"), str)
        else header.get("Scan>channels")
    )
    if isinstance(channels_str, str) and "Z (m)" in channels_str:
        scan_channels_ok = True
    else:
        scan_channels_ok = False

    # Locate Z in DATA_INFO (its payload row is data[2 * i_z]).
    data_info = header.get("DATA_INFO")
    i_z = None
    if data_info is not None:
        for row_index, row in data_info.iterrows():
            name = str(row["Name"]).strip()
            unit = str(row["Unit"]).strip()
            if name == "Z" and unit == "m":
                i_z = int(row_index)
                break

    if scan_channels_ok or i_z is not None:
        if i_z is None:
            raise ValueError("Z (m) appears in Scan>channels but not in DATA_INFO")
        return i_z

    raise ValueError("no Z (m) channel found in Scan>channels or DATA_INFO")


def correct_forward(data: np.ndarray, i_z: int) -> np.ndarray:
    """Plane-fit subtract then shift the finite minimum to zero (NaN-safe).

    ``data`` is the loader payload of shape ``(2 * n_channels, ny, nx)``; the
    Z forward scan is the ``2 * i_z``-th row.  The loader has already
    normalised the scan direction, so no extra flip/transpose/slice is applied.
    """
    forward = data[2 * i_z]
    corrected = subtractMeanPlane(forward)
    finite = np.isfinite(corrected)
    if finite.any():
        corrected = corrected - corrected[finite].min()
    return corrected


def write_csv(path: Path, arr: np.ndarray) -> None:
    """Write the corrected array as tab-separated, header-less %g CSV."""
    np.savetxt(path, arr, delimiter="\t", fmt="%g")


def convert_one(sxm_path: Path, out_dir: Path) -> tuple[Path, str]:
    """Convert a single .sxm file, returning (output_path, filename)."""
    loader = NanonisFileLoader(str(sxm_path))
    header = loader.header
    if loader.file_type != "sxm":
        raise ValueError(f"not an .sxm file: {sxm_path.suffix}")
    filename = derive_filename(header, sxm_path.name)
    i_z = find_channel_index(header)
    corrected = correct_forward(loader.data, i_z)
    out_path = out_dir / filename
    write_csv(out_path, corrected)
    return out_path, filename


def collect_paths(positional: list[str], list_file: str | None) -> list[Path]:
    """Merge positional .sxm paths with paths read from --list FILE."""
    paths = [Path(p) for p in positional]
    if list_file:
        with Path(list_file).open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    paths.append(Path(line))
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sxm", nargs="*", help="one or more .sxm files")
    parser.add_argument("-o", "--output", default=".", help="output directory")
    parser.add_argument(
        "--list",
        metavar="FILE",
        default=None,
        help="text file with one .sxm path per line; blank lines and # lines ignored",
    )
    args = parser.parse_args(argv)

    paths = collect_paths(args.sxm, args.list)
    if not paths:
        parser.error("no .sxm paths given (positional or via --list)")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    succeeded, failed = 0, 0
    for path in paths:
        try:
            out_name, _ = convert_one(path, out_dir)
        except Exception as exc:  # one bad file must not stop the batch
            failed += 1
            print(f"FAIL  {path}: {exc}", file=sys.stderr)
        else:
            succeeded += 1
            print(f"OK    {path}  ->  {out_name}")

    print(f"done: {succeeded} succeeded, {failed} failed")
    return 1 if succeeded == 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
