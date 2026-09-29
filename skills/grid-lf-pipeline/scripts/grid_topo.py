#!/usr/bin/env python
"""Extract and preprocess the topography map of a Nanonis grid file.

A grid file (``.3ds``, or its ``.h5`` conversion) keeps the tip height of every
pixel in the ``/params`` table, column ``Z (m)`` -- it is *not* one of the
``/channels``.  This script turns that column into the plain square CSV that the
``lawler-fujita-correction`` skill takes as its topography input, applying the
three preprocessing steps in this fixed order::

    1. z -= np.median(z, axis=1, keepdims=True)   # per-row background
    2. z  = subtractMeanPlane(z.T).T              # least-squares plane
    3. z -= z.min()                               # min -> 0

``axis=1`` is the fast (X) axis, so each slow-scan line loses its own level.
Step 2 calls the repository helper
``stm_data_processing.utils.plot_funcs.subtractMeanPlane`` (NaN safe) on the
transposed matrix: transposing does not change the plane that gets fitted (the
basis ``{1, i, j}`` spans the same surface either way), it only changes the
last-digit rounding of the plane evaluation, and the transposed call is what
reproduces the approved reference topography byte for byte once the convention
flip below is undone.

**Orientation (top-down convention).**  A ``.3ds`` records its pixels
bottom-up: in :mod:`stm_data_processing.io.nanonis_loader` the 3ds path walks a
single flat pixel index and writes block ``n`` into ``/params`` *and* into the
grid channels alike, flipping neither, so ``/data`` and ``/params`` are in one
and the same frame.  The loader's sxm path (``_reform_sxm_data``) is the one
that normalises an image: it mirrors the backward-scan rows with ``fliplr``, and
flips the row order with ``flipud`` when ``SCAN_DIR`` is ``up`` -- i.e. an sxm
comes back top-down.  This skill adopts that same top-down convention for grid
products, so the extracted z map is flipped once, right here at the source::

    z = np.flipud(z)

``/data`` and ``/params`` remain in one frame with each other (both flipped),
which is what the rest of the chain relies on: the displacement field is fitted
on this top-down CSV and every map that is later fed to the applier is flipped
into the same top-down frame.

The input is an ``.h5`` produced by ``python -m stm_data_processing.io.grid2h5``
(see ``src/stm_data_processing/io/grid2h5.py``); a ``.3ds`` source is converted
first with that same module, and the conversion command is printed before it
runs.  The scan size in nm is read from the ``.3ds``/h5 header (``Scan``'s
``Scanfield``, otherwise the header's ``Grid settings``); ``-L`` overrides it.

Usage::

    .venv/bin/python <this script> GRID.h5  -o TOPO.csv
    .venv/bin/python <this script> GRID.3ds -o TOPO.csv [-L 30]
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

# stm_data_processing imports matplotlib, which wants a writable config dir.
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "grid_lf_mplconfig")
)

import h5py
import numpy as np

from stm_data_processing.utils.plot_funcs import subtractMeanPlane

#: ``/params`` column holding the tip height of each pixel.
Z_COLUMN = "Z (m)"
#: A grid file records its extent as one ``;``-separated record,
#: ``x0;y0;width;height;angle`` (metres, radians); the width sits at index 2.
GRID_SIZE_INDEX = 2


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input", help="grid .h5 (or .3ds) file")
    parser.add_argument("-o", "--out", required=True, help="output topography CSV")
    parser.add_argument(
        "-L",
        "--size-nm",
        type=float,
        default=None,
        help="scan size in nm; default: read from the file header",
    )
    return parser.parse_args(argv)


def convert_3ds(src: Path) -> Path:
    """Convert ``src`` (.3ds) to h5 with the repository converter, print the command."""
    out = src.with_suffix(".h5")
    cmd = [
        sys.executable,
        "-m",
        "stm_data_processing.io.grid2h5",
        str(src),
        "-o",
        str(out),
    ]
    print("# 3ds -> h5: " + " ".join(shlex.quote(part) for part in cmd), flush=True)
    subprocess.run(cmd, check=True)
    return out


def scan_size_nm(h5_path: Path) -> float:
    """Scan width in nm from the header (``Scan``'s ``Scanfield``, else ``Grid settings``).

    Both fields are the same ``;``-separated ``x0;y0;width;height;angle`` record
    that Nanonis writes into the ``.3ds`` header; ``grid2h5`` mirrors the
    ``Scan`` module into ``/header/Scan`` and the top-level ``Grid settings``
    line into the ``/header`` attributes.
    """
    with h5py.File(h5_path, "r") as handle:
        raw = None
        scan = handle.get("header/Scan")
        if scan is not None and "Scanfield" in scan.attrs:
            raw = scan.attrs["Scanfield"]
        if raw is None and "Grid settings" in handle["header"].attrs:
            raw = handle["header"].attrs["Grid settings"]
    if raw is None:
        raise SystemExit(
            f"ERROR: {h5_path} carries no Scanfield / Grid settings; pass -L"
        )
    fields = str(raw).split(";")
    if len(fields) <= GRID_SIZE_INDEX:
        raise SystemExit(f"ERROR: cannot read a scan size out of {raw!r}; pass -L")
    width_m = float(fields[GRID_SIZE_INDEX])
    return width_m * 1e9


def load_z_map(h5_path: Path) -> np.ndarray:
    """``(ny, nx)`` tip-height map from ``/params``'s ``Z (m)`` column, top-down.

    The only flip of the whole chain (see the module docstring): a ``.3ds``
    records bottom-up -- the loader's 3ds path writes one flat pixel index into
    ``/params`` and the grid channels alike, flipping neither, so ``/data`` and
    ``/params`` share one frame -- while an sxm is normalised to top-down by
    ``_reform_sxm_data``.  Grid products adopt the sxm convention, so the map is
    flipped once here, at the source, and stays top-down from here on.
    """
    with h5py.File(h5_path, "r") as handle:
        columns = [
            name.decode() if isinstance(name, bytes) else str(name)
            for name in handle["param_columns"][:]
        ]
        if Z_COLUMN not in columns:
            raise SystemExit(
                f"ERROR: {h5_path} has no {Z_COLUMN!r} in /param_columns "
                f"(found {columns})"
            )
        z = np.asarray(handle["params"][:, columns.index(Z_COLUMN)], dtype=float)
        ny, nx = handle["data"].shape[:2]
    if z.size != ny * nx:
        raise SystemExit(f"ERROR: {h5_path}: {z.size} Z values for a {ny} x {nx} grid")
    return np.flipud(z.reshape(ny, nx))


def preprocess(z: np.ndarray) -> np.ndarray:
    """The three preprocessing steps of this skill, in order (see the module docstring)."""
    z = z - np.median(z, axis=1, keepdims=True)
    z = subtractMeanPlane(z.T).T
    return z - z.min()


def main(argv=None):
    args = parse_args(argv)
    src = Path(args.input)
    h5_path = convert_3ds(src) if src.suffix.lower() == ".3ds" else src

    z = load_z_map(h5_path)
    size_nm = args.size_nm if args.size_nm is not None else scan_size_nm(h5_path)
    topo = preprocess(z)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out, topo, delimiter=",", fmt="%.10e")

    print(f"# input: {h5_path}  ({topo.shape[0]} x {topo.shape[1]} px)")
    print(f"# field of view: {size_nm:g} nm" + (" (from -L)" if args.size_nm else ""))
    print(
        f"# Z (m) range {z.min():.4e} .. {z.max():.4e} m -> "
        f"topo {topo.min():.4e} .. {topo.max():.4e} m"
    )
    print(f"# written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
