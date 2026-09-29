"""Convert one Nanonis ``.3ds`` grid-spectroscopy file into one HDF5 file.

The converter reads the source through :class:`NanonisFileLoader` and writes a
single product through :mod:`stm_data_processing.io.h5_convention`:

``/data``
    The grid cube, ``float32``, shaped ``(ny, nx, nch, npts)`` (``nx`` is the
    fast axis, so ``data[iy, ix]`` is row ``iy``, column ``ix``).  ``Grid dim``
    and ``Points`` fix the shape; ``source_dtype`` and ``axis_order`` record
    what was written.
``/channels``, ``/channel_units``
    The channel names and the unit parsed from each name's trailing
    parenthesised token.
``/bias``
    The ``(npts,)`` sweep axis with its ``units`` and the ``bias_source`` it
    was derived from: the channel named by the header's ``Sweep Signal`` when
    every pixel holds the same row (``'measured_channel'``), otherwise the
    linspace between the first pixel's ``Sweep Start`` / ``Sweep End``
    parameters (``'linspace'``).
``/params``, ``/param_columns``
    The per-pixel parameter table as the loader reports it, with the column
    names taken from the header's ``Fixed parameters`` and ``Experiment
    parameters``.
``/header``
    The loader's ``header`` mirrored key by key: a scalar becomes an attribute
    and a module dict becomes a subgroup of attributes (a single-attribute
    module too).  The dump is data-driven, so a key added by a future Nanonis
    version lands in the product without a code change; a module name is
    escaped for the link namespace (``%`` -> ``%25``, ``/`` -> ``%2F``).

A truncated ``.3ds`` still converts: the loader NaN-pads the pixels the file
does not carry, so the cube keeps the header-declared shape.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from stm_data_processing.io import h5_convention
from stm_data_processing.io.nanonis_loader import NanonisFileLoader

#: Dtype of the source payload, recorded on ``/data``.
SOURCE_DTYPE = ">f4"
#: Meaning of the ``/data`` axes, recorded on ``/data``.
AXIS_ORDER = "(ny, nx, nch, npts)"
#: Header key naming the channel that holds the measured sweep.
SWEEP_SIGNAL = "Sweep Signal"
#: Parameter columns holding the per-pixel sweep bounds.
SWEEP_START = "Sweep Start"
SWEEP_END = "Sweep End"
#: Header keys listing the parameter column names, in table order.
PARAMETER_KEYS = ("Fixed parameters", "Experiment parameters")
#: Unit of the sweep bounds (Nanonis writes them in volts).
BIAS_UNITS = "V"
#: ``bias_source`` values recorded on ``/bias``.
SOURCE_MEASURED = "measured_channel"
SOURCE_LINSPACE = "linspace"


class Grid2H5Error(ValueError):
    """Raised when a ``.3ds`` source cannot be converted."""


def channel_unit(name: str) -> str:
    """Return the unit from ``name``'s trailing parenthesised token.

    ``'Current (A)' -> 'A'``, ``'DSP 7280 X (%)' -> '%'``; ``''`` when the
    name carries no trailing ``(...)``.
    """
    match = re.search(r"\(([^()]*)\)\s*$", name)
    return match.group(1) if match else ""


def _string_array(values: list[str]) -> np.ndarray:
    """Wrap ``values`` as an h5py variable-length UTF-8 string array."""
    return np.asarray(values, dtype=h5py.string_dtype("utf-8"))


def _parse_grid_dim(value: str) -> tuple[int, int]:
    """Parse the header ``Grid dim`` string into ``(nx, ny)``."""
    parts = str(value).replace(" ", "").split("x")
    if len(parts) != 2:
        error_msg = f"cannot parse 'Grid dim' value {value!r} into (nx, ny)"
        raise Grid2H5Error(error_msg)
    return int(parts[0]), int(parts[1])


def _group_name(name: str) -> str:
    """Return the group name for a header module name.

    A ``/`` would otherwise create a nested group and silently merge two
    modules, so it is escaped - together with ``%``, which is escaped first so
    the mapping stays injective (``A/B`` and ``A%2FB`` stay distinct).
    """
    return name.replace("%", "%25").replace("/", "%2F")


def _header_text(value: Any) -> str:
    """Return the text stored for one header value."""
    return value if isinstance(value, str) else str(value)


def _sweep_axis(
    header: dict[str, Any],
    data: np.ndarray,
    params: Any,
    channels: list[str],
    npts: int,
) -> tuple[np.ndarray, str, str]:
    """Return ``(bias, bias_source, units)`` for the sweep axis.

    The measured channel wins when every pixel carries the same sweep row;
    otherwise the axis is the linspace between the first pixel's sweep bounds.
    """
    signal = str(header.get(SWEEP_SIGNAL, "") or "")
    if signal and signal in channels:
        rows = np.asarray(data[:, channels.index(signal), :], dtype=np.float64)
        if np.array_equal(rows, np.broadcast_to(rows[0], rows.shape)):
            return rows[0], SOURCE_MEASURED, channel_unit(signal) or BIAS_UNITS
    missing = [name for name in (SWEEP_START, SWEEP_END) if name not in params]
    if missing:
        error_msg = (
            f"parameter table lacks the sweep bound column(s): {', '.join(missing)}"
        )
        raise Grid2H5Error(error_msg)
    bias = np.linspace(
        float(params[SWEEP_START].iloc[0]),
        float(params[SWEEP_END].iloc[0]),
        int(npts),
    )
    return bias, SOURCE_LINSPACE, BIAS_UNITS


def _write_header(handle: h5py.File, header: dict[str, Any]) -> None:
    """Mirror ``header`` into ``/header``: scalars and dict leaves as attrs."""
    group = handle.create_group("header")
    for key, value in header.items():
        if isinstance(value, dict):
            subgroup = group.create_group(_group_name(str(key)))
            for leaf, leaf_value in value.items():
                subgroup.attrs[str(leaf)] = _header_text(leaf_value)
        else:
            group.attrs[str(key)] = _header_text(value)


def grid_to_h5(source: str | Path, destination: str | Path | None = None) -> Path:
    """Convert the ``.3ds`` file ``source`` and return the product's path.

    ``destination`` defaults to ``source`` with an ``.h5`` suffix.  Raises
    :class:`Grid2H5Error` when the source is not a ``.3ds`` file or its header
    does not describe the payload.
    """
    src = Path(source)
    if not src.exists():
        raise Grid2H5Error(f"input file does not exist: {src}")
    if src.suffix.lower() != ".3ds":
        error_msg = f"input file is not a .3ds file (suffix {src.suffix!r}): {src}"
        raise Grid2H5Error(error_msg)
    dest = Path(destination) if destination is not None else src.with_suffix(".h5")

    loader = NanonisFileLoader(str(src))
    header = loader.header
    missing = [field for field in ("Grid dim", "Points") if field not in header]
    if missing:
        error_msg = f"3ds header lacks required field(s): {', '.join(missing)}"
        raise Grid2H5Error(error_msg)
    data = np.asarray(loader.data)
    params = loader.parameters
    channels = [str(channel) for channel in loader.channels]
    nx, ny = _parse_grid_dim(header["Grid dim"])
    npts = int(str(header["Points"]).strip().strip('"'))
    nch = len(channels)
    if data.shape != (nx * ny, nch, npts):
        error_msg = (
            f"data shape {data.shape} does not match the header grid "
            f"{nx}x{ny} x {nch} channels x {npts} points"
        )
        raise Grid2H5Error(error_msg)

    cube = np.ascontiguousarray(data.reshape(ny, nx, nch, npts).astype(np.float32))
    bias, bias_source, bias_units = _sweep_axis(header, data, params, channels, npts)
    param_columns = [
        item
        for key in PARAMETER_KEYS
        for item in str(header.get(key, "")).split(";")
        if item
    ]

    dest.parent.mkdir(parents=True, exist_ok=True)
    # Read the date of an existing product before the file is recreated, so a
    # re-conversion keeps it.
    creation_date = h5_convention.read_creation_date(dest)
    with h5py.File(dest, "w") as handle:
        h5_convention.write_file_metadata(
            handle,
            generator="grid2h5",
            creation_date=creation_date,
        )
        h5_convention.create_dataset(
            handle,
            "data",
            cube,
            attrs={"source_dtype": SOURCE_DTYPE, "axis_order": AXIS_ORDER},
        )
        h5_convention.create_dataset(
            handle,
            "channels",
            _string_array(channels),
            compression=None,
        )
        h5_convention.create_dataset(
            handle,
            "channel_units",
            _string_array([channel_unit(name) for name in channels]),
            compression=None,
        )
        h5_convention.create_dataset(
            handle,
            "bias",
            bias,
            units=bias_units,
            attrs={"bias_source": bias_source},
        )
        h5_convention.create_dataset(handle, "params", params.values.astype(np.float64))
        h5_convention.create_dataset(
            handle,
            "param_columns",
            _string_array(param_columns),
            compression=None,
        )
        _write_header(handle, header)
    return dest


def main(argv: list[str] | None = None) -> int:
    """CLI wrapper over :func:`grid_to_h5`.

    Usage::

        python -m stm_data_processing.io.grid2h5 SRC.3ds [-o OUT.h5]

    Without ``-o`` the output is ``SRC.h5``.  Exits 1 with a message when the
    source cannot be converted.
    """
    parser = argparse.ArgumentParser(
        prog="python -m stm_data_processing.io.grid2h5",
        description="Convert a Nanonis .3ds grid-spectroscopy file to HDF5.",
    )
    parser.add_argument("source", help="input .3ds grid-spectroscopy file")
    parser.add_argument(
        "-o",
        "--out",
        default=None,
        help="output .h5 path (default: <source>.h5)",
    )
    args = parser.parse_args(argv)

    try:
        out = grid_to_h5(args.source, args.out)
    except (Grid2H5Error, OSError, ValueError, KeyError) as exc:
        print(f"grid2h5: error: {exc}", file=sys.stderr)
        return 1
    print(f"grid2h5: wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
