"""HDF5 writer of this skill: the repository h5 convention, mirrored here.

The numbers and the rules are the same as the package's
(``src/stm_data_processing/io/h5_convention.py``) and are documented in
``docs/hdf5_convention.md``:

* one compression algorithm and level: ``gzip`` at level 4
  (:data:`COMPRESSION` / :data:`COMPRESSION_OPTS`);
* explicit chunking (never h5py's implicit auto-chunking): a chunk holds at most
  :data:`CHUNK_TARGET_BYTES` uncompressed bytes, the trailing (fastest-varying)
  axes are filled first and the leading axes keep a chunk length of one when the
  budget is spent (:func:`chunk_shape`); a dataset smaller than the budget gets
  ``chunks == shape``, so compression always applies;
* every dataset is created with ``track_times=False`` and, when it has units, a
  ``units`` attribute; dimensionless quantities carry no ``units`` attribute;
* every file carries the root attributes ``schema_version`` (int,
  :data:`SCHEMA_VERSION`), ``generator`` (the producing script's name) and
  ``creation_date`` (ISO-8601 with a UTC offset).

Scalars cannot be chunked or filtered in HDF5, so they are stored as one-element
arrays (see ``stm_lf_correct.py``'s transform bundle) instead of 0-d datasets.

This module is deliberately dependency-free: it imports only the standard
library, ``numpy`` and ``h5py``, so the h5 products of the skill do not depend on
the package's h5 module.  The skill's own, documented package dependency (the
``bragg_peak`` / ``plot_funcs`` imports behind the ``--stm-lib`` path, and the
guarded lazy ``preview_plot.gwyddion`` colour map in ``correction_lib.py`` /
``lf_lib.py``) is untouched by this module.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np

#: Version of the on-disk layout described by this module.
SCHEMA_VERSION = 1
#: Single compression algorithm for the whole repository.
COMPRESSION = "gzip"
#: Single compression level for the whole repository (gzip: 0-9).
COMPRESSION_OPTS = 4
#: Uncompressed byte budget per chunk.
CHUNK_TARGET_BYTES = 1 << 20

SCHEMA_VERSION_ATTR = "schema_version"
GENERATOR_ATTR = "generator"
CREATION_DATE_ATTR = "creation_date"
UNITS_ATTR = "units"


def chunk_shape(
    shape: Sequence[int],
    itemsize: int,
    target_bytes: int = CHUNK_TARGET_BYTES,
) -> tuple[int, ...] | None:
    """Explicit chunk shape for a dataset of ``shape`` and ``itemsize``.

    Returns ``None`` for scalar or empty datasets (chunking is meaningless
    there).  Otherwise the trailing axes are filled first up to
    ``target_bytes`` uncompressed bytes and the remaining axes keep a chunk
    length of one.
    """
    if not shape or 0 in shape or itemsize <= 0:
        return None

    chunks = [1] * len(shape)
    budget = max(int(target_bytes) // int(itemsize), 1)
    for axis in range(len(shape) - 1, -1, -1):
        length = int(shape[axis])
        taken = min(length, budget)
        chunks[axis] = max(taken, 1)
        budget = max(budget // chunks[axis], 1)
    return tuple(chunks)


def utc_now_iso() -> str:
    """Current local time as an ISO-8601 string with UTC offset."""
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def create_dataset(
    parent: h5py.Group | h5py.File,
    name: str,
    data: np.ndarray,
    *,
    units: str | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> h5py.Dataset:
    """Create ``name`` under ``parent`` following the repository convention.

    Parameters
    ----------
    parent : h5py.Group or h5py.File
        Group that receives the dataset.
    name : str
        Dataset name.
    data : np.ndarray
        Data to store (at least one-dimensional: HDF5 cannot chunk or filter a
        scalar); ``dtype`` and shape are preserved as given.
    units : str or None, optional
        Physical unit of the data, stored as the dataset ``units`` attribute.
        Omit for dimensionless data.
    attrs : mapping or None, optional
        Extra dataset attributes.
    """
    array = np.asarray(data)
    chunks = chunk_shape(array.shape, array.dtype.itemsize)

    kwargs: dict[str, Any] = {
        "track_times": False,
        "compression": COMPRESSION,
        "compression_opts": int(COMPRESSION_OPTS),
    }
    if chunks is not None:
        kwargs["chunks"] = tuple(int(axis) for axis in chunks)

    dataset = parent.create_dataset(name, data=array, **kwargs)
    if units is not None:
        dataset.attrs[UNITS_ATTR] = str(units)
    for key, value in (attrs or {}).items():
        dataset.attrs[key] = value
    return dataset


def write_file_metadata(
    handle: h5py.File,
    *,
    generator: str,
    units: Mapping[str, str] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> None:
    """Write the attributes every h5 file of this repository must carry.

    Parameters
    ----------
    handle : h5py.File
        Open file (mode ``"w"``).
    generator : str
        Tool that produced the file (the producing script's name).
    units : mapping or None, optional
        ``quantity -> unit`` pairs stored as ``units_<quantity>`` attributes.
    extra : mapping or None, optional
        Additional file attributes; ``None`` values are skipped.
    """
    handle.attrs[SCHEMA_VERSION_ATTR] = int(SCHEMA_VERSION)
    handle.attrs[GENERATOR_ATTR] = str(generator)
    handle.attrs[CREATION_DATE_ATTR] = utc_now_iso()
    for quantity, unit in (units or {}).items():
        handle.attrs[f"units_{quantity}"] = str(unit)
    for key, value in (extra or {}).items():
        if value is not None:
            handle.attrs[key] = value


def write_file(
    path: str | Path,
    datasets: Mapping[str, tuple[np.ndarray, str | None]],
    *,
    generator: str,
    units: Mapping[str, str] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Create ``path`` (mode ``"w"``) and write every dataset of ``datasets``.

    ``datasets`` maps a dataset name to ``(array, units)``, where ``units`` is
    ``None`` for a dimensionless quantity (then no ``units`` attribute is
    written).  The parent directory is created when missing.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(target, "w") as handle:
        for name, (data, dataset_units) in datasets.items():
            create_dataset(handle, name, data, units=dataset_units)
        write_file_metadata(handle, generator=generator, units=units, extra=extra)
    return target
