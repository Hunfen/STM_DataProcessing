"""The package-wide HDF5 convention: chunking, compression and metadata.

Every ``*.h5`` product written by this package is created through this module,
so one rule set governs all of them (see ``docs/hdf5_convention.md``):

* :func:`create_dataset` is the ONLY place that calls ``create_dataset`` on an
  ``h5py`` group.  It fixes the chunk shape (:func:`chunk_shape`), the
  compression algorithm/level (:data:`COMPRESSION` / :data:`COMPRESSION_OPTS`)
  and stamps the per-dataset ``units`` attribute when the data has units.
* :func:`write_file_metadata` writes the three attributes every file must
  carry: ``schema_version``, ``generator`` and ``creation_date``, plus optional
  ``units_<quantity>`` attributes.

Chunking rule (explicit, not h5py's implicit auto-chunking): a dataset is
sliced into chunks of at most :data:`CHUNK_TARGET_BYTES` uncompressed bytes,
filling the trailing (fastest-varying) axes first and leaving the leading axes
at one element when the budget is already spent.  ``chunks == shape`` is used
for datasets smaller than the budget - still explicit chunked storage, so
compression always applies.  Rationale: the readers here slice along the
trailing axes (rows of a q/k grid, the last band axis), so trailing-complete
chunks make those reads contiguous, while the 1 MiB budget caps the memory a
single chunk needs during read/modify/write.

Compression rule: one algorithm and level for the whole package - ``gzip`` at
level 4.  Before this convention the writers disagreed (``ek2d_io`` gzip/4,
``qpi_io`` and ``susceptibility_io`` gzip/6).  Level 4 is the measured knee for
these float arrays: it is within a few percent of level 6 in size while writing
noticeably faster, which matters for the largest products (susceptibility and
``ek2d`` grids) and for the parallel driver that rewrites its result file on
every resume.
"""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np

logger = logging.getLogger(__name__)

#: Version of the on-disk layout described by this module.
SCHEMA_VERSION = 1
#: Single compression algorithm for the whole package.
COMPRESSION = "gzip"
#: Single compression level for the whole package (gzip: 0-9).
COMPRESSION_OPTS = 4
#: Uncompressed byte budget per chunk.
CHUNK_TARGET_BYTES = 1 << 20
#: Default value of the ``generator`` attribute.
DEFAULT_GENERATOR = "stm_data_processing"

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


def read_creation_date(path: str | Path) -> str | None:
    """Return the ``creation_date`` already recorded in ``path``, else ``None``.

    Writers use this so that rewriting a product with different content (the
    parallel driver's resume/repair path, which assembles under a temporary
    name and replaces the target) keeps the original creation date, and with it
    the byte-identity of a rewrite that changed no data.
    """
    target = Path(path)
    if not target.exists():
        return None
    try:
        with h5py.File(target, "r") as handle:
            value = handle.attrs.get(CREATION_DATE_ATTR)
    except OSError:
        return None
    return None if value is None else str(value)


def create_dataset(
    parent: h5py.Group | h5py.File,
    name: str,
    data: np.ndarray,
    *,
    units: str | None = None,
    attrs: Mapping[str, Any] | None = None,
    compression: str | None = COMPRESSION,
    compression_opts: int | None = COMPRESSION_OPTS,
    chunks: Sequence[int] | None = None,
) -> h5py.Dataset:
    """Create ``name`` under ``parent`` following the package convention.

    Parameters
    ----------
    parent : h5py.Group or h5py.File
        Group that receives the dataset.
    name : str
        Dataset name.
    data : np.ndarray
        Data to store.
    units : str or None, optional
        Physical unit of the data, stored as the dataset ``units`` attribute.
        Omit for dimensionless data.
    attrs : mapping or None, optional
        Extra dataset attributes.
    compression : str or None, optional
        Compression algorithm; defaults to :data:`COMPRESSION`.  ``None``
        stores the dataset uncompressed.
    compression_opts : int or None, optional
        Compression level; only passed for ``gzip`` and defaults to
        :data:`COMPRESSION_OPTS`.
    chunks : sequence of int or None, optional
        Explicit chunk shape; defaults to :func:`chunk_shape`.
    """
    array = np.asarray(data)
    if chunks is None:
        chunks = chunk_shape(array.shape, array.dtype.itemsize)

    kwargs: dict[str, Any] = {"track_times": False}
    if compression is not None:
        kwargs["compression"] = compression
        if compression == COMPRESSION and compression_opts is not None:
            kwargs["compression_opts"] = int(compression_opts)
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
    generator: str = DEFAULT_GENERATOR,
    creation_date: str | None = None,
    units: Mapping[str, str] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> None:
    """Write the attributes every file of this package must carry.

    Parameters
    ----------
    handle : h5py.File
        Open file (mode ``"w"``).
    generator : str, optional
        Tool that produced the file; defaults to :data:`DEFAULT_GENERATOR`.
    creation_date : str or None, optional
        Value to keep for an existing product (see
        :func:`read_creation_date`); the current time is stamped when omitted.
    units : mapping or None, optional
        ``quantity -> unit`` pairs stored as ``units_<quantity>`` attributes.
    extra : mapping or None, optional
        Additional file attributes; ``None`` values are skipped.
    """
    handle.attrs[SCHEMA_VERSION_ATTR] = int(SCHEMA_VERSION)
    handle.attrs[GENERATOR_ATTR] = str(generator)
    handle.attrs[CREATION_DATE_ATTR] = (
        str(creation_date) if creation_date is not None else utc_now_iso()
    )
    for quantity, unit in (units or {}).items():
        handle.attrs[f"units_{quantity}"] = str(unit)
    for key, value in (extra or {}).items():
        if value is not None:
            handle.attrs[key] = value
