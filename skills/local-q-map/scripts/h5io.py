"""The HDF5 convention of the local-q-map skill: chunks, compression, metadata.

The repository fixes one HDF5 rule set in
``src/stm_data_processing/io/h5_convention.py``.  A skill is self-contained and
must not import ``stm_data_processing``, so this module carries the same rule set
for this skill, once: it is the only place here that calls h5py's
``create_dataset``, and every ``*.h5`` product of the skill goes through
:func:`write_product`.

* :func:`create_dataset` fixes the chunk shape (:func:`chunk_shape`), the
  compression algorithm/level (:data:`COMPRESSION` / :data:`COMPRESSION_OPTS`)
  and stamps the per-dataset ``units`` attribute where the quantity has one.
* :func:`write_file_metadata` writes the three attributes every file carries:
  ``schema_version``, ``generator`` and ``creation_date``.

Chunking rule (explicit, never h5py's implicit auto-chunking): a dataset is
sliced into chunks of at most :data:`CHUNK_TARGET_BYTES` uncompressed bytes,
filling the trailing (fastest-varying) axes first and leaving the leading axes at
one element once the budget is spent.  ``chunks == shape`` is used for datasets
smaller than the budget - still explicit chunked storage, so compression always
applies.  Readers of these maps slice rows of the canvas, so trailing-complete
chunks make those reads contiguous while the 1 MiB budget caps the memory a single
chunk needs during read/modify/write.

Compression rule: ``gzip`` at level 4, the repository-wide knee for float arrays
(within a few percent of level 6 in size, noticeably faster to write).
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Mapping, Sequence
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
#: Default value of the ``generator`` attribute.
DEFAULT_GENERATOR = "stm_local_q_map.py"

SCHEMA_VERSION_ATTR = "schema_version"
GENERATOR_ATTR = "generator"
CREATION_DATE_ATTR = "creation_date"
UNITS_ATTR = "units"

#: The datasets of one per-q product, in schema order.
PRODUCT_DATASETS = ("field", "amplitude", "theta", "mask")


def chunk_shape(
    shape: Sequence[int],
    itemsize: int,
    target_bytes: int = CHUNK_TARGET_BYTES,
) -> tuple[int, ...] | None:
    """Explicit chunk shape for a dataset of ``shape`` and ``itemsize``.

    Returns ``None`` for scalar or empty datasets (chunking is meaningless there).
    Otherwise the trailing axes are filled first up to ``target_bytes``
    uncompressed bytes and the remaining axes keep a chunk length of one.
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
    """Create ``name`` under ``parent`` following the convention.

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
    """
    array = np.asarray(data)
    chunks = chunk_shape(array.shape, array.dtype.itemsize)
    kwargs: dict[str, Any] = {
        "track_times": False,
        "compression": COMPRESSION,
        "compression_opts": int(COMPRESSION_OPTS),
    }
    if chunks is not None:
        kwargs["chunks"] = chunks

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
    extra: Mapping[str, Any] | None = None,
) -> None:
    """Write the attributes every product file of this skill must carry.

    Parameters
    ----------
    handle : h5py.File
        Open file (mode ``"w"``).
    generator : str, optional
        Tool that produced the file; defaults to :data:`DEFAULT_GENERATOR`.
    creation_date : str or None, optional
        Timestamp to stamp; the current local time is written when omitted.
    extra : mapping or None, optional
        Additional file attributes; ``None`` values are skipped.
    """
    handle.attrs[SCHEMA_VERSION_ATTR] = int(SCHEMA_VERSION)
    handle.attrs[GENERATOR_ATTR] = str(generator)
    handle.attrs[CREATION_DATE_ATTR] = (
        str(creation_date) if creation_date is not None else utc_now_iso()
    )
    for key, value in (extra or {}).items():
        if value is not None:
            handle.attrs[key] = value


def write_product(
    path: str | Path,
    datasets: Iterable[tuple[str, np.ndarray, str | None]],
    *,
    generator: str = DEFAULT_GENERATOR,
    creation_date: str | None = None,
) -> Path:
    """Write one product file: the metadata plus every ``(name, data, units)``.

    ``datasets`` is an ordered iterable of ``(dataset name, array, unit)`` triples;
    a ``None`` unit means dimensionless and writes no ``units`` attribute.
    Returns the path of the written file.
    """
    target = Path(path)
    with h5py.File(target, "w") as handle:
        write_file_metadata(handle, generator=generator, creation_date=creation_date)
        for name, data, units in datasets:
            create_dataset(handle, name, data, units=units)
    return target
