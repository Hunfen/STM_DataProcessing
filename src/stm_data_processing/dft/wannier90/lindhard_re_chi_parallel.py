"""Multi-process driver for the static real Lindhard response.

:mod:`lindhard_re_chi` evaluates the response one q1 row at a time, and a row
depends on nothing but the mesh, the model and the blocking parameters.  This
module therefore splits the q1 rows into contiguous slices, evaluates every
slice in its own spawned process and stitches the slabs back together into
exactly the array the single-process run would have produced.

Guarantees
----------
* **Bitwise assembly.**  ``calculate(q_index_range=(start, stop))`` returns the
  raw FFT-order rows of the full-mesh result bit for bit, so a slice evaluated
  in a worker is copied through unchanged.  ``assemble_slices`` only permutes
  rows/columns (the chi0(q) = chi0(-q) mirror, which is an exact copy) before
  the single ``fftshift`` of the whole map.
* **Checkpoints.**  Every worker atomically lands an HDF5 shard
  ``rows_<start>_<stop>.h5`` - written through
  :mod:`stm_data_processing.io.h5_convention`, checkpoint metadata as file
  attributes - plus a ``.done`` marker, so ``resume=True`` re-dispatches
  only the missing slices.
* **Failure containment.**  A worker that exits non-zero (including a BLAS
  segmentation fault, which leaves no traceback) aborts the remaining dispatch,
  keeps the completed checkpoints and returns a non-zero exit code without
  writing the final HDF5 file.

The module logs through ``logging.getLogger(__name__)``; :func:`configure_logging`
installs the console/file handlers used by the command line entry point.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import sys
import time
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from stm_data_processing.dft.wannier90.lindhard_re_chi import (
    _HK_ROW_BLOCK,
    _MAX_BAND_BLOCK,
    _MAX_BLOCK_ENTRIES,
    _MODULE_TYPE,
    RealLindhardCalculator,
    array_digest,
    format_bytes,
    peak_rss_bytes,
)
from stm_data_processing.dft.wannier90.mlwf_hamiltonian import MLWFHamiltonian
from stm_data_processing.io.h5_convention import (
    CREATION_DATE_ATTR,
    read_creation_date,
)
from stm_data_processing.io.susceptibility_io import save_susceptibility_to_h5
from stm_data_processing.io.w90hr_loader import Wannier90HRLoader
from stm_data_processing.parallel.shard_driver import (
    RunStats,
    ShardSpec,
    ShardStore,
    available_memory_bytes,
    log_aggregate_progress,
    log_plan,
    memory_guard,
    plan_slices,
    run_shards,
)
from stm_data_processing.parallel.shard_driver import (
    scan_shards as scan_checkpoints,
)
from stm_data_processing.utils.miscellaneous import extend_qpi, frac_to_real_2d

logger = logging.getLogger(__name__)

#: Logger of the calculation module whose ``progress`` records are relayed from
#: a worker to its parent.
_LINDHARD_LOGGER = "stm_data_processing.dft.wannier90.lindhard_re_chi"

_ARRAY_KEYS = ("data", "intraband", "interband")
#: ``generator`` recorded in every shard file (see h5_convention).
_SHARD_GENERATOR = "lindhard_re_chi_parallel"
#: Suffix used for metadata entries that have to be stored as JSON text
#: (nested mappings such as the per-array digests); HDF5 attributes only
#: hold scalars, strings and regular numeric arrays.
_SHARD_JSON_SUFFIX = "_json"

#: Console/file format: the parent logs ``pid=...``, a worker adds ``worker=<id>``.
LOG_FORMAT = (
    "%(asctime)s %(levelname)s pid=%(process)d%(worker_tag)s %(name)s: %(message)s"
)

_HEAVY_STAGE_BYTES_PER_ENTRY = 115  # see _MAX_BLOCK_ENTRIES in the engine
#: Fixed per-worker overhead (interpreter, NumPy, h5py and its read buffers).
#: Calibrated so that :func:`estimate_worker_rss_bytes` stays an upper bound of
#: the measured peak RSS (see the estimate-vs-measured table in
#: ``var/lindhard_repair/r4_estimate_check.py``) without making the
#: memory guard refuse a legitimate multi-worker run.
_STATIC_OVERHEAD_BYTES = 1_100_000_000


# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
class _WorkerTagFilter(logging.Filter):
    """Add the ``worker_tag`` field (``""`` in the parent, ``" worker=<id>"``)."""

    def __init__(self, worker_id: int | None = None) -> None:
        super().__init__()
        self.worker_tag = "" if worker_id is None else f" worker={worker_id}"

    def filter(self, record: logging.LogRecord) -> bool:
        record.worker_tag = self.worker_tag
        return True


def configure_logging(
    level: str | int = "INFO",
    *,
    log_file: str | Path | None = None,
    console: bool = True,
    worker_id: int | None = None,
) -> None:
    """Install the root handlers used by the driver and by its workers.

    The format always carries the process id; a worker additionally tags every
    record with ``worker=<id>``.
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    if isinstance(level, str):
        resolved = getattr(logging, level.upper(), None)
        if not isinstance(resolved, int) or isinstance(resolved, bool):
            raise ValueError(f"unknown log level: {level!r}")
        root.setLevel(resolved)
    else:
        root.setLevel(int(level))

    formatter = logging.Formatter(LOG_FORMAT, datefmt="%H:%M:%S")
    worker_filter = _WorkerTagFilter(worker_id)
    handlers: list[logging.Handler] = []
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, mode="a", encoding="utf-8"))
    if console:
        handlers.append(logging.StreamHandler(sys.stderr))
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.addFilter(worker_filter)
        root.addHandler(handler)

    # The relay in a worker has to see INFO records of the engine even when the
    # console level is higher, so pin the engine logger to INFO.
    logging.getLogger(_LINDHARD_LOGGER).setLevel(logging.INFO)


def _ensure_file_log(log_file: str | Path) -> None:
    """Make ``log_file`` really receive the records of this process.

    ``run_parallel(log_file=...)`` used to accept the argument and never use it,
    so a direct API caller got no file log at all.  The handler is only
    installed when no handler of the root logger already writes to that file
    (the command line entry point installs one through
    :func:`configure_logging`), which keeps that path unchanged and avoids
    duplicated lines; the root level is lifted to INFO only in that case, so the
    fresh file is not silently limited to WARNING and above.
    """
    target = Path(log_file).resolve()
    root = logging.getLogger()
    for handler in root.handlers:
        if isinstance(handler, logging.FileHandler) and (
            Path(handler.baseFilename).resolve() == target
        ):
            return
    target.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt="%H:%M:%S"))
    handler.addFilter(_WorkerTagFilter())
    root.addHandler(handler)
    if root.level == logging.NOTSET or root.level > logging.INFO:
        root.setLevel(logging.INFO)


class _ProgressRelay(logging.Handler):
    """Forward the machine-readable progress extra of the engine to a queue.

    The engine attaches ``lindhard_progress = (rows_done, rows_total,
    pixels_done)`` to every progress record, which lets the parent aggregate
    live progress without parsing log text.
    """

    def __init__(self, progress_queue: Any, worker_id: int) -> None:
        super().__init__(level=logging.INFO)
        self._queue = progress_queue
        self._worker_id = worker_id

    def emit(self, record: logging.LogRecord) -> None:
        payload = getattr(record, "lindhard_progress", None)
        if payload is None:
            return
        with contextlib.suppress(Exception):  # progress is best effort
            self._queue.put_nowait((self._worker_id, payload))


# ----------------------------------------------------------------------
# Planning
# ----------------------------------------------------------------------
def plan_row_slices(
    nk: int, n_workers: int, *, mirror: bool = False
) -> list[tuple[int, int]]:
    """Split the q1 rows into pixel-balanced contiguous ``(start, stop)`` ranges.

    Parameters
    ----------
    nk : int
        Mesh size per reciprocal direction (``nk >= 1``).
    n_workers : int
        Requested number of processes (``>= 1``).  More workers than rows yield
        one slice per row, so the plan never contains an empty slice.
    mirror : bool, default False
        Plan only the rows needed to rebuild the map from
        ``chi0(q) = chi0(-q)``, i.e. ``iq1 in [0, nk // 2]``
        (``nk // 2 + 1`` rows; ``iq1 = 0`` is its own mirror image).  For odd
        ``nk`` this is ``[0, (nk - 1) // 2]``.

    Returns
    -------
    list of tuple of int
        Disjoint, contiguous, ascending half-open row ranges whose row counts
        differ by at most one.
    """
    if nk < 1:
        raise ValueError(f"nk must be positive, got {nk}")

    n_rows = nk // 2 + 1 if mirror else nk
    return plan_slices(n_rows, n_workers)


def _validate_row_slices(
    slices: Sequence[tuple[int, int]], nk: int
) -> tuple[np.ndarray, np.ndarray]:
    """Check the slices and return the per-row coverage and the row mask."""
    coverage = np.zeros(nk, dtype=int)
    for row_range in slices:
        start, stop = int(row_range[0]), int(row_range[1])
        if not 0 <= start < stop <= nk:
            raise ValueError(
                f"row slice ({start}, {stop}) violates 0 <= start < stop <= nk ({nk})"
            )
        coverage[start:stop] += 1
    if np.any(coverage > 1):
        duplicates = np.flatnonzero(coverage > 1).tolist()
        raise ValueError(f"row slices overlap on q1 rows {duplicates}")
    return coverage, coverage > 0


def read_model_shape(model_dir: str | Path, seedname: str) -> tuple[int, int] | None:
    """Read ``(num_wann, nrpts)`` of a Wannier90 model without loading it.

    The HDF5 variant keeps both numbers in the file attributes and the plain
    ``hr.dat`` carries them in its three-line header, so the parent process can
    size the memory estimate (and identify the model in the checkpoint
    signature) without reading the hundreds of megabytes of hopping matrices.
    Returns ``None`` when neither file can be read, in which case the caller
    falls back to a placeholder and says so.

    The header logic mirrors ``Wannier90HRLoader._load_hr``, so both agree on
    the same file.
    """
    folder = Path(model_dir)
    h5_path = folder / f"{seedname}_hr.h5"
    if h5_path.exists():
        try:
            with h5py.File(h5_path, "r") as handle:
                return int(handle.attrs["num_wann"]), int(handle.attrs["nrpts"])
        except OSError, KeyError, ValueError:
            return None
    dat_path = folder / f"{seedname}_hr.dat"
    if not dat_path.exists():
        return None
    try:
        with dat_path.open("r", encoding="utf-8", errors="ignore") as handle:
            first_line = handle.readline()
            if first_line.strip().lower().startswith("written on"):
                num_wann = int(handle.readline().split()[0])
                nrpts = int(handle.readline().split()[0])
            else:
                num_wann = int(first_line.split()[0])
                nrpts = int(handle.readline().split()[0])
    except OSError, ValueError, IndexError:
        return None
    return num_wann, nrpts


def model_identity(
    model_dir: str | Path, seedname: str
) -> tuple[int | None, list | None]:
    """Cheap identity of a model for the checkpoint compatibility signature.

    Returns ``(num_wann, bvecs)``; either can be ``None`` when it cannot be read
    cheaply (then the corresponding entry is skipped by the comparison instead
    of silently accepting a shard).  ``bvecs`` uses the very loader the workers
    use, so the parent and the shard metadata carry identical numbers.
    """
    shape = read_model_shape(model_dir, seedname)
    num_wann = None if shape is None else int(shape[0])
    bvecs = None
    try:
        loaded = Wannier90HRLoader._load_bvecs(Path(model_dir), seedname)
    except Exception:  # a missing/odd lattice file just disables this check
        loaded = None
    if loaded is not None:
        bvecs = np.asarray(loaded, dtype=float).tolist()
    return num_wann, bvecs


def estimate_worker_rss_bytes(
    nk: int, num_wann: int, n_orb: int | None, nrpts: int | None = None
) -> int:
    """Resident-set estimate of one worker process, in bytes (upper bound).

    Sums the arrays the engine keeps alive: the band energies and the selected
    eigenvectors of the full k mesh, the ``_HK_ROW_BLOCK``-row H(k) build
    transient and the (capped) q-sum working tensors, plus the tight-binding
    model arrays (``nrpts`` hopping matrices of ``num_wann**2`` complex128, which
    the worker loads eagerly) and a fixed overhead covering the interpreter,
    NumPy, h5py and the load buffers.

    The result is calibrated to be an *upper bound* of the measured per-worker
    peak RSS (see ``var/lindhard_repair/r4_estimate_check.py`` for the
    estimate-vs-measured table); ``nrpts=None`` only drops the model-array term,
    which the caller should avoid when the model is readable.
    """
    rows = nk * nk
    evals = rows * num_wann * 8
    evecs = 0 if n_orb is None else rows * n_orb * num_wann * 16
    hk_block = min(rows, _HK_ROW_BLOCK) * num_wann * num_wann * 16
    band_block = min(_MAX_BAND_BLOCK, num_wann)
    entries = min(rows * band_block * num_wann, _MAX_BLOCK_ENTRIES)
    working = int(entries * _HEAVY_STAGE_BYTES_PER_ENTRY)
    model = 0 if nrpts is None else int(nrpts) * num_wann * num_wann * 16
    return int(evals + evecs + hk_block + working + model + _STATIC_OVERHEAD_BYTES)


# ----------------------------------------------------------------------
# Checkpoints
# ----------------------------------------------------------------------


def _array_sha256(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values, dtype=np.float64)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def _digest_arrays(arrays: Mapping[str, np.ndarray]) -> dict[str, dict[str, float]]:
    return {key: array_digest(arrays[key]) for key in _ARRAY_KEYS}


def assemble_from_checkpoints(
    checkpoint_dir: str | Path,
    *,
    nk: int,
    mirror: bool = False,
    q_range: tuple[float, float] | None = None,
    eta: float = 5e-3,
    chemical_potential: float = 0.0,
    temperature: float = 4.2,
    include_matrix_elements: bool = True,
    orbital_select: Sequence[int] | np.ndarray | None = None,
    degeneracy_tolerance: float = 1e-12,
) -> tuple[dict[str, Any], list[tuple[int, int]]]:
    """Assemble the result from every complete slice in ``checkpoint_dir``.

    ``bvecs`` and the band count are taken from the checkpoint metadata, so the
    returned dict is complete even when no worker had to be dispatched (a fully
    resumed run).  Returns ``(result, slices)``.
    """
    ckpt_dir = Path(checkpoint_dir)
    slices = scan_checkpoints(ckpt_dir)
    if not slices:
        raise ValueError(f"no complete checkpoint slices in {ckpt_dir}")
    store = ShardStore(
        ckpt_dir,
        array_keys=_ARRAY_KEYS,
        identity={},
        label=_LABEL,
        logger=logger,
    )
    blocks = []
    metas = []
    for row_range in slices:
        stored = store.load(row_range)
        if stored is None:
            raise ValueError(
                f"checkpoint rows=[{row_range[0]}, {row_range[1]}) is incomplete"
            )
        blocks.append(stored["arrays"])
        metas.append(stored["meta"])
    stored_bvecs = metas[0].get("bvecs")
    result = assemble_slices(
        slices,
        blocks,
        nk=nk,
        mirror=mirror,
        bvecs=None if stored_bvecs is None else np.asarray(stored_bvecs, dtype=float),
        q_range=q_range,
        eta=eta,
        chemical_potential=chemical_potential,
        temperature=temperature,
        include_matrix_elements=include_matrix_elements,
        orbital_select=orbital_select,
        num_wann=int(metas[0]["num_wann"]),
        degeneracy_tolerance=degeneracy_tolerance,
    )
    return result, slices


# ----------------------------------------------------------------------
# Worker
# ----------------------------------------------------------------------
def _worker(
    worker_id: int,
    model_dir: str,
    seedname: str,
    nk: int,
    eta: float,
    row_range: tuple[int, int],
    calc_kwargs: dict[str, Any],
    store: ShardStore,
    result_queue: Any,
    progress_queue: Any,
    log_config: dict[str, Any] | None,
) -> None:
    """Evaluate one q1 row slice and land its checkpoint.

    Module level on purpose and restricted to string/number/dict arguments, so
    the function is importable and picklable under the ``spawn`` start method:
    the model and the calculator are rebuilt inside the worker instead of being
    inherited from the parent.
    """
    config = dict(log_config or {})
    configure_logging(
        config.get("level", "INFO"),
        log_file=config.get("file"),
        console=bool(config.get("console", False)),
        worker_id=worker_id,
    )
    relay = _ProgressRelay(progress_queue, worker_id)
    logging.getLogger(_LINDHARD_LOGGER).addHandler(relay)

    start, stop = int(row_range[0]), int(row_range[1])
    logger.info(
        "[LindhardParallel] worker=%d start rows=[%d, %d) pixels=%d pid=%d",
        worker_id,
        start,
        stop,
        (stop - start) * nk,
        os.getpid(),
    )
    started = time.perf_counter()
    try:
        hamiltonian = MLWFHamiltonian.from_seedname(model_dir, seedname)
        engine_kwargs = {
            "band_block": calc_kwargs.get("band_block"),
            "block_entries": calc_kwargs.get("block_entries"),
        }
        calculator = RealLindhardCalculator(
            hamiltonian, nk=nk, eta=eta, **engine_kwargs
        )
        result = calculator.calculate(
            chemical_potential=float(calc_kwargs.get("chemical_potential", 0.0)),
            temperature=float(calc_kwargs.get("temperature", 4.2)),
            orbital_select=calc_kwargs.get("orbital_select"),
            include_matrix_elements=bool(
                calc_kwargs.get("include_matrix_elements", True)
            ),
            degeneracy_tolerance=float(calc_kwargs.get("degeneracy_tolerance", 1e-12)),
            q_index_range=(start, stop),
            progress_interval_s=float(calc_kwargs.get("progress_interval_s", 0.0)),
        )
        arrays = {key: np.asarray(result[key], dtype=np.float64) for key in _ARRAY_KEYS}
        elapsed = time.perf_counter() - started
        rss_peak = peak_rss_bytes()
        bvecs = hamiltonian.bvecs
        meta = {
            "worker_id": worker_id,
            "row_range": [start, stop],
            "nk": nk,
            "eta": eta,
            "num_wann": int(hamiltonian.num_wann),
            "orbital_select": result["metadata"]["orbital_select"].tolist(),
            "chemical_potential": float(calc_kwargs.get("chemical_potential", 0.0)),
            "temperature": float(calc_kwargs.get("temperature", 4.2)),
            "include_matrix_elements": bool(
                calc_kwargs.get("include_matrix_elements", True)
            ),
            "degeneracy_tolerance": float(
                calc_kwargs.get("degeneracy_tolerance", 1e-12)
            ),
            "mirror": bool(calc_kwargs.get("mirror", False)),
            "bvecs": None if bvecs is None else np.asarray(bvecs, dtype=float).tolist(),
            "sha256": {key: _array_sha256(arrays[key]) for key in _ARRAY_KEYS},
            "digest": _digest_arrays(arrays),
            "elapsed_s": elapsed,
            "rss_peak_bytes": rss_peak,
            "created_unix": time.time(),
        }
        shard = store.write((start, stop), arrays, meta)
        result_queue.put(
            {
                "status": "ok",
                "worker_id": worker_id,
                "row_range": [start, stop],
                "rows": stop - start,
                "pixels": (stop - start) * nk,
                "num_wann": int(hamiltonian.num_wann),
                "orbital_select": result["metadata"]["orbital_select"].tolist(),
                "bvecs": None if bvecs is None else np.asarray(bvecs, dtype=float),
                "sha256": meta["sha256"],
                "digest": meta["digest"],
                "elapsed_s": elapsed,
                "rss_peak_bytes": rss_peak,
                "shard": str(shard),
            }
        )
        logger.info(
            "[LindhardParallel] worker=%d done rows=[%d, %d) s=%.2f rss_peak=%s",
            worker_id,
            start,
            stop,
            elapsed,
            format_bytes(rss_peak),
        )
    except BaseException:
        logger.error("[LindhardParallel] worker=%d failed", worker_id, exc_info=True)
        _safe_put(
            result_queue,
            {
                "status": "error",
                "worker_id": worker_id,
                "row_range": [start, stop],
                "error": traceback.format_exc(),
            },
        )
        raise
    finally:
        logging.getLogger(_LINDHARD_LOGGER).removeHandler(relay)


def _safe_put(target: Any, item: Any) -> None:
    with contextlib.suppress(Exception):  # pragma: no cover - queue teardown
        target.put_nowait(item)


# ----------------------------------------------------------------------
# Assembly
# ----------------------------------------------------------------------
def _mirrored_columns(nk: int) -> np.ndarray:
    return (nk - np.arange(nk)) % nk


def assemble_slices(
    slices: Sequence[tuple[int, int]],
    blocks: Sequence[Mapping[str, np.ndarray]],
    nk: int,
    *,
    mirror: bool = False,
    bvecs: np.ndarray | None = None,
    q_range: tuple[float, float] | None = None,
    eta: float = 5e-3,
    chemical_potential: float = 0.0,
    temperature: float = 4.2,
    include_matrix_elements: bool = True,
    orbital_select: Sequence[int] | np.ndarray | None = None,
    num_wann: int | None = None,
    degeneracy_tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Stitch raw FFT-order row slices into the ``calculate()`` result dict.

    Parameters
    ----------
    slices : sequence of (int, int)
        Half-open q1 row ranges, parallel to ``blocks``.
    blocks : sequence of mapping
        One mapping per slice with the ``data`` / ``intraband`` / ``interband``
        arrays of shape ``(stop - start, nk)`` in raw FFT order.
    nk : int
        Mesh size per reciprocal direction.
    mirror : bool, default False
        When True the slices need not cover every row: a missing row ``iq1`` is
        filled by the *exact* copy ``row[iq1][iq2] = row[(nk - iq1) % nk]
        [(nk - iq2) % nk]`` of the static evenness ``chi0(q) = chi0(-q)``.
    bvecs, q_range, eta, chemical_potential, temperature,
    include_matrix_elements, orbital_select, num_wann, degeneracy_tolerance
        Same meaning as in
        :meth:`~stm_data_processing.dft.wannier90.lindhard_re_chi.RealLindhardCalculator.calculate`;
        ``orbital_select=None`` is reported as ``arange(num_wann)`` when
        ``num_wann`` is known.

    Returns
    -------
    dict
        The same keys and shapes as ``calculate()``: the three data arrays are
        fftshifted exactly once (q=(0,0) at ``(nk//2, nk//2)``) and the
        fractional grids are ``fftshift(fftfreq(nk))``.

    Raises
    ------
    ValueError
        When the slices are invalid, overlap, or do not cover every row (with
        ``mirror=False``) resp. can not be completed by the mirror (with
        ``mirror=True``), or when a block has the wrong shape.
    """
    if nk < 1:
        raise ValueError(f"nk must be positive, got {nk}")
    if len(slices) != len(blocks):
        raise ValueError(f"got {len(slices)} slices but {len(blocks)} blocks")

    _, covered = _validate_row_slices(slices, nk)
    if mirror:
        missing = np.flatnonzero(~covered)
        if missing.size:
            sources = (nk - missing) % nk
            if np.any(~covered[sources]):
                raise ValueError(
                    "row slices can not be completed by the chi0(q) = chi0(-q) "
                    f"mirror: rows {missing.tolist()} and their mirror rows are "
                    "both missing"
                )
    elif np.any(~covered):
        raise ValueError(
            "row slices must cover every q1 row exactly once, missing "
            f"{np.flatnonzero(~covered).tolist()}"
        )

    arrays = {key: np.empty((nk, nk), dtype=float) for key in _ARRAY_KEYS}
    for row_range, block in zip(slices, blocks, strict=True):
        start, stop = int(row_range[0]), int(row_range[1])
        for key in _ARRAY_KEYS:
            values = np.asarray(block[key])
            if values.shape != (stop - start, nk):
                raise ValueError(
                    f"block {key} for rows ({start}, {stop}) has shape "
                    f"{values.shape}, expected {(stop - start, nk)}"
                )
            arrays[key][start:stop] = values

    if mirror:
        columns = _mirrored_columns(nk)
        for iq1 in np.flatnonzero(~covered):
            source = int((nk - iq1) % nk)
            for key in _ARRAY_KEYS:
                arrays[key][iq1] = arrays[key][source][columns]

    for key in _ARRAY_KEYS:
        arrays[key] = np.fft.fftshift(arrays[key], axes=(0, 1))
    q_values = np.fft.fftshift(np.fft.fftfreq(nk))
    q1_grid, q2_grid = np.meshgrid(q_values, q_values, indexing="ij")

    total = arrays["data"]
    intraband = arrays["intraband"]
    interband = arrays["interband"]

    if mirror:
        # The mirrored rows are exact copies, so the map is symmetric up to the
        # machine-precision evenness of the rows the engine evaluated itself
        # (chi0(q) and chi0(-q) are two independent floating-point sums).  The
        # permutation below is the fftshifted form of iq -> (nk - iq) % nk.
        shifted_raw = (np.arange(nk) - nk // 2) % nk
        mirror_index = ((nk - shifted_raw) % nk + nk // 2) % nk
        symmetry_error = float(
            np.max(np.abs(total - total[np.ix_(mirror_index, mirror_index)]))
        )
        tolerance = 1e-12 * max(1.0, float(np.max(np.abs(total))))
        if symmetry_error > tolerance:
            logger.error(
                "[LindhardParallel] assembled map violates chi0(q) = chi0(-q): "
                "max|chi0(q) - chi0(-q)|=%.3e (tolerance %.3e); the mirror "
                "permutation or the row coverage is wrong",
                symmetry_error,
                tolerance,
            )
        else:
            logger.info(
                "[LindhardParallel] mirror assembly: max|chi0(q) - chi0(-q)| = "
                "%.3e (tolerance %.3e; the mirrored rows are exact copies of "
                "their source rows)",
                symmetry_error,
                tolerance,
            )

    if q_range is not None:
        base_q1_grid, base_q2_grid = q1_grid, q2_grid
        total, q1_grid, q2_grid = extend_qpi(
            total, base_q1_grid, base_q2_grid, q_range[0], q_range[1]
        )
        intraband, _, _ = extend_qpi(
            intraband, base_q1_grid, base_q2_grid, q_range[0], q_range[1]
        )
        interband, _, _ = extend_qpi(
            interband, base_q1_grid, base_q2_grid, q_range[0], q_range[1]
        )

    qx_grid, qy_grid = frac_to_real_2d(q1_grid, q2_grid, bvecs)
    selected = orbital_select
    if selected is None and num_wann is not None:
        selected = np.arange(int(num_wann))
    if selected is not None:
        selected = np.asarray(selected, dtype=int)

    metadata: dict[str, Any] = {
        "module_type": _MODULE_TYPE,
        "eta": eta,
        "nk": nk,
        "nq": nk,
        "bvecs": bvecs,
        "chemical_potential": chemical_potential,
        "temperature": temperature,
        "include_matrix_elements": include_matrix_elements,
        "orbital_select": selected,
        "degeneracy_tolerance": degeneracy_tolerance,
        "note": (
            "Re chi0 in the document convention (doc 7.1), fftshifted so "
            "that q=(0,0) sits at index (nk//2, nk//2)"
        ),
    }

    return {
        "data": total,
        "intraband": intraband,
        "interband": interband,
        "q1_grid": q1_grid,
        "q2_grid": q2_grid,
        "qx_grid": qx_grid,
        "qy_grid": qy_grid,
        "metadata": metadata,
    }


def write_result_h5(
    result: Mapping[str, Any],
    output_path: str | Path,
    *,
    bvecs: np.ndarray | None = None,
    eta: float = 5e-3,
    nq: int | None = None,
    chemical_potential: float = 0.0,
    temperature: float = 4.2,
    orbital_select: Sequence[int] | np.ndarray | None = None,
    projection: str | None = None,
) -> Path:
    """Write the assembled result atomically through ``save_susceptibility_to_h5``.

    Every attribute is written by the single HDF5 write path (no post-hoc
    ``h5py`` patching): ``module_type``, ``eta``, ``nq``, ``chemical_potential``,
    ``temperature``, ``orbital_select`` and ``projection``, plus the ``bvecs``
    dataset (and the ``susceptibility`` dataset itself).  The file is assembled
    under a temporary name and moved into place, so a failed write never leaves
    a half-written product behind.
    """
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(f"{target.name}.tmp-{os.getpid()}")

    extra: dict[str, Any] = {
        "chemical_potential": chemical_potential,
        "temperature": temperature,
    }
    # The product is assembled under a temporary name, so the creation date of
    # the file being replaced has to be carried over explicitly: an atomic
    # rewrite that changes no data must stay byte-identical (the resume/repair
    # checks rely on it), and a fresh timestamp would break that.
    previous_creation_date = read_creation_date(target)
    if previous_creation_date is not None:
        extra[CREATION_DATE_ATTR] = previous_creation_date
    if orbital_select is not None:
        extra["orbital_select"] = np.asarray(orbital_select, dtype=int)
    if projection is not None:
        extra["projection"] = projection

    try:
        save_susceptibility_to_h5(
            susceptibility=np.asarray(result["data"], dtype=float),
            output_path=str(tmp_path),
            module_type=_MODULE_TYPE,
            bvecs=bvecs,
            eta=eta,
            nq=int(nq if nq is not None else result["data"].shape[0]),
            **extra,
        )
        tmp_path.replace(target)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return target


# ----------------------------------------------------------------------
# Parent
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# Shard-and-merge driver wiring
# ----------------------------------------------------------------------
def _memory_guard(
    per_worker_bytes: int, n_workers: int, max_mem_gb: float | None = None
) -> None:
    """Refuse a worker count whose estimated RSS exceeds the budget.

    Thin adapter over :func:`stm_data_processing.parallel.shard_driver.
    memory_guard` that keeps this module's log label.
    """
    memory_guard(per_worker_bytes, n_workers, max_mem_gb, label=_LABEL)


def _load_shard(
    ckpt_dir: str | Path, row_range: tuple[int, int]
) -> dict[str, Any] | None:
    """Load one shard through the driver's store (compatibility adapter)."""
    store = ShardStore(
        ckpt_dir,
        array_keys=_ARRAY_KEYS,
        identity={},
        label=_LABEL,
        logger=logger,
    )
    return store.load(row_range)


#: Everything the driver's own log records are labelled with.
_LABEL = "LindhardParallel"


def run_parallel(
    model_dir: str,
    seedname: str,
    nk: int,
    *,
    eta: float = 5e-3,
    temperature: float = 4.2,
    chemical_potential: float = 0.0,
    orbital_select: Sequence[int] | None = None,
    include_matrix_elements: bool = True,
    degeneracy_tolerance: float = 1e-12,
    band_block: int | None = None,
    block_entries: int | None = None,
    n_workers: int = 2,
    mirror: bool = False,
    output_path: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
    resume: bool = False,
    q_range: tuple[float, float] | None = None,
    projection: str | None = None,
    progress_interval_s: float = 30.0,
    start_method: str = "spawn",
    log_file: str | Path | None = None,
    worker_log_dir: str | Path | None = None,
    max_mem_gb: float | None = None,
    dry_run: bool = False,
) -> int:
    """Evaluate Re chi0(q, 0) in ``n_workers`` spawned processes.

    Parameters
    ----------
    log_file : str or Path, optional
        Also write the log records of *this* process to that file (worker
        detail logs always go to ``worker_log_dir``).  See
        :func:`_ensure_file_log`: when the caller already registered a file
        handler for the same path (as the command line entry point does through
        :func:`configure_logging`), nothing is added and no line is duplicated.

    Returns
    -------
    int
        ``0`` on success, ``130`` when interrupted by SIGINT, non-zero
        otherwise.  The HDF5 product is written only after every slice has been
        assembled and verified, so a failed run leaves no half-written file.

    Notes
    -----
    The shard-and-merge mechanics (slice planning, dispatch, resume filter,
    memory budget, merge ordering and the shard files) live in
    :mod:`stm_data_processing.parallel.shard_driver`; this function only
    declares the physics through a :class:`~stm_data_processing.parallel.
    shard_driver.ShardSpec`.
    """
    if log_file is not None:
        _ensure_file_log(log_file)
    if nk < 1:
        raise ValueError(f"nk must be positive, got {nk}")
    if eta <= 0:
        raise ValueError(f"eta must be positive, got {eta}")

    # Real model size, read from the hr.h5 attributes or the three-line hr.dat
    # header: the parent never loads the hopping matrices itself, but it must
    # not guess the orbital count either (the estimate and the checkpoint
    # signature both depend on it).
    model_num_wann, model_bvecs = model_identity(model_dir, seedname)
    model_shape = read_model_shape(model_dir, seedname)
    if model_shape is None:
        logger.warning(
            "[%s] cannot read the size of %s/%s (no *_hr.h5 and no "
            "*_hr.dat header); the memory estimate falls back to a 75-orbital "
            "placeholder and the checkpoint signature skips the model identity",
            _LABEL,
            model_dir,
            seedname,
        )
        estimate_num_wann, estimate_nrpts = 75, 1681
    else:
        estimate_num_wann, estimate_nrpts = model_shape
        logger.info(
            "[%s] model %s/%s: num_wann=%d nrpts=%d",
            _LABEL,
            model_dir,
            seedname,
            estimate_num_wann,
            estimate_nrpts,
        )

    signature = {
        "nk": nk,
        "eta": float(eta),
        "chemical_potential": float(chemical_potential),
        "temperature": float(temperature),
        "include_matrix_elements": bool(include_matrix_elements),
        "degeneracy_tolerance": float(degeneracy_tolerance),
        "orbital_select": None
        if orbital_select is None
        else [int(o) for o in orbital_select],
        # Model identity: a shard computed for another model must never be
        # reused, and the worker metadata already carries both numbers.
        "num_wann": model_num_wann,
        "bvecs": model_bvecs,
    }

    estimate_n_orb = (
        estimate_num_wann if orbital_select is None else len(orbital_select)
    )
    per_worker = estimate_worker_rss_bytes(
        nk, estimate_num_wann, estimate_n_orb, estimate_nrpts
    )

    worker_progress_interval = (
        0.0 if progress_interval_s <= 0 else min(float(progress_interval_s), 5.0)
    )
    payload = {
        "band_block": band_block,
        "block_entries": block_entries,
        "chemical_potential": float(chemical_potential),
        "temperature": float(temperature),
        "orbital_select": None
        if orbital_select is None
        else [int(o) for o in orbital_select],
        "include_matrix_elements": bool(include_matrix_elements),
        "degeneracy_tolerance": float(degeneracy_tolerance),
        "mirror": bool(mirror),
        "progress_interval_s": worker_progress_interval,
        "model_dir": str(model_dir),
        "seedname": str(seedname),
        "nk": int(nk),
        "eta": float(eta),
        "worker_log_dir": None if worker_log_dir is None else str(worker_log_dir),
    }

    store = ShardStore(
        Path() if checkpoint_dir is None else Path(checkpoint_dir),
        array_keys=_ARRAY_KEYS,
        identity=signature,
        generator=_SHARD_GENERATOR,
        label=_LABEL,
        logger=logger,
    )

    def _plan() -> list[tuple[int, int]]:
        return plan_row_slices(nk, n_workers, mirror=mirror)

    def _progress_units(key: tuple[int, int]) -> tuple[int, int]:
        rows = key[1] - key[0]
        return rows, rows * nk

    def _worker_process(
        context: Any,
        index: int,
        key: tuple[int, int],
        worker_payload: Mapping[str, Any],
        result_queue: Any,
        progress_queue: Any,
    ) -> Any:
        log_dir = Path(worker_payload["worker_log_dir"] or store.directory)
        return context.Process(
            target=_worker,
            name=f"lindhard-worker-{index}",
            args=(
                index,
                worker_payload["model_dir"],
                worker_payload["seedname"],
                worker_payload["nk"],
                worker_payload["eta"],
                (int(key[0]), int(key[1])),
                worker_payload,
                store,
                result_queue,
                progress_queue,
                {
                    "level": "INFO",
                    "file": str(log_dir / f"worker_{index}.log"),
                    "console": True,
                },
            ),
        )

    def _log_plan(keys: Sequence[tuple[int, int]], per_worker_bytes: int) -> None:
        log_plan(
            keys,
            nk,
            mirror,
            per_worker_bytes,
            available_memory_bytes(),
            max_mem_gb,
            label=_LABEL,
        )

    if n_workers > nk and not mirror:
        logger.warning(
            "[%s] %d workers requested for %d q1 rows: only %d slices are dispatched",
            _LABEL,
            n_workers,
            nk,
            min(n_workers, nk),
        )

    def _log_progress(stats: RunStats) -> None:
        log_aggregate_progress(
            stats.rows_done,
            stats.rows_total,
            stats.pixels_done,
            stats.pixels_total,
            stats.started,
            stats.live_workers,
            stats.rss_peak_bytes,
            label=_LABEL,
        )

    def _describe_plan(keys: Sequence[tuple[int, int]]) -> str:
        rows_total = sum(stop - start for start, stop in keys)
        return (
            f"DRY-RUN workers={len(keys)} nk={nk} mirror={mirror} rows={rows_total} "
            f"pixels={rows_total * nk} num_wann={estimate_num_wann} "
            f"nrpts={estimate_nrpts} segments="
            + ",".join(f"[{start},{stop})" for start, stop in keys)
        )

    def _finalize(
        loaded: list[dict[str, Any]],
        reports: Mapping[int, Mapping[str, Any]],
        stats: RunStats,
    ) -> int:
        bvecs_holder = loaded[0]["meta"].get("bvecs")
        bvecs = None if bvecs_holder is None else np.asarray(bvecs_holder, dtype=float)
        num_wann = int(loaded[0]["meta"]["num_wann"])
        blocks = []
        metas: list[dict[str, Any]] = []
        blocks = [shard["arrays"] for shard in loaded]
        metas = [shard["meta"] for shard in loaded]

        for index, report in reports.items():
            row_range = (report["row_range"][0], report["row_range"][1])
            stored = store.load(row_range)
            if stored is None:
                continue
            for key in _ARRAY_KEYS:
                if _array_sha256(stored["arrays"][key]) != report["sha256"][key]:
                    logger.error(
                        "[LindhardParallel] worker=%d digest mismatch for "
                        "'%s' rows=[%d, %d): the checkpoint no longer "
                        "matches what the worker computed",
                        index,
                        key,
                        row_range[0],
                        row_range[1],
                    )
                    return 6

        # bvecs and the band count come from the checkpoints, so a
        # fully resumed run (no worker dispatched) still writes the
        # complete attribute set.
        stored_bvecs = metas[0].get("bvecs")
        bvecs = None if stored_bvecs is None else np.asarray(stored_bvecs, dtype=float)
        num_wann = int(metas[0]["num_wann"])

        # The primitive-BZ map is assembled first: it is what the worker
        # digests are checked against and what the HDF5 product stores.
        primitive = assemble_slices(
            stats.slices,
            blocks,
            nk=nk,
            mirror=mirror,
            bvecs=None if bvecs is None else np.asarray(bvecs, dtype=float),
            q_range=None,
            eta=float(eta),
            chemical_potential=float(chemical_potential),
            temperature=float(temperature),
            include_matrix_elements=bool(include_matrix_elements),
            orbital_select=orbital_select,
            num_wann=num_wann,
            degeneracy_tolerance=float(degeneracy_tolerance),
        )

        if not _verify_assembled_digests(reports, primitive, nk):
            return 8

        if q_range is None:
            result = primitive
        else:
            # Same crop as calculate(): the returned dict is extended to
            # [q_range), the HDF5 product keeps the primitive mesh.
            result = assemble_slices(
                stats.slices,
                blocks,
                nk=nk,
                mirror=mirror,
                bvecs=None if bvecs is None else np.asarray(bvecs, dtype=float),
                q_range=q_range,
                eta=float(eta),
                chemical_potential=float(chemical_potential),
                temperature=float(temperature),
                include_matrix_elements=bool(include_matrix_elements),
                orbital_select=orbital_select,
                num_wann=num_wann,
                degeneracy_tolerance=float(degeneracy_tolerance),
            )

        if output_path is not None:
            label = projection
            if label is None:
                label = "full" if orbital_select is None else "custom"
            try:
                write_result_h5(
                    primitive,
                    output_path,
                    bvecs=None if bvecs is None else np.asarray(bvecs, dtype=float),
                    eta=float(eta),
                    nq=nk,
                    chemical_potential=float(chemical_potential),
                    temperature=float(temperature),
                    orbital_select=result["metadata"]["orbital_select"],
                    projection=label,
                )
            except Exception:
                logger.error(
                    "[LindhardParallel] writing %s failed; the assembled "
                    "result is discarded and no partial product is left "
                    "behind",
                    output_path,
                    exc_info=True,
                )
                return 7
            logger.info(
                "[LindhardParallel] wrote %s (%.2f MB)",
                output_path,
                Path(output_path).stat().st_size / 1024**2,
            )

        digest = _digest_arrays({key: result[key] for key in _ARRAY_KEYS})
        logger.info(
            "[LindhardParallel] summary rows=%d pixels=%d workers=%d "
            "mirror=%s total_s=%.3f stats.rss_peak_bytes=%s "
            "max|data-(intra+inter)|=%.3e",
            stats.rows_total,
            stats.pixels_total,
            stats.workers,
            mirror,
            time.perf_counter() - stats.started,
            format_bytes(stats.rss_peak_bytes),
            float(
                np.max(
                    np.abs(result["data"] - (result["intraband"] + result["interband"]))
                )
            ),
        )
        for key in _ARRAY_KEYS:
            logger.info(
                "[LindhardParallel] digest %s sum=%.12e max=%.12e min=%.12e",
                key,
                digest[key]["sum"],
                digest[key]["max"],
                digest[key]["min"],
            )

    spec = ShardSpec(
        label=_LABEL,
        store=store,
        plan=_plan,
        progress_units=_progress_units,
        worker_process=_worker_process,
        worker_payload=lambda: payload,
        log_plan=_log_plan,
        log_progress=_log_progress,
        describe_plan=_describe_plan,
        finalize=_finalize,
        worker_log_dir=worker_log_dir,
        logger=logger,
        per_worker_bytes=per_worker,
        max_mem_gb=max_mem_gb,
    )
    return run_shards(
        spec,
        n_workers=n_workers,
        checkpoint_dir=checkpoint_dir,
        resume=resume,
        start_method=start_method,
        progress_interval_s=progress_interval_s,
        dry_run=dry_run,
    )


def _verify_assembled_digests(
    reports: Mapping[int, Mapping[str, Any]],
    assembled: Mapping[str, Any],
    nk: int,
) -> bool:
    """Check every worker digest against the rows it landed in.

    A worker reports the sum/max/min digest of the raw FFT-order rows it
    computed; the assembled map carries exactly those rows (``fftshift`` only
    permutes indices), so the digests must agree bit for bit.  A mismatch means
    the slices were placed at the wrong offset or a checkpoint was replaced
    between the worker and the assembly.
    """
    columns = (np.arange(nk) + nk // 2) % nk
    ok = True
    for index, report in sorted(reports.items()):
        start, stop = int(report["row_range"][0]), int(report["row_range"][1])
        rows = (np.arange(start, stop) + nk // 2) % nk
        for key in _ARRAY_KEYS:
            block = np.asarray(assembled[key])[np.ix_(rows, columns)]
            digest = array_digest(block)
            expected = report["digest"][key]
            if digest != expected:
                ok = False
                logger.error(
                    "[LindhardParallel] worker=%d digest mismatch for '%s' "
                    "rows=[%d, %d): worker sum=%.12e max=%.12e min=%.12e vs "
                    "assembled sum=%.12e max=%.12e min=%.12e",
                    index,
                    key,
                    start,
                    stop,
                    expected["sum"],
                    expected["max"],
                    expected["min"],
                    digest["sum"],
                    digest["max"],
                    digest["min"],
                )
    if ok and reports:
        logger.info(
            "[LindhardParallel] worker digests verified against the assembled "
            "rows of %d dispatched slice(s)",
            len(reports),
        )
    return ok
