"""Reusable shard-and-merge parallel driver.

The model: the work is split into disjoint *shards*; every shard is computed by
its own worker process into its own checkpoint file (an HDF5 shard written
through :mod:`stm_data_processing.io.h5_convention`), and the parent merges the
shards **in plan order** into the final product once all of them are present.
Per-worker files plus a merge step are the only viable parallel write pattern
here because ``h5py.get_config().mpi`` is False (serial-only HDF5).

A caller declares its work through :class:`ShardSpec` - see
``docs/parallel_shard_driver.md`` for a walkthrough.  The driver owns:

* the shard lifecycle - atomic shard writes, the ``.done`` completion marker,
  corruption-tolerant reads, compatibility checking and removal after a
  successful merge (:class:`ShardStore`);
* slice planning and the resume filter (only missing shards are dispatched);
* worker dispatch, progress accounting, failure handling, SIGINT handling and
  the aggregate progress log;
* the worker/memory budget check before anything is started
  (:func:`memory_guard` with :func:`available_memory_bytes`).

The caller owns the physics: how a shard maps to a row range, what a worker
computes, how the shards are merged, and how the product is published.
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import queue as queue_module
import signal
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from stm_data_processing.dft.wannier90.lindhard_re_chi import (
    format_bytes,
    format_hms,
    peak_rss_bytes,
)
from stm_data_processing.io.h5_convention import create_dataset, write_file_metadata

logger = logging.getLogger(__name__)

#: Suffix used for shard metadata entries that HDF5 cannot store directly.
_SHARD_JSON_SUFFIX = "_json"


def _shard_stem(row_range: tuple[int, int]) -> str:
    return f"rows_{int(row_range[0])}_{int(row_range[1])}"


def _shard_paths(ckpt_dir: Path, row_range: tuple[int, int]) -> tuple[Path, Path]:
    """``(shard_file, done_marker)`` of one slice.

    A shard is a single HDF5 file written through
    :mod:`stm_data_processing.io.h5_convention` - one dataset per result array
    plus the checkpoint metadata as file attributes - together with a ``.done``
    marker that is only created once the shard is complete.
    """
    stem = _shard_stem(row_range)
    return ckpt_dir / f"{stem}.h5", ckpt_dir / f"{stem}.done"


def _shard_attrs(meta: Mapping[str, Any]) -> dict[str, Any]:
    """Checkpoint metadata -> HDF5 attributes.

    Scalars, strings and regular numeric arrays are stored as they are; nested
    mappings (the per-array digests) and any ragged sequence become JSON text
    under ``<key>_json``, because HDF5 attributes cannot hold them.
    """
    attrs: dict[str, Any] = {}
    for key, value in meta.items():
        if value is None:
            continue
        if isinstance(value, Mapping):
            attrs[f"{key}{_SHARD_JSON_SUFFIX}"] = json.dumps(value, sort_keys=True)
            continue
        converted = _shard_attr_value(value)
        if isinstance(converted, np.ndarray) and converted.dtype == object:
            attrs[f"{key}{_SHARD_JSON_SUFFIX}"] = json.dumps(value, sort_keys=True)
        else:
            attrs[key] = converted
    return attrs


def _shard_meta(handle: h5py.File) -> dict[str, Any]:
    """HDF5 attributes -> checkpoint metadata (the inverse of _shard_attrs)."""
    meta: dict[str, Any] = {}
    for key in handle.attrs:
        if key.endswith(_SHARD_JSON_SUFFIX):
            meta[key[: -len(_SHARD_JSON_SUFFIX)]] = json.loads(str(handle.attrs[key]))
        else:
            meta[key] = _shard_meta_value(handle.attrs[key])
    return meta


def _shard_attr_value(value: Any) -> Any:
    """Checkpoint metadata -> something ``h5py`` can store as an attribute."""
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return np.asarray(value)
    return value


def _shard_meta_value(value: Any) -> Any:
    """HDF5 attribute -> the JSON-equivalent Python value used before.

    The metadata is compared entry by entry by :func:`_shard_is_compatible`, so
    it has to come back exactly as the JSON sidecar delivered it: scalars as
    Python scalars, arrays as nested lists.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _write_shard(
    ckpt_dir: Path,
    row_range: tuple[int, int],
    arrays: Mapping[str, np.ndarray],
    meta: Mapping[str, Any],
    *,
    array_keys: Sequence[str],
    generator: str = "shard_driver",
) -> Path:
    """Atomically land the HDF5 shard plus the ``.done`` marker."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    h5_path, done_path = _shard_paths(ckpt_dir, row_range)

    tmp_h5 = h5_path.with_name(h5_path.name + ".tmp")
    with h5py.File(tmp_h5, "w") as handle:
        for key in array_keys:
            create_dataset(handle, key, np.asarray(arrays[key], dtype=np.float64))
        write_file_metadata(
            handle,
            generator=generator,
            extra=_shard_attrs(meta),
        )
    tmp_h5.replace(h5_path)

    tmp_done = done_path.with_name(done_path.name + ".tmp")
    tmp_done.write_text("ok\n", encoding="utf-8")
    tmp_done.replace(done_path)
    return h5_path


def _load_shard(
    ckpt_dir: Path,
    row_range: tuple[int, int],
    *,
    array_keys: Sequence[str],
    label: str = "shard-driver",
    logger: logging.Logger | None = None,
) -> dict[str, Any] | None:
    """Load a complete checkpoint slice, or ``None`` when it is absent/broken.

    Any failure to read the slice counts as "broken": a checkpoint left behind
    by a killed worker can be empty, truncated, not an HDF5 file at all
    (``OSError`` from the ``h5py`` opener) or carry mismatched shapes, and
    ``resume=True`` has to discard and recompute it instead of aborting the
    whole run.  The failure is logged at WARNING level, so a discarded shard is
    never silent.
    """
    log = logger if logger is not None else logging.getLogger(__name__)
    h5_path, done_path = _shard_paths(ckpt_dir, row_range)
    if not (h5_path.exists() and done_path.exists()):
        return None
    try:
        with h5py.File(h5_path, "r") as handle:
            arrays = {
                key: np.asarray(handle[key][:], dtype=np.float64) for key in array_keys
            }
            meta = _shard_meta(handle)
    except Exception as exc:  # every read failure means the shard is broken
        log.warning(
            f"[{label}] unreadable checkpoint %s (%s): %s",
            h5_path,
            type(exc).__name__,
            exc,
        )
        return None
    expected_rows = int(row_range[1]) - int(row_range[0])
    for key in array_keys:
        if arrays[key].shape[0] != expected_rows:
            log.warning(
                f"[{label}] checkpoint %s has %d rows, expected %d",
                h5_path,
                arrays[key].shape[0],
                expected_rows,
            )
            return None
    return {"arrays": arrays, "meta": meta}


def _shard_is_compatible(meta: Mapping[str, Any], signature: Mapping[str, Any]) -> bool:
    """True when a stored slice was produced by the same physical problem.

    The signature covers the physical settings *and* the model identity
    (``num_wann``, ``bvecs``), so a shard from another model is recomputed
    instead of being silently assembled into a mixed-model result.  An entry
    that is ``None`` in the signature means "not readable in this run" and is
    skipped; a shard that lacks an entry the signature does know counts as
    incompatible.
    """
    for key in signature:
        stored = meta.get(key)
        wanted = signature[key]
        if key == "orbital_select":
            # The requested None ("every orbital") is stored expanded to
            # arange(num_wann), so it matches any full orbital list.
            if stored is None:
                return False
            stored_list = [int(value) for value in stored]
            if wanted is None:
                num_wann = meta.get("num_wann")
                if num_wann is None or stored_list != list(range(int(num_wann))):
                    return False
            elif stored_list != [int(value) for value in wanted]:
                return False
        elif wanted is None:
            continue
        elif isinstance(wanted, (list, tuple)) or isinstance(stored, (list, tuple)):
            if list(stored or []) != list(wanted or []):
                return False
        elif stored != wanted:
            return False
    return True


class CheckpointLock:
    """Mutual exclusion for one checkpoint directory (``.lock`` + live pid).

    The lock records the owning pid and is reclaimed when that pid is gone, so
    a crashed run does not block the next one.
    """

    def __init__(self, path: Path, *, label: str = "shard-driver") -> None:
        self.label = label
        self.path = path
        self.acquired = False

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for attempt in (0, 1):
            try:
                handle = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                owner = self._read_owner()
                if attempt == 0 and owner is not None and not pid_alive(owner):
                    logger.warning(
                        f"[{self.label}] reclaiming stale checkpoint lock %s "
                        "of dead pid %d",
                        self.path,
                        owner,
                    )
                    self.path.unlink(missing_ok=True)
                    continue
                raise RuntimeError(
                    f"the checkpoint directory is locked by another run: {self.path}"
                    + (f" (pid {owner})" if owner is not None else "")
                ) from None
            with os.fdopen(handle, "w", encoding="ascii") as stream:
                stream.write(f"{os.getpid()}\n")
            self.acquired = True
            return

    def _read_owner(self) -> int | None:
        try:
            return int(self.path.read_text(encoding="ascii").split()[0])
        except OSError, ValueError, IndexError:
            return None

    def release(self) -> None:
        if self.acquired:
            self.path.unlink(missing_ok=True)
            self.acquired = False


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return True
    return True


def terminate_all(processes: Mapping[int, Any]) -> None:
    for process in processes.values():
        if process.is_alive():
            process.terminate()
    for process in processes.values():
        process.join(timeout=10)


def remove_tree(path: Path) -> None:
    try:
        for child in sorted(
            path.rglob("*"), key=lambda item: len(item.parts), reverse=True
        ):
            if child.is_dir():
                child.rmdir()
            else:
                child.unlink(missing_ok=True)
        path.rmdir()
    except OSError:  # pragma: no cover - best effort cleanup
        pass


def available_memory_bytes() -> int | None:
    """Memory the kernel reports as available, or ``None`` when unknown.

    Linux parses ``MemAvailable`` from ``/proc/meminfo`` (the reclaimable
    amount, which is the number the guard is specified against).  Other
    platforms use ``sysconf``: the free page count when the platform exposes it
    (not macOS) and otherwise the total physical memory, which is an upper
    bound rather than the available amount.  ``None`` when neither is known, in
    which case the guard only enforces an explicit ``max_mem_gb``.
    """
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        try:
            for line in meminfo.read_text(encoding="ascii").splitlines():
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
        except OSError, ValueError, IndexError:
            return None
        return None
    page_size = None
    for name in ("SC_PAGE_SIZE", "SC_AVPHYS_PAGES", "SC_PHYS_PAGES"):
        try:
            value = int(os.sysconf(name))
        except ValueError, OSError, AttributeError:
            value = None
        if name == "SC_PAGE_SIZE":
            page_size = value
            continue
        if value is not None and value > 0 and page_size:
            return value * page_size
    return None


def log_plan(
    slices: Sequence[tuple[int, int]],
    nk: int,
    mirror: bool,
    per_worker_bytes: int,
    available: int | None,
    max_mem_gb: float | None,
    *,
    label: str = "shard-driver",
) -> None:
    total_pixels = sum((stop - start) * nk for start, stop in slices)
    logger.info(
        f"[{label}] plan nk=%d workers=%d mirror=%s rows=%d pixels=%d",
        nk,
        len(slices),
        mirror,
        sum(stop - start for start, stop in slices),
        total_pixels,
    )
    for index, (start, stop) in enumerate(slices):
        rows = stop - start
        pixels = rows * nk
        logger.info(
            f"[{label}]   worker=%d rows=[%d, %d) rows_n=%d pixels=%d share=%.1f%%",
            index,
            start,
            stop,
            rows,
            pixels,
            100.0 * pixels / total_pixels,
        )
    total_bytes = per_worker_bytes * len(slices)
    logger.info(
        f"[{label}] memory estimate per_worker=%s total=%s available=%s limit=%s",
        format_bytes(per_worker_bytes),
        format_bytes(total_bytes),
        "unknown" if available is None else format_bytes(available),
        "none" if max_mem_gb is None else f"{max_mem_gb:.2f}GB",
    )


def memory_guard(
    per_worker_bytes: int,
    n_workers: int,
    max_mem_gb: float | None,
    *,
    label: str = "shard-driver",
) -> None:
    """Refuse to start when the estimate exceeds the memory budget.

    The default budget is 80% of the memory the kernel reports as available;
    ``max_mem_gb`` overrides it.  When the available memory is unknown and no
    explicit budget was given, the check is skipped with a warning.
    """
    estimated = per_worker_bytes * n_workers
    available = available_memory_bytes()
    if max_mem_gb is None:
        if available is None:
            logger.warning(
                f"[{label}] memory guard skipped: available memory is "
                "unknown on this platform; estimated need %s",
                format_bytes(estimated),
            )
            return
        budget = 0.8 * available
    else:
        budget = float(max_mem_gb) * 1024**3
    if estimated > budget:
        raise MemoryError(
            f"estimated peak memory {format_bytes(estimated)} "
            f"({n_workers} workers x {format_bytes(per_worker_bytes)}) exceeds the "
            f"budget {format_bytes(budget)}"
            + (
                f" (80% of the {format_bytes(available)} the kernel reports as "
                "available)"
                if max_mem_gb is None
                else " (--max-mem-gb)"
            )
            + "; reduce --workers or raise --max-mem-gb explicitly"
        )


def log_aggregate_progress(
    rows_done: int,
    rows_total: int,
    pixels_done: int,
    pixels_total: int,
    started: float,
    active_workers: int,
    rss_peak: int,
    *,
    label: str = "shard-driver",
) -> None:
    elapsed = time.perf_counter() - started
    rate = pixels_done / elapsed if elapsed > 0 else 0.0
    remaining = (pixels_total - pixels_done) / rate if rate > 0 else float("inf")
    logger.info(
        f"[{label}] progress rows=%d/%d (%.1f%%) px=%d/%d px/s=%.0f "
        "elapsed=%.1fs eta=%s active_workers=%d rss_peak=%s",
        rows_done,
        rows_total,
        100.0 * pixels_done / pixels_total,
        pixels_done,
        pixels_total,
        rate,
        elapsed,
        format_hms(remaining),
        active_workers,
        format_bytes(rss_peak),
    )


def absorb_progress(
    message: Any,
    pending: Sequence[tuple[int, int]],
    worker_rows: dict[tuple[int, int], int],
    worker_pixels: dict[tuple[int, int], int],
) -> None:
    worker_id, payload = message
    if worker_id >= len(pending):
        return
    row_range = pending[worker_id]
    rows_done, _rows_total, pixels_done = payload
    worker_rows[row_range] = max(worker_rows[row_range], int(rows_done))
    worker_pixels[row_range] = max(worker_pixels[row_range], int(pixels_done))


def tail(path: Path, lines: int = 12) -> str:
    """Last ``lines`` lines of a text file (best effort)."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return "<no worker log>"
    return "\n".join(text[-lines:]) if text else "<empty worker log>"


# ----------------------------------------------------------------------
# Slice planning
# ----------------------------------------------------------------------
def plan_slices(work_size: int, workers: int) -> list[tuple[int, int]]:
    """Split ``work_size`` units into pixel-balanced contiguous ranges.

    The returned ranges are disjoint, contiguous, ascending and half-open, and
    their lengths differ by at most one - this is the shard-key policy the
    driver assumes (``key == (start, stop)`` over the caller's work axis).
    """
    if work_size < 1:
        raise ValueError(f"work_size must be positive, got {work_size}")
    if workers < 1:
        raise ValueError(f"workers must be positive, got {workers}")
    n_slices = min(int(workers), int(work_size))
    base, remainder = divmod(int(work_size), n_slices)
    slices: list[tuple[int, int]] = []
    start = 0
    for index in range(n_slices):
        stop = start + base + (1 if index < remainder else 0)
        slices.append((start, stop))
        start = stop
    return slices


# ----------------------------------------------------------------------
# Shard store
# ----------------------------------------------------------------------
class ShardStore:
    """HDF5 shard files of one checkpoint directory.

    ``array_keys`` are the datasets every shard stores (all ``float64``);
    ``identity`` is the caller's checkpoint signature - a shard whose stored
    metadata disagrees with it on any known key is *incompatible* and is
    recomputed rather than merged (see :meth:`compatible`).
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        array_keys: Sequence[str],
        identity: Mapping[str, Any],
        generator: str = "shard_driver",
        label: str = "shard-driver",
        logger: logging.Logger | None = None,
    ) -> None:
        self.directory = Path(directory)
        self.array_keys = tuple(array_keys)
        self.identity = dict(identity)
        self.generator = generator
        self.label = label
        self.logger = logger if logger is not None else logging.getLogger(__name__)

    def bind(self, directory: str | Path) -> None:
        """Point the store at the directory the driver actually uses."""
        self.directory = Path(directory)

    def paths(self, key: tuple[int, int]) -> tuple[Path, Path]:
        return _shard_paths(self.directory, key)

    def write(
        self,
        key: tuple[int, int],
        arrays: Mapping[str, np.ndarray],
        meta: Mapping[str, Any],
    ) -> Path:
        return _write_shard(
            self.directory,
            key,
            arrays,
            meta,
            array_keys=self.array_keys,
            generator=self.generator,
        )

    def load(self, key: tuple[int, int]) -> dict[str, Any] | None:
        return _load_shard(
            self.directory,
            key,
            label=self.label,
            array_keys=self.array_keys,
            logger=self.logger,
        )

    def scan(self) -> list[tuple[int, int]]:
        return scan_shards(self.directory)

    def compatible(self, meta: Mapping[str, Any]) -> bool:
        return _shard_is_compatible(meta, self.identity)


# ----------------------------------------------------------------------
# Worker/memory budget
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class RunStats:
    """Numbers the driver measured for the run it just finished."""

    label: str
    slices: tuple[tuple[int, int], ...]
    workers: int
    rows_done: int
    pixels_done: int
    rows_total: int
    pixels_total: int
    live_workers: int
    started: float
    elapsed_s: float
    rss_peak_bytes: int


@dataclass(frozen=True)
class ShardSpec:
    """Everything a calculator must declare to use the driver.

    Attributes
    ----------
    label : str
        Name used in this driver's own log records (``[<label>] ...``).
    store : ShardStore
        Where the shards live and how compatibility is decided.
    plan : callable
        ``() -> list[key]``: the shard keys of this run, in merge order.
    progress_units : callable
        ``key -> (rows, pixels)``: what one shard contributes to the progress
        accounting and to the final report.
    worker_process : callable
        ``(context, index, key, result_queue, progress_queue) -> Process``:
        build (not start) the worker that computes ``key``.
    worker_payload : callable
        ``() -> Mapping``: picklable arguments handed to every worker.
    log_plan : callable
        ``(keys, per_worker_bytes) -> None``: log the plan in the caller's own
        wording (the driver calls it before the start and again after the
        resume filter).
    log_progress : callable
        ``(RunStats) -> None``: the caller's aggregate-progress record.
    describe_plan : callable or None
        ``(keys) -> str``: the line printed by ``dry_run``; ``None`` prints
        nothing extra.
    describe_worker : callable
        ``(key) -> str``: what the driver prints when a shard is dispatched.
    finalize : callable
        ``(loaded, reports, stats) -> int``: merge the shards **in plan order**
        (the driver loaded them in that order), publish the product and return
        an exit code (``0`` on success).  Digest verification, the product
        write and the closing summary belong here.
    worker_log_dir : str or Path or None
        Where the per-worker detail logs go (defaults to the checkpoint dir).
    """

    label: str
    store: ShardStore
    plan: Callable[[], list[tuple[int, int]]]
    progress_units: Callable[[tuple[int, int]], tuple[int, int]]
    worker_process: Callable[..., Any]
    worker_payload: Callable[[], Mapping[str, Any]]
    log_plan: Callable[[Sequence[tuple[int, int]], int], None]
    log_progress: Callable[[RunStats], None]
    finalize: Callable[
        [list[dict[str, Any]], Mapping[int, Mapping[str, Any]], RunStats], int
    ]
    describe_plan: Callable[[Sequence[tuple[int, int]]], str] | None = None
    worker_log_dir: str | Path | None = None
    logger: logging.Logger | None = None
    per_worker_bytes: int = 0
    max_mem_gb: float | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)


def run_shards(
    spec: ShardSpec,
    *,
    n_workers: int,
    checkpoint_dir: str | Path | None = None,
    resume: bool = False,
    start_method: str = "spawn",
    progress_interval_s: float = 30.0,
    dry_run: bool = False,
) -> int:
    """Dispatch the missing shards of ``spec`` and merge them into the product.

    Returns ``0`` on success, ``130`` when interrupted by SIGINT, ``2`` when the
    checkpoint directory is locked, ``3`` when the memory budget refuses the
    run, ``4`` when a worker fails and ``5`` when a shard or its report is
    missing; ``spec.finalize`` may return further codes of its own.
    """
    if n_workers < 1:
        raise ValueError(f"n_workers must be positive, got {n_workers}")
    if start_method not in ("spawn", "fork", "forkserver"):
        raise ValueError(f"unsupported start method: {start_method!r}")

    label = spec.label
    log = logger if spec.logger is None else spec.logger
    keys = list(spec.plan())
    units = {key: spec.progress_units(key) for key in keys}
    rows_total = sum(rows for rows, _ in units.values())
    pixels_total = sum(pixels for _, pixels in units.values())

    if dry_run:
        spec.log_plan(keys, spec.per_worker_bytes)
        if spec.describe_plan is not None:
            print(spec.describe_plan(keys))
        return 0

    auto_checkpoint = checkpoint_dir is None
    if auto_checkpoint:
        ckpt_dir = Path(tempfile.mkdtemp(prefix=f"{label.lower()}_ckpt_"))
    else:
        ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(spec.worker_log_dir) if spec.worker_log_dir is not None else ckpt_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    spec.store.bind(ckpt_dir)

    exit_code = 0
    lock = CheckpointLock(ckpt_dir / ".lock")
    try:
        try:
            lock.acquire()
        except RuntimeError as exc:
            log.error("[%s] %s", label, exc)
            return 2

        pending: list[tuple[int, int]] = []
        for key in keys:
            if resume:
                stored = spec.store.load(key)
                if stored is not None and spec.store.compatible(stored["meta"]):
                    log.info(
                        "[%s] resume: rows=[%d, %d) already complete",
                        label,
                        key[0],
                        key[1],
                    )
                    continue
                if stored is not None:
                    log.warning(
                        "[%s] resume: checkpoint rows=[%d, %d) does not match the "
                        "requested parameters and is recomputed",
                        label,
                        key[0],
                        key[1],
                    )
            pending.append(key)

        spec.log_plan(keys, spec.per_worker_bytes)
        try:
            memory_guard(spec.per_worker_bytes, len(pending), spec.max_mem_gb)
        except MemoryError as exc:
            log.error("[%s] refusing to start: %s", label, exc)
            return 3

        if not pending:
            log.info("[%s] nothing to dispatch: all slices complete", label)

        context = mp.get_context(start_method)
        progress_queue = context.Queue()
        result_queue = context.Queue()
        processes: dict[int, Any] = {}
        reports: dict[int, dict[str, Any]] = {}
        worker_rows = dict.fromkeys(pending, 0)
        worker_pixels = dict.fromkeys(pending, 0)

        started = time.perf_counter()
        last_progress = started
        rss_peak = peak_rss_bytes()
        stop_requested = {"value": False}

        def _request_stop(signum: int, frame: Any) -> None:  # pragma: no cover
            stop_requested["value"] = True

        previous_handler: Any = None
        try:
            previous_handler = signal.signal(signal.SIGINT, _request_stop)
        except ValueError:
            previous_handler = None

        worker_payload = spec.worker_payload()
        try:
            for index, key in enumerate(pending):
                process = spec.worker_process(
                    context, index, key, worker_payload, result_queue, progress_queue
                )
                process.start()
                processes[index] = process
                log.info(
                    "[%s] dispatched worker=%d pid=%d rows=[%d, %d)",
                    label,
                    index,
                    process.pid,
                    key[0],
                    key[1],
                )

            while processes:
                if stop_requested["value"]:
                    log.error(
                        "[%s] SIGINT received: terminating %d live worker(s); "
                        "completed checkpoints are kept",
                        label,
                        len(processes),
                    )
                    terminate_all(processes)
                    exit_code = 130
                    break

                try:
                    message = progress_queue.get(timeout=0.25)
                except queue_module.Empty:
                    message = None
                while message is not None:
                    absorb_progress(message, pending, worker_rows, worker_pixels)
                    try:
                        message = progress_queue.get_nowait()
                    except queue_module.Empty:
                        message = None

                while True:
                    try:
                        report = result_queue.get_nowait()
                    except queue_module.Empty:
                        break
                    reports[int(report["worker_id"])] = report

                failed = None
                for index, process in list(processes.items()):
                    if process.is_alive():
                        continue
                    process.join()
                    del processes[index]
                    key = pending[index]
                    if process.exitcode != 0:
                        failed = (index, process.exitcode, key)
                        break
                    worker_rows[key] = key[1] - key[0]
                    worker_pixels[key] = units[key][1]
                    rss_peak = max(
                        rss_peak,
                        int(reports.get(index, {}).get("rss_peak_bytes", 0) or 0),
                    )

                if failed is not None:
                    index, exitcode, key = failed
                    log_path = log_dir / f"worker_{index}.log"
                    log.error(
                        "[%s] worker=%d (rows=[%d, %d)) exited with code %s; aborting "
                        "the remaining dispatch and keeping the completed "
                        "checkpoints. Last lines of %s:\n%s",
                        label,
                        index,
                        key[0],
                        key[1],
                        exitcode,
                        log_path,
                        tail(log_path),
                    )
                    error_report = reports.get(index)
                    if (
                        error_report is not None
                        and error_report.get("status") == "error"
                    ):
                        log.error(
                            "[%s] worker=%d traceback:\n%s",
                            label,
                            index,
                            error_report.get("error"),
                        )
                    terminate_all(processes)
                    return 4

                now = time.perf_counter()
                if (
                    progress_interval_s > 0
                    and now - last_progress >= progress_interval_s
                ):
                    spec.log_progress(
                        _stats(
                            spec,
                            keys,
                            pending,
                            worker_rows,
                            worker_pixels,
                            rows_total,
                            pixels_total,
                            started,
                            rss_peak,
                        )
                    )
                    last_progress = now

            if exit_code == 0:
                if progress_interval_s > 0:
                    spec.log_progress(
                        _stats(
                            spec,
                            keys,
                            pending,
                            worker_rows,
                            worker_pixels,
                            rows_total,
                            pixels_total,
                            started,
                            rss_peak,
                        )
                    )

                if len(reports) != len(pending):
                    log.error(
                        "[%s] %d of %d workers reported a result; no output is written",
                        label,
                        len(reports),
                        len(pending),
                    )
                    return 5

                loaded: list[dict[str, Any]] = []
                for key in keys:
                    stored = spec.store.load(key)
                    if stored is None:
                        log.error(
                            "[%s] missing checkpoint for rows=[%d, %d)",
                            label,
                            key[0],
                            key[1],
                        )
                        return 5
                    loaded.append(stored)

                stats = _stats(
                    spec,
                    keys,
                    pending,
                    worker_rows,
                    worker_pixels,
                    rows_total,
                    pixels_total,
                    started,
                    rss_peak,
                )
                code = spec.finalize(loaded, reports, stats)
                if code:
                    return code
        finally:
            if previous_handler is not None:
                signal.signal(signal.SIGINT, previous_handler)
    finally:
        lock.release()
        if auto_checkpoint and exit_code == 0:
            remove_tree(ckpt_dir)

    return exit_code


def _stats(
    spec: ShardSpec,
    keys: Sequence[tuple[int, int]],
    pending: Sequence[tuple[int, int]],
    worker_rows: Mapping[tuple[int, int], int],
    worker_pixels: Mapping[tuple[int, int], int],
    rows_total: int,
    pixels_total: int,
    started: float,
    rss_peak: int,
) -> RunStats:
    return RunStats(
        label=spec.label,
        slices=tuple(keys),
        workers=len(pending),
        rows_done=sum(worker_rows.get(key, 0) for key in keys),
        pixels_done=sum(worker_pixels.get(key, 0) for key in keys),
        rows_total=rows_total,
        pixels_total=pixels_total,
        live_workers=len(pending)
        - sum(1 for key in pending if worker_rows.get(key, 0) >= key[1] - key[0]),
        started=started,
        elapsed_s=time.perf_counter() - started,
        rss_peak_bytes=rss_peak,
    )


def scan_shards(checkpoint_dir: str | Path) -> list[tuple[int, int]]:
    """Row ranges that have a complete shard in ``checkpoint_dir``."""
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.is_dir():
        return []
    found: list[tuple[int, int]] = []
    for done_path in sorted(ckpt_dir.glob("rows_*_*.done")):
        parts = done_path.stem.split("_")
        if len(parts) != 3:
            continue
        try:
            start, stop = int(parts[1]), int(parts[2])
        except ValueError:
            continue
        h5_path, done_path = _shard_paths(ckpt_dir, (start, stop))
        if h5_path.exists() and done_path.exists():
            found.append((start, stop))
    return sorted(found)
