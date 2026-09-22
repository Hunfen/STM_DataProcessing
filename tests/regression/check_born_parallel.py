"""Regression checks for the opt-in parallel Born QPI path.

Covers the shard-and-merge adoption in :class:`BornQPI`:

  (a) serial == parallel: bit-for-bit identical layers AND identical h5 products
      (the clock is frozen so the convention's ``creation_date`` cannot differ);
  (b) checkpoint lifecycle: shards are HDF5, written through ``io.h5_convention``,
      carry the full shard identity, the automatic checkpoint directory is
      removed after a successful merge, and an explicit one is kept for resume;
  (c) resume: a real interrupted run (SIGINT in a subprocess) resumes from the
      shards on disk, recomputes ONLY the missing energies and still produces the
      full, bit-correct product;
  (d) identity: shards produced with another broadening, another model or
      another k-grid are rejected as incompatible and recomputed, with the
      operator-visible warning;
  (e) the CPU/GPU backend hazard: a parallel request under the GPU backend falls
      back to the serial path with a WARNING instead of forking workers that
      would inherit a CUDA context;
  (f) the shared layer function still evaluates the pre-adoption serial formula
      bit for bit, and the serial loop and the worker both call it, so a drift in
      it cannot hide behind (a) (which compares the parallel path against the
      serial one and would move with it).

The check is synthetic (a mock Hamiltonian, per-run temporary directories, no
external data and no fixed scratch path the other checks share), so it runs in
CI rather than being deselected.

Run from the repository root:
    .venv/bin/python tests/regression/check_born_parallel.py
"""

from __future__ import annotations

import hashlib
import inspect
import logging
import re
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import h5py
import numpy as np

_REGRESSION_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _REGRESSION_DIR.parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from stm_data_processing.dft.wannier90.mlwf_gk import GreenFunction  # noqa: E402
from stm_data_processing.io import h5_convention  # noqa: E402
from stm_data_processing.parallel import shard_driver  # noqa: E402
from stm_data_processing.stm import qpi_born  # noqa: E402
from stm_data_processing.stm.qpi_born import BornQPI  # noqa: E402

_NK = 64
_ETA = 0.05
_N_ENERGIES = 64
_CHILD_WORKERS = 4
_FROZEN_DATE = "2026-01-01T00:00:00+00:00"

#: The driver's own dispatch/resume log records (parsed to prove what ran).
_DISPATCH_RE = re.compile(r"dispatched worker=\d+ pid=\d+ rows=\[(\d+), (\d+)\)")
_REUSED_RE = re.compile(r"resume: rows=\[(\d+), (\d+)\) already complete")

logger = logging.getLogger("check_born_parallel")

_CHILD_SCRIPT = '''
"""Synthetic Born QPI run used by check_born_parallel (interrupt/resume)."""
import logging
import pathlib
import sys

import numpy as np

REPO = pathlib.Path({repo!r})
sys.path.insert(0, str(REPO / "src"))

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)

from stm_data_processing.stm.qpi_born import BornQPI


class MockMLWFHamiltonian:
    def __init__(self, num_wann=2, seed=11):
        self.num_wann = num_wann
        self.bvecs = None
        self._rng = np.random.default_rng(seed)

    def hk(self, k_points):
        n = len(k_points)
        nw = self.num_wann
        h = self._rng.normal(size=(n, nw, nw)) + 1j * self._rng.normal(size=(n, nw, nw))
        return (h + np.conj(np.swapaxes(h, 1, 2))) / 2


if __name__ == "__main__":
    resume = "--resume" in sys.argv
    calc = BornQPI(MockMLWFHamiltonian(), nk={nk}, eta={eta!r})
    energies = np.linspace(-1.0, 1.0, {n_energies})
    result = calc.calculate(
        energies,
        q_range=None,
        output_path={out!r},
        n_workers={workers},
        checkpoint_dir={ckpt!r},
        resume=resume,
    )
    print("CHILD-OK", result["qpi_layers"].shape)
'''


class MockMLWFHamiltonian:
    """Minimal stand-in exposing the MLWFHamiltonian interface BornQPI needs."""

    def __init__(self, num_wann: int = 2, seed: int = 11) -> None:
        self.num_wann = num_wann
        self.bvecs = None
        self._rng = np.random.default_rng(seed)

    def hk(self, k_points: np.ndarray) -> np.ndarray:
        """Return a batch of Hermitian H(k) matrices (deterministic per seed)."""
        n = len(k_points)
        nw = self.num_wann
        h = self._rng.normal(size=(n, nw, nw)) + 1j * self._rng.normal(size=(n, nw, nw))
        return (h + np.conj(np.swapaxes(h, 1, 2))) / 2


def _calculator(
    nk: int = 8,
    eta: float = _ETA,
    num_wann: int = 2,
    seed: int = 11,
) -> BornQPI:
    return BornQPI(MockMLWFHamiltonian(num_wann=num_wann, seed=seed), nk=nk, eta=eta)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _RecordCollector(logging.Handler):
    """Capture the log records a check asserts on."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def _file_fingerprint(path: Path) -> dict:
    """Datasets (name -> sha256 of the raw bytes) plus attrs minus the clock."""
    fingerprint: dict = {"datasets": {}, "attrs": {}}
    with h5py.File(path, "r") as handle:
        for name in sorted(handle):
            dataset = handle[name]
            fingerprint["datasets"][name] = (
                dataset.shape,
                str(dataset.dtype),
                hashlib.sha256(np.ascontiguousarray(dataset[:]).tobytes()).hexdigest(),
            )
        fingerprint["attrs"] = {
            key: repr(handle.attrs[key])
            for key in handle.attrs
            if key != h5_convention.CREATION_DATE_ATTR
        }
    return fingerprint


def _grid_sha256(calc: BornQPI) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(calc.hk_grid, dtype=np.complex128).tobytes()
    ).hexdigest()


def check_bit_identity() -> None:
    """(a) the parallel layers and the parallel h5 product are identical."""
    compared = 0
    for nk, num_wann, n_energies, workers in ((8, 2, 21, 3), (16, 3, 9, 4)):
        calc = _calculator(nk=nk, num_wann=num_wann)
        energies = np.linspace(-1.0, 1.0, n_energies)
        root = Path(tempfile.mkdtemp(prefix="born_parallel_"))
        serial_path, parallel_path = root / "serial.h5", root / "parallel.h5"

        original_clock = h5_convention.utc_now_iso
        h5_convention.utc_now_iso = lambda: _FROZEN_DATE
        try:
            serial = calc.calculate(
                energies, q_range=None, output_path=str(serial_path)
            )
            parallel = calc.calculate(
                energies,
                q_range=None,
                output_path=str(parallel_path),
                n_workers=workers,
                checkpoint_dir=root / "ckpt",
            )
        finally:
            h5_convention.utc_now_iso = original_clock

        a = serial["qpi_layers"]
        b = parallel["qpi_layers"]
        assert a.shape == b.shape and a.dtype == b.dtype, (
            a.shape,
            b.shape,
            a.dtype,
            b.dtype,
        )
        # A vacuous comparison would be one where the layers carry no signal.
        assert np.all(np.isfinite(a)), "the serial layers are not finite"
        assert float(np.ptp(a)) > 0.0, "the serial layers are constant"
        deviation = float(np.max(np.abs(a - b)))
        assert deviation == 0.0, f"parallel layers differ from serial by {deviation}"
        assert np.array_equal(a.view(np.uint8), b.view(np.uint8)), (
            "parallel layers are not bit-identical to the serial ones"
        )
        assert sorted(serial) == sorted(parallel), "return structure changed"
        assert np.array_equal(serial["q1_grid"], parallel["q1_grid"])
        assert np.array_equal(serial["q2_grid"], parallel["q2_grid"])

        serial_sha, parallel_sha = _sha256(serial_path), _sha256(parallel_path)
        assert serial_sha == parallel_sha, (
            "product files differ:\n"
            f"  serial   {serial_sha}\n  parallel {parallel_sha}\n"
            f"  fingerprints equal: "
            f"{_file_fingerprint(serial_path) == _file_fingerprint(parallel_path)}"
        )
        print(
            f"  [a] nk={nk} num_wann={num_wann} nω={n_energies} workers={workers}: "
            f"serial == parallel, max|delta| = {deviation:.1e} (bitwise identical by "
            f"uint8 view), product sha256 = {serial_sha[:16]}... identical "
            f"({serial_path.stat().st_size} bytes)"
        )
        compared += 1

    assert compared >= 1, "check (a) compared nothing"
    # the serial path stays the DEFAULT: it must not touch the shard driver
    calc = _calculator()
    energies = np.linspace(-1.0, 1.0, 5)
    created: list[Path] = []
    original_mkdtemp = shard_driver.tempfile.mkdtemp

    def _recording_mkdtemp(*args: object, **kwargs: object) -> str:
        path = original_mkdtemp(*args, **kwargs)
        created.append(Path(path))
        return path

    shard_driver.tempfile.mkdtemp = _recording_mkdtemp
    try:
        default = calc.calculate(energies, q_range=None)
    finally:
        shard_driver.tempfile.mkdtemp = original_mkdtemp
    assert not created, "the default (n_workers=None) run used the parallel driver"
    assert default["qpi_layers"].shape == (len(energies), calc.nk, calc.nk)

    # n_workers <= 1 must stay on the serial loop too (and write no shard)
    ckpt = Path(tempfile.mkdtemp(prefix="born_parallel_")) / "ckpt"
    single = calc.calculate(energies, q_range=None, n_workers=1, checkpoint_dir=ckpt)
    assert not ckpt.exists(), "n_workers=1 wrote shards instead of staying serial"
    assert np.array_equal(
        single["qpi_layers"].view(np.uint8), default["qpi_layers"].view(np.uint8)
    ), "n_workers=1 changed the serial result"
    print(
        f"  [a] serial default unchanged over {compared} scale(s): n_workers=None and "
        f"n_workers=1 ran the serial loop without touching the shard driver and gave "
        f"identical layers {default['qpi_layers'].shape}"
    )


def check_checkpoint_lifecycle() -> None:
    """(b) HDF5 shards with the full identity, explicit dir kept, auto dir removed."""
    calc = _calculator()
    energies = np.linspace(-1.0, 1.0, 9)
    workers = 3
    root = Path(tempfile.mkdtemp(prefix="born_parallel_"))
    ckpt = root / "ckpt"
    result = calc.calculate(
        energies, q_range=None, n_workers=workers, checkpoint_dir=ckpt
    )

    names = sorted(p.name for p in ckpt.iterdir())
    h5_shards = [name for name in names if name.endswith(".h5")]
    expected_shards = len(shard_driver.plan_slices(len(energies), workers))
    assert len(h5_shards) == expected_shards, (
        f"expected {expected_shards} shards, found {h5_shards}"
    )
    assert all((ckpt / name.replace(".h5", ".done")).exists() for name in h5_shards)

    # the product really is the plan-order concatenation of the shards
    pieces = []
    for key in shard_driver.plan_slices(len(energies), workers):
        piece_path = ckpt / f"rows_{key[0]}_{key[1]}.h5"
        with h5py.File(piece_path, "r") as handle:
            pieces.append(np.asarray(handle["qpi_layers"][:], dtype=np.float64))
    assert len(pieces) == expected_shards, "the plan-order shard load lost a shard"
    concatenated = np.concatenate(pieces, axis=0)
    assert concatenated.shape == result["qpi_layers"].shape, (
        concatenated.shape,
        result["qpi_layers"].shape,
    )
    assert np.array_equal(
        concatenated.view(np.uint8), result["qpi_layers"].view(np.uint8)
    ), "the merged layers are not the plan-order concatenation of the shards"
    print(
        f"  [b] the product is the plan-order concatenation of the "
        f"{expected_shards} shards bit for bit ({concatenated.shape})"
    )

    shard_path = ckpt / h5_shards[0]
    with h5py.File(shard_path, "r") as handle:
        dataset = handle["qpi_layers"]
        shape, dtype = dataset.shape, dataset.dtype
        chunks = dataset.chunks
        compression = dataset.compression
        opts = dataset.compression_opts
        attrs = {key: handle.attrs[key] for key in sorted(handle.attrs)}
    for required in ("schema_version", "generator", "creation_date"):
        assert required in attrs, f"shard {shard_path.name} lacks '{required}'"
    for required in (
        "nk",
        "eta",
        "num_wann",
        "n_energies",
        "hk_grid_sha256",
        "V_sha256",
    ):
        assert required in attrs, (
            f"shard {shard_path.name} lacks identity key '{required}'"
        )
    assert str(attrs["generator"]) == "BornQPI", attrs["generator"]
    stored_sha = attrs["hk_grid_sha256"]
    stored_sha = (
        stored_sha.decode() if isinstance(stored_sha, bytes) else str(stored_sha)
    )
    assert stored_sha == _grid_sha256(calc), (
        "the shard identity does not digest the hk_grid actually used"
    )
    assert int(attrs["nk"]) == calc.nk and int(attrs["n_energies"]) == len(energies)
    assert compression == h5_convention.COMPRESSION, compression
    assert opts == h5_convention.COMPRESSION_OPTS, opts
    assert chunks is not None, "the shard dataset is not chunked"
    assert shape[1:] == (calc.nk, calc.nk), shape
    print(
        f"  [b] {len(h5_shards)} shard(s) {h5_shards}: dataset 'qpi_layers' {shape} "
        f"{dtype} chunks={chunks} compression={compression}/{opts}; "
        f"identity attrs nk/eta/num_wann/n_energies/hk_grid_sha256/V_sha256 present"
    )
    print(f"  [b] explicit checkpoint dir kept for resume: {names}")

    created: list[Path] = []
    original_mkdtemp = shard_driver.tempfile.mkdtemp

    def _recording_mkdtemp(*args: object, **kwargs: object) -> str:
        path = original_mkdtemp(*args, **kwargs)
        created.append(Path(path))
        return path

    shard_driver.tempfile.mkdtemp = _recording_mkdtemp
    try:
        calc.calculate(energies, q_range=None, n_workers=workers)
    finally:
        shard_driver.tempfile.mkdtemp = original_mkdtemp

    assert created, "the automatic checkpoint directory was never created"
    assert not created[0].exists(), (
        f"the automatic checkpoint directory {created[0]} survived a successful merge"
    )
    print(f"  [b] automatic checkpoint dir {created[0].name} removed after the merge")

    # A merge whose layer total is not len(energy_range) must fail loudly: the
    # shards here cover one energy less than requested (plan patched), so
    # finalize has to return its non-zero code and no product may appear.
    short_root = Path(tempfile.mkdtemp(prefix="born_parallel_"))
    short_product = short_root / "short.h5"
    original_plan = qpi_born.plan_slices
    qpi_born.plan_slices = lambda work_size, n: original_plan(work_size - 1, n)
    try:
        try:
            calc.calculate(
                energies,
                q_range=None,
                output_path=str(short_product),
                n_workers=workers,
                checkpoint_dir=short_root / "ckpt",
            )
        except RuntimeError as exc:
            assert "exit code 6" in str(exc), str(exc)
        else:
            raise AssertionError(
                "a merge with fewer layers than len(energy_range) was accepted"
            )
    finally:
        qpi_born.plan_slices = original_plan
    assert not short_product.exists(), (
        "a product was written even though the merge was short"
    )
    print(
        "  [b] a short merge (len(plan) < len(energy_range)) -> finalize code 6 -> "
        "RuntimeError, no product written"
    )


def _run_child(script: Path, *extra: str) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, str(script), *extra],
        cwd=str(_REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def check_interrupt_resume() -> None:
    """(c) a real SIGINT leaves complete shards; resume recomputes only the rest."""
    root = Path(tempfile.mkdtemp(prefix="born_parallel_"))
    ckpt = root / "ckpt"
    product = root / "resumed.h5"
    script = root / "child.py"
    script.write_text(
        _CHILD_SCRIPT.format(
            repo=str(_REPO_ROOT),
            nk=_NK,
            eta=_ETA,
            n_energies=_N_ENERGIES,
            workers=_CHILD_WORKERS,
            out=str(product),
            ckpt=str(ckpt),
        ),
        encoding="utf-8",
    )

    # The plan splits the energy axis into one contiguous slice per worker, so
    # the child's shard count is the number of workers it asks for.
    plan = shard_driver.plan_slices(_N_ENERGIES, _CHILD_WORKERS)
    total_shards = len(plan)
    process = _run_child(script)
    deadline = time.time() + 120
    complete: list[str] = []
    while time.time() < deadline and process.poll() is None:
        complete = (
            sorted(p.name for p in ckpt.glob("rows_*.done")) if ckpt.exists() else []
        )
        if complete:
            break
        time.sleep(0.002)
    process.send_signal(signal.SIGINT)
    _, stderr = process.communicate(timeout=180)
    on_disk = sorted(p.name for p in ckpt.glob("rows_*.done"))
    assert on_disk, f"the interrupted run left no complete shard: {stderr[-2000:]}"
    assert process.returncode in (130, -2, 1), (
        f"an interrupted parallel run must end as an interrupt, not as a failure "
        f"(130 in a shell, -2 as Popen reports a process killed by SIGINT, 1 for "
        f"the propagated KeyboardInterrupt), got {process.returncode}: "
        f"{stderr[-2000:]}"
    )
    assert "SIGINT received" in stderr, stderr[-2000:]
    # 130 must surface as KeyboardInterrupt naming the kept checkpoint directory.
    assert "KeyboardInterrupt" in stderr, stderr[-3000:]
    assert f"completed shards kept in {ckpt}" in stderr, stderr[-3000:]
    assert len(on_disk) < total_shards, (
        f"the run was not interrupted mid-flight: all {total_shards} shards are on disk"
    )
    kept_ranges = {
        tuple(int(part) for part in Path(name).stem.split("_")[1:]) for name in on_disk
    }
    print(
        f"  [c] interrupted run: exit={process.returncode}, "
        f"{len(on_disk)} of {total_shards} complete shard(s) kept ({on_disk}), "
        f"{total_shards - len(on_disk)} still missing; KeyboardInterrupt names the "
        f"kept dir"
    )

    resumed = _run_child(script, "--resume")
    stdout, stderr = resumed.communicate(timeout=300)
    assert resumed.returncode == 0, stderr[-3000:]
    assert "CHILD-OK" in stdout, stdout[-1000:]
    dispatched = {
        (int(start), int(stop)) for start, stop in _DISPATCH_RE.findall(stderr)
    }
    reused = {(int(start), int(stop)) for start, stop in _REUSED_RE.findall(stderr)}
    missing = set(plan) - kept_ranges
    assert reused == kept_ranges, (
        f"resume did not reuse every complete shard: {sorted(reused)} vs "
        f"{sorted(kept_ranges)}"
    )
    assert dispatched == missing, (
        f"resume dispatched {sorted(dispatched)}, expected exactly the missing "
        f"{sorted(missing)}"
    )
    assert sum(stop - start for start, stop in dispatched) == (
        _N_ENERGIES - sum(stop - start for start, stop in kept_ranges)
    )
    print(
        f"  [c] resume: dispatched only {sorted(dispatched)} (the "
        f"{sum(stop - start for start, stop in dispatched)} missing energies), "
        f"reused {sorted(reused)} from disk"
    )

    # The resumed run must still produce the full product, bit for bit.
    energies = np.linspace(-1.0, 1.0, _N_ENERGIES)
    reference = _calculator(nk=_NK).calculate(energies, q_range=None)["qpi_layers"]
    with h5py.File(product, "r") as handle:
        resumed_layers = np.asarray(handle["qpi_layers"][:], dtype=np.float64)
    assert resumed_layers.shape == reference.shape, (
        resumed_layers.shape,
        reference.shape,
    )
    assert np.all(np.isfinite(resumed_layers)), "the resumed product is not finite"
    assert np.array_equal(resumed_layers.view(np.uint8), reference.view(np.uint8)), (
        "the resumed product is not bit-identical to the serial one"
    )
    print(
        f"  [c] resumed product {resumed_layers.shape} is bit-identical to the serial "
        f"reference ({len(energies)} energies)"
    )


def check_incompatible_shard() -> None:
    """(d) shards from different parameters are rejected, not merged."""
    energies = np.linspace(-1.0, 1.0, 9)
    workers = 3
    shard_count = len(shard_driver.plan_slices(len(energies), workers))
    variants = {
        "different eta (broadening)": {"nk": 8, "eta": _ETA * 1.5, "seed": 11},
        "another model (different hk_grid)": {"nk": 8, "eta": _ETA, "seed": 12},
        "another k-grid (different nk)": {"nk": 16, "eta": _ETA, "seed": 11},
    }
    checked = 0
    for name, variant in variants.items():
        root = Path(tempfile.mkdtemp(prefix="born_parallel_"))
        ckpt = root / "ckpt"
        baseline = _calculator()
        baseline.calculate(
            energies, q_range=None, n_workers=workers, checkpoint_dir=ckpt
        )
        on_disk = sorted(p.name for p in ckpt.glob("rows_*.done"))
        assert len(on_disk) == shard_count, on_disk

        handler = _RecordCollector()
        qpi_born.logger.addHandler(handler)
        try:
            other = _calculator(**variant)
            result = other.calculate(
                energies,
                q_range=None,
                n_workers=workers,
                checkpoint_dir=ckpt,
                resume=True,
            )
        finally:
            qpi_born.logger.removeHandler(handler)

        messages = [record.getMessage() for record in handler.records]
        rejected = [
            message
            for message in messages
            if "does not match the requested parameters" in message
        ]
        assert len(rejected) == shard_count, (
            f"{name}: {len(rejected)} incompatibility warning(s) for {shard_count} "
            f"shard(s): {messages}"
        )
        assert len(result["qpi_layers"]) == len(energies)
        assert result["qpi_layers"].shape == (
            len(energies),
            variant["nk"],
            variant["nk"],
        )
        assert float(np.ptp(result["qpi_layers"])) > 0.0
        print(
            f"  [d] {name} -> all {len(rejected)} shard(s) rejected and recomputed "
            f"with the warning {rejected[0][:88]!r}"
        )
        checked += 1

    assert checked == len(variants), "check (d) exercised fewer variants than declared"


def check_gpu_backend_falls_back() -> None:
    """(e) a parallel request under the GPU backend falls back to serial."""
    calc = _calculator()
    energies = np.linspace(-1.0, 1.0, 5)
    root = Path(tempfile.mkdtemp(prefix="born_parallel_"))
    ckpt = root / "ckpt"

    original_backend, original_cp = qpi_born.BACKEND, qpi_born.cp
    cuda_calls: list[float] = []

    def _stub_cuda(omega: float) -> np.ndarray:
        # Stand-in for the CUDA path (cupy is not installed here): it must be the
        # branch the GPU backend would take, so record the energy and reuse the
        # CPU arithmetic to keep the check self-contained.
        cuda_calls.append(float(omega))
        return calc._compute_Gkq(omega)

    handler = _RecordCollector()
    qpi_born.logger.addHandler(handler)
    qpi_born.BACKEND = "gpu"
    qpi_born.cp = object()  # makes `BACKEND == "gpu"` the live backend
    original_cuda = calc._compute_Gkq_cuda
    calc._compute_Gkq_cuda = _stub_cuda  # type: ignore[method-assign]
    try:
        result = calc.calculate(
            energies,
            q_range=None,
            n_workers=3,
            checkpoint_dir=ckpt,
        )
    finally:
        calc._compute_Gkq_cuda = original_cuda  # type: ignore[method-assign]
        qpi_born.BACKEND, qpi_born.cp = original_backend, original_cp
        qpi_born.logger.removeHandler(handler)

    messages = [record.getMessage() for record in handler.records]
    warned = [message for message in messages if "CUDA context" in message]
    assert warned, f"no GPU-backend warning was emitted: {messages}"
    assert cuda_calls == [float(energy) for energy in energies], (
        f"the GPU backend did not take the serial CUDA path for every energy: "
        f"{cuda_calls}"
    )
    assert not ckpt.exists(), "the fallback still created shards"
    assert result["qpi_layers"].shape == (len(energies), calc.nk, calc.nk)
    assert float(np.ptp(result["qpi_layers"])) > 0.0
    print(
        f"  [e] GPU backend + n_workers=3: WARNING {warned[0][:104]!r}, serial CUDA "
        f"path used for all {len(cuda_calls)} energies, no shards written"
    )


def _reference_layer(
    hk_grid: np.ndarray, V: np.ndarray, eta: float, omega: float
) -> np.ndarray:
    """The layer formula exactly as the pre-adoption serial loop expressed it.

    ``_compute_Gkq`` used to call ``GreenFunction.compute_green`` and then inline
    the ``GV`` trace, the FFT correlation and ``fftshift``, while ``calculate``
    divided the returned map by ``np.pi``; the parallel adoption moved both
    bodies into ``_born_layer``, so this is their frozen reference form (the
    Green's function still comes from the untouched library object).
    """
    num_wann = hk_grid.shape[-1]
    nk = hk_grid.shape[0]
    g0_k = GreenFunction(MockMLWFHamiltonian(num_wann=num_wann), eta=eta).compute_green(
        hk_grid, omega
    )

    gv = np.einsum("ijab,bc->ijac", g0_k, V, optimize=True)

    qpi = np.zeros((nk, nk), dtype=np.float64)
    for c in range(num_wann):
        fft_gv_c = np.conj(np.fft.fftn(np.conj(gv[:, :, :, c]), axes=(0, 1)))
        fft_g0_c = np.fft.fftn(g0_k[:, :, c, :], axes=(0, 1))
        corr_c = np.fft.ifftn(fft_gv_c * fft_g0_c, axes=(0, 1))
        qpi += -np.imag(np.sum(corr_c, axis=2))

    return np.fft.fftshift(qpi) / np.pi


def check_serial_formula_frozen() -> None:
    """(f) the shared layer function still evaluates the pre-refactor formula.

    Checks (a)-(e) compare the parallel path against the serial one; if
    ``_born_layer`` drifted, both sides would drift together and every one of
    those comparisons would still pass.  This check pins the arithmetic the
    serial loop used before the adoption and compares it bit for bit.
    """
    # Both paths must go through the same function for that pin to mean anything.
    assert "_born_layer(" in inspect.getsource(qpi_born.BornQPI._compute_Gkq), (
        "the serial CPU path no longer calls the shared _born_layer"
    )
    assert "_born_layer(" in inspect.getsource(qpi_born._qpi_born_worker), (
        "the parallel worker no longer calls the shared _born_layer"
    )

    compared = 0
    for nk, num_wann in ((8, 2), (16, 3)):
        calc = _calculator(nk=nk, num_wann=num_wann)
        # The serial wrapper must not grow arithmetic of its own: it has to
        # return the shared layer of its own hk_grid/V/eta unchanged.
        for energy in (-0.37, 0.61):
            wrapper = calc._compute_Gkq(energy)
            shared = qpi_born._born_layer(calc.hk_grid, calc.V, calc.eta, energy)
            assert np.array_equal(wrapper.view(np.uint8), shared.view(np.uint8)), (
                f"_compute_Gkq no longer returns _born_layer at nk={nk}, E={energy}"
            )
        for potential in (calc.V, 0.25 * calc.V + 0.1 * (calc.V + np.eye(num_wann))):
            for energy in (-0.37, 0.0, 0.61):
                expected = _reference_layer(calc.hk_grid, potential, calc.eta, energy)
                actual = qpi_born._born_layer(calc.hk_grid, potential, calc.eta, energy)
                assert float(np.ptp(expected)) > 0.0, (
                    f"the reference layer is constant at nk={nk}, E={energy}: the "
                    "comparison would be vacuous"
                )
                assert np.array_equal(expected.view(np.uint8), actual.view(np.uint8)), (
                    f"_born_layer drifted from the pre-adoption formula at nk={nk}, "
                    f"num_wann={num_wann}, E={energy}: "
                    f"max|delta| = {float(np.max(np.abs(expected - actual)))}"
                )
                # Control: a layer that drifted by a single ulp must NOT compare
                # equal, otherwise this check could not detect the drift it exists
                # for (a re-typed constant such as 1/π = 1/3.14159265 moves every
                # element of the layer).
                assert not np.array_equal(
                    np.nextafter(expected, np.inf).view(np.uint8), actual.view(np.uint8)
                ), "the comparison cannot detect a one-ulp drift"
                compared += 1

    assert compared > 0, "check (f) compared nothing"
    print(
        f"  [f] the shared layer function still matches the pre-refactor serial "
        f"formula bit for bit and the serial loop and the worker both call it "
        f"({compared} E x V x model combinations, one-ulp control rejected)"
    )


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    checks = [
        ("(a) serial == parallel, bit for bit", check_bit_identity),
        ("(b) checkpoint lifecycle", check_checkpoint_lifecycle),
        ("(c) interrupt then resume", check_interrupt_resume),
        ("(d) incompatible shard rejected", check_incompatible_shard),
        ("(e) GPU backend falls back to serial", check_gpu_backend_falls_back),
        ("(f) serial formula frozen", check_serial_formula_frozen),
    ]
    failed = []
    for name, function in checks:
        try:
            function()
            print(f"[PASS] {name}")
        except Exception as exc:
            print(f"[FAIL] {name}: {exc}")
            logger.exception("check %s failed", name)
            failed.append(name)
    print()
    if failed:
        print(f"RESULT: FAILED ({len(failed)} check(s) failed): {', '.join(failed)}")
        raise SystemExit(1)
    print("RESULT: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
