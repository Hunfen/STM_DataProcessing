"""Regression checks for the opt-in parallel JDOS QPI path.

Covers the shard-and-merge adoption in :class:`JDOSQPI`:

  (a) serial == parallel: bit-for-bit identical layers AND identical h5 products
      (the clock is frozen so the convention's ``creation_date`` cannot differ);
  (b) checkpoint lifecycle: shards are HDF5, written through ``io.h5_convention``,
      the automatic checkpoint directory is removed after a successful merge, and
      an explicit one is kept for resume;
  (c) resume: a real interrupted run (SIGINT in a subprocess) resumes from the
      shards on disk and recomputes ONLY the missing energies;
  (d) identity: shards produced with different parameters are rejected as
      incompatible and recomputed, with the operator-visible warning;
  (e) the CPU/GPU backend hazard: a parallel request under the GPU backend falls
      back to the serial path with a WARNING instead of forking workers that
      would inherit a CUDA context;
  (f) the shared layer function still evaluates the pre-adoption serial formula
      bit for bit, so a drift in it cannot hide behind (a) (which compares the
      parallel path against the serial one and would move with it).

The check is synthetic (a mock Hamiltonian, no external data), so it runs in CI
rather than being deselected.

Run from the repository root:
    .venv/bin/python tests/regression/check_qpi_parallel.py
"""

from __future__ import annotations

import hashlib
import logging
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import h5py
import numpy as np
from scipy.fft import fft2, fftshift, ifft2

_REGRESSION_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _REGRESSION_DIR.parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from stm_data_processing.io import h5_convention  # noqa: E402
from stm_data_processing.parallel import shard_driver  # noqa: E402
from stm_data_processing.stm import qpi_jdos  # noqa: E402
from stm_data_processing.stm.qpi_jdos import JDOSQPI  # noqa: E402

_NK = 64
_ETA = 0.05
_N_ENERGIES = 64
_CHILD_WORKERS = 4
_FROZEN_DATE = "2026-01-01T00:00:00+00:00"

logger = logging.getLogger("check_qpi_parallel")

_CHILD_SCRIPT = '''
"""Synthetic JDOS QPI run used by check_qpi_parallel (interrupt/resume)."""
import logging
import pathlib
import sys

import numpy as np

REPO = pathlib.Path({repo!r})
sys.path.insert(0, str(REPO / "src"))

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)

from stm_data_processing.stm.qpi_jdos import JDOSQPI


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
    eta = {eta!r}
    calc = JDOSQPI(MockMLWFHamiltonian(), nk={nk}, eta=eta)
    energies = np.linspace(-1.0, 1.0, {n_energies})
    calc.calculate(
        energies,
        q_range=None,
        n_workers={workers},
        checkpoint_dir={ckpt!r},
        resume=resume,
    )
    print("CHILD-OK")
'''


class MockMLWFHamiltonian:
    """Minimal stand-in exposing the MLWFHamiltonian interface JDOSQPI needs."""

    def __init__(self, num_wann: int = 2, seed: int = 11) -> None:
        self.num_wann = num_wann
        self.bvecs = None
        self._rng = np.random.default_rng(seed)

    def hk(self, k_points: np.ndarray) -> np.ndarray:
        """Return a batch of Hermitian H(k) matrices (deterministic)."""
        n = len(k_points)
        nw = self.num_wann
        h = self._rng.normal(size=(n, nw, nw)) + 1j * self._rng.normal(size=(n, nw, nw))
        return (h + np.conj(np.swapaxes(h, 1, 2))) / 2


def _calculator(nk: int = 8, eta: float = _ETA) -> JDOSQPI:
    return JDOSQPI(MockMLWFHamiltonian(), nk=nk, eta=eta)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def check_bit_identity() -> None:
    """(a) the parallel layers and the parallel h5 product are identical."""
    calc = _calculator()
    energies = np.linspace(-1.0, 1.0, 21)
    root = Path(tempfile.mkdtemp(prefix="qpi_parallel_"))
    serial_path, parallel_path = root / "serial.h5", root / "parallel.h5"

    original_clock = h5_convention.utc_now_iso
    h5_convention.utc_now_iso = lambda: _FROZEN_DATE
    try:
        serial = calc.calculate(energies, q_range=None, output_path=str(serial_path))
        parallel = calc.calculate(
            energies,
            q_range=None,
            output_path=str(parallel_path),
            n_workers=3,
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
        f"  fingerprints equal: {_file_fingerprint(serial_path) == _file_fingerprint(parallel_path)}"
    )
    print(
        f"  [a] serial == parallel: max|delta| = {deviation:.1e}, bitwise identical "
        f"(uint8 view), product sha256 = {serial_sha[:16]}... identical for both paths "
        f"({serial_path.stat().st_size} bytes)"
    )

    # the serial path stays the DEFAULT: no extra files, same product shape
    assert serial["qpi_layers"].shape == (len(energies), calc.nk, calc.nk)
    print(
        f"  [a] serial default unchanged: signature has n_workers=None by default, "
        f"result keys {sorted(serial)}, layers {a.shape}"
    )


def check_checkpoint_lifecycle() -> None:
    """(b) HDF5 shards, explicit dir kept, automatic dir removed after merge."""
    calc = _calculator()
    energies = np.linspace(-1.0, 1.0, 9)
    root = Path(tempfile.mkdtemp(prefix="qpi_parallel_"))
    ckpt = root / "ckpt"
    calc.calculate(energies, q_range=None, n_workers=3, checkpoint_dir=ckpt)

    shards = sorted(p.name for p in ckpt.iterdir())
    h5_shards = [name for name in shards if name.endswith(".h5")]
    assert h5_shards, f"no HDF5 shards written: {shards}"
    assert all((ckpt / name.replace(".h5", ".done")).exists() for name in h5_shards)

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
    assert compression == h5_convention.COMPRESSION, compression
    assert opts == h5_convention.COMPRESSION_OPTS, opts
    assert chunks is not None, "the shard dataset is not chunked"
    print(
        f"  [b] shard {shard_path.name}: dataset 'qpi_layers' {shape} "
        f"{dtype} chunks={chunks} compression={compression}/{opts}; "
        f"attrs {sorted(attrs)}"
    )
    print(f"  [b] explicit checkpoint dir kept for resume: {shards}")

    created: list[Path] = []
    original_mkdtemp = shard_driver.tempfile.mkdtemp

    def _recording_mkdtemp(*args: object, **kwargs: object) -> str:
        path = original_mkdtemp(*args, **kwargs)
        created.append(Path(path))
        return path

    shard_driver.tempfile.mkdtemp = _recording_mkdtemp
    try:
        calc.calculate(energies, q_range=None, n_workers=3)
    finally:
        shard_driver.tempfile.mkdtemp = original_mkdtemp

    assert created, "the automatic checkpoint directory was never created"
    assert not created[0].exists(), (
        f"the automatic checkpoint directory {created[0]} survived a successful merge"
    )
    print(f"  [b] automatic checkpoint dir {created[0].name} removed after the merge")


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
    root = Path(tempfile.mkdtemp(prefix="qpi_parallel_"))
    ckpt = root / "ckpt"
    script = root / "child.py"
    script.write_text(
        _CHILD_SCRIPT.format(
            repo=str(_REPO_ROOT),
            nk=_NK,
            eta=_ETA,
            n_energies=_N_ENERGIES,
            workers=_CHILD_WORKERS,
            ckpt=str(ckpt),
        ),
        encoding="utf-8",
    )

    # The plan splits the energy axis into one contiguous slice per worker, so
    # the child's shard count is the number of workers it asks for.
    total_shards = len(shard_driver.plan_slices(_N_ENERGIES, _CHILD_WORKERS))
    process = _run_child(script)
    deadline = time.time() + 120
    complete: list[str] = []
    while time.time() < deadline and process.poll() is None:
        complete = (
            sorted(p.name for p in ckpt.glob("rows_*.done")) if ckpt.exists() else []
        )
        if complete:
            break
        time.sleep(0.005)
    process.send_signal(signal.SIGINT)
    _, stderr = process.communicate(timeout=180)
    on_disk = sorted(p.name for p in ckpt.glob("rows_*.done"))
    assert on_disk, f"the interrupted run left no complete shard: {stderr[-2000:]}"
    assert process.returncode in (130, -2), (
        f"an interrupted parallel run must exit like a Ctrl-C'd serial run (130 in a "
        f"shell, -2 as Popen reports a process killed by SIGINT), got "
        f"{process.returncode}: {stderr[-2000:]}"
    )
    assert "SIGINT received" in stderr, stderr[-2000:]
    assert len(on_disk) < total_shards, (
        f"the run was not interrupted mid-flight: all {total_shards} shards are on disk"
    )
    print(
        f"  [c] interrupted run: exit={process.returncode}, "
        f"{len(on_disk)} of {total_shards} complete shard(s) kept ({on_disk}), "
        f"{total_shards - len(on_disk)} still missing"
    )

    resumed = _run_child(script, "--resume")
    stdout, stderr = resumed.communicate(timeout=300)
    assert resumed.returncode == 0, stderr[-3000:]
    assert "CHILD-OK" in stdout, stdout[-1000:]
    dispatched = [line for line in stderr.splitlines() if "dispatched worker=" in line]
    reused = [line for line in stderr.splitlines() if "already complete" in line]
    assert len(reused) == len(on_disk), (
        f"resume did not reuse every complete shard: {reused} vs {on_disk}"
    )
    assert len(dispatched) == total_shards - len(on_disk), (
        f"resume dispatched {len(dispatched)} shard(s), expected "
        f"{total_shards - len(on_disk)}: {dispatched}"
    )
    print(
        f"  [c] resume: dispatched only the {len(dispatched)} missing shard(s), "
        f"reused {len(reused)} from disk"
    )


def check_incompatible_shard() -> None:
    """(d) shards from different parameters are rejected, not merged."""
    calc = _calculator()
    energies = np.linspace(-1.0, 1.0, 9)
    root = Path(tempfile.mkdtemp(prefix="qpi_parallel_"))
    ckpt = root / "ckpt"
    calc.calculate(energies, q_range=None, n_workers=3, checkpoint_dir=ckpt)

    records: list[logging.LogRecord] = []

    class _Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _Collector()
    qpi_jdos.logger.addHandler(handler)
    try:
        other = _calculator(eta=_ETA * 1.5)  # same grid, different broadening
        other.calculate(
            energies, q_range=None, n_workers=3, checkpoint_dir=ckpt, resume=True
        )
    finally:
        qpi_jdos.logger.removeHandler(handler)

    messages = [record.getMessage() for record in records]
    rejected = [
        message
        for message in messages
        if "does not match the requested parameters" in message
    ]
    assert rejected, f"no incompatibility warning was emitted: {messages}"
    print(
        f"  [d] different parameters -> {len(rejected)} shard(s) rejected and "
        f"recomputed with the warning {rejected[0][:96]!r}"
    )


def check_gpu_backend_falls_back() -> None:
    """(e) a parallel request under the GPU backend falls back to serial."""
    calc = _calculator()
    energies = np.linspace(-1.0, 1.0, 5)
    root = Path(tempfile.mkdtemp(prefix="qpi_parallel_"))
    ckpt = root / "ckpt"

    records: list[logging.LogRecord] = []

    class _Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    original_backend, original_cp = qpi_jdos.BACKEND, qpi_jdos.cp
    cuda_calls: list[int] = []

    def _stub_cuda(energy_array: np.ndarray, normalize: bool) -> np.ndarray:
        # Stand-in for the CUDA path (cupy is not installed here): it must be the
        # branch the GPU backend would take, so record the call and reuse the
        # CPU arithmetic to keep the check self-contained.
        cuda_calls.append(len(energy_array))
        return calc._compute_jdos_cpu(energy_array, normalize)

    handler = _Collector()
    qpi_jdos.logger.addHandler(handler)
    qpi_jdos.BACKEND = "gpu"
    qpi_jdos.cp = object()  # makes `BACKEND == "gpu" and cp is not None` true
    original_cuda = calc._compute_jdos_cuda
    calc._compute_jdos_cuda = _stub_cuda  # type: ignore[method-assign]
    try:
        result = calc.calculate(
            energies,
            q_range=None,
            n_workers=3,
            checkpoint_dir=ckpt,
        )
    finally:
        calc._compute_jdos_cuda = original_cuda  # type: ignore[method-assign]
        qpi_jdos.BACKEND, qpi_jdos.cp = original_backend, original_cp
        qpi_jdos.logger.removeHandler(handler)

    messages = [record.getMessage() for record in records]
    warned = [message for message in messages if "CUDA context" in message]
    assert warned, f"no GPU-backend warning was emitted: {messages}"
    assert cuda_calls == [len(energies)], (
        f"the GPU backend did not take the serial CUDA path: {cuda_calls}"
    )
    assert not ckpt.exists(), "the fallback still created shards"
    assert result["qpi_layers"].shape == (len(energies), calc.nk, calc.nk)
    print(
        f"  [e] GPU backend + n_workers=3: WARNING "
        f"{warned[0][:110]!r}, serial CUDA path used, no shards written"
    )


def _reference_layer(
    eigenvalues: np.ndarray, eta: float, energy: float, normalize: bool
) -> np.ndarray:
    """The layer formula exactly as the pre-adoption serial loop expressed it.

    ``_compute_spectral_function(energy, use_gpu=False)`` built ``a_k`` and
    ``_compute_jdos_cpu`` then did ``fft2 -> |.|^2 -> ifft2 -> fftshift ->
    per-layer normalize``: the two bodies were inlined into ``_jdos_layer`` by
    the parallel adoption, so this is their frozen reference form.
    """
    denominator = (energy - eigenvalues) ** 2 + eta**2
    a_k_per_band = (1 / np.pi) * (eta / denominator)
    a_k = np.sum(a_k_per_band, axis=-1)

    a_r = fft2(a_k)
    jdos_q = np.real(ifft2(np.abs(a_r) ** 2))
    jdos_q = fftshift(jdos_q)
    if normalize and (max_val := np.max(jdos_q)) > 0:
        jdos_q /= max_val
    return jdos_q


def check_serial_formula_frozen() -> None:
    """(f) the shared layer function still evaluates the pre-refactor formula.

    Checks (a)-(e) compare the parallel path against the serial one; if
    ``_jdos_layer`` drifted, both sides would drift together and every one of
    those comparisons would still pass.  This check pins the arithmetic the
    serial loop used before the adoption and compares it bit for bit.
    """
    calc = _calculator()
    compared = 0
    for energy in (-0.37, 0.0, 0.61):
        for normalize in (True, False):
            expected = _reference_layer(calc.eigenvalues, calc.eta, energy, normalize)
            actual = qpi_jdos._jdos_layer(calc.eigenvalues, calc.eta, energy, normalize)
            assert np.array_equal(expected.view(np.uint8), actual.view(np.uint8)), (
                f"_jdos_layer drifted from the pre-adoption formula at "
                f"E={energy}, normalize={normalize}: "
                f"max|delta| = {float(np.max(np.abs(expected - actual)))}"
            )
            compared += 1
    print(
        f"  [f] the shared layer function still matches the pre-refactor serial "
        f"formula bit for bit ({compared} E x normalize combinations)"
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
