import hashlib
import logging
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from scipy.fft import fft2, fftshift, ifft2

from stm_data_processing.config import BACKEND
from stm_data_processing.dft.wannier90.mlwf_ek2d import EK2DCalculator
from stm_data_processing.dft.wannier90.mlwf_hamiltonian import MLWFHamiltonian
from stm_data_processing.io.qpi_io import save_qpi_to_h5
from stm_data_processing.parallel.shard_driver import (
    RunStats,
    ShardSpec,
    ShardStore,
    available_memory_bytes,
    log_plan,
    plan_slices,
    run_shards,
)
from stm_data_processing.utils.miscellaneous import extend_qpi, frac_to_real_2d, k_to_q

if BACKEND == "gpu":
    import cupy as cp
else:
    cp = None

logger = logging.getLogger(__name__)

#: Log label of the parallel path and the ``generator`` of its shards.
_QPI_LABEL = "JDOSQPI"
_ARRAY_KEYS = ("qpi_layers",)


def _jdos_layer(
    eigenvalues: np.ndarray,
    eta: float,
    energy: float,
    normalize: bool,
) -> np.ndarray:
    """One energy layer: ``A(k, E) -> |FFT|^2 -> fftshift -> per-layer normalize``.

    Shared by the serial loop and the parallel worker so the two paths cannot
    drift apart.  The normalization is per layer (each layer is divided by its
    own maximum), which is exactly why slicing the energy axis is an exact
    decomposition: no layer depends on any other one.
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


def _qpi_jdos_worker(
    worker_id: int,
    key: tuple[int, int],
    store: ShardStore,
    payload: dict[str, Any],
    result_queue: Any,
    progress_queue: Any,
) -> None:
    """Compute the QPI layers of ONE energy shard and land its checkpoint.

    Module level and restricted to plain array/dict arguments so it survives the
    ``spawn`` start method.  The shard key is the half-open energy index range
    ``(start, stop)``; the layer arithmetic is :func:`_jdos_layer`, the same
    function the serial loop calls.
    """
    started = time.perf_counter()
    energies = payload["energies"]
    eigenvalues = payload["eigenvalues"]
    eta = float(payload["eta"])
    normalize = bool(payload["normalize"])
    start, stop = int(key[0]), int(key[1])

    try:
        layers = np.empty((stop - start, *eigenvalues.shape[:2]))
        for index, energy in enumerate(energies[start:stop]):
            layers[index] = _jdos_layer(eigenvalues, eta, energy, normalize)
            progress_queue.put((worker_id, (index + 1, stop - start, index + 1)))

        store.write(key, {"qpi_layers": layers}, dict(payload["meta"]))
    except BaseException:
        result_queue.put(
            {
                "status": "error",
                "worker_id": worker_id,
                "row_range": [start, stop],
                "error": traceback.format_exc(),
            }
        )
        raise

    result_queue.put(
        {
            "status": "ok",
            "worker_id": worker_id,
            "row_range": [start, stop],
            "elapsed_s": time.perf_counter() - started,
        }
    )


class JDOSQPI:
    """Class for calculating JDOS QPI (Quasiparticle Interference).

    This class uses MLWFHamiltonian to compute eigenvalues internally.
    k-mesh is fixed to [-0.5, 0.5) range.
    """

    _CUDA_SAFETY_FRACTION: float = 0.75
    _CUDA_HARD_MAX_BATCH: int = 32

    def __init__(
        self,
        hamiltonian: MLWFHamiltonian,
        nk: int = 256,
        eta: float = 0.001,
    ) -> None:
        """Initialize QPI calculator.

        Parameters
        ----------
        hamiltonian : MLWFHamiltonian
            Initialized Hamiltonian object.
        nk : int
            Number of k-points in each dimension (nkx = nky = nk).
        eta : float
            Spectral broadening parameter, default is 0.001

        """
        self._validate_hamiltonian(hamiltonian)
        self.ham: MLWFHamiltonian = hamiltonian
        self.num_wann: int = hamiltonian.num_wann
        self.nk: int = int(nk)
        self.eta: float = float(eta)

        k_vals = np.linspace(-0.5, 0.5, self.nk, endpoint=False)
        self.k1_grid, self.k2_grid = np.meshgrid(k_vals, k_vals, indexing="ij")
        self.q1_grid, self.q2_grid = self.k1_grid.copy(), self.k2_grid.copy()

        # Compute eigenvalues using EK2DCalculator
        self.eigenvalues = self._initialize_eigenvalues()
        logger.info("[JDOSQPI] Diagonalization completed.")

    def _validate_hamiltonian(self, hamiltonian: MLWFHamiltonian) -> None:
        """Validate the MLWFHamiltonian object.

        Parameters
        ----------
        hamiltonian : MLWFHamiltonian
            Hamiltonian object to validate.

        Raises
        ------
        ValueError
            If the Hamiltonian is not properly initialized.
        """
        if not hasattr(hamiltonian, "num_wann") or hamiltonian.num_wann is None:
            raise ValueError("Invalid MLWFHamiltonian: num_wann is not initialized.")
        if hamiltonian.num_wann <= 0:
            raise ValueError(
                f"Invalid MLWFHamiltonian: num_wann must be positive, "
                f"got {hamiltonian.num_wann}."
            )

    def _initialize_eigenvalues(
        self,
    ) -> np.ndarray:
        """Diagonalize Hamiltonian using EK2DCalculator on CPU or GPU.

        Returns
        -------
        np.ndarray
            Eigenvalues on CPU (nk, nk, num_wann). Always returned as numpy array.
        """
        logger.info(
            f"[JDOSQPI] Diagonalizing Hamiltonian on {self.nk}x{self.nk} grid on {BACKEND}...",
        )

        ek2d_calc = EK2DCalculator(self.ham, nk=self.nk)

        if BACKEND == "gpu" and cp is not None:
            # Compute on GPU, then convert to CPU for storage.
            # _compute_eigen_cuda returns (evals, evecs) with evals shape
            # (nk*nk, num_wann); reshape to the (nk, nk, num_wann) layout used
            # by _compute_spectral_function (bands on the last axis).
            e_gpu, _ = ek2d_calc._compute_eigen_cuda(self.ham)
            e_gpu = cp.reshape(e_gpu, (self.nk, self.nk, self.num_wann))
            e_cpu = cp.asnumpy(e_gpu)
            return e_cpu
        else:
            # _compute_eigen returns (evals, evecs) with evals shape
            # (nk*nk, num_wann); reshape to (nk, nk, num_wann).
            e_cpu, _ = ek2d_calc._compute_eigen(self.ham)
            e_cpu = np.ascontiguousarray(
                np.reshape(e_cpu, (self.nk, self.nk, self.num_wann))
            )
            return e_cpu

    def _compute_spectral_function(
        self,
        energy: Any,
        use_gpu: bool = False,
    ) -> np.ndarray | Any:
        """Calculate the spectral function A(k, E) on CPU or GPU."""
        xp = cp if use_gpu and cp is not None else np
        e_k = self.eigenvalues

        # Convert to GPU array if needed
        if use_gpu and cp is not None:
            e_k = cp.asarray(e_k)

        energy = xp.asarray(energy)

        # e_k shape: (nk, nk, num_wann)
        if energy.ndim == 0:
            # Scalar energy: (nk, nk, num_wann)
            denominator = (energy - e_k) ** 2 + self.eta**2
            a_k_per_band = (1 / xp.pi) * (self.eta / denominator)
            a_k = xp.sum(a_k_per_band, axis=-1)
        else:
            # Array energy: (n_energies, nk, nk, num_wann)
            denominator = (
                energy[:, None, None, None] - e_k[None, :, :, :]
            ) ** 2 + self.eta**2
            a_k_per_band = (1 / xp.pi) * (self.eta / denominator)
            a_k = xp.sum(a_k_per_band, axis=3)
        return a_k

    def _compute_jdos_cpu(
        self,
        energy_array: np.ndarray,
        normalize: bool = True,
    ) -> np.ndarray:
        """Calculate JDOS-based QPI pattern using CPU (NumPy)."""
        nkx, nky = self.k1_grid.shape
        qpi_layers = np.empty((len(energy_array), nkx, nky))

        logger.info(
            "  [CPU] Computing QPI for %d energies on %dx%d grid...",
            len(energy_array),
            nkx,
            nky,
        )

        for i, energy in enumerate(energy_array):
            jdos_q = _jdos_layer(self.eigenvalues, self.eta, energy, normalize)

            qpi_layers[i] = jdos_q

            if nkx >= 64 and (i + 1) % max(len(energy_array) // 10, 1) == 0:
                percent = round(100 * (i + 1) / len(energy_array))
                logger.info(
                    "    Progress: %3d%% (%d/%d)", percent, i + 1, len(energy_array)
                )

        return qpi_layers

    def _compute_jdos_parallel(
        self,
        energy_array: np.ndarray,
        normalize: bool,
        n_workers: int,
        checkpoint_dir: str | Path | None,
        resume: bool,
    ) -> np.ndarray:
        """Same layers as :meth:`_compute_jdos_cpu`, computed by shard workers.

        The energy axis is the shard key: each worker computes a contiguous
        index range with :func:`_jdos_layer` and lands it as an HDF5 shard
        through :class:`~stm_data_processing.parallel.shard_driver.ShardStore`;
        the parent concatenates the shards **in plan order**, which is the
        serial order.  Everything else (resume, the memory guard, the shard
        teardown after a successful merge, the incompatibility warning) comes
        from :func:`~stm_data_processing.parallel.shard_driver.run_shards`.
        """
        n_energies = len(energy_array)
        slices = plan_slices(n_energies, n_workers)

        signature = {
            "nk": self.nk,
            "eta": self.eta,
            "normalize": bool(normalize),
            "num_wann": self.num_wann,
            "n_energies": n_energies,
            # The eigenvalues are what every layer is built from, so a shard
            # from another model (or another k-grid) must never be reused.
            "eigenvalues_sha256": hashlib.sha256(
                np.ascontiguousarray(self.eigenvalues, dtype=np.float64).tobytes()
            ).hexdigest(),
        }

        # Memory bound: a worker holds the eigenvalue array plus its layers
        # (`A(k)`, `|FFT|^2` and the output layer are all nk x nk float64).
        per_layer_bytes = 3 * self.nk * self.nk * 8
        max_per_worker = max(stop - start for start, stop in slices)
        per_worker_bytes = int(
            self.eigenvalues.nbytes + max_per_worker * per_layer_bytes
        )

        meta = {
            "module_type": "jdos",
            "nq": self.nk,
            "num_wann": self.num_wann,
            **signature,
        }
        store = ShardStore(
            checkpoint_dir if checkpoint_dir is not None else Path(),
            array_keys=_ARRAY_KEYS,
            identity=signature,
            generator=_QPI_LABEL,
            label=_QPI_LABEL,
            logger=logger,
        )
        payload = {
            "energies": np.asarray(energy_array, dtype=np.float64),
            "eigenvalues": self.eigenvalues,
            "eta": self.eta,
            "normalize": bool(normalize),
            "meta": meta,
        }
        merged: list[np.ndarray] = []

        def _finalize(
            loaded: list[dict[str, Any]],
            reports: Any,
            stats: RunStats,
        ) -> int:
            layers = [
                np.asarray(shard["arrays"]["qpi_layers"], dtype=np.float64)
                for shard in loaded
            ]
            total = sum(layer.shape[0] for layer in layers)
            if total != n_energies:
                logger.error(
                    "[%s] merge found %d energy layers, expected %d",
                    _QPI_LABEL,
                    total,
                    n_energies,
                )
                return 6
            merged.append(np.concatenate(layers, axis=0))
            logger.info(
                "[%s] merged %d shard(s): %d energies in %.3f s",
                _QPI_LABEL,
                len(loaded),
                total,
                stats.elapsed_s,
            )
            return 0

        def _worker_process(
            context: Any,
            index: int,
            key: tuple[int, int],
            worker_payload: dict[str, Any],
            result_queue: Any,
            progress_queue: Any,
        ) -> Any:
            return context.Process(
                target=_qpi_jdos_worker,
                name=f"jdos-worker-{index}",
                args=(index, key, store, worker_payload, result_queue, progress_queue),
            )

        spec = ShardSpec(
            label=_QPI_LABEL,
            store=store,
            plan=lambda: plan_slices(n_energies, n_workers),
            progress_units=lambda key: (key[1] - key[0], key[1] - key[0]),
            worker_process=_worker_process,
            worker_payload=lambda: payload,
            log_plan=lambda keys, per_worker: log_plan(
                keys,
                self.nk,
                False,
                per_worker,
                available_memory_bytes(),
                None,
                label=_QPI_LABEL,
            ),
            log_progress=lambda stats: logger.info(
                "[%s] progress: %d/%d energies, %d worker(s) live",
                _QPI_LABEL,
                stats.rows_done,
                stats.rows_total,
                stats.live_workers,
            ),
            describe_plan=lambda keys: (
                f"DRY-RUN workers={len(keys)} nk={self.nk} n_energies={n_energies} "
                "segments=" + ",".join(f"[{start},{stop})" for start, stop in keys)
            ),
            finalize=_finalize,
            logger=logger,
            per_worker_bytes=per_worker_bytes,
        )

        exit_code = run_shards(
            spec,
            n_workers=n_workers,
            checkpoint_dir=checkpoint_dir,
            resume=resume,
        )
        if exit_code == 130:
            # The driver caught SIGINT and kept the completed checkpoints.  The
            # serial loop would simply propagate KeyboardInterrupt to the caller,
            # so the parallel path must not turn an operator's Ctrl-C into a
            # reported failure.
            kept = checkpoint_dir if checkpoint_dir is not None else "the automatic dir"
            raise KeyboardInterrupt(
                f"parallel JDOS QPI run interrupted; completed shards kept in {kept}"
            )
        if exit_code != 0 or not merged:
            raise RuntimeError(
                f"parallel JDOS QPI run failed with exit code {exit_code}"
            )
        return merged[0]

    def _compute_jdos_cuda(
        self,
        energy_array: np.ndarray,
        normalize: bool = True,
    ) -> np.ndarray:
        """Calculate JDOS-based QPI pattern using GPU (CuPy)."""
        if cp is None:
            raise RuntimeError("CuPy not available but GPU backend requested.")

        start_time = time.time()

        energy_array_gpu = cp.asarray(energy_array)

        nkx, nky = self.k1_grid.shape
        n_energies = energy_array_gpu.size

        # === Batch size determination ===
        e_k_bytes = self.eigenvalues.nbytes
        mem_per_energy = e_k_bytes * 5
        free_mem = cp.cuda.Device().mem_info[0]
        batch_size = max(1, int(free_mem * self._CUDA_SAFETY_FRACTION / mem_per_energy))
        batch_size = min(batch_size, self._CUDA_HARD_MAX_BATCH)
        batch_size = min(batch_size, n_energies)

        if nkx >= 1024 and nky >= 1024:
            batch_size = min(batch_size, 8)
        elif nkx >= 512 and nky >= 512:
            batch_size = min(batch_size, 16)

        num_batches = (n_energies + batch_size - 1) // batch_size

        logger.info(
            "  [CUDA] Computing QPI for %d energies on %dx%d grid...",
            n_energies,
            nkx,
            nky,
        )
        logger.info(
            "  [CUDA] Using batch size: %d (total batches: %d)",
            batch_size,
            num_batches,
        )

        qpi_layers_gpu = cp.zeros((n_energies, nkx, nky), dtype=cp.float64)

        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, n_energies)
            batch_energies = energy_array_gpu[start:end]

            a_k_batch = self._compute_spectral_function(batch_energies, use_gpu=True)

            a_r = cp.fft.fft2(a_k_batch, axes=(-2, -1))
            jdos_batch = cp.real(cp.fft.ifft2(cp.abs(a_r) ** 2, axes=(-2, -1)))
            jdos_batch = cp.fft.fftshift(jdos_batch, axes=(-2, -1))

            if normalize:
                max_vals = cp.max(jdos_batch, axis=(-2, -1), keepdims=True)
                jdos_batch = cp.where(max_vals > 0, jdos_batch / max_vals, jdos_batch)

            qpi_layers_gpu[start:end] = jdos_batch

            del a_k_batch, a_r, jdos_batch
            cp.get_default_memory_pool().free_all_blocks()

            progress = (batch_idx + 1) / num_batches * 100
            free_mem, total_mem = cp.cuda.Device().mem_info
            used_mem = total_mem - free_mem
            logger.info(
                "    Progress: %3d%% (Batch %d/%d, GPU mem: %.2f/%.2f GB)",
                progress,
                batch_idx + 1,
                num_batches,
                used_mem / 1e9,
                total_mem / 1e9,
            )

        qpi_layers = cp.asnumpy(qpi_layers_gpu)

        elapsed_time = time.time() - start_time
        logger.info(
            "  [CUDA] Done in %.2f s (%.2f ms/energy)",
            elapsed_time,
            elapsed_time / n_energies * 1000,
        )

        return qpi_layers

    def calculate(
        self,
        energy_range: float | np.ndarray | list[float],
        q_range: tuple[float, float] | None = (-0.5, 0.5),
        normalize: bool = True,
        output_path: str | None = None,
        n_workers: int | None = None,
        checkpoint_dir: str | Path | None = None,
        resume: bool = False,
    ) -> dict[str, Any]:
        """Unified QPI calculator with automatic CPU/GPU selection.

        Parameters
        ----------
        energy_range : float, list[float], or np.ndarray
            Target energy or energies (in eV) at which to compute the QPI.
        q_range : tuple[float, float] or None, optional (default=(-0.5, 0.5))
            Dimensionless q-range [q_min, q_max] for cropping the QPI result.
            If None, no cropping is applied.
        normalize : bool, default True
            Whether to normalize the QPI intensity to unit maximum.
        output_path : str, optional
            Path to save output to HDF5.
        n_workers : int or None, optional (default None = serial)
            OPT-IN parallel path: split the energy axis into that many shards,
            compute each in its own worker process and merge them in plan order
            through :mod:`stm_data_processing.parallel.shard_driver`.  The result
            is bit-identical to the serial path (every layer is independent and
            normalized on its own), so this only trades wall time for extra
            processes and per-shard checkpoint files.
        checkpoint_dir : str or Path or None, optional
            Where the parallel path keeps its shards.  ``None`` uses a temporary
            directory that is removed after a successful merge; an explicit
            directory is kept so a later ``resume=True`` run can reuse it.
        resume : bool, default False
            With ``n_workers`` and an explicit ``checkpoint_dir``: only the
            shards that are missing or were produced with different parameters
            (different ``nk``/``eta``/``normalize``/eigenvalues) are recomputed.

        Raises
        ------
        KeyboardInterrupt
            If the parallel run is interrupted (Ctrl-C): the workers are
            terminated, the shards already completed stay on disk and the next
            ``resume=True`` run recomputes only the rest.
        RuntimeError
            If a parallel worker fails or a shard is missing after the merge.

        Returns
        -------
        dict[str, Any]
            Dictionary containing:
            - 'qpi_layers': QPI intensity array (optionally extended)
            - 'q1_grid', 'q2_grid': Fractional q-grids (optionally extended)
            - 'qx_grid', 'qy_grid': Real-space q-grids (if bvecs available)
            - 'metadata': Dict containing calculation parameters and bvecs

        """
        nk = self.nk
        energy_array: np.ndarray = np.asarray(energy_range, dtype=np.float64).ravel()

        logger.info(
            "[INFO] Starting QPI calculation on %s (nk=%d, nω=%d)",
            BACKEND,
            nk,
            len(energy_array),
        )

        use_gpu = BACKEND == "gpu" and cp is not None
        parallel = n_workers is not None and int(n_workers) > 1
        if parallel and use_gpu:
            logger.warning(
                "[JDOSQPI] n_workers=%s requested while the GPU backend is active: "
                "spawned workers would inherit a CUDA context, which is unsafe, so "
                "this run falls back to the serial CUDA path",
                n_workers,
            )
            parallel = False
        if parallel and len(energy_array) < 2:
            logger.warning(
                "[JDOSQPI] n_workers=%s requested for a single energy: using the "
                "serial path",
                n_workers,
            )
            parallel = False

        if parallel:
            qpi_layers = self._compute_jdos_parallel(
                energy_array,
                normalize,
                int(n_workers),
                checkpoint_dir,
                resume,
            )
        elif use_gpu:
            qpi_layers = self._compute_jdos_cuda(energy_array, normalize)
        else:
            qpi_layers = self._compute_jdos_cpu(energy_array, normalize)

        # Compute original grids for saving (before extension)
        q1_grid_orig, q2_grid_orig = k_to_q(self.k1_grid, self.k2_grid)

        if output_path is not None:
            logger.info("Saving QPI to %s", output_path)
            save_qpi_to_h5(
                qpi_layers=qpi_layers,
                output_path=output_path,
                energy_range=energy_array,
                module_type="jdos",
                bvecs=self.ham.bvecs,
                eta=self.eta,
                normalize=normalize,
                V=None,
                nq=self.nk,
            )

        # Apply extension/cropping for return value
        if q_range is not None:
            qpi_layers, q1_grid, q2_grid = extend_qpi(
                qpi_layers, q1_grid_orig, q2_grid_orig, q_range[0], q_range[1]
            )
        else:
            q1_grid, q2_grid = q1_grid_orig, q2_grid_orig

        qx_grid, qy_grid = frac_to_real_2d(q1_grid, q2_grid, self.ham.bvecs)

        # Construct metadata to match load_qpi_from_h5 structure
        metadata = {
            "module_type": "jdos",
            "eta": self.eta,
            "normalize": normalize,
            "nq": nk,
            "energy_range": energy_array,
            "bands": None,
            "bvecs": self.ham.bvecs,
            "V": None,
            "mask": None,
        }

        result: dict[str, Any] = {
            "qpi_layers": qpi_layers,
            "q1_grid": q1_grid,
            "q2_grid": q2_grid,
            "qx_grid": qx_grid,
            "qy_grid": qy_grid,
            "metadata": metadata,
        }

        logger.info("QPI calculation completed.")

        return result
