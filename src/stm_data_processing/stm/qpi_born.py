import hashlib
import logging
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from stm_data_processing.config import BACKEND
from stm_data_processing.dft.wannier90.mlwf_gk import GreenFunction
from stm_data_processing.dft.wannier90.mlwf_hamiltonian import MLWFHamiltonian
from stm_data_processing.io.qpi_io import frac_to_real_2d, save_qpi_to_h5
from stm_data_processing.parallel.shard_driver import (
    RunStats,
    ShardSpec,
    ShardStore,
    available_memory_bytes,
    log_plan,
    plan_slices,
    run_shards,
)
from stm_data_processing.utils.miscellaneous import extend_qpi

if BACKEND == "gpu":
    import cupy as cp
else:
    cp = None

logger = logging.getLogger(__name__)

#: Log label of the parallel path and the ``generator`` of its shards.
_QPI_LABEL = "BornQPI"
_ARRAY_KEYS = ("qpi_layers",)


def _born_layer(
    hk_grid: np.ndarray,
    V: np.ndarray,
    eta: float,
    omega: float,
) -> np.ndarray:
    """One energy layer: ``G0(k, ω) -> Σ_k Tr[G0(k) V G0(k+q)] -> fftshift / π``.

    Shared by the serial CPU loop and the parallel worker so the two paths cannot
    drift apart.  Nothing here reads another energy's layer - the ``1/π`` factor
    is applied per layer and no quantity is normalized across energies - which is
    exactly why slicing the energy axis is an exact decomposition and why the
    parallel result is bit-identical to the serial one.

    The Green's-function arithmetic duplicates
    :meth:`~stm_data_processing.dft.wannier90.mlwf_gk.GreenFunction.compute_green`
    on :mod:`numpy` (the worker holds the Hamiltonian grid, not the
    Hamiltonian object, and the parallel path is CPU-only by construction).
    """
    nk, num_wann = hk_grid.shape[0], hk_grid.shape[-1]
    z = float(omega) + 1j * float(eta)
    eye = np.eye(num_wann, dtype=np.complex128)
    eye_batch = np.broadcast_to(eye, hk_grid.shape)
    g0_k = np.linalg.solve(z * eye_batch - hk_grid, eye_batch)  # (nk, nk, nw, nw)

    # Precompute GV[k,a,c] = Σ_b G0[k,a,b] · V[b,c]
    gv = np.einsum("ijab,bc->ijac", g0_k, V, optimize=True)

    # FFT-based correlation theorem: computes all nk*nk q-points at once
    # instead of looping over q (the O(nk^2) Python loop this replaces).
    #
    # s(q) = Σ_k Tr[G0(k) · V · G0(k+q)]
    #      = Σ_{a,c} Σ_k GV[k,a,c] · G0[k+q,c,a]
    #
    # For complex fields the correct no-conjugate formula is (same as the
    # CUDA path): IFFT( FFT(A*)* · FFT(B) ) = Σ_k A[k] · B[k+q] with
    # A = GV[:, :, a, c] and B = G0[:, :, c, a] (verified against the
    # direct np.roll + einsum reference to machine precision).
    #
    # Loop over the c orbital index only for memory efficiency; inside each
    # iteration all q-points and all a orbitals are vectorized.
    qpi: np.ndarray = np.zeros((nk, nk), dtype=np.float64)
    for c in range(num_wann):
        fft_gv_c = np.conj(np.fft.fftn(np.conj(gv[:, :, :, c]), axes=(0, 1)))
        fft_g0_c = np.fft.fftn(g0_k[:, :, c, :], axes=(0, 1))
        corr_c = np.fft.ifftn(fft_gv_c * fft_g0_c, axes=(0, 1))
        qpi += -np.imag(np.sum(corr_c, axis=2))

    return np.fft.fftshift(qpi) / np.pi


def _qpi_born_worker(
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
    ``(start, stop)``; the layer arithmetic is :func:`_born_layer`, the same
    function the serial CPU loop calls.
    """
    started = time.perf_counter()
    energies = payload["energies"]
    hk_grid = payload["hk_grid"]
    V = payload["V"]
    eta = float(payload["eta"])
    start, stop = int(key[0]), int(key[1])

    try:
        layers = np.empty((stop - start, *hk_grid.shape[:2]), dtype=np.float64)
        for index, omega in enumerate(energies[start:stop]):
            layers[index] = _born_layer(hk_grid, V, eta, omega)
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


class BornQPI:
    """
    QPI calculator using Born approximation.

    δρ(q,ω) = -(1/π) Im Σ_k Tr[G0(k,ω) V G0(k+q,ω)]

    Notes
    -----
    - CPU path computes directly in NumPy.
    - CUDA path keeps G0(k) on GPU and evaluates q-points in blocks.
    - The order in calculate_qpi() is intentionally kept as:
        1) save_qpi_to_h5
        2) extend_qpi
        3) frac_to_real_2d
    """

    _CUDA_SAFETY_FRACTION: float = 0.25
    _CUDA_HARD_MAX_BLOCK: int = 64

    def __init__(
        self,
        hamiltonian: MLWFHamiltonian,
        nk: int = 256,
        eta: float = 0.005,
    ) -> None:
        self._validate_hamiltonian(hamiltonian)
        self.ham: MLWFHamiltonian = hamiltonian
        self.num_wann: int | None = hamiltonian.num_wann
        self.nk: int = int(nk)
        self.eta: float = float(eta)
        self.gf: GreenFunction = GreenFunction(hamiltonian, eta=eta)

        self.V: np.ndarray = np.eye(self.num_wann, dtype=np.complex128)

        k_vals = np.linspace(-0.5, 0.5, self.nk, endpoint=False)
        self.k1_grid, self.k2_grid = np.meshgrid(k_vals, k_vals, indexing="ij")
        self.q1_grid, self.q2_grid = self.k1_grid.copy(), self.k2_grid.copy()

        k_points = np.column_stack(
            [
                self.k1_grid.ravel(),
                self.k2_grid.ravel(),
                np.zeros(self.nk * self.nk, dtype=np.float64),
            ]
        )

        if BACKEND == "gpu":
            self.hk_grid: cp.ndarray = cp.asarray(
                self.ham.hk(k_points).reshape(
                    self.nk, self.nk, self.num_wann, self.num_wann
                )
            )
            self.V_gpu: cp.ndarray = cp.asarray(self.V, dtype=cp.complex128)
            self.V_cpu: np.ndarray = self.V
        else:
            self.hk_grid: np.ndarray = self.ham.hk(k_points).reshape(
                self.nk, self.nk, self.num_wann, self.num_wann
            )
            self.V_gpu: cp.ndarray | None = None
            self.V_cpu: np.ndarray = self.V

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

    def _estimate_q_block_size(
        self,
        g0_gpu: Any,
        safety_fraction: float | None = None,
        hard_max_block: int | None = None,
    ) -> int:
        """
        Estimate a reasonable q-block size for CUDA batch evaluation.

        Main temporary tensor in block mode:
            Gq_block ~ (Bq, nk, nk, nw, nw)

        We keep the block conservative because there are additional temporary
        arrays from advanced indexing and einsum.
        """
        if safety_fraction is None:
            safety_fraction = self._CUDA_SAFETY_FRACTION
        if hard_max_block is None:
            hard_max_block = self._CUDA_HARD_MAX_BLOCK

        free_mem, _ = cp.cuda.Device().mem_info
        nk = self.nk
        nw = self.num_wann

        bytes_per_complex = np.dtype(np.complex128).itemsize
        bytes_per_q = nk * nk * nw * nw * bytes_per_complex

        usable = max(int(free_mem * safety_fraction), 1)
        block = max(1, usable // max(bytes_per_q, 1))
        block = min(block, hard_max_block)

        return int(max(1, block))

    # ============================================================
    # Compute Gkq core math (CPU/GPU)
    # ============================================================

    def _compute_Gkq(
        self,
        omega: float,
    ) -> np.ndarray:
        """One energy layer of the serial CPU path (all math in :func:`_born_layer`)."""
        nk = self.nk

        logger.info(f"  [CPU] Computing QPI at ω = {omega:.4f} eV (nk={nk})...")

        layer = _born_layer(self.hk_grid, self.V, self.eta, omega)

        logger.info(f"  [CPU] Done (nk={nk}).")

        return layer

    def _compute_Gkq_cuda(
        self,
        omega: float,
    ) -> np.ndarray:
        """
        CUDA implementation using FFT-based convolution theorem.

        For complex fields, the correct formula for Σ_k A[k] · B[k+q] is:
          IFFT( FFT(A*)* · FFT(B) )

        This avoids the unwanted conjugate that standard cross-correlation introduces.

        Derivation:
          Standard: IFFT(FFT(A)* · FFT(B)) = Σ_k A[k]* · B[k+q]  ← has conjugate
          We need:  Σ_k A[k] · B[k+q]  ← no conjugate
          Solution: Use A* as input: IFFT(FFT(A*)* · FFT(B)) = Σ_k A[k] · B[k+q]  ✓
        """
        nk = self.nk
        nw = self.num_wann

        logger.info(f"  [CUDA] Computing QPI at ω = {omega:.4f} eV (nk={nk})...")

        g0_k_gpu = self.gf.compute_green(self.hk_grid, omega)  # shape: (nk, nk, nw, nw)
        v_gpu = cp.asarray(self.V, dtype=cp.complex128)

        # Precompute GV[k,a,c] = Σ_b G[k,a,b] · V[b,c]
        gv_gpu = cp.einsum("ijab,bc->ijac", g0_k_gpu, v_gpu, optimize=True)

        qpi_gpu = cp.zeros((nk, nk), dtype=cp.float64)

        logger.info("  [CUDA] Using FFT correlation with orbital blocks...")

        for a in range(nw):
            for c in range(nw):
                gv_ac = gv_gpu[:, :, a, c]
                g0_ca = g0_k_gpu[:, :, c, a]

                # FFT with proper conjugate handling for complex fields
                # Formula: IFFT( FFT(A*)* · FFT(B) )
                fft_gv_conj = cp.fft.fftn(cp.conj(gv_ac))
                fft_gv_flipped = cp.conj(fft_gv_conj)
                fft_g0 = cp.fft.fftn(g0_ca)

                corr = cp.fft.ifftn(fft_gv_flipped * fft_g0)
                qpi_gpu += -cp.imag(corr)

                del gv_ac, g0_ca, fft_gv_conj, fft_gv_flipped, fft_g0, corr

        qpi_gpu = cp.fft.fftshift(qpi_gpu)

        logger.info("  [CUDA] Done.")

        # The per-layer 1/π factor lives in the layer function of this module
        # (CPU: _born_layer), so both backends return the finished layer.
        return cp.asnumpy(qpi_gpu / np.pi)

    # ============================================================
    # Public
    # ============================================================

    def _compute_born_parallel(
        self,
        energy_array: np.ndarray,
        n_workers: int,
        checkpoint_dir: str | Path | None,
        resume: bool,
    ) -> np.ndarray:
        """Same layers as :meth:`_compute_Gkq`, computed by shard workers.

        The energy axis is the shard key: each worker computes a contiguous
        index range with :func:`_born_layer` and lands it as an HDF5 shard
        through :class:`~stm_data_processing.parallel.shard_driver.ShardStore`;
        the parent concatenates the shards **in plan order**, which is the
        serial order.  Everything else (resume, the memory guard, the shard
        teardown after a successful merge, the incompatibility warning) comes
        from :func:`~stm_data_processing.parallel.shard_driver.run_shards`.

        The driver's ``start_method='spawn'`` default is kept deliberately: a
        spawned worker starts a fresh interpreter that re-imports this module,
        so it can never inherit a CUDA context, a CuPy memory pool or a live
        ``GreenFunction`` from the parent.  ``fork`` would share all three, so
        the parallel path is restricted to the CPU backend anyway (see
        :meth:`calculate`).
        """
        n_energies = len(energy_array)
        slices = plan_slices(n_energies, n_workers)
        hk_grid = self.hk_grid
        V = self.V

        signature = {
            "nk": self.nk,
            "eta": self.eta,
            "num_wann": self.num_wann,
            "n_energies": n_energies,
            # The Hamiltonian grid carries both the model and the k-grid every
            # layer is built from, so a shard from another model (or another
            # k-grid) must never be reused.
            "hk_grid_sha256": hashlib.sha256(
                np.ascontiguousarray(hk_grid, dtype=np.complex128).tobytes()
            ).hexdigest(),
            # V is a per-call argument: layers of another scattering potential
            # are a different physical problem with the same shard keys.
            "V_sha256": hashlib.sha256(
                np.ascontiguousarray(V, dtype=np.complex128).tobytes()
            ).hexdigest(),
        }

        # Memory bound: one worker holds the ``(nk, nk, nw, nw)`` complex128
        # hk_grid it received as payload, the matching identity batch the solver
        # materializes, the ``z*I - H`` operand (plus the ``z*I`` temporary that
        # feeds it, hence the slack matrix), ``G0`` and ``GV``; the per-orbital
        # FFT scratch is four ``(nk, nk, nw)`` complex128 planes, and the layers
        # the worker accumulates are ``max_share`` ``nk x nk`` float64 maps.
        matrix_bytes = 16 * self.nk * self.nk * self.num_wann * self.num_wann
        fft_plane_bytes = 16 * self.nk * self.nk * self.num_wann
        max_share = max(stop - start for start, stop in slices)
        per_worker_bytes = int(
            V.nbytes
            + 6 * matrix_bytes
            + 4 * fft_plane_bytes
            + max_share * self.nk * self.nk * 8
        )

        meta = {
            "module_type": "born",
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
            "hk_grid": hk_grid,
            "V": V,
            "eta": self.eta,
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
                target=_qpi_born_worker,
                name=f"born-worker-{index}",
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
                f"parallel Born QPI run interrupted; completed shards kept in {kept}"
            )
        if exit_code != 0 or not merged:
            raise RuntimeError(
                f"parallel Born QPI run failed with exit code {exit_code}"
            )
        return merged[0]

    def calculate(
        self,
        energy_range: float | np.ndarray | list[float],
        q_range: tuple[float, float] | None = (-0.5, 0.5),
        V: np.ndarray | None = None,
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
        V : np.ndarray or None, optional
            Scattering potential matrix, shape (num_wann, num_wann).  The
            identity matrix is used when None.
        output_path : str, optional
            Path to save output to HDF5.
        n_workers : int or None, optional (default None = serial)
            OPT-IN parallel path: split the energy axis into that many shards,
            compute each in its own worker process and merge them in plan order
            through :mod:`stm_data_processing.parallel.shard_driver`.  Every
            layer is independent (nothing is normalized across energies), so the
            result is bit-identical to the serial path; this only trades wall
            time for extra processes and per-shard checkpoint files.  The path
            is CPU-only: under the GPU backend, or for a single energy, the
            request falls back to the serial path with a WARNING.
        checkpoint_dir : str or Path or None, optional
            Where the parallel path keeps its shards.  ``None`` uses a temporary
            directory that is removed after a successful merge; an explicit
            directory is kept so a later ``resume=True`` run can reuse it.
        resume : bool, default False
            With ``n_workers`` and an explicit ``checkpoint_dir``: only the
            shards that are missing or were produced with different parameters
            (different ``nk``/``eta``/``hk_grid``/``V``) are recomputed.

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

        if V is not None:
            V = np.asarray(V, dtype=np.complex128)
            if V.shape != (self.num_wann, self.num_wann):
                raise ValueError(
                    f"V must be ({self.num_wann}, {self.num_wann}), got {V.shape}"
                )
            self.V = V
        else:
            self.V = np.eye(self.num_wann, dtype=np.complex128)

        logger.info(
            "[INFO] Starting QPI calculation on %s (nk=%d, nω=%d)",
            BACKEND,
            nk,
            len(energy_array),
        )

        use_gpu = BACKEND == "gpu"
        parallel = n_workers is not None and int(n_workers) > 1
        if parallel and use_gpu:
            logger.warning(
                "[%s] n_workers=%s requested while the GPU backend is active: "
                "spawned workers would inherit a CUDA context, which is unsafe, so "
                "this run falls back to the serial CUDA path",
                _QPI_LABEL,
                n_workers,
            )
            parallel = False
        if parallel and len(energy_array) < 2:
            logger.warning(
                "[%s] n_workers=%s requested for a single energy: using the "
                "serial path",
                _QPI_LABEL,
                n_workers,
            )
            parallel = False

        if parallel:
            qpi_layers = self._compute_born_parallel(
                energy_array,
                int(n_workers),
                checkpoint_dir,
                resume,
            )
        else:
            compute_func = self._compute_Gkq_cuda if use_gpu else self._compute_Gkq
            qpi_layers = np.empty((len(energy_array), nk, nk), dtype=np.float64)
            for ie, omega in enumerate(energy_array):
                logger.info(
                    "[%d/%d] Energy: %.4f eV", ie + 1, len(energy_array), float(omega)
                )
                qpi_layers[ie] = compute_func(float(omega))  # includes the 1/π

        if output_path is not None:
            logger.info("Saving QPI to %s", output_path)
            save_qpi_to_h5(
                qpi_layers=qpi_layers,
                output_path=output_path,
                energy_range=energy_array,
                module_type="born",
                bvecs=self.ham.bvecs,
                V=self.V,
                nq=self.nk,
                eta=self.eta,
                normalize=False,  # Born QPI is stored unnormalized
                bands=None,
            )

        if q_range is not None:
            qpi_layers_ext, q1_grid_ext, q2_grid_ext = extend_qpi(
                qpi_layers,
                self.q1_grid,
                self.q2_grid,
                q_range[0],
                q_range[1],
            )
        else:
            qpi_layers_ext = qpi_layers
            q1_grid_ext = self.q1_grid
            q2_grid_ext = self.q2_grid

        qx_grid, qy_grid = frac_to_real_2d(q1_grid_ext, q2_grid_ext, self.ham.bvecs)

        metadata = {
            "module_type": "born",
            "eta": self.eta,
            "normalize": False,  # Born QPI is stored unnormalized
            "nq": nk,
            "energy_range": energy_array,
            "bands": None,
            "bvecs": self.ham.bvecs,
            "V": self.V,
            "mask": None,
        }

        result: dict[str, Any] = {
            "qpi_layers": qpi_layers_ext,
            "q1_grid": q1_grid_ext,
            "q2_grid": q2_grid_ext,
            "qx_grid": qx_grid,
            "qy_grid": qy_grid,
            "metadata": metadata,
        }

        logger.info("QPI calculation completed.")

        return result
