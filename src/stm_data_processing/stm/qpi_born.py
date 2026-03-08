import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..dft.wannier90.mlwf_gk import GreenFunction
from ..dft.wannier90.mlwf_hamiltonian import (
    _CUPY_USABLE,
    MLWFHamiltonian,
    cp,
)
from ..io.qpi_io import save_qpi_to_h5
from ..utils.miscellaneous import extend_qpi, frac_to_real_2d


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
        eta: float = 0.001,
    ) -> None:
        self.ham: MLWFHamiltonian = hamiltonian
        self.gf: GreenFunction = GreenFunction(hamiltonian, eta=eta)
        self.num_wann: int | None = hamiltonian.num_wann
        self.nk: int = int(nk)
        self.eta: float = float(eta)
        self.logger: logging.Logger = logging.getLogger(__name__)

        if self.num_wann is None:
            raise RuntimeError("Hamiltonian not loaded: num_wann is None.")

        self.V: NDArray[np.complex128] = np.ones(
            (self.num_wann, self.num_wann), dtype=np.complex128
        )

        k_vals = np.linspace(-0.5, 0.5, self.nk, endpoint=False)
        self.k1_grid: NDArray[np.float64]
        self.k2_grid: NDArray[np.float64]
        self.q1_grid: NDArray[np.float64]
        self.q2_grid: NDArray[np.float64]
        self.k1_grid, self.k2_grid = np.meshgrid(k_vals, k_vals, indexing="ij")
        self.q1_grid, self.q2_grid = self.k1_grid.copy(), self.k2_grid.copy()
        k_points = np.column_stack(
            [
                self.k1_grid.ravel(),
                self.k2_grid.ravel(),
                np.zeros(nk * nk, dtype=np.float64),
            ]
        )
        self.hk_grid_cpu: NDArray[np.complex128] = self.ham.hk(k_points).reshape(
            self.nk, self.nk, self.num_wann, self.num_wann
        )

        self.hk_grid_gpu: Any | None = None
        if self._use_cuda():
            self.hk_grid_gpu = cp.asarray(self.hk_grid_cpu)

    # ============================================================
    # Helpers
    # ============================================================

    def _use_cuda(self) -> bool:
        return (
            _CUPY_USABLE
            and cp is not None
            and hasattr(self.ham, "_backend")
            and self.ham._backend == "gpu"
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

    def _prepare_g0_grid_cpu(self, omega: float) -> NDArray[np.complex128]:
        """
        Compute G0(k, omega) using cached Hamiltonian grid on CPU.
        """
        return self.gf._green_from_hk_cpu(self.hk_grid_cpu, omega)

    def _prepare_g0_grid_gpu(self, omega: float) -> Any:
        """
        Compute G0(k, omega) using cached Hamiltonian grid on GPU.
        """
        if self.hk_grid_gpu is None:
            raise RuntimeError("GPU Hamiltonian grid not initialized.")
        return self.gf._green_from_hk_gpu(self.hk_grid_gpu, omega)

    # ============================================================
    # CPU
    # ============================================================

    def _compute_Gkq(
        self,
        omega: float,
        normalize: bool = True,
    ) -> NDArray[np.float64]:
        nk = self.nk
        norm_factor = np.pi * nk * nk if normalize else np.pi

        self.logger.info("  [CPU] Computing QPI at ω = %.4f eV (nk=%d)...", omega, nk)
        g0_k = self._prepare_g0_grid_cpu(omega)

        qpi: NDArray[np.float64] = np.zeros((nk, nk), dtype=np.float64)
        total_q = nk * nk
        self.logger.info("  [CPU] Processing %d q-points...", total_q)

        for q1 in range(nk):
            for q2 in range(nk):
                # G(k+q), consistent with CUDA path
                g0_kq = np.roll(g0_k, shift=(-q1, -q2), axis=(0, 1))
                s = np.einsum("ijab,bc,ijca->", g0_k, self.V, g0_kq, optimize=True)
                qpi[q1, q2] = -np.imag(s) / norm_factor

                done = q1 * nk + q2 + 1
                if nk >= 64 and done % max(total_q // 10, 1) == 0:
                    percent = round(100 * done / total_q)
                    self.logger.info(
                        "    Progress: %3d%% (%d/%d)", percent, done, total_q
                    )

        return np.fft.fftshift(qpi)

    # ============================================================
    # CUDA
    # ============================================================

    def _compute_Gkq_cuda(
        self,
        omega: float,
        normalize: bool = True,
    ) -> NDArray[np.float64]:
        """
        CUDA implementation with q-block batching.

        Strategy
        --------
        1. Build full G0(k) on GPU once.
        2. Precompute GV(k) = G0(k) @ V on GPU.
        3. Process q-points in blocks:
             s(q) = sum_k Tr[ GV(k) G0(k+q) ]
        4. Keep intermediate arrays on GPU; return to CPU only at the end.
        """
        if not _CUPY_USABLE or cp is None:
            raise RuntimeError("CUDA version requested but CuPy/CUDA is not available.")

        nk = self.nk
        nw = self.num_wann
        norm_factor = np.pi * nk * nk if normalize else np.pi

        self.logger.info("  [CUDA] Computing QPI at ω = %.4f eV (nk=%d)...", omega, nk)
        g0_k_gpu = self._prepare_g0_grid_gpu(omega)  # (nk, nk, nw, nw)
        v_gpu = cp.asarray(self.V, dtype=cp.complex128)

        # GV(k) = G(k)V, shape (nk, nk, a, c)
        gv_gpu = cp.einsum("ijab,bc->ijac", g0_k_gpu, v_gpu, optimize=True)

        qpi_flat_gpu = cp.empty(nk * nk, dtype=cp.float64)

        q1_all = cp.arange(nk * nk, dtype=cp.int32) // nk
        q2_all = cp.arange(nk * nk, dtype=cp.int32) % nk

        q_block = self._estimate_q_block_size(g0_k_gpu)
        total_q = nk * nk
        self.logger.info(
            "  [CUDA] Processing %d q-points in blocks of %d...", total_q, q_block
        )

        i_idx = cp.arange(nk, dtype=cp.int32)[None, :, None]
        j_idx = cp.arange(nk, dtype=cp.int32)[None, None, :]

        done = 0
        for start in range(0, total_q, q_block):
            stop = min(start + q_block, total_q)

            q1_block = q1_all[start:stop]  # (Bq,)
            q2_block = q2_all[start:stop]  # (Bq,)
            bq = stop - start

            # (Bq, nk, nk)
            idx1 = (i_idx + q1_block[:, None, None]) % nk
            idx2 = (j_idx + q2_block[:, None, None]) % nk

            # G(k+q): (Bq, nk, nk, nw, nw)
            g0_kq_block = g0_k_gpu[idx1, idx2]

            # s(q) = Σ_{i,j,a,c} GV(i,j,a,c) * Gq(i,j,c,a)
            s_block = cp.einsum("ijac,qijca->q", gv_gpu, g0_kq_block, optimize=True)
            qpi_flat_gpu[start:stop] = -cp.imag(s_block) / norm_factor

            done = stop
            if nk >= 64 and done % max(total_q // 10, 1) < q_block:
                percent = round(100 * done / total_q)
                self.logger.info("    Progress: %3d%% (%d/%d)", percent, done, total_q)

            del idx1, idx2, g0_kq_block, s_block

        qpi_gpu = qpi_flat_gpu.reshape(nk, nk)
        qpi_gpu = cp.fft.fftshift(qpi_gpu)

        self.logger.info("  [CUDA] Done.")
        return cp.asnumpy(qpi_gpu)

    # ============================================================
    # Public
    # ============================================================
    def calculate_qpi(
        self,
        energy_range: float | NDArray[np.float64] | list[float],
        q_range: tuple[float, float] | None = (-0.5, 0.5),
        V: NDArray[np.complex128] | None = None,
        normalize: bool = True,
        output_path: str | None = None,
    ) -> dict[str, Any]:
        nk = self.nk

        # Ensure energy_array is always a 1D numpy array of float64
        energy_array: NDArray[np.float64] = np.asarray(
            energy_range, dtype=np.float64
        ).ravel()

        # Use local variable for V to avoid modifying self.V (state safety)
        current_V: NDArray[np.complex128]
        if V is not None:
            V = np.asarray(V, dtype=np.complex128)
            if V.shape != (self.num_wann, self.num_wann):
                raise ValueError(
                    f"V must be ({self.num_wann}, {self.num_wann}), got {V.shape}"
                )
            current_V = V
        else:
            current_V = np.ones((self.num_wann, self.num_wann), dtype=np.complex128)

        # Temporarily set self.V for compute methods (they access self.V)
        original_V = self.V
        self.V = current_V

        use_cuda = self._use_cuda()
        backend_str = "CUDA" if use_cuda else "CPU"
        self.logger.info(
            "[INFO] Starting QPI calculation on %s (nk=%d, nω=%d)",
            backend_str,
            nk,
            len(energy_array),
        )

        compute_func = self._compute_Gkq_cuda if use_cuda else self._compute_Gkq

        qpi_layers: NDArray[np.float64] = np.empty(
            (len(energy_array), nk, nk), dtype=np.float64
        )
        for ie, omega in enumerate(energy_array):
            self.logger.info(
                "[%d/%d] Energy: %.4f eV", ie + 1, len(energy_array), float(omega)
            )
            qpi_layers[ie] = compute_func(float(omega), normalize)

        if output_path is not None:
            self.logger.info("[INFO] Saving QPI to %s", output_path)
            save_qpi_to_h5(
                qpi_layers=qpi_layers,
                output_path=output_path,
                energy_range=energy_array,
                normalize=normalize,
                bvecs=self.ham.bvecs,
                V=current_V,
                eta=self.eta,
                nq=self.nk,
            )

        if q_range is not None:
            qpi_layers, q1_grid_ext, q2_grid_ext = extend_qpi(
                qpi_layers,
                self.q1_grid,
                self.q2_grid,
                q_range[0],
                q_range[1],
            )
        else:
            q1_grid_ext = self.q1_grid
            q2_grid_ext = self.q2_grid

        qx_grid, qy_grid = frac_to_real_2d(q1_grid_ext, q2_grid_ext, self.ham.bvecs)

        result: dict[str, Any] = {
            "qpi_layers": qpi_layers,
            "qx_grid": qx_grid,
            "qy_grid": qy_grid,
            "q1_grid": q1_grid_ext,
            "q2_grid": q2_grid_ext,
            "bvecs": self.ham.bvecs,
        }

        self.logger.info("[INFO] QPI calculation completed.")
        self.V = original_V

        return result
