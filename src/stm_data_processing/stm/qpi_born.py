import logging
from typing import Any

import numpy as np

from stm_data_processing.config import BACKEND
from stm_data_processing.dft.wannier90.mlwf_gk import GreenFunction
from stm_data_processing.dft.wannier90.mlwf_hamiltonian import MLWFHamiltonian
from stm_data_processing.io.qpi_io import frac_to_real_2d, save_qpi_to_h5
from stm_data_processing.utils.miscellaneous import extend_qpi

if BACKEND == "gpu":
    import cupy as cp
else:
    cp = None

logger = logging.getLogger(__name__)


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
        nk = self.nk

        logger.info(f"  [CPU] Computing QPI at ω = {omega:.4f} eV (nk={nk})...")

        g0_k = self.gf.compute_green(self.hk_grid_cpu, omega)  # shape: (nk, nk, nw, nw)

        qpi: np.ndarray = np.zeros((nk, nk), dtype=np.float64)
        total_q = nk * nk
        logger.info(f"  [CPU] Processing {total_q} q-points...")

        for q1 in range(nk):
            for q2 in range(nk):
                g0_kq = np.roll(g0_k, shift=(-q1, -q2), axis=(0, 1))
                s = np.einsum("ijab,bc,ijca->", g0_k, self.V, g0_kq, optimize=True)
                qpi[q1, q2] = -np.imag(s)  # np.pi not included yet

                done = q1 * nk + q2 + 1
                percent = round(100 * done / total_q)
                logger.info(f"    Progress: {percent:3d}% ({done}/{total_q})")

        return np.fft.fftshift(qpi)

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

        g0_k_gpu = self.gf.compute_green(
            self.hk_grid_gpu, omega
        )  # shape: (nk, nk, nw, nw)
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

        return cp.asnumpy(qpi_gpu)

    # ============================================================
    # Public
    # ============================================================

    def calculate(
        self,
        energy_range: float | np.ndarray | list[float],
        q_range: tuple[float, float] | None = (-0.5, 0.5),
        V: np.ndarray | None = None,
        output_path: str | None = None,
    ) -> dict[str, Any]:
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

        compute_func = self._compute_Gkq_cuda if BACKEND == "gpu" else self._compute_Gkq

        qpi_layers: np.ndarray = np.empty((len(energy_array), nk, nk), dtype=np.float64)
        for ie, omega in enumerate(energy_array):
            logger.info(
                "[%d/%d] Energy: %.4f eV", ie + 1, len(energy_array), float(omega)
            )
            qpi_layers[ie] = compute_func(float(omega)) / np.pi  # np.pi is here.

        if output_path is not None:
            logger.info("[INFO] Saving QPI to %s", output_path)
            save_qpi_to_h5(
                qpi_layers=qpi_layers,
                output_path=output_path,
                energy_range=energy_array,
                module_type="born",
                bvecs=self.ham.bvecs,
                V=self.V,
                nq=self.nk,
                eta=self.eta,
                normalize=True,
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
            "normalize": True,
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

        logger.info("[INFO] QPI calculation completed.")

        return result
