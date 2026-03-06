import numpy as np

from ..dft.wannier90.mlwf_gk import GreenFunction
from ..dft.wannier90.mlwf_hamiltonian import _CUPY_USABLE, MLWFHamiltonian, cp
from ..utils.miscellaneous import extend_qpi, frac_to_real_2d, save_qpi_to_h5


class BornQPI:
    """
    QPI calculator using Born approximation.

    δρ(q,ω) = -(1/π) Im Σ_k Tr[G0(k,ω) V G0(k+q,ω)]
    """

    def __init__(self, hamiltonian: MLWFHamiltonian, nk: int = 256, eta: float = 0.001):
        self.ham = hamiltonian
        self.gf = GreenFunction(hamiltonian)
        self.num_wann = hamiltonian.num_wann
        self.nk = nk
        self.eta = eta

        self.V = np.ones((self.num_wann, self.num_wann), dtype=complex)

        k_vals = np.linspace(-0.5, 0.5, nk, endpoint=False)
        self.k1_grid, self.k2_grid = np.meshgrid(k_vals, k_vals, indexing="ij")
        self.q1_grid, self.q2_grid = self.k1_grid.copy(), self.k2_grid.copy()

    def _compute_Gkq(self, omega: float, normalize: bool = True) -> np.ndarray:

        nk = self.nk
        norm_factor = np.pi * nk * nk if normalize else np.pi

        G0_k = np.zeros((nk, nk, self.num_wann, self.num_wann), dtype=complex)

        for i in range(nk):
            for j in range(nk):
                k_frac = (self.k1_grid[i, j], self.k2_grid[i, j], 0.0)
                G0_k[i, j] = self.gf.G0(k_frac, omega, self.eta)

        qpi = np.zeros((nk, nk))

        total_q = nk * nk
        print(f"  [CPU] Processing {total_q} q-points...")

        for q1 in range(nk):
            for q2 in range(nk):
                G0_kq = np.roll(G0_k, shift=(q1, q2), axis=(0, 1))

                s = np.einsum("ijab,bc,ijca->", G0_k, self.V, G0_kq)

                qpi[q1, q2] = -np.imag(s) / norm_factor

                # Show progress every 10% (only if nk is large enough)
                if nk >= 64 and (q1 * nk + q2 + 1) % (total_q // 10) == 0:
                    percent = 10 * ((q1 * nk + q2 + 1) // (total_q // 10))
                    print(f"    Progress: {percent:3d}% ({q1 * nk + q2 + 1}/{total_q})")

        return np.fft.fftshift(qpi)

    def _compute_Gkq_cuda(self, omega: float, normalize: bool = True) -> np.ndarray:
        """
        CUDA-accelerated version of _compute_Gkq with optimized memory usage.

        Avoids O(nk^4) memory by computing QPI one q-point at a time,
        but vectorizing over k-points using einsum.
        """
        if not _CUPY_USABLE or cp is None:
            raise RuntimeError("CUDA version requested but CuPy/CUDA is not available.")

        print(f"  [CUDA] Computing QPI at ω = {omega:.4f} eV (nk={self.nk})...")

        nk = self.nk
        norm_factor = np.pi * nk * nk if normalize else np.pi

        k_points = np.stack(
            [self.k1_grid.flatten(), self.k2_grid.flatten(), np.zeros(nk * nk)], axis=1
        )

        H_k = self.ham.hk(k_points)
        H_k_gpu = cp.asarray(H_k) if isinstance(H_k, np.ndarray) else H_k
        H_k_gpu = H_k_gpu.reshape(nk, nk, self.num_wann, self.num_wann)

        omega_complex = omega + 1j * self.eta
        eye_gpu = cp.eye(self.num_wann, dtype=cp.complex128)
        G0_k_gpu = cp.linalg.inv(omega_complex * eye_gpu[None, None, :, :] - H_k_gpu)

        V_gpu = cp.asarray(self.V)
        GV_gpu = cp.einsum("ijab,bc->ijac", G0_k_gpu, V_gpu)

        qpi_gpu = cp.zeros((nk, nk), dtype=cp.float64)

        total_q = nk * nk
        print(f"  [CUDA] Processing {total_q} q-points...")

        for q1 in range(nk):
            for q2 in range(nk):
                G0_kq = cp.roll(G0_k_gpu, shift=(q1, q2), axis=(0, 1))
                s = cp.einsum("ijca,ijac->", GV_gpu, G0_kq)
                qpi_gpu[q1, q2] = -cp.imag(s) / norm_factor

                if nk >= 64 and (q1 * nk + q2 + 1) % (total_q // 10) == 0:
                    percent = 10 * ((q1 * nk + q2 + 1) // (total_q // 10))
                    print(f"    Progress: {percent:3d}% ({q1 * nk + q2 + 1}/{total_q})")

        qpi_gpu = cp.fft.fftshift(qpi_gpu)
        print("  [CUDA] Done.")
        return cp.asnumpy(qpi_gpu)

    def calculate_qpi(
        self,
        energy_range: float | np.ndarray | list[float],
        q_range: tuple[float, float] | None = (-0.5, 0.5),
        V: np.ndarray | None = None,
        normalize: bool = True,
        output_path: str | None = None,
    ):

        nk = self.nk
        energy_array = np.atleast_1d(energy_range)
        is_scalar = np.isscalar(energy_range)

        if V is not None:
            V = np.asarray(V)
            if V.shape != (self.num_wann, self.num_wann):
                raise ValueError(
                    f"V must be ({self.num_wann}, {self.num_wann}), got {V.shape}"
                )
            self.V = V

        use_cuda = (
            _CUPY_USABLE
            and cp is not None
            and hasattr(self.ham, "_backend")
            and self.ham._backend == "gpu"
        )

        backend_str = "CUDA" if use_cuda else "CPU"
        print(
            f"[INFO] Starting QPI calculation on {backend_str} (nk={nk}, nω={len(energy_array)})"
        )

        compute_func = self._compute_Gkq_cuda if use_cuda else self._compute_Gkq

        qpi_layers = np.empty((len(energy_array), nk, nk))
        for ie, omega in enumerate(energy_array):
            print(f"[{ie + 1}/{len(energy_array)}] Energy: {omega:.4f} eV")
            qpi_layers[ie] = compute_func(omega, normalize)

        result = {
            "qpi_layers": qpi_layers,
            "q1_grid": self.q1_grid,
            "q2_grid": self.q2_grid,
        }

        if output_path is not None:
            print(f"[INFO] Saving QPI to {output_path}")
            save_qpi_to_h5(
                qpi_layers=qpi_layers,
                output_path=output_path,
                energy_range=energy_range,
                normalize=normalize,
                bvecs=self.ham.bvecs,
                V=self.V,
                eta=self.eta,
                nq=self.nk,
            )

        if q_range is not None:
            qpi_layers, q1_grid_ext, q2_grid_ext = extend_qpi(
                qpi_layers, self.q1_grid, self.q2_grid, q_range[0], q_range[1]
            )
        qx_grid, qy_grid = frac_to_real_2d(q1_grid_ext, q2_grid_ext, self.ham.bvecs)

        result = {
            "qpi_layers": qpi_layers,
            "qx_grid": qx_grid,
            "qy_grid": qy_grid,
            "q1_grid": q1_grid_ext,
            "q2_grid": q2_grid_ext,
            "bvecs": self.ham.bvecs,
        }
        print("[INFO] QPI calculation completed.")
        return result
