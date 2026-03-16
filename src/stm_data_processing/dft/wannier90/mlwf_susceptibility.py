import logging
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pyfftw

from stm_data_processing.config import get_backend, get_xp
from stm_data_processing.dft.wannier90.mlwf_gk import GreenFunction
from stm_data_processing.dft.wannier90.mlwf_hamiltonian import MLWFHamiltonian
from stm_data_processing.io.susceptibility_io import save_susceptibility_to_h5
from stm_data_processing.utils.miscellaneous import (
    extend_qpi,
    frac_to_real_2d,
    k_to_q,
)

logger = logging.getLogger(__name__)


class SusceptibilityCalculator_wang2012:
    """Class for calculating Lindhard susceptibility chi0(q) from tight-binding Hamiltonian.
    Accelatration reference:
    DOI: https://doi.org/10.1103/PhysRevB.85.224529
    """

    def __init__(
        self,
        hamiltonian: MLWFHamiltonian,
        nk: int = 256,
        eta: float = 5e-3,
        minit: np.ndarray | None = None,
        mfin: np.ndarray | None = None,
    ):
        if not isinstance(hamiltonian, MLWFHamiltonian):
            raise TypeError(
                f"Expected MLWFHamiltonian, got {type(hamiltonian).__name__}"
            )

        self.ham = hamiltonian
        self.nk = nk
        self.eta = eta

        logger.info(
            f"[SusceptibilityCalculator] Initialized with backend={get_backend()}"
        )

        self._minit: np.ndarray | None = None
        self._mfin: np.ndarray | None = None
        self._init_orbital_matrices(minit, mfin)

        self._hk_grid = None
        self._gf = None
        self._k_points = None
        self._k1_grid = None
        self._k2_grid = None
        self._q1_grid = None
        self._q2_grid = None

    # Properties
    @property
    def num_wann(self):
        """Number of Wannier functions."""
        return self.ham.num_wann

    @property
    def minit(self):
        """Initial state orbital selection matrix."""
        return self._minit

    @property
    def mfin(self):
        """Final state orbital selection matrix."""
        return self._mfin

    @property
    def xp(self):
        return get_xp()

    @property
    def hk_grid(self):
        """Lazy load hk_grid, cached per instance."""
        if self._hk_grid is None:
            self._hk_grid = self.xp.asarray(self.ham.hk(self.k_points))
        return self._hk_grid

    @property
    def gf(self):
        """GreenFunction instance."""
        if self._gf is None:
            self._gf = GreenFunction(self.ham, eta=self.eta)
        return self._gf

    @property
    def k_points(self):
        """k_frac grid points."""
        if self._k_points is None:
            k_vals = np.linspace(-0.5, 0.5, self.nk, endpoint=False)
            k1, k2 = np.meshgrid(k_vals, k_vals, indexing="ij")

            self._k1_grid = k1
            self._k2_grid = k2
            self._q1_grid = k1.copy()
            self._q2_grid = k2.copy()

            self._k_points = np.column_stack(
                [
                    k1.ravel(),
                    k2.ravel(),
                    np.zeros(self.nk * self.nk, dtype=np.float64),
                ]
            )
        return self._k_points

    @property
    def k1_grid(self):
        """k1 grid (initialized with k_points)."""
        _ = self.k_points  # Ensure initialization
        return self._k1_grid

    @property
    def k2_grid(self):
        """k2 grid (initialized with k_points)."""
        _ = self.k_points  # Ensure initialization
        return self._k2_grid

    @property
    def q1_grid(self):
        """q1 grid (copy of k1_grid)."""
        _ = self.k_points  # Ensure initialization
        return self._q1_grid

    @property
    def q2_grid(self):
        """q2 grid (copy of k2_grid)."""
        _ = self.k_points  # Ensure initialization
        return self._q2_grid

    # Core math
    def _compute_single_particle_spectra(self, omega):
        """Compute spectral function A(k, omega) = -Im[G(k, omega)]/pi for all k-points."""
        hk_grid = self.hk_grid
        gr_k = self.gf.compute_green(hk_grid, omega)
        return -self.xp.imag(gr_k) / self.xp.pi

    def _compute_imag_chi_cuda(self, omega_limit: float, resolution: float):
        """CUDA version of occupied-unoccupied susceptibility calculation."""
        import cupy as cp

        logger.info(
            f"Starting CUDA occupied-unoccupied susceptibility: omega_limit={omega_limit}, resolution={resolution}"
        )

        nw = self.num_wann
        nk = self.nk

        MAX_GPU_MEMORY_FRACTION = 0.75
        mem_pool = self.xp.get_default_memory_pool()
        mem_pool.set_limit(size=int(24 * 1024**3 * MAX_GPU_MEMORY_FRACTION))

        logger.info(
            f"[CUDA] Memory pool limit set to {MAX_GPU_MEMORY_FRACTION * 100:.0f}% of 24GB"
        )

        n_eps = int(np.round(np.abs(omega_limit) / resolution)) + 1
        eps_occ = np.linspace(-np.abs(omega_limit), 0.0, n_eps)
        eps_unocc = eps_occ + omega_limit

        n_spectra_total = 2 * n_eps

        logger.info(
            f"Total energy points: {n_eps}, k-points: {nk}x{nk}, Wannier functions: {nw}"
        )
        logger.info(f"[CUDA] Total spectral function computations: {n_spectra_total}")

        chi_q_accum = self.xp.zeros((nk, nk), dtype=self.xp.float64)

        logger.info("[CUDA] Computing spectral functions and convolution...")
        spectra_count = 0
        for i in range(n_eps):
            spectra_occ = self._compute_single_particle_spectra(eps_occ[i])
            spectra_occ_2d = self.xp.ascontiguousarray(
                spectra_occ.reshape(nk, nk, nw, nw)
            )
            del spectra_occ
            spectra_count += 1

            spectra_unocc = self._compute_single_particle_spectra(eps_unocc[i])
            spectra_unocc_2d = self.xp.ascontiguousarray(
                spectra_unocc.reshape(nk, nk, nw, nw)
            )
            del spectra_unocc
            spectra_count += 1

            b_occ = self.xp.fft.fftn(spectra_occ_2d, axes=(0, 1))
            b_occ_shifted = self.xp.fft.fftshift(b_occ, axes=(0, 1))
            del spectra_occ_2d, b_occ

            b_unocc = self.xp.fft.fftn(spectra_unocc_2d, axes=(0, 1))
            b_unocc_shifted = self.xp.fft.fftshift(b_unocc, axes=(0, 1))
            del spectra_unocc_2d, b_unocc

            b_prod = self.xp.einsum("ijab,ijba->ij", b_occ_shifted, b_unocc_shifted)
            del b_occ_shifted, b_unocc_shifted

            conv_q = self.xp.fft.ifftn(
                self.xp.fft.ifftshift(b_prod, axes=(0, 1)), axes=(0, 1)
            ).real
            del b_prod

            chi_q_accum += conv_q
            del conv_q

            if spectra_count % 20 == 0:
                mem_pool.free_unused_blocks()

            if (
                spectra_count % max(1, n_spectra_total // 10) == 0
                or spectra_count == n_spectra_total
            ):
                progress = spectra_count / n_spectra_total * 100
                used_mem = mem_pool.used_bytes() / 1024**3
                logger.info(
                    f"[CUDA] Spectral function progress: {progress:.1f}% ({spectra_count}/{n_spectra_total}), "
                    f"GPU Memory: {used_mem:.2f} GB"
                )

        chi_q_accum = self.xp.fft.fftshift(chi_q_accum)
        chi_q = -np.abs(resolution) / (2 * np.pi) * chi_q_accum
        chi_q = self.xp.asnumpy(chi_q)

        mem_pool.free_all_blocks()
        cp.get_default_memory_pool().free_all_blocks()

        logger.info("[CUDA] Susceptibility calculation completed.")

        return chi_q

    def _compute_imag_chi(self, omega_limit: float, resolution: float):
        """Compute Im[chi(q)] with orbital selection matrices."""
        import importlib.util

        PYFFTW_AVAILABLE = importlib.util.find_spec("pyfftw") is not None
        if not PYFFTW_AVAILABLE:
            logger.warning("pyFFTW not available, falling back to numpy.fft")

        nw = self.num_wann
        nk = self.nk

        n_eps = int(np.round(np.abs(omega_limit) / resolution)) + 1
        eps_occ = np.linspace(-np.abs(omega_limit), 0.0, n_eps)
        eps_unocc = eps_occ + omega_limit

        n_spectra_total = 2 * n_eps
        logger.info(
            f"Occupied energy range: [{eps_occ[0]:.3f}, {eps_occ[-1]:.3f}] eV ({n_eps} points), "
            f"Unoccupied energy range: [{eps_unocc[0]:.3f}, {eps_unocc[-1]:.3f}] eV ({n_eps} points)"
        )
        logger.info(f"[CPU] Total spectral function computations: {n_spectra_total}")

        chi_q_accum = np.zeros((nk, nk), dtype=np.float64)

        if PYFFTW_AVAILABLE:
            logger.info("[CPU] Starting pyFFTW plan initialization...")

            num_threads = (
                max(1, len(os.sched_getaffinity(0)))
                if hasattr(os, "sched_getaffinity")
                else os.cpu_count()
            )
            num_threads = 10

            logger.info(f"[CPU] Using {num_threads} threads, nk={nk}, num_wann={nw}")

            wisdom_path = self._get_fftw_wisdom_path(nk, nw)

            spectra_shape = (nk, nk, nw, nw)
            fft_axes = (0, 1)

            logger.info("[CPU] Creating FFT plan for occupied states...")
            fft_occ, input_occ = self._init_fftw_plan(
                spectra_shape, fft_axes, num_threads, wisdom_path
            )
            logger.info("[CPU] FFT plan for occupied states completed.")

            logger.info("[CPU] Creating FFT plan for unoccupied states...")
            fft_unocc, input_unocc = self._init_fftw_plan(
                spectra_shape, fft_axes, num_threads, wisdom_path
            )
            logger.info("[CPU] FFT plan for unoccupied states completed.")

            logger.info("[CPU] Creating IFFT plan for convolution...")
            conv_shape = (nk, nk)
            ifft_conv, input_conv = self._init_fftw_plan(
                conv_shape, fft_axes, num_threads, wisdom_path
            )
            logger.info("[CPU] IFFT plan for convolution completed.")

            logger.info(f"[CPU] pyFFTW initialized with {num_threads} threads")
        else:
            fft_occ = fft_unocc = ifft_conv = None
            input_occ = input_unocc = input_conv = None

        logger.info(
            "[CPU] Computing spectral functions and convolution (streaming mode)..."
        )
        spectra_count = 0

        for i in range(n_eps):
            spectra_occ = np.asarray(self._compute_single_particle_spectra(eps_occ[i]))
            spectra_occ = spectra_occ.reshape(nk, nk, nw, nw)

            spectra_occ = np.einsum("ac,ijcb->ijab", self._minit, spectra_occ)
            spectra_count += 1

            spectra_unocc = np.asarray(
                self._compute_single_particle_spectra(eps_unocc[i])
            )
            spectra_unocc = spectra_unocc.reshape(nk, nk, nw, nw)

            spectra_unocc = np.einsum("ac,ijcb->ijab", self._mfin, spectra_unocc)
            spectra_count += 1

            if PYFFTW_AVAILABLE:
                input_occ[:] = spectra_occ
                b_occ = fft_occ()
                b_occ_shifted = np.fft.fftshift(b_occ, axes=(0, 1))

                input_unocc[:] = spectra_unocc
                b_unocc = fft_unocc()
                b_unocc_shifted = np.fft.fftshift(b_unocc, axes=(0, 1))
            else:
                b_occ = np.fft.fftn(spectra_occ, axes=(0, 1))
                b_occ_shifted = np.fft.fftshift(b_occ, axes=(0, 1))
                b_unocc = np.fft.fftn(spectra_unocc, axes=(0, 1))
                b_unocc_shifted = np.fft.fftshift(b_unocc, axes=(0, 1))

            del spectra_occ, spectra_unocc

            b_prod = np.einsum("ijab,ijba->ij", b_occ_shifted, b_unocc_shifted)
            del b_occ_shifted, b_unocc_shifted, b_occ, b_unocc

            if PYFFTW_AVAILABLE:
                input_conv[:] = np.fft.ifftshift(b_prod, axes=(0, 1))
                conv_q = ifft_conv().real
            else:
                conv_q = np.fft.ifftn(
                    np.fft.ifftshift(b_prod, axes=(0, 1)), axes=(0, 1)
                ).real

            del b_prod

            chi_q_accum += conv_q
            del conv_q

            if (
                spectra_count % max(1, n_spectra_total // 10) == 0
                or spectra_count == n_spectra_total
            ):
                progress = spectra_count / n_spectra_total * 100
                logger.info(
                    f"[CPU] Spectral function progress: {progress:.1f}% ({spectra_count}/{n_spectra_total})"
                )

        chi_q = np.fft.fftshift(chi_q_accum)
        chi_q = -np.abs(resolution) / (2 * np.pi) * chi_q

        logger.info("[CPU] Susceptibility calculation completed.")
        return chi_q

    # Helper
    def _get_fftw_wisdom_path(self, nk: int, nw: int) -> str:
        """Get FFTW wisdom file path for specific grid size."""
        wisdom_dir = Path(__file__).parent / "fftw_wisdom"
        wisdom_dir.mkdir(parents=True, exist_ok=True)
        return str(wisdom_dir / f"fftw_wisdom_nk{nk}_nw{nw}.json")

    def _init_fftw_plan(
        self, shape: tuple, fft_axes: tuple, num_threads: int, wisdom_path: str
    ):
        wisdom_path_obj = Path(wisdom_path)
        use_wisdom = False

        if wisdom_path_obj.exists():
            try:
                with wisdom_path_obj.open("rb") as f:
                    wisdom_tuple = pickle.load(f)
                pyfftw.import_wisdom(wisdom_tuple)
                logger.info(f"[CPU] Loaded FFTW wisdom from {wisdom_path}")
                use_wisdom = True
            except Exception as e:
                logger.warning(f"[CPU] Failed to load wisdom: {e}")

        fftw_flags = ["FFTW_ESTIMATE"] if use_wisdom else ["FFTW_MEASURE"]

        input_arr = pyfftw.empty_aligned(shape, dtype="complex128")
        output_arr = pyfftw.empty_aligned(shape, dtype="complex128")
        fft_plan = pyfftw.FFTW(
            input_arr,
            output_arr,
            axes=fft_axes,
            flags=fftw_flags,
            threads=num_threads,
        )

        if not use_wisdom:
            wisdom_tuple = pyfftw.export_wisdom()
            wisdom_path_obj.parent.mkdir(parents=True, exist_ok=True)
            with wisdom_path_obj.open("wb") as f:
                pickle.dump(wisdom_tuple, f)
            logger.info(f"[CPU] Saved FFTW wisdom to {wisdom_path}")

        return fft_plan, input_arr

    def _init_orbital_matrices(
        self, minit: np.ndarray | None = None, mfin: np.ndarray | None = None
    ):
        """Initialize orbital selection matrices."""
        nw = self.num_wann

        if minit is None:
            self._minit = np.eye(nw, dtype=np.float64)
        else:
            self._verify_matrix(minit, "minit")
            self._minit = np.asarray(minit, dtype=np.float64)

        if mfin is None:
            self._mfin = np.eye(nw, dtype=np.float64)
        else:
            self._verify_matrix(mfin, "mfin")
            self._mfin = np.asarray(mfin, dtype=np.float64)

    def _verify_matrix(self, matrix: np.ndarray, name: str):
        """Verify orbital selection matrix shape."""
        nw = self.num_wann
        if matrix.shape != (nw, nw):
            raise ValueError(f"{name} must have shape ({nw}, {nw}), got {matrix.shape}")

    # Public API
    def calculate(
        self,
        omega_limit: float,
        resolution: float,
        q_range: tuple[float, float] | None = (-0.5, 0.5),
        output_path: str | None = None,
    ) -> dict[str, Any]:
        """Calculate static Lindhard susceptibility chi0(q)."""
        nk = self.nk

        # Obtain Backend every time.
        current_backend = get_backend()
        logger.info(
            f"[INFO] Starting susceptibility calculation on {current_backend} "
            f"(nk={nk}, omega_limit={omega_limit:.3f}, resolution={resolution:.3e})",
        )

        if current_backend == "gpu":
            chi_q = self._compute_imag_chi_cuda(omega_limit, resolution)
        else:
            chi_q = self._compute_imag_chi(omega_limit, resolution)

        if output_path is not None:
            logger.info("[INFO] Saving susceptibility results to: %s", output_path)
            save_susceptibility_to_h5(
                susceptibility=chi_q,
                outpath=output_path,
                module_type="Imaginary Lindhard",
                bevecs=self.ham.bvecs,
                eta=self.eta,
                omega_limit=omega_limit,
                resolution=resolution,
                nq=nk,
                minit=self._minit,
                mfin=self._mfin,
            )

        q1_grid_orig, q2_grid_orig = k_to_q(self.k1_grid, self.k2_grid)

        if q_range is not None:
            chi_q, q1_grid, q2_grid = extend_qpi(
                chi_q, q1_grid_orig, q2_grid_orig, q_range[0], q_range[1]
            )
        else:
            q1_grid, q2_grid = q1_grid_orig, q2_grid_orig

        qx_grid, qy_grid = frac_to_real_2d(q1_grid, q2_grid, self.ham.bvecs)

        metadata = {
            "module_type": "imag_Lindhard",
            "eta": self.eta,
            "omega_limit": omega_limit,
            "resolution": resolution,
            "nq": nk,
            "bvecs": self.ham.bvecs,
            "minit": self._minit,
            "mfin": self._mfin,
        }

        result: dict[str, Any] = {
            "data": chi_q,
            "q1_grid": q1_grid,
            "q2_grid": q2_grid,
            "qx_grid": qx_grid,
            "qy_grid": qy_grid,
            "metadata": metadata,
        }

        logger.info("[INFO] Susceptibility calculation completed.")

        return result

    def set_orbital_selection(
        self, minit: np.ndarray | None = None, mfin: np.ndarray | None = None
    ):
        """Update orbital selection matrices."""
        if minit is not None:
            self._verify_matrix(minit, "minit")
            self._minit = np.asarray(minit, dtype=np.float64)

        if mfin is not None:
            self._verify_matrix(mfin, "mfin")
            self._mfin = np.asarray(mfin, dtype=np.float64)

        logger.info(
            f"[INFO] Orbital selection updated: minit shape={self._minit.shape}, "
            f"mfin shape={self._mfin.shape}"
        )

    def clear_cache(self):
        """Clear cached data. Call this after switching backend."""
        self._hk_grid = None
        self._gf = None
        self._k_points = None
        self._k1_grid = None
        self._k2_grid = None
        self._q1_grid = None
        self._q2_grid = None
        logger.info("[SusceptibilityCalculator] Cache cleared.")
