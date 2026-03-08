import logging
import os
import time

import numpy as np

from stm_data_processing.io.ek2d_io import EK2DIO
from stm_data_processing.utils.miscellaneous import frac_to_real_2d

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

from stm_data_processing.dft.wannier90.mlwf_hamiltonian import MLWFHamiltonian

logger = logging.getLogger(__name__)


class EK2DCalculator:
    """
    A class for calculating 2D band structure contour maps from Wannier90 Hamiltonian data.

    This class provides methods to compute band structures in the primitive Brillouin zone
    and extend them to larger k-space windows. For file I/O operations, use EK2DIO class.
    For loading Wannier90 data, use MLWFHamiltonian class.
    """

    def __init__(
        self,
        mlwf_hamiltonian: MLWFHamiltonian,
        nk: int = 256,
    ):
        """
        Initialize the EK2DCalculator with a pre-loaded MLWFHamiltonian instance.

        Parameters
        ----------
        mlwf_hamiltonian : MLWFHamiltonian
            Pre-initialized Wannier Hamiltonian instance with loaded data.
            Use MLWFHamiltonian.from_seedname() or MLWFHamiltonian() to create.
        nk : int, default 256
            Number of k-points in each direction.

        Examples
        --------
        >>> from stm_data_processing.dft.wannier90 import MLWFHamiltonian, EK2DCalculator
        >>> # Load data and create Hamiltonian
        >>> hamiltonian = MLWFHamiltonian.from_seedname(folder="/w90folder", seedname="seedname")
        >>> # Create calculator
        >>> calculator = EK2DCalculator(hamiltonian)
        >>> # Calculate band structure
        >>> result = calculator.calculate()
        """
        self.wh = mlwf_hamiltonian
        self.bvecs = self.wh.bvecs
        self.use_gpu = self.wh._backend == "gpu"
        self.nk = nk

        # Initialize k-grid and k-points in primitive BZ [-0.5, 0.5)
        k_vals = np.linspace(-0.5, 0.5, nk, endpoint=False)
        self.k1_grid, self.k2_grid = np.meshgrid(k_vals, k_vals, indexing="ij")
        self.k_points = np.c_[
            self.k1_grid.ravel(), self.k2_grid.ravel(), np.zeros(nk**2)
        ]

        if self.use_gpu and CUPY_AVAILABLE:
            logger.info("GPU backend detected: using CUDA acceleration.")
        elif self.use_gpu and not CUPY_AVAILABLE:
            logger.warning("GPU backend requested but CuPy not available.")

    def _compute_ek2d(
        self,
        mlwf_hamiltonian: MLWFHamiltonian,
    ) -> np.ndarray:
        """
        Calculate band contour map from Wannier90 Hamiltonian on CPU.

        Parameters
        ----------
        mlwf_hamiltonian : MLWFHamiltonian
            The Wannier Hamiltonian instance with loaded data.

        Returns
        -------
        e : (num_wann, nk, nk) ndarray
            Band energies contour map for all Wannier bands.
        """
        logger.info(
            f"Calculating band structure on {self.nk}x{self.nk} k-grid (CPU)..."
        )

        # Compute H(k) for all points at once (vectorized)
        hk_matrices = mlwf_hamiltonian.hk(self.k_points)

        # Diagonalize batched Hamiltonians
        # hk_matrices shape: (N, num_wann, num_wann)
        evals = np.linalg.eigvalsh(hk_matrices)

        # Reshape to (num_wann, nk, nk)
        # eigvalsh returns eigenvalues in ascending order, shape (N, num_wann)
        evals = evals.T.reshape((mlwf_hamiltonian.num_wann, self.nk, self.nk))

        logger.info("Band structure calculation complete.")
        logger.info(f"  Energy range: {evals.min():.4f} -> {evals.max():.4f} eV")

        return evals

    def _compute_ek2d_cuda(
        self,
        mlwf_hamiltonian: MLWFHamiltonian,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate band contour map from Wannier90 Hamiltonian using CUDA acceleration.

        Parameters
        ----------
        mlwf_hamiltonian : MLWFHamiltonian
            The Wannier Hamiltonian instance with loaded data.

        Returns
        -------
        e : (num_wann, nk, nk) ndarray
            Band energies for all bands.
        k1_grid : (nk, nk) ndarray
            Fractional k1 coordinates along b1 direction.
        k2_grid : (nk, nk) ndarray
            Fractional k2 coordinates along b2 direction.
        """
        # Ensure MLWFHamiltonian returns CuPy arrays for GPU diagonalization
        os.environ["MLWF_GPU_RETURN"] = "cupy"

        # Move k-points to GPU
        k_points_gpu = cp.asarray(self.k_points)

        logger.info(
            f"Calculating band structure on {self.nk}x{self.nk} k-grid (GPU)..."
        )

        # Compute H(k) for all points at once (vectorized, returns CuPy array)
        hk_matrices_gpu = mlwf_hamiltonian.hk(k_points_gpu)

        # Diagonalize batched Hamiltonians on GPU
        # Allocate memory for results
        num_wann = mlwf_hamiltonian.num_wann
        total_kpoints = self.nk * self.nk
        e_gpu = cp.zeros((num_wann, total_kpoints), dtype=cp.float64)

        # Batch diagonalization to manage memory if needed
        # cp.linalg.eigh supports batching, but we loop to be safe with memory
        batch_size = 1024
        num_batches = (total_kpoints + batch_size - 1) // batch_size

        start_total_time = time.time()
        total_eig_time = 0.0

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_kpoints)

            eig_start = time.time()
            hk_batch = hk_matrices_gpu[start_idx:end_idx]
            # eigvalsh returns (batch_size, num_wann)
            evals_batch = cp.linalg.eigvalsh(hk_batch)
            e_gpu[:, start_idx:end_idx] = evals_batch.T
            total_eig_time += time.time() - eig_start

            # Print progress
            if (batch_idx + 1) % max(
                1, num_batches // 10
            ) == 0 or batch_idx == num_batches - 1:
                progress = (batch_idx + 1) / num_batches * 100
                elapsed_total = time.time() - start_total_time
                logger.info(
                    f"  Progress: {progress:.1f}%, "
                    f"Batch {batch_idx + 1}/{num_batches}, "
                    f"Elapsed: {elapsed_total:.1f}s, "
                    f"Eig: {total_eig_time:.1f}s"
                )

        # Print performance summary
        logger.info("\nPerformance summary:")
        logger.info(f"  Total diagonalization time: {total_eig_time:.2f}s")
        logger.info(
            f"  Average time per k-point: {total_eig_time / total_kpoints * 1000:.2f}ms"
        )

        # Reshape results to 3D array
        e_gpu = e_gpu.reshape((num_wann, self.nk, self.nk))

        # Convert results back to CPU
        e = cp.asnumpy(e_gpu)

        logger.info("Band energy contour map complete (GPU).")
        logger.info(f"  Energy range: {e.min():.4f} -> {e.max():.4f} eV")

        # Explicitly free temporary GPU memory
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

        return e, self.k1_grid, self.k2_grid

    def calculate(
        self,
        k_range: tuple[float, float] | None = None,
        save_to_file: str | None = None,
        hr_file: str | None = None,
        std_file: str | None = None,
        folder: str | None = None,
        seedname: str | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Compute a band energy contour map from pre-loaded Wannier90 Hamiltonian data.

        By default, computes over the primitive Brillouin zone [-0.5, 0.5) x [-0.5, 0.5).
        If `k_range` is provided, the result is extended periodically to cover [kmin, kmax).

        Parameters
        ----------
        k_range : tuple[float, float], optional
            (kmin, kmax) defining the target fractional k-space window [kmin, kmax).
            If None, only the primitive BZ [-0.5, 0.5) is computed.
        save_to_file : str, optional
            If provided, saves the **primitive BZ result** (before extension) to an HDF5 file
            using EK2DIO. The `.h5` extension is appended if not present.
        hr_file, std_file, folder, seedname : str, optional
            Metadata for saving to HDF5. If not provided, uses values from MLWFHamiltonian
            if available.

        Returns
        -------
        dict
            A dictionary containing:
            - 'energies': (num_wann, Nk, Nk) array of band energies in eV.
            - 'kx', 'ky': (Nk, Nk) arrays of real-space k-coordinates in 1/Angstrom.
            - 'k1_grid', 'k2_grid': (Nk, Nk) arrays of fractional k-coordinates.
            - 'bvecs': (3, 3) array of reciprocal lattice vectors in 1/Angstrom, or None.

        Notes
        -----
        For saving/loading band structure data, use the EK2DIO class instead.
        """
        # Step 1: Calculate base E(k)
        if self.use_gpu and CUPY_AVAILABLE:
            logger.info("Using GPU-accelerated calculation...")
            e, k1_grid, k2_grid = self._compute_ek2d_cuda(self.wh)
        else:
            logger.info("Using CPU calculation...")
            e = self._compute_ek2d(self.wh)
            k1_grid, k2_grid = self.k1_grid, self.k2_grid

        # Step 2: Save primitive BZ result if requested — BEFORE extension
        if save_to_file:
            EK2DIO.save_ek2d(
                energies=e,
                k1_grid=k1_grid,
                k2_grid=k2_grid,
                filename=save_to_file,
                mlwf_hamiltonian=self.wh,
            )

        # Step 3: Optionally extend to larger k-range
        if k_range is not None:
            logger.info(
                f"Extending band structure to fractional k-range: [{k_range[0]}, {k_range[1]})"
            )
            base_ek2d = {
                "energies": e,
                "k1_grid": k1_grid,
                "k2_grid": k2_grid,
            }
            extended = EK2DIO._extend_ek2d_static(base_ek2d, k_range)
            e = extended["energies"]
            k1_grid = extended["k1_grid"]
            k2_grid = extended["k2_grid"]

        # Step 4: Compute real-space coordinates
        kx, ky = frac_to_real_2d(k1_grid, k2_grid, self.wh.bvecs)

        return {
            "energies": e,
            "kx": kx,
            "ky": ky,
            "k1_grid": k1_grid,
            "k2_grid": k2_grid,
            "bvecs": self.wh.bvecs,
        }
