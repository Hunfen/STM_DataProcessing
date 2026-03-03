import time
from pathlib import Path

import h5py
import numpy as np

from stm_data_processing.utils.lattice_loader import LatticeLoader
from stm_data_processing.utils.miscellaneous import frac_to_real_2d

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

from .mlwf_hamiltonian import MLWFHamiltonian


class EK2DCalculator:
    """
    A class for calculating and managing 2D band structure contour maps from Wannier90 Hamiltonian data.

    This class provides methods to compute band structures in the primitive Brillouin zone,
    extend them to larger k-space windows, and handle file I/O operations.
    """

    def __init__(
        self,
        hr_file: str | None = None,
        std_file: str | None = None,
        folder: str | None = None,
        seedname: str | None = None,
        use_gpu: bool | None = None,
    ):
        """
        Initialize the EK2DCalculator with Wannier90 Hamiltonian data.

        Parameters
        ----------
        hr_file : str, optional
            Path to Wannier90 hr.dat file. Deprecated; use `folder` and `seedname` instead.
        std_file : str, optional
            Path to Wannier90 .wout file for reciprocal lattice vectors.
        folder : str, optional
            Path to folder containing Wannier90 output files (e.g., `.hr.dat`, `.wout`).
        seedname : str, optional
            Seedname for Wannier90 files (e.g., `seedname.hr.dat`).
        use_gpu : bool, optional
            Whether to use GPU acceleration if CuPy is available. If None (default),
            GPU is used automatically when available.

        Raises
        ------
        ValueError
            If neither (`folder` and `seedname`) nor `hr_file` are provided.
        """
        self.use_gpu = use_gpu if use_gpu is not None else CUPY_AVAILABLE
        if self.use_gpu and not CUPY_AVAILABLE:
            print("Warning: GPU requested but CuPy not available. Falling back to CPU.")
            self.use_gpu = False
        elif self.use_gpu:
            print("GPU detected: using CUDA acceleration.")

        # Save input paths for later reloading
        self.hr_file = Path(hr_file).resolve() if hr_file else None
        self.std_file = Path(std_file).resolve() if std_file else None
        self.folder = Path(folder).resolve() if folder else None
        self.seedname = seedname

        # Initialize MLWFHamiltonian
        if folder is not None and seedname is not None:
            self.wh = MLWFHamiltonian(
                folder=folder, seedname=seedname, use_gpu=self.use_gpu
            )
            self.bvecs = self.wh.bvecs
        elif hr_file is not None:
            self.wh = MLWFHamiltonian(use_gpu=self.use_gpu)
            self.wh.load_hr_file(hr_file)
            if std_file is not None:
                lattice_obj = LatticeLoader(filename=std_file)
                self.bvecs = lattice_obj.bvecs.copy()
            else:
                self.bvecs = None
                print("Note: No std file provided. kx & ky set to be None.")
        else:
            raise ValueError("Either provide folder and seedname, or hr_file.")

    def _calculate_ek2d(
        self,
        mlwf_hamiltonian: MLWFHamiltonian,
        nk: int = 256,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate band contour map from Wannier90 Hamiltonian.

        Parameters
        ----------
        mlwf_hamiltonian : MLWFHamiltonian
            The Wannier Hamiltonian instance with loaded data.
        nk : int
            Number of k-points in each direction

        Returns
        -------
        e : (num_wann, nk, nk) ndarray
            Band energies contour map for all Wannier bands
        k1_grid : (nk, nk) ndarray
            Fractional k1 coordinates along b1 direction
        k2_grid : (nk, nk) ndarray
            Fractional k2 coordinates along b2 direction
        """
        # Generate fractional k-grid in the range [-0.5, 0.5)
        kmin, kmax = -0.5, 0.5
        k1 = np.linspace(kmin, kmax, nk, endpoint=False)
        k2 = np.linspace(kmin, kmax, nk, endpoint=False)
        k1_grid, k2_grid = np.meshgrid(k1, k2, indexing="ij")

        # Calculate E(k) in fractional coordinates for all bands
        e = np.zeros((mlwf_hamiltonian.num_wann, nk, nk))

        print(f"Calculating band structure on {nk}x{nk} k-grid (CPU)...")
        for i in range(nk):
            for j in range(nk):
                k_frac = (k1_grid[i, j], k2_grid[i, j], 0.0)
                evals = np.linalg.eigvalsh(mlwf_hamiltonian.hk(k_frac))
                e[:, i, j] = evals  # Save all band energies

        print("Band structure calculation complete.")
        print(f"  Energy range: {e.min():.4f} → {e.max():.4f} eV")

        return e, k1_grid, k2_grid

    def _calculate_ek2d_cuda(
        self,
        mlwf_hamiltonian: MLWFHamiltonian,
        nk: int = 256,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate band contour map from Wannier90 Hamiltonian using CUDA acceleration.

        Parameters
        ----------
        mlwf_hamiltonian : MLWFHamiltonian
            The Wannier Hamiltonian instance with loaded data.
        nk : int
            Number of k-points in each direction

        Returns
        -------
        e : (num_wann, nk, nk) ndarray
            Band energies for all bands
        k1_grid : (nk, nk) ndarray
            Fractional k1 coordinates along b1 direction
        k2_grid : (nk, nk) ndarray
            Fractional k2 coordinates along b2 direction

        Raises
        ------
        RuntimeError
            If GPU acceleration is not available.
        """
        # Initialize GPU arrays if needed
        mlwf_hamiltonian._initialize_gpu_arrays()

        # Generate fractional k-grid on GPU in the range [-0.5, 0.5)
        kmin, kmax = -0.5, 0.5
        k1 = cp.linspace(kmin, kmax, nk, endpoint=False)
        k2 = cp.linspace(kmin, kmax, nk, endpoint=False)
        k1_grid_gpu, k2_grid_gpu = cp.meshgrid(k1, k2, indexing="ij")

        # Store fractional coordinates (convert to numpy for output)
        k1_grid = cp.asnumpy(k1_grid_gpu)
        k2_grid = cp.asnumpy(k2_grid_gpu)

        # Flatten k-grid for parallel processing - keep on GPU
        k_points_flat = cp.zeros((nk * nk, 3), dtype=cp.float64)
        k_points_flat[:, 0] = k1_grid_gpu.ravel()
        k_points_flat[:, 1] = k2_grid_gpu.ravel()
        # kz is always 0 for 2D band structure

        print(f"Calculating band structure on {nk}x{nk} k-grid (GPU)...")

        # Get dimensions
        nrpts = mlwf_hamiltonian.r_list_gpu.shape[0]
        num_wann = mlwf_hamiltonian.num_wann
        total_kpoints = nk * nk

        # Allocate memory for results on GPU
        e_gpu = cp.zeros((num_wann, total_kpoints), dtype=cp.float64)

        # Process k-points in batches to avoid memory issues
        # Calculate optimal batch size based on available GPU memory
        free_mem, total_mem = cp.cuda.Device().mem_info  # Get current free memory

        # Memory estimation per k-point in batch:
        # - hk_batch: batch_size * num_wann * num_wann * 16 bytes (complex128)
        # - eigenvalues: batch_size * num_wann * 8 bytes (float64)
        # - intermediate arrays: batch_size * nrpts * 16 bytes (complex128 for phases)
        # - dot_products: batch_size * nrpts * 8 bytes (float64)
        mem_per_kpoint = (
            num_wann * num_wann * 16  # hk matrix (complex128)
            + num_wann * 8  # eigenvalues (float64)
            + nrpts * 16  # phases (complex128)
            + nrpts * 8  # dot_products (float64)
        )

        # Use up to 70% of currently free GPU memory for working arrays
        usable_mem = free_mem * 0.7
        max_batch_size = int(usable_mem / mem_per_kpoint)

        # Ensure batch_size is at least 64 and at most total_kpoints
        batch_size = max(64, min(max_batch_size, total_kpoints))

        num_batches = (total_kpoints + batch_size - 1) // batch_size

        print(f"  Using batch size: {batch_size} (total batches: {num_batches})")
        print(f"  Memory per k-point: {mem_per_kpoint / 1024:.1f} KB")
        print(
            f"  Estimated memory for batch: {batch_size * mem_per_kpoint / (1024**3):.2f} GB"
        )
        print(f"  Free GPU memory: {free_mem / (1024**3):.2f} GB")

        # Pre-compute constants
        two_pi_i = 2j * cp.pi

        total_hk_time = 0.0
        total_eig_time = 0.0
        start_total_time = time.time()

        for batch_idx in range(num_batches):
            batch_start_time = time.time()
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_kpoints)
            batch_size_current = end_idx - start_idx

            # Extract batch of k-points
            k_batch = k_points_flat[start_idx:end_idx]  # shape: (batch_size_current, 3)

            # OPTIMIZATION 1: Use einsum for efficient matrix multiplication
            # Compute all dot products at once: k_batch (B,3) @ r_list_gpu.T (3,R) -> (B,R)
            hk_start = time.time()
            dot_products = cp.dot(
                k_batch, mlwf_hamiltonian.r_list_gpu.T
            )  # shape: (batch_size_current, nrpts)

            # Compute phase factors for all k-points and R vectors
            phases = cp.exp(
                two_pi_i * dot_products
            )  # shape: (batch_size_current, nrpts)

            # Apply degeneracy weighting
            weights = (
                phases / mlwf_hamiltonian.ndegen_gpu
            )  # broadcasting: (batch_size_current, nrpts)

            # OPTIMIZATION 2: Memory-efficient summation without creating large intermediate arrays
            # Sum over R vectors: Σ_R H(R) * weight(R)
            hk_batch = cp.zeros(
                (batch_size_current, num_wann, num_wann), dtype=cp.complex128
            )

            # Process R vectors one by one to avoid memory issues
            for ir in range(nrpts):
                # Multiply each H(R) matrix by its corresponding weight for all k-points in batch
                weight_ir = weights[:, ir]  # shape: (batch_size_current,)
                h_ir = mlwf_hamiltonian.h_list_gpu[ir]  # shape: (num_wann, num_wann)

                # Broadcasting: weight_ir[:, None, None] has shape (batch_size_current, 1, 1)
                # h_ir has shape (num_wann, num_wann)
                # Result has shape (batch_size_current, num_wann, num_wann)
                hk_batch += weight_ir[:, None, None] * h_ir[None, :, :]

            hk_time = time.time() - hk_start
            total_hk_time += hk_time

            # OPTIMIZATION 3: Batch diagonalization
            eig_start = time.time()
            # Process matrices in smaller sub-batches for better memory management
            sub_batch_size = min(
                64, batch_size_current
            )  # Smaller batches for diagonalization

            for sub_start in range(0, batch_size_current, sub_batch_size):
                sub_end = min(sub_start + sub_batch_size, batch_size_current)
                sub_size = sub_end - sub_start

                # Extract sub-batch of Hamiltonian matrices
                hk_sub = hk_batch[
                    sub_start:sub_end
                ]  # shape: (sub_size, num_wann, num_wann)

                # Process each matrix in the sub-batch
                for i in range(sub_size):
                    # Use eigh for Hermitian matrices (faster and more stable than eigvalsh)
                    evals = cp.linalg.eigh(hk_sub[i])[0]  # Only eigenvalues needed
                    e_gpu[:, start_idx + sub_start + i] = evals

            eig_time = time.time() - eig_start
            total_eig_time += eig_time

            # Print progress with memory usage information
            if (batch_idx + 1) % max(
                1, num_batches // 10
            ) == 0 or batch_idx == num_batches - 1:
                progress = (batch_idx + 1) / num_batches * 100
                free_mem, total_mem = cp.cuda.Device().mem_info
                memory_used = (total_mem - free_mem) / 1e9  # Used memory in GB
                memory_total = total_mem / 1e9  # Total memory in GB

                elapsed_total = time.time() - start_total_time
                batches_per_sec = (
                    (batch_idx + 1) / elapsed_total if elapsed_total > 0 else 0
                )

                print(
                    f"  Progress: {progress:.1f}%, "
                    f"Batch {batch_idx + 1}/{num_batches}, "
                    f"Elapsed: {elapsed_total:.1f}s, "
                    f"Rate: {batches_per_sec:.1f} batches/s, "
                    f"HK: {hk_time:.1f}s, Eig: {eig_time:.1f}s, "
                    f"GPU memory: {memory_used:.2f}/{memory_total:.2f} GB"
                )

        # Print performance summary
        print("\nPerformance summary:")
        print(f"  Total HK computation time: {total_hk_time:.2f}s")
        print(f"  Total diagonalization time: {total_eig_time:.2f}s")
        print(
            f"  Average time per k-point: {(total_hk_time + total_eig_time) / total_kpoints * 1000:.2f}ms"
        )

        # Reshape results to 3D array
        e_gpu = e_gpu.reshape((num_wann, nk, nk))

        # Convert results back to CPU
        e = cp.asnumpy(e_gpu)

        print("Band energy contour map complete (GPU).")
        print(f"  Energy range: {e.min():.4f} → {e.max():.4f} eV")
        # Explicitly free temporary GPU memory used during computation
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

        return e, k1_grid, k2_grid

    def calculate(
        self,
        nk: int = 256,
        k_range: tuple[float, float] | None = None,
        save_to_file: str | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Compute a band energy contour map from pre-loaded Wannier90 Hamiltonian data.

        By default, computes over the primitive Brillouin zone [-0.5, 0.5) x [-0.5, 0.5).
        If `k_range` is provided, the result is extended periodically to cover [kmin, kmax).

        Parameters
        ----------
        nk : int, default 256
            Number of k-points along each direction within the primitive Brillouin zone.
        k_range : tuple[float, float], optional
            (kmin, kmax) defining the target fractional k-space window [kmin, kmax).
            If None, only the primitive BZ [-0.5, 0.5) is computed.
        save_to_file : str, optional
            If provided, saves the **primitive BZ result** (before extension) to an HDF5 file.
            The `.h5` extension is appended if not present. This helps save disk space.

        Returns
        -------
        dict
            A dictionary containing the following keys:
            - 'energies': (num_wann, Nk, Nk) array of band energies in eV.
            - 'kx', 'ky': (Nk, Nk) arrays of real-space k-coordinates in 1/Å.
                          Will be None if reciprocal vectors (`bvecs`) are unavailable.
            - 'k1_grid', 'k2_grid': (Nk, Nk) arrays of fractional k-coordinates.
            - 'bvecs': (3, 3) array of reciprocal lattice vectors in 1/Å, or None.
        """
        # Step 1: Calculate base E(k) in primitive BZ [-0.5, 0.5)
        if self.use_gpu and CUPY_AVAILABLE:
            print("Using GPU-accelerated calculation...")
            e, k1_grid, k2_grid = self._calculate_ek2d_cuda(self.wh, nk)
        else:
            print("Using CPU calculation...")
            e, k1_grid, k2_grid = self._calculate_ek2d(self.wh, nk)

        # Step 2: Save primitive BZ result (small size) if requested — BEFORE extension
        if save_to_file:
            self._save_ek2d(e, k1_grid, k2_grid, save_to_file, self.wh.bvecs)

        # Step 3: Optionally extend to larger k-range
        if k_range is not None:
            print(
                f"Extending band structure to fractional k-range: [{k_range[0]}, {k_range[1]})"
            )
            base_ek2d = {
                "energies": e,
                "k1_grid": k1_grid,
                "k2_grid": k2_grid,
            }
            extended = self._extend_ek2d(base_ek2d, k_range)
            e = extended["energies"]
            k1_grid = extended["k1_grid"]
            k2_grid = extended["k2_grid"]

        # Step 4: Compute real-space coordinates (using possibly extended grids)
        kx, ky = frac_to_real_2d(k1_grid, k2_grid, self.wh.bvecs)

        return {
            "energies": e,
            "kx": kx,
            "ky": ky,
            "k1_grid": k1_grid,
            "k2_grid": k2_grid,
            "bvecs": self.wh.bvecs,
        }

    def _extend_ek2d(
        self,
        ek2d: dict[str, np.ndarray],
        k_range: tuple[float, float],
    ) -> dict[str, np.ndarray]:
        """
        Extend a band contour map from the primitive Brillouin zone [-0.5, 0.5)
        to an arbitrary fractional k-space window [kmin, kmax).

        This function takes the output from calculate and extends it
        periodically to cover the specified k-space window while preserving
        the original k-point density.

        Parameters
        ----------
        ek2d : dict
            Dictionary returned by calculate containing:
            - 'energies': (num_wann, nk, nk) array of band energies computed within [-0.5, 0.5)
            - 'k1_grid': (nk, nk) array of fractional k1 coordinates in [-0.5, 0.5)
            - 'k2_grid': (nk, nk) array of fractional k2 coordinates in [-0.5, 0.5)
        k_range : tuple[float, float]
            (kmin, kmax) defining the target fractional k-space window [kmin, kmax).

        Returns
        -------
        dict[str, np.ndarray]
            A dictionary with the same structure as the input `ek2d`, containing:
            - 'energies': (num_wann, Nk1, Nk2) array of extended band energies
            - 'k1_grid': (Nk1, Nk2) array of extended fractional k1 coordinates
            - 'k2_grid': (Nk1, Nk2) array of extended fractional k2 coordinates

        Notes
        -----
        - The input data must be computed within the primitive Brillouin zone [-0.5, 0.5)
        - The output map will have the same k-point density as the input
        - The function handles both positive and negative kmin/kmax values
        - The result is cropped exactly to [kmin, kmax) without extrapolation
        """
        # Extract base data
        e_base = ek2d["energies"]
        k1_base = ek2d["k1_grid"]
        k2_base = ek2d["k2_grid"]

        kmin, kmax = k_range
        nk = k1_base.shape[0]

        # -------- 1. extend ----------
        n_min = int(np.floor(kmin + 0.5))
        n_max = int(np.ceil(kmax - 0.5))

        shifts = np.arange(n_min, n_max + 1)

        nband = e_base.shape[0]
        nk_big = nk * len(shifts)

        e_big = np.zeros((nband, nk_big, nk_big))
        k1_big = np.zeros((nk_big, nk_big))
        k2_big = np.zeros((nk_big, nk_big))

        for ix, sx in enumerate(shifts):
            for iy, sy in enumerate(shifts):
                x0 = ix * nk
                x1 = (ix + 1) * nk
                y0 = iy * nk
                y1 = (iy + 1) * nk

                e_big[:, x0:x1, y0:y1] = e_base
                k1_big[x0:x1, y0:y1] = k1_base + sx
                k2_big[x0:x1, y0:y1] = k2_base + sy

        # -------- 2. crop ----------
        mask_x = (k1_big[:, 0] >= kmin) & (k1_big[:, 0] < kmax)
        mask_y = (k2_big[0, :] >= kmin) & (k2_big[0, :] < kmax)

        k1_ext = k1_big[np.ix_(mask_x, mask_y)]
        k2_ext = k2_big[np.ix_(mask_x, mask_y)]
        e_ext = e_big[:, mask_x, :][:, :, mask_y]

        return {
            "energies": e_ext,
            "k1_grid": k1_ext,
            "k2_grid": k2_ext,
        }

    def load_ek2d(
        self,
        filename: str,
        k_range: tuple[float, float] | None = None,
    ) -> dict[str, np.ndarray | None]:
        start_time = time.time()
        filepath = Path(filename)
        if not filepath.exists():
            raise FileNotFoundError(f"HDF5 file not found: {filename}")

        if not filename.lower().endswith((".h5", ".hdf5")):
            print(f"Warning: File '{filename}' does not have .h5 or .hdf5 extension")

        try:
            with h5py.File(filename, "r") as f:
                for key in ["energies", "k1_grid", "k2_grid"]:
                    if key not in f:
                        raise ValueError(
                            f"Required dataset '{key}' not found in {filename}"
                        )

                energies = f["energies"][:]
                k1_grid = f["k1_grid"][:]
                k2_grid = f["k2_grid"][:]
                bvecs = f["bvecs"][:] if "bvecs" in f else None
                metadata = dict(f.attrs)
        except Exception as e:
            raise ValueError(f"Failed to load '{filename}': {e!s}") from e

        # Reinitialize self.wh from saved paths if possible
        hr_file = metadata.get("hr_file")
        std_file = metadata.get("std_file")
        folder = metadata.get("folder")
        seedname = metadata.get("seedname")

        hamiltonian_rebuilt = False
        if folder and seedname:
            try:
                self.wh = MLWFHamiltonian(
                    folder=folder, seedname=seedname, use_gpu=self.use_gpu
                )
                self.bvecs = self.wh.bvecs
                hamiltonian_rebuilt = True
                print("Reinitialized MLWFHamiltonian from folder + seedname.")
            except Exception as e:
                print(f"Failed to reinitialize from folder/seedname: {e}")
        elif hr_file:
            try:
                self.wh = MLWFHamiltonian(use_gpu=self.use_gpu)
                self.wh.load_hr_file(hr_file)
                if std_file:
                    lattice_obj = LatticeLoader(filename=std_file)
                    self.bvecs = lattice_obj.bvecs.copy()
                else:
                    self.bvecs = bvecs  # fallback to saved bvecs
                hamiltonian_rebuilt = True
                print("Reinitialized MLWFHamiltonian from hr_file.")
            except Exception as e:
                print(f"Failed to reinitialize from hr_file: {e}")

        if not hamiltonian_rebuilt:
            print(
                "Warning: Could not reinitialize MLWFHamiltonian. Using saved bvecs only."
            )
            if bvecs is not None:
                # At least ensure bvecs is consistent
                if hasattr(self, "wh") and hasattr(self.wh, "bvecs"):
                    self.wh.bvecs = bvecs.copy()
                self.bvecs = bvecs

        # Compute real-space coordinates
        if bvecs is not None:
            kx, ky = frac_to_real_2d(k1_grid, k2_grid, bvecs)
        else:
            kx, ky = None, None
            print("Note: No bvecs available. Real-space kx, ky set to None.")

        # Extend if needed
        if k_range is not None:
            print(f"Extending to k-range: [{k_range[0]}, {k_range[1]})")
            base_ek2d = {"energies": energies, "k1_grid": k1_grid, "k2_grid": k2_grid}
            extended = self._extend_ek2d(base_ek2d, k_range)
            energies = extended["energies"]
            k1_grid = extended["k1_grid"]
            k2_grid = extended["k2_grid"]
            if bvecs is not None:
                kx, ky = frac_to_real_2d(k1_grid, k2_grid, bvecs)

        # Summary
        num_wann, nk1, nk2 = energies.shape
        print(f"Loaded from {filename} | Bands: {num_wann} | Grid: {nk1}x{nk2}")
        print(f"Energy range: {energies.min():.4f} → {energies.max():.4f} eV")
        print(f"Load time: {time.time() - start_time:.2f} s")

        return {
            "energies": energies,
            "kx": kx,
            "ky": ky,
            "k1_grid": k1_grid,
            "k2_grid": k2_grid,
            "bvecs": bvecs,
            "metadata": metadata,
        }

    def _save_ek2d(
        self,
        energies: np.ndarray,
        k1_grid: np.ndarray,
        k2_grid: np.ndarray,
        filename: str,
        bvecs: np.ndarray | None = None,
    ) -> None:
        start_time = time.time()
        num_wann, nk, _ = energies.shape
        total_points = nk * nk

        print("Preparing to save band structure data to HDF5 file...")
        print(f"  Number of bands: {num_wann}")
        print(f"  k-grid size: {nk}x{nk}")
        print(f"  Total k-points: {total_points:,}")

        if not filename.lower().endswith((".h5", ".hdf5")):
            filename += ".h5"
            print("  Note: Added .h5 extension to filename")

        with h5py.File(filename, "w") as f:
            f.create_dataset(
                "energies", data=energies, compression="gzip", compression_opts=4
            )
            f.create_dataset(
                "k1_grid", data=k1_grid, compression="gzip", compression_opts=4
            )
            f.create_dataset(
                "k2_grid", data=k2_grid, compression="gzip", compression_opts=4
            )

            # Metadata
            f.attrs["num_wann"] = num_wann
            f.attrs["nk"] = nk
            f.attrs["total_points"] = total_points
            f.attrs["units_energy"] = "eV"
            f.attrs["units_k_frac"] = "reciprocal lattice units"
            f.attrs["creation_date"] = time.strftime("%Y-%m-%d %H:%M:%S")
            f.attrs["generator"] = "EK2DCalculator"

            # Save source file paths for full reinitialization
            if self.hr_file is not None:
                f.attrs["hr_file"] = str(self.hr_file)
            if self.std_file is not None:
                f.attrs["std_file"] = str(self.std_file)
            if self.folder is not None:
                f.attrs["folder"] = str(self.folder)
            if self.seedname is not None:
                f.attrs["seedname"] = self.seedname

            if bvecs is not None:
                f.create_dataset("bvecs", data=bvecs)

        elapsed_time = time.time() - start_time
        print(f"Band structure data saved to {filename}")
        print(f"  Save time: {elapsed_time:.2f} seconds")
