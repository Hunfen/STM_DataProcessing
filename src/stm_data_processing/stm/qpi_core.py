import numpy as np
from scipy.fft import fft2, fftshift, ifft2

try:
    import cupy as cp

    CUPY_AVAILABLE = True
    print("CuPy is available. GPU acceleration enabled for QPI calculations.")
except ImportError:
    CUPY_AVAILABLE = False
    print("CuPy not found. Using CPU-only mode for QPI calculations.")


class QPICalculator:
    """Class for calculating QPI (Quasiparticle Interference) from band contours.

    This class combines spectral function calculation with QPI-JDOS computation
    to analyze quasiparticle interference patterns from band structure data.
    """

    def __init__(self, eta=0.001):
        """Initialize QPI calculator.

        Parameters
        ----------
        eta : float, optional
            Spectral broadening parameter, default is 0.001

        """
        self.eta = eta

    def calculate_spectral_function(self, e_k, energy, bands="all", mask=None):
        """Calculate the spectral function A(k, E).

        Parameters
        ----------
        e_k : array-like
            Energy values for each k-point (shape: [nkx, nky] or [num_wann, nkx, nky])
        energy : float
            Energy value at which to evaluate the spectral function
        bands : 'all' or list of int, optional
            Band indices to include in the spectral function calculation:
            - 'all' (default): Sum over all bands
            - list of int: Sum over specific band indices
        mask : array-like, optional
            Boolean mask to apply to k-points (True = include, False = exclude)
            Shape should match the spatial dimensions of e_k

        Returns
        -------
        a_k : numpy.ndarray
            Spectral function values (shape: [nkx, nky])

        """
        # Ensure input is a numpy array for consistent operations
        e_k = np.asarray(e_k)

        # Calculate spectral function using Lorentzian formula
        # A(k, E) = (1/π) * η / [(E - E_k)^2 + η^2]
        if e_k.ndim == 2:  # Single band case
            denominator = (energy - e_k) ** 2 + self.eta**2
            a_k = (1 / np.pi) * (self.eta / denominator)
        else:  # Multi-band case
            denominator = (energy - e_k) ** 2 + self.eta**2
            a_k_per_band = (1 / np.pi) * (self.eta / denominator)

            # Handle band selection with ternary operator
            a_k = (
                np.sum(a_k_per_band, axis=0)
                if bands == "all"
                else np.sum(a_k_per_band[bands], axis=0)
            )

        # Apply mask if provided
        if mask is not None:
            mask = np.asarray(mask)

            # Ensure mask matches the spatial dimensions of the output
            if a_k.shape[-2:] != mask.shape:
                raise ValueError(
                    f"Mask shape {mask.shape} does not match spatial dimensions of a_k {a_k.shape[-2:]}",
                )

            # Apply mask (set masked values to zero)
            a_k = a_k * mask

        return a_k

    def calculate_jdos_qpi(
        self,
        k1_grid: np.ndarray,
        k2_grid: np.ndarray,
        e_k: np.ndarray,
        energy_range: float | np.ndarray | list[float],
        bvecs: np.ndarray | None = None,
        bands: str | list[int] = "all",
        normalize: bool = True,
        mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Calculate the joint density of states (JDOS)-based quasiparticle interference (QPI) pattern using CPU (NumPy).

        This method computes the QPI signal in momentum space by evaluating the Fourier transform of the
        local density of states (LDOS) autocorrelation, approximated via the JDOS formalism. The spectral
        function is modeled as a Lorentzian with broadening parameter `self.eta`.

        Parameters
        ----------
        k1_grid : np.ndarray
            2D array of k1 coordinates on the reciprocal-space grid.
        k2_grid : np.ndarray
            2D array of k2 coordinates on the reciprocal-space grid.
        e_k : np.ndarray
            Band energies on the k-grid. Shape is either (nkx, nky) for a single band or (nbands, nkx, nky)
            for multiple bands.
        energy_range : float or array-like
            Energy (or energies) at which to compute the QPI.
        bvecs : np.ndarray, optional
            Reciprocal lattice basis vectors of shape (2, 2), used to convert dimensionless q-grids to physical units.
            If None, q-grids are left in dimensionless units.
        bands : str or array-like, optional
            Specifies which bands to include. Use "all" to sum over all bands, or provide an array of band indices.
        normalize : bool, optional
            If True, normalize each QPI map to its maximum value. Default is True.
        mask : np.ndarray, optional
            Real-space mask applied to the spectral function before Fourier transform. Must be broadcastable to (nkx, nky).

        Returns
        -------
        tuple
            - jdos_scan : np.ndarray
                Computed QPI map(s). Shape is (nkx, nky) if `energy_range` is scalar, or (n_energies, nkx, nky) otherwise.
            - qx_grid : np.ndarray
                Physical qx-coordinates of the QPI grid (if `bvecs` provided), otherwise same as `q1_grid`.
            - qy_grid : np.ndarray
                Physical qy-coordinates of the QPI grid (if `bvecs` provided), otherwise same as `q2_grid`.
            - q1_grid : np.ndarray
                Dimensionless q1-coordinates (lattice units).
            - q2_grid : np.ndarray
                Dimensionless q2-coordinates (lattice units).
        """
        # ... existing code ...
        energy_array = np.atleast_1d(energy_range)
        is_scalar = np.isscalar(energy_range)

        # Precompute q-grids
        nkx, nky = k1_grid.shape
        dk1 = float(k1_grid[1, 0] - k1_grid[0, 0]) if nkx > 1 else 1.0
        dk2 = float(k2_grid[0, 1] - k2_grid[0, 0]) if nky > 1 else 1.0
        q1_vals = (np.arange(nkx) - nkx // 2) * dk1
        q2_vals = (np.arange(nky) - nky // 2) * dk2
        q1_grid, q2_grid = np.meshgrid(q1_vals, q2_vals, indexing="ij")
        if bvecs is not None:
            qx_grid = q1_grid * bvecs[0, 0] + q2_grid * bvecs[1, 0]
            qy_grid = q1_grid * bvecs[0, 1] + q2_grid * bvecs[1, 1]
        else:
            qx_grid, qy_grid = q1_grid, q2_grid

        # Allocate result
        jdos_scan = np.empty((len(energy_array), nkx, nky))

        for i, energy in enumerate(energy_array):
            # Spectral function
            if e_k.ndim == 2:
                denom = (energy - e_k) ** 2 + self.eta**2
                a_k = (1 / np.pi) * (self.eta / denom)
            else:
                denom = (energy - e_k) ** 2 + self.eta**2
                a_k_per_band = (1 / np.pi) * (self.eta / denom)
                a_k = (
                    np.sum(a_k_per_band, axis=0)
                    if bands == "all"
                    else np.sum(a_k_per_band[bands], axis=0)
                )

            if mask is not None:
                a_k = a_k * mask

            # JDOS
            a_r = fft2(a_k)
            jdos_q = np.real(ifft2(np.abs(a_r) ** 2))
            jdos_q = fftshift(jdos_q)
            if normalize and (max_val := np.max(jdos_q)) > 0:
                jdos_q /= max_val

            jdos_scan[i] = jdos_q

        return (
            (jdos_scan[0] if is_scalar else jdos_scan),
            qx_grid,
            qy_grid,
            q1_grid,
            q2_grid,
        )

    def calculate_jdos_qpi_cuda(
        self,
        k1_grid: np.ndarray,
        k2_grid: np.ndarray,
        e_k: np.ndarray,
        energy_range: float | np.ndarray | list[float],
        bvecs: np.ndarray | None = None,
        bands: str | list[int] = "all",
        normalize: bool = True,
        mask: np.ndarray | None = None,
        batch_size: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Calculate the joint density of states (JDOS)-based quasiparticle interference (QPI) pattern using GPU (CuPy).

        This method leverages GPU acceleration via CuPy to compute QPI maps over one or multiple energies.
        It uses a batched approach to manage GPU memory efficiently, automatically determining a safe batch size
        based on available GPU memory unless explicitly specified. The spectral function is modeled as a Lorentzian
        with broadening parameter `self.eta`.

        Parameters
        ----------
        k1_grid : np.ndarray
            2D array of k1 coordinates on the reciprocal-space grid (CPU, NumPy).
        k2_grid : np.ndarray
            2D array of k2 coordinates on the reciprocal-space grid (CPU, NumPy).
        e_k : np.ndarray
            Band energies on the k-grid. Shape is either (nkx, nky) for a single band or (nbands, nkx, nky)
            for multiple bands (CPU, NumPy).
        energy_range : float or array-like
            Energy (or energies) at which to compute the QPI.
        bvecs : np.ndarray, optional
            Reciprocal lattice basis vectors of shape (2, 2), used to convert dimensionless q-grids to physical units.
            If None, q-grids are left in dimensionless units.
        bands : str or array-like, optional
            Specifies which bands to include. Use "all" to sum over all bands, or provide an array of band indices.
        normalize : bool, optional
            If True, normalize each QPI map to its maximum value. Default is True.
        mask : np.ndarray, optional
            Real-space mask applied to the spectral function before Fourier transform. Must be broadcastable to (nkx, nky).
        batch_size : int, optional
            Number of energies to process per GPU batch. If None, the method estimates a safe batch size based on
            available GPU memory and grid dimensions.

        Returns
        -------
        tuple
            - jdos_scan : np.ndarray
                Computed QPI map(s) on CPU. Shape is (nkx, nky) if `energy_range` is scalar,
                or (n_energies, nkx, nky) otherwise.
            - qx_grid : np.ndarray
                Physical qx-coordinates of the QPI grid (if `bvecs` provided), otherwise same as `q1_grid`.
            - qy_grid : np.ndarray
                Physical qy-coordinates of the QPI grid (if `bvecs` provided), otherwise same as `q2_grid`.
            - q1_grid : np.ndarray
                Dimensionless q1-coordinates (lattice units).
            - q2_grid : np.ndarray
                Dimensionless q2-coordinates (lattice units).

        Raises
        ------
        RuntimeError
            If CuPy is not available.

        Notes
        -----
        This function prints progress updates, memory usage, and timing statistics during execution.
        """
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy is required for GPU version.")
        import time

        start_time = time.time()

        energy_array = cp.asarray(np.atleast_1d(energy_range))
        is_scalar = np.isscalar(energy_range)

        e_k_gpu = cp.asarray(e_k)
        mask_gpu = cp.asarray(mask) if mask is not None else None

        nkx, nky = k1_grid.shape
        n_energies = energy_array.size

        # Determine batch size
        if batch_size is None:
            e_k_bytes = e_k_gpu.nbytes
            mem_per_energy = e_k_bytes * 5  # 5x safety factor

            free_mem = cp.cuda.Device().mem_info[0]
            # Use at most 50% of total GPU memory for batch
            batch_size = max(1, int(free_mem * 0.75 / mem_per_energy))
            batch_size = min(batch_size, n_energies)

            # Safety cap for large grids
            if nkx >= 1024 and nky >= 1024:
                batch_size = min(batch_size, 8)
            elif nkx >= 512 and nky >= 512:
                batch_size = min(batch_size, 16)

        num_batches = (n_energies + batch_size - 1) // batch_size

        # >>>> Print task overview <<<<
        print(f"GPU QPI energy scan: {n_energies} energies on {nkx}x{nky} grid")
        print(f"  Using batch size: {batch_size} (total batches: {num_batches})")

        jdos_scan_gpu = cp.zeros((n_energies, nkx, nky), dtype=cp.float64)

        # Process in batches
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, n_energies)
            batch_energies = energy_array[start:end]

            # Vectorized computation over batch
            if e_k_gpu.ndim == 2:
                denom = (
                    batch_energies[:, None, None] - e_k_gpu[None, :, :]
                ) ** 2 + self.eta**2
                a_k_batch = (1 / cp.pi) * (self.eta / denom)
            else:
                denom = (
                    batch_energies[:, None, None, None] - e_k_gpu[None, :, :, :]
                ) ** 2 + self.eta**2
                a_k_per_band = (1 / cp.pi) * (self.eta / denom)
                if bands == "all":
                    a_k_batch = cp.sum(a_k_per_band, axis=1)
                else:
                    bands_idx = cp.array(bands) if isinstance(bands, list) else bands
                    a_k_batch = cp.sum(a_k_per_band[:, bands_idx, :, :], axis=1)

            if mask_gpu is not None:
                a_k_batch *= mask_gpu

            a_r = cp.fft.fft2(a_k_batch, axes=(-2, -1))
            jdos_batch = cp.real(cp.fft.ifft2(cp.abs(a_r) ** 2, axes=(-2, -1)))
            jdos_batch = cp.fft.fftshift(jdos_batch, axes=(-2, -1))

            if normalize:
                max_vals = cp.max(jdos_batch, axis=(-2, -1), keepdims=True)
                jdos_batch = cp.where(max_vals > 0, jdos_batch / max_vals, jdos_batch)

            jdos_scan_gpu[start:end] = jdos_batch

            del a_k_batch, a_r, jdos_batch, denom
            cp.get_default_memory_pool().free_all_blocks()

            # >>>> Print progress and memory usage <<<<
            progress = (batch_idx + 1) / num_batches * 100
            free_mem, total_mem = cp.cuda.Device().mem_info
            used_mem = total_mem - free_mem
            print(
                f"  Progress: {progress:.1f}%, Batch {batch_idx + 1}/{num_batches}, "
                f"GPU memory: {used_mem / 1e9:.2f}/{total_mem / 1e9:.2f} GB"
            )

        # Compute q-grids on CPU
        dk1 = float(k1_grid[1, 0] - k1_grid[0, 0]) if nkx > 1 else 1.0
        dk2 = float(k2_grid[0, 1] - k2_grid[0, 0]) if nky > 1 else 1.0
        q1_vals = (np.arange(nkx) - nkx // 2) * dk1
        q2_vals = (np.arange(nky) - nky // 2) * dk2
        q1_grid, q2_grid = np.meshgrid(q1_vals, q2_vals, indexing="ij")
        if bvecs is not None:
            qx_grid = q1_grid * bvecs[0, 0] + q2_grid * bvecs[1, 0]
            qy_grid = q1_grid * bvecs[0, 1] + q2_grid * bvecs[1, 1]
        else:
            qx_grid, qy_grid = q1_grid, q2_grid

        jdos_scan = cp.asnumpy(jdos_scan_gpu)

        # >>>> Final summary <<<<
        elapsed_time = time.time() - start_time
        print(f"GPU QPI energy scan complete in {elapsed_time:.2f} seconds")
        print(f"  Average time per energy: {elapsed_time / n_energies * 1000:.2f} ms")

        return (
            (jdos_scan[0] if is_scalar else jdos_scan),
            qx_grid,
            qy_grid,
            q1_grid,
            q2_grid,
        )

    def calculate_qpi(
        self,
        k1_grid: np.ndarray,
        k2_grid: np.ndarray,
        e_k: np.ndarray,
        energy_range: float | np.ndarray | list[float],
        bvecs: np.ndarray | None = None,
        bands: str | list[int] = "all",
        normalize: bool = True,
        mask: np.ndarray | None = None,
        use_gpu: bool = False,
        batch_size: int | None = None,
        output_path: str | None = None,
    ) -> dict[str, np.ndarray]:
        """Unified QPI calculator with automatic CPU/GPU selection and optional HDF5 saving.

        Parameters
        ----------
        k1_grid : np.ndarray
            2D array of k1 coordinates on the reciprocal-space grid.
        k2_grid : np.ndarray
            2D array of k2 coordinates on the reciprocal-space grid.
        e_k : np.ndarray
            Band energies on the k-grid. Shape is either (nkx, nky) or (nbands, nkx, nky).
        energy_range : float or array-like
            Energy (or energies) at which to compute the QPI.
        bvecs : np.ndarray, optional
            Reciprocal lattice basis vectors of shape (2, 2).
        bands : str or array-like, optional
            Band indices to include; "all" or list of ints.
        normalize : bool, optional
            Whether to normalize each QPI map to its max. Default is True.
        mask : np.ndarray, optional
            Real-space mask applied before FFT.
        use_gpu : bool, optional
            If True and CuPy available, use GPU acceleration. Default is False.
        batch_size : int, optional
            Batch size for GPU computation (only used if `use_gpu=True`).
        output_path : str or None, optional
            If provided, save the result dictionary as an HDF5 file at this path.

        Returns
        -------
        result : dict
            {
                "jdos": numpy.ndarray,          # JDOS(q, E)
                "qx_grid": numpy.ndarray,       # 2D meshgrid (for convenience)
                "qy_grid": numpy.ndarray,
                "q1_grid": numpy.ndarray,
                "q2_grid": numpy.ndarray,
            }

        Notes
        -----
        If `output_path` is given, the result is saved in an HDF5 file with compact storage:
        - 2D grids are stored as 1D coordinate arrays + shape.
        - Input parameters affecting physics are stored as metadata.
        The file is overwritten if it already exists.
        """
        if use_gpu and CUPY_AVAILABLE:
            jdos, qx, qy, q1, q2 = self.calculate_jdos_qpi_cuda(
                k1_grid,
                k2_grid,
                e_k,
                energy_range,
                bvecs,
                bands,
                normalize,
                mask,
                batch_size,
            )
        else:
            if use_gpu and not CUPY_AVAILABLE:
                print("Warning: CuPy not available. Falling back to CPU.")
            jdos, qx, qy, q1, q2 = self.calculate_jdos_qpi(
                k1_grid, k2_grid, e_k, energy_range, bvecs, bands, normalize, mask
            )

        result = {
            "jdos": jdos,
            "qx_grid": qx,
            "qy_grid": qy,
            "q1_grid": q1,
            "q2_grid": q2,
        }

        # Save to HDF5 if output_path is provided
        if output_path is not None:
            import h5py

            # Extract 1D coordinates from 2D grids (assuming regular meshgrid with indexing='ij')
            nkx, nky = qx.shape
            qx_1d = qx[:, 0]  # all rows, first column → varies along x
            qy_1d = qy[0, :]  # first row, all columns → varies along y
            q1_1d = q1[:, 0]
            q2_1d = q2[0, :]

            with h5py.File(output_path, "w") as f:
                print(f"Saving QPI results to: {output_path}")

                # Save main data with compression
                print("  Compressing and saving 'jdos'...")
                f.create_dataset(
                    "jdos", data=jdos, compression="gzip", compression_opts=6
                )

                # Save 1D coordinate arrays (compact representation)
                print("  Saving 1D q-coordinate arrays...")
                f.create_dataset("q1", data=q1_1d)
                f.create_dataset("q2", data=q2_1d)

                # Save metadata
                f.attrs["eta"] = self.eta
                f.attrs["energy_range"] = (
                    energy_range
                    if np.isscalar(energy_range)
                    else np.array(energy_range)
                )
                f.attrs["bands"] = (
                    "all" if bands == "all" else np.array(bands, dtype=int)
                )
                f.attrs["normalize"] = normalize
                f.attrs["grid_shape"] = [nkx, nky]

                # Save optional arrays
                if bvecs is not None:
                    f.create_dataset("bvecs", data=bvecs)
                    print("  Saved 'bvecs'.")
                if mask is not None:
                    f.create_dataset("mask", data=mask)
                    print("  Saved 'mask'.")

            # Final summary using pathlib
            from pathlib import Path

            file_size = Path(output_path).stat().st_size
            size_mb = file_size / (1024 * 1024)
            jdos_shape = jdos.shape
            print(f"✅ QPI data saved successfully to: {output_path}")
            print(f"   - File size: {size_mb:.2f} MB")
            print(f"   - JDOS shape: {jdos_shape}")
            print(f"   - Grid shape: ({nkx}, {nky})")

        return result

    @staticmethod
    def load_qpi_from_h5(file_path: str):
        """Load QPI results from an HDF5 file saved by `calculate_qpi`.

        Reconstructs the full result dictionary, including 2D meshgrids for q-coordinates.

        Parameters
        ----------
        file_path : str
            Path to the .h5 file saved by `calculate_qpi`.

        Returns
        -------
        result : dict
            {
                "jdos": np.ndarray,
                "qx_grid": np.ndarray,  # reconstructed 2D meshgrid
                "qy_grid": np.ndarray,
                "q1_grid": np.ndarray,
                "q2_grid": np.ndarray,
                "metadata": dict,       # all attrs + optional bvecs/mask
            }

        Notes
        -----
        The 2D grids are reconstructed using `np.meshgrid(..., indexing='ij')` from 1D coordinates.
        """
        import h5py

        with h5py.File(file_path, "r") as f:
            # Load main data
            jdos = f["jdos"][:]

            # Load 1D coordinates
            q1_1d = f["q1"][:]
            q2_1d = f["q2"][:]

            # Reconstruct 2D grids
            q1_grid, q2_grid = np.meshgrid(q1_1d, q2_1d, indexing="ij")

            # Build metadata dict
            metadata = {}
            for key, value in f.attrs.items():
                metadata[key] = value.copy() if hasattr(value, "copy") else value

            # Load optional datasets
            if "bvecs" in f:
                bvecs = f["bvecs"][:]
                metadata["bvecs"] = f["bvecs"][:]
            if "mask" in f:
                metadata["mask"] = f["mask"][:]

            # Compute physical qx, qy grids if bvecs available
            if bvecs is not None:
                qx_grid = q1_grid * bvecs[0, 0] + q2_grid * bvecs[1, 0]
                qy_grid = q1_grid * bvecs[0, 1] + q2_grid * bvecs[1, 1]
            else:
                qx_grid = q1_grid
                qy_grid = q2_grid

            result = {
                "jdos": jdos,
                "qx_grid": qx_grid,
                "qy_grid": qy_grid,
                "q1_grid": q1_grid,
                "q2_grid": q2_grid,
                "metadata": metadata,
            }

        return result
