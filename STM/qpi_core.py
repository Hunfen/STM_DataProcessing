import numpy as np
from scipy.fft import fft2, ifft2, fftshift

try:
    import cupy as cp

    CUPY_AVAILABLE = True
    print("CuPy is available. GPU acceleration enabled for QPI calculations.")
except ImportError:
    CUPY_AVAILABLE = False
    print("CuPy not found. Using CPU-only mode for QPI calculations.")


class QPICalculator:
    """
    Class for calculating QPI (Quasiparticle Interference) from band contours.

    This class combines spectral function calculation with QPI-JDOS computation
    to analyze quasiparticle interference patterns from band structure data.
    """

    def __init__(self, eta=0.01):
        """
        Initialize QPI calculator.

        Parameters:
        -----------
        eta : float, optional
            Spectral broadening parameter, default is 0.01
        """
        self.eta = eta

    def calculate_spectral_function(self, e_k, energy, bands="all", mask=None):
        """
        Calculate the spectral function A(k, E).

        Parameters:
        -----------
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

        Returns:
        --------
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
            a_k = np.sum(a_k_per_band, axis=0) if bands == "all" else np.sum(a_k_per_band[bands], axis=0)

        # Apply mask if provided
        if mask is not None:
            mask = np.asarray(mask)

            # Ensure mask matches the spatial dimensions of the output
            if a_k.shape[-2:] != mask.shape:
                raise ValueError(f"Mask shape {mask.shape} does not match spatial dimensions of a_k {a_k.shape[-2:]}")

            # Apply mask (set masked values to zero)
            a_k = a_k * mask

        return a_k

    def calculate_qpi_jdos(self, a_k, k1_grid, k2_grid, bvecs=None, normalize=True):
        """
        Calculate QPI-JDOS in fractional coordinates, then optionally transform to real coordinates.

        Using convolution theorem: JDOS(q) = FFT^{-1}[ |FFT[A(k)]|^2 ]

        Parameters:
        -----------
        a_k : numpy.ndarray
            Spectral function A(k, E) calculated on fractional k-grid (shape: [nkx, nky])
        k1_grid : array-like
            Fractional k1 coordinate mesh grid points (shape: [nkx, nky])
        k2_grid : array-like
            Fractional k2 coordinate mesh grid points (shape: [nkx, nky])
        bvecs : numpy.ndarray, optional
            Reciprocal lattice vectors [[b1x, b1y], [b2x, b2y]]
            If provided, transforms q to real coordinates
        normalize : bool, optional
            If True, normalize JDOS to have maximum value of 1 (default: True)

        Returns:
        --------
        jdos_q : numpy.ndarray
            JDOS(q, E), normalized if normalize=True (shape: [nkx, nky])
        qx_grid : numpy.ndarray
            qx in real coordinates (if bvecs provided) or fractional coordinates (shape: [nkx, nky])
        qy_grid : numpy.ndarray
            qy in real coordinates (if bvecs provided) or fractional coordinates (shape: [nkx, nky])
        q1_grid : numpy.ndarray
            q in fractional coordinate 1 (shape: [nkx, nky])
        q2_grid : numpy.ndarray
            q in fractional coordinate 2 (shape: [nkx, nky])
        """
        nkx, nky = a_k.shape

        # Validate input shapes
        if k1_grid.shape != (nkx, nky) or k2_grid.shape != (nkx, nky):
            raise ValueError(f"Shape mismatch: a_k {a_k.shape}, k1_grid {k1_grid.shape}, k2_grid {k2_grid.shape}")

        # Calculate autocorrelation using convolution theorem
        a_r = fft2(a_k)
        power_spectrum = np.abs(a_r) ** 2
        jdos_q = np.real(ifft2(power_spectrum))

        # Apply fftshift to center zero frequency
        jdos_q = fftshift(jdos_q)

        # Normalize to maximum value of 1 (if requested)
        if normalize:
            max_val = np.max(jdos_q)
            if max_val > 0:
                jdos_q = jdos_q / max_val

        # Robustly infer dk1, dk2 from the provided mesh (assumes uniform grid)
        # Take one step along each axis
        dk1 = float(k1_grid[1, 0] - k1_grid[0, 0]) if nkx > 1 else 1.0
        dk2 = float(k2_grid[0, 1] - k2_grid[0, 0]) if nky > 1 else 1.0

        # After fftshift, index 0 corresponds to -N//2, center corresponds to 0
        q1_vals = (np.arange(nkx) - nkx // 2) * dk1
        q2_vals = (np.arange(nky) - nky // 2) * dk2

        q1_grid, q2_grid = np.meshgrid(q1_vals, q2_vals, indexing="ij")

        if bvecs is not None:
            # Transform to real coordinates
            qx_real = q1_grid * bvecs[0, 0] + q2_grid * bvecs[1, 0]
            qy_real = q1_grid * bvecs[0, 1] + q2_grid * bvecs[1, 1]
            return jdos_q, qx_real, qy_real, q1_grid, q2_grid
        else:
            return jdos_q, q1_grid, q2_grid, q1_grid, q2_grid

    def calculate_qpi_from_contour(
        self, k1_grid, k2_grid, e_k, energy, bvecs=None, bands="all", normalize=True, mask=None
    ):
        """
        Complete QPI calculation from band contour data.

        This method combines spectral function calculation and QPI-JDOS computation
        into a single workflow.

        Parameters:
        -----------
        k1_grid : array-like
            Fractional k1 coordinate mesh grid points (shape: [nkx, nky])
            This should be the k1_grid returned by calculate_contourmap()
        k2_grid : array-like
            Fractional k2 coordinate mesh grid points (shape: [nkx, nky])
            This should be the k2_grid returned by calculate_contourmap()
        e_k : array-like
            Energy values for each k-point (shape: [nkx, nky] or [num_wann, nkx, nky])
        energy : float
            Energy value at which to evaluate the spectral function
        bvecs : numpy.ndarray, optional
            Reciprocal lattice vectors [[b1x, b1y], [b2x, b2y]]
        bands : 'all' or list of int, optional
            Band indices to include in the spectral function calculation
        normalize : bool, optional
            If True, normalize JDOS to have maximum value of 1 (default: True)
        mask : array-like, optional
            Boolean mask to apply to k-points (True = include, False = exclude)
            Shape should match the spatial dimensions of e_k

        Returns:
        --------
        jdos_q : numpy.ndarray
            JDOS(q, E), normalized if normalize=True
        qx_grid : numpy.ndarray
            qx in real coordinates (if bvecs provided) or fractional coordinates
        qy_grid : numpy.ndarray
            qy in real coordinates (if bvecs provided) or fractional coordinates
        q1_grid : numpy.ndarray
            q in fractional coordinate 1
        q2_grid : numpy.ndarray
            q in fractional coordinate 2
        a_k : numpy.ndarray
            Spectral function A(k, E) used in the calculation
        """
        # Calculate spectral function
        a_k = self.calculate_spectral_function(e_k, energy, bands, mask)

        # Calculate QPI-JDOS
        if bvecs is not None:
            jdos_q, qx_grid, qy_grid, q1_grid, q2_grid = self.calculate_qpi_jdos(
                a_k, k1_grid, k2_grid, bvecs, normalize
            )
        else:
            jdos_q, qx_grid, qy_grid, q1_grid, q2_grid = self.calculate_qpi_jdos(a_k, k1_grid, k2_grid, None, normalize)

        return jdos_q, qx_grid, qy_grid, q1_grid, q2_grid, a_k

    def calculate_qpi_energy_scan(
        self, kx_mesh, ky_mesh, e_k, energy_range, bvecs=None, bands="all", normalize=True, mask=None
    ):
        """
        Calculate QPI for a range of energies.

        Parameters:
        -----------
        kx_mesh : array-like
            K-space x-coordinate mesh grid points (shape: [nkx, nky])
        ky_mesh : array-like
            K-space y-coordinate mesh grid points (shape: [nkx, nky])
        e_k : array-like
            Energy values for each k-point (shape: [nkx, nky] or [num_wann, nkx, nky])
        energy_range : array-like
            Array of energy values to scan
        bvecs : numpy.ndarray, optional
            Reciprocal lattice vectors [[b1x, b1y], [b2x, b2y]]
        bands : 'all' or list of int, optional
            Band indices to include in the spectral function calculation
        normalize : bool, optional
            If True, normalize JDOS to have maximum value of 1 (default: True)
        mask : array-like, optional
            Boolean mask to apply to k-points (True = include, False = exclude)
            Shape should match the spatial dimensions of e_k

        Returns:
        --------
        jdos_scan : numpy.ndarray
            JDOS(q, E) for each energy (shape: [len(energy_range), nkx, nky])
        qx_grid : numpy.ndarray
            qx grid (same for all energies)
        qy_grid : numpy.ndarray
            qy grid (same for all energies)
        q1_grid : numpy.ndarray
            q in fractional coordinate 1 (same for all energies)
        q2_grid : numpy.ndarray
            q in fractional coordinate 2 (same for all energies)
        """
        # Get grid shape from first calculation
        test_a_k = self.calculate_spectral_function(e_k, energy_range[0], bands, mask)
        nkx, nky = test_a_k.shape

        # Initialize array for storing results
        jdos_scan = np.zeros((len(energy_range), nkx, nky))

        # Calculate QPI for each energy
        for i, energy in enumerate(energy_range):
            jdos_q, qx_grid, qy_grid, q1_grid, q2_grid, _ = self.calculate_qpi_from_contour(
                kx_mesh, ky_mesh, e_k, energy, bvecs, bands, normalize, mask
            )
            jdos_scan[i] = jdos_q

        return jdos_scan, qx_grid, qy_grid, q1_grid, q2_grid

    def calculate_qpi_energy_scan_cuda(
        self, k1_grid, k2_grid, e_k, energy_range, bvecs=None, bands="all", normalize=True, batch_size=None, mask=None
    ):
        """
        Calculate QPI for a range of energies using CUDA acceleration.

        Parameters:
        -----------
        k1_grid : array-like
            Fractional k1 coordinate mesh grid points (shape: [nkx, nky])
        k2_grid : array-like
            Fractional k2 coordinate mesh grid points (shape: [nkx, nky])
        e_k : array-like
            Energy values for each k-point (shape: [nkx, nky] or [num_wann, nkx, nky])
        energy_range : array-like
            Array of energy values to scan
        bvecs : numpy.ndarray, optional
            Reciprocal lattice vectors [[b1x, b1y], [b2x, b2y]]
        bands : 'all' or list of int, optional
            Band indices to include in the spectral function calculation
        normalize : bool, optional
            If True, normalize JDOS to have maximum value of 1 (default: True)
        batch_size : int, optional
            Number of energies to process in each batch. If None, automatically determined.
        mask : array-like, optional
            Boolean mask to apply to k-points (True = include, False = exclude)
            Shape should match the spatial dimensions of e_k

        Returns:
        --------
        jdos_scan : numpy.ndarray
            JDOS(q, E) for each energy (shape: [len(energy_range), nkx, nky])
        qx_grid : numpy.ndarray
            qx grid (same for all energies)
        qy_grid : numpy.ndarray
            qy grid (same for all energies)
        q1_grid : numpy.ndarray
            q in fractional coordinate 1 (same for all energies)
        q2_grid : numpy.ndarray
            q in fractional coordinate 2 (same for all energies)

        Raises:
        -------
        RuntimeError
            If CuPy is not available for GPU acceleration.
        """
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy is not available. Cannot use GPU acceleration.")

        import time

        start_time = time.time()

        # Convert inputs to GPU arrays
        e_k_gpu = cp.asarray(e_k)
        energy_range_gpu = cp.asarray(energy_range)

        # Convert mask to GPU array if provided
        mask_gpu = cp.asarray(mask) if mask is not None else None

        # Get dimensions
        n_energies = len(energy_range)
        nkx, nky = k1_grid.shape

        # Initialize result array on GPU
        jdos_scan_gpu = cp.zeros((n_energies, nkx, nky), dtype=cp.float64)

        # Determine batch size if not provided
        if batch_size is None:
            # More conservative memory estimation
            mem_per_spectral_func = nkx * nky * 8  # bytes for single spectral function
            if e_k_gpu.ndim == 3:
                num_bands = e_k_gpu.shape[0]
                mem_per_spectral_func *= num_bands

            # Use at most 30% of GPU memory for working arrays (more conservative)
            gpu_memory = cp.cuda.Device().mem_info[0]  # Total GPU memory in bytes
            max_batch_size = int((gpu_memory * 0.3) / mem_per_spectral_func)

            # Ensure reasonable batch size
            batch_size = min(max(1, max_batch_size), n_energies)
            batch_size = max(batch_size, min(5, n_energies))  # Smaller minimum batch size

        num_batches = (n_energies + batch_size - 1) // batch_size

        print(f"GPU QPI energy scan: {n_energies} energies on {nkx}x{nky} grid")
        print(f"  Using batch size: {batch_size} (total batches: {num_batches})")

        # Pre-compute constants
        eta = self.eta
        pi = cp.pi

        # Process energies in batches
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, n_energies)

            # Process each energy individually to avoid memory spikes
            for i in range(batch_start, batch_end):
                energy = energy_range_gpu[i]

                # Calculate spectral function for single energy
                if e_k_gpu.ndim == 3:
                    # Multi-band case: e_k_gpu shape is (num_wann, nkx, nky)
                    # Calculate denominator: (E - E_k)^2 + η^2
                    denominator = (energy - e_k_gpu) ** 2 + eta**2

                    # Calculate spectral function per band
                    a_k_per_band = (1 / pi) * (eta / denominator)

                    # Sum over bands using ternary operator
                    if bands == "all":
                        a_k = cp.sum(a_k_per_band, axis=0)
                    else:
                        # Convert bands to cupy array if it's a list
                        if isinstance(bands, list):
                            bands_gpu = cp.array(bands)
                            a_k = cp.sum(a_k_per_band[bands_gpu, :, :], axis=0)
                        else:
                            a_k = cp.sum(a_k_per_band[bands, :, :], axis=0)
                else:
                    # Single band case: e_k_gpu shape is (nkx, nky)
                    # Calculate denominator: (E - E_k)^2 + η^2
                    denominator = (energy - e_k_gpu) ** 2 + eta**2

                    # Calculate spectral function
                    a_k = (1 / pi) * (eta / denominator)

                # Apply mask if provided
                if mask_gpu is not None:
                    a_k = a_k * mask_gpu

                # Calculate QPI-JDOS for this energy
                # Calculate autocorrelation using convolution theorem on GPU
                a_r = cp.fft.fft2(a_k)
                power_spectrum = cp.abs(a_r) ** 2
                jdos_q = cp.real(cp.fft.ifft2(power_spectrum))

                # Apply fftshift to center zero frequency
                jdos_q = cp.fft.fftshift(jdos_q)

                # Normalize if requested
                if normalize:
                    max_val = cp.max(jdos_q)
                    if max_val > 0:
                        jdos_q = jdos_q / max_val

                # Store result
                jdos_scan_gpu[i] = jdos_q

                # Clean up intermediate variables to free GPU memory
                del a_k, a_r, power_spectrum, jdos_q, denominator
                if e_k_gpu.ndim == 3:
                    del a_k_per_band

            # Print progress
            progress = (batch_idx + 1) / num_batches * 100
            memory_used = cp.cuda.Device().mem_info[1] / 1e9  # Used memory in GB
            memory_total = cp.cuda.Device().mem_info[0] / 1e9  # Total memory in GB
            print(
                f"  Progress: {progress:.1f}%, Batch {batch_idx + 1}/{num_batches}, "
                f"GPU memory: {memory_used:.2f}/{memory_total:.2f} GB"
            )

        # Calculate q-grids (same for all energies)
        # Robustly infer dk1, dk2 from the provided mesh (assumes uniform grid)
        dk1 = float(k1_grid[1, 0] - k1_grid[0, 0]) if nkx > 1 else 1.0
        dk2 = float(k2_grid[0, 1] - k2_grid[0, 0]) if nky > 1 else 1.0

        # After fftshift, index 0 corresponds to -N//2, center corresponds to 0
        q1_vals = (np.arange(nkx) - nkx // 2) * dk1
        q2_vals = (np.arange(nky) - nky // 2) * dk2

        q1_grid, q2_grid = np.meshgrid(q1_vals, q2_vals, indexing="ij")

        if bvecs is not None:
            # Transform to real coordinates
            qx_grid = q1_grid * bvecs[0, 0] + q2_grid * bvecs[1, 0]
            qy_grid = q1_grid * bvecs[0, 1] + q2_grid * bvecs[1, 1]
        else:
            qx_grid = q1_grid
            qy_grid = q2_grid

        # Convert results back to CPU
        jdos_scan = cp.asnumpy(jdos_scan_gpu)

        elapsed_time = time.time() - start_time
        print(f"GPU QPI energy scan complete in {elapsed_time:.2f} seconds")
        print(f"  Average time per energy: {elapsed_time / n_energies * 1000:.2f} ms")

        return jdos_scan, qx_grid, qy_grid, q1_grid, q2_grid
