import os
import time

import h5py
import numpy as np

# Try to import cupy for GPU support
try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

from .mlwf_hamiltonian import MLWFHamiltonian


class GPUContext:
    """
    Context manager for GPU operations to minimize data transfers.

    Usage:
    ```python
    with GPUContext(wh) as ctx:
        # All operations stay on GPU
        hk_gpu = ctx.hk_cuda(k, return_gpu=True)
        # ... other GPU operations
    ```
    """

    def __init__(self, wannier_hamiltonian: MLWFHamiltonian):
        """
        Initialize GPU context.

        Parameters
        ----------
        wannier_hamiltonian : MLWFHamiltonian
            The MLWFHamiltonian instance to use.
        """
        self.wh = wannier_hamiltonian

    def __enter__(self):
        """Enter the GPU context."""
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy is not available. Cannot use GPU context.")

        # Ensure GPU arrays are initialized
        self.wh._initialize_gpu_arrays()

        # Store original methods to restore later
        self.original_hk_cuda = self.wh.hk_cuda

        # Create wrapped methods that default to return_gpu=True
        def hk_cuda_wrapped(k_frac, return_gpu=True, **kwargs):
            return self.original_hk_cuda(k_frac, return_gpu=return_gpu, **kwargs)

        # Replace methods
        self.wh.hk_cuda = hk_cuda_wrapped

        return self.wh

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the GPU context and restore original methods."""
        # Restore original methods
        self.wh.hk_cuda = self.original_hk_cuda


def calculate_contourmap(
    mlwf_hamiltonian: MLWFHamiltonian,
    wlog_file: str | None = None,
    nk: int = 256,
    kmin: float = -1,
    kmax: float = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate band contour map from Wannier90 Hamiltonian.

    Parameters
    ----------
    mlwf_hamiltonian : MLWFHamiltonian
        The Wannier Hamiltonian instance with loaded data.
    wlog_file : str, optional
        Path to Wannier90 .wout or OpenMX .out file for reciprocal vectors.
        If None, uses auto-detected wlog_file from load_from_seedname().
    nk : int
        Number of k-points in each direction
    kmin, kmax : float
        Fractional k-space sampling range in units of reciprocal lattice vectors

    Returns
    -------
    e : (num_wann, nk, nk)
        Band energies contour map for all Wannier bands
    kx : (nk, nk) ndarray
        Real-space kx coordinates in 1/Ångström
    ky : (nk, nk) ndarray
        Real-space ky coordinates in 1/Ångström
    k1_grid : (nk, nk) ndarray
        Fractional k1 coordinates along b1 direction
    k2_grid : (nk, nk) ndarray
        Fractional k2 coordinates along b2 direction
    bvecs : (3, 3) ndarray
        Reciprocal lattice vectors in 1/Ångström

    Notes
    -----
    This method automatically uses GPU acceleration if available and enabled.
    Use set_use_gpu() to control GPU usage.
    """
    if mlwf_hamiltonian.h_list is None:
        raise ValueError(
            "Hamiltonian data not loaded. Call load_hr_file() or load_from_seedname() first."
        )

    # Determine wlog_file if not provided
    if wlog_file is None:
        if (
            hasattr(mlwf_hamiltonian, "wlog_file")
            and mlwf_hamiltonian.wlog_file is not None
        ):
            wlog_file = mlwf_hamiltonian.wlog_file
            print(f"  Using auto-detected wlog_file: {wlog_file}")
        elif (
            mlwf_hamiltonian.folder is not None
            and mlwf_hamiltonian.seedname is not None
        ):
            # Try to find wlog_file in folder
            wout_file = os.path.join(
                mlwf_hamiltonian.folder, f"{mlwf_hamiltonian.seedname}.wout"
            )
            out_file = os.path.join(
                mlwf_hamiltonian.folder, f"{mlwf_hamiltonian.seedname}.out"
            )

            if os.path.exists(wout_file):
                wlog_file = wout_file
                print(f"  Found and using wout file: {wout_file}")
            elif os.path.exists(out_file):
                wlog_file = out_file
                print(f"  Found and using out file: {out_file}")
            else:
                raise ValueError(
                    f"wlog_file not provided and no wout/out file found in {mlwf_hamiltonian.folder}. "
                    f"Please provide wlog_file or ensure {mlwf_hamiltonian.seedname}.wout or {mlwf_hamiltonian.seedname}.out exists."
                )
        else:
            raise ValueError(
                "wlog_file not provided and folder/seedname not set. "
                "Please provide wlog_file or set folder/seedname in initialization."
            )

    # Use GPU version if available and enabled
    if mlwf_hamiltonian.use_gpu:
        return calculate_contourmap_cuda(mlwf_hamiltonian, wlog_file, nk, kmin, kmax)

    # Original CPU implementation
    # Read reciprocal vectors
    bvecs = mlwf_hamiltonian.load_reciprocal_vectors(wlog_file)

    if bvecs is None:
        raise ValueError(
            f"Failed to load reciprocal vectors from {wlog_file}. "
            "Please check that the file contains reciprocal lattice vectors."
        )

    # Generate fractional k-grid
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

    # Convert fractional k to real-space coordinates
    kx = k1_grid * bvecs[0, 0] + k2_grid * bvecs[1, 0]
    ky = k1_grid * bvecs[0, 1] + k2_grid * bvecs[1, 1]

    print("Band structure calculation complete.")
    print(f"  Energy range: {e.min():.4f} → {e.max():.4f} eV")

    return e, kx, ky, k1_grid, k2_grid, bvecs


def calculate_contourmap_cuda(
    mlwf_hamiltonian: MLWFHamiltonian,
    wlog_file: str | None = None,
    nk: int = 256,
    kmin: float = -1,
    kmax: float = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate band contour map from Wannier90 Hamiltonian using CUDA acceleration.

    Parameters
    ----------
    mlwf_hamiltonian : MLWFHamiltonian
        The Wannier Hamiltonian instance with loaded data.
    wlog_file : str, optional
        Path to Wannier90 .wout or OpenMX .out file for reciprocal vectors.
        If None, uses auto-detected wlog_file from load_from_seedname().
    nk : int
        Number of k-points in each direction
    kmin, kmax : float
        Fractional k-space sampling range in units of reciprocal lattice vectors

    Returns
    -------
    e : (num_wann, nk, nk) ndarray
        Band energies for all bands
    kx : (nk, nk) ndarray
        Real-space kx coordinates in 1/Ångström
    ky : (nk, nk) ndarray
        Real-space ky coordinates in 1/Ångström
    k1_grid : (nk, nk) ndarray
        Fractional k1 coordinates along b1 direction
    k2_grid : (nk, nk) ndarray
        Fractional k2 coordinates along b2 direction
    bvecs : (3, 3) ndarray
        Reciprocal lattice vectors in 1/Ångström

    Raises
    ------
    RuntimeError
        If GPU acceleration is not available.
    """
    if mlwf_hamiltonian.h_list is None:
        raise ValueError(
            "Hamiltonian data not loaded. Call load_hr_file() or load_from_seedname() first."
        )

    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy is not available. Cannot use GPU acceleration.")

    # Determine wlog_file if not provided
    if wlog_file is None:
        if (
            hasattr(mlwf_hamiltonian, "wlog_file")
            and mlwf_hamiltonian.wlog_file is not None
        ):
            wlog_file = mlwf_hamiltonian.wlog_file
            print(f"  Using auto-detected wlog_file: {wlog_file}")
        elif (
            mlwf_hamiltonian.folder is not None
            and mlwf_hamiltonian.seedname is not None
        ):
            # Try to find wlog_file in folder
            wout_file = os.path.join(
                mlwf_hamiltonian.folder, f"{mlwf_hamiltonian.seedname}.wout"
            )
            out_file = os.path.join(
                mlwf_hamiltonian.folder, f"{mlwf_hamiltonian.seedname}.out"
            )

            if os.path.exists(wout_file):
                wlog_file = wout_file
                print(f"  Found and using wout file: {wout_file}")
            elif os.path.exists(out_file):
                wlog_file = out_file
                print(f"  Found and using out file: {out_file}")
            else:
                raise ValueError(
                    f"wlog_file not provided and no wout/out file found in {mlwf_hamiltonian.folder}. "
                    f"Please provide wlog_file or ensure {mlwf_hamiltonian.seedname}.wout or {mlwf_hamiltonian.seedname}.out exists."
                )
        else:
            raise ValueError(
                "wlog_file not provided and folder/seedname not set. "
                "Please provide wlog_file or set folder/seedname in initialization."
            )

    # Initialize GPU arrays if needed
    mlwf_hamiltonian._initialize_gpu_arrays()

    bvecs = mlwf_hamiltonian.load_reciprocal_vectors(wlog_file)

    # Check if reciprocal vectors were loaded successfully
    if bvecs is None:
        raise ValueError(
            f"Failed to load reciprocal vectors from {wlog_file}. "
            "Please check that the file contains reciprocal lattice vectors."
        )

    # Generate fractional k-grid on GPU
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
    gpu_memory = cp.cuda.Device().mem_info[0]  # Total GPU memory in bytes

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

    # Conservative estimate: use at most 30% of GPU memory for working arrays
    # We need to leave memory for other operations and system use
    max_batch_size = int((gpu_memory * 0.3) / mem_per_kpoint)

    # Ensure batch_size is reasonable: at least 256, at most total_kpoints
    # But also ensure we have multiple batches for progress reporting
    min_batch_for_progress = max(256, total_kpoints // 10)  # At least 10 batches
    batch_size = min(max(256, max_batch_size, min_batch_for_progress), total_kpoints)

    # If batch_size is too large (close to total_kpoints), reduce it
    if batch_size > total_kpoints * 0.8:
        batch_size = max(256, total_kpoints // 10)

    num_batches = (total_kpoints + batch_size - 1) // batch_size

    print(f"  Using batch size: {batch_size} (total batches: {num_batches})")
    print(f"  Memory per k-point: {mem_per_kpoint / 1024:.1f} KB")
    print(
        f"  Estimated memory for batch: {batch_size * mem_per_kpoint / (1024**3):.2f} GB"
    )

    # Pre-compute constants
    two_pi_i = 2j * cp.pi

    # Track timing for performance analysis
    import time

    total_hk_time = 0.0
    total_eig_time = 0.0

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
        phases = cp.exp(two_pi_i * dot_products)  # shape: (batch_size_current, nrpts)

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
        batch_time = time.time() - batch_start_time
        if (batch_idx + 1) % max(
            1, num_batches // 10
        ) == 0 or batch_idx == num_batches - 1:
            progress = (batch_idx + 1) / num_batches * 100
            memory_used = cp.cuda.Device().mem_info[1] / 1e9  # Used memory in GB
            memory_total = cp.cuda.Device().mem_info[0] / 1e9  # Total memory in GB
            print(
                f"  Progress: {progress:.1f}%, "
                f"Batch {batch_idx + 1}/{num_batches}, "
                f"Time: {batch_time:.1f}s (HK: {hk_time:.1f}s, Eig: {eig_time:.1f}s), "
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

    # Convert k-grid to real-space coordinates (CPU)
    k1_cpu = np.linspace(kmin, kmax, nk, endpoint=False)
    k2_cpu = np.linspace(kmin, kmax, nk, endpoint=False)
    k1_grid_cpu, k2_grid_cpu = np.meshgrid(k1_cpu, k2_cpu, indexing="ij")

    kx = k1_grid_cpu * bvecs[0, 0] + k2_grid_cpu * bvecs[1, 0]
    ky = k1_grid_cpu * bvecs[0, 1] + k2_grid_cpu * bvecs[1, 1]

    # Convert results back to CPU
    e = cp.asnumpy(e_gpu)

    print("Band energy contour map complete (GPU).")
    print(f"  Energy range: {e.min():.4f} → {e.max():.4f} eV")

    return e, kx, ky, k1_grid, k2_grid, bvecs


def _save_band_contourmap(
    energies: np.ndarray,
    kx: np.ndarray,
    ky: np.ndarray,
    filename: str,
    k1_grid: np.ndarray | None = None,
    k2_grid: np.ndarray | None = None,
    bvecs: np.ndarray | None = None,
) -> None:
    """
    Save band contourmap data to HDF5 file for high-speed storage.

    Parameters
    ----------
    energies : np.ndarray
        (num_wann, nk, nk) array of band energies (eV)
    kx : np.ndarray
        (nk, nk) array of real-space kx coordinates (1/Å)
    ky : np.ndarray
        (nk, nk) array of real-space ky coordinates (1/Å)
    filename : str
        Output filename. If not ending with .h5 or .hdf5, .h5 will be appended.
    k1_grid : np.ndarray, optional
        (nk, nk) array of fractional k1 coordinates along b1 direction
    k2_grid : np.ndarray, optional
        (nk, nk) array of fractional k2 coordinates along b2 direction
    bvecs : np.ndarray, optional
        (3, 3) array of reciprocal lattice vectors in 1/Ångström

    Notes
    -----
    The HDF5 file contains the following datasets and attributes:

    **Datasets:**
    - ``energies``: (num_wann, nk, nk) array of band energies
    - ``kx``: (nk, nk) array of kx coordinates
    - ``ky``: (nk, nk) array of ky coordinates
    - ``k1_grid``: (nk, nk) array of fractional k1 coordinates (if provided)
    - ``k2_grid``: (nk, nk) array of fractional k2 coordinates (if provided)

    **Attributes:**
    - ``num_wann``: Number of Wannier functions
    - ``nk``: Grid size
    - ``total_points``: Total number of k-points
    - ``units_energy``: "eV"
    - ``units_k``: "1/Å"
    - ``units_k_frac``: "reciprocal lattice units"
    - ``creation_date``: Timestamp
    - ``generator``: "wannier90_contourmap"
    - ``energies_shape``, ``kx_shape``, ``ky_shape``: Dataset shapes
    - ``bvecs``: Reciprocal lattice vectors as flattened array (if provided)
    - ``bvecs_shape``: Shape of bvecs array (if provided)

    This function uses HDF5 format which provides:
    - Faster I/O performance
    - Compression support (gzip compression is applied)
    - Hierarchical data organization
    - Efficient storage of large datasets
    """

    start_time = time.time()
    num_wann, nk, _ = energies.shape
    total_points = nk * nk

    print("Preparing to save band structure data to HDF5 file...")
    print(f"  Number of bands: {num_wann}")
    print(f"  k-grid size: {nk}x{nk}")
    print(f"  Total k-points: {total_points:,}")
    print(f"  Total data points: {num_wann * total_points:,}")

    # Ensure filename ends with .h5 or .hdf5
    if not (filename.lower().endswith(".h5") or filename.lower().endswith(".hdf5")):
        filename = filename + ".h5"
        print("  Note: Added .h5 extension to filename")

    # Create HDF5 file
    with h5py.File(filename, "w") as f:
        # Store raw data (preserve original dimensions)
        f.create_dataset(
            "energies", data=energies, compression="gzip", compression_opts=4
        )
        f.create_dataset("kx", data=kx, compression="gzip", compression_opts=4)
        f.create_dataset("ky", data=ky, compression="gzip", compression_opts=4)

        # Store fractional coordinates if provided
        if k1_grid is not None:
            f.create_dataset(
                "k1_grid", data=k1_grid, compression="gzip", compression_opts=4
            )
        if k2_grid is not None:
            f.create_dataset(
                "k2_grid", data=k2_grid, compression="gzip", compression_opts=4
            )

        # Store metadata as attributes
        f.attrs["num_wann"] = num_wann
        f.attrs["nk"] = nk
        f.attrs["total_points"] = total_points
        f.attrs["units_energy"] = "eV"
        f.attrs["units_k"] = "1/Å"
        f.attrs["units_k_frac"] = "reciprocal lattice units"
        f.attrs["creation_date"] = time.strftime("%Y-%m-%d %H:%M:%S")
        f.attrs["generator"] = "wannier90_contourmap"

        # Store shape information
        f.attrs["energies_shape"] = energies.shape
        f.attrs["kx_shape"] = kx.shape
        f.attrs["ky_shape"] = ky.shape
        if k1_grid is not None:
            f.attrs["k1_grid_shape"] = k1_grid.shape
        if k2_grid is not None:
            f.attrs["k2_grid_shape"] = k2_grid.shape

        # Store reciprocal lattice vectors if provided
        if bvecs is not None:
            # Store as flattened array for HDF5 attribute
            f.attrs["bvecs"] = bvecs.flatten()
            f.attrs["bvecs_shape"] = bvecs.shape
            print(f"  Saved reciprocal lattice vectors (shape: {bvecs.shape})")

    elapsed_time = time.time() - start_time
    print(f"Band structure data saved to {filename}")
    print("  Format: HDF5 (compressed with gzip)")
    print(f"  File size: {os.path.getsize(filename) / (1024 * 1024):.2f} MB")
    print(f"  Save time: {elapsed_time:.2f} seconds")


def wannier90_contourmap(
    hr_file: str | None = None,
    wlog_file: str | None = None,
    folder: str | None = None,
    seedname: str | None = None,
    nk: int = 256,
    kmin: float = -1,
    kmax: float = 1,
    use_gpu: bool = True,
    save_to_file: str | None = None,
) -> dict[str, np.ndarray]:
    """
    Quick analysis of Wannier90 band structure.

    Parameters
    ----------
    hr_file : str, optional
        Path to Wannier90 hr.dat file. Deprecated, use folder and seedname instead.
    wlog_file : str, optional
        Path to Wannier90 .wout file for reciprocal vectors.
        If None and folder/seedname are provided, uses seedname.wout from folder.
    folder : str, optional
        Path to folder containing Wannier90 files.
    seedname : str, optional
        Seedname for Wannier90 files.
    nk : int
        Number of k-points in each direction
    kmin : float, optional
        Minimum k-value in fractional coordinates. Default is -1.
    kmax : float, optional
        Maximum k-value in fractional coordinates. Default is 1.
    use_gpu : bool, optional
        Whether to use GPU acceleration if available. Default is True.
    save_to_file : str, optional
        If provided, save the results to the specified HDF5 file.
        The function automatically adds .h5 extension if not present.
        Default is None (no saving).

    Returns
    -------
    dict
        A dictionary containing the following keys:
        - 'energies': (num_wann, nk, nk) array of band energies (eV)
        - 'kx': (nk, nk) array of real-space kx coordinates (1/Å)
        - 'ky': (nk, nk) array of real-space ky coordinates (1/Å)
        - 'k1_grid': (nk, nk) array of fractional k1 coordinates along b1 direction
        - 'k2_grid': (nk, nk) array of fractional k2 coordinates along b2 direction
        - 'bvecs': (3, 3) array of reciprocal lattice vectors in 1/Ångström

    Notes
    -----
    Either provide hr_file (deprecated) or folder+seedname.
    If folder and seedname are provided, they take precedence over hr_file.
    """
    # Determine which initialization method to use
    if folder is not None and seedname is not None:
        # New method using folder and seedname
        wh = MLWFHamiltonian(folder=folder, seedname=seedname, use_gpu=use_gpu)

        # If wlog_file is not provided, let calculate_contourmap handle it
        # (it will use folder and seedname to construct the path)
        if wlog_file is None:
            print("Note: wlog_file not provided, will use seedname.wout from folder")
    elif hr_file is not None:
        # Legacy method using hr_file directly
        print(
            "Warning: Using deprecated hr_file parameter. Consider using folder and seedname instead."
        )
        wh = MLWFHamiltonian(use_gpu=use_gpu)
        wh.load_hr_file(hr_file)

        # For legacy method, wlog_file must be provided
        if wlog_file is None:
            raise ValueError(
                "wlog_file must be provided when using hr_file parameter. Or use folder and seedname instead."
            )
    else:
        raise ValueError("Either provide folder and seedname, or hr_file (deprecated).")

    # Calculate contour map
    e, kx, ky, k1_grid, k2_grid, bvecs = calculate_contourmap(
        wh, wlog_file, nk, kmin, kmax
    )

    # Save to HDF5 file if requested
    if save_to_file:
        _save_band_contourmap(e, kx, ky, save_to_file, k1_grid, k2_grid, bvecs)

    return {
        "energies": e,
        "kx": kx,
        "ky": ky,
        "k1_grid": k1_grid,
        "k2_grid": k2_grid,
        "bvecs": bvecs,
    }


def load_band_contourmap(
    filename: str,
) -> dict[str, np.ndarray]:
    """
    Load band contourmap data from HDF5 file saved by _save_band_contourmap.

    Parameters
    ----------
    filename : str
        Path to HDF5 file (.h5 or .hdf5 extension)

    Returns
    -------
    dict
        A dictionary containing the following keys:
        - 'energies': (num_wann, nk, nk) array of band energies (eV)
        - 'kx': (nk, nk) array of real-space kx coordinates (1/Å)
        - 'ky': (nk, nk) array of real-space ky coordinates (1/Å)
        - 'k1_grid': (nk, nk) array of fractional k1 coordinates along b1 direction (if saved)
        - 'k2_grid': (nk, nk) array of fractional k2 coordinates along b2 direction (if saved)
        - 'bvecs': (3, 3) array of reciprocal lattice vectors in 1/Ångström (if saved)

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If the file format is invalid or data is corrupted.

    Examples
    --------
    >>> # Load previously saved band structure data
    >>> data = load_band_contourmap("band_structure.h5")
    >>> print(
    ...     f"Loaded {data['energies'].shape[0]} bands on {data['energies'].shape[1]}x{data['energies'].shape[2]} k-grid"
    ... )
    """

    start_time = time.time()

    # Check if file exists
    if not os.path.exists(filename):
        raise FileNotFoundError(f"HDF5 file not found: {filename}")

    # Ensure file is HDF5 format
    if not (filename.lower().endswith(".h5") or filename.lower().endswith(".hdf5")):
        print(f"Warning: File '{filename}' does not have .h5 or .hdf5 extension")

    try:
        with h5py.File(filename, "r") as f:
            # Check if required datasets exist
            required_datasets = ["energies", "kx", "ky"]
            for dataset in required_datasets:
                if dataset not in f:
                    raise ValueError(
                        f"Required dataset '{dataset}' not found in HDF5 file"
                    )

            # Load required data
            energies = f["energies"][:]
            kx = f["kx"][:]
            ky = f["ky"][:]

            # Load optional fractional coordinates
            k1_grid = None
            k2_grid = None
            if "k1_grid" in f:
                k1_grid = f["k1_grid"][:]
            if "k2_grid" in f:
                k2_grid = f["k2_grid"][:]

            # Load reciprocal lattice vectors if saved
            bvecs = None
            if "bvecs" in f.attrs and "bvecs_shape" in f.attrs:
                bvecs_flat = f.attrs["bvecs"]
                bvecs_shape = f.attrs["bvecs_shape"]
                bvecs = bvecs_flat.reshape(bvecs_shape)

            # Validate data shape consistency
            num_wann, nk1, nk2 = energies.shape
            if kx.shape != (nk1, nk2):
                raise ValueError(
                    f"Inconsistent shapes: energies {energies.shape}, kx {kx.shape}"
                )
            if ky.shape != (nk1, nk2):
                raise ValueError(
                    f"Inconsistent shapes: energies {energies.shape}, ky {ky.shape}"
                )
            if k1_grid is not None and k1_grid.shape != (nk1, nk2):
                raise ValueError(
                    f"Inconsistent shapes: energies {energies.shape}, k1_grid {k1_grid.shape}"
                )
            if k2_grid is not None and k2_grid.shape != (nk1, nk2):
                raise ValueError(
                    f"Inconsistent shapes: energies {energies.shape}, k2_grid {k2_grid.shape}"
                )

            # Optional: Load metadata
            metadata = {}
            for key in f.attrs:
                metadata[key] = f.attrs[key]

            print(f"Loaded band structure data from {filename}")
            print(f"  Number of bands: {num_wann}")
            print(f"  k-grid size: {nk1}x{nk2}")
            print(f"  Total k-points: {nk1 * nk2:,}")

            # Check if fractional coordinates were loaded
            if k1_grid is not None and k2_grid is not None:
                print("  Fractional coordinates: Loaded")
            elif k1_grid is not None or k2_grid is not None:
                print("  Fractional coordinates: Partially loaded")
            else:
                print("  Fractional coordinates: Not found in file (old format)")

            # Check if reciprocal vectors were loaded
            if bvecs is not None:
                print(f"  Reciprocal lattice vectors: Loaded (shape: {bvecs.shape})")
            else:
                print("  Reciprocal lattice vectors: Not found in file (old format)")

            # Print some useful metadata
            if "units_energy" in metadata:
                print(f"  Energy units: {metadata['units_energy']}")
            if "units_k" in metadata:
                print(f"  k-space units: {metadata['units_k']}")
            if "units_k_frac" in metadata:
                print(f"  Fractional k-space units: {metadata['units_k_frac']}")
            if "creation_date" in metadata:
                print(f"  Created: {metadata['creation_date']}")

            # Validate energy range
            print(f"  Energy range: {energies.min():.4f} → {energies.max():.4f} eV")

    except Exception as e:
        raise ValueError(f"Failed to load HDF5 file '{filename}': {str(e)}") from e

    elapsed_time = time.time() - start_time
    print(f"  Load time: {elapsed_time:.2f} seconds")
    return {
        "energies": energies,
        "kx": kx,
        "ky": ky,
        "k1_grid": k1_grid,
        "k2_grid": k2_grid,
        "bvecs": bvecs,
    }
