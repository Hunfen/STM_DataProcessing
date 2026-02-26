from pathlib import Path

import h5py
import numpy as np
from scipy.fft import fft2, fftshift, ifft2

from stm_data_processing.utils.reciprocal_space import frac_to_real

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

    def _calculate_spectral_function(self, e_k, energy, bands="all", mask=None):
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

    def _calculate_jdos(
        self,
        ek2d: dict[str, np.ndarray],
        energy_range: float | np.ndarray | list[float],
        bands: str | list[int] = "all",
        normalize: bool = True,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Calculate JDOS-based QPI pattern using CPU (NumPy)."""
        k1_grid = ek2d["k1_grid"]
        k2_grid = ek2d["k2_grid"]
        e_k = ek2d["energies"]

        energy_array = np.atleast_1d(energy_range)
        is_scalar = np.isscalar(energy_range)

        nkx, nky = k1_grid.shape
        jdos_scan = np.empty((len(energy_array), nkx, nky))

        for i, energy in enumerate(energy_array):
            a_k = self._calculate_spectral_function(e_k, energy, bands=bands, mask=mask)

            a_r = fft2(a_k)
            jdos_q = np.real(ifft2(np.abs(a_r) ** 2))
            jdos_q = fftshift(jdos_q)
            if normalize and (max_val := np.max(jdos_q)) > 0:
                jdos_q /= max_val

            jdos_scan[i] = jdos_q

        return jdos_scan[0] if is_scalar else jdos_scan

    def _calculate_jdos_cuda(
        self,
        ek2d: dict[str, np.ndarray],
        energy_range: float | np.ndarray | list[float],
        bands: str | list[int] = "all",
        normalize: bool = True,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Calculate JDOS-based QPI pattern using GPU (CuPy).

        Parameters
        ----------
        ek2d : dict
            Dictionary returned by wannier90_ek2d_unit, containing:
            - 'energies': (num_wann, nk, nk) or (nk, nk) array of band energies.
            - 'k1_grid', 'k2_grid': (nk, nk) fractional k-grids (used for shape).
        energy_range : float or array-like
            Energy or energies at which to compute JDOS.
        bands : str or list[int], default "all"
            Which bands to include. "all" uses all bands.
        normalize : bool, default True
            Whether to normalize each JDOS map to its maximum.
        mask : np.ndarray or None
            Optional spatial mask applied in k-space.

        Returns
        -------
        np.ndarray
            JDOS scan array on CPU. Shape is (len(energy_range), nk, nk) or (nk, nk) if scalar.
        """
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy is required for GPU version.")
        import time

        e_k = ek2d["energies"]
        k1_grid = ek2d["k1_grid"]

        start_time = time.time()

        energy_array = cp.asarray(np.atleast_1d(energy_range))
        is_scalar = np.isscalar(energy_range)

        e_k_gpu = cp.asarray(e_k)
        mask_gpu = cp.asarray(mask) if mask is not None else None

        nkx, nky = k1_grid.shape
        n_energies = energy_array.size

        # === Fully automatic batch size determination ===
        e_k_bytes = e_k_gpu.nbytes
        mem_per_energy = e_k_bytes * 5  # rough estimate for intermediate arrays
        free_mem = cp.cuda.Device().mem_info[0]
        batch_size = max(1, int(free_mem * 0.75 / mem_per_energy))
        batch_size = min(batch_size, n_energies)

        # Additional safety limits for large grids
        if nkx >= 1024 and nky >= 1024:
            batch_size = min(batch_size, 8)
        elif nkx >= 512 and nky >= 512:
            batch_size = min(batch_size, 16)

        num_batches = (n_energies + batch_size - 1) // batch_size

        print(f"GPU QPI energy scan: {n_energies} energies on {nkx}x{nky} grid")
        print(
            f"  Using auto-determined batch size: {batch_size} (total batches: {num_batches})"
        )

        jdos_scan_gpu = cp.zeros((n_energies, nkx, nky), dtype=cp.float64)

        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, n_energies)
            batch_energies = energy_array[start:end]

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

            progress = (batch_idx + 1) / num_batches * 100
            free_mem, total_mem = cp.cuda.Device().mem_info
            used_mem = total_mem - free_mem
            print(
                f"  Progress: {progress:.1f}%, Batch {batch_idx + 1}/{num_batches}, "
                f"GPU memory: {used_mem / 1e9:.2f}/{total_mem / 1e9:.2f} GB"
            )

        jdos_scan = cp.asnumpy(jdos_scan_gpu)

        elapsed_time = time.time() - start_time
        print(f"GPU QPI energy scan complete in {elapsed_time:.2f} seconds")
        print(f"  Average time per energy: {elapsed_time / n_energies * 1000:.2f} ms")

        return jdos_scan[0] if is_scalar else jdos_scan

    def calculate_qpi(
        self,
        ek2d: dict[str, np.ndarray],
        energy_range: float | np.ndarray | list[float],
        q_range: tuple[float, float] | None = (-0.5, 0.5),
        bands: str | list[int] = "all",
        normalize: bool = True,
        mask: np.ndarray | None = None,
        use_gpu: bool = False,
        output_path: str | None = None,
    ) -> dict[str, np.ndarray]:
        """Unified QPI calculator with automatic CPU/GPU selection and optional HDF5 saving.

        Parameters
        ----------
        ek2d : dict[str, np.ndarray]
            Dictionary containing k-space band structure data, typically from
            `wannier90_ek2d_unit`, with required keys:
            - 'energies': (nbands, nkx, nky) array of band energies.
            - 'k1_grid', 'k2_grid': (nkx, nky) fractional k-grids.
            Optional key:
            - 'bvecs': (2, 2) reciprocal lattice vectors for physical q-space conversion.
        energy_range : float, list[float], or np.ndarray
            Target energy or energies (in eV) at which to compute the QPI.
        q_range : tuple[float, float] or None, optional (default=(-0.5, 0.5))
            Dimensionless q-range [q_min, q_max] for cropping the QPI result.
            If None, no cropping is applied.
        bands : str or list[int], default "all"
            Band indices to include in the calculation. Use "all" for all bands.
        normalize : bool, default True
            Whether to normalize the QPI intensity to unit maximum.
        mask : np.ndarray or None, optional
            Optional (nkx, nky) boolean or float mask applied in k-space.
        use_gpu : bool, default False
            Enable GPU acceleration if CuPy is available.
        output_path : str or None, optional
            If provided, save the result to an HDF5 file at this path.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary containing:
            - 'intensity': (nq, nq) or (len(energy_range), nq, nq) QPI intensity map(s).
            - 'qx_grid', 'qy_grid': Physical q-space grids (None if bvecs not provided).
            - 'q1_grid', 'q2_grid': Dimensionless q-space grids.
            - 'bvecs': (3, 3) array of reciprocal lattice vectors in 1/Å, or None.

        """
        e_k = ek2d["energies"]
        bvecs = ek2d.get("bvecs")

        if use_gpu and CUPY_AVAILABLE:
            qpi = self._calculate_jdos_cuda(ek2d, energy_range, bands, normalize, mask)
        else:
            if use_gpu and not CUPY_AVAILABLE:
                print("Warning: CuPy not available. Falling back to CPU.")
            qpi = self._calculate_jdos(ek2d, energy_range, bands, normalize, mask)

        q1_grid, q2_grid = self._k_to_q(ek2d["k1_grid"], ek2d["k2_grid"])

        qpi, q1_grid, q2_grid = self._extend_qpi(
            qpi, q1_grid, q2_grid, q_range[0], q_range[1]
        )

        qx_grid, qy_grid = frac_to_real(q1_grid, q2_grid, bvecs)

        result = {
            "intensity": qpi,
            "qx_grid": qx_grid,
            "qy_grid": qy_grid,
            "q1_grid": q1_grid,
            "q2_grid": q2_grid,
            "bvecs": bvecs,
        }

        if output_path is not None:
            self._save_qpi_to_h5(
                qpi=result,
                output_path=output_path,
                energy_range=energy_range,
                bands=bands,
                normalize=normalize,
                mask=mask,
                bvecs=bvecs,
            )

        return result

    @staticmethod
    def _extend_qpi(
        qpi_base: np.ndarray,
        q1_base: np.ndarray,
        q2_base: np.ndarray,
        qmin: float,
        qmax: float,
    ):
        """
        Extend QPI with strictly preserved q-density
        and exact [qmin, qmax) cropping.
        Supports both 2D (nk, nk) and 3D (nband, nk, nk) qpi_base.
        """

        # Ensure qpi_base is 3D
        was_2d = False
        if qpi_base.ndim == 2:
            qpi_base = qpi_base[np.newaxis, :, :]
            was_2d = True
        elif qpi_base.ndim != 3:
            raise ValueError(f"qpi_base must be 2D or 3D, got {qpi_base.ndim}D")

        nband, nq, _ = qpi_base.shape

        # -------- 1. extend ----------
        n_min = int(np.floor(qmin + 0.5))
        n_max = int(np.ceil(qmax - 0.5))
        shifts = np.arange(n_min, n_max + 1)
        nq_big = nq * len(shifts)

        qpi_big = np.zeros((nband, nq_big, nq_big))
        q1_big = np.zeros((nq_big, nq_big))
        q2_big = np.zeros((nq_big, nq_big))

        for ix, sx in enumerate(shifts):
            for iy, sy in enumerate(shifts):
                x0 = ix * nq
                x1 = (ix + 1) * nq
                y0 = iy * nq
                y1 = (iy + 1) * nq

                qpi_big[:, x0:x1, y0:y1] = qpi_base
                q1_big[x0:x1, y0:y1] = q1_base + sx
                q2_big[x0:x1, y0:y1] = q2_base + sy

        # -------- 2. crop ----------
        mask_x = (q1_big[:, 0] >= qmin) & (q1_big[:, 0] < qmax)
        mask_y = (q2_big[0, :] >= qmin) & (q2_big[0, :] < qmax)

        q1_ext = q1_big[np.ix_(mask_x, mask_y)]
        q2_ext = q2_big[np.ix_(mask_x, mask_y)]
        qpi_ext = qpi_big[:, mask_x, :][:, :, mask_y]

        # Squeeze back to 2D if input was 2D
        if was_2d:
            qpi_ext = qpi_ext[0]

        return qpi_ext, q1_ext, q2_ext

    @staticmethod
    def _k_to_q(
        k1_grid: np.ndarray,
        k2_grid: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert k-space grids to dimensionless q-space grids."""
        nkx, nky = k1_grid.shape
        dk1 = float(k1_grid[1, 0] - k1_grid[0, 0]) if nkx > 1 else 1.0
        dk2 = float(k2_grid[0, 1] - k2_grid[0, 0]) if nky > 1 else 1.0
        q1_vals = (np.arange(nkx) - nkx // 2) * dk1
        q2_vals = (np.arange(nky) - nky // 2) * dk2
        q1_grid, q2_grid = np.meshgrid(q1_vals, q2_vals, indexing="ij")
        return q1_grid, q2_grid

    @staticmethod
    def _save_qpi_to_h5(
        qpi: dict[str, np.ndarray],
        output_path: str,
        energy_range: float | np.ndarray | list[float],
        bands: str | list[int] = "all",
        normalize: bool = True,
        bvecs: np.ndarray | None = None,
        mask: np.ndarray | None = None,
        eta: float = 0.001,
        compression: str = "gzip",
        compression_opts: int = 6,
    ) -> None:
        """Save QPI results to an HDF5 file with compact storage.

        Parameters
        ----------
        qpi : dict[str, np.ndarray]
            Output dictionary from calculate_qpi, containing:
            - 'intensity', 'q1_grid', 'q2_grid'
            (qx_grid and qy_grid are not saved but can be reconstructed from q1/q2 and bvecs)
        output_path : str
            Path to save the HDF5 file.
        energy_range : float or array-like
            Energy (or energies) used in the calculation.
        bands : str or list of int, optional
            Band indices included in the calculation.
        normalize : bool, optional
            Whether the JDOS was normalized.
        bvecs : np.ndarray, optional
            Reciprocal lattice basis vectors (saved if provided).
        mask : np.ndarray, optional
            Real-space mask used in the calculation.
        eta : float, optional
            Spectral broadening parameter.
        compression : str, optional
            Compression algorithm for the JDOS dataset.
        compression_opts : int, optional
            Compression level for the JDOS dataset.
        """
        intensity = qpi["intensity"]
        q1_grid = qpi["q1_grid"]
        q2_grid = qpi["q2_grid"]

        # Extract 1D coordinates from dimensionless q-grids only
        nkx, nky = q1_grid.shape
        q1_1d = q1_grid[:, 0]
        q2_1d = q2_grid[0, :]

        with h5py.File(output_path, "w") as f:
            print(f"Saving QPI results to: {output_path}")

            f.create_dataset(
                "intensity",
                data=intensity,
                compression=compression,
                compression_opts=compression_opts,
            )

            f.create_dataset("q1", data=q1_1d)
            f.create_dataset("q2", data=q2_1d)

            f.attrs["eta"] = eta
            f.attrs["energy_range"] = (
                energy_range if np.isscalar(energy_range) else np.array(energy_range)
            )
            f.attrs["bands"] = "all" if bands == "all" else np.array(bands, dtype=int)
            f.attrs["normalize"] = normalize
            f.attrs["grid_shape"] = [nkx, nky]

            if bvecs is not None:
                f.create_dataset("bvecs", data=bvecs)
                print("  Saved 'bvecs'.")
            if mask is not None:
                f.create_dataset("mask", data=mask)
                print("  Saved 'mask'.")

        file_size = Path(output_path).stat().st_size
        size_mb = file_size / (1024 * 1024)
        intensity_shape = intensity.shape
        print(f"✅ QPI data saved successfully to: {output_path}")
        print(f"   - File size: {size_mb:.2f} MB")
        print(f"   - Intensity shape: {intensity_shape}")
        print(f"   - Grid shape: ({nkx}, {nky})")

    @staticmethod
    def load_qpi_from_h5(self, file_path: str) -> dict[str, np.ndarray | dict]:
        """Load QPI results from an HDF5 file saved by `calculate_qpi`.

        Reconstructs a dictionary that closely matches the output of `calculate_qpi`.

        Parameters
        ----------
        file_path : str
            Path to the .h5 file saved by `calculate_qpi`.

        Returns
        -------
        result : dict
            Dictionary with keys:
            - 'intensity': QPI intensity array.
            - 'qx_grid', 'qy_grid': Physical q-grids (None if bvecs not saved).
            - 'q1_grid', 'q2_grid': Dimensionless q-grids.
            - 'bvecs': Reciprocal lattice vectors or None.
            All other saved attributes (e.g., energy_range, bands, normalize) are included
            as top-level keys for consistency with `calculate_qpi`'s context.

        Notes
        -----
        The 2D grids are reconstructed using `np.meshgrid(..., indexing='ij')` from 1D coordinates.
        """
        with h5py.File(file_path, "r") as f:
            # Load main data
            intensity = f["intensity"][:]

            # Load 1D coordinates
            q1_1d = f["q1"][:]
            q2_1d = f["q2"][:]

            # Reconstruct 2D dimensionless grids
            q1_grid, q2_grid = np.meshgrid(q1_1d, q2_1d, indexing="ij")

            # Load optional bvecs
            bvecs = f["bvecs"][:] if "bvecs" in f else None
            qx_grid, qy_grid = frac_to_real(q1_grid, q2_grid, bvecs)

            # Build result dict matching calculate_qpi output structure
            result = {
                "intensity": intensity,
                "qx_grid": qx_grid,
                "qy_grid": qy_grid,
                "q1_grid": q1_grid,
                "q2_grid": q2_grid,
                "bvecs": bvecs,
            }

            # Add scalar/array attributes as top-level keys for convenience
            for key, value in f.attrs.items():
                if key == "energy_range":
                    # Convert back to scalar if originally scalar
                    if np.ndim(value) == 0:
                        result[key] = float(value)
                    else:
                        result[key] = np.array(value)
                elif key == "bands":
                    if isinstance(value, np.ndarray):
                        result[key] = value.tolist()
                    else:
                        result[key] = value  # "all"
                else:
                    result[key] = value

            # Load optional mask if present
            if "mask" in f:
                result["mask"] = f["mask"][:]

        return result
