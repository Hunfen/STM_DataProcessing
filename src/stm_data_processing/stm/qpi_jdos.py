import numpy as np
from scipy.fft import fft2, fftshift, ifft2

from stm_data_processing.utils.miscellaneous import extend_qpi, frac_to_real_2d, k_to_q

try:
    import cupy as cp

    CUPY_AVAILABLE = True
    print("CuPy is available. GPU acceleration enabled for QPI calculations.")
except ImportError:
    CUPY_AVAILABLE = False
    print("CuPy not found. Using CPU-only mode for QPI calculations.")


class JDOSQPI:
    """Class for calculating JDOS QPI (Quasiparticle Interference).

    This class uses precalculated eigen values for joint-DOS QPI computation.
    """

    def __init__(self, eta=0.001):
        """Initialize QPI calculator.

        Parameters
        ----------
        eta : float, optional
            Spectral broadening parameter, default is 0.001

        """
        self.eta = eta

    def _calculate_spectral_function(self, e_k, energy, orbitals="all", mask=None):
        """Calculate the spectral function A(k, E).

        Parameters
        ----------
        e_k : array-like
            Energy values for each k-point (shape: [nkx, nky] or [num_wann, nkx, nky])
        energy : float
            Energy value at which to evaluate the spectral function
        orbitals : 'all' or list of int, optional
            Band indices to include in the spectral function calculation:
            - 'all' (default): Sum over all orbitals
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
                if orbitals == "all"
                else np.sum(a_k_per_band[orbitals], axis=0)
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

    def _compute_jdos(
        self,
        ek2d: dict[str, np.ndarray],
        energy_range: float | np.ndarray | list[float],
        orbitals: str | list[int] = "all",
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
            a_k = self._calculate_spectral_function(
                e_k, energy, orbitals=orbitals, mask=mask
            )

            a_r = fft2(a_k)
            jdos_q = np.real(ifft2(np.abs(a_r) ** 2))
            jdos_q = fftshift(jdos_q)
            if normalize and (max_val := np.max(jdos_q)) > 0:
                jdos_q /= max_val

            jdos_scan[i] = jdos_q

        return jdos_scan[0] if is_scalar else jdos_scan

    def _compute_jdos_cuda(
        self,
        ek2d: dict[str, np.ndarray],
        energy_range: float | np.ndarray | list[float],
        orbitals: str | list[int] = "all",
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
        orbitals : str or list[int], default "all"
            Which orbital to include. "all" uses all orbitals.
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
                if orbitals == "all":
                    a_k_batch = cp.sum(a_k_per_band, axis=1)
                else:
                    orbitals_idx = (
                        cp.array(orbitals) if isinstance(orbitals, list) else orbitals
                    )
                    a_k_batch = cp.sum(a_k_per_band[:, orbitals_idx, :, :], axis=1)

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

    def calculate(
        self,
        ek2d: dict[str, np.ndarray],
        energy_range: float | np.ndarray | list[float],
        q_range: tuple[float, float] | None = (-0.5, 0.5),
        orbitals: str | list[int] = "all",
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
            - 'energies': (norbitals, nkx, nky) array of band energies.
            - 'k1_grid', 'k2_grid': (nkx, nky) fractional k-grids.
            Optional key:
            - 'bvecs': (2, 2) reciprocal lattice vectors for physical q-space conversion.
        energy_range : float, list[float], or np.ndarray
            Target energy or energies (in eV) at which to compute the QPI.
        q_range : tuple[float, float] or None, optional (default=(-0.5, 0.5))
            Dimensionless q-range [q_min, q_max] for cropping the QPI result.
            If None, no cropping is applied.
        orbitals : str or list[int], default "all"
            Band indices to include in the calculation. Use "all" for all orbitals.
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
            Dictionary containing the cropped/extended QPI result:
            - 'intensity': (nq, nq) or (len(energy_range), nq, nq) QPI intensity map(s),
              after optional q-range cropping.
            - 'q1_grid', 'q2_grid': (nq, nq) dimensionless q-space grids corresponding to `intensity`.
            - 'qx_grid', 'qy_grid': (nq, nq) physical q-space grids in 1/Å;
              `None` if `bvecs` was not provided in `ek2d`.
            - 'bvecs': (2, 2) or (3, 3) reciprocal lattice vectors in 1/Å as provided in `ek2d`,
              or `None` if not present.

        """
        e_k = ek2d["energies"]
        bvecs = ek2d.get("bvecs")

        if use_gpu and CUPY_AVAILABLE:
            qpi = self._compute_jdos_cuda(ek2d, energy_range, orbitals, normalize, mask)
        else:
            if use_gpu and not CUPY_AVAILABLE:
                print("Warning: CuPy not available. Falling back to CPU.")
            qpi = self._compute_jdos(ek2d, energy_range, orbitals, normalize, mask)

        q1_grid, q2_grid = k_to_q(ek2d["k1_grid"], ek2d["k2_grid"])

        result = {
            "intensity": qpi,
            "q1_grid": q1_grid,
            "q2_grid": q2_grid,
        }

        if output_path is not None:
            self._save_qpi_to_h5(
                qpi=result,
                output_path=output_path,
                energy_range=energy_range,
                orbitals=orbitals,
                normalize=normalize,
                mask=mask,
                bvecs=bvecs,
            )

        if q_range is not None:
            qpi, q1_grid, q2_grid = extend_qpi(
                qpi, q1_grid, q2_grid, q_range[0], q_range[1]
            )

        qx_grid, qy_grid = frac_to_real_2d(q1_grid, q2_grid, bvecs)

        result = {
            "intensity": qpi,
            "qx_grid": qx_grid,
            "qy_grid": qy_grid,
            "q1_grid": q1_grid,
            "q2_grid": q2_grid,
            "bvecs": bvecs,
        }
        return result
