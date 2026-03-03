import time
from pathlib import Path

import h5py
import numpy as np
from scipy.special import expit

from stm_data_processing.utils.lattice_loader import BVecs

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


class SusceptibilityCalculator:
    """Class for calculating static Lindhard susceptibility chi0(q) from band structure data."""

    def __init__(self, eta: float = 20e-3):
        """Initialize susceptibility calculator.

        Parameters
        ----------
        eta : float
            Small imaginary broadening in the denominator (same units as energies);
            typical values range from 1e-3 to 2e-2 eV.
        """
        self.eta = eta

    @staticmethod
    def _fermi(e, mu: float = 0, T: float = 1.5):
        """Fermi-Dirac distribution function.

        Parameters
        ----------
        e : array_like
            Energy values.
        mu : float, optional
            Chemical potential. Default is 0.
        T : float, optional
            Temperature in Kelvin. Default is 1.5.

        Returns
        -------
        array_like
            Fermi-Dirac occupation numbers.
        """
        if T <= 1e-12:
            return (e < mu).astype(float)
        x = (e - mu) / T
        return expit(-x)

    @staticmethod
    def _window(e: np.ndarray, mu: float, Ec: float, delta: float) -> np.ndarray:
        """Smooth energy window W(|e-mu|) ~ 1 inside |e-mu|<Ec, decays outside.

        If Ec<=0 -> no window (all ones).

        Parameters
        ----------
        e : np.ndarray
            Energy values.
        mu : float
            Chemical potential.
        Ec : float
            Half-width of the energy window.
        delta : float
            Smoothness parameter for the window edge.

        Returns
        -------
        np.ndarray
            Window function values.
        """
        if Ec <= 0:
            return np.ones_like(e, dtype=np.float64)
        if delta <= 0:
            # hard cutoff
            return (np.abs(e - mu) <= Ec).astype(np.float64)
        x = (np.abs(e - mu) - Ec) / delta
        x = np.clip(x, -80.0, 80.0)
        return 1.0 / (1.0 + np.exp(x))

    @staticmethod
    def _extend_chi0(
        chi0_base: np.ndarray,
        q1_base: np.ndarray,
        q2_base: np.ndarray,
        qmin: float,
        qmax: float,
    ):
        """
        Extend susceptibility chi0 with strictly preserved q-density
        and exact [qmin, qmax) cropping.

        Parameters
        ----------
        chi0_base : np.ndarray
            (nk, nk) complex array.
        q1_base, q2_base : np.ndarray
            Base q-grid in fractional coordinates [-0.5, 0.5).
        qmin, qmax : float
            Desired q-range in fractional units.

        Returns
        -------
        chi0_ext, q1_ext, q2_ext
        """

        if chi0_base.ndim != 2:
            raise ValueError("chi0_base must be 2D (nk, nk)")

        nq, nq2 = chi0_base.shape
        if nq != nq2:
            raise ValueError("chi0_base must be square")

        # ---------- 1. determine integer shifts ----------
        n_min = int(np.floor(qmin + 0.5))
        n_max = int(np.ceil(qmax - 0.5))
        shifts = np.arange(n_min, n_max + 1)

        nq_big = nq * len(shifts)

        chi0_big = np.zeros((nq_big, nq_big), dtype=chi0_base.dtype)
        q1_big = np.zeros((nq_big, nq_big))
        q2_big = np.zeros((nq_big, nq_big))

        # ---------- 2. tile ----------
        for ix, sx in enumerate(shifts):
            for iy, sy in enumerate(shifts):
                x0 = ix * nq
                x1 = (ix + 1) * nq
                y0 = iy * nq
                y1 = (iy + 1) * nq

                chi0_big[x0:x1, y0:y1] = chi0_base
                q1_big[x0:x1, y0:y1] = q1_base + sx
                q2_big[x0:x1, y0:y1] = q2_base + sy

        # ---------- 3. crop ----------
        mask_x = (q1_big[:, 0] >= qmin) & (q1_big[:, 0] < qmax)
        mask_y = (q2_big[0, :] >= qmin) & (q2_big[0, :] < qmax)

        chi0_ext = chi0_big[np.ix_(mask_x, mask_y)]
        q1_ext = q1_big[np.ix_(mask_x, mask_y)]
        q2_ext = q2_big[np.ix_(mask_x, mask_y)]

        return chi0_ext, q1_ext, q2_ext

    def _calculate_chi0(
        self,
        energies: np.ndarray,
        f_k: np.ndarray,
        w_k: np.ndarray,
        q1_grid: np.ndarray,
        q2_grid: np.ndarray,
        matrix_element: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the static Lindhard susceptibility chi0(q) on an aligned q-grid.

        Uses array rolling to obtain E(k+q). Implemented in CPU serial mode.

        Parameters
        ----------
        energies : np.ndarray
            Band energies E_n(k) on a uniform fractional k-grid in [-0.5, 0.5).
            Shape: (nband, nk, nk).
        f_k : np.ndarray
            Fermi-Dirac occupation numbers f_n(k) = f(E_n(k)).
            Shape: (nband, nk, nk).
        w_k : np.ndarray
            Energy window factors W_n(k) = W(|E_n(k)-mu|).
            Shape: (nband, nk, nk).
        q1_grid : np.ndarray
            Precomputed aligned q-grid in fractional coordinates, wrapped to [-0.5, 0.5).
            Shape: (nk, nk).
        q2_grid : np.ndarray
            Precomputed aligned q-grid in fractional coordinates, wrapped to [-0.5, 0.5).
            Shape: (nk, nk).
        matrix_element : np.ndarray | None, optional
            Optional momentum matrix elements M_{mn}(k). If provided, they modulate the susceptibility.
            Shape: (nband, nband, nk, nk).

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            q1_grid, q2_grid : Same as input.
            chi0 : Static susceptibility chi0(q). Shape: (nk, nk), dtype: complex128.
        """
        _, nk1, nk2 = energies.shape

        # Validate input data
        if nk1 != nk2:
            raise ValueError(f"Expected square k-grid, got ({nk1}, {nk2})")

        if f_k.shape != energies.shape:
            raise ValueError(
                f"f_k shape {f_k.shape} must match energies shape {energies.shape}"
            )
        if w_k.shape != energies.shape:
            raise ValueError(
                f"w_k shape {w_k.shape} must match energies shape {energies.shape}"
            )

        chi0 = np.zeros((nk1, nk2), dtype=np.complex128)

        # Serial loops over q (but vectorized over k-grid within each q using roll)
        for iq1 in range(nk1):
            dq1 = iq1
            for iq2 in range(nk2):
                dq2 = iq2

                e_kq = np.roll(energies, shift=(-dq1, -dq2), axis=(1, 2))
                f_kq = np.roll(f_k, shift=(-dq1, -dq2), axis=(1, 2))
                w_kq = np.roll(w_k, shift=(-dq1, -dq2), axis=(1, 2))

                denom = (e_kq[None, ...] - energies[:, None, ...]) + (1j * self.eta)
                numer = f_k[:, None, ...] - f_kq[None, ...]
                win = w_k[:, None, ...] * w_kq[None, ...]

                if matrix_element is None:
                    chi0[iq1, iq2] = np.sum(win * numer / denom)
                else:
                    chi0[iq1, iq2] = np.sum(matrix_element * win * numer / denom)
        chi0 = np.fft.fftshift(chi0, axes=(0, 1))
        return chi0

    def _calculate_chi0_cuda(
        self,
        energies: np.ndarray,
        f_k: np.ndarray,
        w_k: np.ndarray,
        q1_grid: np.ndarray,
        q2_grid: np.ndarray,
        matrix_element: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute chi0(q) on GPU with reduced memory usage and optional matrix element support.

        Uses band-chunking to avoid O(nband^2 * nk^2) memory blowup.
        """
        nband, nk1, nk2 = energies.shape
        if nk1 != nk2:
            raise ValueError(f"Expected square k-grid, got ({nk1}, {nk2})")
        if f_k.shape != energies.shape or w_k.shape != energies.shape:
            raise ValueError("Shape mismatch in input arrays.")
        if matrix_element is not None and matrix_element.shape != (
            nband,
            nband,
            nk1,
            nk2,
        ):
            raise ValueError(
                f"matrix_element shape {matrix_element.shape} must be (nband, nband, nk1, nk2)"
            )

        print(
            f"🚀 Starting GPU calculation: nband={nband}, grid={nk1}x{nk2}, eta={self.eta:.3f}, "
            f"matrix_element={'enabled' if matrix_element is not None else 'disabled'}"
        )
        start_time = time.time()

        # Transfer to GPU
        energies_gpu = cp.asarray(energies)
        f_k_gpu = cp.asarray(f_k)
        w_k_gpu = cp.asarray(w_k)

        if matrix_element is not None:
            matrix_element_gpu = cp.asarray(matrix_element)
        else:
            matrix_element_gpu = None

        chi0_gpu = cp.zeros((nk1, nk2), dtype=cp.complex128)

        # Chunk size for (m, n) bands — tune based on GPU memory
        band_chunk_size = min(8, nband)  # e.g., 8 → max 64 band pairs per chunk

        total_q = nk1 * nk2
        processed_q = 0

        for iq1 in range(nk1):
            for iq2 in range(nk2):
                processed_q += 1
                if processed_q % max(1, total_q // 10) == 0:
                    elapsed = time.time() - start_time
                    est_total = elapsed / processed_q * total_q
                    print(
                        f"📦 Processed {processed_q}/{total_q} q-points "
                        f"(elapsed: {elapsed:.1f}s, est. remaining: {est_total - elapsed:.1f}s)"
                    )

                # Shifted quantities at k+q
                e_kq = cp.roll(energies_gpu, shift=(-iq1, -iq2), axis=(1, 2))
                f_kq = cp.roll(f_k_gpu, shift=(-iq1, -iq2), axis=(1, 2))
                w_kq = cp.roll(w_k_gpu, shift=(-iq1, -iq2), axis=(1, 2))

                chi0_q_val = cp.array(0.0 + 0.0j, dtype=cp.complex128)

                # Process (m, n) in chunks
                for m_start in range(0, nband, band_chunk_size):
                    m_end = min(m_start + band_chunk_size, nband)
                    for n_start in range(0, nband, band_chunk_size):
                        n_end = min(n_start + band_chunk_size, nband)

                        # Extract slices
                        e_n = energies_gpu[n_start:n_end, :, :]  # (nb2, nk1, nk2)
                        f_n = f_k_gpu[n_start:n_end, :, :]
                        w_n = w_k_gpu[n_start:n_end, :, :]

                        e_mq = e_kq[m_start:m_end, :, :]  # (nb1, nk1, nk2)
                        f_mq = f_kq[m_start:m_end, :, :]
                        w_mq = w_kq[m_start:m_end, :, :]

                        # Broadcast to (nb1, nb2, nk1, nk2)
                        denom = (
                            e_n[None, :, :, :] - e_mq[:, None, :, :]
                        ) + 1j * self.eta
                        numer = f_n[None, :, :, :] - f_mq[:, None, :, :]
                        win = w_n[None, :, :, :] * w_mq[:, None, :, :]

                        if matrix_element_gpu is not None:
                            M_mn = matrix_element_gpu[
                                m_start:m_end, n_start:n_end, :, :
                            ]  # (nb1, nb2, nk1, nk2)
                            term = M_mn * win * numer / denom
                        else:
                            term = win * numer / denom

                        # Sum over all bands and k-points
                        chi0_q_val += cp.sum(term)

                chi0_gpu[iq1, iq2] = chi0_q_val

                # Optional: free memory periodically (helps on small GPUs)
                if iq1 % 4 == 0 and iq2 == 0:
                    cp.get_default_memory_pool().free_all_blocks()

        print("💾 Transferring result from GPU...")
        chi0 = cp.asnumpy(chi0_gpu)
        chi0 = np.fft.fftshift(chi0, axes=(0, 1))

        total_time = time.time() - start_time
        print(f"✅ GPU computation completed in {total_time:.2f} seconds.")
        return chi0

    def calculate_chi0(
        self,
        ek2d: dict[str, np.ndarray],
        q_range: tuple[float, float] | None = (-0.5, 0.5),
        mu: float = 0.0,
        T: float = 0.0,
        Ec: float = 0.3,
        delta: float = 0.02,
        matrix_element: np.ndarray | None = None,
        use_gpu: bool = False,
        save_path: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the static Lindhard susceptibility chi0(q) on an aligned q-grid.

        Uses array rolling to obtain E(k+q).

        Parameters
        ----------
        ek2d : dict
            Dictionary returned by wannier90_ek2d containing:
            - 'energies': (nband, nk, nk) float array of band energies E_n(k)
            - 'k1_grid', 'k2_grid': (nk, nk) arrays of fractional k-coordinates in [-0.5, 0.5)
            - 'bvecs': reciprocal lattice vectors
        q_range : tuple[float, float] | None, optional
            Desired q-range in fractional units. Default is (-0.5, 0.5).
        mu : float, optional
            Chemical potential (in the same units as energies, e.g., eV). Default is 0.0.
        T : float, optional
            Thermal energy (in K). Setting T=0 corresponds to zero temperature (step-function occupation).
            Default is 0.0.
        Ec : float, optional
            Half-width of the energy window around mu (in eV). If Ec <= 0, no windowing is applied.
            Default is 0.3.
        delta : float, optional
            Smoothness parameter for the window edge (in eV). If delta <= 0, a hard cutoff is used.
            Default is 0.02.
        matrix_element : np.ndarray | None, optional
            Optional momentum matrix elements M_{mn}(k). If provided, they modulate the susceptibility.
            Shape: (nband, nband, nk, nk).
        use_gpu : bool, optional
            If True, use GPU acceleration with CUDA. Default is False.
        save_path : str | None, optional
            If provided, save the results to an HDF5 file at this path.

        Returns
        -------
        dict
            Dictionary containing:
            - 'chi0': (nk, nk) complex128 ndarray, static susceptibility chi0(q)
            - 'q1_grid', 'q2_grid': (nk, nk) ndarray, aligned q-grid in fractional coordinates
            - 'qx', 'qy': (nk, nk) ndarray, q-grid in real coordinates
            - 'bvecs': reciprocal lattice vectors
        """
        energies = np.asarray(ek2d["energies"])
        k1_grid = ek2d["k1_grid"]
        k2_grid = ek2d["k2_grid"]
        bvecs_obj = BVecs(bvecs_array=ek2d["bvecs"])

        if not (
            np.all(k1_grid >= -0.5)
            and np.all(k1_grid < 0.5)
            and np.all(k2_grid >= -0.5)
            and np.all(k2_grid < 0.5)
        ):
            raise ValueError(
                "Input k-grid must be in primitive Brillouin zone [-0.5, 0.5)"
            )

        _, nk1, nk2 = energies.shape
        if nk1 != nk2:
            raise ValueError(f"Expected square k-grid, got ({nk1}, {nk2})")

        # Build q-grid
        dk = 1.0 / nk1
        q = (np.arange(nk1) - nk1 // 2) * dk  # [-0.5, 0.5)
        q1_grid, q2_grid = np.meshgrid(q, q, indexing="ij")

        f_k = self._fermi(energies, mu=mu, T=T).astype(np.float64)
        w_k = self._window(energies, mu=mu, Ec=Ec, delta=delta).astype(np.float64)

        if use_gpu:
            if not CUPY_AVAILABLE:
                print("Warning: CuPy not available. Falling back to CPU.")
                use_gpu = False
            else:
                chi0 = self._calculate_chi0_cuda(
                    energies=energies,
                    f_k=f_k,
                    w_k=w_k,
                    q1_grid=q1_grid,
                    q2_grid=q2_grid,
                    matrix_element=matrix_element,
                )
        else:
            chi0 = self._calculate_chi0(
                energies=energies,
                f_k=f_k,
                w_k=w_k,
                q1_grid=q1_grid,
                q2_grid=q2_grid,
                matrix_element=matrix_element,
            )

        # Extend chi0 to desired q_range
        if q_range is not None:
            chi0, q1_grid, q2_grid = self._extend_chi0(
                chi0, q1_grid, q2_grid, q_range[0], q_range[1]
            )

        qx, qy = bvecs_obj.frac_to_real(q1_grid, q2_grid)

        result = {
            "chi0": chi0,
            "q1_grid": q1_grid,
            "q2_grid": q2_grid,
            "qx": qx,
            "qy": qy,
            "bvecs": bvecs_obj.get_bvecs(),
        }

        if save_path is not None:
            self._save_susceptibility_to_h5(
                chi0_result=result,
                output_path=save_path,
                mu=mu,
                T=T,
                Ec=Ec,
                delta=delta,
                matrix_element_used=(matrix_element is not None),
                use_gpu=use_gpu,
            )

        return result

    def _save_susceptibility_to_h5(
        self,
        chi0_result: dict[str, np.ndarray],
        output_path: str,
        mu: float = 0.0,
        T: float = 0.0,
        Ec: float = 0.3,
        delta: float = 0.02,
        matrix_element_used: bool = False,
        use_gpu: bool = False,
        compression: str = "gzip",
        compression_opts: int = 6,
    ) -> None:
        """Save susceptibility results to an HDF5 file with compact storage.

        Parameters
        ----------
        chi0_result : dict[str, np.ndarray]
            Output dictionary from calculate_chi0, containing:
            - 'chi0': susceptibility array
            - 'q1_grid', 'q2_grid': fractional q-grids
            - 'qx', 'qy': real-space q-grids
            - 'bvecs': reciprocal lattice vectors
        output_path : str
            Path to save the HDF5 file.
        mu : float, optional
            Chemical potential used in the calculation.
        T : float, optional
            Temperature used in the calculation.
        Ec : float, optional
            Energy window half-width used in the calculation.
        delta : float, optional
            Smoothness parameter for the window edge.
        matrix_element_used : bool, optional
            Whether matrix elements were used in the calculation.
        use_gpu : bool, optional
            Whether GPU acceleration was used.
        compression : str, optional
            Compression algorithm for the susceptibility dataset.
        compression_opts : int, optional
            Compression level for the susceptibility dataset.
        """
        chi0 = chi0_result["chi0"]
        q1_grid = chi0_result["q1_grid"]
        q2_grid = chi0_result["q2_grid"]
        bvecs = chi0_result["bvecs"]

        # Extract 1D coordinates from fractional q-grids
        nq1, nq2 = q1_grid.shape
        q1_1d = q1_grid[:, 0]
        q2_1d = q2_grid[0, :]

        with h5py.File(output_path, "w") as f:
            print(f"Saving susceptibility results to: {output_path}")

            # Save main datasets
            f.create_dataset(
                "chi0",
                data=chi0,
                compression=compression,
                compression_opts=compression_opts,
            )

            f.create_dataset("q1", data=q1_1d)
            f.create_dataset("q2", data=q2_1d)
            f.create_dataset("bvecs", data=bvecs)

            # Save attributes
            f.attrs["eta"] = self.eta
            f.attrs["mu"] = mu
            f.attrs["T"] = T
            f.attrs["Ec"] = Ec
            f.attrs["delta"] = delta
            f.attrs["matrix_element_used"] = matrix_element_used
            f.attrs["use_gpu"] = use_gpu
            f.attrs["grid_shape"] = [nq1, nq2]

        file_size = Path(output_path).stat().st_size
        size_mb = file_size / (1024 * 1024)
        chi0_shape = chi0.shape
        print(f"✅ Susceptibility data saved successfully to: {output_path}")
        print(f"   - File size: {size_mb:.2f} MB")
        print(f"   - chi0 shape: {chi0_shape}")
        print(f"   - Grid shape: ({nq1}, {nq2})")

    @staticmethod
    def load_susceptibility_from_h5(file_path: str) -> dict[str, np.ndarray | dict]:
        """Load susceptibility results from an HDF5 file saved by `_save_susceptibility_to_h5`.

        Reconstructs a dictionary that closely matches the output of `calculate_chi0`.

        Parameters
        ----------
        file_path : str
            Path to the .h5 file saved by `_save_susceptibility_to_h5`.

        Returns
        -------
        result : dict
            Dictionary with keys:
            - 'chi0': susceptibility array
            - 'q1_grid', 'q2_grid': fractional q-grids
            - 'qx', 'qy': real-space q-grids (reconstructed from q1/q2 and bvecs)
            - 'bvecs': reciprocal lattice vectors
            All saved attributes (e.g., eta, mu, T, Ec) are included as top-level keys.

        Notes
        -----
        The 2D grids are reconstructed using `np.meshgrid(..., indexing='ij')` from 1D coordinates.
        """
        with h5py.File(file_path, "r") as f:
            # Load main data
            chi0 = f["chi0"][:]
            bvecs = f["bvecs"][:]

            # Load 1D coordinates
            q1_1d = f["q1"][:]
            q2_1d = f["q2"][:]

            # Reconstruct 2D fractional grids
            q1_grid, q2_grid = np.meshgrid(q1_1d, q2_1d, indexing="ij")

            # Convert to real-space coordinates
            bvecs_obj = BVecs(bvecs_array=bvecs)
            qx, qy = bvecs_obj.frac_to_real(q1_grid, q2_grid)

            # Build result dict matching calculate_chi0 output structure
            result = {
                "chi0": chi0,
                "q1_grid": q1_grid,
                "q2_grid": q2_grid,
                "qx": qx,
                "qy": qy,
                "bvecs": bvecs,
            }

            # Add scalar/array attributes as top-level keys
            for key, value in f.attrs.items():
                if key == "grid_shape":
                    result[key] = list(value)
                else:
                    result[key] = value

        return result
