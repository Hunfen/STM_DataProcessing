import logging
from functools import cached_property
from typing import Any

import numpy as np
import psutil

from stm_data_processing.config import BACKEND
from stm_data_processing.dft.wannier90.mlwf_ek2d import EK2DCalculator
from stm_data_processing.dft.wannier90.mlwf_hamiltonian import MLWFHamiltonian
from stm_data_processing.utils.miscellaneous import (
    fermi,
    fermi_cuda,
    frac_to_real_2d,
)

if BACKEND == "gpu":
    import cupy as cp
else:
    cp = None

logger = logging.getLogger(__name__)


class BareLindhardCalculator:
    """Class for calculating bare Lindhard susceptibility."""

    def __init__(self, hamiltonian: MLWFHamiltonian, nk: int = 256, eta: float = 5e-3):
        if not isinstance(hamiltonian, MLWFHamiltonian):
            raise TypeError(
                f"Expected MLWFHamiltonian, got {type(hamiltonian).__name__}"
            )

        self.ham = hamiltonian
        self.ek2d = EK2DCalculator(hamiltonian, nk)
        self.nk = nk
        self.eta = eta
        self.xp = cp if BACKEND == "gpu" else np

    @property
    def num_wann(self):
        """Number of Wannier functions."""
        return self.ham.num_wann

    def _init_k_points(self):
        """k_frac (cached)."""
        k_vals = np.linspace(-0.5, 0.5, self.nk, endpoint=False)
        k1, k2 = np.meshgrid(k_vals, k_vals, indexing="ij")

        self.k1_grid = k1
        self.k2_grid = k2
        self.q1_grid = k1.copy()
        self.q2_grid = k2.copy()

    @cached_property
    def eigen(self):
        """Eigen values for given Hamiltonian."""
        evals, evecs = self.ek2d.calculate_eigh()
        # evals reshape: (N_kpoints, num_wann)
        # evecs reshape: (N_kpoints, num_wann, num_wann) or None
        return evals.reshape((self.nk, self.nk, self.num_wann)), evecs.reshape(
            (self.nk, self.nk, self.num_wann, self.num_wann)
        )

    def _compute_bare_lindhard(
        self,
        ef: float = 0.0,
        temperature: float = 4.2,
        orb_sel: np.ndarray[int] | None = None,
        q_chunk_size: int = 8,
        precompute_weight: bool = False,
    ) -> np.ndarray:
        """
        Bare Lindhard susceptibility with vectorized computation.

        Parameters
        ----------
        ef : float, default 0.0
            Fermi level in eV.
        temperature : float, default 4.2
            Temperature in K.
        orb_sel : np.ndarray[int] | None
            Selected orbitals for projection.
        q_chunk_size : int, default 16
            Number of q1-points to process simultaneously.
        precompute_weight : bool, default False
            If True, pre-compute orbital weights (faster but uses more memory).
            If False, compute on-the-fly (slower but memory efficient).

        Returns
        -------
        np.ndarray
            Bare Lindhard susceptibility with shape (nk1, nk2).
        """
        evals, evecs = self.eigen
        nk1, nk2, num_wann = evals.shape
        num_kpts = nk1 * nk2

        # Compute Fermi distribution for all bands
        f_k = np.zeros((nk1, nk2, num_wann), dtype=np.float64)
        for n in range(num_wann):
            f_k[:, :, n] = fermi(evals[:, :, n], mu=ef, T=temperature)

        # Orbital selection mask
        if orb_sel is None:
            orb_mask = np.ones(num_wann, dtype=bool)
        else:
            orb_mask = np.zeros(num_wann, dtype=bool)
            orb_mask[orb_sel] = True

        # Pre-compute weights or not
        if precompute_weight:
            weight_mn = np.zeros((num_wann, num_wann, nk1, nk2), dtype=np.float64)
            for m in range(num_wann):
                for n in range(num_wann):
                    overlap = np.zeros((nk1, nk2), dtype=np.complex128)
                    for a in range(num_wann):
                        if orb_mask[a]:
                            overlap += evecs[:, :, a, m] * np.conj(evecs[:, :, a, n])
                    weight_mn[m, n, :, :] = np.abs(overlap) ** 2
            logger.info(f"Pre-computed weights: {weight_mn.nbytes / 1024**3:.2f} GB")
        else:
            weight_mn = None
            logger.info("On-the-fly weight computation (memory efficient)")

        # Initialize susceptibility
        chi_q = np.zeros((nk1, nk2), dtype=np.complex128)

        # Process q1-points in chunks
        for q1_start in range(0, nk1, q_chunk_size):
            q1_end = min(q1_start + q_chunk_size, nk1)

            # Create index arrays
            k1_idx = np.arange(nk1)
            k2_idx = np.arange(nk2)
            q1_idx = np.arange(q1_start, q1_end)

            # Process each band pair
            for m in range(num_wann):
                for n in range(num_wann):
                    f_m = f_k[:, :, m]
                    f_n = f_k[:, :, n]

                    if np.max(np.abs(f_m - f_n)) < 1e-10:
                        continue

                    # Get weight
                    if precompute_weight:
                        weight = weight_mn[m, n, :, :]
                    else:
                        # Compute on-the-fly
                        overlap = np.zeros((nk1, nk2), dtype=np.complex128)
                        for a in range(num_wann):
                            if orb_mask[a]:
                                overlap += evecs[:, :, a, m] * np.conj(
                                    evecs[:, :, a, n]
                                )
                        weight = np.abs(overlap) ** 2

                    eps_m = evals[:, :, m]
                    eps_n = evals[:, :, n]

                    # Process each q2
                    for q2 in range(nk2):
                        # Calculate k+q indices with periodic boundary conditions
                        k1q_idx = (k1_idx[:, None] + q1_idx[None, :]) % nk1
                        k2q_idx = (k2_idx + q2) % nk2

                        # Get shifted values
                        eps_n_shift = eps_n[k1q_idx, k2q_idx[:, None]]
                        f_n_shift = f_n[k1q_idx, k2q_idx[:, None]]

                        # Reshape for broadcasting
                        f_m_expanded = f_m[:, :, None]  # shape: (nk1, nk2, 1)
                        eps_m_expanded = eps_m[:, :, None]  # shape: (nk1, nk2, 1)
                        weight_expanded = weight[:, :, None]  # shape: (nk1, nk2, 1)

                        # Calculate contribution
                        numerator = f_m_expanded - f_n_shift  # shape: (nk1, nk2, n_q1)
                        denominator = eps_n_shift - eps_m_expanded + 1j * self.eta

                        # Sum over k1, k2
                        chi_contrib = np.sum(
                            weight_expanded * numerator / denominator, axis=(0, 1)
                        )

                        # Add to result
                        chi_q[q1_start:q1_end, q2] += chi_contrib / num_kpts
        return chi_q

    def _compute_bare_lindhard_cuda(
        self,
        ef: float = 0.0,
        temperature: float = 4.2,
        orb_sel: np.ndarray[int] | None = None,
        q_chunk_size: int = 8,
    ) -> np.ndarray:
        """
        GPU-accelerated bare Lindhard susceptibility calculation using CuPy.
        Vectorized over q-points and k-points, but loops over band-pairs for memory efficiency.

        Parameters
        ----------
        ef : float, default 0.0
            Fermi level in eV.
        temperature : float, default 4.2
            Temperature in K.
        orb_sel : np.ndarray[int] | None
            Selected orbital indices.
        q_chunk_size : int, default 8
            Number of q-points to process simultaneously (vectorized).

        Returns
        -------
        cp.ndarray
            Bare Lindhard susceptibility with shape (nk1, nk2).
        """

        # 1. Prepare data
        evals, evecs = self.eigen
        nk1, nk2, num_wann = evals.shape
        num_kpts = nk1 * nk2

        logger.info(
            f"Starting GPU bare Lindhard calculation: "
            f"k-grid ({nk1}x{nk2}), {num_wann} Wannier functions, "
            f"{num_kpts} k-points"
        )

        # Transfer to GPU
        evals_gpu = cp.asarray(evals, dtype=cp.float64)
        evecs_gpu = cp.asarray(evecs, dtype=cp.complex128)

        # Handle orbital selection
        if orb_sel is None:
            orb_indices = cp.arange(num_wann, dtype=cp.int32)
        else:
            orb_indices = cp.asarray(orb_sel, dtype=cp.int32)

        num_orb = len(orb_indices)
        logger.info(f"Using {num_orb} orbitals for susceptibility calculation")

        # 2. Pre-compute Fermi distribution
        f_k_gpu = fermi_cuda(evals_gpu, ef, temperature)

        # 3. Flatten k-grid for indexing
        k1_flat, k2_flat = cp.meshgrid(
            cp.arange(nk1, dtype=cp.int32),
            cp.arange(nk2, dtype=cp.int32),
            indexing="ij",
        )
        k_indices = cp.column_stack([k1_flat.ravel(), k2_flat.ravel()])
        k_flat_idx = k_indices[:, 0] * nk2 + k_indices[:, 1]  # (N_kpts,)

        # 4. Initialize result
        chi_q_gpu = cp.zeros((nk1, nk2), dtype=cp.complex128)
        q1_vals = cp.arange(nk1, dtype=cp.int32)
        q2_vals = cp.arange(nk2, dtype=cp.int32)
        q_grid = cp.column_stack([q1_vals.repeat(nk2), cp.tile(q2_vals, nk1)])
        num_qpts = len(q_grid)

        # ========== DEBUG: Check q-grid mapping ==========
        logger.debug("=== DEBUG: q-grid mapping ===")
        logger.debug(f"q_grid shape: {q_grid.shape}")
        logger.debug(f"q_grid first 5 rows: {cp.asnumpy(q_grid[:5])}")
        logger.debug(
            f"q_grid middle 5 rows: {cp.asnumpy(q_grid[nk1 * nk2 // 2 - 2 : nk1 * nk2 // 2 + 3])}"
        )
        logger.debug(f"q_grid last 5 rows: {cp.asnumpy(q_grid[-5:])}")

        # Find where q=(0,0) is in the grid
        q0_mask = (q_grid[:, 0] == 0) & (q_grid[:, 1] == 0)
        q0_idx = cp.where(q0_mask)[0]
        logger.debug(f"q=(0,0) index in flattened array: {cp.asnumpy(q0_idx)}")

        # Calculate expected center position
        expected_center = (nk1 // 2) * nk2 + (nk2 // 2)
        logger.debug(f"Expected center index (nk1//2, nk2//2): {expected_center}")
        logger.debug(f"Array shape: {chi_q_gpu.shape}")
        # ========== END DEBUG ==========

        # 5. Reshape data for easier indexing
        evals_flat = evals_gpu.reshape(num_kpts, num_wann)
        f_k_flat = f_k_gpu.reshape(num_kpts, num_wann)
        # FIX: evecs shape is (num_kpts, num_wann, num_wann) = (k, orb, band)
        # We need evecs_flat[k, :, n] to get band n's Wannier coefficients
        evecs_flat = evecs_gpu.reshape(num_kpts, num_wann, num_wann)

        # 6. Process q-points in chunks
        num_q_chunks = (num_qpts + q_chunk_size - 1) // q_chunk_size

        logger.info(
            f"Processing {num_qpts} q-points in {num_q_chunks} chunks "
            f"(chunk size: {q_chunk_size}), {num_wann * num_wann} band pairs per chunk"
        )

        for q_idx, q_start in enumerate(range(0, num_qpts, q_chunk_size)):
            q_end = min(q_start + q_chunk_size, num_qpts)
            q_chunk = q_grid[q_start:q_end]  # Shape: (N_q_chunk, 2)

            if q_idx % 10 == 0:
                logger.info(
                    f"Processing q-chunk {q_idx + 1}/{num_q_chunks} "
                    f"(q-points {q_start}:{q_end})"
                )

            # ========== DEBUG: Check first chunk q-values ==========
            if q_idx == 0:
                logger.debug("=== DEBUG: First q-chunk values ===")
                logger.debug(f"q_chunk (first 5): {cp.asnumpy(q_chunk[:5])}")
                logger.debug(f"q_start: {q_start}, q_end: {q_end}")
            # ========== END DEBUG ==========

            # Calculate k+q indices with periodic boundary
            kq_indices = (q_chunk[:, None, :] + k_indices[None, :, :]) % cp.array(
                [nk1, nk2], dtype=cp.int32
            )
            kq_flat_idx = kq_indices[:, :, 0] * nk2 + kq_indices[:, :, 1]

            # ========== DEBUG: Check k+q calculation for q=(0,0) ==========
            if q_idx == 0:  # First chunk should contain q=(0,0)
                # Find q=(0,0) in this chunk
                q0_in_chunk_mask = (q_chunk[:, 0] == 0) & (q_chunk[:, 1] == 0)
                if cp.any(q0_in_chunk_mask):
                    q0_local_idx = cp.where(q0_in_chunk_mask)[0][0]
                    logger.debug("=== DEBUG: k+q calculation for q=(0,0) ===")
                    logger.debug(f"q=(0,0) local index in chunk: {q0_local_idx}")
                    logger.debug(
                        f"kq_indices for q=(0,0), first 5 k-points: {cp.asnumpy(kq_indices[q0_local_idx, :5])}"
                    )
                    logger.debug(f"k_indices first 5: {cp.asnumpy(k_indices[:5])}")
                    # k+q should equal k when q=(0,0)
                    kq_should_equal_k = cp.all(kq_indices[q0_local_idx] == k_indices)
                    logger.debug(f"k+q == k when q=(0,0): {kq_should_equal_k}")
            # ========== END DEBUG ==========

            # Process each band pair (m, n)
            for m in range(num_wann):
                for n in range(num_wann):
                    # Get energies
                    eps_m = evals_flat[k_flat_idx, m]  # shape: (num_kpts,)
                    eps_n = evals_flat[kq_flat_idx, n]  # shape: (q_chunk, num_kpts)

                    # Get Fermi functions
                    f_m = f_k_flat[k_flat_idx, m]  # shape: (num_kpts,)
                    f_n = f_k_flat[kq_flat_idx, n]  # shape: (q_chunk, num_kpts)

                    # Get eigenvectors - FIX: correct orbital indexing
                    # evecs_flat[k, :, n] gives Wannier coefficients for band n at k-point k
                    # shape: (num_kpts, num_wann) -> (num_kpts, num_orb) after indexing
                    u_km = evecs_flat[k_flat_idx, :, m][
                        :, orb_indices
                    ]  # shape: (num_kpts, num_orb)
                    u_kqn = evecs_flat[kq_flat_idx, :, n][
                        :, :, orb_indices
                    ]  # shape: (q_chunk, num_kpts, num_orb)

                    # Calculate overlap - sum over orbitals
                    # u_km[None, :, :] -> (1, num_kpts, num_orb)
                    # u_kqn -> (q_chunk, num_kpts, num_orb)
                    overlap = cp.sum(
                        u_km[None, :, :] * cp.conj(u_kqn), axis=2
                    )  # shape: (q_chunk, num_kpts)
                    weight = cp.abs(overlap) ** 2

                    # Calculate Lindhard term
                    numerator = f_m[None, :] - f_n  # shape: (q_chunk, num_kpts)
                    denominator = (
                        eps_n - eps_m[None, :] + 1j * self.eta
                    )  # shape: (q_chunk, num_kpts)

                    # Sum over k-points
                    chi_contrib = cp.sum(
                        weight * numerator / denominator, axis=1
                    )  # shape: (q_chunk,)

                    # Add to result
                    chi_q_gpu.ravel()[q_start:q_end] += chi_contrib / num_kpts

            # Clean up memory
            del kq_indices, kq_flat_idx

        # ========== DEBUG: Check final chi_q result ==========
        logger.debug("=== DEBUG: Final chi_q result ===")
        chi_q_np = cp.asnumpy(chi_q_gpu)
        logger.debug(f"chi_q shape: {chi_q_np.shape}")
        logger.debug(f"chi_q dtype: {chi_q_np.dtype}")
        logger.debug(f"chi_q max |value|: {np.max(np.abs(chi_q_np))}")
        logger.debug(
            f"chi_q max |value| position: {np.unravel_index(np.argmax(np.abs(chi_q_np)), chi_q_np.shape)}"
        )
        logger.debug(f"chi_q[0, 0]: {chi_q_np[0, 0]}")
        logger.debug(f"chi_q[nk1//2, nk2//2]: {chi_q_np[nk1 // 2, nk2 // 2]}")
        logger.debug(
            f"chi_q center 5x5:\n{chi_q_np[nk1 // 2 - 2 : nk1 // 2 + 3, nk2 // 2 - 2 : nk2 // 2 + 3]}"
        )
        logger.debug(f"chi_q corner 5x5:\n{chi_q_np[:5, :5]}")

        # Check if maximum is at expected center
        max_pos = np.unravel_index(np.argmax(np.abs(chi_q_np)), chi_q_np.shape)
        expected_center = (nk1 // 2, nk2 // 2)
        logger.debug(f"Max at {max_pos}, expected center at {expected_center}")
        logger.debug(f"Max at center: {max_pos == expected_center}")
        logger.debug(f"q_grid[2080] = {cp.asnumpy(q_grid[2080])}")
        logger.debug(f"chi_q_gpu[32, 32] = {chi_q_gpu[32, 32]}")
        # ========== END DEBUG ==========

        logger.info("Bare Lindhard calculation completed.")

        return chi_q_gpu

    def calculate(
        self,
        q_range: tuple[float, float] | None = None,
        temperature: float = 4.2,
        orbital_select: list[int] | None = None,
        output_path: str | None = None,
    ):
        orb_sel = orbital_select or list(range(self.num_wann))
        orb_sel = np.array(orb_sel, dtype=int)

        # Get optimal chunk size for both CPU and GPU
        q_chunk_size = self._get_optimal_q_chunk_size(precompute_weight=False)

        if BACKEND == "gpu" and cp is not None:
            chi_q = self._compute_bare_lindhard_cuda(
                ef=0,
                temperature=temperature,
                orb_sel=orb_sel,
                q_chunk_size=q_chunk_size,
            )
            chi_q = cp.asnumpy(chi_q)
            cp.get_default_memory_pool().free_all_blocks()
        else:
            precompute_weight = False
            chi_q = self._compute_bare_lindhard(
                ef=0,
                temperature=temperature,
                orb_sel=orb_sel,
                q_chunk_size=q_chunk_size,
                precompute_weight=precompute_weight,
            )

        # Apply fftshift to center q=(0,0) for visualization
        # chi_q is indexed by integer q-points (0 to nk-1)
        # Physical q=(0,0) is at index (nk//2, nk//2)
        # fftshift moves q=(0,0) to the center of the array
        chi_q = np.fft.fftshift(chi_q, axes=(0, 1))

        # Generate coordinate grids that match the fftshifted data
        # q=0 should be at the center of the grid
        q_vals = np.linspace(-0.5, 0.5, self.nk, endpoint=False)
        q_vals = np.fft.fftshift(q_vals)  # Shift coordinate values to center q=0
        q1_grid, q2_grid = np.meshgrid(q_vals, q_vals, indexing="ij")

        # Convert to real-space q-grids if bvecs are available
        qx_grid, qy_grid = frac_to_real_2d(q1_grid, q2_grid, self.ham.bvecs)

        # Construct metadata
        metadata = {
            "module_type": "susceptibility",
            "eta": self.eta,
            "nq": self.nk,
            "bvecs": self.ham.bvecs,
            "note": "Results have been fftshifted to center q=(0,0)",
        }

        result: dict[str, Any] = {
            "data": chi_q,
            "q1_grid": q1_grid,
            "q2_grid": q2_grid,
            "qx_grid": qx_grid,
            "qy_grid": qy_grid,
            "metadata": metadata,
        }

        return result

    def _get_optimal_q_chunk_size(self, precompute_weight: bool = False) -> int:
        """
        Automatically determine optimal q_chunk_size based on system memory and computation parameters.
        Supports both CPU (RAM) and GPU (VRAM) backends.

        Parameters
        ----------
        precompute_weight : bool
            Whether to pre-compute weight matrices.

        Returns
        -------
        int
            Recommended q_chunk_size.
        """
        nk1, nk2 = self.nk, self.nk
        num_wann = self.num_wann
        num_kpts = nk1 * nk2

        # Check if GPU backend is active
        is_gpu = BACKEND == "gpu" and cp is not None

        if is_gpu:
            # GPU: Use CuPy to query available VRAM
            try:
                mem_info = cp.cuda.Device().mem_info
                available_memory = mem_info[0]  # Free memory in bytes
                total_memory = mem_info[1]  # Total memory in bytes
                logger.info(
                    f"GPU Memory - Total: {total_memory / 1024**3:.2f} GB, "
                    f"Free: {available_memory / 1024**3:.2f} GB"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to query GPU memory: {e}, using conservative estimate"
                )
                available_memory = 4 * 1024**3  # Conservative default: 4 GB
        else:
            # CPU: Use psutil to query available RAM
            available_memory = psutil.virtual_memory().available
            logger.info(f"System RAM Available: {available_memory / 1024**3:.1f} GB")

        # Estimate base memory requirements
        bytes_per_eval = 8 if is_gpu else 16
        base_memory = (
            num_kpts * num_wann * bytes_per_eval  # evals
            + num_kpts * num_wann * num_wann * 16  # evecs
            + num_kpts * num_wann * 8  # f_k
        )

        # Additional memory for pre-computed weights (CPU only typically)
        if precompute_weight and not is_gpu:
            base_memory += num_kpts * num_wann * num_wann * 8  # weight_mn

        # Temporary array memory per q-chunk
        # NEW: Only (q_chunk, kpts) arrays, no band-pair dimension
        # Arrays: overlap, weight, numerator, denominator, eps_n, f_n, u_kqn
        num_temp_arrays = 7
        bytes_per_element = 16  # complex128 (conservative)
        temp_memory_per_chunk = num_temp_arrays * num_kpts * bytes_per_element

        # Calculate memory available for temporary arrays
        if is_gpu:
            # Reserve 30% of free VRAM for CUDA context and other overhead
            memory_for_temp = available_memory * 0.7 - base_memory
        else:
            # Reserve 2 GB for system overhead
            memory_for_temp = available_memory - base_memory - 2 * 1024**3

        if memory_for_temp <= 0:
            logger.warning("Insufficient available memory, using minimum q_chunk_size")
            return 1

        # Calculate maximum possible q_chunk_size
        max_q_chunk = max(1, int(memory_for_temp / temp_memory_per_chunk))

        # Log memory breakdown for debugging
        logger.debug(
            f"Memory breakdown - Base: {base_memory / 1024**3:.2f} GB, "
            f"Per chunk: {temp_memory_per_chunk / 1024**3:.4f} GB, "
            f"Available for temp: {memory_for_temp / 1024**3:.2f} GB"
        )

        # Select strategy based on available memory and backend
        if is_gpu:
            # GPU: Now much more memory efficient, can use larger chunks
            if available_memory > 16 * 1024**3:  # > 16 GB VRAM
                optimal_size = min(256, max_q_chunk, nk1)
            elif available_memory > 8 * 1024**3:  # > 8 GB VRAM
                optimal_size = min(128, max_q_chunk, nk1)
            elif available_memory > 4 * 1024**3:  # > 4 GB VRAM
                optimal_size = min(64, max_q_chunk, nk1)
            else:  # <= 4 GB VRAM
                optimal_size = min(32, max_q_chunk, nk1)
        else:
            # CPU: Conservative, smaller chunks
            if available_memory > 64 * 1024**3:
                optimal_size = min(32, max_q_chunk, nk1)
            elif available_memory > 32 * 1024**3:
                optimal_size = min(16, max_q_chunk, nk1)
            elif available_memory > 16 * 1024**3:
                optimal_size = min(8, max_q_chunk, nk1)
            elif available_memory > 8 * 1024**3:
                optimal_size = min(4, max_q_chunk, nk1)
            else:
                optimal_size = min(2, max_q_chunk, nk1)

        # Safety cap: never exceed nk1
        optimal_size = min(optimal_size, nk1)

        logger.info(
            f"Backend: {'GPU' if is_gpu else 'CPU'}, "
            f"Available memory: {available_memory / 1024**3:.1f} GB, "
            f"Recommended q_chunk_size: {optimal_size}"
        )

        return optimal_size
