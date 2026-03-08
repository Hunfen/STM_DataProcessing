import logging
import time
from pathlib import Path

import h5py
import numpy as np

from stm_data_processing.dft.wannier90.mlwf_hamiltonian import MLWFHamiltonian
from stm_data_processing.utils.miscellaneous import frac_to_real_2d

logger = logging.getLogger(__name__)


class EK2DIO:
    """
    A class for handling I/O operations of 2D band structure data.

    This class provides methods to save and load band structure contour maps
    to/from HDF5 files, including metadata for reinitialization.
    """

    @staticmethod
    def _extend_ek2d_static(
        ek2d: dict[str, np.ndarray],
        k_range: tuple[float, float],
    ) -> dict[str, np.ndarray]:
        """
        Extend a band contour map from the primitive Brillouin zone [-0.5, 0.5)
        to an arbitrary fractional k-space window [kmin, kmax).

        Parameters
        ----------
        ek2d : dict[str, np.ndarray]
            Dictionary containing:
            - 'energies': (num_wann, nk, nk) array of band energies.
            - 'k1_grid': (nk, nk) array of fractional k1 coordinates.
            - 'k2_grid': (nk, nk) array of fractional k2 coordinates.
        k_range : tuple[float, float]
            (kmin, kmax) defining the target fractional k-space window [kmin, kmax).

        Returns
        -------
        dict[str, np.ndarray]
            Extended band structure data with the same structure as input.
        """
        e_base = ek2d["energies"]
        k1_base = ek2d["k1_grid"]
        k2_base = ek2d["k2_grid"]

        kmin, kmax = k_range
        nk = k1_base.shape[0]

        n_min = int(np.floor(kmin + 0.5))
        n_max = int(np.ceil(kmax - 0.5))

        shifts = np.arange(n_min, n_max + 1)
        n_shifts = len(shifts)

        nband = e_base.shape[0]
        nk_big = nk * n_shifts

        e_big = np.zeros((nband, nk_big, nk_big), dtype=e_base.dtype)
        k1_big = np.zeros((nk_big, nk_big), dtype=k1_base.dtype)
        k2_big = np.zeros((nk_big, nk_big), dtype=k2_base.dtype)

        for ix, sx in enumerate(shifts):
            for iy, sy in enumerate(shifts):
                x0 = ix * nk
                x1 = (ix + 1) * nk
                y0 = iy * nk
                y1 = (iy + 1) * nk

                e_big[:, x0:x1, y0:y1] = e_base
                k1_big[x0:x1, y0:y1] = k1_base + sx
                k2_big[x0:x1, y0:y1] = k2_base + sy

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

    @staticmethod
    def save_ek2d(
        energies: np.ndarray,
        k1_grid: np.ndarray,
        k2_grid: np.ndarray,
        filename: str,
        mlwf_hamiltonian: MLWFHamiltonian,
    ) -> None:
        """
        Save band structure data to an HDF5 file.

        Parameters
        ----------
        energies : np.ndarray
            (num_wann, nk, nk) array of band energies in eV.
        k1_grid : np.ndarray
            (nk, nk) array of fractional k1 coordinates.
        k2_grid : np.ndarray
            (nk, nk) array of fractional k2 coordinates.
        filename : str
            Output file path. `.h5` extension is appended if not present.
        mlwf_hamiltonian : MLWFHamiltonian
            MLWFHamiltonian instance. bvecs, folder, and seedname are
            automatically extracted from this object.
        """
        start_time = time.time()
        num_wann, nk, _ = energies.shape
        total_points = nk * nk

        logger.info("Preparing to save band structure data to HDF5 file...")
        logger.info("  Number of bands: %d", num_wann)
        logger.info("  k-grid size: %dx%d", nk, nk)
        logger.info("  Total k-points: %d", total_points)

        if not filename.lower().endswith((".h5", ".hdf5")):
            filename += ".h5"
            logger.info("  Note: Added .h5 extension to filename")

        bvecs = mlwf_hamiltonian.bvecs
        folder = mlwf_hamiltonian.folder
        seedname = mlwf_hamiltonian.seedname

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

            if bvecs is not None:
                f.create_dataset("bvecs", data=bvecs)

            f.attrs["num_wann"] = num_wann
            f.attrs["nk"] = nk
            f.attrs["total_points"] = total_points
            f.attrs["units_energy"] = "eV"
            f.attrs["units_k_frac"] = "reciprocal lattice units"
            f.attrs["creation_date"] = time.strftime("%Y-%m-%d %H:%M:%S")
            f.attrs["generator"] = "EK2DCalculator"

            if folder is not None:
                f.attrs["folder"] = str(Path(folder).resolve())
            if seedname is not None:
                f.attrs["seedname"] = seedname

        elapsed_time = time.time() - start_time
        logger.info("Band structure data saved to %s", filename)
        logger.info("  Save time: %.2f seconds", elapsed_time)

    @staticmethod
    def load_ek2d(
        filename: str,
        k_range: tuple[float, float] | None = None,
    ) -> dict[str, np.ndarray | None]:
        """
        Load band structure data from an HDF5 file.

        Parameters
        ----------
        filename : str
            Path to HDF5 file containing band structure data.
        k_range : tuple[float, float], optional
            (kmin, kmax) defining the target fractional k-space window [kmin, kmax).
            If None, only the stored data is returned without extension.

        Returns
        -------
        dict[str, np.ndarray | None]
            A dictionary containing:
            - 'energies': (num_wann, Nk1, Nk2) array of band energies in eV.
            - 'kx', 'ky': (Nk1, Nk2) arrays of real-space k-coordinates in 1/Å, or None.
            - 'k1_grid', 'k2_grid': (Nk1, Nk2) arrays of fractional k-coordinates.
            - 'bvecs': (3, 3) array of reciprocal lattice vectors in 1/Å, or None.
            - 'metadata': dict of file attributes.
        Raises
        ------
        FileNotFoundError
            If the specified file does not exist.
        ValueError
            If required datasets are missing or file loading fails.
        """
        start_time = time.time()
        filepath = Path(filename)
        if not filepath.exists():
            raise FileNotFoundError(f"HDF5 file not found: {filename}")

        if not filename.lower().endswith((".h5", ".hdf5")):
            logger.warning("File '%s' does not have .h5 or .hdf5 extension", filename)

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

        if bvecs is not None:
            kx, ky = frac_to_real_2d(k1_grid, k2_grid, bvecs)
        else:
            kx, ky = None, None
            logger.info("No bvecs available. Real-space kx, ky set to None.")

        if k_range is not None:
            logger.info("Extending to k-range: [%s, %s)", k_range[0], k_range[1])
            base_ek2d = {"energies": energies, "k1_grid": k1_grid, "k2_grid": k2_grid}
            extended = EK2DIO._extend_ek2d_static(base_ek2d, k_range)
            energies = extended["energies"]
            k1_grid = extended["k1_grid"]
            k2_grid = extended["k2_grid"]
            if bvecs is not None:
                kx, ky = frac_to_real_2d(k1_grid, k2_grid, bvecs)

        num_wann, nk1, nk2 = energies.shape
        logger.info(
            "Loaded from %s | Bands: %d | Grid: %dx%d", filename, num_wann, nk1, nk2
        )
        logger.info("Energy range: %.4f -> %.4f eV", energies.min(), energies.max())
        logger.info("Load time: %.2f s", time.time() - start_time)

        return {
            "energies": energies,
            "kx": kx,
            "ky": ky,
            "k1_grid": k1_grid,
            "k2_grid": k2_grid,
            "bvecs": bvecs,
            "metadata": metadata,
        }
