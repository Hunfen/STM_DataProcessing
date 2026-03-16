"""IO module for saving and loading susceptibility calculation results."""

import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from stm_data_processing.utils.miscellaneous import extend_qpi, frac_to_real_2d

logger = logging.getLogger(__name__)


def save_susceptibility_to_h5(
    susceptibility: np.ndarray,
    output_path: str,
    module_type: str = "susceptibility",
    bvecs: np.ndarray | None = None,
    eta: float = 5e-3,
    omega_limit: float | None = None,
    resolution: float | None = None,
    nq: int = 256,
    compression: str = "gzip",
    compression_opts: int = 6,
    **metadata_kwargs,
) -> None:
    """Save susceptibility results to an HDF5 file.

    Saves unextended susceptibility data with the default [-0.5, 0.5) range.
    Extension and real-space coordinate conversion are handled during loading.

    Parameters
    ----------
    susceptibility : np.ndarray
        The susceptibility array, shape (nq, nq).
    output_path : str
        Path to the output HDF5 file.
    module_type : str, optional
        Module type identifier. Default is 'susceptibility'.
    bvecs : np.ndarray or None, optional
        Reciprocal lattice basis vectors, shape (2, 2) or (3, 3).
    eta : float, optional
        Lorentzian broadening parameter. Default is 5e-3.
    omega_limit : float or None, optional
        Energy limit for integration (in eV). Saved as attribute if provided.
    resolution : float or None, optional
        Energy resolution for integration (in eV). Saved as attribute if provided.
    nq : int, optional
        Number of q-points in each dimension. Default is 256.
    compression : str, optional
        Compression algorithm. Default is 'gzip'.
    compression_opts : int, optional
        Compression level (0~9). Default is 6.
    **metadata_kwargs
        Additional metadata to save as attributes.
    """
    with h5py.File(output_path, "w") as f:
        logger.info(f"Saving susceptibility results to: {output_path}")

        f.create_dataset(
            "susceptibility",
            data=susceptibility,
            compression=compression,
            compression_opts=compression_opts,
        )

        f.attrs["module_type"] = module_type
        f.attrs["eta"] = eta
        f.attrs["nq"] = nq

        if omega_limit is not None:
            f.attrs["omega_limit"] = omega_limit
        if resolution is not None:
            f.attrs["resolution"] = resolution

        if bvecs is not None:
            f.create_dataset("bvecs", data=bvecs)
            logger.info("  Saved 'bvecs'.")

        for key, value in metadata_kwargs.items():
            if value is not None:
                try:
                    f.attrs[key] = value
                except Exception as e:
                    logger.warning(f"Could not save attribute '{key}': {e}")

    file_size = Path(output_path).stat().st_size
    size_mb = file_size / (1024 * 1024)
    logger.info(f"Susceptibility data saved successfully to: {output_path}")
    logger.info(f"   - File size: {size_mb:.2f} MB")
    logger.info(f"   - susceptibility shape: {susceptibility.shape}")
    logger.info(f"   - Grid shape: ({nq}, {nq})")


def load_susceptibility_from_h5(
    h5_path: str,
    q_range: tuple[float, float] | None = None,
) -> dict[str, np.ndarray | dict[str, Any]]:
    """Load susceptibility results from an HDF5 file.

    Reconstructs grids and optionally extends susceptibility based on q_range.

    Parameters
    ----------
    h5_path : str
        Path to the input HDF5 file.
    q_range : tuple[float, float] or None, optional
        Target (q_min, q_max) range for extended susceptibility grid. If provided,
        `extend_qpi` is used to interpolate the susceptibility onto the new range.

    Returns
    -------
    dict
        Dictionary containing:
        - 'data': loaded (optionally extended) susceptibility array
        - 'q1_grid', 'q2_grid': dimensionless fractional grids (extended if q_range given)
        - 'qx_grid', 'qy_grid': real reciprocal-space grids (if bvecs available)
        - 'metadata': dict of all loaded attributes and optional arrays
    """
    with h5py.File(h5_path, "r") as f:
        logger.info("Loading susceptibility data from: %s", h5_path)

        susceptibility = f["susceptibility"][:]

        bvecs = f["bvecs"][:] if "bvecs" in f else None

        module_type = f.attrs.get("module_type", "susceptibility")
        eta = f.attrs.get("eta", 5e-3)
        nq = f.attrs.get("nq", 256)
        omega_limit = f.attrs.get("omega_limit", None)
        resolution = f.attrs.get("resolution", None)

        metadata_extra = {}
        for key in f.attrs:
            if key not in [
                "module_type",
                "eta",
                "nq",
                "omega_limit",
                "resolution",
            ]:
                metadata_extra[key] = f.attrs[key]

        file_size = Path(h5_path).stat().st_size
        size_mb = file_size / (1024 * 1024)
        logger.info("Susceptibility data loaded successfully from: %s", h5_path)
        logger.info("   - File size: %.2f MB", size_mb)
        logger.info("   - susceptibility shape: %s", susceptibility.shape)
        logger.info("   - Grid size (nq): %d", nq)
        logger.info("   - Module type: %s", module_type)

    q_vals = np.linspace(-0.5, 0.5, nq, endpoint=False)
    q1_grid, q2_grid = np.meshgrid(q_vals, q_vals, indexing="ij")

    if q_range is not None:
        susceptibility_ext, q1_grid_ext, q2_grid_ext = extend_qpi(
            susceptibility, q1_grid, q2_grid, q_range[0], q_range[1]
        )
    else:
        susceptibility_ext = susceptibility
        q1_grid_ext = q1_grid
        q2_grid_ext = q2_grid

    qx_grid, qy_grid = frac_to_real_2d(q1_grid_ext, q2_grid_ext, bvecs)

    metadata = {
        "module_type": module_type,
        "eta": eta,
        "nq": nq,
        "omega_limit": omega_limit,
        "resolution": resolution,
        "bvecs": bvecs,
        **metadata_extra,
    }

    result = {
        "data": susceptibility_ext,
        "q1_grid": q1_grid_ext,
        "q2_grid": q2_grid_ext,
        "qx_grid": qx_grid,
        "qy_grid": qy_grid,
        "metadata": metadata,
    }

    return result
