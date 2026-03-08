import logging
from pathlib import Path

import h5py
import numpy as np

from ..utils.miscellaneous import extend_qpi, frac_to_real_2d

logger = logging.getLogger(__name__)


def save_qpi_to_h5(
    qpi_layers: np.ndarray,
    output_path: str,
    energy_range: float | np.ndarray | list[float],
    bands: str | list[int] | None = None,
    normalize: bool = True,
    bvecs: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    V: np.ndarray | None = None,
    eta: float = 0.001,
    nq: int = 256,
    compression: str = "gzip",
    compression_opts: int = 6,
) -> None:
    """Save QPI results to an HDF5 file.

    The qpi_layers array is always saved. Optional metadata (bands, bvecs, mask, V, q_range)
    are stored only if provided. Grid resolution is recorded via the 'nq' attribute.
    If q_range is given, it is saved as a dataset; otherwise, the grid is assumed uniform
    with shape (nq, nq).

    Parameters
    ----------
    qpi_layers : np.ndarray
        The QPI intensity array, typically of shape (nq, nq) or (n_energies, nq, nq).
    output_path : str
        Path to the output HDF5 file.
    energy_range : float or array-like
        Energy value(s) used in the QPI calculation.
    bands : str, list of int, or None, optional
        Band indices included in the calculation. Saved as an attribute if not None.
    normalize : bool, optional
        Whether the qpi_layers was normalized. Default is True.
    bvecs : np.ndarray or None, optional
        Reciprocal lattice basis vectors. Saved as a dataset if provided.
    mask : np.ndarray or None, optional
        Real-space mask applied during the calculation. Saved if provided.
    V : np.ndarray or None, optional
        Real-space potential used in the simulation. Saved if provided.
    eta : float, optional
        Lorentzian broadening parameter (in energy units). Default is 0.001.
    nq : int, optional
        Number of grid points along each q-direction. Default is 256.
        Stored as an attribute to indicate grid resolution.
    compression : str, optional
        Compression algorithm for the qpi_layers dataset. Default is 'gzip'.
    compression_opts : int, optional
        Compression level (0~9). Default is 6.
    """
    with h5py.File(output_path, "w") as f:
        logger.info("Saving QPI results to: %s", output_path)

        f.create_dataset(
            "qpi_layers",
            data=qpi_layers,
            compression=compression,
            compression_opts=compression_opts,
        )
        f.attrs["eta"] = eta
        f.attrs["normalize"] = normalize
        f.attrs["nq"] = nq
        f.attrs["energy_range"] = (
            energy_range if np.isscalar(energy_range) else np.array(energy_range)
        )
        if bands is not None:
            f.attrs["bands"] = "all" if bands == "all" else np.array(bands, dtype=int)
        if bvecs is not None:
            f.create_dataset("bvecs", data=bvecs)
            logger.info("  Saved 'bvecs'.")
        if mask is not None:
            f.create_dataset("mask", data=mask)
            logger.info("  Saved 'mask'.")
        if V is not None:
            f.create_dataset("V", data=V)
            logger.info("  Saved 'V'.")

    file_size = Path(output_path).stat().st_size
    size_mb = file_size / (1024 * 1024)
    qpi_layers_shape = qpi_layers.shape
    logger.info("QPI data saved successfully to: %s", output_path)
    logger.info("   - File size: %.2f MB", size_mb)
    logger.info("   - qpi_layers shape: %s", qpi_layers_shape)
    logger.info("   - Grid shape: (%d, %d)", nq, nq)


def load_qpi_from_h5(
    h5_path: str,
    q_range: tuple[float, float] | None = None,
) -> dict[str, np.ndarray | dict]:
    """Load QPI results from an HDF5 file saved by `save_qpi_to_h5`.

    Reconstructs reciprocal-space grids and returns the full QPI data structure.

    Parameters
    ----------
    h5_path : str
        Path to the input HDF5 file.
    q_range : tuple[float, float] or None, optional
        Target (q_min, q_max) range for extended QPI grid. If provided,
        `extend_qpi` is used to interpolate the qpi_layers onto the new range.

    Returns
    -------
    dict
        Dictionary containing:
        - 'qpi_layers': loaded qpi_layers array
        - 'q1_grid', 'q2_grid': dimensionless fractional grids
        - 'qx_grid', 'qy_grid': real reciprocal-space grids (if bvecs available)
        - 'metadata': dict of all loaded attributes and optional arrays
    """
    with h5py.File(h5_path, "r") as f:
        logger.info("Loading QPI data from: %s", h5_path)

        qpi_layers = f["qpi_layers"][:]

        bvecs = f["bvecs"][:] if "bvecs" in f else None
        mask = f["mask"][:] if "mask" in f else None
        V = f["V"][:] if "V" in f else None

        eta = f.attrs.get("eta", 0.001)
        normalize = f.attrs.get("normalize", True)
        nq = f.attrs.get("nq", 256)
        energy_range = f.attrs.get("energy_range", None)
        bands = f.attrs.get("bands", None)

        file_size = Path(h5_path).stat().st_size
        size_mb = file_size / (1024 * 1024)
        logger.info("QPI data loaded successfully from: %s", h5_path)
        logger.info("   - File size: %.2f MB", size_mb)
        logger.info("   - qpi_layers shape: %s", qpi_layers.shape)
        logger.info("   - Grid size (nq): %d", nq)
        if bvecs is not None:
            logger.info("  Loaded 'bvecs'.")
        if mask is not None:
            logger.info("  Loaded 'mask'.")
        if V is not None:
            logger.info("  Loaded 'V'.")

    q_vals = np.linspace(-0.5, 0.5, nq, endpoint=False)
    q1_grid, q2_grid = np.meshgrid(q_vals, q_vals, indexing="ij")

    if q_range is not None:
        qpi_layers_ext, q1_grid_ext, q2_grid_ext = extend_qpi(
            qpi_layers, q1_grid, q2_grid, q_range[0], q_range[1]
        )
    else:
        qpi_layers_ext = qpi_layers
        q1_grid_ext = q1_grid
        q2_grid_ext = q2_grid

    if bvecs is not None:
        qx_grid, qy_grid = frac_to_real_2d(q1_grid_ext, q2_grid_ext, bvecs)
    else:
        qx_grid = qy_grid = None

    metadata = {
        "eta": eta,
        "normalize": normalize,
        "nq": nq,
        "energy_range": energy_range,
        "bands": bands,
        "bvecs": bvecs,
        "mask": mask,
        "V": V,
    }

    result = {
        "qpi_layers": qpi_layers_ext,
        "q1_grid": q1_grid_ext,
        "q2_grid": q2_grid_ext,
        "qx_grid": qx_grid,
        "qy_grid": qy_grid,
        "metadata": metadata,
    }

    return result
