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
    module_type: str,
    bvecs: np.ndarray | None = None,
    eta: float = 0.001,
    normalize: bool = True,
    nq: int = 256,
    V: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    bands: str | list[int] | None = None,
    compression: str = "gzip",
    compression_opts: int = 6,
    **metadata_kwargs,
) -> None:
    """Save QPI results to an HDF5 file.

    Saves unextended qpi_layers with q1_grid, q2_grid (default [-0.5, 0.5) range).
    Extension and real-space coordinate conversion are handled during loading.

    Parameters
    ----------
    qpi_layers : np.ndarray
        The QPI intensity array, shape (n_energies, nq, nq) or (nq, nq).
    output_path : str
        Path to the output HDF5 file.
    energy_range : float or array-like
        Energy value(s) used in the QPI calculation.
    module_type : str
        Module type identifier, e.g., 'jdos' or 'born'.
    bvecs : np.ndarray or None, optional
        Reciprocal lattice basis vectors, shape (3, 3).
    eta : float, optional
        Lorentzian broadening parameter. Default is 0.001.
    normalize : bool, optional
        Whether the qpi_layers was normalized. Default is True.
    V : np.ndarray or None, optional
        Scattering potential matrix. Saved if provided.
    mask : np.ndarray or None, optional
        Real-space mask. Saved if provided.
    bands : str, list of int, or None, optional
        Band indices. Saved as an attribute if not None.
    compression : str, optional
        Compression algorithm. Default is 'gzip'.
    compression_opts : int, optional
        Compression level (0~9). Default is 6.
    **metadata_kwargs
        Additional metadata to save as attributes.
    """
    with h5py.File(output_path, "w") as f:
        logger.info("Saving QPI results to: %s", output_path)

        f.create_dataset(
            "qpi_layers",
            data=qpi_layers,
            compression=compression,
            compression_opts=compression_opts,
        )

        f.attrs["module_type"] = module_type
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
        if V is not None:
            f.create_dataset("V", data=V)
            logger.info("  Saved 'V'.")
        if mask is not None:
            f.create_dataset("mask", data=mask)
            logger.info("  Saved 'mask'.")

        for key, value in metadata_kwargs.items():
            if value is not None:
                f.attrs[key] = value

    file_size = Path(output_path).stat().st_size
    size_mb = file_size / (1024 * 1024)
    logger.info("QPI data saved successfully to: %s", output_path)
    logger.info("   - File size: %.2f MB", size_mb)
    logger.info("   - qpi_layers shape: %s", qpi_layers.shape)
    logger.info("   - Grid shape: (%d, %d)", nq, nq)


def load_qpi_from_h5(
    h5_path: str,
    q_range: tuple[float, float] | None = None,
) -> dict[str, np.ndarray | dict]:
    """Load QPI results from an HDF5 file.

    Reconstructs grids and optionally extends qpi_layers based on q_range.

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
        - 'qpi_layers': loaded (optionally extended) qpi_layers array
        - 'q1_grid', 'q2_grid': dimensionless fractional grids (extended if q_range given)
        - 'qx_grid', 'qy_grid': real reciprocal-space grids (if bvecs available)
        - 'metadata': dict of all loaded attributes and optional arrays
    """
    with h5py.File(h5_path, "r") as f:
        logger.info("Loading QPI data from: %s", h5_path)

        qpi_layers = f["qpi_layers"][:]

        bvecs = f["bvecs"][:] if "bvecs" in f else None
        V = f["V"][:] if "V" in f else None
        mask = f["mask"][:] if "mask" in f else None

        module_type = f.attrs.get("module_type", "unknown")
        eta = f.attrs.get("eta", 0.001)
        normalize = f.attrs.get("normalize", True)
        nq = f.attrs.get("nq", 256)
        energy_range = f.attrs.get("energy_range", None)
        bands = f.attrs.get("bands", None)

        metadata_extra = {}
        for key in f.attrs:
            if key not in [
                "module_type",
                "eta",
                "normalize",
                "nq",
                "energy_range",
                "bands",
            ]:
                metadata_extra[key] = f.attrs[key]

        file_size = Path(h5_path).stat().st_size
        size_mb = file_size / (1024 * 1024)
        logger.info("QPI data loaded successfully from: %s", h5_path)
        logger.info("   - File size: %.2f MB", size_mb)
        logger.info("   - qpi_layers shape: %s", qpi_layers.shape)
        logger.info("   - Grid size (nq): %d", nq)
        logger.info("   - Module type: %s", module_type)

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
        "module_type": module_type,
        "eta": eta,
        "normalize": normalize,
        "nq": nq,
        "energy_range": energy_range,
        "bands": bands,
        "bvecs": bvecs,
        "V": V,
        "mask": mask,
        **metadata_extra,
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
