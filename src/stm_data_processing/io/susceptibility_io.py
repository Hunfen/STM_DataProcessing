"""IO module for saving and loading susceptibility calculation results.

Storage convention
------------------
The two susceptibility modules
(``lindhard_re_chi.RealLindhardCalculator`` and
``mlwf_im_susceptibility.SusceptibilityCalculator_wang2012``) store the
*fftshifted* primitive-BZ array, so the q=(0,0) pixel sits at index
``(nq//2, nq//2)`` and its fractional labels are the discrete FFT
frequencies ``np.fft.fftshift(np.fft.fftfreq(nq))``.  Those labels are
exactly the transfer momenta the data pixels were evaluated at, for both
even and odd ``nq`` (a centered ``linspace`` spanning ``[-0.5, 0.5)`` is
off by half a cell for odd ``nq``).

No grid array is stored in the file: the loader rebuilds the grid from the
``nq`` attribute, which makes the file format independent of the grid
resolution.  The ``module_type`` attribute records which response is
stored (``real_Lindhard`` / ``imag_Lindhard``) and is identical to the
``metadata['module_type']`` returned by the calculating module.
"""

import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from stm_data_processing.utils.miscellaneous import extend_qpi, frac_to_real_2d

from .h5_convention import (
    COMPRESSION,
    COMPRESSION_OPTS,
    create_dataset,
    read_creation_date,
    write_file_metadata,
)

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
    compression: str = COMPRESSION,
    compression_opts: int = COMPRESSION_OPTS,
    **metadata_kwargs,
) -> None:
    """Save susceptibility results to an HDF5 file.

    Saves the fftshifted primitive-BZ susceptibility (q=(0,0) at index
    ``(nq//2, nq//2)``, matching ``fftshift(fftfreq(nq))`` labels).  No grid is
    stored; :func:`load_susceptibility_from_h5` rebuilds it from ``nq``.
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
        Compression algorithm. Default is the package convention (``gzip``).
    compression_opts : int, optional
        Compression level (0~9). Default is the package convention (4).
    **metadata_kwargs
        Additional metadata to save as attributes.
    """
    # Read the creation date of a product we are about to overwrite: an atomic
    # rewrite that changes no data must keep the file byte-identical (the
    # parallel driver's resume/repair path relies on it), so the original date
    # is carried over instead of being re-stamped.  A caller may also inject it
    # through ``**metadata_kwargs`` (``write_result_h5`` does, because it
    # assembles under a temporary name); it has to reach
    # :func:`write_file_metadata` rather than the metadata loop below, because
    # writing the same attribute twice changes HDF5's attribute layout and with
    # it the file bytes.
    provided_creation_date = metadata_kwargs.pop("creation_date", None)
    previous_creation_date = provided_creation_date or read_creation_date(output_path)

    with h5py.File(output_path, "w") as f:
        logger.info(f"Saving susceptibility results to: {output_path}")

        create_dataset(
            f,
            "susceptibility",
            susceptibility,
            units="1/eV",
            compression=compression,
            compression_opts=compression_opts,
        )

        write_file_metadata(
            f,
            creation_date=previous_creation_date,
            extra={
                "module_type": module_type,
                "eta": eta,
                "nq": nq,
                "omega_limit": omega_limit,
                "resolution": resolution,
            },
        )

        if bvecs is not None:
            create_dataset(f, "bvecs", bvecs, units="1/angstrom")
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

    The stored data is fftshifted, so the grid is rebuilt as the discrete FFT
    frequency grid ``fftshift(fftfreq(nq))`` (q=(0,0) at index ``nq//2``), which
    labels every data pixel exactly for both even and odd ``nq``.  Files written
    before this convention used a ``linspace`` grid that was off by half a cell
    for odd ``nq``; they carry the same data and no stored grid, so they are
    read back with the corrected grid automatically.

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

    # Discrete FFT frequency grid: the labels of the fftshifted data pixels.
    # A centered linspace grid spanning [-0.5, 0.5) coincides with this one
    # only for even nq; for odd nq it is offset by half a cell.
    q_vals = np.fft.fftshift(np.fft.fftfreq(nq))
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
