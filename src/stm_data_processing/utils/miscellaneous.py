from pathlib import Path

import h5py
import numpy as np


def frac_to_real_2d(
    grid1: np.ndarray,
    grid2: np.ndarray,
    bvecs: np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Convert dimensionless fractional grids to real-space reciprocal coordinates.

    Given 2D grids in fractional reciprocal space (e.g., along b₁ and b₂ directions),
    this function maps them to physical reciprocal-space coordinates (in Å⁻¹) using
    the provided reciprocal lattice vectors.

    Parameters
    ----------
    grid1, grid2 : np.ndarray
        2D arrays of the same shape representing coordinates in fractional reciprocal space
        (i.e., coefficients along the first and second reciprocal lattice vectors).
    bvecs : np.ndarray or None
        Reciprocal lattice basis vectors of shape (2, 2) or (3, 3), in units of Å⁻¹.
        Only the first two rows and columns are used. If None, the conversion is skipped.

    Returns
    -------
    gridx, gridy : np.ndarray or None
        Physical reciprocal-space coordinate grids in Å⁻¹. Both are None if `bvecs` is None.
    """
    if bvecs is not None:
        gridx = grid1 * bvecs[0, 0] + grid2 * bvecs[1, 0]
        gridy = grid1 * bvecs[0, 1] + grid2 * bvecs[1, 1]
        return gridx, gridy
    else:
        return None, None


def fft_q_limits(frame_size_nm, n):
    """
    frame_size_nm : float
        real-space frame size (nm), assumed square
    n : int
        pixel number (n x n)

    Returns
    -------
    ([qx_min, qx_max], [qy_min, qy_max]) in Å^{-1}
    """
    q_max = np.pi * n / (frame_size_nm * 10.0)  # Å^{-1}

    return ([-q_max, q_max], [-q_max, q_max])


def extend_qpi(
    qpi_layers: np.ndarray,
    q1_base: np.ndarray,
    q2_base: np.ndarray,
    qmin: float,
    qmax: float,
):
    """
    Extend QPI with strictly preserved q-density
    and exact [qmin, qmax) cropping.
    Supports both 2D (nk, nk) and 3D (nband, nk, nk) qpi_layers.
    """

    # Ensure qpi_layers is 3D
    was_2d = False
    if qpi_layers.ndim == 2:
        qpi_layers = qpi_layers[np.newaxis, :, :]
        was_2d = True
    elif qpi_layers.ndim != 3:
        raise ValueError(f"qpi_layers must be 2D or 3D, got {qpi_layers.ndim}D")

    nband, nq, _ = qpi_layers.shape

    # -------- 1. extend ----------
    n_min = int(np.floor(qmin + 0.5))
    n_max = int(np.ceil(qmax - 0.5))
    shifts = np.arange(n_min, n_max + 1)
    nq_big = nq * len(shifts)

    print(f"Extending QPI from [{qmin:.3f}, {qmax:.3f}) with shifts: {shifts.tolist()}")
    print(f"Original grid size: {nq} × {nq} → Extended grid size: {nq_big} × {nq_big}")

    qpi_big = np.zeros((nband, nq_big, nq_big))
    q1_big = np.zeros((nq_big, nq_big))
    q2_big = np.zeros((nq_big, nq_big))

    for ix, sx in enumerate(shifts):
        for iy, sy in enumerate(shifts):
            x0 = ix * nq
            x1 = (ix + 1) * nq
            y0 = iy * nq
            y1 = (iy + 1) * nq

            qpi_big[:, x0:x1, y0:y1] = qpi_layers
            q1_big[x0:x1, y0:y1] = q1_base + sx
            q2_big[x0:x1, y0:y1] = q2_base + sy

    # -------- 2. crop ----------
    mask_x = (q1_big[:, 0] >= qmin) & (q1_big[:, 0] < qmax)
    mask_y = (q2_big[0, :] >= qmin) & (q2_big[0, :] < qmax)

    q1_ext = q1_big[np.ix_(mask_x, mask_y)]
    q2_ext = q2_big[np.ix_(mask_x, mask_y)]
    qpi_ext = qpi_big[:, mask_x, :][:, :, mask_y]

    nq_final = q1_ext.shape[0]
    print(
        f"Cropped to final grid: {nq_final} × {nq_final} over [{qmin:.3f}, {qmax:.3f})"
    )

    # Squeeze back to 2D if input was 2D
    if was_2d:
        qpi_ext = qpi_ext[0]

    return qpi_ext, q1_ext, q2_ext


def k_to_q(
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
        print(f"Saving QPI results to: {output_path}")

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
            print("  Saved 'bvecs'.")
        if mask is not None:
            f.create_dataset("mask", data=mask)
            print("  Saved 'mask'.")
        if V is not None:
            f.create_dataset("V", data=V)
            print("  Saved 'V'.")

    file_size = Path(output_path).stat().st_size
    size_mb = file_size / (1024 * 1024)
    qpi_layers_shape = qpi_layers.shape
    print(f"✅ QPI data saved successfully to: {output_path}")
    print(f"   - File size: {size_mb:.2f} MB")
    print(f"   - qpi_layers shape: {qpi_layers_shape}")
    print(f"   - Grid shape: ({nq}, {nq})")


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
        print(f"Loading QPI data from: {h5_path}")

        # Load mandatory dataset
        qpi_layers = f["qpi_layers"][:]

        # Load optional datasets
        bvecs = f["bvecs"][:] if "bvecs" in f else None
        mask = f["mask"][:] if "mask" in f else None
        V = f["V"][:] if "V" in f else None

        # Load attributes
        eta = f.attrs.get("eta", 0.001)
        normalize = f.attrs.get("normalize", True)
        nq = f.attrs.get("nq", 256)
        energy_range = f.attrs.get("energy_range", None)
        bands = f.attrs.get("bands", None)

        # Print summary
        file_size = Path(h5_path).stat().st_size
        size_mb = file_size / (1024 * 1024)
        print(f"✅ QPI data loaded successfully from: {h5_path}")
        print(f"   - File size: {size_mb:.2f} MB")
        print(f"   - qpi_layers shape: {qpi_layers.shape}")
        print(f"   - Grid size (nq): {nq}")
        if bvecs is not None:
            print("  Loaded 'bvecs'.")
        if mask is not None:
            print("  Loaded 'mask'.")
        if V is not None:
            print("  Loaded 'V'.")

    # Reconstruct base fractional grids (dimensionless, [-0.5, 0.5))
    q_vals = np.linspace(-0.5, 0.5, nq, endpoint=False)
    q1_grid, q2_grid = np.meshgrid(q_vals, q_vals, indexing="ij")

    # Optionally extend to target q_range
    if q_range is not None:
        qpi_layers_ext, q1_grid_ext, q2_grid_ext = extend_qpi(
            qpi_layers, q1_grid, q2_grid, q_range[0], q_range[1]
        )
    else:
        qpi_layers_ext = qpi_layers
        q1_grid_ext = q1_grid
        q2_grid_ext = q2_grid

    # Convert to real reciprocal space if bvecs available
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
