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
        return None, None


def fft_q_limits(frame_size_nm: float, n: int) -> tuple[list[float], list[float]]:
    """Compute reciprocal-space limits for FFT output in Å⁻¹.

    Parameters
    ----------
    frame_size_nm : float
        Real-space frame size in nanometers (assumed square).
    n : int
        Number of pixels along each dimension (n x n grid).
    Returns
    -------
    tuple[list[float], list[float]]
        Reciprocal-space limits as ([qx_min, qx_max], [qy_min, qy_max]) in Å⁻¹.
    """
    q_max = np.pi * n / (frame_size_nm * 10.0)  # Convert nm to Å (1 nm = 10 Å)
    return ([-q_max, q_max], [-q_max, q_max])


def extend_qpi(
    qpi_layers: np.ndarray,
    q1_base: np.ndarray,
    q2_base: np.ndarray,
    qmin: float,
    qmax: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extend QPI with strictly preserved q-density
    and exact [qmin, qmax) cropping.
    Supports both 2D (nk, nk) and 3D (nband, nk, nk) qpi_layers.

    Parameters
    ----------
    qpi_layers : np.ndarray
        Input QPI data array of shape (nk, nk) or (nband, nk, nk).
    q1_base, q2_base : np.ndarray
        Base fractional coordinate grids of shape (nk, nk).
    qmin, qmax : float
        Reciprocal space bounds for cropping in fractional coordinates.

    Returns
    -------
    qpi_ext : np.ndarray
        Extended and cropped QPI array with same dimensionality as input.
    q1_ext, q2_ext : np.ndarray
        Corresponding extended and cropped coordinate grids.
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

    qpi_big = np.zeros((nband, nq_big, nq_big), dtype=qpi_layers.dtype)
    q1_big = np.zeros((nq_big, nq_big), dtype=q1_base.dtype)
    q2_big = np.zeros((nq_big, nq_big), dtype=q2_base.dtype)
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

    # Squeeze back to 2D if input was 2D
    if was_2d:
        qpi_ext = qpi_ext[0]

    return qpi_ext, q1_ext, q2_ext


def k_to_q(
    k1_grid: np.ndarray,
    k2_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert k-space grids to dimensionless q-space grids.

    Parameters
    ----------
    k1_grid, k2_grid : np.ndarray
        2D arrays of shape (nkx, nky) representing k-space coordinate grids.

    Returns
    -------
    q1_grid, q2_grid : np.ndarray
        Corresponding dimensionless q-space coordinate grids of the same shape.
    """
    nkx, nky = k1_grid.shape
    dk1 = float(k1_grid[1, 0] - k1_grid[0, 0]) if nkx > 1 else 1.0
    dk2 = float(k2_grid[0, 1] - k2_grid[0, 0]) if nky > 1 else 1.0
    q1_vals = (np.arange(nkx) - nkx // 2) * dk1
    q2_vals = (np.arange(nky) - nky // 2) * dk2
    q1_grid, q2_grid = np.meshgrid(q1_vals, q2_vals, indexing="ij")
    return q1_grid, q2_grid
