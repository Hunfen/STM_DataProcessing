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
