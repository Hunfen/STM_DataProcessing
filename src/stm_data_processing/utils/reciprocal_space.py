from pathlib import Path

import numpy as np


def frac_to_real(
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


class BVecs:
    """
    Class for handling reciprocal lattice vectors (b-vectors).

    This class provides methods to read and parse reciprocal lattice vectors
    from Wannier90 .wout files or OpenMX .out files, and convert fractional
    coordinates to real-space reciprocal coordinates.

    Attributes
    ----------
    bvecs : np.ndarray or None
        Reciprocal lattice vectors b1, b2, b3 in 1/Ångström, shape (3, 3).
        None if not loaded.
    """

    def __init__(
        self, filename: str | None = None, bvecs_array: np.ndarray | None = None
    ):
        """
        Initialize BVecs.

        Parameters
        ----------
        filename : str, optional
            Path to Wannier90 .wout file or OpenMX .out file.
            If provided, reciprocal vectors will be loaded automatically.
        bvecs_array : np.ndarray, optional
            Directly provide reciprocal lattice vectors as a numpy array.
            Should be of shape (3, 3) with vectors b1, b2, b3 in 1/Ångström.
            If both filename and bvecs_array are provided, bvecs_array takes precedence.
        """
        self.bvecs = None

        if bvecs_array is not None:
            self.set_bvecs_from_array(bvecs_array)
        elif filename is not None:
            self.load_reciprocal_vectors(filename)

    def set_bvecs_from_array(self, bvecs_array: np.ndarray) -> None:
        """
        Set reciprocal lattice vectors directly from a numpy array.

        Parameters
        ----------
        bvecs_array : np.ndarray
            Reciprocal lattice vectors as a numpy array.
            Can be of shape (2, 2) for 2D systems or (3, 3) for 3D systems.
            For 2D arrays, the third row and column will be filled with zeros.
            Should contain vectors b1, b2, b3 in 1/Ångström.

        Raises
        ------
        ValueError
            If the input array does not have shape (2, 2) or (3, 3).
        """
        if not isinstance(bvecs_array, np.ndarray):
            raise TypeError("bvecs_array must be a numpy array")

        if bvecs_array.shape == (2, 2):
            # Convert (2, 2) to (3, 3) by adding zeros for third dimension
            self.bvecs = np.zeros((3, 3))
            self.bvecs[:2, :2] = bvecs_array
        elif bvecs_array.shape == (3, 3):
            self.bvecs = bvecs_array.copy()
        else:
            raise ValueError(
                f"bvecs_array must have shape (2, 2) or (3, 3), got {bvecs_array.shape}"
            )

    def load_reciprocal_vectors(self, filename: str) -> np.ndarray | None:
        """
        Read reciprocal lattice vectors from Wannier90 .wout file or OpenMX .out file.

        Parameters
        ----------
        filename : str
            Path to Wannier90 .wout file or OpenMX .out file.

        Returns
        -------
        bvecs : (3,3) ndarray or None
            Reciprocal lattice vectors b1, b2, b3 in 1/Ångström.
            Returns None if not found.

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist.
        """
        b1 = b2 = b3 = None

        # Determine file type based on extension
        file_ext = Path(filename).suffix.lower()

        try:
            with Path(filename).open(encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

                for i, line in enumerate(lines):
                    # For .wout files (Wannier90 format)
                    if file_ext == ".wout":
                        if (
                            "Reciprocal-Space Vectors" in line
                            or "Reciprocal Vectors" in line
                        ):
                            # Read next 3 lines for b1, b2, b3
                            for j in range(1, 4):
                                if i + j < len(lines):
                                    b_line = lines[i + j]
                                    if "b_1" in b_line:
                                        b1 = self._parse_wannier_vector_line(b_line)
                                    elif "b_2" in b_line:
                                        b2 = self._parse_wannier_vector_line(b_line)
                                    elif "b_3" in b_line:
                                        b3 = self._parse_wannier_vector_line(b_line)
                            break

                    # For .out files (OpenMX format)
                    elif file_ext == ".out":
                        if "Reciprocal vector b1" in line:
                            b1 = self._parse_openmx_vector_line(line)
                        elif "Reciprocal vector b2" in line:
                            b2 = self._parse_openmx_vector_line(line)
                        elif "Reciprocal vector b3" in line:
                            b3 = self._parse_openmx_vector_line(line)
                            # If we found all three vectors, break
                            if b1 is not None and b2 is not None and b3 is not None:
                                break

        except FileNotFoundError as e:
            raise FileNotFoundError(f"File not found: {filename}") from e

        if b1 is None or b2 is None or b3 is None:
            self.bvecs = None
            return None

        self.bvecs = np.vstack([b1, b2, b3])
        return self.bvecs

    @staticmethod
    def _parse_wannier_vector_line(line: str) -> np.ndarray:
        """
        Helper to extract a vector from a Wannier90 line.

        Example format: 'b_1     1.474634   0.851380   0.000000'

        Parameters
        ----------
        line : str
            Line containing vector data.

        Returns
        -------
        np.ndarray
            Vector components as float array of shape (3,).
        """
        vals = line.split()
        return np.array(list(map(float, vals[1:4])))

    @staticmethod
    def _parse_openmx_vector_line(line: str) -> np.ndarray:
        """
        Helper to extract a vector from an OpenMX line.

        Example format: '#  Reciprocal vector b1 (1/Ang):   2.55414  1.47463  0.00000'

        Parameters
        ----------
        line : str
            Line containing vector data.

        Returns
        -------
        np.ndarray
            Vector components as float array of shape (3,).
        """
        right = line.split(":", 1)[1]
        vals = right.split()
        return np.array(list(map(float, vals[:3])))

    def frac_to_real(
        self,
        grid1: np.ndarray,
        grid2: np.ndarray,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        Convert dimensionless fractional grids to real-space reciprocal coordinates.

        Given 2D grids in fractional reciprocal space (e.g., along b₁ and b₂ directions),
        this function maps them to physical reciprocal-space coordinates (in Å⁻¹) using
        the stored reciprocal lattice vectors.

        Parameters
        ----------
        grid1, grid2 : np.ndarray
            2D arrays of the same shape representing coordinates in fractional reciprocal space
            (i.e., coefficients along the first and second reciprocal lattice vectors).

        Returns
        -------
        gridx, gridy : np.ndarray or None
            Physical reciprocal-space coordinate grids in Å⁻¹.
            Both are None if `bvecs` is not loaded.
        """
        if self.bvecs is not None:
            gridx = grid1 * self.bvecs[0, 0] + grid2 * self.bvecs[1, 0]
            gridy = grid1 * self.bvecs[0, 1] + grid2 * self.bvecs[1, 1]
            return gridx, gridy
        else:
            return None, None

    def get_bvecs(self) -> np.ndarray | None:
        """
        Get the reciprocal lattice vectors.

        Returns
        -------
        np.ndarray or None
            Reciprocal lattice vectors b1, b2, b3 in 1/Ångström, shape (3, 3).
            None if not loaded.
        """
        return self.bvecs

    def __repr__(self) -> str:
        """String representation of BVecs."""
        if self.bvecs is None:
            return "BVecs(not loaded)"
        return f"BVecs(bvecs=\n{self.bvecs})"


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
