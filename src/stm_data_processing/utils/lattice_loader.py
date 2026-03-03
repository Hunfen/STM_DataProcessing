from pathlib import Path

import numpy as np

from .lattice import LATTICE


class LatticeLoader:
    """
    Factory class for creating LATTICE instances from reciprocal lattice vectors.

    This class provides methods to read and parse reciprocal lattice vectors
    from Wannier90 .wout files or OpenMX .out files, or from a provided array,
    and then use them to initialize a LATTICE object.

    The LATTICE object is created with the parsed real-space vectors (avecs)
    derived from the reciprocal vectors (bvecs).
    """

    def __init__(self):
        """
        Initialize LatticeLoader factory.
        """
        pass

    @staticmethod
    def create_lattice(
        filename: str | None = None, bvecs_array: np.ndarray | None = None
    ) -> LATTICE:
        """
        Create a LATTICE instance from reciprocal lattice vectors.

        Parameters
        ----------
        filename : str, optional
            Path to Wannier90 .wout file or OpenMX .out file.
            If provided, reciprocal vectors will be loaded automatically.
        bvecs_array : np.ndarray, optional
            Directly provide reciprocal lattice vectors as a numpy array.
            Should be of shape (2, 2) for 2D systems or (3, 3) for 3D systems.
            For 2D arrays, the third row and column will be filled with zeros.
            Should contain vectors b1, b2, b3 in 1/Ångström.
            If both filename and bvecs_array are provided, bvecs_array takes precedence.

        Returns
        -------
        LATTICE
            A fully initialized LATTICE object.

        Raises
        ------
        ValueError
            If neither filename nor bvecs_array is provided, or if parsing fails.
        FileNotFoundError
            If the specified file does not exist.
        """
        if bvecs_array is not None:
            bvecs = LatticeLoader._set_bvecs_from_array(bvecs_array)
        elif filename is not None:
            bvecs = LatticeLoader._load_reciprocal_vectors(filename)
        else:
            raise ValueError("Either filename or bvecs_array must be provided")

        if bvecs is None:
            raise ValueError("Failed to load or set reciprocal vectors")

        return LATTICE(bvecs=bvecs)

    @staticmethod
    def _set_bvecs_from_array(bvecs_array: np.ndarray) -> np.ndarray:
        """
        Set reciprocal lattice vectors directly from a numpy array.

        Parameters
        ----------
        bvecs_array : np.ndarray
            Reciprocal lattice vectors as a numpy array.
            Can be of shape (2, 2) for 2D systems or (3, 3) for 3D systems.
            For 2D arrays, the third row and column will be filled with zeros.
            Should contain vectors b1, b2, b3 in 1/Ångström.

        Returns
        -------
        np.ndarray
            Standardized reciprocal lattice vectors of shape (3, 3).

        Raises
        ------
        ValueError
            If the input array does not have shape (2, 2) or (3, 3).
        TypeError
            If the input is not a numpy array.
        """
        if not isinstance(bvecs_array, np.ndarray):
            raise TypeError("bvecs_array must be a numpy array")

        if bvecs_array.shape == (2, 2):
            # Convert (2, 2) to (3, 3) by adding zeros for third dimension
            bvecs = np.zeros((3, 3))
            bvecs[:2, :2] = bvecs_array
            return bvecs
        elif bvecs_array.shape == (3, 3):
            return bvecs_array.copy()
        else:
            raise ValueError(
                f"bvecs_array must have shape (2, 2) or (3, 3), got {bvecs_array.shape}"
            )

    @staticmethod
    def _load_reciprocal_vectors(filename: str) -> np.ndarray | None:
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
                                        b1 = LatticeLoader._parse_wannier_vector_line(
                                            b_line
                                        )
                                    elif "b_2" in b_line:
                                        b2 = LatticeLoader._parse_wannier_vector_line(
                                            b_line
                                        )
                                    elif "b_3" in b_line:
                                        b3 = LatticeLoader._parse_wannier_vector_line(
                                            b_line
                                        )
                            break

                    # For .out files (OpenMX format)
                    elif file_ext == ".out":
                        if "Reciprocal vector b1" in line:
                            b1 = LatticeLoader._parse_openmx_vector_line(line)
                        elif "Reciprocal vector b2" in line:
                            b2 = LatticeLoader._parse_openmx_vector_line(line)
                        elif "Reciprocal vector b3" in line:
                            b3 = LatticeLoader._parse_openmx_vector_line(line)
                            # If we found all three vectors, break
                            if b1 is not None and b2 is not None and b3 is not None:
                                break

        except FileNotFoundError as e:
            raise FileNotFoundError(f"File not found: {filename}") from e

        if b1 is None or b2 is None or b3 is None:
            return None

        return np.vstack([b1, b2, b3])

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
