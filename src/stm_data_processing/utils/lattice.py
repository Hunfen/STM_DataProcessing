import numpy as np

from .lattice_operations import LatticeOperations


class LATTICE:
    """
    Class for lattice operations in reciprocal and real space.

    This class provides methods for:
    1. Managing reciprocal lattice vectors (bvecs) and real space lattice vectors (avecs)
    2. Converting between reciprocal and real space representations
    3. Applying linear transformations (rotation, supercell, subcell)
    4. Creating geometric masks for selecting regions in q-space

    The lattice can be initialized with either reciprocal vectors (bvecs) or
    real space vectors (avecs), or both. If only one is provided, the other
    is automatically calculated using the standard relationship: a_i · b_j = 2π δ_ij.
    """

    def __init__(self, avecs=None, bvecs=None, degree=0):
        """
        Initialize lattice calculator.

        Parameters
        ----------
        avecs : numpy.ndarray, optional
            Real space lattice vectors in 3D space. Shape: (3, 3).
        bvecs : numpy.ndarray, optional
            Reciprocal lattice vectors in 3D space. Shape: (3, 3).
        degree : float, optional
            Rotation angle in degrees to apply to lattice vectors. Default is 0.

        Raises
        ------
        ValueError
            If neither bvecs nor avecs is provided.
        """
        if bvecs is None and avecs is None:
            raise ValueError("Either bvecs or avecs must be provided")

        # Store rotation angle
        self.degree = degree
        self.radian = np.deg2rad(degree)

        # Initialize vectors
        self._init_vectors(avecs, bvecs)

        # Apply rotation if needed
        if degree != 0:
            self._rotate_vectors()

        # Initialize operations instance
        self.ops = LatticeOperations(self)

    def _init_vectors(self, avecs, bvecs):
        """
        Initialize lattice vectors from provided inputs.

        Parameters
        ----------
        avecs : numpy.ndarray or None
            Real space lattice vectors.
        bvecs : numpy.ndarray or None
            Reciprocal lattice vectors.
        """
        # Store provided vectors
        if bvecs is not None:
            if not isinstance(bvecs, np.ndarray) or bvecs.shape != (3, 3):
                raise ValueError("bvecs must be a 3x3 numpy array")
            self.bvecs = bvecs  # Removed rounding
        else:
            self.bvecs = None

        if avecs is not None:
            if not isinstance(avecs, np.ndarray) or avecs.shape != (3, 3):
                raise ValueError("avecs must be a 3x3 numpy array")
            self.avecs = avecs  # Removed rounding
        else:
            self.avecs = None

        # Calculate missing vectors if needed
        if bvecs is not None and avecs is None:
            self.avecs = self._reciprocal_to_real()

        elif avecs is not None and bvecs is None:
            self.bvecs = self._real_to_reciprocal()

    def _transform(self, T):
        """Return transformed real-space lattice vectors (3x3 array)."""
        T = np.asarray(T, dtype=float)
        if T.shape != (3, 3):
            raise ValueError("T must be a 3x3 array")
        if abs(np.linalg.det(T)) < 1e-12:
            raise ValueError("T is singular")
        return T @ self.avecs

    def _rotate_vectors(self):
        theta = self.radian
        rot_matrix = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ]
        )
        self.avecs = self._transform(rot_matrix)
        self.bvecs = self._real_to_reciprocal()

    @property
    def a(self):
        """
        Get real space lattice vectors.

        Returns
        -------
        numpy.ndarray
            Real space lattice vectors with shape (3, 3).
        """
        return self.avecs

    @property
    def b(self):
        """
        Get reciprocal lattice vectors.

        Returns
        -------
        numpy.ndarray
            Reciprocal lattice vectors with shape (3, 3).
        """
        return self.bvecs

    @property
    def b1(self):
        return self.bvecs[0] if self.bvecs is not None else None

    @property
    def b2(self):
        return self.bvecs[1] if self.bvecs is not None else None

    @property
    def b3(self):
        return self.bvecs[2] if self.bvecs is not None else None

    @property
    def a1(self):
        return self.avecs[0] if self.avecs is not None else None

    @property
    def a2(self):
        return self.avecs[1] if self.avecs is not None else None

    @property
    def a3(self):
        return self.avecs[2] if self.avecs is not None else None

    def _reciprocal_to_real(self):
        """
        Convert reciprocal lattice vectors to real space lattice vectors.
        Internal method - not meant to be called directly.

        Returns
        -------
        avecs : ndarray, shape (3, 3)
            Real space lattice vectors. Each row is a lattice vector.
            Relationship: a_i · b_j = 2π δ_ij
        """
        if self.bvecs is None:
            raise ValueError("Reciprocal vectors not initialized")

        # Calculate volume of reciprocal cell
        volume = np.dot(self.b1, np.cross(self.b2, self.b3))

        if abs(volume) < 1e-12:
            raise ValueError("Reciprocal lattice vectors are linearly dependent")

        # Calculate real space vectors using standard formula:
        # a1 = 2π (b2 × b3) / (b1 · (b2 × b3))
        # a2 = 2π (b3 × b1) / (b1 · (b2 × b3))
        # a3 = 2π (b1 × b2) / (b1 · (b2 × b3))
        avecs = np.zeros((3, 3), dtype=float)
        avecs[0] = 2 * np.pi * np.cross(self.b2, self.b3) / volume
        avecs[1] = 2 * np.pi * np.cross(self.b3, self.b1) / volume
        avecs[2] = 2 * np.pi * np.cross(self.b1, self.b2) / volume

        # Removed rounding
        return avecs

    def _real_to_reciprocal(self):
        """
        Convert real space lattice vectors to reciprocal lattice vectors.
        Internal method - not meant to be called directly.

        Returns
        -------
        bvecs : ndarray, shape (3, 3)
            Reciprocal lattice vectors. Each row is a reciprocal lattice vector.
            Relationship: a_i · b_j = 2π δ_ij
        """
        if self.avecs is None:
            raise ValueError("Real space vectors not initialized")

        # Calculate volume of real space cell
        volume = np.dot(self.a1, np.cross(self.a2, self.a3))

        if abs(volume) < 1e-12:
            raise ValueError("Real space lattice vectors are linearly dependent")

        # Calculate reciprocal vectors using standard formula:
        # b1 = 2π (a2 × a3) / (a1 · (a2 × a3))
        # b2 = 2π (a3 × a1) / (a1 · (a2 × a3))
        # b3 = 2π (a1 × a2) / (a1 · (a2 × a3))
        bvecs = np.zeros((3, 3), dtype=float)
        bvecs[0] = 2 * np.pi * np.cross(self.a2, self.a3) / volume
        bvecs[1] = 2 * np.pi * np.cross(self.a3, self.a1) / volume
        bvecs[2] = 2 * np.pi * np.cross(self.a1, self.a2) / volume

        # Removed rounding
        return bvecs

    def supercell(self, transformation_matrix):
        """Create a supercell using a 2x2 or 3x3 integer transformation matrix."""
        if self.avecs is None:
            raise ValueError("Real space vectors not initialized")

        T = np.asarray(transformation_matrix, dtype=float)

        if T.shape == (2, 2):
            print(
                "Warning: 2x2 transformation matrix provided. Assuming in-plane (xy) supercell."
            )
            T_full = np.eye(3)
            T_full[:2, :2] = T
            T = T_full
        elif T.shape != (3, 3):
            raise ValueError("transformation_matrix must be 2x2 or 3x3")

        if not np.allclose(T, np.round(T), atol=1e-12):
            raise ValueError("All elements of transformation_matrix must be integers")

        T = np.round(T).astype(int)
        new_avecs = self._transform(T)
        return LATTICE(avecs=new_avecs)

    def subcell(self, transformation_matrix):
        """Create a subcell by applying the inverse of an integer supercell transformation matrix."""
        if self.avecs is None:
            raise ValueError("Real space vectors not initialized")

        T = np.asarray(transformation_matrix, dtype=float)

        if T.shape == (2, 2):
            print(
                "Warning: 2x2 transformation matrix provided. Assuming in-plane (xy) subcell."
            )
            T_full = np.eye(3)
            T_full[:2, :2] = T
            T = T_full
        elif T.shape != (3, 3):
            raise ValueError("transformation_matrix must be 2x2 or 3x3")

        if not np.allclose(T, np.round(T), atol=1e-12):
            raise ValueError("All elements of transformation_matrix must be integers")

        det_T = np.linalg.det(T)
        if abs(det_T) < 1e-12:
            raise ValueError("transformation_matrix is singular and cannot be inverted")

        try:
            T_inv = np.linalg.inv(T)
        except np.linalg.LinAlgError as e:
            raise ValueError(f"Failed to invert transformation_matrix: {e}") from e

        new_avecs = self._transform(T_inv)
        return LATTICE(avecs=new_avecs)

    def rotate(self, degree):
        """
        Return a new LATTICE instance rotated by the specified angle around the z-axis.

        Parameters
        ----------
        degree : float
            Rotation angle in degrees (positive = counterclockwise).

        Returns
        -------
        LATTICE
            A new LATTICE object with rotated real-space and reciprocal vectors.
        """
        if self.avecs is None:
            raise ValueError(
                "Real space vectors must be initialized to perform rotation"
            )

        theta = np.deg2rad(degree)
        rot_matrix = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ]
        )
        new_avecs = rot_matrix @ self.avecs
        return LATTICE(avecs=new_avecs)
