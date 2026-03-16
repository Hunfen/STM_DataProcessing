import mpmath as mp
import numpy as np

from .lattice_operations import LatticeOperations


class _PrecisionConfig:
    """Precision configuration manager"""

    _dps = 50  # Decimal places, default 50

    @classmethod
    def get_dps(cls):
        return cls._dps

    @classmethod
    def set_dps(cls, dps):
        """
        Set calculation precision (decimal places)

        Parameters
        ----------
        dps : int
            Decimal significant digits, recommended 50-100
        """
        cls._dps = int(dps)
        mp.mp.dps = cls._dps


# Initialize mpmath precision
mp.mp.dps = _PrecisionConfig.get_dps()


class LATTICE:
    """
    Lattice container with a consistent crystallographic convention.

    High-Precision Calculation Notes
    --------------------------------
    Uses mpmath for arbitrary-precision calculations internally.
    Input/output remains compatible with numpy float64.
    Adjust calculation precision via LATTICE.set_precision().

    Row-wise Storage Convention
    ---------------------------
    All lattice vectors (both real-space and reciprocal-space) are stored
    row-wise throughout this module:

    avecs : (3, 3) ndarray
        Real-space lattice vectors stored row-wise:
            avecs[0] = a1 (first row)
            avecs[1] = a2 (second row)
            avecs[2] = a3 (third row)

    bvecs : (3, 3) ndarray
        Reciprocal lattice vectors stored row-wise:
            bvecs[0] = b1 (first row)
            bvecs[1] = b2 (second row)
            bvecs[2] = b3 (third row)

    Transformation Convention
    -------------------------
    For a crystallographic integer transformation matrix M, the supercell
    transformation is defined as:

        A_super = M.T @ A_prim

    where A is the (3,3) matrix whose rows are lattice vectors.

    Therefore:
        supercell(M): A_new = M.T @ A_old
        subcell(M):   A_new = inv(M).T @ A_old

    This convention is applied consistently to both real-space (avecs)
    and reciprocal-space (bvecs) vectors.
    """

    _precision_dps = 50

    def __init__(self, avecs=None, bvecs=None, degree=0.0):
        if avecs is None and bvecs is None:
            raise ValueError("Either bvecs or avecs must be provided")

        self.degree = float(degree)
        self.radian = np.deg2rad(self.degree)

        self.avecs, self.bvecs = None, None
        self._init_vectors(avecs=avecs, bvecs=bvecs)

        if self.degree != 0.0:
            self._apply_rotation_inplace(self.degree)

        self.ops = LatticeOperations(self)

    # ------------------------------------------------------------------
    # Precision Configuration (Class Methods)
    # ------------------------------------------------------------------
    @classmethod
    def set_precision(cls, dps):
        """
        Set global calculation precision (decimal places)

        Parameters
        ----------
        dps : int
            Decimal significant digits, recommended 50-100
            Default: 50
            Ultra-high precision: 100-200
        """
        cls._precision_dps = int(dps)
        _PrecisionConfig.set_dps(dps)

    @classmethod
    def get_precision(cls):
        """Get current precision setting (decimal places)"""
        return cls._precision_dps

    # ------------------------------------------------------------------
    # initialization helpers
    # ------------------------------------------------------------------

    def _init_vectors(self, avecs=None, bvecs=None):
        """
        Initialize lattice vectors ensuring consistency between real and
        reciprocal space.

        Uses high-precision calculations to ensure consistency of initial vectors.

        Row-wise convention: both avecs and bvecs are stored as (3,3) arrays
        where each row corresponds to a lattice vector.
        """
        if avecs is not None:
            avecs = self._validate_matrix(avecs, "avecs")
        if bvecs is not None:
            bvecs = self._validate_matrix(bvecs, "bvecs")

        if avecs is None and bvecs is not None:
            avecs = self._reciprocal_to_real(bvecs)
        elif bvecs is None and avecs is not None:
            bvecs = self._real_to_reciprocal(avecs)

        if avecs is None or bvecs is None:
            raise ValueError("At least one of avecs or bvecs must be provided")

        b_from_a = self._real_to_reciprocal(avecs)
        if not np.allclose(b_from_a, bvecs, atol=1e-10, rtol=1e-10):
            raise ValueError(
                "Provided avecs and bvecs are inconsistent with a_i · b_j = 2π δ_ij"
            )

        self.avecs, self.bvecs = avecs, bvecs

    @staticmethod
    def _validate_matrix(mat, name):
        """
        Validate that input is a non-singular 3x3 matrix.

        Parameters
        ----------
        mat : array_like
            Input matrix to validate.
        name : str
            Name of the matrix for error messages.

        Returns
        -------
        ndarray
            Validated 3x3 float array.
        """
        arr = np.asarray(mat, dtype=np.float64)
        if arr.shape != (3, 3):
            raise ValueError(f"{name} must be a 3x3 array")
        if abs(np.linalg.det(arr)) < 1e-12:
            raise ValueError(f"{name} is singular or nearly singular")
        return arr

    @staticmethod
    def _promote_transform(T):
        """
        Promote a 2x2 transformation matrix to 3x3 by embedding in the
        xy-plane (z-axis unchanged).

        Row-wise convention: the transformation matrix M operates on
        row-vector stored lattice vectors via M.T @ A.
        """
        T = np.asarray(T, dtype=np.float64)

        if T.shape == (2, 2):
            T_full = np.eye(3, dtype=np.float64)
            T_full[:2, :2] = T
            T = T_full
        elif T.shape != (3, 3):
            raise ValueError("transformation_matrix must be 2x2 or 3x3")

        return T

    @staticmethod
    def _validate_transformation_matrix(T, atol=1e-12):
        """
        Validate that transformation matrix is non-singular.

        Parameters
        ----------
        T : ndarray
            Transformation matrix to validate.
        atol : float
            Tolerance for singularity check.

        Returns
        -------
        ndarray
            Validated transformation matrix (float64).

        Raises
        ------
        ValueError
            If matrix is singular or nearly singular.
        """
        T = np.asarray(T, dtype=np.float64)
        if abs(np.linalg.det(T)) < atol:
            raise ValueError("transformation_matrix is singular or nearly singular")
        return T

    # ------------------------------------------------------------------
    # High-Precision Utility Methods (Internal use of mpmath)
    # ------------------------------------------------------------------
    @staticmethod
    def _np_to_mp(mat):
        """
        Convert numpy float64 array to mpmath matrix.

        Parameters
        ----------
        mat : ndarray
            Input numpy array.

        Returns
        -------
        mp.matrix
            High-precision mpmath matrix.
        """
        return mp.matrix([[mp.mpf(x) for x in row] for row in mat])

    @staticmethod
    def _mp_to_np(mat_mp):
        """
        Convert mpmath matrix to numpy float64 array.

        Parameters
        ----------
        mat_mp : mp.matrix
            High-precision mpmath matrix.

        Returns
        -------
        ndarray
            Numpy float64 array.
        """
        rows = mat_mp.rows
        cols = mat_mp.cols
        return np.array(
            [[float(mat_mp[i, j]) for j in range(cols)] for i in range(rows)],
            dtype=np.float64,
        )

    # ------------------------------------------------------------------
    # core lattice algebra
    # ------------------------------------------------------------------

    @staticmethod
    def _real_to_reciprocal(avecs):
        """
        Compute reciprocal lattice vectors from real-space vectors.

        Uses mpmath for high-precision calculations.

        Row-wise storage convention:
            Input:  avecs[i] = ai (i-th row is the i-th real-space vector)
            Output: bvecs[i] = bi (i-th row is the i-th reciprocal vector)

        The reciprocal vectors satisfy: ai · bj = 2π δij
        """
        avecs_mp = LATTICE._np_to_mp(avecs)
        a1, a2, a3 = avecs_mp[0, :], avecs_mp[1, :], avecs_mp[2, :]

        volume = (
            a1[0] * (a2[1] * a3[2] - a2[2] * a3[1])
            + a1[1] * (a2[2] * a3[0] - a2[0] * a3[2])
            + a1[2] * (a2[0] * a3[1] - a2[1] * a3[0])
        )

        if abs(volume) < mp.mpf("1e-40"):
            raise ValueError("Real-space lattice vectors are linearly dependent")

        two_pi = mp.mpf("2") * mp.pi

        # Cross products in high precision
        b1 = (
            two_pi
            * mp.matrix(
                [
                    a2[1] * a3[2] - a2[2] * a3[1],
                    a2[2] * a3[0] - a2[0] * a3[2],
                    a2[0] * a3[1] - a2[1] * a3[0],
                ]
            )
            / volume
        )

        b2 = (
            two_pi
            * mp.matrix(
                [
                    a3[1] * a1[2] - a3[2] * a1[1],
                    a3[2] * a1[0] - a3[0] * a1[2],
                    a3[0] * a1[1] - a3[1] * a1[0],
                ]
            )
            / volume
        )

        b3 = (
            two_pi
            * mp.matrix(
                [
                    a1[1] * a2[2] - a1[2] * a2[1],
                    a1[2] * a2[0] - a1[0] * a2[2],
                    a1[0] * a2[1] - a1[1] * a2[0],
                ]
            )
            / volume
        )

        # Fix: Construct 3x3 matrix explicitly from vector components
        bvecs_mp = mp.matrix(
            [[b1[0], b1[1], b1[2]], [b2[0], b2[1], b2[2]], [b3[0], b3[1], b3[2]]]
        )
        return LATTICE._mp_to_np(bvecs_mp)

    @staticmethod
    def _reciprocal_to_real(bvecs):
        """
        Compute real-space lattice vectors from reciprocal vectors.

        Uses mpmath for high-precision calculations.

        Row-wise storage convention:
            Input:  bvecs[i] = bi (i-th row is the i-th reciprocal vector)
            Output: avecs[i] = ai (i-th row is the i-th real-space vector)

        The real-space vectors satisfy: ai · bj = 2π δij
        """
        bvecs_mp = LATTICE._np_to_mp(bvecs)
        b1, b2, b3 = bvecs_mp[0, :], bvecs_mp[1, :], bvecs_mp[2, :]

        volume_rec = (
            b1[0] * (b2[1] * b3[2] - b2[2] * b3[1])
            + b1[1] * (b2[2] * b3[0] - b2[0] * b3[2])
            + b1[2] * (b2[0] * b3[1] - b2[1] * b3[0])
        )

        if abs(volume_rec) < mp.mpf("1e-40"):
            raise ValueError("Reciprocal lattice vectors are linearly dependent")

        two_pi = mp.mpf("2") * mp.pi

        a1 = (
            two_pi
            * mp.matrix(
                [
                    b2[1] * b3[2] - b2[2] * b3[1],
                    b2[2] * b3[0] - b2[0] * b3[2],
                    b2[0] * b3[1] - b2[1] * b3[0],
                ]
            )
            / volume_rec
        )

        a2 = (
            two_pi
            * mp.matrix(
                [
                    b3[1] * b1[2] - b3[2] * b1[1],
                    b3[2] * b1[0] - b3[0] * b1[2],
                    b3[0] * b1[1] - b3[1] * b1[0],
                ]
            )
            / volume_rec
        )

        a3 = (
            two_pi
            * mp.matrix(
                [
                    b1[1] * b2[2] - b1[2] * b2[1],
                    b1[2] * b2[0] - b1[0] * b2[2],
                    b1[0] * b2[1] - b1[1] * b2[0],
                ]
            )
            / volume_rec
        )

        # Fix: Construct 3x3 matrix explicitly from vector components
        avecs_mp = mp.matrix(
            [[a1[0], a1[1], a1[2]], [a2[0], a2[1], a2[2]], [a3[0], a3[1], a3[2]]]
        )
        return LATTICE._mp_to_np(avecs_mp)

    # ------------------------------------------------------------------
    # public properties
    # ------------------------------------------------------------------
    @property
    def a(self):
        return np.linalg.norm(self.avecs, axis=1)

    @property
    def b(self):
        return np.linalg.norm(self.bvecs, axis=1)

    @property
    def a1(self):
        return self.avecs[0]

    @property
    def a2(self):
        return self.avecs[1]

    @property
    def a3(self):
        return self.avecs[2]

    @property
    def b1(self):
        return self.bvecs[0]

    @property
    def b2(self):
        return self.bvecs[1]

    @property
    def b3(self):
        return self.bvecs[2]

    @property
    def volume(self):
        return float(np.dot(self.a1, np.cross(self.a2, self.a3)))

    @property
    def reciprocal_volume(self):
        return float(np.dot(self.b1, np.cross(self.b2, self.b3)))

    # ------------------------------------------------------------------
    # cell transforms
    # ------------------------------------------------------------------

    def supercell(self, transformation_matrix):
        """
        Create a supercell from the current cell.

        Uses mpmath for high-precision transformation calculations.

        Row-wise convention: new lattice vectors are computed as
            A_new = M.T @ A_old

        where M is the integer transformation matrix and A is the (3,3)
        matrix with lattice vectors as rows.

        Parameters
        ----------
        transformation_matrix : array_like
            2x2 or 3x3 integer transformation matrix M.

        Returns
        -------
        LATTICE
            New LATTICE object representing the supercell.
        """
        if self.avecs is None:
            raise ValueError("Real-space vectors not initialized")

        M = self._promote_transform(transformation_matrix)
        M = self._validate_transformation_matrix(M)

        avecs_mp = self._np_to_mp(self.avecs)
        M_mp = self._np_to_mp(M)

        # A_new = M.T @ A_old (row-wise convention)
        new_avecs_mp = M_mp.T * avecs_mp
        new_avecs = self._mp_to_np(new_avecs_mp)

        return LATTICE(avecs=new_avecs)

    def subcell(self, transformation_matrix):
        """
        Recover the primitive/subcell from a supercell.

        Uses mpmath for high-precision inverse matrix transformation calculations.

        Row-wise convention: new lattice vectors are computed as
            A_new = inv(M).T @ A_old

        where M is the integer transformation matrix and A is the (3,3)
        matrix with lattice vectors as rows.

        Parameters
        ----------
        transformation_matrix : array_like
            2x2 or 3x3 integer transformation matrix M.

        Returns
        -------
        LATTICE
            New LATTICE object representing the subcell.
        """
        if self.avecs is None:
            raise ValueError("Real-space vectors not initialized")

        M = self._promote_transform(transformation_matrix)
        M = self._validate_transformation_matrix(M)

        avecs_mp = self._np_to_mp(self.avecs)
        M_mp = self._np_to_mp(M)

        # A_new = inv(M).T @ A_old
        M_inv_mp = mp.inverse(M_mp)
        new_avecs_mp = M_inv_mp.T * avecs_mp
        new_avecs = self._mp_to_np(new_avecs_mp)

        return LATTICE(avecs=new_avecs)

    # ------------------------------------------------------------------
    # rotations
    # ------------------------------------------------------------------
    def _apply_rotation_inplace(self, degree):
        """
        Apply rotation around the z-axis in-place.

        Uses mpmath for high-precision rotation matrix calculations.

        Row-wise convention: each row vector v is rotated as v' = v @ R.T

        Parameters
        ----------
        degree : float
            Rotation angle in degrees (counter-clockwise around z-axis).
        """
        theta = mp.radians(mp.mpf(str(degree)))

        cos_t = mp.cos(theta)
        sin_t = mp.sin(theta)

        R_mp = mp.matrix(
            [
                [cos_t, -sin_t, mp.mpf("0")],
                [sin_t, cos_t, mp.mpf("0")],
                [mp.mpf("0"), mp.mpf("0"), mp.mpf("1")],
            ]
        )

        avecs_mp = self._np_to_mp(self.avecs)
        # row-vector form -> v' = v @ R^T
        new_avecs_mp = avecs_mp * R_mp.T
        self.avecs = self._mp_to_np(new_avecs_mp)
        self.bvecs = self._real_to_reciprocal(self.avecs)

    def rotate(self, degree):
        """
        Create a rotated copy of the lattice.

        Uses mpmath for high-precision rotation matrix calculations.

        Row-wise convention: each row vector v is rotated as v' = v @ R.T

        Parameters
        ----------
        degree : float
            Rotation angle in degrees (counter-clockwise around z-axis).

        Returns
        -------
        LATTICE
            New LATTICE object with rotated lattice vectors.
        """
        if self.avecs is None:
            raise ValueError(
                "Real-space vectors must be initialized to perform rotation"
            )

        theta = mp.radians(mp.mpf(str(degree)))

        cos_t = mp.cos(theta)
        sin_t = mp.sin(theta)

        R_mp = mp.matrix(
            [
                [cos_t, -sin_t, mp.mpf("0")],
                [sin_t, cos_t, mp.mpf("0")],
                [mp.mpf("0"), mp.mpf("0"), mp.mpf("1")],
            ]
        )

        avecs_mp = self._np_to_mp(self.avecs)
        new_avecs_mp = avecs_mp * R_mp.T
        new_avecs = self._mp_to_np(new_avecs_mp)

        return LATTICE(avecs=new_avecs)

    def verify_consistency(self, atol=1e-12):
        """
        Verify lattice vector consistency (a_i · b_j = 2π δ_ij)

        Parameters
        ----------
        atol : float
            Allowed absolute error tolerance.

        Returns
        -------
        bool
            Returns True if consistency check passes.

        Raises
        ------
        ValueError
            If consistency check fails.
        """
        dot_product = self.avecs @ self.bvecs.T
        expected = 2 * np.pi * np.eye(3)

        if not np.allclose(dot_product, expected, atol=atol, rtol=atol):
            max_error = np.max(np.abs(dot_product - expected))
            raise ValueError(
                f"Lattice consistency check failed. Max error: {max_error:.2e}"
            )
        return True

    def get_transform_error(self, original_lattice):
        """
        Calculate transformation error between current and original lattice (for verifying round-trip transformation precision)

        Parameters
        ----------
        original_lattice : LATTICE
            Original lattice object.

        Returns
        -------
        float
            Maximum relative error.
        """
        error_a = np.max(
            np.abs(self.avecs - original_lattice.avecs)
            / (np.abs(original_lattice.avecs) + 1e-15)
        )
        return float(error_a)

    def __repr__(self):
        return (
            f"LATTICE(precision={self._precision_dps}dps, "
            f"volume={self.volume:.6f}, degree={self.degree:.2f})"
        )

    def __str__(self):
        lines = [
            f"LATTICE (Precision: {self._precision_dps} decimal places)",
            "Real-space lattice vectors (avecs):",
            f"  a1 = {self.a1}",
            f"  a2 = {self.a2}",
            f"  a3 = {self.a3}",
            "Reciprocal-space lattice vectors (bvecs):",
            f"  b1 = {self.b1}",
            f"  b2 = {self.b2}",
            f"  b3 = {self.b3}",
            f"Unit cell volume: {self.volume:.6f}",
            f"Rotation angle: {self.degree:.2f}°",
        ]
        return "\n".join(lines)
