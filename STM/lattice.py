import numpy as np


class LATTICE2D:
    """
    Class for 2D lattice operations in reciprocal and real space.

    This class provides methods for:
    1. Managing reciprocal lattice vectors (bvecs) and real space lattice vectors (avecs)
    2. Converting between reciprocal and real space representations
    3. Generating reciprocal lattice vectors with C3 symmetry
    4. Creating geometric masks for selecting regions in q-space

    The class can be initialized with either reciprocal vectors (bvecs) or
    real space vectors (avecs), or both. If only one is provided, the other
    is automatically calculated using the standard lattice relationships.
    """

    def __init__(self, avecs=None, bvecs=None, degree=0):
        """
        Initialize 2D lattice calculator.

        Parameters:
        -----------
        avecs : numpy.ndarray, optional
            Real space lattice vectors in 3D space.
            Shape: (3, 3) where each row is a lattice vector.
        bvecs : numpy.ndarray, optional
            Reciprocal lattice vectors in 3D space.
            Shape: (3, 3) where each row is a reciprocal lattice vector.
        degree : float, optional
            Rotation angle in degrees to apply to lattice vectors.
            Default is 0 (no rotation).

        Raises:
        -------
        ValueError:
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
            self.bvecs = np.round(bvecs, 7)  # Round to 7 decimal places
            self.b1 = self.bvecs[0]
            self.b2 = self.bvecs[1]
            self.b3 = self.bvecs[2]
        else:
            self.bvecs = None
            self.b1 = self.b2 = self.b3 = None

        if avecs is not None:
            if not isinstance(avecs, np.ndarray) or avecs.shape != (3, 3):
                raise ValueError("avecs must be a 3x3 numpy array")
            self.avecs = np.round(avecs, 7)  # Round to 7 decimal places
            self.a1 = self.avecs[0]
            self.a2 = self.avecs[1]
            self.a3 = self.avecs[2]
        else:
            self.avecs = None
            self.a1 = self.a2 = self.a3 = None

        # Calculate missing vectors if needed
        if bvecs is not None and avecs is None:
            self.avecs = self._reciprocal_to_real()
            self.a1 = self.avecs[0]
            self.a2 = self.avecs[1]
            self.a3 = self.avecs[2]
        elif avecs is not None and bvecs is None:
            self.bvecs = self._real_to_reciprocal()
            self.b1 = self.bvecs[0]
            self.b2 = self.bvecs[1]
            self.b3 = self.bvecs[2]

    def _rotate_vectors(self):
        """
        Rotate lattice vectors by the specified angle.
        Only rotates the first two vectors (in-plane vectors).
        """
        # Create 2D rotation matrix
        theta = self.radian
        rot_matrix = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])

        # Rotate real space vectors
        if self.avecs is not None:
            self.avecs = self.avecs @ rot_matrix.T
            self.avecs = np.round(self.avecs, 7)  # Round to 7 decimal places
            self.a1 = self.avecs[0]
            self.a2 = self.avecs[1]
            # a3 remains unchanged

        # Rotate reciprocal space vectors
        if self.bvecs is not None:
            self.bvecs = self.bvecs @ rot_matrix.T
            self.bvecs = np.round(self.bvecs, 7)  # Round to 7 decimal places
            self.b1 = self.bvecs[0]
            self.b2 = self.bvecs[1]
            # b3 remains unchanged

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

    @staticmethod
    def _wrap_centered(x: float) -> float:
        """Map x to [-0.5, 0.5)."""
        return x - np.floor(x + 0.5)

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

        # Round to 7 decimal places
        avecs = np.round(avecs, 7)

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

        # Round to 7 decimal places
        bvecs = np.round(bvecs, 7)

        return bvecs

    def supercell(self, transformation_matrix):
        """
        Create a supercell by applying a transformation matrix to the 2D lattice vectors.

        Parameters
        ----------
        transformation_matrix : array-like of shape (2, 2)
            Transformation matrix for the 2D lattice vectors.
            The new lattice vectors are calculated as:
                a1_super = a1*t11 + a2*t12
                a2_super = a1*t21 + a2*t22
            where t_matrix = [[t11, t12], [t21, t22]]

        Returns
        -------
        LATTICE2D
            New LATTICE2D instance representing the supercell.

        Raises
        ------
        ValueError:
            If transformation_matrix is not a 2x2 array,
            or if its determinant is zero (singular transformation).

        Examples
        --------
        >>> # Create a 2x2 supercell
        >>> supercell = lattice.supercell([[2, 0], [0, 2]])

        >>> # Create a rotated supercell
        >>> supercell = lattice.supercell([[1, 1], [-1, 1]])
        """
        if self.avecs is None:
            raise ValueError("Real space vectors not initialized")

        # Convert to numpy array and validate
        t_matrix = np.asarray(transformation_matrix, dtype=float)

        if t_matrix.shape != (2, 2):
            raise ValueError("transformation_matrix must be a 2x2 array")

        det_t = np.linalg.det(t_matrix)
        if abs(det_t) < 1e-12:
            raise ValueError("transformation_matrix is singular (determinant is zero)")

        # Extract in-plane components of original lattice vectors
        a1_2d = self.a1[:2]  # First two components of a1
        a2_2d = self.a2[:2]  # First two components of a2

        # Create 2x2 matrix of original in-plane vectors (each column is a vector)
        a_2d = np.column_stack([a1_2d, a2_2d])  # Shape: (2, 2)

        # Apply transformation: [a1_super, a2_super] = [a1, a2] @ t_matrix
        # where t_matrix = [[t11, t12], [t21, t22]]
        a_prime_2d = a_2d @ t_matrix  # Shape: (2, 2)

        # Create new 3D lattice vectors
        avecs_super = self.avecs.copy()

        # Update in-plane components
        avecs_super[0, :2] = a_prime_2d[:, 0]  # New a1 (in-plane)
        avecs_super[1, :2] = a_prime_2d[:, 1]  # New a2 (in-plane)

        # a3 remains unchanged (out-of-plane vector)
        # Note: a3 is typically [0, 0, c] for 2D materials

        # Create new LATTICE2D instance with supercell vectors
        return LATTICE2D(avecs=avecs_super)

    def get_bragg_points_supercell_in_1x1_fftshift(
        self,
        transformation_matrix,
        include_origin: bool = False,
        return_uv: bool = False,
        return_mn: bool = False,
        tol: float = 1e-10,
    ):
        """
        Generate ALL supercell Bragg points folded into the 1x1 FFTshifted window
        (primitive reciprocal cell centered at Gamma): u,v in [-0.5, 0.5).

        Convention consistent with this class:
            supercell(): [a1', a2'] = [a1, a2] @ T
        Therefore reciprocal transforms as:
            [b1', b2'] = [b1, b2] @ (T^{-1})^T

        Parameters
        ----------
        transformation_matrix : array-like (2,2)
            Real-space supercell transform T.
        include_origin : bool
            Include Gamma (0,0).
        return_uv : bool
            Also return folded reduced coords (u,v) in primitive reciprocal basis.
        return_mn : bool
            Also return the integer coefficients (m,n) in supercell reciprocal basis b1',b2'
            that generated each (folded) point.
        tol : float
            Numerical tolerance for boundary & dedup.

        Returns
        -------
        points : np.ndarray, shape (2,N)
            (qx,qy) Cartesian positions of Bragg points inside fftshifted 1x1 window.
        (optional) uv : np.ndarray, shape (2,N)
            Folded reduced coords in primitive reciprocal basis (inside [-0.5,0.5)).
        (optional) mn : np.ndarray, shape (2,N)
            Integers (m,n) in supercell reciprocal basis used to generate the point.

        Notes
        -----
        In exact arithmetic, the number of unique points inside the 1x1 window is |det(T)|.
        """
        if self.bvecs is None:
            raise ValueError("Reciprocal vectors (bvecs) not initialized")

        t = np.asarray(transformation_matrix, dtype=float)
        if t.shape != (2, 2):
            raise ValueError("transformation_matrix must be a 2x2 array")
        det_t = np.linalg.det(t)
        if abs(det_t) < 1e-14:
            raise ValueError("transformation_matrix is singular")

        # primitive reciprocal basis in xy (columns)
        b1_xy = np.asarray(self.b1[:2], dtype=float)
        b2_xy = np.asarray(self.b2[:2], dtype=float)
        b = np.column_stack([b1_xy, b2_xy])  # (2,2)
        if abs(np.linalg.det(b)) < 1e-14:
            raise ValueError("Primitive reciprocal basis (b1,b2) is singular in xy")
        b_inv = np.linalg.inv(b)

        # supercell reciprocal basis in xy:
        # B' = B @ (T^{-1})^T
        b_p = b @ np.linalg.inv(t).T

        # enumerate candidate (m,n) in a safe finite range, then fold+dedup
        m_max = int(np.ceil(np.max(np.abs(t)))) + 1
        m_range = range(-4 * m_max, 4 * m_max + 1)
        n_range = range(-4 * m_max, 4 * m_max + 1)

        # unique by quantized folded reduced coords
        kept = {}  # key -> (u,v,qx,qy,m,n)

        for m in m_range:
            for n in n_range:
                if (not include_origin) and (m == 0 and n == 0):
                    continue

                g_xy = b_p @ np.array([m, n], dtype=float)  # Cartesian qx,qy

                # reduced coords in primitive basis
                u, v = (b_inv @ g_xy).tolist()

                # fold into fftshifted window [-0.5,0.5)
                u_f = self._wrap_centered(u)
                v_f = self._wrap_centered(v)

                # strict half-open window check (numerical safe)
                if not (-0.5 - tol <= u_f < 0.5 - tol and -0.5 - tol <= v_f < 0.5 - tol):
                    continue

                # reconstruct folded Cartesian point consistently
                g_fold = b @ np.array([u_f, v_f], dtype=float)

                key = (int(np.round(u_f / tol)), int(np.round(v_f / tol)))
                if key not in kept:
                    kept[key] = (u_f, v_f, g_fold[0], g_fold[1], m, n)
                else:
                    old = kept[key]
                    if (abs(m) + abs(n)) < (abs(old[4]) + abs(old[5])):
                        kept[key] = (u_f, v_f, g_fold[0], g_fold[1], m, n)

        if not kept:
            points = np.empty((2, 0))
            outs = [points]
            if return_uv:
                outs.append(np.empty((2, 0)))
            if return_mn:
                outs.append(np.empty((2, 0), dtype=int))
            return tuple(outs) if len(outs) > 1 else points

        # stable ordering
        items = sorted(kept.values(), key=lambda x: (x[0], x[1]))
        uv = np.array([[it[0], it[1]] for it in items], dtype=float).T  # (2,N)
        pts = np.array([[it[2], it[3]] for it in items], dtype=float).T  # (2,N)
        mn = np.array([[it[4], it[5]] for it in items], dtype=int).T  # (2,N)

        outs = [pts]
        if return_uv:
            outs.append(uv)
        if return_mn:
            outs.append(mn)
        return tuple(outs) if len(outs) > 1 else pts


class LatticeOperations:
    """
    Class for geometric operations on LATTICE2D instances.

    This class provides methods for:
    1. Generating reciprocal lattice vectors with C3 symmetry
    2. Sorting polygon vertices
    3. Creating polygon masks for q-space selection
    4. Checking point-on-segment relationships
    """

    def __init__(self, lattice):
        """
        Initialize LatticeOperations with a LATTICE2D instance.

        Parameters
        ----------
        lattice : LATTICE2D
            LATTICE2D instance to operate on.
        """
        if not isinstance(lattice, LATTICE2D):
            raise TypeError("lattice must be an instance of LATTICE2D")
        self.lattice = lattice

    def extend_vecs_c3(self, include_neg=True, tol=1e-10, sort=False):
        """
        Generate reciprocal lattice vectors by applying C3 symmetry to the primitive
        reciprocal lattice vectors b1 and b2.

        This function takes the in-plane components of the primitive reciprocal
        lattice vectors b1 and b2, and generates all symmetry-equivalent vectors
        under C3 rotation symmetry (120° and 240° rotations).

        Parameters
        ----------
        include_neg : bool
            Include -G vectors as well.
        tol : float
            Tolerance for deduplication.
        sort : bool
            If True, sort the vectors in clockwise order around the origin.
            Default is False.

        Returns
        -------
        Gs : ndarray, shape (2, n)
            All unique reciprocal lattice vectors generated by C3 symmetry
            from the primitive vectors {b1, b2} (and optionally their negatives).
            If sort=True, vectors are sorted in clockwise order around the origin.

        Notes
        -----
        - With include_neg=False: generates 6 vectors (2 primitive vectors × 3 rotations)
        - With include_neg=True: generates 12 vectors (4 vectors × 3 rotations)
        - These vectors form a C3-symmetric star pattern in reciprocal space
        - This is NOT the first Brillouin zone, but rather a symmetry-extended
          set of reciprocal lattice vectors

        Examples
        --------
        >>> # Generate C3-symmetric vectors from primitive reciprocal vectors
        >>> vectors = lattice.extend_1x1_C3(include_neg=True)
        >>> # vectors.shape is (2, 12) for hexagonal lattices
        """
        b1 = np.array([self.lattice.bvecs[0, 0], self.lattice.bvecs[0, 1]], dtype=float)
        b2 = np.array([self.lattice.bvecs[1, 0], self.lattice.bvecs[1, 1]], dtype=float)

        def rot(theta):
            c, s = np.cos(theta), np.sin(theta)
            return np.array([[c, -s], [s, c]], dtype=float)

        r120 = rot(2 * np.pi / 3)
        r240 = rot(4 * np.pi / 3)

        seeds = [b1, b2]
        if include_neg:
            seeds += [-b1, -b2]

        vectors = []
        for g in seeds:
            vectors.append(g)
            vectors.append(r120 @ g)
            vectors.append(r240 @ g)

        vectors = np.array(vectors, dtype=float)  # (N,2)

        # deduplicate with tolerance
        keys = np.round(vectors / tol).astype(np.int64)
        _, idx = np.unique(keys, axis=0, return_index=True)
        vectors = vectors[np.sort(idx)]  # (n,2)

        # Sort vectors if requested
        if sort:
            vectors = self._sort_vectors_clockwise(vectors)

        return vectors.T  # (2,n)

    def _sort_vectors_clockwise(self, vectors):
        """
        Sort 2D vectors in clockwise order around the origin.

        Parameters
        ----------
        vectors : ndarray, shape (n, 2)
            Array of 2D vectors to sort.

        Returns
        -------
        sorted_vectors : ndarray, shape (n, 2)
            Vectors sorted in clockwise order around the origin.
        """
        if len(vectors) == 0:
            return vectors

        # Calculate angles of each vector relative to origin
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])

        # Sort by angle in descending order for clockwise arrangement
        sorted_indices = np.argsort(-angles)

        return vectors[sorted_indices]

    def _sort_polygon_clockwise(self, polygon):
        """
        Sort polygon vertices in clockwise order.

        Parameters
        ----------
        polygon : array-like of shape (2, N)
            Polygon vertices, first row x-coordinates, second row y-coordinates.

        Returns
        -------
        sorted_polygon : ndarray of shape (2, N)
            Polygon vertices sorted in clockwise order.
        """
        # Convert to numpy array
        p = np.asarray(polygon, dtype=float)

        # Validate input
        if p.shape[0] != 2:
            raise ValueError("polygon must have shape (2, N)")

        n_vertices = p.shape[1]
        if n_vertices < 3:
            raise ValueError("polygon must have at least 3 vertices")

        # Calculate centroid (mean of all points)
        centroid = np.mean(p, axis=1, keepdims=True)

        # Calculate angles of each point relative to centroid
        angles = np.arctan2(p[1:2] - centroid[1], p[0:1] - centroid[0])

        # Sort by angle in descending order for clockwise arrangement
        sorted_indices = np.argsort(-angles[0])

        return p[:, sorted_indices]

    def get_1stbz_vertices(self, tol=1e-10):
        """
        Compute the six vertices of the first Brillouin zone for a hexagonal lattice.

        This method:
        1. Generates the 6 nearest reciprocal lattice vectors using C3 symmetry
        2. Sorts them by angle
        3. Computes the intersection of perpendicular bisectors of adjacent vectors
        4. Returns the 6 BZ vertices in clockwise order

        Returns
        -------
        vertices : ndarray, shape (2, 6)
            The six vertices of the first Brillouin zone.
        """
        # Get 6 nearest reciprocal vectors (exclude negatives to avoid duplicates)
        gs = self.extend_vecs_c3(include_neg=False, tol=tol, sort=True)  # shape (2, 6)

        if gs.shape[1] != 6:
            raise ValueError("Expected 6 reciprocal vectors for hexagonal lattice")

        vertices = []
        n = gs.shape[1]

        for i in range(n):
            g1 = gs[:, i]
            g2 = gs[:, (i + 1) % n]

            # Set up linear system: a @ k = b
            # where a = [[g1_x, g1_y], [g2_x, g2_y]]
            # and b = [0.5 * |g1|^2, 0.5 * |g2|^2]
            a = np.vstack([g1, g2])
            b = 0.5 * np.array([np.dot(g1, g1), np.dot(g2, g2)])

            # Solve a @ k = b
            try:
                k = np.linalg.solve(a, b)
                vertices.append(k)
            except np.linalg.LinAlgError as e:
                raise RuntimeError(f"Failed to solve for BZ vertex between g{i} and g{(i + 1) % n}") from e

        vertices = np.array(vertices).T  # shape (2, 6)

        # Sort in clockwise order (input and output are both (2, N))
        return self._sort_polygon_clockwise(vertices)


def create_polygon_mask(qx_grid, qy_grid, polygon, eps=1e-12):
    """
    Create a boolean mask for points inside a polygon (including boundary points).

    Parameters
    ----------
    qx_grid : array-like of shape (nx, ny)
        Grid of x-coordinates (Cartesian or fractional)
    qy_grid : array-like of shape (nx, ny)
        Grid of y-coordinates (Cartesian or fractional)
    polygon : array-like of shape (2, M)
        Polygon vertices, first row x-coordinates, second row y-coordinates.
        Must use the same coordinate system as qx_grid/qy_grid.
    eps : float
        Numerical tolerance for boundary checks.

    Returns
    -------
    mask : ndarray of shape (nx, ny)
        Boolean mask indicating which points are inside the polygon.
    """
    # Convert inputs to numpy arrays
    qx_grid = np.asarray(qx_grid, dtype=float)
    qy_grid = np.asarray(qy_grid, dtype=float)
    p = np.asarray(polygon, dtype=float)

    # Validate inputs
    if qx_grid.shape != qy_grid.shape:
        raise ValueError("qx_grid and qy_grid must have the same shape")
    if p.shape[0] != 2:
        raise ValueError("polygon must have shape (2, M)")

    original_shape = qx_grid.shape
    n_vertices = p.shape[1]

    if n_vertices < 3:
        raise ValueError("polygon must have at least 3 vertices")

    # Sort polygon vertices in clockwise order
    p = _sort_polygon_clockwise_external(p)

    # Flatten grids for processing
    px_flat = qx_grid.ravel()
    py_flat = qy_grid.ravel()
    n_points = len(px_flat)

    # Initialize mask
    mask = np.zeros(n_points, dtype=bool)

    # Check for points on boundary first
    for j in range(n_vertices):
        x1, y1 = p[0, j], p[1, j]
        x2, y2 = p[0, (j + 1) % n_vertices], p[1, (j + 1) % n_vertices]

        # Check which points are on this segment
        for i in range(n_points):
            if not mask[i]:  # Only check if not already marked
                px, py = px_flat[i], py_flat[i]
                if _point_on_segment_external(px, py, x1, y1, x2, y2, eps=eps):
                    mask[i] = True

    # Ray casting algorithm for points not on boundary
    not_on_boundary = ~mask
    px_test = px_flat[not_on_boundary]
    py_test = py_flat[not_on_boundary]

    # Count crossings for each point
    crossings = np.zeros(np.sum(not_on_boundary), dtype=int)

    for j in range(n_vertices):
        x1, y1 = p[0, j], p[1, j]
        x2, y2 = p[0, (j + 1) % n_vertices], p[1, (j + 1) % n_vertices]

        # Ray casting: check if edge crosses horizontal ray from point to +inf
        cond = (y1 > py_test) != (y2 > py_test)
        intersect_idx = np.where(cond)[0]

        if len(intersect_idx) > 0:
            # Calculate x-coordinate of intersection
            xinters = x1 + (py_test[intersect_idx] - y1) * (x2 - x1) / (y2 - y1)
            # Count intersections to the right of the point
            right_side = xinters > px_test[intersect_idx]
            crossings[intersect_idx] += right_side.astype(int)

    # Odd number of crossings means point is inside
    inside_not_on_boundary = (crossings % 2) == 1

    # Combine results: points on boundary are inside, plus points with odd crossings
    mask[not_on_boundary] = inside_not_on_boundary

    # Reshape mask to original grid shape
    mask = mask.reshape(original_shape)

    return mask


def _sort_polygon_clockwise_external(polygon):
    """External version of _sort_polygon_clockwise for standalone use."""
    p = np.asarray(polygon, dtype=float)

    if p.shape[0] != 2:
        raise ValueError("polygon must have shape (2, N)")

    n_vertices = p.shape[1]
    if n_vertices < 3:
        raise ValueError("polygon must have at least 3 vertices")

    centroid = np.mean(p, axis=1, keepdims=True)
    angles = np.arctan2(p[1:2] - centroid[1], p[0:1] - centroid[0])
    sorted_indices = np.argsort(-angles[0])

    return p[:, sorted_indices]


def _point_on_segment_external(px, py, x1, y1, x2, y2, eps=1e-12):
    """External version of _point_on_segment for standalone use."""
    cross_product = (py - y1) * (x2 - x1) - (px - x1) * (y2 - y1)
    if abs(cross_product) > eps:
        return False

    if abs(x2 - x1) > abs(y2 - y1):
        if x1 > x2:
            x1, x2 = x2, x1
        return x1 - eps <= px <= x2 + eps
    else:
        if y1 > y2:
            y1, y2 = y2, y1
        return y1 - eps <= py <= y2 + eps
