"""Kinematic (single-scattering) geometry for surface electron diffraction.

Implements the shared geometric core of LEED (low-energy electron
diffraction, 20-500 eV) and RHEED (reflection high-energy electron
diffraction, 5-100 keV): surface 2D lattice -> 2D reciprocal lattice rods
-> Ewald sphere -> 2D Laue condition. Only the geometric positions of the
diffraction spots / streaks are computed, never dynamical intensities.

Conventions
-----------
- The surface lies in the xy plane; the outward surface normal points along
  +z.
- ``k_in`` and ``k_out`` are the incident / outgoing wave vectors and elastic
  scattering fixes ``|k_in| = |k_out| = k``.
- The 2D lattice basis vectors ``a1 = (a1x, a1y)`` and ``a2 = (a2x, a2y)``
  lie in the xy plane.
- Lengths are in Angstrom, wave vectors in Angstrom^-1, energies in eV.
"""

import numpy as np

TWO_PI = 2.0 * np.pi


def _as_vector(value, name, size):
    """Return ``value`` as a 1-D float array of exactly ``size`` entries."""
    vec = np.asarray(value, dtype=float)
    if vec.shape != (size,):
        raise ValueError(f"{name} must be a vector of shape ({size},), got {vec.shape}")
    return vec


def surface_reciprocal(a1, a2):
    """Return the 2D reciprocal basis vectors of a surface lattice.

    The basis satisfies ``a_i . b_j = 2 pi delta_ij`` for the lattice
    vectors ``a1``, ``a2`` in the surface plane: with the scalar 2D cross
    product ``cross = a1x*a2y - a1y*a2x``,

    ``b1 = 2 pi (a2y, -a2x) / cross``
    ``b2 = 2 pi (-a1y, a1x) / cross``.

    Parameters
    ----------
    a1, a2 : array_like
        Real-space lattice basis vectors in the xy plane, each of shape (2,).

    Returns
    -------
    b1, b2 : ndarray
        Reciprocal basis vectors, each of shape (2,).

    Raises
    ------
    ValueError
        If the lattice vectors are (nearly) collinear, i.e. the 2D area
        ``a1 x a2`` vanishes.
    """
    a1 = _as_vector(a1, "a1", 2)
    a2 = _as_vector(a2, "a2", 2)
    cross = a1[0] * a2[1] - a1[1] * a2[0]
    scale = np.hypot(a1[0], a1[1]) * np.hypot(a2[0], a2[1])
    if abs(cross) <= 1e-12 * max(scale, 1.0):
        raise ValueError("surface lattice vectors must not be (nearly) collinear")
    b1 = np.array([a2[1], -a2[0]]) * (TWO_PI / cross)
    b2 = np.array([-a1[1], a1[0]]) * (TWO_PI / cross)
    return b1, b2


def electron_wavenumber(energy_eV, relativistic=False):
    """Free-electron wavenumber for a kinetic energy given in eV.

    Non-relativistic de Broglie wavelength ``lambda = sqrt(150.4 / E)`` A,
    hence ``k = 2 pi sqrt(E / 150.4)`` in A^-1. With ``relativistic=True``
    the result is multiplied by ``sqrt(1 + E / (2 m_e c^2))``, where
    ``m_e c^2 = 511000 eV`` (``2 m_e c^2 = 1022000 eV``).

    Parameters
    ----------
    energy_eV : float or ndarray
        Electron kinetic energy in eV; may be an array.
    relativistic : bool
        Apply the lowest-order relativistic correction (default False).

    Returns
    -------
    k : float or ndarray
        Wavenumber in A^-1.
    """
    energy_eV = np.asarray(energy_eV, dtype=float)
    k = TWO_PI * np.sqrt(energy_eV / 150.4)
    if relativistic:
        k = k * np.sqrt(1.0 + energy_eV / 1_022_000.0)
    return k


def solve_laue(b1, b2, k_in, h_max=None, k_max=None):
    """Cut the Ewald sphere with the reciprocal lattice rods (Laue condition).

    A rod at the 2D reciprocal vector ``g = h*b1 + k*b2`` is visible when
    parallel momentum conservation, ``k_out_par = k_in_par + g``, leaves a
    real vertical component ``k_out_z**2 = k**2 - |k_out_par|**2 >= 0`` with
    ``k = |k_in|``; the Ewald-sphere cut then gives the two exit wave vectors
    ``k_out = (k_out_par, +-k_out_z)``.

    Rod indices are tested in the rectangle ``|h| <= h_max``,
    ``|k| <= k_max``. An omitted bound is chosen so that every rod able to
    intersect the sphere (``|k_in_par + g| <= k``) is tested; this reduces
    to the disk ``|g| <= k`` for normal incidence (``k_in_par = 0``).

    Parameters
    ----------
    b1, b2 : array_like
        Reciprocal basis vectors, each of shape (2,).
    k_in : array_like
        Incident wave vector of shape (3,); its length fixes the
        Ewald-sphere radius.
    h_max, k_max : int or None
        Rod-index bounds; omitted entries are chosen automatically.

    Returns
    -------
    rods : list of dict
        One entry per visible rod, with keys ``h`` and ``k`` (rod indices),
        ``g_x`` / ``g_y`` (the 2D reciprocal vector) and ``k_out``, a (2, 3)
        array of the exit wave vectors with the ``+z`` solution first. The
        two rows coincide at exact tangency of the sphere and the rod.
    """
    b1 = _as_vector(b1, "b1", 2)
    b2 = _as_vector(b2, "b2", 2)
    k_in = _as_vector(k_in, "k_in", 3)
    kk = float(np.linalg.norm(k_in))
    if kk == 0.0:
        raise ValueError("k_in must be a non-zero wave vector")
    k_in_par = k_in[:2]

    if h_max is None or k_max is None:
        # The reachable region |k_in_par + g| <= kk lies inside |g| <= kk +
        # |k_in_par|, i.e. an ellipse in (h, k) space whose axis extents are
        # reach * sqrt((M^-1)_ii) with M = A^T A and A = [b1 b2] (columns).
        metric = np.column_stack([b1, b2])
        metric_inv = np.linalg.inv(metric.T @ metric)
        reach = kk + float(np.linalg.norm(k_in_par))
        if h_max is None:
            h_max = int(np.ceil(reach * np.sqrt(metric_inv[0, 0])))
        if k_max is None:
            k_max = int(np.ceil(reach * np.sqrt(metric_inv[1, 1])))
    h_lim, k_lim = int(h_max), int(k_max)

    h_idx = np.arange(-h_lim, h_lim + 1)
    k_idx = np.arange(-k_lim, k_lim + 1)
    h_grid, k_grid = np.meshgrid(h_idx, k_idx, indexing="ij")
    flat_h = h_grid.ravel()
    flat_k = k_grid.ravel()
    g = np.outer(flat_h, b1) + np.outer(flat_k, b2)
    k_out_par = g + k_in_par
    disc = kk * kk - np.sum(k_out_par * k_out_par, axis=1)

    tol = 1e-12 * max(kk * kk, 1.0)
    visible = disc >= -tol
    if not np.any(visible):
        return []
    k_out_z = np.sqrt(np.maximum(disc[visible], 0.0))
    par = k_out_par[visible]
    k_out_all = np.stack(
        [np.column_stack([par, k_out_z]), np.column_stack([par, -k_out_z])],
        axis=1,
    )
    rows_h = flat_h[visible]
    rows_k = flat_k[visible]
    rows_g = g[visible]

    rods = []
    for i in range(rows_h.shape[0]):
        rods.append(
            {
                "h": int(rows_h[i]),
                "k": int(rows_k[i]),
                "g_x": rows_g[i, 0],
                "g_y": rows_g[i, 1],
                "k_out": k_out_all[i],
            }
        )
    return rods


def leed_pattern(a1, a2, energy_eV, h_max=None, k_max=None):
    """LEED spot positions for normal incidence.

    The beam enters along -z (``k_in = (0, 0, -k)``) and only beams leaving
    the surface (``k_out_z > 0``) are kept. Because the parallel momentum
    transfer vanishes for normal incidence, the position of a spot on the
    screen is simply the reciprocal vector ``k_out_par = g``.

    Parameters
    ----------
    a1, a2 : array_like
        Real-space lattice basis vectors, each of shape (2,).
    energy_eV : float
        Electron kinetic energy in eV.
    h_max, k_max : int or None
        Passed through to :func:`solve_laue`.

    Returns
    -------
    spots : list of dict
        One entry per diffraction spot, with keys ``h`` and ``k`` (rod
        indices), ``g_x`` / ``g_y`` (reciprocal vector / screen position)
        and ``k_out``, the 3D exit wave vector of the upward beam.
    """
    b1, b2 = surface_reciprocal(a1, a2)
    wavenumber = float(electron_wavenumber(energy_eV))
    k_in = np.array([0.0, 0.0, -wavenumber])
    spots = []
    for rod in solve_laue(b1, b2, k_in, h_max=h_max, k_max=k_max):
        for k_out in rod["k_out"]:
            if k_out[2] > 0.0:
                spots.append(
                    {
                        "h": rod["h"],
                        "k": rod["k"],
                        "g_x": rod["g_x"],
                        "g_y": rod["g_y"],
                        "k_out": k_out,
                    }
                )
    return spots


def rheed_pattern(a1, a2, energy_eV, grazing_angle_deg, h_max=None, k_max=None):
    """RHEED streak geometry for grazing incidence in the xz plane.

    The beam travels along +x tilted toward the surface by the grazing
    angle ``alpha`` (angle with the surface), i.e.
    ``k_in = (k cos(alpha), 0, -k sin(alpha))``. Each rod cut by the Ewald
    sphere yields two exit wave vectors (upward and downward); adjacent rods
    then form the near-grazing streaks seen on a RHEED screen.

    Parameters
    ----------
    a1, a2 : array_like
        Real-space lattice basis vectors, each of shape (2,).
    energy_eV : float
        Electron kinetic energy in eV.
    grazing_angle_deg : float
        Grazing angle in degrees (angle between the beam and the surface).
    h_max, k_max : int or None
        Passed through to :func:`solve_laue`.

    Returns
    -------
    streaks : list of dict
        One entry per exit direction, with keys ``h`` and ``k`` (rod
        indices), ``g_x`` / ``g_y`` (the 2D reciprocal vector) and
        ``k_out``, the 3D exit wave vector of one of the two solutions.
    """
    b1, b2 = surface_reciprocal(a1, a2)
    wavenumber = float(electron_wavenumber(energy_eV))
    alpha = np.deg2rad(grazing_angle_deg)
    k_in = np.array([wavenumber * np.cos(alpha), 0.0, -wavenumber * np.sin(alpha)])
    streaks = []
    for rod in solve_laue(b1, b2, k_in, h_max=h_max, k_max=k_max):
        for k_out in rod["k_out"]:
            streaks.append(
                {
                    "h": rod["h"],
                    "k": rod["k"],
                    "g_x": rod["g_x"],
                    "g_y": rod["g_y"],
                    "k_out": k_out,
                }
            )
    return streaks
