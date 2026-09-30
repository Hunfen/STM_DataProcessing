"""Shared helpers of the lawler-fujita-correction skill (self-contained copy).

The skill implements the Lawler-Fujita lattice-phase correction (Fujita et al.,
PNAS 2014, SI Text section 4) for hexagonal-lattice STM topographs: a local
lock-in on two (or three) first-order Bragg wave vectors gives the local lattice
phase, the phase difference from the gauge gives a slowly varying displacement
field ``u(r)``, and the image is resampled along that field.

Everything here is geometry: no physical statement is made anywhere.

Conventions used by every script of this skill (they are fixed by the end-to-end
self-test, which requires the corrected image to be the ideal lattice):

    T(r)     = ideal(r - u(r))          measured image, u = lattice displacement
    psi_i(r) = lowpass[ T(r) exp(-i Q_i . r) ]     complex lock-in field
    theta_i  = arg psi_i(r) = -Q_i . u(r)          local lattice phase
    u(r)     = Q^-1 (theta_bar - theta(r)),  theta_bar = 0 (gauge: mean phase)
    corrected(r) = T(r + u(r))                    warp along the field

``Q`` is the 2x2 matrix whose rows are the two reference wave vectors in
rad/nm, ``theta`` the column of the two phases.
"""

from __future__ import annotations

import sys

import matplotlib
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.fft import dctn, idctn
from scipy.ndimage import distance_transform_edt, map_coordinates

BAD_COLOR = "#b0b0b0"


# --------------------------------------------------------------------------- #
# style, colormap
# --------------------------------------------------------------------------- #
def setup_style():
    """Plot style of the skill (no usetex: broken with mpl 3.10 + TeX Live 2026)."""
    matplotlib.rcParams.update(
        {
            "text.usetex": False,
            "mathtext.fontset": "cm",
            "font.family": "serif",
            "font.serif": ["Palatino"],
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


_GWYDDION_ANCHORS = {  # mirrors stm_data_processing.stm.preview_plot.cdict_gwyddion
    "red": [
        (0.0, 0.0, 0.0),
        (0.344671, 0.658824, 0.658824),
        (0.687075, 0.953506, 0.953506),
        (1.0, 1.0, 1.0),
    ],
    "green": [
        (0.0, 0.0, 0.0),
        (0.344671, 0.156863, 0.156863),
        (0.687075, 0.759686, 0.759686),
        (1.0, 1.0, 1.0),
    ],
    "blue": [
        (0.0, 0.0, 0.0),
        (0.344671, 0.0588235, 0.0588235),
        (0.687075, 0.363821, 0.363821),
        (1.0, 1.0, 1.0),
    ],
}


def load_colormap(stm_lib=None):
    """Topography colormap: the package's ``gwyddion``, else the same anchors."""
    if stm_lib:
        sys.path.insert(0, str(stm_lib))
    try:
        from stm_data_processing.stm.preview_plot import gwyddion

        return gwyddion.copy(), "package:stm_data_processing.stm.preview_plot.gwyddion"
    except Exception:  # any import problem falls back to the anchors
        return (
            LinearSegmentedColormap("gwyddion", segmentdata=_GWYDDION_ANCHORS, N=4096),
            "builtin:identical anchors of cdict_gwyddion",
        )


# --------------------------------------------------------------------------- #
# radius clustering / ideal-radius anchoring of the detected reflections
# --------------------------------------------------------------------------- #
#: Relative radius tolerance of the *anchored* ring search, the tolerance that
#: the ring detection actually uses: the 1x1 radius is known from the lattice
#: constant and the field of view (:func:`ideal_radius_px`), so the ring is
#: gathered in a window around that radius.  The members of the shipped 30 nm
#: grids spread over 3-6 % in radius (the ring is an ellipse), which no
#: reflection-to-reflection tolerance can capture without a magic number.
DEFAULT_RING_TOL = 0.10
#: Relative radius tolerance of the radius *clustering* of the detected
#: reflections.  It only tabulates the detected rings for the report and the
#: self-check; the search for the 1x1 ring itself is anchored on the ideal radius
#: (``DEFAULT_RING_TOL``) and does not read it.  This constant is the single
#: source of the number for both this module and the CLI default; the package
#: constant ``bragg_peak.rings._RING_RADIUS_TOL`` (0.03) belongs to the package's
#: own ring inference and is never used here.  The value stays the historical
#: 0.02 because every product published so far records that number in its
#: correction report.
DEFAULT_RING_CLUSTER_TOL = 0.02
#: Angle tolerance (degrees) of the hexagonal triple.  A ring is three
#: independent reflections 60 degrees apart, so the two partners of the
#: strongest member sit +/-120 degrees away (and the Friedel partners mirror
#: them); a distorted lattice ring misses 60 degrees by up to a few degrees.
DEFAULT_HEXAGON_ANGLE_TOL_DEG = 25.0
#: Where the two partners of the strongest ring member sit, in degrees.
HEXAGON_PARTNER_ANGLES_DEG = (120.0, -120.0)


def group_rings(reflections, tol_frac=DEFAULT_RING_CLUSTER_TOL, min_members=6):
    """Cluster reflections by radius (greedy, strongest first).

    ``reflections`` are tuples whose entries 2, 3 and 4 are amplitude, snr and
    radius.  Returns a list of dicts ``{radius, members, total_amplitude}``
    sorted by descending total amplitude.
    """
    rings = []
    for record in reflections:
        placed = False
        for ring in rings:
            if abs(record[4] - ring["radius"]) <= tol_frac * ring["radius"]:
                ring["members"].append(record)
                total = sum(member[2] for member in ring["members"])
                ring["radius"] = float(
                    np.average(
                        [m[4] for m in ring["members"]],
                        weights=[m[2] for m in ring["members"]],
                    )
                )
                ring["total_amplitude"] = float(total)
                placed = True
                break
        if not placed:
            rings.append(
                {"radius": record[4], "members": [record], "total_amplitude": record[2]}
            )
    keep = [ring for ring in rings if len(ring["members"]) >= min_members]
    keep.sort(key=lambda ring: -ring["total_amplitude"])
    return keep


def ring_angle_deg(reflection):
    """Direction of a reflection record, in degrees."""
    return float(np.degrees(np.arctan2(float(reflection[1]), float(reflection[0]))))


def _signed_angle_difference(first_deg, second_deg):
    """``first - second`` folded into (-180, 180] degrees."""
    return (float(first_deg) - float(second_deg) + 180.0) % 360.0 - 180.0


def hexagon_ring_near_radius(
    reflections,
    radius_ideal_px,
    tol_frac=DEFAULT_RING_TOL,
    *,
    angle_tol_deg=DEFAULT_HEXAGON_ANGLE_TOL_DEG,
):
    """Hexagonal ring anchored on a *known* ideal radius, or None.

    ``reflections`` are the tuples of :func:`group_rings` (entries 2, 3 and 4 are
    amplitude, snr and radius).  Only reflections whose radius sits within
    ``tol_frac`` of ``radius_ideal_px`` are considered: the ideal 1x1 radius
    follows from the lattice constant and the field of view, so the search does
    not have to guess which rung of the radial ladder the data shows, and it
    cannot drift onto a weaker low-order ring.

    The ring is then *defined* by six-fold symmetry: the strongest reflection of
    the window fixes the reference direction and the ring requires its two
    partners ``+/-120`` degrees away, i.e. three independent members 60 degrees
    apart.  A partner is the *strongest* reflection inside the angle tolerance:
    the angle window only selects the direction, and inside an annulus that wide
    a weak local blob of the |FFT| background can easily sit closer to the exact
    angle than the ring member itself.  The returned ring carries those three
    measured reflections plus their mirrored partners (Friedel pairs of a real
    |FFT|), so every consumer sees the six members of a hexagonal ring.  Returns
    None when the window holds fewer than three reflections or when the
    hexagonal triple is not there.
    """
    window = [
        reflection
        for reflection in reflections
        if abs(float(reflection[4]) / float(radius_ideal_px) - 1.0) <= tol_frac
    ]
    if len(window) < 3:
        return None
    ordered = sorted(window, key=lambda reflection: -float(reflection[2]))
    reference_angle = ring_angle_deg(ordered[0])

    def partner(target_deg):
        best, best_key = None, None
        for reflection in ordered[1:]:
            distance = abs(
                _signed_angle_difference(ring_angle_deg(reflection), reference_angle)
                - target_deg
            )
            if distance > angle_tol_deg:
                continue
            key = (-float(reflection[2]), distance)
            if best_key is None or key < best_key:
                best, best_key = reflection, key
        return best

    partners = [partner(target) for target in HEXAGON_PARTNER_ANGLES_DEG]
    if any(entry is None for entry in partners) or partners[0] is partners[1]:
        return None
    triple = [ordered[0], *partners]
    members = triple + [
        (-float(m[0]), -float(m[1]), float(m[2]), float(m[3]), float(m[4]))
        for m in triple
    ]
    return {
        "radius": float(
            np.average([m[4] for m in members], weights=[m[2] for m in members])
        ),
        "members": members,
        "total_amplitude": float(sum(m[2] for m in members)),
    }


# --------------------------------------------------------------------------- #
# Lawler-Fujita core
# --------------------------------------------------------------------------- #
def ideal_radius_px(size_nm, a_nm):
    """Radius of the first-order (1x1) hexagonal ring in FFT pixels."""
    return 2.0 * float(size_nm) / (np.sqrt(3.0) * float(a_nm))


def wave_vectors_nm_inv(q_px, size_nm):
    """Wave vector of an FFT pixel index as the physical vector in rad/nm.

    A peak at FFT index ``q_px`` modulates the image as
    ``exp(i 2 pi (q_px . r_px) / n)`` with ``r_px`` the pixel index, hence
    ``K = 2 pi q_px / size_nm`` in rad/nm and a displacement ``u`` in nm gives the
    local phase ``theta = K . u``.
    """
    return 2.0 * np.pi * np.asarray(q_px, dtype=float) / float(size_nm)


def _wrapped_index(delta, n):
    """Signed distance on the periodic FFT index grid."""
    return (delta + n / 2.0) % n - n / 2.0


def demodulate(image, q_px, lambda_nm, size_nm, nan_policy="mean"):
    """Complex lock-in field of one Bragg peak: low-pass of ``image * exp(-i Q r)``.

    The low-pass is the Gaussian ``exp(-lambda^2 |k|^2 / 2)`` in q space around
    ``q_px`` (the paper's disk of radius ``delta_q = 1/lambda``): ``lambda_nm`` is
    the real-space scale of the distortion kept.  ``|psi|`` is the local amplitude
    of the lattice, ``arg psi`` the local lattice phase.

    NaN pixels are filled with the finite mean before the FFT (policy ``mean``) or
    with zero (policy ``zero``); they are flagged invalid by the caller.
    """
    n = int(image.shape[0])
    filled = np.array(image, dtype=float, copy=True)
    missing = ~np.isfinite(filled)
    if missing.any():
        fill = float(np.mean(filled[~missing])) if nan_policy == "mean" else 0.0
        filled[missing] = fill
    spectrum = np.fft.fft2(filled - float(np.mean(filled)))
    index = np.arange(n, dtype=float)
    dq_x = _wrapped_index(index[None, :] - float(q_px[0]), n) / float(size_nm)
    dq_y = _wrapped_index(index[:, None] - float(q_px[1]), n) / float(size_nm)
    radius = np.hypot(dq_x, dq_y)
    mask = np.exp(-0.5 * (float(lambda_nm) * radius) ** 2)
    field = np.fft.ifft2(spectrum * mask)
    rows, cols = np.mgrid[:n, :n]
    phase_ramp = -2.0 * np.pi * (float(q_px[0]) * cols + float(q_px[1]) * rows) / n
    return field * np.exp(1j * phase_ramp)


def unwrap_phase(wrapped):
    """Unwrap a wrapped phase map by least squares (Poisson, DCT, Neumann edges).

    The wrapped phase is turned into wrapped differences, whose divergence is the
    right-hand side of a discrete Poisson equation; the solution is the phase whose
    gradient best matches those differences.  A sequential row/column unwrap is not
    used instead because a single phase node (the complex lock-in field has nodes)
    seeds a 2 pi dislocation that the sequential pass then carries along an entire
    row.  With the least-squares solution such a defect stays local.

    The Neumann (mirror) boundary of the cosine transform is the natural choice: it
    assumes no phase gradient outside the image.
    """
    phase = np.asarray(wrapped, dtype=float)
    rows, cols = phase.shape
    dx = np.zeros_like(phase)
    dy = np.zeros_like(phase)
    dx[:, :-1] = _wrap_radians(phase[:, 1:] - phase[:, :-1])
    dy[:-1, :] = _wrap_radians(phase[1:, :] - phase[:-1, :])
    divergence = np.zeros_like(phase)
    divergence[:, :-1] += dx[:, :-1]
    divergence[:, 1:] -= dx[:, :-1]
    divergence[:-1, :] += dy[:-1, :]
    divergence[1:, :] -= dy[:-1, :]
    eigenvalues = (
        2.0 * np.cos(np.pi * np.arange(rows) / rows)[:, None]
        + 2.0 * np.cos(np.pi * np.arange(cols) / cols)[None, :]
        - 4.0
    )
    eigenvalues[0, 0] = 1.0
    solution = dctn(divergence, type=2, norm=None) / eigenvalues
    return idctn(solution, type=2, norm=None)


def _wrap_radians(values):
    """Wrap angles into (-pi, pi]."""
    return (np.asarray(values, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


def lockin_phase(image, q_px, lambda_nm, size_nm, amplitude_fraction=0.10):
    """Local lattice phase of one direction plus its amplitude and validity mask.

    Returns ``(theta, amplitude, valid, threshold)`` where ``theta`` is unwrapped
    and already carried to the ``theta_bar = 0`` gauge (its mean over the valid
    pixels is subtracted), ``amplitude`` is ``|psi|``, ``valid`` marks the pixels
    whose lock-in amplitude reaches ``amplitude_fraction`` of the median amplitude
    and ``threshold`` is that absolute amplitude threshold.

    The unwrap runs on the same mask: where ``|psi|`` is below the threshold the
    phase is not defined (the complex field has a node) and its wrapped value would
    seed a 2 pi dislocation that the sequential unwrap then carries along a whole
    row, so those pixels take the wrapped phase of the nearest reliable pixel
    instead.  They stay invalid in ``valid``.
    """
    field = demodulate(image, q_px, lambda_nm, size_nm)
    amplitude = np.abs(field)
    reference = float(np.median(amplitude))
    threshold = float(amplitude_fraction) * reference
    valid = np.isfinite(amplitude) & (amplitude >= threshold)
    wrapped = np.angle(field)
    if not bool(valid.all()):
        if not bool(valid.any()):
            raise ValueError(
                "the lock-in field is below the amplitude threshold everywhere"
            )
        indices = distance_transform_edt(
            ~valid, return_distances=False, return_indices=True
        )
        wrapped = wrapped[tuple(indices)]
    theta = unwrap_phase(wrapped)
    if np.any(valid):
        theta = theta - float(np.mean(theta[valid]))
    return theta, amplitude, valid, threshold


def displacement_from_phase(theta_a, theta_b, q_a_px, q_b_px, size_nm):
    """Displacement field in nm from the two local phases.

    ``u = K^-1 (theta_bar - theta)`` with ``K`` the 2x2 matrix of the reference
    wave vectors in rad/nm and ``theta_bar = 0`` by the gauge fixed in
    :func:`lockin_phase`.  Returns an array of shape ``(2, n, n)`` holding
    ``u_x`` and ``u_y`` in nm.
    """
    k_matrix = np.vstack(
        [wave_vectors_nm_inv(q_a_px, size_nm), wave_vectors_nm_inv(q_b_px, size_nm)]
    )
    determinant = float(np.linalg.det(k_matrix))
    if not np.isfinite(determinant) or abs(determinant) < 1e-12:
        raise ValueError("the two reference wave vectors are collinear")
    phases = np.stack(
        [np.asarray(theta_a, dtype=float), np.asarray(theta_b, dtype=float)]
    )
    flat = np.reshape(phases, (2, -1))
    solved = np.linalg.solve(k_matrix, -flat)
    return np.reshape(solved, (2, *np.asarray(theta_a).shape))


def warp_by_field(image, u_nm, size_nm, valid=None, pad=10, order=3):
    """Resample ``image`` along the displacement field: ``out(r) = image(r + u(r))``.

    ``u_nm`` has shape ``(2, n, n)`` (``u_x``, ``u_y`` in nm).  The canvas grows by
    ``2 * half`` pixels, ``half = ceil(max |u|) + pad``, so every displaced sample
    stays inside it; outside the input the result is NaN.  The nm-per-pixel scale
    is kept, hence the corrected field of view is ``size_nm * n_out / n``.
    ``valid`` marks the pixels whose displacement is trustworthy; elsewhere the
    displacement is set to zero (that pixel is copied rather than moved).
    """
    array = np.asarray(image, dtype=float)
    n = int(array.shape[0])
    scale = float(size_nm) / n  # nm per pixel
    u_px = np.asarray(u_nm, dtype=float) / scale
    if valid is not None:
        invalid = ~np.asarray(valid, dtype=bool)
        u_px = np.where(invalid[None, :, :], 0.0, u_px)
    finite = np.isfinite(u_px)
    u_px = np.where(finite, u_px, 0.0)
    magnitude = float(np.max(np.abs(u_px))) if u_px.size else 0.0
    half = int(np.ceil(magnitude)) + int(pad)
    n_out = n + 2 * half
    u_x_pad = np.pad(u_px[0], half, mode="constant", constant_values=0.0)
    u_y_pad = np.pad(u_px[1], half, mode="constant", constant_values=0.0)
    rows, cols = np.mgrid[:n_out, :n_out]
    row = (rows - half) + u_y_pad
    col = (cols - half) + u_x_pad
    # Missing input pixels would poison the spline prefilter, so the data is filled
    # for the interpolation and the missing mask is carried through separately.
    present = np.isfinite(array)
    fill_value = float(np.mean(array[present])) if bool(present.any()) else 0.0
    prepared = np.where(present, array, fill_value)
    corrected = map_coordinates(
        prepared,
        [row, col],
        order=int(order),
        mode="constant",
        cval=np.nan,
        prefilter=int(order) > 1,
    )
    if not bool(present.all()):
        coverage = map_coordinates(
            present.astype(float),
            [row, col],
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
        corrected = np.where(coverage >= 0.5, corrected, np.nan)
    return corrected, n_out, half, magnitude


def resample_field(u_nm, valid, size_nm_reference, n_target, size_nm_target, order=1):
    """Carry a displacement field (nm) from the reference grid to a target grid.

    ``u`` is physical, so only the grid changes: the target pixel centres are
    mapped to physical coordinates and the reference field is interpolated there.
    An identical grid is returned verbatim, which makes the apply stage reproduce
    the fit stage bit for bit.  Outside the reference canvas the field is NaN.
    """
    array = np.asarray(u_nm, dtype=float)
    n_reference = int(array.shape[1])
    if int(n_target) == n_reference and float(size_nm_target) == float(
        size_nm_reference
    ):
        return array, np.asarray(valid, dtype=bool)
    pixel_reference = float(size_nm_reference) / n_reference
    pixel_target = float(size_nm_target) / int(n_target)
    rows, cols = np.mgrid[: int(n_target), : int(n_target)]
    # pixel centre i covers the middle of pixel i, in both grids
    target_x = (cols + 0.5) * pixel_target
    target_y = (rows + 0.5) * pixel_target
    src_x = target_x / pixel_reference - 0.5
    src_y = target_y / pixel_reference - 0.5
    resampled = np.stack(
        [
            map_coordinates(
                array[component],
                [src_y, src_x],
                order=int(order),
                mode="constant",
                cval=np.nan,
                prefilter=int(order) > 1,
            )
            for component in (0, 1)
        ]
    )
    mask = (
        map_coordinates(
            np.asarray(valid, dtype=float),
            [src_y, src_x],
            order=0,
            mode="constant",
            cval=0.0,
        )
        > 0.5
    )
    return resampled, mask


def max_phase_step(phase):
    """Largest per-pixel phase step of an unwrapped phase map, in radians.

    The unwrap is unambiguous while the true step stays below pi; the fit reports
    this number as the sampling diagnostic of the lock-in.
    """
    difference = np.hypot(*np.gradient(np.asarray(phase, dtype=float)))
    return float(np.max(difference))


def wrapped_residual(phases_deg):
    """Wrapped RMS of a sum of phase maps, in degrees (consistency diagnostics)."""
    total = np.zeros_like(np.asarray(phases_deg[0], dtype=float))
    for phase in phases_deg:
        total = total + np.asarray(phase, dtype=float)
    wrapped = (total + 180.0) % 360.0 - 180.0
    return float(np.sqrt(np.mean(wrapped**2))), wrapped


# --------------------------------------------------------------------------- #
# artifact writing
# --------------------------------------------------------------------------- #
def save_map(path, data, cmap="inferno", vmin=None, vmax=None, dpi=150, size=4.0):
    """Write one LF map as a bare PNG preview (gwyddion-style clean frame)."""
    import matplotlib.pyplot as plt

    cmap_object = plt.get_cmap(cmap).copy()
    cmap_object.set_bad(color=BAD_COLOR)
    finite = np.asarray(data, dtype=float)
    fig, ax = plt.subplots(figsize=(size, size))
    ax.imshow(finite, cmap=cmap_object, origin="lower", vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
