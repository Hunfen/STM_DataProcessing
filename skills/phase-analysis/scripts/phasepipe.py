"""Geometry and phase-extraction pipeline used by the synthetic methodology study.

The stage order mirrors the topographic phase pipeline of the analysis:

    image -> apodised FFT2 -> reflection detection (sub-pixel)
          -> ring radii -> the (1x1, r3) ring pair identified by the radius ratio
          -> per-reflection Gaussian-window demodulation (the local-q-map engine)
          -> theta(r) = arg(psi_q(r))                     (no q.r ramp)
          -> phase statistics over every valid pixel -> lattice-referenced gauge fix
          -> paper-style pairwise phase differences of two reflections

Phase convention
----------------
Every per-reflection field comes from ``localqmap.demodulate`` (the engine of the
sibling ``local-q-map`` skill), reached through ``gaussian_field``:

    psi_q(r)   = FFT^-1{ FFT[ T(r) exp(-i q.r) ] * exp(-Lambda^2 |k|^2 / 2) }
    theta_q(r) = arg psi_q(r)  ~  +phi_q(r)          (the demodulated convention)

so the map carries no ``q.r`` ramp and no per-reflection constant: a reflection at
``q`` whose real-space content is ``A cos(q.r + phi)`` contributes
``psi_q ~ (A/2) exp(+i phi)``.  Because the window's ``k = 0`` weight is exactly
one, ``sum_r psi_q(r) = sum_r T(r) exp(-i q.r)`` holds exactly for every
``Lambda``; with a real ``T`` the Friedel relation ``psi_{-q} = conj(psi_q)`` is
exact, and a circular roll of the canvas acts as the exact translation

    psi'_q(r) = exp(-i q.delta) * psi_q(r - delta)       (T'(r) = T(r - delta))

which reduces to a constant phase offset ``exp(-i q.delta)`` wherever the
demodulated field itself is constant (a single plane wave at exactly ``q``).

Names
-----
Only two geometric objects are ever referenced:

``ring_1x1``  the outer ring of the pair, used as the reference ring (its six
              reflections define the origin through the least-squares fit).
``ring_r3``   the ring whose radius is ``1/sqrt(3)`` of ``ring_1x1`` (the pairing
              is accepted only inside a tolerance, otherwise the study reports
              "r3 ring not found" instead of guessing).

Nothing in this module attaches a physical meaning to a radius or a phase; the
wavevectors are plain FFT-pixel coordinates and a "ring" is a set of reflections
at a common radius.

The synthetic generator builds a real image as a sum of plane waves

    x(r) = ref_amp * sum_{j in ring_1x1} cos(q_j . r + ref_phase)
         + sum_d chi_d(r) * a_d * sum_{j in ring_r3} cos(q_j . r + Phi_d)

so that a region ``d`` of the sample contributes ``w_d = a_d * |region_d|`` to
every reflection of ``ring_r3`` with one common phase ``Phi_d``.  That is the
mathematical model the estimators are validated on; the weights and phases are
known exactly, which is what makes the study a ground-truth experiment.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from phasemath import TWO_PI, circ_mean, linear_median_fwhm, weighted_stats

_HERE = Path(__file__).resolve()


def local_q_map_scripts():
    """Directory holding the engine of the sibling skill ``local-q-map``.

    The two skills are installed side by side, so the engine sits either next to
    this skill directory (``<skills>/phase-analysis/scripts`` -> ``<skills>``,
    which covers a repository checkout *and* the installed library
    ``~/.agents/skills``) or in the installed library while this skill runs from a
    copy somewhere else (a staging or scratch tree).  Both are searched; nothing
    is copied and the engine itself is never modified.
    """
    candidates = [
        _HERE.parents[2] / "local-q-map" / "scripts",
        Path.home() / ".agents" / "skills" / "local-q-map" / "scripts",
    ]
    for candidate in candidates:
        if (candidate / "localqmap.py").is_file():
            return candidate
    raise ModuleNotFoundError(
        "the demodulation engine localqmap.py of the sibling skill local-q-map "
        "was not found; looked in "
        + ", ".join(str(candidate) for candidate in candidates)
    )


# The demodulation engine of the sibling skill ``local-q-map`` is used as delivered.
_LOCAL_Q_MAP_SCRIPTS = local_q_map_scripts()
if str(_LOCAL_Q_MAP_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_LOCAL_Q_MAP_SCRIPTS))

import localqmap  # noqa: E402  (needs the path insert above)

SQRT3 = float(np.sqrt(3.0))


# --------------------------------------------------------------------------- #
# Fourier convention (mirrors the package compute_fft2)
# --------------------------------------------------------------------------- #
def window_array(name, n):
    if name is None or str(name).lower() in ("none", "rect", "boxcar"):
        return np.ones((n, n), dtype=float)
    key = str(name).lower()
    if key == "hann":
        return np.outer(np.hanning(n), np.hanning(n))
    if key == "hamming":
        return np.outer(np.hamming(n), np.hamming(n))
    raise ValueError(f"unsupported window {name!r}")


def compute_fft2(image, window="hann"):
    """``fftshift(fft2(image * window))`` -- the spectrum used for detection."""
    arr = np.asarray(image, dtype=float)
    arr = np.nan_to_num(arr, nan=float(np.nanmean(arr)))
    return np.fft.fftshift(np.fft.fft2(arr * window_array(window, arr.shape[0])))


def gaussian_field(topo, q_px, lambda_nm, nm_per_px):
    """Demodulated complex local field ``psi_q`` of one reflection.

    Thin wrapper over the sibling skill's engine: the wavevector is converted from
    the FFT-pixel convention used everywhere in this skill (an offset from the DC
    bin, i.e. ``numpy.fft.fftfreq(n) * n``) into ``q_rad_px = 2 pi q_px / N`` and
    handed to ``localqmap.demodulate`` together with the Gaussian window width
    ``lambda_nm`` (the local-q-map default is 3.0 nm).  The returned field is
    ``psi_q(r) ~ (A/2) exp(+i phi_q(r))``: no ``q.r`` ramp, no per-reflection
    constant, and ``arg psi_q`` is the demodulated phase used by every estimator
    of this skill.
    """
    array = np.asarray(topo, dtype=float)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"the canvas must be square, got {array.shape}")
    n = int(array.shape[0])
    q_rad_px = (TWO_PI / n) * np.asarray(q_px, dtype=float)
    return localqmap.demodulate(
        array, q_rad_px=q_rad_px, lambda_nm=lambda_nm, nm_per_px=nm_per_px
    )


def window_px(lambda_nm, nm_per_px):
    """The Gaussian window width of the engine in pixels (``Lambda / nm_per_px``)."""
    return float(lambda_nm) / float(nm_per_px)


# --------------------------------------------------------------------------- #
# reflection detection
# --------------------------------------------------------------------------- #
def local_maxima(mag, min_radius=6.0):
    """3x3 strict local maxima of ``mag`` outside ``min_radius`` of the DC bin."""
    n = mag.shape[0]
    centre = n // 2
    interior = np.zeros_like(mag, dtype=bool)
    interior[1:-1, 1:-1] = True
    yy, xx = np.mgrid[:n, :n]
    interior &= np.hypot(xx - centre, yy - centre) >= min_radius
    best = np.ones_like(mag, dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            shifted = np.roll(np.roll(mag, dy, axis=0), dx, axis=1)
            best &= mag > shifted
    return np.flatnonzero((best & interior).ravel())


def subpixel_offset(patch):
    """Parabolic sub-bin offset (dy, dx) of the maximum of a 3x3 patch."""
    iy, ix = np.unravel_index(int(np.argmax(patch)), patch.shape)
    if not (1 <= iy <= patch.shape[0] - 2 and 1 <= ix <= patch.shape[1] - 2):
        return 0.0, 0.0

    def axis(zm, z0, zp):
        denom = zm - 2.0 * z0 + zp
        if denom >= 0.0:
            return 0.0
        return float(np.clip(0.5 * (zm - zp) / denom, -0.5, 0.5))

    log_patch = np.log(np.maximum(patch, np.finfo(float).tiny))
    dy = axis(log_patch[iy - 1, ix], log_patch[iy, ix], log_patch[iy + 1, ix])
    dx = axis(log_patch[iy, ix - 1], log_patch[iy, ix], log_patch[iy, ix + 1])
    return dy, dx


def detect_reflections(mag, min_radius=6.0, snr_floor=6.0, max_peaks=400):
    """Sub-pixel reflection list ``[(qx, qy, amplitude, snr, radius)]``.

    Noise scale: the median absolute deviation of ``mag`` outside the central
    ``min_radius`` region, scaled to a Gaussian sigma.  Only 3x3 local maxima
    above ``snr_floor`` sigma are kept.
    """
    n = mag.shape[0]
    centre = n // 2
    yy, xx = np.mgrid[:n, :n]
    outside = np.hypot(xx - centre, yy - centre) >= min_radius
    values = mag[outside]
    mad = float(np.median(np.abs(values - np.median(values)))) / 0.6744897501960817
    sigma = mad if mad > 0 else float(np.std(values))
    threshold = float(np.median(values)) + snr_floor * sigma
    flat = local_maxima(mag, min_radius)
    records = []
    for index in flat:
        iy, ix = divmod(int(index), n)
        if mag[iy, ix] < threshold:
            continue
        patch = mag[iy - 1 : iy + 2, ix - 1 : ix + 2]
        dy, dx = subpixel_offset(patch)
        qx = ix + dx - centre
        qy = iy + dy - centre
        records.append(
            (
                float(qx),
                float(qy),
                float(mag[iy, ix]),
                float(mag[iy, ix] / sigma if sigma > 0 else np.inf),
                float(np.hypot(qx, qy)),
            )
        )
    records.sort(key=lambda item: -item[2])
    return records[:max_peaks]


def group_rings(reflections, tol_frac=0.02, min_members=6):
    """Cluster reflections by radius (greedy, strongest first).

    Returns a list of dicts ``{radius, members, total_amplitude}`` sorted by
    descending total amplitude; ``members`` are the reflection tuples.
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


def select_ring_pair(rings, ratio=SQRT3, tol_frac=0.03, expect=6):
    """Identify the ``(ring_1x1, ring_r3)`` pair by the radius ratio only.

    Every pair of rings with at least ``expect`` members is considered; the pair
    whose radius ratio is closest to ``ratio`` wins, provided it is inside
    ``tol_frac`` (relative).  Returns ``(outer, inner, ratio)`` or ``None`` when
    no pair qualifies -- the caller must report the honest "r3 ring not found"
    branch rather than guessing an unpaired ring.
    """
    best = None
    for i, outer in enumerate(rings):
        for j, inner in enumerate(rings):
            if i == j or inner["radius"] >= outer["radius"]:
                continue
            if len(outer["members"]) < expect or len(inner["members"]) < expect:
                continue
            observed = outer["radius"] / inner["radius"]
            deviation = abs(observed - ratio) / ratio
            if deviation > tol_frac:
                continue
            if best is None or deviation < best[3]:
                best = (outer, inner, observed, deviation)
    if best is None:
        return None
    return best[0], best[1], best[2]


def ring_six(members, expect=6):
    """The six strongest members of a ring, ordered by polar angle."""
    ordered = sorted(members, key=lambda item: -item[2])[:expect]
    ordered.sort(key=lambda item: np.arctan2(item[1], item[0]))
    return ordered


def snap_to_true_vectors(members, true_vectors, expect=6):
    """Replace each detected member's wavevector by the nearest true one.

    Used by the synthetic study to separate the *phase* estimators from the
    *peak-position* estimator: with a mixed-phase field, the magnitude maximum of
    a reflection is displaced from the wavevector itself, so the detected q
    carries a configuration-dependent bias (measured and reported separately).
    The mask centre and the ring membership still come from the detection.
    """
    ordered = sorted(members, key=lambda item: -item[2])[:expect]
    snapped, used = [], set()
    for record in sorted(ordered, key=lambda item: np.arctan2(item[1], item[0])):
        best, best_distance = None, np.inf
        for index, vector in enumerate(true_vectors):
            if index in used:
                continue
            angle = abs(
                np.arctan2(record[1], record[0]) - np.arctan2(vector[1], vector[0])
            )
            angle = min(angle, TWO_PI - angle)
            if angle < best_distance:
                best, best_distance = index, angle
        used.add(best)
        vector = true_vectors[best]
        snapped.append(
            (
                float(vector[0]),
                float(vector[1]),
                float(record[2]),
                float(record[3]),
                float(np.hypot(*vector)),
            )
        )
    return snapped


# --------------------------------------------------------------------------- #
# per-reflection phase field and statistics
# --------------------------------------------------------------------------- #
def theta_field(psi):
    """``theta(r) = arg(psi(r))`` wrapped into ``[0, 2 pi)``.

    The single phase convention of this skill: the demodulated field already
    carries no ``q.r`` ramp, so the phase of a reflection is its argument with no
    further carrier removal and no per-reflection constant.
    """
    return np.mod(np.angle(np.asarray(psi)), TWO_PI)


def sample_mask(amp, valid):
    """The single sample of every per-reflection estimator: all valid pixels.

    ``valid`` is the canvas validity (``isfinite`` of the input CSV, i.e. every
    pixel the geometry correction did not turn into NaN).  The only pixels removed
    here are those that carry no weight at all -- a non-finite or exactly zero
    demodulated amplitude, which contributes nothing to an amplitude-weighted
    estimator.  There is no percentile, no threshold and no selection rule: the
    sample is fixed by the data, and the maps of the atlas are drawn with exactly
    this mask, so the visible pixels of a figure and the pixels of its statistics
    are one and the same set.
    """
    amp = np.asarray(amp, dtype=float)
    return np.asarray(valid, dtype=bool) & np.isfinite(amp) & (amp > 0.0)


def reflection_stats(theta, amp, valid, bins=3600, smooth_deg=2.0):
    """Phase and amplitude statistics of one reflection on the single sample.

    The sample is ``sample_mask(amp, valid)``: every valid pixel, weighted by the
    local demodulated amplitude ``|psi(r)|``.  The amplitude-weighted circular mean
    over that sample equals ``arg sum_r T(r) exp(-i q.r)`` (the engine's window has
    ``k = 0`` weight one, so the mean does not depend on the window width), while
    the median / IQR / FWHM / clusters describe the shape of that same sample.
    """
    sample = sample_mask(amp, valid)
    phase = weighted_stats(theta[sample], amp[sample], bins=bins, smooth_deg=smooth_deg)
    amp_median, amp_fwhm, _ = linear_median_fwhm(amp[sample])
    _mean, resultant, _total = circ_mean(theta[sample], amp[sample])
    count = int(np.count_nonzero(sample))
    return {
        "phase": phase,
        "n_samples": count,
        "amp_mean": float(np.mean(amp[sample])) if count else float("nan"),
        "amp_median": float(amp_median),
        "amp_fwhm": float(amp_fwhm),
        "resultant_R": float(resultant),
    }


# --------------------------------------------------------------------------- #
# paper-style pairwise phase-difference analysis
# --------------------------------------------------------------------------- #
WITHIN_PAIRS = ((0, 1), (2, 3), (4, 5))


def pair_phase_diff_field(psi_j, psi_k):
    """``D_jk(r) = wrap(arg psi_j(r) - arg psi_k(r))`` in ``(-pi, pi]``.

    ``arg(z_j * conj(z_k)) = arg z_j - arg z_k`` (mod ``2 pi``), so the difference
    of two demodulated fields is one complex product; neither field carries a
    ``q.r`` ramp, so the difference carries none either.
    """
    product = np.asarray(psi_j, dtype=complex) * np.conj(
        np.asarray(psi_k, dtype=complex)
    )
    return np.asarray(np.angle(product), dtype=float)


def pair_amplitude_diff_field(psi_j, psi_k):
    """``a_jk(r) = (|psi_j| - |psi_k|) / (|psi_j| + |psi_k|)`` in ``[-1, 1]``."""
    left = np.abs(np.asarray(psi_j, dtype=complex))
    right = np.abs(np.asarray(psi_k, dtype=complex))
    total = left + right
    out = np.zeros_like(total)
    good = total > 0.0
    out[good] = (left[good] - right[good]) / total[good]
    return out


def pair_weight_field(psi_j, psi_k):
    """``|psi_j psi_k|``, the amplitude weight of a pair sample."""
    return np.abs(np.asarray(psi_j, dtype=complex)) * np.abs(
        np.asarray(psi_k, dtype=complex)
    )


def pair_analysis(
    psi_j,
    psi_k,
    valid_j,
    valid_k,
    bins=3600,
    smooth_deg=2.0,
):
    """The three fields and the statistics of one reflection pair.

    The sample of a pair is the intersection of the two validity masks, restricted
    to pixels with a non-zero weight ``|psi_j psi_k|`` (``sample_mask``).  The
    phase-difference statistics use the circular estimators of ``phasemath`` with
    that weight; the amplitude-difference statistic is the weighted linear median
    of ``a_jk`` over the same sample.  The pair histogram is a one-dimensional
    circular histogram of ``D`` itself over ``(-pi, pi]`` and is built by the atlas
    from the returned field, weight and mask, so the drawn distribution and the
    statistics are the same sample.
    """
    weight = pair_weight_field(psi_j, psi_k)
    overlap = np.asarray(valid_j, dtype=bool) & np.asarray(valid_k, dtype=bool)
    mask = sample_mask(weight, overlap)
    phase_diff = pair_phase_diff_field(psi_j, psi_k)
    amp_diff = pair_amplitude_diff_field(psi_j, psi_k)
    stats = weighted_stats(
        phase_diff[mask], weight[mask], bins=bins, smooth_deg=smooth_deg
    )
    amp_median, amp_fwhm, _ = linear_median_fwhm(amp_diff[mask], weight[mask])
    return {
        "phase_diff": phase_diff,
        "amp_diff": amp_diff,
        "weight": weight,
        "mask": mask,
        "phase_diff_mean_deg": float(stats["mean_deg"]),
        "phase_diff_median_deg": float(stats["median_deg"]),
        "phase_diff_R": float(stats["resultant_R"]),
        "phase_diff_fwhm_deg": float(stats["fwhm_deg"]),
        "phase_diff_fwhm_deconv_deg": float(stats["fwhm_deconv_deg"]),
        "phase_diff_iqr_deg": float(stats["iqr_deg"]),
        "phase_diff_circ_std_deg": float(stats["circ_std_deg"]),
        "phase_diff_n_clusters": int(stats["n_clusters"]),
        "phase_diff_clusters": stats["clusters"],
        "amp_diff_median": float(amp_median),
        "amp_diff_fwhm": float(amp_fwhm),
        "n_valid": int(np.count_nonzero(mask)),
    }


def within_ring_pairs(records, expect=6):
    """Index pairs ``(j, k)`` analysed inside one ring: (p0,p1), (p2,p3), (p4,p5).

    The peaks of a ring are numbered clockwise from 12 o'clock, so these three
    pairs join reflections 60 degrees apart and deliberately avoid the Friedel
    pairs (p0,p3), (p1,p4), (p2,p5): for a Friedel pair ``psi_{-q} = conj(psi_q)``,
    so its phase-difference field is a trivial function of one field instead of an
    independent pair observable.
    """
    return [(j, k) for j, k in WITHIN_PAIRS if k < len(records)]


def friedel_pairs(peaks_q):
    """Index pairs of the reflection list whose wavevectors are antiparallel."""
    pairs = []
    used = set()
    for i, qi in enumerate(peaks_q):
        if i in used:
            continue
        best, best_norm = None, np.inf
        for j, qj in enumerate(peaks_q):
            if j == i or j in used:
                continue
            norm = float(np.hypot(qi[0] + qj[0], qi[1] + qj[1]))
            if norm < best_norm:
                best, best_norm = j, norm
        if best is not None:
            pairs.append((i, best, best_norm))
            used.update({i, best})
    return pairs


def triple_selection(peaks_q):
    """Index triple with the smallest ``|sum q|``, chosen deterministically.

    A ring of six reflections contains two triples whose wavevectors sum to zero,
    and they are antipodal to each other; both are equally legal and their phase
    sums differ by an overall sign.  Ties are therefore broken by the smallest
    polar angle of the triple, so the choice is reproducible, and callers report
    the mirror explicitly (``triple_summary``).
    """
    import itertools

    best = None
    for combo in itertools.combinations(range(len(peaks_q)), 3):
        norm = float(
            np.hypot(
                sum(peaks_q[i][0] for i in combo), sum(peaks_q[i][1] for i in combo)
            )
        )
        first = min(np.arctan2(peaks_q[i][1], peaks_q[i][0]) % TWO_PI for i in combo)
        key = (round(norm, 6), first)
        if best is None or key < best[0]:
            best = (key, combo, norm)
    return best[1], best[2]


# --------------------------------------------------------------------------- #
# ring level analysis
# --------------------------------------------------------------------------- #
def analyse_ring(
    topo,
    valid,
    ring,
    lambda_nm,
    nm_per_px,
    bins=3600,
    smooth_deg=2.0,
    expect=6,
    prefix="",
    peaks_override=None,
):
    """Phase statistics of all reflections of one ring.

    Every field is demodulated with ``gaussian_field`` at the wavevector of the
    reflection, so the phase of a reflection is ``theta = arg(psi)`` with no ramp
    and no per-reflection constant.

    ``peaks_override`` replaces the detected six reflections by a supplied list of
    the same tuples (used by the synthetic study to demodulate with the
    ground-truth wavevectors).  ``psi`` maps the peak name to the complex
    demodulated field, and ``fields`` to ``(q_px, theta, amp)`` for the gauge fix,
    the per-pixel triple product and the pairwise analysis.

    Returns ``(records, fields, psi)``.
    """
    peaks = (
        ring_six(ring["members"], expect=expect)
        if peaks_override is None
        else list(peaks_override)
    )
    n = int(np.asarray(topo).shape[0])
    centre = n // 2
    records, fields, psi_map = [], {}, {}
    for i, (qx, qy, amplitude, snr, radius) in enumerate(peaks):
        name = f"{prefix}p{i}"
        psi = gaussian_field(topo, (qx, qy), lambda_nm, nm_per_px)
        theta = theta_field(psi)
        amp = np.abs(psi)
        stats = reflection_stats(theta, amp, valid, bins=bins, smooth_deg=smooth_deg)
        records.append(
            {
                "name": name,
                "q_px": (float(qx), float(qy)),
                "integer": (round(qx) + centre, round(qy) + centre),
                "subpixel": (float(qx - round(qx)), float(qy - round(qy))),
                "radius_px": float(radius),
                "fft_amplitude": float(amplitude),
                "snr": float(snr),
                "stats": stats,
            }
        )
        fields[name] = ((float(qx), float(qy)), theta, amp)
        psi_map[name] = psi
    return records, fields, psi_map


def pair_summary(records, key="phase", quantity="mean_deg"):
    """Friedel pair sums and their deviation from 360 degrees."""
    qs = [record["q_px"] for record in records]
    out = []
    for i, j, norm in friedel_pairs(qs):
        a = records[i]["stats"][key][quantity]
        b = records[j]["stats"][key][quantity]
        total = (a + b) % 360.0
        out.append(
            {
                "pair": (records[i]["name"], records[j]["name"]),
                "sum_q_norm_px": norm,
                "sum_deg": total,
                "deviation_from_360_deg": abs(total - 360.0)
                if total > 180.0
                else total,
                "a_deg": a,
                "b_deg": b,
            }
        )
    return out


def triple_summary(
    records,
    fields,
    valid,
    key="phase",
    quantity="mean_deg",
    bins=3600,
    smooth_deg=2.0,
):
    """Scalar triple sum and the per-pixel triple-product phase of one ring.

    ``scalar``  the three per-reflection phases added as angles
                (``= 3 * Phi_bar`` in the mixture model, exactly gauge invariant
                when the three wavevectors sum to zero).
    ``field``   the per-pixel sum of the three demodulated phase fields, weighted by
                the product of the three amplitudes: the estimator this skill calls
                the triple product.  Both are computed on every valid pixel; only
                the mean, the median, R, the FWHM and the cluster count are kept in
                the JSON (the triple-product map and its distribution are no longer
                drawn).
    """
    qs = [record["q_px"] for record in records]
    combo, q_sum_norm = triple_selection(qs)
    names = [records[i]["name"] for i in combo]
    values = [records[i]["stats"][key][quantity] for i in combo]
    scalar = float(np.sum(values) % 360.0)
    phi_sum = np.zeros_like(fields[names[0]][1])
    amp_prod = np.ones_like(fields[names[0]][1])
    for name in names:
        phi_sum = phi_sum + fields[name][1]
        amp_prod = amp_prod * fields[name][2]
    theta = np.mod(phi_sum, TWO_PI)
    sample = sample_mask(amp_prod, valid)
    field = weighted_stats(
        theta[sample], amp_prod[sample], bins=bins, smooth_deg=smooth_deg
    )
    # Friedel conjugates are exact negatives, so the antipodal triple is the mirror
    mirror_scalar = float((-scalar) % 360.0)
    return {
        "combo": names,
        "antipodal_combo": [
            (-qx, -qy) for qx, qy in (records[i]["q_px"] for i in combo)
        ],
        "q_sum_norm_px": q_sum_norm,
        "scalar_sum_deg": scalar,
        "scalar_sum_mirror_deg": mirror_scalar,
        "field": field,
        "field_mirror_mean_deg": float((-field["mean_deg"]) % 360.0),
        "field_n_samples": int(np.count_nonzero(sample)),
        "theta_field": theta,
        "amp_product": amp_prod,
    }


# --------------------------------------------------------------------------- #
# synthetic ground truth
# --------------------------------------------------------------------------- #
def _domain_mask(n, domain):
    """Soft region indicator ``chi_d(r)`` of one synthetic domain.

    Kinds: ``full`` (everything), ``band`` (a vertical stripe between the area
    fractions ``lo`` and ``hi``), ``disk`` and ``disk_c`` (its complement).  All
    edges are smooth on a scale of a few pixels so that the region boundary does
    not leak into the reflection masks.
    """
    kind = domain.get("kind", "full")
    if kind in ("full", "background"):
        return np.ones((n, n), dtype=float)
    yy, xx = np.mgrid[:n, :n]
    if kind == "band":
        edge = float(domain.get("edge", 6.0))
        lo, hi = float(domain["lo"]), float(domain["hi"])
        gate = np.ones(n, dtype=float)
        if lo > 0.0:
            gate = gate * 0.5 * (1.0 + np.tanh((np.arange(n) - lo * n) / edge))
        if hi < 1.0:
            gate = gate * 0.5 * (1.0 - np.tanh((np.arange(n) - hi * n) / edge))
        return np.repeat(gate[None, :], n, axis=0)
    if kind in ("disk", "disk_c"):
        cx, cy = domain.get("centre", (0.3, 0.55))
        radius = float(domain["radius_frac"]) * n
        edge = float(domain.get("edge", max(4.0, 0.06 * radius)))
        distance = np.hypot(xx - cx * n, yy - cy * n)
        chi = 0.5 * (1.0 - np.tanh((distance - radius) / edge))
        return 1.0 - chi if kind == "disk_c" else chi
    raise ValueError(f"unsupported domain kind {kind!r}")


def ring_wave(n, vectors, phase_rad):
    """Real ring modulation ``Re[e^{i Phi} sum_j e^{i q_j.r}]`` = ``sum_j cos(q_j.r + Phi)``.

    Only the three independent members of the ring are summed here.  That is the
    point: a ring of six reflections is three Friedel pairs, so the six are NOT
    independent -- adding six plain cosines with the same nominal phase lets each
    pair resolve into ``2 cos(q.r) cos(Phi)`` and erases the phase of the pair
    (the conjugate member adds ``e^{-i Phi}`` at the very same FFT bin).  Summing
    the three independent members puts ``Phi`` on them and ``-Phi`` on their
    conjugates, which is the relation any real image must satisfy.

    ``phase_rad`` is the phase of the pattern at the origin of the pixel grid; the
    pattern is moved by ``d`` pixels by adding ``(2*pi/n) q_j.d`` to *each* wave,
    i.e. by rolling the finished image.
    """
    yy, xx = np.mgrid[:n, :n]
    total = np.zeros((n, n), dtype=float)
    for qx, qy in vectors:
        total += np.cos((TWO_PI / n) * (qx * xx + qy * yy) + phase_rad)
    return total


def independent_triple(vectors):
    """Three of the six ring vectors whose wavevector sum is zero.

    Ordered by polar angle in ``[0, 2 pi)``; this is the triple that carries the
    phase ``Phi`` in the synthetic generator, its antipodal partner carries
    ``-Phi`` (see ``antipodal_triple``).
    """
    ordered = sorted(vectors, key=lambda item: np.arctan2(item[1], item[0]) % TWO_PI)
    return [ordered[0], ordered[2], ordered[4]]


def antipodal_triple(vectors):
    """The other triple with ``sum q = 0`` (the negative of ``independent_triple``)."""
    return [(-qx, -qy) for qx, qy in independent_triple(vectors)]


def synth_image(
    n,
    r1_px,
    domains,
    ref_amp=1.0,
    ref_phase_deg=0.0,
    noise=0.0,
    seed=0,
    rotation_deg=30.0,
    harmonics=(1,),
    rng=None,
):
    """Ground-truth synthetic image: reference ring plus region-dependent r3 ring.

    The reference ring (radius ``r1_px``) carries one phase everywhere; the r3
    ring (radius ``r1_px / sqrt(3)``, rotated by ``rotation_deg``) carries the
    phase ``Phi_d`` of the region the pixel belongs to.  ``domains`` is a list of
    dicts with ``kind`` (``background``/``disk``), ``amp``, ``phase_deg`` and, for
    a disk, ``centre`` and ``radius``.  ``harmonics`` multiplies the r3 wavevectors
    by the listed integers, which adds higher harmonics of the same phase (used to
    check that the estimators do not depend on the harmonic content).
    """
    generator = np.random.default_rng(seed) if rng is None else rng
    angles = np.arange(6) * np.pi / 3.0
    ref_vectors = [(r1_px * np.cos(a), r1_px * np.sin(a)) for a in angles]
    rotate = np.radians(rotation_deg)
    r3_radius = r1_px / SQRT3
    r3_vectors = [
        (r3_radius * np.cos(a + rotate), r3_radius * np.sin(a + rotate)) for a in angles
    ]
    ref_triple = independent_triple(ref_vectors)
    image = ring_wave(n, ref_triple, np.radians(ref_phase_deg)) * ref_amp
    for domain in domains:
        chi = _domain_mask(n, domain)
        for harmonic in harmonics:
            vectors = [
                (harmonic * qx, harmonic * qy)
                for qx, qy in independent_triple(r3_vectors)
            ]
            phase = harmonic * np.radians(domain["phase_deg"])
            image = image + chi * domain["amp"] * ring_wave(n, vectors, phase)
    if noise:
        image = image + float(noise) * generator.standard_normal(image.shape)
    return image


def true_mixture(domains, n, r1_px, window="hann"):
    """Exact weights, phasor sum and per-domain phases of a synthetic configuration.

    The weight of a region in the *analysed* field is not its plain area: the
    pipeline multiplies the image by an apodisation window before the FFT, so a
    reflection's complex amplitude is

        F(q) = (1/2) sum_d a_d e^{i Phi_d} sum_r chi_d(r) W(r),

    i.e. the regions are weighted by their *windowed* integrals
    ``w_d = a_d sum_r chi_d(r) W(r)``.  Both weightings are reported: the windowed
    one is the ground truth for every FFT-domain quantity (the scalar phase sum,
    the coherence, the global peak phase), while the plain area fraction is what a
    per-pixel count would see.
    """
    window_profile = window_array(window, n) if window else np.ones((n, n))
    weights, area_weights, phases, phasors = [], [], [], []
    for domain in domains:
        chi = _domain_mask(n, domain)
        weights.append(float(domain["amp"] * np.sum(chi * window_profile)))
        area_weights.append(float(domain["amp"] * np.sum(chi)))
        phases.append(float(domain["phase_deg"] % 360.0))
        phasors.append(np.exp(1j * np.radians(domain["phase_deg"])))
    weights = np.asarray(weights)
    area_weights = np.asarray(area_weights)
    total = float(np.sum(weights))
    total_area = float(np.sum(area_weights))
    phasor = np.sum(weights * np.asarray(phasors))
    area_phasor = np.sum(area_weights * np.asarray(phasors))
    return {
        "weights": weights,
        "area_weights": area_weights,
        "area_fractions": np.asarray(
            [float(np.mean(_domain_mask(n, d))) for d in domains]
        ),
        "weight_fractions": weights / total,
        "area_weight_fractions": area_weights / total_area,
        "phases_deg": np.asarray(phases),
        "phasor_sum": phasor,
        "phi_bar_deg": float(np.degrees(np.angle(phasor)) % 360.0),
        "coherence": float(abs(phasor) / total),
        "three_phi_bar_deg": float(3.0 * np.degrees(np.angle(phasor)) % 360.0),
        "phi_bar_area_deg": float(np.degrees(np.angle(area_phasor)) % 360.0),
        "coherence_area": float(abs(area_phasor) / total_area),
        "three_phi_bar_area_deg": float(
            3.0 * np.degrees(np.angle(area_phasor)) % 360.0
        ),
    }


# --------------------------------------------------------------------------- #
# geometry stage (same resampling rule as the package correction)
# --------------------------------------------------------------------------- #
def affine_resample(image, stretch, pad=10, order=3):
    """Resample so that the observed stretch is undone (package rule).

    ``stretch`` is the symmetric positive-definite 2x2 ``M`` with
    ``q_obs = q_ideal @ M``.  The output satisfies ``output = input[A y + b]`` with
    ``A = P M^-1 P`` (``P`` the axis swap), the canvas grown to contain every
    transformed corner plus ``pad`` and the centres aligned -- identical to the
    geometry of ``stm_data_processing.utils.bragg_peak.correct``, but with the
    stretch supplied from the ground truth instead of estimated from a detection.
    """
    from scipy.ndimage import affine_transform

    arr = np.asarray(image, dtype=float)
    n = arr.shape[0]
    swap = np.array([[0.0, 1.0], [1.0, 0.0]])
    matrix = swap @ np.linalg.inv(np.asarray(stretch, dtype=float)) @ swap
    corners = (
        np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]]) * (n - 1) / 2.0
    )
    extent = np.abs(np.linalg.inv(matrix) @ corners.T).max(axis=1)
    n_out = 2 * (int(np.ceil(float(extent.max()))) + int(pad)) + 1
    offset = (n - 1) / 2.0 - matrix @ np.array([(n_out - 1) / 2.0, (n_out - 1) / 2.0])
    out = affine_transform(
        arr,
        matrix,
        offset=offset,
        output_shape=(n_out, n_out),
        order=int(order),
        mode="constant",
        cval=np.nan,
        prefilter=int(order) > 1,
    )
    return out, matrix, n_out, offset


def stretched_image(image, stretch, pad=10, order=3):
    """Apply the stretch ``M`` to the pattern: the result peaks at ``q_in @ M``.

    ``affine_resample(img, S)`` undoes the stretch ``S``, i.e. it moves a peak of
    ``img`` from ``q`` to ``q @ S^-1``; applying ``M`` therefore means resampling
    with ``S = M^-1``.  Used to build synthetic images with a known anisotropy.
    """
    return affine_resample(
        image, np.linalg.inv(np.asarray(stretch, dtype=float)), pad=pad, order=order
    )


# --------------------------------------------------------------------------- #
# ring identification: the reference ring is an explicit, reportable choice
# --------------------------------------------------------------------------- #
def find_ring_at_ratio(rings, reference_radius_px, ratio=1.0 / SQRT3, tol_frac=0.03):
    """Ring whose radius is ``ratio`` times the reference radius, or ``None``.

    The lookup is a plain nearest-radius search inside a relative tolerance; it
    never guesses: when nothing is inside the tolerance the caller must report
    "not found".
    """
    target = float(reference_radius_px) * float(ratio)
    best, best_dev = None, np.inf
    for ring in rings:
        deviation = abs(ring["radius"] - target) / target
        if deviation <= tol_frac and deviation < best_dev:
            best, best_dev = ring, deviation
    return best


def choose_rings(
    rings,
    anchor="auto",
    reference_radius_px=None,
    match_tol=0.03,
    pair_tol=0.03,
    r3_tol=0.03,
    expect=6,
    ratio=SQRT3,
):
    """Pick ``ring_1x1`` (the reference ring) and ``ring_r3`` = ring_1x1 / sqrt(3).

    The reference ring is a *choice*, never an implicit assumption: the caller
    states it with ``anchor`` and the function reports which path it took.

    ``anchor`` values
        ``radius``    ``reference_radius_px`` names the reference radius (nearest
                      ring inside ``match_tol``).
        ``strongest`` the ring with the largest total reflection amplitude.
        ``outer`` / ``inner``  that member of the ring pair whose radius ratio is
                      closest to ``sqrt(3)`` (inside ``pair_tol``).
        ``auto``      the strongest ring that has a ring at ``1/sqrt(3)`` of its
                      radius; failing that the outer member of the ratio pair;
                      failing that a ring whose ``sqrt(3)`` multiple exists (the
                      "the innermost ring is the r3 ring" case is then reported
                      explicitly).

    Returns a dict with ``status`` (``ok`` / ``r3_not_found`` / ``no_rings``),
    ``method`` (text), ``ring_1x1``, ``ring_r3``, ``ratio``, ``sqrt3_deviation``.
    """
    usable = [ring for ring in rings if len(ring["members"]) >= int(expect)]
    result = {
        "status": "no_rings",
        "method": (f"no ring with at least {int(expect)} members"),
        "ring_1x1": None,
        "ring_r3": None,
        "ratio": float("nan"),
        "sqrt3_deviation": float("nan"),
    }
    if not usable:
        return result
    by_amplitude = sorted(usable, key=lambda ring: -ring["total_amplitude"])
    reference, method = None, ""

    if anchor == "radius":
        if reference_radius_px is None:
            raise ValueError("--reference-radius-px is required with --anchor radius")
        nearest = min(
            usable, key=lambda ring: abs(ring["radius"] - reference_radius_px)
        )
        deviation = abs(nearest["radius"] - reference_radius_px) / reference_radius_px
        if deviation > match_tol:
            result["method"] = (
                f"no ring within {match_tol:.1%} of the requested "
                f"reference radius {reference_radius_px:.4f} px "
                f"(nearest {nearest['radius']:.4f} px, "
                f"{deviation:.2%} away)"
            )
            return result
        reference = nearest
        method = f"explicit reference radius {reference_radius_px:.4f} px"
    elif anchor == "strongest":
        reference = by_amplitude[0]
        method = "strongest ring by total reflection amplitude"
    elif anchor in ("outer", "inner"):
        pair = select_ring_pair(
            usable, ratio=ratio, tol_frac=pair_tol, expect=int(expect)
        )
        if pair is None:
            result["method"] = (
                f"no ring pair with a radius ratio inside {pair_tol:.1%} of sqrt(3)"
            )
            return result
        reference = pair[0] if anchor == "outer" else pair[1]
        method = (
            f"{anchor} member of the ratio-{pair[2]:.4f} ring pair "
            f"({pair[0]['radius']:.4f} px / {pair[1]['radius']:.4f} px)"
        )
    elif anchor == "auto":
        for ring in by_amplitude:
            if (
                find_ring_at_ratio(
                    usable, ring["radius"], ratio=1.0 / SQRT3, tol_frac=r3_tol
                )
                is not None
            ):
                reference = ring
                method = (
                    "strongest ring that has a ring at 1/sqrt(3) of its "
                    "radius inside tolerance"
                )
                break
        if reference is None:
            pair = select_ring_pair(
                usable, ratio=ratio, tol_frac=pair_tol, expect=int(expect)
            )
            if pair is not None:
                reference = pair[0]
                method = (
                    f"outer member of the ratio-{pair[2]:.4f} ring pair "
                    f"(no direct 1/sqrt(3) match for the strongest rings)"
                )
        if reference is None:
            for ring in by_amplitude:
                above = find_ring_at_ratio(
                    usable, ring["radius"], ratio=SQRT3, tol_frac=r3_tol
                )
                if above is not None:
                    reference = above
                    method = (
                        "the innermost ring sits at 1/sqrt(3) of the "
                        "reference ring: the reference is the ring above it"
                    )
                    break
    else:
        raise ValueError(f"unsupported anchor {anchor!r}")

    if reference is None:
        result["status"] = "r3_not_found"
        if not result["method"] or result["method"].startswith("no ring with at least"):
            result["method"] = "no reference ring could be established: " + (
                method
                or "no ring pair with a sqrt(3) radius ratio "
                "and no ring at 1/sqrt(3) or sqrt(3) of "
                "another ring inside tolerance"
            )
        return result

    result["ring_1x1"] = reference
    partner = find_ring_at_ratio(
        usable, reference["radius"], ratio=1.0 / SQRT3, tol_frac=r3_tol
    )
    if partner is None:
        result["status"] = "r3_not_found"
        result["method"] = (
            method + f"; reference radius {reference['radius']:.4f} px "
            f"but no ring at 1/sqrt(3) of it inside {r3_tol:.1%} "
            f"(target {reference['radius'] / SQRT3:.4f} px)"
        )
        return result
    result["status"] = "ok"
    result["method"] = method
    result["ring_r3"] = partner
    result["ratio"] = float(reference["radius"] / partner["radius"])
    result["sqrt3_deviation"] = float(abs(result["ratio"] - SQRT3) / SQRT3)
    return result


def clock_order_index(qx, qy):
    """Key that numbers a ring's reflections from 12 o'clock clockwise.

    ``p0`` is the reflection closest to the ``+qy`` direction and the following
    ones run clockwise in the ``(qx, qy)`` plane.  ``qy`` grows towards the top of
    the plotted image (``origin='lower'``), so this is the usual "12 o'clock
    first, clockwise" convention, stated once and used for both rings.
    """
    angle = float(np.arctan2(qy, qx))
    return float((np.pi / 2.0 - angle) % TWO_PI)


def order_ring_members(members, expect=6):
    """The six strongest members of a ring, numbered clockwise from 12 o'clock."""
    strongest = sorted(members, key=lambda item: -item[2])[: int(expect)]
    return sorted(strongest, key=lambda item: clock_order_index(item[0], item[1]))
