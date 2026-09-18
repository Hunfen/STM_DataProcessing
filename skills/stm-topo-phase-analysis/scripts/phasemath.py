"""Weighted circular phase statistics and lattice-referenced gauge fixing.

Pure mathematics: angles, weights and the phase convention of the STM
topographic phase pipeline.  Nothing in this module interprets a number
physically; every routine is a plain estimator on a circular sample.

Conventions
-----------
A *phase sample* is a pair ``(phi_i, w_i)`` with ``phi_i`` in ``[0, 2*pi)`` and
``w_i >= 0``.  The pipeline builds it from the single-peak complex iFFT ``psi``
of a reflection at FFT wavevector ``q`` (in FFT pixels, i.e. the same units as
``numpy.fft.fftfreq * n``) as

    phi(r) = angle(psi(r)) - (2*pi/N) * q . (r - c),      c = N // 2

i.e. the carrier is removed with the canvas centre as the reference point.
Moving the reference point by ``d`` adds the *constant* ``-(2*pi/N) q.d`` to
every ``phi`` of that reflection; that is the only "gauge" freedom of a single
reflection.  Every estimator below is exactly equivariant under
``phi -> phi + const`` (medians, cluster centres and FWHM shift with the
constant; R, weights, cluster counts and widths do not), so a gauge change
translates the reported angles and leaves the shape untouched.

Estimators
----------
``circ_mean``      amplitude-weighted circular mean and resultant length R.
``circ_median``    exact minimiser of the weighted circular L1 objective
                   ``g(theta) = sum_i w_i |wrap_pm_pi(theta - phi_i)|``; the
                   minimiser set is reported so that a broad or multimodal
                   sample is flagged instead of silently returning a number.
``circ_quantiles`` weighted quantiles of the sample unrolled around the median.
``circ_spread``    circular standard deviation and the von Mises kappa estimate.
``phase_histogram``/``smooth_circular``  amplitude-weighted density and its
                   wrap-around Gaussian smoothing.
``fwhm_deg``       full width at half maximum of the smoothed histogram.
``cluster_list``   cluster centres / weights of the smoothed histogram.
``linear_median_fwhm``  the same two statistics for a non-circular sample
                   (used for the amplitude distribution).
``weighted_stats`` one call returning the whole scalar set.

Gauge fixing
------------
``fit_origin``     weighted least squares of ``phi_j = (2*pi/n) q_j . r0 + c``
                   modulo ``2*pi`` over the reference ring's reflections, with a
                   multi-start Gauss-Newton iteration; all local minima found by
                   the multi-start are returned, because two of them can differ
                   by a real-space lattice vector of the reference ring and thus
                   define two different (but equally legal) gauges.
``gauge_phase``    ``phi~ = phi - (2*pi/n) q . r0 - c``, the phase referred to
                   the fitted origin.  Applied to the reference ring it lands on
                   0 (that is the definition of the fit); applied to any other
                   reflection it makes that reflection comparable across the
                   field of view.
"""

from __future__ import annotations

import numpy as np

TWO_PI = 2.0 * np.pi
DEG = 180.0 / np.pi


# --------------------------------------------------------------------------- #
# elementary angle helpers
# --------------------------------------------------------------------------- #
def wrap_pm_pi(a):
    """Wrap angles to (-pi, pi]."""
    return (np.asarray(a, dtype=float) + np.pi) % TWO_PI - np.pi


def wrap_2pi(a):
    """Wrap angles to [0, 2*pi)."""
    return np.mod(np.asarray(a, dtype=float), TWO_PI)


def to_deg(a):
    return float(np.degrees(np.mod(a, TWO_PI)))


def ang_diff_deg(a, b):
    """Signed circular difference a - b in degrees, in (-180, 180]."""
    return float(np.degrees(wrap_pm_pi(np.radians(a - b))))


def dist_to_ladder_deg(angle_deg, ladder_deg):
    """Circular distance of an angle to the nearest value of a ladder."""
    ladder = np.asarray(ladder_deg, dtype=float)
    diff = (angle_deg - ladder + 180.0) % 360.0 - 180.0
    return float(np.min(np.abs(diff)))


# --------------------------------------------------------------------------- #
# circular statistics
# --------------------------------------------------------------------------- #
def _prep(phi, w):
    phi = np.mod(np.asarray(phi, dtype=float), TWO_PI)
    if w is None:
        w = np.ones_like(phi)
    w = np.asarray(w, dtype=float)
    keep = (w > 0.0) & np.isfinite(w) & np.isfinite(phi)
    return phi[keep], w[keep]


def circ_mean(phi, w=None):
    """Amplitude-weighted circular mean (rad in [0, 2pi)) and resultant R."""
    phi, w = _prep(phi, w)
    total = float(np.sum(w))
    if phi.size == 0 or total <= 0.0:
        return float("nan"), 0.0, 0.0
    z = np.sum(w * np.exp(1j * phi))
    return float(np.angle(z) % TWO_PI), float(np.abs(z) / total), total


def circ_median(phi, w=None):
    """Exact weighted circular median.

    Returns ``(median_rad, span_deg, n_minimisers)``.  The objective

        g(theta) = sum_i w_i * |wrap_pm_pi(theta - phi_i)|

    is piecewise linear with breakpoints at the samples, and its slope is
    ``s(theta) = 2*A(theta) - W`` with ``A(theta)`` the weight inside the arc
    ``(theta - pi, theta]``.  ``A`` jumps at two kinks per sample, ``phi_i``
    (entry, ``+w_i``) and ``phi_i + pi`` (exit, ``-w_i``), so ``s`` is a circular
    step function: the minimisers are the kinks where ``s`` crosses zero from
    below.  Several crossings can exist (one per mode) and the minima need not be
    degenerate, so ``g`` is evaluated exactly at every crossing and the global
    one is kept.  ``span_deg`` is the width of the minimiser set (0 for a unique
    median; a broad or uniform sample gives a wide set and is flagged instead of
    silently returning a number).
    """
    phi, w = _prep(phi, w)
    if phi.size == 0:
        return float("nan"), float("nan"), 0
    order = np.argsort(phi, kind="stable")
    p, ww = phi[order], w[order]
    total = float(np.sum(ww))
    if p.size == 1:
        return float(p[0]), 0.0, 1
    tol = 1e-9 * total
    base = float(p[0])
    rel = np.concatenate([np.mod(p - base, TWO_PI), np.mod(p + np.pi - base, TWO_PI)])
    delta = np.concatenate([ww, -ww])
    rel = np.where(rel <= 0.0, TWO_PI, rel)          # entries at the base: already in
    order = np.argsort(rel, kind="stable")
    rel, delta = rel[order], delta[order]
    # merge coincident kinks: an entry and an exit at the same angle cancel, and a
    # transient (zero-width) kink must not be mistaken for a slope crossing
    starts = np.concatenate([[0], np.flatnonzero(np.diff(rel) > 1e-12) + 1])
    rel = rel[starts]
    delta = np.add.reduceat(delta, starts)
    offset = np.mod(base - p, TWO_PI)                # in [0, 2pi)
    a0 = float(np.sum(ww[offset < np.pi]))           # A(base^+) = weight in (base-pi, base]
    a_after = a0 + np.cumsum(delta)
    s_after = 2.0 * a_after - total
    s_before = np.concatenate([[2.0 * a0 - total], s_after[:-1]])
    cross = np.flatnonzero((s_before < -tol) & (s_after >= -tol))
    if cross.size == 0:
        # no kink turns the slope from negative to zero: the minimiser set is the
        # (possibly wrapping) run of kinks where the slope is already zero
        flat = (s_before <= tol) & (s_after >= -tol)
        if not flat.any():
            idx = int(np.argmin(np.abs(s_after)))
            return float(np.mod(base + rel[idx], TWO_PI)), 0.0, 1
        if flat.all():
            start = base + rel[0]
            span = TWO_PI
        else:
            offset = int(np.flatnonzero(~flat)[0])
            rolled = np.roll(flat, -offset)
            first = int(np.flatnonzero(rolled)[0])
            last = first
            while last + 1 < rolled.size and rolled[last + 1]:
                last += 1
            head, tail = offset + first, offset + last
            start = base + rel[head % rel.size]
            span = float(rel[tail % rel.size] - rel[head % rel.size]
                         + (TWO_PI if tail >= rel.size else 0.0))
        inside = int(np.count_nonzero(np.mod(p - start, TWO_PI) <= span + 1e-12))
        return float(np.mod(start + span / 2.0, TWO_PI)), float(
            np.degrees(span)), inside
    candidates = base + rel[cross]
    values = np.empty(cross.size)
    for start in range(0, cross.size, 16):
        chunk = candidates[start:start + 16]
        values[start:start + 16] = np.sum(
            ww[None, :] * np.abs(wrap_pm_pi(chunk[:, None] - p[None, :])), axis=1)
    best = int(np.argmin(values))
    k0 = int(cross[best])
    if s_after[k0] > tol:
        return float(np.mod(base + rel[k0], TWO_PI)), 0.0, 1
    nxt = np.flatnonzero(s_after[k0 + 1:] > tol)
    if nxt.size == 0:
        return float(np.mod(base + rel[k0], TWO_PI)), 0.0, 1
    k1 = k0 + 1 + int(nxt[0])
    if np.any(s_after[k0 + 1:k1] < -tol):
        return float(np.mod(base + rel[k0], TWO_PI)), 0.0, 1
    span = float(rel[k1] - rel[k0])
    start = base + rel[k0]
    inside = int(np.count_nonzero(np.mod(p - start, TWO_PI) <= span + 1e-12))
    return float(np.mod(start + span / 2.0, TWO_PI)), float(np.degrees(span)), inside


def circ_quantiles(phi, w=None, probs=(0.25, 0.5, 0.75), centre=None):
    """Weighted quantiles of the sample unrolled around ``centre`` (default: median).

    The sample is rotated so that ``centre`` is at angle 0 and its angles are then
    treated as a linear sample in ``(-pi, pi]``.  This is meaningful for samples
    that do not wrap the whole circle; the returned width is flagged by
    ``circ_spread`` otherwise.
    """
    phi, w = _prep(phi, w)
    if phi.size == 0:
        return [float("nan")] * len(probs)
    if centre is None:
        centre, _, _ = circ_median(phi, w)
    rel = wrap_pm_pi(phi - centre)
    order = np.argsort(rel, kind="stable")
    r, ww = rel[order], w[order]
    cum = np.cumsum(ww) / float(np.sum(ww))
    out = []
    for prob in probs:
        k = int(np.searchsorted(cum, prob, side="left"))
        k = min(k, r.size - 1)
        lo = cum[k - 1] if k > 0 else 0.0
        hi = cum[k]
        frac = 0.0 if hi <= lo else (prob - lo) / (hi - lo)
        value = r[k - 1] + frac * (r[k] - r[k - 1]) if k > 0 else r[0]
        out.append(float(np.mod(value + centre, TWO_PI)))
    return out


def circ_spread(phi, w=None):
    """Circular standard deviation (rad), von Mises kappa and resultant R."""
    _, r_length, _ = circ_mean(phi, w)
    if not np.isfinite(r_length) or r_length <= 0.0:
        return float("nan"), 0.0, float(r_length)
    if r_length >= 1.0:
        # perfectly concentrated sample (e.g. a single-bin mask): sigma = 0 while
        # kappa diverges, which the closed forms below cannot express
        return 0.0, float("inf"), 1.0
    if r_length < 0.53:
        kappa = 2.0 * r_length + r_length ** 3 + 5.0 * r_length ** 5 / 6.0
    else:
        kappa = -0.4 + 1.39 * r_length + 0.43 / (1.0 - r_length)
    std = float(np.sqrt(-2.0 * np.log(r_length)))
    return std, float(kappa), float(r_length)


# --------------------------------------------------------------------------- #
# histogram based estimators
# --------------------------------------------------------------------------- #
def phase_histogram(phi, w=None, bins=3600):
    """Amplitude-weighted count histogram over [0, 2*pi) with ``bins`` bins."""
    phi, w = _prep(phi, w)
    h, _ = np.histogram(phi, bins=bins, range=(0.0, TWO_PI), weights=w)
    return h.astype(float)


def smooth_circular(h, sigma_bins):
    """Wrap-around Gaussian smoothing of a circular histogram."""
    h = np.asarray(h, dtype=float)
    if sigma_bins <= 0.0:
        return h.copy()
    n = h.size
    half = int(np.ceil(4.0 * sigma_bins))
    half = max(1, min(half, n - 1))
    offsets = np.arange(-half, half + 1)
    kernel = np.exp(-0.5 * (offsets / sigma_bins) ** 2)
    kernel /= kernel.sum()
    padded = np.concatenate([h[-half:], h, h[:half]])
    conv = np.convolve(padded, kernel, mode="same")
    return conv[half:half + n]


def _parabolic_peak(y_prev, y_peak, y_next):
    denom = y_prev - 2.0 * y_peak + y_next
    if denom >= 0.0:
        return 0.0
    return float(np.clip(0.5 * (y_prev - y_next) / denom, -0.5, 0.5))


def fwhm_deg(phi, w=None, bins=3600, sigma_deg=2.0):
    """Full width at half maximum (deg) of the smoothed circular histogram.

    Returns ``(fwhm_deg, peak_deg, peak_height, half_level, ok)``.  ``ok`` is
    False when the smoothed histogram never drops to half of its maximum inside
    the circle (a too broad or multi-peaked sample), in which case ``fwhm_deg``
    is NaN and only ``peak_deg``/``peak_height`` are meaningful.
    """
    h = phase_histogram(phi, w, bins)
    if h.sum() <= 0.0:
        return float("nan"), float("nan"), 0.0, 0.0, False
    hs = smooth_circular(h, sigma_deg * bins / 360.0)
    top = float(hs.max())
    if top <= 0.0:
        return float("nan"), float("nan"), 0.0, 0.0, False
    peak = int(np.argmax(hs))
    half = top / 2.0
    step = 360.0 / bins
    peak_deg = float((peak + 0.5) * step) % 360.0

    def crossing(direction):
        k = 1
        while k < bins:
            prev = hs[(peak + direction * (k - 1)) % bins]
            here = hs[(peak + direction * k) % bins]
            if here <= half <= prev and prev > here:
                frac = (prev - half) / (prev - here)
                return (k - 1) + frac
            if here > half:
                k += 1
                continue
            return float(k)
        return None

    right = crossing(+1)
    left = crossing(-1)
    if right is None or left is None:
        return float("nan"), peak_deg, top, half, False
    return float((right + left) * step), peak_deg, top, half, True


def deconvolve_fwhm_deg(fwhm, sigma_deg):
    """Remove the smoothing kernel from an FWHM (Gaussian-equivalent width).

    The reported FWHM is the width of the *smoothed* histogram, i.e. the sample
    width folded with the smoothing kernel: ``FWHM_obs^2 = FWHM_true^2 +
    (2.3548 sigma_smooth)^2`` for Gaussian shapes.  The deconvolved value is a
    Gaussian-equivalent estimate, not an assumption-free one; it is undefined
    (NaN) when the observed width is not larger than the kernel width.
    """
    kernel = 2.354820045 * float(sigma_deg)
    if not np.isfinite(fwhm) or fwhm <= kernel:
        return float("nan")
    return float(np.sqrt(fwhm ** 2 - kernel ** 2))


def cluster_list(phi, w=None, bins=3600, sigma_deg=3.0, min_frac=0.25,
                 merge_deg=30.0, window_deg=15.0, refine_iters=10):
    """Clusters of the smoothed circular histogram.

    Rule: ``bins`` on [0, 2pi), wrap-around Gaussian smoothing of width
    ``sigma_deg``, strict local maxima of the smoothed histogram that reach
    ``min_frac`` of its maximum, parabolic sub-bin refinement, greedy merging of
    centres closer than ``merge_deg`` (the higher one survives), then
    ``refine_iters`` rounds of amplitude-weighted circular-mean refinement inside
    a ``+-window_deg`` window.  Returns a list of dicts with ``centre_deg``,
    ``weight_fraction`` (nearest-centre assignment of the samples) and
    ``height`` (smoothed histogram value at the centre).
    """
    phi, w = _prep(phi, w)
    if phi.size == 0:
        return []
    h = phase_histogram(phi, w, bins)
    if h.sum() <= 0.0:
        return []
    hs = smooth_circular(h, sigma_deg * bins / 360.0)
    step = 360.0 / bins
    top = float(hs.max())
    coarse = []
    for i in range(bins):
        if hs[i] <= 0.0:
            continue
        if not (hs[i] >= hs[i - 1] and hs[i] > hs[(i + 1) % bins]):
            continue
        if hs[i] < min_frac * top:
            continue
        delta = _parabolic_peak(hs[i - 1], hs[i], hs[(i + 1) % bins])
        coarse.append((((i + delta) * step) % 360.0, float(hs[i])))
    coarse.sort(key=lambda item: -item[1])
    kept = []
    for centre, height in coarse:
        if all(min(abs(centre - c), 360.0 - abs(centre - c)) >= merge_deg
               for c, _ in kept):
            kept.append((centre, height))
    if not kept:
        return []
    centres = [np.radians(c) for c, _ in kept]
    window = np.radians(window_deg)
    for _ in range(refine_iters):
        refined = []
        for centre in centres:
            dist = np.abs(wrap_pm_pi(phi - centre))
            sel = dist <= window
            if not sel.any():
                refined.append(centre)
                continue
            z = np.sum(w[sel] * np.exp(1j * phi[sel]))
            refined.append(float(np.angle(z)) % TWO_PI if abs(z) > 0 else centre)
        centres = refined
    diff = np.degrees(phi)[:, None] - np.degrees(np.asarray(centres))[None, :]
    if centres:
        labels = np.argmin(np.abs(wrap_pm_pi(np.radians(diff))), axis=1)
        weight = np.bincount(labels, weights=w, minlength=len(centres))
        weight = weight / weight.sum() if weight.sum() > 0 else weight
    else:
        weight = np.array([])
    out = []
    for i, (centre, height) in enumerate(kept):
        # the refined centre stays inside the +-window of the seed
        out.append({
            "centre_deg": float(np.degrees(centres[i]) % 360.0),
            "height": float(height),
            "weight_fraction": float(weight[i]) if i < len(weight) else float("nan"),
        })
    out.sort(key=lambda item: item["centre_deg"])
    return out


def linear_median_fwhm(x, w=None, bins=512, smooth_bins=2.0):
    """Median and FWHM (same units as ``x``) of a non-circular weighted sample."""
    x = np.asarray(x, dtype=float)
    if w is None:
        w = np.ones_like(x)
    w = np.asarray(w, dtype=float)
    keep = np.isfinite(x) & np.isfinite(w) & (w > 0.0)
    x, w = x[keep], w[keep]
    if x.size == 0 or float(np.sum(w)) <= 0.0:
        return float("nan"), float("nan"), float("nan")
    order = np.argsort(x, kind="stable")
    xs, ws = x[order], w[order]
    cum = np.cumsum(ws) / float(np.sum(ws))
    k = int(np.searchsorted(cum, 0.5, side="left"))
    median = float(xs[min(k, xs.size - 1)])
    lo, hi = float(xs[0]), float(xs[-1])
    if hi <= lo:
        return median, 0.0, hi
    edges = np.linspace(lo, hi, bins + 1)
    h, _ = np.histogram(xs, bins=edges, weights=ws)
    h = smooth_circular(h, smooth_bins)  # same wrapped kernel, harmless off-circle
    top = float(h.max())
    if top <= 0.0:
        return median, float("nan"), float("nan")
    peak = int(np.argmax(h))
    half = top / 2.0
    step = (hi - lo) / bins

    def crossing(direction):
        k = 1
        while 0 <= peak + direction * k < h.size:
            prev = h[peak + direction * (k - 1)]
            here = h[peak + direction * k]
            if here <= half <= prev and prev > here:
                return (k - 1) + (prev - half) / (prev - here)
            if here > half:
                k += 1
                continue
            return float(k)
        return None

    right = crossing(+1)
    left = crossing(-1)
    if right is None or left is None:
        return median, float("nan"), top
    return median, float((right + left) * step), top


def weighted_stats(phi, w=None, bins=3600, smooth_deg=2.0, cluster_kw=None):
    """All scalar phase statistics of one amplitude-weighted sample."""
    cluster_kw = dict(cluster_kw or {})
    mean, r_length, total = circ_mean(phi, w)
    median, median_span, n_min = circ_median(phi, w)
    std, kappa, _ = circ_spread(phi, w)
    width, peak_deg, peak_h, _, ok = fwhm_deg(phi, w, bins=bins, sigma_deg=smooth_deg)
    quantiles = circ_quantiles(phi, w, (0.25, 0.5, 0.75), centre=median)
    clusters = cluster_list(phi, w, bins=bins, sigma_deg=max(smooth_deg, 3.0),
                            **cluster_kw)
    return {
        "fwhm_deconv_deg": deconvolve_fwhm_deg(width, smooth_deg),
        "n_samples": int(np.size(np.asarray(phi).ravel())),
        "weight_sum": float(total),
        "mean_deg": to_deg(mean),
        "resultant_R": float(r_length),
        "median_deg": to_deg(median),
        "median_span_deg": float(median_span),
        "n_median_minimisers": int(n_min),
        "q25_deg": to_deg(quantiles[0]),
        "q75_deg": to_deg(quantiles[2]),
        "iqr_deg": abs(ang_diff_deg(to_deg(quantiles[2]), to_deg(quantiles[0]))),
        "circ_std_deg": float(np.degrees(std)) if np.isfinite(std) else float("nan"),
        "kappa": float(kappa),
        "fwhm_deg": float(width),
        "fwhm_peak_deg": float(peak_deg),
        "fwhm_ok": bool(ok),
        "peak_height": float(peak_h),
        "n_clusters": len(clusters),
        "clusters": clusters,
    }


# --------------------------------------------------------------------------- #
# lattice-referenced gauge fixing
# --------------------------------------------------------------------------- #
def _origin_residual(qs, phases, n, r0, c):
    model = (TWO_PI / n) * (qs @ np.asarray(r0, dtype=float)) + c
    return wrap_pm_pi(np.asarray(phases, dtype=float) - model)


def fit_origin(qs, phases, n, c_free=True, n_start=36, tol=1e-12, max_iter=80):
    """Least squares of ``phi_j = (2*pi/n) q_j . r0 + c`` (mod 2*pi).

    Gauss-Newton with wrapped residuals, restarted from ``n_start`` values of the
    common offset ``c`` on [0, 2pi); the wrapped objective has several local
    minima (two of them can differ by a direct lattice vector of the reference
    ring), so every distinct local minimum is returned.

    Returns ``(best, minima)`` where ``best`` is a dict with ``r0_px``,
    ``c_deg``, ``rms_deg`` and ``minima`` is the list of all distinct minima,
    sorted by RMS.
    """
    qs = np.asarray(qs, dtype=float)
    phases = np.asarray(phases, dtype=float)
    if qs.ndim != 2 or qs.shape[1] != 2 or qs.shape[0] < 2:
        raise ValueError("fit_origin needs at least two (qx, qy) rows")
    results = []
    for start in np.linspace(0.0, TWO_PI, int(n_start), endpoint=False):
        u = np.zeros(3)
        u[2] = start
        for _ in range(int(max_iter)):
            residual = _origin_residual(qs, phases, n, u[:2], u[2])
            jac = np.c_[-((TWO_PI / n) * qs[:, 0]), -((TWO_PI / n) * qs[:, 1]),
                        -np.ones(qs.shape[0]) if c_free else np.zeros(qs.shape[0])]
            delta, *_ = np.linalg.lstsq(jac, -residual, rcond=None)
            u = u + delta
            if float(np.max(np.abs(delta))) < tol:
                break
        residual = _origin_residual(qs, phases, n, u[:2], u[2])
        rms = float(np.degrees(np.sqrt(np.mean(residual ** 2))))
        results.append((rms, float(u[0]), float(u[1]), float(u[2] % TWO_PI)))
    results.sort(key=lambda item: item[0])
    minima, seen = [], []
    for rms, x0, y0, c in results:
        if any(np.hypot(x0 - sx, y0 - sy) < 1e-6 for sx, sy in seen):
            continue
        seen.append((x0, y0))
        minima.append({"rms_deg": rms, "r0_px": (x0, y0), "c_deg": to_deg(c),
                       "c_rad": c})
    best = dict(minima[0])
    best["minima"] = minima
    return best, minima


def gauge_phase(phi, q, r0, c_rad, n):
    """``phi~ = phi - (2*pi/n) q . r0 - c``; the reference ring lands on 0."""
    phi = np.asarray(phi, dtype=float)
    shift = (TWO_PI / n) * (q[0] * r0[0] + q[1] * r0[1]) + c_rad
    return np.mod(phi - shift, TWO_PI)


def branch_relabel_deg(q, r0_a, r0_b, n):
    """Phase difference induced on a reflection by replacing ``r0_a`` with ``r0_b``."""
    delta = (TWO_PI / n) * (q[0] * (r0_b[0] - r0_a[0]) + q[1] * (r0_b[1] - r0_a[1]))
    return to_deg(delta)
