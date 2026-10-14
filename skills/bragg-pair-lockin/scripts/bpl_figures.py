"""Figures of the bragg-pair-lockin skill.

Three kinds of figure, all with English/mathtext captions:

* the two bare previews of a point (through the sibling convention
  ``localqmap.save_map``: no axes, no frame, NaN drawn in the bad colour),
* the bare preview of one pair's ``B(r)`` map,
* the titled figures: the pooled ``B(r)`` summary and the four-step set of one
  selected point (FFT with the marked pair / q-space zoom with the window /
  the modulated field before the low-pass / the low-passed local field).

Every caption prints both spellings of the point and the route of its
reconstruction, so the figure alone says which q and which reconstruction it
shows.
"""

from __future__ import annotations

from pathlib import Path

import bpl_lib as bpl
import numpy as np

DPI = 150
BAD_COLOUR = "0.55"
FFT_ZOOM = 8.0
#: Dynamic range shown by the q-space zoom, in decades of |F|.
FFT_ZOOM_DECADES = 2.0
STEP_COLOURS = ("#00ff7f", "#00e5ff", "#ff4fd8", "#ffd34d", "#b08cff", "#ff8f6b")


def setup_style():
    """Plot style of the skill (no usetex: broken with mpl 3.10 + TeX Live 2026)."""
    import matplotlib

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


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _with_bad(cmap, colour=BAD_COLOUR):
    cmap = cmap.copy()
    cmap.set_bad(colour)
    return cmap


def _sci(value, digits=2):
    """Matplotlib-math scientific notation, e.g. ``$2.85\\times10^{-11}$``."""
    if not np.isfinite(value) or value == 0.0:
        return "$0$"
    exponent = int(np.floor(np.log10(abs(float(value)))))
    mantissa = float(value) / 10.0**exponent
    return rf"${mantissa:.{digits}f}\times10^{{{exponent}}}$"


def caption_for(point, entry):
    """Two spellings plus the route, for a figure caption."""
    write = point.writings
    return (
        rf"$q$ {write['text']}   |q| = {entry['q_nm_inv_abs']:.4f} nm$^{{-1}}$ "
        rf"= {entry['q_nm_inv_abs'] / (2.0 * np.pi):.4f} cycles/nm   "
        rf"route = {entry['route']}   ($\lambda$ = {entry['lambda_nm']:.4g} nm)"
    )


def read_back_caption(figure):
    """Every text actually carried by the rendered figure, joined.

    A caption that was never handed to an artist cannot be recovered here, so
    the callers compare this string with what they meant to draw: a title that
    was declared but not drawn is caught instead of being reported as present.
    """
    pieces = [text.get_text() for text in figure.texts]
    for axis in figure.axes:
        pieces.append(axis.get_title())
        pieces.extend(text.get_text() for text in axis.texts)
    return "\n".join(pieces)


def assert_caption_drawn(figure, caption, where):
    """Fail loudly when the caption is not part of the rendered figure."""
    drawn = read_back_caption(figure)
    if caption not in drawn:
        raise RuntimeError(
            f"{where}: the caption was not drawn on the figure (searched "
            f"{caption[:60]!r})"
        )
    return drawn


#: Significant digits of every coherence number printed in a caption.  The same
#: formatter is used by the self-test when it checks a caption against the report.
COHERENCE_CAPTION_DIGITS = 6


def caption_number(value):
    """The single formatting of a coherence number in a figure caption."""
    if value is None:
        return "n/a"
    value = float(value)
    if not np.isfinite(value):
        return f"{value:g}"
    return f"{value:.{COHERENCE_CAPTION_DIGITS}g}"


def coherence_caption(point, entry, record):
    """Both spellings, the route, xi, |C|(0.75 max_lag) and both strains.

    The numbers are the ones stored in the report (``record``); the self-test
    reformats them with :func:`caption_number` and requires every one of them to
    appear in this caption.
    """
    return (
        rf"$q$ {point.writings['text']}   "
        rf"[{point.writings['cli_hk']} | {point.writings['cli_n_times_basis']}]   "
        rf"route = {entry['route']}   "
        rf"$\xi$ = {caption_number(record['xi_nm'])} nm   "
        rf"$|C|(0.75\,\Delta r)$ = {caption_number(record['anchors']['0.75'])}   "
        rf"$\epsilon_x$ = {caption_number(record['ramp']['x']['strain_equivalent'])}   "
        rf"$\epsilon_y$ = {caption_number(record['ramp']['y']['strain_equivalent'])}"
    )


def bare_map(path, data, cmap="inferno", vmin=None, vmax=None):
    """One bare 1:1 data map (sibling convention: no axes, no title, NaN bad)."""
    lq = bpl.lq
    lq.save_map(path, np.asarray(data, dtype=float), cmap=cmap, vmin=vmin, vmax=vmax)
    return Path(path)


def point_previews(outdir, stem, label, valid, amplitude, theta_deg, mask):
    """The three bare previews of one point, through the sibling writer."""
    lq = bpl.lq
    paths = {
        "amplitude_png": Path(outdir) / f"{stem}_{label}_amplitude.png",
        "theta_png": Path(outdir) / f"{stem}_{label}_theta.png",
        "mask_png": Path(outdir) / f"{stem}_{label}_mask.png",
    }
    lq.save_map(
        paths["amplitude_png"], np.where(valid, amplitude, np.nan), cmap="inferno"
    )
    lq.save_map(
        paths["theta_png"],
        np.where(valid, theta_deg, np.nan),
        cmap="twilight",
        vmin=-180.0,
        vmax=180.0,
    )
    lq.save_map(
        paths["mask_png"],
        np.asarray(mask, dtype=float),
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
    )
    return paths


def step5_phase_coherence(outdir, point, entry, coherence, nm_per_px):
    """The two-panel phase-coherence figure of one point (into ``steps_<label>``).

    Top: the real radial ``|C(Delta r)|`` curve with the two reference lines
    ``|C| = 0`` and ``|C| = 1`` (nothing else is drawn: no shuffled, no
    randomised, no constant-phase control).  Bottom: the 2D ``|C(dx, dy)|`` of the
    same point over ``+-max_lag_nm``, viridis 0..1, the centre marked with a cross
    and the colorbar outside the data area.  Returns the written path.
    """
    plt = _pyplot()
    directory = Path(outdir) / f"steps_{point.label}"
    directory.mkdir(parents=True, exist_ok=True)
    record = coherence.record
    max_lag_nm = float(coherence.max_lag_nm)
    caption = coherence_caption(point, entry, record)
    magnitude = np.abs(coherence.profile)

    figure, axes = plt.subplots(
        2, 1, figsize=(7.6, 10.4), dpi=DPI, gridspec_kw={"height_ratios": [1.0, 1.15]}
    )
    axes[0].plot(coherence.lag_nm, magnitude, "-", color="#1f77b4", lw=2.0)
    axes[0].axhline(1.0, color="0.55", ls="--", lw=0.9)
    axes[0].axhline(0.0, color="0.55", ls=":", lw=0.9)
    axes[0].set_xlim(0.0, max_lag_nm)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_xlabel(r"$|\Delta r|$   (nm)", fontsize=11)
    axes[0].set_ylabel(r"$|C(\Delta r)|$", fontsize=11)
    axes[0].grid(alpha=0.25)
    axes[0].set_title(
        caption + "\n" + r"dashed: $|C|=1$, dotted: $|C|=0$", fontsize=10.5
    )

    centre = int(coherence.c.shape[0]) // 2
    half = round(max_lag_nm / float(nm_per_px))
    available = int(min(centre, coherence.c.shape[0] - centre - 1))
    take = int(min(half, available))
    window = np.full((2 * half + 1, 2 * half + 1), np.nan)
    window[half - take : half + take + 1, half - take : half + take + 1] = np.abs(
        coherence.c[
            centre - take : centre + take + 1, centre - take : centre + take + 1
        ]
    )
    extent = (-max_lag_nm, max_lag_nm, -max_lag_nm, max_lag_nm)
    image = axes[1].imshow(
        window, origin="lower", cmap="viridis", vmin=0.0, vmax=1.0, extent=extent
    )
    axes[1].plot([0.0], [0.0], "+", color="w", ms=10, mew=1.5)
    axes[1].set_xlabel("dx   (nm)", fontsize=11)
    axes[1].set_ylabel("dy   (nm)", fontsize=11)
    axes[1].set_title(
        f"{point.label}: "
        r"$|C(dx, dy)|$, centre = 0 lag, window $\pm$"
        f"{max_lag_nm:.4g} nm (= max_lag)\nthe correlation canvas of a "
        rf"{coherence.c.shape[0]} px field only provides $\pm$"
        f"{available * float(nm_per_px):.4g} nm (= L/2), so the four corners are "
        r"undefined and stay blank",
        fontsize=10.0,
    )
    figure.colorbar(image, ax=axes[1], fraction=0.046, pad=0.22, label=r"$|C|$")
    figure.suptitle(
        "5  phase-difference coherence of this point's own field   "
        r"$C(\Delta r) = \langle e^{i[\theta(r)-\theta(r+\Delta r)]}\rangle$"
        f"\nmask eroded by 0.65 lambda = {coherence.border_px:.4g} px, "
        f"{record['n_valid_px']} valid px, radial profile to {max_lag_nm:.4g} nm",
        fontsize=11,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    path = directory / bpl.STEP_FIGURES[4]
    assert_caption_drawn(figure, caption, path.name)
    figure.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def b_summary(outdir, entries, size_nm):
    """One panel per pair with a shared absolute colour scale."""
    plt = _pyplot()
    finite = (
        np.concatenate([e["B"][np.isfinite(e["B"])] for e in entries])
        if entries
        else np.zeros(1)
    )
    vmax = float(np.percentile(np.abs(finite), 99.5)) if finite.size else 1.0
    vmin = -vmax
    columns = len(entries)
    figure = plt.figure(figsize=(6.4 * columns, 6.6), dpi=DPI)
    grid = figure.add_gridspec(1, columns, wspace=0.06)
    cmap = _with_bad(plt.get_cmap("viridis"))
    image = None
    for slot, entry in enumerate(entries):
        axis = figure.add_subplot(grid[0, slot])
        image = axis.imshow(entry["B"], origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(
            f"B(r) for {entry['label']}\n"
            + entry["caption"]
            + f"\nrms = {entry['B_rms'] * 1e12:.1f} pm",
            fontsize=9,
        )
        axis.set_xticks([])
        axis.set_yticks([])
    bar = figure.add_axes([0.905, 0.15, 0.012, 0.7])
    figure.colorbar(image, cax=bar, label="B(r)  (same units as the input topography)")
    figure.suptitle(
        "Bragg-pair reconstruction B(r) = 2 Re[psi_q exp(i q.r)] "
        f"(two-sided where the criteria fail)  |  field of view {size_nm:.4f} nm  |  "
        f"shared absolute colour scale +/-{vmax:.3e}",
        fontsize=12,
    )
    path = Path(outdir) / "B_pairs_summary.png"
    for entry in entries:
        assert_caption_drawn(figure, entry["caption"], path.name)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


def step_figures(
    outdir,
    point,
    entry,
    canvas,
    magnitude,
    nm_per_px,
    field,
    amplitude,
    theta,
    valid,
    others,
):
    """The four step figures of one selected point, into ``steps_<label>``/.

    Returns the list of written paths.
    """
    plt = _pyplot()
    import matplotlib.patheffects as path_effects
    from matplotlib.patches import Circle

    directory = Path(outdir) / f"steps_{point.label}"
    directory.mkdir(parents=True, exist_ok=True)
    stroke = [path_effects.withStroke(linewidth=2.5, foreground="black")]
    n_px = int(np.asarray(canvas).shape[0])
    q_px = np.asarray(entry["q_px"], dtype=float)
    q_rad_px = np.asarray(entry["q_rad_px"], dtype=float)
    lambda_nm = float(entry["lambda_nm"])
    caption = caption_for(point, entry)
    written = []

    # ---------------------------------------------------------------- step 1
    reference_radius = max(
        [float(np.hypot(*np.asarray(other["q_px"], dtype=float))) for other in others]
        + [float(np.hypot(*q_px)), 1.0]
    )
    half = int(min(n_px // 2 - 1, max(16.0, 1.35 * reference_radius)))
    center = n_px // 2
    crop = magnitude[
        center - half : center + half + 1, center - half : center + half + 1
    ]
    low, high = (float(value) for value in np.percentile(magnitude, [5.0, 99.5]))
    span = max(np.log1p(high) - np.log1p(low), np.finfo(float).tiny)
    plane = np.clip((np.log1p(crop) - np.log1p(low)) / span, 0.0, 1.0)
    figure, axis = plt.subplots(figsize=(7.4, 7.4), dpi=DPI)
    axis.imshow(plane, cmap="plasma", origin="lower")
    for index, other in enumerate(others):
        colour = STEP_COLOURS[index % len(STEP_COLOURS)]
        other_px = np.asarray(other["q_px"], dtype=float)
        for sign in (1.0, -1.0):
            axis.add_patch(
                Circle(
                    (half + sign * other_px[0], half + sign * other_px[1]),
                    18.0 if other["label"] == point.label else 11.0,
                    fill=False,
                    lw=2.4 if other["label"] == point.label else 1.2,
                    color=colour,
                )
            )
        axis.text(
            half + other_px[0] + 20,
            half + other_px[1] + 20,
            other["label"],
            color=colour,
            fontsize=12,
            path_effects=stroke,
        )
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title(
        f"1  FFT of the corrected canvas, crop +/-{half} px "
        f"(+/-{half * 2.0 * np.pi / n_px:.2f} cycles/nm)\n"
        + caption
        + "\ncircles mark q and -q of every selected point; the filled one is this point",
        fontsize=10,
    )
    path = directory / bpl.STEP_FIGURES[0]
    assert_caption_drawn(figure, caption, path.name)
    figure.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    written.append(path)

    # ---------------------------------------------------------------- step 2
    gx = round(center + q_px[0])
    gy = round(center + q_px[1])
    span_px = int(FFT_ZOOM)
    zoom = magnitude[gy - span_px : gy + span_px + 1, gx - span_px : gx + span_px + 1]
    zoom_decades = np.log10(np.maximum(zoom, np.finfo(float).tiny))
    sigma_q_px = float(
        bpl.lq.rad_px_to_px_offsets(
            np.array([1.0 / bpl.lq.lambda_px(lambda_nm, nm_per_px), 0.0]), n_px
        )[0]
    )
    figure, axis = plt.subplots(figsize=(6.6, 6.6), dpi=DPI)
    axis.imshow(
        zoom_decades,
        origin="lower",
        cmap="plasma",
        interpolation="nearest",
        extent=(-span_px, span_px, -span_px, span_px),
        vmin=float(zoom_decades.max()) - FFT_ZOOM_DECADES,
        vmax=float(zoom_decades.max()),
    )
    for index, factor in enumerate((1.0, 2.0, 3.0)):
        axis.add_patch(
            Circle(
                (0.0, 0.0),
                factor * sigma_q_px,
                fill=False,
                color="#00e5ff",
                lw=2.4 if index == 0 else 1.3,
                alpha=1.0 if index == 0 else 0.7,
            )
        )
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_xlabel(r"$q - q_{\rm point}$   (q-px)", fontsize=10)
    axis.set_title(
        "2  q-space zoom on this point\n"
        + caption
        + rf"\nlow-pass keeps $\pm q$ only; circles $1\sigma_q,\,2\sigma_q,\,3\sigma_q$ "
        rf"($\sigma_q = {sigma_q_px:.3f}$ q-px)",
        fontsize=10,
    )
    path = directory / bpl.STEP_FIGURES[1]
    assert_caption_drawn(figure, caption, path.name)
    figure.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    written.append(path)

    # ---------------------------------------------------------------- step 3
    rows = np.arange(n_px, dtype=float)[:, None]
    columns = np.arange(n_px, dtype=float)[None, :]
    modulated = np.asarray(canvas, dtype=float) * np.exp(
        -1j * (float(q_rad_px[0]) * columns + float(q_rad_px[1]) * rows)
    )
    modulus = np.abs(modulated)
    angle = np.degrees(np.angle(modulated))
    period_px = float(bpl.TWO_PI / float(np.hypot(*q_rad_px)))
    fov = n_px * nm_per_px
    extent = (0.0, fov, 0.0, fov)
    colour_max = float(np.nanpercentile(modulus[np.isfinite(modulus)], 99.9))
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 6.4), dpi=DPI)
    axes[0].imshow(
        modulus,
        origin="lower",
        cmap=_with_bad(plt.get_cmap("viridis")),
        extent=extent,
        vmin=0.0,
        vmax=colour_max,
    )
    second = axes[1].imshow(
        angle,
        origin="lower",
        cmap=_with_bad(plt.get_cmap("twilight")),
        extent=extent,
        vmin=-180.0,
        vmax=180.0,
    )
    for axis in axes:
        axis.set_xlabel("nm", fontsize=10)
    axes[0].set_title(r"$|X|$   (input units)    window $[0,\,p_{99.9}]$", fontsize=11)
    axes[1].set_title(
        r"$\arg X$   (deg)    window $[-180^\circ,\,180^\circ]$", fontsize=11
    )
    figure.colorbar(second, ax=axes[1], fraction=0.046, pad=0.03)
    figure.suptitle(
        r"3  modulated field before the low-pass   $X(r) = T(r)\,e^{-i q\cdot r}$"
        "\n"
        + caption
        + rf"   carrier period {period_px:.2f} px = {period_px * nm_per_px:.4f} nm",
        fontsize=11,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    path = directory / bpl.STEP_FIGURES[2]
    assert_caption_drawn(figure, caption, path.name)
    figure.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    written.append(path)

    # ---------------------------------------------------------------- step 4
    shown_amplitude = np.where(valid, amplitude, np.nan)
    mean_deg = float(np.degrees(np.angle(np.mean(np.exp(1j * theta[valid]))))) % 360.0
    delta_deg = np.degrees(
        np.angle(np.exp(1j * theta) * np.exp(-1j * np.radians(mean_deg)))
    )
    shown_phase = np.where(valid, delta_deg, np.nan)
    low_amplitude = float(np.nanpercentile(shown_amplitude, 2.0))
    high_amplitude = float(np.nanpercentile(shown_amplitude, 99.5))
    phase_window = max(1.0, float(np.nanpercentile(np.abs(shown_phase), 98.0)))
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 6.4), dpi=DPI)
    axes[0].imshow(
        shown_amplitude,
        origin="lower",
        cmap=_with_bad(plt.get_cmap("inferno")),
        extent=extent,
        vmin=low_amplitude,
        vmax=high_amplitude,
    )
    second = axes[1].imshow(
        shown_phase,
        origin="lower",
        cmap=_with_bad(plt.get_cmap("twilight")),
        extent=extent,
        vmin=-phase_window,
        vmax=phase_window,
    )
    for axis in axes:
        axis.set_xlabel("nm", fontsize=10)
    axes[0].set_title(
        r"$|\psi|$   (input units)    window $p_2$-$p_{99.5}$: "
        + f"{_sci(low_amplitude)} ... {_sci(high_amplitude)}",
        fontsize=11,
    )
    axes[1].set_title(
        rf"$\arg\psi - \langle\arg\psi\rangle$   (deg)    window $\pm{phase_window:.0f}^\circ$",
        fontsize=11,
    )
    figure.colorbar(second, ax=axes[1], fraction=0.046, pad=0.03)
    figure.suptitle(
        r"4  local field after the low-pass   $\psi(r) = \mathrm{LPF}[\,T(r)\,e^{-i q\cdot r}\,]$"
        "\n" + caption + rf"   $\langle\arg\psi\rangle = {mean_deg:.2f}^\circ$, valid "
        rf"{100.0 * float(np.mean(valid)):.2f} %, grey = invalid",
        fontsize=11,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    path = directory / bpl.STEP_FIGURES[3]
    assert_caption_drawn(figure, caption, path.name)
    figure.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    written.append(path)
    return written
