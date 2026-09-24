"""Figure atlas of the phase analysis: drawing, manifest and the annotation contract.

An atlas figure is not just a picture: every number drawn on it is also written
into a machine readable manifest entry and into the PNG ``tEXt`` chunk
``stm-atlas``.  One dictionary (``annotation``) is the single source of truth --
the same values are rendered into the title, stored in the manifest and embedded
in the image -- so "the number on the figure" and "the number in the JSON" can be
compared programmatically (``check_manifest``), by this script and by anybody
else, without OCR.

Contract of one figure (``Atlas.add``):

    file        PNG inside the atlas directory (basename only)
    group       ``ring_1x1`` or ``ring_r3``
    kind        ``per_peak`` (one reflection), ``summary`` (one ring) or
                ``pairwise`` (one reflection pair)
    peak        peak index for ``per_peak`` figures, ``None`` otherwise
    pair        pair key ``<j>+<k>`` for ``pairwise`` figures, ``None`` otherwise
    label       human prefix of the title
    annotation  {field: value} -- the numbers *as printed* (rounded)
    paths       {field: JSON path} -- where the exact value lives in
                ``phase_stats.json`` (dotted path with list indices)
    panels      the panels of the figure, left to right (row by row in a grid)
    mappable    True when the figure draws a 2D map / density image
    colorbars   number of colour bars of the figure
    norm        the colour scale of a mappable figure, as a machine readable
                record ``{scale, kind, vmin, vmax[, linthresh, linscale]}``
    title       ``label`` + ``" | "`` + rendering of ``annotation``

Two layout rules of this module are enforced programmatically, not just followed
by hand:

* every map / density figure carries at least one colour bar (``Atlas.add``
  raises when a mappable figure has none);
* every colour bar lives **outside** every data axes of its figure, i.e. the
  intersection area of the colour-bar axes rectangle and any data axes rectangle
  is exactly zero (``Atlas.add`` raises otherwise, ``check_manifest`` re-audits it
  from the manifest);

and one colour-scale rule:

* the colour scales are global, not per figure: every theta(r) map spans
  ``[0, 2 pi]``, all amplitude maps of the whole atlas share one symmetric-log
  scale (one ``vmax`` and one ``linthresh`` over all twelve demodulated fields),
  every ``D(r)`` map spans ``[-pi, pi]`` and every ``a(r)`` map spans ``[-1, 1]``
  (``check_manifest`` compares the recorded scale of every figure with the global
  declaration in ``phase_stats.json``).

``check_manifest`` re-derives the title from ``label`` + ``annotation``, compares
every annotated number with the exact value found at its JSON path (within half a
unit in the last printed place), re-reads the PNG (existence, size, PIL
openability, embedded annotation) and re-checks the colour-bar and colour-scale
rules.  ``python atlas.py --check MANIFEST.json`` runs exactly that check, so a
third party can re-audit a delivered atlas.

No physical statement is made anywhere in this module: rings, wavevectors,
phases and statistics only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize, SymLogNorm

BAD_COLOR = "#b0b0b0"
GROUPS = ("ring_1x1", "ring_r3")
KINDS = ("per_peak", "summary", "pairwise")
# Font size of the figure-level title (the manifest title, rendered wrapped, so a
# long annotation stays inside the canvas instead of stretching it).
TITLE_FONTSIZE_MAP = 8
TITLE_FONTSIZE_DIST = 9
TITLE_FONTSIZE_GRID = 11
# A saved figure is refused when the top strip of its pixels carries no ink at all:
# that is exactly the defect of a figure written without any title (its title band
# was pure white while the manifest still declared a title).
TITLE_BAND_FRACTION = 0.10
TITLE_BAND_MIN_INK = 20
TITLE_BAND_MAX_LUMINANCE = 220

# The global colour scales of the whole atlas.  They are constants of the skill,
# so any two figures of a run can be compared by eye.
THETA_VMIN = 0.0
THETA_VMAX = 2.0 * np.pi
PHASE_DIFF_VMIN = -np.pi
PHASE_DIFF_VMAX = np.pi
AMP_DIFF_VMIN = -1.0
AMP_DIFF_VMAX = 1.0
AMPLITUDE_LINSCALE = 0.5
FFT2_VMIN = 0.0
FFT2_VMAX = 1.0

# Decimals used when a field is printed.  ``annotate`` rounds to these, so the
# printed number and the compared number are the same object by construction.
FIELD_DECIMALS = {
    "radius_px": 2,
    "snr": 1,
    "qx_px": 2,
    "qy_px": 2,
    "peak": 0,
    "median_deg": 3,
    "mean_deg": 3,
    "fwhm_deg": 3,
    "fwhm_deconv_deg": 3,
    "iqr_deg": 3,
    "R": 5,
    "n_clusters": 0,
    "n_peaks": 0,
    "n_pairs": 0,
    "n_valid": 0,
    "q_sum_px": 4,
    "theta_deg": 3,
    "ratio": 6,
    "sqrt3_dev_pct": 4,
    "rms_deg": 4,
    "amp_median": 6,
    "total_deg": 3,
    "mirror_deg": 3,
    "pixels": 0,
    "n_figures": 0,
    "ref_dev_rms_deg": 4,
}
DEFAULT_DECIMALS = 3
# canonical order of the annotated fields inside a figure title
FIELD_ORDER = (
    "peak",
    "qx_px",
    "qy_px",
    "radius_px",
    "snr",
    "n_peaks",
    "n_pairs",
    "n_valid",
    "pixels",
    "median_deg",
    "mean_deg",
    "fwhm_deg",
    "fwhm_deconv_deg",
    "iqr_deg",
    "n_clusters",
    "R",
    "theta_deg",
    "q_sum_px",
    "ref_dev_rms_deg",
    "rms_deg",
    "ratio",
    "sqrt3_dev_pct",
    "total_deg",
    "mirror_deg",
    "amp_median",
    "n_figures",
    "figures",
)


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
    """Topography colormap: the package's ``gwyddion``, else the same anchors.

    Returns ``(cmap, source)`` so the log can state which object was used; the
    built-in fallback uses the identical anchor points, hence identical colours.
    """
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
# global colour scales
# --------------------------------------------------------------------------- #
def theta_norm():
    """Colour scale of every theta(r) map: ``[0, 2 pi]``, global."""
    return Normalize(vmin=THETA_VMIN, vmax=THETA_VMAX)


def phase_diff_norm():
    """Colour scale of every ``D(r)`` map: ``[-pi, pi]``, global."""
    return Normalize(vmin=PHASE_DIFF_VMIN, vmax=PHASE_DIFF_VMAX)


def amp_diff_norm():
    """Colour scale of every ``a(r)`` map: ``[-1, 1]``, global."""
    return Normalize(vmin=AMP_DIFF_VMIN, vmax=AMP_DIFF_VMAX)


def fft2_norm():
    """Colour scale of the q-space figure: the clipped ``[0, 1]`` log amplitude."""
    return Normalize(vmin=FFT2_VMIN, vmax=FFT2_VMAX)


def amplitude_norm(vmax, linthresh):
    """The one symmetric-log scale shared by all twelve amplitude maps.

    ``vmin = 0`` and ``vmax`` = the largest valid ``|psi|`` of the whole run, and
    ``linthresh`` = the **pooled median of the positive valid** ``|psi|`` **of the
    whole run**: one global statistic over every positive sample value of the twelve
    demodulated fields (not the minimum of those values), so the two ends of the
    scale are global numbers.  Every amplitude figure of a run is drawn with this one
    norm, so the twelve maps can be compared directly.
    """
    return SymLogNorm(
        linthresh=float(linthresh),
        linscale=AMPLITUDE_LINSCALE,
        vmin=0.0,
        vmax=float(vmax),
    )


def theta_cmap():
    """The colour map of every theta(r) map: ``hsv`` with a visible bad colour.

    ``set_bad`` paints the pixels that carry no value (the NaN padding of the
    corrected canvas and the pixels outside the sample) in the same grey the
    amplitude maps use, so a reader sees where the analysis stops instead of a
    transparent hole.
    """
    cmap = plt.get_cmap("hsv").copy()
    cmap.set_bad(color=BAD_COLOR)
    return cmap


def norm_record(scale, norm):
    """Machine readable record of one colour scale (the manifest and JSON share it).

    ``scale`` names the global scale (``theta`` / ``amplitude`` / ``phase_diff`` /
    ``amp_diff`` / ``fft2``); ``check_manifest`` compares every figure record with
    the declaration of that name inside ``phase_stats.json``, so "all theta maps
    use the same span" is a checked statement, not a comment.
    """
    record = {
        "scale": str(scale),
        "kind": "symlog" if isinstance(norm, SymLogNorm) else "linear",
        "vmin": float(norm.vmin),
        "vmax": float(norm.vmax),
    }
    if isinstance(norm, SymLogNorm):
        record["linthresh"] = float(norm.linthresh)
        # SymLogNorm does not expose ``linscale`` again, so the constant of this
        # module (the one every amplitude map was built with) is recorded instead.
        record["linscale"] = float(getattr(norm, "linscale", AMPLITUDE_LINSCALE))
    return record


# --------------------------------------------------------------------------- #
# colour-bar placement audit
# --------------------------------------------------------------------------- #
def bbox_overlap_area(first, second):
    """Area of the intersection of two rectangles in figure coordinates."""
    width = min(first.x1, second.x1) - max(first.x0, second.x0)
    height = min(first.y1, second.y1) - max(first.y0, second.y0)
    if width <= 0.0 or height <= 0.0:
        return 0.0
    return float(width * height)


def colorbar_layout(data_axes, colorbar_axes):
    """Audit one figure: ``(colorbars, outside, worst_overlap_area)``.

    ``outside`` is True when the rectangle of every colour bar is disjoint from the
    rectangle of every data axes; the return value is what ``Atlas.add`` asserts
    and what the manifest records, so the rule "the colour bar is drawn outside the
    data area" is a property of the delivered files.
    """
    overlaps = [
        bbox_overlap_area(axes.get_position(), bar.get_position())
        for bar in colorbar_axes
        for axes in data_axes
    ]
    return (
        len(list(colorbar_axes)),
        all(area <= 0.0 for area in overlaps),
        float(max(overlaps)) if overlaps else 0.0,
    )


# --------------------------------------------------------------------------- #
# annotation contract
# --------------------------------------------------------------------------- #
def round_fields(values):
    """Round every annotated value to the number of decimals it is printed with."""
    out = {}
    for field, value in values.items():
        decimals = FIELD_DECIMALS.get(field, DEFAULT_DECIMALS)
        out[field] = (
            round(float(value)) if decimals == 0 else round(float(value), decimals)
        )
    return out


def format_field(field, value):
    decimals = FIELD_DECIMALS.get(field, DEFAULT_DECIMALS)
    if decimals == 0:
        return f"{field}={round(float(value))}"
    return f"{field}={float(value):.{decimals}f}"


def annotate(annotation):
    """Render a rounded annotation dict into the exact string drawn on the figure.

    The field order is canonical (``FIELD_ORDER``, unknown fields last,
    alphabetically), so the rendering does not depend on the insertion order of
    the dictionary that was loaded back from JSON.
    """
    fields = sorted(
        annotation,
        key=lambda field: (
            FIELD_ORDER.index(field) if field in FIELD_ORDER else len(FIELD_ORDER),
            field,
        ),
    )
    return " | ".join(format_field(field, annotation[field]) for field in fields)


def title_for(label, annotation):
    return f"{label} | {annotate(annotation)}"


# --------------------------------------------------------------------------- #
# canvas title contract
# --------------------------------------------------------------------------- #
def rendered_title_texts(fig):
    """Every text a figure really renders as a title, read back from the canvas.

    Collected from the artist objects that matplotlib will draw: the figure-level
    title (``fig._suptitle``), the centre / left / right title of every axes, and
    the free figure texts.  This is the function the gate of ``Atlas.add`` uses, so
    "the figure carries its title" is a statement about the canvas and not about
    the manifest.
    """
    out = []
    suptitle = getattr(fig, "_suptitle", None)
    texts = []
    if suptitle is not None:
        texts.append(suptitle)
    for axes in fig.axes:
        for text in (
            getattr(axes, "title", None),
            getattr(axes, "_left_title", None),
            getattr(axes, "_right_title", None),
        ):
            if text is not None:
                texts.append(text)
    texts.extend(fig.texts)
    for text in texts:
        value = text.get_text()
        if value and value not in out:  # the figure title is also in fig.texts
            out.append(value)
    return out


def require_canvas_title(fig, title):
    """Assert that the canvas carries exactly ``title`` as a rendered title.

    Raised when the figure has no title at all, when its title is a different
    string (a typo, a renamed label, an annotation rendered from other numbers), or
    when it only declares the title in the manifest / PNG text chunk while the
    canvas stays empty.
    """
    rendered = rendered_title_texts(fig)
    if title not in rendered:
        raise AssertionError(
            f"the canvas does not carry the manifest title {title!r}; rendered "
            f"title text(s): {rendered if rendered else 'none'}"
        )
    return rendered


def set_canvas_title(fig, title, fontsize=TITLE_FONTSIZE_MAP):
    """Draw the manifest title on the canvas as a wrapped figure-level title.

    ``wrap=True`` keeps the long annotation inside the canvas (matplotlib wraps it
    at draw time) while ``Text.get_text`` still returns the one-line string, so the
    text that the gate compares is exactly the string of the manifest.
    """
    fig.suptitle(title, fontsize=fontsize, wrap=True)
    return fig._suptitle


def axes_title_texts(axes_list):
    """The non-empty title texts of the given axes (centre / left / right).

    This is what a *data axis* draws on its own, as opposed to the figure-level
    title: a single-axes figure must not carry any of these (the user asked for one
    title line per figure), while every panel of a multi-panel grid carries its own
    identifier here.
    """
    out = []
    for axes in axes_list:
        for text in (
            getattr(axes, "title", None),
            getattr(axes, "_left_title", None),
            getattr(axes, "_right_title", None),
        ):
            if text is not None and text.get_text():
                out.append(text.get_text())
    return out


def title_strip_px(fig, data_axes, dpi):
    """Height in pixels of the saved region above the topmost data axes.

    That strip is the room a figure has for its titles; measured from the tight
    bounding box that ``savefig(bbox_inches="tight")`` will use, so the recorded
    number is the length the checker has to look at in the saved PNG.  A figure
    written with no title above its axes leaves this strip empty.
    """
    try:
        renderer = fig.canvas.get_renderer()
        bbox = fig.get_tightbbox(renderer)  # inches
        top_inch = (
            max((axes.get_position().y1 for axes in data_axes), default=0.9)
            * fig.get_figheight()
        )
        return max(0, round((bbox.y1 - top_inch) * float(dpi)))
    except Exception:  # pragma: no cover - a renderer-less canvas falls back
        return max(1, round(TITLE_BAND_FRACTION * fig.get_figheight() * float(dpi)))


def title_band_ink(image, rows=None):
    """Number of dark pixels in the top ``rows`` of a saved PNG (the title strip).

    A figure written without any title above its data axes leaves that strip pure
    white; the counter is the pixel-level counterpart of the canvas-title gate, read
    back from the file instead of from the manifest.
    """
    array = np.asarray(image.convert("L"))
    if rows is None:
        rows = max(1, int(array.shape[0] * TITLE_BAND_FRACTION))
    band = array[: max(1, int(rows)), :]
    return int(np.count_nonzero(band < TITLE_BAND_MAX_LUMINANCE))


def json_path(root, path):
    """Value at a dotted path (``a.b.0.c``) inside a nested dict/list structure."""
    current = root
    for part in str(path).split("."):
        current = current[int(part)] if isinstance(current, list) else current[part]
    return current


def check_manifest(
    manifest_path,
    stats_path=None,
    expected_figures=None,
    expected_per_ring=None,
    verbose=True,
):
    """Re-audit an atlas manifest.  Returns ``(ok, failures, lines)``.

    Checks per figure: the file exists and is non-empty, PIL opens it with the
    recorded pixel size, the embedded ``stm-atlas`` text equals the manifest
    entry, the recorded title is exactly the rendering of the recorded numbers,
    the title band of the saved pixels carries ink (a figure saved without a title
    has a pure white top strip), the ``kind`` / ``peak`` / ``pair`` triple is
    self-consistent, a map / density figure carries a colour bar that is disjoint
    from every data axes, and (when ``stats_path`` is given) every annotated number
    equals the exact value at its JSON path within half a unit in the last printed
    place and every recorded colour scale equals the global declaration.
    """
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    atlas_dir = manifest_path.parent
    stats = json.loads(Path(stats_path).read_text()) if stats_path else None
    figures = manifest.get("figures", [])
    failures, lines = [], []

    def report(text):
        lines.append(text)
        if verbose:
            print(text)

    declared_norms = ((stats or {}).get("atlas") or {}).get("norms") or {}
    seen = set()
    counts = {}
    for entry in figures:
        name = entry["file"]
        group = entry.get("group", "?")
        counts[group] = counts.get(group, 0) + 1
        if name in seen:
            failures.append(f"{name}: duplicate manifest entry")
        seen.add(name)
        path = atlas_dir / name
        if not path.is_file():
            failures.append(f"{name}: missing")
            continue
        if path.stat().st_size <= 0:
            failures.append(f"{name}: zero size")
            continue
        try:
            from PIL import Image
        except ImportError:  # pragma: no cover - PIL ships with the environment
            failures.append("PIL is not importable: cannot audit the PNGs")
            break
        with Image.open(path) as image:
            image.load()
            size = list(image.size)
            embedded = image.text.get("stm-atlas") if hasattr(image, "text") else None
            band_ink = title_band_ink(image, entry.get("title_strip_px"))
        if size != list(entry["size_px"]):
            failures.append(f"{name}: size {size} != manifest {entry['size_px']}")
        if not entry.get("title_strip_px"):
            failures.append(
                f"{name}: the manifest records no title strip above the data axes"
            )
        elif band_ink < TITLE_BAND_MIN_INK:
            failures.append(
                f"{name}: the {entry['title_strip_px']} px strip above the data axes "
                f"carries {band_ink} ink pixel(s) (< {TITLE_BAND_MIN_INK}): the figure "
                f"was written without a title on the canvas"
            )
        if not embedded:
            failures.append(f"{name}: no stm-atlas text chunk")
        else:
            payload = json.loads(embedded)
            for key in (
                "title",
                "canvas_title",
                "axes_titles",
                "panel_axes",
                "title_strip_px",
                "title_band_ink",
                "label",
                "annotation",
                "panels",
                "mappable",
                "colorbars",
                "colorbar_outside",
                "norm",
            ):
                if payload.get(key) != entry.get(key):
                    failures.append(f"{name}: embedded {key} differs from the manifest")
        rebuilt = title_for(entry["label"], entry["annotation"])
        if rebuilt != entry["title"]:
            failures.append(f"{name}: title is not the rendering of the annotation")
        panel_axes = entry.get("panel_axes")
        axes_titles = entry.get("axes_titles")
        if not isinstance(axes_titles, list) or not isinstance(panel_axes, int):
            failures.append(f"{name}: the manifest records no axes-title accounting")
        elif panel_axes == 1 and axes_titles:
            failures.append(
                f"{name}: a single-axes figure must carry one title line only, but "
                f"the data axis draws {axes_titles!r}"
            )
        elif panel_axes > 1 and len(axes_titles) != panel_axes:
            failures.append(
                f"{name}: {panel_axes} panels but {len(axes_titles)} panel "
                f"title(s): {axes_titles!r}"
            )
        if entry.get("canvas_title") != entry["title"]:
            failures.append(
                f"{name}: the recorded canvas title {entry.get('canvas_title')!r} is "
                f"not the manifest title; the figure was written without its title "
                f"on the canvas"
            )
        if not entry.get("panels"):
            failures.append(f"{name}: no panel list in the manifest")
        if "mappable" not in entry:
            failures.append(f"{name}: no map/density flag in the manifest")
        if entry.get("mappable") and not entry.get("colorbars"):
            failures.append(f"{name}: map / density figure without a colour bar")
        if entry.get("mappable") and not entry.get("colorbar_outside"):
            failures.append(
                f"{name}: colour bar overlaps a data axes "
                f"(overlap area {entry.get('colorbar_overlap')})"
            )
        if entry.get("mappable") and not entry.get("norm"):
            failures.append(f"{name}: map / density figure without a colour scale")
        norm = entry.get("norm") or {}
        if norm:
            scale = norm.get("scale")
            expected = declared_norms.get(scale)
            if expected is None:
                failures.append(
                    f"{name}: colour scale {scale!r} is not declared in "
                    f"phase_stats.json (declared: {sorted(declared_norms)})"
                )
            else:
                for key in ("kind", "vmin", "vmax", "linthresh", "linscale"):
                    if key in expected and norm.get(key) != expected[key]:
                        failures.append(
                            f"{name}: colour scale {scale} {key} = {norm.get(key)} "
                            f"vs the global declaration {expected[key]}"
                        )
        kind = entry.get("kind")
        if group not in GROUPS:
            failures.append(f"{name}: unknown group {group!r}")
        if kind not in KINDS:
            failures.append(f"{name}: unknown kind {kind!r}")
        elif kind == "per_peak" and entry.get("peak") is None:
            failures.append(f"{name}: kind per_peak without a peak index")
        elif kind == "summary" and (
            entry.get("peak") is not None or entry.get("pair") is not None
        ):
            failures.append(f"{name}: kind summary with a peak index or a pair key")
        elif kind == "pairwise" and not entry.get("pair"):
            failures.append(f"{name}: kind pairwise without a pair key")
        elif kind == "per_peak" and entry.get("pair") is not None:
            failures.append(f"{name}: kind per_peak with a pair key")
        if stats is not None:
            for field, value in entry["annotation"].items():
                where = entry.get("paths", {}).get(field)
                if where is None:
                    failures.append(f"{name}: no JSON path for {field}")
                    continue
                try:
                    exact = float(json_path(stats, where))
                except KeyError, IndexError, TypeError:
                    failures.append(f"{name}: JSON path {where} not found for {field}")
                    continue
                decimals = FIELD_DECIMALS.get(field, DEFAULT_DECIMALS)
                tolerance = 0.5 * 10.0 ** (-decimals) + 1e-12
                if abs(float(value) - exact) > tolerance:
                    failures.append(
                        f"{name}: {field} printed {value} vs {exact} at "
                        f"{where} (tolerance {tolerance:g})"
                    )
    for group, info in (manifest.get("per_group") or {}).items():
        if counts.get(group, 0) != info.get("figures"):
            failures.append(
                f"group {group}: {counts.get(group, 0)} figures in the "
                f"manifest list vs {info.get('figures')} declared"
            )
    if expected_figures is not None and len(figures) != expected_figures:
        failures.append(
            f"{len(figures)} figures in total vs {expected_figures} declared"
        )
    if expected_per_ring is not None:
        for group in GROUPS:
            if counts.get(group, 0) != expected_per_ring:
                failures.append(
                    f"group {group}: {counts.get(group, 0)} figures vs "
                    f"{expected_per_ring} expected"
                )
    report(f"manifest: {manifest_path}")
    report(
        f"figures checked: {len(figures)} (per group: "
        f"{', '.join(f'{k}={v}' for k, v in sorted(counts.items()))})"
    )
    report(f"failures: {len(failures)}")
    for failure in failures[:40]:
        report(f"  - {failure}")
    return (not failures), failures, lines


# --------------------------------------------------------------------------- #
# atlas writer
# --------------------------------------------------------------------------- #
class Atlas:
    """Write figures (PNG) together with their manifest entries."""

    def __init__(self, outdir, cmap, dpi=300):
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.cmap = cmap
        self.dpi = int(dpi)
        self.figures = []

    # -- contract ---------------------------------------------------------- #
    def add(
        self,
        fig,
        name,
        label,
        values,
        paths,
        group,
        kind,
        peak=None,
        pair=None,
        panels=None,
        mappable=False,
        data_axes=(),
        colorbar_axes=(),
        norm=None,
    ):
        """Save ``fig`` as ``name`` and register its annotation contract.

        ``panels`` names every panel of the figure left to right (and row by row
        for a grid), so the manifest states what a file actually contains.
        ``mappable`` marks a figure that draws a 2D map or density image; such a
        figure must carry a colour bar, and the colour bar must be disjoint from
        every data axes -- both are asserted here, before the file is written.
        """
        annotation = round_fields(values)
        title = title_for(label, annotation)
        require_canvas_title(fig, title)
        axes_titles = axes_title_texts(list(data_axes))
        if len(data_axes) == 1 and axes_titles:
            raise AssertionError(
                f"{name}: a single-axes figure carries the figure-level annotation "
                f"title only; this data axis still draws its own short title "
                f"{axes_titles!r}"
            )
        if len(data_axes) > 1 and len(axes_titles) != len(data_axes):
            raise AssertionError(
                f"{name}: every panel of a multi-panel figure must keep its own "
                f"identifier title; {len(data_axes)} panels but "
                f"{len(axes_titles)} panel title(s): {axes_titles!r}"
            )
        colorbars, outside, overlap = colorbar_layout(
            list(data_axes), list(colorbar_axes)
        )
        if mappable and colorbars == 0:
            raise AssertionError(
                f"{name}: a map / density figure must carry a colour bar"
            )
        if mappable and not outside:
            raise AssertionError(
                f"{name}: the colour bar overlaps a data axes "
                f"(intersection area {overlap:g} in figure coordinates)"
            )
        path = self.outdir / name
        strip = title_strip_px(fig, list(data_axes), self.dpi)
        payload = {
            "title": title,
            "canvas_title": title,
            "axes_titles": list(axes_titles),
            "panel_axes": len(data_axes),
            "title_strip_px": int(strip),
            "label": label,
            "annotation": annotation,
            "panels": list(panels or []),
            "mappable": bool(mappable),
            "colorbars": int(colorbars),
            "colorbar_outside": bool(outside),
            "colorbar_overlap": float(overlap),
            "norm": dict(norm) if norm else None,
        }
        entry = self._embed(path, payload, fig, strip)
        entry.update(
            {
                "group": group,
                "kind": kind,
                "peak": peak,
                "pair": pair,
                "paths": dict(paths),
            }
        )
        self.figures.append(entry)
        return entry

    def _embed(self, path, payload, fig, strip):
        from PIL import Image, PngImagePlugin

        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        with Image.open(path) as image:
            image.load()
            size = list(image.size)
            ink = title_band_ink(image, strip)
            payload["title_band_ink"] = int(ink)
            info = PngImagePlugin.PngInfo()
            for key, value in (image.info or {}).items():
                if isinstance(value, str) and key not in ("stm-atlas",):
                    info.add_text(key, value)
            info.add_text("stm-atlas", json.dumps(payload, sort_keys=True))
            image.save(path, format="PNG", pnginfo=info)
        if strip <= 0 or ink < TITLE_BAND_MIN_INK:
            raise AssertionError(
                f"{path.name}: the saved canvas leaves {strip} px above its data axes "
                f"carrying {ink} ink pixel(s) (< {TITLE_BAND_MIN_INK}): the title is "
                f"not drawn inside the canvas"
            )
        return {"file": path.name, "size_px": size, **payload}

    # -- panels ------------------------------------------------------------ #
    def titled(self, fig, label, values, fontsize=TITLE_FONTSIZE_MAP):
        """Draw this figure's manifest title on the canvas and return the text.

        The string is ``title_for(label, round_fields(values))`` -- the same one
        ``add`` stores in the manifest and embeds in the PNG -- rendered wrapped so
        a long annotation stays inside the canvas.  ``add`` re-derives that string
        and refuses a figure that does not carry it, so a drawing method cannot
        quietly produce a titleless figure.
        """
        title = title_for(label, round_fields(values))
        set_canvas_title(fig, title, fontsize=fontsize)
        return title

    def _distribution_panel(
        self, ax, hist, edges, xlabel, ylabel, label, show_title=False
    ):
        """One phase-distribution panel with plain phase ticks.

        The panel carries the distribution and the phase ticks only: there is no
        reference-line overlay and no second, re-scaled axis.  ``show_title`` draws
        the panel's own identifier (used by the multi-panel grids); a single-axes
        figure leaves it off, because its figure-level annotation title is already
        the identifying line.
        """
        edges = np.asarray(edges, dtype=float)
        centres = 0.5 * (edges[:-1] + edges[1:])
        ax.plot(centres, hist, color="steelblue", lw=1.4)
        ax.set_xlim(float(edges[0]), float(edges[-1]))
        if edges[0] < 0.0:
            ticks = [-np.pi, -np.pi / 2.0, 0.0, np.pi / 2.0, np.pi]
            labels = ["-pi", "-pi/2", "0", "pi/2", "pi"]
        else:
            ticks = [
                0.0,
                np.pi / 2.0,
                np.pi,
                3.0 * np.pi / 2.0,
                2.0 * np.pi,
            ]
            labels = ["0", "pi/2", "pi", "3pi/2", "2pi"]
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_xlabel(f"{xlabel} (phase angle)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=12)
        if show_title:
            ax.set_title(label, fontsize=12)

    def _colorbar(self, fig, mappable, ax, **kwargs):
        """Draw a colour bar in its own axes and return that axes."""
        return fig.colorbar(mappable, ax=ax, **kwargs).ax

    # -- per-reflection figures -------------------------------------------- #
    def amplitude_map(self, amp, sample, norm, label, values, paths, group, name, peak):
        """``|psi(r)|`` map of one reflection on the shared symmetric-log scale."""
        cmap = self.cmap.copy()
        cmap.set_bad(color=BAD_COLOR)
        fig, ax = plt.subplots(figsize=(6.5, 6))
        image = ax.imshow(
            np.where(sample, amp, np.nan), cmap=cmap, origin="lower", norm=norm
        )
        # single-axes figure: the figure-level annotation title is the only title
        ax.set_xticks([])
        ax.set_yticks([])
        bar = self._colorbar(fig, image, ax, fraction=0.046, pad=0.04)
        self.titled(fig, label, values, fontsize=TITLE_FONTSIZE_MAP)
        fig.tight_layout()
        return self.add(
            fig,
            name,
            label,
            values,
            paths,
            group,
            "per_peak",
            peak,
            panels=["amplitude |psi(r)| (global symlog scale)"],
            mappable=True,
            data_axes=[ax],
            colorbar_axes=[bar],
            norm=norm_record("amplitude", norm),
        )

    def theta_map(self, theta, sample, label, values, paths, group, name, peak):
        """``theta(r)`` map of one reflection over the single sample.

        One panel, no other content: the visible pixels of this map are exactly the
        pixels the statistics of that reflection use (``pixels`` in the title is the
        size of that sample, read from ``phase_stats.json`` by the checker).
        """
        cmap = theta_cmap()
        fig, ax = plt.subplots(figsize=(6.5, 6))
        image = ax.imshow(
            np.where(sample, theta, np.nan),
            cmap=cmap,
            origin="lower",
            vmin=THETA_VMIN,
            vmax=THETA_VMAX,
        )
        # single-axes figure: the figure-level annotation title is the only title
        ax.set_xticks([])
        ax.set_yticks([])
        bar = self._colorbar(fig, image, ax, fraction=0.046, pad=0.04)
        self.titled(fig, label, values, fontsize=TITLE_FONTSIZE_MAP)
        fig.tight_layout()
        return self.add(
            fig,
            name,
            label,
            values,
            paths,
            group,
            "per_peak",
            peak,
            panels=["theta(r) map over the sample of the statistics"],
            mappable=True,
            data_axes=[ax],
            colorbar_axes=[bar],
            norm=norm_record("theta", theta_norm()),
        )

    def theta_distribution(self, hist, edges, label, values, paths, group, name, peak):
        """Amplitude-weighted theta distribution of one reflection (one panel)."""
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        self._distribution_panel(
            ax,
            hist,
            edges,
            "theta (rad)",
            "amplitude-weighted density",
            label,
        )
        self.titled(fig, label, values, fontsize=TITLE_FONTSIZE_DIST)
        fig.tight_layout()
        return self.add(
            fig,
            name,
            label,
            values,
            paths,
            group,
            "per_peak",
            peak,
            panels=["theta distribution over the sample of the statistics"],
            data_axes=[ax],
        )

    # -- ring summaries ---------------------------------------------------- #
    def grid_theta_histograms(self, items, label, values, paths, group, name):
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        for ax, item in zip(axes.ravel(), items, strict=True):
            self._distribution_panel(
                ax,
                item["hist"],
                item["edges"],
                "theta (rad)",
                "density",
                item["label"],
                show_title=True,
            )
        self.titled(fig, label, values, fontsize=TITLE_FONTSIZE_GRID)
        fig.tight_layout()
        return self.add(
            fig,
            name,
            label,
            values,
            paths,
            group,
            "summary",
            panels=["six theta distributions (2x3)"],
            data_axes=list(axes.ravel()),
        )

    def grid_theta_maps(self, items, label, values, paths, group, name):
        """The six theta(r) maps of one ring on one shared global scale.

        The colour bar of a grid is placed by hand in the strip that is carved out
        of the right edge of the grid, so "outside the data area" holds for a
        multi-axes figure as well (the automatic placement of matplotlib overlaps
        the last column, which the audit of ``Atlas.add`` refuses).
        """
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        image = None
        for ax, item in zip(axes.ravel(), items, strict=True):
            image = ax.imshow(
                np.where(item["sample"], item["theta"], np.nan),
                cmap=theta_cmap(),
                origin="lower",
                vmin=THETA_VMIN,
                vmax=THETA_VMAX,
            )
            ax.set_title(item["label"], fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
        self.titled(fig, label, values, fontsize=TITLE_FONTSIZE_GRID)
        fig.tight_layout()
        flat = list(axes.ravel())
        positions = [ax.get_position() for ax in flat]
        left = min(position.x0 for position in positions)
        right = max(position.x1 for position in positions)
        bottom = min(position.y0 for position in positions)
        top = max(position.y1 for position in positions)
        span, keep = right - left, 0.86
        for ax in flat:
            position = ax.get_position()
            ax.set_position(
                [
                    left + (position.x0 - left) * keep,
                    position.y0,
                    position.width * keep,
                    position.height,
                ]
            )
        bar = fig.add_axes(
            [left + span * (keep + 0.02), bottom, span * 0.06, top - bottom]
        )
        fig.colorbar(image, cax=bar)
        return self.add(
            fig,
            name,
            label,
            values,
            paths,
            group,
            "summary",
            panels=["six theta(r) maps (2x3), one shared global scale"],
            mappable=True,
            data_axes=flat,
            colorbar_axes=[bar],
            norm=norm_record("theta", theta_norm()),
        )

    def ring_members(self, fft2, centres, label, values, paths, group, name):
        """FFT2 log amplitude with the six ring members marked and labelled.

        The label of a member is placed radially *outside* its marker, away from the
        FFT2 centre, so it never covers the reflection itself; the index is the same
        ``p0 .. p5`` index the per-reflection figures and ``phase_stats.json`` use.
        """
        magnitude = np.abs(fft2)
        lo, hi = np.percentile(magnitude, [5, 99.5])
        logged = np.log(1.0 + magnitude)
        span = np.log(1.0 + hi) - np.log(1.0 + lo)
        norm = np.clip((logged - np.log(1.0 + lo)) / span, 0.0, 1.0)
        fig, ax = plt.subplots(figsize=(6.5, 6))
        image = ax.imshow(
            norm, cmap="inferno", origin="lower", vmin=FFT2_VMIN, vmax=FFT2_VMAX
        )
        centre_q = (fft2.shape[1] / 2.0, fft2.shape[0] / 2.0)
        for index, (qx, qy) in enumerate(centres):
            ax.plot(
                [float(qx)],
                [float(qy)],
                marker="o",
                markersize=9,
                markerfacecolor="none",
                markeredgecolor="#39ff14",
                markeredgewidth=1.2,
            )
            radial = np.array([float(qx) - centre_q[0], float(qy) - centre_q[1]])
            length = float(np.hypot(*radial))
            unit = radial / length if length > 0 else np.array([1.0, 0.0])
            tip = np.array([float(qx), float(qy)]) + 12.0 * unit
            ax.text(
                float(tip[0]),
                float(tip[1]),
                f"p{index}",
                color="#39ff14",
                fontsize=10,
                fontweight="bold",
                ha="center",
                va="center",
            )
        ax.set_xlim(0, fft2.shape[1])
        ax.set_ylim(0, fft2.shape[0])
        ax.set_xticks([])
        ax.set_yticks([])
        bar = self._colorbar(
            fig, image, ax, fraction=0.046, pad=0.04, label="normalised log |FFT2|"
        )
        self.titled(fig, label, values, fontsize=TITLE_FONTSIZE_GRID)
        fig.tight_layout()
        return self.add(
            fig,
            name,
            label,
            values,
            paths,
            group,
            "summary",
            panels=[
                "FFT2 log amplitude with the six ring members "
                "(p0-p5 labelled at the rims)"
            ],
            mappable=True,
            data_axes=[ax],
            colorbar_axes=[bar],
            norm=norm_record("fft2", fft2_norm()),
        )

    # -- pairwise figures -------------------------------------------------- #
    def pair_field(self, field, sample, label, values, paths, group, name, pair, kind):
        """``D(r)`` or ``a(r)`` map of one reflection pair."""
        if kind == "phase_diff":
            cmap, norm = "seismic", phase_diff_norm()
            panel = "D(r) = wrap(arg psi_j - arg psi_k)"
        else:
            cmap, norm = "coolwarm", amp_diff_norm()
            panel = "a(r) = (|psi_j| - |psi_k|) / (|psi_j| + |psi_k|)"
        fig, ax = plt.subplots(figsize=(6.5, 6))
        image = ax.imshow(
            np.where(sample, field, np.nan),
            cmap=cmap,
            origin="lower",
            vmin=float(norm.vmin),
            vmax=float(norm.vmax),
        )
        # single-axes figure: the figure-level annotation title is the only title
        ax.set_xticks([])
        ax.set_yticks([])
        bar = self._colorbar(fig, image, ax, fraction=0.046, pad=0.04)
        self.titled(fig, label, values, fontsize=TITLE_FONTSIZE_MAP)
        fig.tight_layout()
        return self.add(
            fig,
            name,
            label,
            values,
            paths,
            group,
            "pairwise",
            None,
            pair,
            panels=[panel],
            mappable=True,
            data_axes=[ax],
            colorbar_axes=[bar],
            norm=norm_record(kind, norm),
        )

    def pair_phase_diff_dist(
        self, hist, edges, label, values, paths, group, name, pair
    ):
        """Amplitude-weighted circular histogram of ``D`` itself over ``(-pi, pi]``.

        The weight of a sample is ``|psi_j psi_k|``; the axis is the signed
        difference, so this panel shows where the two reflections actually sit
        relative to each other instead of a re-scaled axis.
        """
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        self._distribution_panel(
            ax,
            hist,
            edges,
            "D = wrap(arg psi_j - arg psi_k) (rad)",
            "amplitude-weighted density",
            label,
        )
        self.titled(fig, label, values, fontsize=TITLE_FONTSIZE_DIST)
        fig.tight_layout()
        return self.add(
            fig,
            name,
            label,
            values,
            paths,
            group,
            "pairwise",
            None,
            pair,
            panels=["D distribution over the pair sample, weight |psi_j psi_k|"],
            data_axes=[ax],
        )

    # -- manifest ---------------------------------------------------------- #
    def manifest(self, extra=None):
        per_group, pair_keys = {}, {}
        for entry in self.figures:
            group = entry["group"]
            info = per_group.setdefault(
                group,
                {
                    "figures": 0,
                    "per_peak": 0,
                    "summary": 0,
                    "pairwise": 0,
                    "mappable": 0,
                    "peaks": 0,
                    "pairs": 0,
                },
            )
            info["figures"] += 1
            if entry["mappable"]:
                info["mappable"] += 1
            if entry["kind"] == "per_peak":
                info["per_peak"] += 1
                info["peaks"] = max(info["peaks"], int(entry["peak"]) + 1)
            elif entry["kind"] == "pairwise":
                info["pairwise"] += 1
                pair_keys.setdefault(group, set()).add(str(entry["pair"]))
            else:
                info["summary"] += 1
        for group, keys in pair_keys.items():
            per_group[group]["pairs"] = len(keys)
        payload = {
            "atlas_version": 4,
            "producer": "stm_phase_analysis.py (skill phase-analysis v4)",
            "groups": list(GROUPS),
            "per_group": per_group,
            "total_figures": len(self.figures),
            "colorbar_rule": (
                "every map / density figure carries at least one colour bar and "
                "every colour bar axes is disjoint from every data axes "
                "(intersection area 0); re-audited per figure by check_manifest"
            ),
            "colour_scale_rule": (
                "global scales: theta maps [0, 2 pi]; all twelve amplitude maps one "
                "symmetric-log scale with a shared vmax and linthresh; D maps "
                "[-pi, pi]; a maps [-1, 1]"
            ),
            "figures": self.figures,
        }
        if extra:
            payload.update(extra)
        return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        required=True,
        metavar="MANIFEST.json",
        help="audit an existing atlas manifest",
    )
    parser.add_argument(
        "--stats",
        default=None,
        help="phase_stats.json to compare the annotated numbers and the colour "
        "scales with",
    )
    parser.add_argument("--expected-figures", type=int, default=None)
    parser.add_argument("--expected-per-ring", type=int, default=None)
    args = parser.parse_args(argv)
    ok, failures, _ = check_manifest(
        args.check,
        stats_path=args.stats,
        expected_figures=args.expected_figures,
        expected_per_ring=args.expected_per_ring,
    )
    if ok:
        print("ATLAS CHECK PASSED")
        return 0
    print(f"ATLAS CHECK FAILED ({len(failures)} failure(s))")
    return 1


if __name__ == "__main__":
    sys.exit(main())
