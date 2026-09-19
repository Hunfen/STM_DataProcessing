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
    ring        ``ring_1x1`` or ``ring_r3``
    kind        ``per_peak`` or ``summary``
    peak        peak index for ``per_peak`` figures, ``None`` otherwise
    label       human prefix of the title
    annotation  {field: value} -- the numbers *as printed* (rounded)
    paths       {field: JSON path} -- where the exact value lives in
                ``phase_stats.json`` (dotted path with list indices)
    title       ``label`` + ``" | "`` + rendering of ``annotation``

``check_manifest`` re-derives the title from ``label`` + ``annotation``, compares
every annotated number with the exact value found at its JSON path (within half a
unit in the last printed place) and re-reads the PNG (existence, size, PIL
openability, embedded annotation).  ``python atlas.py --check MANIFEST.json``
runs exactly that check, so a third party can re-audit a delivered atlas.

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
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, SymLogNorm  # noqa: E402

BAD_COLOR = "#b0b0b0"
LADDER_ANGLES_DEG = (0.0, 120.0, 240.0)
LADDER_RAD = tuple(np.radians(a) for a in LADDER_ANGLES_DEG)

# Decimals used when a field is printed.  ``annotate`` rounds to these, so the
# printed number and the compared number are the same object by construction.
FIELD_DECIMALS = {
    "radius_px": 2, "snr": 1, "qx_px": 2, "qy_px": 2, "peak": 0,
    "median_deg": 3, "mean_deg": 3, "fwhm_deg": 3, "fwhm_deconv_deg": 3,
    "iqr_deg": 3, "R": 5, "n_clusters": 0, "mask_radius_px": 0, "n_peaks": 0,
    "q_sum_px": 4, "theta_deg": 3, "ladder_dist_deg": 3, "ratio": 6,
    "sqrt3_dev_pct": 4, "rms_deg": 4, "amp_median": 6, "total_deg": 3,
    "mirror_deg": 3, "pixels": 0, "n_figures": 0, "ref_dev_rms_deg": 4,
}
DEFAULT_DECIMALS = 3
# canonical order of the annotated fields inside a figure title
FIELD_ORDER = ("peak", "qx_px", "qy_px", "radius_px", "snr", "mask_radius_px", "n_peaks",
               "pixels", "median_deg", "mean_deg", "fwhm_deg", "fwhm_deconv_deg",
               "iqr_deg", "n_clusters", "R", "theta_deg", "ladder_dist_deg",
               "q_sum_px", "ref_dev_rms_deg", "rms_deg", "ratio", "sqrt3_dev_pct",
               "total_deg", "mirror_deg", "amp_median", "n_figures", "figures")


# --------------------------------------------------------------------------- #
# style, colormap
# --------------------------------------------------------------------------- #
def setup_style():
    """Plot style of the skill (no usetex: broken with mpl 3.10 + TeX Live 2026)."""
    matplotlib.rcParams.update({
        "text.usetex": False,
        "mathtext.fontset": "cm",
        "font.family": "serif",
        "font.serif": ["Palatino"],
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


_GWYDDION_ANCHORS = {  # mirrors stm_data_processing.stm.preview_plot.cdict_gwyddion
    "red": [(0.0, 0.0, 0.0), (0.344671, 0.658824, 0.658824),
            (0.687075, 0.953506, 0.953506), (1.0, 1.0, 1.0)],
    "green": [(0.0, 0.0, 0.0), (0.344671, 0.156863, 0.156863),
              (0.687075, 0.759686, 0.759686), (1.0, 1.0, 1.0)],
    "blue": [(0.0, 0.0, 0.0), (0.344671, 0.0588235, 0.0588235),
             (0.687075, 0.363821, 0.363821), (1.0, 1.0, 1.0)],
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
    except Exception:  # noqa: BLE001 - any import problem falls back to the anchors
        return (LinearSegmentedColormap("gwyddion", segmentdata=_GWYDDION_ANCHORS,
                                        N=4096),
                "builtin:identical anchors of cdict_gwyddion")


# --------------------------------------------------------------------------- #
# annotation contract
# --------------------------------------------------------------------------- #
def round_fields(values):
    """Round every annotated value to the number of decimals it is printed with."""
    out = {}
    for field, value in values.items():
        decimals = FIELD_DECIMALS.get(field, DEFAULT_DECIMALS)
        out[field] = int(round(float(value))) if decimals == 0 else round(float(value), decimals)
    return out


def format_field(field, value):
    decimals = FIELD_DECIMALS.get(field, DEFAULT_DECIMALS)
    if decimals == 0:
        return f"{field}={int(round(float(value)))}"
    return f"{field}={float(value):.{decimals}f}"


def annotate(annotation):
    """Render a rounded annotation dict into the exact string drawn on the figure.

    The field order is canonical (``FIELD_ORDER``, unknown fields last,
    alphabetically), so the rendering does not depend on the insertion order of
    the dictionary that was loaded back from JSON.
    """
    fields = sorted(annotation, key=lambda field: (FIELD_ORDER.index(field)
                                                   if field in FIELD_ORDER else len(FIELD_ORDER),
                                                   field))
    return " | ".join(format_field(field, annotation[field]) for field in fields)


def title_for(label, annotation):
    return f"{label} | {annotate(annotation)}"


def json_path(root, path):
    """Value at a dotted path (``a.b.0.c``) inside a nested dict/list structure."""
    current = root
    for part in str(path).split("."):
        current = current[int(part)] if isinstance(current, list) else current[part]
    return current


def check_manifest(manifest_path, stats_path=None, expected_figures=None,
                   expected_per_ring=None, verbose=True):
    """Re-audit an atlas manifest.  Returns ``(ok, failures, lines)``.

    Checks per figure: the file exists and is non-empty, PIL opens it with the
    recorded pixel size, the embedded ``stm-atlas`` text equals the manifest
    entry, the recorded title is exactly the rendering of the recorded numbers,
    and (when ``stats_path`` is given) every annotated number equals the exact
    value at its JSON path within half a unit in the last printed place.
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

    seen = set()
    counts = {}
    for entry in figures:
        name = entry["file"]
        ring = entry.get("ring", "?")
        counts[ring] = counts.get(ring, 0) + 1
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
        if size != list(entry["size_px"]):
            failures.append(f"{name}: size {size} != manifest {entry['size_px']}")
        if not embedded:
            failures.append(f"{name}: no stm-atlas text chunk")
        else:
            payload = json.loads(embedded)
            for key in ("title", "label", "annotation", "panels"):
                if payload.get(key) != entry.get(key):
                    failures.append(f"{name}: embedded {key} differs from the manifest")
        rebuilt = title_for(entry["label"], entry["annotation"])
        if rebuilt != entry["title"]:
            failures.append(f"{name}: title is not the rendering of the annotation")
        if not entry.get("panels"):
            failures.append(f"{name}: no panel list in the manifest")
        if entry.get("kind") == "per_peak" and entry.get("peak") is None:
            failures.append(f"{name}: kind per_peak without a peak index")
        if entry.get("kind") == "summary" and entry.get("peak") is not None:
            failures.append(f"{name}: kind summary with a peak index")
        if stats is not None:
            for field, value in entry["annotation"].items():
                where = entry.get("paths", {}).get(field)
                if where is None:
                    failures.append(f"{name}: no JSON path for {field}")
                    continue
                try:
                    exact = float(json_path(stats, where))
                except (KeyError, IndexError, TypeError):
                    failures.append(f"{name}: JSON path {where} not found for {field}")
                    continue
                decimals = FIELD_DECIMALS.get(field, DEFAULT_DECIMALS)
                tolerance = 0.5 * 10.0 ** (-decimals) + 1e-12
                if abs(float(value) - exact) > tolerance:
                    failures.append(f"{name}: {field} printed {value} vs {exact} at "
                                    f"{where} (tolerance {tolerance:g})")
    lines = manifest.get("reference_lines_deg")
    if lines != [0.0, 120.0, 240.0]:
        failures.append(f"manifest: reference_lines_deg is {lines}, expected the "
                        f"2 pi k / 3 ladder [0, 120, 240]")
    if not manifest.get("reference_lines_note"):
        failures.append("manifest: no reference_lines_note for the histogram lines")
    per_ring = manifest.get("per_ring", {})
    for ring, info in per_ring.items():
        if counts.get(ring, 0) != info.get("figures"):
            failures.append(f"ring {ring}: {counts.get(ring, 0)} figures in the "
                            f"manifest list vs {info.get('figures')} declared")
    if expected_figures is not None and len(figures) != expected_figures:
        failures.append(f"{len(figures)} figures in total vs {expected_figures} declared")
    if expected_per_ring is not None:
        for ring in ("ring_1x1", "ring_r3"):
            if counts.get(ring, 0) != expected_per_ring:
                failures.append(f"ring {ring}: {counts.get(ring, 0)} figures vs "
                                f"{expected_per_ring} expected")
    report(f"manifest: {manifest_path}")
    report(f"figures checked: {len(figures)} (per ring: "
           f"{', '.join(f'{k}={v}' for k, v in sorted(counts.items()))})")
    report(f"failures: {len(failures)}")
    for failure in failures[:40]:
        report(f"  - {failure}")
    return (not failures), failures, lines


# --------------------------------------------------------------------------- #
# atlas writer
# --------------------------------------------------------------------------- #
class Atlas:
    """Write figures (PNG) together with their manifest entries."""

    def __init__(self, outdir, cmap, dpi=300, ring=None):
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.cmap = cmap
        self.dpi = int(dpi)
        self.ring = ring
        self.figures = []

    # -- contract ---------------------------------------------------------- #
    def add(self, fig, name, label, values, paths, ring, kind, peak=None,
            panels=None):
        """Save ``fig`` as ``name`` and register its annotation contract.

        ``panels`` names every panel of the figure left to right (and row by row
        for a grid), so the manifest states what a file actually contains -- in
        particular that the per-peak ``phi(r)`` map is a panel of the
        ``*_phi_dist.png`` figure rather than a separate file.
        """
        annotation = round_fields(values)
        title = title_for(label, annotation)
        path = self.outdir / name
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        entry = self._embed(path, {"title": title, "label": label,
                                   "annotation": annotation,
                                   "panels": list(panels or [])})
        entry.update({"ring": ring, "kind": kind, "peak": peak, "paths": dict(paths)})
        self.figures.append(entry)
        return entry

    def _embed(self, path, payload):
        from PIL import Image, PngImagePlugin

        with Image.open(path) as image:
            image.load()
            size = list(image.size)
            info = PngImagePlugin.PngInfo()
            for key, value in (image.info or {}).items():
                if isinstance(value, str) and key not in ("stm-atlas",):
                    info.add_text(key, value)
            info.add_text("stm-atlas", json.dumps(payload, sort_keys=True))
            image.save(path, format="PNG", pnginfo=info)
        return {"file": path.name, "size_px": size, **payload}

    # -- panels ------------------------------------------------------------ #
    def _pair_figure(self, psi, valid, label, values, paths, name, ring, kind,
                     peak=None, mask_note=None,
                     panels=("amplitude (log)", "phase")):
        amp = np.where(valid, np.abs(psi), np.nan)
        phase = np.where(valid, np.angle(psi), np.nan)
        sample = np.abs(psi)[valid]
        cmap_amp = self.cmap.copy()
        cmap_amp.set_bad(color=BAD_COLOR)
        cmap_phase = plt.get_cmap("seismic").copy()
        cmap_phase.set_bad(color=BAD_COLOR)
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        if sample.size and sample.max() > 0:
            linear = sample[sample > 0].min() if np.any(sample > 0) else sample.max()
            norm = SymLogNorm(linthresh=max(linear, np.finfo(float).tiny),
                              linscale=0.5, vmin=0.0, vmax=sample.max())
        else:
            norm = None
        image = axes[0].imshow(amp, cmap=cmap_amp, origin="lower", norm=norm)
        axes[0].set_title(f"{label} - amplitude (log)", fontsize=13)
        fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)
        image = axes[1].imshow(phase, cmap=cmap_phase, origin="lower",
                               vmin=-np.pi, vmax=np.pi)
        axes[1].set_title(f"{label} - phase", fontsize=13)
        fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        if mask_note:
            fig.suptitle(mask_note, fontsize=11)
        fig.tight_layout()
        return self.add(fig, name, label, values, paths, ring, kind, peak,
                        panels=panels)

    # -- public figures ---------------------------------------------------- #
    def qspace_mask(self, fft2, centres, mask_radius, label, values, paths, ring, name):
        magnitude = np.abs(fft2)
        lo, hi = np.percentile(magnitude, [5, 99.5])
        logged = np.log(1.0 + magnitude)
        span = np.log(1.0 + hi) - np.log(1.0 + lo)
        norm = np.clip((logged - np.log(1.0 + lo)) / span, 0.0, 1.0)
        fig, ax = plt.subplots(figsize=(6.5, 6))
        ax.imshow(norm, cmap="inferno", origin="lower")
        # The peak index of a circle is its position in ``centres``, i.e. the same
        # index the per-peak figures and phase_stats.json use (p0 .. p5); the label
        # is placed radially outside the circle, away from the FFT2 centre, so it
        # never covers the masked reflection itself.
        centre_q = (fft2.shape[1] / 2.0, fft2.shape[0] / 2.0)
        for index, (qx, qy) in enumerate(centres):
            ax.add_patch(plt.Circle((qx, qy), mask_radius, fill=False,
                                    color="#39ff14", lw=1.2))
            radial = np.array([float(qx) - centre_q[0], float(qy) - centre_q[1]])
            length = float(np.hypot(*radial))
            unit = radial / length if length > 0 else np.array([1.0, 0.0])
            tip = np.array([float(qx), float(qy)]) + (mask_radius + 8.0) * unit
            ax.text(float(tip[0]), float(tip[1]), f"p{index}", color="#39ff14",
                    fontsize=10, fontweight="bold", ha="center", va="center")
        ax.set_xlim(0, fft2.shape[1])
        ax.set_ylim(0, fft2.shape[0])
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        return self.add(fig, name, label, values, paths, ring, "summary",
                        panels=["FFT2 log amplitude with the ring masks "
                                "(p0-p5 labelled at the circle rims)"])

    def mask_pair(self, psi, valid, label, values, paths, ring, name, kind, peak,
                  mask_note=None):
        return self._pair_figure(psi, valid, label, values, paths, name, ring, kind,
                                 peak, mask_note)

    def _hist_panel(self, ax, hist, edges, xlabel, ylabel, label):
        """One phase-distribution panel; the dashed lines are the 2 pi k / 3 ladder.

        The axis label spells the convention out (``2 pi k / 3, k = 0, 1, 2`` in
        radians and ``0/120/240 deg``), so the reference lines of every histogram
        in the atlas are self-describing.
        """
        ax.plot((edges[:-1] + edges[1:]) / 2.0, hist, color="steelblue", lw=1.4)
        for angle in LADDER_RAD:
            ax.axvline(angle, color="0.35", ls="--", lw=1.0)
        ax.set_xlabel(f"{xlabel}   [dashed: 2 pi k / 3, k = 0,1,2  =  0/120/240 deg]",
                      fontsize=11)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(label, fontsize=12)
        ax.set_xticks(list(LADDER_RAD) + [np.pi])
        ax.set_xticklabels(["0", "2pi/3", "4pi/3", "pi"], fontsize=11)

    def phi_distribution(self, phi, good, amp, hist, edges, label, values, paths,
                         ring, name, kind, peak):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
        image = axes[0].imshow(np.where(good, phi, np.nan), cmap="hsv", origin="lower",
                               vmin=0.0, vmax=2 * np.pi)
        axes[0].set_title(f"{label} - phi(r) map", fontsize=13)
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)
        self._hist_panel(axes[1], hist, edges, "phi (rad)",
                         "amplitude-weighted density", f"{label} - phi distribution")
        fig.tight_layout()
        return self.add(fig, name, label, values, paths, ring, kind, peak,
                        panels=["phi(r) map", "phi distribution"])

    def grid_histograms(self, items, label, values, paths, ring, name):
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        for ax, item in zip(axes.ravel(), items):
            self._hist_panel(ax, item["hist"], item["edges"], "phi (rad)", "density",
                             item["label"])
        fig.tight_layout()
        return self.add(fig, name, label, values, paths, ring, "summary",
                        panels=["six phi distributions (2x3)"])

    def grid_maps(self, items, label, values, paths, ring, name):
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        image = None
        for ax, item in zip(axes.ravel(), items):
            image = ax.imshow(np.where(item["good"], item["phi"], np.nan), cmap="hsv",
                              origin="lower", vmin=0.0, vmax=2 * np.pi)
            ax.set_title(item["label"], fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
        if image is not None:
            fig.colorbar(image, ax=axes, fraction=0.02, pad=0.02)
        fig.tight_layout()
        return self.add(fig, name, label, values, paths, ring, "summary",
                        panels=["six phi(r) maps (2x3)"])

    def theta_field(self, theta, good, amp, hist, edges, label, values, paths,
                    ring, name):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
        image = axes[0].imshow(np.where(good, theta, np.nan), cmap="hsv", origin="lower",
                               vmin=0.0, vmax=2 * np.pi)
        axes[0].set_title(f"{label} - theta(r) map", fontsize=13)
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)
        self._hist_panel(axes[1], hist, edges, "theta (rad)",
                         "amplitude-weighted density", f"{label} - theta distribution")
        fig.tight_layout()
        return self.add(fig, name, label, values, paths, ring, "summary",
                        panels=["theta(r) map", "theta distribution"])

    def case_histograms(self, items, label, values, paths, ring, name):
        fig, axes = plt.subplots(1, 3, figsize=(17, 5))
        bins = np.linspace(-np.pi, np.pi, 361)
        for ax, item in zip(axes.ravel(), items):
            ax.hist(item["phase"], bins=bins, color="steelblue", edgecolor="none")
            ax.set_xlabel("phase (rad)", fontsize=12)
            ax.set_ylabel("count", fontsize=12)
            ax.set_title(f"{item['label']} (360 bins)", fontsize=12)
            ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
            ax.set_xticklabels(["0", "pi/2", "pi", "3pi/2", "2pi"], fontsize=11)
        fig.tight_layout()
        return self.add(fig, name, label, values, paths, ring, "summary",
                        panels=["FFT2 phase histogram",
                                "complement mask phase histogram",
                                "mask phase histogram"])

    # -- manifest ---------------------------------------------------------- #
    def manifest(self, extra=None):
        per_ring = {}
        for entry in self.figures:
            ring = entry["ring"]
            info = per_ring.setdefault(ring, {"figures": 0, "per_peak": 0, "summary": 0,
                                              "peaks": 0})
            info["figures"] += 1
            if entry["kind"] == "per_peak":
                info["per_peak"] += 1
                info["peaks"] = max(info["peaks"], int(entry["peak"]) + 1)
            else:
                info["summary"] += 1
        payload = {
            "atlas_version": 1,
            "producer": "stm_phase_analysis.py (skill stm-topo-phase-analysis v2)",
            "per_ring": per_ring,
            "total_figures": len(self.figures),
            "figures": self.figures,
        }
        if extra:
            payload.update(extra)
        return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", required=True, metavar="MANIFEST.json",
                        help="audit an existing atlas manifest")
    parser.add_argument("--stats", default=None,
                        help="phase_stats.json to compare the annotated numbers with")
    parser.add_argument("--expected-figures", type=int, default=None)
    parser.add_argument("--expected-per-ring", type=int, default=None)
    args = parser.parse_args(argv)
    ok, failures, _ = check_manifest(args.check, stats_path=args.stats,
                                     expected_figures=args.expected_figures,
                                     expected_per_ring=args.expected_per_ring)
    if ok:
        print("ATLAS CHECK PASSED")
        return 0
    print(f"ATLAS CHECK FAILED ({len(failures)} failure(s))")
    return 1


if __name__ == "__main__":
    sys.exit(main())
