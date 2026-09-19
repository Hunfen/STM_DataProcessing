"""Shared helpers of the affine-correction skill (self-contained copy; do not import the phase-analysis skill)."""
from __future__ import annotations

import sys

import matplotlib
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

BAD_COLOR = "#b0b0b0"


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
    except Exception:  # any import problem falls back to the anchors
        return (LinearSegmentedColormap("gwyddion", segmentdata=_GWYDDION_ANCHORS,
                                        N=4096),
                "builtin:identical anchors of cdict_gwyddion")


# --------------------------------------------------------------------------- #
# radius clustering of the detected reflections
# --------------------------------------------------------------------------- #
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
                ring["radius"] = float(np.average([m[4] for m in ring["members"]],
                                                  weights=[m[2] for m in ring["members"]]))
                ring["total_amplitude"] = float(total)
                placed = True
                break
        if not placed:
            rings.append({"radius": record[4], "members": [record],
                          "total_amplitude": record[2]})
    keep = [ring for ring in rings if len(ring["members"]) >= min_members]
    keep.sort(key=lambda ring: -ring["total_amplitude"])
    return keep
