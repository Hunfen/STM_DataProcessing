"""Geometry and phase statistics of a corrected topography image (two rings).

For the six reflections of two rings and with exactly the same definitions for
both, the script measures the demodulated phase field of every reflection

    psi_q(r)   = FFT^-1{ FFT[ T(r) exp(-i q.r) ] * exp(-Lambda^2 |k|^2 / 2) }
    theta_q(r) = arg psi_q(r)          ~  +phi_q(r)     (no q.r ramp)

with the Gaussian-window engine of the sibling ``local-q-map`` skill
(``--lambda-nm``, default 3.0 nm), plus the lattice-referenced gauge fix that makes
the two rings comparable, plus the pairwise analysis of two reflections at a time
(phase-difference field, its amplitude-weighted circular histogram, and the
normalised amplitude-difference field), and it writes the complete figure atlas
together with one JSON that carries every number (a drawn number is the JSON
number by construction; see ``atlas.py``).

One sample rule holds everywhere, for every estimator and for every figure: the
sample of a reflection is **every valid pixel of the corrected canvas** (all
pixels the correction did not turn into NaN), weighted by the demodulated
amplitude ``|psi(r)|``; the sample of a pair is the intersection of the two
validity masks, weighted by ``|psi_j psi_k|``.  There is no percentile, no
threshold and no other selection rule anywhere in this skill.

The rings are named

    ring_1x1   the reference ring: its six reflections pin the lattice origin r0
    ring_r3    the ring at 1/sqrt(3) of the ring_1x1 radius, located by that
               radius ratio alone

The reference ring is never assumed: ``--anchor`` states how it is chosen and the
log reports which path was taken.  When no ring sits at 1/sqrt(3) of the
reference radius the script reports "r3 ring not found" and stops instead of
analysing an unpaired ring.

Scope: geometry and circular statistics only.  The script attaches no physical
meaning to a radius or a phase and prints no physical statement.

Usage:
    cd /path/to/STM_DataProcessing
    MPLCONFIGDIR=<writable> PYTHONDONTWRITEBYTECODE=1 \
        .venv/bin/python <this script> CORRECTED.csv -o OUT_DIR -L <field of view nm> \
        [--anchor auto|strongest|outer|inner|radius] [--reference-radius-px R] \
        [--detector auto|package|builtin] [--lambda-nm 3.0] [--no-figures]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import namedtuple
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import atlas as at  # noqa: E402
import phasemath as pm  # noqa: E402
import phasepipe as pp  # noqa: E402

SKILL_VERSION = "4.0"
PER_PEAK_FIGURES = ("amplitude", "theta_map", "theta_dist")
SUMMARY_FIGURES = (
    "theta_hist_summary",
    "theta_map_summary",
    "ring_members_qspace",
)
PAIR_FIGURES = ("phase_diff", "phase_diff_dist", "amp_diff")
PEAKS_PER_RING = 6
WITHIN_PAIRS_PER_RING = 3
FIGURES_PER_RING = (
    PEAKS_PER_RING * len(PER_PEAK_FIGURES)
    + len(SUMMARY_FIGURES)
    + WITHIN_PAIRS_PER_RING * len(PAIR_FIGURES)
)
TOTAL_FIGURES = 2 * FIGURES_PER_RING

# The two rings that are analysed; the atlas groups are exactly these two (the
# atlas figures every reflection of a ring and every pair inside a ring).
RING_TAGS = ("ring_1x1", "ring_r3")

# Bins of the atlas phase histograms: 0.1 deg over the circle.
HISTOGRAM_BINS = 361

# The annotated numbers of a pairwise figure and the JSON keys they are read from.
PAIR_ANNOTATION = {
    "mean_deg": "phase_diff_mean",
    "median_deg": "phase_diff_median",
    "R": "phase_diff_R",
    "fwhm_deg": "phase_diff_fwhm_deg",
    "n_clusters": "phase_diff_n_clusters",
    "amp_median": "amp_diff_median",
    "n_valid": "n_valid",
}
PAIR_FIGURE_FIELDS = {
    "phase_diff": ("mean_deg", "median_deg", "R", "fwhm_deg", "n_valid"),
    "phase_diff_dist": (
        "mean_deg",
        "median_deg",
        "R",
        "fwhm_deg",
        "n_clusters",
        "n_valid",
    ),
    "amp_diff": ("amp_median", "n_valid"),
}

# Options that were removed before v4; passing one is an explicit error instead of
# a silent "analysed something else".
REMOVED_OPTIONS = {
    "--gate": (
        "the amplitude-threshold sample selection was removed in v4: the sample of "
        "every statistic and of every map is every valid (non-NaN) pixel of the "
        "corrected canvas, weighted by |psi(r)|; there is no threshold to set"
    ),
    "--pct": (
        "the percentile option of the removed hard-mask engine was already gone in "
        "v3; the local-q-map Gaussian window needs no percentile"
    ),
}

# One "field of view <value> nm" occurrence of a correction log
FovMatch = namedtuple("FovMatch", "value size_nm corrected label line")

# A correction log states the field of view of the *input* canvas first and the
# field of view of the *corrected* canvas later (the correction resamples onto a
# larger canvas at a constant nm/px, so the corrected field of view is the one a
# phase analysis of the corrected CSV needs).
FOV_FROM_LOG_HELP = (
    "read the field of view from a correction log; the "
    "'# corrected canvas: ... field of view <value> nm' line of "
    "stm_topo_correct.py wins, otherwise the last 'field of view <value> nm' "
    "line of the log is used"
)


# --------------------------------------------------------------------------- #
# command line and small helpers
# --------------------------------------------------------------------------- #
def reject_removed_options(argv):
    """Fail loudly when an option that this version removed is passed.

    A removed option must never be silently ignored: the run would then produce a
    different analysis than the caller asked for.  Every entry of
    ``REMOVED_OPTIONS`` produces its own message and a non-zero exit code.
    """
    for token in list(argv):
        name = token.split("=", 1)[0]
        if name in REMOVED_OPTIONS:
            raise SystemExit(
                f"error: {name} is not an option of this script: "
                f"{REMOVED_OPTIONS[name]}"
            )


def parse_args(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    reject_removed_options(argv)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "input", help="corrected topography CSV (square, may contain NaN)"
    )
    parser.add_argument(
        "-o",
        "--outdir",
        default=None,
        help="output directory (default: next to the input CSV)",
    )
    parser.add_argument(
        "-L",
        "--size-nm",
        type=float,
        default=None,
        help="field of view of the corrected image in nm",
    )
    parser.add_argument("--size-nm-from-log", default=None, help=FOV_FROM_LOG_HELP)
    parser.add_argument(
        "--fft2",
        default=None,
        help="complex FFT2 npy; default: computed from the CSV with the "
        "same FFT routine as the detector path",
    )
    parser.add_argument(
        "--delimiter", default=",", help="delimiter of the corrected CSV (default ',')"
    )
    parser.add_argument(
        "--detector",
        default="auto",
        choices=("auto", "package", "builtin"),
        help="reflection detector: the package bragg_peak detection "
        "(default when importable) or the builtin one",
    )
    parser.add_argument(
        "--stm-lib",
        default="/Users/hunfen/Documents/GitHub/STM_DataProcessing/src",
        help="STM_DataProcessing src directory (package detection and "
        "the gwyddion colormap)",
    )
    parser.add_argument(
        "--lambda-nm",
        type=float,
        default=3.0,
        help="Gaussian window width of the local-q-map demodulation "
        "engine in nm (default 3.0)",
    )
    parser.add_argument(
        "--anchor",
        default="auto",
        choices=("auto", "strongest", "outer", "inner", "radius"),
        help="how the reference ring (ring_1x1) is chosen (default auto)",
    )
    parser.add_argument(
        "--reference-radius-px",
        type=float,
        default=None,
        help="reference ring radius in FFT pixels (--anchor radius)",
    )
    parser.add_argument(
        "--match-tol",
        type=float,
        default=0.03,
        help="relative tolerance of the explicit reference radius",
    )
    parser.add_argument(
        "--pair-tol",
        type=float,
        default=0.03,
        help="relative tolerance of the sqrt(3) radius ratio used to form a ring pair",
    )
    parser.add_argument(
        "--r3-tol",
        type=float,
        default=0.03,
        help="relative tolerance of the 1/sqrt(3) lookup that locates the r3 ring",
    )
    parser.add_argument(
        "--ring-cluster-tol",
        type=float,
        default=0.02,
        help="relative radius tolerance of the ring clustering",
    )
    parser.add_argument(
        "--project-q",
        dest="project_q",
        action="store_true",
        default=True,
        help="shift the three independent wavevectors so that their sum "
        "is exactly zero before the triple product (default on)",
    )
    parser.add_argument(
        "--no-project-q",
        dest="project_q",
        action="store_false",
        help="keep the detected wavevectors (a non-zero sum turns theta "
        "into a canvas-wide ramp)",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=3600,
        help="histogram bins over 2 pi (default 3600 = 0.1 deg)",
    )
    parser.add_argument(
        "--smooth-deg",
        type=float,
        default=2.0,
        help="circular Gaussian smoothing of the histogram used for the "
        "FWHM (default 2 deg)",
    )
    parser.add_argument(
        "--patch-half",
        type=int,
        default=8,
        help="sub-pixel localizer patch half width of the package detector (default 8)",
    )
    parser.add_argument("--dpi", type=int, default=300, help="figure resolution")
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="statistics only: write no atlas figure",
    )
    return parser.parse_args(argv)


def parse_fov_log(text):
    """All 'field of view <value> nm' occurrences of a correction log, in order.

    The pattern accepts the optional 'corrected canvas: ' label, so one scan
    covers both the input-canvas line and the corrected-canvas line of
    ``stm_topo_correct.py``.
    """
    pattern = re.compile(r"field of view\s+([\d.]+)\s+nm")
    matches = []
    for line in text.splitlines():
        hit = pattern.search(line)
        if not hit:
            continue
        matches.append(
            FovMatch(
                value=float(hit.group(1)),
                size_nm=hit.group(1),
                corrected="corrected canvas" in line,
                label=(
                    "corrected canvas line"
                    if "corrected canvas" in line
                    else "input canvas line"
                ),
                line=line.strip(),
            )
        )
    return matches


def field_of_view(args):
    """The field of view in nm plus a one-line provenance string.

    ``--size-nm-from-log`` reads a correction log: the corrected-canvas line is
    the field of view of the CSV that is being analysed, while the input-canvas
    line is the field of view *before* the correction (a different canvas), so
    the corrected-canvas line wins.  Without it the last occurrence wins: a log
    written by ``stm_topo_correct.py`` ends with the corrected canvas, and a
    plain log with a single line has that line as its last one.  Neither present
    is an error.
    """
    headline = "# field of view from log"
    if args.size_nm is not None:
        value = float(args.size_nm)
        return value, f"{headline}: {value:.4f} nm (command line -L/--size-nm)"
    if args.size_nm_from_log:
        source = Path(args.size_nm_from_log)
        if not source.is_file():
            raise SystemExit(f"--size-nm-from-log {source}: no such file")
        try:
            text = source.read_text()
        except OSError as exc:  # report the log, do not traceback
            raise SystemExit(
                f"--size-nm-from-log {source}: cannot read it ({exc})"
            ) from exc
        matches = parse_fov_log(text)
        if matches:
            chosen = next((item for item in matches if item.corrected), matches[-1])
            return chosen.value, (
                f"{headline}: {chosen.value:.4f} nm ({chosen.label}; {source})"
            )
        raise SystemExit(f"no 'field of view <value> nm' line in {source}")
    raise SystemExit("a field of view is required: pass -L <nm> or --size-nm-from-log")


def load_package(stm_lib):
    """Import the package detection/FFT, or report why it is unavailable."""
    if stm_lib:
        sys.path.insert(0, str(stm_lib))
    try:
        from stm_data_processing.utils.bragg_peak import (
            compute_fft2,
            detect_bragg_peaks,
        )
    except Exception as exc:  # any import failure -> builtin detector
        return None, f"{type(exc).__name__}: {exc}"
    return (compute_fft2, detect_bragg_peaks), "imported"


def write_csv(path, rows, keys=None):
    path = Path(path)
    if not rows:
        path.write_text("")
        return
    keys = keys or sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(keys)
        for row in rows:
            values = []
            for key in keys:
                value = row.get(key, "")
                if isinstance(value, (list, tuple)):
                    value = "|".join(f"{float(item):.6f}" for item in value)
                elif isinstance(value, float):
                    value = f"{value:.6f}"
                values.append(value)
            writer.writerow(values)


def dump_json(payload, indent=2):
    """``json.dumps`` with integer arrays kept on a single line.

    ``hist_counts`` of every pair is a ``[bins_x, bins_y]`` integer array; the
    default of one number per line would turn the JSON into a multi-million-line
    file, so integer lists are serialized compactly and substituted back.
    """
    token = "___stm-compact-integer-array___"

    def compact(node):
        if isinstance(node, dict):
            return {key: compact(value) for key, value in node.items()}
        if isinstance(node, list):
            if node and all(isinstance(value, int) for value in node):
                return token + json.dumps(node) + token
            return [compact(value) for value in node]
        return node

    text = json.dumps(compact(payload), indent=indent)
    return text.replace(f'"{token}', "").replace(f'{token}"', "")


def fmt(value, decimals=3):
    if value is None:
        return "n/a"
    if isinstance(value, float) and not np.isfinite(value):
        return "nan"
    return f"{value:.{decimals}f}"


def cluster_widths(phi, weights, clusters, window_deg=15.0):
    """Amplitude-weighted circular standard deviation of each cluster's samples."""
    widths = []
    for cluster in clusters:
        centre = np.radians(cluster["centre_deg"])
        selected = np.abs(pm.wrap_pm_pi(phi - centre)) <= np.radians(window_deg)
        if np.count_nonzero(selected) < 2:
            widths.append(float("nan"))
            continue
        _mean, resultant, _total = pm.circ_mean(phi[selected], weights[selected])
        widths.append(
            float(
                np.degrees(np.sqrt(-2.0 * np.log(resultant)))
                if 0.0 < resultant < 1.0
                else 0.0
            )
        )
    return widths


def reference_lattice_vectors(peaks_q):
    """Direct-lattice vectors ``a1, a2`` of a reflection set (``q_i . a_j = 2 pi d_ij``).

    Two independent detected wavevectors are picked deterministically (the two
    with the largest separation in angle); the direct lattice they generate maps
    every member of the ring onto ``2 pi`` multiples, which is exactly the
    degeneracy of the origin fit.
    """
    vectors = np.asarray(peaks_q, dtype=float)
    best = (None, -1.0)
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            cross = abs(float(np.linalg.det(np.array([vectors[i], vectors[j]]))))
            if cross > best[1]:
                best = ((i, j), cross)
    i, j = best[0]
    basis = np.column_stack([vectors[i], vectors[j]])
    inverse = np.linalg.inv(basis)
    return (2.0 * np.pi * inverse[0], 2.0 * np.pi * inverse[1])


def hexagon_residual_px(peaks_q):
    """How far the detected wavevectors sit from a perfect hexagon, in pixels.

    The ring's mean radius and the circular mean of the six polar angles define an
    ideal hexagon; the residual of every peak against it measures the quality of
    the detected geometry (a mixed phase field biases the peak positions, so this
    is a peak-position uncertainty, not only a detector property).
    """
    vectors = np.asarray(peaks_q, dtype=float)
    radii = np.hypot(vectors[:, 0], vectors[:, 1])
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    mean_radius = float(np.mean(radii))
    resultant = np.exp(1j * 6.0 * angles).mean()  # hexagon has 60 deg symmetry
    rotation = float(np.angle(resultant) / 6.0)
    ideal = np.array(
        [
            [
                mean_radius * np.cos(rotation + k * np.pi / 3.0),
                mean_radius * np.sin(rotation + k * np.pi / 3.0),
            ]
            for k in range(6)
        ]
    )
    distances = [
        float(np.min(np.hypot(ideal[:, 0] - qx, ideal[:, 1] - qy)))
        for qx, qy in vectors
    ]
    return {
        "mean_radius_px": mean_radius,
        "radial_rms_px": float(np.sqrt(np.mean((radii - mean_radius) ** 2))),
        "hexagon_residual_rms_px": float(np.sqrt(np.mean(np.square(distances)))),
        "hexagon_residual_max_px": float(np.max(distances)),
    }


def project_members(members):
    """Shift the three independent wavevectors so that their sum is exactly zero."""
    combo, _ = pp.triple_selection([(member[0], member[1]) for member in members])
    shift = (
        np.array([members[index][:2] for index in combo], dtype=float).sum(axis=0) / 3.0
    )
    prepared = []
    for index, member in enumerate(members):
        if index in combo:
            qx, qy = float(member[0] - shift[0]), float(member[1] - shift[1])
            prepared.append((qx, qy, member[2], member[3], float(np.hypot(qx, qy))))
        else:
            prepared.append(member)
    return prepared, shift


# --------------------------------------------------------------------------- #
# analysis of one ring (identical code path for both rings)
# --------------------------------------------------------------------------- #
def analyse_ring(tag, ring, topo, valid, lambda_nm, nm_per_px, args):
    detections = pp.order_ring_members(ring["members"], expect=PEAKS_PER_RING)
    q_sum_detected = float(
        np.hypot(sum(m[0] for m in detections), sum(m[1] for m in detections))
    )
    if args.project_q:
        members, shift = project_members(detections)
    else:
        members, shift = detections, np.zeros(2)
    records, fields, psi = pp.analyse_ring(
        topo,
        valid,
        ring,
        lambda_nm,
        nm_per_px,
        bins=args.bins,
        smooth_deg=args.smooth_deg,
        prefix=tag + "_",
        peaks_override=members,
    )

    peaks = []
    for index, (record, detected) in enumerate(zip(records, detections, strict=True)):
        name = record["name"]
        stats = record["stats"]
        phase = stats["phase"]
        theta_peak, amp = np.asarray(fields[name][1]), np.asarray(fields[name][2])
        sample = pp.sample_mask(amp, valid)
        peaks.append(
            {
                "name": name,
                "index": index,
                "qx_px": float(record["q_px"][0]),
                "qy_px": float(record["q_px"][1]),
                "qx_detected_px": float(detected[0]),
                "qy_detected_px": float(detected[1]),
                "integer": [int(record["integer"][0]), int(record["integer"][1])],
                "radius_px": float(record["radius_px"]),
                "snr": float(record["snr"]),
                "fft_amplitude": float(record["fft_amplitude"]),
                "n_samples": int(stats["n_samples"]),
                "n_zero_weight_px": int(
                    np.count_nonzero(np.asarray(valid, dtype=bool) & ~sample)
                ),
                "phase_mean_deg": float(phase["mean_deg"]),
                "phase_median_deg": float(phase["median_deg"]),
                "phase_R": float(phase["resultant_R"]),
                "phase_circ_std_deg": float(phase["circ_std_deg"]),
                "phase_fwhm_deg": float(phase["fwhm_deg"]),
                "phase_fwhm_deconv_deg": float(phase["fwhm_deconv_deg"]),
                "phase_iqr_deg": float(phase["iqr_deg"]),
                "phase_median_span_deg": float(phase["median_span_deg"]),
                "phase_kappa": float(phase["kappa"]),
                "n_clusters": int(phase["n_clusters"]),
                "cluster_centres_deg": [
                    float(c["centre_deg"]) for c in phase["clusters"]
                ],
                "cluster_weight_fractions": [
                    float(c["weight_fraction"]) for c in phase["clusters"]
                ],
                "cluster_widths_deg": cluster_widths(
                    theta_peak[sample], amp[sample], phase["clusters"]
                ),
                "amp_mean": float(stats["amp_mean"]),
                "amp_median": float(stats["amp_median"]),
                "amp_fwhm": float(stats["amp_fwhm"]),
            }
        )

    friedel = {}
    for row in pp.pair_summary(records, "phase", "mean_deg"):
        for left, right in (row["pair"], tuple(reversed(row["pair"]))):
            friedel[left] = {
                "partner": right,
                "sum_deg": float(row["sum_deg"]),
                "deviation_from_360_deg": float(row["deviation_from_360_deg"]),
                "sum_q_norm_px": float(row["sum_q_norm_px"]),
            }
    triple = pp.triple_summary(
        records,
        fields,
        valid,
        key="phase",
        quantity="mean_deg",
        bins=args.bins,
        smooth_deg=args.smooth_deg,
    )
    by_name = {record["name"]: record for record in records}
    combo = list(triple["combo"])
    three = {
        "members": combo,
        "values_deg": [
            float(by_name[name]["stats"]["phase"]["mean_deg"]) for name in combo
        ],
        "sum_deg": float(triple["scalar_sum_deg"]),
        "mirror_sum_deg": float(triple["scalar_sum_mirror_deg"]),
        "q_sum_detected_px": q_sum_detected,
        "q_sum_used_px": float(
            np.hypot(
                sum(by_name[name]["q_px"][0] for name in combo),
                sum(by_name[name]["q_px"][1] for name in combo),
            )
        ),
        "projection_shift_px": [float(shift[0]), float(shift[1])],
    }
    theta = np.asarray(triple["theta_field"])
    amp_product = np.asarray(triple["amp_product"])
    sample_theta = pp.sample_mask(amp_product, valid)
    hist, edges = np.histogram(
        theta[sample_theta],
        bins=HISTOGRAM_BINS,
        range=(0.0, 2 * np.pi),
        weights=amp_product[sample_theta],
        density=True,
    )
    return {
        "ring": tag,
        "radius_px": float(np.mean([row["radius_px"] for row in peaks])),
        "n_peaks": len(peaks),
        "n_members": len(ring["members"]),
        "q_sum_detected_px": q_sum_detected,
        "q_position": hexagon_residual_px([record["q_px"] for record in records]),
        "theta_ramp_span_deg_if_unprojected": float(360.0 * q_sum_detected),
        "q_position_note": (
            "the detected peak positions carry a configuration dependent "
            "sub-pixel bias when a reflection is a mixture of regions "
            "(measured up to 0.911 px in the methods study); projecting "
            "the wavevector sum to zero removes the net ramp of theta "
            "but not that per-peak bias, so read |sum q| and the hexagon "
            "residual as the peak-position uncertainty"
        ),
        "project_q": bool(args.project_q),
        "projection_shift_px": [float(shift[0]), float(shift[1])],
        "peaks": peaks,
        "friedel": friedel,
        "three_independent": three,
        "triple_product": {
            "members": combo,
            "q_sum_px": three["q_sum_used_px"],
            "n_samples": int(triple["field_n_samples"]),
            "theta_mean_deg": float(triple["field"]["mean_deg"]),
            "theta_median_deg": float(triple["field"]["median_deg"]),
            "theta_R": float(triple["field"]["resultant_R"]),
            "theta_fwhm_deg": float(triple["field"]["fwhm_deg"]),
            "theta_fwhm_deconv_deg": float(triple["field"]["fwhm_deconv_deg"]),
            "theta_n_clusters": int(triple["field"]["n_clusters"]),
            "theta_circ_std_deg": float(triple["field"]["circ_std_deg"]),
            "theta_mirror_mean_deg": float(triple["field_mirror_mean_deg"]),
            "antipodal_members": [
                [-float(by_name[name]["q_px"][0]), -float(by_name[name]["q_px"][1])]
                for name in combo
            ],
        },
        # kept out of the JSON: large per-pixel objects
        "theta_hist": {"hist": hist, "edges": edges},
        "theta_field": theta,
        "amp_product": amp_product,
        "records": records,
        "fields": fields,
        "psi": psi,
        "detections": detections,
        "members": members,
        "valid": valid,
    }


# --------------------------------------------------------------------------- #
# pairwise phase-difference analysis (the three pairs inside one ring)
# --------------------------------------------------------------------------- #
def pairwise_group(analysis, json_group, pairs, args):
    """All pairs of one group, with the fields, the statistics and the JSON paths.

    ``json_group`` is the key of the group inside ``phase_stats.json``
    (``within_1x1`` / ``within_r3``), which is also the ``group`` recorded in the
    atlas manifest; ``pairs`` is a list of ``((ring_j, index_j), (ring_k, index_k))``
    whose position is the number used in the JSON paths.
    """
    entries = []
    for index, (left, right) in enumerate(pairs):
        tag_j, j = left
        tag_k, k = right
        record_j = analysis[tag_j]["records"][j]
        record_k = analysis[tag_k]["records"][k]
        psi_j = analysis[tag_j]["psi"][record_j["name"]]
        psi_k = analysis[tag_k]["psi"][record_k["name"]]
        result = pp.pair_analysis(
            psi_j,
            psi_k,
            analysis[tag_j]["valid"],
            analysis[tag_k]["valid"],
            bins=args.bins,
            smooth_deg=args.smooth_deg,
        )
        base = f"pairwise.groups.{json_group}.pairs.{index}"
        entries.append(
            {
                "index": index,
                "j": record_j["name"],
                "k": record_k["name"],
                "j_index": int(j),
                "k_index": int(k),
                "pair": {
                    "j": record_j["name"],
                    "k": record_k["name"],
                    "group": json_group,
                    "key": f"{record_j['name']}+{record_k['name']}",
                },
                "q_j": [float(record_j["q_px"][0]), float(record_j["q_px"][1])],
                "q_k": [float(record_k["q_px"][0]), float(record_k["q_px"][1])],
                "phase_diff_mean": result["phase_diff_mean_deg"],
                "phase_diff_median": result["phase_diff_median_deg"],
                "phase_diff_R": result["phase_diff_R"],
                "phase_diff_fwhm_deg": result["phase_diff_fwhm_deg"],
                "phase_diff_n_clusters": result["phase_diff_n_clusters"],
                "phase_diff_fwhm_deconv_deg": result["phase_diff_fwhm_deconv_deg"],
                "phase_diff_iqr_deg": result["phase_diff_iqr_deg"],
                "phase_diff_circ_std_deg": result["phase_diff_circ_std_deg"],
                "phase_diff_cluster_centres_deg": [
                    float(cluster["centre_deg"])
                    for cluster in result["phase_diff_clusters"]
                ],
                "amp_diff_median": result["amp_diff_median"],
                "amp_diff_fwhm": result["amp_diff_fwhm"],
                "n_valid": result["n_valid"],
                "paths": {
                    name: f"{base}.{name}"
                    for name in (
                        "j",
                        "k",
                        "phase_diff_mean",
                        "phase_diff_median",
                        "phase_diff_R",
                        "phase_diff_fwhm_deg",
                        "phase_diff_n_clusters",
                        "amp_diff_median",
                        "n_valid",
                    )
                },
                "figure_prefix": f"{json_group.replace('within_', 'ring_')}_pair_{j}_{k}",
                "field": {
                    "phase_diff": result["phase_diff"],
                    "amp_diff": result["amp_diff"],
                    "weight": result["weight"],
                    "mask": result["mask"],
                },
            }
        )
    return entries


def pairwise_analysis(analysis, args):
    """The pairs inside each ring: (p0,p1), (p2,p3), (p4,p5) of the clockwise order.

    These three pairs join reflections 60 degrees apart and deliberately avoid the
    Friedel pairs, whose difference field is a trivial function of one field.  Pairs
    that join one reflection of each ring are not analysed: the atlas of this
    version reports every quantity inside a ring.
    """
    within_1x1 = [
        (("ring_1x1", j), ("ring_1x1", k))
        for j, k in pp.within_ring_pairs(analysis["ring_1x1"]["records"])
    ]
    within_r3 = [
        (("ring_r3", j), ("ring_r3", k))
        for j, k in pp.within_ring_pairs(analysis["ring_r3"]["records"])
    ]
    groups = {}
    for json_group, pairs in (("within_1x1", within_1x1), ("within_r3", within_r3)):
        groups[json_group] = {
            "group": json_group,
            "figure_group": json_group.replace("within_", "ring_"),
            "n_pairs": len(pairs),
            "pairs": pairwise_group(analysis, json_group, pairs, args),
        }
    return groups


def ring_summary(analysis):
    """Headline scalars of one ring (atlas annotations and comparison table)."""
    medians = np.radians([row["phase_median_deg"] for row in analysis["peaks"]])
    mean, resultant, _ = pm.circ_mean(medians)
    fwhms = [row["phase_fwhm_deg"] for row in analysis["peaks"]]
    finite = [value for value in fwhms if np.isfinite(value)]
    return {
        "mean_deg": float(np.degrees(mean) % 360.0),
        "R": float(resultant),
        "n_clusters_max": int(max(row["n_clusters"] for row in analysis["peaks"])),
        "fwhm_mean_deg": float(np.mean(finite)) if finite else float("nan"),
    }


def friedel_triples(analysis):
    seen, out = set(), []
    for name, info in analysis["friedel"].items():
        key = tuple(sorted((name, info["partner"])))
        if key in seen:
            continue
        seen.add(key)
        out.append((key[0], key[1], info["sum_deg"]))
    return sorted(out)


# --------------------------------------------------------------------------- #
# atlas
# --------------------------------------------------------------------------- #
def weighted_histogram(values, weights, bins, span=(0.0, 2 * np.pi)):
    """Amplitude-weighted density histogram with a zero-total guard.

    ``numpy.histogram(..., density=True)`` divides by the total weight, so an empty
    sample would produce NaNs and a warning; a pair or a peak with no valid pixel
    is reported as an all-zero density instead.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    edges = np.linspace(float(span[0]), float(span[1]), int(bins) + 1)
    total = float(np.sum(weights))
    if values.size == 0 or total <= 0.0:
        return np.zeros(int(bins)), edges
    hist, _ = np.histogram(
        values,
        bins=int(bins),
        range=(float(span[0]), float(span[1])),
        weights=weights,
        density=True,
    )
    return hist, edges


def pair_d_histogram(phase_diff, mask, weight, bins=HISTOGRAM_BINS):
    """Amplitude-weighted circular histogram of ``D`` over ``(-pi, pi]``."""
    return weighted_histogram(
        np.asarray(phase_diff, dtype=float)[mask],
        np.asarray(weight, dtype=float)[mask],
        bins,
        span=(-np.pi, np.pi),
    )


def pair_annotation(entry, kind):
    """Annotated numbers and JSON paths of one pairwise figure."""
    values, paths = {}, {}
    for field in PAIR_FIGURE_FIELDS[kind]:
        key = PAIR_ANNOTATION[field]
        values[field] = entry[key]
        paths[field] = entry["paths"][key]
    return values, paths


def global_amplitudes(analysis):
    """The one symmetric-log scale of all twelve amplitude maps.

    ``vmax`` is the largest valid ``|psi(r)|`` of the twelve demodulated fields of
    the run (six reflections of each ring) and ``linthresh`` is the **pooled median
    of their positive valid amplitudes**: both are global numbers that two
    amplitude figures of the same run share, so the maps stay directly comparable.

    The linear region is deliberately a *global statistic* and not the minimum of
    the positive values: the minimum positive ``|psi|`` of a 2117 x 2117 canvas is
    several decades below the bulk of the data, which squeezed the whole map into
    the top ~15 % of the colour bar (measured t_med 0.87-0.96) and made the weak and
    the strong reflections look alike.  Centring the linear region on the pooled
    median keeps the same global ``vmin``/``vmax`` while spreading the bulk of the
    data over the middle of the bar.
    """
    maxima = []
    total = 0
    for tag in RING_TAGS:
        ring = analysis[tag]
        valid = ring["valid"]
        for record in ring["records"]:
            amp = np.asarray(ring["fields"][record["name"]][2], dtype=float)
            sample = pp.sample_mask(amp, valid)
            if not np.any(sample):
                continue
            values = amp[sample]
            maxima.append(float(values.max()))
            total += int(np.count_nonzero(values > 0.0))
    if not maxima:
        raise SystemExit("no valid demodulated amplitude: cannot set a colour scale")
    if total == 0:
        raise SystemExit("no positive demodulated amplitude: cannot set a colour scale")
    # one preallocated buffer instead of twelve copies plus a concatenation: the
    # pooled median is taken over every positive valid sample value of the run
    pooled = np.empty(total, dtype=float)
    offset = 0
    for tag in RING_TAGS:
        ring = analysis[tag]
        valid = ring["valid"]
        for record in ring["records"]:
            amp = np.asarray(ring["fields"][record["name"]][2], dtype=float)
            sample = pp.sample_mask(amp, valid)
            if not np.any(sample):
                continue
            positive = amp[sample]
            positive = positive[positive > 0.0]
            pooled[offset : offset + positive.size] = positive
            offset += positive.size
    linthresh = float(np.median(pooled, overwrite_input=True))
    del pooled
    return max(maxima), linthresh


def build_atlas(outdir, analysis, pairwise, cmap, args, r0, c0, size_nm, detector):
    """Render ``FIGURES_PER_RING`` figures per ring (60 in total) and register them.

    Per ring: 3 figures for each of the six reflections (amplitude, theta map, theta
    distribution), 3 ring summaries (theta histogram grid, theta map grid, q-space
    ring members) and 3 figures for each of the three pairs inside the ring
    (D(r), the D distribution, a(r)) -- 18 + 3 + 9 = 30, i.e. 60 for both rings.
    """
    book = at.Atlas(outdir, cmap, dpi=args.dpi)
    lines = []

    def emit(text=""):
        print(text)
        lines.append(text)

    amp_vmax, amp_linthresh = global_amplitudes(analysis)
    amps = at.amplitude_norm(amp_vmax, amp_linthresh)
    norms = {
        "amplitude": at.norm_record("amplitude", amps),
        "theta": at.norm_record("theta", at.theta_norm()),
        "phase_diff": at.norm_record("phase_diff", at.phase_diff_norm()),
        "amp_diff": at.norm_record("amp_diff", at.amp_diff_norm()),
        "fft2": at.norm_record("fft2", at.fft2_norm()),
    }
    emit(
        f"# atlas colour scales (global, one per quantity class): theta maps "
        f"[{at.THETA_VMIN:g}, {at.THETA_VMAX:.6f}] rad; the {len(RING_TAGS) * PEAKS_PER_RING} "
        f"amplitude maps share one symlog scale vmin 0, vmax {amp_vmax:.6e} (the largest "
        f"valid |psi| of the run), linthresh {amp_linthresh:.6e} (the pooled median of "
        f"the positive valid |psi| of the run), linscale {at.AMPLITUDE_LINSCALE:g}; "
        f"D maps [{-np.pi:.6f}, {np.pi:.6f}] rad; a maps [-1, 1]"
    )
    emit(
        "# atlas titles: every figure carries its manifest title on the canvas "
        "(wrapped, figure-level) and the writer refuses a figure without it"
    )
    emit(
        "# atlas colour bars: every map / density figure carries its own colour bar "
        "axes, disjoint from every data axes (asserted per figure before writing)"
    )

    def peak_values(row):
        return {
            "peak": row["index"],
            "qx_px": row["qx_px"],
            "qy_px": row["qy_px"],
            "radius_px": row["radius_px"],
            "snr": row["snr"],
            "pixels": row["n_samples"],
            "median_deg": row["phase_median_deg"],
            "fwhm_deg": row["phase_fwhm_deg"],
            "n_clusters": row["n_clusters"],
            "R": row["phase_R"],
            "amp_median": row["amp_median"],
        }

    def peak_paths(tag, row):
        base = f"rings_analysis.{tag}.peaks.{row['index']}"
        return {
            "peak": f"{base}.index",
            "qx_px": f"{base}.qx_px",
            "qy_px": f"{base}.qy_px",
            "radius_px": f"{base}.radius_px",
            "snr": f"{base}.snr",
            "pixels": f"{base}.n_samples",
            "median_deg": f"{base}.phase_median_deg",
            "fwhm_deg": f"{base}.phase_fwhm_deg",
            "n_clusters": f"{base}.n_clusters",
            "R": f"{base}.phase_R",
            "amp_median": f"{base}.amp_median",
        }

    def pair_figures(pairs, group, label_prefix):
        for entry in pairs:
            pair = entry["pair"]
            key = pair["key"]
            field = entry["field"]
            covers = entry["figure_prefix"]
            for kind in PAIR_FIGURES:
                values, paths = pair_annotation(entry, kind)
                name = f"{covers}_{kind}.png"
                label = f"{label_prefix} {pair['j']} {pair['k']} {kind}"
                if kind == "phase_diff_dist":
                    hist, edges = pair_d_histogram(
                        field["phase_diff"], field["mask"], field["weight"]
                    )
                    book.pair_phase_diff_dist(
                        hist, edges, label, values, paths, group, name, key
                    )
                else:
                    book.pair_field(
                        field[kind],
                        field["mask"],
                        label,
                        values,
                        paths,
                        group,
                        name,
                        key,
                        kind,
                    )

    for tag in RING_TAGS:
        ring = analysis[tag]
        peaks, records = ring["peaks"], ring["records"]
        valid, fields = ring["valid"], ring["fields"]
        ring_radius = ring["radius_px"]
        summary = ring_summary(ring)

        for row, record in zip(peaks, records, strict=True):
            number = row["index"]
            theta = np.asarray(fields[record["name"]][1])
            amp = np.asarray(fields[record["name"]][2])
            sample = pp.sample_mask(amp, valid)
            values, paths = peak_values(row), peak_paths(tag, row)
            hist, edges = weighted_histogram(theta[sample], amp[sample], HISTOGRAM_BINS)
            book.amplitude_map(
                amp,
                sample,
                amps,
                f"{tag} p{number} amplitude",
                values,
                paths,
                tag,
                f"{tag}_p{number}_amplitude.png",
                number,
            )
            book.theta_map(
                theta,
                sample,
                f"{tag} p{number} theta(r)",
                values,
                paths,
                tag,
                f"{tag}_p{number}_theta_map.png",
                number,
            )
            book.theta_distribution(
                hist,
                edges,
                f"{tag} p{number} theta distribution",
                values,
                paths,
                tag,
                f"{tag}_p{number}_theta_dist.png",
                number,
            )

        ring_values = {"radius_px": ring_radius, "n_peaks": ring["n_peaks"]}
        ring_paths = {
            "radius_px": f"rings_analysis.{tag}.radius_px",
            "n_peaks": f"rings_analysis.{tag}.n_peaks",
        }
        book.ring_members(
            ring["fft2"],
            [record["integer"] for record in records],
            f"{tag} ring members",
            ring_values,
            ring_paths,
            tag,
            f"{tag}_ring_members_qspace.png",
        )

        hist_items, map_items = [], []
        for row, record in zip(peaks, records, strict=True):
            theta = np.asarray(fields[record["name"]][1])
            amp = np.asarray(fields[record["name"]][2])
            sample = pp.sample_mask(amp, valid)
            hist, edges = weighted_histogram(theta[sample], amp[sample], HISTOGRAM_BINS)
            hist_items.append(
                {
                    "hist": hist,
                    "edges": edges,
                    "label": f"p{row['index']} (median "
                    f"{row['phase_median_deg']:.3f} deg)",
                }
            )
            map_items.append(
                {"theta": theta, "sample": sample, "label": f"p{row['index']} theta(r)"}
            )
        book.grid_theta_histograms(
            hist_items,
            f"{tag} theta histograms of the six reflections",
            {
                "radius_px": ring_radius,
                "n_peaks": ring["n_peaks"],
                "mean_deg": summary["mean_deg"],
                "R": summary["R"],
                "n_clusters": summary["n_clusters_max"],
            },
            {
                "radius_px": f"rings_analysis.{tag}.radius_px",
                "n_peaks": f"rings_analysis.{tag}.n_peaks",
                "mean_deg": f"rings_analysis.{tag}.summary.mean_deg",
                "R": f"rings_analysis.{tag}.summary.R",
                "n_clusters": f"rings_analysis.{tag}.summary.n_clusters_max",
            },
            tag,
            f"{tag}_theta_hist_summary.png",
        )
        book.grid_theta_maps(
            map_items,
            f"{tag} theta(r) maps of the six reflections",
            ring_values,
            ring_paths,
            tag,
            f"{tag}_theta_map_summary.png",
        )

        group_key = "within_1x1" if tag == "ring_1x1" else "within_r3"
        pair_figures(pairwise[group_key]["pairs"], tag, f"{tag} within-pair")
        emit(
            f"# atlas: {tag} -> {FIGURES_PER_RING} figures "
            f"({len(PER_PEAK_FIGURES)} per peak x {ring['n_peaks']} + "
            f"{len(SUMMARY_FIGURES)} summary + "
            f"{WITHIN_PAIRS_PER_RING} pairs x {len(PAIR_FIGURES)})"
        )

    emit(f"# atlas: {TOTAL_FIGURES} figures in total, no figure spans the two rings")

    manifest = book.manifest(
        extra={
            "field_of_view_nm": float(size_nm),
            "detector": detector,
            "engine": "local-q-map gaussian window",
            "lambda_nm": float(args.lambda_nm),
            "r0_px": [float(r0[0]), float(r0[1])],
            "c_deg": float(c0),
            "figures_per_ring": FIGURES_PER_RING,
            "figures_total": TOTAL_FIGURES,
            "within_pairs": "p0-p1, p2-p3, p4-p5 per ring",
            "sample": (
                "every valid (non-NaN) pixel of the corrected canvas, weight |psi(r)|; "
                "a pair sample is the intersection of the two validity masks with "
                "weight |psi_j psi_k|"
            ),
            "norms": norms,
        }
    )
    return manifest, lines


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv=None):
    args = parse_args(argv)
    csv_path = Path(args.input)
    outdir = Path(args.outdir) if args.outdir else csv_path.parent
    outdir.mkdir(parents=True, exist_ok=True)
    size_nm, fov_source = field_of_view(args)
    log_lines = []

    def emit(text=""):
        print(text)
        log_lines.append(text)

    topo = np.loadtxt(csv_path, delimiter=args.delimiter)
    if topo.ndim != 2 or topo.shape[0] != topo.shape[1]:
        raise SystemExit(f"{csv_path}: expected a square 2D matrix, got {topo.shape}")
    valid = np.isfinite(topo)
    n = int(topo.shape[0])
    nm_per_px = float(size_nm) / float(n)
    window = pp.window_px(args.lambda_nm, nm_per_px)

    package, package_detail = load_package(args.stm_lib)
    if args.detector == "package" and package is None:
        raise SystemExit(
            f"--detector package requested but the import failed: {package_detail}"
        )
    use_package = args.detector == "package" or (
        args.detector == "auto" and package is not None
    )

    if args.fft2:
        fft2 = np.load(args.fft2)
        fft_source = f"npy:{args.fft2}"
    elif use_package:
        fft2 = package[0](topo, size_nm, subtract_plane=False)
        fft_source = "package:bragg_peak.compute_fft2(subtract_plane=False)"
    else:
        fft2 = pp.compute_fft2(topo, "hann")
        fft_source = "builtin:phasepipe.compute_fft2(window=hann)"

    if use_package:
        detection = package[1](
            topo,
            size_nm,
            patch_half=args.patch_half,
            subtract_plane=False,
            return_fft2=False,
        )
        records = [
            (
                float(peak.q_px[0]),
                float(peak.q_px[1]),
                float(peak.amplitude),
                float(peak.snr),
                float(np.hypot(*peak.q_px)),
            )
            for peak in detection.peaks
        ]
        detector = "package:bragg_peak.detect_bragg_peaks"
        detector_detail = (
            f"{len(detection.peaks)} peaks, basis source "
            f"{detection.meta.get('basis_source')}"
        )
    else:
        records = pp.detect_reflections(np.abs(fft2))
        detector = "builtin:phasepipe.detect_reflections"
        detector_detail = f"{len(records)} peaks"

    emit(
        "# phase statistics of a corrected topography image (geometry and "
        "mathematics only; skill version " + SKILL_VERSION + ")"
    )
    emit(f"# input: {csv_path}")
    emit(fov_source)
    emit(
        f"# canvas: {n} x {n} px, field of view {size_nm} nm "
        f"({nm_per_px:.6f} nm/px), valid pixels {100 * float(np.mean(valid)):.2f} %"
    )
    emit(
        f"# FFT2: {fft_source} (reflection detection and the q-space figure; the "
        f"phase fields come from the engine below)"
    )
    emit(f"# detector: {detector} ({detector_detail})")
    emit(
        f"# engine: gaussian window via local-q-map (lambda = {args.lambda_nm:g} nm = "
        f"{window:.3f} px); sample: every valid (non-NaN) pixel weighted by |psi(r)|; "
        f"histogram {args.bins} bins with {args.smooth_deg:g} deg smoothing"
    )
    emit(
        "# phase definition: psi_q(r) = FFT^-1{ FFT[ T(r) exp(-i q.r) ] * "
        "exp(-lambda^2 |k|^2 / 2) }, theta_q(r) = arg(psi_q(r)) with NO q.(r - c) ramp "
        "and no per-reflection constant (theta ~ +phi of the reflection); the phase "
        "VALUE of a reflection is the amplitude-weighted circular mean over every valid "
        "pixel, equal to arg sum_r T(r) exp(-i q.r) exactly whatever the window width "
        "is, and the median / IQR / FWHM / clusters describe that same sample"
    )

    rings = pp.group_rings(records, tol_frac=args.ring_cluster_tol, min_members=6)
    emit("")
    emit("== detected rings (radius px, members, total |F|) ==")
    for ring in rings:
        emit(
            f"#   radius {ring['radius']:10.4f}  members {len(ring['members']):2d}  "
            f"|F| {ring['total_amplitude']:.4e}"
        )
    ring_rows = [
        {
            "radius_px": float(ring["radius"]),
            "n_members": len(ring["members"]),
            "total_amplitude": float(ring["total_amplitude"]),
            "member_radii_px": [
                float(m[4]) for m in sorted(ring["members"], key=lambda item: item[4])
            ],
            "max_snr": float(max(m[3] for m in ring["members"])),
        }
        for ring in rings
    ]
    write_csv(outdir / "ring_candidates.csv", ring_rows)

    choice = pp.choose_rings(
        rings,
        anchor=args.anchor,
        reference_radius_px=args.reference_radius_px,
        match_tol=args.match_tol,
        pair_tol=args.pair_tol,
        r3_tol=args.r3_tol,
        expect=PEAKS_PER_RING,
    )
    selection = {
        "status": choice["status"],
        "method": choice["method"],
        "anchor": args.anchor,
        "tolerances": {
            "match": args.match_tol,
            "pair": args.pair_tol,
            "r3": args.r3_tol,
            "ring_cluster": args.ring_cluster_tol,
        },
        "ring_1x1_radius_px": (
            float(choice["ring_1x1"]["radius"])
            if choice["ring_1x1"] is not None
            else None
        ),
        "ring_r3_radius_px": (
            float(choice["ring_r3"]["radius"])
            if choice["ring_r3"] is not None
            else None
        ),
        "ratio": float(choice["ratio"]) if np.isfinite(choice["ratio"]) else None,
        "sqrt3_deviation": (
            float(choice["sqrt3_deviation"])
            if np.isfinite(choice["sqrt3_deviation"])
            else None
        ),
        "sqrt3": float(pp.SQRT3),
    }
    emit("")
    if choice["status"] != "ok":
        emit(f"ring pair: NOT FOUND ({choice['status']})")
        emit(
            f"# reference ring selection: --anchor {args.anchor} -> {choice['method']}"
        )
        if choice["status"] == "no_rings":
            emit("# no ring with at least six members was detected on this image")
        emit(
            "# the r3 ring is NOT reported: no ring sits at 1/sqrt(3) of the "
            "reference radius inside the tolerance, and the analysis stops here "
            "instead of analysing an unpaired ring"
        )
        emit(
            "# options: raise --r3-tol, name the reference ring explicitly "
            "(--anchor radius --reference-radius-px), or fix the geometry first"
        )
        payload = {
            "skill": "phase-analysis",
            "skill_version": SKILL_VERSION,
            "status": choice["status"],
            "input": str(csv_path),
            "canvas_px": n,
            "field_of_view_nm": size_nm,
            "field_of_view_source": fov_source,
            "detector": detector,
            "rings": ring_rows,
            "ring_selection": selection,
        }
        (outdir / "phase_stats.json").write_text(dump_json(payload) + "\n")
        emit(
            f"# written: {outdir / 'phase_stats.json'}, "
            f"{outdir / 'ring_candidates.csv'}, {outdir / 'phase_stats.log'}"
        )
        (outdir / "phase_stats.log").write_text("\n".join(log_lines) + "\n")
        return 2

    ring_1x1, ring_r3 = choice["ring_1x1"], choice["ring_r3"]
    emit(
        f"ring pair: ring_1x1 = {ring_1x1['radius']:.4f} px, ring_r3 = "
        f"{ring_r3['radius']:.4f} px, ratio = {choice['ratio']:.6f} "
        f"(sqrt(3) = {pp.SQRT3:.6f}, deviation "
        f"{100 * choice['sqrt3_deviation']:.4f} %)"
    )
    emit(f"# reference ring selection: --anchor {args.anchor} -> {choice['method']}")
    emit(
        f"# r3 position: 1/sqrt(3) of the reference radius "
        f"({ring_1x1['radius'] / pp.SQRT3:.4f} px target, tolerance "
        f"{args.r3_tol:.1%})"
    )

    analysis = {}
    for tag, ring in (("ring_1x1", ring_1x1), ("ring_r3", ring_r3)):
        analysis[tag] = analyse_ring(
            tag, ring, topo, valid, args.lambda_nm, nm_per_px, args
        )
        analysis[tag]["fft2"] = fft2
        summary = ring_summary(analysis[tag])
        emit("")
        emit(
            f"== {tag}: {analysis[tag]['n_peaks']} reflections, mean radius "
            f"{analysis[tag]['radius_px']:.4f} px =="
        )
        emit(
            f"# |sum q| over the three independent reflections: detected "
            f"{analysis[tag]['q_sum_detected_px']:.4f} px, used "
            f"{analysis[tag]['three_independent']['q_sum_used_px']:.4f} px "
            f"({'projected to zero' if args.project_q else 'not projected'})"
        )
        position = analysis[tag]["q_position"]
        emit(
            f"# peak-position quality: hexagon residual rms "
            f"{position['hexagon_residual_rms_px']:.4f} px (max "
            f"{position['hexagon_residual_max_px']:.4f} px), radial rms "
            f"{position['radial_rms_px']:.4f} px about {position['mean_radius_px']:.4f} px; "
            f"without the projection the wavevector sum would ramp theta by "
            f"{analysis[tag]['theta_ramp_span_deg_if_unprojected']:.3f} deg across the "
            f"canvas"
        )
        for row in analysis[tag]["peaks"]:
            emit(
                f"   {row['name']}  q=({row['qx_px']:9.3f},{row['qy_px']:9.3f}) px  "
                f"snr={row['snr']:10.1f}  radius={row['radius_px']:9.3f} px"
            )
            emit(
                f"      phase (circular mean over {row['n_samples']} valid pixels) "
                f"{row['phase_mean_deg']:9.4f} deg  R {row['phase_R']:.5f}  | median "
                f"{row['phase_median_deg']:9.4f} deg"
            )
            emit(
                f"      distribution shape: median {row['phase_median_deg']:9.4f} deg, "
                f"fwhm {fmt(row['phase_fwhm_deg'])} deg (deconvolved "
                f"{fmt(row['phase_fwhm_deconv_deg'])}), iqr "
                f"{fmt(row['phase_iqr_deg'])} deg, clusters {row['n_clusters']} "
                f"at {[round(c, 3) for c in row['cluster_centres_deg']]} widths "
                f"{[None if not np.isfinite(w) else round(w, 3) for w in row['cluster_widths_deg']]}"
                f" weights {[round(w, 4) for w in row['cluster_weight_fractions']]}"
            )
            emit(
                f"      amplitude median {row['amp_median']:.4e}, fwhm "
                f"{row['amp_fwhm']:.4e}; pixels dropped for a zero weight "
                f"{row['n_zero_weight_px']} of {row['n_samples'] + row['n_zero_weight_px']}"
            )
        three = analysis[tag]["three_independent"]
        triple = analysis[tag]["triple_product"]
        emit(
            "   Friedel pair sums (identity of a real image): "
            + "; ".join(
                f"{left}+{right} = {value:.5f} deg"
                for left, right, value in friedel_triples(analysis[tag])
            )
        )
        emit(
            f"   three independent phases {three['members']}: values "
            f"{[round(v, 4) for v in three['values_deg']]}, sum "
            f"{three['sum_deg']:.4f} deg (mirror {three['mirror_sum_deg']:.4f} deg)"
        )
        emit(
            f"   per-pixel triple product theta over {triple['n_samples']} valid pixels: "
            f"mean {triple['theta_mean_deg']:.4f} deg (mirror "
            f"{triple['theta_mirror_mean_deg']:.4f}), median "
            f"{triple['theta_median_deg']:.4f} deg, R {triple['theta_R']:.5f}, "
            f"fwhm {fmt(triple['theta_fwhm_deg'])} deg, clusters "
            f"{triple['theta_n_clusters']}"
        )
        emit(
            f"   ring summary of the six phase medians: circular mean "
            f"{summary['mean_deg']:.4f} deg, R {summary['R']:.5f}, max clusters "
            f"{summary['n_clusters_max']}, mean fwhm {fmt(summary['fwhm_mean_deg'])} deg"
        )

    # ---- gauge fix: the six ring_1x1 reflections pin the lattice origin ---- #
    recs = analysis["ring_1x1"]["records"]
    qs = [record["q_px"] for record in recs]
    phases = [np.radians(record["stats"]["phase"]["mean_deg"]) for record in recs]
    best, minima = pm.fit_origin(qs, phases, n)
    r0, c0 = best["r0_px"], best["c_rad"]
    emit("")
    emit("== lattice-referenced gauge fix (the ring_1x1 reflections pin the origin) ==")
    emit(
        f"# least squares of phi_j = (2 pi / N) q_j . r0 + c over the six ring_1x1 "
        f"reflections: r0 = ({r0[0]:.4f}, {r0[1]:.4f}) px, c = {best['c_deg']:.4f} deg, "
        f"rms residual {best['rms_deg']:.4f} deg, {len(minima)} local minimum(a)"
    )
    emit(
        "# that rms is a MODEL-CONSISTENCY diagnostic, not phase noise: the six peaks "
        "are three Friedel pairs (+Phi / -Phi), so a single origin cannot represent "
        "them unless the phases are origin-consistent; the q-sum-zero triple fit below "
        "has rms 0 by construction"
    )
    reference_deviations = []
    for record in recs:
        fixed = (
            np.degrees(
                pm.gauge_phase(
                    np.radians(record["stats"]["phase"]["mean_deg"]),
                    record["q_px"],
                    r0,
                    c0,
                    n,
                )
            )
            % 360.0
        )
        reference_deviations.append(float(pm.dist_to_deg(fixed, 0.0)))
    ref_rms = float(np.sqrt(np.mean(np.square(reference_deviations))))
    emit(
        f"# the six gauge-fixed ring_1x1 phases deviate from 0 by "
        f"{[round(value, 3) for value in reference_deviations]} deg (rms "
        f"{ref_rms:.4f} deg); the ring_r3 phases inherit this reference uncertainty"
    )
    emit(
        "# the least-squares solution set is {best r0} plus every direct-lattice "
        "translation of the reference ring: such a translation adds an exact multiple "
        "of 2 pi to each of the six model phases, so the residual is bit-identical and "
        "the fit can never separate them (structural degeneracy, not a numerical error)"
    )
    band = ref_rms / 2.1213  # sigma_peak / 3 with sigma_peak = rms_residual / sqrt(0.5)
    emit(
        f"# systematic band of a single-peak absolute phase: sigma_peak ~ "
        f"{ref_rms:.4f}/0.707 = {ref_rms / 0.707:.4f} deg, so a reflection at 1/sqrt(3) "
        f"of the reference radius inherits sigma_peak/3 = {band:.4f} deg (systematic: "
        f"it does not shrink with more pixels)"
    )

    triple_index, _q_sum = pp.triple_selection([tuple(q) for q in qs])
    triple_fit, _triple_minima = pm.fit_origin(
        [tuple(qs[index]) for index in triple_index],
        [phases[index] for index in triple_index],
        n,
    )
    emit(
        f"# origin fit convention: SIX reflections (all three Friedel pairs) are "
        f"fitted against one common offset and one origin; residual rms "
        f"{best['rms_deg']:.4f} deg. The three reflections with q-sum zero alone "
        f"determine (r0, c) exactly, so their rms is {triple_fit['rms_deg']:.3e} deg "
        f"by construction (3 equations, 3 unknowns) and carries no diagnostic "
        f"information. The six-peak residual is the model-consistency number: it "
        f"grows when the six phases are not realisable by a single origin (an "
        f"uneven / heterogeneous reference ring, or a synthetic pattern whose "
        f"mirrors carry the opposite sign)"
    )

    a1, a2 = reference_lattice_vectors(qs)
    basis = np.column_stack([a1, a2])
    branch_table = []
    for index, minimum in enumerate(minima):
        delta = np.asarray(minimum["r0_px"], dtype=float) - np.asarray(r0, dtype=float)
        coefficients = np.linalg.solve(basis, delta)
        lattice = bool(
            abs(coefficients[0] - round(coefficients[0])) < 1e-6
            and abs(coefficients[1] - round(coefficients[1])) < 1e-6
        )
        shifts = [
            float(
                np.degrees(
                    np.mod(
                        (2.0 * np.pi / n) * float(np.dot(record["q_px"], delta)),
                        2.0 * np.pi,
                    )
                )
            )
            for record in analysis["ring_r3"]["records"]
        ]
        branch_table.append(
            {
                "branch": index,
                "r0_px": [float(minimum["r0_px"][0]), float(minimum["r0_px"][1])],
                "c_deg": float(minimum["c_deg"]),
                "rms_deg": float(minimum["rms_deg"]),
                "delta_r0_px": [float(delta[0]), float(delta[1])],
                "delta_r0_lattice_coefficients": [
                    float(coefficients[0]),
                    float(coefficients[1]),
                ],
                "lattice_translation_of_the_best_fit": lattice,
                "induced_ring_r3_phase_shifts_deg": shifts,
            }
        )
    lattice_branches = sum(
        1 for row in branch_table if row["lattice_translation_of_the_best_fit"]
    )
    emit(
        f"# branch table: {len(branch_table)} local minimum(a), {lattice_branches} of "
        f"them direct-lattice translations of the best fit (bit-identical residual); a "
        f"translation shifts the ring_r3 phases by (2 pi / N) q.L per reflection"
    )
    emit(
        "# the induced shifts are listed per branch in the JSON (gauge.branch_table): a "
        "point-lattice translation of r0 moves the absolute ring_r3 phases, so a "
        "single-peak absolute phase is comparable only after the branch is stated; the "
        "closing sums and the per-peak distribution shape are invariant"
    )

    for tag in ("ring_1x1", "ring_r3"):
        for row, record in zip(
            analysis[tag]["peaks"], analysis[tag]["records"], strict=True
        ):
            fixed = (
                np.degrees(
                    pm.gauge_phase(
                        np.radians(row["phase_mean_deg"]),
                        record["q_px"],
                        r0,
                        c0,
                        n,
                    )
                )
                % 360.0
            )
            row["gauge_mean_deg"] = float(fixed)
            row["friedel_partner"] = analysis[tag]["friedel"][row["name"]]["partner"]
            row["friedel_sum_deg"] = analysis[tag]["friedel"][row["name"]]["sum_deg"]
            row["friedel_deviation_deg"] = analysis[tag]["friedel"][row["name"]][
                "deviation_from_360_deg"
            ]

    # fix the JSON-only fields the atlas paths point at
    for tag in ("ring_1x1", "ring_r3"):
        ring = analysis[tag]
        ring["summary"] = ring_summary(ring)

    # ---- pairwise analysis ----------------------------------------------- #
    pairwise = pairwise_analysis(analysis, args)
    emit("")
    emit(
        "== pairwise phase differences (D = wrap(arg psi_j - arg psi_k), "
        "a = (|psi_j| - |psi_k|) / (|psi_j| + |psi_k|)) =="
    )
    emit(
        f"# groups: within_1x1 {pairwise['within_1x1']['n_pairs']} pairs, within_r3 "
        f"{pairwise['within_r3']['n_pairs']} pairs; sample = intersection of the two "
        f"validity masks, weight |psi_j psi_k|; D distribution = amplitude-weighted "
        f"circular histogram over (-pi, pi] with {HISTOGRAM_BINS} bins"
    )
    for group_name in ("within_1x1", "within_r3"):
        for entry in pairwise[group_name]["pairs"]:
            emit(
                f"   [{group_name}] D({entry['j']} - {entry['k']}): mean "
                f"{entry['phase_diff_mean']:9.4f} deg, median "
                f"{entry['phase_diff_median']:9.4f} deg, R {entry['phase_diff_R']:.5f}, "
                f"fwhm {fmt(entry['phase_diff_fwhm_deg'])} deg, clusters "
                f"{entry['phase_diff_n_clusters']}, a median "
                f"{entry['amp_diff_median']:+.6f}, valid pixels {entry['n_valid']}"
            )

    # ---- atlas ----------------------------------------------------------- #
    manifest = None
    cmap, cmap_source = at.load_colormap(args.stm_lib)
    emit(f"# colormap: {cmap_source}")
    if not args.no_figures:
        at.setup_style()
        manifest, atlas_lines = build_atlas(
            outdir, analysis, pairwise, cmap, args, r0, c0, size_nm, detector
        )
        log_lines.extend(atlas_lines)

    # ---- JSON ------------------------------------------------------------ #
    payload = {
        "skill": "phase-analysis",
        "skill_version": SKILL_VERSION,
        "status": "ok",
        "input": str(csv_path),
        "canvas_px": n,
        "field_of_view_nm": size_nm,
        "field_of_view_source": fov_source,
        "nm_per_px": size_nm / n,
        "valid_fraction": float(np.mean(valid)),
        "detector": detector,
        "detector_detail": detector_detail,
        "fft2_source": fft_source,
        "package_import": package_detail,
        "engine": "local-q-map gaussian window (theta = arg psi, no q.r ramp)",
        "lambda_nm": float(args.lambda_nm),
        "window_px": float(window),
        "sample": (
            "every valid (non-NaN) pixel of the corrected canvas, weight |psi(r)|; "
            "a pair sample is the intersection of the two validity masks, weight "
            "|psi_j psi_k|; there is no threshold anywhere"
        ),
        "bins": args.bins,
        "smooth_deg": args.smooth_deg,
        "patch_half": args.patch_half,
        "ring_cluster_tol": args.ring_cluster_tol,
        "rings": ring_rows,
        "ring_selection": selection,
        "gauge": {
            "reference_ring": "ring_1x1",
            "definition": "gauge_phase(phi, q) = phi - (2 pi / N) q.r0 - c (mod 2 pi)",
            "sign_convention_note": "the opposite sign convention differs by a constant "
            "rotation of every reported phase; the shapes and "
            "concentrations of the distributions are identical",
            "r0_px": [float(r0[0]), float(r0[1])],
            "c_deg": float(best["c_deg"]),
            "rms_deg": float(best["rms_deg"]),
            "rms_deg_note": (
                "model-consistency diagnostic of the six-peak fit, not "
                "phase noise: the six reflections are three Friedel pairs "
                "(+Phi / -Phi), so one origin represents them only if the "
                "six phases are origin-consistent; see triple_fit for the "
                "q-sum-zero variant whose rms is 0 by construction"
            ),
            "n_minima": len(minima),
            "reference_deviations_deg": reference_deviations,
            "reference_rms_deg": ref_rms,
            "fit_convention": "six reflections (three Friedel pairs) vs one common "
            "offset and one origin; the q-sum-zero triple determines "
            "(r0, c) exactly and is reported for comparison",
            "triple_fit": {
                "members": [int(index) for index in triple_index],
                "rms_deg": float(triple_fit["rms_deg"]),
                "r0_px": [float(triple_fit["r0_px"][0]), float(triple_fit["r0_px"][1])],
                "c_deg": float(triple_fit["c_deg"]),
                "delta_r0_vs_six_peak_px": [
                    float(triple_fit["r0_px"][0] - r0[0]),
                    float(triple_fit["r0_px"][1] - r0[1]),
                ],
            },
            "sigma_peak_deg": float(ref_rms / 0.707),
            "single_peak_systematic_band_deg": float(ref_rms / 2.1213),
            "solution_set": (
                "{best r0} plus every direct-lattice translation of the "
                "reference ring; a translation adds an exact multiple of "
                "2 pi to the six model phases, so the residual is "
                "bit-identical and the fit cannot separate them"
            ),
            "lattice_vectors": [
                {
                    "a1": [
                        float(reference_lattice_vectors(qs)[0][0]),
                        float(reference_lattice_vectors(qs)[0][1]),
                    ],
                    "a2": [
                        float(reference_lattice_vectors(qs)[1][0]),
                        float(reference_lattice_vectors(qs)[1][1]),
                    ],
                }
            ],
            "lattice_branches": int(lattice_branches),
            "branch_table": branch_table,
        },
        "rings_analysis": {},
        "pairwise": {
            "definition": {
                "phase_diff_field": "D_jk(r) = wrap(arg psi_j(r) - arg psi_k(r)) = "
                "arg(psi_j conj(psi_k)), in (-pi, pi]",
                "amp_diff_field": "a_jk(r) = (|psi_j(r)| - |psi_k(r)|) / "
                "(|psi_j(r)| + |psi_k(r)|), in [-1, 1]",
                "sample": "intersection of the two validity masks, restricted to the "
                "pixels where |psi_j psi_k| > 0",
                "weight": "|psi_j(r) psi_k(r)|",
                "d_distribution": "amplitude-weighted circular histogram of D itself "
                "over (-pi, pi] (the histogram drawn by RING_pair_<j>_<k>_"
                "phase_diff_dist.png)",
                "within_pairs": "p0-p1, p2-p3, p4-p5 of the clockwise peak numbering "
                "(60 deg apart; the Friedel pairs p0-p3, p1-p4, p2-p5 "
                "are avoided because psi_{-q} = conj(psi_q) makes their "
                "difference field a trivial function of one field)",
            },
            "bins": {
                "d": int(HISTOGRAM_BINS),
                "d_range": [-float(np.pi), float(np.pi)],
            },
            "groups": {
                name: {
                    "group": group["group"],
                    "figure_group": group["figure_group"],
                    "n_pairs": group["n_pairs"],
                    # the per-pixel fields stay out of the JSON (as for the rings)
                    "pairs": [
                        {key: value for key, value in entry.items() if key != "field"}
                        for entry in group["pairs"]
                    ],
                }
                for name, group in pairwise.items()
            },
        },
        "atlas": {
            "dir": str(outdir),
            "figures": 0,
            "per_ring": {},
            "manifest": "atlas_manifest.json",
            "declared_per_ring": {
                "per_peak": len(PER_PEAK_FIGURES),
                "peaks": PEAKS_PER_RING,
                "summary": len(SUMMARY_FIGURES),
                "pairwise": WITHIN_PAIRS_PER_RING * len(PAIR_FIGURES),
                "total": FIGURES_PER_RING,
            },
            "declared_total": TOTAL_FIGURES,
        },
        "conventions": {
            "peak_order": "p0 is the reflection closest to +qy (12 o'clock), the "
            "following ones run clockwise in the (q_x, q_y) plane",
            "sample": "every valid (non-NaN) pixel of the corrected canvas, weight "
            "|psi(r)|; a pair sample is the intersection of the two validity "
            "masks with weight |psi_j psi_k|; the pixels excluded are exactly "
            "those with a non-finite or zero weight",
            "phase_value": "amplitude-weighted circular mean over the sample = "
            "arg sum_r psi_q(r) = arg sum_r T(r) "
            "exp(-i q.r) (exact for every window width)",
            "phase_shape": "median/IQR/FWHM/clusters of the same sample "
            f"(bins {args.bins}, smoothing "
            f"{args.smooth_deg} deg)",
            "engine": "gaussian window demodulation from the sibling local-q-map skill: "
            "psi_q(r) = FFT^-1{ FFT[ T(r) exp(-i q.r) ] * "
            "exp(-lambda^2 |k|^2 / 2) }, lambda = "
            f"{args.lambda_nm:g} nm = {window:.4f} px",
            "theta": "theta_q(r) = arg psi_q(r) ~ +phi_q(r): the demodulated phase "
            "carries no q.(r - c) ramp and no per-reflection constant",
            "identities": [
                "sum_r psi_q(r) = sum_r T(r) exp(-i q.r)   (window independent)",
                "psi_{-q}(r) = conj(psi_q(r))              (T real; Friedel)",
                "psi_q(r - delta) = exp(-i q.delta) psi_q(r)   (translation)",
            ],
            "change_from_v2": "the raw phase of a single reflection is shifted by "
            "-pi (q_x + q_y) with respect to the v2 carrier "
            "convention; that constant is absorbed by r0 -> "
            "r0 + (N/2, N/2) in the origin fit, so every "
            "gauge-fixed phase, closing sum, distribution shape and "
            "pairwise difference is unchanged",
            "fwhm_raw": "FWHM of the smoothed amplitude-weighted histogram",
            "fwhm_deconv": "Gaussian-equivalent deconvolution of the smoothing kernel",
            "triangle_independent": "three reflections whose wavevectors sum to zero; "
            "the other triple is their antipodal mirror",
            "triple_product": "sum of the three independent per-pixel phase fields "
            f"(wavevector sum projected to zero: {args.project_q}); the "
            "JSON keeps its mean/median/R/FWHM/cluster count, no figure "
            "is drawn from it",
            "pairwise": "D_jk = wrap(arg psi_j - arg psi_k), a_jk = (|psi_j| - |psi_k|) "
            "/ (|psi_j| + |psi_k|); the pair figures are the D(r) map, the "
            "amplitude-weighted circular histogram of D over (-pi, pi] and "
            "the a(r) map on the pair sample",
            "colour_scales": "global: every theta(r) map spans [0, 2 pi]; the twelve "
            "amplitude maps share one symmetric-log scale (one vmax, one "
            "linthresh); every D(r) map spans [-pi, pi]; every a(r) map "
            "spans [-1, 1]; the records are in atlas.norms and are "
            "compared with the manifest by atlas.py --check",
            "colour_bars": "every map / density figure carries a colour bar in its own "
            "axes, disjoint from every data axes (asserted while writing "
            "and re-audited by atlas.py --check)",
            "gauge": "theta~ = theta - (2 pi / N) q.r0 - c with (r0, c) from the six "
            "ring_1x1 reflections; the solution set also contains every "
            "direct-lattice translation of r0 (identical residual), which "
            "shifts the ring_r3 phases by (2 pi / N) q.L per reflection, so "
            "ring_r3 absolute phases are comparable only after the r0 "
            "branch is stated",
            "invariant_under_the_r0_choice": "Friedel pair sums, the three-independent "
            "sum and the per-pixel triple product, the "
            "per-peak distribution shape (R, FWHM, "
            "cluster count and widths) and the pairwise fields",
            "single_peak_absolute_phase": "reference only: state which branch of r0 was "
            "used and attach the systematic band "
            "sigma_peak/3 derived from the reference-ring "
            "residual rms",
        },
    }
    for tag in ("ring_1x1", "ring_r3"):
        ring = analysis[tag]
        exported = {
            key: value
            for key, value in ring.items()
            if key
            not in (
                "records",
                "fields",
                "psi",
                "detections",
                "members",
                "valid",
                "fft2",
                "theta_hist",
                "theta_field",
                "amp_product",
            )
        }
        exported["theta_hist"] = {
            "bin_edges_deg": [
                float(np.degrees(value)) for value in ring["theta_hist"]["edges"]
            ],
            "density": [float(value) for value in ring["theta_hist"]["hist"]],
        }
        payload["rings_analysis"][tag] = exported
    if manifest is not None:
        per_group = manifest["per_group"]
        payload["atlas"].update(
            {
                "figures": int(manifest["total_figures"]),
                "per_group": {group: dict(info) for group, info in per_group.items()},
                "per_ring": {
                    tag: int(per_group.get(tag, {}).get("figures", 0))
                    for tag in RING_TAGS
                },
                "norms": {
                    name: dict(record) for name, record in manifest["norms"].items()
                },
            }
        )
        (outdir / "atlas_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
    (outdir / "phase_stats.json").write_text(dump_json(payload) + "\n")

    # ---- CSV tables ------------------------------------------------------ #
    write_csv(
        outdir / "phase_stats.csv",
        [row for tag in ("ring_1x1", "ring_r3") for row in analysis[tag]["peaks"]],
    )
    relation_rows = []
    for tag in ("ring_1x1", "ring_r3"):
        for left, right, value in friedel_triples(analysis[tag]):
            info = analysis[tag]["friedel"][left]
            relation_rows.append(
                {
                    "ring": tag,
                    "relation": "friedel_pair_sum",
                    "subject": left,
                    "partner": right,
                    "value_deg": value,
                    "deviation_deg": info["deviation_from_360_deg"],
                    "note": f"identity of a real image; |sum q| = "
                    f"{info['sum_q_norm_px']:.3e} px",
                }
            )
        three = analysis[tag]["three_independent"]
        triple = analysis[tag]["triple_product"]
        relation_rows.append(
            {
                "ring": tag,
                "relation": "three_independent_sum",
                "subject": "+".join(three["members"]),
                "partner": "",
                "value_deg": three["sum_deg"],
                "deviation_deg": "",
                "note": f"= 3 Phi_bar (mod 360); mirror sum "
                f"{three['mirror_sum_deg']:.4f} deg; |sum q| used "
                f"{three['q_sum_used_px']:.4f} px",
            }
        )
        relation_rows.append(
            {
                "ring": tag,
                "relation": "per_pixel_triple_product",
                "subject": "+".join(triple["members"]),
                "partner": "",
                "value_deg": triple["theta_mean_deg"],
                "deviation_deg": "",
                "note": f"theta(r) = 3 Phi_d inside a region; R = {triple['theta_R']:.5f}, "
                f"fwhm {triple['theta_fwhm_deg']:.4f} deg, mirror "
                f"{triple['theta_mirror_mean_deg']:.4f} deg",
            }
        )
    write_csv(outdir / "phase_relations.csv", relation_rows)

    comparison = []
    for tag in ("ring_1x1", "ring_r3"):
        summary = analysis[tag]["summary"]
        three = analysis[tag]["three_independent"]
        triple = analysis[tag]["triple_product"]
        comparison.append(
            {
                "ring": tag,
                "radius_px": analysis[tag]["radius_px"],
                "n_peaks": analysis[tag]["n_peaks"],
                "circular_mean_of_medians_deg": summary["mean_deg"],
                "n_clusters_max": summary["n_clusters_max"],
                "mean_fwhm_deg": summary["fwhm_mean_deg"],
                "friedel_max_deviation_deg": max(
                    info["deviation_from_360_deg"]
                    for info in analysis[tag]["friedel"].values()
                ),
                "three_sum_deg": three["sum_deg"],
                "theta_mean_deg": triple["theta_mean_deg"],
                "theta_median_deg": triple["theta_median_deg"],
                "theta_R": triple["theta_R"],
                "theta_fwhm_deg": triple["theta_fwhm_deg"],
                "theta_n_clusters": triple["theta_n_clusters"],
                "q_sum_used_px": triple["q_sum_px"],
            }
        )
    write_csv(outdir / "phase_ring_comparison.csv", comparison)

    emit("")
    emit("== reading guide ==")
    emit(
        "# phase value       -> amplitude-weighted circular mean over every valid "
        "pixel = arg sum_r T(r) exp(-i q.r); it is window independent and carries no ramp"
    )
    emit(
        "# pairwise D / a    -> D_jk = wrap(arg psi_j - arg psi_k), a_jk = "
        "(|psi_j| - |psi_k|)/(|psi_j| + |psi_k|) on the intersection of the two "
        "validity masks; the D distribution is the amplitude-weighted circular "
        "histogram of D over (-pi, pi]; the within-ring pairs avoid the Friedel pairs, "
        "whose difference field is a trivial function of one field"
    )
    emit(
        "# median/IQR/FWHM   -> shape of the same sample the maps are drawn from; the "
        "sample is every valid pixel, so no condition has to be quoted with them"
    )
    emit("# cluster count     -> how many distinct phases are present in the sample")
    emit(
        "# amplitude         -> whether a reflection is still there (a balanced mixture "
        "can cancel it); the amplitude maps of a run share one colour scale"
    )
    emit(
        "# three-phase sum   -> the weighted circular mean of the phases of the three "
        "independent reflections; a value-range statement, not a count of regions"
    )
    emit(
        "# Friedel pair sum  -> identity of a real image; a non-zero value means an "
        "implementation error, not a sample property"
    )
    emit(
        "# gauge fix         -> the ring_1x1 reflections pin r0; the solution set also "
        "contains every direct-lattice translation of r0 (identical residual)"
    )
    emit(
        "# robust quantities -> Friedel pair sums, the three-independent sum, the "
        "per-pixel triple product, the per-peak distribution shape (R/FWHM/clusters/"
        "widths) and the pairwise D/a fields: these are the numbers to compare across "
        "conventions"
    )
    emit(
        "# reference only    -> a single-peak absolute phase changes with the r0 branch "
        "(by (2 pi / N) q.L); quote it together with the branch and the systematic band "
        "sigma_peak/3 printed above"
    )
    emit(
        "# mirror ambiguity  -> the other closing triple is antipodal, so a signed theta "
        "mirrors around 0 (the mirror mean is printed next to every theta value)"
    )
    emit(
        "# two rings, one definition: every quantity above is computed with the same "
        "estimator and the same sample rule for ring_1x1 and ring_r3"
    )
    emit("")
    emit(
        f"# written: {outdir / 'phase_stats.json'}, {outdir / 'phase_stats.csv'}, "
        f"{outdir / 'phase_relations.csv'}, {outdir / 'phase_ring_comparison.csv'}, "
        f"{outdir / 'ring_candidates.csv'}, {outdir / 'phase_stats.log'}"
        + (
            f", {outdir / 'atlas_manifest.json'} and "
            f"{payload['atlas']['figures']} atlas figures"
            if manifest
            else ""
        )
    )
    (outdir / "phase_stats.log").write_text("\n".join(log_lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
