"""Geometry and phase statistics of a corrected topography image (two rings).

For the six reflections of two rings and with exactly the same definitions for
both, the script measures the demodulated phase field

    psi(r)  = ifft2(ifftshift(mask * FFT2))          (one reflection per mask)
    phi(r)  = angle(psi(r)) - (2*pi/N) q.(r - c),    c = N // 2

together with the lattice-referenced gauge fix that makes the two rings
comparable, and writes the complete figure atlas plus one JSON that carries every
number (a drawn number is the JSON number by construction; see ``atlas.py``).

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
        [--detector auto|package|builtin] [--pct 5] [--gate p50] [--no-figures]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import atlas as at  # noqa: E402
import phasepipe as pp  # noqa: E402
import phasemath as pm  # noqa: E402

SKILL_VERSION = "2.0"
LADDER_DEG = (0.0, 120.0, 240.0)
PER_PEAK_FIGURES = ("mask_in", "mask_out", "phi_dist")
SUMMARY_FIGURES = ("qspace_mask", "mask_all_in", "mask_all_out",
                   "case_phase_histograms", "phi_hist_summary", "phi_map_summary",
                   "theta_field")
PEAKS_PER_RING = 6
FIGURES_PER_RING = PEAKS_PER_RING * len(PER_PEAK_FIGURES) + len(SUMMARY_FIGURES)


# --------------------------------------------------------------------------- #
# command line and small helpers
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input", help="corrected topography CSV (square, may contain NaN)")
    parser.add_argument("-o", "--outdir", default=None,
                        help="output directory (default: next to the input CSV)")
    parser.add_argument("-L", "--size-nm", type=float, default=None,
                        help="field of view of the corrected image in nm")
    parser.add_argument("--size-nm-from-log", default=None,
                        help="read the field of view from a correction log "
                             "(the line 'field of view <value> nm')")
    parser.add_argument("--fft2", default=None,
                        help="complex FFT2 npy; default: computed from the CSV with the "
                             "same FFT routine as the detector path")
    parser.add_argument("--delimiter", default=",",
                        help="delimiter of the corrected CSV (default ',')")
    parser.add_argument("--detector", default="auto",
                        choices=("auto", "package", "builtin"),
                        help="reflection detector: the package bragg_peak detection "
                             "(default when importable) or the builtin one")
    parser.add_argument("--stm-lib", default="/Users/hunfen/Documents/GitHub/"
                                             "STM_DataProcessing/src",
                        help="STM_DataProcessing src directory (package detection and "
                             "the gwyddion colormap)")
    parser.add_argument("--pct", type=float, default=5.0,
                        help="single-reflection mask radius in percent of the image "
                             "side (default 5)")
    parser.add_argument("--gate", default="p50",
                        help="amplitude gate of the phase histogram: pNN percentile, "
                             "'none' for every valid pixel (default p50)")
    parser.add_argument("--anchor", default="auto",
                        choices=("auto", "strongest", "outer", "inner", "radius"),
                        help="how the reference ring (ring_1x1) is chosen (default auto)")
    parser.add_argument("--reference-radius-px", type=float, default=None,
                        help="reference ring radius in FFT pixels (--anchor radius)")
    parser.add_argument("--match-tol", type=float, default=0.03,
                        help="relative tolerance of the explicit reference radius")
    parser.add_argument("--pair-tol", type=float, default=0.03,
                        help="relative tolerance of the sqrt(3) radius ratio used to "
                             "form a ring pair")
    parser.add_argument("--r3-tol", type=float, default=0.03,
                        help="relative tolerance of the 1/sqrt(3) lookup that locates "
                             "the r3 ring")
    parser.add_argument("--ring-cluster-tol", type=float, default=0.02,
                        help="relative radius tolerance of the ring clustering")
    parser.add_argument("--project-q", dest="project_q", action="store_true",
                        default=True,
                        help="shift the three independent wavevectors so that their sum "
                             "is exactly zero before the triple product (default on)")
    parser.add_argument("--no-project-q", dest="project_q", action="store_false",
                        help="keep the detected wavevectors (a non-zero sum turns theta "
                             "into a canvas-wide ramp)")
    parser.add_argument("--bins", type=int, default=3600,
                        help="histogram bins over 2 pi (default 3600 = 0.1 deg)")
    parser.add_argument("--smooth-deg", type=float, default=2.0,
                        help="circular Gaussian smoothing of the histogram used for the "
                             "FWHM (default 2 deg)")
    parser.add_argument("--patch-half", type=int, default=8,
                        help="sub-pixel localizer patch half width of the package "
                             "detector (default 8)")
    parser.add_argument("--dpi", type=int, default=300, help="figure resolution")
    parser.add_argument("--no-figures", action="store_true",
                        help="statistics only: write no atlas figure")
    return parser.parse_args(argv)


def field_of_view(args):
    if args.size_nm is not None:
        return float(args.size_nm)
    if args.size_nm_from_log:
        text = Path(args.size_nm_from_log).read_text()
        match = re.search(r"field of view ([\d.]+) nm", text)
        if match:
            return float(match.group(1))
        raise SystemExit(f"no 'field of view <value> nm' line in {args.size_nm_from_log}")
    raise SystemExit("a field of view is required: pass -L <nm> or --size-nm-from-log")


def load_package(stm_lib):
    """Import the package detection/FFT, or report why it is unavailable."""
    if stm_lib:
        sys.path.insert(0, str(stm_lib))
    try:
        from stm_data_processing.utils.bragg_peak import compute_fft2, detect_bragg_peaks
    except Exception as exc:  # noqa: BLE001 - any import failure -> builtin detector
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


def fmt(value, decimals=3):
    if value is None:
        return "n/a"
    if isinstance(value, float) and not np.isfinite(value):
        return "nan"
    return f"{value:.{decimals}f}"


def fold_to_120(phi, weights):
    """Statistics of the distribution folded onto the 120 degree interval.

    ``phi mod 120 deg`` is mapped back onto the full circle (times 3) so the
    ordinary circular estimators apply unchanged; this distribution is invariant
    under a global 120 degree relabelling of the phases.
    """
    folded = np.mod(np.degrees(phi), 120.0)
    return pm.weighted_stats(np.radians(3.0 * folded), weights, bins=360, smooth_deg=2.0)


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
        widths.append(float(np.degrees(np.sqrt(-2.0 * np.log(resultant)))
                            if 0.0 < resultant < 1.0 else 0.0))
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
    ideal = np.array([[mean_radius * np.cos(rotation + k * np.pi / 3.0),
                       mean_radius * np.sin(rotation + k * np.pi / 3.0)]
                      for k in range(6)])
    distances = [float(np.min(np.hypot(ideal[:, 0] - qx, ideal[:, 1] - qy)))
                 for qx, qy in vectors]
    return {"mean_radius_px": mean_radius,
            "radial_rms_px": float(np.sqrt(np.mean((radii - mean_radius) ** 2))),
            "hexagon_residual_rms_px": float(np.sqrt(np.mean(np.square(distances)))),
            "hexagon_residual_max_px": float(np.max(distances))}


def project_members(members):
    """Shift the three independent wavevectors so that their sum is exactly zero."""
    combo, _ = pp.triple_selection([(member[0], member[1]) for member in members])
    shift = np.array([members[index][:2] for index in combo], dtype=float).sum(axis=0) / 3.0
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
def analyse_ring(tag, ring, fft2, valid, mask_radius, args):
    detections = pp.order_ring_members(ring["members"], expect=PEAKS_PER_RING)
    q_sum_detected = float(np.hypot(sum(m[0] for m in detections),
                                    sum(m[1] for m in detections)))
    if args.project_q:
        members, shift = project_members(detections)
    else:
        members, shift = detections, np.zeros(2)
    records, fields = pp.analyse_ring(fft2, valid, ring, mask_radius, gate=args.gate,
                                      bins=args.bins, smooth_deg=args.smooth_deg,
                                      prefix=tag + "_", peaks_override=members)
    psi = {}
    for record in records:
        mask = pp.circle_mask(fft2.shape, [record["integer"]], mask_radius)
        psi[record["name"]] = {
            "in": pp.complex_ifft(np.where(mask, fft2, 0.0)),
            "out": pp.complex_ifft(np.where(~mask, fft2, 0.0)),
            "mask": mask,
        }

    peaks = []
    for index, (record, detected) in enumerate(zip(records, detections)):
        name = record["name"]
        stats = record["stats"]
        ungated, gated = stats["phase_ungated"], stats["phase_gated"]
        phi, amp = np.asarray(fields[name][1]), np.asarray(fields[name][2])
        folded = fold_to_120(phi[valid], amp[valid])
        peaks.append({
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
            "phase_ungated_mean_deg": float(ungated["mean_deg"]),
            "phase_ungated_median_deg": float(ungated["median_deg"]),
            "phase_ungated_R": float(ungated["resultant_R"]),
            "phase_ungated_circ_std_deg": float(ungated["circ_std_deg"]),
            "phase_gated_mean_deg": float(gated["mean_deg"]),
            "phase_gated_median_deg": float(gated["median_deg"]),
            "phase_gated_R": float(gated["resultant_R"]),
            "phase_gated_fwhm_deg": float(gated["fwhm_deg"]),
            "phase_gated_fwhm_deconv_deg": float(gated["fwhm_deconv_deg"]),
            "phase_gated_iqr_deg": float(gated["iqr_deg"]),
            "phase_gated_median_span_deg": float(gated["median_span_deg"]),
            "n_clusters": int(gated["n_clusters"]),
            "cluster_centres_deg": [float(c["centre_deg"]) for c in gated["clusters"]],
            "cluster_weight_fractions": [float(c["weight_fraction"])
                                         for c in gated["clusters"]],
            "cluster_widths_deg": cluster_widths(phi[valid], amp[valid],
                                                 gated["clusters"]),
            "amp_median": float(stats["amp_median"]),
            "amp_fwhm": float(stats["amp_fwhm"]),
            "gate_fraction": float(stats["gate_fraction"]),
            "folded120_median_deg": float(folded["median_deg"]),
            "folded120_mean_deg": float(folded["mean_deg"]),
            "folded120_R": float(folded["resultant_R"]),
            "folded120_fwhm_deg": float(folded["fwhm_deg"]),
            "folded120_n_clusters": int(folded["n_clusters"]),
        })

    friedel = {}
    for row in pp.pair_summary(records, "phase_ungated", "mean_deg"):
        for left, right in (row["pair"], tuple(reversed(row["pair"]))):
            friedel[left] = {"partner": right,
                             "sum_deg": float(row["sum_deg"]),
                             "deviation_from_360_deg": float(row["deviation_from_360_deg"]),
                             "sum_q_norm_px": float(row["sum_q_norm_px"])}
    triple = pp.triple_summary(records, fields, valid, key="phase_ungated",
                               quantity="mean_deg", bins=args.bins,
                               smooth_deg=args.smooth_deg, gate=args.gate)
    by_name = {record["name"]: record for record in records}
    combo = list(triple["combo"])
    three = {
        "members": combo,
        "values_deg": [float(by_name[name]["stats"]["phase_ungated"]["mean_deg"])
                       for name in combo],
        "values_mod120_deg": [float(by_name[name]["stats"]["phase_ungated"]["mean_deg"]
                                    % 120.0) for name in combo],
        "sum_deg": float(triple["scalar_sum_deg"]),
        "sum_mod120_deg": float(triple["scalar_sum_mod120_deg"]),
        "mirror_sum_deg": float(triple["scalar_sum_mirror_deg"]),
        "ladder_distance_deg": float(triple["scalar_ladder_dist_deg"]),
        "q_sum_detected_px": q_sum_detected,
        "q_sum_used_px": float(np.hypot(sum(by_name[name]["q_px"][0] for name in combo),
                                        sum(by_name[name]["q_px"][1] for name in combo))),
        "projection_shift_px": [float(shift[0]), float(shift[1])],
    }
    theta = np.asarray(triple["theta_field"])
    amp_product = np.asarray(triple["amp_product"])
    good_theta = pp.gate_mask(amp_product, valid, args.gate)
    hist, edges = np.histogram(theta[good_theta], bins=361, range=(0.0, 2 * np.pi),
                               weights=amp_product[good_theta], density=True)
    return {
        "ring": tag,
        "radius_px": float(np.mean([row["radius_px"] for row in peaks])),
        "n_peaks": len(peaks),
        "n_members": len(ring["members"]),
        "mask_radius_px": int(mask_radius),
        "q_sum_detected_px": q_sum_detected,
        "q_position": hexagon_residual_px([record["q_px"] for record in records]),
        "theta_ramp_span_deg_if_unprojected": float(360.0 * q_sum_detected),
        "q_position_note": ("the detected peak positions carry a configuration dependent "
                            "sub-pixel bias when a reflection is a mixture of regions "
                            "(measured up to 0.911 px in the methods study); projecting "
                            "the wavevector sum to zero removes the net ramp of theta "
                            "but not that per-peak bias, so read |sum q| and the hexagon "
                            "residual as the peak-position uncertainty"),
        "project_q": bool(args.project_q),
        "projection_shift_px": [float(shift[0]), float(shift[1])],
        "peaks": peaks,
        "friedel": friedel,
        "three_independent": three,
        "triple_product": {
            "members": combo,
            "q_sum_px": three["q_sum_used_px"],
            "theta_mean_deg": float(triple["field_gated"]["mean_deg"]),
            "theta_median_deg": float(triple["field_gated"]["median_deg"]),
            "theta_R": float(triple["field_gated"]["resultant_R"]),
            "theta_fwhm_deg": float(triple["field_gated"]["fwhm_deg"]),
            "theta_n_clusters": int(triple["field_gated"]["n_clusters"]),
            "theta_ladder_distance_deg": float(triple["field_ladder_dist_deg"]),
            "theta_mirror_mean_deg": float(triple["field_mirror_mean_deg"]),
            "theta_mean_3x_mod360_deg": float(triple["field_mean_3x_mod360_deg"]),
            "antipodal_members": [[-float(by_name[name]["q_px"][0]),
                                   -float(by_name[name]["q_px"][1])] for name in combo],
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
        "fft2": fft2,
    }


def ring_summary(analysis):
    """Headline scalars of one ring (atlas annotations and comparison table)."""
    medians = np.radians([row["phase_gated_median_deg"] for row in analysis["peaks"]])
    mean, resultant, _ = pm.circ_mean(medians)
    fwhms = [row["phase_gated_fwhm_deg"] for row in analysis["peaks"]]
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
def build_atlas(outdir, analysis, cmap, args, r0, c0, size_nm, detector, emit):
    """Render the complete atlas (25 figures per ring) and register the contract."""
    book = at.Atlas(outdir, cmap, dpi=args.dpi)
    for tag in ("ring_1x1", "ring_r3"):
        ring = analysis[tag]
        peaks, records = ring["peaks"], ring["records"]
        valid, mask_radius = ring["valid"], ring["mask_radius_px"]
        fields, psi = ring["fields"], ring["psi"]
        ring_radius = ring["radius_px"]
        summary = ring_summary(ring)
        triple = ring["triple_product"]
        centres = [record["integer"] for record in records]

        def peak_values(row, tag=tag):
            return {"peak": row["index"], "qx_px": row["qx_px"], "qy_px": row["qy_px"],
                    "radius_px": row["radius_px"], "snr": row["snr"],
                    "median_deg": row["phase_gated_median_deg"],
                    "fwhm_deg": row["phase_gated_fwhm_deg"],
                    "n_clusters": row["n_clusters"], "R": row["phase_ungated_R"]}

        def peak_paths(row, tag=tag):
            base = f"rings_analysis.{tag}.peaks.{row['index']}"
            return {"peak": f"{base}.index", "qx_px": f"{base}.qx_px",
                    "qy_px": f"{base}.qy_px", "radius_px": f"{base}.radius_px",
                    "snr": f"{base}.snr", "median_deg": f"{base}.phase_gated_median_deg",
                    "fwhm_deg": f"{base}.phase_gated_fwhm_deg",
                    "n_clusters": f"{base}.n_clusters",
                    "R": f"{base}.phase_ungated_R"}

        for row, record in zip(peaks, records):
            name = record["name"]
            phi = np.asarray(fields[name][1])
            amp = np.asarray(fields[name][2])
            values, paths = peak_values(row), peak_paths(row)
            book.mask_pair(psi[name]["in"], valid, f"{tag} p{row['index']} mask IN",
                           values, paths, tag, f"{tag}_p{row['index']}_mask_in.png",
                           "per_peak", row["index"],
                           mask_note=f"mask radius {mask_radius} px")
            book.mask_pair(psi[name]["out"], valid, f"{tag} p{row['index']} mask OUT",
                           values, paths, tag, f"{tag}_p{row['index']}_mask_out.png",
                           "per_peak", row["index"],
                           mask_note=f"complement of the same {mask_radius} px mask")
            good = pp.gate_mask(amp, valid, args.gate)
            hist, edges = np.histogram(phi[good], bins=361, range=(0.0, 2 * np.pi),
                                       weights=amp[good], density=True)
            book.phi_distribution(phi, good, amp, hist, edges,
                                  f"{tag} p{row['index']} phi(r)", values, paths, tag,
                                  f"{tag}_p{row['index']}_phi_dist.png", "per_peak",
                                  row["index"])

        ring_values = {"radius_px": ring_radius, "n_peaks": ring["n_peaks"],
                       "mask_radius_px": mask_radius}
        ring_paths = {"radius_px": f"rings_analysis.{tag}.radius_px",
                      "n_peaks": f"rings_analysis.{tag}.n_peaks",
                      "mask_radius_px": f"rings_analysis.{tag}.mask_radius_px"}
        book.qspace_mask(ring["fft2"], centres, mask_radius, f"{tag} q-space mask",
                         ring_values, ring_paths, tag, f"{tag}_qspace_mask.png")

        mask_all = pp.circle_mask(ring["fft2"].shape, centres, mask_radius)
        psi_all_in = pp.complex_ifft(np.where(mask_all, ring["fft2"], 0.0))
        psi_all_out = pp.complex_ifft(np.where(~mask_all, ring["fft2"], 0.0))
        book.mask_pair(psi_all_in, valid, f"{tag} all peaks mask IN", ring_values,
                       ring_paths, tag, f"{tag}_mask_all_in.png", "summary", None,
                       mask_note=f"mask on the six reflections, radius {mask_radius} px")
        book.mask_pair(psi_all_out, valid, f"{tag} all peaks mask OUT", ring_values,
                       ring_paths, tag, f"{tag}_mask_all_out.png", "summary", None,
                       mask_note="complement of the six-reflection mask")
        book.case_histograms(
            [{"label": "FFT2 (no mask)", "phase": np.angle(ring["fft2"]).ravel()},
             {"label": f"{tag} mask OUT", "phase": np.angle(psi_all_out)[valid]},
             {"label": f"{tag} mask IN", "phase": np.angle(psi_all_in)[valid]}],
            f"{tag} phase histograms",
            {"radius_px": ring_radius, "n_peaks": ring["n_peaks"],
             "pixels": int(np.count_nonzero(mask_all))},
            {"radius_px": f"rings_analysis.{tag}.radius_px",
             "n_peaks": f"rings_analysis.{tag}.n_peaks",
             "pixels": f"rings_analysis.{tag}.mask_all_pixels"},
            tag, f"{tag}_case_phase_histograms.png")

        hist_items, map_items = [], []
        for row, record in zip(peaks, records):
            name = record["name"]
            phi = np.asarray(fields[name][1])
            amp = np.asarray(fields[name][2])
            good = pp.gate_mask(amp, valid, args.gate)
            hist, edges = np.histogram(phi[good], bins=361, range=(0.0, 2 * np.pi),
                                       weights=amp[good], density=True)
            hist_items.append({"hist": hist, "edges": edges,
                               "label": f"p{row['index']} (median "
                                        f"{row['phase_gated_median_deg']:.3f} deg)"})
            map_items.append({"phi": phi, "good": good,
                              "label": f"p{row['index']} phi(r)"})
        book.grid_histograms(hist_items,
                             f"{tag} phi histograms of the six reflections",
                             {"radius_px": ring_radius, "n_peaks": ring["n_peaks"],
                              "mean_deg": summary["mean_deg"], "R": summary["R"],
                              "n_clusters": summary["n_clusters_max"]},
                             {"radius_px": f"rings_analysis.{tag}.radius_px",
                              "n_peaks": f"rings_analysis.{tag}.n_peaks",
                              "mean_deg": f"rings_analysis.{tag}.summary.mean_deg",
                              "R": f"rings_analysis.{tag}.summary.R",
                              "n_clusters": f"rings_analysis.{tag}.summary.n_clusters_max"},
                             tag, f"{tag}_phi_hist_summary.png")
        book.grid_maps(map_items, f"{tag} phi(r) maps of the six reflections",
                       ring_values, ring_paths, tag, f"{tag}_phi_map_summary.png")

        theta = np.asarray(ring["theta_field"])
        amp_product = np.asarray(ring["amp_product"])
        good_theta = pp.gate_mask(amp_product, valid, args.gate)
        hist, edges = np.histogram(theta[good_theta], bins=361, range=(0.0, 2 * np.pi),
                                   weights=amp_product[good_theta], density=True)
        book.theta_field(theta, good_theta, amp_product, hist, edges,
                         f"{tag} theta(r)",
                         {"theta_deg": triple["theta_mean_deg"],
                          "ladder_dist_deg": triple["theta_ladder_distance_deg"],
                          "R": triple["theta_R"], "q_sum_px": triple["q_sum_px"],
                          "n_peaks": ring["n_peaks"]},
                         {"theta_deg":
                          f"rings_analysis.{tag}.triple_product.theta_mean_deg",
                          "ladder_dist_deg":
                          f"rings_analysis.{tag}.triple_product.theta_ladder_distance_deg",
                          "R": f"rings_analysis.{tag}.triple_product.theta_R",
                          "q_sum_px": f"rings_analysis.{tag}.triple_product.q_sum_px",
                          "n_peaks": f"rings_analysis.{tag}.n_peaks"},
                         tag, f"{tag}_theta_field.png")
        emit(f"# atlas: {tag} -> {FIGURES_PER_RING} figures "
             f"({len(PER_PEAK_FIGURES)} per peak x {ring['n_peaks']} + "
             f"{len(SUMMARY_FIGURES)} summary)")
    return book.manifest(extra={"field_of_view_nm": float(size_nm),
                                "detector": detector,
                                "r0_px": [float(r0[0]), float(r0[1])],
                                "c_deg": float(c0),
                                "figures_per_ring": FIGURES_PER_RING,
                                "reference_lines_deg": [0.0, 120.0, 240.0],
                                "reference_lines_note": ("2 pi k / 3, k = 0, 1, 2 "
                                                         "(0/120/240 deg): the dashed "
                                                         "lines of every phase "
                                                         "histogram in this atlas")})


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv=None):
    args = parse_args(argv)
    csv_path = Path(args.input)
    outdir = Path(args.outdir) if args.outdir else csv_path.parent
    outdir.mkdir(parents=True, exist_ok=True)
    size_nm = field_of_view(args)
    log_lines = []

    def emit(text=""):
        print(text)
        log_lines.append(text)

    topo = np.loadtxt(csv_path, delimiter=args.delimiter)
    if topo.ndim != 2 or topo.shape[0] != topo.shape[1]:
        raise SystemExit(f"{csv_path}: expected a square 2D matrix, got {topo.shape}")
    valid = np.isfinite(topo)
    n = int(topo.shape[0])
    mask_radius = int(args.pct / 100.0 * n)

    package, package_detail = load_package(args.stm_lib)
    if args.detector == "package" and package is None:
        raise SystemExit(f"--detector package requested but the import failed: "
                         f"{package_detail}")
    use_package = (args.detector == "package"
                   or (args.detector == "auto" and package is not None))

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
        detection = package[1](topo, size_nm, patch_half=args.patch_half,
                               subtract_plane=False, return_fft2=False)
        records = [(float(peak.q_px[0]), float(peak.q_px[1]), float(peak.amplitude),
                    float(peak.snr), float(np.hypot(*peak.q_px)))
                   for peak in detection.peaks]
        detector = "package:bragg_peak.detect_bragg_peaks"
        detector_detail = (f"{len(detection.peaks)} peaks, basis source "
                           f"{detection.meta.get('basis_source')}")
    else:
        records = pp.detect_reflections(np.abs(fft2))
        detector = "builtin:phasepipe.detect_reflections"
        detector_detail = f"{len(records)} peaks"

    emit("# phase statistics of a corrected topography image (geometry and "
         "mathematics only; skill version " + SKILL_VERSION + ")")
    emit(f"# input: {csv_path}")
    emit(f"# canvas: {n} x {n} px, field of view {size_nm} nm ({size_nm / n:.6f} nm/px), "
         f"valid pixels {100 * float(np.mean(valid)):.2f} %")
    emit(f"# FFT2: {fft_source}")
    emit(f"# detector: {detector} ({detector_detail})")
    emit(f"# mask radius {args.pct:g} % of the side = {mask_radius} px; gate {args.gate}; "
         f"histogram {args.bins} bins with {args.smooth_deg:g} deg smoothing")
    emit("# phase definitions: phi(r) = angle(psi(r)) - (2 pi / N) q.(r - c), c = N // 2; "
         "the phase VALUE of a reflection is the amplitude-weighted circular mean over "
         "all valid pixels (ungated = the reflection's global phase); median, FWHM and "
         "clusters describe the amplitude-gated sample and always state the gate")

    rings = pp.group_rings(records, tol_frac=args.ring_cluster_tol, min_members=6)
    emit("")
    emit("== detected rings (radius px, members, total |F|) ==")
    for ring in rings:
        emit(f"#   radius {ring['radius']:10.4f}  members {len(ring['members']):2d}  "
             f"|F| {ring['total_amplitude']:.4e}")
    ring_rows = [{"radius_px": float(ring["radius"]),
                  "n_members": len(ring["members"]),
                  "total_amplitude": float(ring["total_amplitude"]),
                  "member_radii_px": [float(m[4]) for m in
                                      sorted(ring["members"], key=lambda item: item[4])],
                  "max_snr": float(max(m[3] for m in ring["members"]))}
                 for ring in rings]
    write_csv(outdir / "ring_candidates.csv", ring_rows)

    choice = pp.choose_rings(rings, anchor=args.anchor,
                             reference_radius_px=args.reference_radius_px,
                             match_tol=args.match_tol, pair_tol=args.pair_tol,
                             r3_tol=args.r3_tol, expect=PEAKS_PER_RING)
    selection = {
        "status": choice["status"],
        "method": choice["method"],
        "anchor": args.anchor,
        "tolerances": {"match": args.match_tol, "pair": args.pair_tol,
                       "r3": args.r3_tol, "ring_cluster": args.ring_cluster_tol},
        "ring_1x1_radius_px": (float(choice["ring_1x1"]["radius"])
                               if choice["ring_1x1"] is not None else None),
        "ring_r3_radius_px": (float(choice["ring_r3"]["radius"])
                              if choice["ring_r3"] is not None else None),
        "ratio": float(choice["ratio"]) if np.isfinite(choice["ratio"]) else None,
        "sqrt3_deviation": (float(choice["sqrt3_deviation"])
                            if np.isfinite(choice["sqrt3_deviation"]) else None),
        "sqrt3": float(pp.SQRT3),
    }
    emit("")
    if choice["status"] != "ok":
        emit(f"ring pair: NOT FOUND ({choice['status']})")
        emit(f"# reference ring selection: --anchor {args.anchor} -> {choice['method']}")
        if choice["status"] == "no_rings":
            emit("# no ring with at least six members was detected on this image")
        emit("# the r3 ring is NOT reported: no ring sits at 1/sqrt(3) of the "
             "reference radius inside the tolerance, and the analysis stops here "
             "instead of analysing an unpaired ring")
        emit("# options: raise --r3-tol, name the reference ring explicitly "
             "(--anchor radius --reference-radius-px), or fix the geometry first")
        payload = {"skill": "stm-topo-phase-analysis", "skill_version": SKILL_VERSION,
                   "status": choice["status"], "input": str(csv_path),
                   "canvas_px": n, "field_of_view_nm": size_nm,
                   "detector": detector, "rings": ring_rows,
                   "ring_selection": selection}
        (outdir / "phase_stats.json").write_text(json.dumps(payload, indent=2) + "\n")
        emit(f"# written: {outdir / 'phase_stats.json'}, "
             f"{outdir / 'ring_candidates.csv'}, {outdir / 'phase_stats.log'}")
        (outdir / "phase_stats.log").write_text("\n".join(log_lines) + "\n")
        return 2

    ring_1x1, ring_r3 = choice["ring_1x1"], choice["ring_r3"]
    emit(f"ring pair: ring_1x1 = {ring_1x1['radius']:.4f} px, ring_r3 = "
         f"{ring_r3['radius']:.4f} px, ratio = {choice['ratio']:.6f} "
         f"(sqrt(3) = {pp.SQRT3:.6f}, deviation "
         f"{100 * choice['sqrt3_deviation']:.4f} %)")
    emit(f"# reference ring selection: --anchor {args.anchor} -> {choice['method']}")
    emit(f"# r3 position: 1/sqrt(3) of the reference radius "
         f"({ring_1x1['radius'] / pp.SQRT3:.4f} px target, tolerance "
         f"{args.r3_tol:.1%})")

    analysis = {}
    for tag, ring in (("ring_1x1", ring_1x1), ("ring_r3", ring_r3)):
        analysis[tag] = analyse_ring(tag, ring, fft2, valid, mask_radius, args)
        summary = ring_summary(analysis[tag])
        emit("")
        emit(f"== {tag}: {analysis[tag]['n_peaks']} reflections, mean radius "
             f"{analysis[tag]['radius_px']:.4f} px ==")
        emit(f"# |sum q| over the three independent reflections: detected "
             f"{analysis[tag]['q_sum_detected_px']:.4f} px, used "
             f"{analysis[tag]['three_independent']['q_sum_used_px']:.4f} px "
             f"({'projected to zero' if args.project_q else 'not projected'})")
        position = analysis[tag]["q_position"]
        emit(f"# peak-position quality: hexagon residual rms "
             f"{position['hexagon_residual_rms_px']:.4f} px (max "
             f"{position['hexagon_residual_max_px']:.4f} px), radial rms "
             f"{position['radial_rms_px']:.4f} px about {position['mean_radius_px']:.4f} px; "
             f"without the projection the wavevector sum would ramp theta by "
             f"{analysis[tag]['theta_ramp_span_deg_if_unprojected']:.3f} deg across the "
             f"canvas")
        for row in analysis[tag]["peaks"]:
            emit(f"   {row['name']}  q=({row['qx_px']:9.3f},{row['qy_px']:9.3f}) px  "
                 f"snr={row['snr']:10.1f}  radius={row['radius_px']:9.3f} px")
            emit(f"      phase (ungated circular mean) {row['phase_ungated_mean_deg']:9.4f}"
                 f" deg  R {row['phase_ungated_R']:.5f}  | median(ungated) "
                 f"{row['phase_ungated_median_deg']:9.4f} deg")
            emit(f"      gated shape: median {row['phase_gated_median_deg']:9.4f} deg, "
                 f"fwhm {fmt(row['phase_gated_fwhm_deg'])} deg (deconvolved "
                 f"{fmt(row['phase_gated_fwhm_deconv_deg'])}), iqr "
                 f"{fmt(row['phase_gated_iqr_deg'])} deg, clusters {row['n_clusters']} "
                 f"at {[round(c, 3) for c in row['cluster_centres_deg']]} widths "
                 f"{[None if not np.isfinite(w) else round(w, 3) for w in row['cluster_widths_deg']]}"
                 f" weights {[round(w, 4) for w in row['cluster_weight_fractions']]}")
            emit(f"      amplitude median {row['amp_median']:.4e}, fwhm "
                 f"{row['amp_fwhm']:.4e}; folded 120 deg: median "
                 f"{row['folded120_median_deg']:9.4f} deg, R {row['folded120_R']:.5f}, "
                 f"fwhm {fmt(row['folded120_fwhm_deg'])} deg")
        three = analysis[tag]["three_independent"]
        triple = analysis[tag]["triple_product"]
        emit("   Friedel pair sums (identity of a real image): "
             + "; ".join(f"{left}+{right} = {value:.5f} deg"
                         for left, right, value in friedel_triples(analysis[tag])))
        emit(f"   three independent phases {three['members']}: values "
             f"{[round(v, 4) for v in three['values_deg']]} mod 120 "
             f"{[round(v, 4) for v in three['values_mod120_deg']]}, sum "
             f"{three['sum_deg']:.4f} deg (mod 120 {three['sum_mod120_deg']:.4f}, mirror "
             f"{three['mirror_sum_deg']:.4f}), distance to the 0/120/240 ladder "
             f"{three['ladder_distance_deg']:.4f} deg")
        emit(f"   per-pixel triple product theta: mean {triple['theta_mean_deg']:.4f} deg "
             f"(mirror {triple['theta_mirror_mean_deg']:.4f}), ladder distance "
             f"{triple['theta_ladder_distance_deg']:.4f} deg, R {triple['theta_R']:.5f}, "
             f"fwhm {fmt(triple['theta_fwhm_deg'])} deg, clusters "
             f"{triple['theta_n_clusters']}, 3 theta mod 360 "
             f"{triple['theta_mean_3x_mod360_deg']:.4f} deg")
        emit(f"   ring summary of the six gated medians: circular mean "
             f"{summary['mean_deg']:.4f} deg, R {summary['R']:.5f}, max clusters "
             f"{summary['n_clusters_max']}, mean fwhm {fmt(summary['fwhm_mean_deg'])} deg")

    # ---- gauge fix: the six ring_1x1 reflections pin the lattice origin ---- #
    recs = analysis["ring_1x1"]["records"]
    qs = [record["q_px"] for record in recs]
    phases = [np.radians(record["stats"]["phase_ungated"]["mean_deg"]) for record in recs]
    best, minima = pm.fit_origin(qs, phases, n)
    r0, c0 = best["r0_px"], best["c_rad"]
    emit("")
    emit("== lattice-referenced gauge fix (the ring_1x1 reflections pin the origin) ==")
    emit(f"# least squares of phi_j = (2 pi / N) q_j . r0 + c over the six ring_1x1 "
         f"reflections: r0 = ({r0[0]:.4f}, {r0[1]:.4f}) px, c = {best['c_deg']:.4f} deg, "
         f"rms residual {best['rms_deg']:.4f} deg, {len(minima)} local minimum(a)")
    emit("# that rms is a MODEL-CONSISTENCY diagnostic, not phase noise: the six peaks "
         "are three Friedel pairs (+Phi / -Phi), so a single origin cannot represent "
         "them unless the phases are origin-consistent; the q-sum-zero triple fit below "
         "has rms 0 by construction")
    reference_deviations = []
    for record in recs:
        fixed = np.degrees(pm.gauge_phase(
            np.radians(record["stats"]["phase_ungated"]["mean_deg"]),
            record["q_px"], r0, c0, n)) % 360.0
        reference_deviations.append(float(pm.dist_to_ladder_deg(fixed, (0.0,))))
    ref_rms = float(np.sqrt(np.mean(np.square(reference_deviations))))
    emit(f"# the six gauge-fixed ring_1x1 phases deviate from 0 by "
         f"{[round(value, 3) for value in reference_deviations]} deg (rms "
         f"{ref_rms:.4f} deg); the ring_r3 phases inherit this reference uncertainty")
    emit("# the least-squares solution set is {best r0} plus every direct-lattice "
         "translation of the reference ring: such a translation adds an exact multiple "
         "of 2 pi to each of the six model phases, so the residual is bit-identical and "
         "the fit can never separate them (structural degeneracy, not a numerical error)")
    band = ref_rms / 2.1213  # sigma_peak / 3 with sigma_peak = rms_residual / sqrt(0.5)
    emit(f"# systematic band of a single-peak absolute phase: sigma_peak ~ "
         f"{ref_rms:.4f}/0.707 = {ref_rms / 0.707:.4f} deg, so a reflection at 1/sqrt(3) "
         f"of the reference radius inherits sigma_peak/3 = {band:.4f} deg (systematic: "
         f"it does not shrink with more pixels)")

    triple_index, _q_sum = pp.triple_selection([tuple(q) for q in qs])
    triple_fit, _triple_minima = pm.fit_origin(
        [tuple(qs[index]) for index in triple_index],
        [phases[index] for index in triple_index], n)
    emit(f"# origin fit convention: SIX reflections (all three Friedel pairs) are "
         f"fitted against one common offset and one origin; residual rms "
         f"{best['rms_deg']:.4f} deg. The three reflections with q-sum zero alone "
         f"determine (r0, c) exactly, so their rms is {triple_fit['rms_deg']:.3e} deg "
         f"by construction (3 equations, 3 unknowns) and carries no diagnostic "
         f"information. The six-peak residual is the model-consistency number: it "
         f"grows when the six phases are not realisable by a single origin (an "
         f"uneven / heterogeneous reference ring, or a synthetic pattern whose "
         f"mirrors carry the opposite sign)")

    a1, a2 = reference_lattice_vectors(qs)
    basis = np.column_stack([a1, a2])
    branch_table = []
    for index, minimum in enumerate(minima):
        delta = np.asarray(minimum["r0_px"], dtype=float) - np.asarray(r0, dtype=float)
        coefficients = np.linalg.solve(basis, delta)
        lattice = bool(abs(coefficients[0] - round(coefficients[0])) < 1e-6
                       and abs(coefficients[1] - round(coefficients[1])) < 1e-6)
        shifts = [float(np.degrees(np.mod((2.0 * np.pi / n)
                                          * float(np.dot(record["q_px"], delta)),
                                          2.0 * np.pi)))
                  for record in analysis["ring_r3"]["records"]]
        multiple_of_120 = bool(all(
            min(abs(np.radians(value)) % np.radians(120.0),
                np.radians(120.0) - abs(np.radians(value)) % np.radians(120.0)) < 1e-9
            for value in shifts))
        branch_table.append({
            "branch": index,
            "r0_px": [float(minimum["r0_px"][0]), float(minimum["r0_px"][1])],
            "c_deg": float(minimum["c_deg"]),
            "rms_deg": float(minimum["rms_deg"]),
            "delta_r0_px": [float(delta[0]), float(delta[1])],
            "delta_r0_lattice_coefficients": [float(coefficients[0]),
                                              float(coefficients[1])],
            "lattice_translation_of_the_best_fit": lattice,
            "induced_ring_r3_phase_shifts_deg": shifts,
            "induced_shifts_are_multiples_of_120deg": multiple_of_120,
        })
    lattice_branches = sum(1 for row in branch_table
                           if row["lattice_translation_of_the_best_fit"])
    emit(f"# branch table: {len(branch_table)} local minimum(a), {lattice_branches} of "
         f"them direct-lattice translations of the best fit (bit-identical residual); a "
         f"translation shifts the ring_r3 phases by (2 pi / N) q.L per reflection")
    if any(row["induced_shifts_are_multiples_of_120deg"]
           for row in branch_table if row["branch"] > 0):
        emit("# those induced shifts happen to be multiples of 120 deg in this data set")
    else:
        emit("# those induced shifts are NOT multiples of 120 deg: the ring_r3 absolute "
             "phases are comparable only after the r0 branch is stated explicitly, and "
             "cross-peak differences and the position of the folded distribution move "
             "with the branch; invariant are the closing sums and the per-peak shape")

    for tag in ("ring_1x1", "ring_r3"):
        for row, record in zip(analysis[tag]["peaks"], analysis[tag]["records"]):
            fixed = np.degrees(pm.gauge_phase(
                np.radians(row["phase_ungated_mean_deg"]), record["q_px"], r0, c0,
                n)) % 360.0
            row["gauge_mean_deg"] = float(fixed)
            row["gauge_mean_mod120_deg"] = float(fixed % 120.0)
            row["friedel_partner"] = analysis[tag]["friedel"][row["name"]]["partner"]
            row["friedel_sum_deg"] = analysis[tag]["friedel"][row["name"]]["sum_deg"]
            row["friedel_deviation_deg"] = analysis[tag]["friedel"][row["name"]][
                "deviation_from_360_deg"]

    # fix the JSON-only fields the atlas paths point at
    for tag in ("ring_1x1", "ring_r3"):
        ring = analysis[tag]
        ring["summary"] = ring_summary(ring)
        mask_all = pp.circle_mask(ring["fft2"].shape,
                                  [record["integer"] for record in ring["records"]],
                                  mask_radius)
        ring["mask_all_pixels"] = int(np.count_nonzero(mask_all))

    # ---- atlas ----------------------------------------------------------- #
    manifest = None
    cmap, cmap_source = at.load_colormap(args.stm_lib)
    emit(f"# colormap: {cmap_source}")
    if not args.no_figures:
        at.setup_style()
        manifest = build_atlas(outdir, analysis, cmap, args, r0, c0, size_nm,
                               detector, emit)

    # ---- JSON ------------------------------------------------------------ #
    payload = {
        "skill": "stm-topo-phase-analysis",
        "skill_version": SKILL_VERSION,
        "status": "ok",
        "input": str(csv_path),
        "canvas_px": n,
        "field_of_view_nm": size_nm,
        "nm_per_px": size_nm / n,
        "valid_fraction": float(np.mean(valid)),
        "detector": detector,
        "detector_detail": detector_detail,
        "fft2_source": fft_source,
        "package_import": package_detail,
        "mask_pct": args.pct,
        "mask_radius_px": mask_radius,
        "gate": args.gate,
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
                                    "rotation of every reported phase; shapes, "
                                    "concentrations and ladder distances are identical",
            "r0_px": [float(r0[0]), float(r0[1])],
            "c_deg": float(best["c_deg"]),
            "rms_deg": float(best["rms_deg"]),
            "rms_deg_note": ("model-consistency diagnostic of the six-peak fit, not "
                             "phase noise: the six reflections are three Friedel pairs "
                             "(+Phi / -Phi), so one origin represents them only if the "
                             "six phases are origin-consistent; see triple_fit for the "
                             "q-sum-zero variant whose rms is 0 by construction"),
            "n_minima": len(minima),
            "reference_deviations_deg": reference_deviations,
            "reference_rms_deg": ref_rms,
            "fit_convention": "six reflections (three Friedel pairs) vs one common "
                              "offset and one origin; the q-sum-zero triple determines "
                              "(r0, c) exactly and is reported for comparison",
            "triple_fit": {"members": [int(index) for index in triple_index],
                           "rms_deg": float(triple_fit["rms_deg"]),
                           "r0_px": [float(triple_fit["r0_px"][0]),
                                     float(triple_fit["r0_px"][1])],
                           "c_deg": float(triple_fit["c_deg"]),
                           "delta_r0_vs_six_peak_px": [
                               float(triple_fit["r0_px"][0] - r0[0]),
                               float(triple_fit["r0_px"][1] - r0[1])]},
            "sigma_peak_deg": float(ref_rms / 0.707),
            "single_peak_systematic_band_deg": float(ref_rms / 2.1213),
            "solution_set": ("{best r0} plus every direct-lattice translation of the "
                             "reference ring; a translation adds an exact multiple of "
                             "2 pi to the six model phases, so the residual is "
                             "bit-identical and the fit cannot separate them"),
            "lattice_vectors": [
                {"a1": [float(reference_lattice_vectors(qs)[0][0]),
                        float(reference_lattice_vectors(qs)[0][1])],
                 "a2": [float(reference_lattice_vectors(qs)[1][0]),
                        float(reference_lattice_vectors(qs)[1][1])]}],
            "lattice_branches": int(lattice_branches),
            "branch_table": branch_table,
        },
        "rings_analysis": {},
        "atlas": {"dir": str(outdir), "figures": 0, "per_ring": {},
                  "manifest": "atlas_manifest.json",
                  "declared_per_ring": {"per_peak": len(PER_PEAK_FIGURES),
                                        "peaks": PEAKS_PER_RING,
                                        "summary": len(SUMMARY_FIGURES),
                                        "total": FIGURES_PER_RING}},
        "conventions": {
            "peak_order": "p0 is the reflection closest to +qy (12 o'clock), the "
                          "following ones run clockwise in the (q_x, q_y) plane",
            "phase_value": "amplitude-weighted circular mean over all valid pixels "
                           "(ungated) = the reflection's global phase",
            "phase_shape": "median/FWHM/clusters of the amplitude-gated sample "
                           f"(gate {args.gate}, bins {args.bins}, smoothing "
                           f"{args.smooth_deg} deg)",
            "fwhm_raw": "FWHM of the smoothed amplitude-weighted histogram",
            "fwhm_deconv": "Gaussian-equivalent deconvolution of the smoothing kernel",
            "triangle_independent": "three reflections whose wavevectors sum to zero; "
                                    "the other triple is their antipodal mirror",
            "triple_product": "sum of the three independent per-pixel phase fields "
                              f"(wavevector sum projected to zero: {args.project_q})",
            "folded_120": "per-pixel phi folded by mod 120 deg and mapped back onto "
                          "the circle; invariant under a global 120 deg relabelling",
            "ladder": "the three reference lines of every phase histogram are "
                      "2 pi k / 3 (0/120/240 deg), a mathematical ladder",
            "gauge": "phi~ = phi - (2 pi / N) q.r0 - c with (r0, c) from the six "
                     "ring_1x1 reflections; the solution set also contains every "
                     "direct-lattice translation of r0 (identical residual), which "
                     "shifts the ring_r3 phases by (2 pi / N) q.L per reflection -- "
                     "generally not a multiple of 120 deg, so ring_r3 absolute phases "
                     "are comparable only after the r0 branch is stated",
            "invariant_under_the_r0_choice": "Friedel pair sums, the three-independent "
                                             "sum and the per-pixel triple product, the "
                                             "per-peak distribution shape (R, FWHM, "
                                             "cluster count and widths), amplitude and "
                                             "coherence",
            "single_peak_absolute_phase": "reference only: state which branch of r0 was "
                                          "used and attach the systematic band "
                                          "sigma_peak/3 derived from the reference-ring "
                                          "residual rms",
        },
    }
    for tag in ("ring_1x1", "ring_r3"):
        ring = analysis[tag]
        exported = {key: value for key, value in ring.items()
                    if key not in ("records", "fields", "psi", "detections", "members",
                                   "valid", "fft2", "theta_hist", "theta_field",
                                   "amp_product")}
        exported["theta_hist"] = {
            "bin_edges_deg": [float(np.degrees(value)) for value in ring["theta_hist"]["edges"]],
            "density": [float(value) for value in ring["theta_hist"]["hist"]],
        }
        payload["rings_analysis"][tag] = exported
    if manifest is not None:
        payload["atlas"].update({
            "figures": int(manifest["total_figures"]),
            "per_ring": {ring: info["figures"]
                         for ring, info in manifest["per_ring"].items()}})
        (outdir / "atlas_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (outdir / "phase_stats.json").write_text(json.dumps(payload, indent=2) + "\n")

    # ---- CSV tables ------------------------------------------------------ #
    write_csv(outdir / "phase_stats.csv",
              [row for tag in ("ring_1x1", "ring_r3") for row in analysis[tag]["peaks"]])
    relation_rows = []
    for tag in ("ring_1x1", "ring_r3"):
        for left, right, value in friedel_triples(analysis[tag]):
            info = analysis[tag]["friedel"][left]
            relation_rows.append({
                "ring": tag, "relation": "friedel_pair_sum", "subject": left,
                "partner": right, "value_deg": value,
                "deviation_deg": info["deviation_from_360_deg"],
                "note": f"identity of a real image; |sum q| = "
                        f"{info['sum_q_norm_px']:.3e} px"})
        three = analysis[tag]["three_independent"]
        triple = analysis[tag]["triple_product"]
        relation_rows.append({
            "ring": tag, "relation": "three_independent_sum",
            "subject": "+".join(three["members"]), "partner": "",
            "value_deg": three["sum_deg"], "deviation_deg": three["ladder_distance_deg"],
            "note": f"= 3 Phi_bar (mod 360); distance to the 0/120/240 ladder; mirror "
                    f"{three['mirror_sum_deg']:.4f} deg; |sum q| used "
                    f"{three['q_sum_used_px']:.4f} px"})
        relation_rows.append({
            "ring": tag, "relation": "per_pixel_triple_product",
            "subject": "+".join(triple["members"]), "partner": "",
            "value_deg": triple["theta_mean_deg"],
            "deviation_deg": triple["theta_ladder_distance_deg"],
            "note": f"theta(r) = 3 Phi_d inside a region; R = {triple['theta_R']:.5f}, "
                    f"fwhm {triple['theta_fwhm_deg']:.4f} deg, mirror "
                    f"{triple['theta_mirror_mean_deg']:.4f} deg"})
    write_csv(outdir / "phase_relations.csv", relation_rows)

    comparison = []
    for tag in ("ring_1x1", "ring_r3"):
        summary = analysis[tag]["summary"]
        three = analysis[tag]["three_independent"]
        triple = analysis[tag]["triple_product"]
        comparison.append({
            "ring": tag,
            "radius_px": analysis[tag]["radius_px"],
            "n_peaks": analysis[tag]["n_peaks"],
            "circular_mean_of_medians_deg": summary["mean_deg"],
            "n_clusters_max": summary["n_clusters_max"],
            "mean_fwhm_deg": summary["fwhm_mean_deg"],
            "friedel_max_deviation_deg": max(info["deviation_from_360_deg"]
                                             for info in analysis[tag]["friedel"].values()),
            "three_sum_deg": three["sum_deg"],
            "three_ladder_distance_deg": three["ladder_distance_deg"],
            "theta_mean_deg": triple["theta_mean_deg"],
            "theta_ladder_distance_deg": triple["theta_ladder_distance_deg"],
            "theta_R": triple["theta_R"],
            "q_sum_used_px": triple["q_sum_px"],
        })
    write_csv(outdir / "phase_ring_comparison.csv", comparison)

    emit("")
    emit("== reading guide ==")
    emit("# phase value       -> amplitude-weighted circular mean (ungated); the "
         "quantity that reproduces the reflection's global phase")
    emit("# median/FWHM       -> shape of the GATED distribution; quoting them requires "
         "the gate")
    emit("# cluster count     -> how many distinct phases are present (independent of "
         "the 0/120/240 ladder)")
    emit("# amplitude/coherence -> whether a reflection is still there (a balanced "
         "mixture can cancel it)")
    emit("# three-phase sum   -> whether the weighted circular mean of the phases sits "
         "near the ladder (a value-range test, not a count of regions)")
    emit("# Friedel pair sum  -> identity of a real image; a non-zero value means an "
         "implementation error, not a sample property")
    emit("# gauge fix         -> the ring_1x1 reflections pin r0; the solution set also "
         "contains every direct-lattice translation of r0 (identical residual)")
    emit("# robust quantities -> Friedel pair sums, the three-independent sum, the "
         "per-pixel triple product, the per-peak distribution shape (R/FWHM/clusters/"
         "widths) and the amplitude/coherence: these are the numbers to compare across "
         "conventions")
    emit("# reference only    -> a single-peak absolute phase changes with the r0 branch "
         "(by (2 pi / N) q.L, generally not a multiple of 120 deg); quote it together "
         "with the branch and the systematic band sigma_peak/3 printed above")
    emit("# mirror ambiguity  -> the other closing triple is antipodal, so a signed theta "
         "and '3 theta mod 360' mirror around 0 while the distance to the 0/120/240 "
         "ladder is unchanged; every theta line above prints both")
    emit("# two rings, one definition: every quantity above is computed with the same "
         "estimator and the same gate for ring_1x1 and ring_r3")
    emit("")
    emit(f"# written: {outdir / 'phase_stats.json'}, {outdir / 'phase_stats.csv'}, "
         f"{outdir / 'phase_relations.csv'}, {outdir / 'phase_ring_comparison.csv'}, "
         f"{outdir / 'ring_candidates.csv'}, {outdir / 'phase_stats.log'}"
         + (f", {outdir / 'atlas_manifest.json'} and "
            f"{payload['atlas']['figures']} atlas figures" if manifest else ""))
    (outdir / "phase_stats.log").write_text("\n".join(log_lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
