"""Lawler-Fujita lattice-phase correction of a topography image.

The correction follows Fujita et al., PNAS 2014, Supplementary Text section 4: a
local lock-in on two first-order Bragg wave vectors of the hexagonal lattice gives
the local lattice phase, the phase difference from the gauge gives a slowly
varying displacement field ``u(r)`` (in nm), and the image is resampled along that
field, ``corrected(r) = T(r + u(r))``.  Peak detection comes from the
``stm_data_processing.utils.bragg_peak`` package; the correction anchors on the
detected 1x1 ring, and everything else -- the lock-in, the phase unwrapping, the
displacement solve, the warp and the plots -- is this skill (see ``lf_lib`` for
the convention table).

Outputs (same conventions as the affine-correction skill):
    <outdir>/<stem>_corrected.csv       corrected topography (square, NaN padded)
    <outdir>/<stem>_corrected_fft2.npy  complex FFT2 (complex128, fftshifted)
    <outdir>/<stem>_corrected.png       topography plot (gwyddion colormap)
    <outdir>/<stem>_corrected_fft.png   FFT plot (inferno, log, percentile norm)
    <outdir>/correction.log             the console report (both contract lines)
    <outdir>/correction_report.json     the same numbers, machine readable

Extra Lawler-Fujita artifacts (npy + preview png each): the local phase maps
``theta_a``, ``theta_b`` and (when a third direction exists) ``theta_c`` [radians,
unwrapped, in the npy; the png shows the same phase wrapped into (-180, 180]
degrees], the lock-in amplitude maps, the displacement components ``u_x`` and
``u_y`` [nm] and the validity mask.

``--save-transform FILE`` writes the transferable bundle: the JSON above plus the
u-field npz next to it, which ``stm_lf_apply.py`` applies to another dataset of the
same scan.

Usage:
    cd /path/to/STM_DataProcessing
    MPLCONFIGDIR=<writable> PYTHONDONTWRITEBYTECODE=1 \
        .venv/bin/python <this script> INPUT.csv -L 50 -o OUT_DIR \
        [--a 0.246] [--lambda-nm 30] [--delimiter ','] [--save-transform BUNDLE.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import lf_lib as lf  # noqa: E402

SKILL_VERSION = "1.0"
BUNDLE_SCHEMA = "lawler-fujita-correction-transform"
BUNDLE_SCHEMA_VERSION = 1
# A pixel of the lock-in amplitude below this fraction of the median amplitude is
# treated as unreliable (no lattice signal -> the local phase means nothing).
DEFAULT_AMPLITUDE_FRACTION = 0.10
# Coverage below this fraction makes the whole field untrustworthy -> fallback.
DEFAULT_MIN_COVERAGE = 0.50
# The phase of a lock-in kernel is unreliable within that many kernel widths of the
# image border (the FFT treats the image as periodic, so the phase there averages
# data from the opposite edge): used for the border-excluded diagnostics.
LAMBDA_BORDER_MARGIN = 0.65


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input", help="topography matrix (whitespace, tab or comma separated)")
    parser.add_argument("-L", "--size-nm", type=float, required=True,
                        help="scan size in nm (square image)")
    parser.add_argument("-o", "--outdir", default=None,
                        help="output directory (default: next to the input file)")
    parser.add_argument("--a", type=float, default=0.246,
                        help="lattice constant of the 1x1 ring in nm (default 0.246)")
    parser.add_argument("--lambda-nm", type=float, default=30.0,
                        help="real-space scale of the distortion kept by the lock-in "
                             "low-pass exp(-lambda^2 k^2 / 2) in nm (default 30.0)")
    parser.add_argument("--amplitude-fraction", type=float,
                        default=DEFAULT_AMPLITUDE_FRACTION,
                        help="lock-in amplitude threshold as a fraction of the median "
                             "amplitude (default 0.10); below it a pixel is invalid")
    parser.add_argument("--min-coverage", type=float, default=DEFAULT_MIN_COVERAGE,
                        help="minimum valid-pixel fraction before the fit falls back "
                             "(default 0.50)")
    parser.add_argument("--ring-tol", type=float, default=0.10,
                        help="relative radius tolerance when matching a detected ring "
                             "to the ideal 1x1 ring radius (default 0.10)")
    parser.add_argument("--pair-angle", type=float, default=120.0, choices=(120.0, 60.0),
                        help="angle between the two lock-in directions (default 120): "
                             "120 puts the third direction on a real ring member, so "
                             "theta_a + theta_b + theta_c = 0 mod 2 pi is a genuine "
                             "consistency check; with 60 the derived third phase is a "
                             "tautology and no third measurement exists")
    parser.add_argument("--pad", type=int, default=10,
                        help="extra canvas margin of the warp in pixels (default 10)")
    parser.add_argument("--order", type=int, default=3,
                        help="spline order of the warp (default 3)")
    parser.add_argument("--patch-half", type=int, default=8,
                        help="sub-pixel localizer patch half width (default 8)")
    parser.add_argument("--ring-cluster-tol", type=float, default=0.02,
                        help="relative radius tolerance of the ring clustering (default 0.02)")
    parser.add_argument("--delimiter", default=None,
                        help="column delimiter (default: auto-detect tab/comma/whitespace; "
                             "the escape '\\t' is accepted)")
    parser.add_argument("--stm-lib", default="/Users/hunfen/Documents/GitHub/"
                                             "STM_DataProcessing/src",
                        help="STM_DataProcessing src directory")
    parser.add_argument("--list-peaks", action="store_true",
                        help="print the detected peak table")
    parser.add_argument("--list-rings", action="store_true",
                        help="print the detected ring table (radius clustering)")
    parser.add_argument("--save-transform", default=None, metavar="FILE",
                        help="also write the transferable bundle: this JSON plus the "
                             "u-field npz next to it (schema "
                             f"{BUNDLE_SCHEMA}, version {BUNDLE_SCHEMA_VERSION})")
    return parser.parse_args(argv)


def resolve_delimiter(raw):
    if raw is None:
        return None
    return raw.replace("\\t", "\t").replace("\\s", " ")


def load_matrix(path, delimiter, loader):
    data = loader(path) if delimiter is None else np.loadtxt(path, delimiter=delimiter)
    if data.ndim != 2 or data.shape[0] != data.shape[1]:
        raise ValueError(f"{path}: expected a square 2D matrix, got {data.shape}")
    return data


def ring_table(peaks, cluster_tol):
    """Radius clustering of the detected peaks (geometry only, no labels)."""
    records = [(float(peak.q_px[0]), float(peak.q_px[1]), float(peak.amplitude),
                float(peak.snr), float(np.hypot(*peak.q_px))) for peak in peaks]
    return lf.group_rings(records, tol_frac=cluster_tol, min_members=6)


def match_ring(rings, radius_ideal, tol):
    """The detected ring whose radius matches the ideal 1x1 radius, or None."""
    best, best_deviation = None, None
    for ring in rings:
        deviation = abs(float(ring["radius"]) / float(radius_ideal) - 1.0)
        if deviation <= tol and (best_deviation is None or deviation < best_deviation):
            best, best_deviation = ring, deviation
    return best


def pick_directions(ring, pair_angle):
    """Pick the ring members used as lock-in directions and build the reference.

    The strongest member is ``a``; ``b`` is the member ``pair_angle`` degrees away
    (counter-clockwise); for ``pair_angle = 120`` the member 120 degrees clockwise is
    the third direction ``c``.  A 60 degree pair has no third direction: there
    ``-(theta_a + theta_b)`` is not the phase of a ring member and no independent
    measurement exists.

    Returns ``(measured, reference, orientation_deg, deviation_deg)``.  ``measured``
    are the detected member records (the distortion-shifted peaks of the data);
    ``reference`` are the lock-in wave vectors actually used.  The reference is the
    *ideal* hexagon: the radius comes from ``--a`` and the orientation from the data
    (the mean deviation of the measured members from exactly 120 degree spacing), so
    ``Q_a + Q_b + Q_c = 0`` holds exactly and the third-direction check tests the
    lock-in rather than the geometry.
    """
    members = sorted(ring["members"], key=lambda member: -member[2])
    angle_a = np.degrees(np.arctan2(members[0][1], members[0][0]))

    def nearest(signed_target_deg):
        best, best_distance = None, None
        for member in members[1:]:
            delta = ((np.degrees(np.arctan2(member[1], member[0])) - angle_a + 180.0)
                     % 360.0) - 180.0
            distance = abs(delta - signed_target_deg)
            if distance <= 25.0 and (best_distance is None or distance < best_distance):
                best, best_distance = member, distance
        return best

    second = nearest(float(pair_angle))
    if second is None:
        raise ValueError(f"no ring member {pair_angle:g} degrees counter-clockwise from "
                         "the strongest member of the matched ring")
    third = nearest(-120.0) if float(pair_angle) == 120.0 else None
    if third is None and float(pair_angle) == 120.0:
        raise ValueError("no ring member 120 degrees clockwise from the strongest "
                         "member: the third-direction consistency cannot be measured")
    measured = [members[0], second] + ([third] if third is not None else [])
    slots = [0.0, float(pair_angle)] + ([-120.0] if third is not None else [])
    deviations = []
    for member, slot in zip(measured, slots, strict=True):
        delta = ((np.degrees(np.arctan2(member[1], member[0])) - angle_a - slot + 180.0)
                 % 360.0) - 180.0
        deviations.append(delta)
    deviation = float(np.mean(deviations))
    orientation = float(angle_a) + deviation
    return measured, slots, orientation, deviation


def reference_vectors(orientation_deg, slots, size_nm, a_nm):
    """Ideal-radius wave vectors of the hexagonal reference at ``orientation_deg``."""
    radius = lf.ideal_radius_px(size_nm, a_nm)
    return np.array([[radius * np.cos(np.radians(orientation_deg + slot)),
                      radius * np.sin(np.radians(orientation_deg + slot))]
                     for slot in slots], dtype=float)


def ring_quality(peaks, size_nm, a_nm, cluster_tol, ring_tol):
    """Residual self-check of a re-detected image: 1x1 ring spread and pair ratio."""
    rings = ring_table(peaks, cluster_tol)
    radius_ideal = lf.ideal_radius_px(size_nm, a_nm)
    matched = match_ring(rings, radius_ideal, ring_tol)
    quality = {"n_rings": len(rings), "ideal_radius_px": float(radius_ideal),
               "ring_1x1": None, "r3_ratio_vs_sqrt3": None,
               "radii_px": [float(ring["radius"]) for ring in rings]}
    if matched is not None:
        radii = np.array(sorted(float(member[4]) for member in matched["members"]))
        quality["ring_1x1"] = {
            "radius_px": float(matched["radius"]),
            "radius_nm_inv": float(matched["radius"]) * 2.0 * np.pi / size_nm,
            "n_members": len(matched["members"]),
            "min_radius_px": float(radii.min()),
            "max_radius_px": float(radii.max()),
            "anisotropy": float((radii.max() - radii.min()) / float(np.mean(radii))),
            "deviation_from_ideal": float(matched["radius"] / radius_ideal - 1.0),
        }
    inner = match_ring(rings, radius_ideal / np.sqrt(3.0), ring_tol)
    if matched is not None and inner is not None:
        ratio = float(matched["radius"]) / float(inner["radius"])
        quality["r3_ratio_vs_sqrt3"] = {
            "ratio": ratio,
            "deviation": float(ratio / np.sqrt(3.0) - 1.0),
            "r3_radius_px": float(inner["radius"]),
        }
    return quality


def u_statistics(u_nm, valid):
    """RMS and maximum of the displacement components over the valid pixels."""
    stats = {}
    for name, component in (("u_x", u_nm[0]), ("u_y", u_nm[1])):
        values = component[valid]
        stats[name] = {
            "rms_nm": float(np.sqrt(np.mean(values ** 2))) if values.size else 0.0,
            "max_nm": float(np.max(np.abs(values))) if values.size else 0.0}
    norm = np.hypot(u_nm[0], u_nm[1])[valid]
    stats["norm"] = {"rms_nm": float(np.sqrt(np.mean(norm ** 2))) if norm.size else 0.0,
                     "max_nm": float(np.max(norm)) if norm.size else 0.0}
    return stats


def write_artifacts(prefix, theta, amplitude, u_nm, valid):
    """Write the LF artifact set (npy + preview png) and return the path dict.

    Every file carries the documented ``_lf_`` marker: ``<stem>_lf_theta_a.npy``,
    ``<stem>_lf_u_x.npy``, ``<stem>_lf_mask.npy`` and so on, so the LF maps cannot be
    confused with the core correction products (``<stem>_corrected.*``).
    """
    written = {}
    entries = [("theta_a", theta[0], "phase"), ("theta_b", theta[1], "phase"),
               ("amplitude_a", amplitude[0], "amplitude"),
               ("amplitude_b", amplitude[1], "amplitude"),
               ("u_x", u_nm[0], "displacement"), ("u_y", u_nm[1], "displacement"),
               ("mask", np.asarray(valid, dtype=float), "mask")]
    if theta.shape[0] > 2:
        entries.append(("theta_c", theta[2], "phase"))
        entries.append(("amplitude_c", amplitude[2], "amplitude"))
    for name, data, kind in entries:
        npy_path = prefix.parent / f"{prefix.name}_lf_{name}.npy"
        png_path = prefix.parent / f"{prefix.name}_lf_{name}.png"
        lf.save_npy(npy_path, data)
        if kind == "phase":
            # The png shows the phase wrapped into (-180, 180] degrees: the npy keeps
            # the unwrapped phase in radians, whose absolute values are not a colour scale.
            wrapped = np.degrees(np.angle(np.exp(1j * np.asarray(data, dtype=float))))
            lf.save_map(png_path, wrapped, cmap="twilight", vmin=-180.0, vmax=180.0)
        elif kind == "amplitude":
            lf.save_map(png_path, data, cmap="inferno")
        elif kind == "mask":
            lf.save_map(png_path, data, cmap="gray", vmin=0.0, vmax=1.0)
        else:
            limit = float(np.max(np.abs(data))) if np.any(data) else 1.0
            lf.save_map(png_path, data, cmap="RdBu_r", vmin=-limit, vmax=limit)
        written[name] = {"npy": str(npy_path), "png": str(png_path)}
    return written


def bundle_payload(report_path, csv_path, args, q_pair_px, report, n_out, u_path, u_stats):
    """Standalone description of the fitted Lawler-Fujita correction."""
    q_pair_px = np.asarray(q_pair_px, dtype=float)
    return {
        "schema": BUNDLE_SCHEMA,
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "source_report": str(Path(report_path).resolve()),
        "source_input": str(Path(csv_path).resolve()),
        "q_a_px": [float(value) for value in q_pair_px[0]],
        "q_b_px": [float(value) for value in q_pair_px[1]],
        "q_a_nm_inv": [float(value)
                       for value in lf.wave_vectors_nm_inv(q_pair_px[0], args.size_nm)],
        "q_b_nm_inv": [float(value)
                       for value in lf.wave_vectors_nm_inv(q_pair_px[1], args.size_nm)],
        "q_matrix_px": [[float(value) for value in row] for row in q_pair_px],
        "lambda_nm": float(args.lambda_nm),
        "a_nm": float(args.a),
        "amplitude_fraction": float(args.amplitude_fraction),
        "amplitude_threshold": (None if report["lockin"]["amplitude_threshold"] is None
                               else float(report["lockin"]["amplitude_threshold"])),
        "gauge": report["gauge"],
        "n_px_reference": int(report["canvas_px"]),
        "n_out_reference": int(n_out),
        "field_of_view_nm_reference": float(args.size_nm),
        "nm_per_px_reference": float(args.size_nm / report["canvas_px"]),
        "pad": int(args.pad),
        "order": int(args.order),
        "method": str(report["method"]),
        "fallback": bool(report["fallback"]),
        "mask_coverage_fraction": float(report["lockin"]["mask_coverage_fraction"]),
        "u_stats_nm": u_stats,
        "u_field_file": str(u_path),
        "transfer_constraints": report["transfer_constraints"],
        "usage": ("apply with: .venv/bin/python "
                  "skills/lawler-fujita-correction/scripts/stm_lf_apply.py INPUT.csv "
                  "--transform <this file> -L SIZE_NM -o OUT"),
    }


def main(argv=None):
    args = parse_args(argv)
    sys.path.insert(0, args.stm_lib)
    from stm_data_processing.utils.bragg_peak import (
        compute_fft2,
        detect_bragg_peaks,
        load_image,
    )
    from stm_data_processing.utils.plot_funcs import subtractMeanPlane

    csv_path = Path(args.input)
    outdir = Path(args.outdir) if args.outdir else csv_path.parent
    outdir.mkdir(parents=True, exist_ok=True)
    stem = csv_path.stem
    log_lines = []

    def emit(text=""):
        print(text)
        log_lines.append(text)

    topo = load_matrix(csv_path, resolve_delimiter(args.delimiter), load_image)
    topo = np.flipud(subtractMeanPlane(topo))  # first row = top scan line
    n = int(topo.shape[0])
    radius_ideal = lf.ideal_radius_px(args.size_nm, args.a)
    gauge = ("theta_bar = 0: the mean local phase over the valid pixels is removed, "
             "equivalently the ideal lattice has an atomic peak at the origin (zero "
             "imaginary part of the average Bragg peak)")

    emit(f"# Lawler-Fujita lattice-phase correction (skill version {SKILL_VERSION})")
    emit(f"# input: {csv_path}")
    emit(f"# canvas {n} x {n} px, field of view {args.size_nm:g} nm "
         f"({args.size_nm / n:.6f} nm/px)")
    emit(f"# ideal 1x1 ring radius {radius_ideal:.4f} px = "
         f"{2 * np.pi * radius_ideal / args.size_nm:.4f} nm^-1 (a = {args.a:g} nm)")

    detection = detect_bragg_peaks(topo, args.size_nm, patch_half=args.patch_half,
                                   subtract_plane=False, return_fft2=False)
    rings = ring_table(detection.peaks, args.ring_cluster_tol)
    if rings:
        emit("")
        emit("== detected rings (radius clustering of the detected peaks) ==")
        for ring in rings:
            radius = float(ring["radius"])
            emit(f"#   radius {radius:10.4f} px = {radius * 2 * np.pi / args.size_nm:9.4f} "
                 f"nm^-1  members {len(ring['members']):2d}  |F| "
                 f"{ring['total_amplitude']:.4e}")
    if args.list_rings:
        emit("")
        emit("ring members (radius px, member radii):")
        for ring in rings:
            radii = sorted(float(member[4]) for member in ring["members"])
            emit(f"  radius {ring['radius']:9.4f}: "
                 + ", ".join(f"{value:.3f}" for value in radii))
    if args.list_peaks:
        emit("")
        emit("detected peaks (qx, qy, |q|, label, snr, quality):")
        for index, peak in enumerate(detection.peaks, 1):
            label = "-" if peak.index_hk is None else f"{peak.index_hk[0]},{peak.index_hk[1]}"
            emit(f"  {index:2d}: q=({peak.q_px[0]:8.2f}, {peak.q_px[1]:8.2f}) "
                 f"|q|={np.hypot(*peak.q_px):8.2f} label={label:>7s} "
                 f"snr={peak.snr:6.2f} {peak.quality}")

    matched = match_ring(rings, radius_ideal, args.ring_tol)
    fallback_reason = None
    if matched is None:
        fallback_reason = (f"no detected ring with at least six members matches the "
                           f"ideal 1x1 radius {radius_ideal:.4f} px within "
                           f"{100 * args.ring_tol:.1f} %")
        emit("")
        emit(f"# WARNING: {fallback_reason}")
    else:
        emit("")
        emit(f"# anchor ring: the detected 1x1 ring (radius {float(matched['radius']):.4f} "
             f"px, {100 * (float(matched['radius']) / radius_ideal - 1):+.3f} % from the "
             f"ideal, {len(matched['members'])} members)")

    q_pair, theta_maps, amplitude_maps, threshold = None, None, None, None
    valid, coverage = None, None
    if fallback_reason is None:
        try:
            directions, slots, orientation, deviation = pick_directions(matched,
                                                                        args.pair_angle)
        except ValueError as exc:
            fallback_reason = str(exc)
            emit(f"# WARNING: {fallback_reason}")
        else:
            names = ["a", "b", "c"][:len(directions)]
            q_measured = np.array([member[:2] for member in directions], dtype=float)
            q_pair = reference_vectors(orientation, slots, args.size_nm, args.a)
            n_directions = int(q_pair.shape[0])
            closure = ("Q_a + Q_b + Q_c = 0" if n_directions == 3
                       else "Q_a + Q_b does not close: no third direction")
            emit(f"# reference hexagon: ideal radius {radius_ideal:.4f} px from a = "
                 f"{args.a:g} nm, orientation {orientation:.4f} deg (measured members "
                 f"deviate {deviation:+.4f} deg from the exact "
                 f"{args.pair_angle:g} deg spacing of this direction set)")
            emit(f"# lock-in directions: {n_directions} "
                 f"({'three' if n_directions == 3 else 'two'}) reference wave vector(s), "
                 f"pairwise spacing exactly {args.pair_angle:g} deg ({closure})")
            for member, q_vector, name in zip(directions, q_pair, names, strict=True):
                emit(f"# lock-in direction {name}: measured q_px = "
                     f"({member[0]:9.3f}, {member[1]:9.3f}) amplitude {member[2]:.4e} -> "
                     f"reference q_px = ({q_vector[0]:9.3f}, {q_vector[1]:9.3f}), |q| = "
                     f"{np.hypot(*q_vector):9.3f} px = "
                     f"{np.hypot(*q_vector) * 2 * np.pi / args.size_nm:9.4f} nm^-1, "
                     f"angle {np.degrees(np.arctan2(q_vector[1], q_vector[0])):7.2f} deg")
            # The lock-in mask has radius 1/lambda around the reference vector; if the
            # measured peak sits further away than that (a mean strain of the data),
            # the mask does not even contain the actual Bragg peak and the extracted
            # field is meaningless -- say so instead of returning it silently.
            offset_cycles = float(np.max(np.hypot(*(q_measured - q_pair).T))
                                  / args.size_nm)
            band_margin = float(args.lambda_nm) * offset_cycles
            emit(f"# reference vs measured peak: offset {offset_cycles:.4f} cycles/nm, "
                 f"band margin lambda x offset = {band_margin:.3f} "
                 f"({'inside' if band_margin <= 1.0 else 'OUTSIDE'} the low-pass radius "
                 f"1/lambda = {1.0 / args.lambda_nm:.4f} cycles/nm)")
            if band_margin > 1.0:
                emit("# WARNING: the low-pass is too narrow for this data: the measured "
                     "Bragg peak lies outside the lock-in radius, so the extracted "
                     "phase/displacement field is not the lattice distortion (lower "
                     "--lambda-nm until lambda x offset < 1)")
            theta_list, amplitude_list, valid_list = [], [], []
            for q_vector in q_pair:
                theta, amplitude, mask, threshold = lf.lockin_phase(
                    topo, q_vector, args.lambda_nm, args.size_nm, args.amplitude_fraction)
                theta_list.append(theta)
                amplitude_list.append(amplitude)
                valid_list.append(mask)
            valid = np.logical_and.reduce(valid_list) & np.isfinite(topo)
            theta_maps = np.stack(theta_list)
            amplitude_maps = np.stack(amplitude_list)
            coverage = float(np.count_nonzero(valid)) / float(valid.size)
            if coverage < args.min_coverage:
                fallback_reason = (
                    f"the lock-in amplitude reaches the threshold on only "
                    f"{100 * coverage:.2f} % of the pixels (< "
                    f"{100 * args.min_coverage:.1f} %): the displacement field would be "
                    "built almost entirely from unreliable pixels")
                emit(f"# WARNING: {fallback_reason}")

    u_nm, u_stats, third, gradients = None, {}, {}, []
    if fallback_reason is None:
        u_nm = lf.displacement_from_phase(theta_maps[0], theta_maps[1], q_pair[0],
                                          q_pair[1], args.size_nm)
        u_stats = u_statistics(u_nm, valid)
        gradients = [lf.max_phase_step(theta_maps[index])
                     for index in range(theta_maps.shape[0])]
        emit("")
        emit("== lock-in ==")
        emit(f"# low-pass exp(-lambda^2 k^2 / 2) with lambda = {args.lambda_nm:g} nm "
             f"(keeps distortion structure on real-space scales >= {args.lambda_nm:g} nm)")
        emit(f"# amplitude threshold {args.amplitude_fraction:g} x median = {threshold:.4e} "
             f"-> valid pixels {100 * coverage:.2f} %")
        emit(f"# largest per-pixel phase step {max(gradients):.4f} rad (the unwrap is "
             f"unambiguous below pi)")
        emit("")
        emit("== displacement field ==")
        emit(f"# gauge: {gauge}")
        emit(f"# u RMS {u_stats['norm']['rms_nm']:.4f} nm (u_x {u_stats['u_x']['rms_nm']:.4f}, "
             f"u_y {u_stats['u_y']['rms_nm']:.4f}), u max {u_stats['norm']['max_nm']:.4f} nm")
        if theta_maps.shape[0] == 3:
            residual, _wrapped = lf.wrapped_residual(np.degrees(theta_maps))
            # The demodulated phase within one lock-in kernel width of the border is
            # contaminated (the FFT wraps the image periodically), so the diagnostic
            # is also evaluated on the interior region.
            margin = min(int(np.ceil(LAMBDA_BORDER_MARGIN * args.lambda_nm
                                    / (args.size_nm / n))), n // 4)
            interior = np.zeros_like(valid)
            interior[margin:n - margin, margin:n - margin] = True
            interior = interior & valid
            interior_rms = None
            if int(np.count_nonzero(interior)) > 0:
                interior_rms = lf.wrapped_residual(np.degrees(theta_maps[:, interior]))[0]
            third = {"available": True, "wrapped_rms_deg": residual,
                     "wrapped_rms_deg_interior": interior_rms,
                     "interior_margin_px": int(margin),
                     "interior_margin_rule": f"{LAMBDA_BORDER_MARGIN:g} x lambda",
                     "interior_pixels": int(np.count_nonzero(interior))}
            emit(f"# third-direction consistency: theta_a + theta_b + theta_c wrapped "
                 f"RMS {residual:.4f} deg (full mask); "
                 f"{interior_rms:.4f} deg inside a {margin} px border margin "
                 f"({LAMBDA_BORDER_MARGIN:g} x lambda)")
        else:
            third = {"available": False,
                     "reason": ("the two lock-in directions are 60 deg apart, so "
                                "-(theta_a + theta_b) is not the phase of a ring member "
                                "and no independent third measurement exists")}
            emit(f"# third-direction consistency: not available ({third['reason']})")
        corrected, n_out, half, magnitude_px = lf.warp_by_field(
            topo, u_nm, args.size_nm, valid=valid, pad=args.pad, order=args.order)
        size_out = float(args.size_nm) * n_out / n
        emit(f"# warp: corrected(r) = T(r + u(r)), spline order {args.order}, "
             f"max |u| {magnitude_px:.3f} px, canvas growth {half} px per side")
    else:
        corrected = np.array(topo, dtype=float, copy=True)
        n_out, half, magnitude_px, size_out = n, 0, 0.0, float(args.size_nm)
        emit("# WARNING: identity fallback: no correction is applied, the input is copied "
             "unchanged with its own canvas and field of view; do not treat this output "
             "as a corrected image")
    nan_fraction = 1.0 - float(np.count_nonzero(np.isfinite(corrected))) / corrected.size
    emit(f"# corrected canvas: {n_out} x {n_out} px, field of view {size_out:.4f} nm "
         f"({size_out / n_out:.6f} nm/px), NaN {100 * nan_fraction:.2f} %")

    emit("")
    emit("== after the correction (re-detected on the corrected image) ==")
    after = detect_bragg_peaks(corrected, size_out, patch_half=args.patch_half,
                               subtract_plane=False, return_fft2=False)
    before_quality = ring_quality(detection.peaks, args.size_nm, args.a,
                                  args.ring_cluster_tol, args.ring_tol)
    after_quality = ring_quality(after.peaks, size_out, args.a, args.ring_cluster_tol,
                                 args.ring_tol)
    if after_quality["ring_1x1"] is None:
        emit("# residual self-check: no 1x1 ring re-detected on the corrected image")
    else:
        ring_after = after_quality["ring_1x1"]
        emit(f"# residual self-check: 1x1 ring radius {ring_after['radius_px']:.4f} px "
             f"({100 * ring_after['deviation_from_ideal']:+.4f} % from the ideal "
             f"{radius_ideal:.4f} px), radius spread {ring_after['min_radius_px']:.4f} .. "
             f"{ring_after['max_radius_px']:.4f} px "
             f"(anisotropy {100 * ring_after['anisotropy']:.4f} %)")
        ring_before = before_quality["ring_1x1"]
        if ring_before is not None:
            emit(f"# residual self-check: 1x1 anisotropy before "
                 f"{100 * ring_before['anisotropy']:.4f} % -> after "
                 f"{100 * ring_after['anisotropy']:.4f} %")
    if after_quality["r3_ratio_vs_sqrt3"] is not None:
        pair = after_quality["r3_ratio_vs_sqrt3"]
        emit(f"# residual self-check: 1x1/r3 ring pair ratio {pair['ratio']:.6f} "
             f"({100 * pair['deviation']:+.4f} % from sqrt(3) = {np.sqrt(3.0):.6f})")
    elif after_quality["ring_1x1"] is not None:
        emit("# residual self-check: no r3 ring on the corrected image, the 1 : sqrt(3) "
             "ring pair cannot be formed")

    out_csv = outdir / f"{stem}_corrected.csv"
    np.savetxt(out_csv, corrected, delimiter=",", fmt="%.10e")
    fft2_corrected = compute_fft2(corrected, size_out, subtract_plane=False)
    out_fft2 = outdir / f"{stem}_corrected_fft2.npy"
    np.save(out_fft2, fft2_corrected)

    lf.setup_style()
    cmap, cmap_source = lf.load_colormap(args.stm_lib)
    emit(f"# colormap: {cmap_source}")
    cmap_bad = cmap.copy()
    cmap_bad.set_bad(color=lf.BAD_COLOR)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(corrected, cmap=cmap_bad, origin="lower")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")
    out_png = outdir / f"{stem}_corrected.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    magnitude = np.abs(fft2_corrected)
    lo, hi = np.percentile(magnitude, [5, 99.5])
    logged = np.log(1.0 + magnitude)
    span = np.log(1.0 + hi) - np.log(1.0 + lo)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(np.clip((logged - np.log(1.0 + lo)) / span, 0.0, 1.0), cmap="inferno",
              origin="lower")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")
    out_fft_png = outdir / f"{stem}_corrected_fft.png"
    fig.savefig(out_fft_png, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    artifacts = {}
    if fallback_reason is None:
        artifacts = write_artifacts(outdir / stem, theta_maps, amplitude_maps, u_nm, valid)
        emit("# LF artifacts: " + ", ".join(f"{name} (npy+png)" for name in artifacts))

    report = {
        "skill": "lawler-fujita-correction",
        "skill_version": SKILL_VERSION,
        "input": str(csv_path),
        "canvas_px": n,
        "field_of_view_nm": float(args.size_nm),
        "nm_per_px": float(args.size_nm / n),
        "a_nm": float(args.a),
        "ideal_radius_px": float(radius_ideal),
        "ideal_radius_nm_inv": float(2 * np.pi * radius_ideal / args.size_nm),
        "lambda_nm": float(args.lambda_nm),
        "amplitude_fraction": float(args.amplitude_fraction),
        "gauge": gauge,
        "method": "identity_fallback" if fallback_reason is not None else "lawler_fujita",
        "fallback": bool(fallback_reason is not None),
        "fallback_reason": fallback_reason,
        "pair_angle_deg": float(args.pair_angle),
        "n_lockin_directions": (None if q_pair is None else int(q_pair.shape[0])),
        "direction_spacing_deg": (None if q_pair is None else float(args.pair_angle)),
        "rings_before": [{"radius_px": float(ring["radius"]),
                          "radius_nm_inv": float(ring["radius"]) * 2 * np.pi / args.size_nm,
                          "n_members": len(ring["members"]),
                          "total_amplitude": float(ring["total_amplitude"])}
                         for ring in rings],
        "anchor_ring": ({"radius_px": float(matched["radius"]),
                         "n_members": len(matched["members"]),
                         "deviation_from_ideal": float(matched["radius"] / radius_ideal - 1.0)}
                        if matched is not None else None),
        "q_a_px": ([float(value) for value in q_pair[0]] if q_pair is not None else None),
        "q_b_px": ([float(value) for value in q_pair[1]] if q_pair is not None else None),
        "q_c_px": ([float(value) for value in q_pair[2]]
                   if q_pair is not None and q_pair.shape[0] > 2 else None),
        "q_measured_px": ([[float(value) for value in row] for row in q_measured]
                          if q_pair is not None else None),
        "hexagon_orientation_deg": (None if q_pair is None else float(orientation)),
        "hexagon_member_deviation_deg": (None if q_pair is None else float(deviation)),
        "q_a_nm_inv": ([float(value)
                        for value in lf.wave_vectors_nm_inv(q_pair[0], args.size_nm)]
                       if q_pair is not None else None),
        "q_b_nm_inv": ([float(value)
                        for value in lf.wave_vectors_nm_inv(q_pair[1], args.size_nm)]
                       if q_pair is not None else None),
        "q_c_nm_inv": ([float(value)
                        for value in lf.wave_vectors_nm_inv(q_pair[2], args.size_nm)]
                       if q_pair is not None and q_pair.shape[0] > 2 else None),
        "q_matrix_px": ([[float(value) for value in row] for row in q_pair]
                        if q_pair is not None else None),
        "lockin": {
            "lambda_nm": float(args.lambda_nm),
            "reference_offset_cycles_per_nm": (None if q_pair is None
                                               else float(offset_cycles)),
            "band_margin_lambda_times_offset": (None if q_pair is None
                                                else float(band_margin)),
            "band_warning": bool(q_pair is not None and band_margin > 1.0),
            "amplitude_fraction": float(args.amplitude_fraction),
            "amplitude_threshold": None if threshold is None else float(threshold),
            "mask_coverage_fraction": 0.0 if coverage is None else float(coverage),
            "max_phase_step_rad": float(max(gradients)) if gradients else None,
        },
        "displacement": {
            "u_stats_nm": u_stats,
            "component_order": "u_x, u_y along the pixel axes (x = column, y = row), nm",
            "gauge": gauge,
        },
        "third_direction": third,
        "canvas_growth_px": int(half),
        "max_displacement_px": float(magnitude_px),
        "n_out": int(n_out),
        "corrected_field_of_view_nm": float(size_out),
        "corrected_nm_per_px": float(size_out / n_out),
        "nan_fraction": float(nan_fraction),
        "residual_self_check": {"before": before_quality, "after": after_quality},
        "transfer_constraints": {
            "same_scan": ("the target must be the same scan and the same field of view: "
                          "the bundle carries a dense displacement field of the "
                          "reference grid"),
            "grid_rescaling": ("a different pixel count is supported: u is physical, so "
                               "it is interpolated onto the target pixel coordinates"),
            "lattice_not_required": ("the target need not show a lattice at all -- "
                                     "transferring the field to simultaneously measured "
                                     "data is the point"),
            "mask_propagation": "the validity mask is resampled with the field",
        },
        "lf_artifacts": artifacts,
        "written": {
            "corrected_csv": str(out_csv),
            "corrected_fft2_npy": str(out_fft2),
            "corrected_png": str(out_png),
            "corrected_fft_png": str(out_fft_png),
            "correction_log": str(outdir / "correction.log"),
            "correction_report": str(outdir / "correction_report.json"),
        },
    }
    report_path = outdir / "correction_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    emit("")
    emit(f"# written: {out_csv}, {out_fft2}, {out_png}, {out_fft_png}, "
         f"{report_path}, {outdir / 'correction.log'}")
    (outdir / "correction.log").write_text("\n".join(log_lines) + "\n")

    if args.save_transform:
        bundle_path = Path(args.save_transform)
        bundle_path.parent.mkdir(parents=True, exist_ok=True)
        u_path = bundle_path.with_suffix(".npz")
        if u_nm is not None:
            np.savez(u_path, u_x=u_nm[0], u_y=u_nm[1], valid=valid,
                     n_px_reference=n, field_of_view_nm_reference=float(args.size_nm),
                     nm_per_px=float(args.size_nm / n))
            payload = bundle_payload(report_path, csv_path, args, q_pair, report, n_out,
                                     u_path, u_stats)
        else:
            payload = bundle_payload(report_path, csv_path, args, np.zeros((2, 2)), report,
                                     n_out, u_path, {})
        bundle_path.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"# bundle written: {bundle_path}"
              + (f" (+ {u_path})" if u_nm is not None
                 else " (fallback: no u field written)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
