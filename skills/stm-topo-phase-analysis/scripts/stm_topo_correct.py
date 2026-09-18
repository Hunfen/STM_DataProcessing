"""Geometry correction of a topography image through the bragg_peak package.

Peak detection, sub-pixel localization and the symmetric positive-definite
stretch ``M`` (pure stretch, no rotation) all come from
``stm_data_processing.utils.bragg_peak``; this script loads the matrix, applies
the skill's preprocessing (``flipud(subtractMeanPlane(...))``, first row = top
scan line), states the reference lattice explicitly and writes the plots and the
report.

The **anchor ring is explicit**: the package's data-anchored detection labels one
ring as the first ring, and that ring need not be the 1x1 ring -- when the r3 ring
sits inside the 1x1 ring (radius ratio 1/sqrt(3)), the detection can anchor on the
r3 ring, and a reference lattice that assumes the 1x1 ring would then try to
stretch the r3 ring up to the 1x1 radius and destroy the geometry.  ``--anchor-ring``
states which physical ring the detected anchor is:

    --anchor-ring 1x1   the anchor ring is the 1x1 ring:  a_ref = a
    --anchor-ring r3    the anchor ring is the r3 ring:   a_ref = sqrt(3) * a

Everything else -- the stretch fit, the resampling rule, the corrected field of
view ``size_nm * n_out / n_px`` -- is unchanged.  The report prints which path was
taken, the fit method and fallback flag, |b1| before and after, the canvas, the
field of view, the NaN fraction and the lattice constant implied by the measured
ring.

Outputs:
    <outdir>/<stem>_corrected.csv       corrected topography (square, NaN padded)
    <outdir>/<stem>_corrected_fft2.npy  complex FFT2 (complex128, fftshifted)
    <outdir>/<stem>_corrected.png       topography plot (gwyddion colormap)
    <outdir>/<stem>_corrected_fft.png   FFT plot (inferno, log, percentile norm)
    <outdir>/correction.log             the console report
    <outdir>/correction_report.json     the same numbers, machine readable

Usage:
    cd /path/to/STM_DataProcessing
    MPLCONFIGDIR=<writable> PYTHONDONTWRITEBYTECODE=1 \
        .venv/bin/python <this script> INPUT.csv -L 50 -o OUT_DIR \
        [--a 0.246] [--anchor-ring 1x1|r3] [--delimiter ','] [--list-peaks]
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import atlas as at  # noqa: E402
import phasepipe as pp  # noqa: E402

SKILL_VERSION = "2.0"
PATCH_HALF = 8  # sub-pixel Gaussian patch half width: 8 -> 17x17 pixels


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input", help="topography matrix (whitespace, tab or comma separated)")
    parser.add_argument("-L", "--size-nm", type=float, required=True,
                        help="scan size in nm (square image)")
    parser.add_argument("-o", "--outdir", default=None,
                        help="output directory (default: next to the input file)")
    parser.add_argument("--a", type=float, default=0.246,
                        help="lattice constant of the 1x1 ring in nm (default 0.246)")
    parser.add_argument("--anchor-ring", default="1x1", choices=("1x1", "r3"),
                        help="which physical ring the data-anchored detection labels "
                             "as the first ring (default 1x1); with r3 the reference "
                             "lattice is sqrt(3) times a")
    parser.add_argument("--stm-lib", default="/Users/hunfen/Documents/GitHub/"
                                             "STM_DataProcessing/src",
                        help="STM_DataProcessing src directory")
    parser.add_argument("--peaks", nargs=4, type=float, default=None,
                        metavar=("Q1X", "Q1Y", "Q2X", "Q2Y"),
                        help="optional manual anchor pair (q offsets in px): restrict the "
                             "stretch fit to the two detected peaks closest to these")
    parser.add_argument("--delimiter", default=None,
                        help="column delimiter (default: auto-detect tab/comma/whitespace; "
                             "the escape '\\t' is accepted)")
    parser.add_argument("--patch-half", type=int, default=PATCH_HALF,
                        help="sub-pixel localizer patch half width (default 8)")
    parser.add_argument("--list-peaks", action="store_true",
                        help="print the detected peak table")
    parser.add_argument("--list-rings", action="store_true",
                        help="print the detected ring table (radius clustering)")
    parser.add_argument("--ring-cluster-tol", type=float, default=0.02,
                        help="relative radius tolerance of the ring clustering")
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


def pin_anchor_pair(detection, anchors):
    """Restrict the stretch fit to the two detected peaks closest to the anchors."""
    labelled = [peak for peak in detection.peaks
                if peak.index_hk is not None and peak.independent]
    if len(labelled) < 2:
        raise ValueError("--peaks needs at least two labelled peaks in the detection")
    chosen, used = [], set()
    for target in (np.array(anchors[:2]), np.array(anchors[2:])):
        order = sorted(range(len(labelled)),
                       key=lambda i: float(np.hypot(*(np.array(labelled[i].q_px) - target))))
        pick = next((i for i in order if i not in used), order[0])
        used.add(pick)
        chosen.append(labelled[pick])
    print("anchor pair: " + ", ".join(
        f"{peak.index_hk} q=({peak.q_px[0]:.2f}, {peak.q_px[1]:.2f})" for peak in chosen))
    return replace(detection, peaks=tuple(chosen))


def lattice_orientation_deg(detection):
    """Counter-clockwise angle of the data's own (1,0) basis direction.

    The detection is data-anchored, so its label gauge comes from the data.  A
    caller-stated reference must carry that same orientation: a symmetric stretch
    has no rotation, and feeding an unrotated ideal basis into it would force the
    fit to absorb the lattice orientation as a spurious shear.
    """
    lattice = getattr(detection, "lattice", None)
    if lattice is None:
        return None
    b1 = np.asarray(lattice.bvecs_nm_inv[0], dtype=float)
    if not np.all(np.isfinite(b1)) or float(np.hypot(*b1)) <= 0.0:
        return None
    return float(np.degrees(np.arctan2(b1[1], b1[0])))


def ring_table(peaks, cluster_tol):
    """Radius clustering of the detected peaks (geometry only, no labels)."""
    records = [(float(peak.q_px[0]), float(peak.q_px[1]), float(peak.amplitude),
                float(peak.snr), float(np.hypot(*peak.q_px))) for peak in peaks]
    return pp.group_rings(records, tol_frac=cluster_tol, min_members=6)


def implied_lattice(radius_nm_inv, a_nm, anchor):
    """Lattice constants implied by a measured ring radius.

    ``radius_nm_inv`` is the measured radius of the *anchored* ring in nm^-1; the
    anchored lattice constant follows from ``|b1| = 4 pi / (sqrt(3) a_ref)`` and the
    1x1 constant is ``a_ref`` for ``--anchor-ring 1x1`` and ``a_ref / sqrt(3)`` for
    ``--anchor-ring r3``.
    """
    a_ref = 4.0 * np.pi / (np.sqrt(3.0) * float(radius_nm_inv))
    a_1x1 = a_ref if anchor == "1x1" else a_ref / np.sqrt(3.0)
    return {"a_ref_nm": float(a_ref), "a_1x1_nm": float(a_1x1),
            "deviation_from_nominal": float(a_1x1 / float(a_nm) - 1.0)}


def main(argv=None):
    args = parse_args(argv)
    sys.path.insert(0, args.stm_lib)
    from stm_data_processing.utils.bragg_peak import (
        LatticeSpec, compute_fft2, correct_bragg_peaks, detect_bragg_peaks, load_image)
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

    detection = detect_bragg_peaks(topo, args.size_nm, patch_half=args.patch_half,
                                   subtract_plane=False, return_fft2=False)
    orientation_deg = lattice_orientation_deg(detection)
    a_ref = float(args.a) if args.anchor_ring == "1x1" else float(args.a) * np.sqrt(3.0)
    spec = LatticeSpec(a_nm=a_ref, symmetry="hexagonal", orientation_deg=orientation_deg)

    emit("# geometry correction through the bragg_peak package "
         f"(skill version {SKILL_VERSION})")
    emit(f"# input: {csv_path}")
    emit(f"# canvas {topo.shape[0]} x {topo.shape[1]} px, field of view "
         f"{args.size_nm:g} nm ({args.size_nm / topo.shape[0]:.6f} nm/px)")
    emit(f"# reference lattice: hexagonal, 1x1 lattice constant a = {args.a:g} nm, "
         f"anchor ring = {args.anchor_ring} -> a_ref = {a_ref:.6f} nm, "
         f"|b1|_ideal = {4 * np.pi / (np.sqrt(3) * a_ref):.4f} nm^-1, orientation "
         f"{'unknown' if orientation_deg is None else f'{orientation_deg:.2f} deg'}"
         f" (from the data), basis_source = {detection.meta.get('basis_source')}")
    emit("# the anchor ring is what the package's data-anchored detection labelled as "
         "the first ring; stating it is the caller's responsibility: with the r3 ring "
         "inside the 1x1 ring (radius ratio 1/sqrt(3)) an anchor assumption of 1x1 "
         "would stretch the r3 ring up to the 1x1 radius")

    rings = ring_table(detection.peaks, args.ring_cluster_tol)
    if rings:
        emit("")
        emit("== detected rings (radius clustering of the detected peaks) ==")
        for ring in rings:
            radius = float(ring["radius"])
            emit(f"#   radius {radius:10.4f} px = {radius * 2 * np.pi / args.size_nm:9.4f} "
                 f"nm^-1  members {len(ring['members']):2d}  |F| "
                 f"{ring['total_amplitude']:.4e}")
        if len(rings) > 1:
            strongest = max(rings, key=lambda ring: ring["total_amplitude"])
            innermost = min(rings, key=lambda ring: ring["radius"])
            ratio = float(innermost["radius"]) / float(strongest["radius"])
            if abs(ratio - 1.0 / np.sqrt(3.0)) < 0.03:
                emit(f"# geometry note: the innermost ring ({innermost['radius']:.4f} px) "
                     f"sits at 1/sqrt(3) of the strongest ring ({strongest['radius']:.4f} "
                     f"px, ratio {ratio:.6f}); if the anchor of the detection is the "
                     f"innermost ring, pass --anchor-ring r3")

    if args.peaks is not None:
        detection = pin_anchor_pair(detection, args.peaks)
    if args.list_peaks:
        emit("")
        emit("detected peaks (qx, qy, |q|, label, snr, quality):")
        for index, peak in enumerate(detection.peaks, 1):
            label = "-" if peak.index_hk is None else f"{peak.index_hk[0]},{peak.index_hk[1]}"
            emit(f"  {index:2d}: q=({peak.q_px[0]:8.2f}, {peak.q_px[1]:8.2f}) "
                 f"|q|={np.hypot(*peak.q_px):8.2f} label={label:>7s} "
                 f"snr={peak.snr:6.2f} {peak.quality}")
    if args.list_rings:
        emit("")
        emit("ring members (radius px, member radii):")
        for ring in rings:
            radii = sorted(float(member[4]) for member in ring["members"])
            emit(f"  radius {ring['radius']:9.4f}: "
                 + ", ".join(f"{value:.3f}" for value in radii))

    correction = correct_bragg_peaks(topo, args.size_nm, lattice=spec, result=detection,
                                     return_fft2=False)
    meta = correction.meta
    b_ideal = float(meta["target_radius_nm_inv"])
    before = implied_lattice(meta["measured_radius_nm_inv"], args.a, args.anchor_ring)
    stretch_scale = float(np.sqrt(np.linalg.det(correction.affine_q)))
    emit("")
    emit("== stretch fit ==")
    emit(f"# {meta['n_labelled']} labelled peak(s), method={meta['method']}, "
         f"fallback={meta['fallback']}, rms={meta['rms_residual_px']:.3f} px")
    emit(f"# symmetric stretch M =\n{correction.affine_q}")
    emit(f"# anchored ring before: {meta['measured_radius_nm_inv']:.4f} nm^-1 "
         f"({correction.measured_radius_px:.2f} px), ideal {b_ideal:.4f} nm^-1 "
         f"({correction.target_radius_px:.2f} px), ratio {correction.residual_ratio:.5f}, "
         f"stretch scale |det M|^(1/2) = {stretch_scale:.5f}")
    emit(f"# implied lattice constant of the anchored ring {before['a_ref_nm']:.4f} nm, "
         f"implied 1x1 lattice constant {before['a_1x1_nm']:.4f} nm "
         f"({100 * before['deviation_from_nominal']:+.3f} % from a = {args.a:g} nm)")
    if meta["fallback"]:
        emit("# WARNING: the fit fell back (method = "
             f"{meta['method']}); the returned array is NOT a corrected image, it is "
             "the input resampled onto a larger NaN canvas -- do not trust it")
    emit(f"# corrected canvas: {correction.n_out} x {correction.n_out} px, field of view "
         f"{correction.size_nm:.4f} nm ({correction.size_nm / correction.n_out:.6f} nm/px), "
         f"NaN {100 * (1 - correction.valid_fraction):.2f} %")

    after = detect_bragg_peaks(correction.image, correction.size_nm, lattice=spec,
                               patch_half=args.patch_half, subtract_plane=False,
                               return_fft2=False)
    b_after = (float(np.hypot(*after.lattice.bvecs_nm_inv[0]))
               if after.lattice is not None else float("nan"))
    after_implied = (implied_lattice(b_after, args.a, args.anchor_ring)
                     if np.isfinite(b_after) else None)
    emit("")
    emit("== after the correction (re-detected on the corrected image) ==")
    emit(f"# anchored ring: {b_after:.4f} nm^-1 "
         f"({abs(b_after - b_ideal) / b_ideal * 100:.3f} % off the ideal "
         f"{b_ideal:.4f} nm^-1)" if np.isfinite(b_after) else
         "# anchored ring: not detected on the corrected image")
    if after_implied:
        emit(f"# implied lattice constant of the anchored ring "
             f"{after_implied['a_ref_nm']:.4f} nm, implied 1x1 lattice constant "
             f"{after_implied['a_1x1_nm']:.4f} nm "
             f"({100 * after_implied['deviation_from_nominal']:+.3f} % from "
             f"a = {args.a:g} nm)")
    after_rings = ring_table(after.peaks, args.ring_cluster_tol)
    if after_rings:
        emit("# corrected rings: " + ", ".join(
            f"{float(ring['radius']):.2f} px = "
            f"{float(ring['radius']) * 2 * np.pi / correction.size_nm:.4f} nm^-1 "
            f"({len(ring['members'])} members)"
            for ring in after_rings))
    # Non-circular anchor self-check: the correct anchor makes the two strongest
    # rings of the corrected image a 1 : sqrt(3) pair, because the correction
    # places only the anchored ring on the ideal radius.  A wrong anchor leaves the
    # other ring at a ratio far from sqrt(3), which is the signature to look at.
    check = {"ratio": None, "deviation": None, "consistent": None,
             "verdict": "unverifiable", "outer_radius_px": None,
             "inner_radius_px": None, "tolerance": float(args.ring_cluster_tol),
             "n_rings": len(after_rings)}
    if len(after_rings) >= 2:
        strongest = sorted(after_rings, key=lambda ring: -ring["total_amplitude"])[:2]
        outer = max(strongest, key=lambda ring: ring["radius"])
        inner = min(strongest, key=lambda ring: ring["radius"])
        ratio = float(outer["radius"]) / float(inner["radius"])
        deviation = abs(ratio - np.sqrt(3.0)) / np.sqrt(3.0)
        check.update({"ratio": ratio, "deviation": deviation,
                      "consistent": bool(deviation <= args.ring_cluster_tol),
                      "outer_radius_px": float(outer["radius"]),
                      "inner_radius_px": float(inner["radius"])})
        check["verdict"] = "consistent" if check["consistent"] else "inconsistent"
        emit(f"# anchor self-check: the two strongest corrected rings are at a radius "
             f"ratio {ratio:.6f} (1/sqrt(3) pair: {np.sqrt(3.0):.6f}, deviation "
             f"{100 * deviation:.4f} %) -> anchor verdict = {check['verdict']}")
    else:
        emit(f"# anchor self-check: only {len(after_rings)} ring(s) with at least six "
             f"members detected on the corrected image -> the 1 : sqrt(3) ring pair "
             f"cannot be confirmed (anchor verdict = {check['verdict']})")
    if check["verdict"] != "consistent":
        emit("# WARNING: the corrected image does not show a 1 : sqrt(3) ring pair. "
             "Either the anchor ring is mis-stated (try the other --anchor-ring) or "
             "the data does not contain such a pair.")

    corrected = np.asarray(correction.image, dtype=float)
    out_csv = outdir / f"{stem}_corrected.csv"
    np.savetxt(out_csv, corrected, delimiter=",", fmt="%.10e")
    fft2_corrected = compute_fft2(corrected, correction.size_nm, subtract_plane=False)
    out_fft2 = outdir / f"{stem}_corrected_fft2.npy"
    np.save(out_fft2, fft2_corrected)

    at.setup_style()
    cmap, cmap_source = at.load_colormap(args.stm_lib)
    emit(f"# colormap: {cmap_source}")
    cmap_bad = cmap.copy()
    cmap_bad.set_bad(color=at.BAD_COLOR)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(corrected, cmap=cmap_bad, origin="lower")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")
    fig.savefig(outdir / f"{stem}_corrected.png", dpi=300, bbox_inches="tight",
                pad_inches=0)
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
    fig.savefig(outdir / f"{stem}_corrected_fft.png", dpi=300, bbox_inches="tight",
                pad_inches=0)
    plt.close(fig)

    report = {
        "skill": "stm-topo-phase-analysis",
        "skill_version": SKILL_VERSION,
        "input": str(csv_path),
        "canvas_px": int(topo.shape[0]),
        "field_of_view_nm": float(args.size_nm),
        "nm_per_px": float(args.size_nm / topo.shape[0]),
        "anchor_ring": args.anchor_ring,
        "a_nm": float(args.a),
        "a_ref_nm": a_ref,
        "orientation_deg": orientation_deg,
        "basis_source": detection.meta.get("basis_source"),
        "method": str(meta["method"]),
        "fallback": bool(meta["fallback"]),
        "n_labelled": int(meta["n_labelled"]),
        "rms_residual_px": float(meta["rms_residual_px"]),
        "affine_q": [[float(value) for value in row] for row in np.asarray(correction.affine_q)],
        "stretch_scale_sqrt_det": stretch_scale,
        "b1_measured_nm_inv_before": float(meta["measured_radius_nm_inv"]),
        "b1_measured_px_before": float(correction.measured_radius_px),
        "b1_ideal_nm_inv": b_ideal,
        "b1_ideal_px": float(correction.target_radius_px),
        "residual_ratio_before": float(correction.residual_ratio),
        "implied_lattice_before": before,
        "n_out": int(correction.n_out),
        "corrected_field_of_view_nm": float(correction.size_nm),
        "corrected_nm_per_px": float(correction.size_nm / correction.n_out),
        "nan_fraction": float(1.0 - correction.valid_fraction),
        "b1_measured_nm_inv_after": b_after if np.isfinite(b_after) else None,
        "implied_lattice_after": after_implied,
        "rings_before": [{"radius_px": float(ring["radius"]),
                          "radius_nm_inv": float(ring["radius"]) * 2 * np.pi / args.size_nm,
                          "n_members": len(ring["members"]),
                          "total_amplitude": float(ring["total_amplitude"])}
                         for ring in rings],
        "rings_after": [{"radius_px": float(ring["radius"]),
                         "radius_nm_inv": float(ring["radius"]) * 2 * np.pi / correction.size_nm,
                         "n_members": len(ring["members"]),
                         "total_amplitude": float(ring["total_amplitude"])}
                        for ring in after_rings],
        "anchor_self_check": check,
        "corrected_csv": str(out_csv),
        "corrected_fft2_npy": str(out_fft2),
    }
    (outdir / "correction_report.json").write_text(json.dumps(report, indent=2) + "\n")
    emit("")
    emit(f"# written: {out_csv}, {out_fft2}, "
         f"{outdir / f'{stem}_corrected.png'}, {outdir / f'{stem}_corrected_fft.png'}, "
         f"{outdir / 'correction_report.json'}, {outdir / 'correction.log'}")
    (outdir / "correction.log").write_text("\n".join(log_lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
