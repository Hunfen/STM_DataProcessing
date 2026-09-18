"""STM topography correction through the bragg_peak package.

The detected peak table, the sub-pixel localization (17x17 Gaussian fit) and the
symmetric positive-definite stretch ``M`` (pure stretch, zero rotation) all come
from ``stm_data_processing.utils.bragg_peak``; this script only loads the matrix,
applies the skill's preprocessing (``flipud(subtractMeanPlane(...))``, first row =
top scan line), states the physical reference lattice explicitly and writes the
plots.

The corrected canvas comes from the package result: ``n_out`` pixels with the same
nm-per-pixel scale, so the corrected field of view is ``size_nm * n_out / n``.

Outputs:
    <outdir>/<stem>_corrected.csv      corrected topography (square, NaN padded)
    <outdir>/<stem>_corrected_fft2.npy complex FFT2 (complex128, fftshifted)
    <outdir>/<stem>_corrected.png      topography plot (gwyddion colormap)
    <outdir>/<stem>_corrected_fft.png  FFT plot (inferno, log, percentile norm)

Usage:
    cd /path/to/STM_DataProcessing
    uv run python skills/stm-topo-phase-analysis/scripts/stm_topo_correct.py \
        INPUT.txt -L 100 -o OUT_DIR [--a 0.246] [--delimiter ',']
"""
import argparse
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "text.usetex": False,  # usetex broken with mpl 3.10 + TeX Live 2026
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "font.serif": ["Palatino"],
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Sub-pixel Gaussian patch half width: 8 -> 17x17 pixels.
PATCH_HALF = 8


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("input", help="topography matrix (whitespace, tab or comma separated)")
    p.add_argument("-L", "--size-nm", type=float, required=True,
                   help="scan size in nm (square image)")
    p.add_argument("-o", "--outdir", default=None,
                   help="output directory (default: next to the input file)")
    p.add_argument("--a", type=float, default=0.246,
                   help="graphene lattice constant in nm (default 0.246)")
    p.add_argument("--stm-lib", default="/Users/hunfen/Documents/GitHub/STM_DataProcessing/src",
                   help="STM_DataProcessing src directory (for gwyddion, subtractMeanPlane)")
    p.add_argument("--peaks", nargs=4, type=float, default=None,
                   metavar=("Q1X", "Q1Y", "Q2X", "Q2Y"),
                   help="optional manual anchor pair (q offsets in px): restrict the "
                        "stretch fit to the two detected peaks closest to these positions")
    p.add_argument("--delimiter", default=None,
                   help="column delimiter (default: auto-detect tab/comma/whitespace; "
                        "the escape '\\t' is accepted)")
    p.add_argument("--list-peaks", action="store_true",
                   help="print the detected peak table")
    return p.parse_args()


def resolve_delimiter(raw):
    """Turn the CLI delimiter (None, ',', the escape '\\t', ...) into a delimiter."""
    if raw is None:
        return None
    return raw.replace("\\t", "\t").replace("\\s", " ")


def load_matrix(path, delimiter, loader):
    """Load a square topography matrix, honouring an explicit delimiter."""
    data = (
        loader(path) if delimiter is None else np.loadtxt(path, delimiter=delimiter)
    )
    if data.ndim != 2 or data.shape[0] != data.shape[1]:
        raise ValueError(f"{path}: expected a square 2D matrix, got {data.shape}")
    return data


def pin_anchor_pair(detection, anchors):
    """Restrict the stretch fit to the two detected peaks closest to the anchors.

    ``anchors`` is ``(q1x, q1y, q2x, q2y)`` in FFT pixels relative to the DC
    centre.  The correction then solves ``M`` from those two labelled peaks only
    and falls back to the package's closed-form two-vector solution when the pair
    is rank deficient (the fallback is reported in ``meta``).
    """
    labelled = [
        peak for peak in detection.peaks
        if peak.index_hk is not None and peak.independent
    ]
    if len(labelled) < 2:
        raise ValueError("--peaks needs at least two labelled peaks in the detection")
    chosen = []
    used = set()
    for target in (np.array(anchors[:2]), np.array(anchors[2:])):
        order = sorted(
            range(len(labelled)),
            key=lambda i: float(np.hypot(*(np.array(labelled[i].q_px) - target))),
        )
        pick = next((i for i in order if i not in used), order[0])
        used.add(pick)
        chosen.append(labelled[pick])
    picked = ", ".join(f"{peak.index_hk} q=({peak.q_px[0]:.2f}, {peak.q_px[1]:.2f})"
                       for peak in chosen)
    print(f"anchor pair: {picked}")
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


def print_peak_table(peaks):
    """Print the detected peak table (q offset, |q|, label, SNR)."""
    print("detected peaks (qx, qy, |q|, label, snr, quality):")
    for index, peak in enumerate(peaks, 1):
        qx, qy = peak.q_px
        label = "-" if peak.index_hk is None else f"{peak.index_hk[0]},{peak.index_hk[1]}"
        print(f"  {index:2d}: q=({qx:8.2f}, {qy:8.2f}) |q|={np.hypot(qx, qy):8.2f} "
              f"label={label:>7s} snr={peak.snr:6.2f} {peak.quality}")


def main():
    args = parse_args()
    sys.path.insert(0, args.stm_lib)
    from stm_data_processing.stm.preview_plot import gwyddion
    from stm_data_processing.utils.bragg_peak import (
        LatticeSpec,
        compute_fft2,
        correct_bragg_peaks,
        detect_bragg_peaks,
        load_image,
    )
    from stm_data_processing.utils.plot_funcs import subtractMeanPlane

    csv_path = Path(args.input)
    outdir = Path(args.outdir) if args.outdir else csv_path.parent
    outdir.mkdir(parents=True, exist_ok=True)
    stem = csv_path.stem

    topo = load_matrix(csv_path, resolve_delimiter(args.delimiter), load_image)
    topo = np.flipud(subtractMeanPlane(topo))  # first row = top scan line

    # Peak detection stays data-anchored (the package's hexagon-on-the-first-ring
    # labeller), then the reference lattice is stated explicitly with the physical
    # constant and the orientation the data itself has: no point-group inference.
    detection = detect_bragg_peaks(topo, args.size_nm, patch_half=PATCH_HALF,
                                   subtract_plane=False, return_fft2=False)
    orientation_deg = lattice_orientation_deg(detection)
    spec = LatticeSpec(a_nm=args.a, symmetry="hexagonal",
                       orientation_deg=orientation_deg)
    print(f"reference lattice: hexagonal a = {args.a:g} nm (|b1| = "
          f"{4 * np.pi / (np.sqrt(3) * args.a):.4f} nm^-1), orientation "
          f"{'unknown' if orientation_deg is None else f'{orientation_deg:.2f} deg'}, "
          f"basis_source = {detection.meta.get('basis_source')}")
    if args.peaks is not None:
        detection = pin_anchor_pair(detection, args.peaks)
    if args.list_peaks:
        print_peak_table(detection.peaks)

    correction = correct_bragg_peaks(topo, args.size_nm, lattice=spec, result=detection,
                                     return_fft2=False)
    meta = correction.meta
    b_ideal = float(meta["target_radius_nm_inv"])
    print(f"fit: {meta['n_labelled']} labelled peak(s), method={meta['method']}, "
          f"fallback={meta['fallback']}, rms={meta['rms_residual_px']:.3f} px")
    print(f"symmetric stretch M =\n{correction.affine_q}")
    scale = float(np.sqrt(np.linalg.det(correction.affine_q)))
    print(f"first ring before: labelled-ring mean {meta['measured_radius_nm_inv']:.4f} nm^-1 "
          f"({correction.measured_radius_px:.2f} px), ratio to ideal "
          f"{correction.residual_ratio:.5f}, stretch scale |det M|^(1/2) = {scale:.5f}")
    print(f"ideal ring: {b_ideal:.4f} nm^-1 ({correction.target_radius_px:.2f} px)")
    print(f"corrected canvas: {correction.n_out} x {correction.n_out}, "
          f"field of view {correction.size_nm:.4f} nm, "
          f"NaN {meta['nan_fraction'] * 100:.2f}%")

    # Measured ring after the correction: re-run the detection on the corrected image.
    after = detect_bragg_peaks(correction.image, correction.size_nm, lattice=spec,
                               patch_half=PATCH_HALF, subtract_plane=False,
                               return_fft2=False)
    b_after = (
        float(np.hypot(*after.lattice.bvecs_nm_inv[0])) if after.lattice is not None
        else float("nan")
    )
    print(f"first ring after: {b_after:.4f} nm^-1 "
          f"({abs(b_after - b_ideal) / b_ideal * 100:.3f}% off ideal)")

    # --- save corrected CSV ---
    corrected = correction.image
    out_csv = outdir / f"{stem}_corrected.csv"
    np.savetxt(out_csv, corrected, delimiter=",", fmt="%.10e")

    # --- complex FFT2 of the corrected data (package FFT, NaN filled with the plane) ---
    fft2_c = compute_fft2(corrected, correction.size_nm, subtract_plane=False)
    out_fft2 = outdir / f"{stem}_corrected_fft2.npy"
    np.save(out_fft2, fft2_c)

    # --- topography plot (gwyddion) ---
    cmap = gwyddion.copy()
    cmap.set_bad(color="#b0b0b0")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(corrected, cmap=cmap, origin="lower")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")
    fig.savefig(outdir / f"{stem}_corrected.png", dpi=300,
                bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    # --- FFT plot (inferno, log, percentile norm) ---
    mag_c = np.abs(fft2_c)
    p_lo, p_hi = np.percentile(mag_c, [5, 99.5])
    log_c = np.log(1 + mag_c)
    norm_c = np.clip((log_c - np.log(1 + p_lo)) / (np.log(1 + p_hi) - np.log(1 + p_lo)), 0, 1)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(norm_c, cmap="inferno", origin="lower")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")
    fig.savefig(outdir / f"{stem}_corrected_fft.png", dpi=300,
                bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    print(f"saved: {out_csv}")
    print(f"saved: {out_fft2}")
    print(f"saved: {outdir / f'{stem}_corrected.png'}")
    print(f"saved: {outdir / f'{stem}_corrected_fft.png'}")


if __name__ == "__main__":
    main()
